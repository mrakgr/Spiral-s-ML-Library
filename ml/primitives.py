kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <curand_kernel.h>
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
struct Tuple3 {
    float v0;
    bool v1;
    __device__ Tuple3() = default;
    __device__ Tuple3(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
    __device__ Tuple3 operator()(Tuple3 tup0, Tuple3 tup1){
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
                return Tuple3{v5, true};
            } else {
                return Tuple3{v0, v1};
            }
        } else {
            if (v3){
                return Tuple3{v2, v3};
            } else {
                return Tuple3{v0, v1};
            }
        }
    }
};
struct Closure6 {
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
struct Tuple4 {
    int v0;
    bool v1;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure7 {
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
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 2048l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
extern "C" __global__ void entry0(int * v0, float * v1, float * v2, int * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int * v9, int * v10, int * v11, int * v12, int * v13, float * v14, int * v15) {
    int v16;
    v16 = threadIdx.x;
    int v17;
    v17 = v16;
    while (while_method_0(v17)){
        bool v19;
        v19 = 0l <= v17;
        bool v20;
        v20 = v19 == false;
        if (v20){
            assert("The index needs to be zero or positive." && v19);
        } else {
        }
        int v22;
        v22 = v17 % 32l;
        int v23;
        v23 = v17 / 32l;
        bool v24;
        v24 = v23 < 64l;
        bool v25;
        v25 = v24 == false;
        if (v25){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
        } else {
        }
        assert("Tensor range check" && 0 <= v23 && v23 < 64l);
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        int v27;
        v27 = 4l * v22;
        int v28;
        v28 = 128l * v23;
        int v29;
        v29 = v28 + v27;
        assert("Tensor range check" && 0 <= v23 && v23 < 64l);
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        float v30[4l];
        float v31[4l];
        int4* v32;
        v32 = reinterpret_cast<int4*>(v1 + v29);
        int4* v33;
        v33 = reinterpret_cast<int4*>(v30 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v32) % 4l == 0 && (unsigned long long)(v33) % 4l == 0);
        *v33 = *v32;
        // Pushing the loop unrolling to: 0
        int v34;
        v34 = 0l;
        #pragma unroll
        while (while_method_1(v34)){
            assert("Tensor range check" && 0 <= v34 && v34 < 4l);
            float v36;
            v36 = v30[v34];
            float v37;
            v37 = 1.0f + v36;
            assert("Tensor range check" && 0 <= v34 && v34 < 4l);
            v31[v34] = v37;
            v34 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v38;
        v38 = reinterpret_cast<int4*>(v31 + 0l);
        int4* v39;
        v39 = reinterpret_cast<int4*>(v1 + v29);
        assert("Pointer alignment check" && (unsigned long long)(v38) % 4l == 0 && (unsigned long long)(v39) % 4l == 0);
        *v39 = *v38;
        v17 += 32l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float v40;
    v40 = 0.0f;
    int v41;
    v41 = threadIdx.x;
    int v42;
    v42 = v41;
    while (while_method_0(v42)){
        bool v44;
        v44 = 0l <= v42;
        bool v45;
        v45 = v44 == false;
        if (v45){
            assert("The index needs to be zero or positive." && v44);
        } else {
        }
        int v47;
        v47 = v42 % 32l;
        int v48;
        v48 = v42 / 32l;
        bool v49;
        v49 = v48 < 64l;
        bool v50;
        v50 = v49 == false;
        if (v50){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v49);
        } else {
        }
        assert("Tensor range check" && 0 <= v48 && v48 < 64l);
        assert("Tensor range check" && 0 <= v47 && v47 < 32l);
        int v52;
        v52 = 4l * v47;
        int v53;
        v53 = 128l * v48;
        int v54;
        v54 = v53 + v52;
        float v55[4l];
        int4* v56;
        v56 = reinterpret_cast<int4*>(v1 + v54);
        int4* v57;
        v57 = reinterpret_cast<int4*>(v55 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v56) % 4l == 0 && (unsigned long long)(v57) % 4l == 0);
        *v57 = *v56;
        int v58; float v59;
        Tuple0 tmp0 = Tuple0{0l, v40};
        v58 = tmp0.v0; v59 = tmp0.v1;
        while (while_method_1(v58)){
            assert("Tensor range check" && 0 <= v58 && v58 < 4l);
            float v61;
            v61 = v55[v58];
            float v62;
            v62 = v59 + v61;
            v59 = v62;
            v58 += 1l ;
        }
        v40 = v59;
        v42 += 32l ;
    }
    auto v63 = cooperative_groups::coalesced_threads();
    Closure0 v64{};
    float v65;
    v65 = cooperative_groups::reduce(v63, v40, v64);
    int v66;
    v66 = threadIdx.x;
    int v67;
    v67 = v66 / 32l;
    __shared__ float v68[1l];
    assert("Tensor range check" && 0 <= v67 && v67 < 1l);
    v68[v67] = v65;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v69;
    v69 = threadIdx.x;
    int v70;
    v70 = v69 % 32l;
    bool v71;
    v71 = v67 == 0l;
    bool v73;
    if (v71){
        bool v72;
        v72 = v70 < 1l;
        v73 = v72;
    } else {
        v73 = false;
    }
    if (v73){
        auto v74 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v70 && v70 < 1l);
        float v75;
        v75 = v68[v70];
        float v76;
        v76 = cooperative_groups::reduce(v74, v75, v64);
        v2[0l] = v76;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v77;
    v77 = threadIdx.x;
    bool v78;
    v78 = 0l <= v77;
    bool v79;
    v79 = v78 == false;
    if (v79){
        assert("The index needs to be zero or positive." && v78);
    } else {
    }
    int v81;
    v81 = v77 % 32l;
    int v82;
    v82 = v77 / 32l;
    bool v83;
    v83 = v82 < 1l;
    bool v84;
    v84 = v83 == false;
    if (v84){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v83);
    } else {
    }
    assert("Tensor range check" && 0 <= v82 && v82 < 1l);
    assert("Tensor range check" && 0 <= v81 && v81 < 32l);
    int v86;
    v86 = 4l * v81;
    int v87;
    v87 = 128l * v82;
    int v88;
    v88 = v87 + v86;
    assert("Tensor range check" && 0 <= v82 && v82 < 1l);
    assert("Tensor range check" && 0 <= v81 && v81 < 32l);
    int v89;
    v89 = 0l;
    while (while_method_2(v89)){
        assert("Tensor range check" && 0 <= v89 && v89 < 64l);
        int v91;
        v91 = 128l * v89;
        int v92;
        v92 = v91 + v88;
        int v93[4l];
        int v94[4l];
        int v95;
        v95 = 0l;
        while (while_method_3(v95)){
            assert("Tensor range check" && 0 <= v95 && v95 < 1l);
            int v97;
            v97 = 4l * v95;
            assert("Tensor range check" && 0 <= v95 && v95 < 1l);
            int v98;
            v98 = 128l * v95;
            int v99;
            v99 = v98 + v92;
            int4* v100;
            v100 = reinterpret_cast<int4*>(v0 + v99);
            int4* v101;
            v101 = reinterpret_cast<int4*>(v93 + v97);
            assert("Pointer alignment check" && (unsigned long long)(v100) % 4l == 0 && (unsigned long long)(v101) % 4l == 0);
            *v101 = *v100;
            v95 += 1l ;
        }
        int v102;
        v102 = 0l;
        while (while_method_3(v102)){
            int v104;
            v104 = 0l;
            while (while_method_1(v104)){
                bool v106;
                v106 = 0l <= v104;
                bool v108;
                if (v106){
                    bool v107;
                    v107 = v104 < 4l;
                    v108 = v107;
                } else {
                    v108 = false;
                }
                bool v109;
                v109 = v108 == false;
                if (v109){
                    assert("The indices should be inside the range of the dimension." && v108);
                } else {
                }
                bool v111;
                v111 = 0l <= v81;
                bool v113;
                if (v111){
                    bool v112;
                    v112 = v81 < 32l;
                    v113 = v112;
                } else {
                    v113 = false;
                }
                bool v114;
                v114 = v113 == false;
                if (v114){
                    assert("The indices should be inside the range of the dimension." && v113);
                } else {
                }
                int v116;
                v116 = v81 * 4l;
                int v117;
                v117 = v104 + v116;
                bool v118;
                v118 = 0l <= v102;
                bool v120;
                if (v118){
                    bool v119;
                    v119 = v102 < 1l;
                    v120 = v119;
                } else {
                    v120 = false;
                }
                bool v121;
                v121 = v120 == false;
                if (v121){
                    assert("The indices should be inside the range of the dimension." && v120);
                } else {
                }
                int v123;
                v123 = v102 * 128l;
                int v124;
                v124 = v117 + v123;
                assert("Tensor range check" && 0 <= v102 && v102 < 1l);
                assert("Tensor range check" && 0 <= v104 && v104 < 4l);
                int v125;
                v125 = 4l * v102;
                int v126;
                v126 = v125 + v104;
                v94[v126] = v124;
                v104 += 1l ;
            }
            v102 += 1l ;
        }
        bool v127;
        v127 = 0l <= v82;
        bool v128;
        v128 = v127 && v83;
        bool v129;
        v129 = v128 == false;
        if (v129){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v128);
        } else {
        }
        bool v131;
        v131 = 0l <= v89;
        bool v133;
        if (v131){
            bool v132;
            v132 = v89 < 64l;
            v133 = v132;
        } else {
            v133 = false;
        }
        bool v134;
        v134 = v133 == false;
        if (v134){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v133);
        } else {
        }
        int v136;
        v136 = v89 + v82;
        assert("Tensor range check" && 0 <= v89 && v89 < 64l);
        int v137;
        v137 = 0l;
        while (while_method_3(v137)){
            assert("Tensor range check" && 0 <= v137 && v137 < 1l);
            int v139;
            v139 = 128l * v137;
            int v140;
            v140 = v139 + v92;
            assert("Tensor range check" && 0 <= v137 && v137 < 1l);
            int v141;
            v141 = 4l * v137;
            int4* v142;
            v142 = reinterpret_cast<int4*>(v93 + v141);
            int4* v143;
            v143 = reinterpret_cast<int4*>(v3 + v140);
            assert("Pointer alignment check" && (unsigned long long)(v142) % 4l == 0 && (unsigned long long)(v143) % 4l == 0);
            *v143 = *v142;
            v137 += 1l ;
        }
        v89 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v144;
    v144 = threadIdx.x;
    bool v145;
    v145 = 0l <= v144;
    bool v146;
    v146 = v145 == false;
    if (v146){
        assert("The index needs to be zero or positive." && v145);
    } else {
    }
    int v148;
    v148 = v144 % 32l;
    int v149;
    v149 = v144 / 32l;
    bool v150;
    v150 = v149 < 1l;
    bool v151;
    v151 = v150 == false;
    if (v151){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v150);
    } else {
    }
    assert("Tensor range check" && 0 <= v149 && v149 < 1l);
    assert("Tensor range check" && 0 <= v148 && v148 < 32l);
    int v153;
    v153 = 4l * v148;
    int v154;
    v154 = 128l * v149;
    int v155;
    v155 = v154 + v153;
    assert("Tensor range check" && 0 <= v149 && v149 < 1l);
    assert("Tensor range check" && 0 <= v148 && v148 < 32l);
    int v156;
    v156 = 0l;
    while (while_method_2(v156)){
        assert("Tensor range check" && 0 <= v156 && v156 < 64l);
        int v158;
        v158 = 128l * v156;
        int v159;
        v159 = v158 + v155;
        float v160[4l];
        int v161[4l];
        int v162;
        v162 = 0l;
        while (while_method_3(v162)){
            assert("Tensor range check" && 0 <= v162 && v162 < 1l);
            int v164;
            v164 = 4l * v162;
            assert("Tensor range check" && 0 <= v162 && v162 < 1l);
            int v165;
            v165 = 128l * v162;
            int v166;
            v166 = v165 + v159;
            int4* v167;
            v167 = reinterpret_cast<int4*>(v1 + v166);
            int4* v168;
            v168 = reinterpret_cast<int4*>(v160 + v164);
            assert("Pointer alignment check" && (unsigned long long)(v167) % 4l == 0 && (unsigned long long)(v168) % 4l == 0);
            *v168 = *v167;
            v162 += 1l ;
        }
        int v169;
        v169 = 0l;
        while (while_method_3(v169)){
            int v171;
            v171 = 0l;
            while (while_method_1(v171)){
                bool v173;
                v173 = 0l <= v171;
                bool v175;
                if (v173){
                    bool v174;
                    v174 = v171 < 4l;
                    v175 = v174;
                } else {
                    v175 = false;
                }
                bool v176;
                v176 = v175 == false;
                if (v176){
                    assert("The indices should be inside the range of the dimension." && v175);
                } else {
                }
                bool v178;
                v178 = 0l <= v148;
                bool v180;
                if (v178){
                    bool v179;
                    v179 = v148 < 32l;
                    v180 = v179;
                } else {
                    v180 = false;
                }
                bool v181;
                v181 = v180 == false;
                if (v181){
                    assert("The indices should be inside the range of the dimension." && v180);
                } else {
                }
                int v183;
                v183 = v148 * 4l;
                int v184;
                v184 = v171 + v183;
                bool v185;
                v185 = 0l <= v169;
                bool v187;
                if (v185){
                    bool v186;
                    v186 = v169 < 1l;
                    v187 = v186;
                } else {
                    v187 = false;
                }
                bool v188;
                v188 = v187 == false;
                if (v188){
                    assert("The indices should be inside the range of the dimension." && v187);
                } else {
                }
                int v190;
                v190 = v169 * 128l;
                int v191;
                v191 = v184 + v190;
                assert("Tensor range check" && 0 <= v169 && v169 < 1l);
                assert("Tensor range check" && 0 <= v171 && v171 < 4l);
                int v192;
                v192 = 4l * v169;
                int v193;
                v193 = v192 + v171;
                v161[v193] = v191;
                v171 += 1l ;
            }
            v169 += 1l ;
        }
        bool v194;
        v194 = 0l <= v149;
        bool v195;
        v195 = v194 && v150;
        bool v196;
        v196 = v195 == false;
        if (v196){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v195);
        } else {
        }
        bool v198;
        v198 = 0l <= v156;
        bool v200;
        if (v198){
            bool v199;
            v199 = v156 < 64l;
            v200 = v199;
        } else {
            v200 = false;
        }
        bool v201;
        v201 = v200 == false;
        if (v201){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v200);
        } else {
        }
        int v203;
        v203 = v156 + v149;
        int v204[4l];
        int v205[4l];
        int v206;
        v206 = 0l;
        while (while_method_3(v206)){
            int v208;
            v208 = 0l;
            while (while_method_1(v208)){
                assert("Tensor range check" && 0 <= v206 && v206 < 1l);
                assert("Tensor range check" && 0 <= v208 && v208 < 4l);
                int v210;
                v210 = 4l * v206;
                int v211;
                v211 = v210 + v208;
                int v212;
                v212 = v161[v211];
                assert("Tensor range check" && 0 <= v206 && v206 < 1l);
                assert("Tensor range check" && 0 <= v208 && v208 < 4l);
                v204[v211] = v203;
                v205[v211] = v212;
                v208 += 1l ;
            }
            v206 += 1l ;
        }
        assert("Tensor range check" && 0 <= v156 && v156 < 64l);
        int v213;
        v213 = 0l;
        while (while_method_3(v213)){
            assert("Tensor range check" && 0 <= v213 && v213 < 1l);
            int v215;
            v215 = 128l * v213;
            int v216;
            v216 = v215 + v159;
            assert("Tensor range check" && 0 <= v213 && v213 < 1l);
            int v217;
            v217 = 4l * v213;
            int4* v218;
            v218 = reinterpret_cast<int4*>(v204 + v217);
            int4* v219;
            v219 = reinterpret_cast<int4*>(v10 + v216);
            assert("Pointer alignment check" && (unsigned long long)(v218) % 4l == 0 && (unsigned long long)(v219) % 4l == 0);
            *v219 = *v218;
            int4* v220;
            v220 = reinterpret_cast<int4*>(v205 + v217);
            int4* v221;
            v221 = reinterpret_cast<int4*>(v11 + v216);
            assert("Pointer alignment check" && (unsigned long long)(v220) % 4l == 0 && (unsigned long long)(v221) % 4l == 0);
            *v221 = *v220;
            v213 += 1l ;
        }
        v156 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v222;
    v222 = threadIdx.x;
    bool v223;
    v223 = 0l <= v222;
    bool v224;
    v224 = v223 == false;
    if (v224){
        assert("The index needs to be zero or positive." && v223);
    } else {
    }
    int v226;
    v226 = v222 % 32l;
    int v227;
    v227 = v222 / 32l;
    bool v228;
    v228 = v227 < 1l;
    bool v229;
    v229 = v228 == false;
    if (v229){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v228);
    } else {
    }
    assert("Tensor range check" && 0 <= v227 && v227 < 1l);
    assert("Tensor range check" && 0 <= v226 && v226 < 32l);
    int v231;
    v231 = 4l * v226;
    int v232;
    v232 = 128l * v227;
    int v233;
    v233 = v232 + v231;
    assert("Tensor range check" && 0 <= v227 && v227 < 1l);
    int v234;
    v234 = 0l;
    while (while_method_2(v234)){
        assert("Tensor range check" && 0 <= v234 && v234 < 64l);
        int v236;
        v236 = 128l * v234;
        int v237;
        v237 = v236 + v233;
        float v238[4l];
        int v239[4l];
        int v240;
        v240 = 0l;
        while (while_method_3(v240)){
            assert("Tensor range check" && 0 <= v240 && v240 < 1l);
            int v242;
            v242 = 4l * v240;
            assert("Tensor range check" && 0 <= v240 && v240 < 1l);
            int v243;
            v243 = 128l * v240;
            int v244;
            v244 = v243 + v237;
            int4* v245;
            v245 = reinterpret_cast<int4*>(v1 + v244);
            int4* v246;
            v246 = reinterpret_cast<int4*>(v238 + v242);
            assert("Pointer alignment check" && (unsigned long long)(v245) % 4l == 0 && (unsigned long long)(v246) % 4l == 0);
            *v246 = *v245;
            v240 += 1l ;
        }
        int v247;
        v247 = 0l;
        while (while_method_3(v247)){
            int v249;
            v249 = 0l;
            while (while_method_1(v249)){
                bool v251;
                v251 = 0l <= v249;
                bool v253;
                if (v251){
                    bool v252;
                    v252 = v249 < 4l;
                    v253 = v252;
                } else {
                    v253 = false;
                }
                bool v254;
                v254 = v253 == false;
                if (v254){
                    assert("The indices should be inside the range of the dimension." && v253);
                } else {
                }
                bool v256;
                v256 = 0l <= v226;
                bool v258;
                if (v256){
                    bool v257;
                    v257 = v226 < 32l;
                    v258 = v257;
                } else {
                    v258 = false;
                }
                bool v259;
                v259 = v258 == false;
                if (v259){
                    assert("The indices should be inside the range of the dimension." && v258);
                } else {
                }
                int v261;
                v261 = v226 * 4l;
                int v262;
                v262 = v249 + v261;
                bool v263;
                v263 = 0l <= v247;
                bool v265;
                if (v263){
                    bool v264;
                    v264 = v247 < 1l;
                    v265 = v264;
                } else {
                    v265 = false;
                }
                bool v266;
                v266 = v265 == false;
                if (v266){
                    assert("The indices should be inside the range of the dimension." && v265);
                } else {
                }
                int v268;
                v268 = v247 * 128l;
                int v269;
                v269 = v262 + v268;
                assert("Tensor range check" && 0 <= v247 && v247 < 1l);
                assert("Tensor range check" && 0 <= v249 && v249 < 4l);
                int v270;
                v270 = 4l * v247;
                int v271;
                v271 = v270 + v249;
                v239[v271] = v269;
                v249 += 1l ;
            }
            v247 += 1l ;
        }
        bool v272;
        v272 = 0l <= v227;
        bool v273;
        v273 = v272 && v228;
        bool v274;
        v274 = v273 == false;
        if (v274){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v273);
        } else {
        }
        bool v276;
        v276 = 0l <= v234;
        bool v278;
        if (v276){
            bool v277;
            v277 = v234 < 64l;
            v278 = v277;
        } else {
            v278 = false;
        }
        bool v279;
        v279 = v278 == false;
        if (v279){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v278);
        } else {
        }
        int v281;
        v281 = v234 + v227;
        assert("Tensor range check" && 0 <= v234 && v234 < 64l);
        v12[v281] = v281;
        v234 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v282;
    v282 = threadIdx.x;
    bool v283;
    v283 = 0l <= v282;
    bool v284;
    v284 = v283 == false;
    if (v284){
        assert("The index needs to be zero or positive." && v283);
    } else {
    }
    int v286;
    v286 = v282 % 32l;
    int v287;
    v287 = v282 / 32l;
    bool v288;
    v288 = v287 < 1l;
    bool v289;
    v289 = v288 == false;
    if (v289){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v288);
    } else {
    }
    assert("Tensor range check" && 0 <= v287 && v287 < 1l);
    assert("Tensor range check" && 0 <= v286 && v286 < 32l);
    int v291;
    v291 = 4l * v286;
    int v292;
    v292 = 128l * v287;
    int v293;
    v293 = v292 + v291;
    assert("Tensor range check" && 0 <= v287 && v287 < 1l);
    assert("Tensor range check" && 0 <= v286 && v286 < 32l);
    int v294;
    v294 = 0l;
    while (while_method_2(v294)){
        assert("Tensor range check" && 0 <= v294 && v294 < 64l);
        int v296;
        v296 = 128l * v294;
        int v297;
        v297 = v296 + v293;
        float v298[4l];
        int v299[4l];
        int v300;
        v300 = 0l;
        while (while_method_3(v300)){
            assert("Tensor range check" && 0 <= v300 && v300 < 1l);
            int v302;
            v302 = 4l * v300;
            assert("Tensor range check" && 0 <= v300 && v300 < 1l);
            int v303;
            v303 = 128l * v300;
            int v304;
            v304 = v303 + v297;
            int4* v305;
            v305 = reinterpret_cast<int4*>(v1 + v304);
            int4* v306;
            v306 = reinterpret_cast<int4*>(v298 + v302);
            assert("Pointer alignment check" && (unsigned long long)(v305) % 4l == 0 && (unsigned long long)(v306) % 4l == 0);
            *v306 = *v305;
            v300 += 1l ;
        }
        int v307;
        v307 = 0l;
        while (while_method_3(v307)){
            int v309;
            v309 = 0l;
            while (while_method_1(v309)){
                bool v311;
                v311 = 0l <= v309;
                bool v313;
                if (v311){
                    bool v312;
                    v312 = v309 < 4l;
                    v313 = v312;
                } else {
                    v313 = false;
                }
                bool v314;
                v314 = v313 == false;
                if (v314){
                    assert("The indices should be inside the range of the dimension." && v313);
                } else {
                }
                bool v316;
                v316 = 0l <= v286;
                bool v318;
                if (v316){
                    bool v317;
                    v317 = v286 < 32l;
                    v318 = v317;
                } else {
                    v318 = false;
                }
                bool v319;
                v319 = v318 == false;
                if (v319){
                    assert("The indices should be inside the range of the dimension." && v318);
                } else {
                }
                int v321;
                v321 = v286 * 4l;
                int v322;
                v322 = v309 + v321;
                bool v323;
                v323 = 0l <= v307;
                bool v325;
                if (v323){
                    bool v324;
                    v324 = v307 < 1l;
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
                v328 = v307 * 128l;
                int v329;
                v329 = v322 + v328;
                assert("Tensor range check" && 0 <= v307 && v307 < 1l);
                assert("Tensor range check" && 0 <= v309 && v309 < 4l);
                int v330;
                v330 = 4l * v307;
                int v331;
                v331 = v330 + v309;
                v299[v331] = v329;
                v309 += 1l ;
            }
            v307 += 1l ;
        }
        bool v332;
        v332 = 0l <= v287;
        bool v333;
        v333 = v332 && v288;
        bool v334;
        v334 = v333 == false;
        if (v334){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v333);
        } else {
        }
        bool v336;
        v336 = 0l <= v294;
        bool v338;
        if (v336){
            bool v337;
            v337 = v294 < 64l;
            v338 = v337;
        } else {
            v338 = false;
        }
        bool v339;
        v339 = v338 == false;
        if (v339){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v338);
        } else {
        }
        int v341;
        v341 = v294 + v287;
        float v342;
        v342 = 0.0f;
        int v343;
        v343 = 0l;
        while (while_method_3(v343)){
            int v345;
            v345 = 0l;
            while (while_method_1(v345)){
                assert("Tensor range check" && 0 <= v343 && v343 < 1l);
                assert("Tensor range check" && 0 <= v345 && v345 < 4l);
                int v347;
                v347 = 4l * v343;
                int v348;
                v348 = v347 + v345;
                float v349;
                v349 = v298[v348];
                float v350;
                v350 = v342 + v349;
                v342 = v350;
                v345 += 1l ;
            }
            v343 += 1l ;
        }
        auto v351 = cooperative_groups::coalesced_threads();
        int v352;
        v352 = threadIdx.x;
        int v353;
        v353 = v352 / 32l;
        auto v354 = cooperative_groups::labeled_partition(v351,v353);
        float v355;
        v355 = cooperative_groups::reduce(v354, v342, v64);
        float v356;
        v356 = v355 / 128.0f;
        float v357[4l];
        int v358;
        v358 = 0l;
        while (while_method_3(v358)){
            int v360;
            v360 = 0l;
            while (while_method_1(v360)){
                assert("Tensor range check" && 0 <= v358 && v358 < 1l);
                assert("Tensor range check" && 0 <= v360 && v360 < 4l);
                int v362;
                v362 = 4l * v358;
                int v363;
                v363 = v362 + v360;
                float v364;
                v364 = v298[v363];
                float v365;
                v365 = v364 - v356;
                float v366;
                v366 = exp(v365);
                assert("Tensor range check" && 0 <= v358 && v358 < 1l);
                assert("Tensor range check" && 0 <= v360 && v360 < 4l);
                v357[v363] = v366;
                v360 += 1l ;
            }
            v358 += 1l ;
        }
        float v367;
        v367 = 0.0f;
        int v368;
        v368 = 0l;
        while (while_method_3(v368)){
            int v370;
            v370 = 0l;
            while (while_method_1(v370)){
                assert("Tensor range check" && 0 <= v368 && v368 < 1l);
                assert("Tensor range check" && 0 <= v370 && v370 < 4l);
                int v372;
                v372 = 4l * v368;
                int v373;
                v373 = v372 + v370;
                float v374;
                v374 = v357[v373];
                float v375;
                v375 = v367 + v374;
                v367 = v375;
                v370 += 1l ;
            }
            v368 += 1l ;
        }
        auto v376 = cooperative_groups::coalesced_threads();
        int v377;
        v377 = threadIdx.x;
        int v378;
        v378 = v377 / 32l;
        auto v379 = cooperative_groups::labeled_partition(v376,v378);
        float v380;
        v380 = cooperative_groups::reduce(v379, v367, v64);
        float v381[4l];
        int v382;
        v382 = 0l;
        while (while_method_3(v382)){
            int v384;
            v384 = 0l;
            while (while_method_1(v384)){
                assert("Tensor range check" && 0 <= v382 && v382 < 1l);
                assert("Tensor range check" && 0 <= v384 && v384 < 4l);
                int v386;
                v386 = 4l * v382;
                int v387;
                v387 = v386 + v384;
                float v388;
                v388 = v357[v387];
                float v389;
                v389 = v388 / v380;
                assert("Tensor range check" && 0 <= v382 && v382 < 1l);
                assert("Tensor range check" && 0 <= v384 && v384 < 4l);
                v381[v387] = v389;
                v384 += 1l ;
            }
            v382 += 1l ;
        }
        assert("Tensor range check" && 0 <= v294 && v294 < 64l);
        int v390;
        v390 = 0l;
        while (while_method_3(v390)){
            assert("Tensor range check" && 0 <= v390 && v390 < 1l);
            int v392;
            v392 = 128l * v390;
            int v393;
            v393 = v392 + v297;
            assert("Tensor range check" && 0 <= v390 && v390 < 1l);
            int v394;
            v394 = 4l * v390;
            int4* v395;
            v395 = reinterpret_cast<int4*>(v381 + v394);
            int4* v396;
            v396 = reinterpret_cast<int4*>(v4 + v393);
            assert("Pointer alignment check" && (unsigned long long)(v395) % 4l == 0 && (unsigned long long)(v396) % 4l == 0);
            *v396 = *v395;
            v390 += 1l ;
        }
        v294 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v397;
    v397 = threadIdx.x;
    bool v398;
    v398 = 0l <= v397;
    bool v399;
    v399 = v398 == false;
    if (v399){
        assert("The index needs to be zero or positive." && v398);
    } else {
    }
    int v401;
    v401 = v397 % 32l;
    int v402;
    v402 = v397 / 32l;
    bool v403;
    v403 = v402 < 1l;
    bool v404;
    v404 = v403 == false;
    if (v404){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v403);
    } else {
    }
    assert("Tensor range check" && 0 <= v402 && v402 < 1l);
    assert("Tensor range check" && 0 <= v401 && v401 < 32l);
    int v406;
    v406 = 4l * v401;
    int v407;
    v407 = 128l * v402;
    int v408;
    v408 = v407 + v406;
    assert("Tensor range check" && 0 <= v402 && v402 < 1l);
    assert("Tensor range check" && 0 <= v401 && v401 < 32l);
    int v409;
    v409 = 0l;
    while (while_method_2(v409)){
        assert("Tensor range check" && 0 <= v409 && v409 < 64l);
        int v411;
        v411 = 128l * v409;
        int v412;
        v412 = v411 + v408;
        float v413[4l];
        int v414[4l];
        int v415;
        v415 = 0l;
        while (while_method_3(v415)){
            assert("Tensor range check" && 0 <= v415 && v415 < 1l);
            int v417;
            v417 = 4l * v415;
            assert("Tensor range check" && 0 <= v415 && v415 < 1l);
            int v418;
            v418 = 128l * v415;
            int v419;
            v419 = v418 + v412;
            int4* v420;
            v420 = reinterpret_cast<int4*>(v1 + v419);
            int4* v421;
            v421 = reinterpret_cast<int4*>(v413 + v417);
            assert("Pointer alignment check" && (unsigned long long)(v420) % 4l == 0 && (unsigned long long)(v421) % 4l == 0);
            *v421 = *v420;
            v415 += 1l ;
        }
        int v422;
        v422 = 0l;
        while (while_method_3(v422)){
            int v424;
            v424 = 0l;
            while (while_method_1(v424)){
                bool v426;
                v426 = 0l <= v424;
                bool v428;
                if (v426){
                    bool v427;
                    v427 = v424 < 4l;
                    v428 = v427;
                } else {
                    v428 = false;
                }
                bool v429;
                v429 = v428 == false;
                if (v429){
                    assert("The indices should be inside the range of the dimension." && v428);
                } else {
                }
                bool v431;
                v431 = 0l <= v401;
                bool v433;
                if (v431){
                    bool v432;
                    v432 = v401 < 32l;
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
                int v436;
                v436 = v401 * 4l;
                int v437;
                v437 = v424 + v436;
                bool v438;
                v438 = 0l <= v422;
                bool v440;
                if (v438){
                    bool v439;
                    v439 = v422 < 1l;
                    v440 = v439;
                } else {
                    v440 = false;
                }
                bool v441;
                v441 = v440 == false;
                if (v441){
                    assert("The indices should be inside the range of the dimension." && v440);
                } else {
                }
                int v443;
                v443 = v422 * 128l;
                int v444;
                v444 = v437 + v443;
                assert("Tensor range check" && 0 <= v422 && v422 < 1l);
                assert("Tensor range check" && 0 <= v424 && v424 < 4l);
                int v445;
                v445 = 4l * v422;
                int v446;
                v446 = v445 + v424;
                v414[v446] = v444;
                v424 += 1l ;
            }
            v422 += 1l ;
        }
        bool v447;
        v447 = 0l <= v402;
        bool v448;
        v448 = v447 && v403;
        bool v449;
        v449 = v448 == false;
        if (v449){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v448);
        } else {
        }
        bool v451;
        v451 = 0l <= v409;
        bool v453;
        if (v451){
            bool v452;
            v452 = v409 < 64l;
            v453 = v452;
        } else {
            v453 = false;
        }
        bool v454;
        v454 = v453 == false;
        if (v454){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v453);
        } else {
        }
        int v456;
        v456 = v409 + v402;
        float v457[4l];
        int v458;
        v458 = 0l;
        while (while_method_3(v458)){
            int v460;
            v460 = 0l;
            while (while_method_1(v460)){
                assert("Tensor range check" && 0 <= v458 && v458 < 1l);
                assert("Tensor range check" && 0 <= v460 && v460 < 4l);
                int v462;
                v462 = 4l * v458;
                int v463;
                v463 = v462 + v460;
                float v464;
                v464 = v413[v463];
                float v465;
                v465 = v464 * v464;
                assert("Tensor range check" && 0 <= v458 && v458 < 1l);
                assert("Tensor range check" && 0 <= v460 && v460 < 4l);
                v457[v463] = v465;
                v460 += 1l ;
            }
            v458 += 1l ;
        }
        float v466;
        v466 = 0.0f;
        int v467;
        v467 = 0l;
        while (while_method_3(v467)){
            int v469;
            v469 = 0l;
            while (while_method_1(v469)){
                assert("Tensor range check" && 0 <= v467 && v467 < 1l);
                assert("Tensor range check" && 0 <= v469 && v469 < 4l);
                int v471;
                v471 = 4l * v467;
                int v472;
                v472 = v471 + v469;
                float v473;
                v473 = v457[v472];
                float v474;
                v474 = v466 + v473;
                v466 = v474;
                v469 += 1l ;
            }
            v467 += 1l ;
        }
        auto v475 = cooperative_groups::coalesced_threads();
        int v476;
        v476 = threadIdx.x;
        int v477;
        v477 = v476 / 32l;
        auto v478 = cooperative_groups::labeled_partition(v475,v477);
        float v479;
        v479 = cooperative_groups::reduce(v478, v466, v64);
        float v480[4l];
        int v481;
        v481 = 0l;
        while (while_method_3(v481)){
            int v483;
            v483 = 0l;
            while (while_method_1(v483)){
                assert("Tensor range check" && 0 <= v481 && v481 < 1l);
                assert("Tensor range check" && 0 <= v483 && v483 < 4l);
                int v485;
                v485 = 4l * v481;
                int v486;
                v486 = v485 + v483;
                float v487;
                v487 = v413[v486];
                bool v488;
                v488 = v479 == 0.0f;
                bool v489;
                v489 = v488 != true;
                float v491;
                if (v489){
                    float v490;
                    v490 = v487 / v479;
                    v491 = v490;
                } else {
                    v491 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v481 && v481 < 1l);
                assert("Tensor range check" && 0 <= v483 && v483 < 4l);
                v480[v486] = v491;
                v483 += 1l ;
            }
            v481 += 1l ;
        }
        assert("Tensor range check" && 0 <= v409 && v409 < 64l);
        int v492;
        v492 = 0l;
        while (while_method_3(v492)){
            assert("Tensor range check" && 0 <= v492 && v492 < 1l);
            int v494;
            v494 = 128l * v492;
            int v495;
            v495 = v494 + v412;
            assert("Tensor range check" && 0 <= v492 && v492 < 1l);
            int v496;
            v496 = 4l * v492;
            int4* v497;
            v497 = reinterpret_cast<int4*>(v480 + v496);
            int4* v498;
            v498 = reinterpret_cast<int4*>(v8 + v495);
            assert("Pointer alignment check" && (unsigned long long)(v497) % 4l == 0 && (unsigned long long)(v498) % 4l == 0);
            *v498 = *v497;
            v492 += 1l ;
        }
        v409 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v499;
    v499 = threadIdx.x;
    bool v500;
    v500 = 0l <= v499;
    bool v501;
    v501 = v500 == false;
    if (v501){
        assert("The index needs to be zero or positive." && v500);
    } else {
    }
    int v503;
    v503 = v499 % 32l;
    int v504;
    v504 = v499 / 32l;
    bool v505;
    v505 = v504 < 1l;
    bool v506;
    v506 = v505 == false;
    if (v506){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v505);
    } else {
    }
    assert("Tensor range check" && 0 <= v504 && v504 < 1l);
    assert("Tensor range check" && 0 <= v503 && v503 < 32l);
    int v508;
    v508 = 4l * v503;
    int v509;
    v509 = 128l * v504;
    int v510;
    v510 = v509 + v508;
    assert("Tensor range check" && 0 <= v504 && v504 < 1l);
    int v511;
    v511 = 0l;
    while (while_method_2(v511)){
        assert("Tensor range check" && 0 <= v511 && v511 < 64l);
        int v513;
        v513 = 128l * v511;
        int v514;
        v514 = v513 + v510;
        float v515[4l];
        int v516[4l];
        int v517;
        v517 = 0l;
        while (while_method_3(v517)){
            assert("Tensor range check" && 0 <= v517 && v517 < 1l);
            int v519;
            v519 = 4l * v517;
            assert("Tensor range check" && 0 <= v517 && v517 < 1l);
            int v520;
            v520 = 128l * v517;
            int v521;
            v521 = v520 + v514;
            int4* v522;
            v522 = reinterpret_cast<int4*>(v1 + v521);
            int4* v523;
            v523 = reinterpret_cast<int4*>(v515 + v519);
            assert("Pointer alignment check" && (unsigned long long)(v522) % 4l == 0 && (unsigned long long)(v523) % 4l == 0);
            *v523 = *v522;
            v517 += 1l ;
        }
        int v524;
        v524 = 0l;
        while (while_method_3(v524)){
            int v526;
            v526 = 0l;
            while (while_method_1(v526)){
                bool v528;
                v528 = 0l <= v526;
                bool v530;
                if (v528){
                    bool v529;
                    v529 = v526 < 4l;
                    v530 = v529;
                } else {
                    v530 = false;
                }
                bool v531;
                v531 = v530 == false;
                if (v531){
                    assert("The indices should be inside the range of the dimension." && v530);
                } else {
                }
                bool v533;
                v533 = 0l <= v503;
                bool v535;
                if (v533){
                    bool v534;
                    v534 = v503 < 32l;
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
                int v538;
                v538 = v503 * 4l;
                int v539;
                v539 = v526 + v538;
                bool v540;
                v540 = 0l <= v524;
                bool v542;
                if (v540){
                    bool v541;
                    v541 = v524 < 1l;
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
                int v545;
                v545 = v524 * 128l;
                int v546;
                v546 = v539 + v545;
                assert("Tensor range check" && 0 <= v524 && v524 < 1l);
                assert("Tensor range check" && 0 <= v526 && v526 < 4l);
                int v547;
                v547 = 4l * v524;
                int v548;
                v548 = v547 + v526;
                v516[v548] = v546;
                v526 += 1l ;
            }
            v524 += 1l ;
        }
        bool v549;
        v549 = 0l <= v504;
        bool v550;
        v550 = v549 && v505;
        bool v551;
        v551 = v550 == false;
        if (v551){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v550);
        } else {
        }
        bool v553;
        v553 = 0l <= v511;
        bool v555;
        if (v553){
            bool v554;
            v554 = v511 < 64l;
            v555 = v554;
        } else {
            v555 = false;
        }
        bool v556;
        v556 = v555 == false;
        if (v556){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v555);
        } else {
        }
        int v558;
        v558 = v511 + v504;
        float v559; int v560;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0l};
        v559 = tmp1.v0; v560 = tmp1.v1;
        int v561;
        v561 = 0l;
        while (while_method_3(v561)){
            int v563;
            v563 = 0l;
            while (while_method_1(v563)){
                assert("Tensor range check" && 0 <= v561 && v561 < 1l);
                assert("Tensor range check" && 0 <= v563 && v563 < 4l);
                int v565;
                v565 = 4l * v561;
                int v566;
                v566 = v565 + v563;
                float v567;
                v567 = v515[v566];
                int v568;
                v568 = v516[v566];
                bool v569;
                v569 = v559 > v567;
                float v570; int v571;
                if (v569){
                    v570 = v559; v571 = v560;
                } else {
                    v570 = v567; v571 = v568;
                }
                v559 = v570;
                v560 = v571;
                v563 += 1l ;
            }
            v561 += 1l ;
        }
        auto v572 = cooperative_groups::coalesced_threads();
        int v573;
        v573 = threadIdx.x;
        int v574;
        v574 = v573 / 32l;
        auto v575 = cooperative_groups::labeled_partition(v572,v574);
        Closure1 v576{};
        float v577; int v578;
        Tuple1 tmp2 = cooperative_groups::reduce(v575, Tuple1{v559, v560}, v576);
        v577 = tmp2.v0; v578 = tmp2.v1;
        assert("Tensor range check" && 0 <= v511 && v511 < 64l);
        v9[v558] = v578;
        v511 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v579;
    v579 = threadIdx.x;
    bool v580;
    v580 = 0l <= v579;
    bool v581;
    v581 = v580 == false;
    if (v581){
        assert("The index needs to be zero or positive." && v580);
    } else {
    }
    int v583;
    v583 = v579 % 32l;
    int v584;
    v584 = v579 / 32l;
    bool v585;
    v585 = v584 < 1l;
    bool v586;
    v586 = v585 == false;
    if (v586){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v585);
    } else {
    }
    assert("Tensor range check" && 0 <= v584 && v584 < 1l);
    assert("Tensor range check" && 0 <= v583 && v583 < 32l);
    int v588;
    v588 = 4l * v583;
    int v589;
    v589 = 128l * v584;
    int v590;
    v590 = v589 + v588;
    assert("Tensor range check" && 0 <= v584 && v584 < 1l);
    assert("Tensor range check" && 0 <= v583 && v583 < 32l);
    int v591;
    v591 = 0l;
    while (while_method_2(v591)){
        assert("Tensor range check" && 0 <= v591 && v591 < 64l);
        int v593;
        v593 = 128l * v591;
        int v594;
        v594 = v593 + v590;
        float v595[4l];
        int v596[4l];
        int v597;
        v597 = 0l;
        while (while_method_3(v597)){
            assert("Tensor range check" && 0 <= v597 && v597 < 1l);
            int v599;
            v599 = 4l * v597;
            assert("Tensor range check" && 0 <= v597 && v597 < 1l);
            int v600;
            v600 = 128l * v597;
            int v601;
            v601 = v600 + v594;
            int4* v602;
            v602 = reinterpret_cast<int4*>(v1 + v601);
            int4* v603;
            v603 = reinterpret_cast<int4*>(v595 + v599);
            assert("Pointer alignment check" && (unsigned long long)(v602) % 4l == 0 && (unsigned long long)(v603) % 4l == 0);
            *v603 = *v602;
            v597 += 1l ;
        }
        int v604;
        v604 = 0l;
        while (while_method_3(v604)){
            int v606;
            v606 = 0l;
            while (while_method_1(v606)){
                bool v608;
                v608 = 0l <= v606;
                bool v610;
                if (v608){
                    bool v609;
                    v609 = v606 < 4l;
                    v610 = v609;
                } else {
                    v610 = false;
                }
                bool v611;
                v611 = v610 == false;
                if (v611){
                    assert("The indices should be inside the range of the dimension." && v610);
                } else {
                }
                bool v613;
                v613 = 0l <= v583;
                bool v615;
                if (v613){
                    bool v614;
                    v614 = v583 < 32l;
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
                int v618;
                v618 = v583 * 4l;
                int v619;
                v619 = v606 + v618;
                bool v620;
                v620 = 0l <= v604;
                bool v622;
                if (v620){
                    bool v621;
                    v621 = v604 < 1l;
                    v622 = v621;
                } else {
                    v622 = false;
                }
                bool v623;
                v623 = v622 == false;
                if (v623){
                    assert("The indices should be inside the range of the dimension." && v622);
                } else {
                }
                int v625;
                v625 = v604 * 128l;
                int v626;
                v626 = v619 + v625;
                assert("Tensor range check" && 0 <= v604 && v604 < 1l);
                assert("Tensor range check" && 0 <= v606 && v606 < 4l);
                int v627;
                v627 = 4l * v604;
                int v628;
                v628 = v627 + v606;
                v596[v628] = v626;
                v606 += 1l ;
            }
            v604 += 1l ;
        }
        bool v629;
        v629 = 0l <= v584;
        bool v630;
        v630 = v629 && v585;
        bool v631;
        v631 = v630 == false;
        if (v631){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v630);
        } else {
        }
        bool v633;
        v633 = 0l <= v591;
        bool v635;
        if (v633){
            bool v634;
            v634 = v591 < 64l;
            v635 = v634;
        } else {
            v635 = false;
        }
        bool v636;
        v636 = v635 == false;
        if (v636){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v635);
        } else {
        }
        int v638;
        v638 = v591 + v584;
        float v639;
        v639 = 0.0f;
        int v640;
        v640 = 0l;
        while (while_method_3(v640)){
            int v642;
            v642 = 0l;
            while (while_method_1(v642)){
                assert("Tensor range check" && 0 <= v640 && v640 < 1l);
                assert("Tensor range check" && 0 <= v642 && v642 < 4l);
                int v644;
                v644 = 4l * v640;
                int v645;
                v645 = v644 + v642;
                float v646;
                v646 = v595[v645];
                float v647;
                v647 = v639 + v646;
                v639 = v647;
                v642 += 1l ;
            }
            v640 += 1l ;
        }
        auto v648 = cooperative_groups::coalesced_threads();
        int v649;
        v649 = threadIdx.x;
        int v650;
        v650 = v649 / 32l;
        auto v651 = cooperative_groups::labeled_partition(v648,v650);
        float v652;
        v652 = cooperative_groups::reduce(v651, v639, v64);
        float v653;
        v653 = v652 / 128.0f;
        float v654[4l];
        int v655;
        v655 = 0l;
        while (while_method_3(v655)){
            int v657;
            v657 = 0l;
            while (while_method_1(v657)){
                assert("Tensor range check" && 0 <= v655 && v655 < 1l);
                assert("Tensor range check" && 0 <= v657 && v657 < 4l);
                int v659;
                v659 = 4l * v655;
                int v660;
                v660 = v659 + v657;
                float v661;
                v661 = v595[v660];
                float v662;
                v662 = v661 - v653;
                float v663;
                v663 = exp(v662);
                assert("Tensor range check" && 0 <= v655 && v655 < 1l);
                assert("Tensor range check" && 0 <= v657 && v657 < 4l);
                v654[v660] = v663;
                v657 += 1l ;
            }
            v655 += 1l ;
        }
        float v664;
        v664 = 0.0f;
        int v665;
        v665 = 0l;
        while (while_method_3(v665)){
            int v667;
            v667 = 0l;
            while (while_method_1(v667)){
                assert("Tensor range check" && 0 <= v665 && v665 < 1l);
                assert("Tensor range check" && 0 <= v667 && v667 < 4l);
                int v669;
                v669 = 4l * v665;
                int v670;
                v670 = v669 + v667;
                float v671;
                v671 = v654[v670];
                float v672;
                v672 = v664 + v671;
                v664 = v672;
                v667 += 1l ;
            }
            v665 += 1l ;
        }
        auto v673 = cooperative_groups::coalesced_threads();
        int v674;
        v674 = threadIdx.x;
        int v675;
        v675 = v674 / 32l;
        auto v676 = cooperative_groups::labeled_partition(v673,v675);
        float v677;
        v677 = cooperative_groups::reduce(v676, v664, v64);
        float v678[4l];
        int v679;
        v679 = 0l;
        while (while_method_3(v679)){
            int v681;
            v681 = 0l;
            while (while_method_1(v681)){
                assert("Tensor range check" && 0 <= v679 && v679 < 1l);
                assert("Tensor range check" && 0 <= v681 && v681 < 4l);
                int v683;
                v683 = 4l * v679;
                int v684;
                v684 = v683 + v681;
                float v685;
                v685 = v654[v684];
                float v686;
                v686 = v685 / v677;
                assert("Tensor range check" && 0 <= v679 && v679 < 1l);
                assert("Tensor range check" && 0 <= v681 && v681 < 4l);
                v678[v684] = v686;
                v681 += 1l ;
            }
            v679 += 1l ;
        }
        float v687[4l];
        float v688;
        v688 = 0.0f;
        int v689;
        v689 = 0l;
        while (while_method_3(v689)){
            assert("Tensor range check" && 0 <= v689 && v689 < 1l);
            int v691;
            v691 = 4l * v689;
            assert("Tensor range check" && 0 <= v689 && v689 < 1l);
            int v692; float v693;
            Tuple0 tmp3 = Tuple0{0l, 0.0f};
            v692 = tmp3.v0; v693 = tmp3.v1;
            while (while_method_1(v692)){
                assert("Tensor range check" && 0 <= v692 && v692 < 4l);
                int v695;
                v695 = v692 + v691;
                float v696;
                v696 = v678[v695];
                float v697;
                v697 = v693 + v696;
                v693 = v697;
                v692 += 1l ;
            }
            auto v698 = cooperative_groups::coalesced_threads();
            int v699;
            v699 = threadIdx.x;
            int v700;
            v700 = v699 / 32l;
            auto v701 = cooperative_groups::labeled_partition(v698,v700);
            Closure2 v702{};
            float v703;
            v703 = cooperative_groups::inclusive_scan(v701, v693, v702);
            float v704;
            v704 = v701.shfl_up(v703,1);
            bool v705;
            v705 = v701.thread_rank() == 0;
            float v706;
            if (v705){
                v706 = 0.0f;
            } else {
                v706 = v704;
            }
            float v707;
            v707 = v701.shfl(v703,v701.num_threads()-1);
            float v708;
            v708 = v688 + v706;
            int v709; float v710;
            Tuple0 tmp4 = Tuple0{0l, v708};
            v709 = tmp4.v0; v710 = tmp4.v1;
            while (while_method_1(v709)){
                assert("Tensor range check" && 0 <= v709 && v709 < 4l);
                int v712;
                v712 = v709 + v691;
                float v713;
                v713 = v678[v712];
                float v714;
                v714 = v710 + v713;
                assert("Tensor range check" && 0 <= v709 && v709 < 4l);
                v687[v712] = v714;
                v710 = v714;
                v709 += 1l ;
            }
            float v715;
            v715 = v688 + v707;
            v688 = v715;
            v689 += 1l ;
        }
        assert("Tensor range check" && 0 <= v591 && v591 < 64l);
        int v716;
        v716 = 0l;
        while (while_method_3(v716)){
            assert("Tensor range check" && 0 <= v716 && v716 < 1l);
            int v718;
            v718 = 128l * v716;
            int v719;
            v719 = v718 + v594;
            assert("Tensor range check" && 0 <= v716 && v716 < 1l);
            int v720;
            v720 = 4l * v716;
            int4* v721;
            v721 = reinterpret_cast<int4*>(v678 + v720);
            int4* v722;
            v722 = reinterpret_cast<int4*>(v6 + v719);
            assert("Pointer alignment check" && (unsigned long long)(v721) % 4l == 0 && (unsigned long long)(v722) % 4l == 0);
            *v722 = *v721;
            int4* v723;
            v723 = reinterpret_cast<int4*>(v687 + v720);
            int4* v724;
            v724 = reinterpret_cast<int4*>(v7 + v719);
            assert("Pointer alignment check" && (unsigned long long)(v723) % 4l == 0 && (unsigned long long)(v724) % 4l == 0);
            *v724 = *v723;
            v716 += 1l ;
        }
        v591 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v725;
    v725 = threadIdx.x;
    bool v726;
    v726 = 0l <= v725;
    bool v727;
    v727 = v726 == false;
    if (v727){
        assert("The index needs to be zero or positive." && v726);
    } else {
    }
    int v729;
    v729 = v725 % 32l;
    int v730;
    v730 = v725 / 32l;
    bool v731;
    v731 = v730 < 1l;
    bool v732;
    v732 = v731 == false;
    if (v732){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v731);
    } else {
    }
    assert("Tensor range check" && 0 <= v730 && v730 < 1l);
    assert("Tensor range check" && 0 <= v729 && v729 < 32l);
    int v734;
    v734 = 4l * v729;
    int v735;
    v735 = 128l * v730;
    int v736;
    v736 = v735 + v734;
    assert("Tensor range check" && 0 <= v730 && v730 < 1l);
    assert("Tensor range check" && 0 <= v729 && v729 < 32l);
    int v737;
    v737 = 0l;
    while (while_method_2(v737)){
        assert("Tensor range check" && 0 <= v737 && v737 < 64l);
        int v739;
        v739 = 128l * v737;
        int v740;
        v740 = v739 + v736;
        int v741[4l];
        int v742[4l];
        int v743;
        v743 = 0l;
        while (while_method_3(v743)){
            assert("Tensor range check" && 0 <= v743 && v743 < 1l);
            int v745;
            v745 = 4l * v743;
            assert("Tensor range check" && 0 <= v743 && v743 < 1l);
            int v746;
            v746 = 128l * v743;
            int v747;
            v747 = v746 + v740;
            int4* v748;
            v748 = reinterpret_cast<int4*>(v0 + v747);
            int4* v749;
            v749 = reinterpret_cast<int4*>(v741 + v745);
            assert("Pointer alignment check" && (unsigned long long)(v748) % 4l == 0 && (unsigned long long)(v749) % 4l == 0);
            *v749 = *v748;
            v743 += 1l ;
        }
        int v750;
        v750 = 0l;
        while (while_method_3(v750)){
            int v752;
            v752 = 0l;
            while (while_method_1(v752)){
                bool v754;
                v754 = 0l <= v752;
                bool v756;
                if (v754){
                    bool v755;
                    v755 = v752 < 4l;
                    v756 = v755;
                } else {
                    v756 = false;
                }
                bool v757;
                v757 = v756 == false;
                if (v757){
                    assert("The indices should be inside the range of the dimension." && v756);
                } else {
                }
                bool v759;
                v759 = 0l <= v729;
                bool v761;
                if (v759){
                    bool v760;
                    v760 = v729 < 32l;
                    v761 = v760;
                } else {
                    v761 = false;
                }
                bool v762;
                v762 = v761 == false;
                if (v762){
                    assert("The indices should be inside the range of the dimension." && v761);
                } else {
                }
                int v764;
                v764 = v729 * 4l;
                int v765;
                v765 = v752 + v764;
                bool v766;
                v766 = 0l <= v750;
                bool v768;
                if (v766){
                    bool v767;
                    v767 = v750 < 1l;
                    v768 = v767;
                } else {
                    v768 = false;
                }
                bool v769;
                v769 = v768 == false;
                if (v769){
                    assert("The indices should be inside the range of the dimension." && v768);
                } else {
                }
                int v771;
                v771 = v750 * 128l;
                int v772;
                v772 = v765 + v771;
                assert("Tensor range check" && 0 <= v750 && v750 < 1l);
                assert("Tensor range check" && 0 <= v752 && v752 < 4l);
                int v773;
                v773 = 4l * v750;
                int v774;
                v774 = v773 + v752;
                v742[v774] = v772;
                v752 += 1l ;
            }
            v750 += 1l ;
        }
        bool v775;
        v775 = 0l <= v730;
        bool v776;
        v776 = v775 && v731;
        bool v777;
        v777 = v776 == false;
        if (v777){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v776);
        } else {
        }
        bool v779;
        v779 = 0l <= v737;
        bool v781;
        if (v779){
            bool v780;
            v780 = v737 < 64l;
            v781 = v780;
        } else {
            v781 = false;
        }
        bool v782;
        v782 = v781 == false;
        if (v782){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v781);
        } else {
        }
        int v784;
        v784 = v737 + v730;
        int v785[4l];
        int v786;
        v786 = 0l;
        int v787;
        v787 = 0l;
        while (while_method_3(v787)){
            assert("Tensor range check" && 0 <= v787 && v787 < 1l);
            int v789;
            v789 = 4l * v787;
            assert("Tensor range check" && 0 <= v787 && v787 < 1l);
            int v790; int v791;
            Tuple2 tmp5 = Tuple2{0l, 0l};
            v790 = tmp5.v0; v791 = tmp5.v1;
            while (while_method_1(v790)){
                assert("Tensor range check" && 0 <= v790 && v790 < 4l);
                int v793;
                v793 = v790 + v789;
                int v794;
                v794 = v741[v793];
                int v795;
                v795 = v791 + v794;
                v791 = v795;
                v790 += 1l ;
            }
            auto v796 = cooperative_groups::coalesced_threads();
            int v797;
            v797 = threadIdx.x;
            int v798;
            v798 = v797 / 32l;
            auto v799 = cooperative_groups::labeled_partition(v796,v798);
            Closure3 v800{};
            int v801;
            v801 = cooperative_groups::inclusive_scan(v799, v791, v800);
            int v802;
            v802 = v799.shfl_up(v801,1);
            bool v803;
            v803 = v799.thread_rank() == 0;
            int v804;
            if (v803){
                v804 = 0l;
            } else {
                v804 = v802;
            }
            int v805;
            v805 = v799.shfl(v801,v799.num_threads()-1);
            int v806;
            v806 = v786 + v804;
            int v807; int v808;
            Tuple2 tmp6 = Tuple2{0l, v806};
            v807 = tmp6.v0; v808 = tmp6.v1;
            while (while_method_1(v807)){
                assert("Tensor range check" && 0 <= v807 && v807 < 4l);
                int v810;
                v810 = v807 + v789;
                int v811;
                v811 = v741[v810];
                assert("Tensor range check" && 0 <= v807 && v807 < 4l);
                v785[v810] = v808;
                int v812;
                v812 = v808 + v811;
                v808 = v812;
                v807 += 1l ;
            }
            int v813;
            v813 = v786 + v805;
            v786 = v813;
            v787 += 1l ;
        }
        assert("Tensor range check" && 0 <= v737 && v737 < 64l);
        int v814;
        v814 = 0l;
        while (while_method_3(v814)){
            assert("Tensor range check" && 0 <= v814 && v814 < 1l);
            int v816;
            v816 = 128l * v814;
            int v817;
            v817 = v816 + v740;
            assert("Tensor range check" && 0 <= v814 && v814 < 1l);
            int v818;
            v818 = 4l * v814;
            int4* v819;
            v819 = reinterpret_cast<int4*>(v785 + v818);
            int4* v820;
            v820 = reinterpret_cast<int4*>(v13 + v817);
            assert("Pointer alignment check" && (unsigned long long)(v819) % 4l == 0 && (unsigned long long)(v820) % 4l == 0);
            *v820 = *v819;
            v814 += 1l ;
        }
        v737 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v821;
    v821 = threadIdx.x;
    bool v822;
    v822 = 0l <= v821;
    bool v823;
    v823 = v822 == false;
    if (v823){
        assert("The index needs to be zero or positive." && v822);
    } else {
    }
    int v825;
    v825 = v821 % 32l;
    int v826;
    v826 = v821 / 32l;
    bool v827;
    v827 = v826 < 1l;
    bool v828;
    v828 = v827 == false;
    if (v828){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v827);
    } else {
    }
    assert("Tensor range check" && 0 <= v826 && v826 < 1l);
    assert("Tensor range check" && 0 <= v825 && v825 < 32l);
    int v830;
    v830 = 4l * v825;
    int v831;
    v831 = 128l * v826;
    int v832;
    v832 = v831 + v830;
    assert("Tensor range check" && 0 <= v826 && v826 < 1l);
    assert("Tensor range check" && 0 <= v825 && v825 < 32l);
    int v833;
    v833 = 0l;
    while (while_method_2(v833)){
        assert("Tensor range check" && 0 <= v833 && v833 < 64l);
        int v835;
        v835 = 128l * v833;
        int v836;
        v836 = v835 + v832;
        float v837[4l];
        int v838[4l];
        int v839;
        v839 = 0l;
        while (while_method_3(v839)){
            assert("Tensor range check" && 0 <= v839 && v839 < 1l);
            int v841;
            v841 = 4l * v839;
            assert("Tensor range check" && 0 <= v839 && v839 < 1l);
            int v842;
            v842 = 128l * v839;
            int v843;
            v843 = v842 + v836;
            int4* v844;
            v844 = reinterpret_cast<int4*>(v1 + v843);
            int4* v845;
            v845 = reinterpret_cast<int4*>(v837 + v841);
            assert("Pointer alignment check" && (unsigned long long)(v844) % 4l == 0 && (unsigned long long)(v845) % 4l == 0);
            *v845 = *v844;
            v839 += 1l ;
        }
        int v846;
        v846 = 0l;
        while (while_method_3(v846)){
            int v848;
            v848 = 0l;
            while (while_method_1(v848)){
                bool v850;
                v850 = 0l <= v848;
                bool v852;
                if (v850){
                    bool v851;
                    v851 = v848 < 4l;
                    v852 = v851;
                } else {
                    v852 = false;
                }
                bool v853;
                v853 = v852 == false;
                if (v853){
                    assert("The indices should be inside the range of the dimension." && v852);
                } else {
                }
                bool v855;
                v855 = 0l <= v825;
                bool v857;
                if (v855){
                    bool v856;
                    v856 = v825 < 32l;
                    v857 = v856;
                } else {
                    v857 = false;
                }
                bool v858;
                v858 = v857 == false;
                if (v858){
                    assert("The indices should be inside the range of the dimension." && v857);
                } else {
                }
                int v860;
                v860 = v825 * 4l;
                int v861;
                v861 = v848 + v860;
                bool v862;
                v862 = 0l <= v846;
                bool v864;
                if (v862){
                    bool v863;
                    v863 = v846 < 1l;
                    v864 = v863;
                } else {
                    v864 = false;
                }
                bool v865;
                v865 = v864 == false;
                if (v865){
                    assert("The indices should be inside the range of the dimension." && v864);
                } else {
                }
                int v867;
                v867 = v846 * 128l;
                int v868;
                v868 = v861 + v867;
                assert("Tensor range check" && 0 <= v846 && v846 < 1l);
                assert("Tensor range check" && 0 <= v848 && v848 < 4l);
                int v869;
                v869 = 4l * v846;
                int v870;
                v870 = v869 + v848;
                v838[v870] = v868;
                v848 += 1l ;
            }
            v846 += 1l ;
        }
        bool v871;
        v871 = 0l <= v826;
        bool v872;
        v872 = v871 && v827;
        bool v873;
        v873 = v872 == false;
        if (v873){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v872);
        } else {
        }
        bool v875;
        v875 = 0l <= v833;
        bool v877;
        if (v875){
            bool v876;
            v876 = v833 < 64l;
            v877 = v876;
        } else {
            v877 = false;
        }
        bool v878;
        v878 = v877 == false;
        if (v878){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v877);
        } else {
        }
        int v880;
        v880 = v833 + v826;
        bool v881[4l];
        int v882;
        v882 = 0l;
        while (while_method_3(v882)){
            int v884;
            v884 = 0l;
            while (while_method_1(v884)){
                assert("Tensor range check" && 0 <= v882 && v882 < 1l);
                assert("Tensor range check" && 0 <= v884 && v884 < 4l);
                int v886;
                v886 = 4l * v882;
                int v887;
                v887 = v886 + v884;
                float v888;
                v888 = v837[v887];
                int v889;
                v889 = v838[v887];
                bool v890;
                v890 = v889 < 4l;
                assert("Tensor range check" && 0 <= v882 && v882 < 1l);
                assert("Tensor range check" && 0 <= v884 && v884 < 4l);
                v881[v887] = v890;
                v884 += 1l ;
            }
            v882 += 1l ;
        }
        int v891[4l];
        int v892;
        v892 = 0l;
        while (while_method_3(v892)){
            int v894;
            v894 = 0l;
            while (while_method_1(v894)){
                assert("Tensor range check" && 0 <= v892 && v892 < 1l);
                assert("Tensor range check" && 0 <= v894 && v894 < 4l);
                int v896;
                v896 = 4l * v892;
                int v897;
                v897 = v896 + v894;
                bool v898;
                v898 = v881[v897];
                int v899;
                if (v898){
                    v899 = 1l;
                } else {
                    v899 = 0l;
                }
                assert("Tensor range check" && 0 <= v892 && v892 < 1l);
                assert("Tensor range check" && 0 <= v894 && v894 < 4l);
                v891[v897] = v899;
                v894 += 1l ;
            }
            v892 += 1l ;
        }
        int v900;
        v900 = 0l;
        int v901;
        v901 = 0l;
        while (while_method_3(v901)){
            int v903;
            v903 = 0l;
            while (while_method_1(v903)){
                assert("Tensor range check" && 0 <= v901 && v901 < 1l);
                assert("Tensor range check" && 0 <= v903 && v903 < 4l);
                int v905;
                v905 = 4l * v901;
                int v906;
                v906 = v905 + v903;
                int v907;
                v907 = v891[v906];
                int v908;
                v908 = v900 + v907;
                v900 = v908;
                v903 += 1l ;
            }
            v901 += 1l ;
        }
        auto v909 = cooperative_groups::coalesced_threads();
        int v910;
        v910 = threadIdx.x;
        int v911;
        v911 = v910 / 32l;
        auto v912 = cooperative_groups::labeled_partition(v909,v911);
        Closure4 v913{};
        int v914;
        v914 = cooperative_groups::reduce(v912, v900, v913);
        float v915[4l];
        int v916;
        v916 = 0l;
        while (while_method_3(v916)){
            int v918;
            v918 = 0l;
            while (while_method_1(v918)){
                assert("Tensor range check" && 0 <= v916 && v916 < 1l);
                assert("Tensor range check" && 0 <= v918 && v918 < 4l);
                int v920;
                v920 = 4l * v916;
                int v921;
                v921 = v920 + v918;
                float v922;
                v922 = v837[v921];
                bool v923;
                v923 = v881[v921];
                float v924;
                if (v923){
                    v924 = v922;
                } else {
                    v924 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v916 && v916 < 1l);
                assert("Tensor range check" && 0 <= v918 && v918 < 4l);
                v915[v921] = v924;
                v918 += 1l ;
            }
            v916 += 1l ;
        }
        float v925;
        v925 = 0.0f;
        int v926;
        v926 = 0l;
        while (while_method_3(v926)){
            int v928;
            v928 = 0l;
            while (while_method_1(v928)){
                assert("Tensor range check" && 0 <= v926 && v926 < 1l);
                assert("Tensor range check" && 0 <= v928 && v928 < 4l);
                int v930;
                v930 = 4l * v926;
                int v931;
                v931 = v930 + v928;
                float v932;
                v932 = v915[v931];
                float v933;
                v933 = v925 + v932;
                v925 = v933;
                v928 += 1l ;
            }
            v926 += 1l ;
        }
        auto v934 = cooperative_groups::coalesced_threads();
        int v935;
        v935 = threadIdx.x;
        int v936;
        v936 = v935 / 32l;
        auto v937 = cooperative_groups::labeled_partition(v934,v936);
        float v938;
        v938 = cooperative_groups::reduce(v937, v925, v64);
        float v939;
        v939 = (float)v914;
        float v940;
        v940 = v938 / v939;
        float v941[4l];
        int v942;
        v942 = 0l;
        while (while_method_3(v942)){
            int v944;
            v944 = 0l;
            while (while_method_1(v944)){
                assert("Tensor range check" && 0 <= v942 && v942 < 1l);
                assert("Tensor range check" && 0 <= v944 && v944 < 4l);
                int v946;
                v946 = 4l * v942;
                int v947;
                v947 = v946 + v944;
                float v948;
                v948 = v837[v947];
                bool v949;
                v949 = v881[v947];
                float v950;
                if (v949){
                    v950 = v948;
                } else {
                    v950 = -1.0f / 0.0f;
                }
                float v951;
                v951 = v950 - v940;
                float v952;
                v952 = exp(v951);
                assert("Tensor range check" && 0 <= v942 && v942 < 1l);
                assert("Tensor range check" && 0 <= v944 && v944 < 4l);
                v941[v947] = v952;
                v944 += 1l ;
            }
            v942 += 1l ;
        }
        float v953;
        v953 = 0.0f;
        int v954;
        v954 = 0l;
        while (while_method_3(v954)){
            int v956;
            v956 = 0l;
            while (while_method_1(v956)){
                assert("Tensor range check" && 0 <= v954 && v954 < 1l);
                assert("Tensor range check" && 0 <= v956 && v956 < 4l);
                int v958;
                v958 = 4l * v954;
                int v959;
                v959 = v958 + v956;
                float v960;
                v960 = v941[v959];
                float v961;
                v961 = v953 + v960;
                v953 = v961;
                v956 += 1l ;
            }
            v954 += 1l ;
        }
        auto v962 = cooperative_groups::coalesced_threads();
        int v963;
        v963 = threadIdx.x;
        int v964;
        v964 = v963 / 32l;
        auto v965 = cooperative_groups::labeled_partition(v962,v964);
        float v966;
        v966 = cooperative_groups::reduce(v965, v953, v64);
        float v967[4l];
        int v968;
        v968 = 0l;
        while (while_method_3(v968)){
            int v970;
            v970 = 0l;
            while (while_method_1(v970)){
                assert("Tensor range check" && 0 <= v968 && v968 < 1l);
                assert("Tensor range check" && 0 <= v970 && v970 < 4l);
                int v972;
                v972 = 4l * v968;
                int v973;
                v973 = v972 + v970;
                float v974;
                v974 = v941[v973];
                float v975;
                v975 = v974 / v966;
                assert("Tensor range check" && 0 <= v968 && v968 < 1l);
                assert("Tensor range check" && 0 <= v970 && v970 < 4l);
                v967[v973] = v975;
                v970 += 1l ;
            }
            v968 += 1l ;
        }
        assert("Tensor range check" && 0 <= v833 && v833 < 64l);
        int v976;
        v976 = 0l;
        while (while_method_3(v976)){
            assert("Tensor range check" && 0 <= v976 && v976 < 1l);
            int v978;
            v978 = 128l * v976;
            int v979;
            v979 = v978 + v836;
            assert("Tensor range check" && 0 <= v976 && v976 < 1l);
            int v980;
            v980 = 4l * v976;
            int4* v981;
            v981 = reinterpret_cast<int4*>(v967 + v980);
            int4* v982;
            v982 = reinterpret_cast<int4*>(v5 + v979);
            assert("Pointer alignment check" && (unsigned long long)(v981) % 4l == 0 && (unsigned long long)(v982) % 4l == 0);
            *v982 = *v981;
            v976 += 1l ;
        }
        v833 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v983;
    v983 = threadIdx.x;
    unsigned long long v984;
    v984 = (unsigned long long)v983;
    curandStatePhilox4_32_10_t v985;
    curand_init(12344321ull,v984,0ull,&v985);
    int v986;
    v986 = threadIdx.x;
    bool v987;
    v987 = 0l <= v986;
    bool v988;
    v988 = v987 == false;
    if (v988){
        assert("The index needs to be zero or positive." && v987);
    } else {
    }
    int v990;
    v990 = v986 % 32l;
    int v991;
    v991 = v986 / 32l;
    bool v992;
    v992 = v991 < 1l;
    bool v993;
    v993 = v992 == false;
    if (v993){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v992);
    } else {
    }
    assert("Tensor range check" && 0 <= v991 && v991 < 1l);
    assert("Tensor range check" && 0 <= v990 && v990 < 32l);
    int v995;
    v995 = 4l * v990;
    int v996;
    v996 = 128l * v991;
    int v997;
    v997 = v996 + v995;
    assert("Tensor range check" && 0 <= v991 && v991 < 1l);
    assert("Tensor range check" && 0 <= v990 && v990 < 32l);
    assert("Tensor range check" && 0 <= v991 && v991 < 1l);
    int v998;
    v998 = 0l;
    while (while_method_2(v998)){
        assert("Tensor range check" && 0 <= v998 && v998 < 64l);
        int v1000;
        v1000 = 128l * v998;
        int v1001;
        v1001 = v1000 + v997;
        float v1002[4l];
        int v1003[4l];
        int v1004;
        v1004 = 0l;
        while (while_method_3(v1004)){
            assert("Tensor range check" && 0 <= v1004 && v1004 < 1l);
            int v1006;
            v1006 = 4l * v1004;
            assert("Tensor range check" && 0 <= v1004 && v1004 < 1l);
            int v1007;
            v1007 = 128l * v1004;
            int v1008;
            v1008 = v1007 + v1001;
            int4* v1009;
            v1009 = reinterpret_cast<int4*>(v1 + v1008);
            int4* v1010;
            v1010 = reinterpret_cast<int4*>(v1002 + v1006);
            assert("Pointer alignment check" && (unsigned long long)(v1009) % 4l == 0 && (unsigned long long)(v1010) % 4l == 0);
            *v1010 = *v1009;
            v1004 += 1l ;
        }
        int v1011;
        v1011 = 0l;
        while (while_method_3(v1011)){
            int v1013;
            v1013 = 0l;
            while (while_method_1(v1013)){
                bool v1015;
                v1015 = 0l <= v1013;
                bool v1017;
                if (v1015){
                    bool v1016;
                    v1016 = v1013 < 4l;
                    v1017 = v1016;
                } else {
                    v1017 = false;
                }
                bool v1018;
                v1018 = v1017 == false;
                if (v1018){
                    assert("The indices should be inside the range of the dimension." && v1017);
                } else {
                }
                bool v1020;
                v1020 = 0l <= v990;
                bool v1022;
                if (v1020){
                    bool v1021;
                    v1021 = v990 < 32l;
                    v1022 = v1021;
                } else {
                    v1022 = false;
                }
                bool v1023;
                v1023 = v1022 == false;
                if (v1023){
                    assert("The indices should be inside the range of the dimension." && v1022);
                } else {
                }
                int v1025;
                v1025 = v990 * 4l;
                int v1026;
                v1026 = v1013 + v1025;
                bool v1027;
                v1027 = 0l <= v1011;
                bool v1029;
                if (v1027){
                    bool v1028;
                    v1028 = v1011 < 1l;
                    v1029 = v1028;
                } else {
                    v1029 = false;
                }
                bool v1030;
                v1030 = v1029 == false;
                if (v1030){
                    assert("The indices should be inside the range of the dimension." && v1029);
                } else {
                }
                int v1032;
                v1032 = v1011 * 128l;
                int v1033;
                v1033 = v1026 + v1032;
                assert("Tensor range check" && 0 <= v1011 && v1011 < 1l);
                assert("Tensor range check" && 0 <= v1013 && v1013 < 4l);
                int v1034;
                v1034 = 4l * v1011;
                int v1035;
                v1035 = v1034 + v1013;
                v1003[v1035] = v1033;
                v1013 += 1l ;
            }
            v1011 += 1l ;
        }
        bool v1036;
        v1036 = 0l <= v991;
        bool v1037;
        v1037 = v1036 && v992;
        bool v1038;
        v1038 = v1037 == false;
        if (v1038){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1037);
        } else {
        }
        bool v1040;
        v1040 = 0l <= v998;
        bool v1042;
        if (v1040){
            bool v1041;
            v1041 = v998 < 64l;
            v1042 = v1041;
        } else {
            v1042 = false;
        }
        bool v1043;
        v1043 = v1042 == false;
        if (v1043){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1042);
        } else {
        }
        int v1045;
        v1045 = v998 + v991;
        float v1046;
        v1046 = 0.0f;
        int v1047;
        v1047 = 0l;
        while (while_method_3(v1047)){
            int v1049;
            v1049 = 0l;
            while (while_method_1(v1049)){
                assert("Tensor range check" && 0 <= v1047 && v1047 < 1l);
                assert("Tensor range check" && 0 <= v1049 && v1049 < 4l);
                int v1051;
                v1051 = 4l * v1047;
                int v1052;
                v1052 = v1051 + v1049;
                float v1053;
                v1053 = v1002[v1052];
                float v1054;
                v1054 = v1046 + v1053;
                v1046 = v1054;
                v1049 += 1l ;
            }
            v1047 += 1l ;
        }
        auto v1055 = cooperative_groups::coalesced_threads();
        int v1056;
        v1056 = threadIdx.x;
        int v1057;
        v1057 = v1056 / 32l;
        auto v1058 = cooperative_groups::labeled_partition(v1055,v1057);
        float v1059;
        v1059 = cooperative_groups::reduce(v1058, v1046, v64);
        float v1060;
        v1060 = v1059 / 128.0f;
        float v1061[4l];
        int v1062;
        v1062 = 0l;
        while (while_method_3(v1062)){
            int v1064;
            v1064 = 0l;
            while (while_method_1(v1064)){
                assert("Tensor range check" && 0 <= v1062 && v1062 < 1l);
                assert("Tensor range check" && 0 <= v1064 && v1064 < 4l);
                int v1066;
                v1066 = 4l * v1062;
                int v1067;
                v1067 = v1066 + v1064;
                float v1068;
                v1068 = v1002[v1067];
                float v1069;
                v1069 = v1068 - v1060;
                float v1070;
                v1070 = exp(v1069);
                assert("Tensor range check" && 0 <= v1062 && v1062 < 1l);
                assert("Tensor range check" && 0 <= v1064 && v1064 < 4l);
                v1061[v1067] = v1070;
                v1064 += 1l ;
            }
            v1062 += 1l ;
        }
        float v1071;
        v1071 = 0.0f;
        int v1072;
        v1072 = 0l;
        while (while_method_3(v1072)){
            int v1074;
            v1074 = 0l;
            while (while_method_1(v1074)){
                assert("Tensor range check" && 0 <= v1072 && v1072 < 1l);
                assert("Tensor range check" && 0 <= v1074 && v1074 < 4l);
                int v1076;
                v1076 = 4l * v1072;
                int v1077;
                v1077 = v1076 + v1074;
                float v1078;
                v1078 = v1061[v1077];
                float v1079;
                v1079 = v1071 + v1078;
                v1071 = v1079;
                v1074 += 1l ;
            }
            v1072 += 1l ;
        }
        auto v1080 = cooperative_groups::coalesced_threads();
        int v1081;
        v1081 = threadIdx.x;
        int v1082;
        v1082 = v1081 / 32l;
        auto v1083 = cooperative_groups::labeled_partition(v1080,v1082);
        float v1084;
        v1084 = cooperative_groups::reduce(v1083, v1071, v64);
        float v1085[4l];
        int v1086;
        v1086 = 0l;
        while (while_method_3(v1086)){
            int v1088;
            v1088 = 0l;
            while (while_method_1(v1088)){
                assert("Tensor range check" && 0 <= v1086 && v1086 < 1l);
                assert("Tensor range check" && 0 <= v1088 && v1088 < 4l);
                int v1090;
                v1090 = 4l * v1086;
                int v1091;
                v1091 = v1090 + v1088;
                float v1092;
                v1092 = v1061[v1091];
                float v1093;
                v1093 = v1092 / v1084;
                assert("Tensor range check" && 0 <= v1086 && v1086 < 1l);
                assert("Tensor range check" && 0 <= v1088 && v1088 < 4l);
                v1085[v1091] = v1093;
                v1088 += 1l ;
            }
            v1086 += 1l ;
        }
        float v1094[4l];
        float v1095;
        v1095 = 0.0f;
        int v1096;
        v1096 = 0l;
        while (while_method_3(v1096)){
            assert("Tensor range check" && 0 <= v1096 && v1096 < 1l);
            int v1098;
            v1098 = 4l * v1096;
            assert("Tensor range check" && 0 <= v1096 && v1096 < 1l);
            int v1099; float v1100;
            Tuple0 tmp7 = Tuple0{0l, 0.0f};
            v1099 = tmp7.v0; v1100 = tmp7.v1;
            while (while_method_1(v1099)){
                assert("Tensor range check" && 0 <= v1099 && v1099 < 4l);
                int v1102;
                v1102 = v1099 + v1098;
                float v1103;
                v1103 = v1085[v1102];
                float v1104;
                v1104 = v1100 + v1103;
                v1100 = v1104;
                v1099 += 1l ;
            }
            auto v1105 = cooperative_groups::coalesced_threads();
            int v1106;
            v1106 = threadIdx.x;
            int v1107;
            v1107 = v1106 / 32l;
            auto v1108 = cooperative_groups::labeled_partition(v1105,v1107);
            Closure2 v1109{};
            float v1110;
            v1110 = cooperative_groups::inclusive_scan(v1108, v1100, v1109);
            float v1111;
            v1111 = v1108.shfl_up(v1110,1);
            bool v1112;
            v1112 = v1108.thread_rank() == 0;
            float v1113;
            if (v1112){
                v1113 = 0.0f;
            } else {
                v1113 = v1111;
            }
            float v1114;
            v1114 = v1108.shfl(v1110,v1108.num_threads()-1);
            float v1115;
            v1115 = v1095 + v1113;
            int v1116; float v1117;
            Tuple0 tmp8 = Tuple0{0l, v1115};
            v1116 = tmp8.v0; v1117 = tmp8.v1;
            while (while_method_1(v1116)){
                assert("Tensor range check" && 0 <= v1116 && v1116 < 4l);
                int v1119;
                v1119 = v1116 + v1098;
                float v1120;
                v1120 = v1085[v1119];
                float v1121;
                v1121 = v1117 + v1120;
                assert("Tensor range check" && 0 <= v1116 && v1116 < 4l);
                v1094[v1119] = v1121;
                v1117 = v1121;
                v1116 += 1l ;
            }
            float v1122;
            v1122 = v1095 + v1114;
            v1095 = v1122;
            v1096 += 1l ;
        }
        float v1123[4l];
        bool v1124[4l];
        int v1125;
        v1125 = 0l;
        while (while_method_3(v1125)){
            int v1127;
            v1127 = 0l;
            while (while_method_1(v1127)){
                assert("Tensor range check" && 0 <= v1125 && v1125 < 1l);
                assert("Tensor range check" && 0 <= v1127 && v1127 < 4l);
                int v1129;
                v1129 = 4l * v1125;
                int v1130;
                v1130 = v1129 + v1127;
                float v1131;
                v1131 = v1094[v1130];
                float v1132;
                v1132 = v1085[v1130];
                bool v1133;
                v1133 = v1132 > 0.0f;
                assert("Tensor range check" && 0 <= v1125 && v1125 < 1l);
                assert("Tensor range check" && 0 <= v1127 && v1127 < 4l);
                v1123[v1130] = v1131;
                v1124[v1130] = v1133;
                v1127 += 1l ;
            }
            v1125 += 1l ;
        }
        float v1134; bool v1135;
        Tuple3 tmp9 = Tuple3{-1.0f / 0.0f, false};
        v1134 = tmp9.v0; v1135 = tmp9.v1;
        int v1136;
        v1136 = 0l;
        while (while_method_3(v1136)){
            int v1138;
            v1138 = 0l;
            while (while_method_1(v1138)){
                assert("Tensor range check" && 0 <= v1136 && v1136 < 1l);
                assert("Tensor range check" && 0 <= v1138 && v1138 < 4l);
                int v1140;
                v1140 = 4l * v1136;
                int v1141;
                v1141 = v1140 + v1138;
                float v1142;
                v1142 = v1123[v1141];
                bool v1143;
                v1143 = v1124[v1141];
                float v1150; bool v1151;
                if (v1135){
                    if (v1143){
                        bool v1144;
                        v1144 = v1134 >= v1142;
                        float v1145;
                        if (v1144){
                            v1145 = v1134;
                        } else {
                            v1145 = v1142;
                        }
                        v1150 = v1145; v1151 = true;
                    } else {
                        v1150 = v1134; v1151 = v1135;
                    }
                } else {
                    if (v1143){
                        v1150 = v1142; v1151 = v1143;
                    } else {
                        v1150 = v1134; v1151 = v1135;
                    }
                }
                v1134 = v1150;
                v1135 = v1151;
                v1138 += 1l ;
            }
            v1136 += 1l ;
        }
        auto v1152 = cooperative_groups::coalesced_threads();
        int v1153;
        v1153 = threadIdx.x;
        int v1154;
        v1154 = v1153 / 32l;
        auto v1155 = cooperative_groups::labeled_partition(v1152,v1154);
        Closure5 v1156{};
        float v1157; bool v1158;
        Tuple3 tmp10 = cooperative_groups::reduce(v1155, Tuple3{v1134, v1135}, v1156);
        v1157 = tmp10.v0; v1158 = tmp10.v1;
        bool v1159;
        v1159 = v1158 == false;
        if (v1159){
            assert("The local reduce must be true." && v1158);
        } else {
        }
        float v1161[4l];
        int v1162[4l];
        int v1163;
        v1163 = 0l;
        while (while_method_3(v1163)){
            int v1165;
            v1165 = 0l;
            while (while_method_1(v1165)){
                assert("Tensor range check" && 0 <= v1163 && v1163 < 1l);
                assert("Tensor range check" && 0 <= v1165 && v1165 < 4l);
                int v1167;
                v1167 = 4l * v1163;
                int v1168;
                v1168 = v1167 + v1165;
                int v1169;
                v1169 = v1003[v1168];
                float v1170;
                v1170 = curand_uniform(&v985);
                assert("Tensor range check" && 0 <= v1163 && v1163 < 1l);
                assert("Tensor range check" && 0 <= v1165 && v1165 < 4l);
                v1161[v1168] = v1170;
                v1162[v1168] = v1169;
                v1165 += 1l ;
            }
            v1163 += 1l ;
        }
        float v1171; int v1172;
        Tuple1 tmp11 = Tuple1{0.0f, 2147483647l};
        v1171 = tmp11.v0; v1172 = tmp11.v1;
        int v1173;
        v1173 = 0l;
        while (while_method_3(v1173)){
            int v1175;
            v1175 = 0l;
            while (while_method_1(v1175)){
                assert("Tensor range check" && 0 <= v1173 && v1173 < 1l);
                assert("Tensor range check" && 0 <= v1175 && v1175 < 4l);
                int v1177;
                v1177 = 4l * v1173;
                int v1178;
                v1178 = v1177 + v1175;
                float v1179;
                v1179 = v1161[v1178];
                int v1180;
                v1180 = v1162[v1178];
                bool v1181;
                v1181 = v1172 < v1180;
                float v1182; int v1183;
                if (v1181){
                    v1182 = v1171; v1183 = v1172;
                } else {
                    v1182 = v1179; v1183 = v1180;
                }
                v1171 = v1182;
                v1172 = v1183;
                v1175 += 1l ;
            }
            v1173 += 1l ;
        }
        auto v1184 = cooperative_groups::coalesced_threads();
        int v1185;
        v1185 = threadIdx.x;
        int v1186;
        v1186 = v1185 / 32l;
        auto v1187 = cooperative_groups::labeled_partition(v1184,v1186);
        Closure6 v1188{};
        float v1189; int v1190;
        Tuple1 tmp12 = cooperative_groups::reduce(v1187, Tuple1{v1171, v1172}, v1188);
        v1189 = tmp12.v0; v1190 = tmp12.v1;
        float v1191;
        v1191 = v1157 * v1189;
        int v1192[4l];
        bool v1193[4l];
        int v1194;
        v1194 = 0l;
        while (while_method_3(v1194)){
            int v1196;
            v1196 = 0l;
            while (while_method_1(v1196)){
                assert("Tensor range check" && 0 <= v1194 && v1194 < 1l);
                assert("Tensor range check" && 0 <= v1196 && v1196 < 4l);
                int v1198;
                v1198 = 4l * v1194;
                int v1199;
                v1199 = v1198 + v1196;
                float v1200;
                v1200 = v1123[v1199];
                bool v1201;
                v1201 = v1124[v1199];
                int v1202;
                v1202 = v1003[v1199];
                int v1205; bool v1206;
                if (v1201){
                    float v1203;
                    v1203 = v1200 - v1191;
                    bool v1204;
                    v1204 = v1203 >= 0.0f;
                    v1205 = v1202; v1206 = v1204;
                } else {
                    v1205 = 2147483647l; v1206 = false;
                }
                assert("Tensor range check" && 0 <= v1194 && v1194 < 1l);
                assert("Tensor range check" && 0 <= v1196 && v1196 < 4l);
                v1192[v1199] = v1205;
                v1193[v1199] = v1206;
                v1196 += 1l ;
            }
            v1194 += 1l ;
        }
        int v1207; bool v1208;
        Tuple4 tmp13 = Tuple4{2147483647l, false};
        v1207 = tmp13.v0; v1208 = tmp13.v1;
        int v1209;
        v1209 = 0l;
        while (while_method_3(v1209)){
            int v1211;
            v1211 = 0l;
            while (while_method_1(v1211)){
                assert("Tensor range check" && 0 <= v1209 && v1209 < 1l);
                assert("Tensor range check" && 0 <= v1211 && v1211 < 4l);
                int v1213;
                v1213 = 4l * v1209;
                int v1214;
                v1214 = v1213 + v1211;
                int v1215;
                v1215 = v1192[v1214];
                bool v1216;
                v1216 = v1193[v1214];
                int v1223; bool v1224;
                if (v1208){
                    if (v1216){
                        bool v1217;
                        v1217 = v1207 < v1215;
                        int v1218;
                        if (v1217){
                            v1218 = v1207;
                        } else {
                            v1218 = v1215;
                        }
                        v1223 = v1218; v1224 = true;
                    } else {
                        v1223 = v1207; v1224 = v1208;
                    }
                } else {
                    if (v1216){
                        v1223 = v1215; v1224 = v1216;
                    } else {
                        v1223 = v1207; v1224 = v1208;
                    }
                }
                v1207 = v1223;
                v1208 = v1224;
                v1211 += 1l ;
            }
            v1209 += 1l ;
        }
        auto v1225 = cooperative_groups::coalesced_threads();
        int v1226;
        v1226 = threadIdx.x;
        int v1227;
        v1227 = v1226 / 32l;
        auto v1228 = cooperative_groups::labeled_partition(v1225,v1227);
        Closure7 v1229{};
        int v1230; bool v1231;
        Tuple4 tmp14 = cooperative_groups::reduce(v1228, Tuple4{v1207, v1208}, v1229);
        v1230 = tmp14.v0; v1231 = tmp14.v1;
        bool v1232;
        v1232 = v1231 == false;
        if (v1232){
            assert("The local reduce must be true." && v1231);
        } else {
        }
        assert("Tensor range check" && 0 <= v998 && v998 < 64l);
        int v1234;
        v1234 = 0l;
        while (while_method_3(v1234)){
            assert("Tensor range check" && 0 <= v1234 && v1234 < 1l);
            int v1236;
            v1236 = 128l * v1234;
            int v1237;
            v1237 = v1236 + v1001;
            assert("Tensor range check" && 0 <= v1234 && v1234 < 1l);
            int v1238;
            v1238 = 4l * v1234;
            int4* v1239;
            v1239 = reinterpret_cast<int4*>(v1085 + v1238);
            int4* v1240;
            v1240 = reinterpret_cast<int4*>(v14 + v1237);
            assert("Pointer alignment check" && (unsigned long long)(v1239) % 4l == 0 && (unsigned long long)(v1240) % 4l == 0);
            *v1240 = *v1239;
            v1234 += 1l ;
        }
        assert("Tensor range check" && 0 <= v998 && v998 < 64l);
        v15[v1045] = v1230;
        v998 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1241;
    v1241 = threadIdx.x;
    unsigned long long v1242;
    v1242 = (unsigned long long)v1241;
    curandStatePhilox4_32_10_t v1243;
    curand_init(12344321ull,v1242,0ull,&v1243);
    int v1244;
    v1244 = threadIdx.x;
    bool v1245;
    v1245 = 0l <= v1244;
    bool v1246;
    v1246 = v1245 == false;
    if (v1246){
        assert("The index needs to be zero or positive." && v1245);
    } else {
    }
    int v1248;
    v1248 = v1244 % 32l;
    int v1249;
    v1249 = v1244 / 32l;
    bool v1250;
    v1250 = v1249 < 1l;
    bool v1251;
    v1251 = v1250 == false;
    if (v1251){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1250);
    } else {
    }
    assert("Tensor range check" && 0 <= v1249 && v1249 < 1l);
    assert("Tensor range check" && 0 <= v1248 && v1248 < 32l);
    int v1253;
    v1253 = 4l * v1248;
    int v1254;
    v1254 = 128l * v1249;
    int v1255;
    v1255 = v1254 + v1253;
    assert("Tensor range check" && 0 <= v1249 && v1249 < 1l);
    assert("Tensor range check" && 0 <= v1248 && v1248 < 32l);
    assert("Tensor range check" && 0 <= v1249 && v1249 < 1l);
    int v1256;
    v1256 = 0l;
    while (while_method_2(v1256)){
        assert("Tensor range check" && 0 <= v1256 && v1256 < 64l);
        int v1258;
        v1258 = 128l * v1256;
        int v1259;
        v1259 = v1258 + v1255;
        float v1260[4l];
        int v1261[4l];
        int v1262;
        v1262 = 0l;
        while (while_method_3(v1262)){
            assert("Tensor range check" && 0 <= v1262 && v1262 < 1l);
            int v1264;
            v1264 = 4l * v1262;
            assert("Tensor range check" && 0 <= v1262 && v1262 < 1l);
            int v1265;
            v1265 = 128l * v1262;
            int v1266;
            v1266 = v1265 + v1259;
            int4* v1267;
            v1267 = reinterpret_cast<int4*>(v1 + v1266);
            int4* v1268;
            v1268 = reinterpret_cast<int4*>(v1260 + v1264);
            assert("Pointer alignment check" && (unsigned long long)(v1267) % 4l == 0 && (unsigned long long)(v1268) % 4l == 0);
            *v1268 = *v1267;
            v1262 += 1l ;
        }
        int v1269;
        v1269 = 0l;
        while (while_method_3(v1269)){
            int v1271;
            v1271 = 0l;
            while (while_method_1(v1271)){
                bool v1273;
                v1273 = 0l <= v1271;
                bool v1275;
                if (v1273){
                    bool v1274;
                    v1274 = v1271 < 4l;
                    v1275 = v1274;
                } else {
                    v1275 = false;
                }
                bool v1276;
                v1276 = v1275 == false;
                if (v1276){
                    assert("The indices should be inside the range of the dimension." && v1275);
                } else {
                }
                bool v1278;
                v1278 = 0l <= v1248;
                bool v1280;
                if (v1278){
                    bool v1279;
                    v1279 = v1248 < 32l;
                    v1280 = v1279;
                } else {
                    v1280 = false;
                }
                bool v1281;
                v1281 = v1280 == false;
                if (v1281){
                    assert("The indices should be inside the range of the dimension." && v1280);
                } else {
                }
                int v1283;
                v1283 = v1248 * 4l;
                int v1284;
                v1284 = v1271 + v1283;
                bool v1285;
                v1285 = 0l <= v1269;
                bool v1287;
                if (v1285){
                    bool v1286;
                    v1286 = v1269 < 1l;
                    v1287 = v1286;
                } else {
                    v1287 = false;
                }
                bool v1288;
                v1288 = v1287 == false;
                if (v1288){
                    assert("The indices should be inside the range of the dimension." && v1287);
                } else {
                }
                int v1290;
                v1290 = v1269 * 128l;
                int v1291;
                v1291 = v1284 + v1290;
                assert("Tensor range check" && 0 <= v1269 && v1269 < 1l);
                assert("Tensor range check" && 0 <= v1271 && v1271 < 4l);
                int v1292;
                v1292 = 4l * v1269;
                int v1293;
                v1293 = v1292 + v1271;
                v1261[v1293] = v1291;
                v1271 += 1l ;
            }
            v1269 += 1l ;
        }
        bool v1294;
        v1294 = 0l <= v1249;
        bool v1295;
        v1295 = v1294 && v1250;
        bool v1296;
        v1296 = v1295 == false;
        if (v1296){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1295);
        } else {
        }
        bool v1298;
        v1298 = 0l <= v1256;
        bool v1300;
        if (v1298){
            bool v1299;
            v1299 = v1256 < 64l;
            v1300 = v1299;
        } else {
            v1300 = false;
        }
        bool v1301;
        v1301 = v1300 == false;
        if (v1301){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1300);
        } else {
        }
        int v1303;
        v1303 = v1256 + v1249;
        bool v1304[4l];
        int v1305;
        v1305 = 0l;
        while (while_method_3(v1305)){
            int v1307;
            v1307 = 0l;
            while (while_method_1(v1307)){
                assert("Tensor range check" && 0 <= v1305 && v1305 < 1l);
                assert("Tensor range check" && 0 <= v1307 && v1307 < 4l);
                int v1309;
                v1309 = 4l * v1305;
                int v1310;
                v1310 = v1309 + v1307;
                float v1311;
                v1311 = v1260[v1310];
                int v1312;
                v1312 = v1261[v1310];
                bool v1313;
                v1313 = v1312 < 11l;
                assert("Tensor range check" && 0 <= v1305 && v1305 < 1l);
                assert("Tensor range check" && 0 <= v1307 && v1307 < 4l);
                v1304[v1310] = v1313;
                v1307 += 1l ;
            }
            v1305 += 1l ;
        }
        int v1314[4l];
        int v1315;
        v1315 = 0l;
        while (while_method_3(v1315)){
            int v1317;
            v1317 = 0l;
            while (while_method_1(v1317)){
                assert("Tensor range check" && 0 <= v1315 && v1315 < 1l);
                assert("Tensor range check" && 0 <= v1317 && v1317 < 4l);
                int v1319;
                v1319 = 4l * v1315;
                int v1320;
                v1320 = v1319 + v1317;
                bool v1321;
                v1321 = v1304[v1320];
                int v1322;
                if (v1321){
                    v1322 = 1l;
                } else {
                    v1322 = 0l;
                }
                assert("Tensor range check" && 0 <= v1315 && v1315 < 1l);
                assert("Tensor range check" && 0 <= v1317 && v1317 < 4l);
                v1314[v1320] = v1322;
                v1317 += 1l ;
            }
            v1315 += 1l ;
        }
        int v1323;
        v1323 = 0l;
        int v1324;
        v1324 = 0l;
        while (while_method_3(v1324)){
            int v1326;
            v1326 = 0l;
            while (while_method_1(v1326)){
                assert("Tensor range check" && 0 <= v1324 && v1324 < 1l);
                assert("Tensor range check" && 0 <= v1326 && v1326 < 4l);
                int v1328;
                v1328 = 4l * v1324;
                int v1329;
                v1329 = v1328 + v1326;
                int v1330;
                v1330 = v1314[v1329];
                int v1331;
                v1331 = v1323 + v1330;
                v1323 = v1331;
                v1326 += 1l ;
            }
            v1324 += 1l ;
        }
        auto v1332 = cooperative_groups::coalesced_threads();
        int v1333;
        v1333 = threadIdx.x;
        int v1334;
        v1334 = v1333 / 32l;
        auto v1335 = cooperative_groups::labeled_partition(v1332,v1334);
        Closure4 v1336{};
        int v1337;
        v1337 = cooperative_groups::reduce(v1335, v1323, v1336);
        float v1338[4l];
        int v1339;
        v1339 = 0l;
        while (while_method_3(v1339)){
            int v1341;
            v1341 = 0l;
            while (while_method_1(v1341)){
                assert("Tensor range check" && 0 <= v1339 && v1339 < 1l);
                assert("Tensor range check" && 0 <= v1341 && v1341 < 4l);
                int v1343;
                v1343 = 4l * v1339;
                int v1344;
                v1344 = v1343 + v1341;
                float v1345;
                v1345 = v1260[v1344];
                bool v1346;
                v1346 = v1304[v1344];
                float v1347;
                if (v1346){
                    v1347 = v1345;
                } else {
                    v1347 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1339 && v1339 < 1l);
                assert("Tensor range check" && 0 <= v1341 && v1341 < 4l);
                v1338[v1344] = v1347;
                v1341 += 1l ;
            }
            v1339 += 1l ;
        }
        float v1348;
        v1348 = 0.0f;
        int v1349;
        v1349 = 0l;
        while (while_method_3(v1349)){
            int v1351;
            v1351 = 0l;
            while (while_method_1(v1351)){
                assert("Tensor range check" && 0 <= v1349 && v1349 < 1l);
                assert("Tensor range check" && 0 <= v1351 && v1351 < 4l);
                int v1353;
                v1353 = 4l * v1349;
                int v1354;
                v1354 = v1353 + v1351;
                float v1355;
                v1355 = v1338[v1354];
                float v1356;
                v1356 = v1348 + v1355;
                v1348 = v1356;
                v1351 += 1l ;
            }
            v1349 += 1l ;
        }
        auto v1357 = cooperative_groups::coalesced_threads();
        int v1358;
        v1358 = threadIdx.x;
        int v1359;
        v1359 = v1358 / 32l;
        auto v1360 = cooperative_groups::labeled_partition(v1357,v1359);
        float v1361;
        v1361 = cooperative_groups::reduce(v1360, v1348, v64);
        float v1362;
        v1362 = (float)v1337;
        float v1363;
        v1363 = v1361 / v1362;
        float v1364[4l];
        int v1365;
        v1365 = 0l;
        while (while_method_3(v1365)){
            int v1367;
            v1367 = 0l;
            while (while_method_1(v1367)){
                assert("Tensor range check" && 0 <= v1365 && v1365 < 1l);
                assert("Tensor range check" && 0 <= v1367 && v1367 < 4l);
                int v1369;
                v1369 = 4l * v1365;
                int v1370;
                v1370 = v1369 + v1367;
                float v1371;
                v1371 = v1260[v1370];
                bool v1372;
                v1372 = v1304[v1370];
                float v1373;
                if (v1372){
                    v1373 = v1371;
                } else {
                    v1373 = -1.0f / 0.0f;
                }
                float v1374;
                v1374 = v1373 - v1363;
                float v1375;
                v1375 = exp(v1374);
                assert("Tensor range check" && 0 <= v1365 && v1365 < 1l);
                assert("Tensor range check" && 0 <= v1367 && v1367 < 4l);
                v1364[v1370] = v1375;
                v1367 += 1l ;
            }
            v1365 += 1l ;
        }
        float v1376;
        v1376 = 0.0f;
        int v1377;
        v1377 = 0l;
        while (while_method_3(v1377)){
            int v1379;
            v1379 = 0l;
            while (while_method_1(v1379)){
                assert("Tensor range check" && 0 <= v1377 && v1377 < 1l);
                assert("Tensor range check" && 0 <= v1379 && v1379 < 4l);
                int v1381;
                v1381 = 4l * v1377;
                int v1382;
                v1382 = v1381 + v1379;
                float v1383;
                v1383 = v1364[v1382];
                float v1384;
                v1384 = v1376 + v1383;
                v1376 = v1384;
                v1379 += 1l ;
            }
            v1377 += 1l ;
        }
        auto v1385 = cooperative_groups::coalesced_threads();
        int v1386;
        v1386 = threadIdx.x;
        int v1387;
        v1387 = v1386 / 32l;
        auto v1388 = cooperative_groups::labeled_partition(v1385,v1387);
        float v1389;
        v1389 = cooperative_groups::reduce(v1388, v1376, v64);
        float v1390[4l];
        int v1391;
        v1391 = 0l;
        while (while_method_3(v1391)){
            int v1393;
            v1393 = 0l;
            while (while_method_1(v1393)){
                assert("Tensor range check" && 0 <= v1391 && v1391 < 1l);
                assert("Tensor range check" && 0 <= v1393 && v1393 < 4l);
                int v1395;
                v1395 = 4l * v1391;
                int v1396;
                v1396 = v1395 + v1393;
                float v1397;
                v1397 = v1364[v1396];
                float v1398;
                v1398 = v1397 / v1389;
                assert("Tensor range check" && 0 <= v1391 && v1391 < 1l);
                assert("Tensor range check" && 0 <= v1393 && v1393 < 4l);
                v1390[v1396] = v1398;
                v1393 += 1l ;
            }
            v1391 += 1l ;
        }
        float v1399[4l];
        float v1400;
        v1400 = 0.0f;
        int v1401;
        v1401 = 0l;
        while (while_method_3(v1401)){
            assert("Tensor range check" && 0 <= v1401 && v1401 < 1l);
            int v1403;
            v1403 = 4l * v1401;
            assert("Tensor range check" && 0 <= v1401 && v1401 < 1l);
            int v1404; float v1405;
            Tuple0 tmp15 = Tuple0{0l, 0.0f};
            v1404 = tmp15.v0; v1405 = tmp15.v1;
            while (while_method_1(v1404)){
                assert("Tensor range check" && 0 <= v1404 && v1404 < 4l);
                int v1407;
                v1407 = v1404 + v1403;
                float v1408;
                v1408 = v1390[v1407];
                float v1409;
                v1409 = v1405 + v1408;
                v1405 = v1409;
                v1404 += 1l ;
            }
            auto v1410 = cooperative_groups::coalesced_threads();
            int v1411;
            v1411 = threadIdx.x;
            int v1412;
            v1412 = v1411 / 32l;
            auto v1413 = cooperative_groups::labeled_partition(v1410,v1412);
            Closure2 v1414{};
            float v1415;
            v1415 = cooperative_groups::inclusive_scan(v1413, v1405, v1414);
            float v1416;
            v1416 = v1413.shfl_up(v1415,1);
            bool v1417;
            v1417 = v1413.thread_rank() == 0;
            float v1418;
            if (v1417){
                v1418 = 0.0f;
            } else {
                v1418 = v1416;
            }
            float v1419;
            v1419 = v1413.shfl(v1415,v1413.num_threads()-1);
            float v1420;
            v1420 = v1400 + v1418;
            int v1421; float v1422;
            Tuple0 tmp16 = Tuple0{0l, v1420};
            v1421 = tmp16.v0; v1422 = tmp16.v1;
            while (while_method_1(v1421)){
                assert("Tensor range check" && 0 <= v1421 && v1421 < 4l);
                int v1424;
                v1424 = v1421 + v1403;
                float v1425;
                v1425 = v1390[v1424];
                float v1426;
                v1426 = v1422 + v1425;
                assert("Tensor range check" && 0 <= v1421 && v1421 < 4l);
                v1399[v1424] = v1426;
                v1422 = v1426;
                v1421 += 1l ;
            }
            float v1427;
            v1427 = v1400 + v1419;
            v1400 = v1427;
            v1401 += 1l ;
        }
        float v1428[4l];
        bool v1429[4l];
        int v1430;
        v1430 = 0l;
        while (while_method_3(v1430)){
            int v1432;
            v1432 = 0l;
            while (while_method_1(v1432)){
                assert("Tensor range check" && 0 <= v1430 && v1430 < 1l);
                assert("Tensor range check" && 0 <= v1432 && v1432 < 4l);
                int v1434;
                v1434 = 4l * v1430;
                int v1435;
                v1435 = v1434 + v1432;
                float v1436;
                v1436 = v1399[v1435];
                float v1437;
                v1437 = v1390[v1435];
                bool v1438;
                v1438 = v1437 > 0.0f;
                assert("Tensor range check" && 0 <= v1430 && v1430 < 1l);
                assert("Tensor range check" && 0 <= v1432 && v1432 < 4l);
                v1428[v1435] = v1436;
                v1429[v1435] = v1438;
                v1432 += 1l ;
            }
            v1430 += 1l ;
        }
        float v1439; bool v1440;
        Tuple3 tmp17 = Tuple3{-1.0f / 0.0f, false};
        v1439 = tmp17.v0; v1440 = tmp17.v1;
        int v1441;
        v1441 = 0l;
        while (while_method_3(v1441)){
            int v1443;
            v1443 = 0l;
            while (while_method_1(v1443)){
                assert("Tensor range check" && 0 <= v1441 && v1441 < 1l);
                assert("Tensor range check" && 0 <= v1443 && v1443 < 4l);
                int v1445;
                v1445 = 4l * v1441;
                int v1446;
                v1446 = v1445 + v1443;
                float v1447;
                v1447 = v1428[v1446];
                bool v1448;
                v1448 = v1429[v1446];
                float v1455; bool v1456;
                if (v1440){
                    if (v1448){
                        bool v1449;
                        v1449 = v1439 >= v1447;
                        float v1450;
                        if (v1449){
                            v1450 = v1439;
                        } else {
                            v1450 = v1447;
                        }
                        v1455 = v1450; v1456 = true;
                    } else {
                        v1455 = v1439; v1456 = v1440;
                    }
                } else {
                    if (v1448){
                        v1455 = v1447; v1456 = v1448;
                    } else {
                        v1455 = v1439; v1456 = v1440;
                    }
                }
                v1439 = v1455;
                v1440 = v1456;
                v1443 += 1l ;
            }
            v1441 += 1l ;
        }
        auto v1457 = cooperative_groups::coalesced_threads();
        int v1458;
        v1458 = threadIdx.x;
        int v1459;
        v1459 = v1458 / 32l;
        auto v1460 = cooperative_groups::labeled_partition(v1457,v1459);
        Closure5 v1461{};
        float v1462; bool v1463;
        Tuple3 tmp18 = cooperative_groups::reduce(v1460, Tuple3{v1439, v1440}, v1461);
        v1462 = tmp18.v0; v1463 = tmp18.v1;
        bool v1464;
        v1464 = v1463 == false;
        if (v1464){
            assert("The local reduce must be true." && v1463);
        } else {
        }
        float v1466[4l];
        int v1467[4l];
        int v1468;
        v1468 = 0l;
        while (while_method_3(v1468)){
            int v1470;
            v1470 = 0l;
            while (while_method_1(v1470)){
                assert("Tensor range check" && 0 <= v1468 && v1468 < 1l);
                assert("Tensor range check" && 0 <= v1470 && v1470 < 4l);
                int v1472;
                v1472 = 4l * v1468;
                int v1473;
                v1473 = v1472 + v1470;
                int v1474;
                v1474 = v1261[v1473];
                float v1475;
                v1475 = curand_uniform(&v1243);
                assert("Tensor range check" && 0 <= v1468 && v1468 < 1l);
                assert("Tensor range check" && 0 <= v1470 && v1470 < 4l);
                v1466[v1473] = v1475;
                v1467[v1473] = v1474;
                v1470 += 1l ;
            }
            v1468 += 1l ;
        }
        float v1476; int v1477;
        Tuple1 tmp19 = Tuple1{0.0f, 2147483647l};
        v1476 = tmp19.v0; v1477 = tmp19.v1;
        int v1478;
        v1478 = 0l;
        while (while_method_3(v1478)){
            int v1480;
            v1480 = 0l;
            while (while_method_1(v1480)){
                assert("Tensor range check" && 0 <= v1478 && v1478 < 1l);
                assert("Tensor range check" && 0 <= v1480 && v1480 < 4l);
                int v1482;
                v1482 = 4l * v1478;
                int v1483;
                v1483 = v1482 + v1480;
                float v1484;
                v1484 = v1466[v1483];
                int v1485;
                v1485 = v1467[v1483];
                bool v1486;
                v1486 = v1477 < v1485;
                float v1487; int v1488;
                if (v1486){
                    v1487 = v1476; v1488 = v1477;
                } else {
                    v1487 = v1484; v1488 = v1485;
                }
                v1476 = v1487;
                v1477 = v1488;
                v1480 += 1l ;
            }
            v1478 += 1l ;
        }
        auto v1489 = cooperative_groups::coalesced_threads();
        int v1490;
        v1490 = threadIdx.x;
        int v1491;
        v1491 = v1490 / 32l;
        auto v1492 = cooperative_groups::labeled_partition(v1489,v1491);
        Closure6 v1493{};
        float v1494; int v1495;
        Tuple1 tmp20 = cooperative_groups::reduce(v1492, Tuple1{v1476, v1477}, v1493);
        v1494 = tmp20.v0; v1495 = tmp20.v1;
        float v1496;
        v1496 = v1462 * v1494;
        int v1497[4l];
        bool v1498[4l];
        int v1499;
        v1499 = 0l;
        while (while_method_3(v1499)){
            int v1501;
            v1501 = 0l;
            while (while_method_1(v1501)){
                assert("Tensor range check" && 0 <= v1499 && v1499 < 1l);
                assert("Tensor range check" && 0 <= v1501 && v1501 < 4l);
                int v1503;
                v1503 = 4l * v1499;
                int v1504;
                v1504 = v1503 + v1501;
                float v1505;
                v1505 = v1428[v1504];
                bool v1506;
                v1506 = v1429[v1504];
                int v1507;
                v1507 = v1261[v1504];
                int v1510; bool v1511;
                if (v1506){
                    float v1508;
                    v1508 = v1505 - v1496;
                    bool v1509;
                    v1509 = v1508 >= 0.0f;
                    v1510 = v1507; v1511 = v1509;
                } else {
                    v1510 = 2147483647l; v1511 = false;
                }
                assert("Tensor range check" && 0 <= v1499 && v1499 < 1l);
                assert("Tensor range check" && 0 <= v1501 && v1501 < 4l);
                v1497[v1504] = v1510;
                v1498[v1504] = v1511;
                v1501 += 1l ;
            }
            v1499 += 1l ;
        }
        int v1512; bool v1513;
        Tuple4 tmp21 = Tuple4{2147483647l, false};
        v1512 = tmp21.v0; v1513 = tmp21.v1;
        int v1514;
        v1514 = 0l;
        while (while_method_3(v1514)){
            int v1516;
            v1516 = 0l;
            while (while_method_1(v1516)){
                assert("Tensor range check" && 0 <= v1514 && v1514 < 1l);
                assert("Tensor range check" && 0 <= v1516 && v1516 < 4l);
                int v1518;
                v1518 = 4l * v1514;
                int v1519;
                v1519 = v1518 + v1516;
                int v1520;
                v1520 = v1497[v1519];
                bool v1521;
                v1521 = v1498[v1519];
                int v1528; bool v1529;
                if (v1513){
                    if (v1521){
                        bool v1522;
                        v1522 = v1512 < v1520;
                        int v1523;
                        if (v1522){
                            v1523 = v1512;
                        } else {
                            v1523 = v1520;
                        }
                        v1528 = v1523; v1529 = true;
                    } else {
                        v1528 = v1512; v1529 = v1513;
                    }
                } else {
                    if (v1521){
                        v1528 = v1520; v1529 = v1521;
                    } else {
                        v1528 = v1512; v1529 = v1513;
                    }
                }
                v1512 = v1528;
                v1513 = v1529;
                v1516 += 1l ;
            }
            v1514 += 1l ;
        }
        auto v1530 = cooperative_groups::coalesced_threads();
        int v1531;
        v1531 = threadIdx.x;
        int v1532;
        v1532 = v1531 / 32l;
        auto v1533 = cooperative_groups::labeled_partition(v1530,v1532);
        Closure7 v1534{};
        int v1535; bool v1536;
        Tuple4 tmp22 = cooperative_groups::reduce(v1533, Tuple4{v1512, v1513}, v1534);
        v1535 = tmp22.v0; v1536 = tmp22.v1;
        bool v1537;
        v1537 = v1536 == false;
        if (v1537){
            assert("The local reduce must be true." && v1536);
        } else {
        }
        assert("Tensor range check" && 0 <= v1256 && v1256 < 64l);
        int v1539;
        v1539 = 0l;
        while (while_method_3(v1539)){
            assert("Tensor range check" && 0 <= v1539 && v1539 < 1l);
            int v1541;
            v1541 = 128l * v1539;
            int v1542;
            v1542 = v1541 + v1259;
            assert("Tensor range check" && 0 <= v1539 && v1539 < 1l);
            int v1543;
            v1543 = 4l * v1539;
            int4* v1544;
            v1544 = reinterpret_cast<int4*>(v1390 + v1543);
            int4* v1545;
            v1545 = reinterpret_cast<int4*>(v14 + v1542);
            assert("Pointer alignment check" && (unsigned long long)(v1544) % 4l == 0 && (unsigned long long)(v1545) % 4l == 0);
            *v1545 = *v1544;
            v1539 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1256 && v1256 < 64l);
        v15[v1303] = v1535;
        v1256 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
extern "C" __global__ void entry1(int * v0, float * v1, float * v2, int * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int * v9, int * v10, int * v11, int * v12, int * v13, float * v14, int * v15) {
    int v16;
    v16 = threadIdx.x;
    int v17;
    v17 = v16;
    while (while_method_0(v17)){
        bool v19;
        v19 = 0l <= v17;
        bool v20;
        v20 = v19 == false;
        if (v20){
            assert("The index needs to be zero or positive." && v19);
        } else {
        }
        int v22;
        v22 = v17 % 16l;
        int v23;
        v23 = v17 / 16l;
        bool v24;
        v24 = v23 < 128l;
        bool v25;
        v25 = v24 == false;
        if (v25){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
        } else {
        }
        assert("Tensor range check" && 0 <= v23 && v23 < 128l);
        assert("Tensor range check" && 0 <= v22 && v22 < 16l);
        int v27;
        v27 = 4l * v22;
        int v28;
        v28 = 64l * v23;
        int v29;
        v29 = v28 + v27;
        assert("Tensor range check" && 0 <= v23 && v23 < 128l);
        assert("Tensor range check" && 0 <= v22 && v22 < 16l);
        float v30[4l];
        float v31[4l];
        int4* v32;
        v32 = reinterpret_cast<int4*>(v1 + v29);
        int4* v33;
        v33 = reinterpret_cast<int4*>(v30 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v32) % 4l == 0 && (unsigned long long)(v33) % 4l == 0);
        *v33 = *v32;
        // Pushing the loop unrolling to: 0
        int v34;
        v34 = 0l;
        #pragma unroll
        while (while_method_1(v34)){
            assert("Tensor range check" && 0 <= v34 && v34 < 4l);
            float v36;
            v36 = v30[v34];
            float v37;
            v37 = 1.0f + v36;
            assert("Tensor range check" && 0 <= v34 && v34 < 4l);
            v31[v34] = v37;
            v34 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v38;
        v38 = reinterpret_cast<int4*>(v31 + 0l);
        int4* v39;
        v39 = reinterpret_cast<int4*>(v1 + v29);
        assert("Pointer alignment check" && (unsigned long long)(v38) % 4l == 0 && (unsigned long long)(v39) % 4l == 0);
        *v39 = *v38;
        v17 += 32l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float v40;
    v40 = 0.0f;
    int v41;
    v41 = threadIdx.x;
    int v42;
    v42 = v41;
    while (while_method_0(v42)){
        bool v44;
        v44 = 0l <= v42;
        bool v45;
        v45 = v44 == false;
        if (v45){
            assert("The index needs to be zero or positive." && v44);
        } else {
        }
        int v47;
        v47 = v42 % 16l;
        int v48;
        v48 = v42 / 16l;
        bool v49;
        v49 = v48 < 128l;
        bool v50;
        v50 = v49 == false;
        if (v50){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v49);
        } else {
        }
        assert("Tensor range check" && 0 <= v48 && v48 < 128l);
        assert("Tensor range check" && 0 <= v47 && v47 < 16l);
        int v52;
        v52 = 4l * v47;
        int v53;
        v53 = 64l * v48;
        int v54;
        v54 = v53 + v52;
        float v55[4l];
        int4* v56;
        v56 = reinterpret_cast<int4*>(v1 + v54);
        int4* v57;
        v57 = reinterpret_cast<int4*>(v55 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v56) % 4l == 0 && (unsigned long long)(v57) % 4l == 0);
        *v57 = *v56;
        int v58; float v59;
        Tuple0 tmp23 = Tuple0{0l, v40};
        v58 = tmp23.v0; v59 = tmp23.v1;
        while (while_method_1(v58)){
            assert("Tensor range check" && 0 <= v58 && v58 < 4l);
            float v61;
            v61 = v55[v58];
            float v62;
            v62 = v59 + v61;
            v59 = v62;
            v58 += 1l ;
        }
        v40 = v59;
        v42 += 32l ;
    }
    auto v63 = cooperative_groups::coalesced_threads();
    Closure0 v64{};
    float v65;
    v65 = cooperative_groups::reduce(v63, v40, v64);
    int v66;
    v66 = threadIdx.x;
    int v67;
    v67 = v66 / 32l;
    __shared__ float v68[1l];
    assert("Tensor range check" && 0 <= v67 && v67 < 1l);
    v68[v67] = v65;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v69;
    v69 = threadIdx.x;
    int v70;
    v70 = v69 % 32l;
    bool v71;
    v71 = v67 == 0l;
    bool v73;
    if (v71){
        bool v72;
        v72 = v70 < 1l;
        v73 = v72;
    } else {
        v73 = false;
    }
    if (v73){
        auto v74 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v70 && v70 < 1l);
        float v75;
        v75 = v68[v70];
        float v76;
        v76 = cooperative_groups::reduce(v74, v75, v64);
        v2[0l] = v76;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v77;
    v77 = threadIdx.x;
    bool v78;
    v78 = 0l <= v77;
    bool v79;
    v79 = v78 == false;
    if (v79){
        assert("The index needs to be zero or positive." && v78);
    } else {
    }
    int v81;
    v81 = v77 % 16l;
    int v82;
    v82 = v77 / 16l;
    bool v83;
    v83 = v82 < 2l;
    bool v84;
    v84 = v83 == false;
    if (v84){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v83);
    } else {
    }
    assert("Tensor range check" && 0 <= v82 && v82 < 2l);
    assert("Tensor range check" && 0 <= v81 && v81 < 16l);
    int v86;
    v86 = 4l * v81;
    int v87;
    v87 = 64l * v82;
    int v88;
    v88 = v87 + v86;
    assert("Tensor range check" && 0 <= v82 && v82 < 2l);
    assert("Tensor range check" && 0 <= v81 && v81 < 16l);
    int v89;
    v89 = 0l;
    while (while_method_2(v89)){
        assert("Tensor range check" && 0 <= v89 && v89 < 64l);
        int v91;
        v91 = 128l * v89;
        int v92;
        v92 = v91 + v88;
        int v93[4l];
        int v94[4l];
        int v95;
        v95 = 0l;
        while (while_method_3(v95)){
            assert("Tensor range check" && 0 <= v95 && v95 < 1l);
            int v97;
            v97 = 4l * v95;
            assert("Tensor range check" && 0 <= v95 && v95 < 1l);
            int v98;
            v98 = 64l * v95;
            int v99;
            v99 = v98 + v92;
            int4* v100;
            v100 = reinterpret_cast<int4*>(v0 + v99);
            int4* v101;
            v101 = reinterpret_cast<int4*>(v93 + v97);
            assert("Pointer alignment check" && (unsigned long long)(v100) % 4l == 0 && (unsigned long long)(v101) % 4l == 0);
            *v101 = *v100;
            v95 += 1l ;
        }
        int v102;
        v102 = 0l;
        while (while_method_3(v102)){
            int v104;
            v104 = 0l;
            while (while_method_1(v104)){
                bool v106;
                v106 = 0l <= v104;
                bool v108;
                if (v106){
                    bool v107;
                    v107 = v104 < 4l;
                    v108 = v107;
                } else {
                    v108 = false;
                }
                bool v109;
                v109 = v108 == false;
                if (v109){
                    assert("The indices should be inside the range of the dimension." && v108);
                } else {
                }
                bool v111;
                v111 = 0l <= v81;
                bool v113;
                if (v111){
                    bool v112;
                    v112 = v81 < 16l;
                    v113 = v112;
                } else {
                    v113 = false;
                }
                bool v114;
                v114 = v113 == false;
                if (v114){
                    assert("The indices should be inside the range of the dimension." && v113);
                } else {
                }
                int v116;
                v116 = v81 * 4l;
                int v117;
                v117 = v104 + v116;
                bool v118;
                v118 = 0l <= v102;
                bool v120;
                if (v118){
                    bool v119;
                    v119 = v102 < 1l;
                    v120 = v119;
                } else {
                    v120 = false;
                }
                bool v121;
                v121 = v120 == false;
                if (v121){
                    assert("The indices should be inside the range of the dimension." && v120);
                } else {
                }
                int v123;
                v123 = v102 * 64l;
                int v124;
                v124 = v117 + v123;
                assert("Tensor range check" && 0 <= v102 && v102 < 1l);
                assert("Tensor range check" && 0 <= v104 && v104 < 4l);
                int v125;
                v125 = 4l * v102;
                int v126;
                v126 = v125 + v104;
                v94[v126] = v124;
                v104 += 1l ;
            }
            v102 += 1l ;
        }
        bool v127;
        v127 = 0l <= v82;
        bool v128;
        v128 = v127 && v83;
        bool v129;
        v129 = v128 == false;
        if (v129){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v128);
        } else {
        }
        bool v131;
        v131 = 0l <= v89;
        bool v133;
        if (v131){
            bool v132;
            v132 = v89 < 64l;
            v133 = v132;
        } else {
            v133 = false;
        }
        bool v134;
        v134 = v133 == false;
        if (v134){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v133);
        } else {
        }
        int v136;
        v136 = v89 * 2l;
        int v137;
        v137 = v136 + v82;
        assert("Tensor range check" && 0 <= v89 && v89 < 64l);
        int v138;
        v138 = 0l;
        while (while_method_3(v138)){
            assert("Tensor range check" && 0 <= v138 && v138 < 1l);
            int v140;
            v140 = 64l * v138;
            int v141;
            v141 = v140 + v92;
            assert("Tensor range check" && 0 <= v138 && v138 < 1l);
            int v142;
            v142 = 4l * v138;
            int4* v143;
            v143 = reinterpret_cast<int4*>(v93 + v142);
            int4* v144;
            v144 = reinterpret_cast<int4*>(v3 + v141);
            assert("Pointer alignment check" && (unsigned long long)(v143) % 4l == 0 && (unsigned long long)(v144) % 4l == 0);
            *v144 = *v143;
            v138 += 1l ;
        }
        v89 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v145;
    v145 = threadIdx.x;
    bool v146;
    v146 = 0l <= v145;
    bool v147;
    v147 = v146 == false;
    if (v147){
        assert("The index needs to be zero or positive." && v146);
    } else {
    }
    int v149;
    v149 = v145 % 16l;
    int v150;
    v150 = v145 / 16l;
    bool v151;
    v151 = v150 < 2l;
    bool v152;
    v152 = v151 == false;
    if (v152){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v151);
    } else {
    }
    assert("Tensor range check" && 0 <= v150 && v150 < 2l);
    assert("Tensor range check" && 0 <= v149 && v149 < 16l);
    int v154;
    v154 = 4l * v149;
    int v155;
    v155 = 64l * v150;
    int v156;
    v156 = v155 + v154;
    assert("Tensor range check" && 0 <= v150 && v150 < 2l);
    assert("Tensor range check" && 0 <= v149 && v149 < 16l);
    int v157;
    v157 = 0l;
    while (while_method_2(v157)){
        assert("Tensor range check" && 0 <= v157 && v157 < 64l);
        int v159;
        v159 = 128l * v157;
        int v160;
        v160 = v159 + v156;
        float v161[4l];
        int v162[4l];
        int v163;
        v163 = 0l;
        while (while_method_3(v163)){
            assert("Tensor range check" && 0 <= v163 && v163 < 1l);
            int v165;
            v165 = 4l * v163;
            assert("Tensor range check" && 0 <= v163 && v163 < 1l);
            int v166;
            v166 = 64l * v163;
            int v167;
            v167 = v166 + v160;
            int4* v168;
            v168 = reinterpret_cast<int4*>(v1 + v167);
            int4* v169;
            v169 = reinterpret_cast<int4*>(v161 + v165);
            assert("Pointer alignment check" && (unsigned long long)(v168) % 4l == 0 && (unsigned long long)(v169) % 4l == 0);
            *v169 = *v168;
            v163 += 1l ;
        }
        int v170;
        v170 = 0l;
        while (while_method_3(v170)){
            int v172;
            v172 = 0l;
            while (while_method_1(v172)){
                bool v174;
                v174 = 0l <= v172;
                bool v176;
                if (v174){
                    bool v175;
                    v175 = v172 < 4l;
                    v176 = v175;
                } else {
                    v176 = false;
                }
                bool v177;
                v177 = v176 == false;
                if (v177){
                    assert("The indices should be inside the range of the dimension." && v176);
                } else {
                }
                bool v179;
                v179 = 0l <= v149;
                bool v181;
                if (v179){
                    bool v180;
                    v180 = v149 < 16l;
                    v181 = v180;
                } else {
                    v181 = false;
                }
                bool v182;
                v182 = v181 == false;
                if (v182){
                    assert("The indices should be inside the range of the dimension." && v181);
                } else {
                }
                int v184;
                v184 = v149 * 4l;
                int v185;
                v185 = v172 + v184;
                bool v186;
                v186 = 0l <= v170;
                bool v188;
                if (v186){
                    bool v187;
                    v187 = v170 < 1l;
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
                int v191;
                v191 = v170 * 64l;
                int v192;
                v192 = v185 + v191;
                assert("Tensor range check" && 0 <= v170 && v170 < 1l);
                assert("Tensor range check" && 0 <= v172 && v172 < 4l);
                int v193;
                v193 = 4l * v170;
                int v194;
                v194 = v193 + v172;
                v162[v194] = v192;
                v172 += 1l ;
            }
            v170 += 1l ;
        }
        bool v195;
        v195 = 0l <= v150;
        bool v196;
        v196 = v195 && v151;
        bool v197;
        v197 = v196 == false;
        if (v197){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v196);
        } else {
        }
        bool v199;
        v199 = 0l <= v157;
        bool v201;
        if (v199){
            bool v200;
            v200 = v157 < 64l;
            v201 = v200;
        } else {
            v201 = false;
        }
        bool v202;
        v202 = v201 == false;
        if (v202){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v201);
        } else {
        }
        int v204;
        v204 = v157 * 2l;
        int v205;
        v205 = v204 + v150;
        int v206[4l];
        int v207[4l];
        int v208;
        v208 = 0l;
        while (while_method_3(v208)){
            int v210;
            v210 = 0l;
            while (while_method_1(v210)){
                assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                assert("Tensor range check" && 0 <= v210 && v210 < 4l);
                int v212;
                v212 = 4l * v208;
                int v213;
                v213 = v212 + v210;
                int v214;
                v214 = v162[v213];
                assert("Tensor range check" && 0 <= v208 && v208 < 1l);
                assert("Tensor range check" && 0 <= v210 && v210 < 4l);
                v206[v213] = v205;
                v207[v213] = v214;
                v210 += 1l ;
            }
            v208 += 1l ;
        }
        assert("Tensor range check" && 0 <= v157 && v157 < 64l);
        int v215;
        v215 = 0l;
        while (while_method_3(v215)){
            assert("Tensor range check" && 0 <= v215 && v215 < 1l);
            int v217;
            v217 = 64l * v215;
            int v218;
            v218 = v217 + v160;
            assert("Tensor range check" && 0 <= v215 && v215 < 1l);
            int v219;
            v219 = 4l * v215;
            int4* v220;
            v220 = reinterpret_cast<int4*>(v206 + v219);
            int4* v221;
            v221 = reinterpret_cast<int4*>(v10 + v218);
            assert("Pointer alignment check" && (unsigned long long)(v220) % 4l == 0 && (unsigned long long)(v221) % 4l == 0);
            *v221 = *v220;
            int4* v222;
            v222 = reinterpret_cast<int4*>(v207 + v219);
            int4* v223;
            v223 = reinterpret_cast<int4*>(v11 + v218);
            assert("Pointer alignment check" && (unsigned long long)(v222) % 4l == 0 && (unsigned long long)(v223) % 4l == 0);
            *v223 = *v222;
            v215 += 1l ;
        }
        v157 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
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
    v228 = v224 % 16l;
    int v229;
    v229 = v224 / 16l;
    bool v230;
    v230 = v229 < 2l;
    bool v231;
    v231 = v230 == false;
    if (v231){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v230);
    } else {
    }
    assert("Tensor range check" && 0 <= v229 && v229 < 2l);
    assert("Tensor range check" && 0 <= v228 && v228 < 16l);
    int v233;
    v233 = 4l * v228;
    int v234;
    v234 = 64l * v229;
    int v235;
    v235 = v234 + v233;
    assert("Tensor range check" && 0 <= v229 && v229 < 2l);
    int v236;
    v236 = 0l;
    while (while_method_2(v236)){
        assert("Tensor range check" && 0 <= v236 && v236 < 64l);
        int v238;
        v238 = 128l * v236;
        int v239;
        v239 = v238 + v235;
        float v240[4l];
        int v241[4l];
        int v242;
        v242 = 0l;
        while (while_method_3(v242)){
            assert("Tensor range check" && 0 <= v242 && v242 < 1l);
            int v244;
            v244 = 4l * v242;
            assert("Tensor range check" && 0 <= v242 && v242 < 1l);
            int v245;
            v245 = 64l * v242;
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
                    v259 = v228 < 16l;
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
                    v266 = v249 < 1l;
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
                v270 = v249 * 64l;
                int v271;
                v271 = v264 + v270;
                assert("Tensor range check" && 0 <= v249 && v249 < 1l);
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
            v279 = v236 < 64l;
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
        v283 = v236 * 2l;
        int v284;
        v284 = v283 + v229;
        assert("Tensor range check" && 0 <= v236 && v236 < 64l);
        int v285;
        v285 = 2l * v236;
        int v286;
        v286 = v285 + v229;
        v12[v286] = v284;
        v236 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v287;
    v287 = threadIdx.x;
    bool v288;
    v288 = 0l <= v287;
    bool v289;
    v289 = v288 == false;
    if (v289){
        assert("The index needs to be zero or positive." && v288);
    } else {
    }
    int v291;
    v291 = v287 % 16l;
    int v292;
    v292 = v287 / 16l;
    bool v293;
    v293 = v292 < 2l;
    bool v294;
    v294 = v293 == false;
    if (v294){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v293);
    } else {
    }
    assert("Tensor range check" && 0 <= v292 && v292 < 2l);
    assert("Tensor range check" && 0 <= v291 && v291 < 16l);
    int v296;
    v296 = 4l * v291;
    int v297;
    v297 = 64l * v292;
    int v298;
    v298 = v297 + v296;
    assert("Tensor range check" && 0 <= v292 && v292 < 2l);
    assert("Tensor range check" && 0 <= v291 && v291 < 16l);
    int v299;
    v299 = 0l;
    while (while_method_2(v299)){
        assert("Tensor range check" && 0 <= v299 && v299 < 64l);
        int v301;
        v301 = 128l * v299;
        int v302;
        v302 = v301 + v298;
        float v303[4l];
        int v304[4l];
        int v305;
        v305 = 0l;
        while (while_method_3(v305)){
            assert("Tensor range check" && 0 <= v305 && v305 < 1l);
            int v307;
            v307 = 4l * v305;
            assert("Tensor range check" && 0 <= v305 && v305 < 1l);
            int v308;
            v308 = 64l * v305;
            int v309;
            v309 = v308 + v302;
            int4* v310;
            v310 = reinterpret_cast<int4*>(v1 + v309);
            int4* v311;
            v311 = reinterpret_cast<int4*>(v303 + v307);
            assert("Pointer alignment check" && (unsigned long long)(v310) % 4l == 0 && (unsigned long long)(v311) % 4l == 0);
            *v311 = *v310;
            v305 += 1l ;
        }
        int v312;
        v312 = 0l;
        while (while_method_3(v312)){
            int v314;
            v314 = 0l;
            while (while_method_1(v314)){
                bool v316;
                v316 = 0l <= v314;
                bool v318;
                if (v316){
                    bool v317;
                    v317 = v314 < 4l;
                    v318 = v317;
                } else {
                    v318 = false;
                }
                bool v319;
                v319 = v318 == false;
                if (v319){
                    assert("The indices should be inside the range of the dimension." && v318);
                } else {
                }
                bool v321;
                v321 = 0l <= v291;
                bool v323;
                if (v321){
                    bool v322;
                    v322 = v291 < 16l;
                    v323 = v322;
                } else {
                    v323 = false;
                }
                bool v324;
                v324 = v323 == false;
                if (v324){
                    assert("The indices should be inside the range of the dimension." && v323);
                } else {
                }
                int v326;
                v326 = v291 * 4l;
                int v327;
                v327 = v314 + v326;
                bool v328;
                v328 = 0l <= v312;
                bool v330;
                if (v328){
                    bool v329;
                    v329 = v312 < 1l;
                    v330 = v329;
                } else {
                    v330 = false;
                }
                bool v331;
                v331 = v330 == false;
                if (v331){
                    assert("The indices should be inside the range of the dimension." && v330);
                } else {
                }
                int v333;
                v333 = v312 * 64l;
                int v334;
                v334 = v327 + v333;
                assert("Tensor range check" && 0 <= v312 && v312 < 1l);
                assert("Tensor range check" && 0 <= v314 && v314 < 4l);
                int v335;
                v335 = 4l * v312;
                int v336;
                v336 = v335 + v314;
                v304[v336] = v334;
                v314 += 1l ;
            }
            v312 += 1l ;
        }
        bool v337;
        v337 = 0l <= v292;
        bool v338;
        v338 = v337 && v293;
        bool v339;
        v339 = v338 == false;
        if (v339){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v338);
        } else {
        }
        bool v341;
        v341 = 0l <= v299;
        bool v343;
        if (v341){
            bool v342;
            v342 = v299 < 64l;
            v343 = v342;
        } else {
            v343 = false;
        }
        bool v344;
        v344 = v343 == false;
        if (v344){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v343);
        } else {
        }
        int v346;
        v346 = v299 * 2l;
        int v347;
        v347 = v346 + v292;
        float v348;
        v348 = 0.0f;
        int v349;
        v349 = 0l;
        while (while_method_3(v349)){
            int v351;
            v351 = 0l;
            while (while_method_1(v351)){
                assert("Tensor range check" && 0 <= v349 && v349 < 1l);
                assert("Tensor range check" && 0 <= v351 && v351 < 4l);
                int v353;
                v353 = 4l * v349;
                int v354;
                v354 = v353 + v351;
                float v355;
                v355 = v303[v354];
                float v356;
                v356 = v348 + v355;
                v348 = v356;
                v351 += 1l ;
            }
            v349 += 1l ;
        }
        auto v357 = cooperative_groups::coalesced_threads();
        int v358;
        v358 = threadIdx.x;
        int v359;
        v359 = v358 / 16l;
        auto v360 = cooperative_groups::labeled_partition(v357,v359);
        float v361;
        v361 = cooperative_groups::reduce(v360, v348, v64);
        float v362;
        v362 = v361 / 64.0f;
        float v363[4l];
        int v364;
        v364 = 0l;
        while (while_method_3(v364)){
            int v366;
            v366 = 0l;
            while (while_method_1(v366)){
                assert("Tensor range check" && 0 <= v364 && v364 < 1l);
                assert("Tensor range check" && 0 <= v366 && v366 < 4l);
                int v368;
                v368 = 4l * v364;
                int v369;
                v369 = v368 + v366;
                float v370;
                v370 = v303[v369];
                float v371;
                v371 = v370 - v362;
                float v372;
                v372 = exp(v371);
                assert("Tensor range check" && 0 <= v364 && v364 < 1l);
                assert("Tensor range check" && 0 <= v366 && v366 < 4l);
                v363[v369] = v372;
                v366 += 1l ;
            }
            v364 += 1l ;
        }
        float v373;
        v373 = 0.0f;
        int v374;
        v374 = 0l;
        while (while_method_3(v374)){
            int v376;
            v376 = 0l;
            while (while_method_1(v376)){
                assert("Tensor range check" && 0 <= v374 && v374 < 1l);
                assert("Tensor range check" && 0 <= v376 && v376 < 4l);
                int v378;
                v378 = 4l * v374;
                int v379;
                v379 = v378 + v376;
                float v380;
                v380 = v363[v379];
                float v381;
                v381 = v373 + v380;
                v373 = v381;
                v376 += 1l ;
            }
            v374 += 1l ;
        }
        auto v382 = cooperative_groups::coalesced_threads();
        int v383;
        v383 = threadIdx.x;
        int v384;
        v384 = v383 / 16l;
        auto v385 = cooperative_groups::labeled_partition(v382,v384);
        float v386;
        v386 = cooperative_groups::reduce(v385, v373, v64);
        float v387[4l];
        int v388;
        v388 = 0l;
        while (while_method_3(v388)){
            int v390;
            v390 = 0l;
            while (while_method_1(v390)){
                assert("Tensor range check" && 0 <= v388 && v388 < 1l);
                assert("Tensor range check" && 0 <= v390 && v390 < 4l);
                int v392;
                v392 = 4l * v388;
                int v393;
                v393 = v392 + v390;
                float v394;
                v394 = v363[v393];
                float v395;
                v395 = v394 / v386;
                assert("Tensor range check" && 0 <= v388 && v388 < 1l);
                assert("Tensor range check" && 0 <= v390 && v390 < 4l);
                v387[v393] = v395;
                v390 += 1l ;
            }
            v388 += 1l ;
        }
        assert("Tensor range check" && 0 <= v299 && v299 < 64l);
        int v396;
        v396 = 0l;
        while (while_method_3(v396)){
            assert("Tensor range check" && 0 <= v396 && v396 < 1l);
            int v398;
            v398 = 64l * v396;
            int v399;
            v399 = v398 + v302;
            assert("Tensor range check" && 0 <= v396 && v396 < 1l);
            int v400;
            v400 = 4l * v396;
            int4* v401;
            v401 = reinterpret_cast<int4*>(v387 + v400);
            int4* v402;
            v402 = reinterpret_cast<int4*>(v4 + v399);
            assert("Pointer alignment check" && (unsigned long long)(v401) % 4l == 0 && (unsigned long long)(v402) % 4l == 0);
            *v402 = *v401;
            v396 += 1l ;
        }
        v299 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v403;
    v403 = threadIdx.x;
    bool v404;
    v404 = 0l <= v403;
    bool v405;
    v405 = v404 == false;
    if (v405){
        assert("The index needs to be zero or positive." && v404);
    } else {
    }
    int v407;
    v407 = v403 % 16l;
    int v408;
    v408 = v403 / 16l;
    bool v409;
    v409 = v408 < 2l;
    bool v410;
    v410 = v409 == false;
    if (v410){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v409);
    } else {
    }
    assert("Tensor range check" && 0 <= v408 && v408 < 2l);
    assert("Tensor range check" && 0 <= v407 && v407 < 16l);
    int v412;
    v412 = 4l * v407;
    int v413;
    v413 = 64l * v408;
    int v414;
    v414 = v413 + v412;
    assert("Tensor range check" && 0 <= v408 && v408 < 2l);
    assert("Tensor range check" && 0 <= v407 && v407 < 16l);
    int v415;
    v415 = 0l;
    while (while_method_2(v415)){
        assert("Tensor range check" && 0 <= v415 && v415 < 64l);
        int v417;
        v417 = 128l * v415;
        int v418;
        v418 = v417 + v414;
        float v419[4l];
        int v420[4l];
        int v421;
        v421 = 0l;
        while (while_method_3(v421)){
            assert("Tensor range check" && 0 <= v421 && v421 < 1l);
            int v423;
            v423 = 4l * v421;
            assert("Tensor range check" && 0 <= v421 && v421 < 1l);
            int v424;
            v424 = 64l * v421;
            int v425;
            v425 = v424 + v418;
            int4* v426;
            v426 = reinterpret_cast<int4*>(v1 + v425);
            int4* v427;
            v427 = reinterpret_cast<int4*>(v419 + v423);
            assert("Pointer alignment check" && (unsigned long long)(v426) % 4l == 0 && (unsigned long long)(v427) % 4l == 0);
            *v427 = *v426;
            v421 += 1l ;
        }
        int v428;
        v428 = 0l;
        while (while_method_3(v428)){
            int v430;
            v430 = 0l;
            while (while_method_1(v430)){
                bool v432;
                v432 = 0l <= v430;
                bool v434;
                if (v432){
                    bool v433;
                    v433 = v430 < 4l;
                    v434 = v433;
                } else {
                    v434 = false;
                }
                bool v435;
                v435 = v434 == false;
                if (v435){
                    assert("The indices should be inside the range of the dimension." && v434);
                } else {
                }
                bool v437;
                v437 = 0l <= v407;
                bool v439;
                if (v437){
                    bool v438;
                    v438 = v407 < 16l;
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
                int v442;
                v442 = v407 * 4l;
                int v443;
                v443 = v430 + v442;
                bool v444;
                v444 = 0l <= v428;
                bool v446;
                if (v444){
                    bool v445;
                    v445 = v428 < 1l;
                    v446 = v445;
                } else {
                    v446 = false;
                }
                bool v447;
                v447 = v446 == false;
                if (v447){
                    assert("The indices should be inside the range of the dimension." && v446);
                } else {
                }
                int v449;
                v449 = v428 * 64l;
                int v450;
                v450 = v443 + v449;
                assert("Tensor range check" && 0 <= v428 && v428 < 1l);
                assert("Tensor range check" && 0 <= v430 && v430 < 4l);
                int v451;
                v451 = 4l * v428;
                int v452;
                v452 = v451 + v430;
                v420[v452] = v450;
                v430 += 1l ;
            }
            v428 += 1l ;
        }
        bool v453;
        v453 = 0l <= v408;
        bool v454;
        v454 = v453 && v409;
        bool v455;
        v455 = v454 == false;
        if (v455){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v454);
        } else {
        }
        bool v457;
        v457 = 0l <= v415;
        bool v459;
        if (v457){
            bool v458;
            v458 = v415 < 64l;
            v459 = v458;
        } else {
            v459 = false;
        }
        bool v460;
        v460 = v459 == false;
        if (v460){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v459);
        } else {
        }
        int v462;
        v462 = v415 * 2l;
        int v463;
        v463 = v462 + v408;
        float v464[4l];
        int v465;
        v465 = 0l;
        while (while_method_3(v465)){
            int v467;
            v467 = 0l;
            while (while_method_1(v467)){
                assert("Tensor range check" && 0 <= v465 && v465 < 1l);
                assert("Tensor range check" && 0 <= v467 && v467 < 4l);
                int v469;
                v469 = 4l * v465;
                int v470;
                v470 = v469 + v467;
                float v471;
                v471 = v419[v470];
                float v472;
                v472 = v471 * v471;
                assert("Tensor range check" && 0 <= v465 && v465 < 1l);
                assert("Tensor range check" && 0 <= v467 && v467 < 4l);
                v464[v470] = v472;
                v467 += 1l ;
            }
            v465 += 1l ;
        }
        float v473;
        v473 = 0.0f;
        int v474;
        v474 = 0l;
        while (while_method_3(v474)){
            int v476;
            v476 = 0l;
            while (while_method_1(v476)){
                assert("Tensor range check" && 0 <= v474 && v474 < 1l);
                assert("Tensor range check" && 0 <= v476 && v476 < 4l);
                int v478;
                v478 = 4l * v474;
                int v479;
                v479 = v478 + v476;
                float v480;
                v480 = v464[v479];
                float v481;
                v481 = v473 + v480;
                v473 = v481;
                v476 += 1l ;
            }
            v474 += 1l ;
        }
        auto v482 = cooperative_groups::coalesced_threads();
        int v483;
        v483 = threadIdx.x;
        int v484;
        v484 = v483 / 16l;
        auto v485 = cooperative_groups::labeled_partition(v482,v484);
        float v486;
        v486 = cooperative_groups::reduce(v485, v473, v64);
        float v487[4l];
        int v488;
        v488 = 0l;
        while (while_method_3(v488)){
            int v490;
            v490 = 0l;
            while (while_method_1(v490)){
                assert("Tensor range check" && 0 <= v488 && v488 < 1l);
                assert("Tensor range check" && 0 <= v490 && v490 < 4l);
                int v492;
                v492 = 4l * v488;
                int v493;
                v493 = v492 + v490;
                float v494;
                v494 = v419[v493];
                bool v495;
                v495 = v486 == 0.0f;
                bool v496;
                v496 = v495 != true;
                float v498;
                if (v496){
                    float v497;
                    v497 = v494 / v486;
                    v498 = v497;
                } else {
                    v498 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v488 && v488 < 1l);
                assert("Tensor range check" && 0 <= v490 && v490 < 4l);
                v487[v493] = v498;
                v490 += 1l ;
            }
            v488 += 1l ;
        }
        assert("Tensor range check" && 0 <= v415 && v415 < 64l);
        int v499;
        v499 = 0l;
        while (while_method_3(v499)){
            assert("Tensor range check" && 0 <= v499 && v499 < 1l);
            int v501;
            v501 = 64l * v499;
            int v502;
            v502 = v501 + v418;
            assert("Tensor range check" && 0 <= v499 && v499 < 1l);
            int v503;
            v503 = 4l * v499;
            int4* v504;
            v504 = reinterpret_cast<int4*>(v487 + v503);
            int4* v505;
            v505 = reinterpret_cast<int4*>(v8 + v502);
            assert("Pointer alignment check" && (unsigned long long)(v504) % 4l == 0 && (unsigned long long)(v505) % 4l == 0);
            *v505 = *v504;
            v499 += 1l ;
        }
        v415 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v506;
    v506 = threadIdx.x;
    bool v507;
    v507 = 0l <= v506;
    bool v508;
    v508 = v507 == false;
    if (v508){
        assert("The index needs to be zero or positive." && v507);
    } else {
    }
    int v510;
    v510 = v506 % 16l;
    int v511;
    v511 = v506 / 16l;
    bool v512;
    v512 = v511 < 2l;
    bool v513;
    v513 = v512 == false;
    if (v513){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v512);
    } else {
    }
    assert("Tensor range check" && 0 <= v511 && v511 < 2l);
    assert("Tensor range check" && 0 <= v510 && v510 < 16l);
    int v515;
    v515 = 4l * v510;
    int v516;
    v516 = 64l * v511;
    int v517;
    v517 = v516 + v515;
    assert("Tensor range check" && 0 <= v511 && v511 < 2l);
    int v518;
    v518 = 0l;
    while (while_method_2(v518)){
        assert("Tensor range check" && 0 <= v518 && v518 < 64l);
        int v520;
        v520 = 128l * v518;
        int v521;
        v521 = v520 + v517;
        float v522[4l];
        int v523[4l];
        int v524;
        v524 = 0l;
        while (while_method_3(v524)){
            assert("Tensor range check" && 0 <= v524 && v524 < 1l);
            int v526;
            v526 = 4l * v524;
            assert("Tensor range check" && 0 <= v524 && v524 < 1l);
            int v527;
            v527 = 64l * v524;
            int v528;
            v528 = v527 + v521;
            int4* v529;
            v529 = reinterpret_cast<int4*>(v1 + v528);
            int4* v530;
            v530 = reinterpret_cast<int4*>(v522 + v526);
            assert("Pointer alignment check" && (unsigned long long)(v529) % 4l == 0 && (unsigned long long)(v530) % 4l == 0);
            *v530 = *v529;
            v524 += 1l ;
        }
        int v531;
        v531 = 0l;
        while (while_method_3(v531)){
            int v533;
            v533 = 0l;
            while (while_method_1(v533)){
                bool v535;
                v535 = 0l <= v533;
                bool v537;
                if (v535){
                    bool v536;
                    v536 = v533 < 4l;
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
                bool v540;
                v540 = 0l <= v510;
                bool v542;
                if (v540){
                    bool v541;
                    v541 = v510 < 16l;
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
                int v545;
                v545 = v510 * 4l;
                int v546;
                v546 = v533 + v545;
                bool v547;
                v547 = 0l <= v531;
                bool v549;
                if (v547){
                    bool v548;
                    v548 = v531 < 1l;
                    v549 = v548;
                } else {
                    v549 = false;
                }
                bool v550;
                v550 = v549 == false;
                if (v550){
                    assert("The indices should be inside the range of the dimension." && v549);
                } else {
                }
                int v552;
                v552 = v531 * 64l;
                int v553;
                v553 = v546 + v552;
                assert("Tensor range check" && 0 <= v531 && v531 < 1l);
                assert("Tensor range check" && 0 <= v533 && v533 < 4l);
                int v554;
                v554 = 4l * v531;
                int v555;
                v555 = v554 + v533;
                v523[v555] = v553;
                v533 += 1l ;
            }
            v531 += 1l ;
        }
        bool v556;
        v556 = 0l <= v511;
        bool v557;
        v557 = v556 && v512;
        bool v558;
        v558 = v557 == false;
        if (v558){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v557);
        } else {
        }
        bool v560;
        v560 = 0l <= v518;
        bool v562;
        if (v560){
            bool v561;
            v561 = v518 < 64l;
            v562 = v561;
        } else {
            v562 = false;
        }
        bool v563;
        v563 = v562 == false;
        if (v563){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v562);
        } else {
        }
        int v565;
        v565 = v518 * 2l;
        int v566;
        v566 = v565 + v511;
        float v567; int v568;
        Tuple1 tmp24 = Tuple1{-1.0f / 0.0f, 0l};
        v567 = tmp24.v0; v568 = tmp24.v1;
        int v569;
        v569 = 0l;
        while (while_method_3(v569)){
            int v571;
            v571 = 0l;
            while (while_method_1(v571)){
                assert("Tensor range check" && 0 <= v569 && v569 < 1l);
                assert("Tensor range check" && 0 <= v571 && v571 < 4l);
                int v573;
                v573 = 4l * v569;
                int v574;
                v574 = v573 + v571;
                float v575;
                v575 = v522[v574];
                int v576;
                v576 = v523[v574];
                bool v577;
                v577 = v567 > v575;
                float v578; int v579;
                if (v577){
                    v578 = v567; v579 = v568;
                } else {
                    v578 = v575; v579 = v576;
                }
                v567 = v578;
                v568 = v579;
                v571 += 1l ;
            }
            v569 += 1l ;
        }
        auto v580 = cooperative_groups::coalesced_threads();
        int v581;
        v581 = threadIdx.x;
        int v582;
        v582 = v581 / 16l;
        auto v583 = cooperative_groups::labeled_partition(v580,v582);
        Closure1 v584{};
        float v585; int v586;
        Tuple1 tmp25 = cooperative_groups::reduce(v583, Tuple1{v567, v568}, v584);
        v585 = tmp25.v0; v586 = tmp25.v1;
        assert("Tensor range check" && 0 <= v518 && v518 < 64l);
        int v587;
        v587 = 2l * v518;
        int v588;
        v588 = v587 + v511;
        v9[v588] = v586;
        v518 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v589;
    v589 = threadIdx.x;
    bool v590;
    v590 = 0l <= v589;
    bool v591;
    v591 = v590 == false;
    if (v591){
        assert("The index needs to be zero or positive." && v590);
    } else {
    }
    int v593;
    v593 = v589 % 16l;
    int v594;
    v594 = v589 / 16l;
    bool v595;
    v595 = v594 < 2l;
    bool v596;
    v596 = v595 == false;
    if (v596){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v595);
    } else {
    }
    assert("Tensor range check" && 0 <= v594 && v594 < 2l);
    assert("Tensor range check" && 0 <= v593 && v593 < 16l);
    int v598;
    v598 = 4l * v593;
    int v599;
    v599 = 64l * v594;
    int v600;
    v600 = v599 + v598;
    assert("Tensor range check" && 0 <= v594 && v594 < 2l);
    assert("Tensor range check" && 0 <= v593 && v593 < 16l);
    int v601;
    v601 = 0l;
    while (while_method_2(v601)){
        assert("Tensor range check" && 0 <= v601 && v601 < 64l);
        int v603;
        v603 = 128l * v601;
        int v604;
        v604 = v603 + v600;
        float v605[4l];
        int v606[4l];
        int v607;
        v607 = 0l;
        while (while_method_3(v607)){
            assert("Tensor range check" && 0 <= v607 && v607 < 1l);
            int v609;
            v609 = 4l * v607;
            assert("Tensor range check" && 0 <= v607 && v607 < 1l);
            int v610;
            v610 = 64l * v607;
            int v611;
            v611 = v610 + v604;
            int4* v612;
            v612 = reinterpret_cast<int4*>(v1 + v611);
            int4* v613;
            v613 = reinterpret_cast<int4*>(v605 + v609);
            assert("Pointer alignment check" && (unsigned long long)(v612) % 4l == 0 && (unsigned long long)(v613) % 4l == 0);
            *v613 = *v612;
            v607 += 1l ;
        }
        int v614;
        v614 = 0l;
        while (while_method_3(v614)){
            int v616;
            v616 = 0l;
            while (while_method_1(v616)){
                bool v618;
                v618 = 0l <= v616;
                bool v620;
                if (v618){
                    bool v619;
                    v619 = v616 < 4l;
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
                bool v623;
                v623 = 0l <= v593;
                bool v625;
                if (v623){
                    bool v624;
                    v624 = v593 < 16l;
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
                int v628;
                v628 = v593 * 4l;
                int v629;
                v629 = v616 + v628;
                bool v630;
                v630 = 0l <= v614;
                bool v632;
                if (v630){
                    bool v631;
                    v631 = v614 < 1l;
                    v632 = v631;
                } else {
                    v632 = false;
                }
                bool v633;
                v633 = v632 == false;
                if (v633){
                    assert("The indices should be inside the range of the dimension." && v632);
                } else {
                }
                int v635;
                v635 = v614 * 64l;
                int v636;
                v636 = v629 + v635;
                assert("Tensor range check" && 0 <= v614 && v614 < 1l);
                assert("Tensor range check" && 0 <= v616 && v616 < 4l);
                int v637;
                v637 = 4l * v614;
                int v638;
                v638 = v637 + v616;
                v606[v638] = v636;
                v616 += 1l ;
            }
            v614 += 1l ;
        }
        bool v639;
        v639 = 0l <= v594;
        bool v640;
        v640 = v639 && v595;
        bool v641;
        v641 = v640 == false;
        if (v641){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v640);
        } else {
        }
        bool v643;
        v643 = 0l <= v601;
        bool v645;
        if (v643){
            bool v644;
            v644 = v601 < 64l;
            v645 = v644;
        } else {
            v645 = false;
        }
        bool v646;
        v646 = v645 == false;
        if (v646){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v645);
        } else {
        }
        int v648;
        v648 = v601 * 2l;
        int v649;
        v649 = v648 + v594;
        float v650;
        v650 = 0.0f;
        int v651;
        v651 = 0l;
        while (while_method_3(v651)){
            int v653;
            v653 = 0l;
            while (while_method_1(v653)){
                assert("Tensor range check" && 0 <= v651 && v651 < 1l);
                assert("Tensor range check" && 0 <= v653 && v653 < 4l);
                int v655;
                v655 = 4l * v651;
                int v656;
                v656 = v655 + v653;
                float v657;
                v657 = v605[v656];
                float v658;
                v658 = v650 + v657;
                v650 = v658;
                v653 += 1l ;
            }
            v651 += 1l ;
        }
        auto v659 = cooperative_groups::coalesced_threads();
        int v660;
        v660 = threadIdx.x;
        int v661;
        v661 = v660 / 16l;
        auto v662 = cooperative_groups::labeled_partition(v659,v661);
        float v663;
        v663 = cooperative_groups::reduce(v662, v650, v64);
        float v664;
        v664 = v663 / 64.0f;
        float v665[4l];
        int v666;
        v666 = 0l;
        while (while_method_3(v666)){
            int v668;
            v668 = 0l;
            while (while_method_1(v668)){
                assert("Tensor range check" && 0 <= v666 && v666 < 1l);
                assert("Tensor range check" && 0 <= v668 && v668 < 4l);
                int v670;
                v670 = 4l * v666;
                int v671;
                v671 = v670 + v668;
                float v672;
                v672 = v605[v671];
                float v673;
                v673 = v672 - v664;
                float v674;
                v674 = exp(v673);
                assert("Tensor range check" && 0 <= v666 && v666 < 1l);
                assert("Tensor range check" && 0 <= v668 && v668 < 4l);
                v665[v671] = v674;
                v668 += 1l ;
            }
            v666 += 1l ;
        }
        float v675;
        v675 = 0.0f;
        int v676;
        v676 = 0l;
        while (while_method_3(v676)){
            int v678;
            v678 = 0l;
            while (while_method_1(v678)){
                assert("Tensor range check" && 0 <= v676 && v676 < 1l);
                assert("Tensor range check" && 0 <= v678 && v678 < 4l);
                int v680;
                v680 = 4l * v676;
                int v681;
                v681 = v680 + v678;
                float v682;
                v682 = v665[v681];
                float v683;
                v683 = v675 + v682;
                v675 = v683;
                v678 += 1l ;
            }
            v676 += 1l ;
        }
        auto v684 = cooperative_groups::coalesced_threads();
        int v685;
        v685 = threadIdx.x;
        int v686;
        v686 = v685 / 16l;
        auto v687 = cooperative_groups::labeled_partition(v684,v686);
        float v688;
        v688 = cooperative_groups::reduce(v687, v675, v64);
        float v689[4l];
        int v690;
        v690 = 0l;
        while (while_method_3(v690)){
            int v692;
            v692 = 0l;
            while (while_method_1(v692)){
                assert("Tensor range check" && 0 <= v690 && v690 < 1l);
                assert("Tensor range check" && 0 <= v692 && v692 < 4l);
                int v694;
                v694 = 4l * v690;
                int v695;
                v695 = v694 + v692;
                float v696;
                v696 = v665[v695];
                float v697;
                v697 = v696 / v688;
                assert("Tensor range check" && 0 <= v690 && v690 < 1l);
                assert("Tensor range check" && 0 <= v692 && v692 < 4l);
                v689[v695] = v697;
                v692 += 1l ;
            }
            v690 += 1l ;
        }
        float v698[4l];
        float v699;
        v699 = 0.0f;
        int v700;
        v700 = 0l;
        while (while_method_3(v700)){
            assert("Tensor range check" && 0 <= v700 && v700 < 1l);
            int v702;
            v702 = 4l * v700;
            assert("Tensor range check" && 0 <= v700 && v700 < 1l);
            int v703; float v704;
            Tuple0 tmp26 = Tuple0{0l, 0.0f};
            v703 = tmp26.v0; v704 = tmp26.v1;
            while (while_method_1(v703)){
                assert("Tensor range check" && 0 <= v703 && v703 < 4l);
                int v706;
                v706 = v703 + v702;
                float v707;
                v707 = v689[v706];
                float v708;
                v708 = v704 + v707;
                v704 = v708;
                v703 += 1l ;
            }
            auto v709 = cooperative_groups::coalesced_threads();
            int v710;
            v710 = threadIdx.x;
            int v711;
            v711 = v710 / 16l;
            auto v712 = cooperative_groups::labeled_partition(v709,v711);
            Closure2 v713{};
            float v714;
            v714 = cooperative_groups::inclusive_scan(v712, v704, v713);
            float v715;
            v715 = v712.shfl_up(v714,1);
            bool v716;
            v716 = v712.thread_rank() == 0;
            float v717;
            if (v716){
                v717 = 0.0f;
            } else {
                v717 = v715;
            }
            float v718;
            v718 = v712.shfl(v714,v712.num_threads()-1);
            float v719;
            v719 = v699 + v717;
            int v720; float v721;
            Tuple0 tmp27 = Tuple0{0l, v719};
            v720 = tmp27.v0; v721 = tmp27.v1;
            while (while_method_1(v720)){
                assert("Tensor range check" && 0 <= v720 && v720 < 4l);
                int v723;
                v723 = v720 + v702;
                float v724;
                v724 = v689[v723];
                float v725;
                v725 = v721 + v724;
                assert("Tensor range check" && 0 <= v720 && v720 < 4l);
                v698[v723] = v725;
                v721 = v725;
                v720 += 1l ;
            }
            float v726;
            v726 = v699 + v718;
            v699 = v726;
            v700 += 1l ;
        }
        assert("Tensor range check" && 0 <= v601 && v601 < 64l);
        int v727;
        v727 = 0l;
        while (while_method_3(v727)){
            assert("Tensor range check" && 0 <= v727 && v727 < 1l);
            int v729;
            v729 = 64l * v727;
            int v730;
            v730 = v729 + v604;
            assert("Tensor range check" && 0 <= v727 && v727 < 1l);
            int v731;
            v731 = 4l * v727;
            int4* v732;
            v732 = reinterpret_cast<int4*>(v689 + v731);
            int4* v733;
            v733 = reinterpret_cast<int4*>(v6 + v730);
            assert("Pointer alignment check" && (unsigned long long)(v732) % 4l == 0 && (unsigned long long)(v733) % 4l == 0);
            *v733 = *v732;
            int4* v734;
            v734 = reinterpret_cast<int4*>(v698 + v731);
            int4* v735;
            v735 = reinterpret_cast<int4*>(v7 + v730);
            assert("Pointer alignment check" && (unsigned long long)(v734) % 4l == 0 && (unsigned long long)(v735) % 4l == 0);
            *v735 = *v734;
            v727 += 1l ;
        }
        v601 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v736;
    v736 = threadIdx.x;
    bool v737;
    v737 = 0l <= v736;
    bool v738;
    v738 = v737 == false;
    if (v738){
        assert("The index needs to be zero or positive." && v737);
    } else {
    }
    int v740;
    v740 = v736 % 16l;
    int v741;
    v741 = v736 / 16l;
    bool v742;
    v742 = v741 < 2l;
    bool v743;
    v743 = v742 == false;
    if (v743){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v742);
    } else {
    }
    assert("Tensor range check" && 0 <= v741 && v741 < 2l);
    assert("Tensor range check" && 0 <= v740 && v740 < 16l);
    int v745;
    v745 = 4l * v740;
    int v746;
    v746 = 64l * v741;
    int v747;
    v747 = v746 + v745;
    assert("Tensor range check" && 0 <= v741 && v741 < 2l);
    assert("Tensor range check" && 0 <= v740 && v740 < 16l);
    int v748;
    v748 = 0l;
    while (while_method_2(v748)){
        assert("Tensor range check" && 0 <= v748 && v748 < 64l);
        int v750;
        v750 = 128l * v748;
        int v751;
        v751 = v750 + v747;
        int v752[4l];
        int v753[4l];
        int v754;
        v754 = 0l;
        while (while_method_3(v754)){
            assert("Tensor range check" && 0 <= v754 && v754 < 1l);
            int v756;
            v756 = 4l * v754;
            assert("Tensor range check" && 0 <= v754 && v754 < 1l);
            int v757;
            v757 = 64l * v754;
            int v758;
            v758 = v757 + v751;
            int4* v759;
            v759 = reinterpret_cast<int4*>(v0 + v758);
            int4* v760;
            v760 = reinterpret_cast<int4*>(v752 + v756);
            assert("Pointer alignment check" && (unsigned long long)(v759) % 4l == 0 && (unsigned long long)(v760) % 4l == 0);
            *v760 = *v759;
            v754 += 1l ;
        }
        int v761;
        v761 = 0l;
        while (while_method_3(v761)){
            int v763;
            v763 = 0l;
            while (while_method_1(v763)){
                bool v765;
                v765 = 0l <= v763;
                bool v767;
                if (v765){
                    bool v766;
                    v766 = v763 < 4l;
                    v767 = v766;
                } else {
                    v767 = false;
                }
                bool v768;
                v768 = v767 == false;
                if (v768){
                    assert("The indices should be inside the range of the dimension." && v767);
                } else {
                }
                bool v770;
                v770 = 0l <= v740;
                bool v772;
                if (v770){
                    bool v771;
                    v771 = v740 < 16l;
                    v772 = v771;
                } else {
                    v772 = false;
                }
                bool v773;
                v773 = v772 == false;
                if (v773){
                    assert("The indices should be inside the range of the dimension." && v772);
                } else {
                }
                int v775;
                v775 = v740 * 4l;
                int v776;
                v776 = v763 + v775;
                bool v777;
                v777 = 0l <= v761;
                bool v779;
                if (v777){
                    bool v778;
                    v778 = v761 < 1l;
                    v779 = v778;
                } else {
                    v779 = false;
                }
                bool v780;
                v780 = v779 == false;
                if (v780){
                    assert("The indices should be inside the range of the dimension." && v779);
                } else {
                }
                int v782;
                v782 = v761 * 64l;
                int v783;
                v783 = v776 + v782;
                assert("Tensor range check" && 0 <= v761 && v761 < 1l);
                assert("Tensor range check" && 0 <= v763 && v763 < 4l);
                int v784;
                v784 = 4l * v761;
                int v785;
                v785 = v784 + v763;
                v753[v785] = v783;
                v763 += 1l ;
            }
            v761 += 1l ;
        }
        bool v786;
        v786 = 0l <= v741;
        bool v787;
        v787 = v786 && v742;
        bool v788;
        v788 = v787 == false;
        if (v788){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v787);
        } else {
        }
        bool v790;
        v790 = 0l <= v748;
        bool v792;
        if (v790){
            bool v791;
            v791 = v748 < 64l;
            v792 = v791;
        } else {
            v792 = false;
        }
        bool v793;
        v793 = v792 == false;
        if (v793){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v792);
        } else {
        }
        int v795;
        v795 = v748 * 2l;
        int v796;
        v796 = v795 + v741;
        int v797[4l];
        int v798;
        v798 = 0l;
        int v799;
        v799 = 0l;
        while (while_method_3(v799)){
            assert("Tensor range check" && 0 <= v799 && v799 < 1l);
            int v801;
            v801 = 4l * v799;
            assert("Tensor range check" && 0 <= v799 && v799 < 1l);
            int v802; int v803;
            Tuple2 tmp28 = Tuple2{0l, 0l};
            v802 = tmp28.v0; v803 = tmp28.v1;
            while (while_method_1(v802)){
                assert("Tensor range check" && 0 <= v802 && v802 < 4l);
                int v805;
                v805 = v802 + v801;
                int v806;
                v806 = v752[v805];
                int v807;
                v807 = v803 + v806;
                v803 = v807;
                v802 += 1l ;
            }
            auto v808 = cooperative_groups::coalesced_threads();
            int v809;
            v809 = threadIdx.x;
            int v810;
            v810 = v809 / 16l;
            auto v811 = cooperative_groups::labeled_partition(v808,v810);
            Closure3 v812{};
            int v813;
            v813 = cooperative_groups::inclusive_scan(v811, v803, v812);
            int v814;
            v814 = v811.shfl_up(v813,1);
            bool v815;
            v815 = v811.thread_rank() == 0;
            int v816;
            if (v815){
                v816 = 0l;
            } else {
                v816 = v814;
            }
            int v817;
            v817 = v811.shfl(v813,v811.num_threads()-1);
            int v818;
            v818 = v798 + v816;
            int v819; int v820;
            Tuple2 tmp29 = Tuple2{0l, v818};
            v819 = tmp29.v0; v820 = tmp29.v1;
            while (while_method_1(v819)){
                assert("Tensor range check" && 0 <= v819 && v819 < 4l);
                int v822;
                v822 = v819 + v801;
                int v823;
                v823 = v752[v822];
                assert("Tensor range check" && 0 <= v819 && v819 < 4l);
                v797[v822] = v820;
                int v824;
                v824 = v820 + v823;
                v820 = v824;
                v819 += 1l ;
            }
            int v825;
            v825 = v798 + v817;
            v798 = v825;
            v799 += 1l ;
        }
        assert("Tensor range check" && 0 <= v748 && v748 < 64l);
        int v826;
        v826 = 0l;
        while (while_method_3(v826)){
            assert("Tensor range check" && 0 <= v826 && v826 < 1l);
            int v828;
            v828 = 64l * v826;
            int v829;
            v829 = v828 + v751;
            assert("Tensor range check" && 0 <= v826 && v826 < 1l);
            int v830;
            v830 = 4l * v826;
            int4* v831;
            v831 = reinterpret_cast<int4*>(v797 + v830);
            int4* v832;
            v832 = reinterpret_cast<int4*>(v13 + v829);
            assert("Pointer alignment check" && (unsigned long long)(v831) % 4l == 0 && (unsigned long long)(v832) % 4l == 0);
            *v832 = *v831;
            v826 += 1l ;
        }
        v748 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v833;
    v833 = threadIdx.x;
    bool v834;
    v834 = 0l <= v833;
    bool v835;
    v835 = v834 == false;
    if (v835){
        assert("The index needs to be zero or positive." && v834);
    } else {
    }
    int v837;
    v837 = v833 % 16l;
    int v838;
    v838 = v833 / 16l;
    bool v839;
    v839 = v838 < 2l;
    bool v840;
    v840 = v839 == false;
    if (v840){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v839);
    } else {
    }
    assert("Tensor range check" && 0 <= v838 && v838 < 2l);
    assert("Tensor range check" && 0 <= v837 && v837 < 16l);
    int v842;
    v842 = 4l * v837;
    int v843;
    v843 = 64l * v838;
    int v844;
    v844 = v843 + v842;
    assert("Tensor range check" && 0 <= v838 && v838 < 2l);
    assert("Tensor range check" && 0 <= v837 && v837 < 16l);
    int v845;
    v845 = 0l;
    while (while_method_2(v845)){
        assert("Tensor range check" && 0 <= v845 && v845 < 64l);
        int v847;
        v847 = 128l * v845;
        int v848;
        v848 = v847 + v844;
        float v849[4l];
        int v850[4l];
        int v851;
        v851 = 0l;
        while (while_method_3(v851)){
            assert("Tensor range check" && 0 <= v851 && v851 < 1l);
            int v853;
            v853 = 4l * v851;
            assert("Tensor range check" && 0 <= v851 && v851 < 1l);
            int v854;
            v854 = 64l * v851;
            int v855;
            v855 = v854 + v848;
            int4* v856;
            v856 = reinterpret_cast<int4*>(v1 + v855);
            int4* v857;
            v857 = reinterpret_cast<int4*>(v849 + v853);
            assert("Pointer alignment check" && (unsigned long long)(v856) % 4l == 0 && (unsigned long long)(v857) % 4l == 0);
            *v857 = *v856;
            v851 += 1l ;
        }
        int v858;
        v858 = 0l;
        while (while_method_3(v858)){
            int v860;
            v860 = 0l;
            while (while_method_1(v860)){
                bool v862;
                v862 = 0l <= v860;
                bool v864;
                if (v862){
                    bool v863;
                    v863 = v860 < 4l;
                    v864 = v863;
                } else {
                    v864 = false;
                }
                bool v865;
                v865 = v864 == false;
                if (v865){
                    assert("The indices should be inside the range of the dimension." && v864);
                } else {
                }
                bool v867;
                v867 = 0l <= v837;
                bool v869;
                if (v867){
                    bool v868;
                    v868 = v837 < 16l;
                    v869 = v868;
                } else {
                    v869 = false;
                }
                bool v870;
                v870 = v869 == false;
                if (v870){
                    assert("The indices should be inside the range of the dimension." && v869);
                } else {
                }
                int v872;
                v872 = v837 * 4l;
                int v873;
                v873 = v860 + v872;
                bool v874;
                v874 = 0l <= v858;
                bool v876;
                if (v874){
                    bool v875;
                    v875 = v858 < 1l;
                    v876 = v875;
                } else {
                    v876 = false;
                }
                bool v877;
                v877 = v876 == false;
                if (v877){
                    assert("The indices should be inside the range of the dimension." && v876);
                } else {
                }
                int v879;
                v879 = v858 * 64l;
                int v880;
                v880 = v873 + v879;
                assert("Tensor range check" && 0 <= v858 && v858 < 1l);
                assert("Tensor range check" && 0 <= v860 && v860 < 4l);
                int v881;
                v881 = 4l * v858;
                int v882;
                v882 = v881 + v860;
                v850[v882] = v880;
                v860 += 1l ;
            }
            v858 += 1l ;
        }
        bool v883;
        v883 = 0l <= v838;
        bool v884;
        v884 = v883 && v839;
        bool v885;
        v885 = v884 == false;
        if (v885){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v884);
        } else {
        }
        bool v887;
        v887 = 0l <= v845;
        bool v889;
        if (v887){
            bool v888;
            v888 = v845 < 64l;
            v889 = v888;
        } else {
            v889 = false;
        }
        bool v890;
        v890 = v889 == false;
        if (v890){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v889);
        } else {
        }
        int v892;
        v892 = v845 * 2l;
        int v893;
        v893 = v892 + v838;
        bool v894[4l];
        int v895;
        v895 = 0l;
        while (while_method_3(v895)){
            int v897;
            v897 = 0l;
            while (while_method_1(v897)){
                assert("Tensor range check" && 0 <= v895 && v895 < 1l);
                assert("Tensor range check" && 0 <= v897 && v897 < 4l);
                int v899;
                v899 = 4l * v895;
                int v900;
                v900 = v899 + v897;
                float v901;
                v901 = v849[v900];
                int v902;
                v902 = v850[v900];
                bool v903;
                v903 = v902 < 4l;
                assert("Tensor range check" && 0 <= v895 && v895 < 1l);
                assert("Tensor range check" && 0 <= v897 && v897 < 4l);
                v894[v900] = v903;
                v897 += 1l ;
            }
            v895 += 1l ;
        }
        int v904[4l];
        int v905;
        v905 = 0l;
        while (while_method_3(v905)){
            int v907;
            v907 = 0l;
            while (while_method_1(v907)){
                assert("Tensor range check" && 0 <= v905 && v905 < 1l);
                assert("Tensor range check" && 0 <= v907 && v907 < 4l);
                int v909;
                v909 = 4l * v905;
                int v910;
                v910 = v909 + v907;
                bool v911;
                v911 = v894[v910];
                int v912;
                if (v911){
                    v912 = 1l;
                } else {
                    v912 = 0l;
                }
                assert("Tensor range check" && 0 <= v905 && v905 < 1l);
                assert("Tensor range check" && 0 <= v907 && v907 < 4l);
                v904[v910] = v912;
                v907 += 1l ;
            }
            v905 += 1l ;
        }
        int v913;
        v913 = 0l;
        int v914;
        v914 = 0l;
        while (while_method_3(v914)){
            int v916;
            v916 = 0l;
            while (while_method_1(v916)){
                assert("Tensor range check" && 0 <= v914 && v914 < 1l);
                assert("Tensor range check" && 0 <= v916 && v916 < 4l);
                int v918;
                v918 = 4l * v914;
                int v919;
                v919 = v918 + v916;
                int v920;
                v920 = v904[v919];
                int v921;
                v921 = v913 + v920;
                v913 = v921;
                v916 += 1l ;
            }
            v914 += 1l ;
        }
        auto v922 = cooperative_groups::coalesced_threads();
        int v923;
        v923 = threadIdx.x;
        int v924;
        v924 = v923 / 16l;
        auto v925 = cooperative_groups::labeled_partition(v922,v924);
        Closure4 v926{};
        int v927;
        v927 = cooperative_groups::reduce(v925, v913, v926);
        float v928[4l];
        int v929;
        v929 = 0l;
        while (while_method_3(v929)){
            int v931;
            v931 = 0l;
            while (while_method_1(v931)){
                assert("Tensor range check" && 0 <= v929 && v929 < 1l);
                assert("Tensor range check" && 0 <= v931 && v931 < 4l);
                int v933;
                v933 = 4l * v929;
                int v934;
                v934 = v933 + v931;
                float v935;
                v935 = v849[v934];
                bool v936;
                v936 = v894[v934];
                float v937;
                if (v936){
                    v937 = v935;
                } else {
                    v937 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v929 && v929 < 1l);
                assert("Tensor range check" && 0 <= v931 && v931 < 4l);
                v928[v934] = v937;
                v931 += 1l ;
            }
            v929 += 1l ;
        }
        float v938;
        v938 = 0.0f;
        int v939;
        v939 = 0l;
        while (while_method_3(v939)){
            int v941;
            v941 = 0l;
            while (while_method_1(v941)){
                assert("Tensor range check" && 0 <= v939 && v939 < 1l);
                assert("Tensor range check" && 0 <= v941 && v941 < 4l);
                int v943;
                v943 = 4l * v939;
                int v944;
                v944 = v943 + v941;
                float v945;
                v945 = v928[v944];
                float v946;
                v946 = v938 + v945;
                v938 = v946;
                v941 += 1l ;
            }
            v939 += 1l ;
        }
        auto v947 = cooperative_groups::coalesced_threads();
        int v948;
        v948 = threadIdx.x;
        int v949;
        v949 = v948 / 16l;
        auto v950 = cooperative_groups::labeled_partition(v947,v949);
        float v951;
        v951 = cooperative_groups::reduce(v950, v938, v64);
        float v952;
        v952 = (float)v927;
        float v953;
        v953 = v951 / v952;
        float v954[4l];
        int v955;
        v955 = 0l;
        while (while_method_3(v955)){
            int v957;
            v957 = 0l;
            while (while_method_1(v957)){
                assert("Tensor range check" && 0 <= v955 && v955 < 1l);
                assert("Tensor range check" && 0 <= v957 && v957 < 4l);
                int v959;
                v959 = 4l * v955;
                int v960;
                v960 = v959 + v957;
                float v961;
                v961 = v849[v960];
                bool v962;
                v962 = v894[v960];
                float v963;
                if (v962){
                    v963 = v961;
                } else {
                    v963 = -1.0f / 0.0f;
                }
                float v964;
                v964 = v963 - v953;
                float v965;
                v965 = exp(v964);
                assert("Tensor range check" && 0 <= v955 && v955 < 1l);
                assert("Tensor range check" && 0 <= v957 && v957 < 4l);
                v954[v960] = v965;
                v957 += 1l ;
            }
            v955 += 1l ;
        }
        float v966;
        v966 = 0.0f;
        int v967;
        v967 = 0l;
        while (while_method_3(v967)){
            int v969;
            v969 = 0l;
            while (while_method_1(v969)){
                assert("Tensor range check" && 0 <= v967 && v967 < 1l);
                assert("Tensor range check" && 0 <= v969 && v969 < 4l);
                int v971;
                v971 = 4l * v967;
                int v972;
                v972 = v971 + v969;
                float v973;
                v973 = v954[v972];
                float v974;
                v974 = v966 + v973;
                v966 = v974;
                v969 += 1l ;
            }
            v967 += 1l ;
        }
        auto v975 = cooperative_groups::coalesced_threads();
        int v976;
        v976 = threadIdx.x;
        int v977;
        v977 = v976 / 16l;
        auto v978 = cooperative_groups::labeled_partition(v975,v977);
        float v979;
        v979 = cooperative_groups::reduce(v978, v966, v64);
        float v980[4l];
        int v981;
        v981 = 0l;
        while (while_method_3(v981)){
            int v983;
            v983 = 0l;
            while (while_method_1(v983)){
                assert("Tensor range check" && 0 <= v981 && v981 < 1l);
                assert("Tensor range check" && 0 <= v983 && v983 < 4l);
                int v985;
                v985 = 4l * v981;
                int v986;
                v986 = v985 + v983;
                float v987;
                v987 = v954[v986];
                float v988;
                v988 = v987 / v979;
                assert("Tensor range check" && 0 <= v981 && v981 < 1l);
                assert("Tensor range check" && 0 <= v983 && v983 < 4l);
                v980[v986] = v988;
                v983 += 1l ;
            }
            v981 += 1l ;
        }
        assert("Tensor range check" && 0 <= v845 && v845 < 64l);
        int v989;
        v989 = 0l;
        while (while_method_3(v989)){
            assert("Tensor range check" && 0 <= v989 && v989 < 1l);
            int v991;
            v991 = 64l * v989;
            int v992;
            v992 = v991 + v848;
            assert("Tensor range check" && 0 <= v989 && v989 < 1l);
            int v993;
            v993 = 4l * v989;
            int4* v994;
            v994 = reinterpret_cast<int4*>(v980 + v993);
            int4* v995;
            v995 = reinterpret_cast<int4*>(v5 + v992);
            assert("Pointer alignment check" && (unsigned long long)(v994) % 4l == 0 && (unsigned long long)(v995) % 4l == 0);
            *v995 = *v994;
            v989 += 1l ;
        }
        v845 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v996;
    v996 = threadIdx.x;
    unsigned long long v997;
    v997 = (unsigned long long)v996;
    curandStatePhilox4_32_10_t v998;
    curand_init(12344321ull,v997,0ull,&v998);
    int v999;
    v999 = threadIdx.x;
    bool v1000;
    v1000 = 0l <= v999;
    bool v1001;
    v1001 = v1000 == false;
    if (v1001){
        assert("The index needs to be zero or positive." && v1000);
    } else {
    }
    int v1003;
    v1003 = v999 % 16l;
    int v1004;
    v1004 = v999 / 16l;
    bool v1005;
    v1005 = v1004 < 2l;
    bool v1006;
    v1006 = v1005 == false;
    if (v1006){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1005);
    } else {
    }
    assert("Tensor range check" && 0 <= v1004 && v1004 < 2l);
    assert("Tensor range check" && 0 <= v1003 && v1003 < 16l);
    int v1008;
    v1008 = 4l * v1003;
    int v1009;
    v1009 = 64l * v1004;
    int v1010;
    v1010 = v1009 + v1008;
    assert("Tensor range check" && 0 <= v1004 && v1004 < 2l);
    assert("Tensor range check" && 0 <= v1003 && v1003 < 16l);
    assert("Tensor range check" && 0 <= v1004 && v1004 < 2l);
    int v1011;
    v1011 = 0l;
    while (while_method_2(v1011)){
        assert("Tensor range check" && 0 <= v1011 && v1011 < 64l);
        int v1013;
        v1013 = 128l * v1011;
        int v1014;
        v1014 = v1013 + v1010;
        float v1015[4l];
        int v1016[4l];
        int v1017;
        v1017 = 0l;
        while (while_method_3(v1017)){
            assert("Tensor range check" && 0 <= v1017 && v1017 < 1l);
            int v1019;
            v1019 = 4l * v1017;
            assert("Tensor range check" && 0 <= v1017 && v1017 < 1l);
            int v1020;
            v1020 = 64l * v1017;
            int v1021;
            v1021 = v1020 + v1014;
            int4* v1022;
            v1022 = reinterpret_cast<int4*>(v1 + v1021);
            int4* v1023;
            v1023 = reinterpret_cast<int4*>(v1015 + v1019);
            assert("Pointer alignment check" && (unsigned long long)(v1022) % 4l == 0 && (unsigned long long)(v1023) % 4l == 0);
            *v1023 = *v1022;
            v1017 += 1l ;
        }
        int v1024;
        v1024 = 0l;
        while (while_method_3(v1024)){
            int v1026;
            v1026 = 0l;
            while (while_method_1(v1026)){
                bool v1028;
                v1028 = 0l <= v1026;
                bool v1030;
                if (v1028){
                    bool v1029;
                    v1029 = v1026 < 4l;
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
                bool v1033;
                v1033 = 0l <= v1003;
                bool v1035;
                if (v1033){
                    bool v1034;
                    v1034 = v1003 < 16l;
                    v1035 = v1034;
                } else {
                    v1035 = false;
                }
                bool v1036;
                v1036 = v1035 == false;
                if (v1036){
                    assert("The indices should be inside the range of the dimension." && v1035);
                } else {
                }
                int v1038;
                v1038 = v1003 * 4l;
                int v1039;
                v1039 = v1026 + v1038;
                bool v1040;
                v1040 = 0l <= v1024;
                bool v1042;
                if (v1040){
                    bool v1041;
                    v1041 = v1024 < 1l;
                    v1042 = v1041;
                } else {
                    v1042 = false;
                }
                bool v1043;
                v1043 = v1042 == false;
                if (v1043){
                    assert("The indices should be inside the range of the dimension." && v1042);
                } else {
                }
                int v1045;
                v1045 = v1024 * 64l;
                int v1046;
                v1046 = v1039 + v1045;
                assert("Tensor range check" && 0 <= v1024 && v1024 < 1l);
                assert("Tensor range check" && 0 <= v1026 && v1026 < 4l);
                int v1047;
                v1047 = 4l * v1024;
                int v1048;
                v1048 = v1047 + v1026;
                v1016[v1048] = v1046;
                v1026 += 1l ;
            }
            v1024 += 1l ;
        }
        bool v1049;
        v1049 = 0l <= v1004;
        bool v1050;
        v1050 = v1049 && v1005;
        bool v1051;
        v1051 = v1050 == false;
        if (v1051){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1050);
        } else {
        }
        bool v1053;
        v1053 = 0l <= v1011;
        bool v1055;
        if (v1053){
            bool v1054;
            v1054 = v1011 < 64l;
            v1055 = v1054;
        } else {
            v1055 = false;
        }
        bool v1056;
        v1056 = v1055 == false;
        if (v1056){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1055);
        } else {
        }
        int v1058;
        v1058 = v1011 * 2l;
        int v1059;
        v1059 = v1058 + v1004;
        float v1060;
        v1060 = 0.0f;
        int v1061;
        v1061 = 0l;
        while (while_method_3(v1061)){
            int v1063;
            v1063 = 0l;
            while (while_method_1(v1063)){
                assert("Tensor range check" && 0 <= v1061 && v1061 < 1l);
                assert("Tensor range check" && 0 <= v1063 && v1063 < 4l);
                int v1065;
                v1065 = 4l * v1061;
                int v1066;
                v1066 = v1065 + v1063;
                float v1067;
                v1067 = v1015[v1066];
                float v1068;
                v1068 = v1060 + v1067;
                v1060 = v1068;
                v1063 += 1l ;
            }
            v1061 += 1l ;
        }
        auto v1069 = cooperative_groups::coalesced_threads();
        int v1070;
        v1070 = threadIdx.x;
        int v1071;
        v1071 = v1070 / 16l;
        auto v1072 = cooperative_groups::labeled_partition(v1069,v1071);
        float v1073;
        v1073 = cooperative_groups::reduce(v1072, v1060, v64);
        float v1074;
        v1074 = v1073 / 64.0f;
        float v1075[4l];
        int v1076;
        v1076 = 0l;
        while (while_method_3(v1076)){
            int v1078;
            v1078 = 0l;
            while (while_method_1(v1078)){
                assert("Tensor range check" && 0 <= v1076 && v1076 < 1l);
                assert("Tensor range check" && 0 <= v1078 && v1078 < 4l);
                int v1080;
                v1080 = 4l * v1076;
                int v1081;
                v1081 = v1080 + v1078;
                float v1082;
                v1082 = v1015[v1081];
                float v1083;
                v1083 = v1082 - v1074;
                float v1084;
                v1084 = exp(v1083);
                assert("Tensor range check" && 0 <= v1076 && v1076 < 1l);
                assert("Tensor range check" && 0 <= v1078 && v1078 < 4l);
                v1075[v1081] = v1084;
                v1078 += 1l ;
            }
            v1076 += 1l ;
        }
        float v1085;
        v1085 = 0.0f;
        int v1086;
        v1086 = 0l;
        while (while_method_3(v1086)){
            int v1088;
            v1088 = 0l;
            while (while_method_1(v1088)){
                assert("Tensor range check" && 0 <= v1086 && v1086 < 1l);
                assert("Tensor range check" && 0 <= v1088 && v1088 < 4l);
                int v1090;
                v1090 = 4l * v1086;
                int v1091;
                v1091 = v1090 + v1088;
                float v1092;
                v1092 = v1075[v1091];
                float v1093;
                v1093 = v1085 + v1092;
                v1085 = v1093;
                v1088 += 1l ;
            }
            v1086 += 1l ;
        }
        auto v1094 = cooperative_groups::coalesced_threads();
        int v1095;
        v1095 = threadIdx.x;
        int v1096;
        v1096 = v1095 / 16l;
        auto v1097 = cooperative_groups::labeled_partition(v1094,v1096);
        float v1098;
        v1098 = cooperative_groups::reduce(v1097, v1085, v64);
        float v1099[4l];
        int v1100;
        v1100 = 0l;
        while (while_method_3(v1100)){
            int v1102;
            v1102 = 0l;
            while (while_method_1(v1102)){
                assert("Tensor range check" && 0 <= v1100 && v1100 < 1l);
                assert("Tensor range check" && 0 <= v1102 && v1102 < 4l);
                int v1104;
                v1104 = 4l * v1100;
                int v1105;
                v1105 = v1104 + v1102;
                float v1106;
                v1106 = v1075[v1105];
                float v1107;
                v1107 = v1106 / v1098;
                assert("Tensor range check" && 0 <= v1100 && v1100 < 1l);
                assert("Tensor range check" && 0 <= v1102 && v1102 < 4l);
                v1099[v1105] = v1107;
                v1102 += 1l ;
            }
            v1100 += 1l ;
        }
        float v1108[4l];
        float v1109;
        v1109 = 0.0f;
        int v1110;
        v1110 = 0l;
        while (while_method_3(v1110)){
            assert("Tensor range check" && 0 <= v1110 && v1110 < 1l);
            int v1112;
            v1112 = 4l * v1110;
            assert("Tensor range check" && 0 <= v1110 && v1110 < 1l);
            int v1113; float v1114;
            Tuple0 tmp30 = Tuple0{0l, 0.0f};
            v1113 = tmp30.v0; v1114 = tmp30.v1;
            while (while_method_1(v1113)){
                assert("Tensor range check" && 0 <= v1113 && v1113 < 4l);
                int v1116;
                v1116 = v1113 + v1112;
                float v1117;
                v1117 = v1099[v1116];
                float v1118;
                v1118 = v1114 + v1117;
                v1114 = v1118;
                v1113 += 1l ;
            }
            auto v1119 = cooperative_groups::coalesced_threads();
            int v1120;
            v1120 = threadIdx.x;
            int v1121;
            v1121 = v1120 / 16l;
            auto v1122 = cooperative_groups::labeled_partition(v1119,v1121);
            Closure2 v1123{};
            float v1124;
            v1124 = cooperative_groups::inclusive_scan(v1122, v1114, v1123);
            float v1125;
            v1125 = v1122.shfl_up(v1124,1);
            bool v1126;
            v1126 = v1122.thread_rank() == 0;
            float v1127;
            if (v1126){
                v1127 = 0.0f;
            } else {
                v1127 = v1125;
            }
            float v1128;
            v1128 = v1122.shfl(v1124,v1122.num_threads()-1);
            float v1129;
            v1129 = v1109 + v1127;
            int v1130; float v1131;
            Tuple0 tmp31 = Tuple0{0l, v1129};
            v1130 = tmp31.v0; v1131 = tmp31.v1;
            while (while_method_1(v1130)){
                assert("Tensor range check" && 0 <= v1130 && v1130 < 4l);
                int v1133;
                v1133 = v1130 + v1112;
                float v1134;
                v1134 = v1099[v1133];
                float v1135;
                v1135 = v1131 + v1134;
                assert("Tensor range check" && 0 <= v1130 && v1130 < 4l);
                v1108[v1133] = v1135;
                v1131 = v1135;
                v1130 += 1l ;
            }
            float v1136;
            v1136 = v1109 + v1128;
            v1109 = v1136;
            v1110 += 1l ;
        }
        float v1137[4l];
        bool v1138[4l];
        int v1139;
        v1139 = 0l;
        while (while_method_3(v1139)){
            int v1141;
            v1141 = 0l;
            while (while_method_1(v1141)){
                assert("Tensor range check" && 0 <= v1139 && v1139 < 1l);
                assert("Tensor range check" && 0 <= v1141 && v1141 < 4l);
                int v1143;
                v1143 = 4l * v1139;
                int v1144;
                v1144 = v1143 + v1141;
                float v1145;
                v1145 = v1108[v1144];
                float v1146;
                v1146 = v1099[v1144];
                bool v1147;
                v1147 = v1146 > 0.0f;
                assert("Tensor range check" && 0 <= v1139 && v1139 < 1l);
                assert("Tensor range check" && 0 <= v1141 && v1141 < 4l);
                v1137[v1144] = v1145;
                v1138[v1144] = v1147;
                v1141 += 1l ;
            }
            v1139 += 1l ;
        }
        float v1148; bool v1149;
        Tuple3 tmp32 = Tuple3{-1.0f / 0.0f, false};
        v1148 = tmp32.v0; v1149 = tmp32.v1;
        int v1150;
        v1150 = 0l;
        while (while_method_3(v1150)){
            int v1152;
            v1152 = 0l;
            while (while_method_1(v1152)){
                assert("Tensor range check" && 0 <= v1150 && v1150 < 1l);
                assert("Tensor range check" && 0 <= v1152 && v1152 < 4l);
                int v1154;
                v1154 = 4l * v1150;
                int v1155;
                v1155 = v1154 + v1152;
                float v1156;
                v1156 = v1137[v1155];
                bool v1157;
                v1157 = v1138[v1155];
                float v1164; bool v1165;
                if (v1149){
                    if (v1157){
                        bool v1158;
                        v1158 = v1148 >= v1156;
                        float v1159;
                        if (v1158){
                            v1159 = v1148;
                        } else {
                            v1159 = v1156;
                        }
                        v1164 = v1159; v1165 = true;
                    } else {
                        v1164 = v1148; v1165 = v1149;
                    }
                } else {
                    if (v1157){
                        v1164 = v1156; v1165 = v1157;
                    } else {
                        v1164 = v1148; v1165 = v1149;
                    }
                }
                v1148 = v1164;
                v1149 = v1165;
                v1152 += 1l ;
            }
            v1150 += 1l ;
        }
        auto v1166 = cooperative_groups::coalesced_threads();
        int v1167;
        v1167 = threadIdx.x;
        int v1168;
        v1168 = v1167 / 16l;
        auto v1169 = cooperative_groups::labeled_partition(v1166,v1168);
        Closure5 v1170{};
        float v1171; bool v1172;
        Tuple3 tmp33 = cooperative_groups::reduce(v1169, Tuple3{v1148, v1149}, v1170);
        v1171 = tmp33.v0; v1172 = tmp33.v1;
        bool v1173;
        v1173 = v1172 == false;
        if (v1173){
            assert("The local reduce must be true." && v1172);
        } else {
        }
        float v1175[4l];
        int v1176[4l];
        int v1177;
        v1177 = 0l;
        while (while_method_3(v1177)){
            int v1179;
            v1179 = 0l;
            while (while_method_1(v1179)){
                assert("Tensor range check" && 0 <= v1177 && v1177 < 1l);
                assert("Tensor range check" && 0 <= v1179 && v1179 < 4l);
                int v1181;
                v1181 = 4l * v1177;
                int v1182;
                v1182 = v1181 + v1179;
                int v1183;
                v1183 = v1016[v1182];
                float v1184;
                v1184 = curand_uniform(&v998);
                assert("Tensor range check" && 0 <= v1177 && v1177 < 1l);
                assert("Tensor range check" && 0 <= v1179 && v1179 < 4l);
                v1175[v1182] = v1184;
                v1176[v1182] = v1183;
                v1179 += 1l ;
            }
            v1177 += 1l ;
        }
        float v1185; int v1186;
        Tuple1 tmp34 = Tuple1{0.0f, 2147483647l};
        v1185 = tmp34.v0; v1186 = tmp34.v1;
        int v1187;
        v1187 = 0l;
        while (while_method_3(v1187)){
            int v1189;
            v1189 = 0l;
            while (while_method_1(v1189)){
                assert("Tensor range check" && 0 <= v1187 && v1187 < 1l);
                assert("Tensor range check" && 0 <= v1189 && v1189 < 4l);
                int v1191;
                v1191 = 4l * v1187;
                int v1192;
                v1192 = v1191 + v1189;
                float v1193;
                v1193 = v1175[v1192];
                int v1194;
                v1194 = v1176[v1192];
                bool v1195;
                v1195 = v1186 < v1194;
                float v1196; int v1197;
                if (v1195){
                    v1196 = v1185; v1197 = v1186;
                } else {
                    v1196 = v1193; v1197 = v1194;
                }
                v1185 = v1196;
                v1186 = v1197;
                v1189 += 1l ;
            }
            v1187 += 1l ;
        }
        auto v1198 = cooperative_groups::coalesced_threads();
        int v1199;
        v1199 = threadIdx.x;
        int v1200;
        v1200 = v1199 / 16l;
        auto v1201 = cooperative_groups::labeled_partition(v1198,v1200);
        Closure6 v1202{};
        float v1203; int v1204;
        Tuple1 tmp35 = cooperative_groups::reduce(v1201, Tuple1{v1185, v1186}, v1202);
        v1203 = tmp35.v0; v1204 = tmp35.v1;
        float v1205;
        v1205 = v1171 * v1203;
        int v1206[4l];
        bool v1207[4l];
        int v1208;
        v1208 = 0l;
        while (while_method_3(v1208)){
            int v1210;
            v1210 = 0l;
            while (while_method_1(v1210)){
                assert("Tensor range check" && 0 <= v1208 && v1208 < 1l);
                assert("Tensor range check" && 0 <= v1210 && v1210 < 4l);
                int v1212;
                v1212 = 4l * v1208;
                int v1213;
                v1213 = v1212 + v1210;
                float v1214;
                v1214 = v1137[v1213];
                bool v1215;
                v1215 = v1138[v1213];
                int v1216;
                v1216 = v1016[v1213];
                int v1219; bool v1220;
                if (v1215){
                    float v1217;
                    v1217 = v1214 - v1205;
                    bool v1218;
                    v1218 = v1217 >= 0.0f;
                    v1219 = v1216; v1220 = v1218;
                } else {
                    v1219 = 2147483647l; v1220 = false;
                }
                assert("Tensor range check" && 0 <= v1208 && v1208 < 1l);
                assert("Tensor range check" && 0 <= v1210 && v1210 < 4l);
                v1206[v1213] = v1219;
                v1207[v1213] = v1220;
                v1210 += 1l ;
            }
            v1208 += 1l ;
        }
        int v1221; bool v1222;
        Tuple4 tmp36 = Tuple4{2147483647l, false};
        v1221 = tmp36.v0; v1222 = tmp36.v1;
        int v1223;
        v1223 = 0l;
        while (while_method_3(v1223)){
            int v1225;
            v1225 = 0l;
            while (while_method_1(v1225)){
                assert("Tensor range check" && 0 <= v1223 && v1223 < 1l);
                assert("Tensor range check" && 0 <= v1225 && v1225 < 4l);
                int v1227;
                v1227 = 4l * v1223;
                int v1228;
                v1228 = v1227 + v1225;
                int v1229;
                v1229 = v1206[v1228];
                bool v1230;
                v1230 = v1207[v1228];
                int v1237; bool v1238;
                if (v1222){
                    if (v1230){
                        bool v1231;
                        v1231 = v1221 < v1229;
                        int v1232;
                        if (v1231){
                            v1232 = v1221;
                        } else {
                            v1232 = v1229;
                        }
                        v1237 = v1232; v1238 = true;
                    } else {
                        v1237 = v1221; v1238 = v1222;
                    }
                } else {
                    if (v1230){
                        v1237 = v1229; v1238 = v1230;
                    } else {
                        v1237 = v1221; v1238 = v1222;
                    }
                }
                v1221 = v1237;
                v1222 = v1238;
                v1225 += 1l ;
            }
            v1223 += 1l ;
        }
        auto v1239 = cooperative_groups::coalesced_threads();
        int v1240;
        v1240 = threadIdx.x;
        int v1241;
        v1241 = v1240 / 16l;
        auto v1242 = cooperative_groups::labeled_partition(v1239,v1241);
        Closure7 v1243{};
        int v1244; bool v1245;
        Tuple4 tmp37 = cooperative_groups::reduce(v1242, Tuple4{v1221, v1222}, v1243);
        v1244 = tmp37.v0; v1245 = tmp37.v1;
        bool v1246;
        v1246 = v1245 == false;
        if (v1246){
            assert("The local reduce must be true." && v1245);
        } else {
        }
        assert("Tensor range check" && 0 <= v1011 && v1011 < 64l);
        int v1248;
        v1248 = 0l;
        while (while_method_3(v1248)){
            assert("Tensor range check" && 0 <= v1248 && v1248 < 1l);
            int v1250;
            v1250 = 64l * v1248;
            int v1251;
            v1251 = v1250 + v1014;
            assert("Tensor range check" && 0 <= v1248 && v1248 < 1l);
            int v1252;
            v1252 = 4l * v1248;
            int4* v1253;
            v1253 = reinterpret_cast<int4*>(v1099 + v1252);
            int4* v1254;
            v1254 = reinterpret_cast<int4*>(v14 + v1251);
            assert("Pointer alignment check" && (unsigned long long)(v1253) % 4l == 0 && (unsigned long long)(v1254) % 4l == 0);
            *v1254 = *v1253;
            v1248 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1011 && v1011 < 64l);
        int v1255;
        v1255 = 2l * v1011;
        int v1256;
        v1256 = v1255 + v1004;
        v15[v1256] = v1244;
        v1011 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1257;
    v1257 = threadIdx.x;
    unsigned long long v1258;
    v1258 = (unsigned long long)v1257;
    curandStatePhilox4_32_10_t v1259;
    curand_init(12344321ull,v1258,0ull,&v1259);
    int v1260;
    v1260 = threadIdx.x;
    bool v1261;
    v1261 = 0l <= v1260;
    bool v1262;
    v1262 = v1261 == false;
    if (v1262){
        assert("The index needs to be zero or positive." && v1261);
    } else {
    }
    int v1264;
    v1264 = v1260 % 16l;
    int v1265;
    v1265 = v1260 / 16l;
    bool v1266;
    v1266 = v1265 < 2l;
    bool v1267;
    v1267 = v1266 == false;
    if (v1267){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1266);
    } else {
    }
    assert("Tensor range check" && 0 <= v1265 && v1265 < 2l);
    assert("Tensor range check" && 0 <= v1264 && v1264 < 16l);
    int v1269;
    v1269 = 4l * v1264;
    int v1270;
    v1270 = 64l * v1265;
    int v1271;
    v1271 = v1270 + v1269;
    assert("Tensor range check" && 0 <= v1265 && v1265 < 2l);
    assert("Tensor range check" && 0 <= v1264 && v1264 < 16l);
    assert("Tensor range check" && 0 <= v1265 && v1265 < 2l);
    int v1272;
    v1272 = 0l;
    while (while_method_2(v1272)){
        assert("Tensor range check" && 0 <= v1272 && v1272 < 64l);
        int v1274;
        v1274 = 128l * v1272;
        int v1275;
        v1275 = v1274 + v1271;
        float v1276[4l];
        int v1277[4l];
        int v1278;
        v1278 = 0l;
        while (while_method_3(v1278)){
            assert("Tensor range check" && 0 <= v1278 && v1278 < 1l);
            int v1280;
            v1280 = 4l * v1278;
            assert("Tensor range check" && 0 <= v1278 && v1278 < 1l);
            int v1281;
            v1281 = 64l * v1278;
            int v1282;
            v1282 = v1281 + v1275;
            int4* v1283;
            v1283 = reinterpret_cast<int4*>(v1 + v1282);
            int4* v1284;
            v1284 = reinterpret_cast<int4*>(v1276 + v1280);
            assert("Pointer alignment check" && (unsigned long long)(v1283) % 4l == 0 && (unsigned long long)(v1284) % 4l == 0);
            *v1284 = *v1283;
            v1278 += 1l ;
        }
        int v1285;
        v1285 = 0l;
        while (while_method_3(v1285)){
            int v1287;
            v1287 = 0l;
            while (while_method_1(v1287)){
                bool v1289;
                v1289 = 0l <= v1287;
                bool v1291;
                if (v1289){
                    bool v1290;
                    v1290 = v1287 < 4l;
                    v1291 = v1290;
                } else {
                    v1291 = false;
                }
                bool v1292;
                v1292 = v1291 == false;
                if (v1292){
                    assert("The indices should be inside the range of the dimension." && v1291);
                } else {
                }
                bool v1294;
                v1294 = 0l <= v1264;
                bool v1296;
                if (v1294){
                    bool v1295;
                    v1295 = v1264 < 16l;
                    v1296 = v1295;
                } else {
                    v1296 = false;
                }
                bool v1297;
                v1297 = v1296 == false;
                if (v1297){
                    assert("The indices should be inside the range of the dimension." && v1296);
                } else {
                }
                int v1299;
                v1299 = v1264 * 4l;
                int v1300;
                v1300 = v1287 + v1299;
                bool v1301;
                v1301 = 0l <= v1285;
                bool v1303;
                if (v1301){
                    bool v1302;
                    v1302 = v1285 < 1l;
                    v1303 = v1302;
                } else {
                    v1303 = false;
                }
                bool v1304;
                v1304 = v1303 == false;
                if (v1304){
                    assert("The indices should be inside the range of the dimension." && v1303);
                } else {
                }
                int v1306;
                v1306 = v1285 * 64l;
                int v1307;
                v1307 = v1300 + v1306;
                assert("Tensor range check" && 0 <= v1285 && v1285 < 1l);
                assert("Tensor range check" && 0 <= v1287 && v1287 < 4l);
                int v1308;
                v1308 = 4l * v1285;
                int v1309;
                v1309 = v1308 + v1287;
                v1277[v1309] = v1307;
                v1287 += 1l ;
            }
            v1285 += 1l ;
        }
        bool v1310;
        v1310 = 0l <= v1265;
        bool v1311;
        v1311 = v1310 && v1266;
        bool v1312;
        v1312 = v1311 == false;
        if (v1312){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1311);
        } else {
        }
        bool v1314;
        v1314 = 0l <= v1272;
        bool v1316;
        if (v1314){
            bool v1315;
            v1315 = v1272 < 64l;
            v1316 = v1315;
        } else {
            v1316 = false;
        }
        bool v1317;
        v1317 = v1316 == false;
        if (v1317){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1316);
        } else {
        }
        int v1319;
        v1319 = v1272 * 2l;
        int v1320;
        v1320 = v1319 + v1265;
        bool v1321[4l];
        int v1322;
        v1322 = 0l;
        while (while_method_3(v1322)){
            int v1324;
            v1324 = 0l;
            while (while_method_1(v1324)){
                assert("Tensor range check" && 0 <= v1322 && v1322 < 1l);
                assert("Tensor range check" && 0 <= v1324 && v1324 < 4l);
                int v1326;
                v1326 = 4l * v1322;
                int v1327;
                v1327 = v1326 + v1324;
                float v1328;
                v1328 = v1276[v1327];
                int v1329;
                v1329 = v1277[v1327];
                bool v1330;
                v1330 = v1329 < 11l;
                assert("Tensor range check" && 0 <= v1322 && v1322 < 1l);
                assert("Tensor range check" && 0 <= v1324 && v1324 < 4l);
                v1321[v1327] = v1330;
                v1324 += 1l ;
            }
            v1322 += 1l ;
        }
        int v1331[4l];
        int v1332;
        v1332 = 0l;
        while (while_method_3(v1332)){
            int v1334;
            v1334 = 0l;
            while (while_method_1(v1334)){
                assert("Tensor range check" && 0 <= v1332 && v1332 < 1l);
                assert("Tensor range check" && 0 <= v1334 && v1334 < 4l);
                int v1336;
                v1336 = 4l * v1332;
                int v1337;
                v1337 = v1336 + v1334;
                bool v1338;
                v1338 = v1321[v1337];
                int v1339;
                if (v1338){
                    v1339 = 1l;
                } else {
                    v1339 = 0l;
                }
                assert("Tensor range check" && 0 <= v1332 && v1332 < 1l);
                assert("Tensor range check" && 0 <= v1334 && v1334 < 4l);
                v1331[v1337] = v1339;
                v1334 += 1l ;
            }
            v1332 += 1l ;
        }
        int v1340;
        v1340 = 0l;
        int v1341;
        v1341 = 0l;
        while (while_method_3(v1341)){
            int v1343;
            v1343 = 0l;
            while (while_method_1(v1343)){
                assert("Tensor range check" && 0 <= v1341 && v1341 < 1l);
                assert("Tensor range check" && 0 <= v1343 && v1343 < 4l);
                int v1345;
                v1345 = 4l * v1341;
                int v1346;
                v1346 = v1345 + v1343;
                int v1347;
                v1347 = v1331[v1346];
                int v1348;
                v1348 = v1340 + v1347;
                v1340 = v1348;
                v1343 += 1l ;
            }
            v1341 += 1l ;
        }
        auto v1349 = cooperative_groups::coalesced_threads();
        int v1350;
        v1350 = threadIdx.x;
        int v1351;
        v1351 = v1350 / 16l;
        auto v1352 = cooperative_groups::labeled_partition(v1349,v1351);
        Closure4 v1353{};
        int v1354;
        v1354 = cooperative_groups::reduce(v1352, v1340, v1353);
        float v1355[4l];
        int v1356;
        v1356 = 0l;
        while (while_method_3(v1356)){
            int v1358;
            v1358 = 0l;
            while (while_method_1(v1358)){
                assert("Tensor range check" && 0 <= v1356 && v1356 < 1l);
                assert("Tensor range check" && 0 <= v1358 && v1358 < 4l);
                int v1360;
                v1360 = 4l * v1356;
                int v1361;
                v1361 = v1360 + v1358;
                float v1362;
                v1362 = v1276[v1361];
                bool v1363;
                v1363 = v1321[v1361];
                float v1364;
                if (v1363){
                    v1364 = v1362;
                } else {
                    v1364 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1356 && v1356 < 1l);
                assert("Tensor range check" && 0 <= v1358 && v1358 < 4l);
                v1355[v1361] = v1364;
                v1358 += 1l ;
            }
            v1356 += 1l ;
        }
        float v1365;
        v1365 = 0.0f;
        int v1366;
        v1366 = 0l;
        while (while_method_3(v1366)){
            int v1368;
            v1368 = 0l;
            while (while_method_1(v1368)){
                assert("Tensor range check" && 0 <= v1366 && v1366 < 1l);
                assert("Tensor range check" && 0 <= v1368 && v1368 < 4l);
                int v1370;
                v1370 = 4l * v1366;
                int v1371;
                v1371 = v1370 + v1368;
                float v1372;
                v1372 = v1355[v1371];
                float v1373;
                v1373 = v1365 + v1372;
                v1365 = v1373;
                v1368 += 1l ;
            }
            v1366 += 1l ;
        }
        auto v1374 = cooperative_groups::coalesced_threads();
        int v1375;
        v1375 = threadIdx.x;
        int v1376;
        v1376 = v1375 / 16l;
        auto v1377 = cooperative_groups::labeled_partition(v1374,v1376);
        float v1378;
        v1378 = cooperative_groups::reduce(v1377, v1365, v64);
        float v1379;
        v1379 = (float)v1354;
        float v1380;
        v1380 = v1378 / v1379;
        float v1381[4l];
        int v1382;
        v1382 = 0l;
        while (while_method_3(v1382)){
            int v1384;
            v1384 = 0l;
            while (while_method_1(v1384)){
                assert("Tensor range check" && 0 <= v1382 && v1382 < 1l);
                assert("Tensor range check" && 0 <= v1384 && v1384 < 4l);
                int v1386;
                v1386 = 4l * v1382;
                int v1387;
                v1387 = v1386 + v1384;
                float v1388;
                v1388 = v1276[v1387];
                bool v1389;
                v1389 = v1321[v1387];
                float v1390;
                if (v1389){
                    v1390 = v1388;
                } else {
                    v1390 = -1.0f / 0.0f;
                }
                float v1391;
                v1391 = v1390 - v1380;
                float v1392;
                v1392 = exp(v1391);
                assert("Tensor range check" && 0 <= v1382 && v1382 < 1l);
                assert("Tensor range check" && 0 <= v1384 && v1384 < 4l);
                v1381[v1387] = v1392;
                v1384 += 1l ;
            }
            v1382 += 1l ;
        }
        float v1393;
        v1393 = 0.0f;
        int v1394;
        v1394 = 0l;
        while (while_method_3(v1394)){
            int v1396;
            v1396 = 0l;
            while (while_method_1(v1396)){
                assert("Tensor range check" && 0 <= v1394 && v1394 < 1l);
                assert("Tensor range check" && 0 <= v1396 && v1396 < 4l);
                int v1398;
                v1398 = 4l * v1394;
                int v1399;
                v1399 = v1398 + v1396;
                float v1400;
                v1400 = v1381[v1399];
                float v1401;
                v1401 = v1393 + v1400;
                v1393 = v1401;
                v1396 += 1l ;
            }
            v1394 += 1l ;
        }
        auto v1402 = cooperative_groups::coalesced_threads();
        int v1403;
        v1403 = threadIdx.x;
        int v1404;
        v1404 = v1403 / 16l;
        auto v1405 = cooperative_groups::labeled_partition(v1402,v1404);
        float v1406;
        v1406 = cooperative_groups::reduce(v1405, v1393, v64);
        float v1407[4l];
        int v1408;
        v1408 = 0l;
        while (while_method_3(v1408)){
            int v1410;
            v1410 = 0l;
            while (while_method_1(v1410)){
                assert("Tensor range check" && 0 <= v1408 && v1408 < 1l);
                assert("Tensor range check" && 0 <= v1410 && v1410 < 4l);
                int v1412;
                v1412 = 4l * v1408;
                int v1413;
                v1413 = v1412 + v1410;
                float v1414;
                v1414 = v1381[v1413];
                float v1415;
                v1415 = v1414 / v1406;
                assert("Tensor range check" && 0 <= v1408 && v1408 < 1l);
                assert("Tensor range check" && 0 <= v1410 && v1410 < 4l);
                v1407[v1413] = v1415;
                v1410 += 1l ;
            }
            v1408 += 1l ;
        }
        float v1416[4l];
        float v1417;
        v1417 = 0.0f;
        int v1418;
        v1418 = 0l;
        while (while_method_3(v1418)){
            assert("Tensor range check" && 0 <= v1418 && v1418 < 1l);
            int v1420;
            v1420 = 4l * v1418;
            assert("Tensor range check" && 0 <= v1418 && v1418 < 1l);
            int v1421; float v1422;
            Tuple0 tmp38 = Tuple0{0l, 0.0f};
            v1421 = tmp38.v0; v1422 = tmp38.v1;
            while (while_method_1(v1421)){
                assert("Tensor range check" && 0 <= v1421 && v1421 < 4l);
                int v1424;
                v1424 = v1421 + v1420;
                float v1425;
                v1425 = v1407[v1424];
                float v1426;
                v1426 = v1422 + v1425;
                v1422 = v1426;
                v1421 += 1l ;
            }
            auto v1427 = cooperative_groups::coalesced_threads();
            int v1428;
            v1428 = threadIdx.x;
            int v1429;
            v1429 = v1428 / 16l;
            auto v1430 = cooperative_groups::labeled_partition(v1427,v1429);
            Closure2 v1431{};
            float v1432;
            v1432 = cooperative_groups::inclusive_scan(v1430, v1422, v1431);
            float v1433;
            v1433 = v1430.shfl_up(v1432,1);
            bool v1434;
            v1434 = v1430.thread_rank() == 0;
            float v1435;
            if (v1434){
                v1435 = 0.0f;
            } else {
                v1435 = v1433;
            }
            float v1436;
            v1436 = v1430.shfl(v1432,v1430.num_threads()-1);
            float v1437;
            v1437 = v1417 + v1435;
            int v1438; float v1439;
            Tuple0 tmp39 = Tuple0{0l, v1437};
            v1438 = tmp39.v0; v1439 = tmp39.v1;
            while (while_method_1(v1438)){
                assert("Tensor range check" && 0 <= v1438 && v1438 < 4l);
                int v1441;
                v1441 = v1438 + v1420;
                float v1442;
                v1442 = v1407[v1441];
                float v1443;
                v1443 = v1439 + v1442;
                assert("Tensor range check" && 0 <= v1438 && v1438 < 4l);
                v1416[v1441] = v1443;
                v1439 = v1443;
                v1438 += 1l ;
            }
            float v1444;
            v1444 = v1417 + v1436;
            v1417 = v1444;
            v1418 += 1l ;
        }
        float v1445[4l];
        bool v1446[4l];
        int v1447;
        v1447 = 0l;
        while (while_method_3(v1447)){
            int v1449;
            v1449 = 0l;
            while (while_method_1(v1449)){
                assert("Tensor range check" && 0 <= v1447 && v1447 < 1l);
                assert("Tensor range check" && 0 <= v1449 && v1449 < 4l);
                int v1451;
                v1451 = 4l * v1447;
                int v1452;
                v1452 = v1451 + v1449;
                float v1453;
                v1453 = v1416[v1452];
                float v1454;
                v1454 = v1407[v1452];
                bool v1455;
                v1455 = v1454 > 0.0f;
                assert("Tensor range check" && 0 <= v1447 && v1447 < 1l);
                assert("Tensor range check" && 0 <= v1449 && v1449 < 4l);
                v1445[v1452] = v1453;
                v1446[v1452] = v1455;
                v1449 += 1l ;
            }
            v1447 += 1l ;
        }
        float v1456; bool v1457;
        Tuple3 tmp40 = Tuple3{-1.0f / 0.0f, false};
        v1456 = tmp40.v0; v1457 = tmp40.v1;
        int v1458;
        v1458 = 0l;
        while (while_method_3(v1458)){
            int v1460;
            v1460 = 0l;
            while (while_method_1(v1460)){
                assert("Tensor range check" && 0 <= v1458 && v1458 < 1l);
                assert("Tensor range check" && 0 <= v1460 && v1460 < 4l);
                int v1462;
                v1462 = 4l * v1458;
                int v1463;
                v1463 = v1462 + v1460;
                float v1464;
                v1464 = v1445[v1463];
                bool v1465;
                v1465 = v1446[v1463];
                float v1472; bool v1473;
                if (v1457){
                    if (v1465){
                        bool v1466;
                        v1466 = v1456 >= v1464;
                        float v1467;
                        if (v1466){
                            v1467 = v1456;
                        } else {
                            v1467 = v1464;
                        }
                        v1472 = v1467; v1473 = true;
                    } else {
                        v1472 = v1456; v1473 = v1457;
                    }
                } else {
                    if (v1465){
                        v1472 = v1464; v1473 = v1465;
                    } else {
                        v1472 = v1456; v1473 = v1457;
                    }
                }
                v1456 = v1472;
                v1457 = v1473;
                v1460 += 1l ;
            }
            v1458 += 1l ;
        }
        auto v1474 = cooperative_groups::coalesced_threads();
        int v1475;
        v1475 = threadIdx.x;
        int v1476;
        v1476 = v1475 / 16l;
        auto v1477 = cooperative_groups::labeled_partition(v1474,v1476);
        Closure5 v1478{};
        float v1479; bool v1480;
        Tuple3 tmp41 = cooperative_groups::reduce(v1477, Tuple3{v1456, v1457}, v1478);
        v1479 = tmp41.v0; v1480 = tmp41.v1;
        bool v1481;
        v1481 = v1480 == false;
        if (v1481){
            assert("The local reduce must be true." && v1480);
        } else {
        }
        float v1483[4l];
        int v1484[4l];
        int v1485;
        v1485 = 0l;
        while (while_method_3(v1485)){
            int v1487;
            v1487 = 0l;
            while (while_method_1(v1487)){
                assert("Tensor range check" && 0 <= v1485 && v1485 < 1l);
                assert("Tensor range check" && 0 <= v1487 && v1487 < 4l);
                int v1489;
                v1489 = 4l * v1485;
                int v1490;
                v1490 = v1489 + v1487;
                int v1491;
                v1491 = v1277[v1490];
                float v1492;
                v1492 = curand_uniform(&v1259);
                assert("Tensor range check" && 0 <= v1485 && v1485 < 1l);
                assert("Tensor range check" && 0 <= v1487 && v1487 < 4l);
                v1483[v1490] = v1492;
                v1484[v1490] = v1491;
                v1487 += 1l ;
            }
            v1485 += 1l ;
        }
        float v1493; int v1494;
        Tuple1 tmp42 = Tuple1{0.0f, 2147483647l};
        v1493 = tmp42.v0; v1494 = tmp42.v1;
        int v1495;
        v1495 = 0l;
        while (while_method_3(v1495)){
            int v1497;
            v1497 = 0l;
            while (while_method_1(v1497)){
                assert("Tensor range check" && 0 <= v1495 && v1495 < 1l);
                assert("Tensor range check" && 0 <= v1497 && v1497 < 4l);
                int v1499;
                v1499 = 4l * v1495;
                int v1500;
                v1500 = v1499 + v1497;
                float v1501;
                v1501 = v1483[v1500];
                int v1502;
                v1502 = v1484[v1500];
                bool v1503;
                v1503 = v1494 < v1502;
                float v1504; int v1505;
                if (v1503){
                    v1504 = v1493; v1505 = v1494;
                } else {
                    v1504 = v1501; v1505 = v1502;
                }
                v1493 = v1504;
                v1494 = v1505;
                v1497 += 1l ;
            }
            v1495 += 1l ;
        }
        auto v1506 = cooperative_groups::coalesced_threads();
        int v1507;
        v1507 = threadIdx.x;
        int v1508;
        v1508 = v1507 / 16l;
        auto v1509 = cooperative_groups::labeled_partition(v1506,v1508);
        Closure6 v1510{};
        float v1511; int v1512;
        Tuple1 tmp43 = cooperative_groups::reduce(v1509, Tuple1{v1493, v1494}, v1510);
        v1511 = tmp43.v0; v1512 = tmp43.v1;
        float v1513;
        v1513 = v1479 * v1511;
        int v1514[4l];
        bool v1515[4l];
        int v1516;
        v1516 = 0l;
        while (while_method_3(v1516)){
            int v1518;
            v1518 = 0l;
            while (while_method_1(v1518)){
                assert("Tensor range check" && 0 <= v1516 && v1516 < 1l);
                assert("Tensor range check" && 0 <= v1518 && v1518 < 4l);
                int v1520;
                v1520 = 4l * v1516;
                int v1521;
                v1521 = v1520 + v1518;
                float v1522;
                v1522 = v1445[v1521];
                bool v1523;
                v1523 = v1446[v1521];
                int v1524;
                v1524 = v1277[v1521];
                int v1527; bool v1528;
                if (v1523){
                    float v1525;
                    v1525 = v1522 - v1513;
                    bool v1526;
                    v1526 = v1525 >= 0.0f;
                    v1527 = v1524; v1528 = v1526;
                } else {
                    v1527 = 2147483647l; v1528 = false;
                }
                assert("Tensor range check" && 0 <= v1516 && v1516 < 1l);
                assert("Tensor range check" && 0 <= v1518 && v1518 < 4l);
                v1514[v1521] = v1527;
                v1515[v1521] = v1528;
                v1518 += 1l ;
            }
            v1516 += 1l ;
        }
        int v1529; bool v1530;
        Tuple4 tmp44 = Tuple4{2147483647l, false};
        v1529 = tmp44.v0; v1530 = tmp44.v1;
        int v1531;
        v1531 = 0l;
        while (while_method_3(v1531)){
            int v1533;
            v1533 = 0l;
            while (while_method_1(v1533)){
                assert("Tensor range check" && 0 <= v1531 && v1531 < 1l);
                assert("Tensor range check" && 0 <= v1533 && v1533 < 4l);
                int v1535;
                v1535 = 4l * v1531;
                int v1536;
                v1536 = v1535 + v1533;
                int v1537;
                v1537 = v1514[v1536];
                bool v1538;
                v1538 = v1515[v1536];
                int v1545; bool v1546;
                if (v1530){
                    if (v1538){
                        bool v1539;
                        v1539 = v1529 < v1537;
                        int v1540;
                        if (v1539){
                            v1540 = v1529;
                        } else {
                            v1540 = v1537;
                        }
                        v1545 = v1540; v1546 = true;
                    } else {
                        v1545 = v1529; v1546 = v1530;
                    }
                } else {
                    if (v1538){
                        v1545 = v1537; v1546 = v1538;
                    } else {
                        v1545 = v1529; v1546 = v1530;
                    }
                }
                v1529 = v1545;
                v1530 = v1546;
                v1533 += 1l ;
            }
            v1531 += 1l ;
        }
        auto v1547 = cooperative_groups::coalesced_threads();
        int v1548;
        v1548 = threadIdx.x;
        int v1549;
        v1549 = v1548 / 16l;
        auto v1550 = cooperative_groups::labeled_partition(v1547,v1549);
        Closure7 v1551{};
        int v1552; bool v1553;
        Tuple4 tmp45 = cooperative_groups::reduce(v1550, Tuple4{v1529, v1530}, v1551);
        v1552 = tmp45.v0; v1553 = tmp45.v1;
        bool v1554;
        v1554 = v1553 == false;
        if (v1554){
            assert("The local reduce must be true." && v1553);
        } else {
        }
        assert("Tensor range check" && 0 <= v1272 && v1272 < 64l);
        int v1556;
        v1556 = 0l;
        while (while_method_3(v1556)){
            assert("Tensor range check" && 0 <= v1556 && v1556 < 1l);
            int v1558;
            v1558 = 64l * v1556;
            int v1559;
            v1559 = v1558 + v1275;
            assert("Tensor range check" && 0 <= v1556 && v1556 < 1l);
            int v1560;
            v1560 = 4l * v1556;
            int4* v1561;
            v1561 = reinterpret_cast<int4*>(v1407 + v1560);
            int4* v1562;
            v1562 = reinterpret_cast<int4*>(v14 + v1559);
            assert("Pointer alignment check" && (unsigned long long)(v1561) % 4l == 0 && (unsigned long long)(v1562) % 4l == 0);
            *v1562 = *v1561;
            v1556 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1272 && v1272 < 64l);
        int v1563;
        v1563 = 2l * v1272;
        int v1564;
        v1564 = v1563 + v1265;
        v15[v1564] = v1552;
        v1272 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
extern "C" __global__ void entry2(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 32l);
    int v9;
    v9 = 16l * v8;
    int v10;
    v10 = threadIdx.x;
    assert("Tensor range check" && 0 <= v10 && v10 < 32l);
    int v11;
    v11 = 16l * v10;
    int v12;
    v12 = threadIdx.x;
    assert("Tensor range check" && 0 <= v12 && v12 < 32l);
    int v13;
    v13 = 16l * v12;
    int v14;
    v14 = threadIdx.x;
    assert("Tensor range check" && 0 <= v14 && v14 < 32l);
    int v15;
    v15 = 16l * v14;
    int v16;
    v16 = threadIdx.x;
    assert("Tensor range check" && 0 <= v16 && v16 < 32l);
    int v17;
    v17 = 16l * v16;
    float * v18;
    v18 = v1+v9;
    int * v20;
    v20 = v2+v15;
    int * v22;
    v22 = v3+v15;
    __shared__ float * v24[32l];
    __shared__ int * v25[32l];
    __shared__ int * v26[32l];
    /* void shared array create v27 */;
    /* void shared array create v28 */;
    int v29;
    v29 = threadIdx.x;
    bool v30;
    v30 = v29 < 32l;
    if (v30){
        assert("Tensor range check" && 0 <= v29 && v29 < 32l);
        v24[v29] = v18;
        v25[v29] = v20;
        v26[v29] = v22;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v31;
    v31 = 0l <= v29;
    bool v32;
    v32 = v31 == false;
    if (v32){
        assert("The index needs to be zero or positive." && v31);
    } else {
    }
    int v34;
    v34 = v29 % 4l;
    int v35;
    v35 = v29 / 4l;
    bool v36;
    v36 = v35 < 8l;
    bool v37;
    v37 = v36 == false;
    if (v37){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v36);
    } else {
    }
    assert("Tensor range check" && 0 <= v35 && v35 < 8l);
    int v39;
    v39 = 0l;
    while (while_method_1(v39)){
        bool v41;
        v41 = 0l <= v35;
        bool v42;
        v42 = v41 && v36;
        bool v43;
        v43 = v42 == false;
        if (v43){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v42);
        } else {
        }
        bool v45;
        v45 = 0l <= v39;
        bool v47;
        if (v45){
            bool v46;
            v46 = v39 < 4l;
            v47 = v46;
        } else {
            v47 = false;
        }
        bool v48;
        v48 = v47 == false;
        if (v48){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v47);
        } else {
        }
        int v50;
        v50 = v39 * 8l;
        int v51;
        v51 = v50 + v35;
        assert("Tensor range check" && 0 <= v39 && v39 < 4l);
        int v52;
        v52 = 8l * v39;
        int v53;
        v53 = v52 + v35;
        float * v54;
        v54 = v24[v53];
        int * v55;
        v55 = v25[v53];
        int * v56;
        v56 = v26[v53];
        /* void array index */;
        assert("Tensor range check" && 0 <= v34 && v34 < 4l);
        int v57;
        v57 = 4l * v34;
        float v58[4l];
        int v59[4l];
        int v60;
        v60 = 0l;
        while (while_method_3(v60)){
            assert("Tensor range check" && 0 <= v60 && v60 < 1l);
            int v62;
            v62 = 4l * v60;
            assert("Tensor range check" && 0 <= v60 && v60 < 1l);
            int v63;
            v63 = 16l * v60;
            int v64;
            v64 = v63 + v57;
            int4* v65;
            v65 = reinterpret_cast<int4*>(v54 + v64);
            int4* v66;
            v66 = reinterpret_cast<int4*>(v58 + v62);
            assert("Pointer alignment check" && (unsigned long long)(v65) % 4l == 0 && (unsigned long long)(v66) % 4l == 0);
            *v66 = *v65;
            v60 += 1l ;
        }
        int v67;
        v67 = 0l;
        while (while_method_3(v67)){
            int v69;
            v69 = 0l;
            while (while_method_1(v69)){
                bool v71;
                v71 = 0l <= v69;
                bool v73;
                if (v71){
                    bool v72;
                    v72 = v69 < 4l;
                    v73 = v72;
                } else {
                    v73 = false;
                }
                bool v74;
                v74 = v73 == false;
                if (v74){
                    assert("The indices should be inside the range of the dimension." && v73);
                } else {
                }
                bool v76;
                v76 = 0l <= v34;
                bool v78;
                if (v76){
                    bool v77;
                    v77 = v34 < 4l;
                    v78 = v77;
                } else {
                    v78 = false;
                }
                bool v79;
                v79 = v78 == false;
                if (v79){
                    assert("The indices should be inside the range of the dimension." && v78);
                } else {
                }
                int v81;
                v81 = v34 * 4l;
                int v82;
                v82 = v69 + v81;
                bool v83;
                v83 = 0l <= v67;
                bool v85;
                if (v83){
                    bool v84;
                    v84 = v67 < 1l;
                    v85 = v84;
                } else {
                    v85 = false;
                }
                bool v86;
                v86 = v85 == false;
                if (v86){
                    assert("The indices should be inside the range of the dimension." && v85);
                } else {
                }
                int v88;
                v88 = v67 * 16l;
                int v89;
                v89 = v82 + v88;
                assert("Tensor range check" && 0 <= v67 && v67 < 1l);
                assert("Tensor range check" && 0 <= v69 && v69 < 4l);
                int v90;
                v90 = 4l * v67;
                int v91;
                v91 = v90 + v69;
                v59[v91] = v89;
                v69 += 1l ;
            }
            v67 += 1l ;
        }
        int v92[4l];
        int v93[4l];
        int v94;
        v94 = 0l;
        while (while_method_3(v94)){
            int v96;
            v96 = 0l;
            while (while_method_1(v96)){
                assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                int v98;
                v98 = 4l * v94;
                int v99;
                v99 = v98 + v96;
                int v100;
                v100 = v59[v99];
                assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                v92[v99] = v51;
                v93[v99] = v100;
                v96 += 1l ;
            }
            v94 += 1l ;
        }
        int v101;
        v101 = 0l;
        while (while_method_3(v101)){
            assert("Tensor range check" && 0 <= v101 && v101 < 1l);
            int v103;
            v103 = 16l * v101;
            int v104;
            v104 = v103 + v57;
            assert("Tensor range check" && 0 <= v101 && v101 < 1l);
            int v105;
            v105 = 4l * v101;
            int4* v106;
            v106 = reinterpret_cast<int4*>(v92 + v105);
            int4* v107;
            v107 = reinterpret_cast<int4*>(v55 + v104);
            assert("Pointer alignment check" && (unsigned long long)(v106) % 4l == 0 && (unsigned long long)(v107) % 4l == 0);
            *v107 = *v106;
            int4* v108;
            v108 = reinterpret_cast<int4*>(v93 + v105);
            int4* v109;
            v109 = reinterpret_cast<int4*>(v56 + v104);
            assert("Pointer alignment check" && (unsigned long long)(v108) % 4l == 0 && (unsigned long long)(v109) % 4l == 0);
            *v109 = *v108;
            v101 += 1l ;
        }
        assert("Tensor range check" && 0 <= v51 && v51 < 32l);
        /* void array set */;
        v39 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    if (v30){
        assert("Tensor range check" && 0 <= v29 && v29 < 32l);
        /* void array index */;
    } else {
        /* void array create */
        /* void array index */;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v111;
    v111 = v1+v9;
    __shared__ float * v113[32l];
    /* void shared array create v114 */;
    __shared__ int v115[32l];
    int v116;
    v116 = threadIdx.x;
    bool v117;
    v117 = v116 < 32l;
    if (v117){
        assert("Tensor range check" && 0 <= v116 && v116 < 32l);
        v113[v116] = v111;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v118;
    v118 = 0l <= v116;
    bool v119;
    v119 = v118 == false;
    if (v119){
        assert("The index needs to be zero or positive." && v118);
    } else {
    }
    int v121;
    v121 = v116 % 4l;
    int v122;
    v122 = v116 / 4l;
    bool v123;
    v123 = v122 < 8l;
    bool v124;
    v124 = v123 == false;
    if (v124){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v123);
    } else {
    }
    assert("Tensor range check" && 0 <= v122 && v122 < 8l);
    int v126;
    v126 = 0l;
    while (while_method_1(v126)){
        bool v128;
        v128 = 0l <= v122;
        bool v129;
        v129 = v128 && v123;
        bool v130;
        v130 = v129 == false;
        if (v130){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v129);
        } else {
        }
        bool v132;
        v132 = 0l <= v126;
        bool v134;
        if (v132){
            bool v133;
            v133 = v126 < 4l;
            v134 = v133;
        } else {
            v134 = false;
        }
        bool v135;
        v135 = v134 == false;
        if (v135){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v134);
        } else {
        }
        int v137;
        v137 = v126 * 8l;
        int v138;
        v138 = v137 + v122;
        assert("Tensor range check" && 0 <= v126 && v126 < 4l);
        int v139;
        v139 = 8l * v126;
        int v140;
        v140 = v139 + v122;
        float * v141;
        v141 = v113[v140];
        /* void array index */;
        assert("Tensor range check" && 0 <= v121 && v121 < 4l);
        int v142;
        v142 = 4l * v121;
        float v143[4l];
        int v144[4l];
        int v145;
        v145 = 0l;
        while (while_method_3(v145)){
            assert("Tensor range check" && 0 <= v145 && v145 < 1l);
            int v147;
            v147 = 4l * v145;
            assert("Tensor range check" && 0 <= v145 && v145 < 1l);
            int v148;
            v148 = 16l * v145;
            int v149;
            v149 = v148 + v142;
            int4* v150;
            v150 = reinterpret_cast<int4*>(v141 + v149);
            int4* v151;
            v151 = reinterpret_cast<int4*>(v143 + v147);
            assert("Pointer alignment check" && (unsigned long long)(v150) % 4l == 0 && (unsigned long long)(v151) % 4l == 0);
            *v151 = *v150;
            v145 += 1l ;
        }
        int v152;
        v152 = 0l;
        while (while_method_3(v152)){
            int v154;
            v154 = 0l;
            while (while_method_1(v154)){
                bool v156;
                v156 = 0l <= v154;
                bool v158;
                if (v156){
                    bool v157;
                    v157 = v154 < 4l;
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
                bool v161;
                v161 = 0l <= v121;
                bool v163;
                if (v161){
                    bool v162;
                    v162 = v121 < 4l;
                    v163 = v162;
                } else {
                    v163 = false;
                }
                bool v164;
                v164 = v163 == false;
                if (v164){
                    assert("The indices should be inside the range of the dimension." && v163);
                } else {
                }
                int v166;
                v166 = v121 * 4l;
                int v167;
                v167 = v154 + v166;
                bool v168;
                v168 = 0l <= v152;
                bool v170;
                if (v168){
                    bool v169;
                    v169 = v152 < 1l;
                    v170 = v169;
                } else {
                    v170 = false;
                }
                bool v171;
                v171 = v170 == false;
                if (v171){
                    assert("The indices should be inside the range of the dimension." && v170);
                } else {
                }
                int v173;
                v173 = v152 * 16l;
                int v174;
                v174 = v167 + v173;
                assert("Tensor range check" && 0 <= v152 && v152 < 1l);
                assert("Tensor range check" && 0 <= v154 && v154 < 4l);
                int v175;
                v175 = 4l * v152;
                int v176;
                v176 = v175 + v154;
                v144[v176] = v174;
                v154 += 1l ;
            }
            v152 += 1l ;
        }
        int v177;
        v177 = 0l;
        while (while_method_3(v177)){
            assert("Tensor range check" && 0 <= v177 && v177 < 1l);
            assert("Tensor range check" && 0 <= v177 && v177 < 1l);
            v177 += 1l ;
        }
        assert("Tensor range check" && 0 <= v138 && v138 < 32l);
        v115[v138] = v138;
        v126 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v182;
    if (v117){
        assert("Tensor range check" && 0 <= v116 && v116 < 32l);
        int v179;
        v179 = v115[v116];
        v182 = v179;
    } else {
        int v180[1l];
        int v181;
        v181 = v180[0l];
        v182 = v181;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v183;
    v183 = threadIdx.x;
    assert("Tensor range check" && 0 <= v183 && v183 < 32l);
    v4[v183] = v182;
    float * v184;
    v184 = v1+v9;
    float * v186;
    v186 = v6+v17;
    __shared__ float * v188[32l];
    __shared__ float * v189[32l];
    /* void shared array create v190 */;
    /* void shared array create v191 */;
    int v192;
    v192 = threadIdx.x;
    bool v193;
    v193 = v192 < 32l;
    if (v193){
        assert("Tensor range check" && 0 <= v192 && v192 < 32l);
        v188[v192] = v184;
        v189[v192] = v186;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v194;
    v194 = 0l <= v192;
    bool v195;
    v195 = v194 == false;
    if (v195){
        assert("The index needs to be zero or positive." && v194);
    } else {
    }
    int v197;
    v197 = v192 % 4l;
    int v198;
    v198 = v192 / 4l;
    bool v199;
    v199 = v198 < 8l;
    bool v200;
    v200 = v199 == false;
    if (v200){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v199);
    } else {
    }
    assert("Tensor range check" && 0 <= v198 && v198 < 8l);
    int v202;
    v202 = 0l;
    while (while_method_1(v202)){
        bool v204;
        v204 = 0l <= v198;
        bool v205;
        v205 = v204 && v199;
        bool v206;
        v206 = v205 == false;
        if (v206){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v205);
        } else {
        }
        bool v208;
        v208 = 0l <= v202;
        bool v210;
        if (v208){
            bool v209;
            v209 = v202 < 4l;
            v210 = v209;
        } else {
            v210 = false;
        }
        bool v211;
        v211 = v210 == false;
        if (v211){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v210);
        } else {
        }
        int v213;
        v213 = v202 * 8l;
        int v214;
        v214 = v213 + v198;
        assert("Tensor range check" && 0 <= v202 && v202 < 4l);
        int v215;
        v215 = 8l * v202;
        int v216;
        v216 = v215 + v198;
        float * v217;
        v217 = v188[v216];
        float * v218;
        v218 = v189[v216];
        /* void array index */;
        assert("Tensor range check" && 0 <= v197 && v197 < 4l);
        int v219;
        v219 = 4l * v197;
        float v220[4l];
        int v221[4l];
        int v222;
        v222 = 0l;
        while (while_method_3(v222)){
            assert("Tensor range check" && 0 <= v222 && v222 < 1l);
            int v224;
            v224 = 4l * v222;
            assert("Tensor range check" && 0 <= v222 && v222 < 1l);
            int v225;
            v225 = 16l * v222;
            int v226;
            v226 = v225 + v219;
            int4* v227;
            v227 = reinterpret_cast<int4*>(v217 + v226);
            int4* v228;
            v228 = reinterpret_cast<int4*>(v220 + v224);
            assert("Pointer alignment check" && (unsigned long long)(v227) % 4l == 0 && (unsigned long long)(v228) % 4l == 0);
            *v228 = *v227;
            v222 += 1l ;
        }
        int v229;
        v229 = 0l;
        while (while_method_3(v229)){
            int v231;
            v231 = 0l;
            while (while_method_1(v231)){
                bool v233;
                v233 = 0l <= v231;
                bool v235;
                if (v233){
                    bool v234;
                    v234 = v231 < 4l;
                    v235 = v234;
                } else {
                    v235 = false;
                }
                bool v236;
                v236 = v235 == false;
                if (v236){
                    assert("The indices should be inside the range of the dimension." && v235);
                } else {
                }
                bool v238;
                v238 = 0l <= v197;
                bool v240;
                if (v238){
                    bool v239;
                    v239 = v197 < 4l;
                    v240 = v239;
                } else {
                    v240 = false;
                }
                bool v241;
                v241 = v240 == false;
                if (v241){
                    assert("The indices should be inside the range of the dimension." && v240);
                } else {
                }
                int v243;
                v243 = v197 * 4l;
                int v244;
                v244 = v231 + v243;
                bool v245;
                v245 = 0l <= v229;
                bool v247;
                if (v245){
                    bool v246;
                    v246 = v229 < 1l;
                    v247 = v246;
                } else {
                    v247 = false;
                }
                bool v248;
                v248 = v247 == false;
                if (v248){
                    assert("The indices should be inside the range of the dimension." && v247);
                } else {
                }
                int v250;
                v250 = v229 * 16l;
                int v251;
                v251 = v244 + v250;
                assert("Tensor range check" && 0 <= v229 && v229 < 1l);
                assert("Tensor range check" && 0 <= v231 && v231 < 4l);
                int v252;
                v252 = 4l * v229;
                int v253;
                v253 = v252 + v231;
                v221[v253] = v251;
                v231 += 1l ;
            }
            v229 += 1l ;
        }
        int v254;
        v254 = 0l;
        while (while_method_3(v254)){
            assert("Tensor range check" && 0 <= v254 && v254 < 1l);
            int v256;
            v256 = 16l * v254;
            int v257;
            v257 = v256 + v219;
            assert("Tensor range check" && 0 <= v254 && v254 < 1l);
            int v258;
            v258 = 4l * v254;
            int4* v259;
            v259 = reinterpret_cast<int4*>(v220 + v258);
            int4* v260;
            v260 = reinterpret_cast<int4*>(v218 + v257);
            assert("Pointer alignment check" && (unsigned long long)(v259) % 4l == 0 && (unsigned long long)(v260) % 4l == 0);
            *v260 = *v259;
            v254 += 1l ;
        }
        assert("Tensor range check" && 0 <= v214 && v214 < 32l);
        /* void array set */;
        v202 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    if (v193){
        assert("Tensor range check" && 0 <= v192 && v192 < 32l);
        /* void array index */;
    } else {
        /* void array create */
        /* void array index */;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v262;
    v262 = v1+v9;
    float * v264;
    v264 = v7+v13;
    __shared__ float * v266[32l];
    __shared__ float * v267[32l];
    /* void shared array create v268 */;
    /* void shared array create v269 */;
    int v270;
    v270 = threadIdx.x;
    bool v271;
    v271 = v270 < 32l;
    if (v271){
        assert("Tensor range check" && 0 <= v270 && v270 < 32l);
        v266[v270] = v262;
        v267[v270] = v264;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v272;
    v272 = 0l <= v270;
    bool v273;
    v273 = v272 == false;
    if (v273){
        assert("The index needs to be zero or positive." && v272);
    } else {
    }
    int v275;
    v275 = v270 % 4l;
    int v276;
    v276 = v270 / 4l;
    bool v277;
    v277 = v276 < 8l;
    bool v278;
    v278 = v277 == false;
    if (v278){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v277);
    } else {
    }
    assert("Tensor range check" && 0 <= v276 && v276 < 8l);
    int v280;
    v280 = 0l;
    while (while_method_1(v280)){
        bool v282;
        v282 = 0l <= v276;
        bool v283;
        v283 = v282 && v277;
        bool v284;
        v284 = v283 == false;
        if (v284){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v283);
        } else {
        }
        bool v286;
        v286 = 0l <= v280;
        bool v288;
        if (v286){
            bool v287;
            v287 = v280 < 4l;
            v288 = v287;
        } else {
            v288 = false;
        }
        bool v289;
        v289 = v288 == false;
        if (v289){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v288);
        } else {
        }
        int v291;
        v291 = v280 * 8l;
        int v292;
        v292 = v291 + v276;
        assert("Tensor range check" && 0 <= v280 && v280 < 4l);
        int v293;
        v293 = 8l * v280;
        int v294;
        v294 = v293 + v276;
        float * v295;
        v295 = v266[v294];
        float * v296;
        v296 = v267[v294];
        /* void array index */;
        assert("Tensor range check" && 0 <= v275 && v275 < 4l);
        int v297;
        v297 = 4l * v275;
        float v298[4l];
        int v299[4l];
        int v300;
        v300 = 0l;
        while (while_method_3(v300)){
            assert("Tensor range check" && 0 <= v300 && v300 < 1l);
            int v302;
            v302 = 4l * v300;
            assert("Tensor range check" && 0 <= v300 && v300 < 1l);
            int v303;
            v303 = 16l * v300;
            int v304;
            v304 = v303 + v297;
            int4* v305;
            v305 = reinterpret_cast<int4*>(v295 + v304);
            int4* v306;
            v306 = reinterpret_cast<int4*>(v298 + v302);
            assert("Pointer alignment check" && (unsigned long long)(v305) % 4l == 0 && (unsigned long long)(v306) % 4l == 0);
            *v306 = *v305;
            v300 += 1l ;
        }
        int v307;
        v307 = 0l;
        while (while_method_3(v307)){
            int v309;
            v309 = 0l;
            while (while_method_1(v309)){
                bool v311;
                v311 = 0l <= v309;
                bool v313;
                if (v311){
                    bool v312;
                    v312 = v309 < 4l;
                    v313 = v312;
                } else {
                    v313 = false;
                }
                bool v314;
                v314 = v313 == false;
                if (v314){
                    assert("The indices should be inside the range of the dimension." && v313);
                } else {
                }
                bool v316;
                v316 = 0l <= v275;
                bool v318;
                if (v316){
                    bool v317;
                    v317 = v275 < 4l;
                    v318 = v317;
                } else {
                    v318 = false;
                }
                bool v319;
                v319 = v318 == false;
                if (v319){
                    assert("The indices should be inside the range of the dimension." && v318);
                } else {
                }
                int v321;
                v321 = v275 * 4l;
                int v322;
                v322 = v309 + v321;
                bool v323;
                v323 = 0l <= v307;
                bool v325;
                if (v323){
                    bool v324;
                    v324 = v307 < 1l;
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
                v328 = v307 * 16l;
                int v329;
                v329 = v322 + v328;
                assert("Tensor range check" && 0 <= v307 && v307 < 1l);
                assert("Tensor range check" && 0 <= v309 && v309 < 4l);
                int v330;
                v330 = 4l * v307;
                int v331;
                v331 = v330 + v309;
                v299[v331] = v329;
                v309 += 1l ;
            }
            v307 += 1l ;
        }
        bool v332[4l];
        int v333;
        v333 = 0l;
        while (while_method_3(v333)){
            int v335;
            v335 = 0l;
            while (while_method_1(v335)){
                assert("Tensor range check" && 0 <= v333 && v333 < 1l);
                assert("Tensor range check" && 0 <= v335 && v335 < 4l);
                int v337;
                v337 = 4l * v333;
                int v338;
                v338 = v337 + v335;
                float v339;
                v339 = v298[v338];
                int v340;
                v340 = v299[v338];
                bool v341;
                v341 = v340 < 3l;
                assert("Tensor range check" && 0 <= v333 && v333 < 1l);
                assert("Tensor range check" && 0 <= v335 && v335 < 4l);
                v332[v338] = v341;
                v335 += 1l ;
            }
            v333 += 1l ;
        }
        float v342[4l];
        int v343;
        v343 = 0l;
        while (while_method_3(v343)){
            int v345;
            v345 = 0l;
            while (while_method_1(v345)){
                assert("Tensor range check" && 0 <= v343 && v343 < 1l);
                assert("Tensor range check" && 0 <= v345 && v345 < 4l);
                int v347;
                v347 = 4l * v343;
                int v348;
                v348 = v347 + v345;
                float v349;
                v349 = v298[v348];
                bool v350;
                v350 = v332[v348];
                float v353;
                if (v350){
                    bool v351;
                    v351 = 0.0f >= v349;
                    if (v351){
                        v353 = 0.0f;
                    } else {
                        v353 = v349;
                    }
                } else {
                    v353 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v343 && v343 < 1l);
                assert("Tensor range check" && 0 <= v345 && v345 < 4l);
                v342[v348] = v353;
                v345 += 1l ;
            }
            v343 += 1l ;
        }
        float v354;
        v354 = 0.0f;
        int v355;
        v355 = 0l;
        while (while_method_3(v355)){
            int v357;
            v357 = 0l;
            while (while_method_1(v357)){
                assert("Tensor range check" && 0 <= v355 && v355 < 1l);
                assert("Tensor range check" && 0 <= v357 && v357 < 4l);
                int v359;
                v359 = 4l * v355;
                int v360;
                v360 = v359 + v357;
                float v361;
                v361 = v342[v360];
                float v362;
                v362 = v354 + v361;
                v354 = v362;
                v357 += 1l ;
            }
            v355 += 1l ;
        }
        auto v363 = cooperative_groups::coalesced_threads();
        int v364;
        v364 = threadIdx.x;
        int v365;
        v365 = v364 / 4l;
        auto v366 = cooperative_groups::labeled_partition(v363,v365);
        Closure0 v367{};
        float v368;
        v368 = cooperative_groups::reduce(v366, v354, v367);
        int v369[4l];
        int v370;
        v370 = 0l;
        while (while_method_3(v370)){
            int v372;
            v372 = 0l;
            while (while_method_1(v372)){
                assert("Tensor range check" && 0 <= v370 && v370 < 1l);
                assert("Tensor range check" && 0 <= v372 && v372 < 4l);
                int v374;
                v374 = 4l * v370;
                int v375;
                v375 = v374 + v372;
                bool v376;
                v376 = v332[v375];
                int v377;
                if (v376){
                    v377 = 1l;
                } else {
                    v377 = 0l;
                }
                assert("Tensor range check" && 0 <= v370 && v370 < 1l);
                assert("Tensor range check" && 0 <= v372 && v372 < 4l);
                v369[v375] = v377;
                v372 += 1l ;
            }
            v370 += 1l ;
        }
        int v378;
        v378 = 0l;
        int v379;
        v379 = 0l;
        while (while_method_3(v379)){
            int v381;
            v381 = 0l;
            while (while_method_1(v381)){
                assert("Tensor range check" && 0 <= v379 && v379 < 1l);
                assert("Tensor range check" && 0 <= v381 && v381 < 4l);
                int v383;
                v383 = 4l * v379;
                int v384;
                v384 = v383 + v381;
                int v385;
                v385 = v369[v384];
                int v386;
                v386 = v378 + v385;
                v378 = v386;
                v381 += 1l ;
            }
            v379 += 1l ;
        }
        auto v387 = cooperative_groups::coalesced_threads();
        int v388;
        v388 = threadIdx.x;
        int v389;
        v389 = v388 / 4l;
        auto v390 = cooperative_groups::labeled_partition(v387,v389);
        Closure4 v391{};
        int v392;
        v392 = cooperative_groups::reduce(v390, v378, v391);
        float v393;
        v393 = (float)v392;
        float v394;
        v394 = 1.0f / v393;
        float v395[4l];
        int v396;
        v396 = 0l;
        while (while_method_3(v396)){
            int v398;
            v398 = 0l;
            while (while_method_1(v398)){
                assert("Tensor range check" && 0 <= v396 && v396 < 1l);
                assert("Tensor range check" && 0 <= v398 && v398 < 4l);
                int v400;
                v400 = 4l * v396;
                int v401;
                v401 = v400 + v398;
                float v402;
                v402 = v342[v401];
                bool v403;
                v403 = v332[v401];
                bool v404;
                v404 = v403 == false;
                float v409;
                if (v404){
                    v409 = 0.0f;
                } else {
                    bool v405;
                    v405 = v368 == 0.0f;
                    bool v406;
                    v406 = v405 != true;
                    if (v406){
                        float v407;
                        v407 = v402 / v368;
                        v409 = v407;
                    } else {
                        v409 = v394;
                    }
                }
                assert("Tensor range check" && 0 <= v396 && v396 < 1l);
                assert("Tensor range check" && 0 <= v398 && v398 < 4l);
                v395[v401] = v409;
                v398 += 1l ;
            }
            v396 += 1l ;
        }
        int v410;
        v410 = 0l;
        while (while_method_3(v410)){
            assert("Tensor range check" && 0 <= v410 && v410 < 1l);
            int v412;
            v412 = 16l * v410;
            int v413;
            v413 = v412 + v297;
            assert("Tensor range check" && 0 <= v410 && v410 < 1l);
            int v414;
            v414 = 4l * v410;
            int4* v415;
            v415 = reinterpret_cast<int4*>(v395 + v414);
            int4* v416;
            v416 = reinterpret_cast<int4*>(v296 + v413);
            assert("Pointer alignment check" && (unsigned long long)(v415) % 4l == 0 && (unsigned long long)(v416) % 4l == 0);
            *v416 = *v415;
            v410 += 1l ;
        }
        assert("Tensor range check" && 0 <= v292 && v292 < 32l);
        /* void array set */;
        v280 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    if (v271){
        assert("Tensor range check" && 0 <= v270 && v270 < 32l);
        /* void array index */;
    } else {
        /* void array create */
        /* void array index */;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v418;
    v418 = v1+v9;
    __shared__ float * v420[32l];
    /* void shared array create v421 */;
    __shared__ int v422[32l];
    int v423;
    v423 = threadIdx.x;
    bool v424;
    v424 = v423 < 32l;
    if (v424){
        assert("Tensor range check" && 0 <= v423 && v423 < 32l);
        v420[v423] = v418;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v425;
    v425 = 0l <= v423;
    bool v426;
    v426 = v425 == false;
    if (v426){
        assert("The index needs to be zero or positive." && v425);
    } else {
    }
    int v428;
    v428 = v423 % 4l;
    int v429;
    v429 = v423 / 4l;
    bool v430;
    v430 = v429 < 8l;
    bool v431;
    v431 = v430 == false;
    if (v431){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v430);
    } else {
    }
    assert("Tensor range check" && 0 <= v429 && v429 < 8l);
    int v433;
    v433 = 0l;
    while (while_method_1(v433)){
        bool v435;
        v435 = 0l <= v429;
        bool v436;
        v436 = v435 && v430;
        bool v437;
        v437 = v436 == false;
        if (v437){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v436);
        } else {
        }
        bool v439;
        v439 = 0l <= v433;
        bool v441;
        if (v439){
            bool v440;
            v440 = v433 < 4l;
            v441 = v440;
        } else {
            v441 = false;
        }
        bool v442;
        v442 = v441 == false;
        if (v442){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v441);
        } else {
        }
        int v444;
        v444 = v433 * 8l;
        int v445;
        v445 = v444 + v429;
        assert("Tensor range check" && 0 <= v433 && v433 < 4l);
        int v446;
        v446 = 8l * v433;
        int v447;
        v447 = v446 + v429;
        float * v448;
        v448 = v420[v447];
        /* void array index */;
        assert("Tensor range check" && 0 <= v428 && v428 < 4l);
        int v449;
        v449 = 4l * v428;
        float v450[4l];
        int v451[4l];
        int v452;
        v452 = 0l;
        while (while_method_3(v452)){
            assert("Tensor range check" && 0 <= v452 && v452 < 1l);
            int v454;
            v454 = 4l * v452;
            assert("Tensor range check" && 0 <= v452 && v452 < 1l);
            int v455;
            v455 = 16l * v452;
            int v456;
            v456 = v455 + v449;
            int4* v457;
            v457 = reinterpret_cast<int4*>(v448 + v456);
            int4* v458;
            v458 = reinterpret_cast<int4*>(v450 + v454);
            assert("Pointer alignment check" && (unsigned long long)(v457) % 4l == 0 && (unsigned long long)(v458) % 4l == 0);
            *v458 = *v457;
            v452 += 1l ;
        }
        int v459;
        v459 = 0l;
        while (while_method_3(v459)){
            int v461;
            v461 = 0l;
            while (while_method_1(v461)){
                bool v463;
                v463 = 0l <= v461;
                bool v465;
                if (v463){
                    bool v464;
                    v464 = v461 < 4l;
                    v465 = v464;
                } else {
                    v465 = false;
                }
                bool v466;
                v466 = v465 == false;
                if (v466){
                    assert("The indices should be inside the range of the dimension." && v465);
                } else {
                }
                bool v468;
                v468 = 0l <= v428;
                bool v470;
                if (v468){
                    bool v469;
                    v469 = v428 < 4l;
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
                v473 = v428 * 4l;
                int v474;
                v474 = v461 + v473;
                bool v475;
                v475 = 0l <= v459;
                bool v477;
                if (v475){
                    bool v476;
                    v476 = v459 < 1l;
                    v477 = v476;
                } else {
                    v477 = false;
                }
                bool v478;
                v478 = v477 == false;
                if (v478){
                    assert("The indices should be inside the range of the dimension." && v477);
                } else {
                }
                int v480;
                v480 = v459 * 16l;
                int v481;
                v481 = v474 + v480;
                assert("Tensor range check" && 0 <= v459 && v459 < 1l);
                assert("Tensor range check" && 0 <= v461 && v461 < 4l);
                int v482;
                v482 = 4l * v459;
                int v483;
                v483 = v482 + v461;
                v451[v483] = v481;
                v461 += 1l ;
            }
            v459 += 1l ;
        }
        int v484;
        v484 = threadIdx.x;
        unsigned long long v485;
        v485 = (unsigned long long)v484;
        curandStatePhilox4_32_10_t v486;
        curand_init(12344321ull,v485,0ull,&v486);
        bool v487[4l];
        int v488;
        v488 = 0l;
        while (while_method_3(v488)){
            int v490;
            v490 = 0l;
            while (while_method_1(v490)){
                assert("Tensor range check" && 0 <= v488 && v488 < 1l);
                assert("Tensor range check" && 0 <= v490 && v490 < 4l);
                int v492;
                v492 = 4l * v488;
                int v493;
                v493 = v492 + v490;
                float v494;
                v494 = v450[v493];
                int v495;
                v495 = v451[v493];
                bool v496;
                v496 = v495 < 3l;
                assert("Tensor range check" && 0 <= v488 && v488 < 1l);
                assert("Tensor range check" && 0 <= v490 && v490 < 4l);
                v487[v493] = v496;
                v490 += 1l ;
            }
            v488 += 1l ;
        }
        int v497[4l];
        int v498;
        v498 = 0l;
        while (while_method_3(v498)){
            int v500;
            v500 = 0l;
            while (while_method_1(v500)){
                assert("Tensor range check" && 0 <= v498 && v498 < 1l);
                assert("Tensor range check" && 0 <= v500 && v500 < 4l);
                int v502;
                v502 = 4l * v498;
                int v503;
                v503 = v502 + v500;
                bool v504;
                v504 = v487[v503];
                int v505;
                if (v504){
                    v505 = 1l;
                } else {
                    v505 = 0l;
                }
                assert("Tensor range check" && 0 <= v498 && v498 < 1l);
                assert("Tensor range check" && 0 <= v500 && v500 < 4l);
                v497[v503] = v505;
                v500 += 1l ;
            }
            v498 += 1l ;
        }
        int v506;
        v506 = 0l;
        int v507;
        v507 = 0l;
        while (while_method_3(v507)){
            int v509;
            v509 = 0l;
            while (while_method_1(v509)){
                assert("Tensor range check" && 0 <= v507 && v507 < 1l);
                assert("Tensor range check" && 0 <= v509 && v509 < 4l);
                int v511;
                v511 = 4l * v507;
                int v512;
                v512 = v511 + v509;
                int v513;
                v513 = v497[v512];
                int v514;
                v514 = v506 + v513;
                v506 = v514;
                v509 += 1l ;
            }
            v507 += 1l ;
        }
        auto v515 = cooperative_groups::coalesced_threads();
        int v516;
        v516 = threadIdx.x;
        int v517;
        v517 = v516 / 4l;
        auto v518 = cooperative_groups::labeled_partition(v515,v517);
        Closure4 v519{};
        int v520;
        v520 = cooperative_groups::reduce(v518, v506, v519);
        float v521[4l];
        int v522;
        v522 = 0l;
        while (while_method_3(v522)){
            int v524;
            v524 = 0l;
            while (while_method_1(v524)){
                assert("Tensor range check" && 0 <= v522 && v522 < 1l);
                assert("Tensor range check" && 0 <= v524 && v524 < 4l);
                int v526;
                v526 = 4l * v522;
                int v527;
                v527 = v526 + v524;
                float v528;
                v528 = v450[v527];
                bool v529;
                v529 = v487[v527];
                float v530;
                if (v529){
                    v530 = v528;
                } else {
                    v530 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v522 && v522 < 1l);
                assert("Tensor range check" && 0 <= v524 && v524 < 4l);
                v521[v527] = v530;
                v524 += 1l ;
            }
            v522 += 1l ;
        }
        float v531;
        v531 = 0.0f;
        int v532;
        v532 = 0l;
        while (while_method_3(v532)){
            int v534;
            v534 = 0l;
            while (while_method_1(v534)){
                assert("Tensor range check" && 0 <= v532 && v532 < 1l);
                assert("Tensor range check" && 0 <= v534 && v534 < 4l);
                int v536;
                v536 = 4l * v532;
                int v537;
                v537 = v536 + v534;
                float v538;
                v538 = v521[v537];
                float v539;
                v539 = v531 + v538;
                v531 = v539;
                v534 += 1l ;
            }
            v532 += 1l ;
        }
        auto v540 = cooperative_groups::coalesced_threads();
        int v541;
        v541 = threadIdx.x;
        int v542;
        v542 = v541 / 4l;
        auto v543 = cooperative_groups::labeled_partition(v540,v542);
        Closure0 v544{};
        float v545;
        v545 = cooperative_groups::reduce(v543, v531, v544);
        float v546;
        v546 = (float)v520;
        float v547;
        v547 = v545 / v546;
        float v548[4l];
        int v549;
        v549 = 0l;
        while (while_method_3(v549)){
            int v551;
            v551 = 0l;
            while (while_method_1(v551)){
                assert("Tensor range check" && 0 <= v549 && v549 < 1l);
                assert("Tensor range check" && 0 <= v551 && v551 < 4l);
                int v553;
                v553 = 4l * v549;
                int v554;
                v554 = v553 + v551;
                float v555;
                v555 = v450[v554];
                bool v556;
                v556 = v487[v554];
                float v557;
                if (v556){
                    v557 = v555;
                } else {
                    v557 = -1.0f / 0.0f;
                }
                float v558;
                v558 = v557 - v547;
                float v559;
                v559 = exp(v558);
                assert("Tensor range check" && 0 <= v549 && v549 < 1l);
                assert("Tensor range check" && 0 <= v551 && v551 < 4l);
                v548[v554] = v559;
                v551 += 1l ;
            }
            v549 += 1l ;
        }
        float v560;
        v560 = 0.0f;
        int v561;
        v561 = 0l;
        while (while_method_3(v561)){
            int v563;
            v563 = 0l;
            while (while_method_1(v563)){
                assert("Tensor range check" && 0 <= v561 && v561 < 1l);
                assert("Tensor range check" && 0 <= v563 && v563 < 4l);
                int v565;
                v565 = 4l * v561;
                int v566;
                v566 = v565 + v563;
                float v567;
                v567 = v548[v566];
                float v568;
                v568 = v560 + v567;
                v560 = v568;
                v563 += 1l ;
            }
            v561 += 1l ;
        }
        auto v569 = cooperative_groups::coalesced_threads();
        int v570;
        v570 = threadIdx.x;
        int v571;
        v571 = v570 / 4l;
        auto v572 = cooperative_groups::labeled_partition(v569,v571);
        float v573;
        v573 = cooperative_groups::reduce(v572, v560, v544);
        float v574[4l];
        int v575;
        v575 = 0l;
        while (while_method_3(v575)){
            int v577;
            v577 = 0l;
            while (while_method_1(v577)){
                assert("Tensor range check" && 0 <= v575 && v575 < 1l);
                assert("Tensor range check" && 0 <= v577 && v577 < 4l);
                int v579;
                v579 = 4l * v575;
                int v580;
                v580 = v579 + v577;
                float v581;
                v581 = v548[v580];
                float v582;
                v582 = v581 / v573;
                assert("Tensor range check" && 0 <= v575 && v575 < 1l);
                assert("Tensor range check" && 0 <= v577 && v577 < 4l);
                v574[v580] = v582;
                v577 += 1l ;
            }
            v575 += 1l ;
        }
        float v583[4l];
        float v584;
        v584 = 0.0f;
        int v585;
        v585 = 0l;
        while (while_method_3(v585)){
            assert("Tensor range check" && 0 <= v585 && v585 < 1l);
            int v587;
            v587 = 4l * v585;
            assert("Tensor range check" && 0 <= v585 && v585 < 1l);
            int v588; float v589;
            Tuple0 tmp46 = Tuple0{0l, 0.0f};
            v588 = tmp46.v0; v589 = tmp46.v1;
            while (while_method_1(v588)){
                assert("Tensor range check" && 0 <= v588 && v588 < 4l);
                int v591;
                v591 = v588 + v587;
                float v592;
                v592 = v574[v591];
                float v593;
                v593 = v589 + v592;
                v589 = v593;
                v588 += 1l ;
            }
            auto v594 = cooperative_groups::coalesced_threads();
            int v595;
            v595 = threadIdx.x;
            int v596;
            v596 = v595 / 4l;
            auto v597 = cooperative_groups::labeled_partition(v594,v596);
            Closure2 v598{};
            float v599;
            v599 = cooperative_groups::inclusive_scan(v597, v589, v598);
            float v600;
            v600 = v597.shfl_up(v599,1);
            bool v601;
            v601 = v597.thread_rank() == 0;
            float v602;
            if (v601){
                v602 = 0.0f;
            } else {
                v602 = v600;
            }
            float v603;
            v603 = v597.shfl(v599,v597.num_threads()-1);
            float v604;
            v604 = v584 + v602;
            int v605; float v606;
            Tuple0 tmp47 = Tuple0{0l, v604};
            v605 = tmp47.v0; v606 = tmp47.v1;
            while (while_method_1(v605)){
                assert("Tensor range check" && 0 <= v605 && v605 < 4l);
                int v608;
                v608 = v605 + v587;
                float v609;
                v609 = v574[v608];
                float v610;
                v610 = v606 + v609;
                assert("Tensor range check" && 0 <= v605 && v605 < 4l);
                v583[v608] = v610;
                v606 = v610;
                v605 += 1l ;
            }
            float v611;
            v611 = v584 + v603;
            v584 = v611;
            v585 += 1l ;
        }
        float v612[4l];
        bool v613[4l];
        int v614;
        v614 = 0l;
        while (while_method_3(v614)){
            int v616;
            v616 = 0l;
            while (while_method_1(v616)){
                assert("Tensor range check" && 0 <= v614 && v614 < 1l);
                assert("Tensor range check" && 0 <= v616 && v616 < 4l);
                int v618;
                v618 = 4l * v614;
                int v619;
                v619 = v618 + v616;
                float v620;
                v620 = v583[v619];
                float v621;
                v621 = v574[v619];
                bool v622;
                v622 = v621 > 0.0f;
                assert("Tensor range check" && 0 <= v614 && v614 < 1l);
                assert("Tensor range check" && 0 <= v616 && v616 < 4l);
                v612[v619] = v620;
                v613[v619] = v622;
                v616 += 1l ;
            }
            v614 += 1l ;
        }
        float v623; bool v624;
        Tuple3 tmp48 = Tuple3{-1.0f / 0.0f, false};
        v623 = tmp48.v0; v624 = tmp48.v1;
        int v625;
        v625 = 0l;
        while (while_method_3(v625)){
            int v627;
            v627 = 0l;
            while (while_method_1(v627)){
                assert("Tensor range check" && 0 <= v625 && v625 < 1l);
                assert("Tensor range check" && 0 <= v627 && v627 < 4l);
                int v629;
                v629 = 4l * v625;
                int v630;
                v630 = v629 + v627;
                float v631;
                v631 = v612[v630];
                bool v632;
                v632 = v613[v630];
                float v639; bool v640;
                if (v624){
                    if (v632){
                        bool v633;
                        v633 = v623 >= v631;
                        float v634;
                        if (v633){
                            v634 = v623;
                        } else {
                            v634 = v631;
                        }
                        v639 = v634; v640 = true;
                    } else {
                        v639 = v623; v640 = v624;
                    }
                } else {
                    if (v632){
                        v639 = v631; v640 = v632;
                    } else {
                        v639 = v623; v640 = v624;
                    }
                }
                v623 = v639;
                v624 = v640;
                v627 += 1l ;
            }
            v625 += 1l ;
        }
        auto v641 = cooperative_groups::coalesced_threads();
        int v642;
        v642 = threadIdx.x;
        int v643;
        v643 = v642 / 4l;
        auto v644 = cooperative_groups::labeled_partition(v641,v643);
        Closure5 v645{};
        float v646; bool v647;
        Tuple3 tmp49 = cooperative_groups::reduce(v644, Tuple3{v623, v624}, v645);
        v646 = tmp49.v0; v647 = tmp49.v1;
        bool v648;
        v648 = v647 == false;
        if (v648){
            assert("The local reduce must be true." && v647);
        } else {
        }
        float v650[4l];
        int v651[4l];
        int v652;
        v652 = 0l;
        while (while_method_3(v652)){
            int v654;
            v654 = 0l;
            while (while_method_1(v654)){
                assert("Tensor range check" && 0 <= v652 && v652 < 1l);
                assert("Tensor range check" && 0 <= v654 && v654 < 4l);
                int v656;
                v656 = 4l * v652;
                int v657;
                v657 = v656 + v654;
                int v658;
                v658 = v451[v657];
                float v659;
                v659 = curand_uniform(&v486);
                assert("Tensor range check" && 0 <= v652 && v652 < 1l);
                assert("Tensor range check" && 0 <= v654 && v654 < 4l);
                v650[v657] = v659;
                v651[v657] = v658;
                v654 += 1l ;
            }
            v652 += 1l ;
        }
        float v660; int v661;
        Tuple1 tmp50 = Tuple1{0.0f, 2147483647l};
        v660 = tmp50.v0; v661 = tmp50.v1;
        int v662;
        v662 = 0l;
        while (while_method_3(v662)){
            int v664;
            v664 = 0l;
            while (while_method_1(v664)){
                assert("Tensor range check" && 0 <= v662 && v662 < 1l);
                assert("Tensor range check" && 0 <= v664 && v664 < 4l);
                int v666;
                v666 = 4l * v662;
                int v667;
                v667 = v666 + v664;
                float v668;
                v668 = v650[v667];
                int v669;
                v669 = v651[v667];
                bool v670;
                v670 = v661 < v669;
                float v671; int v672;
                if (v670){
                    v671 = v660; v672 = v661;
                } else {
                    v671 = v668; v672 = v669;
                }
                v660 = v671;
                v661 = v672;
                v664 += 1l ;
            }
            v662 += 1l ;
        }
        auto v673 = cooperative_groups::coalesced_threads();
        int v674;
        v674 = threadIdx.x;
        int v675;
        v675 = v674 / 4l;
        auto v676 = cooperative_groups::labeled_partition(v673,v675);
        Closure6 v677{};
        float v678; int v679;
        Tuple1 tmp51 = cooperative_groups::reduce(v676, Tuple1{v660, v661}, v677);
        v678 = tmp51.v0; v679 = tmp51.v1;
        float v680;
        v680 = v646 * v678;
        int v681[4l];
        bool v682[4l];
        int v683;
        v683 = 0l;
        while (while_method_3(v683)){
            int v685;
            v685 = 0l;
            while (while_method_1(v685)){
                assert("Tensor range check" && 0 <= v683 && v683 < 1l);
                assert("Tensor range check" && 0 <= v685 && v685 < 4l);
                int v687;
                v687 = 4l * v683;
                int v688;
                v688 = v687 + v685;
                float v689;
                v689 = v612[v688];
                bool v690;
                v690 = v613[v688];
                int v691;
                v691 = v451[v688];
                int v694; bool v695;
                if (v690){
                    float v692;
                    v692 = v689 - v680;
                    bool v693;
                    v693 = v692 >= 0.0f;
                    v694 = v691; v695 = v693;
                } else {
                    v694 = 2147483647l; v695 = false;
                }
                assert("Tensor range check" && 0 <= v683 && v683 < 1l);
                assert("Tensor range check" && 0 <= v685 && v685 < 4l);
                v681[v688] = v694;
                v682[v688] = v695;
                v685 += 1l ;
            }
            v683 += 1l ;
        }
        int v696; bool v697;
        Tuple4 tmp52 = Tuple4{2147483647l, false};
        v696 = tmp52.v0; v697 = tmp52.v1;
        int v698;
        v698 = 0l;
        while (while_method_3(v698)){
            int v700;
            v700 = 0l;
            while (while_method_1(v700)){
                assert("Tensor range check" && 0 <= v698 && v698 < 1l);
                assert("Tensor range check" && 0 <= v700 && v700 < 4l);
                int v702;
                v702 = 4l * v698;
                int v703;
                v703 = v702 + v700;
                int v704;
                v704 = v681[v703];
                bool v705;
                v705 = v682[v703];
                int v712; bool v713;
                if (v697){
                    if (v705){
                        bool v706;
                        v706 = v696 < v704;
                        int v707;
                        if (v706){
                            v707 = v696;
                        } else {
                            v707 = v704;
                        }
                        v712 = v707; v713 = true;
                    } else {
                        v712 = v696; v713 = v697;
                    }
                } else {
                    if (v705){
                        v712 = v704; v713 = v705;
                    } else {
                        v712 = v696; v713 = v697;
                    }
                }
                v696 = v712;
                v697 = v713;
                v700 += 1l ;
            }
            v698 += 1l ;
        }
        auto v714 = cooperative_groups::coalesced_threads();
        int v715;
        v715 = threadIdx.x;
        int v716;
        v716 = v715 / 4l;
        auto v717 = cooperative_groups::labeled_partition(v714,v716);
        Closure7 v718{};
        int v719; bool v720;
        Tuple4 tmp53 = cooperative_groups::reduce(v717, Tuple4{v696, v697}, v718);
        v719 = tmp53.v0; v720 = tmp53.v1;
        bool v721;
        v721 = v720 == false;
        if (v721){
            assert("The local reduce must be true." && v720);
        } else {
        }
        int v723;
        v723 = 0l;
        while (while_method_3(v723)){
            assert("Tensor range check" && 0 <= v723 && v723 < 1l);
            assert("Tensor range check" && 0 <= v723 && v723 < 1l);
            v723 += 1l ;
        }
        assert("Tensor range check" && 0 <= v445 && v445 < 32l);
        v422[v445] = v719;
        v433 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v728;
    if (v424){
        assert("Tensor range check" && 0 <= v423 && v423 < 32l);
        int v725;
        v725 = v422[v423];
        v728 = v725;
    } else {
        int v726[1l];
        int v727;
        v727 = v726[0l];
        v728 = v727;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v729;
    v729 = threadIdx.x;
    assert("Tensor range check" && 0 <= v729 && v729 < 32l);
    v5[v729] = v728;
    return ;
}
extern "C" __global__ void entry3(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 32l);
    int v9;
    v9 = 256l * v8;
    int v10;
    v10 = threadIdx.x;
    assert("Tensor range check" && 0 <= v10 && v10 < 32l);
    int v11;
    v11 = 256l * v10;
    int v12;
    v12 = threadIdx.x;
    assert("Tensor range check" && 0 <= v12 && v12 < 32l);
    int v13;
    v13 = 256l * v12;
    int v14;
    v14 = threadIdx.x;
    assert("Tensor range check" && 0 <= v14 && v14 < 32l);
    int v15;
    v15 = 256l * v14;
    int v16;
    v16 = threadIdx.x;
    assert("Tensor range check" && 0 <= v16 && v16 < 32l);
    int v17;
    v17 = 256l * v16;
    float * v18;
    v18 = v1+v9;
    int * v20;
    v20 = v2+v15;
    int * v22;
    v22 = v3+v15;
    __shared__ float * v24[32l];
    __shared__ int * v25[32l];
    __shared__ int * v26[32l];
    /* void shared array create v27 */;
    /* void shared array create v28 */;
    int v29;
    v29 = threadIdx.x;
    bool v30;
    v30 = v29 < 32l;
    if (v30){
        assert("Tensor range check" && 0 <= v29 && v29 < 32l);
        v24[v29] = v18;
        v25[v29] = v20;
        v26[v29] = v22;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v31;
    v31 = 0l <= v29;
    bool v32;
    v32 = v31 == false;
    if (v32){
        assert("The index needs to be zero or positive." && v31);
    } else {
    }
    int v34;
    v34 = v29 % 32l;
    int v35;
    v35 = v29 / 32l;
    bool v36;
    v36 = v35 < 1l;
    bool v37;
    v37 = v36 == false;
    if (v37){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v36);
    } else {
    }
    assert("Tensor range check" && 0 <= v35 && v35 < 1l);
    int v39;
    v39 = 0l;
    while (while_method_4(v39)){
        bool v41;
        v41 = 0l <= v35;
        bool v42;
        v42 = v41 && v36;
        bool v43;
        v43 = v42 == false;
        if (v43){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v42);
        } else {
        }
        bool v45;
        v45 = 0l <= v39;
        bool v47;
        if (v45){
            bool v46;
            v46 = v39 < 32l;
            v47 = v46;
        } else {
            v47 = false;
        }
        bool v48;
        v48 = v47 == false;
        if (v48){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v47);
        } else {
        }
        int v50;
        v50 = v39 + v35;
        assert("Tensor range check" && 0 <= v39 && v39 < 32l);
        float * v51;
        v51 = v24[v50];
        int * v52;
        v52 = v25[v50];
        int * v53;
        v53 = v26[v50];
        /* void array index */;
        assert("Tensor range check" && 0 <= v34 && v34 < 32l);
        int v54;
        v54 = 4l * v34;
        float v55[8l];
        int v56[8l];
        int v57;
        v57 = 0l;
        while (while_method_5(v57)){
            assert("Tensor range check" && 0 <= v57 && v57 < 2l);
            int v59;
            v59 = 4l * v57;
            assert("Tensor range check" && 0 <= v57 && v57 < 2l);
            int v60;
            v60 = 128l * v57;
            int v61;
            v61 = v60 + v54;
            int4* v62;
            v62 = reinterpret_cast<int4*>(v51 + v61);
            int4* v63;
            v63 = reinterpret_cast<int4*>(v55 + v59);
            assert("Pointer alignment check" && (unsigned long long)(v62) % 4l == 0 && (unsigned long long)(v63) % 4l == 0);
            *v63 = *v62;
            v57 += 1l ;
        }
        int v64;
        v64 = 0l;
        while (while_method_5(v64)){
            int v66;
            v66 = 0l;
            while (while_method_1(v66)){
                bool v68;
                v68 = 0l <= v66;
                bool v70;
                if (v68){
                    bool v69;
                    v69 = v66 < 4l;
                    v70 = v69;
                } else {
                    v70 = false;
                }
                bool v71;
                v71 = v70 == false;
                if (v71){
                    assert("The indices should be inside the range of the dimension." && v70);
                } else {
                }
                bool v73;
                v73 = 0l <= v34;
                bool v75;
                if (v73){
                    bool v74;
                    v74 = v34 < 32l;
                    v75 = v74;
                } else {
                    v75 = false;
                }
                bool v76;
                v76 = v75 == false;
                if (v76){
                    assert("The indices should be inside the range of the dimension." && v75);
                } else {
                }
                int v78;
                v78 = v34 * 4l;
                int v79;
                v79 = v66 + v78;
                bool v80;
                v80 = 0l <= v64;
                bool v82;
                if (v80){
                    bool v81;
                    v81 = v64 < 2l;
                    v82 = v81;
                } else {
                    v82 = false;
                }
                bool v83;
                v83 = v82 == false;
                if (v83){
                    assert("The indices should be inside the range of the dimension." && v82);
                } else {
                }
                int v85;
                v85 = v64 * 128l;
                int v86;
                v86 = v79 + v85;
                assert("Tensor range check" && 0 <= v64 && v64 < 2l);
                assert("Tensor range check" && 0 <= v66 && v66 < 4l);
                int v87;
                v87 = 4l * v64;
                int v88;
                v88 = v87 + v66;
                v56[v88] = v86;
                v66 += 1l ;
            }
            v64 += 1l ;
        }
        int v89[8l];
        int v90[8l];
        int v91;
        v91 = 0l;
        while (while_method_5(v91)){
            int v93;
            v93 = 0l;
            while (while_method_1(v93)){
                assert("Tensor range check" && 0 <= v91 && v91 < 2l);
                assert("Tensor range check" && 0 <= v93 && v93 < 4l);
                int v95;
                v95 = 4l * v91;
                int v96;
                v96 = v95 + v93;
                int v97;
                v97 = v56[v96];
                assert("Tensor range check" && 0 <= v91 && v91 < 2l);
                assert("Tensor range check" && 0 <= v93 && v93 < 4l);
                v89[v96] = v50;
                v90[v96] = v97;
                v93 += 1l ;
            }
            v91 += 1l ;
        }
        int v98;
        v98 = 0l;
        while (while_method_5(v98)){
            assert("Tensor range check" && 0 <= v98 && v98 < 2l);
            int v100;
            v100 = 128l * v98;
            int v101;
            v101 = v100 + v54;
            assert("Tensor range check" && 0 <= v98 && v98 < 2l);
            int v102;
            v102 = 4l * v98;
            int4* v103;
            v103 = reinterpret_cast<int4*>(v89 + v102);
            int4* v104;
            v104 = reinterpret_cast<int4*>(v52 + v101);
            assert("Pointer alignment check" && (unsigned long long)(v103) % 4l == 0 && (unsigned long long)(v104) % 4l == 0);
            *v104 = *v103;
            int4* v105;
            v105 = reinterpret_cast<int4*>(v90 + v102);
            int4* v106;
            v106 = reinterpret_cast<int4*>(v53 + v101);
            assert("Pointer alignment check" && (unsigned long long)(v105) % 4l == 0 && (unsigned long long)(v106) % 4l == 0);
            *v106 = *v105;
            v98 += 1l ;
        }
        assert("Tensor range check" && 0 <= v50 && v50 < 32l);
        /* void array set */;
        v39 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    if (v30){
        assert("Tensor range check" && 0 <= v29 && v29 < 32l);
        /* void array index */;
    } else {
        /* void array create */
        /* void array index */;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v108;
    v108 = v1+v9;
    __shared__ float * v110[32l];
    /* void shared array create v111 */;
    __shared__ int v112[32l];
    int v113;
    v113 = threadIdx.x;
    bool v114;
    v114 = v113 < 32l;
    if (v114){
        assert("Tensor range check" && 0 <= v113 && v113 < 32l);
        v110[v113] = v108;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v115;
    v115 = 0l <= v113;
    bool v116;
    v116 = v115 == false;
    if (v116){
        assert("The index needs to be zero or positive." && v115);
    } else {
    }
    int v118;
    v118 = v113 % 32l;
    int v119;
    v119 = v113 / 32l;
    bool v120;
    v120 = v119 < 1l;
    bool v121;
    v121 = v120 == false;
    if (v121){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v120);
    } else {
    }
    assert("Tensor range check" && 0 <= v119 && v119 < 1l);
    int v123;
    v123 = 0l;
    while (while_method_4(v123)){
        bool v125;
        v125 = 0l <= v119;
        bool v126;
        v126 = v125 && v120;
        bool v127;
        v127 = v126 == false;
        if (v127){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v126);
        } else {
        }
        bool v129;
        v129 = 0l <= v123;
        bool v131;
        if (v129){
            bool v130;
            v130 = v123 < 32l;
            v131 = v130;
        } else {
            v131 = false;
        }
        bool v132;
        v132 = v131 == false;
        if (v132){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v131);
        } else {
        }
        int v134;
        v134 = v123 + v119;
        assert("Tensor range check" && 0 <= v123 && v123 < 32l);
        float * v135;
        v135 = v110[v134];
        /* void array index */;
        assert("Tensor range check" && 0 <= v118 && v118 < 32l);
        int v136;
        v136 = 4l * v118;
        float v137[8l];
        int v138[8l];
        int v139;
        v139 = 0l;
        while (while_method_5(v139)){
            assert("Tensor range check" && 0 <= v139 && v139 < 2l);
            int v141;
            v141 = 4l * v139;
            assert("Tensor range check" && 0 <= v139 && v139 < 2l);
            int v142;
            v142 = 128l * v139;
            int v143;
            v143 = v142 + v136;
            int4* v144;
            v144 = reinterpret_cast<int4*>(v135 + v143);
            int4* v145;
            v145 = reinterpret_cast<int4*>(v137 + v141);
            assert("Pointer alignment check" && (unsigned long long)(v144) % 4l == 0 && (unsigned long long)(v145) % 4l == 0);
            *v145 = *v144;
            v139 += 1l ;
        }
        int v146;
        v146 = 0l;
        while (while_method_5(v146)){
            int v148;
            v148 = 0l;
            while (while_method_1(v148)){
                bool v150;
                v150 = 0l <= v148;
                bool v152;
                if (v150){
                    bool v151;
                    v151 = v148 < 4l;
                    v152 = v151;
                } else {
                    v152 = false;
                }
                bool v153;
                v153 = v152 == false;
                if (v153){
                    assert("The indices should be inside the range of the dimension." && v152);
                } else {
                }
                bool v155;
                v155 = 0l <= v118;
                bool v157;
                if (v155){
                    bool v156;
                    v156 = v118 < 32l;
                    v157 = v156;
                } else {
                    v157 = false;
                }
                bool v158;
                v158 = v157 == false;
                if (v158){
                    assert("The indices should be inside the range of the dimension." && v157);
                } else {
                }
                int v160;
                v160 = v118 * 4l;
                int v161;
                v161 = v148 + v160;
                bool v162;
                v162 = 0l <= v146;
                bool v164;
                if (v162){
                    bool v163;
                    v163 = v146 < 2l;
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
                int v167;
                v167 = v146 * 128l;
                int v168;
                v168 = v161 + v167;
                assert("Tensor range check" && 0 <= v146 && v146 < 2l);
                assert("Tensor range check" && 0 <= v148 && v148 < 4l);
                int v169;
                v169 = 4l * v146;
                int v170;
                v170 = v169 + v148;
                v138[v170] = v168;
                v148 += 1l ;
            }
            v146 += 1l ;
        }
        int v171;
        v171 = 0l;
        while (while_method_5(v171)){
            assert("Tensor range check" && 0 <= v171 && v171 < 2l);
            assert("Tensor range check" && 0 <= v171 && v171 < 2l);
            v171 += 1l ;
        }
        assert("Tensor range check" && 0 <= v134 && v134 < 32l);
        v112[v134] = v134;
        v123 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v176;
    if (v114){
        assert("Tensor range check" && 0 <= v113 && v113 < 32l);
        int v173;
        v173 = v112[v113];
        v176 = v173;
    } else {
        int v174[1l];
        int v175;
        v175 = v174[0l];
        v176 = v175;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v177;
    v177 = threadIdx.x;
    assert("Tensor range check" && 0 <= v177 && v177 < 32l);
    v4[v177] = v176;
    float * v178;
    v178 = v1+v9;
    float * v180;
    v180 = v6+v17;
    __shared__ float * v182[32l];
    __shared__ float * v183[32l];
    /* void shared array create v184 */;
    /* void shared array create v185 */;
    int v186;
    v186 = threadIdx.x;
    bool v187;
    v187 = v186 < 32l;
    if (v187){
        assert("Tensor range check" && 0 <= v186 && v186 < 32l);
        v182[v186] = v178;
        v183[v186] = v180;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v188;
    v188 = 0l <= v186;
    bool v189;
    v189 = v188 == false;
    if (v189){
        assert("The index needs to be zero or positive." && v188);
    } else {
    }
    int v191;
    v191 = v186 % 32l;
    int v192;
    v192 = v186 / 32l;
    bool v193;
    v193 = v192 < 1l;
    bool v194;
    v194 = v193 == false;
    if (v194){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v193);
    } else {
    }
    assert("Tensor range check" && 0 <= v192 && v192 < 1l);
    int v196;
    v196 = 0l;
    while (while_method_4(v196)){
        bool v198;
        v198 = 0l <= v192;
        bool v199;
        v199 = v198 && v193;
        bool v200;
        v200 = v199 == false;
        if (v200){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v199);
        } else {
        }
        bool v202;
        v202 = 0l <= v196;
        bool v204;
        if (v202){
            bool v203;
            v203 = v196 < 32l;
            v204 = v203;
        } else {
            v204 = false;
        }
        bool v205;
        v205 = v204 == false;
        if (v205){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v204);
        } else {
        }
        int v207;
        v207 = v196 + v192;
        assert("Tensor range check" && 0 <= v196 && v196 < 32l);
        float * v208;
        v208 = v182[v207];
        float * v209;
        v209 = v183[v207];
        /* void array index */;
        assert("Tensor range check" && 0 <= v191 && v191 < 32l);
        int v210;
        v210 = 4l * v191;
        float v211[8l];
        int v212[8l];
        int v213;
        v213 = 0l;
        while (while_method_5(v213)){
            assert("Tensor range check" && 0 <= v213 && v213 < 2l);
            int v215;
            v215 = 4l * v213;
            assert("Tensor range check" && 0 <= v213 && v213 < 2l);
            int v216;
            v216 = 128l * v213;
            int v217;
            v217 = v216 + v210;
            int4* v218;
            v218 = reinterpret_cast<int4*>(v208 + v217);
            int4* v219;
            v219 = reinterpret_cast<int4*>(v211 + v215);
            assert("Pointer alignment check" && (unsigned long long)(v218) % 4l == 0 && (unsigned long long)(v219) % 4l == 0);
            *v219 = *v218;
            v213 += 1l ;
        }
        int v220;
        v220 = 0l;
        while (while_method_5(v220)){
            int v222;
            v222 = 0l;
            while (while_method_1(v222)){
                bool v224;
                v224 = 0l <= v222;
                bool v226;
                if (v224){
                    bool v225;
                    v225 = v222 < 4l;
                    v226 = v225;
                } else {
                    v226 = false;
                }
                bool v227;
                v227 = v226 == false;
                if (v227){
                    assert("The indices should be inside the range of the dimension." && v226);
                } else {
                }
                bool v229;
                v229 = 0l <= v191;
                bool v231;
                if (v229){
                    bool v230;
                    v230 = v191 < 32l;
                    v231 = v230;
                } else {
                    v231 = false;
                }
                bool v232;
                v232 = v231 == false;
                if (v232){
                    assert("The indices should be inside the range of the dimension." && v231);
                } else {
                }
                int v234;
                v234 = v191 * 4l;
                int v235;
                v235 = v222 + v234;
                bool v236;
                v236 = 0l <= v220;
                bool v238;
                if (v236){
                    bool v237;
                    v237 = v220 < 2l;
                    v238 = v237;
                } else {
                    v238 = false;
                }
                bool v239;
                v239 = v238 == false;
                if (v239){
                    assert("The indices should be inside the range of the dimension." && v238);
                } else {
                }
                int v241;
                v241 = v220 * 128l;
                int v242;
                v242 = v235 + v241;
                assert("Tensor range check" && 0 <= v220 && v220 < 2l);
                assert("Tensor range check" && 0 <= v222 && v222 < 4l);
                int v243;
                v243 = 4l * v220;
                int v244;
                v244 = v243 + v222;
                v212[v244] = v242;
                v222 += 1l ;
            }
            v220 += 1l ;
        }
        int v245;
        v245 = 0l;
        while (while_method_5(v245)){
            assert("Tensor range check" && 0 <= v245 && v245 < 2l);
            int v247;
            v247 = 128l * v245;
            int v248;
            v248 = v247 + v210;
            assert("Tensor range check" && 0 <= v245 && v245 < 2l);
            int v249;
            v249 = 4l * v245;
            int4* v250;
            v250 = reinterpret_cast<int4*>(v211 + v249);
            int4* v251;
            v251 = reinterpret_cast<int4*>(v209 + v248);
            assert("Pointer alignment check" && (unsigned long long)(v250) % 4l == 0 && (unsigned long long)(v251) % 4l == 0);
            *v251 = *v250;
            v245 += 1l ;
        }
        assert("Tensor range check" && 0 <= v207 && v207 < 32l);
        /* void array set */;
        v196 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    if (v187){
        assert("Tensor range check" && 0 <= v186 && v186 < 32l);
        /* void array index */;
    } else {
        /* void array create */
        /* void array index */;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v253;
    v253 = v1+v9;
    float * v255;
    v255 = v7+v13;
    __shared__ float * v257[32l];
    __shared__ float * v258[32l];
    /* void shared array create v259 */;
    /* void shared array create v260 */;
    int v261;
    v261 = threadIdx.x;
    bool v262;
    v262 = v261 < 32l;
    if (v262){
        assert("Tensor range check" && 0 <= v261 && v261 < 32l);
        v257[v261] = v253;
        v258[v261] = v255;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v263;
    v263 = 0l <= v261;
    bool v264;
    v264 = v263 == false;
    if (v264){
        assert("The index needs to be zero or positive." && v263);
    } else {
    }
    int v266;
    v266 = v261 % 32l;
    int v267;
    v267 = v261 / 32l;
    bool v268;
    v268 = v267 < 1l;
    bool v269;
    v269 = v268 == false;
    if (v269){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v268);
    } else {
    }
    assert("Tensor range check" && 0 <= v267 && v267 < 1l);
    int v271;
    v271 = 0l;
    while (while_method_4(v271)){
        bool v273;
        v273 = 0l <= v267;
        bool v274;
        v274 = v273 && v268;
        bool v275;
        v275 = v274 == false;
        if (v275){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v274);
        } else {
        }
        bool v277;
        v277 = 0l <= v271;
        bool v279;
        if (v277){
            bool v278;
            v278 = v271 < 32l;
            v279 = v278;
        } else {
            v279 = false;
        }
        bool v280;
        v280 = v279 == false;
        if (v280){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v279);
        } else {
        }
        int v282;
        v282 = v271 + v267;
        assert("Tensor range check" && 0 <= v271 && v271 < 32l);
        float * v283;
        v283 = v257[v282];
        float * v284;
        v284 = v258[v282];
        /* void array index */;
        assert("Tensor range check" && 0 <= v266 && v266 < 32l);
        int v285;
        v285 = 4l * v266;
        float v286[8l];
        int v287[8l];
        int v288;
        v288 = 0l;
        while (while_method_5(v288)){
            assert("Tensor range check" && 0 <= v288 && v288 < 2l);
            int v290;
            v290 = 4l * v288;
            assert("Tensor range check" && 0 <= v288 && v288 < 2l);
            int v291;
            v291 = 128l * v288;
            int v292;
            v292 = v291 + v285;
            int4* v293;
            v293 = reinterpret_cast<int4*>(v283 + v292);
            int4* v294;
            v294 = reinterpret_cast<int4*>(v286 + v290);
            assert("Pointer alignment check" && (unsigned long long)(v293) % 4l == 0 && (unsigned long long)(v294) % 4l == 0);
            *v294 = *v293;
            v288 += 1l ;
        }
        int v295;
        v295 = 0l;
        while (while_method_5(v295)){
            int v297;
            v297 = 0l;
            while (while_method_1(v297)){
                bool v299;
                v299 = 0l <= v297;
                bool v301;
                if (v299){
                    bool v300;
                    v300 = v297 < 4l;
                    v301 = v300;
                } else {
                    v301 = false;
                }
                bool v302;
                v302 = v301 == false;
                if (v302){
                    assert("The indices should be inside the range of the dimension." && v301);
                } else {
                }
                bool v304;
                v304 = 0l <= v266;
                bool v306;
                if (v304){
                    bool v305;
                    v305 = v266 < 32l;
                    v306 = v305;
                } else {
                    v306 = false;
                }
                bool v307;
                v307 = v306 == false;
                if (v307){
                    assert("The indices should be inside the range of the dimension." && v306);
                } else {
                }
                int v309;
                v309 = v266 * 4l;
                int v310;
                v310 = v297 + v309;
                bool v311;
                v311 = 0l <= v295;
                bool v313;
                if (v311){
                    bool v312;
                    v312 = v295 < 2l;
                    v313 = v312;
                } else {
                    v313 = false;
                }
                bool v314;
                v314 = v313 == false;
                if (v314){
                    assert("The indices should be inside the range of the dimension." && v313);
                } else {
                }
                int v316;
                v316 = v295 * 128l;
                int v317;
                v317 = v310 + v316;
                assert("Tensor range check" && 0 <= v295 && v295 < 2l);
                assert("Tensor range check" && 0 <= v297 && v297 < 4l);
                int v318;
                v318 = 4l * v295;
                int v319;
                v319 = v318 + v297;
                v287[v319] = v317;
                v297 += 1l ;
            }
            v295 += 1l ;
        }
        bool v320[8l];
        int v321;
        v321 = 0l;
        while (while_method_5(v321)){
            int v323;
            v323 = 0l;
            while (while_method_1(v323)){
                assert("Tensor range check" && 0 <= v321 && v321 < 2l);
                assert("Tensor range check" && 0 <= v323 && v323 < 4l);
                int v325;
                v325 = 4l * v321;
                int v326;
                v326 = v325 + v323;
                float v327;
                v327 = v286[v326];
                int v328;
                v328 = v287[v326];
                bool v329;
                v329 = v328 < 3l;
                assert("Tensor range check" && 0 <= v321 && v321 < 2l);
                assert("Tensor range check" && 0 <= v323 && v323 < 4l);
                v320[v326] = v329;
                v323 += 1l ;
            }
            v321 += 1l ;
        }
        float v330[8l];
        int v331;
        v331 = 0l;
        while (while_method_5(v331)){
            int v333;
            v333 = 0l;
            while (while_method_1(v333)){
                assert("Tensor range check" && 0 <= v331 && v331 < 2l);
                assert("Tensor range check" && 0 <= v333 && v333 < 4l);
                int v335;
                v335 = 4l * v331;
                int v336;
                v336 = v335 + v333;
                float v337;
                v337 = v286[v336];
                bool v338;
                v338 = v320[v336];
                float v341;
                if (v338){
                    bool v339;
                    v339 = 0.0f >= v337;
                    if (v339){
                        v341 = 0.0f;
                    } else {
                        v341 = v337;
                    }
                } else {
                    v341 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v331 && v331 < 2l);
                assert("Tensor range check" && 0 <= v333 && v333 < 4l);
                v330[v336] = v341;
                v333 += 1l ;
            }
            v331 += 1l ;
        }
        float v342;
        v342 = 0.0f;
        int v343;
        v343 = 0l;
        while (while_method_5(v343)){
            int v345;
            v345 = 0l;
            while (while_method_1(v345)){
                assert("Tensor range check" && 0 <= v343 && v343 < 2l);
                assert("Tensor range check" && 0 <= v345 && v345 < 4l);
                int v347;
                v347 = 4l * v343;
                int v348;
                v348 = v347 + v345;
                float v349;
                v349 = v330[v348];
                float v350;
                v350 = v342 + v349;
                v342 = v350;
                v345 += 1l ;
            }
            v343 += 1l ;
        }
        auto v351 = cooperative_groups::coalesced_threads();
        int v352;
        v352 = threadIdx.x;
        int v353;
        v353 = v352 / 32l;
        auto v354 = cooperative_groups::labeled_partition(v351,v353);
        Closure0 v355{};
        float v356;
        v356 = cooperative_groups::reduce(v354, v342, v355);
        int v357[8l];
        int v358;
        v358 = 0l;
        while (while_method_5(v358)){
            int v360;
            v360 = 0l;
            while (while_method_1(v360)){
                assert("Tensor range check" && 0 <= v358 && v358 < 2l);
                assert("Tensor range check" && 0 <= v360 && v360 < 4l);
                int v362;
                v362 = 4l * v358;
                int v363;
                v363 = v362 + v360;
                bool v364;
                v364 = v320[v363];
                int v365;
                if (v364){
                    v365 = 1l;
                } else {
                    v365 = 0l;
                }
                assert("Tensor range check" && 0 <= v358 && v358 < 2l);
                assert("Tensor range check" && 0 <= v360 && v360 < 4l);
                v357[v363] = v365;
                v360 += 1l ;
            }
            v358 += 1l ;
        }
        int v366;
        v366 = 0l;
        int v367;
        v367 = 0l;
        while (while_method_5(v367)){
            int v369;
            v369 = 0l;
            while (while_method_1(v369)){
                assert("Tensor range check" && 0 <= v367 && v367 < 2l);
                assert("Tensor range check" && 0 <= v369 && v369 < 4l);
                int v371;
                v371 = 4l * v367;
                int v372;
                v372 = v371 + v369;
                int v373;
                v373 = v357[v372];
                int v374;
                v374 = v366 + v373;
                v366 = v374;
                v369 += 1l ;
            }
            v367 += 1l ;
        }
        auto v375 = cooperative_groups::coalesced_threads();
        int v376;
        v376 = threadIdx.x;
        int v377;
        v377 = v376 / 32l;
        auto v378 = cooperative_groups::labeled_partition(v375,v377);
        Closure4 v379{};
        int v380;
        v380 = cooperative_groups::reduce(v378, v366, v379);
        float v381;
        v381 = (float)v380;
        float v382;
        v382 = 1.0f / v381;
        float v383[8l];
        int v384;
        v384 = 0l;
        while (while_method_5(v384)){
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
                v390 = v330[v389];
                bool v391;
                v391 = v320[v389];
                bool v392;
                v392 = v391 == false;
                float v397;
                if (v392){
                    v397 = 0.0f;
                } else {
                    bool v393;
                    v393 = v356 == 0.0f;
                    bool v394;
                    v394 = v393 != true;
                    if (v394){
                        float v395;
                        v395 = v390 / v356;
                        v397 = v395;
                    } else {
                        v397 = v382;
                    }
                }
                assert("Tensor range check" && 0 <= v384 && v384 < 2l);
                assert("Tensor range check" && 0 <= v386 && v386 < 4l);
                v383[v389] = v397;
                v386 += 1l ;
            }
            v384 += 1l ;
        }
        int v398;
        v398 = 0l;
        while (while_method_5(v398)){
            assert("Tensor range check" && 0 <= v398 && v398 < 2l);
            int v400;
            v400 = 128l * v398;
            int v401;
            v401 = v400 + v285;
            assert("Tensor range check" && 0 <= v398 && v398 < 2l);
            int v402;
            v402 = 4l * v398;
            int4* v403;
            v403 = reinterpret_cast<int4*>(v383 + v402);
            int4* v404;
            v404 = reinterpret_cast<int4*>(v284 + v401);
            assert("Pointer alignment check" && (unsigned long long)(v403) % 4l == 0 && (unsigned long long)(v404) % 4l == 0);
            *v404 = *v403;
            v398 += 1l ;
        }
        assert("Tensor range check" && 0 <= v282 && v282 < 32l);
        /* void array set */;
        v271 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    if (v262){
        assert("Tensor range check" && 0 <= v261 && v261 < 32l);
        /* void array index */;
    } else {
        /* void array create */
        /* void array index */;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v406;
    v406 = v1+v9;
    __shared__ float * v408[32l];
    /* void shared array create v409 */;
    __shared__ int v410[32l];
    int v411;
    v411 = threadIdx.x;
    bool v412;
    v412 = v411 < 32l;
    if (v412){
        assert("Tensor range check" && 0 <= v411 && v411 < 32l);
        v408[v411] = v406;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v413;
    v413 = 0l <= v411;
    bool v414;
    v414 = v413 == false;
    if (v414){
        assert("The index needs to be zero or positive." && v413);
    } else {
    }
    int v416;
    v416 = v411 % 32l;
    int v417;
    v417 = v411 / 32l;
    bool v418;
    v418 = v417 < 1l;
    bool v419;
    v419 = v418 == false;
    if (v419){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v418);
    } else {
    }
    assert("Tensor range check" && 0 <= v417 && v417 < 1l);
    int v421;
    v421 = 0l;
    while (while_method_4(v421)){
        bool v423;
        v423 = 0l <= v417;
        bool v424;
        v424 = v423 && v418;
        bool v425;
        v425 = v424 == false;
        if (v425){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v424);
        } else {
        }
        bool v427;
        v427 = 0l <= v421;
        bool v429;
        if (v427){
            bool v428;
            v428 = v421 < 32l;
            v429 = v428;
        } else {
            v429 = false;
        }
        bool v430;
        v430 = v429 == false;
        if (v430){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v429);
        } else {
        }
        int v432;
        v432 = v421 + v417;
        assert("Tensor range check" && 0 <= v421 && v421 < 32l);
        float * v433;
        v433 = v408[v432];
        /* void array index */;
        assert("Tensor range check" && 0 <= v416 && v416 < 32l);
        int v434;
        v434 = 4l * v416;
        float v435[8l];
        int v436[8l];
        int v437;
        v437 = 0l;
        while (while_method_5(v437)){
            assert("Tensor range check" && 0 <= v437 && v437 < 2l);
            int v439;
            v439 = 4l * v437;
            assert("Tensor range check" && 0 <= v437 && v437 < 2l);
            int v440;
            v440 = 128l * v437;
            int v441;
            v441 = v440 + v434;
            int4* v442;
            v442 = reinterpret_cast<int4*>(v433 + v441);
            int4* v443;
            v443 = reinterpret_cast<int4*>(v435 + v439);
            assert("Pointer alignment check" && (unsigned long long)(v442) % 4l == 0 && (unsigned long long)(v443) % 4l == 0);
            *v443 = *v442;
            v437 += 1l ;
        }
        int v444;
        v444 = 0l;
        while (while_method_5(v444)){
            int v446;
            v446 = 0l;
            while (while_method_1(v446)){
                bool v448;
                v448 = 0l <= v446;
                bool v450;
                if (v448){
                    bool v449;
                    v449 = v446 < 4l;
                    v450 = v449;
                } else {
                    v450 = false;
                }
                bool v451;
                v451 = v450 == false;
                if (v451){
                    assert("The indices should be inside the range of the dimension." && v450);
                } else {
                }
                bool v453;
                v453 = 0l <= v416;
                bool v455;
                if (v453){
                    bool v454;
                    v454 = v416 < 32l;
                    v455 = v454;
                } else {
                    v455 = false;
                }
                bool v456;
                v456 = v455 == false;
                if (v456){
                    assert("The indices should be inside the range of the dimension." && v455);
                } else {
                }
                int v458;
                v458 = v416 * 4l;
                int v459;
                v459 = v446 + v458;
                bool v460;
                v460 = 0l <= v444;
                bool v462;
                if (v460){
                    bool v461;
                    v461 = v444 < 2l;
                    v462 = v461;
                } else {
                    v462 = false;
                }
                bool v463;
                v463 = v462 == false;
                if (v463){
                    assert("The indices should be inside the range of the dimension." && v462);
                } else {
                }
                int v465;
                v465 = v444 * 128l;
                int v466;
                v466 = v459 + v465;
                assert("Tensor range check" && 0 <= v444 && v444 < 2l);
                assert("Tensor range check" && 0 <= v446 && v446 < 4l);
                int v467;
                v467 = 4l * v444;
                int v468;
                v468 = v467 + v446;
                v436[v468] = v466;
                v446 += 1l ;
            }
            v444 += 1l ;
        }
        int v469;
        v469 = threadIdx.x;
        unsigned long long v470;
        v470 = (unsigned long long)v469;
        curandStatePhilox4_32_10_t v471;
        curand_init(12344321ull,v470,0ull,&v471);
        bool v472[8l];
        int v473;
        v473 = 0l;
        while (while_method_5(v473)){
            int v475;
            v475 = 0l;
            while (while_method_1(v475)){
                assert("Tensor range check" && 0 <= v473 && v473 < 2l);
                assert("Tensor range check" && 0 <= v475 && v475 < 4l);
                int v477;
                v477 = 4l * v473;
                int v478;
                v478 = v477 + v475;
                float v479;
                v479 = v435[v478];
                int v480;
                v480 = v436[v478];
                bool v481;
                v481 = v480 < 3l;
                assert("Tensor range check" && 0 <= v473 && v473 < 2l);
                assert("Tensor range check" && 0 <= v475 && v475 < 4l);
                v472[v478] = v481;
                v475 += 1l ;
            }
            v473 += 1l ;
        }
        int v482[8l];
        int v483;
        v483 = 0l;
        while (while_method_5(v483)){
            int v485;
            v485 = 0l;
            while (while_method_1(v485)){
                assert("Tensor range check" && 0 <= v483 && v483 < 2l);
                assert("Tensor range check" && 0 <= v485 && v485 < 4l);
                int v487;
                v487 = 4l * v483;
                int v488;
                v488 = v487 + v485;
                bool v489;
                v489 = v472[v488];
                int v490;
                if (v489){
                    v490 = 1l;
                } else {
                    v490 = 0l;
                }
                assert("Tensor range check" && 0 <= v483 && v483 < 2l);
                assert("Tensor range check" && 0 <= v485 && v485 < 4l);
                v482[v488] = v490;
                v485 += 1l ;
            }
            v483 += 1l ;
        }
        int v491;
        v491 = 0l;
        int v492;
        v492 = 0l;
        while (while_method_5(v492)){
            int v494;
            v494 = 0l;
            while (while_method_1(v494)){
                assert("Tensor range check" && 0 <= v492 && v492 < 2l);
                assert("Tensor range check" && 0 <= v494 && v494 < 4l);
                int v496;
                v496 = 4l * v492;
                int v497;
                v497 = v496 + v494;
                int v498;
                v498 = v482[v497];
                int v499;
                v499 = v491 + v498;
                v491 = v499;
                v494 += 1l ;
            }
            v492 += 1l ;
        }
        auto v500 = cooperative_groups::coalesced_threads();
        int v501;
        v501 = threadIdx.x;
        int v502;
        v502 = v501 / 32l;
        auto v503 = cooperative_groups::labeled_partition(v500,v502);
        Closure4 v504{};
        int v505;
        v505 = cooperative_groups::reduce(v503, v491, v504);
        float v506[8l];
        int v507;
        v507 = 0l;
        while (while_method_5(v507)){
            int v509;
            v509 = 0l;
            while (while_method_1(v509)){
                assert("Tensor range check" && 0 <= v507 && v507 < 2l);
                assert("Tensor range check" && 0 <= v509 && v509 < 4l);
                int v511;
                v511 = 4l * v507;
                int v512;
                v512 = v511 + v509;
                float v513;
                v513 = v435[v512];
                bool v514;
                v514 = v472[v512];
                float v515;
                if (v514){
                    v515 = v513;
                } else {
                    v515 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v507 && v507 < 2l);
                assert("Tensor range check" && 0 <= v509 && v509 < 4l);
                v506[v512] = v515;
                v509 += 1l ;
            }
            v507 += 1l ;
        }
        float v516;
        v516 = 0.0f;
        int v517;
        v517 = 0l;
        while (while_method_5(v517)){
            int v519;
            v519 = 0l;
            while (while_method_1(v519)){
                assert("Tensor range check" && 0 <= v517 && v517 < 2l);
                assert("Tensor range check" && 0 <= v519 && v519 < 4l);
                int v521;
                v521 = 4l * v517;
                int v522;
                v522 = v521 + v519;
                float v523;
                v523 = v506[v522];
                float v524;
                v524 = v516 + v523;
                v516 = v524;
                v519 += 1l ;
            }
            v517 += 1l ;
        }
        auto v525 = cooperative_groups::coalesced_threads();
        int v526;
        v526 = threadIdx.x;
        int v527;
        v527 = v526 / 32l;
        auto v528 = cooperative_groups::labeled_partition(v525,v527);
        Closure0 v529{};
        float v530;
        v530 = cooperative_groups::reduce(v528, v516, v529);
        float v531;
        v531 = (float)v505;
        float v532;
        v532 = v530 / v531;
        float v533[8l];
        int v534;
        v534 = 0l;
        while (while_method_5(v534)){
            int v536;
            v536 = 0l;
            while (while_method_1(v536)){
                assert("Tensor range check" && 0 <= v534 && v534 < 2l);
                assert("Tensor range check" && 0 <= v536 && v536 < 4l);
                int v538;
                v538 = 4l * v534;
                int v539;
                v539 = v538 + v536;
                float v540;
                v540 = v435[v539];
                bool v541;
                v541 = v472[v539];
                float v542;
                if (v541){
                    v542 = v540;
                } else {
                    v542 = -1.0f / 0.0f;
                }
                float v543;
                v543 = v542 - v532;
                float v544;
                v544 = exp(v543);
                assert("Tensor range check" && 0 <= v534 && v534 < 2l);
                assert("Tensor range check" && 0 <= v536 && v536 < 4l);
                v533[v539] = v544;
                v536 += 1l ;
            }
            v534 += 1l ;
        }
        float v545;
        v545 = 0.0f;
        int v546;
        v546 = 0l;
        while (while_method_5(v546)){
            int v548;
            v548 = 0l;
            while (while_method_1(v548)){
                assert("Tensor range check" && 0 <= v546 && v546 < 2l);
                assert("Tensor range check" && 0 <= v548 && v548 < 4l);
                int v550;
                v550 = 4l * v546;
                int v551;
                v551 = v550 + v548;
                float v552;
                v552 = v533[v551];
                float v553;
                v553 = v545 + v552;
                v545 = v553;
                v548 += 1l ;
            }
            v546 += 1l ;
        }
        auto v554 = cooperative_groups::coalesced_threads();
        int v555;
        v555 = threadIdx.x;
        int v556;
        v556 = v555 / 32l;
        auto v557 = cooperative_groups::labeled_partition(v554,v556);
        float v558;
        v558 = cooperative_groups::reduce(v557, v545, v529);
        float v559[8l];
        int v560;
        v560 = 0l;
        while (while_method_5(v560)){
            int v562;
            v562 = 0l;
            while (while_method_1(v562)){
                assert("Tensor range check" && 0 <= v560 && v560 < 2l);
                assert("Tensor range check" && 0 <= v562 && v562 < 4l);
                int v564;
                v564 = 4l * v560;
                int v565;
                v565 = v564 + v562;
                float v566;
                v566 = v533[v565];
                float v567;
                v567 = v566 / v558;
                assert("Tensor range check" && 0 <= v560 && v560 < 2l);
                assert("Tensor range check" && 0 <= v562 && v562 < 4l);
                v559[v565] = v567;
                v562 += 1l ;
            }
            v560 += 1l ;
        }
        float v568[8l];
        float v569;
        v569 = 0.0f;
        int v570;
        v570 = 0l;
        while (while_method_5(v570)){
            assert("Tensor range check" && 0 <= v570 && v570 < 2l);
            int v572;
            v572 = 4l * v570;
            assert("Tensor range check" && 0 <= v570 && v570 < 2l);
            int v573; float v574;
            Tuple0 tmp54 = Tuple0{0l, 0.0f};
            v573 = tmp54.v0; v574 = tmp54.v1;
            while (while_method_1(v573)){
                assert("Tensor range check" && 0 <= v573 && v573 < 4l);
                int v576;
                v576 = v573 + v572;
                float v577;
                v577 = v559[v576];
                float v578;
                v578 = v574 + v577;
                v574 = v578;
                v573 += 1l ;
            }
            auto v579 = cooperative_groups::coalesced_threads();
            int v580;
            v580 = threadIdx.x;
            int v581;
            v581 = v580 / 32l;
            auto v582 = cooperative_groups::labeled_partition(v579,v581);
            Closure2 v583{};
            float v584;
            v584 = cooperative_groups::inclusive_scan(v582, v574, v583);
            float v585;
            v585 = v582.shfl_up(v584,1);
            bool v586;
            v586 = v582.thread_rank() == 0;
            float v587;
            if (v586){
                v587 = 0.0f;
            } else {
                v587 = v585;
            }
            float v588;
            v588 = v582.shfl(v584,v582.num_threads()-1);
            float v589;
            v589 = v569 + v587;
            int v590; float v591;
            Tuple0 tmp55 = Tuple0{0l, v589};
            v590 = tmp55.v0; v591 = tmp55.v1;
            while (while_method_1(v590)){
                assert("Tensor range check" && 0 <= v590 && v590 < 4l);
                int v593;
                v593 = v590 + v572;
                float v594;
                v594 = v559[v593];
                float v595;
                v595 = v591 + v594;
                assert("Tensor range check" && 0 <= v590 && v590 < 4l);
                v568[v593] = v595;
                v591 = v595;
                v590 += 1l ;
            }
            float v596;
            v596 = v569 + v588;
            v569 = v596;
            v570 += 1l ;
        }
        float v597[8l];
        bool v598[8l];
        int v599;
        v599 = 0l;
        while (while_method_5(v599)){
            int v601;
            v601 = 0l;
            while (while_method_1(v601)){
                assert("Tensor range check" && 0 <= v599 && v599 < 2l);
                assert("Tensor range check" && 0 <= v601 && v601 < 4l);
                int v603;
                v603 = 4l * v599;
                int v604;
                v604 = v603 + v601;
                float v605;
                v605 = v568[v604];
                float v606;
                v606 = v559[v604];
                bool v607;
                v607 = v606 > 0.0f;
                assert("Tensor range check" && 0 <= v599 && v599 < 2l);
                assert("Tensor range check" && 0 <= v601 && v601 < 4l);
                v597[v604] = v605;
                v598[v604] = v607;
                v601 += 1l ;
            }
            v599 += 1l ;
        }
        float v608; bool v609;
        Tuple3 tmp56 = Tuple3{-1.0f / 0.0f, false};
        v608 = tmp56.v0; v609 = tmp56.v1;
        int v610;
        v610 = 0l;
        while (while_method_5(v610)){
            int v612;
            v612 = 0l;
            while (while_method_1(v612)){
                assert("Tensor range check" && 0 <= v610 && v610 < 2l);
                assert("Tensor range check" && 0 <= v612 && v612 < 4l);
                int v614;
                v614 = 4l * v610;
                int v615;
                v615 = v614 + v612;
                float v616;
                v616 = v597[v615];
                bool v617;
                v617 = v598[v615];
                float v624; bool v625;
                if (v609){
                    if (v617){
                        bool v618;
                        v618 = v608 >= v616;
                        float v619;
                        if (v618){
                            v619 = v608;
                        } else {
                            v619 = v616;
                        }
                        v624 = v619; v625 = true;
                    } else {
                        v624 = v608; v625 = v609;
                    }
                } else {
                    if (v617){
                        v624 = v616; v625 = v617;
                    } else {
                        v624 = v608; v625 = v609;
                    }
                }
                v608 = v624;
                v609 = v625;
                v612 += 1l ;
            }
            v610 += 1l ;
        }
        auto v626 = cooperative_groups::coalesced_threads();
        int v627;
        v627 = threadIdx.x;
        int v628;
        v628 = v627 / 32l;
        auto v629 = cooperative_groups::labeled_partition(v626,v628);
        Closure5 v630{};
        float v631; bool v632;
        Tuple3 tmp57 = cooperative_groups::reduce(v629, Tuple3{v608, v609}, v630);
        v631 = tmp57.v0; v632 = tmp57.v1;
        bool v633;
        v633 = v632 == false;
        if (v633){
            assert("The local reduce must be true." && v632);
        } else {
        }
        float v635[8l];
        int v636[8l];
        int v637;
        v637 = 0l;
        while (while_method_5(v637)){
            int v639;
            v639 = 0l;
            while (while_method_1(v639)){
                assert("Tensor range check" && 0 <= v637 && v637 < 2l);
                assert("Tensor range check" && 0 <= v639 && v639 < 4l);
                int v641;
                v641 = 4l * v637;
                int v642;
                v642 = v641 + v639;
                int v643;
                v643 = v436[v642];
                float v644;
                v644 = curand_uniform(&v471);
                assert("Tensor range check" && 0 <= v637 && v637 < 2l);
                assert("Tensor range check" && 0 <= v639 && v639 < 4l);
                v635[v642] = v644;
                v636[v642] = v643;
                v639 += 1l ;
            }
            v637 += 1l ;
        }
        float v645; int v646;
        Tuple1 tmp58 = Tuple1{0.0f, 2147483647l};
        v645 = tmp58.v0; v646 = tmp58.v1;
        int v647;
        v647 = 0l;
        while (while_method_5(v647)){
            int v649;
            v649 = 0l;
            while (while_method_1(v649)){
                assert("Tensor range check" && 0 <= v647 && v647 < 2l);
                assert("Tensor range check" && 0 <= v649 && v649 < 4l);
                int v651;
                v651 = 4l * v647;
                int v652;
                v652 = v651 + v649;
                float v653;
                v653 = v635[v652];
                int v654;
                v654 = v636[v652];
                bool v655;
                v655 = v646 < v654;
                float v656; int v657;
                if (v655){
                    v656 = v645; v657 = v646;
                } else {
                    v656 = v653; v657 = v654;
                }
                v645 = v656;
                v646 = v657;
                v649 += 1l ;
            }
            v647 += 1l ;
        }
        auto v658 = cooperative_groups::coalesced_threads();
        int v659;
        v659 = threadIdx.x;
        int v660;
        v660 = v659 / 32l;
        auto v661 = cooperative_groups::labeled_partition(v658,v660);
        Closure6 v662{};
        float v663; int v664;
        Tuple1 tmp59 = cooperative_groups::reduce(v661, Tuple1{v645, v646}, v662);
        v663 = tmp59.v0; v664 = tmp59.v1;
        float v665;
        v665 = v631 * v663;
        int v666[8l];
        bool v667[8l];
        int v668;
        v668 = 0l;
        while (while_method_5(v668)){
            int v670;
            v670 = 0l;
            while (while_method_1(v670)){
                assert("Tensor range check" && 0 <= v668 && v668 < 2l);
                assert("Tensor range check" && 0 <= v670 && v670 < 4l);
                int v672;
                v672 = 4l * v668;
                int v673;
                v673 = v672 + v670;
                float v674;
                v674 = v597[v673];
                bool v675;
                v675 = v598[v673];
                int v676;
                v676 = v436[v673];
                int v679; bool v680;
                if (v675){
                    float v677;
                    v677 = v674 - v665;
                    bool v678;
                    v678 = v677 >= 0.0f;
                    v679 = v676; v680 = v678;
                } else {
                    v679 = 2147483647l; v680 = false;
                }
                assert("Tensor range check" && 0 <= v668 && v668 < 2l);
                assert("Tensor range check" && 0 <= v670 && v670 < 4l);
                v666[v673] = v679;
                v667[v673] = v680;
                v670 += 1l ;
            }
            v668 += 1l ;
        }
        int v681; bool v682;
        Tuple4 tmp60 = Tuple4{2147483647l, false};
        v681 = tmp60.v0; v682 = tmp60.v1;
        int v683;
        v683 = 0l;
        while (while_method_5(v683)){
            int v685;
            v685 = 0l;
            while (while_method_1(v685)){
                assert("Tensor range check" && 0 <= v683 && v683 < 2l);
                assert("Tensor range check" && 0 <= v685 && v685 < 4l);
                int v687;
                v687 = 4l * v683;
                int v688;
                v688 = v687 + v685;
                int v689;
                v689 = v666[v688];
                bool v690;
                v690 = v667[v688];
                int v697; bool v698;
                if (v682){
                    if (v690){
                        bool v691;
                        v691 = v681 < v689;
                        int v692;
                        if (v691){
                            v692 = v681;
                        } else {
                            v692 = v689;
                        }
                        v697 = v692; v698 = true;
                    } else {
                        v697 = v681; v698 = v682;
                    }
                } else {
                    if (v690){
                        v697 = v689; v698 = v690;
                    } else {
                        v697 = v681; v698 = v682;
                    }
                }
                v681 = v697;
                v682 = v698;
                v685 += 1l ;
            }
            v683 += 1l ;
        }
        auto v699 = cooperative_groups::coalesced_threads();
        int v700;
        v700 = threadIdx.x;
        int v701;
        v701 = v700 / 32l;
        auto v702 = cooperative_groups::labeled_partition(v699,v701);
        Closure7 v703{};
        int v704; bool v705;
        Tuple4 tmp61 = cooperative_groups::reduce(v702, Tuple4{v681, v682}, v703);
        v704 = tmp61.v0; v705 = tmp61.v1;
        bool v706;
        v706 = v705 == false;
        if (v706){
            assert("The local reduce must be true." && v705);
        } else {
        }
        int v708;
        v708 = 0l;
        while (while_method_5(v708)){
            assert("Tensor range check" && 0 <= v708 && v708 < 2l);
            assert("Tensor range check" && 0 <= v708 && v708 < 2l);
            v708 += 1l ;
        }
        assert("Tensor range check" && 0 <= v432 && v432 < 32l);
        v410[v432] = v704;
        v421 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v713;
    if (v412){
        assert("Tensor range check" && 0 <= v411 && v411 < 32l);
        int v710;
        v710 = v410[v411];
        v713 = v710;
    } else {
        int v711[1l];
        int v712;
        v712 = v711[0l];
        v713 = v712;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v714;
    v714 = threadIdx.x;
    assert("Tensor range check" && 0 <= v714 && v714 < 32l);
    v5[v714] = v713;
    return ;
}
extern "C" __global__ void entry4(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14) {
    auto v15 = cooperative_groups::this_grid();
    int v16;
    v16 = threadIdx.x;
    int v17;
    v17 = v16;
    while (while_method_0(v17)){
        bool v19;
        v19 = 0l <= v17;
        bool v20;
        v20 = v19 == false;
        if (v20){
            assert("The index needs to be zero or positive." && v19);
        } else {
        }
        int v22;
        v22 = v17 % 16l;
        int v23;
        v23 = v17 / 16l;
        bool v24;
        v24 = v23 < 128l;
        bool v25;
        v25 = v24 == false;
        if (v25){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
        } else {
        }
        assert("Tensor range check" && 0 <= v23 && v23 < 128l);
        assert("Tensor range check" && 0 <= v22 && v22 < 16l);
        int v27;
        v27 = 4l * v22;
        int v28;
        v28 = 64l * v23;
        int v29;
        v29 = v28 + v27;
        assert("Tensor range check" && 0 <= v23 && v23 < 128l);
        assert("Tensor range check" && 0 <= v22 && v22 < 16l);
        float v30[4l];
        float v31[4l];
        int4* v32;
        v32 = reinterpret_cast<int4*>(v1 + v29);
        int4* v33;
        v33 = reinterpret_cast<int4*>(v30 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v32) % 4l == 0 && (unsigned long long)(v33) % 4l == 0);
        *v33 = *v32;
        // Pushing the loop unrolling to: 0
        int v34;
        v34 = 0l;
        #pragma unroll
        while (while_method_1(v34)){
            assert("Tensor range check" && 0 <= v34 && v34 < 4l);
            float v36;
            v36 = v30[v34];
            float v37;
            v37 = 1.0f + v36;
            assert("Tensor range check" && 0 <= v34 && v34 < 4l);
            v31[v34] = v37;
            v34 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v38;
        v38 = reinterpret_cast<int4*>(v31 + 0l);
        int4* v39;
        v39 = reinterpret_cast<int4*>(v1 + v29);
        assert("Pointer alignment check" && (unsigned long long)(v38) % 4l == 0 && (unsigned long long)(v39) % 4l == 0);
        *v39 = *v38;
        v17 += 32l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v40;
    v40 = threadIdx.x;
    bool v41;
    v41 = 0l <= v40;
    bool v42;
    v42 = v41 == false;
    if (v42){
        assert("The index needs to be zero or positive." && v41);
    } else {
    }
    int v44;
    v44 = v40 % 16l;
    int v45;
    v45 = v40 / 16l;
    bool v46;
    v46 = v45 < 2l;
    bool v47;
    v47 = v46 == false;
    if (v47){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v46);
    } else {
    }
    assert("Tensor range check" && 0 <= v45 && v45 < 2l);
    assert("Tensor range check" && 0 <= v44 && v44 < 16l);
    int v49;
    v49 = 4l * v44;
    int v50;
    v50 = 64l * v45;
    int v51;
    v51 = v50 + v49;
    assert("Tensor range check" && 0 <= v45 && v45 < 2l);
    assert("Tensor range check" && 0 <= v44 && v44 < 16l);
    int v52;
    v52 = blockIdx.x;
    int v53;
    v53 = v52;
    while (while_method_2(v53)){
        bool v55;
        v55 = 0l <= v53;
        bool v56;
        v56 = v55 == false;
        if (v56){
            assert("The index needs to be zero or positive." && v55);
        } else {
        }
        bool v58;
        v58 = v53 < 64l;
        bool v59;
        v59 = v58 == false;
        if (v59){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v58);
        } else {
        }
        assert("Tensor range check" && 0 <= v53 && v53 < 64l);
        int v61;
        v61 = 128l * v53;
        int v62;
        v62 = v61 + v51;
        int v63[4l];
        int v64[4l];
        int v65;
        v65 = 0l;
        while (while_method_3(v65)){
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            int v67;
            v67 = 4l * v65;
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            int v68;
            v68 = 64l * v65;
            int v69;
            v69 = v68 + v62;
            int4* v70;
            v70 = reinterpret_cast<int4*>(v0 + v69);
            int4* v71;
            v71 = reinterpret_cast<int4*>(v63 + v67);
            assert("Pointer alignment check" && (unsigned long long)(v70) % 4l == 0 && (unsigned long long)(v71) % 4l == 0);
            *v71 = *v70;
            v65 += 1l ;
        }
        int v72;
        v72 = 0l;
        while (while_method_3(v72)){
            int v74;
            v74 = 0l;
            while (while_method_1(v74)){
                bool v76;
                v76 = 0l <= v74;
                bool v78;
                if (v76){
                    bool v77;
                    v77 = v74 < 4l;
                    v78 = v77;
                } else {
                    v78 = false;
                }
                bool v79;
                v79 = v78 == false;
                if (v79){
                    assert("The indices should be inside the range of the dimension." && v78);
                } else {
                }
                bool v81;
                v81 = 0l <= v44;
                bool v83;
                if (v81){
                    bool v82;
                    v82 = v44 < 16l;
                    v83 = v82;
                } else {
                    v83 = false;
                }
                bool v84;
                v84 = v83 == false;
                if (v84){
                    assert("The indices should be inside the range of the dimension." && v83);
                } else {
                }
                int v86;
                v86 = v44 * 4l;
                int v87;
                v87 = v74 + v86;
                bool v88;
                v88 = 0l <= v72;
                bool v90;
                if (v88){
                    bool v89;
                    v89 = v72 < 1l;
                    v90 = v89;
                } else {
                    v90 = false;
                }
                bool v91;
                v91 = v90 == false;
                if (v91){
                    assert("The indices should be inside the range of the dimension." && v90);
                } else {
                }
                int v93;
                v93 = v72 * 64l;
                int v94;
                v94 = v87 + v93;
                assert("Tensor range check" && 0 <= v72 && v72 < 1l);
                assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                int v95;
                v95 = 4l * v72;
                int v96;
                v96 = v95 + v74;
                v64[v96] = v94;
                v74 += 1l ;
            }
            v72 += 1l ;
        }
        bool v97;
        v97 = 0l <= v45;
        bool v98;
        v98 = v97 && v46;
        bool v99;
        v99 = v98 == false;
        if (v99){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v98);
        } else {
        }
        bool v101;
        v101 = v55 && v58;
        bool v102;
        v102 = v101 == false;
        if (v102){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v101);
        } else {
        }
        int v104;
        v104 = v53 * 2l;
        int v105;
        v105 = v104 + v45;
        assert("Tensor range check" && 0 <= v53 && v53 < 64l);
        int v106;
        v106 = 0l;
        while (while_method_3(v106)){
            assert("Tensor range check" && 0 <= v106 && v106 < 1l);
            int v108;
            v108 = 64l * v106;
            int v109;
            v109 = v108 + v62;
            assert("Tensor range check" && 0 <= v106 && v106 < 1l);
            int v110;
            v110 = 4l * v106;
            int4* v111;
            v111 = reinterpret_cast<int4*>(v63 + v110);
            int4* v112;
            v112 = reinterpret_cast<int4*>(v2 + v109);
            assert("Pointer alignment check" && (unsigned long long)(v111) % 4l == 0 && (unsigned long long)(v112) % 4l == 0);
            *v112 = *v111;
            v106 += 1l ;
        }
        v53 += 24l ;
    }
    v15.sync() ;
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
    v117 = v113 % 16l;
    int v118;
    v118 = v113 / 16l;
    bool v119;
    v119 = v118 < 2l;
    bool v120;
    v120 = v119 == false;
    if (v120){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v119);
    } else {
    }
    assert("Tensor range check" && 0 <= v118 && v118 < 2l);
    assert("Tensor range check" && 0 <= v117 && v117 < 16l);
    int v122;
    v122 = 4l * v117;
    int v123;
    v123 = 64l * v118;
    int v124;
    v124 = v123 + v122;
    assert("Tensor range check" && 0 <= v118 && v118 < 2l);
    assert("Tensor range check" && 0 <= v117 && v117 < 16l);
    int v125;
    v125 = blockIdx.x;
    int v126;
    v126 = v125;
    while (while_method_2(v126)){
        bool v128;
        v128 = 0l <= v126;
        bool v129;
        v129 = v128 == false;
        if (v129){
            assert("The index needs to be zero or positive." && v128);
        } else {
        }
        bool v131;
        v131 = v126 < 64l;
        bool v132;
        v132 = v131 == false;
        if (v132){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v131);
        } else {
        }
        assert("Tensor range check" && 0 <= v126 && v126 < 64l);
        int v134;
        v134 = 128l * v126;
        int v135;
        v135 = v134 + v124;
        float v136[4l];
        int v137[4l];
        int v138;
        v138 = 0l;
        while (while_method_3(v138)){
            assert("Tensor range check" && 0 <= v138 && v138 < 1l);
            int v140;
            v140 = 4l * v138;
            assert("Tensor range check" && 0 <= v138 && v138 < 1l);
            int v141;
            v141 = 64l * v138;
            int v142;
            v142 = v141 + v135;
            int4* v143;
            v143 = reinterpret_cast<int4*>(v1 + v142);
            int4* v144;
            v144 = reinterpret_cast<int4*>(v136 + v140);
            assert("Pointer alignment check" && (unsigned long long)(v143) % 4l == 0 && (unsigned long long)(v144) % 4l == 0);
            *v144 = *v143;
            v138 += 1l ;
        }
        int v145;
        v145 = 0l;
        while (while_method_3(v145)){
            int v147;
            v147 = 0l;
            while (while_method_1(v147)){
                bool v149;
                v149 = 0l <= v147;
                bool v151;
                if (v149){
                    bool v150;
                    v150 = v147 < 4l;
                    v151 = v150;
                } else {
                    v151 = false;
                }
                bool v152;
                v152 = v151 == false;
                if (v152){
                    assert("The indices should be inside the range of the dimension." && v151);
                } else {
                }
                bool v154;
                v154 = 0l <= v117;
                bool v156;
                if (v154){
                    bool v155;
                    v155 = v117 < 16l;
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
                int v159;
                v159 = v117 * 4l;
                int v160;
                v160 = v147 + v159;
                bool v161;
                v161 = 0l <= v145;
                bool v163;
                if (v161){
                    bool v162;
                    v162 = v145 < 1l;
                    v163 = v162;
                } else {
                    v163 = false;
                }
                bool v164;
                v164 = v163 == false;
                if (v164){
                    assert("The indices should be inside the range of the dimension." && v163);
                } else {
                }
                int v166;
                v166 = v145 * 64l;
                int v167;
                v167 = v160 + v166;
                assert("Tensor range check" && 0 <= v145 && v145 < 1l);
                assert("Tensor range check" && 0 <= v147 && v147 < 4l);
                int v168;
                v168 = 4l * v145;
                int v169;
                v169 = v168 + v147;
                v137[v169] = v167;
                v147 += 1l ;
            }
            v145 += 1l ;
        }
        bool v170;
        v170 = 0l <= v118;
        bool v171;
        v171 = v170 && v119;
        bool v172;
        v172 = v171 == false;
        if (v172){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v171);
        } else {
        }
        bool v174;
        v174 = v128 && v131;
        bool v175;
        v175 = v174 == false;
        if (v175){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v174);
        } else {
        }
        int v177;
        v177 = v126 * 2l;
        int v178;
        v178 = v177 + v118;
        int v179[4l];
        int v180[4l];
        int v181;
        v181 = 0l;
        while (while_method_3(v181)){
            int v183;
            v183 = 0l;
            while (while_method_1(v183)){
                assert("Tensor range check" && 0 <= v181 && v181 < 1l);
                assert("Tensor range check" && 0 <= v183 && v183 < 4l);
                int v185;
                v185 = 4l * v181;
                int v186;
                v186 = v185 + v183;
                int v187;
                v187 = v137[v186];
                assert("Tensor range check" && 0 <= v181 && v181 < 1l);
                assert("Tensor range check" && 0 <= v183 && v183 < 4l);
                v179[v186] = v178;
                v180[v186] = v187;
                v183 += 1l ;
            }
            v181 += 1l ;
        }
        assert("Tensor range check" && 0 <= v126 && v126 < 64l);
        int v188;
        v188 = 0l;
        while (while_method_3(v188)){
            assert("Tensor range check" && 0 <= v188 && v188 < 1l);
            int v190;
            v190 = 64l * v188;
            int v191;
            v191 = v190 + v135;
            assert("Tensor range check" && 0 <= v188 && v188 < 1l);
            int v192;
            v192 = 4l * v188;
            int4* v193;
            v193 = reinterpret_cast<int4*>(v179 + v192);
            int4* v194;
            v194 = reinterpret_cast<int4*>(v9 + v191);
            assert("Pointer alignment check" && (unsigned long long)(v193) % 4l == 0 && (unsigned long long)(v194) % 4l == 0);
            *v194 = *v193;
            int4* v195;
            v195 = reinterpret_cast<int4*>(v180 + v192);
            int4* v196;
            v196 = reinterpret_cast<int4*>(v10 + v191);
            assert("Pointer alignment check" && (unsigned long long)(v195) % 4l == 0 && (unsigned long long)(v196) % 4l == 0);
            *v196 = *v195;
            v188 += 1l ;
        }
        v126 += 24l ;
    }
    v15.sync() ;
    int v197;
    v197 = threadIdx.x;
    bool v198;
    v198 = 0l <= v197;
    bool v199;
    v199 = v198 == false;
    if (v199){
        assert("The index needs to be zero or positive." && v198);
    } else {
    }
    int v201;
    v201 = v197 % 16l;
    int v202;
    v202 = v197 / 16l;
    bool v203;
    v203 = v202 < 2l;
    bool v204;
    v204 = v203 == false;
    if (v204){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v203);
    } else {
    }
    assert("Tensor range check" && 0 <= v202 && v202 < 2l);
    assert("Tensor range check" && 0 <= v201 && v201 < 16l);
    int v206;
    v206 = 4l * v201;
    int v207;
    v207 = 64l * v202;
    int v208;
    v208 = v207 + v206;
    assert("Tensor range check" && 0 <= v202 && v202 < 2l);
    int v209;
    v209 = blockIdx.x;
    int v210;
    v210 = v209;
    while (while_method_2(v210)){
        bool v212;
        v212 = 0l <= v210;
        bool v213;
        v213 = v212 == false;
        if (v213){
            assert("The index needs to be zero or positive." && v212);
        } else {
        }
        bool v215;
        v215 = v210 < 64l;
        bool v216;
        v216 = v215 == false;
        if (v216){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v215);
        } else {
        }
        assert("Tensor range check" && 0 <= v210 && v210 < 64l);
        int v218;
        v218 = 128l * v210;
        int v219;
        v219 = v218 + v208;
        float v220[4l];
        int v221[4l];
        int v222;
        v222 = 0l;
        while (while_method_3(v222)){
            assert("Tensor range check" && 0 <= v222 && v222 < 1l);
            int v224;
            v224 = 4l * v222;
            assert("Tensor range check" && 0 <= v222 && v222 < 1l);
            int v225;
            v225 = 64l * v222;
            int v226;
            v226 = v225 + v219;
            int4* v227;
            v227 = reinterpret_cast<int4*>(v1 + v226);
            int4* v228;
            v228 = reinterpret_cast<int4*>(v220 + v224);
            assert("Pointer alignment check" && (unsigned long long)(v227) % 4l == 0 && (unsigned long long)(v228) % 4l == 0);
            *v228 = *v227;
            v222 += 1l ;
        }
        int v229;
        v229 = 0l;
        while (while_method_3(v229)){
            int v231;
            v231 = 0l;
            while (while_method_1(v231)){
                bool v233;
                v233 = 0l <= v231;
                bool v235;
                if (v233){
                    bool v234;
                    v234 = v231 < 4l;
                    v235 = v234;
                } else {
                    v235 = false;
                }
                bool v236;
                v236 = v235 == false;
                if (v236){
                    assert("The indices should be inside the range of the dimension." && v235);
                } else {
                }
                bool v238;
                v238 = 0l <= v201;
                bool v240;
                if (v238){
                    bool v239;
                    v239 = v201 < 16l;
                    v240 = v239;
                } else {
                    v240 = false;
                }
                bool v241;
                v241 = v240 == false;
                if (v241){
                    assert("The indices should be inside the range of the dimension." && v240);
                } else {
                }
                int v243;
                v243 = v201 * 4l;
                int v244;
                v244 = v231 + v243;
                bool v245;
                v245 = 0l <= v229;
                bool v247;
                if (v245){
                    bool v246;
                    v246 = v229 < 1l;
                    v247 = v246;
                } else {
                    v247 = false;
                }
                bool v248;
                v248 = v247 == false;
                if (v248){
                    assert("The indices should be inside the range of the dimension." && v247);
                } else {
                }
                int v250;
                v250 = v229 * 64l;
                int v251;
                v251 = v244 + v250;
                assert("Tensor range check" && 0 <= v229 && v229 < 1l);
                assert("Tensor range check" && 0 <= v231 && v231 < 4l);
                int v252;
                v252 = 4l * v229;
                int v253;
                v253 = v252 + v231;
                v221[v253] = v251;
                v231 += 1l ;
            }
            v229 += 1l ;
        }
        bool v254;
        v254 = 0l <= v202;
        bool v255;
        v255 = v254 && v203;
        bool v256;
        v256 = v255 == false;
        if (v256){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v255);
        } else {
        }
        bool v258;
        v258 = v212 && v215;
        bool v259;
        v259 = v258 == false;
        if (v259){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v258);
        } else {
        }
        int v261;
        v261 = v210 * 2l;
        int v262;
        v262 = v261 + v202;
        assert("Tensor range check" && 0 <= v210 && v210 < 64l);
        int v263;
        v263 = 2l * v210;
        int v264;
        v264 = v263 + v202;
        v11[v264] = v262;
        v210 += 24l ;
    }
    v15.sync() ;
    int v265;
    v265 = threadIdx.x;
    bool v266;
    v266 = 0l <= v265;
    bool v267;
    v267 = v266 == false;
    if (v267){
        assert("The index needs to be zero or positive." && v266);
    } else {
    }
    int v269;
    v269 = v265 % 16l;
    int v270;
    v270 = v265 / 16l;
    bool v271;
    v271 = v270 < 2l;
    bool v272;
    v272 = v271 == false;
    if (v272){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v271);
    } else {
    }
    assert("Tensor range check" && 0 <= v270 && v270 < 2l);
    assert("Tensor range check" && 0 <= v269 && v269 < 16l);
    int v274;
    v274 = 4l * v269;
    int v275;
    v275 = 64l * v270;
    int v276;
    v276 = v275 + v274;
    assert("Tensor range check" && 0 <= v270 && v270 < 2l);
    assert("Tensor range check" && 0 <= v269 && v269 < 16l);
    int v277;
    v277 = blockIdx.x;
    int v278;
    v278 = v277;
    while (while_method_2(v278)){
        bool v280;
        v280 = 0l <= v278;
        bool v281;
        v281 = v280 == false;
        if (v281){
            assert("The index needs to be zero or positive." && v280);
        } else {
        }
        bool v283;
        v283 = v278 < 64l;
        bool v284;
        v284 = v283 == false;
        if (v284){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v283);
        } else {
        }
        assert("Tensor range check" && 0 <= v278 && v278 < 64l);
        int v286;
        v286 = 128l * v278;
        int v287;
        v287 = v286 + v276;
        float v288[4l];
        int v289[4l];
        int v290;
        v290 = 0l;
        while (while_method_3(v290)){
            assert("Tensor range check" && 0 <= v290 && v290 < 1l);
            int v292;
            v292 = 4l * v290;
            assert("Tensor range check" && 0 <= v290 && v290 < 1l);
            int v293;
            v293 = 64l * v290;
            int v294;
            v294 = v293 + v287;
            int4* v295;
            v295 = reinterpret_cast<int4*>(v1 + v294);
            int4* v296;
            v296 = reinterpret_cast<int4*>(v288 + v292);
            assert("Pointer alignment check" && (unsigned long long)(v295) % 4l == 0 && (unsigned long long)(v296) % 4l == 0);
            *v296 = *v295;
            v290 += 1l ;
        }
        int v297;
        v297 = 0l;
        while (while_method_3(v297)){
            int v299;
            v299 = 0l;
            while (while_method_1(v299)){
                bool v301;
                v301 = 0l <= v299;
                bool v303;
                if (v301){
                    bool v302;
                    v302 = v299 < 4l;
                    v303 = v302;
                } else {
                    v303 = false;
                }
                bool v304;
                v304 = v303 == false;
                if (v304){
                    assert("The indices should be inside the range of the dimension." && v303);
                } else {
                }
                bool v306;
                v306 = 0l <= v269;
                bool v308;
                if (v306){
                    bool v307;
                    v307 = v269 < 16l;
                    v308 = v307;
                } else {
                    v308 = false;
                }
                bool v309;
                v309 = v308 == false;
                if (v309){
                    assert("The indices should be inside the range of the dimension." && v308);
                } else {
                }
                int v311;
                v311 = v269 * 4l;
                int v312;
                v312 = v299 + v311;
                bool v313;
                v313 = 0l <= v297;
                bool v315;
                if (v313){
                    bool v314;
                    v314 = v297 < 1l;
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
                int v318;
                v318 = v297 * 64l;
                int v319;
                v319 = v312 + v318;
                assert("Tensor range check" && 0 <= v297 && v297 < 1l);
                assert("Tensor range check" && 0 <= v299 && v299 < 4l);
                int v320;
                v320 = 4l * v297;
                int v321;
                v321 = v320 + v299;
                v289[v321] = v319;
                v299 += 1l ;
            }
            v297 += 1l ;
        }
        bool v322;
        v322 = 0l <= v270;
        bool v323;
        v323 = v322 && v271;
        bool v324;
        v324 = v323 == false;
        if (v324){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v323);
        } else {
        }
        bool v326;
        v326 = v280 && v283;
        bool v327;
        v327 = v326 == false;
        if (v327){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v326);
        } else {
        }
        int v329;
        v329 = v278 * 2l;
        int v330;
        v330 = v329 + v270;
        float v331;
        v331 = 0.0f;
        int v332;
        v332 = 0l;
        while (while_method_3(v332)){
            int v334;
            v334 = 0l;
            while (while_method_1(v334)){
                assert("Tensor range check" && 0 <= v332 && v332 < 1l);
                assert("Tensor range check" && 0 <= v334 && v334 < 4l);
                int v336;
                v336 = 4l * v332;
                int v337;
                v337 = v336 + v334;
                float v338;
                v338 = v288[v337];
                float v339;
                v339 = v331 + v338;
                v331 = v339;
                v334 += 1l ;
            }
            v332 += 1l ;
        }
        auto v340 = cooperative_groups::coalesced_threads();
        int v341;
        v341 = threadIdx.x;
        int v342;
        v342 = v341 / 16l;
        auto v343 = cooperative_groups::labeled_partition(v340,v342);
        Closure0 v344{};
        float v345;
        v345 = cooperative_groups::reduce(v343, v331, v344);
        float v346;
        v346 = v345 / 64.0f;
        float v347[4l];
        int v348;
        v348 = 0l;
        while (while_method_3(v348)){
            int v350;
            v350 = 0l;
            while (while_method_1(v350)){
                assert("Tensor range check" && 0 <= v348 && v348 < 1l);
                assert("Tensor range check" && 0 <= v350 && v350 < 4l);
                int v352;
                v352 = 4l * v348;
                int v353;
                v353 = v352 + v350;
                float v354;
                v354 = v288[v353];
                float v355;
                v355 = v354 - v346;
                float v356;
                v356 = exp(v355);
                assert("Tensor range check" && 0 <= v348 && v348 < 1l);
                assert("Tensor range check" && 0 <= v350 && v350 < 4l);
                v347[v353] = v356;
                v350 += 1l ;
            }
            v348 += 1l ;
        }
        float v357;
        v357 = 0.0f;
        int v358;
        v358 = 0l;
        while (while_method_3(v358)){
            int v360;
            v360 = 0l;
            while (while_method_1(v360)){
                assert("Tensor range check" && 0 <= v358 && v358 < 1l);
                assert("Tensor range check" && 0 <= v360 && v360 < 4l);
                int v362;
                v362 = 4l * v358;
                int v363;
                v363 = v362 + v360;
                float v364;
                v364 = v347[v363];
                float v365;
                v365 = v357 + v364;
                v357 = v365;
                v360 += 1l ;
            }
            v358 += 1l ;
        }
        auto v366 = cooperative_groups::coalesced_threads();
        int v367;
        v367 = threadIdx.x;
        int v368;
        v368 = v367 / 16l;
        auto v369 = cooperative_groups::labeled_partition(v366,v368);
        float v370;
        v370 = cooperative_groups::reduce(v369, v357, v344);
        float v371[4l];
        int v372;
        v372 = 0l;
        while (while_method_3(v372)){
            int v374;
            v374 = 0l;
            while (while_method_1(v374)){
                assert("Tensor range check" && 0 <= v372 && v372 < 1l);
                assert("Tensor range check" && 0 <= v374 && v374 < 4l);
                int v376;
                v376 = 4l * v372;
                int v377;
                v377 = v376 + v374;
                float v378;
                v378 = v347[v377];
                float v379;
                v379 = v378 / v370;
                assert("Tensor range check" && 0 <= v372 && v372 < 1l);
                assert("Tensor range check" && 0 <= v374 && v374 < 4l);
                v371[v377] = v379;
                v374 += 1l ;
            }
            v372 += 1l ;
        }
        assert("Tensor range check" && 0 <= v278 && v278 < 64l);
        int v380;
        v380 = 0l;
        while (while_method_3(v380)){
            assert("Tensor range check" && 0 <= v380 && v380 < 1l);
            int v382;
            v382 = 64l * v380;
            int v383;
            v383 = v382 + v287;
            assert("Tensor range check" && 0 <= v380 && v380 < 1l);
            int v384;
            v384 = 4l * v380;
            int4* v385;
            v385 = reinterpret_cast<int4*>(v371 + v384);
            int4* v386;
            v386 = reinterpret_cast<int4*>(v3 + v383);
            assert("Pointer alignment check" && (unsigned long long)(v385) % 4l == 0 && (unsigned long long)(v386) % 4l == 0);
            *v386 = *v385;
            v380 += 1l ;
        }
        v278 += 24l ;
    }
    v15.sync() ;
    int v387;
    v387 = threadIdx.x;
    bool v388;
    v388 = 0l <= v387;
    bool v389;
    v389 = v388 == false;
    if (v389){
        assert("The index needs to be zero or positive." && v388);
    } else {
    }
    int v391;
    v391 = v387 % 16l;
    int v392;
    v392 = v387 / 16l;
    bool v393;
    v393 = v392 < 2l;
    bool v394;
    v394 = v393 == false;
    if (v394){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v393);
    } else {
    }
    assert("Tensor range check" && 0 <= v392 && v392 < 2l);
    assert("Tensor range check" && 0 <= v391 && v391 < 16l);
    int v396;
    v396 = 4l * v391;
    int v397;
    v397 = 64l * v392;
    int v398;
    v398 = v397 + v396;
    assert("Tensor range check" && 0 <= v392 && v392 < 2l);
    assert("Tensor range check" && 0 <= v391 && v391 < 16l);
    int v399;
    v399 = blockIdx.x;
    int v400;
    v400 = v399;
    while (while_method_2(v400)){
        bool v402;
        v402 = 0l <= v400;
        bool v403;
        v403 = v402 == false;
        if (v403){
            assert("The index needs to be zero or positive." && v402);
        } else {
        }
        bool v405;
        v405 = v400 < 64l;
        bool v406;
        v406 = v405 == false;
        if (v406){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v405);
        } else {
        }
        assert("Tensor range check" && 0 <= v400 && v400 < 64l);
        int v408;
        v408 = 128l * v400;
        int v409;
        v409 = v408 + v398;
        float v410[4l];
        int v411[4l];
        int v412;
        v412 = 0l;
        while (while_method_3(v412)){
            assert("Tensor range check" && 0 <= v412 && v412 < 1l);
            int v414;
            v414 = 4l * v412;
            assert("Tensor range check" && 0 <= v412 && v412 < 1l);
            int v415;
            v415 = 64l * v412;
            int v416;
            v416 = v415 + v409;
            int4* v417;
            v417 = reinterpret_cast<int4*>(v1 + v416);
            int4* v418;
            v418 = reinterpret_cast<int4*>(v410 + v414);
            assert("Pointer alignment check" && (unsigned long long)(v417) % 4l == 0 && (unsigned long long)(v418) % 4l == 0);
            *v418 = *v417;
            v412 += 1l ;
        }
        int v419;
        v419 = 0l;
        while (while_method_3(v419)){
            int v421;
            v421 = 0l;
            while (while_method_1(v421)){
                bool v423;
                v423 = 0l <= v421;
                bool v425;
                if (v423){
                    bool v424;
                    v424 = v421 < 4l;
                    v425 = v424;
                } else {
                    v425 = false;
                }
                bool v426;
                v426 = v425 == false;
                if (v426){
                    assert("The indices should be inside the range of the dimension." && v425);
                } else {
                }
                bool v428;
                v428 = 0l <= v391;
                bool v430;
                if (v428){
                    bool v429;
                    v429 = v391 < 16l;
                    v430 = v429;
                } else {
                    v430 = false;
                }
                bool v431;
                v431 = v430 == false;
                if (v431){
                    assert("The indices should be inside the range of the dimension." && v430);
                } else {
                }
                int v433;
                v433 = v391 * 4l;
                int v434;
                v434 = v421 + v433;
                bool v435;
                v435 = 0l <= v419;
                bool v437;
                if (v435){
                    bool v436;
                    v436 = v419 < 1l;
                    v437 = v436;
                } else {
                    v437 = false;
                }
                bool v438;
                v438 = v437 == false;
                if (v438){
                    assert("The indices should be inside the range of the dimension." && v437);
                } else {
                }
                int v440;
                v440 = v419 * 64l;
                int v441;
                v441 = v434 + v440;
                assert("Tensor range check" && 0 <= v419 && v419 < 1l);
                assert("Tensor range check" && 0 <= v421 && v421 < 4l);
                int v442;
                v442 = 4l * v419;
                int v443;
                v443 = v442 + v421;
                v411[v443] = v441;
                v421 += 1l ;
            }
            v419 += 1l ;
        }
        bool v444;
        v444 = 0l <= v392;
        bool v445;
        v445 = v444 && v393;
        bool v446;
        v446 = v445 == false;
        if (v446){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v445);
        } else {
        }
        bool v448;
        v448 = v402 && v405;
        bool v449;
        v449 = v448 == false;
        if (v449){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v448);
        } else {
        }
        int v451;
        v451 = v400 * 2l;
        int v452;
        v452 = v451 + v392;
        float v453[4l];
        int v454;
        v454 = 0l;
        while (while_method_3(v454)){
            int v456;
            v456 = 0l;
            while (while_method_1(v456)){
                assert("Tensor range check" && 0 <= v454 && v454 < 1l);
                assert("Tensor range check" && 0 <= v456 && v456 < 4l);
                int v458;
                v458 = 4l * v454;
                int v459;
                v459 = v458 + v456;
                float v460;
                v460 = v410[v459];
                float v461;
                v461 = v460 * v460;
                assert("Tensor range check" && 0 <= v454 && v454 < 1l);
                assert("Tensor range check" && 0 <= v456 && v456 < 4l);
                v453[v459] = v461;
                v456 += 1l ;
            }
            v454 += 1l ;
        }
        float v462;
        v462 = 0.0f;
        int v463;
        v463 = 0l;
        while (while_method_3(v463)){
            int v465;
            v465 = 0l;
            while (while_method_1(v465)){
                assert("Tensor range check" && 0 <= v463 && v463 < 1l);
                assert("Tensor range check" && 0 <= v465 && v465 < 4l);
                int v467;
                v467 = 4l * v463;
                int v468;
                v468 = v467 + v465;
                float v469;
                v469 = v453[v468];
                float v470;
                v470 = v462 + v469;
                v462 = v470;
                v465 += 1l ;
            }
            v463 += 1l ;
        }
        auto v471 = cooperative_groups::coalesced_threads();
        int v472;
        v472 = threadIdx.x;
        int v473;
        v473 = v472 / 16l;
        auto v474 = cooperative_groups::labeled_partition(v471,v473);
        Closure0 v475{};
        float v476;
        v476 = cooperative_groups::reduce(v474, v462, v475);
        float v477[4l];
        int v478;
        v478 = 0l;
        while (while_method_3(v478)){
            int v480;
            v480 = 0l;
            while (while_method_1(v480)){
                assert("Tensor range check" && 0 <= v478 && v478 < 1l);
                assert("Tensor range check" && 0 <= v480 && v480 < 4l);
                int v482;
                v482 = 4l * v478;
                int v483;
                v483 = v482 + v480;
                float v484;
                v484 = v410[v483];
                bool v485;
                v485 = v476 == 0.0f;
                bool v486;
                v486 = v485 != true;
                float v488;
                if (v486){
                    float v487;
                    v487 = v484 / v476;
                    v488 = v487;
                } else {
                    v488 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v478 && v478 < 1l);
                assert("Tensor range check" && 0 <= v480 && v480 < 4l);
                v477[v483] = v488;
                v480 += 1l ;
            }
            v478 += 1l ;
        }
        assert("Tensor range check" && 0 <= v400 && v400 < 64l);
        int v489;
        v489 = 0l;
        while (while_method_3(v489)){
            assert("Tensor range check" && 0 <= v489 && v489 < 1l);
            int v491;
            v491 = 64l * v489;
            int v492;
            v492 = v491 + v409;
            assert("Tensor range check" && 0 <= v489 && v489 < 1l);
            int v493;
            v493 = 4l * v489;
            int4* v494;
            v494 = reinterpret_cast<int4*>(v477 + v493);
            int4* v495;
            v495 = reinterpret_cast<int4*>(v7 + v492);
            assert("Pointer alignment check" && (unsigned long long)(v494) % 4l == 0 && (unsigned long long)(v495) % 4l == 0);
            *v495 = *v494;
            v489 += 1l ;
        }
        v400 += 24l ;
    }
    v15.sync() ;
    int v496;
    v496 = threadIdx.x;
    bool v497;
    v497 = 0l <= v496;
    bool v498;
    v498 = v497 == false;
    if (v498){
        assert("The index needs to be zero or positive." && v497);
    } else {
    }
    int v500;
    v500 = v496 % 16l;
    int v501;
    v501 = v496 / 16l;
    bool v502;
    v502 = v501 < 2l;
    bool v503;
    v503 = v502 == false;
    if (v503){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v502);
    } else {
    }
    assert("Tensor range check" && 0 <= v501 && v501 < 2l);
    assert("Tensor range check" && 0 <= v500 && v500 < 16l);
    int v505;
    v505 = 4l * v500;
    int v506;
    v506 = 64l * v501;
    int v507;
    v507 = v506 + v505;
    assert("Tensor range check" && 0 <= v501 && v501 < 2l);
    int v508;
    v508 = blockIdx.x;
    int v509;
    v509 = v508;
    while (while_method_2(v509)){
        bool v511;
        v511 = 0l <= v509;
        bool v512;
        v512 = v511 == false;
        if (v512){
            assert("The index needs to be zero or positive." && v511);
        } else {
        }
        bool v514;
        v514 = v509 < 64l;
        bool v515;
        v515 = v514 == false;
        if (v515){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v514);
        } else {
        }
        assert("Tensor range check" && 0 <= v509 && v509 < 64l);
        int v517;
        v517 = 128l * v509;
        int v518;
        v518 = v517 + v507;
        float v519[4l];
        int v520[4l];
        int v521;
        v521 = 0l;
        while (while_method_3(v521)){
            assert("Tensor range check" && 0 <= v521 && v521 < 1l);
            int v523;
            v523 = 4l * v521;
            assert("Tensor range check" && 0 <= v521 && v521 < 1l);
            int v524;
            v524 = 64l * v521;
            int v525;
            v525 = v524 + v518;
            int4* v526;
            v526 = reinterpret_cast<int4*>(v1 + v525);
            int4* v527;
            v527 = reinterpret_cast<int4*>(v519 + v523);
            assert("Pointer alignment check" && (unsigned long long)(v526) % 4l == 0 && (unsigned long long)(v527) % 4l == 0);
            *v527 = *v526;
            v521 += 1l ;
        }
        int v528;
        v528 = 0l;
        while (while_method_3(v528)){
            int v530;
            v530 = 0l;
            while (while_method_1(v530)){
                bool v532;
                v532 = 0l <= v530;
                bool v534;
                if (v532){
                    bool v533;
                    v533 = v530 < 4l;
                    v534 = v533;
                } else {
                    v534 = false;
                }
                bool v535;
                v535 = v534 == false;
                if (v535){
                    assert("The indices should be inside the range of the dimension." && v534);
                } else {
                }
                bool v537;
                v537 = 0l <= v500;
                bool v539;
                if (v537){
                    bool v538;
                    v538 = v500 < 16l;
                    v539 = v538;
                } else {
                    v539 = false;
                }
                bool v540;
                v540 = v539 == false;
                if (v540){
                    assert("The indices should be inside the range of the dimension." && v539);
                } else {
                }
                int v542;
                v542 = v500 * 4l;
                int v543;
                v543 = v530 + v542;
                bool v544;
                v544 = 0l <= v528;
                bool v546;
                if (v544){
                    bool v545;
                    v545 = v528 < 1l;
                    v546 = v545;
                } else {
                    v546 = false;
                }
                bool v547;
                v547 = v546 == false;
                if (v547){
                    assert("The indices should be inside the range of the dimension." && v546);
                } else {
                }
                int v549;
                v549 = v528 * 64l;
                int v550;
                v550 = v543 + v549;
                assert("Tensor range check" && 0 <= v528 && v528 < 1l);
                assert("Tensor range check" && 0 <= v530 && v530 < 4l);
                int v551;
                v551 = 4l * v528;
                int v552;
                v552 = v551 + v530;
                v520[v552] = v550;
                v530 += 1l ;
            }
            v528 += 1l ;
        }
        bool v553;
        v553 = 0l <= v501;
        bool v554;
        v554 = v553 && v502;
        bool v555;
        v555 = v554 == false;
        if (v555){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v554);
        } else {
        }
        bool v557;
        v557 = v511 && v514;
        bool v558;
        v558 = v557 == false;
        if (v558){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v557);
        } else {
        }
        int v560;
        v560 = v509 * 2l;
        int v561;
        v561 = v560 + v501;
        float v562; int v563;
        Tuple1 tmp62 = Tuple1{-1.0f / 0.0f, 0l};
        v562 = tmp62.v0; v563 = tmp62.v1;
        int v564;
        v564 = 0l;
        while (while_method_3(v564)){
            int v566;
            v566 = 0l;
            while (while_method_1(v566)){
                assert("Tensor range check" && 0 <= v564 && v564 < 1l);
                assert("Tensor range check" && 0 <= v566 && v566 < 4l);
                int v568;
                v568 = 4l * v564;
                int v569;
                v569 = v568 + v566;
                float v570;
                v570 = v519[v569];
                int v571;
                v571 = v520[v569];
                bool v572;
                v572 = v562 > v570;
                float v573; int v574;
                if (v572){
                    v573 = v562; v574 = v563;
                } else {
                    v573 = v570; v574 = v571;
                }
                v562 = v573;
                v563 = v574;
                v566 += 1l ;
            }
            v564 += 1l ;
        }
        auto v575 = cooperative_groups::coalesced_threads();
        int v576;
        v576 = threadIdx.x;
        int v577;
        v577 = v576 / 16l;
        auto v578 = cooperative_groups::labeled_partition(v575,v577);
        Closure1 v579{};
        float v580; int v581;
        Tuple1 tmp63 = cooperative_groups::reduce(v578, Tuple1{v562, v563}, v579);
        v580 = tmp63.v0; v581 = tmp63.v1;
        assert("Tensor range check" && 0 <= v509 && v509 < 64l);
        int v582;
        v582 = 2l * v509;
        int v583;
        v583 = v582 + v501;
        v8[v583] = v581;
        v509 += 24l ;
    }
    v15.sync() ;
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
    v588 = v584 % 16l;
    int v589;
    v589 = v584 / 16l;
    bool v590;
    v590 = v589 < 2l;
    bool v591;
    v591 = v590 == false;
    if (v591){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v590);
    } else {
    }
    assert("Tensor range check" && 0 <= v589 && v589 < 2l);
    assert("Tensor range check" && 0 <= v588 && v588 < 16l);
    int v593;
    v593 = 4l * v588;
    int v594;
    v594 = 64l * v589;
    int v595;
    v595 = v594 + v593;
    assert("Tensor range check" && 0 <= v589 && v589 < 2l);
    assert("Tensor range check" && 0 <= v588 && v588 < 16l);
    int v596;
    v596 = blockIdx.x;
    int v597;
    v597 = v596;
    while (while_method_2(v597)){
        bool v599;
        v599 = 0l <= v597;
        bool v600;
        v600 = v599 == false;
        if (v600){
            assert("The index needs to be zero or positive." && v599);
        } else {
        }
        bool v602;
        v602 = v597 < 64l;
        bool v603;
        v603 = v602 == false;
        if (v603){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v602);
        } else {
        }
        assert("Tensor range check" && 0 <= v597 && v597 < 64l);
        int v605;
        v605 = 128l * v597;
        int v606;
        v606 = v605 + v595;
        float v607[4l];
        int v608[4l];
        int v609;
        v609 = 0l;
        while (while_method_3(v609)){
            assert("Tensor range check" && 0 <= v609 && v609 < 1l);
            int v611;
            v611 = 4l * v609;
            assert("Tensor range check" && 0 <= v609 && v609 < 1l);
            int v612;
            v612 = 64l * v609;
            int v613;
            v613 = v612 + v606;
            int4* v614;
            v614 = reinterpret_cast<int4*>(v1 + v613);
            int4* v615;
            v615 = reinterpret_cast<int4*>(v607 + v611);
            assert("Pointer alignment check" && (unsigned long long)(v614) % 4l == 0 && (unsigned long long)(v615) % 4l == 0);
            *v615 = *v614;
            v609 += 1l ;
        }
        int v616;
        v616 = 0l;
        while (while_method_3(v616)){
            int v618;
            v618 = 0l;
            while (while_method_1(v618)){
                bool v620;
                v620 = 0l <= v618;
                bool v622;
                if (v620){
                    bool v621;
                    v621 = v618 < 4l;
                    v622 = v621;
                } else {
                    v622 = false;
                }
                bool v623;
                v623 = v622 == false;
                if (v623){
                    assert("The indices should be inside the range of the dimension." && v622);
                } else {
                }
                bool v625;
                v625 = 0l <= v588;
                bool v627;
                if (v625){
                    bool v626;
                    v626 = v588 < 16l;
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
                v630 = v588 * 4l;
                int v631;
                v631 = v618 + v630;
                bool v632;
                v632 = 0l <= v616;
                bool v634;
                if (v632){
                    bool v633;
                    v633 = v616 < 1l;
                    v634 = v633;
                } else {
                    v634 = false;
                }
                bool v635;
                v635 = v634 == false;
                if (v635){
                    assert("The indices should be inside the range of the dimension." && v634);
                } else {
                }
                int v637;
                v637 = v616 * 64l;
                int v638;
                v638 = v631 + v637;
                assert("Tensor range check" && 0 <= v616 && v616 < 1l);
                assert("Tensor range check" && 0 <= v618 && v618 < 4l);
                int v639;
                v639 = 4l * v616;
                int v640;
                v640 = v639 + v618;
                v608[v640] = v638;
                v618 += 1l ;
            }
            v616 += 1l ;
        }
        bool v641;
        v641 = 0l <= v589;
        bool v642;
        v642 = v641 && v590;
        bool v643;
        v643 = v642 == false;
        if (v643){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v642);
        } else {
        }
        bool v645;
        v645 = v599 && v602;
        bool v646;
        v646 = v645 == false;
        if (v646){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v645);
        } else {
        }
        int v648;
        v648 = v597 * 2l;
        int v649;
        v649 = v648 + v589;
        float v650;
        v650 = 0.0f;
        int v651;
        v651 = 0l;
        while (while_method_3(v651)){
            int v653;
            v653 = 0l;
            while (while_method_1(v653)){
                assert("Tensor range check" && 0 <= v651 && v651 < 1l);
                assert("Tensor range check" && 0 <= v653 && v653 < 4l);
                int v655;
                v655 = 4l * v651;
                int v656;
                v656 = v655 + v653;
                float v657;
                v657 = v607[v656];
                float v658;
                v658 = v650 + v657;
                v650 = v658;
                v653 += 1l ;
            }
            v651 += 1l ;
        }
        auto v659 = cooperative_groups::coalesced_threads();
        int v660;
        v660 = threadIdx.x;
        int v661;
        v661 = v660 / 16l;
        auto v662 = cooperative_groups::labeled_partition(v659,v661);
        Closure0 v663{};
        float v664;
        v664 = cooperative_groups::reduce(v662, v650, v663);
        float v665;
        v665 = v664 / 64.0f;
        float v666[4l];
        int v667;
        v667 = 0l;
        while (while_method_3(v667)){
            int v669;
            v669 = 0l;
            while (while_method_1(v669)){
                assert("Tensor range check" && 0 <= v667 && v667 < 1l);
                assert("Tensor range check" && 0 <= v669 && v669 < 4l);
                int v671;
                v671 = 4l * v667;
                int v672;
                v672 = v671 + v669;
                float v673;
                v673 = v607[v672];
                float v674;
                v674 = v673 - v665;
                float v675;
                v675 = exp(v674);
                assert("Tensor range check" && 0 <= v667 && v667 < 1l);
                assert("Tensor range check" && 0 <= v669 && v669 < 4l);
                v666[v672] = v675;
                v669 += 1l ;
            }
            v667 += 1l ;
        }
        float v676;
        v676 = 0.0f;
        int v677;
        v677 = 0l;
        while (while_method_3(v677)){
            int v679;
            v679 = 0l;
            while (while_method_1(v679)){
                assert("Tensor range check" && 0 <= v677 && v677 < 1l);
                assert("Tensor range check" && 0 <= v679 && v679 < 4l);
                int v681;
                v681 = 4l * v677;
                int v682;
                v682 = v681 + v679;
                float v683;
                v683 = v666[v682];
                float v684;
                v684 = v676 + v683;
                v676 = v684;
                v679 += 1l ;
            }
            v677 += 1l ;
        }
        auto v685 = cooperative_groups::coalesced_threads();
        int v686;
        v686 = threadIdx.x;
        int v687;
        v687 = v686 / 16l;
        auto v688 = cooperative_groups::labeled_partition(v685,v687);
        float v689;
        v689 = cooperative_groups::reduce(v688, v676, v663);
        float v690[4l];
        int v691;
        v691 = 0l;
        while (while_method_3(v691)){
            int v693;
            v693 = 0l;
            while (while_method_1(v693)){
                assert("Tensor range check" && 0 <= v691 && v691 < 1l);
                assert("Tensor range check" && 0 <= v693 && v693 < 4l);
                int v695;
                v695 = 4l * v691;
                int v696;
                v696 = v695 + v693;
                float v697;
                v697 = v666[v696];
                float v698;
                v698 = v697 / v689;
                assert("Tensor range check" && 0 <= v691 && v691 < 1l);
                assert("Tensor range check" && 0 <= v693 && v693 < 4l);
                v690[v696] = v698;
                v693 += 1l ;
            }
            v691 += 1l ;
        }
        float v699[4l];
        float v700;
        v700 = 0.0f;
        int v701;
        v701 = 0l;
        while (while_method_3(v701)){
            assert("Tensor range check" && 0 <= v701 && v701 < 1l);
            int v703;
            v703 = 4l * v701;
            assert("Tensor range check" && 0 <= v701 && v701 < 1l);
            int v704; float v705;
            Tuple0 tmp64 = Tuple0{0l, 0.0f};
            v704 = tmp64.v0; v705 = tmp64.v1;
            while (while_method_1(v704)){
                assert("Tensor range check" && 0 <= v704 && v704 < 4l);
                int v707;
                v707 = v704 + v703;
                float v708;
                v708 = v690[v707];
                float v709;
                v709 = v705 + v708;
                v705 = v709;
                v704 += 1l ;
            }
            auto v710 = cooperative_groups::coalesced_threads();
            int v711;
            v711 = threadIdx.x;
            int v712;
            v712 = v711 / 16l;
            auto v713 = cooperative_groups::labeled_partition(v710,v712);
            Closure2 v714{};
            float v715;
            v715 = cooperative_groups::inclusive_scan(v713, v705, v714);
            float v716;
            v716 = v713.shfl_up(v715,1);
            bool v717;
            v717 = v713.thread_rank() == 0;
            float v718;
            if (v717){
                v718 = 0.0f;
            } else {
                v718 = v716;
            }
            float v719;
            v719 = v713.shfl(v715,v713.num_threads()-1);
            float v720;
            v720 = v700 + v718;
            int v721; float v722;
            Tuple0 tmp65 = Tuple0{0l, v720};
            v721 = tmp65.v0; v722 = tmp65.v1;
            while (while_method_1(v721)){
                assert("Tensor range check" && 0 <= v721 && v721 < 4l);
                int v724;
                v724 = v721 + v703;
                float v725;
                v725 = v690[v724];
                float v726;
                v726 = v722 + v725;
                assert("Tensor range check" && 0 <= v721 && v721 < 4l);
                v699[v724] = v726;
                v722 = v726;
                v721 += 1l ;
            }
            float v727;
            v727 = v700 + v719;
            v700 = v727;
            v701 += 1l ;
        }
        assert("Tensor range check" && 0 <= v597 && v597 < 64l);
        int v728;
        v728 = 0l;
        while (while_method_3(v728)){
            assert("Tensor range check" && 0 <= v728 && v728 < 1l);
            int v730;
            v730 = 64l * v728;
            int v731;
            v731 = v730 + v606;
            assert("Tensor range check" && 0 <= v728 && v728 < 1l);
            int v732;
            v732 = 4l * v728;
            int4* v733;
            v733 = reinterpret_cast<int4*>(v690 + v732);
            int4* v734;
            v734 = reinterpret_cast<int4*>(v5 + v731);
            assert("Pointer alignment check" && (unsigned long long)(v733) % 4l == 0 && (unsigned long long)(v734) % 4l == 0);
            *v734 = *v733;
            int4* v735;
            v735 = reinterpret_cast<int4*>(v699 + v732);
            int4* v736;
            v736 = reinterpret_cast<int4*>(v6 + v731);
            assert("Pointer alignment check" && (unsigned long long)(v735) % 4l == 0 && (unsigned long long)(v736) % 4l == 0);
            *v736 = *v735;
            v728 += 1l ;
        }
        v597 += 24l ;
    }
    v15.sync() ;
    int v737;
    v737 = threadIdx.x;
    bool v738;
    v738 = 0l <= v737;
    bool v739;
    v739 = v738 == false;
    if (v739){
        assert("The index needs to be zero or positive." && v738);
    } else {
    }
    int v741;
    v741 = v737 % 16l;
    int v742;
    v742 = v737 / 16l;
    bool v743;
    v743 = v742 < 2l;
    bool v744;
    v744 = v743 == false;
    if (v744){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v743);
    } else {
    }
    assert("Tensor range check" && 0 <= v742 && v742 < 2l);
    assert("Tensor range check" && 0 <= v741 && v741 < 16l);
    int v746;
    v746 = 4l * v741;
    int v747;
    v747 = 64l * v742;
    int v748;
    v748 = v747 + v746;
    assert("Tensor range check" && 0 <= v742 && v742 < 2l);
    assert("Tensor range check" && 0 <= v741 && v741 < 16l);
    int v749;
    v749 = blockIdx.x;
    int v750;
    v750 = v749;
    while (while_method_2(v750)){
        bool v752;
        v752 = 0l <= v750;
        bool v753;
        v753 = v752 == false;
        if (v753){
            assert("The index needs to be zero or positive." && v752);
        } else {
        }
        bool v755;
        v755 = v750 < 64l;
        bool v756;
        v756 = v755 == false;
        if (v756){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v755);
        } else {
        }
        assert("Tensor range check" && 0 <= v750 && v750 < 64l);
        int v758;
        v758 = 128l * v750;
        int v759;
        v759 = v758 + v748;
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
            v765 = 64l * v762;
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
                v778 = 0l <= v741;
                bool v780;
                if (v778){
                    bool v779;
                    v779 = v741 < 16l;
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
                v783 = v741 * 4l;
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
                v790 = v769 * 64l;
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
        v794 = 0l <= v742;
        bool v795;
        v795 = v794 && v743;
        bool v796;
        v796 = v795 == false;
        if (v796){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v795);
        } else {
        }
        bool v798;
        v798 = v752 && v755;
        bool v799;
        v799 = v798 == false;
        if (v799){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v798);
        } else {
        }
        int v801;
        v801 = v750 * 2l;
        int v802;
        v802 = v801 + v742;
        int v803[4l];
        int v804;
        v804 = 0l;
        int v805;
        v805 = 0l;
        while (while_method_3(v805)){
            assert("Tensor range check" && 0 <= v805 && v805 < 1l);
            int v807;
            v807 = 4l * v805;
            assert("Tensor range check" && 0 <= v805 && v805 < 1l);
            int v808; int v809;
            Tuple2 tmp66 = Tuple2{0l, 0l};
            v808 = tmp66.v0; v809 = tmp66.v1;
            while (while_method_1(v808)){
                assert("Tensor range check" && 0 <= v808 && v808 < 4l);
                int v811;
                v811 = v808 + v807;
                int v812;
                v812 = v760[v811];
                int v813;
                v813 = v809 + v812;
                v809 = v813;
                v808 += 1l ;
            }
            auto v814 = cooperative_groups::coalesced_threads();
            int v815;
            v815 = threadIdx.x;
            int v816;
            v816 = v815 / 16l;
            auto v817 = cooperative_groups::labeled_partition(v814,v816);
            Closure3 v818{};
            int v819;
            v819 = cooperative_groups::inclusive_scan(v817, v809, v818);
            int v820;
            v820 = v817.shfl_up(v819,1);
            bool v821;
            v821 = v817.thread_rank() == 0;
            int v822;
            if (v821){
                v822 = 0l;
            } else {
                v822 = v820;
            }
            int v823;
            v823 = v817.shfl(v819,v817.num_threads()-1);
            int v824;
            v824 = v804 + v822;
            int v825; int v826;
            Tuple2 tmp67 = Tuple2{0l, v824};
            v825 = tmp67.v0; v826 = tmp67.v1;
            while (while_method_1(v825)){
                assert("Tensor range check" && 0 <= v825 && v825 < 4l);
                int v828;
                v828 = v825 + v807;
                int v829;
                v829 = v760[v828];
                assert("Tensor range check" && 0 <= v825 && v825 < 4l);
                v803[v828] = v826;
                int v830;
                v830 = v826 + v829;
                v826 = v830;
                v825 += 1l ;
            }
            int v831;
            v831 = v804 + v823;
            v804 = v831;
            v805 += 1l ;
        }
        assert("Tensor range check" && 0 <= v750 && v750 < 64l);
        int v832;
        v832 = 0l;
        while (while_method_3(v832)){
            assert("Tensor range check" && 0 <= v832 && v832 < 1l);
            int v834;
            v834 = 64l * v832;
            int v835;
            v835 = v834 + v759;
            assert("Tensor range check" && 0 <= v832 && v832 < 1l);
            int v836;
            v836 = 4l * v832;
            int4* v837;
            v837 = reinterpret_cast<int4*>(v803 + v836);
            int4* v838;
            v838 = reinterpret_cast<int4*>(v12 + v835);
            assert("Pointer alignment check" && (unsigned long long)(v837) % 4l == 0 && (unsigned long long)(v838) % 4l == 0);
            *v838 = *v837;
            v832 += 1l ;
        }
        v750 += 24l ;
    }
    v15.sync() ;
    int v839;
    v839 = threadIdx.x;
    bool v840;
    v840 = 0l <= v839;
    bool v841;
    v841 = v840 == false;
    if (v841){
        assert("The index needs to be zero or positive." && v840);
    } else {
    }
    int v843;
    v843 = v839 % 16l;
    int v844;
    v844 = v839 / 16l;
    bool v845;
    v845 = v844 < 2l;
    bool v846;
    v846 = v845 == false;
    if (v846){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v845);
    } else {
    }
    assert("Tensor range check" && 0 <= v844 && v844 < 2l);
    assert("Tensor range check" && 0 <= v843 && v843 < 16l);
    int v848;
    v848 = 4l * v843;
    int v849;
    v849 = 64l * v844;
    int v850;
    v850 = v849 + v848;
    assert("Tensor range check" && 0 <= v844 && v844 < 2l);
    assert("Tensor range check" && 0 <= v843 && v843 < 16l);
    int v851;
    v851 = blockIdx.x;
    int v852;
    v852 = v851;
    while (while_method_2(v852)){
        bool v854;
        v854 = 0l <= v852;
        bool v855;
        v855 = v854 == false;
        if (v855){
            assert("The index needs to be zero or positive." && v854);
        } else {
        }
        bool v857;
        v857 = v852 < 64l;
        bool v858;
        v858 = v857 == false;
        if (v858){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v857);
        } else {
        }
        assert("Tensor range check" && 0 <= v852 && v852 < 64l);
        int v860;
        v860 = 128l * v852;
        int v861;
        v861 = v860 + v850;
        float v862[4l];
        int v863[4l];
        int v864;
        v864 = 0l;
        while (while_method_3(v864)){
            assert("Tensor range check" && 0 <= v864 && v864 < 1l);
            int v866;
            v866 = 4l * v864;
            assert("Tensor range check" && 0 <= v864 && v864 < 1l);
            int v867;
            v867 = 64l * v864;
            int v868;
            v868 = v867 + v861;
            int4* v869;
            v869 = reinterpret_cast<int4*>(v1 + v868);
            int4* v870;
            v870 = reinterpret_cast<int4*>(v862 + v866);
            assert("Pointer alignment check" && (unsigned long long)(v869) % 4l == 0 && (unsigned long long)(v870) % 4l == 0);
            *v870 = *v869;
            v864 += 1l ;
        }
        int v871;
        v871 = 0l;
        while (while_method_3(v871)){
            int v873;
            v873 = 0l;
            while (while_method_1(v873)){
                bool v875;
                v875 = 0l <= v873;
                bool v877;
                if (v875){
                    bool v876;
                    v876 = v873 < 4l;
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
                bool v880;
                v880 = 0l <= v843;
                bool v882;
                if (v880){
                    bool v881;
                    v881 = v843 < 16l;
                    v882 = v881;
                } else {
                    v882 = false;
                }
                bool v883;
                v883 = v882 == false;
                if (v883){
                    assert("The indices should be inside the range of the dimension." && v882);
                } else {
                }
                int v885;
                v885 = v843 * 4l;
                int v886;
                v886 = v873 + v885;
                bool v887;
                v887 = 0l <= v871;
                bool v889;
                if (v887){
                    bool v888;
                    v888 = v871 < 1l;
                    v889 = v888;
                } else {
                    v889 = false;
                }
                bool v890;
                v890 = v889 == false;
                if (v890){
                    assert("The indices should be inside the range of the dimension." && v889);
                } else {
                }
                int v892;
                v892 = v871 * 64l;
                int v893;
                v893 = v886 + v892;
                assert("Tensor range check" && 0 <= v871 && v871 < 1l);
                assert("Tensor range check" && 0 <= v873 && v873 < 4l);
                int v894;
                v894 = 4l * v871;
                int v895;
                v895 = v894 + v873;
                v863[v895] = v893;
                v873 += 1l ;
            }
            v871 += 1l ;
        }
        bool v896;
        v896 = 0l <= v844;
        bool v897;
        v897 = v896 && v845;
        bool v898;
        v898 = v897 == false;
        if (v898){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v897);
        } else {
        }
        bool v900;
        v900 = v854 && v857;
        bool v901;
        v901 = v900 == false;
        if (v901){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v900);
        } else {
        }
        int v903;
        v903 = v852 * 2l;
        int v904;
        v904 = v903 + v844;
        bool v905[4l];
        int v906;
        v906 = 0l;
        while (while_method_3(v906)){
            int v908;
            v908 = 0l;
            while (while_method_1(v908)){
                assert("Tensor range check" && 0 <= v906 && v906 < 1l);
                assert("Tensor range check" && 0 <= v908 && v908 < 4l);
                int v910;
                v910 = 4l * v906;
                int v911;
                v911 = v910 + v908;
                float v912;
                v912 = v862[v911];
                int v913;
                v913 = v863[v911];
                bool v914;
                v914 = v913 < 4l;
                assert("Tensor range check" && 0 <= v906 && v906 < 1l);
                assert("Tensor range check" && 0 <= v908 && v908 < 4l);
                v905[v911] = v914;
                v908 += 1l ;
            }
            v906 += 1l ;
        }
        int v915[4l];
        int v916;
        v916 = 0l;
        while (while_method_3(v916)){
            int v918;
            v918 = 0l;
            while (while_method_1(v918)){
                assert("Tensor range check" && 0 <= v916 && v916 < 1l);
                assert("Tensor range check" && 0 <= v918 && v918 < 4l);
                int v920;
                v920 = 4l * v916;
                int v921;
                v921 = v920 + v918;
                bool v922;
                v922 = v905[v921];
                int v923;
                if (v922){
                    v923 = 1l;
                } else {
                    v923 = 0l;
                }
                assert("Tensor range check" && 0 <= v916 && v916 < 1l);
                assert("Tensor range check" && 0 <= v918 && v918 < 4l);
                v915[v921] = v923;
                v918 += 1l ;
            }
            v916 += 1l ;
        }
        int v924;
        v924 = 0l;
        int v925;
        v925 = 0l;
        while (while_method_3(v925)){
            int v927;
            v927 = 0l;
            while (while_method_1(v927)){
                assert("Tensor range check" && 0 <= v925 && v925 < 1l);
                assert("Tensor range check" && 0 <= v927 && v927 < 4l);
                int v929;
                v929 = 4l * v925;
                int v930;
                v930 = v929 + v927;
                int v931;
                v931 = v915[v930];
                int v932;
                v932 = v924 + v931;
                v924 = v932;
                v927 += 1l ;
            }
            v925 += 1l ;
        }
        auto v933 = cooperative_groups::coalesced_threads();
        int v934;
        v934 = threadIdx.x;
        int v935;
        v935 = v934 / 16l;
        auto v936 = cooperative_groups::labeled_partition(v933,v935);
        Closure4 v937{};
        int v938;
        v938 = cooperative_groups::reduce(v936, v924, v937);
        float v939[4l];
        int v940;
        v940 = 0l;
        while (while_method_3(v940)){
            int v942;
            v942 = 0l;
            while (while_method_1(v942)){
                assert("Tensor range check" && 0 <= v940 && v940 < 1l);
                assert("Tensor range check" && 0 <= v942 && v942 < 4l);
                int v944;
                v944 = 4l * v940;
                int v945;
                v945 = v944 + v942;
                float v946;
                v946 = v862[v945];
                bool v947;
                v947 = v905[v945];
                float v948;
                if (v947){
                    v948 = v946;
                } else {
                    v948 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v940 && v940 < 1l);
                assert("Tensor range check" && 0 <= v942 && v942 < 4l);
                v939[v945] = v948;
                v942 += 1l ;
            }
            v940 += 1l ;
        }
        float v949;
        v949 = 0.0f;
        int v950;
        v950 = 0l;
        while (while_method_3(v950)){
            int v952;
            v952 = 0l;
            while (while_method_1(v952)){
                assert("Tensor range check" && 0 <= v950 && v950 < 1l);
                assert("Tensor range check" && 0 <= v952 && v952 < 4l);
                int v954;
                v954 = 4l * v950;
                int v955;
                v955 = v954 + v952;
                float v956;
                v956 = v939[v955];
                float v957;
                v957 = v949 + v956;
                v949 = v957;
                v952 += 1l ;
            }
            v950 += 1l ;
        }
        auto v958 = cooperative_groups::coalesced_threads();
        int v959;
        v959 = threadIdx.x;
        int v960;
        v960 = v959 / 16l;
        auto v961 = cooperative_groups::labeled_partition(v958,v960);
        Closure0 v962{};
        float v963;
        v963 = cooperative_groups::reduce(v961, v949, v962);
        float v964;
        v964 = (float)v938;
        float v965;
        v965 = v963 / v964;
        float v966[4l];
        int v967;
        v967 = 0l;
        while (while_method_3(v967)){
            int v969;
            v969 = 0l;
            while (while_method_1(v969)){
                assert("Tensor range check" && 0 <= v967 && v967 < 1l);
                assert("Tensor range check" && 0 <= v969 && v969 < 4l);
                int v971;
                v971 = 4l * v967;
                int v972;
                v972 = v971 + v969;
                float v973;
                v973 = v862[v972];
                bool v974;
                v974 = v905[v972];
                float v975;
                if (v974){
                    v975 = v973;
                } else {
                    v975 = -1.0f / 0.0f;
                }
                float v976;
                v976 = v975 - v965;
                float v977;
                v977 = exp(v976);
                assert("Tensor range check" && 0 <= v967 && v967 < 1l);
                assert("Tensor range check" && 0 <= v969 && v969 < 4l);
                v966[v972] = v977;
                v969 += 1l ;
            }
            v967 += 1l ;
        }
        float v978;
        v978 = 0.0f;
        int v979;
        v979 = 0l;
        while (while_method_3(v979)){
            int v981;
            v981 = 0l;
            while (while_method_1(v981)){
                assert("Tensor range check" && 0 <= v979 && v979 < 1l);
                assert("Tensor range check" && 0 <= v981 && v981 < 4l);
                int v983;
                v983 = 4l * v979;
                int v984;
                v984 = v983 + v981;
                float v985;
                v985 = v966[v984];
                float v986;
                v986 = v978 + v985;
                v978 = v986;
                v981 += 1l ;
            }
            v979 += 1l ;
        }
        auto v987 = cooperative_groups::coalesced_threads();
        int v988;
        v988 = threadIdx.x;
        int v989;
        v989 = v988 / 16l;
        auto v990 = cooperative_groups::labeled_partition(v987,v989);
        float v991;
        v991 = cooperative_groups::reduce(v990, v978, v962);
        float v992[4l];
        int v993;
        v993 = 0l;
        while (while_method_3(v993)){
            int v995;
            v995 = 0l;
            while (while_method_1(v995)){
                assert("Tensor range check" && 0 <= v993 && v993 < 1l);
                assert("Tensor range check" && 0 <= v995 && v995 < 4l);
                int v997;
                v997 = 4l * v993;
                int v998;
                v998 = v997 + v995;
                float v999;
                v999 = v966[v998];
                float v1000;
                v1000 = v999 / v991;
                assert("Tensor range check" && 0 <= v993 && v993 < 1l);
                assert("Tensor range check" && 0 <= v995 && v995 < 4l);
                v992[v998] = v1000;
                v995 += 1l ;
            }
            v993 += 1l ;
        }
        assert("Tensor range check" && 0 <= v852 && v852 < 64l);
        int v1001;
        v1001 = 0l;
        while (while_method_3(v1001)){
            assert("Tensor range check" && 0 <= v1001 && v1001 < 1l);
            int v1003;
            v1003 = 64l * v1001;
            int v1004;
            v1004 = v1003 + v861;
            assert("Tensor range check" && 0 <= v1001 && v1001 < 1l);
            int v1005;
            v1005 = 4l * v1001;
            int4* v1006;
            v1006 = reinterpret_cast<int4*>(v992 + v1005);
            int4* v1007;
            v1007 = reinterpret_cast<int4*>(v4 + v1004);
            assert("Pointer alignment check" && (unsigned long long)(v1006) % 4l == 0 && (unsigned long long)(v1007) % 4l == 0);
            *v1007 = *v1006;
            v1001 += 1l ;
        }
        v852 += 24l ;
    }
    v15.sync() ;
    int v1008;
    v1008 = threadIdx.x;
    unsigned long long v1009;
    v1009 = (unsigned long long)v1008;
    curandStatePhilox4_32_10_t v1010;
    curand_init(12344321ull,v1009,0ull,&v1010);
    int v1011;
    v1011 = threadIdx.x;
    bool v1012;
    v1012 = 0l <= v1011;
    bool v1013;
    v1013 = v1012 == false;
    if (v1013){
        assert("The index needs to be zero or positive." && v1012);
    } else {
    }
    int v1015;
    v1015 = v1011 % 16l;
    int v1016;
    v1016 = v1011 / 16l;
    bool v1017;
    v1017 = v1016 < 2l;
    bool v1018;
    v1018 = v1017 == false;
    if (v1018){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1017);
    } else {
    }
    assert("Tensor range check" && 0 <= v1016 && v1016 < 2l);
    assert("Tensor range check" && 0 <= v1015 && v1015 < 16l);
    int v1020;
    v1020 = 4l * v1015;
    int v1021;
    v1021 = 64l * v1016;
    int v1022;
    v1022 = v1021 + v1020;
    assert("Tensor range check" && 0 <= v1016 && v1016 < 2l);
    assert("Tensor range check" && 0 <= v1015 && v1015 < 16l);
    assert("Tensor range check" && 0 <= v1016 && v1016 < 2l);
    int v1023;
    v1023 = blockIdx.x;
    int v1024;
    v1024 = v1023;
    while (while_method_2(v1024)){
        bool v1026;
        v1026 = 0l <= v1024;
        bool v1027;
        v1027 = v1026 == false;
        if (v1027){
            assert("The index needs to be zero or positive." && v1026);
        } else {
        }
        bool v1029;
        v1029 = v1024 < 64l;
        bool v1030;
        v1030 = v1029 == false;
        if (v1030){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1029);
        } else {
        }
        assert("Tensor range check" && 0 <= v1024 && v1024 < 64l);
        int v1032;
        v1032 = 128l * v1024;
        int v1033;
        v1033 = v1032 + v1022;
        float v1034[4l];
        int v1035[4l];
        int v1036;
        v1036 = 0l;
        while (while_method_3(v1036)){
            assert("Tensor range check" && 0 <= v1036 && v1036 < 1l);
            int v1038;
            v1038 = 4l * v1036;
            assert("Tensor range check" && 0 <= v1036 && v1036 < 1l);
            int v1039;
            v1039 = 64l * v1036;
            int v1040;
            v1040 = v1039 + v1033;
            int4* v1041;
            v1041 = reinterpret_cast<int4*>(v1 + v1040);
            int4* v1042;
            v1042 = reinterpret_cast<int4*>(v1034 + v1038);
            assert("Pointer alignment check" && (unsigned long long)(v1041) % 4l == 0 && (unsigned long long)(v1042) % 4l == 0);
            *v1042 = *v1041;
            v1036 += 1l ;
        }
        int v1043;
        v1043 = 0l;
        while (while_method_3(v1043)){
            int v1045;
            v1045 = 0l;
            while (while_method_1(v1045)){
                bool v1047;
                v1047 = 0l <= v1045;
                bool v1049;
                if (v1047){
                    bool v1048;
                    v1048 = v1045 < 4l;
                    v1049 = v1048;
                } else {
                    v1049 = false;
                }
                bool v1050;
                v1050 = v1049 == false;
                if (v1050){
                    assert("The indices should be inside the range of the dimension." && v1049);
                } else {
                }
                bool v1052;
                v1052 = 0l <= v1015;
                bool v1054;
                if (v1052){
                    bool v1053;
                    v1053 = v1015 < 16l;
                    v1054 = v1053;
                } else {
                    v1054 = false;
                }
                bool v1055;
                v1055 = v1054 == false;
                if (v1055){
                    assert("The indices should be inside the range of the dimension." && v1054);
                } else {
                }
                int v1057;
                v1057 = v1015 * 4l;
                int v1058;
                v1058 = v1045 + v1057;
                bool v1059;
                v1059 = 0l <= v1043;
                bool v1061;
                if (v1059){
                    bool v1060;
                    v1060 = v1043 < 1l;
                    v1061 = v1060;
                } else {
                    v1061 = false;
                }
                bool v1062;
                v1062 = v1061 == false;
                if (v1062){
                    assert("The indices should be inside the range of the dimension." && v1061);
                } else {
                }
                int v1064;
                v1064 = v1043 * 64l;
                int v1065;
                v1065 = v1058 + v1064;
                assert("Tensor range check" && 0 <= v1043 && v1043 < 1l);
                assert("Tensor range check" && 0 <= v1045 && v1045 < 4l);
                int v1066;
                v1066 = 4l * v1043;
                int v1067;
                v1067 = v1066 + v1045;
                v1035[v1067] = v1065;
                v1045 += 1l ;
            }
            v1043 += 1l ;
        }
        bool v1068;
        v1068 = 0l <= v1016;
        bool v1069;
        v1069 = v1068 && v1017;
        bool v1070;
        v1070 = v1069 == false;
        if (v1070){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1069);
        } else {
        }
        bool v1072;
        v1072 = v1026 && v1029;
        bool v1073;
        v1073 = v1072 == false;
        if (v1073){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1072);
        } else {
        }
        int v1075;
        v1075 = v1024 * 2l;
        int v1076;
        v1076 = v1075 + v1016;
        float v1077;
        v1077 = 0.0f;
        int v1078;
        v1078 = 0l;
        while (while_method_3(v1078)){
            int v1080;
            v1080 = 0l;
            while (while_method_1(v1080)){
                assert("Tensor range check" && 0 <= v1078 && v1078 < 1l);
                assert("Tensor range check" && 0 <= v1080 && v1080 < 4l);
                int v1082;
                v1082 = 4l * v1078;
                int v1083;
                v1083 = v1082 + v1080;
                float v1084;
                v1084 = v1034[v1083];
                float v1085;
                v1085 = v1077 + v1084;
                v1077 = v1085;
                v1080 += 1l ;
            }
            v1078 += 1l ;
        }
        auto v1086 = cooperative_groups::coalesced_threads();
        int v1087;
        v1087 = threadIdx.x;
        int v1088;
        v1088 = v1087 / 16l;
        auto v1089 = cooperative_groups::labeled_partition(v1086,v1088);
        Closure0 v1090{};
        float v1091;
        v1091 = cooperative_groups::reduce(v1089, v1077, v1090);
        float v1092;
        v1092 = v1091 / 64.0f;
        float v1093[4l];
        int v1094;
        v1094 = 0l;
        while (while_method_3(v1094)){
            int v1096;
            v1096 = 0l;
            while (while_method_1(v1096)){
                assert("Tensor range check" && 0 <= v1094 && v1094 < 1l);
                assert("Tensor range check" && 0 <= v1096 && v1096 < 4l);
                int v1098;
                v1098 = 4l * v1094;
                int v1099;
                v1099 = v1098 + v1096;
                float v1100;
                v1100 = v1034[v1099];
                float v1101;
                v1101 = v1100 - v1092;
                float v1102;
                v1102 = exp(v1101);
                assert("Tensor range check" && 0 <= v1094 && v1094 < 1l);
                assert("Tensor range check" && 0 <= v1096 && v1096 < 4l);
                v1093[v1099] = v1102;
                v1096 += 1l ;
            }
            v1094 += 1l ;
        }
        float v1103;
        v1103 = 0.0f;
        int v1104;
        v1104 = 0l;
        while (while_method_3(v1104)){
            int v1106;
            v1106 = 0l;
            while (while_method_1(v1106)){
                assert("Tensor range check" && 0 <= v1104 && v1104 < 1l);
                assert("Tensor range check" && 0 <= v1106 && v1106 < 4l);
                int v1108;
                v1108 = 4l * v1104;
                int v1109;
                v1109 = v1108 + v1106;
                float v1110;
                v1110 = v1093[v1109];
                float v1111;
                v1111 = v1103 + v1110;
                v1103 = v1111;
                v1106 += 1l ;
            }
            v1104 += 1l ;
        }
        auto v1112 = cooperative_groups::coalesced_threads();
        int v1113;
        v1113 = threadIdx.x;
        int v1114;
        v1114 = v1113 / 16l;
        auto v1115 = cooperative_groups::labeled_partition(v1112,v1114);
        float v1116;
        v1116 = cooperative_groups::reduce(v1115, v1103, v1090);
        float v1117[4l];
        int v1118;
        v1118 = 0l;
        while (while_method_3(v1118)){
            int v1120;
            v1120 = 0l;
            while (while_method_1(v1120)){
                assert("Tensor range check" && 0 <= v1118 && v1118 < 1l);
                assert("Tensor range check" && 0 <= v1120 && v1120 < 4l);
                int v1122;
                v1122 = 4l * v1118;
                int v1123;
                v1123 = v1122 + v1120;
                float v1124;
                v1124 = v1093[v1123];
                float v1125;
                v1125 = v1124 / v1116;
                assert("Tensor range check" && 0 <= v1118 && v1118 < 1l);
                assert("Tensor range check" && 0 <= v1120 && v1120 < 4l);
                v1117[v1123] = v1125;
                v1120 += 1l ;
            }
            v1118 += 1l ;
        }
        float v1126[4l];
        float v1127;
        v1127 = 0.0f;
        int v1128;
        v1128 = 0l;
        while (while_method_3(v1128)){
            assert("Tensor range check" && 0 <= v1128 && v1128 < 1l);
            int v1130;
            v1130 = 4l * v1128;
            assert("Tensor range check" && 0 <= v1128 && v1128 < 1l);
            int v1131; float v1132;
            Tuple0 tmp68 = Tuple0{0l, 0.0f};
            v1131 = tmp68.v0; v1132 = tmp68.v1;
            while (while_method_1(v1131)){
                assert("Tensor range check" && 0 <= v1131 && v1131 < 4l);
                int v1134;
                v1134 = v1131 + v1130;
                float v1135;
                v1135 = v1117[v1134];
                float v1136;
                v1136 = v1132 + v1135;
                v1132 = v1136;
                v1131 += 1l ;
            }
            auto v1137 = cooperative_groups::coalesced_threads();
            int v1138;
            v1138 = threadIdx.x;
            int v1139;
            v1139 = v1138 / 16l;
            auto v1140 = cooperative_groups::labeled_partition(v1137,v1139);
            Closure2 v1141{};
            float v1142;
            v1142 = cooperative_groups::inclusive_scan(v1140, v1132, v1141);
            float v1143;
            v1143 = v1140.shfl_up(v1142,1);
            bool v1144;
            v1144 = v1140.thread_rank() == 0;
            float v1145;
            if (v1144){
                v1145 = 0.0f;
            } else {
                v1145 = v1143;
            }
            float v1146;
            v1146 = v1140.shfl(v1142,v1140.num_threads()-1);
            float v1147;
            v1147 = v1127 + v1145;
            int v1148; float v1149;
            Tuple0 tmp69 = Tuple0{0l, v1147};
            v1148 = tmp69.v0; v1149 = tmp69.v1;
            while (while_method_1(v1148)){
                assert("Tensor range check" && 0 <= v1148 && v1148 < 4l);
                int v1151;
                v1151 = v1148 + v1130;
                float v1152;
                v1152 = v1117[v1151];
                float v1153;
                v1153 = v1149 + v1152;
                assert("Tensor range check" && 0 <= v1148 && v1148 < 4l);
                v1126[v1151] = v1153;
                v1149 = v1153;
                v1148 += 1l ;
            }
            float v1154;
            v1154 = v1127 + v1146;
            v1127 = v1154;
            v1128 += 1l ;
        }
        float v1155[4l];
        bool v1156[4l];
        int v1157;
        v1157 = 0l;
        while (while_method_3(v1157)){
            int v1159;
            v1159 = 0l;
            while (while_method_1(v1159)){
                assert("Tensor range check" && 0 <= v1157 && v1157 < 1l);
                assert("Tensor range check" && 0 <= v1159 && v1159 < 4l);
                int v1161;
                v1161 = 4l * v1157;
                int v1162;
                v1162 = v1161 + v1159;
                float v1163;
                v1163 = v1126[v1162];
                float v1164;
                v1164 = v1117[v1162];
                bool v1165;
                v1165 = v1164 > 0.0f;
                assert("Tensor range check" && 0 <= v1157 && v1157 < 1l);
                assert("Tensor range check" && 0 <= v1159 && v1159 < 4l);
                v1155[v1162] = v1163;
                v1156[v1162] = v1165;
                v1159 += 1l ;
            }
            v1157 += 1l ;
        }
        float v1166; bool v1167;
        Tuple3 tmp70 = Tuple3{-1.0f / 0.0f, false};
        v1166 = tmp70.v0; v1167 = tmp70.v1;
        int v1168;
        v1168 = 0l;
        while (while_method_3(v1168)){
            int v1170;
            v1170 = 0l;
            while (while_method_1(v1170)){
                assert("Tensor range check" && 0 <= v1168 && v1168 < 1l);
                assert("Tensor range check" && 0 <= v1170 && v1170 < 4l);
                int v1172;
                v1172 = 4l * v1168;
                int v1173;
                v1173 = v1172 + v1170;
                float v1174;
                v1174 = v1155[v1173];
                bool v1175;
                v1175 = v1156[v1173];
                float v1182; bool v1183;
                if (v1167){
                    if (v1175){
                        bool v1176;
                        v1176 = v1166 >= v1174;
                        float v1177;
                        if (v1176){
                            v1177 = v1166;
                        } else {
                            v1177 = v1174;
                        }
                        v1182 = v1177; v1183 = true;
                    } else {
                        v1182 = v1166; v1183 = v1167;
                    }
                } else {
                    if (v1175){
                        v1182 = v1174; v1183 = v1175;
                    } else {
                        v1182 = v1166; v1183 = v1167;
                    }
                }
                v1166 = v1182;
                v1167 = v1183;
                v1170 += 1l ;
            }
            v1168 += 1l ;
        }
        auto v1184 = cooperative_groups::coalesced_threads();
        int v1185;
        v1185 = threadIdx.x;
        int v1186;
        v1186 = v1185 / 16l;
        auto v1187 = cooperative_groups::labeled_partition(v1184,v1186);
        Closure5 v1188{};
        float v1189; bool v1190;
        Tuple3 tmp71 = cooperative_groups::reduce(v1187, Tuple3{v1166, v1167}, v1188);
        v1189 = tmp71.v0; v1190 = tmp71.v1;
        bool v1191;
        v1191 = v1190 == false;
        if (v1191){
            assert("The local reduce must be true." && v1190);
        } else {
        }
        float v1193[4l];
        int v1194[4l];
        int v1195;
        v1195 = 0l;
        while (while_method_3(v1195)){
            int v1197;
            v1197 = 0l;
            while (while_method_1(v1197)){
                assert("Tensor range check" && 0 <= v1195 && v1195 < 1l);
                assert("Tensor range check" && 0 <= v1197 && v1197 < 4l);
                int v1199;
                v1199 = 4l * v1195;
                int v1200;
                v1200 = v1199 + v1197;
                int v1201;
                v1201 = v1035[v1200];
                float v1202;
                v1202 = curand_uniform(&v1010);
                assert("Tensor range check" && 0 <= v1195 && v1195 < 1l);
                assert("Tensor range check" && 0 <= v1197 && v1197 < 4l);
                v1193[v1200] = v1202;
                v1194[v1200] = v1201;
                v1197 += 1l ;
            }
            v1195 += 1l ;
        }
        float v1203; int v1204;
        Tuple1 tmp72 = Tuple1{0.0f, 2147483647l};
        v1203 = tmp72.v0; v1204 = tmp72.v1;
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
                v1211 = v1193[v1210];
                int v1212;
                v1212 = v1194[v1210];
                bool v1213;
                v1213 = v1204 < v1212;
                float v1214; int v1215;
                if (v1213){
                    v1214 = v1203; v1215 = v1204;
                } else {
                    v1214 = v1211; v1215 = v1212;
                }
                v1203 = v1214;
                v1204 = v1215;
                v1207 += 1l ;
            }
            v1205 += 1l ;
        }
        auto v1216 = cooperative_groups::coalesced_threads();
        int v1217;
        v1217 = threadIdx.x;
        int v1218;
        v1218 = v1217 / 16l;
        auto v1219 = cooperative_groups::labeled_partition(v1216,v1218);
        Closure6 v1220{};
        float v1221; int v1222;
        Tuple1 tmp73 = cooperative_groups::reduce(v1219, Tuple1{v1203, v1204}, v1220);
        v1221 = tmp73.v0; v1222 = tmp73.v1;
        float v1223;
        v1223 = v1189 * v1221;
        int v1224[4l];
        bool v1225[4l];
        int v1226;
        v1226 = 0l;
        while (while_method_3(v1226)){
            int v1228;
            v1228 = 0l;
            while (while_method_1(v1228)){
                assert("Tensor range check" && 0 <= v1226 && v1226 < 1l);
                assert("Tensor range check" && 0 <= v1228 && v1228 < 4l);
                int v1230;
                v1230 = 4l * v1226;
                int v1231;
                v1231 = v1230 + v1228;
                float v1232;
                v1232 = v1155[v1231];
                bool v1233;
                v1233 = v1156[v1231];
                int v1234;
                v1234 = v1035[v1231];
                int v1237; bool v1238;
                if (v1233){
                    float v1235;
                    v1235 = v1232 - v1223;
                    bool v1236;
                    v1236 = v1235 >= 0.0f;
                    v1237 = v1234; v1238 = v1236;
                } else {
                    v1237 = 2147483647l; v1238 = false;
                }
                assert("Tensor range check" && 0 <= v1226 && v1226 < 1l);
                assert("Tensor range check" && 0 <= v1228 && v1228 < 4l);
                v1224[v1231] = v1237;
                v1225[v1231] = v1238;
                v1228 += 1l ;
            }
            v1226 += 1l ;
        }
        int v1239; bool v1240;
        Tuple4 tmp74 = Tuple4{2147483647l, false};
        v1239 = tmp74.v0; v1240 = tmp74.v1;
        int v1241;
        v1241 = 0l;
        while (while_method_3(v1241)){
            int v1243;
            v1243 = 0l;
            while (while_method_1(v1243)){
                assert("Tensor range check" && 0 <= v1241 && v1241 < 1l);
                assert("Tensor range check" && 0 <= v1243 && v1243 < 4l);
                int v1245;
                v1245 = 4l * v1241;
                int v1246;
                v1246 = v1245 + v1243;
                int v1247;
                v1247 = v1224[v1246];
                bool v1248;
                v1248 = v1225[v1246];
                int v1255; bool v1256;
                if (v1240){
                    if (v1248){
                        bool v1249;
                        v1249 = v1239 < v1247;
                        int v1250;
                        if (v1249){
                            v1250 = v1239;
                        } else {
                            v1250 = v1247;
                        }
                        v1255 = v1250; v1256 = true;
                    } else {
                        v1255 = v1239; v1256 = v1240;
                    }
                } else {
                    if (v1248){
                        v1255 = v1247; v1256 = v1248;
                    } else {
                        v1255 = v1239; v1256 = v1240;
                    }
                }
                v1239 = v1255;
                v1240 = v1256;
                v1243 += 1l ;
            }
            v1241 += 1l ;
        }
        auto v1257 = cooperative_groups::coalesced_threads();
        int v1258;
        v1258 = threadIdx.x;
        int v1259;
        v1259 = v1258 / 16l;
        auto v1260 = cooperative_groups::labeled_partition(v1257,v1259);
        Closure7 v1261{};
        int v1262; bool v1263;
        Tuple4 tmp75 = cooperative_groups::reduce(v1260, Tuple4{v1239, v1240}, v1261);
        v1262 = tmp75.v0; v1263 = tmp75.v1;
        bool v1264;
        v1264 = v1263 == false;
        if (v1264){
            assert("The local reduce must be true." && v1263);
        } else {
        }
        assert("Tensor range check" && 0 <= v1024 && v1024 < 64l);
        int v1266;
        v1266 = 0l;
        while (while_method_3(v1266)){
            assert("Tensor range check" && 0 <= v1266 && v1266 < 1l);
            int v1268;
            v1268 = 64l * v1266;
            int v1269;
            v1269 = v1268 + v1033;
            assert("Tensor range check" && 0 <= v1266 && v1266 < 1l);
            int v1270;
            v1270 = 4l * v1266;
            int4* v1271;
            v1271 = reinterpret_cast<int4*>(v1117 + v1270);
            int4* v1272;
            v1272 = reinterpret_cast<int4*>(v13 + v1269);
            assert("Pointer alignment check" && (unsigned long long)(v1271) % 4l == 0 && (unsigned long long)(v1272) % 4l == 0);
            *v1272 = *v1271;
            v1266 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1024 && v1024 < 64l);
        int v1273;
        v1273 = 2l * v1024;
        int v1274;
        v1274 = v1273 + v1016;
        v14[v1274] = v1262;
        v1024 += 24l ;
    }
    v15.sync() ;
    int v1275;
    v1275 = threadIdx.x;
    unsigned long long v1276;
    v1276 = (unsigned long long)v1275;
    curandStatePhilox4_32_10_t v1277;
    curand_init(12344321ull,v1276,0ull,&v1277);
    int v1278;
    v1278 = threadIdx.x;
    bool v1279;
    v1279 = 0l <= v1278;
    bool v1280;
    v1280 = v1279 == false;
    if (v1280){
        assert("The index needs to be zero or positive." && v1279);
    } else {
    }
    int v1282;
    v1282 = v1278 % 16l;
    int v1283;
    v1283 = v1278 / 16l;
    bool v1284;
    v1284 = v1283 < 2l;
    bool v1285;
    v1285 = v1284 == false;
    if (v1285){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1284);
    } else {
    }
    assert("Tensor range check" && 0 <= v1283 && v1283 < 2l);
    assert("Tensor range check" && 0 <= v1282 && v1282 < 16l);
    int v1287;
    v1287 = 4l * v1282;
    int v1288;
    v1288 = 64l * v1283;
    int v1289;
    v1289 = v1288 + v1287;
    assert("Tensor range check" && 0 <= v1283 && v1283 < 2l);
    assert("Tensor range check" && 0 <= v1282 && v1282 < 16l);
    assert("Tensor range check" && 0 <= v1283 && v1283 < 2l);
    int v1290;
    v1290 = blockIdx.x;
    int v1291;
    v1291 = v1290;
    while (while_method_2(v1291)){
        bool v1293;
        v1293 = 0l <= v1291;
        bool v1294;
        v1294 = v1293 == false;
        if (v1294){
            assert("The index needs to be zero or positive." && v1293);
        } else {
        }
        bool v1296;
        v1296 = v1291 < 64l;
        bool v1297;
        v1297 = v1296 == false;
        if (v1297){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1296);
        } else {
        }
        assert("Tensor range check" && 0 <= v1291 && v1291 < 64l);
        int v1299;
        v1299 = 128l * v1291;
        int v1300;
        v1300 = v1299 + v1289;
        float v1301[4l];
        int v1302[4l];
        int v1303;
        v1303 = 0l;
        while (while_method_3(v1303)){
            assert("Tensor range check" && 0 <= v1303 && v1303 < 1l);
            int v1305;
            v1305 = 4l * v1303;
            assert("Tensor range check" && 0 <= v1303 && v1303 < 1l);
            int v1306;
            v1306 = 64l * v1303;
            int v1307;
            v1307 = v1306 + v1300;
            int4* v1308;
            v1308 = reinterpret_cast<int4*>(v1 + v1307);
            int4* v1309;
            v1309 = reinterpret_cast<int4*>(v1301 + v1305);
            assert("Pointer alignment check" && (unsigned long long)(v1308) % 4l == 0 && (unsigned long long)(v1309) % 4l == 0);
            *v1309 = *v1308;
            v1303 += 1l ;
        }
        int v1310;
        v1310 = 0l;
        while (while_method_3(v1310)){
            int v1312;
            v1312 = 0l;
            while (while_method_1(v1312)){
                bool v1314;
                v1314 = 0l <= v1312;
                bool v1316;
                if (v1314){
                    bool v1315;
                    v1315 = v1312 < 4l;
                    v1316 = v1315;
                } else {
                    v1316 = false;
                }
                bool v1317;
                v1317 = v1316 == false;
                if (v1317){
                    assert("The indices should be inside the range of the dimension." && v1316);
                } else {
                }
                bool v1319;
                v1319 = 0l <= v1282;
                bool v1321;
                if (v1319){
                    bool v1320;
                    v1320 = v1282 < 16l;
                    v1321 = v1320;
                } else {
                    v1321 = false;
                }
                bool v1322;
                v1322 = v1321 == false;
                if (v1322){
                    assert("The indices should be inside the range of the dimension." && v1321);
                } else {
                }
                int v1324;
                v1324 = v1282 * 4l;
                int v1325;
                v1325 = v1312 + v1324;
                bool v1326;
                v1326 = 0l <= v1310;
                bool v1328;
                if (v1326){
                    bool v1327;
                    v1327 = v1310 < 1l;
                    v1328 = v1327;
                } else {
                    v1328 = false;
                }
                bool v1329;
                v1329 = v1328 == false;
                if (v1329){
                    assert("The indices should be inside the range of the dimension." && v1328);
                } else {
                }
                int v1331;
                v1331 = v1310 * 64l;
                int v1332;
                v1332 = v1325 + v1331;
                assert("Tensor range check" && 0 <= v1310 && v1310 < 1l);
                assert("Tensor range check" && 0 <= v1312 && v1312 < 4l);
                int v1333;
                v1333 = 4l * v1310;
                int v1334;
                v1334 = v1333 + v1312;
                v1302[v1334] = v1332;
                v1312 += 1l ;
            }
            v1310 += 1l ;
        }
        bool v1335;
        v1335 = 0l <= v1283;
        bool v1336;
        v1336 = v1335 && v1284;
        bool v1337;
        v1337 = v1336 == false;
        if (v1337){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1336);
        } else {
        }
        bool v1339;
        v1339 = v1293 && v1296;
        bool v1340;
        v1340 = v1339 == false;
        if (v1340){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1339);
        } else {
        }
        int v1342;
        v1342 = v1291 * 2l;
        int v1343;
        v1343 = v1342 + v1283;
        bool v1344[4l];
        int v1345;
        v1345 = 0l;
        while (while_method_3(v1345)){
            int v1347;
            v1347 = 0l;
            while (while_method_1(v1347)){
                assert("Tensor range check" && 0 <= v1345 && v1345 < 1l);
                assert("Tensor range check" && 0 <= v1347 && v1347 < 4l);
                int v1349;
                v1349 = 4l * v1345;
                int v1350;
                v1350 = v1349 + v1347;
                float v1351;
                v1351 = v1301[v1350];
                int v1352;
                v1352 = v1302[v1350];
                bool v1353;
                v1353 = v1352 < 11l;
                assert("Tensor range check" && 0 <= v1345 && v1345 < 1l);
                assert("Tensor range check" && 0 <= v1347 && v1347 < 4l);
                v1344[v1350] = v1353;
                v1347 += 1l ;
            }
            v1345 += 1l ;
        }
        int v1354[4l];
        int v1355;
        v1355 = 0l;
        while (while_method_3(v1355)){
            int v1357;
            v1357 = 0l;
            while (while_method_1(v1357)){
                assert("Tensor range check" && 0 <= v1355 && v1355 < 1l);
                assert("Tensor range check" && 0 <= v1357 && v1357 < 4l);
                int v1359;
                v1359 = 4l * v1355;
                int v1360;
                v1360 = v1359 + v1357;
                bool v1361;
                v1361 = v1344[v1360];
                int v1362;
                if (v1361){
                    v1362 = 1l;
                } else {
                    v1362 = 0l;
                }
                assert("Tensor range check" && 0 <= v1355 && v1355 < 1l);
                assert("Tensor range check" && 0 <= v1357 && v1357 < 4l);
                v1354[v1360] = v1362;
                v1357 += 1l ;
            }
            v1355 += 1l ;
        }
        int v1363;
        v1363 = 0l;
        int v1364;
        v1364 = 0l;
        while (while_method_3(v1364)){
            int v1366;
            v1366 = 0l;
            while (while_method_1(v1366)){
                assert("Tensor range check" && 0 <= v1364 && v1364 < 1l);
                assert("Tensor range check" && 0 <= v1366 && v1366 < 4l);
                int v1368;
                v1368 = 4l * v1364;
                int v1369;
                v1369 = v1368 + v1366;
                int v1370;
                v1370 = v1354[v1369];
                int v1371;
                v1371 = v1363 + v1370;
                v1363 = v1371;
                v1366 += 1l ;
            }
            v1364 += 1l ;
        }
        auto v1372 = cooperative_groups::coalesced_threads();
        int v1373;
        v1373 = threadIdx.x;
        int v1374;
        v1374 = v1373 / 16l;
        auto v1375 = cooperative_groups::labeled_partition(v1372,v1374);
        Closure4 v1376{};
        int v1377;
        v1377 = cooperative_groups::reduce(v1375, v1363, v1376);
        float v1378[4l];
        int v1379;
        v1379 = 0l;
        while (while_method_3(v1379)){
            int v1381;
            v1381 = 0l;
            while (while_method_1(v1381)){
                assert("Tensor range check" && 0 <= v1379 && v1379 < 1l);
                assert("Tensor range check" && 0 <= v1381 && v1381 < 4l);
                int v1383;
                v1383 = 4l * v1379;
                int v1384;
                v1384 = v1383 + v1381;
                float v1385;
                v1385 = v1301[v1384];
                bool v1386;
                v1386 = v1344[v1384];
                float v1387;
                if (v1386){
                    v1387 = v1385;
                } else {
                    v1387 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1379 && v1379 < 1l);
                assert("Tensor range check" && 0 <= v1381 && v1381 < 4l);
                v1378[v1384] = v1387;
                v1381 += 1l ;
            }
            v1379 += 1l ;
        }
        float v1388;
        v1388 = 0.0f;
        int v1389;
        v1389 = 0l;
        while (while_method_3(v1389)){
            int v1391;
            v1391 = 0l;
            while (while_method_1(v1391)){
                assert("Tensor range check" && 0 <= v1389 && v1389 < 1l);
                assert("Tensor range check" && 0 <= v1391 && v1391 < 4l);
                int v1393;
                v1393 = 4l * v1389;
                int v1394;
                v1394 = v1393 + v1391;
                float v1395;
                v1395 = v1378[v1394];
                float v1396;
                v1396 = v1388 + v1395;
                v1388 = v1396;
                v1391 += 1l ;
            }
            v1389 += 1l ;
        }
        auto v1397 = cooperative_groups::coalesced_threads();
        int v1398;
        v1398 = threadIdx.x;
        int v1399;
        v1399 = v1398 / 16l;
        auto v1400 = cooperative_groups::labeled_partition(v1397,v1399);
        Closure0 v1401{};
        float v1402;
        v1402 = cooperative_groups::reduce(v1400, v1388, v1401);
        float v1403;
        v1403 = (float)v1377;
        float v1404;
        v1404 = v1402 / v1403;
        float v1405[4l];
        int v1406;
        v1406 = 0l;
        while (while_method_3(v1406)){
            int v1408;
            v1408 = 0l;
            while (while_method_1(v1408)){
                assert("Tensor range check" && 0 <= v1406 && v1406 < 1l);
                assert("Tensor range check" && 0 <= v1408 && v1408 < 4l);
                int v1410;
                v1410 = 4l * v1406;
                int v1411;
                v1411 = v1410 + v1408;
                float v1412;
                v1412 = v1301[v1411];
                bool v1413;
                v1413 = v1344[v1411];
                float v1414;
                if (v1413){
                    v1414 = v1412;
                } else {
                    v1414 = -1.0f / 0.0f;
                }
                float v1415;
                v1415 = v1414 - v1404;
                float v1416;
                v1416 = exp(v1415);
                assert("Tensor range check" && 0 <= v1406 && v1406 < 1l);
                assert("Tensor range check" && 0 <= v1408 && v1408 < 4l);
                v1405[v1411] = v1416;
                v1408 += 1l ;
            }
            v1406 += 1l ;
        }
        float v1417;
        v1417 = 0.0f;
        int v1418;
        v1418 = 0l;
        while (while_method_3(v1418)){
            int v1420;
            v1420 = 0l;
            while (while_method_1(v1420)){
                assert("Tensor range check" && 0 <= v1418 && v1418 < 1l);
                assert("Tensor range check" && 0 <= v1420 && v1420 < 4l);
                int v1422;
                v1422 = 4l * v1418;
                int v1423;
                v1423 = v1422 + v1420;
                float v1424;
                v1424 = v1405[v1423];
                float v1425;
                v1425 = v1417 + v1424;
                v1417 = v1425;
                v1420 += 1l ;
            }
            v1418 += 1l ;
        }
        auto v1426 = cooperative_groups::coalesced_threads();
        int v1427;
        v1427 = threadIdx.x;
        int v1428;
        v1428 = v1427 / 16l;
        auto v1429 = cooperative_groups::labeled_partition(v1426,v1428);
        float v1430;
        v1430 = cooperative_groups::reduce(v1429, v1417, v1401);
        float v1431[4l];
        int v1432;
        v1432 = 0l;
        while (while_method_3(v1432)){
            int v1434;
            v1434 = 0l;
            while (while_method_1(v1434)){
                assert("Tensor range check" && 0 <= v1432 && v1432 < 1l);
                assert("Tensor range check" && 0 <= v1434 && v1434 < 4l);
                int v1436;
                v1436 = 4l * v1432;
                int v1437;
                v1437 = v1436 + v1434;
                float v1438;
                v1438 = v1405[v1437];
                float v1439;
                v1439 = v1438 / v1430;
                assert("Tensor range check" && 0 <= v1432 && v1432 < 1l);
                assert("Tensor range check" && 0 <= v1434 && v1434 < 4l);
                v1431[v1437] = v1439;
                v1434 += 1l ;
            }
            v1432 += 1l ;
        }
        float v1440[4l];
        float v1441;
        v1441 = 0.0f;
        int v1442;
        v1442 = 0l;
        while (while_method_3(v1442)){
            assert("Tensor range check" && 0 <= v1442 && v1442 < 1l);
            int v1444;
            v1444 = 4l * v1442;
            assert("Tensor range check" && 0 <= v1442 && v1442 < 1l);
            int v1445; float v1446;
            Tuple0 tmp76 = Tuple0{0l, 0.0f};
            v1445 = tmp76.v0; v1446 = tmp76.v1;
            while (while_method_1(v1445)){
                assert("Tensor range check" && 0 <= v1445 && v1445 < 4l);
                int v1448;
                v1448 = v1445 + v1444;
                float v1449;
                v1449 = v1431[v1448];
                float v1450;
                v1450 = v1446 + v1449;
                v1446 = v1450;
                v1445 += 1l ;
            }
            auto v1451 = cooperative_groups::coalesced_threads();
            int v1452;
            v1452 = threadIdx.x;
            int v1453;
            v1453 = v1452 / 16l;
            auto v1454 = cooperative_groups::labeled_partition(v1451,v1453);
            Closure2 v1455{};
            float v1456;
            v1456 = cooperative_groups::inclusive_scan(v1454, v1446, v1455);
            float v1457;
            v1457 = v1454.shfl_up(v1456,1);
            bool v1458;
            v1458 = v1454.thread_rank() == 0;
            float v1459;
            if (v1458){
                v1459 = 0.0f;
            } else {
                v1459 = v1457;
            }
            float v1460;
            v1460 = v1454.shfl(v1456,v1454.num_threads()-1);
            float v1461;
            v1461 = v1441 + v1459;
            int v1462; float v1463;
            Tuple0 tmp77 = Tuple0{0l, v1461};
            v1462 = tmp77.v0; v1463 = tmp77.v1;
            while (while_method_1(v1462)){
                assert("Tensor range check" && 0 <= v1462 && v1462 < 4l);
                int v1465;
                v1465 = v1462 + v1444;
                float v1466;
                v1466 = v1431[v1465];
                float v1467;
                v1467 = v1463 + v1466;
                assert("Tensor range check" && 0 <= v1462 && v1462 < 4l);
                v1440[v1465] = v1467;
                v1463 = v1467;
                v1462 += 1l ;
            }
            float v1468;
            v1468 = v1441 + v1460;
            v1441 = v1468;
            v1442 += 1l ;
        }
        float v1469[4l];
        bool v1470[4l];
        int v1471;
        v1471 = 0l;
        while (while_method_3(v1471)){
            int v1473;
            v1473 = 0l;
            while (while_method_1(v1473)){
                assert("Tensor range check" && 0 <= v1471 && v1471 < 1l);
                assert("Tensor range check" && 0 <= v1473 && v1473 < 4l);
                int v1475;
                v1475 = 4l * v1471;
                int v1476;
                v1476 = v1475 + v1473;
                float v1477;
                v1477 = v1440[v1476];
                float v1478;
                v1478 = v1431[v1476];
                bool v1479;
                v1479 = v1478 > 0.0f;
                assert("Tensor range check" && 0 <= v1471 && v1471 < 1l);
                assert("Tensor range check" && 0 <= v1473 && v1473 < 4l);
                v1469[v1476] = v1477;
                v1470[v1476] = v1479;
                v1473 += 1l ;
            }
            v1471 += 1l ;
        }
        float v1480; bool v1481;
        Tuple3 tmp78 = Tuple3{-1.0f / 0.0f, false};
        v1480 = tmp78.v0; v1481 = tmp78.v1;
        int v1482;
        v1482 = 0l;
        while (while_method_3(v1482)){
            int v1484;
            v1484 = 0l;
            while (while_method_1(v1484)){
                assert("Tensor range check" && 0 <= v1482 && v1482 < 1l);
                assert("Tensor range check" && 0 <= v1484 && v1484 < 4l);
                int v1486;
                v1486 = 4l * v1482;
                int v1487;
                v1487 = v1486 + v1484;
                float v1488;
                v1488 = v1469[v1487];
                bool v1489;
                v1489 = v1470[v1487];
                float v1496; bool v1497;
                if (v1481){
                    if (v1489){
                        bool v1490;
                        v1490 = v1480 >= v1488;
                        float v1491;
                        if (v1490){
                            v1491 = v1480;
                        } else {
                            v1491 = v1488;
                        }
                        v1496 = v1491; v1497 = true;
                    } else {
                        v1496 = v1480; v1497 = v1481;
                    }
                } else {
                    if (v1489){
                        v1496 = v1488; v1497 = v1489;
                    } else {
                        v1496 = v1480; v1497 = v1481;
                    }
                }
                v1480 = v1496;
                v1481 = v1497;
                v1484 += 1l ;
            }
            v1482 += 1l ;
        }
        auto v1498 = cooperative_groups::coalesced_threads();
        int v1499;
        v1499 = threadIdx.x;
        int v1500;
        v1500 = v1499 / 16l;
        auto v1501 = cooperative_groups::labeled_partition(v1498,v1500);
        Closure5 v1502{};
        float v1503; bool v1504;
        Tuple3 tmp79 = cooperative_groups::reduce(v1501, Tuple3{v1480, v1481}, v1502);
        v1503 = tmp79.v0; v1504 = tmp79.v1;
        bool v1505;
        v1505 = v1504 == false;
        if (v1505){
            assert("The local reduce must be true." && v1504);
        } else {
        }
        float v1507[4l];
        int v1508[4l];
        int v1509;
        v1509 = 0l;
        while (while_method_3(v1509)){
            int v1511;
            v1511 = 0l;
            while (while_method_1(v1511)){
                assert("Tensor range check" && 0 <= v1509 && v1509 < 1l);
                assert("Tensor range check" && 0 <= v1511 && v1511 < 4l);
                int v1513;
                v1513 = 4l * v1509;
                int v1514;
                v1514 = v1513 + v1511;
                int v1515;
                v1515 = v1302[v1514];
                float v1516;
                v1516 = curand_uniform(&v1277);
                assert("Tensor range check" && 0 <= v1509 && v1509 < 1l);
                assert("Tensor range check" && 0 <= v1511 && v1511 < 4l);
                v1507[v1514] = v1516;
                v1508[v1514] = v1515;
                v1511 += 1l ;
            }
            v1509 += 1l ;
        }
        float v1517; int v1518;
        Tuple1 tmp80 = Tuple1{0.0f, 2147483647l};
        v1517 = tmp80.v0; v1518 = tmp80.v1;
        int v1519;
        v1519 = 0l;
        while (while_method_3(v1519)){
            int v1521;
            v1521 = 0l;
            while (while_method_1(v1521)){
                assert("Tensor range check" && 0 <= v1519 && v1519 < 1l);
                assert("Tensor range check" && 0 <= v1521 && v1521 < 4l);
                int v1523;
                v1523 = 4l * v1519;
                int v1524;
                v1524 = v1523 + v1521;
                float v1525;
                v1525 = v1507[v1524];
                int v1526;
                v1526 = v1508[v1524];
                bool v1527;
                v1527 = v1518 < v1526;
                float v1528; int v1529;
                if (v1527){
                    v1528 = v1517; v1529 = v1518;
                } else {
                    v1528 = v1525; v1529 = v1526;
                }
                v1517 = v1528;
                v1518 = v1529;
                v1521 += 1l ;
            }
            v1519 += 1l ;
        }
        auto v1530 = cooperative_groups::coalesced_threads();
        int v1531;
        v1531 = threadIdx.x;
        int v1532;
        v1532 = v1531 / 16l;
        auto v1533 = cooperative_groups::labeled_partition(v1530,v1532);
        Closure6 v1534{};
        float v1535; int v1536;
        Tuple1 tmp81 = cooperative_groups::reduce(v1533, Tuple1{v1517, v1518}, v1534);
        v1535 = tmp81.v0; v1536 = tmp81.v1;
        float v1537;
        v1537 = v1503 * v1535;
        int v1538[4l];
        bool v1539[4l];
        int v1540;
        v1540 = 0l;
        while (while_method_3(v1540)){
            int v1542;
            v1542 = 0l;
            while (while_method_1(v1542)){
                assert("Tensor range check" && 0 <= v1540 && v1540 < 1l);
                assert("Tensor range check" && 0 <= v1542 && v1542 < 4l);
                int v1544;
                v1544 = 4l * v1540;
                int v1545;
                v1545 = v1544 + v1542;
                float v1546;
                v1546 = v1469[v1545];
                bool v1547;
                v1547 = v1470[v1545];
                int v1548;
                v1548 = v1302[v1545];
                int v1551; bool v1552;
                if (v1547){
                    float v1549;
                    v1549 = v1546 - v1537;
                    bool v1550;
                    v1550 = v1549 >= 0.0f;
                    v1551 = v1548; v1552 = v1550;
                } else {
                    v1551 = 2147483647l; v1552 = false;
                }
                assert("Tensor range check" && 0 <= v1540 && v1540 < 1l);
                assert("Tensor range check" && 0 <= v1542 && v1542 < 4l);
                v1538[v1545] = v1551;
                v1539[v1545] = v1552;
                v1542 += 1l ;
            }
            v1540 += 1l ;
        }
        int v1553; bool v1554;
        Tuple4 tmp82 = Tuple4{2147483647l, false};
        v1553 = tmp82.v0; v1554 = tmp82.v1;
        int v1555;
        v1555 = 0l;
        while (while_method_3(v1555)){
            int v1557;
            v1557 = 0l;
            while (while_method_1(v1557)){
                assert("Tensor range check" && 0 <= v1555 && v1555 < 1l);
                assert("Tensor range check" && 0 <= v1557 && v1557 < 4l);
                int v1559;
                v1559 = 4l * v1555;
                int v1560;
                v1560 = v1559 + v1557;
                int v1561;
                v1561 = v1538[v1560];
                bool v1562;
                v1562 = v1539[v1560];
                int v1569; bool v1570;
                if (v1554){
                    if (v1562){
                        bool v1563;
                        v1563 = v1553 < v1561;
                        int v1564;
                        if (v1563){
                            v1564 = v1553;
                        } else {
                            v1564 = v1561;
                        }
                        v1569 = v1564; v1570 = true;
                    } else {
                        v1569 = v1553; v1570 = v1554;
                    }
                } else {
                    if (v1562){
                        v1569 = v1561; v1570 = v1562;
                    } else {
                        v1569 = v1553; v1570 = v1554;
                    }
                }
                v1553 = v1569;
                v1554 = v1570;
                v1557 += 1l ;
            }
            v1555 += 1l ;
        }
        auto v1571 = cooperative_groups::coalesced_threads();
        int v1572;
        v1572 = threadIdx.x;
        int v1573;
        v1573 = v1572 / 16l;
        auto v1574 = cooperative_groups::labeled_partition(v1571,v1573);
        Closure7 v1575{};
        int v1576; bool v1577;
        Tuple4 tmp83 = cooperative_groups::reduce(v1574, Tuple4{v1553, v1554}, v1575);
        v1576 = tmp83.v0; v1577 = tmp83.v1;
        bool v1578;
        v1578 = v1577 == false;
        if (v1578){
            assert("The local reduce must be true." && v1577);
        } else {
        }
        assert("Tensor range check" && 0 <= v1291 && v1291 < 64l);
        int v1580;
        v1580 = 0l;
        while (while_method_3(v1580)){
            assert("Tensor range check" && 0 <= v1580 && v1580 < 1l);
            int v1582;
            v1582 = 64l * v1580;
            int v1583;
            v1583 = v1582 + v1300;
            assert("Tensor range check" && 0 <= v1580 && v1580 < 1l);
            int v1584;
            v1584 = 4l * v1580;
            int4* v1585;
            v1585 = reinterpret_cast<int4*>(v1431 + v1584);
            int4* v1586;
            v1586 = reinterpret_cast<int4*>(v13 + v1583);
            assert("Pointer alignment check" && (unsigned long long)(v1585) % 4l == 0 && (unsigned long long)(v1586) % 4l == 0);
            *v1586 = *v1585;
            v1580 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1291 && v1291 < 64l);
        int v1587;
        v1587 = 2l * v1291;
        int v1588;
        v1588 = v1587 + v1283;
        v14[v1588] = v1576;
        v1291 += 24l ;
    }
    v15.sync() ;
    return ;
}
extern "C" __global__ void entry5(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14) {
    auto v15 = cooperative_groups::this_grid();
    int v16;
    v16 = threadIdx.x;
    int v17;
    v17 = v16;
    while (while_method_0(v17)){
        bool v19;
        v19 = 0l <= v17;
        bool v20;
        v20 = v19 == false;
        if (v20){
            assert("The index needs to be zero or positive." && v19);
        } else {
        }
        int v22;
        v22 = v17 % 32l;
        int v23;
        v23 = v17 / 32l;
        bool v24;
        v24 = v23 < 64l;
        bool v25;
        v25 = v24 == false;
        if (v25){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
        } else {
        }
        assert("Tensor range check" && 0 <= v23 && v23 < 64l);
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        int v27;
        v27 = 4l * v22;
        int v28;
        v28 = 128l * v23;
        int v29;
        v29 = v28 + v27;
        assert("Tensor range check" && 0 <= v23 && v23 < 64l);
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        float v30[4l];
        float v31[4l];
        int4* v32;
        v32 = reinterpret_cast<int4*>(v1 + v29);
        int4* v33;
        v33 = reinterpret_cast<int4*>(v30 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v32) % 4l == 0 && (unsigned long long)(v33) % 4l == 0);
        *v33 = *v32;
        // Pushing the loop unrolling to: 0
        int v34;
        v34 = 0l;
        #pragma unroll
        while (while_method_1(v34)){
            assert("Tensor range check" && 0 <= v34 && v34 < 4l);
            float v36;
            v36 = v30[v34];
            float v37;
            v37 = 1.0f + v36;
            assert("Tensor range check" && 0 <= v34 && v34 < 4l);
            v31[v34] = v37;
            v34 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v38;
        v38 = reinterpret_cast<int4*>(v31 + 0l);
        int4* v39;
        v39 = reinterpret_cast<int4*>(v1 + v29);
        assert("Pointer alignment check" && (unsigned long long)(v38) % 4l == 0 && (unsigned long long)(v39) % 4l == 0);
        *v39 = *v38;
        v17 += 32l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v40;
    v40 = threadIdx.x;
    bool v41;
    v41 = 0l <= v40;
    bool v42;
    v42 = v41 == false;
    if (v42){
        assert("The index needs to be zero or positive." && v41);
    } else {
    }
    int v44;
    v44 = v40 % 32l;
    int v45;
    v45 = v40 / 32l;
    bool v46;
    v46 = v45 < 1l;
    bool v47;
    v47 = v46 == false;
    if (v47){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v46);
    } else {
    }
    assert("Tensor range check" && 0 <= v45 && v45 < 1l);
    assert("Tensor range check" && 0 <= v44 && v44 < 32l);
    int v49;
    v49 = 4l * v44;
    int v50;
    v50 = 128l * v45;
    int v51;
    v51 = v50 + v49;
    assert("Tensor range check" && 0 <= v45 && v45 < 1l);
    assert("Tensor range check" && 0 <= v44 && v44 < 32l);
    int v52;
    v52 = blockIdx.x;
    int v53;
    v53 = v52;
    while (while_method_2(v53)){
        bool v55;
        v55 = 0l <= v53;
        bool v56;
        v56 = v55 == false;
        if (v56){
            assert("The index needs to be zero or positive." && v55);
        } else {
        }
        bool v58;
        v58 = v53 < 64l;
        bool v59;
        v59 = v58 == false;
        if (v59){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v58);
        } else {
        }
        assert("Tensor range check" && 0 <= v53 && v53 < 64l);
        int v61;
        v61 = 128l * v53;
        int v62;
        v62 = v61 + v51;
        int v63[4l];
        int v64[4l];
        int v65;
        v65 = 0l;
        while (while_method_3(v65)){
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            int v67;
            v67 = 4l * v65;
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            int v68;
            v68 = 128l * v65;
            int v69;
            v69 = v68 + v62;
            int4* v70;
            v70 = reinterpret_cast<int4*>(v0 + v69);
            int4* v71;
            v71 = reinterpret_cast<int4*>(v63 + v67);
            assert("Pointer alignment check" && (unsigned long long)(v70) % 4l == 0 && (unsigned long long)(v71) % 4l == 0);
            *v71 = *v70;
            v65 += 1l ;
        }
        int v72;
        v72 = 0l;
        while (while_method_3(v72)){
            int v74;
            v74 = 0l;
            while (while_method_1(v74)){
                bool v76;
                v76 = 0l <= v74;
                bool v78;
                if (v76){
                    bool v77;
                    v77 = v74 < 4l;
                    v78 = v77;
                } else {
                    v78 = false;
                }
                bool v79;
                v79 = v78 == false;
                if (v79){
                    assert("The indices should be inside the range of the dimension." && v78);
                } else {
                }
                bool v81;
                v81 = 0l <= v44;
                bool v83;
                if (v81){
                    bool v82;
                    v82 = v44 < 32l;
                    v83 = v82;
                } else {
                    v83 = false;
                }
                bool v84;
                v84 = v83 == false;
                if (v84){
                    assert("The indices should be inside the range of the dimension." && v83);
                } else {
                }
                int v86;
                v86 = v44 * 4l;
                int v87;
                v87 = v74 + v86;
                bool v88;
                v88 = 0l <= v72;
                bool v90;
                if (v88){
                    bool v89;
                    v89 = v72 < 1l;
                    v90 = v89;
                } else {
                    v90 = false;
                }
                bool v91;
                v91 = v90 == false;
                if (v91){
                    assert("The indices should be inside the range of the dimension." && v90);
                } else {
                }
                int v93;
                v93 = v72 * 128l;
                int v94;
                v94 = v87 + v93;
                assert("Tensor range check" && 0 <= v72 && v72 < 1l);
                assert("Tensor range check" && 0 <= v74 && v74 < 4l);
                int v95;
                v95 = 4l * v72;
                int v96;
                v96 = v95 + v74;
                v64[v96] = v94;
                v74 += 1l ;
            }
            v72 += 1l ;
        }
        bool v97;
        v97 = 0l <= v45;
        bool v98;
        v98 = v97 && v46;
        bool v99;
        v99 = v98 == false;
        if (v99){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v98);
        } else {
        }
        bool v101;
        v101 = v55 && v58;
        bool v102;
        v102 = v101 == false;
        if (v102){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v101);
        } else {
        }
        int v104;
        v104 = v53 + v45;
        assert("Tensor range check" && 0 <= v53 && v53 < 64l);
        int v105;
        v105 = 0l;
        while (while_method_3(v105)){
            assert("Tensor range check" && 0 <= v105 && v105 < 1l);
            int v107;
            v107 = 128l * v105;
            int v108;
            v108 = v107 + v62;
            assert("Tensor range check" && 0 <= v105 && v105 < 1l);
            int v109;
            v109 = 4l * v105;
            int4* v110;
            v110 = reinterpret_cast<int4*>(v63 + v109);
            int4* v111;
            v111 = reinterpret_cast<int4*>(v2 + v108);
            assert("Pointer alignment check" && (unsigned long long)(v110) % 4l == 0 && (unsigned long long)(v111) % 4l == 0);
            *v111 = *v110;
            v105 += 1l ;
        }
        v53 += 24l ;
    }
    v15.sync() ;
    int v112;
    v112 = threadIdx.x;
    bool v113;
    v113 = 0l <= v112;
    bool v114;
    v114 = v113 == false;
    if (v114){
        assert("The index needs to be zero or positive." && v113);
    } else {
    }
    int v116;
    v116 = v112 % 32l;
    int v117;
    v117 = v112 / 32l;
    bool v118;
    v118 = v117 < 1l;
    bool v119;
    v119 = v118 == false;
    if (v119){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v118);
    } else {
    }
    assert("Tensor range check" && 0 <= v117 && v117 < 1l);
    assert("Tensor range check" && 0 <= v116 && v116 < 32l);
    int v121;
    v121 = 4l * v116;
    int v122;
    v122 = 128l * v117;
    int v123;
    v123 = v122 + v121;
    assert("Tensor range check" && 0 <= v117 && v117 < 1l);
    assert("Tensor range check" && 0 <= v116 && v116 < 32l);
    int v124;
    v124 = blockIdx.x;
    int v125;
    v125 = v124;
    while (while_method_2(v125)){
        bool v127;
        v127 = 0l <= v125;
        bool v128;
        v128 = v127 == false;
        if (v128){
            assert("The index needs to be zero or positive." && v127);
        } else {
        }
        bool v130;
        v130 = v125 < 64l;
        bool v131;
        v131 = v130 == false;
        if (v131){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v130);
        } else {
        }
        assert("Tensor range check" && 0 <= v125 && v125 < 64l);
        int v133;
        v133 = 128l * v125;
        int v134;
        v134 = v133 + v123;
        float v135[4l];
        int v136[4l];
        int v137;
        v137 = 0l;
        while (while_method_3(v137)){
            assert("Tensor range check" && 0 <= v137 && v137 < 1l);
            int v139;
            v139 = 4l * v137;
            assert("Tensor range check" && 0 <= v137 && v137 < 1l);
            int v140;
            v140 = 128l * v137;
            int v141;
            v141 = v140 + v134;
            int4* v142;
            v142 = reinterpret_cast<int4*>(v1 + v141);
            int4* v143;
            v143 = reinterpret_cast<int4*>(v135 + v139);
            assert("Pointer alignment check" && (unsigned long long)(v142) % 4l == 0 && (unsigned long long)(v143) % 4l == 0);
            *v143 = *v142;
            v137 += 1l ;
        }
        int v144;
        v144 = 0l;
        while (while_method_3(v144)){
            int v146;
            v146 = 0l;
            while (while_method_1(v146)){
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
                v153 = 0l <= v116;
                bool v155;
                if (v153){
                    bool v154;
                    v154 = v116 < 32l;
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
                v158 = v116 * 4l;
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
                v165 = v144 * 128l;
                int v166;
                v166 = v159 + v165;
                assert("Tensor range check" && 0 <= v144 && v144 < 1l);
                assert("Tensor range check" && 0 <= v146 && v146 < 4l);
                int v167;
                v167 = 4l * v144;
                int v168;
                v168 = v167 + v146;
                v136[v168] = v166;
                v146 += 1l ;
            }
            v144 += 1l ;
        }
        bool v169;
        v169 = 0l <= v117;
        bool v170;
        v170 = v169 && v118;
        bool v171;
        v171 = v170 == false;
        if (v171){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v170);
        } else {
        }
        bool v173;
        v173 = v127 && v130;
        bool v174;
        v174 = v173 == false;
        if (v174){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v173);
        } else {
        }
        int v176;
        v176 = v125 + v117;
        int v177[4l];
        int v178[4l];
        int v179;
        v179 = 0l;
        while (while_method_3(v179)){
            int v181;
            v181 = 0l;
            while (while_method_1(v181)){
                assert("Tensor range check" && 0 <= v179 && v179 < 1l);
                assert("Tensor range check" && 0 <= v181 && v181 < 4l);
                int v183;
                v183 = 4l * v179;
                int v184;
                v184 = v183 + v181;
                int v185;
                v185 = v136[v184];
                assert("Tensor range check" && 0 <= v179 && v179 < 1l);
                assert("Tensor range check" && 0 <= v181 && v181 < 4l);
                v177[v184] = v176;
                v178[v184] = v185;
                v181 += 1l ;
            }
            v179 += 1l ;
        }
        assert("Tensor range check" && 0 <= v125 && v125 < 64l);
        int v186;
        v186 = 0l;
        while (while_method_3(v186)){
            assert("Tensor range check" && 0 <= v186 && v186 < 1l);
            int v188;
            v188 = 128l * v186;
            int v189;
            v189 = v188 + v134;
            assert("Tensor range check" && 0 <= v186 && v186 < 1l);
            int v190;
            v190 = 4l * v186;
            int4* v191;
            v191 = reinterpret_cast<int4*>(v177 + v190);
            int4* v192;
            v192 = reinterpret_cast<int4*>(v9 + v189);
            assert("Pointer alignment check" && (unsigned long long)(v191) % 4l == 0 && (unsigned long long)(v192) % 4l == 0);
            *v192 = *v191;
            int4* v193;
            v193 = reinterpret_cast<int4*>(v178 + v190);
            int4* v194;
            v194 = reinterpret_cast<int4*>(v10 + v189);
            assert("Pointer alignment check" && (unsigned long long)(v193) % 4l == 0 && (unsigned long long)(v194) % 4l == 0);
            *v194 = *v193;
            v186 += 1l ;
        }
        v125 += 24l ;
    }
    v15.sync() ;
    int v195;
    v195 = threadIdx.x;
    bool v196;
    v196 = 0l <= v195;
    bool v197;
    v197 = v196 == false;
    if (v197){
        assert("The index needs to be zero or positive." && v196);
    } else {
    }
    int v199;
    v199 = v195 % 32l;
    int v200;
    v200 = v195 / 32l;
    bool v201;
    v201 = v200 < 1l;
    bool v202;
    v202 = v201 == false;
    if (v202){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v201);
    } else {
    }
    assert("Tensor range check" && 0 <= v200 && v200 < 1l);
    assert("Tensor range check" && 0 <= v199 && v199 < 32l);
    int v204;
    v204 = 4l * v199;
    int v205;
    v205 = 128l * v200;
    int v206;
    v206 = v205 + v204;
    assert("Tensor range check" && 0 <= v200 && v200 < 1l);
    int v207;
    v207 = blockIdx.x;
    int v208;
    v208 = v207;
    while (while_method_2(v208)){
        bool v210;
        v210 = 0l <= v208;
        bool v211;
        v211 = v210 == false;
        if (v211){
            assert("The index needs to be zero or positive." && v210);
        } else {
        }
        bool v213;
        v213 = v208 < 64l;
        bool v214;
        v214 = v213 == false;
        if (v214){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v213);
        } else {
        }
        assert("Tensor range check" && 0 <= v208 && v208 < 64l);
        int v216;
        v216 = 128l * v208;
        int v217;
        v217 = v216 + v206;
        float v218[4l];
        int v219[4l];
        int v220;
        v220 = 0l;
        while (while_method_3(v220)){
            assert("Tensor range check" && 0 <= v220 && v220 < 1l);
            int v222;
            v222 = 4l * v220;
            assert("Tensor range check" && 0 <= v220 && v220 < 1l);
            int v223;
            v223 = 128l * v220;
            int v224;
            v224 = v223 + v217;
            int4* v225;
            v225 = reinterpret_cast<int4*>(v1 + v224);
            int4* v226;
            v226 = reinterpret_cast<int4*>(v218 + v222);
            assert("Pointer alignment check" && (unsigned long long)(v225) % 4l == 0 && (unsigned long long)(v226) % 4l == 0);
            *v226 = *v225;
            v220 += 1l ;
        }
        int v227;
        v227 = 0l;
        while (while_method_3(v227)){
            int v229;
            v229 = 0l;
            while (while_method_1(v229)){
                bool v231;
                v231 = 0l <= v229;
                bool v233;
                if (v231){
                    bool v232;
                    v232 = v229 < 4l;
                    v233 = v232;
                } else {
                    v233 = false;
                }
                bool v234;
                v234 = v233 == false;
                if (v234){
                    assert("The indices should be inside the range of the dimension." && v233);
                } else {
                }
                bool v236;
                v236 = 0l <= v199;
                bool v238;
                if (v236){
                    bool v237;
                    v237 = v199 < 32l;
                    v238 = v237;
                } else {
                    v238 = false;
                }
                bool v239;
                v239 = v238 == false;
                if (v239){
                    assert("The indices should be inside the range of the dimension." && v238);
                } else {
                }
                int v241;
                v241 = v199 * 4l;
                int v242;
                v242 = v229 + v241;
                bool v243;
                v243 = 0l <= v227;
                bool v245;
                if (v243){
                    bool v244;
                    v244 = v227 < 1l;
                    v245 = v244;
                } else {
                    v245 = false;
                }
                bool v246;
                v246 = v245 == false;
                if (v246){
                    assert("The indices should be inside the range of the dimension." && v245);
                } else {
                }
                int v248;
                v248 = v227 * 128l;
                int v249;
                v249 = v242 + v248;
                assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                assert("Tensor range check" && 0 <= v229 && v229 < 4l);
                int v250;
                v250 = 4l * v227;
                int v251;
                v251 = v250 + v229;
                v219[v251] = v249;
                v229 += 1l ;
            }
            v227 += 1l ;
        }
        bool v252;
        v252 = 0l <= v200;
        bool v253;
        v253 = v252 && v201;
        bool v254;
        v254 = v253 == false;
        if (v254){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v253);
        } else {
        }
        bool v256;
        v256 = v210 && v213;
        bool v257;
        v257 = v256 == false;
        if (v257){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v256);
        } else {
        }
        int v259;
        v259 = v208 + v200;
        assert("Tensor range check" && 0 <= v208 && v208 < 64l);
        v11[v259] = v259;
        v208 += 24l ;
    }
    v15.sync() ;
    int v260;
    v260 = threadIdx.x;
    bool v261;
    v261 = 0l <= v260;
    bool v262;
    v262 = v261 == false;
    if (v262){
        assert("The index needs to be zero or positive." && v261);
    } else {
    }
    int v264;
    v264 = v260 % 32l;
    int v265;
    v265 = v260 / 32l;
    bool v266;
    v266 = v265 < 1l;
    bool v267;
    v267 = v266 == false;
    if (v267){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v266);
    } else {
    }
    assert("Tensor range check" && 0 <= v265 && v265 < 1l);
    assert("Tensor range check" && 0 <= v264 && v264 < 32l);
    int v269;
    v269 = 4l * v264;
    int v270;
    v270 = 128l * v265;
    int v271;
    v271 = v270 + v269;
    assert("Tensor range check" && 0 <= v265 && v265 < 1l);
    assert("Tensor range check" && 0 <= v264 && v264 < 32l);
    int v272;
    v272 = blockIdx.x;
    int v273;
    v273 = v272;
    while (while_method_2(v273)){
        bool v275;
        v275 = 0l <= v273;
        bool v276;
        v276 = v275 == false;
        if (v276){
            assert("The index needs to be zero or positive." && v275);
        } else {
        }
        bool v278;
        v278 = v273 < 64l;
        bool v279;
        v279 = v278 == false;
        if (v279){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v278);
        } else {
        }
        assert("Tensor range check" && 0 <= v273 && v273 < 64l);
        int v281;
        v281 = 128l * v273;
        int v282;
        v282 = v281 + v271;
        float v283[4l];
        int v284[4l];
        int v285;
        v285 = 0l;
        while (while_method_3(v285)){
            assert("Tensor range check" && 0 <= v285 && v285 < 1l);
            int v287;
            v287 = 4l * v285;
            assert("Tensor range check" && 0 <= v285 && v285 < 1l);
            int v288;
            v288 = 128l * v285;
            int v289;
            v289 = v288 + v282;
            int4* v290;
            v290 = reinterpret_cast<int4*>(v1 + v289);
            int4* v291;
            v291 = reinterpret_cast<int4*>(v283 + v287);
            assert("Pointer alignment check" && (unsigned long long)(v290) % 4l == 0 && (unsigned long long)(v291) % 4l == 0);
            *v291 = *v290;
            v285 += 1l ;
        }
        int v292;
        v292 = 0l;
        while (while_method_3(v292)){
            int v294;
            v294 = 0l;
            while (while_method_1(v294)){
                bool v296;
                v296 = 0l <= v294;
                bool v298;
                if (v296){
                    bool v297;
                    v297 = v294 < 4l;
                    v298 = v297;
                } else {
                    v298 = false;
                }
                bool v299;
                v299 = v298 == false;
                if (v299){
                    assert("The indices should be inside the range of the dimension." && v298);
                } else {
                }
                bool v301;
                v301 = 0l <= v264;
                bool v303;
                if (v301){
                    bool v302;
                    v302 = v264 < 32l;
                    v303 = v302;
                } else {
                    v303 = false;
                }
                bool v304;
                v304 = v303 == false;
                if (v304){
                    assert("The indices should be inside the range of the dimension." && v303);
                } else {
                }
                int v306;
                v306 = v264 * 4l;
                int v307;
                v307 = v294 + v306;
                bool v308;
                v308 = 0l <= v292;
                bool v310;
                if (v308){
                    bool v309;
                    v309 = v292 < 1l;
                    v310 = v309;
                } else {
                    v310 = false;
                }
                bool v311;
                v311 = v310 == false;
                if (v311){
                    assert("The indices should be inside the range of the dimension." && v310);
                } else {
                }
                int v313;
                v313 = v292 * 128l;
                int v314;
                v314 = v307 + v313;
                assert("Tensor range check" && 0 <= v292 && v292 < 1l);
                assert("Tensor range check" && 0 <= v294 && v294 < 4l);
                int v315;
                v315 = 4l * v292;
                int v316;
                v316 = v315 + v294;
                v284[v316] = v314;
                v294 += 1l ;
            }
            v292 += 1l ;
        }
        bool v317;
        v317 = 0l <= v265;
        bool v318;
        v318 = v317 && v266;
        bool v319;
        v319 = v318 == false;
        if (v319){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v318);
        } else {
        }
        bool v321;
        v321 = v275 && v278;
        bool v322;
        v322 = v321 == false;
        if (v322){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v321);
        } else {
        }
        int v324;
        v324 = v273 + v265;
        float v325;
        v325 = 0.0f;
        int v326;
        v326 = 0l;
        while (while_method_3(v326)){
            int v328;
            v328 = 0l;
            while (while_method_1(v328)){
                assert("Tensor range check" && 0 <= v326 && v326 < 1l);
                assert("Tensor range check" && 0 <= v328 && v328 < 4l);
                int v330;
                v330 = 4l * v326;
                int v331;
                v331 = v330 + v328;
                float v332;
                v332 = v283[v331];
                float v333;
                v333 = v325 + v332;
                v325 = v333;
                v328 += 1l ;
            }
            v326 += 1l ;
        }
        auto v334 = cooperative_groups::coalesced_threads();
        int v335;
        v335 = threadIdx.x;
        int v336;
        v336 = v335 / 32l;
        auto v337 = cooperative_groups::labeled_partition(v334,v336);
        Closure0 v338{};
        float v339;
        v339 = cooperative_groups::reduce(v337, v325, v338);
        float v340;
        v340 = v339 / 128.0f;
        float v341[4l];
        int v342;
        v342 = 0l;
        while (while_method_3(v342)){
            int v344;
            v344 = 0l;
            while (while_method_1(v344)){
                assert("Tensor range check" && 0 <= v342 && v342 < 1l);
                assert("Tensor range check" && 0 <= v344 && v344 < 4l);
                int v346;
                v346 = 4l * v342;
                int v347;
                v347 = v346 + v344;
                float v348;
                v348 = v283[v347];
                float v349;
                v349 = v348 - v340;
                float v350;
                v350 = exp(v349);
                assert("Tensor range check" && 0 <= v342 && v342 < 1l);
                assert("Tensor range check" && 0 <= v344 && v344 < 4l);
                v341[v347] = v350;
                v344 += 1l ;
            }
            v342 += 1l ;
        }
        float v351;
        v351 = 0.0f;
        int v352;
        v352 = 0l;
        while (while_method_3(v352)){
            int v354;
            v354 = 0l;
            while (while_method_1(v354)){
                assert("Tensor range check" && 0 <= v352 && v352 < 1l);
                assert("Tensor range check" && 0 <= v354 && v354 < 4l);
                int v356;
                v356 = 4l * v352;
                int v357;
                v357 = v356 + v354;
                float v358;
                v358 = v341[v357];
                float v359;
                v359 = v351 + v358;
                v351 = v359;
                v354 += 1l ;
            }
            v352 += 1l ;
        }
        auto v360 = cooperative_groups::coalesced_threads();
        int v361;
        v361 = threadIdx.x;
        int v362;
        v362 = v361 / 32l;
        auto v363 = cooperative_groups::labeled_partition(v360,v362);
        float v364;
        v364 = cooperative_groups::reduce(v363, v351, v338);
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
                v372 = v341[v371];
                float v373;
                v373 = v372 / v364;
                assert("Tensor range check" && 0 <= v366 && v366 < 1l);
                assert("Tensor range check" && 0 <= v368 && v368 < 4l);
                v365[v371] = v373;
                v368 += 1l ;
            }
            v366 += 1l ;
        }
        assert("Tensor range check" && 0 <= v273 && v273 < 64l);
        int v374;
        v374 = 0l;
        while (while_method_3(v374)){
            assert("Tensor range check" && 0 <= v374 && v374 < 1l);
            int v376;
            v376 = 128l * v374;
            int v377;
            v377 = v376 + v282;
            assert("Tensor range check" && 0 <= v374 && v374 < 1l);
            int v378;
            v378 = 4l * v374;
            int4* v379;
            v379 = reinterpret_cast<int4*>(v365 + v378);
            int4* v380;
            v380 = reinterpret_cast<int4*>(v3 + v377);
            assert("Pointer alignment check" && (unsigned long long)(v379) % 4l == 0 && (unsigned long long)(v380) % 4l == 0);
            *v380 = *v379;
            v374 += 1l ;
        }
        v273 += 24l ;
    }
    v15.sync() ;
    int v381;
    v381 = threadIdx.x;
    bool v382;
    v382 = 0l <= v381;
    bool v383;
    v383 = v382 == false;
    if (v383){
        assert("The index needs to be zero or positive." && v382);
    } else {
    }
    int v385;
    v385 = v381 % 32l;
    int v386;
    v386 = v381 / 32l;
    bool v387;
    v387 = v386 < 1l;
    bool v388;
    v388 = v387 == false;
    if (v388){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v387);
    } else {
    }
    assert("Tensor range check" && 0 <= v386 && v386 < 1l);
    assert("Tensor range check" && 0 <= v385 && v385 < 32l);
    int v390;
    v390 = 4l * v385;
    int v391;
    v391 = 128l * v386;
    int v392;
    v392 = v391 + v390;
    assert("Tensor range check" && 0 <= v386 && v386 < 1l);
    assert("Tensor range check" && 0 <= v385 && v385 < 32l);
    int v393;
    v393 = blockIdx.x;
    int v394;
    v394 = v393;
    while (while_method_2(v394)){
        bool v396;
        v396 = 0l <= v394;
        bool v397;
        v397 = v396 == false;
        if (v397){
            assert("The index needs to be zero or positive." && v396);
        } else {
        }
        bool v399;
        v399 = v394 < 64l;
        bool v400;
        v400 = v399 == false;
        if (v400){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v399);
        } else {
        }
        assert("Tensor range check" && 0 <= v394 && v394 < 64l);
        int v402;
        v402 = 128l * v394;
        int v403;
        v403 = v402 + v392;
        float v404[4l];
        int v405[4l];
        int v406;
        v406 = 0l;
        while (while_method_3(v406)){
            assert("Tensor range check" && 0 <= v406 && v406 < 1l);
            int v408;
            v408 = 4l * v406;
            assert("Tensor range check" && 0 <= v406 && v406 < 1l);
            int v409;
            v409 = 128l * v406;
            int v410;
            v410 = v409 + v403;
            int4* v411;
            v411 = reinterpret_cast<int4*>(v1 + v410);
            int4* v412;
            v412 = reinterpret_cast<int4*>(v404 + v408);
            assert("Pointer alignment check" && (unsigned long long)(v411) % 4l == 0 && (unsigned long long)(v412) % 4l == 0);
            *v412 = *v411;
            v406 += 1l ;
        }
        int v413;
        v413 = 0l;
        while (while_method_3(v413)){
            int v415;
            v415 = 0l;
            while (while_method_1(v415)){
                bool v417;
                v417 = 0l <= v415;
                bool v419;
                if (v417){
                    bool v418;
                    v418 = v415 < 4l;
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
                bool v422;
                v422 = 0l <= v385;
                bool v424;
                if (v422){
                    bool v423;
                    v423 = v385 < 32l;
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
                v427 = v385 * 4l;
                int v428;
                v428 = v415 + v427;
                bool v429;
                v429 = 0l <= v413;
                bool v431;
                if (v429){
                    bool v430;
                    v430 = v413 < 1l;
                    v431 = v430;
                } else {
                    v431 = false;
                }
                bool v432;
                v432 = v431 == false;
                if (v432){
                    assert("The indices should be inside the range of the dimension." && v431);
                } else {
                }
                int v434;
                v434 = v413 * 128l;
                int v435;
                v435 = v428 + v434;
                assert("Tensor range check" && 0 <= v413 && v413 < 1l);
                assert("Tensor range check" && 0 <= v415 && v415 < 4l);
                int v436;
                v436 = 4l * v413;
                int v437;
                v437 = v436 + v415;
                v405[v437] = v435;
                v415 += 1l ;
            }
            v413 += 1l ;
        }
        bool v438;
        v438 = 0l <= v386;
        bool v439;
        v439 = v438 && v387;
        bool v440;
        v440 = v439 == false;
        if (v440){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v439);
        } else {
        }
        bool v442;
        v442 = v396 && v399;
        bool v443;
        v443 = v442 == false;
        if (v443){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v442);
        } else {
        }
        int v445;
        v445 = v394 + v386;
        float v446[4l];
        int v447;
        v447 = 0l;
        while (while_method_3(v447)){
            int v449;
            v449 = 0l;
            while (while_method_1(v449)){
                assert("Tensor range check" && 0 <= v447 && v447 < 1l);
                assert("Tensor range check" && 0 <= v449 && v449 < 4l);
                int v451;
                v451 = 4l * v447;
                int v452;
                v452 = v451 + v449;
                float v453;
                v453 = v404[v452];
                float v454;
                v454 = v453 * v453;
                assert("Tensor range check" && 0 <= v447 && v447 < 1l);
                assert("Tensor range check" && 0 <= v449 && v449 < 4l);
                v446[v452] = v454;
                v449 += 1l ;
            }
            v447 += 1l ;
        }
        float v455;
        v455 = 0.0f;
        int v456;
        v456 = 0l;
        while (while_method_3(v456)){
            int v458;
            v458 = 0l;
            while (while_method_1(v458)){
                assert("Tensor range check" && 0 <= v456 && v456 < 1l);
                assert("Tensor range check" && 0 <= v458 && v458 < 4l);
                int v460;
                v460 = 4l * v456;
                int v461;
                v461 = v460 + v458;
                float v462;
                v462 = v446[v461];
                float v463;
                v463 = v455 + v462;
                v455 = v463;
                v458 += 1l ;
            }
            v456 += 1l ;
        }
        auto v464 = cooperative_groups::coalesced_threads();
        int v465;
        v465 = threadIdx.x;
        int v466;
        v466 = v465 / 32l;
        auto v467 = cooperative_groups::labeled_partition(v464,v466);
        Closure0 v468{};
        float v469;
        v469 = cooperative_groups::reduce(v467, v455, v468);
        float v470[4l];
        int v471;
        v471 = 0l;
        while (while_method_3(v471)){
            int v473;
            v473 = 0l;
            while (while_method_1(v473)){
                assert("Tensor range check" && 0 <= v471 && v471 < 1l);
                assert("Tensor range check" && 0 <= v473 && v473 < 4l);
                int v475;
                v475 = 4l * v471;
                int v476;
                v476 = v475 + v473;
                float v477;
                v477 = v404[v476];
                bool v478;
                v478 = v469 == 0.0f;
                bool v479;
                v479 = v478 != true;
                float v481;
                if (v479){
                    float v480;
                    v480 = v477 / v469;
                    v481 = v480;
                } else {
                    v481 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v471 && v471 < 1l);
                assert("Tensor range check" && 0 <= v473 && v473 < 4l);
                v470[v476] = v481;
                v473 += 1l ;
            }
            v471 += 1l ;
        }
        assert("Tensor range check" && 0 <= v394 && v394 < 64l);
        int v482;
        v482 = 0l;
        while (while_method_3(v482)){
            assert("Tensor range check" && 0 <= v482 && v482 < 1l);
            int v484;
            v484 = 128l * v482;
            int v485;
            v485 = v484 + v403;
            assert("Tensor range check" && 0 <= v482 && v482 < 1l);
            int v486;
            v486 = 4l * v482;
            int4* v487;
            v487 = reinterpret_cast<int4*>(v470 + v486);
            int4* v488;
            v488 = reinterpret_cast<int4*>(v7 + v485);
            assert("Pointer alignment check" && (unsigned long long)(v487) % 4l == 0 && (unsigned long long)(v488) % 4l == 0);
            *v488 = *v487;
            v482 += 1l ;
        }
        v394 += 24l ;
    }
    v15.sync() ;
    int v489;
    v489 = threadIdx.x;
    bool v490;
    v490 = 0l <= v489;
    bool v491;
    v491 = v490 == false;
    if (v491){
        assert("The index needs to be zero or positive." && v490);
    } else {
    }
    int v493;
    v493 = v489 % 32l;
    int v494;
    v494 = v489 / 32l;
    bool v495;
    v495 = v494 < 1l;
    bool v496;
    v496 = v495 == false;
    if (v496){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v495);
    } else {
    }
    assert("Tensor range check" && 0 <= v494 && v494 < 1l);
    assert("Tensor range check" && 0 <= v493 && v493 < 32l);
    int v498;
    v498 = 4l * v493;
    int v499;
    v499 = 128l * v494;
    int v500;
    v500 = v499 + v498;
    assert("Tensor range check" && 0 <= v494 && v494 < 1l);
    int v501;
    v501 = blockIdx.x;
    int v502;
    v502 = v501;
    while (while_method_2(v502)){
        bool v504;
        v504 = 0l <= v502;
        bool v505;
        v505 = v504 == false;
        if (v505){
            assert("The index needs to be zero or positive." && v504);
        } else {
        }
        bool v507;
        v507 = v502 < 64l;
        bool v508;
        v508 = v507 == false;
        if (v508){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v507);
        } else {
        }
        assert("Tensor range check" && 0 <= v502 && v502 < 64l);
        int v510;
        v510 = 128l * v502;
        int v511;
        v511 = v510 + v500;
        float v512[4l];
        int v513[4l];
        int v514;
        v514 = 0l;
        while (while_method_3(v514)){
            assert("Tensor range check" && 0 <= v514 && v514 < 1l);
            int v516;
            v516 = 4l * v514;
            assert("Tensor range check" && 0 <= v514 && v514 < 1l);
            int v517;
            v517 = 128l * v514;
            int v518;
            v518 = v517 + v511;
            int4* v519;
            v519 = reinterpret_cast<int4*>(v1 + v518);
            int4* v520;
            v520 = reinterpret_cast<int4*>(v512 + v516);
            assert("Pointer alignment check" && (unsigned long long)(v519) % 4l == 0 && (unsigned long long)(v520) % 4l == 0);
            *v520 = *v519;
            v514 += 1l ;
        }
        int v521;
        v521 = 0l;
        while (while_method_3(v521)){
            int v523;
            v523 = 0l;
            while (while_method_1(v523)){
                bool v525;
                v525 = 0l <= v523;
                bool v527;
                if (v525){
                    bool v526;
                    v526 = v523 < 4l;
                    v527 = v526;
                } else {
                    v527 = false;
                }
                bool v528;
                v528 = v527 == false;
                if (v528){
                    assert("The indices should be inside the range of the dimension." && v527);
                } else {
                }
                bool v530;
                v530 = 0l <= v493;
                bool v532;
                if (v530){
                    bool v531;
                    v531 = v493 < 32l;
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
                int v535;
                v535 = v493 * 4l;
                int v536;
                v536 = v523 + v535;
                bool v537;
                v537 = 0l <= v521;
                bool v539;
                if (v537){
                    bool v538;
                    v538 = v521 < 1l;
                    v539 = v538;
                } else {
                    v539 = false;
                }
                bool v540;
                v540 = v539 == false;
                if (v540){
                    assert("The indices should be inside the range of the dimension." && v539);
                } else {
                }
                int v542;
                v542 = v521 * 128l;
                int v543;
                v543 = v536 + v542;
                assert("Tensor range check" && 0 <= v521 && v521 < 1l);
                assert("Tensor range check" && 0 <= v523 && v523 < 4l);
                int v544;
                v544 = 4l * v521;
                int v545;
                v545 = v544 + v523;
                v513[v545] = v543;
                v523 += 1l ;
            }
            v521 += 1l ;
        }
        bool v546;
        v546 = 0l <= v494;
        bool v547;
        v547 = v546 && v495;
        bool v548;
        v548 = v547 == false;
        if (v548){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v547);
        } else {
        }
        bool v550;
        v550 = v504 && v507;
        bool v551;
        v551 = v550 == false;
        if (v551){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v550);
        } else {
        }
        int v553;
        v553 = v502 + v494;
        float v554; int v555;
        Tuple1 tmp84 = Tuple1{-1.0f / 0.0f, 0l};
        v554 = tmp84.v0; v555 = tmp84.v1;
        int v556;
        v556 = 0l;
        while (while_method_3(v556)){
            int v558;
            v558 = 0l;
            while (while_method_1(v558)){
                assert("Tensor range check" && 0 <= v556 && v556 < 1l);
                assert("Tensor range check" && 0 <= v558 && v558 < 4l);
                int v560;
                v560 = 4l * v556;
                int v561;
                v561 = v560 + v558;
                float v562;
                v562 = v512[v561];
                int v563;
                v563 = v513[v561];
                bool v564;
                v564 = v554 > v562;
                float v565; int v566;
                if (v564){
                    v565 = v554; v566 = v555;
                } else {
                    v565 = v562; v566 = v563;
                }
                v554 = v565;
                v555 = v566;
                v558 += 1l ;
            }
            v556 += 1l ;
        }
        auto v567 = cooperative_groups::coalesced_threads();
        int v568;
        v568 = threadIdx.x;
        int v569;
        v569 = v568 / 32l;
        auto v570 = cooperative_groups::labeled_partition(v567,v569);
        Closure1 v571{};
        float v572; int v573;
        Tuple1 tmp85 = cooperative_groups::reduce(v570, Tuple1{v554, v555}, v571);
        v572 = tmp85.v0; v573 = tmp85.v1;
        assert("Tensor range check" && 0 <= v502 && v502 < 64l);
        v8[v553] = v573;
        v502 += 24l ;
    }
    v15.sync() ;
    int v574;
    v574 = threadIdx.x;
    bool v575;
    v575 = 0l <= v574;
    bool v576;
    v576 = v575 == false;
    if (v576){
        assert("The index needs to be zero or positive." && v575);
    } else {
    }
    int v578;
    v578 = v574 % 32l;
    int v579;
    v579 = v574 / 32l;
    bool v580;
    v580 = v579 < 1l;
    bool v581;
    v581 = v580 == false;
    if (v581){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v580);
    } else {
    }
    assert("Tensor range check" && 0 <= v579 && v579 < 1l);
    assert("Tensor range check" && 0 <= v578 && v578 < 32l);
    int v583;
    v583 = 4l * v578;
    int v584;
    v584 = 128l * v579;
    int v585;
    v585 = v584 + v583;
    assert("Tensor range check" && 0 <= v579 && v579 < 1l);
    assert("Tensor range check" && 0 <= v578 && v578 < 32l);
    int v586;
    v586 = blockIdx.x;
    int v587;
    v587 = v586;
    while (while_method_2(v587)){
        bool v589;
        v589 = 0l <= v587;
        bool v590;
        v590 = v589 == false;
        if (v590){
            assert("The index needs to be zero or positive." && v589);
        } else {
        }
        bool v592;
        v592 = v587 < 64l;
        bool v593;
        v593 = v592 == false;
        if (v593){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v592);
        } else {
        }
        assert("Tensor range check" && 0 <= v587 && v587 < 64l);
        int v595;
        v595 = 128l * v587;
        int v596;
        v596 = v595 + v585;
        float v597[4l];
        int v598[4l];
        int v599;
        v599 = 0l;
        while (while_method_3(v599)){
            assert("Tensor range check" && 0 <= v599 && v599 < 1l);
            int v601;
            v601 = 4l * v599;
            assert("Tensor range check" && 0 <= v599 && v599 < 1l);
            int v602;
            v602 = 128l * v599;
            int v603;
            v603 = v602 + v596;
            int4* v604;
            v604 = reinterpret_cast<int4*>(v1 + v603);
            int4* v605;
            v605 = reinterpret_cast<int4*>(v597 + v601);
            assert("Pointer alignment check" && (unsigned long long)(v604) % 4l == 0 && (unsigned long long)(v605) % 4l == 0);
            *v605 = *v604;
            v599 += 1l ;
        }
        int v606;
        v606 = 0l;
        while (while_method_3(v606)){
            int v608;
            v608 = 0l;
            while (while_method_1(v608)){
                bool v610;
                v610 = 0l <= v608;
                bool v612;
                if (v610){
                    bool v611;
                    v611 = v608 < 4l;
                    v612 = v611;
                } else {
                    v612 = false;
                }
                bool v613;
                v613 = v612 == false;
                if (v613){
                    assert("The indices should be inside the range of the dimension." && v612);
                } else {
                }
                bool v615;
                v615 = 0l <= v578;
                bool v617;
                if (v615){
                    bool v616;
                    v616 = v578 < 32l;
                    v617 = v616;
                } else {
                    v617 = false;
                }
                bool v618;
                v618 = v617 == false;
                if (v618){
                    assert("The indices should be inside the range of the dimension." && v617);
                } else {
                }
                int v620;
                v620 = v578 * 4l;
                int v621;
                v621 = v608 + v620;
                bool v622;
                v622 = 0l <= v606;
                bool v624;
                if (v622){
                    bool v623;
                    v623 = v606 < 1l;
                    v624 = v623;
                } else {
                    v624 = false;
                }
                bool v625;
                v625 = v624 == false;
                if (v625){
                    assert("The indices should be inside the range of the dimension." && v624);
                } else {
                }
                int v627;
                v627 = v606 * 128l;
                int v628;
                v628 = v621 + v627;
                assert("Tensor range check" && 0 <= v606 && v606 < 1l);
                assert("Tensor range check" && 0 <= v608 && v608 < 4l);
                int v629;
                v629 = 4l * v606;
                int v630;
                v630 = v629 + v608;
                v598[v630] = v628;
                v608 += 1l ;
            }
            v606 += 1l ;
        }
        bool v631;
        v631 = 0l <= v579;
        bool v632;
        v632 = v631 && v580;
        bool v633;
        v633 = v632 == false;
        if (v633){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v632);
        } else {
        }
        bool v635;
        v635 = v589 && v592;
        bool v636;
        v636 = v635 == false;
        if (v636){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v635);
        } else {
        }
        int v638;
        v638 = v587 + v579;
        float v639;
        v639 = 0.0f;
        int v640;
        v640 = 0l;
        while (while_method_3(v640)){
            int v642;
            v642 = 0l;
            while (while_method_1(v642)){
                assert("Tensor range check" && 0 <= v640 && v640 < 1l);
                assert("Tensor range check" && 0 <= v642 && v642 < 4l);
                int v644;
                v644 = 4l * v640;
                int v645;
                v645 = v644 + v642;
                float v646;
                v646 = v597[v645];
                float v647;
                v647 = v639 + v646;
                v639 = v647;
                v642 += 1l ;
            }
            v640 += 1l ;
        }
        auto v648 = cooperative_groups::coalesced_threads();
        int v649;
        v649 = threadIdx.x;
        int v650;
        v650 = v649 / 32l;
        auto v651 = cooperative_groups::labeled_partition(v648,v650);
        Closure0 v652{};
        float v653;
        v653 = cooperative_groups::reduce(v651, v639, v652);
        float v654;
        v654 = v653 / 128.0f;
        float v655[4l];
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
                v662 = v597[v661];
                float v663;
                v663 = v662 - v654;
                float v664;
                v664 = exp(v663);
                assert("Tensor range check" && 0 <= v656 && v656 < 1l);
                assert("Tensor range check" && 0 <= v658 && v658 < 4l);
                v655[v661] = v664;
                v658 += 1l ;
            }
            v656 += 1l ;
        }
        float v665;
        v665 = 0.0f;
        int v666;
        v666 = 0l;
        while (while_method_3(v666)){
            int v668;
            v668 = 0l;
            while (while_method_1(v668)){
                assert("Tensor range check" && 0 <= v666 && v666 < 1l);
                assert("Tensor range check" && 0 <= v668 && v668 < 4l);
                int v670;
                v670 = 4l * v666;
                int v671;
                v671 = v670 + v668;
                float v672;
                v672 = v655[v671];
                float v673;
                v673 = v665 + v672;
                v665 = v673;
                v668 += 1l ;
            }
            v666 += 1l ;
        }
        auto v674 = cooperative_groups::coalesced_threads();
        int v675;
        v675 = threadIdx.x;
        int v676;
        v676 = v675 / 32l;
        auto v677 = cooperative_groups::labeled_partition(v674,v676);
        float v678;
        v678 = cooperative_groups::reduce(v677, v665, v652);
        float v679[4l];
        int v680;
        v680 = 0l;
        while (while_method_3(v680)){
            int v682;
            v682 = 0l;
            while (while_method_1(v682)){
                assert("Tensor range check" && 0 <= v680 && v680 < 1l);
                assert("Tensor range check" && 0 <= v682 && v682 < 4l);
                int v684;
                v684 = 4l * v680;
                int v685;
                v685 = v684 + v682;
                float v686;
                v686 = v655[v685];
                float v687;
                v687 = v686 / v678;
                assert("Tensor range check" && 0 <= v680 && v680 < 1l);
                assert("Tensor range check" && 0 <= v682 && v682 < 4l);
                v679[v685] = v687;
                v682 += 1l ;
            }
            v680 += 1l ;
        }
        float v688[4l];
        float v689;
        v689 = 0.0f;
        int v690;
        v690 = 0l;
        while (while_method_3(v690)){
            assert("Tensor range check" && 0 <= v690 && v690 < 1l);
            int v692;
            v692 = 4l * v690;
            assert("Tensor range check" && 0 <= v690 && v690 < 1l);
            int v693; float v694;
            Tuple0 tmp86 = Tuple0{0l, 0.0f};
            v693 = tmp86.v0; v694 = tmp86.v1;
            while (while_method_1(v693)){
                assert("Tensor range check" && 0 <= v693 && v693 < 4l);
                int v696;
                v696 = v693 + v692;
                float v697;
                v697 = v679[v696];
                float v698;
                v698 = v694 + v697;
                v694 = v698;
                v693 += 1l ;
            }
            auto v699 = cooperative_groups::coalesced_threads();
            int v700;
            v700 = threadIdx.x;
            int v701;
            v701 = v700 / 32l;
            auto v702 = cooperative_groups::labeled_partition(v699,v701);
            Closure2 v703{};
            float v704;
            v704 = cooperative_groups::inclusive_scan(v702, v694, v703);
            float v705;
            v705 = v702.shfl_up(v704,1);
            bool v706;
            v706 = v702.thread_rank() == 0;
            float v707;
            if (v706){
                v707 = 0.0f;
            } else {
                v707 = v705;
            }
            float v708;
            v708 = v702.shfl(v704,v702.num_threads()-1);
            float v709;
            v709 = v689 + v707;
            int v710; float v711;
            Tuple0 tmp87 = Tuple0{0l, v709};
            v710 = tmp87.v0; v711 = tmp87.v1;
            while (while_method_1(v710)){
                assert("Tensor range check" && 0 <= v710 && v710 < 4l);
                int v713;
                v713 = v710 + v692;
                float v714;
                v714 = v679[v713];
                float v715;
                v715 = v711 + v714;
                assert("Tensor range check" && 0 <= v710 && v710 < 4l);
                v688[v713] = v715;
                v711 = v715;
                v710 += 1l ;
            }
            float v716;
            v716 = v689 + v708;
            v689 = v716;
            v690 += 1l ;
        }
        assert("Tensor range check" && 0 <= v587 && v587 < 64l);
        int v717;
        v717 = 0l;
        while (while_method_3(v717)){
            assert("Tensor range check" && 0 <= v717 && v717 < 1l);
            int v719;
            v719 = 128l * v717;
            int v720;
            v720 = v719 + v596;
            assert("Tensor range check" && 0 <= v717 && v717 < 1l);
            int v721;
            v721 = 4l * v717;
            int4* v722;
            v722 = reinterpret_cast<int4*>(v679 + v721);
            int4* v723;
            v723 = reinterpret_cast<int4*>(v5 + v720);
            assert("Pointer alignment check" && (unsigned long long)(v722) % 4l == 0 && (unsigned long long)(v723) % 4l == 0);
            *v723 = *v722;
            int4* v724;
            v724 = reinterpret_cast<int4*>(v688 + v721);
            int4* v725;
            v725 = reinterpret_cast<int4*>(v6 + v720);
            assert("Pointer alignment check" && (unsigned long long)(v724) % 4l == 0 && (unsigned long long)(v725) % 4l == 0);
            *v725 = *v724;
            v717 += 1l ;
        }
        v587 += 24l ;
    }
    v15.sync() ;
    int v726;
    v726 = threadIdx.x;
    bool v727;
    v727 = 0l <= v726;
    bool v728;
    v728 = v727 == false;
    if (v728){
        assert("The index needs to be zero or positive." && v727);
    } else {
    }
    int v730;
    v730 = v726 % 32l;
    int v731;
    v731 = v726 / 32l;
    bool v732;
    v732 = v731 < 1l;
    bool v733;
    v733 = v732 == false;
    if (v733){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v732);
    } else {
    }
    assert("Tensor range check" && 0 <= v731 && v731 < 1l);
    assert("Tensor range check" && 0 <= v730 && v730 < 32l);
    int v735;
    v735 = 4l * v730;
    int v736;
    v736 = 128l * v731;
    int v737;
    v737 = v736 + v735;
    assert("Tensor range check" && 0 <= v731 && v731 < 1l);
    assert("Tensor range check" && 0 <= v730 && v730 < 32l);
    int v738;
    v738 = blockIdx.x;
    int v739;
    v739 = v738;
    while (while_method_2(v739)){
        bool v741;
        v741 = 0l <= v739;
        bool v742;
        v742 = v741 == false;
        if (v742){
            assert("The index needs to be zero or positive." && v741);
        } else {
        }
        bool v744;
        v744 = v739 < 64l;
        bool v745;
        v745 = v744 == false;
        if (v745){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v744);
        } else {
        }
        assert("Tensor range check" && 0 <= v739 && v739 < 64l);
        int v747;
        v747 = 128l * v739;
        int v748;
        v748 = v747 + v737;
        int v749[4l];
        int v750[4l];
        int v751;
        v751 = 0l;
        while (while_method_3(v751)){
            assert("Tensor range check" && 0 <= v751 && v751 < 1l);
            int v753;
            v753 = 4l * v751;
            assert("Tensor range check" && 0 <= v751 && v751 < 1l);
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
                v767 = 0l <= v730;
                bool v769;
                if (v767){
                    bool v768;
                    v768 = v730 < 32l;
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
                v772 = v730 * 4l;
                int v773;
                v773 = v760 + v772;
                bool v774;
                v774 = 0l <= v758;
                bool v776;
                if (v774){
                    bool v775;
                    v775 = v758 < 1l;
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
                assert("Tensor range check" && 0 <= v758 && v758 < 1l);
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
        v783 = 0l <= v731;
        bool v784;
        v784 = v783 && v732;
        bool v785;
        v785 = v784 == false;
        if (v785){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v784);
        } else {
        }
        bool v787;
        v787 = v741 && v744;
        bool v788;
        v788 = v787 == false;
        if (v788){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v787);
        } else {
        }
        int v790;
        v790 = v739 + v731;
        int v791[4l];
        int v792;
        v792 = 0l;
        int v793;
        v793 = 0l;
        while (while_method_3(v793)){
            assert("Tensor range check" && 0 <= v793 && v793 < 1l);
            int v795;
            v795 = 4l * v793;
            assert("Tensor range check" && 0 <= v793 && v793 < 1l);
            int v796; int v797;
            Tuple2 tmp88 = Tuple2{0l, 0l};
            v796 = tmp88.v0; v797 = tmp88.v1;
            while (while_method_1(v796)){
                assert("Tensor range check" && 0 <= v796 && v796 < 4l);
                int v799;
                v799 = v796 + v795;
                int v800;
                v800 = v749[v799];
                int v801;
                v801 = v797 + v800;
                v797 = v801;
                v796 += 1l ;
            }
            auto v802 = cooperative_groups::coalesced_threads();
            int v803;
            v803 = threadIdx.x;
            int v804;
            v804 = v803 / 32l;
            auto v805 = cooperative_groups::labeled_partition(v802,v804);
            Closure3 v806{};
            int v807;
            v807 = cooperative_groups::inclusive_scan(v805, v797, v806);
            int v808;
            v808 = v805.shfl_up(v807,1);
            bool v809;
            v809 = v805.thread_rank() == 0;
            int v810;
            if (v809){
                v810 = 0l;
            } else {
                v810 = v808;
            }
            int v811;
            v811 = v805.shfl(v807,v805.num_threads()-1);
            int v812;
            v812 = v792 + v810;
            int v813; int v814;
            Tuple2 tmp89 = Tuple2{0l, v812};
            v813 = tmp89.v0; v814 = tmp89.v1;
            while (while_method_1(v813)){
                assert("Tensor range check" && 0 <= v813 && v813 < 4l);
                int v816;
                v816 = v813 + v795;
                int v817;
                v817 = v749[v816];
                assert("Tensor range check" && 0 <= v813 && v813 < 4l);
                v791[v816] = v814;
                int v818;
                v818 = v814 + v817;
                v814 = v818;
                v813 += 1l ;
            }
            int v819;
            v819 = v792 + v811;
            v792 = v819;
            v793 += 1l ;
        }
        assert("Tensor range check" && 0 <= v739 && v739 < 64l);
        int v820;
        v820 = 0l;
        while (while_method_3(v820)){
            assert("Tensor range check" && 0 <= v820 && v820 < 1l);
            int v822;
            v822 = 128l * v820;
            int v823;
            v823 = v822 + v748;
            assert("Tensor range check" && 0 <= v820 && v820 < 1l);
            int v824;
            v824 = 4l * v820;
            int4* v825;
            v825 = reinterpret_cast<int4*>(v791 + v824);
            int4* v826;
            v826 = reinterpret_cast<int4*>(v12 + v823);
            assert("Pointer alignment check" && (unsigned long long)(v825) % 4l == 0 && (unsigned long long)(v826) % 4l == 0);
            *v826 = *v825;
            v820 += 1l ;
        }
        v739 += 24l ;
    }
    v15.sync() ;
    int v827;
    v827 = threadIdx.x;
    bool v828;
    v828 = 0l <= v827;
    bool v829;
    v829 = v828 == false;
    if (v829){
        assert("The index needs to be zero or positive." && v828);
    } else {
    }
    int v831;
    v831 = v827 % 32l;
    int v832;
    v832 = v827 / 32l;
    bool v833;
    v833 = v832 < 1l;
    bool v834;
    v834 = v833 == false;
    if (v834){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v833);
    } else {
    }
    assert("Tensor range check" && 0 <= v832 && v832 < 1l);
    assert("Tensor range check" && 0 <= v831 && v831 < 32l);
    int v836;
    v836 = 4l * v831;
    int v837;
    v837 = 128l * v832;
    int v838;
    v838 = v837 + v836;
    assert("Tensor range check" && 0 <= v832 && v832 < 1l);
    assert("Tensor range check" && 0 <= v831 && v831 < 32l);
    int v839;
    v839 = blockIdx.x;
    int v840;
    v840 = v839;
    while (while_method_2(v840)){
        bool v842;
        v842 = 0l <= v840;
        bool v843;
        v843 = v842 == false;
        if (v843){
            assert("The index needs to be zero or positive." && v842);
        } else {
        }
        bool v845;
        v845 = v840 < 64l;
        bool v846;
        v846 = v845 == false;
        if (v846){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v845);
        } else {
        }
        assert("Tensor range check" && 0 <= v840 && v840 < 64l);
        int v848;
        v848 = 128l * v840;
        int v849;
        v849 = v848 + v838;
        float v850[4l];
        int v851[4l];
        int v852;
        v852 = 0l;
        while (while_method_3(v852)){
            assert("Tensor range check" && 0 <= v852 && v852 < 1l);
            int v854;
            v854 = 4l * v852;
            assert("Tensor range check" && 0 <= v852 && v852 < 1l);
            int v855;
            v855 = 128l * v852;
            int v856;
            v856 = v855 + v849;
            int4* v857;
            v857 = reinterpret_cast<int4*>(v1 + v856);
            int4* v858;
            v858 = reinterpret_cast<int4*>(v850 + v854);
            assert("Pointer alignment check" && (unsigned long long)(v857) % 4l == 0 && (unsigned long long)(v858) % 4l == 0);
            *v858 = *v857;
            v852 += 1l ;
        }
        int v859;
        v859 = 0l;
        while (while_method_3(v859)){
            int v861;
            v861 = 0l;
            while (while_method_1(v861)){
                bool v863;
                v863 = 0l <= v861;
                bool v865;
                if (v863){
                    bool v864;
                    v864 = v861 < 4l;
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
                bool v868;
                v868 = 0l <= v831;
                bool v870;
                if (v868){
                    bool v869;
                    v869 = v831 < 32l;
                    v870 = v869;
                } else {
                    v870 = false;
                }
                bool v871;
                v871 = v870 == false;
                if (v871){
                    assert("The indices should be inside the range of the dimension." && v870);
                } else {
                }
                int v873;
                v873 = v831 * 4l;
                int v874;
                v874 = v861 + v873;
                bool v875;
                v875 = 0l <= v859;
                bool v877;
                if (v875){
                    bool v876;
                    v876 = v859 < 1l;
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
                v880 = v859 * 128l;
                int v881;
                v881 = v874 + v880;
                assert("Tensor range check" && 0 <= v859 && v859 < 1l);
                assert("Tensor range check" && 0 <= v861 && v861 < 4l);
                int v882;
                v882 = 4l * v859;
                int v883;
                v883 = v882 + v861;
                v851[v883] = v881;
                v861 += 1l ;
            }
            v859 += 1l ;
        }
        bool v884;
        v884 = 0l <= v832;
        bool v885;
        v885 = v884 && v833;
        bool v886;
        v886 = v885 == false;
        if (v886){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v885);
        } else {
        }
        bool v888;
        v888 = v842 && v845;
        bool v889;
        v889 = v888 == false;
        if (v889){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v888);
        } else {
        }
        int v891;
        v891 = v840 + v832;
        bool v892[4l];
        int v893;
        v893 = 0l;
        while (while_method_3(v893)){
            int v895;
            v895 = 0l;
            while (while_method_1(v895)){
                assert("Tensor range check" && 0 <= v893 && v893 < 1l);
                assert("Tensor range check" && 0 <= v895 && v895 < 4l);
                int v897;
                v897 = 4l * v893;
                int v898;
                v898 = v897 + v895;
                float v899;
                v899 = v850[v898];
                int v900;
                v900 = v851[v898];
                bool v901;
                v901 = v900 < 4l;
                assert("Tensor range check" && 0 <= v893 && v893 < 1l);
                assert("Tensor range check" && 0 <= v895 && v895 < 4l);
                v892[v898] = v901;
                v895 += 1l ;
            }
            v893 += 1l ;
        }
        int v902[4l];
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
                bool v909;
                v909 = v892[v908];
                int v910;
                if (v909){
                    v910 = 1l;
                } else {
                    v910 = 0l;
                }
                assert("Tensor range check" && 0 <= v903 && v903 < 1l);
                assert("Tensor range check" && 0 <= v905 && v905 < 4l);
                v902[v908] = v910;
                v905 += 1l ;
            }
            v903 += 1l ;
        }
        int v911;
        v911 = 0l;
        int v912;
        v912 = 0l;
        while (while_method_3(v912)){
            int v914;
            v914 = 0l;
            while (while_method_1(v914)){
                assert("Tensor range check" && 0 <= v912 && v912 < 1l);
                assert("Tensor range check" && 0 <= v914 && v914 < 4l);
                int v916;
                v916 = 4l * v912;
                int v917;
                v917 = v916 + v914;
                int v918;
                v918 = v902[v917];
                int v919;
                v919 = v911 + v918;
                v911 = v919;
                v914 += 1l ;
            }
            v912 += 1l ;
        }
        auto v920 = cooperative_groups::coalesced_threads();
        int v921;
        v921 = threadIdx.x;
        int v922;
        v922 = v921 / 32l;
        auto v923 = cooperative_groups::labeled_partition(v920,v922);
        Closure4 v924{};
        int v925;
        v925 = cooperative_groups::reduce(v923, v911, v924);
        float v926[4l];
        int v927;
        v927 = 0l;
        while (while_method_3(v927)){
            int v929;
            v929 = 0l;
            while (while_method_1(v929)){
                assert("Tensor range check" && 0 <= v927 && v927 < 1l);
                assert("Tensor range check" && 0 <= v929 && v929 < 4l);
                int v931;
                v931 = 4l * v927;
                int v932;
                v932 = v931 + v929;
                float v933;
                v933 = v850[v932];
                bool v934;
                v934 = v892[v932];
                float v935;
                if (v934){
                    v935 = v933;
                } else {
                    v935 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v927 && v927 < 1l);
                assert("Tensor range check" && 0 <= v929 && v929 < 4l);
                v926[v932] = v935;
                v929 += 1l ;
            }
            v927 += 1l ;
        }
        float v936;
        v936 = 0.0f;
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
                v943 = v926[v942];
                float v944;
                v944 = v936 + v943;
                v936 = v944;
                v939 += 1l ;
            }
            v937 += 1l ;
        }
        auto v945 = cooperative_groups::coalesced_threads();
        int v946;
        v946 = threadIdx.x;
        int v947;
        v947 = v946 / 32l;
        auto v948 = cooperative_groups::labeled_partition(v945,v947);
        Closure0 v949{};
        float v950;
        v950 = cooperative_groups::reduce(v948, v936, v949);
        float v951;
        v951 = (float)v925;
        float v952;
        v952 = v950 / v951;
        float v953[4l];
        int v954;
        v954 = 0l;
        while (while_method_3(v954)){
            int v956;
            v956 = 0l;
            while (while_method_1(v956)){
                assert("Tensor range check" && 0 <= v954 && v954 < 1l);
                assert("Tensor range check" && 0 <= v956 && v956 < 4l);
                int v958;
                v958 = 4l * v954;
                int v959;
                v959 = v958 + v956;
                float v960;
                v960 = v850[v959];
                bool v961;
                v961 = v892[v959];
                float v962;
                if (v961){
                    v962 = v960;
                } else {
                    v962 = -1.0f / 0.0f;
                }
                float v963;
                v963 = v962 - v952;
                float v964;
                v964 = exp(v963);
                assert("Tensor range check" && 0 <= v954 && v954 < 1l);
                assert("Tensor range check" && 0 <= v956 && v956 < 4l);
                v953[v959] = v964;
                v956 += 1l ;
            }
            v954 += 1l ;
        }
        float v965;
        v965 = 0.0f;
        int v966;
        v966 = 0l;
        while (while_method_3(v966)){
            int v968;
            v968 = 0l;
            while (while_method_1(v968)){
                assert("Tensor range check" && 0 <= v966 && v966 < 1l);
                assert("Tensor range check" && 0 <= v968 && v968 < 4l);
                int v970;
                v970 = 4l * v966;
                int v971;
                v971 = v970 + v968;
                float v972;
                v972 = v953[v971];
                float v973;
                v973 = v965 + v972;
                v965 = v973;
                v968 += 1l ;
            }
            v966 += 1l ;
        }
        auto v974 = cooperative_groups::coalesced_threads();
        int v975;
        v975 = threadIdx.x;
        int v976;
        v976 = v975 / 32l;
        auto v977 = cooperative_groups::labeled_partition(v974,v976);
        float v978;
        v978 = cooperative_groups::reduce(v977, v965, v949);
        float v979[4l];
        int v980;
        v980 = 0l;
        while (while_method_3(v980)){
            int v982;
            v982 = 0l;
            while (while_method_1(v982)){
                assert("Tensor range check" && 0 <= v980 && v980 < 1l);
                assert("Tensor range check" && 0 <= v982 && v982 < 4l);
                int v984;
                v984 = 4l * v980;
                int v985;
                v985 = v984 + v982;
                float v986;
                v986 = v953[v985];
                float v987;
                v987 = v986 / v978;
                assert("Tensor range check" && 0 <= v980 && v980 < 1l);
                assert("Tensor range check" && 0 <= v982 && v982 < 4l);
                v979[v985] = v987;
                v982 += 1l ;
            }
            v980 += 1l ;
        }
        assert("Tensor range check" && 0 <= v840 && v840 < 64l);
        int v988;
        v988 = 0l;
        while (while_method_3(v988)){
            assert("Tensor range check" && 0 <= v988 && v988 < 1l);
            int v990;
            v990 = 128l * v988;
            int v991;
            v991 = v990 + v849;
            assert("Tensor range check" && 0 <= v988 && v988 < 1l);
            int v992;
            v992 = 4l * v988;
            int4* v993;
            v993 = reinterpret_cast<int4*>(v979 + v992);
            int4* v994;
            v994 = reinterpret_cast<int4*>(v4 + v991);
            assert("Pointer alignment check" && (unsigned long long)(v993) % 4l == 0 && (unsigned long long)(v994) % 4l == 0);
            *v994 = *v993;
            v988 += 1l ;
        }
        v840 += 24l ;
    }
    v15.sync() ;
    int v995;
    v995 = threadIdx.x;
    unsigned long long v996;
    v996 = (unsigned long long)v995;
    curandStatePhilox4_32_10_t v997;
    curand_init(12344321ull,v996,0ull,&v997);
    int v998;
    v998 = threadIdx.x;
    bool v999;
    v999 = 0l <= v998;
    bool v1000;
    v1000 = v999 == false;
    if (v1000){
        assert("The index needs to be zero or positive." && v999);
    } else {
    }
    int v1002;
    v1002 = v998 % 32l;
    int v1003;
    v1003 = v998 / 32l;
    bool v1004;
    v1004 = v1003 < 1l;
    bool v1005;
    v1005 = v1004 == false;
    if (v1005){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1004);
    } else {
    }
    assert("Tensor range check" && 0 <= v1003 && v1003 < 1l);
    assert("Tensor range check" && 0 <= v1002 && v1002 < 32l);
    int v1007;
    v1007 = 4l * v1002;
    int v1008;
    v1008 = 128l * v1003;
    int v1009;
    v1009 = v1008 + v1007;
    assert("Tensor range check" && 0 <= v1003 && v1003 < 1l);
    assert("Tensor range check" && 0 <= v1002 && v1002 < 32l);
    assert("Tensor range check" && 0 <= v1003 && v1003 < 1l);
    int v1010;
    v1010 = blockIdx.x;
    int v1011;
    v1011 = v1010;
    while (while_method_2(v1011)){
        bool v1013;
        v1013 = 0l <= v1011;
        bool v1014;
        v1014 = v1013 == false;
        if (v1014){
            assert("The index needs to be zero or positive." && v1013);
        } else {
        }
        bool v1016;
        v1016 = v1011 < 64l;
        bool v1017;
        v1017 = v1016 == false;
        if (v1017){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1016);
        } else {
        }
        assert("Tensor range check" && 0 <= v1011 && v1011 < 64l);
        int v1019;
        v1019 = 128l * v1011;
        int v1020;
        v1020 = v1019 + v1009;
        float v1021[4l];
        int v1022[4l];
        int v1023;
        v1023 = 0l;
        while (while_method_3(v1023)){
            assert("Tensor range check" && 0 <= v1023 && v1023 < 1l);
            int v1025;
            v1025 = 4l * v1023;
            assert("Tensor range check" && 0 <= v1023 && v1023 < 1l);
            int v1026;
            v1026 = 128l * v1023;
            int v1027;
            v1027 = v1026 + v1020;
            int4* v1028;
            v1028 = reinterpret_cast<int4*>(v1 + v1027);
            int4* v1029;
            v1029 = reinterpret_cast<int4*>(v1021 + v1025);
            assert("Pointer alignment check" && (unsigned long long)(v1028) % 4l == 0 && (unsigned long long)(v1029) % 4l == 0);
            *v1029 = *v1028;
            v1023 += 1l ;
        }
        int v1030;
        v1030 = 0l;
        while (while_method_3(v1030)){
            int v1032;
            v1032 = 0l;
            while (while_method_1(v1032)){
                bool v1034;
                v1034 = 0l <= v1032;
                bool v1036;
                if (v1034){
                    bool v1035;
                    v1035 = v1032 < 4l;
                    v1036 = v1035;
                } else {
                    v1036 = false;
                }
                bool v1037;
                v1037 = v1036 == false;
                if (v1037){
                    assert("The indices should be inside the range of the dimension." && v1036);
                } else {
                }
                bool v1039;
                v1039 = 0l <= v1002;
                bool v1041;
                if (v1039){
                    bool v1040;
                    v1040 = v1002 < 32l;
                    v1041 = v1040;
                } else {
                    v1041 = false;
                }
                bool v1042;
                v1042 = v1041 == false;
                if (v1042){
                    assert("The indices should be inside the range of the dimension." && v1041);
                } else {
                }
                int v1044;
                v1044 = v1002 * 4l;
                int v1045;
                v1045 = v1032 + v1044;
                bool v1046;
                v1046 = 0l <= v1030;
                bool v1048;
                if (v1046){
                    bool v1047;
                    v1047 = v1030 < 1l;
                    v1048 = v1047;
                } else {
                    v1048 = false;
                }
                bool v1049;
                v1049 = v1048 == false;
                if (v1049){
                    assert("The indices should be inside the range of the dimension." && v1048);
                } else {
                }
                int v1051;
                v1051 = v1030 * 128l;
                int v1052;
                v1052 = v1045 + v1051;
                assert("Tensor range check" && 0 <= v1030 && v1030 < 1l);
                assert("Tensor range check" && 0 <= v1032 && v1032 < 4l);
                int v1053;
                v1053 = 4l * v1030;
                int v1054;
                v1054 = v1053 + v1032;
                v1022[v1054] = v1052;
                v1032 += 1l ;
            }
            v1030 += 1l ;
        }
        bool v1055;
        v1055 = 0l <= v1003;
        bool v1056;
        v1056 = v1055 && v1004;
        bool v1057;
        v1057 = v1056 == false;
        if (v1057){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1056);
        } else {
        }
        bool v1059;
        v1059 = v1013 && v1016;
        bool v1060;
        v1060 = v1059 == false;
        if (v1060){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1059);
        } else {
        }
        int v1062;
        v1062 = v1011 + v1003;
        float v1063;
        v1063 = 0.0f;
        int v1064;
        v1064 = 0l;
        while (while_method_3(v1064)){
            int v1066;
            v1066 = 0l;
            while (while_method_1(v1066)){
                assert("Tensor range check" && 0 <= v1064 && v1064 < 1l);
                assert("Tensor range check" && 0 <= v1066 && v1066 < 4l);
                int v1068;
                v1068 = 4l * v1064;
                int v1069;
                v1069 = v1068 + v1066;
                float v1070;
                v1070 = v1021[v1069];
                float v1071;
                v1071 = v1063 + v1070;
                v1063 = v1071;
                v1066 += 1l ;
            }
            v1064 += 1l ;
        }
        auto v1072 = cooperative_groups::coalesced_threads();
        int v1073;
        v1073 = threadIdx.x;
        int v1074;
        v1074 = v1073 / 32l;
        auto v1075 = cooperative_groups::labeled_partition(v1072,v1074);
        Closure0 v1076{};
        float v1077;
        v1077 = cooperative_groups::reduce(v1075, v1063, v1076);
        float v1078;
        v1078 = v1077 / 128.0f;
        float v1079[4l];
        int v1080;
        v1080 = 0l;
        while (while_method_3(v1080)){
            int v1082;
            v1082 = 0l;
            while (while_method_1(v1082)){
                assert("Tensor range check" && 0 <= v1080 && v1080 < 1l);
                assert("Tensor range check" && 0 <= v1082 && v1082 < 4l);
                int v1084;
                v1084 = 4l * v1080;
                int v1085;
                v1085 = v1084 + v1082;
                float v1086;
                v1086 = v1021[v1085];
                float v1087;
                v1087 = v1086 - v1078;
                float v1088;
                v1088 = exp(v1087);
                assert("Tensor range check" && 0 <= v1080 && v1080 < 1l);
                assert("Tensor range check" && 0 <= v1082 && v1082 < 4l);
                v1079[v1085] = v1088;
                v1082 += 1l ;
            }
            v1080 += 1l ;
        }
        float v1089;
        v1089 = 0.0f;
        int v1090;
        v1090 = 0l;
        while (while_method_3(v1090)){
            int v1092;
            v1092 = 0l;
            while (while_method_1(v1092)){
                assert("Tensor range check" && 0 <= v1090 && v1090 < 1l);
                assert("Tensor range check" && 0 <= v1092 && v1092 < 4l);
                int v1094;
                v1094 = 4l * v1090;
                int v1095;
                v1095 = v1094 + v1092;
                float v1096;
                v1096 = v1079[v1095];
                float v1097;
                v1097 = v1089 + v1096;
                v1089 = v1097;
                v1092 += 1l ;
            }
            v1090 += 1l ;
        }
        auto v1098 = cooperative_groups::coalesced_threads();
        int v1099;
        v1099 = threadIdx.x;
        int v1100;
        v1100 = v1099 / 32l;
        auto v1101 = cooperative_groups::labeled_partition(v1098,v1100);
        float v1102;
        v1102 = cooperative_groups::reduce(v1101, v1089, v1076);
        float v1103[4l];
        int v1104;
        v1104 = 0l;
        while (while_method_3(v1104)){
            int v1106;
            v1106 = 0l;
            while (while_method_1(v1106)){
                assert("Tensor range check" && 0 <= v1104 && v1104 < 1l);
                assert("Tensor range check" && 0 <= v1106 && v1106 < 4l);
                int v1108;
                v1108 = 4l * v1104;
                int v1109;
                v1109 = v1108 + v1106;
                float v1110;
                v1110 = v1079[v1109];
                float v1111;
                v1111 = v1110 / v1102;
                assert("Tensor range check" && 0 <= v1104 && v1104 < 1l);
                assert("Tensor range check" && 0 <= v1106 && v1106 < 4l);
                v1103[v1109] = v1111;
                v1106 += 1l ;
            }
            v1104 += 1l ;
        }
        float v1112[4l];
        float v1113;
        v1113 = 0.0f;
        int v1114;
        v1114 = 0l;
        while (while_method_3(v1114)){
            assert("Tensor range check" && 0 <= v1114 && v1114 < 1l);
            int v1116;
            v1116 = 4l * v1114;
            assert("Tensor range check" && 0 <= v1114 && v1114 < 1l);
            int v1117; float v1118;
            Tuple0 tmp90 = Tuple0{0l, 0.0f};
            v1117 = tmp90.v0; v1118 = tmp90.v1;
            while (while_method_1(v1117)){
                assert("Tensor range check" && 0 <= v1117 && v1117 < 4l);
                int v1120;
                v1120 = v1117 + v1116;
                float v1121;
                v1121 = v1103[v1120];
                float v1122;
                v1122 = v1118 + v1121;
                v1118 = v1122;
                v1117 += 1l ;
            }
            auto v1123 = cooperative_groups::coalesced_threads();
            int v1124;
            v1124 = threadIdx.x;
            int v1125;
            v1125 = v1124 / 32l;
            auto v1126 = cooperative_groups::labeled_partition(v1123,v1125);
            Closure2 v1127{};
            float v1128;
            v1128 = cooperative_groups::inclusive_scan(v1126, v1118, v1127);
            float v1129;
            v1129 = v1126.shfl_up(v1128,1);
            bool v1130;
            v1130 = v1126.thread_rank() == 0;
            float v1131;
            if (v1130){
                v1131 = 0.0f;
            } else {
                v1131 = v1129;
            }
            float v1132;
            v1132 = v1126.shfl(v1128,v1126.num_threads()-1);
            float v1133;
            v1133 = v1113 + v1131;
            int v1134; float v1135;
            Tuple0 tmp91 = Tuple0{0l, v1133};
            v1134 = tmp91.v0; v1135 = tmp91.v1;
            while (while_method_1(v1134)){
                assert("Tensor range check" && 0 <= v1134 && v1134 < 4l);
                int v1137;
                v1137 = v1134 + v1116;
                float v1138;
                v1138 = v1103[v1137];
                float v1139;
                v1139 = v1135 + v1138;
                assert("Tensor range check" && 0 <= v1134 && v1134 < 4l);
                v1112[v1137] = v1139;
                v1135 = v1139;
                v1134 += 1l ;
            }
            float v1140;
            v1140 = v1113 + v1132;
            v1113 = v1140;
            v1114 += 1l ;
        }
        float v1141[4l];
        bool v1142[4l];
        int v1143;
        v1143 = 0l;
        while (while_method_3(v1143)){
            int v1145;
            v1145 = 0l;
            while (while_method_1(v1145)){
                assert("Tensor range check" && 0 <= v1143 && v1143 < 1l);
                assert("Tensor range check" && 0 <= v1145 && v1145 < 4l);
                int v1147;
                v1147 = 4l * v1143;
                int v1148;
                v1148 = v1147 + v1145;
                float v1149;
                v1149 = v1112[v1148];
                float v1150;
                v1150 = v1103[v1148];
                bool v1151;
                v1151 = v1150 > 0.0f;
                assert("Tensor range check" && 0 <= v1143 && v1143 < 1l);
                assert("Tensor range check" && 0 <= v1145 && v1145 < 4l);
                v1141[v1148] = v1149;
                v1142[v1148] = v1151;
                v1145 += 1l ;
            }
            v1143 += 1l ;
        }
        float v1152; bool v1153;
        Tuple3 tmp92 = Tuple3{-1.0f / 0.0f, false};
        v1152 = tmp92.v0; v1153 = tmp92.v1;
        int v1154;
        v1154 = 0l;
        while (while_method_3(v1154)){
            int v1156;
            v1156 = 0l;
            while (while_method_1(v1156)){
                assert("Tensor range check" && 0 <= v1154 && v1154 < 1l);
                assert("Tensor range check" && 0 <= v1156 && v1156 < 4l);
                int v1158;
                v1158 = 4l * v1154;
                int v1159;
                v1159 = v1158 + v1156;
                float v1160;
                v1160 = v1141[v1159];
                bool v1161;
                v1161 = v1142[v1159];
                float v1168; bool v1169;
                if (v1153){
                    if (v1161){
                        bool v1162;
                        v1162 = v1152 >= v1160;
                        float v1163;
                        if (v1162){
                            v1163 = v1152;
                        } else {
                            v1163 = v1160;
                        }
                        v1168 = v1163; v1169 = true;
                    } else {
                        v1168 = v1152; v1169 = v1153;
                    }
                } else {
                    if (v1161){
                        v1168 = v1160; v1169 = v1161;
                    } else {
                        v1168 = v1152; v1169 = v1153;
                    }
                }
                v1152 = v1168;
                v1153 = v1169;
                v1156 += 1l ;
            }
            v1154 += 1l ;
        }
        auto v1170 = cooperative_groups::coalesced_threads();
        int v1171;
        v1171 = threadIdx.x;
        int v1172;
        v1172 = v1171 / 32l;
        auto v1173 = cooperative_groups::labeled_partition(v1170,v1172);
        Closure5 v1174{};
        float v1175; bool v1176;
        Tuple3 tmp93 = cooperative_groups::reduce(v1173, Tuple3{v1152, v1153}, v1174);
        v1175 = tmp93.v0; v1176 = tmp93.v1;
        bool v1177;
        v1177 = v1176 == false;
        if (v1177){
            assert("The local reduce must be true." && v1176);
        } else {
        }
        float v1179[4l];
        int v1180[4l];
        int v1181;
        v1181 = 0l;
        while (while_method_3(v1181)){
            int v1183;
            v1183 = 0l;
            while (while_method_1(v1183)){
                assert("Tensor range check" && 0 <= v1181 && v1181 < 1l);
                assert("Tensor range check" && 0 <= v1183 && v1183 < 4l);
                int v1185;
                v1185 = 4l * v1181;
                int v1186;
                v1186 = v1185 + v1183;
                int v1187;
                v1187 = v1022[v1186];
                float v1188;
                v1188 = curand_uniform(&v997);
                assert("Tensor range check" && 0 <= v1181 && v1181 < 1l);
                assert("Tensor range check" && 0 <= v1183 && v1183 < 4l);
                v1179[v1186] = v1188;
                v1180[v1186] = v1187;
                v1183 += 1l ;
            }
            v1181 += 1l ;
        }
        float v1189; int v1190;
        Tuple1 tmp94 = Tuple1{0.0f, 2147483647l};
        v1189 = tmp94.v0; v1190 = tmp94.v1;
        int v1191;
        v1191 = 0l;
        while (while_method_3(v1191)){
            int v1193;
            v1193 = 0l;
            while (while_method_1(v1193)){
                assert("Tensor range check" && 0 <= v1191 && v1191 < 1l);
                assert("Tensor range check" && 0 <= v1193 && v1193 < 4l);
                int v1195;
                v1195 = 4l * v1191;
                int v1196;
                v1196 = v1195 + v1193;
                float v1197;
                v1197 = v1179[v1196];
                int v1198;
                v1198 = v1180[v1196];
                bool v1199;
                v1199 = v1190 < v1198;
                float v1200; int v1201;
                if (v1199){
                    v1200 = v1189; v1201 = v1190;
                } else {
                    v1200 = v1197; v1201 = v1198;
                }
                v1189 = v1200;
                v1190 = v1201;
                v1193 += 1l ;
            }
            v1191 += 1l ;
        }
        auto v1202 = cooperative_groups::coalesced_threads();
        int v1203;
        v1203 = threadIdx.x;
        int v1204;
        v1204 = v1203 / 32l;
        auto v1205 = cooperative_groups::labeled_partition(v1202,v1204);
        Closure6 v1206{};
        float v1207; int v1208;
        Tuple1 tmp95 = cooperative_groups::reduce(v1205, Tuple1{v1189, v1190}, v1206);
        v1207 = tmp95.v0; v1208 = tmp95.v1;
        float v1209;
        v1209 = v1175 * v1207;
        int v1210[4l];
        bool v1211[4l];
        int v1212;
        v1212 = 0l;
        while (while_method_3(v1212)){
            int v1214;
            v1214 = 0l;
            while (while_method_1(v1214)){
                assert("Tensor range check" && 0 <= v1212 && v1212 < 1l);
                assert("Tensor range check" && 0 <= v1214 && v1214 < 4l);
                int v1216;
                v1216 = 4l * v1212;
                int v1217;
                v1217 = v1216 + v1214;
                float v1218;
                v1218 = v1141[v1217];
                bool v1219;
                v1219 = v1142[v1217];
                int v1220;
                v1220 = v1022[v1217];
                int v1223; bool v1224;
                if (v1219){
                    float v1221;
                    v1221 = v1218 - v1209;
                    bool v1222;
                    v1222 = v1221 >= 0.0f;
                    v1223 = v1220; v1224 = v1222;
                } else {
                    v1223 = 2147483647l; v1224 = false;
                }
                assert("Tensor range check" && 0 <= v1212 && v1212 < 1l);
                assert("Tensor range check" && 0 <= v1214 && v1214 < 4l);
                v1210[v1217] = v1223;
                v1211[v1217] = v1224;
                v1214 += 1l ;
            }
            v1212 += 1l ;
        }
        int v1225; bool v1226;
        Tuple4 tmp96 = Tuple4{2147483647l, false};
        v1225 = tmp96.v0; v1226 = tmp96.v1;
        int v1227;
        v1227 = 0l;
        while (while_method_3(v1227)){
            int v1229;
            v1229 = 0l;
            while (while_method_1(v1229)){
                assert("Tensor range check" && 0 <= v1227 && v1227 < 1l);
                assert("Tensor range check" && 0 <= v1229 && v1229 < 4l);
                int v1231;
                v1231 = 4l * v1227;
                int v1232;
                v1232 = v1231 + v1229;
                int v1233;
                v1233 = v1210[v1232];
                bool v1234;
                v1234 = v1211[v1232];
                int v1241; bool v1242;
                if (v1226){
                    if (v1234){
                        bool v1235;
                        v1235 = v1225 < v1233;
                        int v1236;
                        if (v1235){
                            v1236 = v1225;
                        } else {
                            v1236 = v1233;
                        }
                        v1241 = v1236; v1242 = true;
                    } else {
                        v1241 = v1225; v1242 = v1226;
                    }
                } else {
                    if (v1234){
                        v1241 = v1233; v1242 = v1234;
                    } else {
                        v1241 = v1225; v1242 = v1226;
                    }
                }
                v1225 = v1241;
                v1226 = v1242;
                v1229 += 1l ;
            }
            v1227 += 1l ;
        }
        auto v1243 = cooperative_groups::coalesced_threads();
        int v1244;
        v1244 = threadIdx.x;
        int v1245;
        v1245 = v1244 / 32l;
        auto v1246 = cooperative_groups::labeled_partition(v1243,v1245);
        Closure7 v1247{};
        int v1248; bool v1249;
        Tuple4 tmp97 = cooperative_groups::reduce(v1246, Tuple4{v1225, v1226}, v1247);
        v1248 = tmp97.v0; v1249 = tmp97.v1;
        bool v1250;
        v1250 = v1249 == false;
        if (v1250){
            assert("The local reduce must be true." && v1249);
        } else {
        }
        assert("Tensor range check" && 0 <= v1011 && v1011 < 64l);
        int v1252;
        v1252 = 0l;
        while (while_method_3(v1252)){
            assert("Tensor range check" && 0 <= v1252 && v1252 < 1l);
            int v1254;
            v1254 = 128l * v1252;
            int v1255;
            v1255 = v1254 + v1020;
            assert("Tensor range check" && 0 <= v1252 && v1252 < 1l);
            int v1256;
            v1256 = 4l * v1252;
            int4* v1257;
            v1257 = reinterpret_cast<int4*>(v1103 + v1256);
            int4* v1258;
            v1258 = reinterpret_cast<int4*>(v13 + v1255);
            assert("Pointer alignment check" && (unsigned long long)(v1257) % 4l == 0 && (unsigned long long)(v1258) % 4l == 0);
            *v1258 = *v1257;
            v1252 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1011 && v1011 < 64l);
        v14[v1062] = v1248;
        v1011 += 24l ;
    }
    v15.sync() ;
    int v1259;
    v1259 = threadIdx.x;
    unsigned long long v1260;
    v1260 = (unsigned long long)v1259;
    curandStatePhilox4_32_10_t v1261;
    curand_init(12344321ull,v1260,0ull,&v1261);
    int v1262;
    v1262 = threadIdx.x;
    bool v1263;
    v1263 = 0l <= v1262;
    bool v1264;
    v1264 = v1263 == false;
    if (v1264){
        assert("The index needs to be zero or positive." && v1263);
    } else {
    }
    int v1266;
    v1266 = v1262 % 32l;
    int v1267;
    v1267 = v1262 / 32l;
    bool v1268;
    v1268 = v1267 < 1l;
    bool v1269;
    v1269 = v1268 == false;
    if (v1269){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1268);
    } else {
    }
    assert("Tensor range check" && 0 <= v1267 && v1267 < 1l);
    assert("Tensor range check" && 0 <= v1266 && v1266 < 32l);
    int v1271;
    v1271 = 4l * v1266;
    int v1272;
    v1272 = 128l * v1267;
    int v1273;
    v1273 = v1272 + v1271;
    assert("Tensor range check" && 0 <= v1267 && v1267 < 1l);
    assert("Tensor range check" && 0 <= v1266 && v1266 < 32l);
    assert("Tensor range check" && 0 <= v1267 && v1267 < 1l);
    int v1274;
    v1274 = blockIdx.x;
    int v1275;
    v1275 = v1274;
    while (while_method_2(v1275)){
        bool v1277;
        v1277 = 0l <= v1275;
        bool v1278;
        v1278 = v1277 == false;
        if (v1278){
            assert("The index needs to be zero or positive." && v1277);
        } else {
        }
        bool v1280;
        v1280 = v1275 < 64l;
        bool v1281;
        v1281 = v1280 == false;
        if (v1281){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1280);
        } else {
        }
        assert("Tensor range check" && 0 <= v1275 && v1275 < 64l);
        int v1283;
        v1283 = 128l * v1275;
        int v1284;
        v1284 = v1283 + v1273;
        float v1285[4l];
        int v1286[4l];
        int v1287;
        v1287 = 0l;
        while (while_method_3(v1287)){
            assert("Tensor range check" && 0 <= v1287 && v1287 < 1l);
            int v1289;
            v1289 = 4l * v1287;
            assert("Tensor range check" && 0 <= v1287 && v1287 < 1l);
            int v1290;
            v1290 = 128l * v1287;
            int v1291;
            v1291 = v1290 + v1284;
            int4* v1292;
            v1292 = reinterpret_cast<int4*>(v1 + v1291);
            int4* v1293;
            v1293 = reinterpret_cast<int4*>(v1285 + v1289);
            assert("Pointer alignment check" && (unsigned long long)(v1292) % 4l == 0 && (unsigned long long)(v1293) % 4l == 0);
            *v1293 = *v1292;
            v1287 += 1l ;
        }
        int v1294;
        v1294 = 0l;
        while (while_method_3(v1294)){
            int v1296;
            v1296 = 0l;
            while (while_method_1(v1296)){
                bool v1298;
                v1298 = 0l <= v1296;
                bool v1300;
                if (v1298){
                    bool v1299;
                    v1299 = v1296 < 4l;
                    v1300 = v1299;
                } else {
                    v1300 = false;
                }
                bool v1301;
                v1301 = v1300 == false;
                if (v1301){
                    assert("The indices should be inside the range of the dimension." && v1300);
                } else {
                }
                bool v1303;
                v1303 = 0l <= v1266;
                bool v1305;
                if (v1303){
                    bool v1304;
                    v1304 = v1266 < 32l;
                    v1305 = v1304;
                } else {
                    v1305 = false;
                }
                bool v1306;
                v1306 = v1305 == false;
                if (v1306){
                    assert("The indices should be inside the range of the dimension." && v1305);
                } else {
                }
                int v1308;
                v1308 = v1266 * 4l;
                int v1309;
                v1309 = v1296 + v1308;
                bool v1310;
                v1310 = 0l <= v1294;
                bool v1312;
                if (v1310){
                    bool v1311;
                    v1311 = v1294 < 1l;
                    v1312 = v1311;
                } else {
                    v1312 = false;
                }
                bool v1313;
                v1313 = v1312 == false;
                if (v1313){
                    assert("The indices should be inside the range of the dimension." && v1312);
                } else {
                }
                int v1315;
                v1315 = v1294 * 128l;
                int v1316;
                v1316 = v1309 + v1315;
                assert("Tensor range check" && 0 <= v1294 && v1294 < 1l);
                assert("Tensor range check" && 0 <= v1296 && v1296 < 4l);
                int v1317;
                v1317 = 4l * v1294;
                int v1318;
                v1318 = v1317 + v1296;
                v1286[v1318] = v1316;
                v1296 += 1l ;
            }
            v1294 += 1l ;
        }
        bool v1319;
        v1319 = 0l <= v1267;
        bool v1320;
        v1320 = v1319 && v1268;
        bool v1321;
        v1321 = v1320 == false;
        if (v1321){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1320);
        } else {
        }
        bool v1323;
        v1323 = v1277 && v1280;
        bool v1324;
        v1324 = v1323 == false;
        if (v1324){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1323);
        } else {
        }
        int v1326;
        v1326 = v1275 + v1267;
        bool v1327[4l];
        int v1328;
        v1328 = 0l;
        while (while_method_3(v1328)){
            int v1330;
            v1330 = 0l;
            while (while_method_1(v1330)){
                assert("Tensor range check" && 0 <= v1328 && v1328 < 1l);
                assert("Tensor range check" && 0 <= v1330 && v1330 < 4l);
                int v1332;
                v1332 = 4l * v1328;
                int v1333;
                v1333 = v1332 + v1330;
                float v1334;
                v1334 = v1285[v1333];
                int v1335;
                v1335 = v1286[v1333];
                bool v1336;
                v1336 = v1335 < 11l;
                assert("Tensor range check" && 0 <= v1328 && v1328 < 1l);
                assert("Tensor range check" && 0 <= v1330 && v1330 < 4l);
                v1327[v1333] = v1336;
                v1330 += 1l ;
            }
            v1328 += 1l ;
        }
        int v1337[4l];
        int v1338;
        v1338 = 0l;
        while (while_method_3(v1338)){
            int v1340;
            v1340 = 0l;
            while (while_method_1(v1340)){
                assert("Tensor range check" && 0 <= v1338 && v1338 < 1l);
                assert("Tensor range check" && 0 <= v1340 && v1340 < 4l);
                int v1342;
                v1342 = 4l * v1338;
                int v1343;
                v1343 = v1342 + v1340;
                bool v1344;
                v1344 = v1327[v1343];
                int v1345;
                if (v1344){
                    v1345 = 1l;
                } else {
                    v1345 = 0l;
                }
                assert("Tensor range check" && 0 <= v1338 && v1338 < 1l);
                assert("Tensor range check" && 0 <= v1340 && v1340 < 4l);
                v1337[v1343] = v1345;
                v1340 += 1l ;
            }
            v1338 += 1l ;
        }
        int v1346;
        v1346 = 0l;
        int v1347;
        v1347 = 0l;
        while (while_method_3(v1347)){
            int v1349;
            v1349 = 0l;
            while (while_method_1(v1349)){
                assert("Tensor range check" && 0 <= v1347 && v1347 < 1l);
                assert("Tensor range check" && 0 <= v1349 && v1349 < 4l);
                int v1351;
                v1351 = 4l * v1347;
                int v1352;
                v1352 = v1351 + v1349;
                int v1353;
                v1353 = v1337[v1352];
                int v1354;
                v1354 = v1346 + v1353;
                v1346 = v1354;
                v1349 += 1l ;
            }
            v1347 += 1l ;
        }
        auto v1355 = cooperative_groups::coalesced_threads();
        int v1356;
        v1356 = threadIdx.x;
        int v1357;
        v1357 = v1356 / 32l;
        auto v1358 = cooperative_groups::labeled_partition(v1355,v1357);
        Closure4 v1359{};
        int v1360;
        v1360 = cooperative_groups::reduce(v1358, v1346, v1359);
        float v1361[4l];
        int v1362;
        v1362 = 0l;
        while (while_method_3(v1362)){
            int v1364;
            v1364 = 0l;
            while (while_method_1(v1364)){
                assert("Tensor range check" && 0 <= v1362 && v1362 < 1l);
                assert("Tensor range check" && 0 <= v1364 && v1364 < 4l);
                int v1366;
                v1366 = 4l * v1362;
                int v1367;
                v1367 = v1366 + v1364;
                float v1368;
                v1368 = v1285[v1367];
                bool v1369;
                v1369 = v1327[v1367];
                float v1370;
                if (v1369){
                    v1370 = v1368;
                } else {
                    v1370 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1362 && v1362 < 1l);
                assert("Tensor range check" && 0 <= v1364 && v1364 < 4l);
                v1361[v1367] = v1370;
                v1364 += 1l ;
            }
            v1362 += 1l ;
        }
        float v1371;
        v1371 = 0.0f;
        int v1372;
        v1372 = 0l;
        while (while_method_3(v1372)){
            int v1374;
            v1374 = 0l;
            while (while_method_1(v1374)){
                assert("Tensor range check" && 0 <= v1372 && v1372 < 1l);
                assert("Tensor range check" && 0 <= v1374 && v1374 < 4l);
                int v1376;
                v1376 = 4l * v1372;
                int v1377;
                v1377 = v1376 + v1374;
                float v1378;
                v1378 = v1361[v1377];
                float v1379;
                v1379 = v1371 + v1378;
                v1371 = v1379;
                v1374 += 1l ;
            }
            v1372 += 1l ;
        }
        auto v1380 = cooperative_groups::coalesced_threads();
        int v1381;
        v1381 = threadIdx.x;
        int v1382;
        v1382 = v1381 / 32l;
        auto v1383 = cooperative_groups::labeled_partition(v1380,v1382);
        Closure0 v1384{};
        float v1385;
        v1385 = cooperative_groups::reduce(v1383, v1371, v1384);
        float v1386;
        v1386 = (float)v1360;
        float v1387;
        v1387 = v1385 / v1386;
        float v1388[4l];
        int v1389;
        v1389 = 0l;
        while (while_method_3(v1389)){
            int v1391;
            v1391 = 0l;
            while (while_method_1(v1391)){
                assert("Tensor range check" && 0 <= v1389 && v1389 < 1l);
                assert("Tensor range check" && 0 <= v1391 && v1391 < 4l);
                int v1393;
                v1393 = 4l * v1389;
                int v1394;
                v1394 = v1393 + v1391;
                float v1395;
                v1395 = v1285[v1394];
                bool v1396;
                v1396 = v1327[v1394];
                float v1397;
                if (v1396){
                    v1397 = v1395;
                } else {
                    v1397 = -1.0f / 0.0f;
                }
                float v1398;
                v1398 = v1397 - v1387;
                float v1399;
                v1399 = exp(v1398);
                assert("Tensor range check" && 0 <= v1389 && v1389 < 1l);
                assert("Tensor range check" && 0 <= v1391 && v1391 < 4l);
                v1388[v1394] = v1399;
                v1391 += 1l ;
            }
            v1389 += 1l ;
        }
        float v1400;
        v1400 = 0.0f;
        int v1401;
        v1401 = 0l;
        while (while_method_3(v1401)){
            int v1403;
            v1403 = 0l;
            while (while_method_1(v1403)){
                assert("Tensor range check" && 0 <= v1401 && v1401 < 1l);
                assert("Tensor range check" && 0 <= v1403 && v1403 < 4l);
                int v1405;
                v1405 = 4l * v1401;
                int v1406;
                v1406 = v1405 + v1403;
                float v1407;
                v1407 = v1388[v1406];
                float v1408;
                v1408 = v1400 + v1407;
                v1400 = v1408;
                v1403 += 1l ;
            }
            v1401 += 1l ;
        }
        auto v1409 = cooperative_groups::coalesced_threads();
        int v1410;
        v1410 = threadIdx.x;
        int v1411;
        v1411 = v1410 / 32l;
        auto v1412 = cooperative_groups::labeled_partition(v1409,v1411);
        float v1413;
        v1413 = cooperative_groups::reduce(v1412, v1400, v1384);
        float v1414[4l];
        int v1415;
        v1415 = 0l;
        while (while_method_3(v1415)){
            int v1417;
            v1417 = 0l;
            while (while_method_1(v1417)){
                assert("Tensor range check" && 0 <= v1415 && v1415 < 1l);
                assert("Tensor range check" && 0 <= v1417 && v1417 < 4l);
                int v1419;
                v1419 = 4l * v1415;
                int v1420;
                v1420 = v1419 + v1417;
                float v1421;
                v1421 = v1388[v1420];
                float v1422;
                v1422 = v1421 / v1413;
                assert("Tensor range check" && 0 <= v1415 && v1415 < 1l);
                assert("Tensor range check" && 0 <= v1417 && v1417 < 4l);
                v1414[v1420] = v1422;
                v1417 += 1l ;
            }
            v1415 += 1l ;
        }
        float v1423[4l];
        float v1424;
        v1424 = 0.0f;
        int v1425;
        v1425 = 0l;
        while (while_method_3(v1425)){
            assert("Tensor range check" && 0 <= v1425 && v1425 < 1l);
            int v1427;
            v1427 = 4l * v1425;
            assert("Tensor range check" && 0 <= v1425 && v1425 < 1l);
            int v1428; float v1429;
            Tuple0 tmp98 = Tuple0{0l, 0.0f};
            v1428 = tmp98.v0; v1429 = tmp98.v1;
            while (while_method_1(v1428)){
                assert("Tensor range check" && 0 <= v1428 && v1428 < 4l);
                int v1431;
                v1431 = v1428 + v1427;
                float v1432;
                v1432 = v1414[v1431];
                float v1433;
                v1433 = v1429 + v1432;
                v1429 = v1433;
                v1428 += 1l ;
            }
            auto v1434 = cooperative_groups::coalesced_threads();
            int v1435;
            v1435 = threadIdx.x;
            int v1436;
            v1436 = v1435 / 32l;
            auto v1437 = cooperative_groups::labeled_partition(v1434,v1436);
            Closure2 v1438{};
            float v1439;
            v1439 = cooperative_groups::inclusive_scan(v1437, v1429, v1438);
            float v1440;
            v1440 = v1437.shfl_up(v1439,1);
            bool v1441;
            v1441 = v1437.thread_rank() == 0;
            float v1442;
            if (v1441){
                v1442 = 0.0f;
            } else {
                v1442 = v1440;
            }
            float v1443;
            v1443 = v1437.shfl(v1439,v1437.num_threads()-1);
            float v1444;
            v1444 = v1424 + v1442;
            int v1445; float v1446;
            Tuple0 tmp99 = Tuple0{0l, v1444};
            v1445 = tmp99.v0; v1446 = tmp99.v1;
            while (while_method_1(v1445)){
                assert("Tensor range check" && 0 <= v1445 && v1445 < 4l);
                int v1448;
                v1448 = v1445 + v1427;
                float v1449;
                v1449 = v1414[v1448];
                float v1450;
                v1450 = v1446 + v1449;
                assert("Tensor range check" && 0 <= v1445 && v1445 < 4l);
                v1423[v1448] = v1450;
                v1446 = v1450;
                v1445 += 1l ;
            }
            float v1451;
            v1451 = v1424 + v1443;
            v1424 = v1451;
            v1425 += 1l ;
        }
        float v1452[4l];
        bool v1453[4l];
        int v1454;
        v1454 = 0l;
        while (while_method_3(v1454)){
            int v1456;
            v1456 = 0l;
            while (while_method_1(v1456)){
                assert("Tensor range check" && 0 <= v1454 && v1454 < 1l);
                assert("Tensor range check" && 0 <= v1456 && v1456 < 4l);
                int v1458;
                v1458 = 4l * v1454;
                int v1459;
                v1459 = v1458 + v1456;
                float v1460;
                v1460 = v1423[v1459];
                float v1461;
                v1461 = v1414[v1459];
                bool v1462;
                v1462 = v1461 > 0.0f;
                assert("Tensor range check" && 0 <= v1454 && v1454 < 1l);
                assert("Tensor range check" && 0 <= v1456 && v1456 < 4l);
                v1452[v1459] = v1460;
                v1453[v1459] = v1462;
                v1456 += 1l ;
            }
            v1454 += 1l ;
        }
        float v1463; bool v1464;
        Tuple3 tmp100 = Tuple3{-1.0f / 0.0f, false};
        v1463 = tmp100.v0; v1464 = tmp100.v1;
        int v1465;
        v1465 = 0l;
        while (while_method_3(v1465)){
            int v1467;
            v1467 = 0l;
            while (while_method_1(v1467)){
                assert("Tensor range check" && 0 <= v1465 && v1465 < 1l);
                assert("Tensor range check" && 0 <= v1467 && v1467 < 4l);
                int v1469;
                v1469 = 4l * v1465;
                int v1470;
                v1470 = v1469 + v1467;
                float v1471;
                v1471 = v1452[v1470];
                bool v1472;
                v1472 = v1453[v1470];
                float v1479; bool v1480;
                if (v1464){
                    if (v1472){
                        bool v1473;
                        v1473 = v1463 >= v1471;
                        float v1474;
                        if (v1473){
                            v1474 = v1463;
                        } else {
                            v1474 = v1471;
                        }
                        v1479 = v1474; v1480 = true;
                    } else {
                        v1479 = v1463; v1480 = v1464;
                    }
                } else {
                    if (v1472){
                        v1479 = v1471; v1480 = v1472;
                    } else {
                        v1479 = v1463; v1480 = v1464;
                    }
                }
                v1463 = v1479;
                v1464 = v1480;
                v1467 += 1l ;
            }
            v1465 += 1l ;
        }
        auto v1481 = cooperative_groups::coalesced_threads();
        int v1482;
        v1482 = threadIdx.x;
        int v1483;
        v1483 = v1482 / 32l;
        auto v1484 = cooperative_groups::labeled_partition(v1481,v1483);
        Closure5 v1485{};
        float v1486; bool v1487;
        Tuple3 tmp101 = cooperative_groups::reduce(v1484, Tuple3{v1463, v1464}, v1485);
        v1486 = tmp101.v0; v1487 = tmp101.v1;
        bool v1488;
        v1488 = v1487 == false;
        if (v1488){
            assert("The local reduce must be true." && v1487);
        } else {
        }
        float v1490[4l];
        int v1491[4l];
        int v1492;
        v1492 = 0l;
        while (while_method_3(v1492)){
            int v1494;
            v1494 = 0l;
            while (while_method_1(v1494)){
                assert("Tensor range check" && 0 <= v1492 && v1492 < 1l);
                assert("Tensor range check" && 0 <= v1494 && v1494 < 4l);
                int v1496;
                v1496 = 4l * v1492;
                int v1497;
                v1497 = v1496 + v1494;
                int v1498;
                v1498 = v1286[v1497];
                float v1499;
                v1499 = curand_uniform(&v1261);
                assert("Tensor range check" && 0 <= v1492 && v1492 < 1l);
                assert("Tensor range check" && 0 <= v1494 && v1494 < 4l);
                v1490[v1497] = v1499;
                v1491[v1497] = v1498;
                v1494 += 1l ;
            }
            v1492 += 1l ;
        }
        float v1500; int v1501;
        Tuple1 tmp102 = Tuple1{0.0f, 2147483647l};
        v1500 = tmp102.v0; v1501 = tmp102.v1;
        int v1502;
        v1502 = 0l;
        while (while_method_3(v1502)){
            int v1504;
            v1504 = 0l;
            while (while_method_1(v1504)){
                assert("Tensor range check" && 0 <= v1502 && v1502 < 1l);
                assert("Tensor range check" && 0 <= v1504 && v1504 < 4l);
                int v1506;
                v1506 = 4l * v1502;
                int v1507;
                v1507 = v1506 + v1504;
                float v1508;
                v1508 = v1490[v1507];
                int v1509;
                v1509 = v1491[v1507];
                bool v1510;
                v1510 = v1501 < v1509;
                float v1511; int v1512;
                if (v1510){
                    v1511 = v1500; v1512 = v1501;
                } else {
                    v1511 = v1508; v1512 = v1509;
                }
                v1500 = v1511;
                v1501 = v1512;
                v1504 += 1l ;
            }
            v1502 += 1l ;
        }
        auto v1513 = cooperative_groups::coalesced_threads();
        int v1514;
        v1514 = threadIdx.x;
        int v1515;
        v1515 = v1514 / 32l;
        auto v1516 = cooperative_groups::labeled_partition(v1513,v1515);
        Closure6 v1517{};
        float v1518; int v1519;
        Tuple1 tmp103 = cooperative_groups::reduce(v1516, Tuple1{v1500, v1501}, v1517);
        v1518 = tmp103.v0; v1519 = tmp103.v1;
        float v1520;
        v1520 = v1486 * v1518;
        int v1521[4l];
        bool v1522[4l];
        int v1523;
        v1523 = 0l;
        while (while_method_3(v1523)){
            int v1525;
            v1525 = 0l;
            while (while_method_1(v1525)){
                assert("Tensor range check" && 0 <= v1523 && v1523 < 1l);
                assert("Tensor range check" && 0 <= v1525 && v1525 < 4l);
                int v1527;
                v1527 = 4l * v1523;
                int v1528;
                v1528 = v1527 + v1525;
                float v1529;
                v1529 = v1452[v1528];
                bool v1530;
                v1530 = v1453[v1528];
                int v1531;
                v1531 = v1286[v1528];
                int v1534; bool v1535;
                if (v1530){
                    float v1532;
                    v1532 = v1529 - v1520;
                    bool v1533;
                    v1533 = v1532 >= 0.0f;
                    v1534 = v1531; v1535 = v1533;
                } else {
                    v1534 = 2147483647l; v1535 = false;
                }
                assert("Tensor range check" && 0 <= v1523 && v1523 < 1l);
                assert("Tensor range check" && 0 <= v1525 && v1525 < 4l);
                v1521[v1528] = v1534;
                v1522[v1528] = v1535;
                v1525 += 1l ;
            }
            v1523 += 1l ;
        }
        int v1536; bool v1537;
        Tuple4 tmp104 = Tuple4{2147483647l, false};
        v1536 = tmp104.v0; v1537 = tmp104.v1;
        int v1538;
        v1538 = 0l;
        while (while_method_3(v1538)){
            int v1540;
            v1540 = 0l;
            while (while_method_1(v1540)){
                assert("Tensor range check" && 0 <= v1538 && v1538 < 1l);
                assert("Tensor range check" && 0 <= v1540 && v1540 < 4l);
                int v1542;
                v1542 = 4l * v1538;
                int v1543;
                v1543 = v1542 + v1540;
                int v1544;
                v1544 = v1521[v1543];
                bool v1545;
                v1545 = v1522[v1543];
                int v1552; bool v1553;
                if (v1537){
                    if (v1545){
                        bool v1546;
                        v1546 = v1536 < v1544;
                        int v1547;
                        if (v1546){
                            v1547 = v1536;
                        } else {
                            v1547 = v1544;
                        }
                        v1552 = v1547; v1553 = true;
                    } else {
                        v1552 = v1536; v1553 = v1537;
                    }
                } else {
                    if (v1545){
                        v1552 = v1544; v1553 = v1545;
                    } else {
                        v1552 = v1536; v1553 = v1537;
                    }
                }
                v1536 = v1552;
                v1537 = v1553;
                v1540 += 1l ;
            }
            v1538 += 1l ;
        }
        auto v1554 = cooperative_groups::coalesced_threads();
        int v1555;
        v1555 = threadIdx.x;
        int v1556;
        v1556 = v1555 / 32l;
        auto v1557 = cooperative_groups::labeled_partition(v1554,v1556);
        Closure7 v1558{};
        int v1559; bool v1560;
        Tuple4 tmp105 = cooperative_groups::reduce(v1557, Tuple4{v1536, v1537}, v1558);
        v1559 = tmp105.v0; v1560 = tmp105.v1;
        bool v1561;
        v1561 = v1560 == false;
        if (v1561){
            assert("The local reduce must be true." && v1560);
        } else {
        }
        assert("Tensor range check" && 0 <= v1275 && v1275 < 64l);
        int v1563;
        v1563 = 0l;
        while (while_method_3(v1563)){
            assert("Tensor range check" && 0 <= v1563 && v1563 < 1l);
            int v1565;
            v1565 = 128l * v1563;
            int v1566;
            v1566 = v1565 + v1284;
            assert("Tensor range check" && 0 <= v1563 && v1563 < 1l);
            int v1567;
            v1567 = 4l * v1563;
            int4* v1568;
            v1568 = reinterpret_cast<int4*>(v1414 + v1567);
            int4* v1569;
            v1569 = reinterpret_cast<int4*>(v13 + v1566);
            assert("Pointer alignment check" && (unsigned long long)(v1568) % 4l == 0 && (unsigned long long)(v1569) % 4l == 0);
            *v1569 = *v1568;
            v1563 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1275 && v1275 < 64l);
        v14[v1326] = v1559;
        v1275 += 24l ;
    }
    v15.sync() ;
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
import sys
import pathlib
def method1(v0 : i32) -> bool:
    v1 = v0 < 64
    del v0
    return v1
def method2(v0 : i32) -> bool:
    v1 = v0 < 128
    del v0
    return v1
def method0(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "input.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method1(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method2(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method3(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "input_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method1(v32):
        v34 = v30
        v35 = v34 >= 1024
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method2(v40):
            v42 = v30
            v43 = v42 >= 1024
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 128
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method5(v0 : i32) -> bool:
    v1 = v0 < 1
    del v0
    return v1
def method4(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v21 = 0
    v22 = "{}"
    print(v22.format('['),end="")
    v23 = 0
    while method5(v23):
        v25 = v21
        v26 = v25 >= 1024
        del v25
        if v26:
            v27 = " ..."
            print(v22.format(v27),end="")
            del v27
            break
        else:
            pass
        del v26
        v28 = v23 == 0
        v29 = v28 != True
        del v28
        if v29:
            v30 = "; "
            print(v22.format(v30),end="")
            del v30
        else:
            pass
        del v29
        v31 = v21 + 1
        v21 = v31
        del v31
        v32 = v0[v23].item()
        v33 = "{:.6f}"
        print(v33.format(v32),end="")
        del v32, v33
        v23 += 1 
    del v0, v21, v23
    print(v22.format(']'),end="")
    del v22
    v34 = "\n"
    print(v34,end="")
    del v34
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method6(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_softmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method1(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method2(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method7(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_masked_softmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method1(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method2(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method8(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_ln.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method1(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method2(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method9(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_argmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method1(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method10(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test2/a/"
    v4 = "output_softmax_scan.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method1(v35):
        v37 = v33
        v38 = v37 >= 8192
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method2(v43):
            v45 = v33
            v46 = v45 >= 8192
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 128
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{:.6f}, {:.6f}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method11(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method1(v32):
        v34 = v30
        v35 = v34 >= 8192
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method2(v40):
            v42 = v30
            v43 = v42 >= 8192
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 128
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method12(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test2/a/"
    v4 = "output_indices_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method1(v35):
        v37 = v33
        v38 = v37 >= 8192
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method2(v43):
            v45 = v33
            v46 = v45 >= 8192
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 128
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{}, {}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method13(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_indices_reduction.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method1(v22):
        v24 = v20
        v25 = v24 >= 8192
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method14(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_sum_exclusive.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method1(v32):
        v34 = v30
        v35 = v34 >= 8192
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method2(v40):
            v42 = v30
            v43 = v42 >= 8192
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 128
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method15(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_softmax'.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method1(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method2(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method16(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_sampling.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method1(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method17(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "input.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method1(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 64
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method18(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "input_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method2(v32):
        v34 = v30
        v35 = v34 >= 1024
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method1(v40):
            v42 = v30
            v43 = v42 >= 1024
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 64
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method19(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "output_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v21 = 0
    v22 = "{}"
    print(v22.format('['),end="")
    v23 = 0
    while method5(v23):
        v25 = v21
        v26 = v25 >= 1024
        del v25
        if v26:
            v27 = " ..."
            print(v22.format(v27),end="")
            del v27
            break
        else:
            pass
        del v26
        v28 = v23 == 0
        v29 = v28 != True
        del v28
        if v29:
            v30 = "; "
            print(v22.format(v30),end="")
            del v30
        else:
            pass
        del v29
        v31 = v21 + 1
        v21 = v31
        del v31
        v32 = v0[v23].item()
        v33 = "{:.6f}"
        print(v33.format(v32),end="")
        del v32, v33
        v23 += 1 
    del v0, v21, v23
    print(v22.format(']'),end="")
    del v22
    v34 = "\n"
    print(v34,end="")
    del v34
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method20(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "output_softmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method1(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 64
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method21(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "output_masked_softmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method1(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 64
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method22(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "output_ln.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method1(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 64
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method23(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "output_argmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method24(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test2/b/"
    v4 = "output_softmax_scan.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method2(v35):
        v37 = v33
        v38 = v37 >= 8192
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method1(v43):
            v45 = v33
            v46 = v45 >= 8192
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 64
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{:.6f}, {:.6f}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method25(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "output_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method2(v32):
        v34 = v30
        v35 = v34 >= 8192
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method1(v40):
            v42 = v30
            v43 = v42 >= 8192
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 64
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method26(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test2/b/"
    v4 = "output_indices_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method2(v35):
        v37 = v33
        v38 = v37 >= 8192
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method1(v43):
            v45 = v33
            v46 = v45 >= 8192
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 64
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{}, {}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method27(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "output_indices_reduction.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 8192
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method28(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "output_sum_exclusive.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method2(v32):
        v34 = v30
        v35 = v34 >= 8192
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method1(v40):
            v42 = v30
            v43 = v42 >= 8192
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 64
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method29(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "output_softmax'.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method1(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 64
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method30(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
    v3 = "output_sampling.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method32(v0 : i32) -> bool:
    v1 = v0 < 32
    del v0
    return v1
def method33(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def method31(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "input_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method32(v33):
        v35 = v31
        v36 = v35 >= 2147483647
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method33(v41):
            v43 = v31
            v44 = v43 >= 2147483647
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 16
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method34(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "output_sample_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method32(v22):
        v24 = v20
        v25 = v24 >= 2147483647
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method35(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/a"
    v4 = "output_indices_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method32(v35):
        v37 = v33
        v38 = v37 >= 2147483647
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method33(v43):
            v45 = v33
            v46 = v45 >= 2147483647
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 16
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{}, {}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method36(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "output_indices_map.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method32(v22):
        v24 = v20
        v25 = v24 >= 2147483647
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method37(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "output_op_map.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method32(v33):
        v35 = v31
        v36 = v35 >= 2147483647
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method33(v41):
            v43 = v31
            v44 = v43 >= 2147483647
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 16
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method38(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/a"
    v4 = "zip_input_output_identity_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method32(v35):
        v37 = v33
        v38 = v37 >= 2147483647
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method33(v43):
            v45 = v33
            v46 = v45 >= 2147483647
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 16
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{:.6f}, {:.6f}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method40(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method39(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "input_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method32(v33):
        v35 = v31
        v36 = v35 >= 2147483647
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method40(v41):
            v43 = v31
            v44 = v43 >= 2147483647
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 256
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method41(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "output_sample_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method32(v22):
        v24 = v20
        v25 = v24 >= 2147483647
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method42(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/b"
    v4 = "output_indices_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method32(v35):
        v37 = v33
        v38 = v37 >= 2147483647
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method40(v43):
            v45 = v33
            v46 = v45 >= 2147483647
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 256
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{}, {}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method43(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "output_indices_map.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method32(v22):
        v24 = v20
        v25 = v24 >= 2147483647
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method44(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "output_op_map.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method32(v33):
        v35 = v31
        v36 = v35 >= 2147483647
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method40(v41):
            v43 = v31
            v44 = v43 >= 2147483647
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 256
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method45(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/b"
    v4 = "zip_input_output_identity_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method32(v35):
        v37 = v33
        v38 = v37 >= 2147483647
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method40(v43):
            v45 = v33
            v46 = v45 >= 2147483647
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 256
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{:.6f}, {:.6f}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method46(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "input.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method1(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 64
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method47(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "input_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method2(v32):
        v34 = v30
        v35 = v34 >= 1024
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method1(v40):
            v42 = v30
            v43 = v42 >= 1024
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 64
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method48(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_softmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method1(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 64
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method49(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_masked_softmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method1(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 64
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method50(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_ln.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method1(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 64
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method51(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_argmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method52(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test4/b/"
    v4 = "output_softmax_scan.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method2(v35):
        v37 = v33
        v38 = v37 >= 8192
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method1(v43):
            v45 = v33
            v46 = v45 >= 8192
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 64
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{:.6f}, {:.6f}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method53(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method2(v32):
        v34 = v30
        v35 = v34 >= 8192
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method1(v40):
            v42 = v30
            v43 = v42 >= 8192
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 64
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method54(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test4/b/"
    v4 = "output_indices_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method2(v35):
        v37 = v33
        v38 = v37 >= 8192
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method1(v43):
            v45 = v33
            v46 = v45 >= 8192
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 64
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{}, {}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method55(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_indices_reduction.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 8192
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method56(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_sum_exclusive.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method2(v32):
        v34 = v30
        v35 = v34 >= 8192
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method1(v40):
            v42 = v30
            v43 = v42 >= 8192
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 64
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method57(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_softmax'.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method1(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 64
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method58(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_sampling.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method59(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "input.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method1(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method2(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method60(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "input_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method1(v32):
        v34 = v30
        v35 = v34 >= 1024
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method2(v40):
            v42 = v30
            v43 = v42 >= 1024
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 128
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method61(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_softmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method1(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method2(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method62(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_masked_softmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method1(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method2(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method63(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_ln.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method1(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method2(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method64(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_argmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method1(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method65(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test4/a/"
    v4 = "output_softmax_scan.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method1(v35):
        v37 = v33
        v38 = v37 >= 8192
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method2(v43):
            v45 = v33
            v46 = v45 >= 8192
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 128
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{:.6f}, {:.6f}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method66(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method1(v32):
        v34 = v30
        v35 = v34 >= 8192
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method2(v40):
            v42 = v30
            v43 = v42 >= 8192
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 128
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method67(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test4/a/"
    v4 = "output_indices_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method1(v35):
        v37 = v33
        v38 = v37 >= 8192
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method2(v43):
            v45 = v33
            v46 = v45 >= 8192
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 128
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{}, {}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57,end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method68(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_indices_reduction.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method1(v22):
        v24 = v20
        v25 = v24 >= 8192
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method69(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_sum_exclusive.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method1(v32):
        v34 = v30
        v35 = v34 >= 8192
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
            del v36
            break
        else:
            pass
        del v35
        v37 = v32 == 0
        v38 = v37 != True
        del v37
        if v38:
            v39 = "; "
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method2(v40):
            v42 = v30
            v43 = v42 >= 8192
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 128
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method70(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_softmax'.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method1(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method2(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method71(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_sampling.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method1(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
            del v26
            break
        else:
            pass
        del v25
        v27 = v22 == 0
        v28 = v27 != True
        del v27
        if v28:
            v29 = "; "
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32,end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def main_body():
    cp.random.seed(12344321)
    v0 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v1 = v0.size
    v2 = 8192 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v6 = cp.empty(1,dtype=cp.float32)
    v7 = cp.empty(8192,dtype=cp.int32)
    v8 = cp.empty(8192,dtype=cp.float32)
    v9 = cp.empty(8192,dtype=cp.float32)
    v10 = cp.empty(8192,dtype=cp.float32)
    v11 = cp.empty(8192,dtype=cp.float32)
    v12 = cp.empty(8192,dtype=cp.float32)
    v13 = cp.empty(64,dtype=cp.int32)
    v14 = cp.empty(8192,dtype=cp.int32)
    v15 = cp.empty(8192,dtype=cp.int32)
    v16 = cp.empty(64,dtype=cp.int32)
    v17 = cp.empty(8192,dtype=cp.int32)
    v18 = cp.empty(8192,dtype=cp.float32)
    v19 = cp.empty(64,dtype=cp.int32)
    v20 = cp.cuda.Device().attributes['MultiProcessorCount']
    v21 = v20 == 24
    del v20
    v22 = v21 == False
    if v22:
        v23 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v21, v23
        del v23
    else:
        pass
    del v21, v22
    v24 = 0
    v25 = raw_module.get_function(f"entry{v24}")
    del v24
    v25.max_dynamic_shared_size_bytes = 0 
    v25((24,),(32,),(v0, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19),shared_mem=0)
    del v25
    method0(v5)
    del v5
    method3(v0)
    del v0
    method4(v6)
    del v6
    method6(v8)
    del v8
    method7(v9)
    del v9
    method8(v12)
    del v12
    method9(v13)
    del v13
    method10(v10, v11)
    del v10, v11
    method11(v7)
    del v7
    method12(v14, v15)
    del v14, v15
    method13(v16)
    del v16
    method14(v17)
    del v17
    method15(v18)
    del v18
    method16(v19)
    del v19
    cp.random.seed(12344321)
    v26 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v27 = v26.size
    v28 = 8192 == v27
    del v27
    v29 = v28 == False
    if v29:
        v30 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v28, v30
        del v30
    else:
        pass
    del v28, v29
    v31 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v32 = cp.empty(1,dtype=cp.float32)
    v33 = cp.empty(8192,dtype=cp.int32)
    v34 = cp.empty(8192,dtype=cp.float32)
    v35 = cp.empty(8192,dtype=cp.float32)
    v36 = cp.empty(8192,dtype=cp.float32)
    v37 = cp.empty(8192,dtype=cp.float32)
    v38 = cp.empty(8192,dtype=cp.float32)
    v39 = cp.empty(128,dtype=cp.int32)
    v40 = cp.empty(8192,dtype=cp.int32)
    v41 = cp.empty(8192,dtype=cp.int32)
    v42 = cp.empty(128,dtype=cp.int32)
    v43 = cp.empty(8192,dtype=cp.int32)
    v44 = cp.empty(8192,dtype=cp.float32)
    v45 = cp.empty(128,dtype=cp.int32)
    v46 = cp.cuda.Device().attributes['MultiProcessorCount']
    v47 = v46 == 24
    del v46
    v48 = v47 == False
    if v48:
        v49 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v47, v49
        del v49
    else:
        pass
    del v47, v48
    v50 = 1
    v51 = raw_module.get_function(f"entry{v50}")
    del v50
    v51.max_dynamic_shared_size_bytes = 0 
    v51((24,),(32,),(v26, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45),shared_mem=0)
    del v51
    method17(v31)
    del v31
    method18(v26)
    del v26
    method19(v32)
    del v32
    method20(v34)
    del v34
    method21(v35)
    del v35
    method22(v38)
    del v38
    method23(v39)
    del v39
    method24(v36, v37)
    del v36, v37
    method25(v33)
    del v33
    method26(v40, v41)
    del v40, v41
    method27(v42)
    del v42
    method28(v43)
    del v43
    method29(v44)
    del v44
    method30(v45)
    del v45
    cp.random.seed(12344321)
    v52 = cp.arange(0,512,1,dtype=cp.float32) # type: ignore
    v53 = v52.size
    v54 = 512 == v53
    del v53
    v55 = v54 == False
    if v55:
        v56 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v54, v56
        del v56
    else:
        pass
    del v54, v55
    v57 = cp.random.normal(0.0,1.0,512,dtype=cp.float32) # type: ignore
    v58 = cp.empty(512,dtype=cp.int32)
    v59 = cp.empty(512,dtype=cp.int32)
    v60 = cp.empty(32,dtype=cp.int32)
    v61 = cp.empty(32,dtype=cp.int32)
    v62 = cp.empty(512,dtype=cp.float32)
    v63 = cp.empty(512,dtype=cp.float32)
    v64 = cp.cuda.Device().attributes['MultiProcessorCount']
    v65 = v64 == 24
    del v64
    v66 = v65 == False
    if v66:
        v67 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v65, v67
        del v67
    else:
        pass
    del v65, v66
    v68 = 2
    v69 = raw_module.get_function(f"entry{v68}")
    del v68
    v69.max_dynamic_shared_size_bytes = 0 
    v69((24,),(32,),(v52, v57, v58, v59, v60, v61, v62, v63),shared_mem=0)
    del v69
    method31(v52)
    del v52
    method34(v61)
    del v61
    method35(v58, v59)
    del v58, v59
    method36(v60)
    del v60
    method37(v63)
    del v63
    method38(v57, v62)
    del v57, v62
    cp.random.seed(12344321)
    v70 = cp.arange(0,8192,1,dtype=cp.float32) # type: ignore
    v71 = v70.size
    v72 = 8192 == v71
    del v71
    v73 = v72 == False
    if v73:
        v74 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v72, v74
        del v74
    else:
        pass
    del v72, v73
    v75 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v76 = cp.empty(8192,dtype=cp.int32)
    v77 = cp.empty(8192,dtype=cp.int32)
    v78 = cp.empty(32,dtype=cp.int32)
    v79 = cp.empty(32,dtype=cp.int32)
    v80 = cp.empty(8192,dtype=cp.float32)
    v81 = cp.empty(8192,dtype=cp.float32)
    v82 = cp.cuda.Device().attributes['MultiProcessorCount']
    v83 = v82 == 24
    del v82
    v84 = v83 == False
    if v84:
        v85 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v83, v85
        del v85
    else:
        pass
    del v83, v84
    v86 = 3
    v87 = raw_module.get_function(f"entry{v86}")
    del v86
    v87.max_dynamic_shared_size_bytes = 0 
    v87((24,),(32,),(v70, v75, v76, v77, v78, v79, v80, v81),shared_mem=0)
    del v87
    method39(v70)
    del v70
    method41(v79)
    del v79
    method42(v76, v77)
    del v76, v77
    method43(v78)
    del v78
    method44(v81)
    del v81
    method45(v75, v80)
    del v75, v80
    cp.random.seed(12344321)
    v88 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v89 = v88.size
    v90 = 8192 == v89
    del v89
    v91 = v90 == False
    if v91:
        v92 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v90, v92
        del v92
    else:
        pass
    del v90, v91
    v93 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v94 = cp.empty(8192,dtype=cp.int32)
    v95 = cp.empty(8192,dtype=cp.float32)
    v96 = cp.empty(8192,dtype=cp.float32)
    v97 = cp.empty(8192,dtype=cp.float32)
    v98 = cp.empty(8192,dtype=cp.float32)
    v99 = cp.empty(8192,dtype=cp.float32)
    v100 = cp.empty(128,dtype=cp.int32)
    v101 = cp.empty(8192,dtype=cp.int32)
    v102 = cp.empty(8192,dtype=cp.int32)
    v103 = cp.empty(128,dtype=cp.int32)
    v104 = cp.empty(8192,dtype=cp.int32)
    v105 = cp.empty(8192,dtype=cp.float32)
    v106 = cp.empty(128,dtype=cp.int32)
    v107 = cp.cuda.Device().attributes['MultiProcessorCount']
    v108 = v107 == 24
    del v107
    v109 = v108 == False
    if v109:
        v110 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v108, v110
        del v110
    else:
        pass
    del v108, v109
    v111 = 4
    v112 = raw_module.get_function(f"entry{v111}")
    del v111
    v112.max_dynamic_shared_size_bytes = 0 
    v112((24,),(32,),(v88, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106),shared_mem=0)
    del v112
    method46(v93)
    del v93
    method47(v88)
    del v88
    method48(v95)
    del v95
    method49(v96)
    del v96
    method50(v99)
    del v99
    method51(v100)
    del v100
    method52(v97, v98)
    del v97, v98
    method53(v94)
    del v94
    method54(v101, v102)
    del v101, v102
    method55(v103)
    del v103
    method56(v104)
    del v104
    method57(v105)
    del v105
    method58(v106)
    del v106
    cp.random.seed(12344321)
    v113 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v114 = v113.size
    v115 = 8192 == v114
    del v114
    v116 = v115 == False
    if v116:
        v117 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v115, v117
        del v117
    else:
        pass
    del v115, v116
    v118 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v119 = cp.empty(8192,dtype=cp.int32)
    v120 = cp.empty(8192,dtype=cp.float32)
    v121 = cp.empty(8192,dtype=cp.float32)
    v122 = cp.empty(8192,dtype=cp.float32)
    v123 = cp.empty(8192,dtype=cp.float32)
    v124 = cp.empty(8192,dtype=cp.float32)
    v125 = cp.empty(64,dtype=cp.int32)
    v126 = cp.empty(8192,dtype=cp.int32)
    v127 = cp.empty(8192,dtype=cp.int32)
    v128 = cp.empty(64,dtype=cp.int32)
    v129 = cp.empty(8192,dtype=cp.int32)
    v130 = cp.empty(8192,dtype=cp.float32)
    v131 = cp.empty(64,dtype=cp.int32)
    v132 = cp.cuda.Device().attributes['MultiProcessorCount']
    v133 = v132 == 24
    del v132
    v134 = v133 == False
    if v134:
        v135 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v133, v135
        del v135
    else:
        pass
    del v133, v134
    v136 = 5
    v137 = raw_module.get_function(f"entry{v136}")
    del v136
    v137.max_dynamic_shared_size_bytes = 0 
    v137((24,),(32,),(v113, v118, v119, v120, v121, v122, v123, v124, v125, v126, v127, v128, v129, v130, v131),shared_mem=0)
    del v137
    method59(v118)
    del v118
    method60(v113)
    del v113
    method61(v120)
    del v120
    method62(v121)
    del v121
    method63(v124)
    del v124
    method64(v125)
    del v125
    method65(v122, v123)
    del v122, v123
    method66(v119)
    del v119
    method67(v126, v127)
    del v126, v127
    method68(v128)
    del v128
    method69(v129)
    del v129
    method70(v130)
    del v130
    return method71(v131)

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
