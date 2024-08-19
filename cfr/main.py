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
struct Tuple5;
__device__ void push__0(int * v0, float * v1, int * v2, int * v3, double * v4, double * v5, float * v6, float * v7, int v8, int & v9, double * v10, double * v11, int v12, int v13, int v14, float v15, float v16);
struct Tuple6;
struct Tuple0 {
    float v0;
    float v1;
    int v2;
    __device__ Tuple0() = default;
    __device__ Tuple0(float t0, float t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple1 {
    float * v0;
    float * v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float * t0, float * t1) : v0(t0), v1(t1) {}
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
struct Tuple2 {
    int v0;
    float v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
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
struct Closure3 {
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
struct Tuple4 {
    float v0;
    int v1;
    __device__ Tuple4() = default;
    __device__ Tuple4(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple4 operator()(Tuple4 tup0, Tuple4 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple4{v0, v1};
        } else {
            return Tuple4{v2, v3};
        }
    }
};
struct Tuple5 {
    int v0;
    bool v1;
    __device__ Tuple5() = default;
    __device__ Tuple5(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
    __device__ Tuple5 operator()(Tuple5 tup0, Tuple5 tup1){
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
                return Tuple5{v5, true};
            } else {
                return Tuple5{v0, v1};
            }
        } else {
            if (v3){
                return Tuple5{v2, v3};
            } else {
                return Tuple5{v0, v1};
            }
        }
    }
};
struct Closure6 {
    int v0;
    __device__ Tuple4 operator()(Tuple4 tup0, Tuple4 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple4{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple4{v3, v4};
            } else {
                return Tuple4{v1, v2};
            }
        }
    }
    __device__ Closure6(int _v0) : v0(_v0) { }
};
struct Tuple6 {
    float * v0;
    float * v1;
    float * v2;
    __device__ Tuple6() = default;
    __device__ Tuple6(float * t0, float * t1, float * t2) : v0(t0), v1(t1), v2(t2) {}
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
__device__ void push__0(int * v0, float * v1, int * v2, int * v3, double * v4, double * v5, float * v6, float * v7, int v8, int & v9, double * v10, double * v11, int v12, int v13, int v14, float v15, float v16){
    int v17 = v9;
    int v18;
    v18 = v17 + 1l;
    v9 = v18;
    assert("Tensor range check" && 0 <= v17 && v17 < 16l);
    assert("Tensor range check" && 0 <= v8 && v8 < 32l);
    int v19;
    v19 = 32l * v17;
    int v20;
    v20 = v19 + v8;
    v0[v20] = v14;
    v1[v20] = v16;
    v2[v20] = v12;
    v3[v20] = v13;
    double v21;
    v21 = (double)v16;
    double v22;
    v22 = log(v21);
    double v23;
    v23 = (double)v15;
    double v24;
    v24 = log(v23);
    assert("Tensor range check" && 0 <= v12 && v12 < 2l);
    double v25;
    v25 = v10[v12];
    double v26;
    v26 = v11[v12];
    double v27;
    v27 = v24 + v25;
    double v28;
    v28 = v22 + v26;
    assert("Tensor range check" && 0 <= v12 && v12 < 2l);
    v10[v12] = v27;
    v11[v12] = v28;
    assert("Tensor range check" && 0 <= v17 && v17 < 16l);
    int v29;
    v29 = 64l * v17;
    assert("Tensor range check" && 0 <= v8 && v8 < 32l);
    int v30;
    v30 = 2l * v8;
    int v31;
    v31 = v30 + v29;
    int v32;
    v32 = 0l;
    while (while_method_0(v32)){
        assert("Tensor range check" && 0 <= v32 && v32 < 2l);
        double v34;
        v34 = v10[v32];
        double v35;
        v35 = v11[v32];
        assert("Tensor range check" && 0 <= v32 && v32 < 2l);
        int v36;
        v36 = v32 + v31;
        v4[v36] = v34;
        v5[v36] = v35;
        v32 += 1l ;
    }
    return ;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 > 0l;
    return v1;
}
extern "C" __global__ void entry0(int * v0, float * v1, int * v2, int * v3, double * v4, double * v5, float * v6, float * v7, float * v8, float * v9, float * v10, float * v11, float * v12) {
    unsigned long long v13;
    v13 = clock64();
    int v14;
    v14 = threadIdx.x;
    unsigned long long v15;
    v15 = (unsigned long long)v14;
    curandStatePhilox4_32_10_t v16;
    curand_init(v13,v15,0ull,&v16);
    int v17;
    v17 = threadIdx.x;
    int v18 = 0l;
    double v19[2l];
    double v20[2l];
    int v21;
    v21 = 0l;
    while (while_method_0(v21)){
        assert("Tensor range check" && 0 <= v21 && v21 < 2l);
        v19[v21] = 0.0;
        v20[v21] = 0.0;
        v21 += 1l ;
    }
    int v23;
    v23 = 235l;
    int v24;
    v24 = 0l;
    Tuple0 v25[1l];
    __shared__ Tuple1 v26[32l];
    __shared__ float v27[32l];
    __shared__ float v28[32l];
    __shared__ int v29[32l];
    int v30;
    v30 = threadIdx.x;
    float * v31;
    v31 = v8+940l;
    float * v33;
    v33 = v9+940l;
    assert("Tensor range check" && 0 <= v30 && v30 < 32l);
    v26[v30] = Tuple1{v31, v33};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v35;
    v35 = threadIdx.x;
    bool v36;
    v36 = 0l <= v35;
    bool v37;
    v37 = v36 == false;
    if (v37){
        assert("The index needs to be zero or positive." && v36);
    } else {
    }
    int v39;
    v39 = v35 % 1l;
    bool v40;
    v40 = v35 < 32l;
    bool v41;
    v41 = v40 == false;
    if (v41){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v40);
    } else {
    }
    assert("Tensor range check" && 0 <= v35 && v35 < 32l);
    assert("Tensor range check" && 0 <= v35 && v35 < 32l);
    int v43;
    v43 = 0l;
    while (while_method_1(v43)){
        assert("Tensor range check" && 0 <= v43 && v43 < 1l);
        int v45;
        v45 = v43 + v35;
        float * v46; float * v47;
        Tuple1 tmp0 = v26[v45];
        v46 = tmp0.v0; v47 = tmp0.v1;
        assert("Tensor range check" && 0 <= v39 && v39 < 1l);
        int v48;
        v48 = 4l * v39;
        float v49[4l];
        float v50[4l];
        int v51[4l];
        int v52;
        v52 = 0l;
        while (while_method_1(v52)){
            assert("Tensor range check" && 0 <= v52 && v52 < 1l);
            int v54;
            v54 = 4l * v52;
            assert("Tensor range check" && 0 <= v52 && v52 < 1l);
            int v55;
            v55 = v54 + v48;
            int4* v56;
            v56 = reinterpret_cast<int4*>(v46 + v55);
            int4* v57;
            v57 = reinterpret_cast<int4*>(v49 + v54);
            assert("Pointer alignment check" && (unsigned long long)(v56) % 4l == 0 && (unsigned long long)(v57) % 4l == 0);
            *v57 = *v56;
            int4* v58;
            v58 = reinterpret_cast<int4*>(v47 + v55);
            int4* v59;
            v59 = reinterpret_cast<int4*>(v50 + v54);
            assert("Pointer alignment check" && (unsigned long long)(v58) % 4l == 0 && (unsigned long long)(v59) % 4l == 0);
            *v59 = *v58;
            v52 += 1l ;
        }
        int v60;
        v60 = 0l;
        while (while_method_1(v60)){
            int v62;
            v62 = 0l;
            while (while_method_2(v62)){
                bool v64;
                v64 = 0l <= v62;
                bool v66;
                if (v64){
                    bool v65;
                    v65 = v62 < 4l;
                    v66 = v65;
                } else {
                    v66 = false;
                }
                bool v67;
                v67 = v66 == false;
                if (v67){
                    assert("The indices should be inside the range of the dimension." && v66);
                } else {
                }
                bool v69;
                v69 = 0l <= v39;
                bool v71;
                if (v69){
                    bool v70;
                    v70 = v39 < 1l;
                    v71 = v70;
                } else {
                    v71 = false;
                }
                bool v72;
                v72 = v71 == false;
                if (v72){
                    assert("The indices should be inside the range of the dimension." && v71);
                } else {
                }
                int v74;
                v74 = v39 * 4l;
                int v75;
                v75 = v62 + v74;
                bool v76;
                v76 = 0l <= v60;
                bool v78;
                if (v76){
                    bool v77;
                    v77 = v60 < 1l;
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
                v81 = v60 * 4l;
                int v82;
                v82 = v75 + v81;
                assert("Tensor range check" && 0 <= v60 && v60 < 1l);
                assert("Tensor range check" && 0 <= v62 && v62 < 4l);
                int v83;
                v83 = 4l * v60;
                int v84;
                v84 = v83 + v62;
                v51[v84] = v82;
                v62 += 1l ;
            }
            v60 += 1l ;
        }
        bool v85;
        v85 = 0l <= v43;
        bool v87;
        if (v85){
            bool v86;
            v86 = v43 < 1l;
            v87 = v86;
        } else {
            v87 = false;
        }
        bool v88;
        v88 = v87 == false;
        if (v88){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v87);
        } else {
        }
        bool v90;
        v90 = v36 && v40;
        bool v91;
        v91 = v90 == false;
        if (v91){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v90);
        } else {
        }
        int v93;
        v93 = v35 + v43;
        bool v94[4l];
        int v95;
        v95 = 0l;
        while (while_method_1(v95)){
            int v97;
            v97 = 0l;
            while (while_method_2(v97)){
                assert("Tensor range check" && 0 <= v95 && v95 < 1l);
                assert("Tensor range check" && 0 <= v97 && v97 < 4l);
                int v99;
                v99 = 4l * v95;
                int v100;
                v100 = v99 + v97;
                float v101;
                v101 = v49[v100];
                int v102;
                v102 = v51[v100];
                bool v103;
                v103 = v102 < 3l;
                assert("Tensor range check" && 0 <= v95 && v95 < 1l);
                assert("Tensor range check" && 0 <= v97 && v97 < 4l);
                v94[v100] = v103;
                v97 += 1l ;
            }
            v95 += 1l ;
        }
        float v104[4l];
        int v105;
        v105 = 0l;
        while (while_method_1(v105)){
            int v107;
            v107 = 0l;
            while (while_method_2(v107)){
                assert("Tensor range check" && 0 <= v105 && v105 < 1l);
                assert("Tensor range check" && 0 <= v107 && v107 < 4l);
                int v109;
                v109 = 4l * v105;
                int v110;
                v110 = v109 + v107;
                float v111;
                v111 = v49[v110];
                bool v112;
                v112 = v94[v110];
                float v115;
                if (v112){
                    bool v113;
                    v113 = 0.0f >= v111;
                    if (v113){
                        v115 = 0.0f;
                    } else {
                        v115 = v111;
                    }
                } else {
                    v115 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v105 && v105 < 1l);
                assert("Tensor range check" && 0 <= v107 && v107 < 4l);
                v104[v110] = v115;
                v107 += 1l ;
            }
            v105 += 1l ;
        }
        float v116;
        v116 = 0.0f;
        int v117;
        v117 = 0l;
        while (while_method_1(v117)){
            int v119;
            v119 = 0l;
            while (while_method_2(v119)){
                assert("Tensor range check" && 0 <= v117 && v117 < 1l);
                assert("Tensor range check" && 0 <= v119 && v119 < 4l);
                int v121;
                v121 = 4l * v117;
                int v122;
                v122 = v121 + v119;
                float v123;
                v123 = v104[v122];
                float v124;
                v124 = v116 + v123;
                v116 = v124;
                v119 += 1l ;
            }
            v117 += 1l ;
        }
        auto v125 = cooperative_groups::coalesced_threads();
        int v126;
        v126 = threadIdx.x;
        auto v127 = cooperative_groups::labeled_partition(v125,v126);
        Closure0 v128{};
        float v129;
        v129 = cooperative_groups::reduce(v127, v116, v128);
        int v130[4l];
        int v131;
        v131 = 0l;
        while (while_method_1(v131)){
            int v133;
            v133 = 0l;
            while (while_method_2(v133)){
                assert("Tensor range check" && 0 <= v131 && v131 < 1l);
                assert("Tensor range check" && 0 <= v133 && v133 < 4l);
                int v135;
                v135 = 4l * v131;
                int v136;
                v136 = v135 + v133;
                bool v137;
                v137 = v94[v136];
                int v138;
                if (v137){
                    v138 = 1l;
                } else {
                    v138 = 0l;
                }
                assert("Tensor range check" && 0 <= v131 && v131 < 1l);
                assert("Tensor range check" && 0 <= v133 && v133 < 4l);
                v130[v136] = v138;
                v133 += 1l ;
            }
            v131 += 1l ;
        }
        int v139;
        v139 = 0l;
        int v140;
        v140 = 0l;
        while (while_method_1(v140)){
            int v142;
            v142 = 0l;
            while (while_method_2(v142)){
                assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                assert("Tensor range check" && 0 <= v142 && v142 < 4l);
                int v144;
                v144 = 4l * v140;
                int v145;
                v145 = v144 + v142;
                int v146;
                v146 = v130[v145];
                int v147;
                v147 = v139 + v146;
                v139 = v147;
                v142 += 1l ;
            }
            v140 += 1l ;
        }
        auto v148 = cooperative_groups::coalesced_threads();
        int v149;
        v149 = threadIdx.x;
        auto v150 = cooperative_groups::labeled_partition(v148,v149);
        Closure1 v151{};
        int v152;
        v152 = cooperative_groups::reduce(v150, v139, v151);
        float v153;
        v153 = (float)v152;
        float v154;
        v154 = 1.0f / v153;
        float v155[4l];
        int v156;
        v156 = 0l;
        while (while_method_1(v156)){
            int v158;
            v158 = 0l;
            while (while_method_2(v158)){
                assert("Tensor range check" && 0 <= v156 && v156 < 1l);
                assert("Tensor range check" && 0 <= v158 && v158 < 4l);
                int v160;
                v160 = 4l * v156;
                int v161;
                v161 = v160 + v158;
                float v162;
                v162 = v104[v161];
                bool v163;
                v163 = v94[v161];
                bool v164;
                v164 = v163 == false;
                float v169;
                if (v164){
                    v169 = 0.0f;
                } else {
                    bool v165;
                    v165 = v129 == 0.0f;
                    bool v166;
                    v166 = v165 != true;
                    if (v166){
                        float v167;
                        v167 = v162 / v129;
                        v169 = v167;
                    } else {
                        v169 = v154;
                    }
                }
                assert("Tensor range check" && 0 <= v156 && v156 < 1l);
                assert("Tensor range check" && 0 <= v158 && v158 < 4l);
                v155[v161] = v169;
                v158 += 1l ;
            }
            v156 += 1l ;
        }
        float v170[4l];
        float v171;
        v171 = 0.0f;
        int v172;
        v172 = 0l;
        while (while_method_1(v172)){
            assert("Tensor range check" && 0 <= v172 && v172 < 1l);
            int v174;
            v174 = 4l * v172;
            assert("Tensor range check" && 0 <= v172 && v172 < 1l);
            int v175; float v176;
            Tuple2 tmp1 = Tuple2{0l, 0.0f};
            v175 = tmp1.v0; v176 = tmp1.v1;
            while (while_method_2(v175)){
                assert("Tensor range check" && 0 <= v175 && v175 < 4l);
                int v178;
                v178 = v175 + v174;
                float v179;
                v179 = v155[v178];
                float v180;
                v180 = v176 + v179;
                v176 = v180;
                v175 += 1l ;
            }
            auto v181 = cooperative_groups::coalesced_threads();
            int v182;
            v182 = threadIdx.x;
            auto v183 = cooperative_groups::labeled_partition(v181,v182);
            Closure2 v184{};
            float v185;
            v185 = cooperative_groups::inclusive_scan(v183, v176, v184);
            float v186;
            v186 = v183.shfl_up(v185,1);
            bool v187;
            v187 = v183.thread_rank() == 0;
            float v188;
            if (v187){
                v188 = 0.0f;
            } else {
                v188 = v186;
            }
            float v189;
            v189 = v183.shfl(v185,v183.num_threads()-1);
            float v190;
            v190 = v171 + v188;
            int v191; float v192;
            Tuple2 tmp2 = Tuple2{0l, v190};
            v191 = tmp2.v0; v192 = tmp2.v1;
            while (while_method_2(v191)){
                assert("Tensor range check" && 0 <= v191 && v191 < 4l);
                int v194;
                v194 = v191 + v174;
                float v195;
                v195 = v155[v194];
                float v196;
                v196 = v192 + v195;
                assert("Tensor range check" && 0 <= v191 && v191 < 4l);
                v170[v194] = v196;
                v192 = v196;
                v191 += 1l ;
            }
            float v197;
            v197 = v171 + v189;
            v171 = v197;
            v172 += 1l ;
        }
        float v198[4l];
        bool v199[4l];
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
                v206 = v170[v205];
                float v207;
                v207 = v155[v205];
                bool v208;
                v208 = v207 > 0.0f;
                assert("Tensor range check" && 0 <= v200 && v200 < 1l);
                assert("Tensor range check" && 0 <= v202 && v202 < 4l);
                v198[v205] = v206;
                v199[v205] = v208;
                v202 += 1l ;
            }
            v200 += 1l ;
        }
        float v209; bool v210;
        Tuple3 tmp3 = Tuple3{-1.0f / 0.0f, false};
        v209 = tmp3.v0; v210 = tmp3.v1;
        int v211;
        v211 = 0l;
        while (while_method_1(v211)){
            int v213;
            v213 = 0l;
            while (while_method_2(v213)){
                assert("Tensor range check" && 0 <= v211 && v211 < 1l);
                assert("Tensor range check" && 0 <= v213 && v213 < 4l);
                int v215;
                v215 = 4l * v211;
                int v216;
                v216 = v215 + v213;
                float v217;
                v217 = v198[v216];
                bool v218;
                v218 = v199[v216];
                float v225; bool v226;
                if (v210){
                    if (v218){
                        bool v219;
                        v219 = v209 >= v217;
                        float v220;
                        if (v219){
                            v220 = v209;
                        } else {
                            v220 = v217;
                        }
                        v225 = v220; v226 = true;
                    } else {
                        v225 = v209; v226 = v210;
                    }
                } else {
                    if (v218){
                        v225 = v217; v226 = v218;
                    } else {
                        v225 = v209; v226 = v210;
                    }
                }
                v209 = v225;
                v210 = v226;
                v213 += 1l ;
            }
            v211 += 1l ;
        }
        auto v227 = cooperative_groups::coalesced_threads();
        int v228;
        v228 = threadIdx.x;
        auto v229 = cooperative_groups::labeled_partition(v227,v228);
        Closure3 v230{};
        float v231; bool v232;
        Tuple3 tmp4 = cooperative_groups::reduce(v229, Tuple3{v209, v210}, v230);
        v231 = tmp4.v0; v232 = tmp4.v1;
        bool v233;
        v233 = v232 == false;
        if (v233){
            assert("The local reduce must be true." && v232);
        } else {
        }
        float v235[4l];
        int v236[4l];
        int v237;
        v237 = 0l;
        while (while_method_1(v237)){
            int v239;
            v239 = 0l;
            while (while_method_2(v239)){
                assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                assert("Tensor range check" && 0 <= v239 && v239 < 4l);
                int v241;
                v241 = 4l * v237;
                int v242;
                v242 = v241 + v239;
                int v243;
                v243 = v51[v242];
                float v244;
                v244 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                assert("Tensor range check" && 0 <= v239 && v239 < 4l);
                v235[v242] = v244;
                v236[v242] = v243;
                v239 += 1l ;
            }
            v237 += 1l ;
        }
        float v245; int v246;
        Tuple4 tmp5 = Tuple4{0.0f, 2147483647l};
        v245 = tmp5.v0; v246 = tmp5.v1;
        int v247;
        v247 = 0l;
        while (while_method_1(v247)){
            int v249;
            v249 = 0l;
            while (while_method_2(v249)){
                assert("Tensor range check" && 0 <= v247 && v247 < 1l);
                assert("Tensor range check" && 0 <= v249 && v249 < 4l);
                int v251;
                v251 = 4l * v247;
                int v252;
                v252 = v251 + v249;
                float v253;
                v253 = v235[v252];
                int v254;
                v254 = v236[v252];
                bool v255;
                v255 = v246 < v254;
                float v256; int v257;
                if (v255){
                    v256 = v245; v257 = v246;
                } else {
                    v256 = v253; v257 = v254;
                }
                v245 = v256;
                v246 = v257;
                v249 += 1l ;
            }
            v247 += 1l ;
        }
        auto v258 = cooperative_groups::coalesced_threads();
        int v259;
        v259 = threadIdx.x;
        auto v260 = cooperative_groups::labeled_partition(v258,v259);
        Closure4 v261{};
        float v262; int v263;
        Tuple4 tmp6 = cooperative_groups::reduce(v260, Tuple4{v245, v246}, v261);
        v262 = tmp6.v0; v263 = tmp6.v1;
        float v264;
        v264 = v231 * v262;
        int v265[4l];
        bool v266[4l];
        int v267;
        v267 = 0l;
        while (while_method_1(v267)){
            int v269;
            v269 = 0l;
            while (while_method_2(v269)){
                assert("Tensor range check" && 0 <= v267 && v267 < 1l);
                assert("Tensor range check" && 0 <= v269 && v269 < 4l);
                int v271;
                v271 = 4l * v267;
                int v272;
                v272 = v271 + v269;
                float v273;
                v273 = v198[v272];
                bool v274;
                v274 = v199[v272];
                int v275;
                v275 = v51[v272];
                int v278; bool v279;
                if (v274){
                    float v276;
                    v276 = v273 - v264;
                    bool v277;
                    v277 = v276 >= 0.0f;
                    v278 = v275; v279 = v277;
                } else {
                    v278 = 2147483647l; v279 = false;
                }
                assert("Tensor range check" && 0 <= v267 && v267 < 1l);
                assert("Tensor range check" && 0 <= v269 && v269 < 4l);
                v265[v272] = v278;
                v266[v272] = v279;
                v269 += 1l ;
            }
            v267 += 1l ;
        }
        int v280; bool v281;
        Tuple5 tmp7 = Tuple5{2147483647l, false};
        v280 = tmp7.v0; v281 = tmp7.v1;
        int v282;
        v282 = 0l;
        while (while_method_1(v282)){
            int v284;
            v284 = 0l;
            while (while_method_2(v284)){
                assert("Tensor range check" && 0 <= v282 && v282 < 1l);
                assert("Tensor range check" && 0 <= v284 && v284 < 4l);
                int v286;
                v286 = 4l * v282;
                int v287;
                v287 = v286 + v284;
                int v288;
                v288 = v265[v287];
                bool v289;
                v289 = v266[v287];
                int v296; bool v297;
                if (v281){
                    if (v289){
                        bool v290;
                        v290 = v280 < v288;
                        int v291;
                        if (v290){
                            v291 = v280;
                        } else {
                            v291 = v288;
                        }
                        v296 = v291; v297 = true;
                    } else {
                        v296 = v280; v297 = v281;
                    }
                } else {
                    if (v289){
                        v296 = v288; v297 = v289;
                    } else {
                        v296 = v280; v297 = v281;
                    }
                }
                v280 = v296;
                v281 = v297;
                v284 += 1l ;
            }
            v282 += 1l ;
        }
        auto v298 = cooperative_groups::coalesced_threads();
        int v299;
        v299 = threadIdx.x;
        auto v300 = cooperative_groups::labeled_partition(v298,v299);
        Closure5 v301{};
        int v302; bool v303;
        Tuple5 tmp8 = cooperative_groups::reduce(v300, Tuple5{v280, v281}, v301);
        v302 = tmp8.v0; v303 = tmp8.v1;
        bool v304;
        v304 = v303 == false;
        if (v304){
            assert("The local reduce must be true." && v303);
        } else {
        }
        bool v306[4l];
        int v307;
        v307 = 0l;
        while (while_method_1(v307)){
            int v309;
            v309 = 0l;
            while (while_method_2(v309)){
                assert("Tensor range check" && 0 <= v307 && v307 < 1l);
                assert("Tensor range check" && 0 <= v309 && v309 < 4l);
                int v311;
                v311 = 4l * v307;
                int v312;
                v312 = v311 + v309;
                float v313;
                v313 = v50[v312];
                int v314;
                v314 = v51[v312];
                bool v315;
                v315 = v314 < 3l;
                assert("Tensor range check" && 0 <= v307 && v307 < 1l);
                assert("Tensor range check" && 0 <= v309 && v309 < 4l);
                v306[v312] = v315;
                v309 += 1l ;
            }
            v307 += 1l ;
        }
        float v316[4l];
        int v317;
        v317 = 0l;
        while (while_method_1(v317)){
            int v319;
            v319 = 0l;
            while (while_method_2(v319)){
                assert("Tensor range check" && 0 <= v317 && v317 < 1l);
                assert("Tensor range check" && 0 <= v319 && v319 < 4l);
                int v321;
                v321 = 4l * v317;
                int v322;
                v322 = v321 + v319;
                float v323;
                v323 = v50[v322];
                bool v324;
                v324 = v306[v322];
                float v327;
                if (v324){
                    bool v325;
                    v325 = 0.0f >= v323;
                    if (v325){
                        v327 = 0.0f;
                    } else {
                        v327 = v323;
                    }
                } else {
                    v327 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v317 && v317 < 1l);
                assert("Tensor range check" && 0 <= v319 && v319 < 4l);
                v316[v322] = v327;
                v319 += 1l ;
            }
            v317 += 1l ;
        }
        float v328;
        v328 = 0.0f;
        int v329;
        v329 = 0l;
        while (while_method_1(v329)){
            int v331;
            v331 = 0l;
            while (while_method_2(v331)){
                assert("Tensor range check" && 0 <= v329 && v329 < 1l);
                assert("Tensor range check" && 0 <= v331 && v331 < 4l);
                int v333;
                v333 = 4l * v329;
                int v334;
                v334 = v333 + v331;
                float v335;
                v335 = v316[v334];
                float v336;
                v336 = v328 + v335;
                v328 = v336;
                v331 += 1l ;
            }
            v329 += 1l ;
        }
        auto v337 = cooperative_groups::coalesced_threads();
        int v338;
        v338 = threadIdx.x;
        auto v339 = cooperative_groups::labeled_partition(v337,v338);
        float v340;
        v340 = cooperative_groups::reduce(v339, v328, v128);
        int v341[4l];
        int v342;
        v342 = 0l;
        while (while_method_1(v342)){
            int v344;
            v344 = 0l;
            while (while_method_2(v344)){
                assert("Tensor range check" && 0 <= v342 && v342 < 1l);
                assert("Tensor range check" && 0 <= v344 && v344 < 4l);
                int v346;
                v346 = 4l * v342;
                int v347;
                v347 = v346 + v344;
                bool v348;
                v348 = v306[v347];
                int v349;
                if (v348){
                    v349 = 1l;
                } else {
                    v349 = 0l;
                }
                assert("Tensor range check" && 0 <= v342 && v342 < 1l);
                assert("Tensor range check" && 0 <= v344 && v344 < 4l);
                v341[v347] = v349;
                v344 += 1l ;
            }
            v342 += 1l ;
        }
        int v350;
        v350 = 0l;
        int v351;
        v351 = 0l;
        while (while_method_1(v351)){
            int v353;
            v353 = 0l;
            while (while_method_2(v353)){
                assert("Tensor range check" && 0 <= v351 && v351 < 1l);
                assert("Tensor range check" && 0 <= v353 && v353 < 4l);
                int v355;
                v355 = 4l * v351;
                int v356;
                v356 = v355 + v353;
                int v357;
                v357 = v341[v356];
                int v358;
                v358 = v350 + v357;
                v350 = v358;
                v353 += 1l ;
            }
            v351 += 1l ;
        }
        auto v359 = cooperative_groups::coalesced_threads();
        int v360;
        v360 = threadIdx.x;
        auto v361 = cooperative_groups::labeled_partition(v359,v360);
        int v362;
        v362 = cooperative_groups::reduce(v361, v350, v151);
        float v363;
        v363 = (float)v362;
        float v364;
        v364 = 1.0f / v363;
        float v365[4l];
        int v366;
        v366 = 0l;
        while (while_method_1(v366)){
            int v368;
            v368 = 0l;
            while (while_method_2(v368)){
                assert("Tensor range check" && 0 <= v366 && v366 < 1l);
                assert("Tensor range check" && 0 <= v368 && v368 < 4l);
                int v370;
                v370 = 4l * v366;
                int v371;
                v371 = v370 + v368;
                float v372;
                v372 = v316[v371];
                bool v373;
                v373 = v306[v371];
                bool v374;
                v374 = v373 == false;
                float v379;
                if (v374){
                    v379 = 0.0f;
                } else {
                    bool v375;
                    v375 = v340 == 0.0f;
                    bool v376;
                    v376 = v375 != true;
                    if (v376){
                        float v377;
                        v377 = v372 / v340;
                        v379 = v377;
                    } else {
                        v379 = v364;
                    }
                }
                assert("Tensor range check" && 0 <= v366 && v366 < 1l);
                assert("Tensor range check" && 0 <= v368 && v368 < 4l);
                v365[v371] = v379;
                v368 += 1l ;
            }
            v366 += 1l ;
        }
        float v380; int v381;
        Tuple4 tmp9 = Tuple4{0.0f, 2147483647l};
        v380 = tmp9.v0; v381 = tmp9.v1;
        int v382;
        v382 = 0l;
        while (while_method_1(v382)){
            int v384;
            v384 = 0l;
            while (while_method_2(v384)){
                assert("Tensor range check" && 0 <= v382 && v382 < 1l);
                assert("Tensor range check" && 0 <= v384 && v384 < 4l);
                int v386;
                v386 = 4l * v382;
                int v387;
                v387 = v386 + v384;
                float v388;
                v388 = v155[v387];
                int v389;
                v389 = v51[v387];
                bool v390;
                v390 = v381 == v302;
                float v394; int v395;
                if (v390){
                    v394 = v380; v395 = v381;
                } else {
                    bool v391;
                    v391 = v389 == v302;
                    if (v391){
                        v394 = v388; v395 = v389;
                    } else {
                        v394 = v380; v395 = v381;
                    }
                }
                v380 = v394;
                v381 = v395;
                v384 += 1l ;
            }
            v382 += 1l ;
        }
        auto v396 = cooperative_groups::coalesced_threads();
        int v397;
        v397 = threadIdx.x;
        auto v398 = cooperative_groups::labeled_partition(v396,v397);
        Closure6 v399{v302};
        float v400; int v401;
        Tuple4 tmp10 = cooperative_groups::reduce(v398, Tuple4{v380, v381}, v399);
        v400 = tmp10.v0; v401 = tmp10.v1;
        bool v402;
        v402 = v401 == 2147483647l;
        bool v403;
        v403 = v402 != true;
        bool v404;
        v404 = v403 == false;
        if (v404){
            assert("Expected a valid action id in get_action." && v403);
        } else {
        }
        float v406; int v407;
        Tuple4 tmp11 = Tuple4{0.0f, 2147483647l};
        v406 = tmp11.v0; v407 = tmp11.v1;
        int v408;
        v408 = 0l;
        while (while_method_1(v408)){
            int v410;
            v410 = 0l;
            while (while_method_2(v410)){
                assert("Tensor range check" && 0 <= v408 && v408 < 1l);
                assert("Tensor range check" && 0 <= v410 && v410 < 4l);
                int v412;
                v412 = 4l * v408;
                int v413;
                v413 = v412 + v410;
                float v414;
                v414 = v365[v413];
                int v415;
                v415 = v51[v413];
                bool v416;
                v416 = v407 == v302;
                float v420; int v421;
                if (v416){
                    v420 = v406; v421 = v407;
                } else {
                    bool v417;
                    v417 = v415 == v302;
                    if (v417){
                        v420 = v414; v421 = v415;
                    } else {
                        v420 = v406; v421 = v407;
                    }
                }
                v406 = v420;
                v407 = v421;
                v410 += 1l ;
            }
            v408 += 1l ;
        }
        auto v422 = cooperative_groups::coalesced_threads();
        int v423;
        v423 = threadIdx.x;
        auto v424 = cooperative_groups::labeled_partition(v422,v423);
        float v425; int v426;
        Tuple4 tmp12 = cooperative_groups::reduce(v424, Tuple4{v406, v407}, v399);
        v425 = tmp12.v0; v426 = tmp12.v1;
        bool v427;
        v427 = v426 == 2147483647l;
        bool v428;
        v428 = v427 != true;
        bool v429;
        v429 = v428 == false;
        if (v429){
            assert("Expected a valid action id in get_action." && v428);
        } else {
        }
        assert("Tensor range check" && 0 <= v43 && v43 < 1l);
        v27[v45] = v425;
        v28[v45] = v400;
        v29[v45] = v302;
        v43 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v431;
    v431 = threadIdx.x;
    assert("Tensor range check" && 0 <= v431 && v431 < 32l);
    float v432;
    v432 = v27[v431];
    float v433;
    v433 = v28[v431];
    int v434;
    v434 = v29[v431];
    v25[0l] = Tuple0{v432, v433, v434};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float v435; float v436; int v437;
    Tuple0 tmp13 = v25[0l];
    v435 = tmp13.v0; v436 = tmp13.v1; v437 = tmp13.v2;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v17, v18, v19, v20, v24, v23, v437, v435, v436);
    int v438;
    v438 = 212l;
    int v439;
    v439 = 1l;
    Tuple0 v440[1l];
    __shared__ Tuple1 v441[32l];
    __shared__ float v442[32l];
    __shared__ float v443[32l];
    __shared__ int v444[32l];
    int v445;
    v445 = threadIdx.x;
    float * v446;
    v446 = v8+848l;
    float * v448;
    v448 = v9+848l;
    assert("Tensor range check" && 0 <= v445 && v445 < 32l);
    v441[v445] = Tuple1{v446, v448};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v450;
    v450 = threadIdx.x;
    bool v451;
    v451 = 0l <= v450;
    bool v452;
    v452 = v451 == false;
    if (v452){
        assert("The index needs to be zero or positive." && v451);
    } else {
    }
    int v454;
    v454 = v450 % 1l;
    bool v455;
    v455 = v450 < 32l;
    bool v456;
    v456 = v455 == false;
    if (v456){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v455);
    } else {
    }
    assert("Tensor range check" && 0 <= v450 && v450 < 32l);
    assert("Tensor range check" && 0 <= v450 && v450 < 32l);
    int v458;
    v458 = 0l;
    while (while_method_1(v458)){
        assert("Tensor range check" && 0 <= v458 && v458 < 1l);
        int v460;
        v460 = v458 + v450;
        float * v461; float * v462;
        Tuple1 tmp14 = v441[v460];
        v461 = tmp14.v0; v462 = tmp14.v1;
        assert("Tensor range check" && 0 <= v454 && v454 < 1l);
        int v463;
        v463 = 4l * v454;
        float v464[4l];
        float v465[4l];
        int v466[4l];
        int v467;
        v467 = 0l;
        while (while_method_1(v467)){
            assert("Tensor range check" && 0 <= v467 && v467 < 1l);
            int v469;
            v469 = 4l * v467;
            assert("Tensor range check" && 0 <= v467 && v467 < 1l);
            int v470;
            v470 = v469 + v463;
            int4* v471;
            v471 = reinterpret_cast<int4*>(v461 + v470);
            int4* v472;
            v472 = reinterpret_cast<int4*>(v464 + v469);
            assert("Pointer alignment check" && (unsigned long long)(v471) % 4l == 0 && (unsigned long long)(v472) % 4l == 0);
            *v472 = *v471;
            int4* v473;
            v473 = reinterpret_cast<int4*>(v462 + v470);
            int4* v474;
            v474 = reinterpret_cast<int4*>(v465 + v469);
            assert("Pointer alignment check" && (unsigned long long)(v473) % 4l == 0 && (unsigned long long)(v474) % 4l == 0);
            *v474 = *v473;
            v467 += 1l ;
        }
        int v475;
        v475 = 0l;
        while (while_method_1(v475)){
            int v477;
            v477 = 0l;
            while (while_method_2(v477)){
                bool v479;
                v479 = 0l <= v477;
                bool v481;
                if (v479){
                    bool v480;
                    v480 = v477 < 4l;
                    v481 = v480;
                } else {
                    v481 = false;
                }
                bool v482;
                v482 = v481 == false;
                if (v482){
                    assert("The indices should be inside the range of the dimension." && v481);
                } else {
                }
                bool v484;
                v484 = 0l <= v454;
                bool v486;
                if (v484){
                    bool v485;
                    v485 = v454 < 1l;
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
                int v489;
                v489 = v454 * 4l;
                int v490;
                v490 = v477 + v489;
                bool v491;
                v491 = 0l <= v475;
                bool v493;
                if (v491){
                    bool v492;
                    v492 = v475 < 1l;
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
                v496 = v475 * 4l;
                int v497;
                v497 = v490 + v496;
                assert("Tensor range check" && 0 <= v475 && v475 < 1l);
                assert("Tensor range check" && 0 <= v477 && v477 < 4l);
                int v498;
                v498 = 4l * v475;
                int v499;
                v499 = v498 + v477;
                v466[v499] = v497;
                v477 += 1l ;
            }
            v475 += 1l ;
        }
        bool v500;
        v500 = 0l <= v458;
        bool v502;
        if (v500){
            bool v501;
            v501 = v458 < 1l;
            v502 = v501;
        } else {
            v502 = false;
        }
        bool v503;
        v503 = v502 == false;
        if (v503){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v502);
        } else {
        }
        bool v505;
        v505 = v451 && v455;
        bool v506;
        v506 = v505 == false;
        if (v506){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v505);
        } else {
        }
        int v508;
        v508 = v450 + v458;
        bool v509[4l];
        int v510;
        v510 = 0l;
        while (while_method_1(v510)){
            int v512;
            v512 = 0l;
            while (while_method_2(v512)){
                assert("Tensor range check" && 0 <= v510 && v510 < 1l);
                assert("Tensor range check" && 0 <= v512 && v512 < 4l);
                int v514;
                v514 = 4l * v510;
                int v515;
                v515 = v514 + v512;
                float v516;
                v516 = v464[v515];
                int v517;
                v517 = v466[v515];
                bool v518;
                v518 = v517 < 3l;
                assert("Tensor range check" && 0 <= v510 && v510 < 1l);
                assert("Tensor range check" && 0 <= v512 && v512 < 4l);
                v509[v515] = v518;
                v512 += 1l ;
            }
            v510 += 1l ;
        }
        float v519[4l];
        int v520;
        v520 = 0l;
        while (while_method_1(v520)){
            int v522;
            v522 = 0l;
            while (while_method_2(v522)){
                assert("Tensor range check" && 0 <= v520 && v520 < 1l);
                assert("Tensor range check" && 0 <= v522 && v522 < 4l);
                int v524;
                v524 = 4l * v520;
                int v525;
                v525 = v524 + v522;
                float v526;
                v526 = v464[v525];
                bool v527;
                v527 = v509[v525];
                float v530;
                if (v527){
                    bool v528;
                    v528 = 0.0f >= v526;
                    if (v528){
                        v530 = 0.0f;
                    } else {
                        v530 = v526;
                    }
                } else {
                    v530 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v520 && v520 < 1l);
                assert("Tensor range check" && 0 <= v522 && v522 < 4l);
                v519[v525] = v530;
                v522 += 1l ;
            }
            v520 += 1l ;
        }
        float v531;
        v531 = 0.0f;
        int v532;
        v532 = 0l;
        while (while_method_1(v532)){
            int v534;
            v534 = 0l;
            while (while_method_2(v534)){
                assert("Tensor range check" && 0 <= v532 && v532 < 1l);
                assert("Tensor range check" && 0 <= v534 && v534 < 4l);
                int v536;
                v536 = 4l * v532;
                int v537;
                v537 = v536 + v534;
                float v538;
                v538 = v519[v537];
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
        auto v542 = cooperative_groups::labeled_partition(v540,v541);
        Closure0 v543{};
        float v544;
        v544 = cooperative_groups::reduce(v542, v531, v543);
        int v545[4l];
        int v546;
        v546 = 0l;
        while (while_method_1(v546)){
            int v548;
            v548 = 0l;
            while (while_method_2(v548)){
                assert("Tensor range check" && 0 <= v546 && v546 < 1l);
                assert("Tensor range check" && 0 <= v548 && v548 < 4l);
                int v550;
                v550 = 4l * v546;
                int v551;
                v551 = v550 + v548;
                bool v552;
                v552 = v509[v551];
                int v553;
                if (v552){
                    v553 = 1l;
                } else {
                    v553 = 0l;
                }
                assert("Tensor range check" && 0 <= v546 && v546 < 1l);
                assert("Tensor range check" && 0 <= v548 && v548 < 4l);
                v545[v551] = v553;
                v548 += 1l ;
            }
            v546 += 1l ;
        }
        int v554;
        v554 = 0l;
        int v555;
        v555 = 0l;
        while (while_method_1(v555)){
            int v557;
            v557 = 0l;
            while (while_method_2(v557)){
                assert("Tensor range check" && 0 <= v555 && v555 < 1l);
                assert("Tensor range check" && 0 <= v557 && v557 < 4l);
                int v559;
                v559 = 4l * v555;
                int v560;
                v560 = v559 + v557;
                int v561;
                v561 = v545[v560];
                int v562;
                v562 = v554 + v561;
                v554 = v562;
                v557 += 1l ;
            }
            v555 += 1l ;
        }
        auto v563 = cooperative_groups::coalesced_threads();
        int v564;
        v564 = threadIdx.x;
        auto v565 = cooperative_groups::labeled_partition(v563,v564);
        Closure1 v566{};
        int v567;
        v567 = cooperative_groups::reduce(v565, v554, v566);
        float v568;
        v568 = (float)v567;
        float v569;
        v569 = 1.0f / v568;
        float v570[4l];
        int v571;
        v571 = 0l;
        while (while_method_1(v571)){
            int v573;
            v573 = 0l;
            while (while_method_2(v573)){
                assert("Tensor range check" && 0 <= v571 && v571 < 1l);
                assert("Tensor range check" && 0 <= v573 && v573 < 4l);
                int v575;
                v575 = 4l * v571;
                int v576;
                v576 = v575 + v573;
                float v577;
                v577 = v519[v576];
                bool v578;
                v578 = v509[v576];
                bool v579;
                v579 = v578 == false;
                float v584;
                if (v579){
                    v584 = 0.0f;
                } else {
                    bool v580;
                    v580 = v544 == 0.0f;
                    bool v581;
                    v581 = v580 != true;
                    if (v581){
                        float v582;
                        v582 = v577 / v544;
                        v584 = v582;
                    } else {
                        v584 = v569;
                    }
                }
                assert("Tensor range check" && 0 <= v571 && v571 < 1l);
                assert("Tensor range check" && 0 <= v573 && v573 < 4l);
                v570[v576] = v584;
                v573 += 1l ;
            }
            v571 += 1l ;
        }
        float v585[4l];
        float v586;
        v586 = 0.0f;
        int v587;
        v587 = 0l;
        while (while_method_1(v587)){
            assert("Tensor range check" && 0 <= v587 && v587 < 1l);
            int v589;
            v589 = 4l * v587;
            assert("Tensor range check" && 0 <= v587 && v587 < 1l);
            int v590; float v591;
            Tuple2 tmp15 = Tuple2{0l, 0.0f};
            v590 = tmp15.v0; v591 = tmp15.v1;
            while (while_method_2(v590)){
                assert("Tensor range check" && 0 <= v590 && v590 < 4l);
                int v593;
                v593 = v590 + v589;
                float v594;
                v594 = v570[v593];
                float v595;
                v595 = v591 + v594;
                v591 = v595;
                v590 += 1l ;
            }
            auto v596 = cooperative_groups::coalesced_threads();
            int v597;
            v597 = threadIdx.x;
            auto v598 = cooperative_groups::labeled_partition(v596,v597);
            Closure2 v599{};
            float v600;
            v600 = cooperative_groups::inclusive_scan(v598, v591, v599);
            float v601;
            v601 = v598.shfl_up(v600,1);
            bool v602;
            v602 = v598.thread_rank() == 0;
            float v603;
            if (v602){
                v603 = 0.0f;
            } else {
                v603 = v601;
            }
            float v604;
            v604 = v598.shfl(v600,v598.num_threads()-1);
            float v605;
            v605 = v586 + v603;
            int v606; float v607;
            Tuple2 tmp16 = Tuple2{0l, v605};
            v606 = tmp16.v0; v607 = tmp16.v1;
            while (while_method_2(v606)){
                assert("Tensor range check" && 0 <= v606 && v606 < 4l);
                int v609;
                v609 = v606 + v589;
                float v610;
                v610 = v570[v609];
                float v611;
                v611 = v607 + v610;
                assert("Tensor range check" && 0 <= v606 && v606 < 4l);
                v585[v609] = v611;
                v607 = v611;
                v606 += 1l ;
            }
            float v612;
            v612 = v586 + v604;
            v586 = v612;
            v587 += 1l ;
        }
        float v613[4l];
        bool v614[4l];
        int v615;
        v615 = 0l;
        while (while_method_1(v615)){
            int v617;
            v617 = 0l;
            while (while_method_2(v617)){
                assert("Tensor range check" && 0 <= v615 && v615 < 1l);
                assert("Tensor range check" && 0 <= v617 && v617 < 4l);
                int v619;
                v619 = 4l * v615;
                int v620;
                v620 = v619 + v617;
                float v621;
                v621 = v585[v620];
                float v622;
                v622 = v570[v620];
                bool v623;
                v623 = v622 > 0.0f;
                assert("Tensor range check" && 0 <= v615 && v615 < 1l);
                assert("Tensor range check" && 0 <= v617 && v617 < 4l);
                v613[v620] = v621;
                v614[v620] = v623;
                v617 += 1l ;
            }
            v615 += 1l ;
        }
        float v624; bool v625;
        Tuple3 tmp17 = Tuple3{-1.0f / 0.0f, false};
        v624 = tmp17.v0; v625 = tmp17.v1;
        int v626;
        v626 = 0l;
        while (while_method_1(v626)){
            int v628;
            v628 = 0l;
            while (while_method_2(v628)){
                assert("Tensor range check" && 0 <= v626 && v626 < 1l);
                assert("Tensor range check" && 0 <= v628 && v628 < 4l);
                int v630;
                v630 = 4l * v626;
                int v631;
                v631 = v630 + v628;
                float v632;
                v632 = v613[v631];
                bool v633;
                v633 = v614[v631];
                float v640; bool v641;
                if (v625){
                    if (v633){
                        bool v634;
                        v634 = v624 >= v632;
                        float v635;
                        if (v634){
                            v635 = v624;
                        } else {
                            v635 = v632;
                        }
                        v640 = v635; v641 = true;
                    } else {
                        v640 = v624; v641 = v625;
                    }
                } else {
                    if (v633){
                        v640 = v632; v641 = v633;
                    } else {
                        v640 = v624; v641 = v625;
                    }
                }
                v624 = v640;
                v625 = v641;
                v628 += 1l ;
            }
            v626 += 1l ;
        }
        auto v642 = cooperative_groups::coalesced_threads();
        int v643;
        v643 = threadIdx.x;
        auto v644 = cooperative_groups::labeled_partition(v642,v643);
        Closure3 v645{};
        float v646; bool v647;
        Tuple3 tmp18 = cooperative_groups::reduce(v644, Tuple3{v624, v625}, v645);
        v646 = tmp18.v0; v647 = tmp18.v1;
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
        while (while_method_1(v652)){
            int v654;
            v654 = 0l;
            while (while_method_2(v654)){
                assert("Tensor range check" && 0 <= v652 && v652 < 1l);
                assert("Tensor range check" && 0 <= v654 && v654 < 4l);
                int v656;
                v656 = 4l * v652;
                int v657;
                v657 = v656 + v654;
                int v658;
                v658 = v466[v657];
                float v659;
                v659 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v652 && v652 < 1l);
                assert("Tensor range check" && 0 <= v654 && v654 < 4l);
                v650[v657] = v659;
                v651[v657] = v658;
                v654 += 1l ;
            }
            v652 += 1l ;
        }
        float v660; int v661;
        Tuple4 tmp19 = Tuple4{0.0f, 2147483647l};
        v660 = tmp19.v0; v661 = tmp19.v1;
        int v662;
        v662 = 0l;
        while (while_method_1(v662)){
            int v664;
            v664 = 0l;
            while (while_method_2(v664)){
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
        auto v675 = cooperative_groups::labeled_partition(v673,v674);
        Closure4 v676{};
        float v677; int v678;
        Tuple4 tmp20 = cooperative_groups::reduce(v675, Tuple4{v660, v661}, v676);
        v677 = tmp20.v0; v678 = tmp20.v1;
        float v679;
        v679 = v646 * v677;
        int v680[4l];
        bool v681[4l];
        int v682;
        v682 = 0l;
        while (while_method_1(v682)){
            int v684;
            v684 = 0l;
            while (while_method_2(v684)){
                assert("Tensor range check" && 0 <= v682 && v682 < 1l);
                assert("Tensor range check" && 0 <= v684 && v684 < 4l);
                int v686;
                v686 = 4l * v682;
                int v687;
                v687 = v686 + v684;
                float v688;
                v688 = v613[v687];
                bool v689;
                v689 = v614[v687];
                int v690;
                v690 = v466[v687];
                int v693; bool v694;
                if (v689){
                    float v691;
                    v691 = v688 - v679;
                    bool v692;
                    v692 = v691 >= 0.0f;
                    v693 = v690; v694 = v692;
                } else {
                    v693 = 2147483647l; v694 = false;
                }
                assert("Tensor range check" && 0 <= v682 && v682 < 1l);
                assert("Tensor range check" && 0 <= v684 && v684 < 4l);
                v680[v687] = v693;
                v681[v687] = v694;
                v684 += 1l ;
            }
            v682 += 1l ;
        }
        int v695; bool v696;
        Tuple5 tmp21 = Tuple5{2147483647l, false};
        v695 = tmp21.v0; v696 = tmp21.v1;
        int v697;
        v697 = 0l;
        while (while_method_1(v697)){
            int v699;
            v699 = 0l;
            while (while_method_2(v699)){
                assert("Tensor range check" && 0 <= v697 && v697 < 1l);
                assert("Tensor range check" && 0 <= v699 && v699 < 4l);
                int v701;
                v701 = 4l * v697;
                int v702;
                v702 = v701 + v699;
                int v703;
                v703 = v680[v702];
                bool v704;
                v704 = v681[v702];
                int v711; bool v712;
                if (v696){
                    if (v704){
                        bool v705;
                        v705 = v695 < v703;
                        int v706;
                        if (v705){
                            v706 = v695;
                        } else {
                            v706 = v703;
                        }
                        v711 = v706; v712 = true;
                    } else {
                        v711 = v695; v712 = v696;
                    }
                } else {
                    if (v704){
                        v711 = v703; v712 = v704;
                    } else {
                        v711 = v695; v712 = v696;
                    }
                }
                v695 = v711;
                v696 = v712;
                v699 += 1l ;
            }
            v697 += 1l ;
        }
        auto v713 = cooperative_groups::coalesced_threads();
        int v714;
        v714 = threadIdx.x;
        auto v715 = cooperative_groups::labeled_partition(v713,v714);
        Closure5 v716{};
        int v717; bool v718;
        Tuple5 tmp22 = cooperative_groups::reduce(v715, Tuple5{v695, v696}, v716);
        v717 = tmp22.v0; v718 = tmp22.v1;
        bool v719;
        v719 = v718 == false;
        if (v719){
            assert("The local reduce must be true." && v718);
        } else {
        }
        bool v721[4l];
        int v722;
        v722 = 0l;
        while (while_method_1(v722)){
            int v724;
            v724 = 0l;
            while (while_method_2(v724)){
                assert("Tensor range check" && 0 <= v722 && v722 < 1l);
                assert("Tensor range check" && 0 <= v724 && v724 < 4l);
                int v726;
                v726 = 4l * v722;
                int v727;
                v727 = v726 + v724;
                float v728;
                v728 = v465[v727];
                int v729;
                v729 = v466[v727];
                bool v730;
                v730 = v729 < 3l;
                assert("Tensor range check" && 0 <= v722 && v722 < 1l);
                assert("Tensor range check" && 0 <= v724 && v724 < 4l);
                v721[v727] = v730;
                v724 += 1l ;
            }
            v722 += 1l ;
        }
        float v731[4l];
        int v732;
        v732 = 0l;
        while (while_method_1(v732)){
            int v734;
            v734 = 0l;
            while (while_method_2(v734)){
                assert("Tensor range check" && 0 <= v732 && v732 < 1l);
                assert("Tensor range check" && 0 <= v734 && v734 < 4l);
                int v736;
                v736 = 4l * v732;
                int v737;
                v737 = v736 + v734;
                float v738;
                v738 = v465[v737];
                bool v739;
                v739 = v721[v737];
                float v742;
                if (v739){
                    bool v740;
                    v740 = 0.0f >= v738;
                    if (v740){
                        v742 = 0.0f;
                    } else {
                        v742 = v738;
                    }
                } else {
                    v742 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v732 && v732 < 1l);
                assert("Tensor range check" && 0 <= v734 && v734 < 4l);
                v731[v737] = v742;
                v734 += 1l ;
            }
            v732 += 1l ;
        }
        float v743;
        v743 = 0.0f;
        int v744;
        v744 = 0l;
        while (while_method_1(v744)){
            int v746;
            v746 = 0l;
            while (while_method_2(v746)){
                assert("Tensor range check" && 0 <= v744 && v744 < 1l);
                assert("Tensor range check" && 0 <= v746 && v746 < 4l);
                int v748;
                v748 = 4l * v744;
                int v749;
                v749 = v748 + v746;
                float v750;
                v750 = v731[v749];
                float v751;
                v751 = v743 + v750;
                v743 = v751;
                v746 += 1l ;
            }
            v744 += 1l ;
        }
        auto v752 = cooperative_groups::coalesced_threads();
        int v753;
        v753 = threadIdx.x;
        auto v754 = cooperative_groups::labeled_partition(v752,v753);
        float v755;
        v755 = cooperative_groups::reduce(v754, v743, v543);
        int v756[4l];
        int v757;
        v757 = 0l;
        while (while_method_1(v757)){
            int v759;
            v759 = 0l;
            while (while_method_2(v759)){
                assert("Tensor range check" && 0 <= v757 && v757 < 1l);
                assert("Tensor range check" && 0 <= v759 && v759 < 4l);
                int v761;
                v761 = 4l * v757;
                int v762;
                v762 = v761 + v759;
                bool v763;
                v763 = v721[v762];
                int v764;
                if (v763){
                    v764 = 1l;
                } else {
                    v764 = 0l;
                }
                assert("Tensor range check" && 0 <= v757 && v757 < 1l);
                assert("Tensor range check" && 0 <= v759 && v759 < 4l);
                v756[v762] = v764;
                v759 += 1l ;
            }
            v757 += 1l ;
        }
        int v765;
        v765 = 0l;
        int v766;
        v766 = 0l;
        while (while_method_1(v766)){
            int v768;
            v768 = 0l;
            while (while_method_2(v768)){
                assert("Tensor range check" && 0 <= v766 && v766 < 1l);
                assert("Tensor range check" && 0 <= v768 && v768 < 4l);
                int v770;
                v770 = 4l * v766;
                int v771;
                v771 = v770 + v768;
                int v772;
                v772 = v756[v771];
                int v773;
                v773 = v765 + v772;
                v765 = v773;
                v768 += 1l ;
            }
            v766 += 1l ;
        }
        auto v774 = cooperative_groups::coalesced_threads();
        int v775;
        v775 = threadIdx.x;
        auto v776 = cooperative_groups::labeled_partition(v774,v775);
        int v777;
        v777 = cooperative_groups::reduce(v776, v765, v566);
        float v778;
        v778 = (float)v777;
        float v779;
        v779 = 1.0f / v778;
        float v780[4l];
        int v781;
        v781 = 0l;
        while (while_method_1(v781)){
            int v783;
            v783 = 0l;
            while (while_method_2(v783)){
                assert("Tensor range check" && 0 <= v781 && v781 < 1l);
                assert("Tensor range check" && 0 <= v783 && v783 < 4l);
                int v785;
                v785 = 4l * v781;
                int v786;
                v786 = v785 + v783;
                float v787;
                v787 = v731[v786];
                bool v788;
                v788 = v721[v786];
                bool v789;
                v789 = v788 == false;
                float v794;
                if (v789){
                    v794 = 0.0f;
                } else {
                    bool v790;
                    v790 = v755 == 0.0f;
                    bool v791;
                    v791 = v790 != true;
                    if (v791){
                        float v792;
                        v792 = v787 / v755;
                        v794 = v792;
                    } else {
                        v794 = v779;
                    }
                }
                assert("Tensor range check" && 0 <= v781 && v781 < 1l);
                assert("Tensor range check" && 0 <= v783 && v783 < 4l);
                v780[v786] = v794;
                v783 += 1l ;
            }
            v781 += 1l ;
        }
        float v795; int v796;
        Tuple4 tmp23 = Tuple4{0.0f, 2147483647l};
        v795 = tmp23.v0; v796 = tmp23.v1;
        int v797;
        v797 = 0l;
        while (while_method_1(v797)){
            int v799;
            v799 = 0l;
            while (while_method_2(v799)){
                assert("Tensor range check" && 0 <= v797 && v797 < 1l);
                assert("Tensor range check" && 0 <= v799 && v799 < 4l);
                int v801;
                v801 = 4l * v797;
                int v802;
                v802 = v801 + v799;
                float v803;
                v803 = v570[v802];
                int v804;
                v804 = v466[v802];
                bool v805;
                v805 = v796 == v717;
                float v809; int v810;
                if (v805){
                    v809 = v795; v810 = v796;
                } else {
                    bool v806;
                    v806 = v804 == v717;
                    if (v806){
                        v809 = v803; v810 = v804;
                    } else {
                        v809 = v795; v810 = v796;
                    }
                }
                v795 = v809;
                v796 = v810;
                v799 += 1l ;
            }
            v797 += 1l ;
        }
        auto v811 = cooperative_groups::coalesced_threads();
        int v812;
        v812 = threadIdx.x;
        auto v813 = cooperative_groups::labeled_partition(v811,v812);
        Closure6 v814{v717};
        float v815; int v816;
        Tuple4 tmp24 = cooperative_groups::reduce(v813, Tuple4{v795, v796}, v814);
        v815 = tmp24.v0; v816 = tmp24.v1;
        bool v817;
        v817 = v816 == 2147483647l;
        bool v818;
        v818 = v817 != true;
        bool v819;
        v819 = v818 == false;
        if (v819){
            assert("Expected a valid action id in get_action." && v818);
        } else {
        }
        float v821; int v822;
        Tuple4 tmp25 = Tuple4{0.0f, 2147483647l};
        v821 = tmp25.v0; v822 = tmp25.v1;
        int v823;
        v823 = 0l;
        while (while_method_1(v823)){
            int v825;
            v825 = 0l;
            while (while_method_2(v825)){
                assert("Tensor range check" && 0 <= v823 && v823 < 1l);
                assert("Tensor range check" && 0 <= v825 && v825 < 4l);
                int v827;
                v827 = 4l * v823;
                int v828;
                v828 = v827 + v825;
                float v829;
                v829 = v780[v828];
                int v830;
                v830 = v466[v828];
                bool v831;
                v831 = v822 == v717;
                float v835; int v836;
                if (v831){
                    v835 = v821; v836 = v822;
                } else {
                    bool v832;
                    v832 = v830 == v717;
                    if (v832){
                        v835 = v829; v836 = v830;
                    } else {
                        v835 = v821; v836 = v822;
                    }
                }
                v821 = v835;
                v822 = v836;
                v825 += 1l ;
            }
            v823 += 1l ;
        }
        auto v837 = cooperative_groups::coalesced_threads();
        int v838;
        v838 = threadIdx.x;
        auto v839 = cooperative_groups::labeled_partition(v837,v838);
        float v840; int v841;
        Tuple4 tmp26 = cooperative_groups::reduce(v839, Tuple4{v821, v822}, v814);
        v840 = tmp26.v0; v841 = tmp26.v1;
        bool v842;
        v842 = v841 == 2147483647l;
        bool v843;
        v843 = v842 != true;
        bool v844;
        v844 = v843 == false;
        if (v844){
            assert("Expected a valid action id in get_action." && v843);
        } else {
        }
        assert("Tensor range check" && 0 <= v458 && v458 < 1l);
        v442[v460] = v840;
        v443[v460] = v815;
        v444[v460] = v717;
        v458 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v846;
    v846 = threadIdx.x;
    assert("Tensor range check" && 0 <= v846 && v846 < 32l);
    float v847;
    v847 = v442[v846];
    float v848;
    v848 = v443[v846];
    int v849;
    v849 = v444[v846];
    v440[0l] = Tuple0{v847, v848, v849};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float v850; float v851; int v852;
    Tuple0 tmp27 = v440[0l];
    v850 = tmp27.v0; v851 = tmp27.v1; v852 = tmp27.v2;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v17, v18, v19, v20, v439, v438, v852, v850, v851);
    int v853;
    v853 = 790l;
    int v854;
    v854 = 0l;
    Tuple0 v855[1l];
    __shared__ Tuple1 v856[32l];
    __shared__ float v857[32l];
    __shared__ float v858[32l];
    __shared__ int v859[32l];
    int v860;
    v860 = threadIdx.x;
    float * v861;
    v861 = v8+3160l;
    float * v863;
    v863 = v9+3160l;
    assert("Tensor range check" && 0 <= v860 && v860 < 32l);
    v856[v860] = Tuple1{v861, v863};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v865;
    v865 = threadIdx.x;
    bool v866;
    v866 = 0l <= v865;
    bool v867;
    v867 = v866 == false;
    if (v867){
        assert("The index needs to be zero or positive." && v866);
    } else {
    }
    int v869;
    v869 = v865 % 1l;
    bool v870;
    v870 = v865 < 32l;
    bool v871;
    v871 = v870 == false;
    if (v871){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v870);
    } else {
    }
    assert("Tensor range check" && 0 <= v865 && v865 < 32l);
    assert("Tensor range check" && 0 <= v865 && v865 < 32l);
    int v873;
    v873 = 0l;
    while (while_method_1(v873)){
        assert("Tensor range check" && 0 <= v873 && v873 < 1l);
        int v875;
        v875 = v873 + v865;
        float * v876; float * v877;
        Tuple1 tmp28 = v856[v875];
        v876 = tmp28.v0; v877 = tmp28.v1;
        assert("Tensor range check" && 0 <= v869 && v869 < 1l);
        int v878;
        v878 = 4l * v869;
        float v879[4l];
        float v880[4l];
        int v881[4l];
        int v882;
        v882 = 0l;
        while (while_method_1(v882)){
            assert("Tensor range check" && 0 <= v882 && v882 < 1l);
            int v884;
            v884 = 4l * v882;
            assert("Tensor range check" && 0 <= v882 && v882 < 1l);
            int v885;
            v885 = v884 + v878;
            int4* v886;
            v886 = reinterpret_cast<int4*>(v876 + v885);
            int4* v887;
            v887 = reinterpret_cast<int4*>(v879 + v884);
            assert("Pointer alignment check" && (unsigned long long)(v886) % 4l == 0 && (unsigned long long)(v887) % 4l == 0);
            *v887 = *v886;
            int4* v888;
            v888 = reinterpret_cast<int4*>(v877 + v885);
            int4* v889;
            v889 = reinterpret_cast<int4*>(v880 + v884);
            assert("Pointer alignment check" && (unsigned long long)(v888) % 4l == 0 && (unsigned long long)(v889) % 4l == 0);
            *v889 = *v888;
            v882 += 1l ;
        }
        int v890;
        v890 = 0l;
        while (while_method_1(v890)){
            int v892;
            v892 = 0l;
            while (while_method_2(v892)){
                bool v894;
                v894 = 0l <= v892;
                bool v896;
                if (v894){
                    bool v895;
                    v895 = v892 < 4l;
                    v896 = v895;
                } else {
                    v896 = false;
                }
                bool v897;
                v897 = v896 == false;
                if (v897){
                    assert("The indices should be inside the range of the dimension." && v896);
                } else {
                }
                bool v899;
                v899 = 0l <= v869;
                bool v901;
                if (v899){
                    bool v900;
                    v900 = v869 < 1l;
                    v901 = v900;
                } else {
                    v901 = false;
                }
                bool v902;
                v902 = v901 == false;
                if (v902){
                    assert("The indices should be inside the range of the dimension." && v901);
                } else {
                }
                int v904;
                v904 = v869 * 4l;
                int v905;
                v905 = v892 + v904;
                bool v906;
                v906 = 0l <= v890;
                bool v908;
                if (v906){
                    bool v907;
                    v907 = v890 < 1l;
                    v908 = v907;
                } else {
                    v908 = false;
                }
                bool v909;
                v909 = v908 == false;
                if (v909){
                    assert("The indices should be inside the range of the dimension." && v908);
                } else {
                }
                int v911;
                v911 = v890 * 4l;
                int v912;
                v912 = v905 + v911;
                assert("Tensor range check" && 0 <= v890 && v890 < 1l);
                assert("Tensor range check" && 0 <= v892 && v892 < 4l);
                int v913;
                v913 = 4l * v890;
                int v914;
                v914 = v913 + v892;
                v881[v914] = v912;
                v892 += 1l ;
            }
            v890 += 1l ;
        }
        bool v915;
        v915 = 0l <= v873;
        bool v917;
        if (v915){
            bool v916;
            v916 = v873 < 1l;
            v917 = v916;
        } else {
            v917 = false;
        }
        bool v918;
        v918 = v917 == false;
        if (v918){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v917);
        } else {
        }
        bool v920;
        v920 = v866 && v870;
        bool v921;
        v921 = v920 == false;
        if (v921){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v920);
        } else {
        }
        int v923;
        v923 = v865 + v873;
        bool v924[4l];
        int v925;
        v925 = 0l;
        while (while_method_1(v925)){
            int v927;
            v927 = 0l;
            while (while_method_2(v927)){
                assert("Tensor range check" && 0 <= v925 && v925 < 1l);
                assert("Tensor range check" && 0 <= v927 && v927 < 4l);
                int v929;
                v929 = 4l * v925;
                int v930;
                v930 = v929 + v927;
                float v931;
                v931 = v879[v930];
                int v932;
                v932 = v881[v930];
                bool v933;
                v933 = v932 < 3l;
                assert("Tensor range check" && 0 <= v925 && v925 < 1l);
                assert("Tensor range check" && 0 <= v927 && v927 < 4l);
                v924[v930] = v933;
                v927 += 1l ;
            }
            v925 += 1l ;
        }
        float v934[4l];
        int v935;
        v935 = 0l;
        while (while_method_1(v935)){
            int v937;
            v937 = 0l;
            while (while_method_2(v937)){
                assert("Tensor range check" && 0 <= v935 && v935 < 1l);
                assert("Tensor range check" && 0 <= v937 && v937 < 4l);
                int v939;
                v939 = 4l * v935;
                int v940;
                v940 = v939 + v937;
                float v941;
                v941 = v879[v940];
                bool v942;
                v942 = v924[v940];
                float v945;
                if (v942){
                    bool v943;
                    v943 = 0.0f >= v941;
                    if (v943){
                        v945 = 0.0f;
                    } else {
                        v945 = v941;
                    }
                } else {
                    v945 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v935 && v935 < 1l);
                assert("Tensor range check" && 0 <= v937 && v937 < 4l);
                v934[v940] = v945;
                v937 += 1l ;
            }
            v935 += 1l ;
        }
        float v946;
        v946 = 0.0f;
        int v947;
        v947 = 0l;
        while (while_method_1(v947)){
            int v949;
            v949 = 0l;
            while (while_method_2(v949)){
                assert("Tensor range check" && 0 <= v947 && v947 < 1l);
                assert("Tensor range check" && 0 <= v949 && v949 < 4l);
                int v951;
                v951 = 4l * v947;
                int v952;
                v952 = v951 + v949;
                float v953;
                v953 = v934[v952];
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
        auto v957 = cooperative_groups::labeled_partition(v955,v956);
        Closure0 v958{};
        float v959;
        v959 = cooperative_groups::reduce(v957, v946, v958);
        int v960[4l];
        int v961;
        v961 = 0l;
        while (while_method_1(v961)){
            int v963;
            v963 = 0l;
            while (while_method_2(v963)){
                assert("Tensor range check" && 0 <= v961 && v961 < 1l);
                assert("Tensor range check" && 0 <= v963 && v963 < 4l);
                int v965;
                v965 = 4l * v961;
                int v966;
                v966 = v965 + v963;
                bool v967;
                v967 = v924[v966];
                int v968;
                if (v967){
                    v968 = 1l;
                } else {
                    v968 = 0l;
                }
                assert("Tensor range check" && 0 <= v961 && v961 < 1l);
                assert("Tensor range check" && 0 <= v963 && v963 < 4l);
                v960[v966] = v968;
                v963 += 1l ;
            }
            v961 += 1l ;
        }
        int v969;
        v969 = 0l;
        int v970;
        v970 = 0l;
        while (while_method_1(v970)){
            int v972;
            v972 = 0l;
            while (while_method_2(v972)){
                assert("Tensor range check" && 0 <= v970 && v970 < 1l);
                assert("Tensor range check" && 0 <= v972 && v972 < 4l);
                int v974;
                v974 = 4l * v970;
                int v975;
                v975 = v974 + v972;
                int v976;
                v976 = v960[v975];
                int v977;
                v977 = v969 + v976;
                v969 = v977;
                v972 += 1l ;
            }
            v970 += 1l ;
        }
        auto v978 = cooperative_groups::coalesced_threads();
        int v979;
        v979 = threadIdx.x;
        auto v980 = cooperative_groups::labeled_partition(v978,v979);
        Closure1 v981{};
        int v982;
        v982 = cooperative_groups::reduce(v980, v969, v981);
        float v983;
        v983 = (float)v982;
        float v984;
        v984 = 1.0f / v983;
        float v985[4l];
        int v986;
        v986 = 0l;
        while (while_method_1(v986)){
            int v988;
            v988 = 0l;
            while (while_method_2(v988)){
                assert("Tensor range check" && 0 <= v986 && v986 < 1l);
                assert("Tensor range check" && 0 <= v988 && v988 < 4l);
                int v990;
                v990 = 4l * v986;
                int v991;
                v991 = v990 + v988;
                float v992;
                v992 = v934[v991];
                bool v993;
                v993 = v924[v991];
                bool v994;
                v994 = v993 == false;
                float v999;
                if (v994){
                    v999 = 0.0f;
                } else {
                    bool v995;
                    v995 = v959 == 0.0f;
                    bool v996;
                    v996 = v995 != true;
                    if (v996){
                        float v997;
                        v997 = v992 / v959;
                        v999 = v997;
                    } else {
                        v999 = v984;
                    }
                }
                assert("Tensor range check" && 0 <= v986 && v986 < 1l);
                assert("Tensor range check" && 0 <= v988 && v988 < 4l);
                v985[v991] = v999;
                v988 += 1l ;
            }
            v986 += 1l ;
        }
        float v1000[4l];
        float v1001;
        v1001 = 0.0f;
        int v1002;
        v1002 = 0l;
        while (while_method_1(v1002)){
            assert("Tensor range check" && 0 <= v1002 && v1002 < 1l);
            int v1004;
            v1004 = 4l * v1002;
            assert("Tensor range check" && 0 <= v1002 && v1002 < 1l);
            int v1005; float v1006;
            Tuple2 tmp29 = Tuple2{0l, 0.0f};
            v1005 = tmp29.v0; v1006 = tmp29.v1;
            while (while_method_2(v1005)){
                assert("Tensor range check" && 0 <= v1005 && v1005 < 4l);
                int v1008;
                v1008 = v1005 + v1004;
                float v1009;
                v1009 = v985[v1008];
                float v1010;
                v1010 = v1006 + v1009;
                v1006 = v1010;
                v1005 += 1l ;
            }
            auto v1011 = cooperative_groups::coalesced_threads();
            int v1012;
            v1012 = threadIdx.x;
            auto v1013 = cooperative_groups::labeled_partition(v1011,v1012);
            Closure2 v1014{};
            float v1015;
            v1015 = cooperative_groups::inclusive_scan(v1013, v1006, v1014);
            float v1016;
            v1016 = v1013.shfl_up(v1015,1);
            bool v1017;
            v1017 = v1013.thread_rank() == 0;
            float v1018;
            if (v1017){
                v1018 = 0.0f;
            } else {
                v1018 = v1016;
            }
            float v1019;
            v1019 = v1013.shfl(v1015,v1013.num_threads()-1);
            float v1020;
            v1020 = v1001 + v1018;
            int v1021; float v1022;
            Tuple2 tmp30 = Tuple2{0l, v1020};
            v1021 = tmp30.v0; v1022 = tmp30.v1;
            while (while_method_2(v1021)){
                assert("Tensor range check" && 0 <= v1021 && v1021 < 4l);
                int v1024;
                v1024 = v1021 + v1004;
                float v1025;
                v1025 = v985[v1024];
                float v1026;
                v1026 = v1022 + v1025;
                assert("Tensor range check" && 0 <= v1021 && v1021 < 4l);
                v1000[v1024] = v1026;
                v1022 = v1026;
                v1021 += 1l ;
            }
            float v1027;
            v1027 = v1001 + v1019;
            v1001 = v1027;
            v1002 += 1l ;
        }
        float v1028[4l];
        bool v1029[4l];
        int v1030;
        v1030 = 0l;
        while (while_method_1(v1030)){
            int v1032;
            v1032 = 0l;
            while (while_method_2(v1032)){
                assert("Tensor range check" && 0 <= v1030 && v1030 < 1l);
                assert("Tensor range check" && 0 <= v1032 && v1032 < 4l);
                int v1034;
                v1034 = 4l * v1030;
                int v1035;
                v1035 = v1034 + v1032;
                float v1036;
                v1036 = v1000[v1035];
                float v1037;
                v1037 = v985[v1035];
                bool v1038;
                v1038 = v1037 > 0.0f;
                assert("Tensor range check" && 0 <= v1030 && v1030 < 1l);
                assert("Tensor range check" && 0 <= v1032 && v1032 < 4l);
                v1028[v1035] = v1036;
                v1029[v1035] = v1038;
                v1032 += 1l ;
            }
            v1030 += 1l ;
        }
        float v1039; bool v1040;
        Tuple3 tmp31 = Tuple3{-1.0f / 0.0f, false};
        v1039 = tmp31.v0; v1040 = tmp31.v1;
        int v1041;
        v1041 = 0l;
        while (while_method_1(v1041)){
            int v1043;
            v1043 = 0l;
            while (while_method_2(v1043)){
                assert("Tensor range check" && 0 <= v1041 && v1041 < 1l);
                assert("Tensor range check" && 0 <= v1043 && v1043 < 4l);
                int v1045;
                v1045 = 4l * v1041;
                int v1046;
                v1046 = v1045 + v1043;
                float v1047;
                v1047 = v1028[v1046];
                bool v1048;
                v1048 = v1029[v1046];
                float v1055; bool v1056;
                if (v1040){
                    if (v1048){
                        bool v1049;
                        v1049 = v1039 >= v1047;
                        float v1050;
                        if (v1049){
                            v1050 = v1039;
                        } else {
                            v1050 = v1047;
                        }
                        v1055 = v1050; v1056 = true;
                    } else {
                        v1055 = v1039; v1056 = v1040;
                    }
                } else {
                    if (v1048){
                        v1055 = v1047; v1056 = v1048;
                    } else {
                        v1055 = v1039; v1056 = v1040;
                    }
                }
                v1039 = v1055;
                v1040 = v1056;
                v1043 += 1l ;
            }
            v1041 += 1l ;
        }
        auto v1057 = cooperative_groups::coalesced_threads();
        int v1058;
        v1058 = threadIdx.x;
        auto v1059 = cooperative_groups::labeled_partition(v1057,v1058);
        Closure3 v1060{};
        float v1061; bool v1062;
        Tuple3 tmp32 = cooperative_groups::reduce(v1059, Tuple3{v1039, v1040}, v1060);
        v1061 = tmp32.v0; v1062 = tmp32.v1;
        bool v1063;
        v1063 = v1062 == false;
        if (v1063){
            assert("The local reduce must be true." && v1062);
        } else {
        }
        float v1065[4l];
        int v1066[4l];
        int v1067;
        v1067 = 0l;
        while (while_method_1(v1067)){
            int v1069;
            v1069 = 0l;
            while (while_method_2(v1069)){
                assert("Tensor range check" && 0 <= v1067 && v1067 < 1l);
                assert("Tensor range check" && 0 <= v1069 && v1069 < 4l);
                int v1071;
                v1071 = 4l * v1067;
                int v1072;
                v1072 = v1071 + v1069;
                int v1073;
                v1073 = v881[v1072];
                float v1074;
                v1074 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v1067 && v1067 < 1l);
                assert("Tensor range check" && 0 <= v1069 && v1069 < 4l);
                v1065[v1072] = v1074;
                v1066[v1072] = v1073;
                v1069 += 1l ;
            }
            v1067 += 1l ;
        }
        float v1075; int v1076;
        Tuple4 tmp33 = Tuple4{0.0f, 2147483647l};
        v1075 = tmp33.v0; v1076 = tmp33.v1;
        int v1077;
        v1077 = 0l;
        while (while_method_1(v1077)){
            int v1079;
            v1079 = 0l;
            while (while_method_2(v1079)){
                assert("Tensor range check" && 0 <= v1077 && v1077 < 1l);
                assert("Tensor range check" && 0 <= v1079 && v1079 < 4l);
                int v1081;
                v1081 = 4l * v1077;
                int v1082;
                v1082 = v1081 + v1079;
                float v1083;
                v1083 = v1065[v1082];
                int v1084;
                v1084 = v1066[v1082];
                bool v1085;
                v1085 = v1076 < v1084;
                float v1086; int v1087;
                if (v1085){
                    v1086 = v1075; v1087 = v1076;
                } else {
                    v1086 = v1083; v1087 = v1084;
                }
                v1075 = v1086;
                v1076 = v1087;
                v1079 += 1l ;
            }
            v1077 += 1l ;
        }
        auto v1088 = cooperative_groups::coalesced_threads();
        int v1089;
        v1089 = threadIdx.x;
        auto v1090 = cooperative_groups::labeled_partition(v1088,v1089);
        Closure4 v1091{};
        float v1092; int v1093;
        Tuple4 tmp34 = cooperative_groups::reduce(v1090, Tuple4{v1075, v1076}, v1091);
        v1092 = tmp34.v0; v1093 = tmp34.v1;
        float v1094;
        v1094 = v1061 * v1092;
        int v1095[4l];
        bool v1096[4l];
        int v1097;
        v1097 = 0l;
        while (while_method_1(v1097)){
            int v1099;
            v1099 = 0l;
            while (while_method_2(v1099)){
                assert("Tensor range check" && 0 <= v1097 && v1097 < 1l);
                assert("Tensor range check" && 0 <= v1099 && v1099 < 4l);
                int v1101;
                v1101 = 4l * v1097;
                int v1102;
                v1102 = v1101 + v1099;
                float v1103;
                v1103 = v1028[v1102];
                bool v1104;
                v1104 = v1029[v1102];
                int v1105;
                v1105 = v881[v1102];
                int v1108; bool v1109;
                if (v1104){
                    float v1106;
                    v1106 = v1103 - v1094;
                    bool v1107;
                    v1107 = v1106 >= 0.0f;
                    v1108 = v1105; v1109 = v1107;
                } else {
                    v1108 = 2147483647l; v1109 = false;
                }
                assert("Tensor range check" && 0 <= v1097 && v1097 < 1l);
                assert("Tensor range check" && 0 <= v1099 && v1099 < 4l);
                v1095[v1102] = v1108;
                v1096[v1102] = v1109;
                v1099 += 1l ;
            }
            v1097 += 1l ;
        }
        int v1110; bool v1111;
        Tuple5 tmp35 = Tuple5{2147483647l, false};
        v1110 = tmp35.v0; v1111 = tmp35.v1;
        int v1112;
        v1112 = 0l;
        while (while_method_1(v1112)){
            int v1114;
            v1114 = 0l;
            while (while_method_2(v1114)){
                assert("Tensor range check" && 0 <= v1112 && v1112 < 1l);
                assert("Tensor range check" && 0 <= v1114 && v1114 < 4l);
                int v1116;
                v1116 = 4l * v1112;
                int v1117;
                v1117 = v1116 + v1114;
                int v1118;
                v1118 = v1095[v1117];
                bool v1119;
                v1119 = v1096[v1117];
                int v1126; bool v1127;
                if (v1111){
                    if (v1119){
                        bool v1120;
                        v1120 = v1110 < v1118;
                        int v1121;
                        if (v1120){
                            v1121 = v1110;
                        } else {
                            v1121 = v1118;
                        }
                        v1126 = v1121; v1127 = true;
                    } else {
                        v1126 = v1110; v1127 = v1111;
                    }
                } else {
                    if (v1119){
                        v1126 = v1118; v1127 = v1119;
                    } else {
                        v1126 = v1110; v1127 = v1111;
                    }
                }
                v1110 = v1126;
                v1111 = v1127;
                v1114 += 1l ;
            }
            v1112 += 1l ;
        }
        auto v1128 = cooperative_groups::coalesced_threads();
        int v1129;
        v1129 = threadIdx.x;
        auto v1130 = cooperative_groups::labeled_partition(v1128,v1129);
        Closure5 v1131{};
        int v1132; bool v1133;
        Tuple5 tmp36 = cooperative_groups::reduce(v1130, Tuple5{v1110, v1111}, v1131);
        v1132 = tmp36.v0; v1133 = tmp36.v1;
        bool v1134;
        v1134 = v1133 == false;
        if (v1134){
            assert("The local reduce must be true." && v1133);
        } else {
        }
        bool v1136[4l];
        int v1137;
        v1137 = 0l;
        while (while_method_1(v1137)){
            int v1139;
            v1139 = 0l;
            while (while_method_2(v1139)){
                assert("Tensor range check" && 0 <= v1137 && v1137 < 1l);
                assert("Tensor range check" && 0 <= v1139 && v1139 < 4l);
                int v1141;
                v1141 = 4l * v1137;
                int v1142;
                v1142 = v1141 + v1139;
                float v1143;
                v1143 = v880[v1142];
                int v1144;
                v1144 = v881[v1142];
                bool v1145;
                v1145 = v1144 < 3l;
                assert("Tensor range check" && 0 <= v1137 && v1137 < 1l);
                assert("Tensor range check" && 0 <= v1139 && v1139 < 4l);
                v1136[v1142] = v1145;
                v1139 += 1l ;
            }
            v1137 += 1l ;
        }
        float v1146[4l];
        int v1147;
        v1147 = 0l;
        while (while_method_1(v1147)){
            int v1149;
            v1149 = 0l;
            while (while_method_2(v1149)){
                assert("Tensor range check" && 0 <= v1147 && v1147 < 1l);
                assert("Tensor range check" && 0 <= v1149 && v1149 < 4l);
                int v1151;
                v1151 = 4l * v1147;
                int v1152;
                v1152 = v1151 + v1149;
                float v1153;
                v1153 = v880[v1152];
                bool v1154;
                v1154 = v1136[v1152];
                float v1157;
                if (v1154){
                    bool v1155;
                    v1155 = 0.0f >= v1153;
                    if (v1155){
                        v1157 = 0.0f;
                    } else {
                        v1157 = v1153;
                    }
                } else {
                    v1157 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1147 && v1147 < 1l);
                assert("Tensor range check" && 0 <= v1149 && v1149 < 4l);
                v1146[v1152] = v1157;
                v1149 += 1l ;
            }
            v1147 += 1l ;
        }
        float v1158;
        v1158 = 0.0f;
        int v1159;
        v1159 = 0l;
        while (while_method_1(v1159)){
            int v1161;
            v1161 = 0l;
            while (while_method_2(v1161)){
                assert("Tensor range check" && 0 <= v1159 && v1159 < 1l);
                assert("Tensor range check" && 0 <= v1161 && v1161 < 4l);
                int v1163;
                v1163 = 4l * v1159;
                int v1164;
                v1164 = v1163 + v1161;
                float v1165;
                v1165 = v1146[v1164];
                float v1166;
                v1166 = v1158 + v1165;
                v1158 = v1166;
                v1161 += 1l ;
            }
            v1159 += 1l ;
        }
        auto v1167 = cooperative_groups::coalesced_threads();
        int v1168;
        v1168 = threadIdx.x;
        auto v1169 = cooperative_groups::labeled_partition(v1167,v1168);
        float v1170;
        v1170 = cooperative_groups::reduce(v1169, v1158, v958);
        int v1171[4l];
        int v1172;
        v1172 = 0l;
        while (while_method_1(v1172)){
            int v1174;
            v1174 = 0l;
            while (while_method_2(v1174)){
                assert("Tensor range check" && 0 <= v1172 && v1172 < 1l);
                assert("Tensor range check" && 0 <= v1174 && v1174 < 4l);
                int v1176;
                v1176 = 4l * v1172;
                int v1177;
                v1177 = v1176 + v1174;
                bool v1178;
                v1178 = v1136[v1177];
                int v1179;
                if (v1178){
                    v1179 = 1l;
                } else {
                    v1179 = 0l;
                }
                assert("Tensor range check" && 0 <= v1172 && v1172 < 1l);
                assert("Tensor range check" && 0 <= v1174 && v1174 < 4l);
                v1171[v1177] = v1179;
                v1174 += 1l ;
            }
            v1172 += 1l ;
        }
        int v1180;
        v1180 = 0l;
        int v1181;
        v1181 = 0l;
        while (while_method_1(v1181)){
            int v1183;
            v1183 = 0l;
            while (while_method_2(v1183)){
                assert("Tensor range check" && 0 <= v1181 && v1181 < 1l);
                assert("Tensor range check" && 0 <= v1183 && v1183 < 4l);
                int v1185;
                v1185 = 4l * v1181;
                int v1186;
                v1186 = v1185 + v1183;
                int v1187;
                v1187 = v1171[v1186];
                int v1188;
                v1188 = v1180 + v1187;
                v1180 = v1188;
                v1183 += 1l ;
            }
            v1181 += 1l ;
        }
        auto v1189 = cooperative_groups::coalesced_threads();
        int v1190;
        v1190 = threadIdx.x;
        auto v1191 = cooperative_groups::labeled_partition(v1189,v1190);
        int v1192;
        v1192 = cooperative_groups::reduce(v1191, v1180, v981);
        float v1193;
        v1193 = (float)v1192;
        float v1194;
        v1194 = 1.0f / v1193;
        float v1195[4l];
        int v1196;
        v1196 = 0l;
        while (while_method_1(v1196)){
            int v1198;
            v1198 = 0l;
            while (while_method_2(v1198)){
                assert("Tensor range check" && 0 <= v1196 && v1196 < 1l);
                assert("Tensor range check" && 0 <= v1198 && v1198 < 4l);
                int v1200;
                v1200 = 4l * v1196;
                int v1201;
                v1201 = v1200 + v1198;
                float v1202;
                v1202 = v1146[v1201];
                bool v1203;
                v1203 = v1136[v1201];
                bool v1204;
                v1204 = v1203 == false;
                float v1209;
                if (v1204){
                    v1209 = 0.0f;
                } else {
                    bool v1205;
                    v1205 = v1170 == 0.0f;
                    bool v1206;
                    v1206 = v1205 != true;
                    if (v1206){
                        float v1207;
                        v1207 = v1202 / v1170;
                        v1209 = v1207;
                    } else {
                        v1209 = v1194;
                    }
                }
                assert("Tensor range check" && 0 <= v1196 && v1196 < 1l);
                assert("Tensor range check" && 0 <= v1198 && v1198 < 4l);
                v1195[v1201] = v1209;
                v1198 += 1l ;
            }
            v1196 += 1l ;
        }
        float v1210; int v1211;
        Tuple4 tmp37 = Tuple4{0.0f, 2147483647l};
        v1210 = tmp37.v0; v1211 = tmp37.v1;
        int v1212;
        v1212 = 0l;
        while (while_method_1(v1212)){
            int v1214;
            v1214 = 0l;
            while (while_method_2(v1214)){
                assert("Tensor range check" && 0 <= v1212 && v1212 < 1l);
                assert("Tensor range check" && 0 <= v1214 && v1214 < 4l);
                int v1216;
                v1216 = 4l * v1212;
                int v1217;
                v1217 = v1216 + v1214;
                float v1218;
                v1218 = v985[v1217];
                int v1219;
                v1219 = v881[v1217];
                bool v1220;
                v1220 = v1211 == v1132;
                float v1224; int v1225;
                if (v1220){
                    v1224 = v1210; v1225 = v1211;
                } else {
                    bool v1221;
                    v1221 = v1219 == v1132;
                    if (v1221){
                        v1224 = v1218; v1225 = v1219;
                    } else {
                        v1224 = v1210; v1225 = v1211;
                    }
                }
                v1210 = v1224;
                v1211 = v1225;
                v1214 += 1l ;
            }
            v1212 += 1l ;
        }
        auto v1226 = cooperative_groups::coalesced_threads();
        int v1227;
        v1227 = threadIdx.x;
        auto v1228 = cooperative_groups::labeled_partition(v1226,v1227);
        Closure6 v1229{v1132};
        float v1230; int v1231;
        Tuple4 tmp38 = cooperative_groups::reduce(v1228, Tuple4{v1210, v1211}, v1229);
        v1230 = tmp38.v0; v1231 = tmp38.v1;
        bool v1232;
        v1232 = v1231 == 2147483647l;
        bool v1233;
        v1233 = v1232 != true;
        bool v1234;
        v1234 = v1233 == false;
        if (v1234){
            assert("Expected a valid action id in get_action." && v1233);
        } else {
        }
        float v1236; int v1237;
        Tuple4 tmp39 = Tuple4{0.0f, 2147483647l};
        v1236 = tmp39.v0; v1237 = tmp39.v1;
        int v1238;
        v1238 = 0l;
        while (while_method_1(v1238)){
            int v1240;
            v1240 = 0l;
            while (while_method_2(v1240)){
                assert("Tensor range check" && 0 <= v1238 && v1238 < 1l);
                assert("Tensor range check" && 0 <= v1240 && v1240 < 4l);
                int v1242;
                v1242 = 4l * v1238;
                int v1243;
                v1243 = v1242 + v1240;
                float v1244;
                v1244 = v1195[v1243];
                int v1245;
                v1245 = v881[v1243];
                bool v1246;
                v1246 = v1237 == v1132;
                float v1250; int v1251;
                if (v1246){
                    v1250 = v1236; v1251 = v1237;
                } else {
                    bool v1247;
                    v1247 = v1245 == v1132;
                    if (v1247){
                        v1250 = v1244; v1251 = v1245;
                    } else {
                        v1250 = v1236; v1251 = v1237;
                    }
                }
                v1236 = v1250;
                v1237 = v1251;
                v1240 += 1l ;
            }
            v1238 += 1l ;
        }
        auto v1252 = cooperative_groups::coalesced_threads();
        int v1253;
        v1253 = threadIdx.x;
        auto v1254 = cooperative_groups::labeled_partition(v1252,v1253);
        float v1255; int v1256;
        Tuple4 tmp40 = cooperative_groups::reduce(v1254, Tuple4{v1236, v1237}, v1229);
        v1255 = tmp40.v0; v1256 = tmp40.v1;
        bool v1257;
        v1257 = v1256 == 2147483647l;
        bool v1258;
        v1258 = v1257 != true;
        bool v1259;
        v1259 = v1258 == false;
        if (v1259){
            assert("Expected a valid action id in get_action." && v1258);
        } else {
        }
        assert("Tensor range check" && 0 <= v873 && v873 < 1l);
        v857[v875] = v1255;
        v858[v875] = v1230;
        v859[v875] = v1132;
        v873 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1261;
    v1261 = threadIdx.x;
    assert("Tensor range check" && 0 <= v1261 && v1261 < 32l);
    float v1262;
    v1262 = v857[v1261];
    float v1263;
    v1263 = v858[v1261];
    int v1264;
    v1264 = v859[v1261];
    v855[0l] = Tuple0{v1262, v1263, v1264};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float v1265; float v1266; int v1267;
    Tuple0 tmp41 = v855[0l];
    v1265 = tmp41.v0; v1266 = tmp41.v1; v1267 = tmp41.v2;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v17, v18, v19, v20, v854, v853, v1267, v1265, v1266);
    int v1268;
    v1268 = 343l;
    int v1269;
    v1269 = 1l;
    Tuple0 v1270[1l];
    __shared__ Tuple1 v1271[32l];
    __shared__ float v1272[32l];
    __shared__ float v1273[32l];
    __shared__ int v1274[32l];
    int v1275;
    v1275 = threadIdx.x;
    float * v1276;
    v1276 = v8+1372l;
    float * v1278;
    v1278 = v9+1372l;
    assert("Tensor range check" && 0 <= v1275 && v1275 < 32l);
    v1271[v1275] = Tuple1{v1276, v1278};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1280;
    v1280 = threadIdx.x;
    bool v1281;
    v1281 = 0l <= v1280;
    bool v1282;
    v1282 = v1281 == false;
    if (v1282){
        assert("The index needs to be zero or positive." && v1281);
    } else {
    }
    int v1284;
    v1284 = v1280 % 1l;
    bool v1285;
    v1285 = v1280 < 32l;
    bool v1286;
    v1286 = v1285 == false;
    if (v1286){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1285);
    } else {
    }
    assert("Tensor range check" && 0 <= v1280 && v1280 < 32l);
    assert("Tensor range check" && 0 <= v1280 && v1280 < 32l);
    int v1288;
    v1288 = 0l;
    while (while_method_1(v1288)){
        assert("Tensor range check" && 0 <= v1288 && v1288 < 1l);
        int v1290;
        v1290 = v1288 + v1280;
        float * v1291; float * v1292;
        Tuple1 tmp42 = v1271[v1290];
        v1291 = tmp42.v0; v1292 = tmp42.v1;
        assert("Tensor range check" && 0 <= v1284 && v1284 < 1l);
        int v1293;
        v1293 = 4l * v1284;
        float v1294[4l];
        float v1295[4l];
        int v1296[4l];
        int v1297;
        v1297 = 0l;
        while (while_method_1(v1297)){
            assert("Tensor range check" && 0 <= v1297 && v1297 < 1l);
            int v1299;
            v1299 = 4l * v1297;
            assert("Tensor range check" && 0 <= v1297 && v1297 < 1l);
            int v1300;
            v1300 = v1299 + v1293;
            int4* v1301;
            v1301 = reinterpret_cast<int4*>(v1291 + v1300);
            int4* v1302;
            v1302 = reinterpret_cast<int4*>(v1294 + v1299);
            assert("Pointer alignment check" && (unsigned long long)(v1301) % 4l == 0 && (unsigned long long)(v1302) % 4l == 0);
            *v1302 = *v1301;
            int4* v1303;
            v1303 = reinterpret_cast<int4*>(v1292 + v1300);
            int4* v1304;
            v1304 = reinterpret_cast<int4*>(v1295 + v1299);
            assert("Pointer alignment check" && (unsigned long long)(v1303) % 4l == 0 && (unsigned long long)(v1304) % 4l == 0);
            *v1304 = *v1303;
            v1297 += 1l ;
        }
        int v1305;
        v1305 = 0l;
        while (while_method_1(v1305)){
            int v1307;
            v1307 = 0l;
            while (while_method_2(v1307)){
                bool v1309;
                v1309 = 0l <= v1307;
                bool v1311;
                if (v1309){
                    bool v1310;
                    v1310 = v1307 < 4l;
                    v1311 = v1310;
                } else {
                    v1311 = false;
                }
                bool v1312;
                v1312 = v1311 == false;
                if (v1312){
                    assert("The indices should be inside the range of the dimension." && v1311);
                } else {
                }
                bool v1314;
                v1314 = 0l <= v1284;
                bool v1316;
                if (v1314){
                    bool v1315;
                    v1315 = v1284 < 1l;
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
                int v1319;
                v1319 = v1284 * 4l;
                int v1320;
                v1320 = v1307 + v1319;
                bool v1321;
                v1321 = 0l <= v1305;
                bool v1323;
                if (v1321){
                    bool v1322;
                    v1322 = v1305 < 1l;
                    v1323 = v1322;
                } else {
                    v1323 = false;
                }
                bool v1324;
                v1324 = v1323 == false;
                if (v1324){
                    assert("The indices should be inside the range of the dimension." && v1323);
                } else {
                }
                int v1326;
                v1326 = v1305 * 4l;
                int v1327;
                v1327 = v1320 + v1326;
                assert("Tensor range check" && 0 <= v1305 && v1305 < 1l);
                assert("Tensor range check" && 0 <= v1307 && v1307 < 4l);
                int v1328;
                v1328 = 4l * v1305;
                int v1329;
                v1329 = v1328 + v1307;
                v1296[v1329] = v1327;
                v1307 += 1l ;
            }
            v1305 += 1l ;
        }
        bool v1330;
        v1330 = 0l <= v1288;
        bool v1332;
        if (v1330){
            bool v1331;
            v1331 = v1288 < 1l;
            v1332 = v1331;
        } else {
            v1332 = false;
        }
        bool v1333;
        v1333 = v1332 == false;
        if (v1333){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1332);
        } else {
        }
        bool v1335;
        v1335 = v1281 && v1285;
        bool v1336;
        v1336 = v1335 == false;
        if (v1336){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1335);
        } else {
        }
        int v1338;
        v1338 = v1280 + v1288;
        bool v1339[4l];
        int v1340;
        v1340 = 0l;
        while (while_method_1(v1340)){
            int v1342;
            v1342 = 0l;
            while (while_method_2(v1342)){
                assert("Tensor range check" && 0 <= v1340 && v1340 < 1l);
                assert("Tensor range check" && 0 <= v1342 && v1342 < 4l);
                int v1344;
                v1344 = 4l * v1340;
                int v1345;
                v1345 = v1344 + v1342;
                float v1346;
                v1346 = v1294[v1345];
                int v1347;
                v1347 = v1296[v1345];
                bool v1348;
                v1348 = v1347 < 3l;
                assert("Tensor range check" && 0 <= v1340 && v1340 < 1l);
                assert("Tensor range check" && 0 <= v1342 && v1342 < 4l);
                v1339[v1345] = v1348;
                v1342 += 1l ;
            }
            v1340 += 1l ;
        }
        float v1349[4l];
        int v1350;
        v1350 = 0l;
        while (while_method_1(v1350)){
            int v1352;
            v1352 = 0l;
            while (while_method_2(v1352)){
                assert("Tensor range check" && 0 <= v1350 && v1350 < 1l);
                assert("Tensor range check" && 0 <= v1352 && v1352 < 4l);
                int v1354;
                v1354 = 4l * v1350;
                int v1355;
                v1355 = v1354 + v1352;
                float v1356;
                v1356 = v1294[v1355];
                bool v1357;
                v1357 = v1339[v1355];
                float v1360;
                if (v1357){
                    bool v1358;
                    v1358 = 0.0f >= v1356;
                    if (v1358){
                        v1360 = 0.0f;
                    } else {
                        v1360 = v1356;
                    }
                } else {
                    v1360 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1350 && v1350 < 1l);
                assert("Tensor range check" && 0 <= v1352 && v1352 < 4l);
                v1349[v1355] = v1360;
                v1352 += 1l ;
            }
            v1350 += 1l ;
        }
        float v1361;
        v1361 = 0.0f;
        int v1362;
        v1362 = 0l;
        while (while_method_1(v1362)){
            int v1364;
            v1364 = 0l;
            while (while_method_2(v1364)){
                assert("Tensor range check" && 0 <= v1362 && v1362 < 1l);
                assert("Tensor range check" && 0 <= v1364 && v1364 < 4l);
                int v1366;
                v1366 = 4l * v1362;
                int v1367;
                v1367 = v1366 + v1364;
                float v1368;
                v1368 = v1349[v1367];
                float v1369;
                v1369 = v1361 + v1368;
                v1361 = v1369;
                v1364 += 1l ;
            }
            v1362 += 1l ;
        }
        auto v1370 = cooperative_groups::coalesced_threads();
        int v1371;
        v1371 = threadIdx.x;
        auto v1372 = cooperative_groups::labeled_partition(v1370,v1371);
        Closure0 v1373{};
        float v1374;
        v1374 = cooperative_groups::reduce(v1372, v1361, v1373);
        int v1375[4l];
        int v1376;
        v1376 = 0l;
        while (while_method_1(v1376)){
            int v1378;
            v1378 = 0l;
            while (while_method_2(v1378)){
                assert("Tensor range check" && 0 <= v1376 && v1376 < 1l);
                assert("Tensor range check" && 0 <= v1378 && v1378 < 4l);
                int v1380;
                v1380 = 4l * v1376;
                int v1381;
                v1381 = v1380 + v1378;
                bool v1382;
                v1382 = v1339[v1381];
                int v1383;
                if (v1382){
                    v1383 = 1l;
                } else {
                    v1383 = 0l;
                }
                assert("Tensor range check" && 0 <= v1376 && v1376 < 1l);
                assert("Tensor range check" && 0 <= v1378 && v1378 < 4l);
                v1375[v1381] = v1383;
                v1378 += 1l ;
            }
            v1376 += 1l ;
        }
        int v1384;
        v1384 = 0l;
        int v1385;
        v1385 = 0l;
        while (while_method_1(v1385)){
            int v1387;
            v1387 = 0l;
            while (while_method_2(v1387)){
                assert("Tensor range check" && 0 <= v1385 && v1385 < 1l);
                assert("Tensor range check" && 0 <= v1387 && v1387 < 4l);
                int v1389;
                v1389 = 4l * v1385;
                int v1390;
                v1390 = v1389 + v1387;
                int v1391;
                v1391 = v1375[v1390];
                int v1392;
                v1392 = v1384 + v1391;
                v1384 = v1392;
                v1387 += 1l ;
            }
            v1385 += 1l ;
        }
        auto v1393 = cooperative_groups::coalesced_threads();
        int v1394;
        v1394 = threadIdx.x;
        auto v1395 = cooperative_groups::labeled_partition(v1393,v1394);
        Closure1 v1396{};
        int v1397;
        v1397 = cooperative_groups::reduce(v1395, v1384, v1396);
        float v1398;
        v1398 = (float)v1397;
        float v1399;
        v1399 = 1.0f / v1398;
        float v1400[4l];
        int v1401;
        v1401 = 0l;
        while (while_method_1(v1401)){
            int v1403;
            v1403 = 0l;
            while (while_method_2(v1403)){
                assert("Tensor range check" && 0 <= v1401 && v1401 < 1l);
                assert("Tensor range check" && 0 <= v1403 && v1403 < 4l);
                int v1405;
                v1405 = 4l * v1401;
                int v1406;
                v1406 = v1405 + v1403;
                float v1407;
                v1407 = v1349[v1406];
                bool v1408;
                v1408 = v1339[v1406];
                bool v1409;
                v1409 = v1408 == false;
                float v1414;
                if (v1409){
                    v1414 = 0.0f;
                } else {
                    bool v1410;
                    v1410 = v1374 == 0.0f;
                    bool v1411;
                    v1411 = v1410 != true;
                    if (v1411){
                        float v1412;
                        v1412 = v1407 / v1374;
                        v1414 = v1412;
                    } else {
                        v1414 = v1399;
                    }
                }
                assert("Tensor range check" && 0 <= v1401 && v1401 < 1l);
                assert("Tensor range check" && 0 <= v1403 && v1403 < 4l);
                v1400[v1406] = v1414;
                v1403 += 1l ;
            }
            v1401 += 1l ;
        }
        float v1415[4l];
        float v1416;
        v1416 = 0.0f;
        int v1417;
        v1417 = 0l;
        while (while_method_1(v1417)){
            assert("Tensor range check" && 0 <= v1417 && v1417 < 1l);
            int v1419;
            v1419 = 4l * v1417;
            assert("Tensor range check" && 0 <= v1417 && v1417 < 1l);
            int v1420; float v1421;
            Tuple2 tmp43 = Tuple2{0l, 0.0f};
            v1420 = tmp43.v0; v1421 = tmp43.v1;
            while (while_method_2(v1420)){
                assert("Tensor range check" && 0 <= v1420 && v1420 < 4l);
                int v1423;
                v1423 = v1420 + v1419;
                float v1424;
                v1424 = v1400[v1423];
                float v1425;
                v1425 = v1421 + v1424;
                v1421 = v1425;
                v1420 += 1l ;
            }
            auto v1426 = cooperative_groups::coalesced_threads();
            int v1427;
            v1427 = threadIdx.x;
            auto v1428 = cooperative_groups::labeled_partition(v1426,v1427);
            Closure2 v1429{};
            float v1430;
            v1430 = cooperative_groups::inclusive_scan(v1428, v1421, v1429);
            float v1431;
            v1431 = v1428.shfl_up(v1430,1);
            bool v1432;
            v1432 = v1428.thread_rank() == 0;
            float v1433;
            if (v1432){
                v1433 = 0.0f;
            } else {
                v1433 = v1431;
            }
            float v1434;
            v1434 = v1428.shfl(v1430,v1428.num_threads()-1);
            float v1435;
            v1435 = v1416 + v1433;
            int v1436; float v1437;
            Tuple2 tmp44 = Tuple2{0l, v1435};
            v1436 = tmp44.v0; v1437 = tmp44.v1;
            while (while_method_2(v1436)){
                assert("Tensor range check" && 0 <= v1436 && v1436 < 4l);
                int v1439;
                v1439 = v1436 + v1419;
                float v1440;
                v1440 = v1400[v1439];
                float v1441;
                v1441 = v1437 + v1440;
                assert("Tensor range check" && 0 <= v1436 && v1436 < 4l);
                v1415[v1439] = v1441;
                v1437 = v1441;
                v1436 += 1l ;
            }
            float v1442;
            v1442 = v1416 + v1434;
            v1416 = v1442;
            v1417 += 1l ;
        }
        float v1443[4l];
        bool v1444[4l];
        int v1445;
        v1445 = 0l;
        while (while_method_1(v1445)){
            int v1447;
            v1447 = 0l;
            while (while_method_2(v1447)){
                assert("Tensor range check" && 0 <= v1445 && v1445 < 1l);
                assert("Tensor range check" && 0 <= v1447 && v1447 < 4l);
                int v1449;
                v1449 = 4l * v1445;
                int v1450;
                v1450 = v1449 + v1447;
                float v1451;
                v1451 = v1415[v1450];
                float v1452;
                v1452 = v1400[v1450];
                bool v1453;
                v1453 = v1452 > 0.0f;
                assert("Tensor range check" && 0 <= v1445 && v1445 < 1l);
                assert("Tensor range check" && 0 <= v1447 && v1447 < 4l);
                v1443[v1450] = v1451;
                v1444[v1450] = v1453;
                v1447 += 1l ;
            }
            v1445 += 1l ;
        }
        float v1454; bool v1455;
        Tuple3 tmp45 = Tuple3{-1.0f / 0.0f, false};
        v1454 = tmp45.v0; v1455 = tmp45.v1;
        int v1456;
        v1456 = 0l;
        while (while_method_1(v1456)){
            int v1458;
            v1458 = 0l;
            while (while_method_2(v1458)){
                assert("Tensor range check" && 0 <= v1456 && v1456 < 1l);
                assert("Tensor range check" && 0 <= v1458 && v1458 < 4l);
                int v1460;
                v1460 = 4l * v1456;
                int v1461;
                v1461 = v1460 + v1458;
                float v1462;
                v1462 = v1443[v1461];
                bool v1463;
                v1463 = v1444[v1461];
                float v1470; bool v1471;
                if (v1455){
                    if (v1463){
                        bool v1464;
                        v1464 = v1454 >= v1462;
                        float v1465;
                        if (v1464){
                            v1465 = v1454;
                        } else {
                            v1465 = v1462;
                        }
                        v1470 = v1465; v1471 = true;
                    } else {
                        v1470 = v1454; v1471 = v1455;
                    }
                } else {
                    if (v1463){
                        v1470 = v1462; v1471 = v1463;
                    } else {
                        v1470 = v1454; v1471 = v1455;
                    }
                }
                v1454 = v1470;
                v1455 = v1471;
                v1458 += 1l ;
            }
            v1456 += 1l ;
        }
        auto v1472 = cooperative_groups::coalesced_threads();
        int v1473;
        v1473 = threadIdx.x;
        auto v1474 = cooperative_groups::labeled_partition(v1472,v1473);
        Closure3 v1475{};
        float v1476; bool v1477;
        Tuple3 tmp46 = cooperative_groups::reduce(v1474, Tuple3{v1454, v1455}, v1475);
        v1476 = tmp46.v0; v1477 = tmp46.v1;
        bool v1478;
        v1478 = v1477 == false;
        if (v1478){
            assert("The local reduce must be true." && v1477);
        } else {
        }
        float v1480[4l];
        int v1481[4l];
        int v1482;
        v1482 = 0l;
        while (while_method_1(v1482)){
            int v1484;
            v1484 = 0l;
            while (while_method_2(v1484)){
                assert("Tensor range check" && 0 <= v1482 && v1482 < 1l);
                assert("Tensor range check" && 0 <= v1484 && v1484 < 4l);
                int v1486;
                v1486 = 4l * v1482;
                int v1487;
                v1487 = v1486 + v1484;
                int v1488;
                v1488 = v1296[v1487];
                float v1489;
                v1489 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v1482 && v1482 < 1l);
                assert("Tensor range check" && 0 <= v1484 && v1484 < 4l);
                v1480[v1487] = v1489;
                v1481[v1487] = v1488;
                v1484 += 1l ;
            }
            v1482 += 1l ;
        }
        float v1490; int v1491;
        Tuple4 tmp47 = Tuple4{0.0f, 2147483647l};
        v1490 = tmp47.v0; v1491 = tmp47.v1;
        int v1492;
        v1492 = 0l;
        while (while_method_1(v1492)){
            int v1494;
            v1494 = 0l;
            while (while_method_2(v1494)){
                assert("Tensor range check" && 0 <= v1492 && v1492 < 1l);
                assert("Tensor range check" && 0 <= v1494 && v1494 < 4l);
                int v1496;
                v1496 = 4l * v1492;
                int v1497;
                v1497 = v1496 + v1494;
                float v1498;
                v1498 = v1480[v1497];
                int v1499;
                v1499 = v1481[v1497];
                bool v1500;
                v1500 = v1491 < v1499;
                float v1501; int v1502;
                if (v1500){
                    v1501 = v1490; v1502 = v1491;
                } else {
                    v1501 = v1498; v1502 = v1499;
                }
                v1490 = v1501;
                v1491 = v1502;
                v1494 += 1l ;
            }
            v1492 += 1l ;
        }
        auto v1503 = cooperative_groups::coalesced_threads();
        int v1504;
        v1504 = threadIdx.x;
        auto v1505 = cooperative_groups::labeled_partition(v1503,v1504);
        Closure4 v1506{};
        float v1507; int v1508;
        Tuple4 tmp48 = cooperative_groups::reduce(v1505, Tuple4{v1490, v1491}, v1506);
        v1507 = tmp48.v0; v1508 = tmp48.v1;
        float v1509;
        v1509 = v1476 * v1507;
        int v1510[4l];
        bool v1511[4l];
        int v1512;
        v1512 = 0l;
        while (while_method_1(v1512)){
            int v1514;
            v1514 = 0l;
            while (while_method_2(v1514)){
                assert("Tensor range check" && 0 <= v1512 && v1512 < 1l);
                assert("Tensor range check" && 0 <= v1514 && v1514 < 4l);
                int v1516;
                v1516 = 4l * v1512;
                int v1517;
                v1517 = v1516 + v1514;
                float v1518;
                v1518 = v1443[v1517];
                bool v1519;
                v1519 = v1444[v1517];
                int v1520;
                v1520 = v1296[v1517];
                int v1523; bool v1524;
                if (v1519){
                    float v1521;
                    v1521 = v1518 - v1509;
                    bool v1522;
                    v1522 = v1521 >= 0.0f;
                    v1523 = v1520; v1524 = v1522;
                } else {
                    v1523 = 2147483647l; v1524 = false;
                }
                assert("Tensor range check" && 0 <= v1512 && v1512 < 1l);
                assert("Tensor range check" && 0 <= v1514 && v1514 < 4l);
                v1510[v1517] = v1523;
                v1511[v1517] = v1524;
                v1514 += 1l ;
            }
            v1512 += 1l ;
        }
        int v1525; bool v1526;
        Tuple5 tmp49 = Tuple5{2147483647l, false};
        v1525 = tmp49.v0; v1526 = tmp49.v1;
        int v1527;
        v1527 = 0l;
        while (while_method_1(v1527)){
            int v1529;
            v1529 = 0l;
            while (while_method_2(v1529)){
                assert("Tensor range check" && 0 <= v1527 && v1527 < 1l);
                assert("Tensor range check" && 0 <= v1529 && v1529 < 4l);
                int v1531;
                v1531 = 4l * v1527;
                int v1532;
                v1532 = v1531 + v1529;
                int v1533;
                v1533 = v1510[v1532];
                bool v1534;
                v1534 = v1511[v1532];
                int v1541; bool v1542;
                if (v1526){
                    if (v1534){
                        bool v1535;
                        v1535 = v1525 < v1533;
                        int v1536;
                        if (v1535){
                            v1536 = v1525;
                        } else {
                            v1536 = v1533;
                        }
                        v1541 = v1536; v1542 = true;
                    } else {
                        v1541 = v1525; v1542 = v1526;
                    }
                } else {
                    if (v1534){
                        v1541 = v1533; v1542 = v1534;
                    } else {
                        v1541 = v1525; v1542 = v1526;
                    }
                }
                v1525 = v1541;
                v1526 = v1542;
                v1529 += 1l ;
            }
            v1527 += 1l ;
        }
        auto v1543 = cooperative_groups::coalesced_threads();
        int v1544;
        v1544 = threadIdx.x;
        auto v1545 = cooperative_groups::labeled_partition(v1543,v1544);
        Closure5 v1546{};
        int v1547; bool v1548;
        Tuple5 tmp50 = cooperative_groups::reduce(v1545, Tuple5{v1525, v1526}, v1546);
        v1547 = tmp50.v0; v1548 = tmp50.v1;
        bool v1549;
        v1549 = v1548 == false;
        if (v1549){
            assert("The local reduce must be true." && v1548);
        } else {
        }
        bool v1551[4l];
        int v1552;
        v1552 = 0l;
        while (while_method_1(v1552)){
            int v1554;
            v1554 = 0l;
            while (while_method_2(v1554)){
                assert("Tensor range check" && 0 <= v1552 && v1552 < 1l);
                assert("Tensor range check" && 0 <= v1554 && v1554 < 4l);
                int v1556;
                v1556 = 4l * v1552;
                int v1557;
                v1557 = v1556 + v1554;
                float v1558;
                v1558 = v1295[v1557];
                int v1559;
                v1559 = v1296[v1557];
                bool v1560;
                v1560 = v1559 < 3l;
                assert("Tensor range check" && 0 <= v1552 && v1552 < 1l);
                assert("Tensor range check" && 0 <= v1554 && v1554 < 4l);
                v1551[v1557] = v1560;
                v1554 += 1l ;
            }
            v1552 += 1l ;
        }
        float v1561[4l];
        int v1562;
        v1562 = 0l;
        while (while_method_1(v1562)){
            int v1564;
            v1564 = 0l;
            while (while_method_2(v1564)){
                assert("Tensor range check" && 0 <= v1562 && v1562 < 1l);
                assert("Tensor range check" && 0 <= v1564 && v1564 < 4l);
                int v1566;
                v1566 = 4l * v1562;
                int v1567;
                v1567 = v1566 + v1564;
                float v1568;
                v1568 = v1295[v1567];
                bool v1569;
                v1569 = v1551[v1567];
                float v1572;
                if (v1569){
                    bool v1570;
                    v1570 = 0.0f >= v1568;
                    if (v1570){
                        v1572 = 0.0f;
                    } else {
                        v1572 = v1568;
                    }
                } else {
                    v1572 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1562 && v1562 < 1l);
                assert("Tensor range check" && 0 <= v1564 && v1564 < 4l);
                v1561[v1567] = v1572;
                v1564 += 1l ;
            }
            v1562 += 1l ;
        }
        float v1573;
        v1573 = 0.0f;
        int v1574;
        v1574 = 0l;
        while (while_method_1(v1574)){
            int v1576;
            v1576 = 0l;
            while (while_method_2(v1576)){
                assert("Tensor range check" && 0 <= v1574 && v1574 < 1l);
                assert("Tensor range check" && 0 <= v1576 && v1576 < 4l);
                int v1578;
                v1578 = 4l * v1574;
                int v1579;
                v1579 = v1578 + v1576;
                float v1580;
                v1580 = v1561[v1579];
                float v1581;
                v1581 = v1573 + v1580;
                v1573 = v1581;
                v1576 += 1l ;
            }
            v1574 += 1l ;
        }
        auto v1582 = cooperative_groups::coalesced_threads();
        int v1583;
        v1583 = threadIdx.x;
        auto v1584 = cooperative_groups::labeled_partition(v1582,v1583);
        float v1585;
        v1585 = cooperative_groups::reduce(v1584, v1573, v1373);
        int v1586[4l];
        int v1587;
        v1587 = 0l;
        while (while_method_1(v1587)){
            int v1589;
            v1589 = 0l;
            while (while_method_2(v1589)){
                assert("Tensor range check" && 0 <= v1587 && v1587 < 1l);
                assert("Tensor range check" && 0 <= v1589 && v1589 < 4l);
                int v1591;
                v1591 = 4l * v1587;
                int v1592;
                v1592 = v1591 + v1589;
                bool v1593;
                v1593 = v1551[v1592];
                int v1594;
                if (v1593){
                    v1594 = 1l;
                } else {
                    v1594 = 0l;
                }
                assert("Tensor range check" && 0 <= v1587 && v1587 < 1l);
                assert("Tensor range check" && 0 <= v1589 && v1589 < 4l);
                v1586[v1592] = v1594;
                v1589 += 1l ;
            }
            v1587 += 1l ;
        }
        int v1595;
        v1595 = 0l;
        int v1596;
        v1596 = 0l;
        while (while_method_1(v1596)){
            int v1598;
            v1598 = 0l;
            while (while_method_2(v1598)){
                assert("Tensor range check" && 0 <= v1596 && v1596 < 1l);
                assert("Tensor range check" && 0 <= v1598 && v1598 < 4l);
                int v1600;
                v1600 = 4l * v1596;
                int v1601;
                v1601 = v1600 + v1598;
                int v1602;
                v1602 = v1586[v1601];
                int v1603;
                v1603 = v1595 + v1602;
                v1595 = v1603;
                v1598 += 1l ;
            }
            v1596 += 1l ;
        }
        auto v1604 = cooperative_groups::coalesced_threads();
        int v1605;
        v1605 = threadIdx.x;
        auto v1606 = cooperative_groups::labeled_partition(v1604,v1605);
        int v1607;
        v1607 = cooperative_groups::reduce(v1606, v1595, v1396);
        float v1608;
        v1608 = (float)v1607;
        float v1609;
        v1609 = 1.0f / v1608;
        float v1610[4l];
        int v1611;
        v1611 = 0l;
        while (while_method_1(v1611)){
            int v1613;
            v1613 = 0l;
            while (while_method_2(v1613)){
                assert("Tensor range check" && 0 <= v1611 && v1611 < 1l);
                assert("Tensor range check" && 0 <= v1613 && v1613 < 4l);
                int v1615;
                v1615 = 4l * v1611;
                int v1616;
                v1616 = v1615 + v1613;
                float v1617;
                v1617 = v1561[v1616];
                bool v1618;
                v1618 = v1551[v1616];
                bool v1619;
                v1619 = v1618 == false;
                float v1624;
                if (v1619){
                    v1624 = 0.0f;
                } else {
                    bool v1620;
                    v1620 = v1585 == 0.0f;
                    bool v1621;
                    v1621 = v1620 != true;
                    if (v1621){
                        float v1622;
                        v1622 = v1617 / v1585;
                        v1624 = v1622;
                    } else {
                        v1624 = v1609;
                    }
                }
                assert("Tensor range check" && 0 <= v1611 && v1611 < 1l);
                assert("Tensor range check" && 0 <= v1613 && v1613 < 4l);
                v1610[v1616] = v1624;
                v1613 += 1l ;
            }
            v1611 += 1l ;
        }
        float v1625; int v1626;
        Tuple4 tmp51 = Tuple4{0.0f, 2147483647l};
        v1625 = tmp51.v0; v1626 = tmp51.v1;
        int v1627;
        v1627 = 0l;
        while (while_method_1(v1627)){
            int v1629;
            v1629 = 0l;
            while (while_method_2(v1629)){
                assert("Tensor range check" && 0 <= v1627 && v1627 < 1l);
                assert("Tensor range check" && 0 <= v1629 && v1629 < 4l);
                int v1631;
                v1631 = 4l * v1627;
                int v1632;
                v1632 = v1631 + v1629;
                float v1633;
                v1633 = v1400[v1632];
                int v1634;
                v1634 = v1296[v1632];
                bool v1635;
                v1635 = v1626 == v1547;
                float v1639; int v1640;
                if (v1635){
                    v1639 = v1625; v1640 = v1626;
                } else {
                    bool v1636;
                    v1636 = v1634 == v1547;
                    if (v1636){
                        v1639 = v1633; v1640 = v1634;
                    } else {
                        v1639 = v1625; v1640 = v1626;
                    }
                }
                v1625 = v1639;
                v1626 = v1640;
                v1629 += 1l ;
            }
            v1627 += 1l ;
        }
        auto v1641 = cooperative_groups::coalesced_threads();
        int v1642;
        v1642 = threadIdx.x;
        auto v1643 = cooperative_groups::labeled_partition(v1641,v1642);
        Closure6 v1644{v1547};
        float v1645; int v1646;
        Tuple4 tmp52 = cooperative_groups::reduce(v1643, Tuple4{v1625, v1626}, v1644);
        v1645 = tmp52.v0; v1646 = tmp52.v1;
        bool v1647;
        v1647 = v1646 == 2147483647l;
        bool v1648;
        v1648 = v1647 != true;
        bool v1649;
        v1649 = v1648 == false;
        if (v1649){
            assert("Expected a valid action id in get_action." && v1648);
        } else {
        }
        float v1651; int v1652;
        Tuple4 tmp53 = Tuple4{0.0f, 2147483647l};
        v1651 = tmp53.v0; v1652 = tmp53.v1;
        int v1653;
        v1653 = 0l;
        while (while_method_1(v1653)){
            int v1655;
            v1655 = 0l;
            while (while_method_2(v1655)){
                assert("Tensor range check" && 0 <= v1653 && v1653 < 1l);
                assert("Tensor range check" && 0 <= v1655 && v1655 < 4l);
                int v1657;
                v1657 = 4l * v1653;
                int v1658;
                v1658 = v1657 + v1655;
                float v1659;
                v1659 = v1610[v1658];
                int v1660;
                v1660 = v1296[v1658];
                bool v1661;
                v1661 = v1652 == v1547;
                float v1665; int v1666;
                if (v1661){
                    v1665 = v1651; v1666 = v1652;
                } else {
                    bool v1662;
                    v1662 = v1660 == v1547;
                    if (v1662){
                        v1665 = v1659; v1666 = v1660;
                    } else {
                        v1665 = v1651; v1666 = v1652;
                    }
                }
                v1651 = v1665;
                v1652 = v1666;
                v1655 += 1l ;
            }
            v1653 += 1l ;
        }
        auto v1667 = cooperative_groups::coalesced_threads();
        int v1668;
        v1668 = threadIdx.x;
        auto v1669 = cooperative_groups::labeled_partition(v1667,v1668);
        float v1670; int v1671;
        Tuple4 tmp54 = cooperative_groups::reduce(v1669, Tuple4{v1651, v1652}, v1644);
        v1670 = tmp54.v0; v1671 = tmp54.v1;
        bool v1672;
        v1672 = v1671 == 2147483647l;
        bool v1673;
        v1673 = v1672 != true;
        bool v1674;
        v1674 = v1673 == false;
        if (v1674){
            assert("Expected a valid action id in get_action." && v1673);
        } else {
        }
        assert("Tensor range check" && 0 <= v1288 && v1288 < 1l);
        v1272[v1290] = v1670;
        v1273[v1290] = v1645;
        v1274[v1290] = v1547;
        v1288 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1676;
    v1676 = threadIdx.x;
    assert("Tensor range check" && 0 <= v1676 && v1676 < 32l);
    float v1677;
    v1677 = v1272[v1676];
    float v1678;
    v1678 = v1273[v1676];
    int v1679;
    v1679 = v1274[v1676];
    v1270[0l] = Tuple0{v1677, v1678, v1679};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float v1680; float v1681; int v1682;
    Tuple0 tmp55 = v1270[0l];
    v1680 = tmp55.v0; v1681 = tmp55.v1; v1682 = tmp55.v2;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v17, v18, v19, v20, v1269, v1268, v1682, v1680, v1681);
    int v1683;
    v1683 = 457l;
    int v1684;
    v1684 = 0l;
    Tuple0 v1685[1l];
    __shared__ Tuple1 v1686[32l];
    __shared__ float v1687[32l];
    __shared__ float v1688[32l];
    __shared__ int v1689[32l];
    int v1690;
    v1690 = threadIdx.x;
    float * v1691;
    v1691 = v8+1828l;
    float * v1693;
    v1693 = v9+1828l;
    assert("Tensor range check" && 0 <= v1690 && v1690 < 32l);
    v1686[v1690] = Tuple1{v1691, v1693};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1695;
    v1695 = threadIdx.x;
    bool v1696;
    v1696 = 0l <= v1695;
    bool v1697;
    v1697 = v1696 == false;
    if (v1697){
        assert("The index needs to be zero or positive." && v1696);
    } else {
    }
    int v1699;
    v1699 = v1695 % 1l;
    bool v1700;
    v1700 = v1695 < 32l;
    bool v1701;
    v1701 = v1700 == false;
    if (v1701){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1700);
    } else {
    }
    assert("Tensor range check" && 0 <= v1695 && v1695 < 32l);
    assert("Tensor range check" && 0 <= v1695 && v1695 < 32l);
    int v1703;
    v1703 = 0l;
    while (while_method_1(v1703)){
        assert("Tensor range check" && 0 <= v1703 && v1703 < 1l);
        int v1705;
        v1705 = v1703 + v1695;
        float * v1706; float * v1707;
        Tuple1 tmp56 = v1686[v1705];
        v1706 = tmp56.v0; v1707 = tmp56.v1;
        assert("Tensor range check" && 0 <= v1699 && v1699 < 1l);
        int v1708;
        v1708 = 4l * v1699;
        float v1709[4l];
        float v1710[4l];
        int v1711[4l];
        int v1712;
        v1712 = 0l;
        while (while_method_1(v1712)){
            assert("Tensor range check" && 0 <= v1712 && v1712 < 1l);
            int v1714;
            v1714 = 4l * v1712;
            assert("Tensor range check" && 0 <= v1712 && v1712 < 1l);
            int v1715;
            v1715 = v1714 + v1708;
            int4* v1716;
            v1716 = reinterpret_cast<int4*>(v1706 + v1715);
            int4* v1717;
            v1717 = reinterpret_cast<int4*>(v1709 + v1714);
            assert("Pointer alignment check" && (unsigned long long)(v1716) % 4l == 0 && (unsigned long long)(v1717) % 4l == 0);
            *v1717 = *v1716;
            int4* v1718;
            v1718 = reinterpret_cast<int4*>(v1707 + v1715);
            int4* v1719;
            v1719 = reinterpret_cast<int4*>(v1710 + v1714);
            assert("Pointer alignment check" && (unsigned long long)(v1718) % 4l == 0 && (unsigned long long)(v1719) % 4l == 0);
            *v1719 = *v1718;
            v1712 += 1l ;
        }
        int v1720;
        v1720 = 0l;
        while (while_method_1(v1720)){
            int v1722;
            v1722 = 0l;
            while (while_method_2(v1722)){
                bool v1724;
                v1724 = 0l <= v1722;
                bool v1726;
                if (v1724){
                    bool v1725;
                    v1725 = v1722 < 4l;
                    v1726 = v1725;
                } else {
                    v1726 = false;
                }
                bool v1727;
                v1727 = v1726 == false;
                if (v1727){
                    assert("The indices should be inside the range of the dimension." && v1726);
                } else {
                }
                bool v1729;
                v1729 = 0l <= v1699;
                bool v1731;
                if (v1729){
                    bool v1730;
                    v1730 = v1699 < 1l;
                    v1731 = v1730;
                } else {
                    v1731 = false;
                }
                bool v1732;
                v1732 = v1731 == false;
                if (v1732){
                    assert("The indices should be inside the range of the dimension." && v1731);
                } else {
                }
                int v1734;
                v1734 = v1699 * 4l;
                int v1735;
                v1735 = v1722 + v1734;
                bool v1736;
                v1736 = 0l <= v1720;
                bool v1738;
                if (v1736){
                    bool v1737;
                    v1737 = v1720 < 1l;
                    v1738 = v1737;
                } else {
                    v1738 = false;
                }
                bool v1739;
                v1739 = v1738 == false;
                if (v1739){
                    assert("The indices should be inside the range of the dimension." && v1738);
                } else {
                }
                int v1741;
                v1741 = v1720 * 4l;
                int v1742;
                v1742 = v1735 + v1741;
                assert("Tensor range check" && 0 <= v1720 && v1720 < 1l);
                assert("Tensor range check" && 0 <= v1722 && v1722 < 4l);
                int v1743;
                v1743 = 4l * v1720;
                int v1744;
                v1744 = v1743 + v1722;
                v1711[v1744] = v1742;
                v1722 += 1l ;
            }
            v1720 += 1l ;
        }
        bool v1745;
        v1745 = 0l <= v1703;
        bool v1747;
        if (v1745){
            bool v1746;
            v1746 = v1703 < 1l;
            v1747 = v1746;
        } else {
            v1747 = false;
        }
        bool v1748;
        v1748 = v1747 == false;
        if (v1748){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1747);
        } else {
        }
        bool v1750;
        v1750 = v1696 && v1700;
        bool v1751;
        v1751 = v1750 == false;
        if (v1751){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1750);
        } else {
        }
        int v1753;
        v1753 = v1695 + v1703;
        bool v1754[4l];
        int v1755;
        v1755 = 0l;
        while (while_method_1(v1755)){
            int v1757;
            v1757 = 0l;
            while (while_method_2(v1757)){
                assert("Tensor range check" && 0 <= v1755 && v1755 < 1l);
                assert("Tensor range check" && 0 <= v1757 && v1757 < 4l);
                int v1759;
                v1759 = 4l * v1755;
                int v1760;
                v1760 = v1759 + v1757;
                float v1761;
                v1761 = v1709[v1760];
                int v1762;
                v1762 = v1711[v1760];
                bool v1763;
                v1763 = v1762 < 3l;
                assert("Tensor range check" && 0 <= v1755 && v1755 < 1l);
                assert("Tensor range check" && 0 <= v1757 && v1757 < 4l);
                v1754[v1760] = v1763;
                v1757 += 1l ;
            }
            v1755 += 1l ;
        }
        float v1764[4l];
        int v1765;
        v1765 = 0l;
        while (while_method_1(v1765)){
            int v1767;
            v1767 = 0l;
            while (while_method_2(v1767)){
                assert("Tensor range check" && 0 <= v1765 && v1765 < 1l);
                assert("Tensor range check" && 0 <= v1767 && v1767 < 4l);
                int v1769;
                v1769 = 4l * v1765;
                int v1770;
                v1770 = v1769 + v1767;
                float v1771;
                v1771 = v1709[v1770];
                bool v1772;
                v1772 = v1754[v1770];
                float v1775;
                if (v1772){
                    bool v1773;
                    v1773 = 0.0f >= v1771;
                    if (v1773){
                        v1775 = 0.0f;
                    } else {
                        v1775 = v1771;
                    }
                } else {
                    v1775 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1765 && v1765 < 1l);
                assert("Tensor range check" && 0 <= v1767 && v1767 < 4l);
                v1764[v1770] = v1775;
                v1767 += 1l ;
            }
            v1765 += 1l ;
        }
        float v1776;
        v1776 = 0.0f;
        int v1777;
        v1777 = 0l;
        while (while_method_1(v1777)){
            int v1779;
            v1779 = 0l;
            while (while_method_2(v1779)){
                assert("Tensor range check" && 0 <= v1777 && v1777 < 1l);
                assert("Tensor range check" && 0 <= v1779 && v1779 < 4l);
                int v1781;
                v1781 = 4l * v1777;
                int v1782;
                v1782 = v1781 + v1779;
                float v1783;
                v1783 = v1764[v1782];
                float v1784;
                v1784 = v1776 + v1783;
                v1776 = v1784;
                v1779 += 1l ;
            }
            v1777 += 1l ;
        }
        auto v1785 = cooperative_groups::coalesced_threads();
        int v1786;
        v1786 = threadIdx.x;
        auto v1787 = cooperative_groups::labeled_partition(v1785,v1786);
        Closure0 v1788{};
        float v1789;
        v1789 = cooperative_groups::reduce(v1787, v1776, v1788);
        int v1790[4l];
        int v1791;
        v1791 = 0l;
        while (while_method_1(v1791)){
            int v1793;
            v1793 = 0l;
            while (while_method_2(v1793)){
                assert("Tensor range check" && 0 <= v1791 && v1791 < 1l);
                assert("Tensor range check" && 0 <= v1793 && v1793 < 4l);
                int v1795;
                v1795 = 4l * v1791;
                int v1796;
                v1796 = v1795 + v1793;
                bool v1797;
                v1797 = v1754[v1796];
                int v1798;
                if (v1797){
                    v1798 = 1l;
                } else {
                    v1798 = 0l;
                }
                assert("Tensor range check" && 0 <= v1791 && v1791 < 1l);
                assert("Tensor range check" && 0 <= v1793 && v1793 < 4l);
                v1790[v1796] = v1798;
                v1793 += 1l ;
            }
            v1791 += 1l ;
        }
        int v1799;
        v1799 = 0l;
        int v1800;
        v1800 = 0l;
        while (while_method_1(v1800)){
            int v1802;
            v1802 = 0l;
            while (while_method_2(v1802)){
                assert("Tensor range check" && 0 <= v1800 && v1800 < 1l);
                assert("Tensor range check" && 0 <= v1802 && v1802 < 4l);
                int v1804;
                v1804 = 4l * v1800;
                int v1805;
                v1805 = v1804 + v1802;
                int v1806;
                v1806 = v1790[v1805];
                int v1807;
                v1807 = v1799 + v1806;
                v1799 = v1807;
                v1802 += 1l ;
            }
            v1800 += 1l ;
        }
        auto v1808 = cooperative_groups::coalesced_threads();
        int v1809;
        v1809 = threadIdx.x;
        auto v1810 = cooperative_groups::labeled_partition(v1808,v1809);
        Closure1 v1811{};
        int v1812;
        v1812 = cooperative_groups::reduce(v1810, v1799, v1811);
        float v1813;
        v1813 = (float)v1812;
        float v1814;
        v1814 = 1.0f / v1813;
        float v1815[4l];
        int v1816;
        v1816 = 0l;
        while (while_method_1(v1816)){
            int v1818;
            v1818 = 0l;
            while (while_method_2(v1818)){
                assert("Tensor range check" && 0 <= v1816 && v1816 < 1l);
                assert("Tensor range check" && 0 <= v1818 && v1818 < 4l);
                int v1820;
                v1820 = 4l * v1816;
                int v1821;
                v1821 = v1820 + v1818;
                float v1822;
                v1822 = v1764[v1821];
                bool v1823;
                v1823 = v1754[v1821];
                bool v1824;
                v1824 = v1823 == false;
                float v1829;
                if (v1824){
                    v1829 = 0.0f;
                } else {
                    bool v1825;
                    v1825 = v1789 == 0.0f;
                    bool v1826;
                    v1826 = v1825 != true;
                    if (v1826){
                        float v1827;
                        v1827 = v1822 / v1789;
                        v1829 = v1827;
                    } else {
                        v1829 = v1814;
                    }
                }
                assert("Tensor range check" && 0 <= v1816 && v1816 < 1l);
                assert("Tensor range check" && 0 <= v1818 && v1818 < 4l);
                v1815[v1821] = v1829;
                v1818 += 1l ;
            }
            v1816 += 1l ;
        }
        float v1830[4l];
        float v1831;
        v1831 = 0.0f;
        int v1832;
        v1832 = 0l;
        while (while_method_1(v1832)){
            assert("Tensor range check" && 0 <= v1832 && v1832 < 1l);
            int v1834;
            v1834 = 4l * v1832;
            assert("Tensor range check" && 0 <= v1832 && v1832 < 1l);
            int v1835; float v1836;
            Tuple2 tmp57 = Tuple2{0l, 0.0f};
            v1835 = tmp57.v0; v1836 = tmp57.v1;
            while (while_method_2(v1835)){
                assert("Tensor range check" && 0 <= v1835 && v1835 < 4l);
                int v1838;
                v1838 = v1835 + v1834;
                float v1839;
                v1839 = v1815[v1838];
                float v1840;
                v1840 = v1836 + v1839;
                v1836 = v1840;
                v1835 += 1l ;
            }
            auto v1841 = cooperative_groups::coalesced_threads();
            int v1842;
            v1842 = threadIdx.x;
            auto v1843 = cooperative_groups::labeled_partition(v1841,v1842);
            Closure2 v1844{};
            float v1845;
            v1845 = cooperative_groups::inclusive_scan(v1843, v1836, v1844);
            float v1846;
            v1846 = v1843.shfl_up(v1845,1);
            bool v1847;
            v1847 = v1843.thread_rank() == 0;
            float v1848;
            if (v1847){
                v1848 = 0.0f;
            } else {
                v1848 = v1846;
            }
            float v1849;
            v1849 = v1843.shfl(v1845,v1843.num_threads()-1);
            float v1850;
            v1850 = v1831 + v1848;
            int v1851; float v1852;
            Tuple2 tmp58 = Tuple2{0l, v1850};
            v1851 = tmp58.v0; v1852 = tmp58.v1;
            while (while_method_2(v1851)){
                assert("Tensor range check" && 0 <= v1851 && v1851 < 4l);
                int v1854;
                v1854 = v1851 + v1834;
                float v1855;
                v1855 = v1815[v1854];
                float v1856;
                v1856 = v1852 + v1855;
                assert("Tensor range check" && 0 <= v1851 && v1851 < 4l);
                v1830[v1854] = v1856;
                v1852 = v1856;
                v1851 += 1l ;
            }
            float v1857;
            v1857 = v1831 + v1849;
            v1831 = v1857;
            v1832 += 1l ;
        }
        float v1858[4l];
        bool v1859[4l];
        int v1860;
        v1860 = 0l;
        while (while_method_1(v1860)){
            int v1862;
            v1862 = 0l;
            while (while_method_2(v1862)){
                assert("Tensor range check" && 0 <= v1860 && v1860 < 1l);
                assert("Tensor range check" && 0 <= v1862 && v1862 < 4l);
                int v1864;
                v1864 = 4l * v1860;
                int v1865;
                v1865 = v1864 + v1862;
                float v1866;
                v1866 = v1830[v1865];
                float v1867;
                v1867 = v1815[v1865];
                bool v1868;
                v1868 = v1867 > 0.0f;
                assert("Tensor range check" && 0 <= v1860 && v1860 < 1l);
                assert("Tensor range check" && 0 <= v1862 && v1862 < 4l);
                v1858[v1865] = v1866;
                v1859[v1865] = v1868;
                v1862 += 1l ;
            }
            v1860 += 1l ;
        }
        float v1869; bool v1870;
        Tuple3 tmp59 = Tuple3{-1.0f / 0.0f, false};
        v1869 = tmp59.v0; v1870 = tmp59.v1;
        int v1871;
        v1871 = 0l;
        while (while_method_1(v1871)){
            int v1873;
            v1873 = 0l;
            while (while_method_2(v1873)){
                assert("Tensor range check" && 0 <= v1871 && v1871 < 1l);
                assert("Tensor range check" && 0 <= v1873 && v1873 < 4l);
                int v1875;
                v1875 = 4l * v1871;
                int v1876;
                v1876 = v1875 + v1873;
                float v1877;
                v1877 = v1858[v1876];
                bool v1878;
                v1878 = v1859[v1876];
                float v1885; bool v1886;
                if (v1870){
                    if (v1878){
                        bool v1879;
                        v1879 = v1869 >= v1877;
                        float v1880;
                        if (v1879){
                            v1880 = v1869;
                        } else {
                            v1880 = v1877;
                        }
                        v1885 = v1880; v1886 = true;
                    } else {
                        v1885 = v1869; v1886 = v1870;
                    }
                } else {
                    if (v1878){
                        v1885 = v1877; v1886 = v1878;
                    } else {
                        v1885 = v1869; v1886 = v1870;
                    }
                }
                v1869 = v1885;
                v1870 = v1886;
                v1873 += 1l ;
            }
            v1871 += 1l ;
        }
        auto v1887 = cooperative_groups::coalesced_threads();
        int v1888;
        v1888 = threadIdx.x;
        auto v1889 = cooperative_groups::labeled_partition(v1887,v1888);
        Closure3 v1890{};
        float v1891; bool v1892;
        Tuple3 tmp60 = cooperative_groups::reduce(v1889, Tuple3{v1869, v1870}, v1890);
        v1891 = tmp60.v0; v1892 = tmp60.v1;
        bool v1893;
        v1893 = v1892 == false;
        if (v1893){
            assert("The local reduce must be true." && v1892);
        } else {
        }
        float v1895[4l];
        int v1896[4l];
        int v1897;
        v1897 = 0l;
        while (while_method_1(v1897)){
            int v1899;
            v1899 = 0l;
            while (while_method_2(v1899)){
                assert("Tensor range check" && 0 <= v1897 && v1897 < 1l);
                assert("Tensor range check" && 0 <= v1899 && v1899 < 4l);
                int v1901;
                v1901 = 4l * v1897;
                int v1902;
                v1902 = v1901 + v1899;
                int v1903;
                v1903 = v1711[v1902];
                float v1904;
                v1904 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v1897 && v1897 < 1l);
                assert("Tensor range check" && 0 <= v1899 && v1899 < 4l);
                v1895[v1902] = v1904;
                v1896[v1902] = v1903;
                v1899 += 1l ;
            }
            v1897 += 1l ;
        }
        float v1905; int v1906;
        Tuple4 tmp61 = Tuple4{0.0f, 2147483647l};
        v1905 = tmp61.v0; v1906 = tmp61.v1;
        int v1907;
        v1907 = 0l;
        while (while_method_1(v1907)){
            int v1909;
            v1909 = 0l;
            while (while_method_2(v1909)){
                assert("Tensor range check" && 0 <= v1907 && v1907 < 1l);
                assert("Tensor range check" && 0 <= v1909 && v1909 < 4l);
                int v1911;
                v1911 = 4l * v1907;
                int v1912;
                v1912 = v1911 + v1909;
                float v1913;
                v1913 = v1895[v1912];
                int v1914;
                v1914 = v1896[v1912];
                bool v1915;
                v1915 = v1906 < v1914;
                float v1916; int v1917;
                if (v1915){
                    v1916 = v1905; v1917 = v1906;
                } else {
                    v1916 = v1913; v1917 = v1914;
                }
                v1905 = v1916;
                v1906 = v1917;
                v1909 += 1l ;
            }
            v1907 += 1l ;
        }
        auto v1918 = cooperative_groups::coalesced_threads();
        int v1919;
        v1919 = threadIdx.x;
        auto v1920 = cooperative_groups::labeled_partition(v1918,v1919);
        Closure4 v1921{};
        float v1922; int v1923;
        Tuple4 tmp62 = cooperative_groups::reduce(v1920, Tuple4{v1905, v1906}, v1921);
        v1922 = tmp62.v0; v1923 = tmp62.v1;
        float v1924;
        v1924 = v1891 * v1922;
        int v1925[4l];
        bool v1926[4l];
        int v1927;
        v1927 = 0l;
        while (while_method_1(v1927)){
            int v1929;
            v1929 = 0l;
            while (while_method_2(v1929)){
                assert("Tensor range check" && 0 <= v1927 && v1927 < 1l);
                assert("Tensor range check" && 0 <= v1929 && v1929 < 4l);
                int v1931;
                v1931 = 4l * v1927;
                int v1932;
                v1932 = v1931 + v1929;
                float v1933;
                v1933 = v1858[v1932];
                bool v1934;
                v1934 = v1859[v1932];
                int v1935;
                v1935 = v1711[v1932];
                int v1938; bool v1939;
                if (v1934){
                    float v1936;
                    v1936 = v1933 - v1924;
                    bool v1937;
                    v1937 = v1936 >= 0.0f;
                    v1938 = v1935; v1939 = v1937;
                } else {
                    v1938 = 2147483647l; v1939 = false;
                }
                assert("Tensor range check" && 0 <= v1927 && v1927 < 1l);
                assert("Tensor range check" && 0 <= v1929 && v1929 < 4l);
                v1925[v1932] = v1938;
                v1926[v1932] = v1939;
                v1929 += 1l ;
            }
            v1927 += 1l ;
        }
        int v1940; bool v1941;
        Tuple5 tmp63 = Tuple5{2147483647l, false};
        v1940 = tmp63.v0; v1941 = tmp63.v1;
        int v1942;
        v1942 = 0l;
        while (while_method_1(v1942)){
            int v1944;
            v1944 = 0l;
            while (while_method_2(v1944)){
                assert("Tensor range check" && 0 <= v1942 && v1942 < 1l);
                assert("Tensor range check" && 0 <= v1944 && v1944 < 4l);
                int v1946;
                v1946 = 4l * v1942;
                int v1947;
                v1947 = v1946 + v1944;
                int v1948;
                v1948 = v1925[v1947];
                bool v1949;
                v1949 = v1926[v1947];
                int v1956; bool v1957;
                if (v1941){
                    if (v1949){
                        bool v1950;
                        v1950 = v1940 < v1948;
                        int v1951;
                        if (v1950){
                            v1951 = v1940;
                        } else {
                            v1951 = v1948;
                        }
                        v1956 = v1951; v1957 = true;
                    } else {
                        v1956 = v1940; v1957 = v1941;
                    }
                } else {
                    if (v1949){
                        v1956 = v1948; v1957 = v1949;
                    } else {
                        v1956 = v1940; v1957 = v1941;
                    }
                }
                v1940 = v1956;
                v1941 = v1957;
                v1944 += 1l ;
            }
            v1942 += 1l ;
        }
        auto v1958 = cooperative_groups::coalesced_threads();
        int v1959;
        v1959 = threadIdx.x;
        auto v1960 = cooperative_groups::labeled_partition(v1958,v1959);
        Closure5 v1961{};
        int v1962; bool v1963;
        Tuple5 tmp64 = cooperative_groups::reduce(v1960, Tuple5{v1940, v1941}, v1961);
        v1962 = tmp64.v0; v1963 = tmp64.v1;
        bool v1964;
        v1964 = v1963 == false;
        if (v1964){
            assert("The local reduce must be true." && v1963);
        } else {
        }
        bool v1966[4l];
        int v1967;
        v1967 = 0l;
        while (while_method_1(v1967)){
            int v1969;
            v1969 = 0l;
            while (while_method_2(v1969)){
                assert("Tensor range check" && 0 <= v1967 && v1967 < 1l);
                assert("Tensor range check" && 0 <= v1969 && v1969 < 4l);
                int v1971;
                v1971 = 4l * v1967;
                int v1972;
                v1972 = v1971 + v1969;
                float v1973;
                v1973 = v1710[v1972];
                int v1974;
                v1974 = v1711[v1972];
                bool v1975;
                v1975 = v1974 < 3l;
                assert("Tensor range check" && 0 <= v1967 && v1967 < 1l);
                assert("Tensor range check" && 0 <= v1969 && v1969 < 4l);
                v1966[v1972] = v1975;
                v1969 += 1l ;
            }
            v1967 += 1l ;
        }
        float v1976[4l];
        int v1977;
        v1977 = 0l;
        while (while_method_1(v1977)){
            int v1979;
            v1979 = 0l;
            while (while_method_2(v1979)){
                assert("Tensor range check" && 0 <= v1977 && v1977 < 1l);
                assert("Tensor range check" && 0 <= v1979 && v1979 < 4l);
                int v1981;
                v1981 = 4l * v1977;
                int v1982;
                v1982 = v1981 + v1979;
                float v1983;
                v1983 = v1710[v1982];
                bool v1984;
                v1984 = v1966[v1982];
                float v1987;
                if (v1984){
                    bool v1985;
                    v1985 = 0.0f >= v1983;
                    if (v1985){
                        v1987 = 0.0f;
                    } else {
                        v1987 = v1983;
                    }
                } else {
                    v1987 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1977 && v1977 < 1l);
                assert("Tensor range check" && 0 <= v1979 && v1979 < 4l);
                v1976[v1982] = v1987;
                v1979 += 1l ;
            }
            v1977 += 1l ;
        }
        float v1988;
        v1988 = 0.0f;
        int v1989;
        v1989 = 0l;
        while (while_method_1(v1989)){
            int v1991;
            v1991 = 0l;
            while (while_method_2(v1991)){
                assert("Tensor range check" && 0 <= v1989 && v1989 < 1l);
                assert("Tensor range check" && 0 <= v1991 && v1991 < 4l);
                int v1993;
                v1993 = 4l * v1989;
                int v1994;
                v1994 = v1993 + v1991;
                float v1995;
                v1995 = v1976[v1994];
                float v1996;
                v1996 = v1988 + v1995;
                v1988 = v1996;
                v1991 += 1l ;
            }
            v1989 += 1l ;
        }
        auto v1997 = cooperative_groups::coalesced_threads();
        int v1998;
        v1998 = threadIdx.x;
        auto v1999 = cooperative_groups::labeled_partition(v1997,v1998);
        float v2000;
        v2000 = cooperative_groups::reduce(v1999, v1988, v1788);
        int v2001[4l];
        int v2002;
        v2002 = 0l;
        while (while_method_1(v2002)){
            int v2004;
            v2004 = 0l;
            while (while_method_2(v2004)){
                assert("Tensor range check" && 0 <= v2002 && v2002 < 1l);
                assert("Tensor range check" && 0 <= v2004 && v2004 < 4l);
                int v2006;
                v2006 = 4l * v2002;
                int v2007;
                v2007 = v2006 + v2004;
                bool v2008;
                v2008 = v1966[v2007];
                int v2009;
                if (v2008){
                    v2009 = 1l;
                } else {
                    v2009 = 0l;
                }
                assert("Tensor range check" && 0 <= v2002 && v2002 < 1l);
                assert("Tensor range check" && 0 <= v2004 && v2004 < 4l);
                v2001[v2007] = v2009;
                v2004 += 1l ;
            }
            v2002 += 1l ;
        }
        int v2010;
        v2010 = 0l;
        int v2011;
        v2011 = 0l;
        while (while_method_1(v2011)){
            int v2013;
            v2013 = 0l;
            while (while_method_2(v2013)){
                assert("Tensor range check" && 0 <= v2011 && v2011 < 1l);
                assert("Tensor range check" && 0 <= v2013 && v2013 < 4l);
                int v2015;
                v2015 = 4l * v2011;
                int v2016;
                v2016 = v2015 + v2013;
                int v2017;
                v2017 = v2001[v2016];
                int v2018;
                v2018 = v2010 + v2017;
                v2010 = v2018;
                v2013 += 1l ;
            }
            v2011 += 1l ;
        }
        auto v2019 = cooperative_groups::coalesced_threads();
        int v2020;
        v2020 = threadIdx.x;
        auto v2021 = cooperative_groups::labeled_partition(v2019,v2020);
        int v2022;
        v2022 = cooperative_groups::reduce(v2021, v2010, v1811);
        float v2023;
        v2023 = (float)v2022;
        float v2024;
        v2024 = 1.0f / v2023;
        float v2025[4l];
        int v2026;
        v2026 = 0l;
        while (while_method_1(v2026)){
            int v2028;
            v2028 = 0l;
            while (while_method_2(v2028)){
                assert("Tensor range check" && 0 <= v2026 && v2026 < 1l);
                assert("Tensor range check" && 0 <= v2028 && v2028 < 4l);
                int v2030;
                v2030 = 4l * v2026;
                int v2031;
                v2031 = v2030 + v2028;
                float v2032;
                v2032 = v1976[v2031];
                bool v2033;
                v2033 = v1966[v2031];
                bool v2034;
                v2034 = v2033 == false;
                float v2039;
                if (v2034){
                    v2039 = 0.0f;
                } else {
                    bool v2035;
                    v2035 = v2000 == 0.0f;
                    bool v2036;
                    v2036 = v2035 != true;
                    if (v2036){
                        float v2037;
                        v2037 = v2032 / v2000;
                        v2039 = v2037;
                    } else {
                        v2039 = v2024;
                    }
                }
                assert("Tensor range check" && 0 <= v2026 && v2026 < 1l);
                assert("Tensor range check" && 0 <= v2028 && v2028 < 4l);
                v2025[v2031] = v2039;
                v2028 += 1l ;
            }
            v2026 += 1l ;
        }
        float v2040; int v2041;
        Tuple4 tmp65 = Tuple4{0.0f, 2147483647l};
        v2040 = tmp65.v0; v2041 = tmp65.v1;
        int v2042;
        v2042 = 0l;
        while (while_method_1(v2042)){
            int v2044;
            v2044 = 0l;
            while (while_method_2(v2044)){
                assert("Tensor range check" && 0 <= v2042 && v2042 < 1l);
                assert("Tensor range check" && 0 <= v2044 && v2044 < 4l);
                int v2046;
                v2046 = 4l * v2042;
                int v2047;
                v2047 = v2046 + v2044;
                float v2048;
                v2048 = v1815[v2047];
                int v2049;
                v2049 = v1711[v2047];
                bool v2050;
                v2050 = v2041 == v1962;
                float v2054; int v2055;
                if (v2050){
                    v2054 = v2040; v2055 = v2041;
                } else {
                    bool v2051;
                    v2051 = v2049 == v1962;
                    if (v2051){
                        v2054 = v2048; v2055 = v2049;
                    } else {
                        v2054 = v2040; v2055 = v2041;
                    }
                }
                v2040 = v2054;
                v2041 = v2055;
                v2044 += 1l ;
            }
            v2042 += 1l ;
        }
        auto v2056 = cooperative_groups::coalesced_threads();
        int v2057;
        v2057 = threadIdx.x;
        auto v2058 = cooperative_groups::labeled_partition(v2056,v2057);
        Closure6 v2059{v1962};
        float v2060; int v2061;
        Tuple4 tmp66 = cooperative_groups::reduce(v2058, Tuple4{v2040, v2041}, v2059);
        v2060 = tmp66.v0; v2061 = tmp66.v1;
        bool v2062;
        v2062 = v2061 == 2147483647l;
        bool v2063;
        v2063 = v2062 != true;
        bool v2064;
        v2064 = v2063 == false;
        if (v2064){
            assert("Expected a valid action id in get_action." && v2063);
        } else {
        }
        float v2066; int v2067;
        Tuple4 tmp67 = Tuple4{0.0f, 2147483647l};
        v2066 = tmp67.v0; v2067 = tmp67.v1;
        int v2068;
        v2068 = 0l;
        while (while_method_1(v2068)){
            int v2070;
            v2070 = 0l;
            while (while_method_2(v2070)){
                assert("Tensor range check" && 0 <= v2068 && v2068 < 1l);
                assert("Tensor range check" && 0 <= v2070 && v2070 < 4l);
                int v2072;
                v2072 = 4l * v2068;
                int v2073;
                v2073 = v2072 + v2070;
                float v2074;
                v2074 = v2025[v2073];
                int v2075;
                v2075 = v1711[v2073];
                bool v2076;
                v2076 = v2067 == v1962;
                float v2080; int v2081;
                if (v2076){
                    v2080 = v2066; v2081 = v2067;
                } else {
                    bool v2077;
                    v2077 = v2075 == v1962;
                    if (v2077){
                        v2080 = v2074; v2081 = v2075;
                    } else {
                        v2080 = v2066; v2081 = v2067;
                    }
                }
                v2066 = v2080;
                v2067 = v2081;
                v2070 += 1l ;
            }
            v2068 += 1l ;
        }
        auto v2082 = cooperative_groups::coalesced_threads();
        int v2083;
        v2083 = threadIdx.x;
        auto v2084 = cooperative_groups::labeled_partition(v2082,v2083);
        float v2085; int v2086;
        Tuple4 tmp68 = cooperative_groups::reduce(v2084, Tuple4{v2066, v2067}, v2059);
        v2085 = tmp68.v0; v2086 = tmp68.v1;
        bool v2087;
        v2087 = v2086 == 2147483647l;
        bool v2088;
        v2088 = v2087 != true;
        bool v2089;
        v2089 = v2088 == false;
        if (v2089){
            assert("Expected a valid action id in get_action." && v2088);
        } else {
        }
        assert("Tensor range check" && 0 <= v1703 && v1703 < 1l);
        v1687[v1705] = v2085;
        v1688[v1705] = v2060;
        v1689[v1705] = v1962;
        v1703 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v2091;
    v2091 = threadIdx.x;
    assert("Tensor range check" && 0 <= v2091 && v2091 < 32l);
    float v2092;
    v2092 = v1687[v2091];
    float v2093;
    v2093 = v1688[v2091];
    int v2094;
    v2094 = v1689[v2091];
    v1685[0l] = Tuple0{v2092, v2093, v2094};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float v2095; float v2096; int v2097;
    Tuple0 tmp69 = v1685[0l];
    v2095 = tmp69.v0; v2096 = tmp69.v1; v2097 = tmp69.v2;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v17, v18, v19, v20, v1684, v1683, v2097, v2095, v2096);
    int v2098;
    v2098 = 3447l;
    int v2099;
    v2099 = 1l;
    Tuple0 v2100[1l];
    __shared__ Tuple1 v2101[32l];
    __shared__ float v2102[32l];
    __shared__ float v2103[32l];
    __shared__ int v2104[32l];
    int v2105;
    v2105 = threadIdx.x;
    float * v2106;
    v2106 = v8+13788l;
    float * v2108;
    v2108 = v9+13788l;
    assert("Tensor range check" && 0 <= v2105 && v2105 < 32l);
    v2101[v2105] = Tuple1{v2106, v2108};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v2110;
    v2110 = threadIdx.x;
    bool v2111;
    v2111 = 0l <= v2110;
    bool v2112;
    v2112 = v2111 == false;
    if (v2112){
        assert("The index needs to be zero or positive." && v2111);
    } else {
    }
    int v2114;
    v2114 = v2110 % 1l;
    bool v2115;
    v2115 = v2110 < 32l;
    bool v2116;
    v2116 = v2115 == false;
    if (v2116){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v2115);
    } else {
    }
    assert("Tensor range check" && 0 <= v2110 && v2110 < 32l);
    assert("Tensor range check" && 0 <= v2110 && v2110 < 32l);
    int v2118;
    v2118 = 0l;
    while (while_method_1(v2118)){
        assert("Tensor range check" && 0 <= v2118 && v2118 < 1l);
        int v2120;
        v2120 = v2118 + v2110;
        float * v2121; float * v2122;
        Tuple1 tmp70 = v2101[v2120];
        v2121 = tmp70.v0; v2122 = tmp70.v1;
        assert("Tensor range check" && 0 <= v2114 && v2114 < 1l);
        int v2123;
        v2123 = 4l * v2114;
        float v2124[4l];
        float v2125[4l];
        int v2126[4l];
        int v2127;
        v2127 = 0l;
        while (while_method_1(v2127)){
            assert("Tensor range check" && 0 <= v2127 && v2127 < 1l);
            int v2129;
            v2129 = 4l * v2127;
            assert("Tensor range check" && 0 <= v2127 && v2127 < 1l);
            int v2130;
            v2130 = v2129 + v2123;
            int4* v2131;
            v2131 = reinterpret_cast<int4*>(v2121 + v2130);
            int4* v2132;
            v2132 = reinterpret_cast<int4*>(v2124 + v2129);
            assert("Pointer alignment check" && (unsigned long long)(v2131) % 4l == 0 && (unsigned long long)(v2132) % 4l == 0);
            *v2132 = *v2131;
            int4* v2133;
            v2133 = reinterpret_cast<int4*>(v2122 + v2130);
            int4* v2134;
            v2134 = reinterpret_cast<int4*>(v2125 + v2129);
            assert("Pointer alignment check" && (unsigned long long)(v2133) % 4l == 0 && (unsigned long long)(v2134) % 4l == 0);
            *v2134 = *v2133;
            v2127 += 1l ;
        }
        int v2135;
        v2135 = 0l;
        while (while_method_1(v2135)){
            int v2137;
            v2137 = 0l;
            while (while_method_2(v2137)){
                bool v2139;
                v2139 = 0l <= v2137;
                bool v2141;
                if (v2139){
                    bool v2140;
                    v2140 = v2137 < 4l;
                    v2141 = v2140;
                } else {
                    v2141 = false;
                }
                bool v2142;
                v2142 = v2141 == false;
                if (v2142){
                    assert("The indices should be inside the range of the dimension." && v2141);
                } else {
                }
                bool v2144;
                v2144 = 0l <= v2114;
                bool v2146;
                if (v2144){
                    bool v2145;
                    v2145 = v2114 < 1l;
                    v2146 = v2145;
                } else {
                    v2146 = false;
                }
                bool v2147;
                v2147 = v2146 == false;
                if (v2147){
                    assert("The indices should be inside the range of the dimension." && v2146);
                } else {
                }
                int v2149;
                v2149 = v2114 * 4l;
                int v2150;
                v2150 = v2137 + v2149;
                bool v2151;
                v2151 = 0l <= v2135;
                bool v2153;
                if (v2151){
                    bool v2152;
                    v2152 = v2135 < 1l;
                    v2153 = v2152;
                } else {
                    v2153 = false;
                }
                bool v2154;
                v2154 = v2153 == false;
                if (v2154){
                    assert("The indices should be inside the range of the dimension." && v2153);
                } else {
                }
                int v2156;
                v2156 = v2135 * 4l;
                int v2157;
                v2157 = v2150 + v2156;
                assert("Tensor range check" && 0 <= v2135 && v2135 < 1l);
                assert("Tensor range check" && 0 <= v2137 && v2137 < 4l);
                int v2158;
                v2158 = 4l * v2135;
                int v2159;
                v2159 = v2158 + v2137;
                v2126[v2159] = v2157;
                v2137 += 1l ;
            }
            v2135 += 1l ;
        }
        bool v2160;
        v2160 = 0l <= v2118;
        bool v2162;
        if (v2160){
            bool v2161;
            v2161 = v2118 < 1l;
            v2162 = v2161;
        } else {
            v2162 = false;
        }
        bool v2163;
        v2163 = v2162 == false;
        if (v2163){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v2162);
        } else {
        }
        bool v2165;
        v2165 = v2111 && v2115;
        bool v2166;
        v2166 = v2165 == false;
        if (v2166){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v2165);
        } else {
        }
        int v2168;
        v2168 = v2110 + v2118;
        bool v2169[4l];
        int v2170;
        v2170 = 0l;
        while (while_method_1(v2170)){
            int v2172;
            v2172 = 0l;
            while (while_method_2(v2172)){
                assert("Tensor range check" && 0 <= v2170 && v2170 < 1l);
                assert("Tensor range check" && 0 <= v2172 && v2172 < 4l);
                int v2174;
                v2174 = 4l * v2170;
                int v2175;
                v2175 = v2174 + v2172;
                float v2176;
                v2176 = v2124[v2175];
                int v2177;
                v2177 = v2126[v2175];
                bool v2178;
                v2178 = v2177 < 3l;
                assert("Tensor range check" && 0 <= v2170 && v2170 < 1l);
                assert("Tensor range check" && 0 <= v2172 && v2172 < 4l);
                v2169[v2175] = v2178;
                v2172 += 1l ;
            }
            v2170 += 1l ;
        }
        float v2179[4l];
        int v2180;
        v2180 = 0l;
        while (while_method_1(v2180)){
            int v2182;
            v2182 = 0l;
            while (while_method_2(v2182)){
                assert("Tensor range check" && 0 <= v2180 && v2180 < 1l);
                assert("Tensor range check" && 0 <= v2182 && v2182 < 4l);
                int v2184;
                v2184 = 4l * v2180;
                int v2185;
                v2185 = v2184 + v2182;
                float v2186;
                v2186 = v2124[v2185];
                bool v2187;
                v2187 = v2169[v2185];
                float v2190;
                if (v2187){
                    bool v2188;
                    v2188 = 0.0f >= v2186;
                    if (v2188){
                        v2190 = 0.0f;
                    } else {
                        v2190 = v2186;
                    }
                } else {
                    v2190 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v2180 && v2180 < 1l);
                assert("Tensor range check" && 0 <= v2182 && v2182 < 4l);
                v2179[v2185] = v2190;
                v2182 += 1l ;
            }
            v2180 += 1l ;
        }
        float v2191;
        v2191 = 0.0f;
        int v2192;
        v2192 = 0l;
        while (while_method_1(v2192)){
            int v2194;
            v2194 = 0l;
            while (while_method_2(v2194)){
                assert("Tensor range check" && 0 <= v2192 && v2192 < 1l);
                assert("Tensor range check" && 0 <= v2194 && v2194 < 4l);
                int v2196;
                v2196 = 4l * v2192;
                int v2197;
                v2197 = v2196 + v2194;
                float v2198;
                v2198 = v2179[v2197];
                float v2199;
                v2199 = v2191 + v2198;
                v2191 = v2199;
                v2194 += 1l ;
            }
            v2192 += 1l ;
        }
        auto v2200 = cooperative_groups::coalesced_threads();
        int v2201;
        v2201 = threadIdx.x;
        auto v2202 = cooperative_groups::labeled_partition(v2200,v2201);
        Closure0 v2203{};
        float v2204;
        v2204 = cooperative_groups::reduce(v2202, v2191, v2203);
        int v2205[4l];
        int v2206;
        v2206 = 0l;
        while (while_method_1(v2206)){
            int v2208;
            v2208 = 0l;
            while (while_method_2(v2208)){
                assert("Tensor range check" && 0 <= v2206 && v2206 < 1l);
                assert("Tensor range check" && 0 <= v2208 && v2208 < 4l);
                int v2210;
                v2210 = 4l * v2206;
                int v2211;
                v2211 = v2210 + v2208;
                bool v2212;
                v2212 = v2169[v2211];
                int v2213;
                if (v2212){
                    v2213 = 1l;
                } else {
                    v2213 = 0l;
                }
                assert("Tensor range check" && 0 <= v2206 && v2206 < 1l);
                assert("Tensor range check" && 0 <= v2208 && v2208 < 4l);
                v2205[v2211] = v2213;
                v2208 += 1l ;
            }
            v2206 += 1l ;
        }
        int v2214;
        v2214 = 0l;
        int v2215;
        v2215 = 0l;
        while (while_method_1(v2215)){
            int v2217;
            v2217 = 0l;
            while (while_method_2(v2217)){
                assert("Tensor range check" && 0 <= v2215 && v2215 < 1l);
                assert("Tensor range check" && 0 <= v2217 && v2217 < 4l);
                int v2219;
                v2219 = 4l * v2215;
                int v2220;
                v2220 = v2219 + v2217;
                int v2221;
                v2221 = v2205[v2220];
                int v2222;
                v2222 = v2214 + v2221;
                v2214 = v2222;
                v2217 += 1l ;
            }
            v2215 += 1l ;
        }
        auto v2223 = cooperative_groups::coalesced_threads();
        int v2224;
        v2224 = threadIdx.x;
        auto v2225 = cooperative_groups::labeled_partition(v2223,v2224);
        Closure1 v2226{};
        int v2227;
        v2227 = cooperative_groups::reduce(v2225, v2214, v2226);
        float v2228;
        v2228 = (float)v2227;
        float v2229;
        v2229 = 1.0f / v2228;
        float v2230[4l];
        int v2231;
        v2231 = 0l;
        while (while_method_1(v2231)){
            int v2233;
            v2233 = 0l;
            while (while_method_2(v2233)){
                assert("Tensor range check" && 0 <= v2231 && v2231 < 1l);
                assert("Tensor range check" && 0 <= v2233 && v2233 < 4l);
                int v2235;
                v2235 = 4l * v2231;
                int v2236;
                v2236 = v2235 + v2233;
                float v2237;
                v2237 = v2179[v2236];
                bool v2238;
                v2238 = v2169[v2236];
                bool v2239;
                v2239 = v2238 == false;
                float v2244;
                if (v2239){
                    v2244 = 0.0f;
                } else {
                    bool v2240;
                    v2240 = v2204 == 0.0f;
                    bool v2241;
                    v2241 = v2240 != true;
                    if (v2241){
                        float v2242;
                        v2242 = v2237 / v2204;
                        v2244 = v2242;
                    } else {
                        v2244 = v2229;
                    }
                }
                assert("Tensor range check" && 0 <= v2231 && v2231 < 1l);
                assert("Tensor range check" && 0 <= v2233 && v2233 < 4l);
                v2230[v2236] = v2244;
                v2233 += 1l ;
            }
            v2231 += 1l ;
        }
        float v2245[4l];
        float v2246;
        v2246 = 0.0f;
        int v2247;
        v2247 = 0l;
        while (while_method_1(v2247)){
            assert("Tensor range check" && 0 <= v2247 && v2247 < 1l);
            int v2249;
            v2249 = 4l * v2247;
            assert("Tensor range check" && 0 <= v2247 && v2247 < 1l);
            int v2250; float v2251;
            Tuple2 tmp71 = Tuple2{0l, 0.0f};
            v2250 = tmp71.v0; v2251 = tmp71.v1;
            while (while_method_2(v2250)){
                assert("Tensor range check" && 0 <= v2250 && v2250 < 4l);
                int v2253;
                v2253 = v2250 + v2249;
                float v2254;
                v2254 = v2230[v2253];
                float v2255;
                v2255 = v2251 + v2254;
                v2251 = v2255;
                v2250 += 1l ;
            }
            auto v2256 = cooperative_groups::coalesced_threads();
            int v2257;
            v2257 = threadIdx.x;
            auto v2258 = cooperative_groups::labeled_partition(v2256,v2257);
            Closure2 v2259{};
            float v2260;
            v2260 = cooperative_groups::inclusive_scan(v2258, v2251, v2259);
            float v2261;
            v2261 = v2258.shfl_up(v2260,1);
            bool v2262;
            v2262 = v2258.thread_rank() == 0;
            float v2263;
            if (v2262){
                v2263 = 0.0f;
            } else {
                v2263 = v2261;
            }
            float v2264;
            v2264 = v2258.shfl(v2260,v2258.num_threads()-1);
            float v2265;
            v2265 = v2246 + v2263;
            int v2266; float v2267;
            Tuple2 tmp72 = Tuple2{0l, v2265};
            v2266 = tmp72.v0; v2267 = tmp72.v1;
            while (while_method_2(v2266)){
                assert("Tensor range check" && 0 <= v2266 && v2266 < 4l);
                int v2269;
                v2269 = v2266 + v2249;
                float v2270;
                v2270 = v2230[v2269];
                float v2271;
                v2271 = v2267 + v2270;
                assert("Tensor range check" && 0 <= v2266 && v2266 < 4l);
                v2245[v2269] = v2271;
                v2267 = v2271;
                v2266 += 1l ;
            }
            float v2272;
            v2272 = v2246 + v2264;
            v2246 = v2272;
            v2247 += 1l ;
        }
        float v2273[4l];
        bool v2274[4l];
        int v2275;
        v2275 = 0l;
        while (while_method_1(v2275)){
            int v2277;
            v2277 = 0l;
            while (while_method_2(v2277)){
                assert("Tensor range check" && 0 <= v2275 && v2275 < 1l);
                assert("Tensor range check" && 0 <= v2277 && v2277 < 4l);
                int v2279;
                v2279 = 4l * v2275;
                int v2280;
                v2280 = v2279 + v2277;
                float v2281;
                v2281 = v2245[v2280];
                float v2282;
                v2282 = v2230[v2280];
                bool v2283;
                v2283 = v2282 > 0.0f;
                assert("Tensor range check" && 0 <= v2275 && v2275 < 1l);
                assert("Tensor range check" && 0 <= v2277 && v2277 < 4l);
                v2273[v2280] = v2281;
                v2274[v2280] = v2283;
                v2277 += 1l ;
            }
            v2275 += 1l ;
        }
        float v2284; bool v2285;
        Tuple3 tmp73 = Tuple3{-1.0f / 0.0f, false};
        v2284 = tmp73.v0; v2285 = tmp73.v1;
        int v2286;
        v2286 = 0l;
        while (while_method_1(v2286)){
            int v2288;
            v2288 = 0l;
            while (while_method_2(v2288)){
                assert("Tensor range check" && 0 <= v2286 && v2286 < 1l);
                assert("Tensor range check" && 0 <= v2288 && v2288 < 4l);
                int v2290;
                v2290 = 4l * v2286;
                int v2291;
                v2291 = v2290 + v2288;
                float v2292;
                v2292 = v2273[v2291];
                bool v2293;
                v2293 = v2274[v2291];
                float v2300; bool v2301;
                if (v2285){
                    if (v2293){
                        bool v2294;
                        v2294 = v2284 >= v2292;
                        float v2295;
                        if (v2294){
                            v2295 = v2284;
                        } else {
                            v2295 = v2292;
                        }
                        v2300 = v2295; v2301 = true;
                    } else {
                        v2300 = v2284; v2301 = v2285;
                    }
                } else {
                    if (v2293){
                        v2300 = v2292; v2301 = v2293;
                    } else {
                        v2300 = v2284; v2301 = v2285;
                    }
                }
                v2284 = v2300;
                v2285 = v2301;
                v2288 += 1l ;
            }
            v2286 += 1l ;
        }
        auto v2302 = cooperative_groups::coalesced_threads();
        int v2303;
        v2303 = threadIdx.x;
        auto v2304 = cooperative_groups::labeled_partition(v2302,v2303);
        Closure3 v2305{};
        float v2306; bool v2307;
        Tuple3 tmp74 = cooperative_groups::reduce(v2304, Tuple3{v2284, v2285}, v2305);
        v2306 = tmp74.v0; v2307 = tmp74.v1;
        bool v2308;
        v2308 = v2307 == false;
        if (v2308){
            assert("The local reduce must be true." && v2307);
        } else {
        }
        float v2310[4l];
        int v2311[4l];
        int v2312;
        v2312 = 0l;
        while (while_method_1(v2312)){
            int v2314;
            v2314 = 0l;
            while (while_method_2(v2314)){
                assert("Tensor range check" && 0 <= v2312 && v2312 < 1l);
                assert("Tensor range check" && 0 <= v2314 && v2314 < 4l);
                int v2316;
                v2316 = 4l * v2312;
                int v2317;
                v2317 = v2316 + v2314;
                int v2318;
                v2318 = v2126[v2317];
                float v2319;
                v2319 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v2312 && v2312 < 1l);
                assert("Tensor range check" && 0 <= v2314 && v2314 < 4l);
                v2310[v2317] = v2319;
                v2311[v2317] = v2318;
                v2314 += 1l ;
            }
            v2312 += 1l ;
        }
        float v2320; int v2321;
        Tuple4 tmp75 = Tuple4{0.0f, 2147483647l};
        v2320 = tmp75.v0; v2321 = tmp75.v1;
        int v2322;
        v2322 = 0l;
        while (while_method_1(v2322)){
            int v2324;
            v2324 = 0l;
            while (while_method_2(v2324)){
                assert("Tensor range check" && 0 <= v2322 && v2322 < 1l);
                assert("Tensor range check" && 0 <= v2324 && v2324 < 4l);
                int v2326;
                v2326 = 4l * v2322;
                int v2327;
                v2327 = v2326 + v2324;
                float v2328;
                v2328 = v2310[v2327];
                int v2329;
                v2329 = v2311[v2327];
                bool v2330;
                v2330 = v2321 < v2329;
                float v2331; int v2332;
                if (v2330){
                    v2331 = v2320; v2332 = v2321;
                } else {
                    v2331 = v2328; v2332 = v2329;
                }
                v2320 = v2331;
                v2321 = v2332;
                v2324 += 1l ;
            }
            v2322 += 1l ;
        }
        auto v2333 = cooperative_groups::coalesced_threads();
        int v2334;
        v2334 = threadIdx.x;
        auto v2335 = cooperative_groups::labeled_partition(v2333,v2334);
        Closure4 v2336{};
        float v2337; int v2338;
        Tuple4 tmp76 = cooperative_groups::reduce(v2335, Tuple4{v2320, v2321}, v2336);
        v2337 = tmp76.v0; v2338 = tmp76.v1;
        float v2339;
        v2339 = v2306 * v2337;
        int v2340[4l];
        bool v2341[4l];
        int v2342;
        v2342 = 0l;
        while (while_method_1(v2342)){
            int v2344;
            v2344 = 0l;
            while (while_method_2(v2344)){
                assert("Tensor range check" && 0 <= v2342 && v2342 < 1l);
                assert("Tensor range check" && 0 <= v2344 && v2344 < 4l);
                int v2346;
                v2346 = 4l * v2342;
                int v2347;
                v2347 = v2346 + v2344;
                float v2348;
                v2348 = v2273[v2347];
                bool v2349;
                v2349 = v2274[v2347];
                int v2350;
                v2350 = v2126[v2347];
                int v2353; bool v2354;
                if (v2349){
                    float v2351;
                    v2351 = v2348 - v2339;
                    bool v2352;
                    v2352 = v2351 >= 0.0f;
                    v2353 = v2350; v2354 = v2352;
                } else {
                    v2353 = 2147483647l; v2354 = false;
                }
                assert("Tensor range check" && 0 <= v2342 && v2342 < 1l);
                assert("Tensor range check" && 0 <= v2344 && v2344 < 4l);
                v2340[v2347] = v2353;
                v2341[v2347] = v2354;
                v2344 += 1l ;
            }
            v2342 += 1l ;
        }
        int v2355; bool v2356;
        Tuple5 tmp77 = Tuple5{2147483647l, false};
        v2355 = tmp77.v0; v2356 = tmp77.v1;
        int v2357;
        v2357 = 0l;
        while (while_method_1(v2357)){
            int v2359;
            v2359 = 0l;
            while (while_method_2(v2359)){
                assert("Tensor range check" && 0 <= v2357 && v2357 < 1l);
                assert("Tensor range check" && 0 <= v2359 && v2359 < 4l);
                int v2361;
                v2361 = 4l * v2357;
                int v2362;
                v2362 = v2361 + v2359;
                int v2363;
                v2363 = v2340[v2362];
                bool v2364;
                v2364 = v2341[v2362];
                int v2371; bool v2372;
                if (v2356){
                    if (v2364){
                        bool v2365;
                        v2365 = v2355 < v2363;
                        int v2366;
                        if (v2365){
                            v2366 = v2355;
                        } else {
                            v2366 = v2363;
                        }
                        v2371 = v2366; v2372 = true;
                    } else {
                        v2371 = v2355; v2372 = v2356;
                    }
                } else {
                    if (v2364){
                        v2371 = v2363; v2372 = v2364;
                    } else {
                        v2371 = v2355; v2372 = v2356;
                    }
                }
                v2355 = v2371;
                v2356 = v2372;
                v2359 += 1l ;
            }
            v2357 += 1l ;
        }
        auto v2373 = cooperative_groups::coalesced_threads();
        int v2374;
        v2374 = threadIdx.x;
        auto v2375 = cooperative_groups::labeled_partition(v2373,v2374);
        Closure5 v2376{};
        int v2377; bool v2378;
        Tuple5 tmp78 = cooperative_groups::reduce(v2375, Tuple5{v2355, v2356}, v2376);
        v2377 = tmp78.v0; v2378 = tmp78.v1;
        bool v2379;
        v2379 = v2378 == false;
        if (v2379){
            assert("The local reduce must be true." && v2378);
        } else {
        }
        bool v2381[4l];
        int v2382;
        v2382 = 0l;
        while (while_method_1(v2382)){
            int v2384;
            v2384 = 0l;
            while (while_method_2(v2384)){
                assert("Tensor range check" && 0 <= v2382 && v2382 < 1l);
                assert("Tensor range check" && 0 <= v2384 && v2384 < 4l);
                int v2386;
                v2386 = 4l * v2382;
                int v2387;
                v2387 = v2386 + v2384;
                float v2388;
                v2388 = v2125[v2387];
                int v2389;
                v2389 = v2126[v2387];
                bool v2390;
                v2390 = v2389 < 3l;
                assert("Tensor range check" && 0 <= v2382 && v2382 < 1l);
                assert("Tensor range check" && 0 <= v2384 && v2384 < 4l);
                v2381[v2387] = v2390;
                v2384 += 1l ;
            }
            v2382 += 1l ;
        }
        float v2391[4l];
        int v2392;
        v2392 = 0l;
        while (while_method_1(v2392)){
            int v2394;
            v2394 = 0l;
            while (while_method_2(v2394)){
                assert("Tensor range check" && 0 <= v2392 && v2392 < 1l);
                assert("Tensor range check" && 0 <= v2394 && v2394 < 4l);
                int v2396;
                v2396 = 4l * v2392;
                int v2397;
                v2397 = v2396 + v2394;
                float v2398;
                v2398 = v2125[v2397];
                bool v2399;
                v2399 = v2381[v2397];
                float v2402;
                if (v2399){
                    bool v2400;
                    v2400 = 0.0f >= v2398;
                    if (v2400){
                        v2402 = 0.0f;
                    } else {
                        v2402 = v2398;
                    }
                } else {
                    v2402 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v2392 && v2392 < 1l);
                assert("Tensor range check" && 0 <= v2394 && v2394 < 4l);
                v2391[v2397] = v2402;
                v2394 += 1l ;
            }
            v2392 += 1l ;
        }
        float v2403;
        v2403 = 0.0f;
        int v2404;
        v2404 = 0l;
        while (while_method_1(v2404)){
            int v2406;
            v2406 = 0l;
            while (while_method_2(v2406)){
                assert("Tensor range check" && 0 <= v2404 && v2404 < 1l);
                assert("Tensor range check" && 0 <= v2406 && v2406 < 4l);
                int v2408;
                v2408 = 4l * v2404;
                int v2409;
                v2409 = v2408 + v2406;
                float v2410;
                v2410 = v2391[v2409];
                float v2411;
                v2411 = v2403 + v2410;
                v2403 = v2411;
                v2406 += 1l ;
            }
            v2404 += 1l ;
        }
        auto v2412 = cooperative_groups::coalesced_threads();
        int v2413;
        v2413 = threadIdx.x;
        auto v2414 = cooperative_groups::labeled_partition(v2412,v2413);
        float v2415;
        v2415 = cooperative_groups::reduce(v2414, v2403, v2203);
        int v2416[4l];
        int v2417;
        v2417 = 0l;
        while (while_method_1(v2417)){
            int v2419;
            v2419 = 0l;
            while (while_method_2(v2419)){
                assert("Tensor range check" && 0 <= v2417 && v2417 < 1l);
                assert("Tensor range check" && 0 <= v2419 && v2419 < 4l);
                int v2421;
                v2421 = 4l * v2417;
                int v2422;
                v2422 = v2421 + v2419;
                bool v2423;
                v2423 = v2381[v2422];
                int v2424;
                if (v2423){
                    v2424 = 1l;
                } else {
                    v2424 = 0l;
                }
                assert("Tensor range check" && 0 <= v2417 && v2417 < 1l);
                assert("Tensor range check" && 0 <= v2419 && v2419 < 4l);
                v2416[v2422] = v2424;
                v2419 += 1l ;
            }
            v2417 += 1l ;
        }
        int v2425;
        v2425 = 0l;
        int v2426;
        v2426 = 0l;
        while (while_method_1(v2426)){
            int v2428;
            v2428 = 0l;
            while (while_method_2(v2428)){
                assert("Tensor range check" && 0 <= v2426 && v2426 < 1l);
                assert("Tensor range check" && 0 <= v2428 && v2428 < 4l);
                int v2430;
                v2430 = 4l * v2426;
                int v2431;
                v2431 = v2430 + v2428;
                int v2432;
                v2432 = v2416[v2431];
                int v2433;
                v2433 = v2425 + v2432;
                v2425 = v2433;
                v2428 += 1l ;
            }
            v2426 += 1l ;
        }
        auto v2434 = cooperative_groups::coalesced_threads();
        int v2435;
        v2435 = threadIdx.x;
        auto v2436 = cooperative_groups::labeled_partition(v2434,v2435);
        int v2437;
        v2437 = cooperative_groups::reduce(v2436, v2425, v2226);
        float v2438;
        v2438 = (float)v2437;
        float v2439;
        v2439 = 1.0f / v2438;
        float v2440[4l];
        int v2441;
        v2441 = 0l;
        while (while_method_1(v2441)){
            int v2443;
            v2443 = 0l;
            while (while_method_2(v2443)){
                assert("Tensor range check" && 0 <= v2441 && v2441 < 1l);
                assert("Tensor range check" && 0 <= v2443 && v2443 < 4l);
                int v2445;
                v2445 = 4l * v2441;
                int v2446;
                v2446 = v2445 + v2443;
                float v2447;
                v2447 = v2391[v2446];
                bool v2448;
                v2448 = v2381[v2446];
                bool v2449;
                v2449 = v2448 == false;
                float v2454;
                if (v2449){
                    v2454 = 0.0f;
                } else {
                    bool v2450;
                    v2450 = v2415 == 0.0f;
                    bool v2451;
                    v2451 = v2450 != true;
                    if (v2451){
                        float v2452;
                        v2452 = v2447 / v2415;
                        v2454 = v2452;
                    } else {
                        v2454 = v2439;
                    }
                }
                assert("Tensor range check" && 0 <= v2441 && v2441 < 1l);
                assert("Tensor range check" && 0 <= v2443 && v2443 < 4l);
                v2440[v2446] = v2454;
                v2443 += 1l ;
            }
            v2441 += 1l ;
        }
        float v2455; int v2456;
        Tuple4 tmp79 = Tuple4{0.0f, 2147483647l};
        v2455 = tmp79.v0; v2456 = tmp79.v1;
        int v2457;
        v2457 = 0l;
        while (while_method_1(v2457)){
            int v2459;
            v2459 = 0l;
            while (while_method_2(v2459)){
                assert("Tensor range check" && 0 <= v2457 && v2457 < 1l);
                assert("Tensor range check" && 0 <= v2459 && v2459 < 4l);
                int v2461;
                v2461 = 4l * v2457;
                int v2462;
                v2462 = v2461 + v2459;
                float v2463;
                v2463 = v2230[v2462];
                int v2464;
                v2464 = v2126[v2462];
                bool v2465;
                v2465 = v2456 == v2377;
                float v2469; int v2470;
                if (v2465){
                    v2469 = v2455; v2470 = v2456;
                } else {
                    bool v2466;
                    v2466 = v2464 == v2377;
                    if (v2466){
                        v2469 = v2463; v2470 = v2464;
                    } else {
                        v2469 = v2455; v2470 = v2456;
                    }
                }
                v2455 = v2469;
                v2456 = v2470;
                v2459 += 1l ;
            }
            v2457 += 1l ;
        }
        auto v2471 = cooperative_groups::coalesced_threads();
        int v2472;
        v2472 = threadIdx.x;
        auto v2473 = cooperative_groups::labeled_partition(v2471,v2472);
        Closure6 v2474{v2377};
        float v2475; int v2476;
        Tuple4 tmp80 = cooperative_groups::reduce(v2473, Tuple4{v2455, v2456}, v2474);
        v2475 = tmp80.v0; v2476 = tmp80.v1;
        bool v2477;
        v2477 = v2476 == 2147483647l;
        bool v2478;
        v2478 = v2477 != true;
        bool v2479;
        v2479 = v2478 == false;
        if (v2479){
            assert("Expected a valid action id in get_action." && v2478);
        } else {
        }
        float v2481; int v2482;
        Tuple4 tmp81 = Tuple4{0.0f, 2147483647l};
        v2481 = tmp81.v0; v2482 = tmp81.v1;
        int v2483;
        v2483 = 0l;
        while (while_method_1(v2483)){
            int v2485;
            v2485 = 0l;
            while (while_method_2(v2485)){
                assert("Tensor range check" && 0 <= v2483 && v2483 < 1l);
                assert("Tensor range check" && 0 <= v2485 && v2485 < 4l);
                int v2487;
                v2487 = 4l * v2483;
                int v2488;
                v2488 = v2487 + v2485;
                float v2489;
                v2489 = v2440[v2488];
                int v2490;
                v2490 = v2126[v2488];
                bool v2491;
                v2491 = v2482 == v2377;
                float v2495; int v2496;
                if (v2491){
                    v2495 = v2481; v2496 = v2482;
                } else {
                    bool v2492;
                    v2492 = v2490 == v2377;
                    if (v2492){
                        v2495 = v2489; v2496 = v2490;
                    } else {
                        v2495 = v2481; v2496 = v2482;
                    }
                }
                v2481 = v2495;
                v2482 = v2496;
                v2485 += 1l ;
            }
            v2483 += 1l ;
        }
        auto v2497 = cooperative_groups::coalesced_threads();
        int v2498;
        v2498 = threadIdx.x;
        auto v2499 = cooperative_groups::labeled_partition(v2497,v2498);
        float v2500; int v2501;
        Tuple4 tmp82 = cooperative_groups::reduce(v2499, Tuple4{v2481, v2482}, v2474);
        v2500 = tmp82.v0; v2501 = tmp82.v1;
        bool v2502;
        v2502 = v2501 == 2147483647l;
        bool v2503;
        v2503 = v2502 != true;
        bool v2504;
        v2504 = v2503 == false;
        if (v2504){
            assert("Expected a valid action id in get_action." && v2503);
        } else {
        }
        assert("Tensor range check" && 0 <= v2118 && v2118 < 1l);
        v2102[v2120] = v2500;
        v2103[v2120] = v2475;
        v2104[v2120] = v2377;
        v2118 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v2506;
    v2506 = threadIdx.x;
    assert("Tensor range check" && 0 <= v2506 && v2506 < 32l);
    float v2507;
    v2507 = v2102[v2506];
    float v2508;
    v2508 = v2103[v2506];
    int v2509;
    v2509 = v2104[v2506];
    v2100[0l] = Tuple0{v2507, v2508, v2509};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float v2510; float v2511; int v2512;
    Tuple0 tmp83 = v2100[0l];
    v2510 = tmp83.v0; v2511 = tmp83.v1; v2512 = tmp83.v2;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v17, v18, v19, v20, v2099, v2098, v2512, v2510, v2511);
    int v2513 = v18;
    int v2514; float v2515;
    Tuple2 tmp84 = Tuple2{v2513, -13.0f};
    v2514 = tmp84.v0; v2515 = tmp84.v1;
    while (while_method_3(v2514)){
        v2514 -= 1l ;
        assert("Tensor range check" && 0 <= v2514 && v2514 < 16l);
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
        int v2517;
        v2517 = 32l * v2514;
        int v2518;
        v2518 = v2517 + v17;
        int v2519;
        v2519 = v0[v2518];
        float v2520;
        v2520 = v1[v2518];
        int v2521;
        v2521 = v2[v2518];
        int v2522;
        v2522 = v3[v2518];
        assert("Tensor range check" && 0 <= v2522 && v2522 < 4096l);
        int v2523;
        v2523 = 4l * v2522;
        assert("Tensor range check" && 0 <= v2514 && v2514 < 16l);
        int v2524;
        v2524 = 64l * v2514;
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
        int v2525;
        v2525 = 2l * v17;
        int v2526;
        v2526 = v2525 + v2524;
        assert("Tensor range check" && 0 <= v2514 && v2514 < 16l);
        int v2527;
        v2527 = 128l * v2514;
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
        int v2528;
        v2528 = 4l * v17;
        int v2529;
        v2529 = v2528 + v2527;
        assert("Tensor range check" && 0 <= v2514 && v2514 < 16l);
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
        v7[v2518] = v2515;
        float v2530[1l];
        __shared__ Tuple6 v2531[32l];
        __shared__ float * v2532[32l];
        __shared__ float v2533[32l];
        int v2534;
        v2534 = threadIdx.x;
        float * v2535;
        v2535 = v9+v2523;
        float * v2537;
        v2537 = v11+v2523;
        float * v2539;
        v2539 = v12+v2523;
        assert("Tensor range check" && 0 <= v2534 && v2534 < 32l);
        v2531[v2534] = Tuple6{v2535, v2537, v2539};
        int v2541;
        v2541 = threadIdx.x;
        float * v2542;
        v2542 = v6+v2529;
        assert("Tensor range check" && 0 <= v2541 && v2541 < 32l);
        v2532[v2541] = v2542;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v2544;
        v2544 = threadIdx.x;
        bool v2545;
        v2545 = 0l <= v2544;
        bool v2546;
        v2546 = v2545 == false;
        if (v2546){
            assert("The index needs to be zero or positive." && v2545);
        } else {
        }
        int v2548;
        v2548 = v2544 % 1l;
        bool v2549;
        v2549 = v2544 < 32l;
        bool v2550;
        v2550 = v2549 == false;
        if (v2550){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v2549);
        } else {
        }
        assert("Tensor range check" && 0 <= v2544 && v2544 < 32l);
        assert("Tensor range check" && 0 <= v2544 && v2544 < 32l);
        assert("Tensor range check" && 0 <= v2544 && v2544 < 32l);
        int v2552;
        v2552 = 0l;
        while (while_method_1(v2552)){
            assert("Tensor range check" && 0 <= v2552 && v2552 < 1l);
            int v2554;
            v2554 = v2552 + v2544;
            float * v2555; float * v2556; float * v2557;
            Tuple6 tmp85 = v2531[v2554];
            v2555 = tmp85.v0; v2556 = tmp85.v1; v2557 = tmp85.v2;
            assert("Tensor range check" && 0 <= v2552 && v2552 < 1l);
            float * v2558;
            v2558 = v2532[v2554];
            assert("Tensor range check" && 0 <= v2548 && v2548 < 1l);
            int v2559;
            v2559 = 4l * v2548;
            float v2560[4l];
            float v2561[4l];
            float v2562[4l];
            int v2563[4l];
            int v2564;
            v2564 = 0l;
            while (while_method_1(v2564)){
                assert("Tensor range check" && 0 <= v2564 && v2564 < 1l);
                int v2566;
                v2566 = 4l * v2564;
                assert("Tensor range check" && 0 <= v2564 && v2564 < 1l);
                int v2567;
                v2567 = v2566 + v2559;
                int4* v2568;
                v2568 = reinterpret_cast<int4*>(v2555 + v2567);
                int4* v2569;
                v2569 = reinterpret_cast<int4*>(v2560 + v2566);
                assert("Pointer alignment check" && (unsigned long long)(v2568) % 4l == 0 && (unsigned long long)(v2569) % 4l == 0);
                *v2569 = *v2568;
                int4* v2570;
                v2570 = reinterpret_cast<int4*>(v2556 + v2567);
                int4* v2571;
                v2571 = reinterpret_cast<int4*>(v2561 + v2566);
                assert("Pointer alignment check" && (unsigned long long)(v2570) % 4l == 0 && (unsigned long long)(v2571) % 4l == 0);
                *v2571 = *v2570;
                int4* v2572;
                v2572 = reinterpret_cast<int4*>(v2557 + v2567);
                int4* v2573;
                v2573 = reinterpret_cast<int4*>(v2562 + v2566);
                assert("Pointer alignment check" && (unsigned long long)(v2572) % 4l == 0 && (unsigned long long)(v2573) % 4l == 0);
                *v2573 = *v2572;
                v2564 += 1l ;
            }
            int v2574;
            v2574 = 0l;
            while (while_method_1(v2574)){
                int v2576;
                v2576 = 0l;
                while (while_method_2(v2576)){
                    bool v2578;
                    v2578 = 0l <= v2576;
                    bool v2580;
                    if (v2578){
                        bool v2579;
                        v2579 = v2576 < 4l;
                        v2580 = v2579;
                    } else {
                        v2580 = false;
                    }
                    bool v2581;
                    v2581 = v2580 == false;
                    if (v2581){
                        assert("The indices should be inside the range of the dimension." && v2580);
                    } else {
                    }
                    bool v2583;
                    v2583 = 0l <= v2548;
                    bool v2585;
                    if (v2583){
                        bool v2584;
                        v2584 = v2548 < 1l;
                        v2585 = v2584;
                    } else {
                        v2585 = false;
                    }
                    bool v2586;
                    v2586 = v2585 == false;
                    if (v2586){
                        assert("The indices should be inside the range of the dimension." && v2585);
                    } else {
                    }
                    int v2588;
                    v2588 = v2548 * 4l;
                    int v2589;
                    v2589 = v2576 + v2588;
                    bool v2590;
                    v2590 = 0l <= v2574;
                    bool v2592;
                    if (v2590){
                        bool v2591;
                        v2591 = v2574 < 1l;
                        v2592 = v2591;
                    } else {
                        v2592 = false;
                    }
                    bool v2593;
                    v2593 = v2592 == false;
                    if (v2593){
                        assert("The indices should be inside the range of the dimension." && v2592);
                    } else {
                    }
                    int v2595;
                    v2595 = v2574 * 4l;
                    int v2596;
                    v2596 = v2589 + v2595;
                    assert("Tensor range check" && 0 <= v2574 && v2574 < 1l);
                    assert("Tensor range check" && 0 <= v2576 && v2576 < 4l);
                    int v2597;
                    v2597 = 4l * v2574;
                    int v2598;
                    v2598 = v2597 + v2576;
                    v2563[v2598] = v2596;
                    v2576 += 1l ;
                }
                v2574 += 1l ;
            }
            bool v2599;
            v2599 = 0l <= v2552;
            bool v2601;
            if (v2599){
                bool v2600;
                v2600 = v2552 < 1l;
                v2601 = v2600;
            } else {
                v2601 = false;
            }
            bool v2602;
            v2602 = v2601 == false;
            if (v2602){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v2601);
            } else {
            }
            bool v2604;
            v2604 = v2545 && v2549;
            bool v2605;
            v2605 = v2604 == false;
            if (v2605){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v2604);
            } else {
            }
            int v2607;
            v2607 = v2544 + v2552;
            float v2608[4l];
            int v2609;
            v2609 = 0l;
            while (while_method_1(v2609)){
                int v2611;
                v2611 = 0l;
                while (while_method_2(v2611)){
                    assert("Tensor range check" && 0 <= v2609 && v2609 < 1l);
                    assert("Tensor range check" && 0 <= v2611 && v2611 < 4l);
                    int v2613;
                    v2613 = 4l * v2609;
                    int v2614;
                    v2614 = v2613 + v2611;
                    float v2615;
                    v2615 = v2561[v2614];
                    float v2616;
                    v2616 = v2562[v2614];
                    bool v2617;
                    v2617 = v2616 == 0.0f;
                    bool v2618;
                    v2618 = v2617 != true;
                    float v2620;
                    if (v2618){
                        float v2619;
                        v2619 = v2615 / v2616;
                        v2620 = v2619;
                    } else {
                        v2620 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v2609 && v2609 < 1l);
                    assert("Tensor range check" && 0 <= v2611 && v2611 < 4l);
                    v2608[v2614] = v2620;
                    v2611 += 1l ;
                }
                v2609 += 1l ;
            }
            bool v2621[4l];
            int v2622;
            v2622 = 0l;
            while (while_method_1(v2622)){
                int v2624;
                v2624 = 0l;
                while (while_method_2(v2624)){
                    assert("Tensor range check" && 0 <= v2622 && v2622 < 1l);
                    assert("Tensor range check" && 0 <= v2624 && v2624 < 4l);
                    int v2626;
                    v2626 = 4l * v2622;
                    int v2627;
                    v2627 = v2626 + v2624;
                    float v2628;
                    v2628 = v2560[v2627];
                    int v2629;
                    v2629 = v2563[v2627];
                    bool v2630;
                    v2630 = v2629 < 3l;
                    assert("Tensor range check" && 0 <= v2622 && v2622 < 1l);
                    assert("Tensor range check" && 0 <= v2624 && v2624 < 4l);
                    v2621[v2627] = v2630;
                    v2624 += 1l ;
                }
                v2622 += 1l ;
            }
            float v2631[4l];
            int v2632;
            v2632 = 0l;
            while (while_method_1(v2632)){
                int v2634;
                v2634 = 0l;
                while (while_method_2(v2634)){
                    assert("Tensor range check" && 0 <= v2632 && v2632 < 1l);
                    assert("Tensor range check" && 0 <= v2634 && v2634 < 4l);
                    int v2636;
                    v2636 = 4l * v2632;
                    int v2637;
                    v2637 = v2636 + v2634;
                    float v2638;
                    v2638 = v2560[v2637];
                    bool v2639;
                    v2639 = v2621[v2637];
                    float v2642;
                    if (v2639){
                        bool v2640;
                        v2640 = 0.0f >= v2638;
                        if (v2640){
                            v2642 = 0.0f;
                        } else {
                            v2642 = v2638;
                        }
                    } else {
                        v2642 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v2632 && v2632 < 1l);
                    assert("Tensor range check" && 0 <= v2634 && v2634 < 4l);
                    v2631[v2637] = v2642;
                    v2634 += 1l ;
                }
                v2632 += 1l ;
            }
            float v2643;
            v2643 = 0.0f;
            int v2644;
            v2644 = 0l;
            while (while_method_1(v2644)){
                int v2646;
                v2646 = 0l;
                while (while_method_2(v2646)){
                    assert("Tensor range check" && 0 <= v2644 && v2644 < 1l);
                    assert("Tensor range check" && 0 <= v2646 && v2646 < 4l);
                    int v2648;
                    v2648 = 4l * v2644;
                    int v2649;
                    v2649 = v2648 + v2646;
                    float v2650;
                    v2650 = v2631[v2649];
                    float v2651;
                    v2651 = v2643 + v2650;
                    v2643 = v2651;
                    v2646 += 1l ;
                }
                v2644 += 1l ;
            }
            auto v2652 = cooperative_groups::coalesced_threads();
            int v2653;
            v2653 = threadIdx.x;
            auto v2654 = cooperative_groups::labeled_partition(v2652,v2653);
            Closure0 v2655{};
            float v2656;
            v2656 = cooperative_groups::reduce(v2654, v2643, v2655);
            int v2657[4l];
            int v2658;
            v2658 = 0l;
            while (while_method_1(v2658)){
                int v2660;
                v2660 = 0l;
                while (while_method_2(v2660)){
                    assert("Tensor range check" && 0 <= v2658 && v2658 < 1l);
                    assert("Tensor range check" && 0 <= v2660 && v2660 < 4l);
                    int v2662;
                    v2662 = 4l * v2658;
                    int v2663;
                    v2663 = v2662 + v2660;
                    bool v2664;
                    v2664 = v2621[v2663];
                    int v2665;
                    if (v2664){
                        v2665 = 1l;
                    } else {
                        v2665 = 0l;
                    }
                    assert("Tensor range check" && 0 <= v2658 && v2658 < 1l);
                    assert("Tensor range check" && 0 <= v2660 && v2660 < 4l);
                    v2657[v2663] = v2665;
                    v2660 += 1l ;
                }
                v2658 += 1l ;
            }
            int v2666;
            v2666 = 0l;
            int v2667;
            v2667 = 0l;
            while (while_method_1(v2667)){
                int v2669;
                v2669 = 0l;
                while (while_method_2(v2669)){
                    assert("Tensor range check" && 0 <= v2667 && v2667 < 1l);
                    assert("Tensor range check" && 0 <= v2669 && v2669 < 4l);
                    int v2671;
                    v2671 = 4l * v2667;
                    int v2672;
                    v2672 = v2671 + v2669;
                    int v2673;
                    v2673 = v2657[v2672];
                    int v2674;
                    v2674 = v2666 + v2673;
                    v2666 = v2674;
                    v2669 += 1l ;
                }
                v2667 += 1l ;
            }
            auto v2675 = cooperative_groups::coalesced_threads();
            int v2676;
            v2676 = threadIdx.x;
            auto v2677 = cooperative_groups::labeled_partition(v2675,v2676);
            Closure1 v2678{};
            int v2679;
            v2679 = cooperative_groups::reduce(v2677, v2666, v2678);
            float v2680;
            v2680 = (float)v2679;
            float v2681;
            v2681 = 1.0f / v2680;
            float v2682[4l];
            int v2683;
            v2683 = 0l;
            while (while_method_1(v2683)){
                int v2685;
                v2685 = 0l;
                while (while_method_2(v2685)){
                    assert("Tensor range check" && 0 <= v2683 && v2683 < 1l);
                    assert("Tensor range check" && 0 <= v2685 && v2685 < 4l);
                    int v2687;
                    v2687 = 4l * v2683;
                    int v2688;
                    v2688 = v2687 + v2685;
                    float v2689;
                    v2689 = v2631[v2688];
                    bool v2690;
                    v2690 = v2621[v2688];
                    bool v2691;
                    v2691 = v2690 == false;
                    float v2696;
                    if (v2691){
                        v2696 = 0.0f;
                    } else {
                        bool v2692;
                        v2692 = v2656 == 0.0f;
                        bool v2693;
                        v2693 = v2692 != true;
                        if (v2693){
                            float v2694;
                            v2694 = v2689 / v2656;
                            v2696 = v2694;
                        } else {
                            v2696 = v2681;
                        }
                    }
                    assert("Tensor range check" && 0 <= v2683 && v2683 < 1l);
                    assert("Tensor range check" && 0 <= v2685 && v2685 < 4l);
                    v2682[v2688] = v2696;
                    v2685 += 1l ;
                }
                v2683 += 1l ;
            }
            float v2697[4l];
            int v2698;
            v2698 = 0l;
            while (while_method_1(v2698)){
                int v2700;
                v2700 = 0l;
                while (while_method_2(v2700)){
                    assert("Tensor range check" && 0 <= v2698 && v2698 < 1l);
                    assert("Tensor range check" && 0 <= v2700 && v2700 < 4l);
                    int v2702;
                    v2702 = 4l * v2698;
                    int v2703;
                    v2703 = v2702 + v2700;
                    float v2704;
                    v2704 = v2608[v2703];
                    int v2705;
                    v2705 = v2563[v2703];
                    bool v2706;
                    v2706 = v2519 == v2705;
                    float v2709;
                    if (v2706){
                        float v2707;
                        v2707 = v2515 - v2704;
                        float v2708;
                        v2708 = v2707 / v2520;
                        v2709 = v2708;
                    } else {
                        v2709 = 0.0f;
                    }
                    float v2710;
                    v2710 = v2709 + v2704;
                    assert("Tensor range check" && 0 <= v2698 && v2698 < 1l);
                    assert("Tensor range check" && 0 <= v2700 && v2700 < 4l);
                    v2697[v2703] = v2710;
                    v2700 += 1l ;
                }
                v2698 += 1l ;
            }
            float v2711[4l];
            int v2712;
            v2712 = 0l;
            while (while_method_1(v2712)){
                int v2714;
                v2714 = 0l;
                while (while_method_2(v2714)){
                    assert("Tensor range check" && 0 <= v2712 && v2712 < 1l);
                    assert("Tensor range check" && 0 <= v2714 && v2714 < 4l);
                    int v2716;
                    v2716 = 4l * v2712;
                    int v2717;
                    v2717 = v2716 + v2714;
                    float v2718;
                    v2718 = v2682[v2717];
                    float v2719;
                    v2719 = v2697[v2717];
                    float v2720;
                    v2720 = v2718 * v2719;
                    assert("Tensor range check" && 0 <= v2712 && v2712 < 1l);
                    assert("Tensor range check" && 0 <= v2714 && v2714 < 4l);
                    v2711[v2717] = v2720;
                    v2714 += 1l ;
                }
                v2712 += 1l ;
            }
            float v2721;
            v2721 = 0.0f;
            int v2722;
            v2722 = 0l;
            while (while_method_1(v2722)){
                int v2724;
                v2724 = 0l;
                while (while_method_2(v2724)){
                    assert("Tensor range check" && 0 <= v2722 && v2722 < 1l);
                    assert("Tensor range check" && 0 <= v2724 && v2724 < 4l);
                    int v2726;
                    v2726 = 4l * v2722;
                    int v2727;
                    v2727 = v2726 + v2724;
                    float v2728;
                    v2728 = v2711[v2727];
                    float v2729;
                    v2729 = v2721 + v2728;
                    v2721 = v2729;
                    v2724 += 1l ;
                }
                v2722 += 1l ;
            }
            auto v2730 = cooperative_groups::coalesced_threads();
            int v2731;
            v2731 = threadIdx.x;
            auto v2732 = cooperative_groups::labeled_partition(v2730,v2731);
            float v2733;
            v2733 = cooperative_groups::reduce(v2732, v2721, v2655);
            float v2734[4l];
            int v2735;
            v2735 = 0l;
            while (while_method_1(v2735)){
                int v2737;
                v2737 = 0l;
                while (while_method_2(v2737)){
                    assert("Tensor range check" && 0 <= v2735 && v2735 < 1l);
                    assert("Tensor range check" && 0 <= v2737 && v2737 < 4l);
                    int v2739;
                    v2739 = 4l * v2735;
                    int v2740;
                    v2740 = v2739 + v2737;
                    float v2741;
                    v2741 = v2697[v2740];
                    double v2742[2l];
                    int v2743;
                    v2743 = 0l;
                    while (while_method_0(v2743)){
                        assert("Tensor range check" && 0 <= v2743 && v2743 < 2l);
                        int v2745;
                        v2745 = v2743 + v2526;
                        double v2746;
                        v2746 = v4[v2745];
                        bool v2747;
                        v2747 = v2521 == v2743;
                        double v2748;
                        if (v2747){
                            v2748 = 0.0;
                        } else {
                            v2748 = v2746;
                        }
                        assert("Tensor range check" && 0 <= v2743 && v2743 < 2l);
                        v2742[v2743] = v2748;
                        v2743 += 1l ;
                    }
                    double v2749;
                    v2749 = 0.0;
                    int v2750;
                    v2750 = 0l;
                    while (while_method_0(v2750)){
                        assert("Tensor range check" && 0 <= v2750 && v2750 < 2l);
                        double v2752;
                        v2752 = v2742[v2750];
                        double v2753;
                        v2753 = v2749 + v2752;
                        v2749 = v2753;
                        v2750 += 1l ;
                    }
                    double v2754;
                    v2754 = 0.0;
                    int v2755;
                    v2755 = 0l;
                    while (while_method_0(v2755)){
                        assert("Tensor range check" && 0 <= v2755 && v2755 < 2l);
                        int v2757;
                        v2757 = v2755 + v2526;
                        double v2758;
                        v2758 = v5[v2757];
                        double v2759;
                        v2759 = v2754 + v2758;
                        v2754 = v2759;
                        v2755 += 1l ;
                    }
                    double v2760;
                    v2760 = v2749 - v2754;
                    double v2761;
                    v2761 = exp(v2760);
                    float v2762;
                    v2762 = (float)v2761;
                    float v2763;
                    v2763 = v2741 - v2733;
                    float v2764;
                    v2764 = v2762 * v2763;
                    assert("Tensor range check" && 0 <= v2735 && v2735 < 1l);
                    assert("Tensor range check" && 0 <= v2737 && v2737 < 4l);
                    v2734[v2740] = v2764;
                    v2737 += 1l ;
                }
                v2735 += 1l ;
            }
            int v2765;
            v2765 = threadIdx.x;
            cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v2766 = console_lock;
            auto v2767 = cooperative_groups::coalesced_threads();
            v2766.acquire();
            int v2768;
            v2768 = 0l;
            printf("{%s = %d; %s = %c","tid", v2765, "update_policy", '[');
            int v2769;
            v2769 = 0l;
            while (while_method_1(v2769)){
                int v2771;
                v2771 = v2768;
                bool v2772;
                v2772 = v2771 >= 100l;
                if (v2772){
                    printf("%s"," ...");
                    break;
                } else {
                }
                bool v2773;
                v2773 = v2769 == 0l;
                bool v2774;
                v2774 = v2773 != true;
                if (v2774){
                    printf("%s","; ");
                } else {
                }
                printf("%c",'[');
                int v2775;
                v2775 = 0l;
                while (while_method_2(v2775)){
                    int v2777;
                    v2777 = v2768;
                    bool v2778;
                    v2778 = v2777 >= 100l;
                    if (v2778){
                        printf("%s"," ...");
                        break;
                    } else {
                    }
                    bool v2779;
                    v2779 = v2775 == 0l;
                    bool v2780;
                    v2780 = v2779 != true;
                    if (v2780){
                        printf("%s","; ");
                    } else {
                    }
                    int v2781;
                    v2781 = v2768 + 1l;
                    v2768 = v2781;
                    int v2782;
                    v2782 = v2769 * 4l;
                    int v2783;
                    v2783 = v2782 + v2775;
                    float v2784;
                    v2784 = v2734[v2783];
                    printf("%f",v2784);
                    v2775 += 1l ;
                }
                printf("%c",']');
                v2769 += 1l ;
            }
            printf("%c",']');
            printf("}\n");
            v2766.release();
            v2767.sync() ;
            assert("Tensor range check" && 0 <= v2552 && v2552 < 1l);
            v2533[v2554] = v2733;
            assert("Tensor range check" && 0 <= v2548 && v2548 < 1l);
            int v2815;
            v2815 = 0l;
            while (while_method_1(v2815)){
                assert("Tensor range check" && 0 <= v2815 && v2815 < 1l);
                int v2817;
                v2817 = 4l * v2815;
                int v2818;
                v2818 = v2817 + v2559;
                assert("Tensor range check" && 0 <= v2815 && v2815 < 1l);
                int4* v2819;
                v2819 = reinterpret_cast<int4*>(v2734 + v2817);
                int4* v2820;
                v2820 = reinterpret_cast<int4*>(v2558 + v2818);
                assert("Pointer alignment check" && (unsigned long long)(v2819) % 4l == 0 && (unsigned long long)(v2820) % 4l == 0);
                *v2820 = *v2819;
                v2815 += 1l ;
            }
            v2552 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v2821;
        v2821 = threadIdx.x;
        assert("Tensor range check" && 0 <= v2821 && v2821 < 32l);
        float v2822;
        v2822 = v2533[v2821];
        v2530[0l] = v2822;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        float v2823;
        v2823 = v2530[0l];
        v2515 = v2823;
    }
    cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v2824 = console_lock;
    auto v2825 = cooperative_groups::coalesced_threads();
    v2824.acquire();
    printf("{%s = %f}\n","fin_reward", v2515);
    v2824.release();
    v2825.sync() ;
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
    v8 = cp.empty(16384,dtype=cp.float32)
    v9 = cp.empty(16384,dtype=cp.float32)
    v10 = cp.empty(16384,dtype=cp.float32)
    v11 = cp.empty(16384,dtype=cp.float32)
    v12 = cp.empty(16384,dtype=cp.float32)
    v13 = 0
    v14 = raw_module.get_function(f"entry{v13}")
    del v13
    v14.max_dynamic_shared_size_bytes = 0 
    v14((1,),(32,),(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12),shared_mem=0)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v14
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
