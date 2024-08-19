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
__device__ void push__0(int * v0, float * v1, int * v2, int * v3, double * v4, double * v5, float * v6, float * v7, float * v8, float * v9, float * v10, float * v11, float * v12, int v13, int & v14, double * v15, double * v16, int v17, int v18, float v19, int v20);
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
struct Closure1 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
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
struct Tuple1 {
    float v0;
    bool v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
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
                return Tuple1{v5, true};
            } else {
                return Tuple1{v0, v1};
            }
        } else {
            if (v3){
                return Tuple1{v2, v3};
            } else {
                return Tuple1{v0, v1};
            }
        }
    }
};
struct Tuple2 {
    float v0;
    int v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple2 operator()(Tuple2 tup0, Tuple2 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple2{v0, v1};
        } else {
            return Tuple2{v2, v3};
        }
    }
};
struct Tuple3 {
    int v0;
    bool v1;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
    __device__ Tuple3 operator()(Tuple3 tup0, Tuple3 tup1){
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
    int v0;
    __device__ Tuple2 operator()(Tuple2 tup0, Tuple2 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple2{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple2{v3, v4};
            } else {
                return Tuple2{v1, v2};
            }
        }
    }
    __device__ Closure6(int _v0) : v0(_v0) { }
};
struct Tuple4 {
    float * v0;
    float * v1;
    float * v2;
    __device__ Tuple4() = default;
    __device__ Tuple4(float * t0, float * t1, float * t2) : v0(t0), v1(t1), v2(t2) {}
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
__device__ void push__0(int * v0, float * v1, int * v2, int * v3, double * v4, double * v5, float * v6, float * v7, float * v8, float * v9, float * v10, float * v11, float * v12, int v13, int & v14, double * v15, double * v16, int v17, int v18, float v19, int v20){
    int v21 = v14;
    int v22;
    v22 = v21 + 1l;
    v14 = v22;
    assert("Tensor range check" && 0 <= v21 && v21 < 16l);
    assert("Tensor range check" && 0 <= v13 && v13 < 32l);
    int v23;
    v23 = 32l * v21;
    int v24;
    v24 = v23 + v13;
    v0[v24] = v20;
    v1[v24] = v19;
    v2[v24] = v17;
    v3[v24] = v18;
    double v25;
    v25 = (double)v19;
    double v26;
    v26 = log(v25);
    assert("Tensor range check" && 0 <= v18 && v18 < 4096l);
    assert("Tensor range check" && 0 <= v20 && v20 < 4l);
    int v27;
    v27 = 4l * v18;
    int v28;
    v28 = v27 + v20;
    float v29;
    v29 = v9[v28];
    double v30;
    v30 = (double)v29;
    double v31;
    v31 = log(v30);
    assert("Tensor range check" && 0 <= v17 && v17 < 2l);
    double v32;
    v32 = v15[v17];
    double v33;
    v33 = v16[v17];
    double v34;
    v34 = v31 + v32;
    double v35;
    v35 = v26 + v33;
    assert("Tensor range check" && 0 <= v17 && v17 < 2l);
    v15[v17] = v34;
    v16[v17] = v35;
    assert("Tensor range check" && 0 <= v21 && v21 < 16l);
    int v36;
    v36 = 64l * v21;
    assert("Tensor range check" && 0 <= v13 && v13 < 32l);
    int v37;
    v37 = 2l * v13;
    int v38;
    v38 = v37 + v36;
    int v39;
    v39 = 0l;
    while (while_method_0(v39)){
        assert("Tensor range check" && 0 <= v39 && v39 < 2l);
        double v41;
        v41 = v15[v39];
        double v42;
        v42 = v16[v39];
        assert("Tensor range check" && 0 <= v39 && v39 < 2l);
        int v43;
        v43 = v39 + v38;
        v4[v43] = v41;
        v5[v43] = v42;
        v39 += 1l ;
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
    __shared__ float * v26[32l];
    __shared__ int v27[32l];
    __shared__ float v28[32l];
    int v29;
    v29 = threadIdx.x;
    float * v30;
    v30 = v8+940l;
    assert("Tensor range check" && 0 <= v29 && v29 < 32l);
    v26[v29] = v30;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v32;
    v32 = threadIdx.x;
    bool v33;
    v33 = 0l <= v32;
    bool v34;
    v34 = v33 == false;
    if (v34){
        assert("The index needs to be zero or positive." && v33);
    } else {
    }
    int v36;
    v36 = v32 % 1l;
    bool v37;
    v37 = v32 < 32l;
    bool v38;
    v38 = v37 == false;
    if (v38){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v37);
    } else {
    }
    assert("Tensor range check" && 0 <= v32 && v32 < 32l);
    assert("Tensor range check" && 0 <= v32 && v32 < 32l);
    int v40;
    v40 = 0l;
    while (while_method_1(v40)){
        assert("Tensor range check" && 0 <= v40 && v40 < 1l);
        int v42;
        v42 = v40 + v32;
        float * v43;
        v43 = v26[v42];
        assert("Tensor range check" && 0 <= v36 && v36 < 1l);
        int v44;
        v44 = 4l * v36;
        float v45[4l];
        int v46[4l];
        int v47;
        v47 = 0l;
        while (while_method_1(v47)){
            assert("Tensor range check" && 0 <= v47 && v47 < 1l);
            int v49;
            v49 = 4l * v47;
            assert("Tensor range check" && 0 <= v47 && v47 < 1l);
            int v50;
            v50 = v49 + v44;
            int4* v51;
            v51 = reinterpret_cast<int4*>(v43 + v50);
            int4* v52;
            v52 = reinterpret_cast<int4*>(v45 + v49);
            assert("Pointer alignment check" && (unsigned long long)(v51) % 4l == 0 && (unsigned long long)(v52) % 4l == 0);
            *v52 = *v51;
            v47 += 1l ;
        }
        int v53;
        v53 = 0l;
        while (while_method_1(v53)){
            int v55;
            v55 = 0l;
            while (while_method_2(v55)){
                bool v57;
                v57 = 0l <= v55;
                bool v59;
                if (v57){
                    bool v58;
                    v58 = v55 < 4l;
                    v59 = v58;
                } else {
                    v59 = false;
                }
                bool v60;
                v60 = v59 == false;
                if (v60){
                    assert("The indices should be inside the range of the dimension." && v59);
                } else {
                }
                bool v62;
                v62 = 0l <= v36;
                bool v64;
                if (v62){
                    bool v63;
                    v63 = v36 < 1l;
                    v64 = v63;
                } else {
                    v64 = false;
                }
                bool v65;
                v65 = v64 == false;
                if (v65){
                    assert("The indices should be inside the range of the dimension." && v64);
                } else {
                }
                int v67;
                v67 = v36 * 4l;
                int v68;
                v68 = v55 + v67;
                bool v69;
                v69 = 0l <= v53;
                bool v71;
                if (v69){
                    bool v70;
                    v70 = v53 < 1l;
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
                v74 = v53 * 4l;
                int v75;
                v75 = v68 + v74;
                assert("Tensor range check" && 0 <= v53 && v53 < 1l);
                assert("Tensor range check" && 0 <= v55 && v55 < 4l);
                int v76;
                v76 = 4l * v53;
                int v77;
                v77 = v76 + v55;
                v46[v77] = v75;
                v55 += 1l ;
            }
            v53 += 1l ;
        }
        bool v78;
        v78 = 0l <= v40;
        bool v80;
        if (v78){
            bool v79;
            v79 = v40 < 1l;
            v80 = v79;
        } else {
            v80 = false;
        }
        bool v81;
        v81 = v80 == false;
        if (v81){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v80);
        } else {
        }
        bool v83;
        v83 = v33 && v37;
        bool v84;
        v84 = v83 == false;
        if (v84){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v83);
        } else {
        }
        int v86;
        v86 = v32 + v40;
        bool v87[4l];
        int v88;
        v88 = 0l;
        while (while_method_1(v88)){
            int v90;
            v90 = 0l;
            while (while_method_2(v90)){
                assert("Tensor range check" && 0 <= v88 && v88 < 1l);
                assert("Tensor range check" && 0 <= v90 && v90 < 4l);
                int v92;
                v92 = 4l * v88;
                int v93;
                v93 = v92 + v90;
                float v94;
                v94 = v45[v93];
                int v95;
                v95 = v46[v93];
                bool v96;
                v96 = v95 < 3l;
                assert("Tensor range check" && 0 <= v88 && v88 < 1l);
                assert("Tensor range check" && 0 <= v90 && v90 < 4l);
                v87[v93] = v96;
                v90 += 1l ;
            }
            v88 += 1l ;
        }
        float v97[4l];
        int v98;
        v98 = 0l;
        while (while_method_1(v98)){
            int v100;
            v100 = 0l;
            while (while_method_2(v100)){
                assert("Tensor range check" && 0 <= v98 && v98 < 1l);
                assert("Tensor range check" && 0 <= v100 && v100 < 4l);
                int v102;
                v102 = 4l * v98;
                int v103;
                v103 = v102 + v100;
                float v104;
                v104 = v45[v103];
                bool v105;
                v105 = v87[v103];
                float v108;
                if (v105){
                    bool v106;
                    v106 = 0.0f >= v104;
                    if (v106){
                        v108 = 0.0f;
                    } else {
                        v108 = v104;
                    }
                } else {
                    v108 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v98 && v98 < 1l);
                assert("Tensor range check" && 0 <= v100 && v100 < 4l);
                v97[v103] = v108;
                v100 += 1l ;
            }
            v98 += 1l ;
        }
        float v109;
        v109 = 0.0f;
        int v110;
        v110 = 0l;
        while (while_method_1(v110)){
            int v112;
            v112 = 0l;
            while (while_method_2(v112)){
                assert("Tensor range check" && 0 <= v110 && v110 < 1l);
                assert("Tensor range check" && 0 <= v112 && v112 < 4l);
                int v114;
                v114 = 4l * v110;
                int v115;
                v115 = v114 + v112;
                float v116;
                v116 = v97[v115];
                float v117;
                v117 = v109 + v116;
                v109 = v117;
                v112 += 1l ;
            }
            v110 += 1l ;
        }
        auto v118 = cooperative_groups::coalesced_threads();
        int v119;
        v119 = threadIdx.x;
        auto v120 = cooperative_groups::labeled_partition(v118,v119);
        Closure0 v121{};
        float v122;
        v122 = cooperative_groups::reduce(v120, v109, v121);
        int v123[4l];
        int v124;
        v124 = 0l;
        while (while_method_1(v124)){
            int v126;
            v126 = 0l;
            while (while_method_2(v126)){
                assert("Tensor range check" && 0 <= v124 && v124 < 1l);
                assert("Tensor range check" && 0 <= v126 && v126 < 4l);
                int v128;
                v128 = 4l * v124;
                int v129;
                v129 = v128 + v126;
                bool v130;
                v130 = v87[v129];
                int v131;
                if (v130){
                    v131 = 1l;
                } else {
                    v131 = 0l;
                }
                assert("Tensor range check" && 0 <= v124 && v124 < 1l);
                assert("Tensor range check" && 0 <= v126 && v126 < 4l);
                v123[v129] = v131;
                v126 += 1l ;
            }
            v124 += 1l ;
        }
        int v132;
        v132 = 0l;
        int v133;
        v133 = 0l;
        while (while_method_1(v133)){
            int v135;
            v135 = 0l;
            while (while_method_2(v135)){
                assert("Tensor range check" && 0 <= v133 && v133 < 1l);
                assert("Tensor range check" && 0 <= v135 && v135 < 4l);
                int v137;
                v137 = 4l * v133;
                int v138;
                v138 = v137 + v135;
                int v139;
                v139 = v123[v138];
                int v140;
                v140 = v132 + v139;
                v132 = v140;
                v135 += 1l ;
            }
            v133 += 1l ;
        }
        auto v141 = cooperative_groups::coalesced_threads();
        int v142;
        v142 = threadIdx.x;
        auto v143 = cooperative_groups::labeled_partition(v141,v142);
        Closure1 v144{};
        int v145;
        v145 = cooperative_groups::reduce(v143, v132, v144);
        float v146;
        v146 = (float)v145;
        float v147;
        v147 = 1.0f / v146;
        float v148[4l];
        int v149;
        v149 = 0l;
        while (while_method_1(v149)){
            int v151;
            v151 = 0l;
            while (while_method_2(v151)){
                assert("Tensor range check" && 0 <= v149 && v149 < 1l);
                assert("Tensor range check" && 0 <= v151 && v151 < 4l);
                int v153;
                v153 = 4l * v149;
                int v154;
                v154 = v153 + v151;
                float v155;
                v155 = v97[v154];
                bool v156;
                v156 = v87[v154];
                bool v157;
                v157 = v156 == false;
                float v162;
                if (v157){
                    v162 = 0.0f;
                } else {
                    bool v158;
                    v158 = v122 == 0.0f;
                    bool v159;
                    v159 = v158 != true;
                    if (v159){
                        float v160;
                        v160 = v155 / v122;
                        v162 = v160;
                    } else {
                        v162 = v147;
                    }
                }
                assert("Tensor range check" && 0 <= v149 && v149 < 1l);
                assert("Tensor range check" && 0 <= v151 && v151 < 4l);
                v148[v154] = v162;
                v151 += 1l ;
            }
            v149 += 1l ;
        }
        float v163[4l];
        float v164;
        v164 = 0.0f;
        int v165;
        v165 = 0l;
        while (while_method_1(v165)){
            assert("Tensor range check" && 0 <= v165 && v165 < 1l);
            int v167;
            v167 = 4l * v165;
            assert("Tensor range check" && 0 <= v165 && v165 < 1l);
            int v168; float v169;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v168 = tmp0.v0; v169 = tmp0.v1;
            while (while_method_2(v168)){
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                int v171;
                v171 = v168 + v167;
                float v172;
                v172 = v148[v171];
                float v173;
                v173 = v169 + v172;
                v169 = v173;
                v168 += 1l ;
            }
            auto v174 = cooperative_groups::coalesced_threads();
            int v175;
            v175 = threadIdx.x;
            auto v176 = cooperative_groups::labeled_partition(v174,v175);
            Closure2 v177{};
            float v178;
            v178 = cooperative_groups::inclusive_scan(v176, v169, v177);
            float v179;
            v179 = v176.shfl_up(v178,1);
            bool v180;
            v180 = v176.thread_rank() == 0;
            float v181;
            if (v180){
                v181 = 0.0f;
            } else {
                v181 = v179;
            }
            float v182;
            v182 = v176.shfl(v178,v176.num_threads()-1);
            float v183;
            v183 = v164 + v181;
            int v184; float v185;
            Tuple0 tmp1 = Tuple0{0l, v183};
            v184 = tmp1.v0; v185 = tmp1.v1;
            while (while_method_2(v184)){
                assert("Tensor range check" && 0 <= v184 && v184 < 4l);
                int v187;
                v187 = v184 + v167;
                float v188;
                v188 = v148[v187];
                float v189;
                v189 = v185 + v188;
                assert("Tensor range check" && 0 <= v184 && v184 < 4l);
                v163[v187] = v189;
                v185 = v189;
                v184 += 1l ;
            }
            float v190;
            v190 = v164 + v182;
            v164 = v190;
            v165 += 1l ;
        }
        float v191[4l];
        bool v192[4l];
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
                v199 = v163[v198];
                float v200;
                v200 = v148[v198];
                bool v201;
                v201 = v200 > 0.0f;
                assert("Tensor range check" && 0 <= v193 && v193 < 1l);
                assert("Tensor range check" && 0 <= v195 && v195 < 4l);
                v191[v198] = v199;
                v192[v198] = v201;
                v195 += 1l ;
            }
            v193 += 1l ;
        }
        float v202; bool v203;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, false};
        v202 = tmp2.v0; v203 = tmp2.v1;
        int v204;
        v204 = 0l;
        while (while_method_1(v204)){
            int v206;
            v206 = 0l;
            while (while_method_2(v206)){
                assert("Tensor range check" && 0 <= v204 && v204 < 1l);
                assert("Tensor range check" && 0 <= v206 && v206 < 4l);
                int v208;
                v208 = 4l * v204;
                int v209;
                v209 = v208 + v206;
                float v210;
                v210 = v191[v209];
                bool v211;
                v211 = v192[v209];
                float v218; bool v219;
                if (v203){
                    if (v211){
                        bool v212;
                        v212 = v202 >= v210;
                        float v213;
                        if (v212){
                            v213 = v202;
                        } else {
                            v213 = v210;
                        }
                        v218 = v213; v219 = true;
                    } else {
                        v218 = v202; v219 = v203;
                    }
                } else {
                    if (v211){
                        v218 = v210; v219 = v211;
                    } else {
                        v218 = v202; v219 = v203;
                    }
                }
                v202 = v218;
                v203 = v219;
                v206 += 1l ;
            }
            v204 += 1l ;
        }
        auto v220 = cooperative_groups::coalesced_threads();
        int v221;
        v221 = threadIdx.x;
        auto v222 = cooperative_groups::labeled_partition(v220,v221);
        Closure3 v223{};
        float v224; bool v225;
        Tuple1 tmp3 = cooperative_groups::reduce(v222, Tuple1{v202, v203}, v223);
        v224 = tmp3.v0; v225 = tmp3.v1;
        bool v226;
        v226 = v225 == false;
        if (v226){
            assert("The local reduce must be true." && v225);
        } else {
        }
        float v228[4l];
        int v229[4l];
        int v230;
        v230 = 0l;
        while (while_method_1(v230)){
            int v232;
            v232 = 0l;
            while (while_method_2(v232)){
                assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                assert("Tensor range check" && 0 <= v232 && v232 < 4l);
                int v234;
                v234 = 4l * v230;
                int v235;
                v235 = v234 + v232;
                int v236;
                v236 = v46[v235];
                float v237;
                v237 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                assert("Tensor range check" && 0 <= v232 && v232 < 4l);
                v228[v235] = v237;
                v229[v235] = v236;
                v232 += 1l ;
            }
            v230 += 1l ;
        }
        float v238; int v239;
        Tuple2 tmp4 = Tuple2{0.0f, 2147483647l};
        v238 = tmp4.v0; v239 = tmp4.v1;
        int v240;
        v240 = 0l;
        while (while_method_1(v240)){
            int v242;
            v242 = 0l;
            while (while_method_2(v242)){
                assert("Tensor range check" && 0 <= v240 && v240 < 1l);
                assert("Tensor range check" && 0 <= v242 && v242 < 4l);
                int v244;
                v244 = 4l * v240;
                int v245;
                v245 = v244 + v242;
                float v246;
                v246 = v228[v245];
                int v247;
                v247 = v229[v245];
                bool v248;
                v248 = v239 < v247;
                float v249; int v250;
                if (v248){
                    v249 = v238; v250 = v239;
                } else {
                    v249 = v246; v250 = v247;
                }
                v238 = v249;
                v239 = v250;
                v242 += 1l ;
            }
            v240 += 1l ;
        }
        auto v251 = cooperative_groups::coalesced_threads();
        int v252;
        v252 = threadIdx.x;
        auto v253 = cooperative_groups::labeled_partition(v251,v252);
        Closure4 v254{};
        float v255; int v256;
        Tuple2 tmp5 = cooperative_groups::reduce(v253, Tuple2{v238, v239}, v254);
        v255 = tmp5.v0; v256 = tmp5.v1;
        float v257;
        v257 = v224 * v255;
        int v258[4l];
        bool v259[4l];
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
                v266 = v191[v265];
                bool v267;
                v267 = v192[v265];
                int v268;
                v268 = v46[v265];
                int v271; bool v272;
                if (v267){
                    float v269;
                    v269 = v266 - v257;
                    bool v270;
                    v270 = v269 >= 0.0f;
                    v271 = v268; v272 = v270;
                } else {
                    v271 = 2147483647l; v272 = false;
                }
                assert("Tensor range check" && 0 <= v260 && v260 < 1l);
                assert("Tensor range check" && 0 <= v262 && v262 < 4l);
                v258[v265] = v271;
                v259[v265] = v272;
                v262 += 1l ;
            }
            v260 += 1l ;
        }
        int v273; bool v274;
        Tuple3 tmp6 = Tuple3{2147483647l, false};
        v273 = tmp6.v0; v274 = tmp6.v1;
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
                int v281;
                v281 = v258[v280];
                bool v282;
                v282 = v259[v280];
                int v289; bool v290;
                if (v274){
                    if (v282){
                        bool v283;
                        v283 = v273 < v281;
                        int v284;
                        if (v283){
                            v284 = v273;
                        } else {
                            v284 = v281;
                        }
                        v289 = v284; v290 = true;
                    } else {
                        v289 = v273; v290 = v274;
                    }
                } else {
                    if (v282){
                        v289 = v281; v290 = v282;
                    } else {
                        v289 = v273; v290 = v274;
                    }
                }
                v273 = v289;
                v274 = v290;
                v277 += 1l ;
            }
            v275 += 1l ;
        }
        auto v291 = cooperative_groups::coalesced_threads();
        int v292;
        v292 = threadIdx.x;
        auto v293 = cooperative_groups::labeled_partition(v291,v292);
        Closure5 v294{};
        int v295; bool v296;
        Tuple3 tmp7 = cooperative_groups::reduce(v293, Tuple3{v273, v274}, v294);
        v295 = tmp7.v0; v296 = tmp7.v1;
        bool v297;
        v297 = v296 == false;
        if (v297){
            assert("The local reduce must be true." && v296);
        } else {
        }
        float v299; int v300;
        Tuple2 tmp8 = Tuple2{0.0f, 2147483647l};
        v299 = tmp8.v0; v300 = tmp8.v1;
        int v301;
        v301 = 0l;
        while (while_method_1(v301)){
            int v303;
            v303 = 0l;
            while (while_method_2(v303)){
                assert("Tensor range check" && 0 <= v301 && v301 < 1l);
                assert("Tensor range check" && 0 <= v303 && v303 < 4l);
                int v305;
                v305 = 4l * v301;
                int v306;
                v306 = v305 + v303;
                float v307;
                v307 = v148[v306];
                int v308;
                v308 = v46[v306];
                bool v309;
                v309 = v300 == v295;
                float v313; int v314;
                if (v309){
                    v313 = v299; v314 = v300;
                } else {
                    bool v310;
                    v310 = v308 == v295;
                    if (v310){
                        v313 = v307; v314 = v308;
                    } else {
                        v313 = v299; v314 = v300;
                    }
                }
                v299 = v313;
                v300 = v314;
                v303 += 1l ;
            }
            v301 += 1l ;
        }
        auto v315 = cooperative_groups::coalesced_threads();
        int v316;
        v316 = threadIdx.x;
        auto v317 = cooperative_groups::labeled_partition(v315,v316);
        Closure6 v318{v295};
        float v319; int v320;
        Tuple2 tmp9 = cooperative_groups::reduce(v317, Tuple2{v299, v300}, v318);
        v319 = tmp9.v0; v320 = tmp9.v1;
        bool v321;
        v321 = v320 == 2147483647l;
        bool v322;
        v322 = v321 != true;
        bool v323;
        v323 = v322 == false;
        if (v323){
            assert("Expected a valid action id in get_action." && v322);
        } else {
        }
        assert("Tensor range check" && 0 <= v40 && v40 < 1l);
        v27[v42] = v295;
        v28[v42] = v319;
        v40 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v325;
    v325 = threadIdx.x;
    assert("Tensor range check" && 0 <= v325 && v325 < 32l);
    int v326;
    v326 = v27[v325];
    float v327;
    v327 = v28[v325];
    v25[0l] = Tuple0{v326, v327};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v328; float v329;
    Tuple0 tmp10 = v25[0l];
    v328 = tmp10.v0; v329 = tmp10.v1;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v17, v18, v19, v20, v24, v23, v329, v328);
    int v330;
    v330 = 212l;
    int v331;
    v331 = 1l;
    Tuple0 v332[1l];
    __shared__ float * v333[32l];
    __shared__ int v334[32l];
    __shared__ float v335[32l];
    int v336;
    v336 = threadIdx.x;
    float * v337;
    v337 = v8+848l;
    assert("Tensor range check" && 0 <= v336 && v336 < 32l);
    v333[v336] = v337;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v339;
    v339 = threadIdx.x;
    bool v340;
    v340 = 0l <= v339;
    bool v341;
    v341 = v340 == false;
    if (v341){
        assert("The index needs to be zero or positive." && v340);
    } else {
    }
    int v343;
    v343 = v339 % 1l;
    bool v344;
    v344 = v339 < 32l;
    bool v345;
    v345 = v344 == false;
    if (v345){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v344);
    } else {
    }
    assert("Tensor range check" && 0 <= v339 && v339 < 32l);
    assert("Tensor range check" && 0 <= v339 && v339 < 32l);
    int v347;
    v347 = 0l;
    while (while_method_1(v347)){
        assert("Tensor range check" && 0 <= v347 && v347 < 1l);
        int v349;
        v349 = v347 + v339;
        float * v350;
        v350 = v333[v349];
        assert("Tensor range check" && 0 <= v343 && v343 < 1l);
        int v351;
        v351 = 4l * v343;
        float v352[4l];
        int v353[4l];
        int v354;
        v354 = 0l;
        while (while_method_1(v354)){
            assert("Tensor range check" && 0 <= v354 && v354 < 1l);
            int v356;
            v356 = 4l * v354;
            assert("Tensor range check" && 0 <= v354 && v354 < 1l);
            int v357;
            v357 = v356 + v351;
            int4* v358;
            v358 = reinterpret_cast<int4*>(v350 + v357);
            int4* v359;
            v359 = reinterpret_cast<int4*>(v352 + v356);
            assert("Pointer alignment check" && (unsigned long long)(v358) % 4l == 0 && (unsigned long long)(v359) % 4l == 0);
            *v359 = *v358;
            v354 += 1l ;
        }
        int v360;
        v360 = 0l;
        while (while_method_1(v360)){
            int v362;
            v362 = 0l;
            while (while_method_2(v362)){
                bool v364;
                v364 = 0l <= v362;
                bool v366;
                if (v364){
                    bool v365;
                    v365 = v362 < 4l;
                    v366 = v365;
                } else {
                    v366 = false;
                }
                bool v367;
                v367 = v366 == false;
                if (v367){
                    assert("The indices should be inside the range of the dimension." && v366);
                } else {
                }
                bool v369;
                v369 = 0l <= v343;
                bool v371;
                if (v369){
                    bool v370;
                    v370 = v343 < 1l;
                    v371 = v370;
                } else {
                    v371 = false;
                }
                bool v372;
                v372 = v371 == false;
                if (v372){
                    assert("The indices should be inside the range of the dimension." && v371);
                } else {
                }
                int v374;
                v374 = v343 * 4l;
                int v375;
                v375 = v362 + v374;
                bool v376;
                v376 = 0l <= v360;
                bool v378;
                if (v376){
                    bool v377;
                    v377 = v360 < 1l;
                    v378 = v377;
                } else {
                    v378 = false;
                }
                bool v379;
                v379 = v378 == false;
                if (v379){
                    assert("The indices should be inside the range of the dimension." && v378);
                } else {
                }
                int v381;
                v381 = v360 * 4l;
                int v382;
                v382 = v375 + v381;
                assert("Tensor range check" && 0 <= v360 && v360 < 1l);
                assert("Tensor range check" && 0 <= v362 && v362 < 4l);
                int v383;
                v383 = 4l * v360;
                int v384;
                v384 = v383 + v362;
                v353[v384] = v382;
                v362 += 1l ;
            }
            v360 += 1l ;
        }
        bool v385;
        v385 = 0l <= v347;
        bool v387;
        if (v385){
            bool v386;
            v386 = v347 < 1l;
            v387 = v386;
        } else {
            v387 = false;
        }
        bool v388;
        v388 = v387 == false;
        if (v388){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v387);
        } else {
        }
        bool v390;
        v390 = v340 && v344;
        bool v391;
        v391 = v390 == false;
        if (v391){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v390);
        } else {
        }
        int v393;
        v393 = v339 + v347;
        bool v394[4l];
        int v395;
        v395 = 0l;
        while (while_method_1(v395)){
            int v397;
            v397 = 0l;
            while (while_method_2(v397)){
                assert("Tensor range check" && 0 <= v395 && v395 < 1l);
                assert("Tensor range check" && 0 <= v397 && v397 < 4l);
                int v399;
                v399 = 4l * v395;
                int v400;
                v400 = v399 + v397;
                float v401;
                v401 = v352[v400];
                int v402;
                v402 = v353[v400];
                bool v403;
                v403 = v402 < 3l;
                assert("Tensor range check" && 0 <= v395 && v395 < 1l);
                assert("Tensor range check" && 0 <= v397 && v397 < 4l);
                v394[v400] = v403;
                v397 += 1l ;
            }
            v395 += 1l ;
        }
        float v404[4l];
        int v405;
        v405 = 0l;
        while (while_method_1(v405)){
            int v407;
            v407 = 0l;
            while (while_method_2(v407)){
                assert("Tensor range check" && 0 <= v405 && v405 < 1l);
                assert("Tensor range check" && 0 <= v407 && v407 < 4l);
                int v409;
                v409 = 4l * v405;
                int v410;
                v410 = v409 + v407;
                float v411;
                v411 = v352[v410];
                bool v412;
                v412 = v394[v410];
                float v415;
                if (v412){
                    bool v413;
                    v413 = 0.0f >= v411;
                    if (v413){
                        v415 = 0.0f;
                    } else {
                        v415 = v411;
                    }
                } else {
                    v415 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v405 && v405 < 1l);
                assert("Tensor range check" && 0 <= v407 && v407 < 4l);
                v404[v410] = v415;
                v407 += 1l ;
            }
            v405 += 1l ;
        }
        float v416;
        v416 = 0.0f;
        int v417;
        v417 = 0l;
        while (while_method_1(v417)){
            int v419;
            v419 = 0l;
            while (while_method_2(v419)){
                assert("Tensor range check" && 0 <= v417 && v417 < 1l);
                assert("Tensor range check" && 0 <= v419 && v419 < 4l);
                int v421;
                v421 = 4l * v417;
                int v422;
                v422 = v421 + v419;
                float v423;
                v423 = v404[v422];
                float v424;
                v424 = v416 + v423;
                v416 = v424;
                v419 += 1l ;
            }
            v417 += 1l ;
        }
        auto v425 = cooperative_groups::coalesced_threads();
        int v426;
        v426 = threadIdx.x;
        auto v427 = cooperative_groups::labeled_partition(v425,v426);
        Closure0 v428{};
        float v429;
        v429 = cooperative_groups::reduce(v427, v416, v428);
        int v430[4l];
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
                bool v437;
                v437 = v394[v436];
                int v438;
                if (v437){
                    v438 = 1l;
                } else {
                    v438 = 0l;
                }
                assert("Tensor range check" && 0 <= v431 && v431 < 1l);
                assert("Tensor range check" && 0 <= v433 && v433 < 4l);
                v430[v436] = v438;
                v433 += 1l ;
            }
            v431 += 1l ;
        }
        int v439;
        v439 = 0l;
        int v440;
        v440 = 0l;
        while (while_method_1(v440)){
            int v442;
            v442 = 0l;
            while (while_method_2(v442)){
                assert("Tensor range check" && 0 <= v440 && v440 < 1l);
                assert("Tensor range check" && 0 <= v442 && v442 < 4l);
                int v444;
                v444 = 4l * v440;
                int v445;
                v445 = v444 + v442;
                int v446;
                v446 = v430[v445];
                int v447;
                v447 = v439 + v446;
                v439 = v447;
                v442 += 1l ;
            }
            v440 += 1l ;
        }
        auto v448 = cooperative_groups::coalesced_threads();
        int v449;
        v449 = threadIdx.x;
        auto v450 = cooperative_groups::labeled_partition(v448,v449);
        Closure1 v451{};
        int v452;
        v452 = cooperative_groups::reduce(v450, v439, v451);
        float v453;
        v453 = (float)v452;
        float v454;
        v454 = 1.0f / v453;
        float v455[4l];
        int v456;
        v456 = 0l;
        while (while_method_1(v456)){
            int v458;
            v458 = 0l;
            while (while_method_2(v458)){
                assert("Tensor range check" && 0 <= v456 && v456 < 1l);
                assert("Tensor range check" && 0 <= v458 && v458 < 4l);
                int v460;
                v460 = 4l * v456;
                int v461;
                v461 = v460 + v458;
                float v462;
                v462 = v404[v461];
                bool v463;
                v463 = v394[v461];
                bool v464;
                v464 = v463 == false;
                float v469;
                if (v464){
                    v469 = 0.0f;
                } else {
                    bool v465;
                    v465 = v429 == 0.0f;
                    bool v466;
                    v466 = v465 != true;
                    if (v466){
                        float v467;
                        v467 = v462 / v429;
                        v469 = v467;
                    } else {
                        v469 = v454;
                    }
                }
                assert("Tensor range check" && 0 <= v456 && v456 < 1l);
                assert("Tensor range check" && 0 <= v458 && v458 < 4l);
                v455[v461] = v469;
                v458 += 1l ;
            }
            v456 += 1l ;
        }
        float v470[4l];
        float v471;
        v471 = 0.0f;
        int v472;
        v472 = 0l;
        while (while_method_1(v472)){
            assert("Tensor range check" && 0 <= v472 && v472 < 1l);
            int v474;
            v474 = 4l * v472;
            assert("Tensor range check" && 0 <= v472 && v472 < 1l);
            int v475; float v476;
            Tuple0 tmp11 = Tuple0{0l, 0.0f};
            v475 = tmp11.v0; v476 = tmp11.v1;
            while (while_method_2(v475)){
                assert("Tensor range check" && 0 <= v475 && v475 < 4l);
                int v478;
                v478 = v475 + v474;
                float v479;
                v479 = v455[v478];
                float v480;
                v480 = v476 + v479;
                v476 = v480;
                v475 += 1l ;
            }
            auto v481 = cooperative_groups::coalesced_threads();
            int v482;
            v482 = threadIdx.x;
            auto v483 = cooperative_groups::labeled_partition(v481,v482);
            Closure2 v484{};
            float v485;
            v485 = cooperative_groups::inclusive_scan(v483, v476, v484);
            float v486;
            v486 = v483.shfl_up(v485,1);
            bool v487;
            v487 = v483.thread_rank() == 0;
            float v488;
            if (v487){
                v488 = 0.0f;
            } else {
                v488 = v486;
            }
            float v489;
            v489 = v483.shfl(v485,v483.num_threads()-1);
            float v490;
            v490 = v471 + v488;
            int v491; float v492;
            Tuple0 tmp12 = Tuple0{0l, v490};
            v491 = tmp12.v0; v492 = tmp12.v1;
            while (while_method_2(v491)){
                assert("Tensor range check" && 0 <= v491 && v491 < 4l);
                int v494;
                v494 = v491 + v474;
                float v495;
                v495 = v455[v494];
                float v496;
                v496 = v492 + v495;
                assert("Tensor range check" && 0 <= v491 && v491 < 4l);
                v470[v494] = v496;
                v492 = v496;
                v491 += 1l ;
            }
            float v497;
            v497 = v471 + v489;
            v471 = v497;
            v472 += 1l ;
        }
        float v498[4l];
        bool v499[4l];
        int v500;
        v500 = 0l;
        while (while_method_1(v500)){
            int v502;
            v502 = 0l;
            while (while_method_2(v502)){
                assert("Tensor range check" && 0 <= v500 && v500 < 1l);
                assert("Tensor range check" && 0 <= v502 && v502 < 4l);
                int v504;
                v504 = 4l * v500;
                int v505;
                v505 = v504 + v502;
                float v506;
                v506 = v470[v505];
                float v507;
                v507 = v455[v505];
                bool v508;
                v508 = v507 > 0.0f;
                assert("Tensor range check" && 0 <= v500 && v500 < 1l);
                assert("Tensor range check" && 0 <= v502 && v502 < 4l);
                v498[v505] = v506;
                v499[v505] = v508;
                v502 += 1l ;
            }
            v500 += 1l ;
        }
        float v509; bool v510;
        Tuple1 tmp13 = Tuple1{-1.0f / 0.0f, false};
        v509 = tmp13.v0; v510 = tmp13.v1;
        int v511;
        v511 = 0l;
        while (while_method_1(v511)){
            int v513;
            v513 = 0l;
            while (while_method_2(v513)){
                assert("Tensor range check" && 0 <= v511 && v511 < 1l);
                assert("Tensor range check" && 0 <= v513 && v513 < 4l);
                int v515;
                v515 = 4l * v511;
                int v516;
                v516 = v515 + v513;
                float v517;
                v517 = v498[v516];
                bool v518;
                v518 = v499[v516];
                float v525; bool v526;
                if (v510){
                    if (v518){
                        bool v519;
                        v519 = v509 >= v517;
                        float v520;
                        if (v519){
                            v520 = v509;
                        } else {
                            v520 = v517;
                        }
                        v525 = v520; v526 = true;
                    } else {
                        v525 = v509; v526 = v510;
                    }
                } else {
                    if (v518){
                        v525 = v517; v526 = v518;
                    } else {
                        v525 = v509; v526 = v510;
                    }
                }
                v509 = v525;
                v510 = v526;
                v513 += 1l ;
            }
            v511 += 1l ;
        }
        auto v527 = cooperative_groups::coalesced_threads();
        int v528;
        v528 = threadIdx.x;
        auto v529 = cooperative_groups::labeled_partition(v527,v528);
        Closure3 v530{};
        float v531; bool v532;
        Tuple1 tmp14 = cooperative_groups::reduce(v529, Tuple1{v509, v510}, v530);
        v531 = tmp14.v0; v532 = tmp14.v1;
        bool v533;
        v533 = v532 == false;
        if (v533){
            assert("The local reduce must be true." && v532);
        } else {
        }
        float v535[4l];
        int v536[4l];
        int v537;
        v537 = 0l;
        while (while_method_1(v537)){
            int v539;
            v539 = 0l;
            while (while_method_2(v539)){
                assert("Tensor range check" && 0 <= v537 && v537 < 1l);
                assert("Tensor range check" && 0 <= v539 && v539 < 4l);
                int v541;
                v541 = 4l * v537;
                int v542;
                v542 = v541 + v539;
                int v543;
                v543 = v353[v542];
                float v544;
                v544 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v537 && v537 < 1l);
                assert("Tensor range check" && 0 <= v539 && v539 < 4l);
                v535[v542] = v544;
                v536[v542] = v543;
                v539 += 1l ;
            }
            v537 += 1l ;
        }
        float v545; int v546;
        Tuple2 tmp15 = Tuple2{0.0f, 2147483647l};
        v545 = tmp15.v0; v546 = tmp15.v1;
        int v547;
        v547 = 0l;
        while (while_method_1(v547)){
            int v549;
            v549 = 0l;
            while (while_method_2(v549)){
                assert("Tensor range check" && 0 <= v547 && v547 < 1l);
                assert("Tensor range check" && 0 <= v549 && v549 < 4l);
                int v551;
                v551 = 4l * v547;
                int v552;
                v552 = v551 + v549;
                float v553;
                v553 = v535[v552];
                int v554;
                v554 = v536[v552];
                bool v555;
                v555 = v546 < v554;
                float v556; int v557;
                if (v555){
                    v556 = v545; v557 = v546;
                } else {
                    v556 = v553; v557 = v554;
                }
                v545 = v556;
                v546 = v557;
                v549 += 1l ;
            }
            v547 += 1l ;
        }
        auto v558 = cooperative_groups::coalesced_threads();
        int v559;
        v559 = threadIdx.x;
        auto v560 = cooperative_groups::labeled_partition(v558,v559);
        Closure4 v561{};
        float v562; int v563;
        Tuple2 tmp16 = cooperative_groups::reduce(v560, Tuple2{v545, v546}, v561);
        v562 = tmp16.v0; v563 = tmp16.v1;
        float v564;
        v564 = v531 * v562;
        int v565[4l];
        bool v566[4l];
        int v567;
        v567 = 0l;
        while (while_method_1(v567)){
            int v569;
            v569 = 0l;
            while (while_method_2(v569)){
                assert("Tensor range check" && 0 <= v567 && v567 < 1l);
                assert("Tensor range check" && 0 <= v569 && v569 < 4l);
                int v571;
                v571 = 4l * v567;
                int v572;
                v572 = v571 + v569;
                float v573;
                v573 = v498[v572];
                bool v574;
                v574 = v499[v572];
                int v575;
                v575 = v353[v572];
                int v578; bool v579;
                if (v574){
                    float v576;
                    v576 = v573 - v564;
                    bool v577;
                    v577 = v576 >= 0.0f;
                    v578 = v575; v579 = v577;
                } else {
                    v578 = 2147483647l; v579 = false;
                }
                assert("Tensor range check" && 0 <= v567 && v567 < 1l);
                assert("Tensor range check" && 0 <= v569 && v569 < 4l);
                v565[v572] = v578;
                v566[v572] = v579;
                v569 += 1l ;
            }
            v567 += 1l ;
        }
        int v580; bool v581;
        Tuple3 tmp17 = Tuple3{2147483647l, false};
        v580 = tmp17.v0; v581 = tmp17.v1;
        int v582;
        v582 = 0l;
        while (while_method_1(v582)){
            int v584;
            v584 = 0l;
            while (while_method_2(v584)){
                assert("Tensor range check" && 0 <= v582 && v582 < 1l);
                assert("Tensor range check" && 0 <= v584 && v584 < 4l);
                int v586;
                v586 = 4l * v582;
                int v587;
                v587 = v586 + v584;
                int v588;
                v588 = v565[v587];
                bool v589;
                v589 = v566[v587];
                int v596; bool v597;
                if (v581){
                    if (v589){
                        bool v590;
                        v590 = v580 < v588;
                        int v591;
                        if (v590){
                            v591 = v580;
                        } else {
                            v591 = v588;
                        }
                        v596 = v591; v597 = true;
                    } else {
                        v596 = v580; v597 = v581;
                    }
                } else {
                    if (v589){
                        v596 = v588; v597 = v589;
                    } else {
                        v596 = v580; v597 = v581;
                    }
                }
                v580 = v596;
                v581 = v597;
                v584 += 1l ;
            }
            v582 += 1l ;
        }
        auto v598 = cooperative_groups::coalesced_threads();
        int v599;
        v599 = threadIdx.x;
        auto v600 = cooperative_groups::labeled_partition(v598,v599);
        Closure5 v601{};
        int v602; bool v603;
        Tuple3 tmp18 = cooperative_groups::reduce(v600, Tuple3{v580, v581}, v601);
        v602 = tmp18.v0; v603 = tmp18.v1;
        bool v604;
        v604 = v603 == false;
        if (v604){
            assert("The local reduce must be true." && v603);
        } else {
        }
        float v606; int v607;
        Tuple2 tmp19 = Tuple2{0.0f, 2147483647l};
        v606 = tmp19.v0; v607 = tmp19.v1;
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
                v614 = v455[v613];
                int v615;
                v615 = v353[v613];
                bool v616;
                v616 = v607 == v602;
                float v620; int v621;
                if (v616){
                    v620 = v606; v621 = v607;
                } else {
                    bool v617;
                    v617 = v615 == v602;
                    if (v617){
                        v620 = v614; v621 = v615;
                    } else {
                        v620 = v606; v621 = v607;
                    }
                }
                v606 = v620;
                v607 = v621;
                v610 += 1l ;
            }
            v608 += 1l ;
        }
        auto v622 = cooperative_groups::coalesced_threads();
        int v623;
        v623 = threadIdx.x;
        auto v624 = cooperative_groups::labeled_partition(v622,v623);
        Closure6 v625{v602};
        float v626; int v627;
        Tuple2 tmp20 = cooperative_groups::reduce(v624, Tuple2{v606, v607}, v625);
        v626 = tmp20.v0; v627 = tmp20.v1;
        bool v628;
        v628 = v627 == 2147483647l;
        bool v629;
        v629 = v628 != true;
        bool v630;
        v630 = v629 == false;
        if (v630){
            assert("Expected a valid action id in get_action." && v629);
        } else {
        }
        assert("Tensor range check" && 0 <= v347 && v347 < 1l);
        v334[v349] = v602;
        v335[v349] = v626;
        v347 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v632;
    v632 = threadIdx.x;
    assert("Tensor range check" && 0 <= v632 && v632 < 32l);
    int v633;
    v633 = v334[v632];
    float v634;
    v634 = v335[v632];
    v332[0l] = Tuple0{v633, v634};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v635; float v636;
    Tuple0 tmp21 = v332[0l];
    v635 = tmp21.v0; v636 = tmp21.v1;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v17, v18, v19, v20, v331, v330, v636, v635);
    int v637;
    v637 = 790l;
    int v638;
    v638 = 0l;
    Tuple0 v639[1l];
    __shared__ float * v640[32l];
    __shared__ int v641[32l];
    __shared__ float v642[32l];
    int v643;
    v643 = threadIdx.x;
    float * v644;
    v644 = v8+3160l;
    assert("Tensor range check" && 0 <= v643 && v643 < 32l);
    v640[v643] = v644;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v646;
    v646 = threadIdx.x;
    bool v647;
    v647 = 0l <= v646;
    bool v648;
    v648 = v647 == false;
    if (v648){
        assert("The index needs to be zero or positive." && v647);
    } else {
    }
    int v650;
    v650 = v646 % 1l;
    bool v651;
    v651 = v646 < 32l;
    bool v652;
    v652 = v651 == false;
    if (v652){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v651);
    } else {
    }
    assert("Tensor range check" && 0 <= v646 && v646 < 32l);
    assert("Tensor range check" && 0 <= v646 && v646 < 32l);
    int v654;
    v654 = 0l;
    while (while_method_1(v654)){
        assert("Tensor range check" && 0 <= v654 && v654 < 1l);
        int v656;
        v656 = v654 + v646;
        float * v657;
        v657 = v640[v656];
        assert("Tensor range check" && 0 <= v650 && v650 < 1l);
        int v658;
        v658 = 4l * v650;
        float v659[4l];
        int v660[4l];
        int v661;
        v661 = 0l;
        while (while_method_1(v661)){
            assert("Tensor range check" && 0 <= v661 && v661 < 1l);
            int v663;
            v663 = 4l * v661;
            assert("Tensor range check" && 0 <= v661 && v661 < 1l);
            int v664;
            v664 = v663 + v658;
            int4* v665;
            v665 = reinterpret_cast<int4*>(v657 + v664);
            int4* v666;
            v666 = reinterpret_cast<int4*>(v659 + v663);
            assert("Pointer alignment check" && (unsigned long long)(v665) % 4l == 0 && (unsigned long long)(v666) % 4l == 0);
            *v666 = *v665;
            v661 += 1l ;
        }
        int v667;
        v667 = 0l;
        while (while_method_1(v667)){
            int v669;
            v669 = 0l;
            while (while_method_2(v669)){
                bool v671;
                v671 = 0l <= v669;
                bool v673;
                if (v671){
                    bool v672;
                    v672 = v669 < 4l;
                    v673 = v672;
                } else {
                    v673 = false;
                }
                bool v674;
                v674 = v673 == false;
                if (v674){
                    assert("The indices should be inside the range of the dimension." && v673);
                } else {
                }
                bool v676;
                v676 = 0l <= v650;
                bool v678;
                if (v676){
                    bool v677;
                    v677 = v650 < 1l;
                    v678 = v677;
                } else {
                    v678 = false;
                }
                bool v679;
                v679 = v678 == false;
                if (v679){
                    assert("The indices should be inside the range of the dimension." && v678);
                } else {
                }
                int v681;
                v681 = v650 * 4l;
                int v682;
                v682 = v669 + v681;
                bool v683;
                v683 = 0l <= v667;
                bool v685;
                if (v683){
                    bool v684;
                    v684 = v667 < 1l;
                    v685 = v684;
                } else {
                    v685 = false;
                }
                bool v686;
                v686 = v685 == false;
                if (v686){
                    assert("The indices should be inside the range of the dimension." && v685);
                } else {
                }
                int v688;
                v688 = v667 * 4l;
                int v689;
                v689 = v682 + v688;
                assert("Tensor range check" && 0 <= v667 && v667 < 1l);
                assert("Tensor range check" && 0 <= v669 && v669 < 4l);
                int v690;
                v690 = 4l * v667;
                int v691;
                v691 = v690 + v669;
                v660[v691] = v689;
                v669 += 1l ;
            }
            v667 += 1l ;
        }
        bool v692;
        v692 = 0l <= v654;
        bool v694;
        if (v692){
            bool v693;
            v693 = v654 < 1l;
            v694 = v693;
        } else {
            v694 = false;
        }
        bool v695;
        v695 = v694 == false;
        if (v695){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v694);
        } else {
        }
        bool v697;
        v697 = v647 && v651;
        bool v698;
        v698 = v697 == false;
        if (v698){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v697);
        } else {
        }
        int v700;
        v700 = v646 + v654;
        bool v701[4l];
        int v702;
        v702 = 0l;
        while (while_method_1(v702)){
            int v704;
            v704 = 0l;
            while (while_method_2(v704)){
                assert("Tensor range check" && 0 <= v702 && v702 < 1l);
                assert("Tensor range check" && 0 <= v704 && v704 < 4l);
                int v706;
                v706 = 4l * v702;
                int v707;
                v707 = v706 + v704;
                float v708;
                v708 = v659[v707];
                int v709;
                v709 = v660[v707];
                bool v710;
                v710 = v709 < 3l;
                assert("Tensor range check" && 0 <= v702 && v702 < 1l);
                assert("Tensor range check" && 0 <= v704 && v704 < 4l);
                v701[v707] = v710;
                v704 += 1l ;
            }
            v702 += 1l ;
        }
        float v711[4l];
        int v712;
        v712 = 0l;
        while (while_method_1(v712)){
            int v714;
            v714 = 0l;
            while (while_method_2(v714)){
                assert("Tensor range check" && 0 <= v712 && v712 < 1l);
                assert("Tensor range check" && 0 <= v714 && v714 < 4l);
                int v716;
                v716 = 4l * v712;
                int v717;
                v717 = v716 + v714;
                float v718;
                v718 = v659[v717];
                bool v719;
                v719 = v701[v717];
                float v722;
                if (v719){
                    bool v720;
                    v720 = 0.0f >= v718;
                    if (v720){
                        v722 = 0.0f;
                    } else {
                        v722 = v718;
                    }
                } else {
                    v722 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v712 && v712 < 1l);
                assert("Tensor range check" && 0 <= v714 && v714 < 4l);
                v711[v717] = v722;
                v714 += 1l ;
            }
            v712 += 1l ;
        }
        float v723;
        v723 = 0.0f;
        int v724;
        v724 = 0l;
        while (while_method_1(v724)){
            int v726;
            v726 = 0l;
            while (while_method_2(v726)){
                assert("Tensor range check" && 0 <= v724 && v724 < 1l);
                assert("Tensor range check" && 0 <= v726 && v726 < 4l);
                int v728;
                v728 = 4l * v724;
                int v729;
                v729 = v728 + v726;
                float v730;
                v730 = v711[v729];
                float v731;
                v731 = v723 + v730;
                v723 = v731;
                v726 += 1l ;
            }
            v724 += 1l ;
        }
        auto v732 = cooperative_groups::coalesced_threads();
        int v733;
        v733 = threadIdx.x;
        auto v734 = cooperative_groups::labeled_partition(v732,v733);
        Closure0 v735{};
        float v736;
        v736 = cooperative_groups::reduce(v734, v723, v735);
        int v737[4l];
        int v738;
        v738 = 0l;
        while (while_method_1(v738)){
            int v740;
            v740 = 0l;
            while (while_method_2(v740)){
                assert("Tensor range check" && 0 <= v738 && v738 < 1l);
                assert("Tensor range check" && 0 <= v740 && v740 < 4l);
                int v742;
                v742 = 4l * v738;
                int v743;
                v743 = v742 + v740;
                bool v744;
                v744 = v701[v743];
                int v745;
                if (v744){
                    v745 = 1l;
                } else {
                    v745 = 0l;
                }
                assert("Tensor range check" && 0 <= v738 && v738 < 1l);
                assert("Tensor range check" && 0 <= v740 && v740 < 4l);
                v737[v743] = v745;
                v740 += 1l ;
            }
            v738 += 1l ;
        }
        int v746;
        v746 = 0l;
        int v747;
        v747 = 0l;
        while (while_method_1(v747)){
            int v749;
            v749 = 0l;
            while (while_method_2(v749)){
                assert("Tensor range check" && 0 <= v747 && v747 < 1l);
                assert("Tensor range check" && 0 <= v749 && v749 < 4l);
                int v751;
                v751 = 4l * v747;
                int v752;
                v752 = v751 + v749;
                int v753;
                v753 = v737[v752];
                int v754;
                v754 = v746 + v753;
                v746 = v754;
                v749 += 1l ;
            }
            v747 += 1l ;
        }
        auto v755 = cooperative_groups::coalesced_threads();
        int v756;
        v756 = threadIdx.x;
        auto v757 = cooperative_groups::labeled_partition(v755,v756);
        Closure1 v758{};
        int v759;
        v759 = cooperative_groups::reduce(v757, v746, v758);
        float v760;
        v760 = (float)v759;
        float v761;
        v761 = 1.0f / v760;
        float v762[4l];
        int v763;
        v763 = 0l;
        while (while_method_1(v763)){
            int v765;
            v765 = 0l;
            while (while_method_2(v765)){
                assert("Tensor range check" && 0 <= v763 && v763 < 1l);
                assert("Tensor range check" && 0 <= v765 && v765 < 4l);
                int v767;
                v767 = 4l * v763;
                int v768;
                v768 = v767 + v765;
                float v769;
                v769 = v711[v768];
                bool v770;
                v770 = v701[v768];
                bool v771;
                v771 = v770 == false;
                float v776;
                if (v771){
                    v776 = 0.0f;
                } else {
                    bool v772;
                    v772 = v736 == 0.0f;
                    bool v773;
                    v773 = v772 != true;
                    if (v773){
                        float v774;
                        v774 = v769 / v736;
                        v776 = v774;
                    } else {
                        v776 = v761;
                    }
                }
                assert("Tensor range check" && 0 <= v763 && v763 < 1l);
                assert("Tensor range check" && 0 <= v765 && v765 < 4l);
                v762[v768] = v776;
                v765 += 1l ;
            }
            v763 += 1l ;
        }
        float v777[4l];
        float v778;
        v778 = 0.0f;
        int v779;
        v779 = 0l;
        while (while_method_1(v779)){
            assert("Tensor range check" && 0 <= v779 && v779 < 1l);
            int v781;
            v781 = 4l * v779;
            assert("Tensor range check" && 0 <= v779 && v779 < 1l);
            int v782; float v783;
            Tuple0 tmp22 = Tuple0{0l, 0.0f};
            v782 = tmp22.v0; v783 = tmp22.v1;
            while (while_method_2(v782)){
                assert("Tensor range check" && 0 <= v782 && v782 < 4l);
                int v785;
                v785 = v782 + v781;
                float v786;
                v786 = v762[v785];
                float v787;
                v787 = v783 + v786;
                v783 = v787;
                v782 += 1l ;
            }
            auto v788 = cooperative_groups::coalesced_threads();
            int v789;
            v789 = threadIdx.x;
            auto v790 = cooperative_groups::labeled_partition(v788,v789);
            Closure2 v791{};
            float v792;
            v792 = cooperative_groups::inclusive_scan(v790, v783, v791);
            float v793;
            v793 = v790.shfl_up(v792,1);
            bool v794;
            v794 = v790.thread_rank() == 0;
            float v795;
            if (v794){
                v795 = 0.0f;
            } else {
                v795 = v793;
            }
            float v796;
            v796 = v790.shfl(v792,v790.num_threads()-1);
            float v797;
            v797 = v778 + v795;
            int v798; float v799;
            Tuple0 tmp23 = Tuple0{0l, v797};
            v798 = tmp23.v0; v799 = tmp23.v1;
            while (while_method_2(v798)){
                assert("Tensor range check" && 0 <= v798 && v798 < 4l);
                int v801;
                v801 = v798 + v781;
                float v802;
                v802 = v762[v801];
                float v803;
                v803 = v799 + v802;
                assert("Tensor range check" && 0 <= v798 && v798 < 4l);
                v777[v801] = v803;
                v799 = v803;
                v798 += 1l ;
            }
            float v804;
            v804 = v778 + v796;
            v778 = v804;
            v779 += 1l ;
        }
        float v805[4l];
        bool v806[4l];
        int v807;
        v807 = 0l;
        while (while_method_1(v807)){
            int v809;
            v809 = 0l;
            while (while_method_2(v809)){
                assert("Tensor range check" && 0 <= v807 && v807 < 1l);
                assert("Tensor range check" && 0 <= v809 && v809 < 4l);
                int v811;
                v811 = 4l * v807;
                int v812;
                v812 = v811 + v809;
                float v813;
                v813 = v777[v812];
                float v814;
                v814 = v762[v812];
                bool v815;
                v815 = v814 > 0.0f;
                assert("Tensor range check" && 0 <= v807 && v807 < 1l);
                assert("Tensor range check" && 0 <= v809 && v809 < 4l);
                v805[v812] = v813;
                v806[v812] = v815;
                v809 += 1l ;
            }
            v807 += 1l ;
        }
        float v816; bool v817;
        Tuple1 tmp24 = Tuple1{-1.0f / 0.0f, false};
        v816 = tmp24.v0; v817 = tmp24.v1;
        int v818;
        v818 = 0l;
        while (while_method_1(v818)){
            int v820;
            v820 = 0l;
            while (while_method_2(v820)){
                assert("Tensor range check" && 0 <= v818 && v818 < 1l);
                assert("Tensor range check" && 0 <= v820 && v820 < 4l);
                int v822;
                v822 = 4l * v818;
                int v823;
                v823 = v822 + v820;
                float v824;
                v824 = v805[v823];
                bool v825;
                v825 = v806[v823];
                float v832; bool v833;
                if (v817){
                    if (v825){
                        bool v826;
                        v826 = v816 >= v824;
                        float v827;
                        if (v826){
                            v827 = v816;
                        } else {
                            v827 = v824;
                        }
                        v832 = v827; v833 = true;
                    } else {
                        v832 = v816; v833 = v817;
                    }
                } else {
                    if (v825){
                        v832 = v824; v833 = v825;
                    } else {
                        v832 = v816; v833 = v817;
                    }
                }
                v816 = v832;
                v817 = v833;
                v820 += 1l ;
            }
            v818 += 1l ;
        }
        auto v834 = cooperative_groups::coalesced_threads();
        int v835;
        v835 = threadIdx.x;
        auto v836 = cooperative_groups::labeled_partition(v834,v835);
        Closure3 v837{};
        float v838; bool v839;
        Tuple1 tmp25 = cooperative_groups::reduce(v836, Tuple1{v816, v817}, v837);
        v838 = tmp25.v0; v839 = tmp25.v1;
        bool v840;
        v840 = v839 == false;
        if (v840){
            assert("The local reduce must be true." && v839);
        } else {
        }
        float v842[4l];
        int v843[4l];
        int v844;
        v844 = 0l;
        while (while_method_1(v844)){
            int v846;
            v846 = 0l;
            while (while_method_2(v846)){
                assert("Tensor range check" && 0 <= v844 && v844 < 1l);
                assert("Tensor range check" && 0 <= v846 && v846 < 4l);
                int v848;
                v848 = 4l * v844;
                int v849;
                v849 = v848 + v846;
                int v850;
                v850 = v660[v849];
                float v851;
                v851 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v844 && v844 < 1l);
                assert("Tensor range check" && 0 <= v846 && v846 < 4l);
                v842[v849] = v851;
                v843[v849] = v850;
                v846 += 1l ;
            }
            v844 += 1l ;
        }
        float v852; int v853;
        Tuple2 tmp26 = Tuple2{0.0f, 2147483647l};
        v852 = tmp26.v0; v853 = tmp26.v1;
        int v854;
        v854 = 0l;
        while (while_method_1(v854)){
            int v856;
            v856 = 0l;
            while (while_method_2(v856)){
                assert("Tensor range check" && 0 <= v854 && v854 < 1l);
                assert("Tensor range check" && 0 <= v856 && v856 < 4l);
                int v858;
                v858 = 4l * v854;
                int v859;
                v859 = v858 + v856;
                float v860;
                v860 = v842[v859];
                int v861;
                v861 = v843[v859];
                bool v862;
                v862 = v853 < v861;
                float v863; int v864;
                if (v862){
                    v863 = v852; v864 = v853;
                } else {
                    v863 = v860; v864 = v861;
                }
                v852 = v863;
                v853 = v864;
                v856 += 1l ;
            }
            v854 += 1l ;
        }
        auto v865 = cooperative_groups::coalesced_threads();
        int v866;
        v866 = threadIdx.x;
        auto v867 = cooperative_groups::labeled_partition(v865,v866);
        Closure4 v868{};
        float v869; int v870;
        Tuple2 tmp27 = cooperative_groups::reduce(v867, Tuple2{v852, v853}, v868);
        v869 = tmp27.v0; v870 = tmp27.v1;
        float v871;
        v871 = v838 * v869;
        int v872[4l];
        bool v873[4l];
        int v874;
        v874 = 0l;
        while (while_method_1(v874)){
            int v876;
            v876 = 0l;
            while (while_method_2(v876)){
                assert("Tensor range check" && 0 <= v874 && v874 < 1l);
                assert("Tensor range check" && 0 <= v876 && v876 < 4l);
                int v878;
                v878 = 4l * v874;
                int v879;
                v879 = v878 + v876;
                float v880;
                v880 = v805[v879];
                bool v881;
                v881 = v806[v879];
                int v882;
                v882 = v660[v879];
                int v885; bool v886;
                if (v881){
                    float v883;
                    v883 = v880 - v871;
                    bool v884;
                    v884 = v883 >= 0.0f;
                    v885 = v882; v886 = v884;
                } else {
                    v885 = 2147483647l; v886 = false;
                }
                assert("Tensor range check" && 0 <= v874 && v874 < 1l);
                assert("Tensor range check" && 0 <= v876 && v876 < 4l);
                v872[v879] = v885;
                v873[v879] = v886;
                v876 += 1l ;
            }
            v874 += 1l ;
        }
        int v887; bool v888;
        Tuple3 tmp28 = Tuple3{2147483647l, false};
        v887 = tmp28.v0; v888 = tmp28.v1;
        int v889;
        v889 = 0l;
        while (while_method_1(v889)){
            int v891;
            v891 = 0l;
            while (while_method_2(v891)){
                assert("Tensor range check" && 0 <= v889 && v889 < 1l);
                assert("Tensor range check" && 0 <= v891 && v891 < 4l);
                int v893;
                v893 = 4l * v889;
                int v894;
                v894 = v893 + v891;
                int v895;
                v895 = v872[v894];
                bool v896;
                v896 = v873[v894];
                int v903; bool v904;
                if (v888){
                    if (v896){
                        bool v897;
                        v897 = v887 < v895;
                        int v898;
                        if (v897){
                            v898 = v887;
                        } else {
                            v898 = v895;
                        }
                        v903 = v898; v904 = true;
                    } else {
                        v903 = v887; v904 = v888;
                    }
                } else {
                    if (v896){
                        v903 = v895; v904 = v896;
                    } else {
                        v903 = v887; v904 = v888;
                    }
                }
                v887 = v903;
                v888 = v904;
                v891 += 1l ;
            }
            v889 += 1l ;
        }
        auto v905 = cooperative_groups::coalesced_threads();
        int v906;
        v906 = threadIdx.x;
        auto v907 = cooperative_groups::labeled_partition(v905,v906);
        Closure5 v908{};
        int v909; bool v910;
        Tuple3 tmp29 = cooperative_groups::reduce(v907, Tuple3{v887, v888}, v908);
        v909 = tmp29.v0; v910 = tmp29.v1;
        bool v911;
        v911 = v910 == false;
        if (v911){
            assert("The local reduce must be true." && v910);
        } else {
        }
        float v913; int v914;
        Tuple2 tmp30 = Tuple2{0.0f, 2147483647l};
        v913 = tmp30.v0; v914 = tmp30.v1;
        int v915;
        v915 = 0l;
        while (while_method_1(v915)){
            int v917;
            v917 = 0l;
            while (while_method_2(v917)){
                assert("Tensor range check" && 0 <= v915 && v915 < 1l);
                assert("Tensor range check" && 0 <= v917 && v917 < 4l);
                int v919;
                v919 = 4l * v915;
                int v920;
                v920 = v919 + v917;
                float v921;
                v921 = v762[v920];
                int v922;
                v922 = v660[v920];
                bool v923;
                v923 = v914 == v909;
                float v927; int v928;
                if (v923){
                    v927 = v913; v928 = v914;
                } else {
                    bool v924;
                    v924 = v922 == v909;
                    if (v924){
                        v927 = v921; v928 = v922;
                    } else {
                        v927 = v913; v928 = v914;
                    }
                }
                v913 = v927;
                v914 = v928;
                v917 += 1l ;
            }
            v915 += 1l ;
        }
        auto v929 = cooperative_groups::coalesced_threads();
        int v930;
        v930 = threadIdx.x;
        auto v931 = cooperative_groups::labeled_partition(v929,v930);
        Closure6 v932{v909};
        float v933; int v934;
        Tuple2 tmp31 = cooperative_groups::reduce(v931, Tuple2{v913, v914}, v932);
        v933 = tmp31.v0; v934 = tmp31.v1;
        bool v935;
        v935 = v934 == 2147483647l;
        bool v936;
        v936 = v935 != true;
        bool v937;
        v937 = v936 == false;
        if (v937){
            assert("Expected a valid action id in get_action." && v936);
        } else {
        }
        assert("Tensor range check" && 0 <= v654 && v654 < 1l);
        v641[v656] = v909;
        v642[v656] = v933;
        v654 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v939;
    v939 = threadIdx.x;
    assert("Tensor range check" && 0 <= v939 && v939 < 32l);
    int v940;
    v940 = v641[v939];
    float v941;
    v941 = v642[v939];
    v639[0l] = Tuple0{v940, v941};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v942; float v943;
    Tuple0 tmp32 = v639[0l];
    v942 = tmp32.v0; v943 = tmp32.v1;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v17, v18, v19, v20, v638, v637, v943, v942);
    int v944;
    v944 = 343l;
    int v945;
    v945 = 1l;
    Tuple0 v946[1l];
    __shared__ float * v947[32l];
    __shared__ int v948[32l];
    __shared__ float v949[32l];
    int v950;
    v950 = threadIdx.x;
    float * v951;
    v951 = v8+1372l;
    assert("Tensor range check" && 0 <= v950 && v950 < 32l);
    v947[v950] = v951;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v953;
    v953 = threadIdx.x;
    bool v954;
    v954 = 0l <= v953;
    bool v955;
    v955 = v954 == false;
    if (v955){
        assert("The index needs to be zero or positive." && v954);
    } else {
    }
    int v957;
    v957 = v953 % 1l;
    bool v958;
    v958 = v953 < 32l;
    bool v959;
    v959 = v958 == false;
    if (v959){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v958);
    } else {
    }
    assert("Tensor range check" && 0 <= v953 && v953 < 32l);
    assert("Tensor range check" && 0 <= v953 && v953 < 32l);
    int v961;
    v961 = 0l;
    while (while_method_1(v961)){
        assert("Tensor range check" && 0 <= v961 && v961 < 1l);
        int v963;
        v963 = v961 + v953;
        float * v964;
        v964 = v947[v963];
        assert("Tensor range check" && 0 <= v957 && v957 < 1l);
        int v965;
        v965 = 4l * v957;
        float v966[4l];
        int v967[4l];
        int v968;
        v968 = 0l;
        while (while_method_1(v968)){
            assert("Tensor range check" && 0 <= v968 && v968 < 1l);
            int v970;
            v970 = 4l * v968;
            assert("Tensor range check" && 0 <= v968 && v968 < 1l);
            int v971;
            v971 = v970 + v965;
            int4* v972;
            v972 = reinterpret_cast<int4*>(v964 + v971);
            int4* v973;
            v973 = reinterpret_cast<int4*>(v966 + v970);
            assert("Pointer alignment check" && (unsigned long long)(v972) % 4l == 0 && (unsigned long long)(v973) % 4l == 0);
            *v973 = *v972;
            v968 += 1l ;
        }
        int v974;
        v974 = 0l;
        while (while_method_1(v974)){
            int v976;
            v976 = 0l;
            while (while_method_2(v976)){
                bool v978;
                v978 = 0l <= v976;
                bool v980;
                if (v978){
                    bool v979;
                    v979 = v976 < 4l;
                    v980 = v979;
                } else {
                    v980 = false;
                }
                bool v981;
                v981 = v980 == false;
                if (v981){
                    assert("The indices should be inside the range of the dimension." && v980);
                } else {
                }
                bool v983;
                v983 = 0l <= v957;
                bool v985;
                if (v983){
                    bool v984;
                    v984 = v957 < 1l;
                    v985 = v984;
                } else {
                    v985 = false;
                }
                bool v986;
                v986 = v985 == false;
                if (v986){
                    assert("The indices should be inside the range of the dimension." && v985);
                } else {
                }
                int v988;
                v988 = v957 * 4l;
                int v989;
                v989 = v976 + v988;
                bool v990;
                v990 = 0l <= v974;
                bool v992;
                if (v990){
                    bool v991;
                    v991 = v974 < 1l;
                    v992 = v991;
                } else {
                    v992 = false;
                }
                bool v993;
                v993 = v992 == false;
                if (v993){
                    assert("The indices should be inside the range of the dimension." && v992);
                } else {
                }
                int v995;
                v995 = v974 * 4l;
                int v996;
                v996 = v989 + v995;
                assert("Tensor range check" && 0 <= v974 && v974 < 1l);
                assert("Tensor range check" && 0 <= v976 && v976 < 4l);
                int v997;
                v997 = 4l * v974;
                int v998;
                v998 = v997 + v976;
                v967[v998] = v996;
                v976 += 1l ;
            }
            v974 += 1l ;
        }
        bool v999;
        v999 = 0l <= v961;
        bool v1001;
        if (v999){
            bool v1000;
            v1000 = v961 < 1l;
            v1001 = v1000;
        } else {
            v1001 = false;
        }
        bool v1002;
        v1002 = v1001 == false;
        if (v1002){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1001);
        } else {
        }
        bool v1004;
        v1004 = v954 && v958;
        bool v1005;
        v1005 = v1004 == false;
        if (v1005){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1004);
        } else {
        }
        int v1007;
        v1007 = v953 + v961;
        bool v1008[4l];
        int v1009;
        v1009 = 0l;
        while (while_method_1(v1009)){
            int v1011;
            v1011 = 0l;
            while (while_method_2(v1011)){
                assert("Tensor range check" && 0 <= v1009 && v1009 < 1l);
                assert("Tensor range check" && 0 <= v1011 && v1011 < 4l);
                int v1013;
                v1013 = 4l * v1009;
                int v1014;
                v1014 = v1013 + v1011;
                float v1015;
                v1015 = v966[v1014];
                int v1016;
                v1016 = v967[v1014];
                bool v1017;
                v1017 = v1016 < 3l;
                assert("Tensor range check" && 0 <= v1009 && v1009 < 1l);
                assert("Tensor range check" && 0 <= v1011 && v1011 < 4l);
                v1008[v1014] = v1017;
                v1011 += 1l ;
            }
            v1009 += 1l ;
        }
        float v1018[4l];
        int v1019;
        v1019 = 0l;
        while (while_method_1(v1019)){
            int v1021;
            v1021 = 0l;
            while (while_method_2(v1021)){
                assert("Tensor range check" && 0 <= v1019 && v1019 < 1l);
                assert("Tensor range check" && 0 <= v1021 && v1021 < 4l);
                int v1023;
                v1023 = 4l * v1019;
                int v1024;
                v1024 = v1023 + v1021;
                float v1025;
                v1025 = v966[v1024];
                bool v1026;
                v1026 = v1008[v1024];
                float v1029;
                if (v1026){
                    bool v1027;
                    v1027 = 0.0f >= v1025;
                    if (v1027){
                        v1029 = 0.0f;
                    } else {
                        v1029 = v1025;
                    }
                } else {
                    v1029 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1019 && v1019 < 1l);
                assert("Tensor range check" && 0 <= v1021 && v1021 < 4l);
                v1018[v1024] = v1029;
                v1021 += 1l ;
            }
            v1019 += 1l ;
        }
        float v1030;
        v1030 = 0.0f;
        int v1031;
        v1031 = 0l;
        while (while_method_1(v1031)){
            int v1033;
            v1033 = 0l;
            while (while_method_2(v1033)){
                assert("Tensor range check" && 0 <= v1031 && v1031 < 1l);
                assert("Tensor range check" && 0 <= v1033 && v1033 < 4l);
                int v1035;
                v1035 = 4l * v1031;
                int v1036;
                v1036 = v1035 + v1033;
                float v1037;
                v1037 = v1018[v1036];
                float v1038;
                v1038 = v1030 + v1037;
                v1030 = v1038;
                v1033 += 1l ;
            }
            v1031 += 1l ;
        }
        auto v1039 = cooperative_groups::coalesced_threads();
        int v1040;
        v1040 = threadIdx.x;
        auto v1041 = cooperative_groups::labeled_partition(v1039,v1040);
        Closure0 v1042{};
        float v1043;
        v1043 = cooperative_groups::reduce(v1041, v1030, v1042);
        int v1044[4l];
        int v1045;
        v1045 = 0l;
        while (while_method_1(v1045)){
            int v1047;
            v1047 = 0l;
            while (while_method_2(v1047)){
                assert("Tensor range check" && 0 <= v1045 && v1045 < 1l);
                assert("Tensor range check" && 0 <= v1047 && v1047 < 4l);
                int v1049;
                v1049 = 4l * v1045;
                int v1050;
                v1050 = v1049 + v1047;
                bool v1051;
                v1051 = v1008[v1050];
                int v1052;
                if (v1051){
                    v1052 = 1l;
                } else {
                    v1052 = 0l;
                }
                assert("Tensor range check" && 0 <= v1045 && v1045 < 1l);
                assert("Tensor range check" && 0 <= v1047 && v1047 < 4l);
                v1044[v1050] = v1052;
                v1047 += 1l ;
            }
            v1045 += 1l ;
        }
        int v1053;
        v1053 = 0l;
        int v1054;
        v1054 = 0l;
        while (while_method_1(v1054)){
            int v1056;
            v1056 = 0l;
            while (while_method_2(v1056)){
                assert("Tensor range check" && 0 <= v1054 && v1054 < 1l);
                assert("Tensor range check" && 0 <= v1056 && v1056 < 4l);
                int v1058;
                v1058 = 4l * v1054;
                int v1059;
                v1059 = v1058 + v1056;
                int v1060;
                v1060 = v1044[v1059];
                int v1061;
                v1061 = v1053 + v1060;
                v1053 = v1061;
                v1056 += 1l ;
            }
            v1054 += 1l ;
        }
        auto v1062 = cooperative_groups::coalesced_threads();
        int v1063;
        v1063 = threadIdx.x;
        auto v1064 = cooperative_groups::labeled_partition(v1062,v1063);
        Closure1 v1065{};
        int v1066;
        v1066 = cooperative_groups::reduce(v1064, v1053, v1065);
        float v1067;
        v1067 = (float)v1066;
        float v1068;
        v1068 = 1.0f / v1067;
        float v1069[4l];
        int v1070;
        v1070 = 0l;
        while (while_method_1(v1070)){
            int v1072;
            v1072 = 0l;
            while (while_method_2(v1072)){
                assert("Tensor range check" && 0 <= v1070 && v1070 < 1l);
                assert("Tensor range check" && 0 <= v1072 && v1072 < 4l);
                int v1074;
                v1074 = 4l * v1070;
                int v1075;
                v1075 = v1074 + v1072;
                float v1076;
                v1076 = v1018[v1075];
                bool v1077;
                v1077 = v1008[v1075];
                bool v1078;
                v1078 = v1077 == false;
                float v1083;
                if (v1078){
                    v1083 = 0.0f;
                } else {
                    bool v1079;
                    v1079 = v1043 == 0.0f;
                    bool v1080;
                    v1080 = v1079 != true;
                    if (v1080){
                        float v1081;
                        v1081 = v1076 / v1043;
                        v1083 = v1081;
                    } else {
                        v1083 = v1068;
                    }
                }
                assert("Tensor range check" && 0 <= v1070 && v1070 < 1l);
                assert("Tensor range check" && 0 <= v1072 && v1072 < 4l);
                v1069[v1075] = v1083;
                v1072 += 1l ;
            }
            v1070 += 1l ;
        }
        float v1084[4l];
        float v1085;
        v1085 = 0.0f;
        int v1086;
        v1086 = 0l;
        while (while_method_1(v1086)){
            assert("Tensor range check" && 0 <= v1086 && v1086 < 1l);
            int v1088;
            v1088 = 4l * v1086;
            assert("Tensor range check" && 0 <= v1086 && v1086 < 1l);
            int v1089; float v1090;
            Tuple0 tmp33 = Tuple0{0l, 0.0f};
            v1089 = tmp33.v0; v1090 = tmp33.v1;
            while (while_method_2(v1089)){
                assert("Tensor range check" && 0 <= v1089 && v1089 < 4l);
                int v1092;
                v1092 = v1089 + v1088;
                float v1093;
                v1093 = v1069[v1092];
                float v1094;
                v1094 = v1090 + v1093;
                v1090 = v1094;
                v1089 += 1l ;
            }
            auto v1095 = cooperative_groups::coalesced_threads();
            int v1096;
            v1096 = threadIdx.x;
            auto v1097 = cooperative_groups::labeled_partition(v1095,v1096);
            Closure2 v1098{};
            float v1099;
            v1099 = cooperative_groups::inclusive_scan(v1097, v1090, v1098);
            float v1100;
            v1100 = v1097.shfl_up(v1099,1);
            bool v1101;
            v1101 = v1097.thread_rank() == 0;
            float v1102;
            if (v1101){
                v1102 = 0.0f;
            } else {
                v1102 = v1100;
            }
            float v1103;
            v1103 = v1097.shfl(v1099,v1097.num_threads()-1);
            float v1104;
            v1104 = v1085 + v1102;
            int v1105; float v1106;
            Tuple0 tmp34 = Tuple0{0l, v1104};
            v1105 = tmp34.v0; v1106 = tmp34.v1;
            while (while_method_2(v1105)){
                assert("Tensor range check" && 0 <= v1105 && v1105 < 4l);
                int v1108;
                v1108 = v1105 + v1088;
                float v1109;
                v1109 = v1069[v1108];
                float v1110;
                v1110 = v1106 + v1109;
                assert("Tensor range check" && 0 <= v1105 && v1105 < 4l);
                v1084[v1108] = v1110;
                v1106 = v1110;
                v1105 += 1l ;
            }
            float v1111;
            v1111 = v1085 + v1103;
            v1085 = v1111;
            v1086 += 1l ;
        }
        float v1112[4l];
        bool v1113[4l];
        int v1114;
        v1114 = 0l;
        while (while_method_1(v1114)){
            int v1116;
            v1116 = 0l;
            while (while_method_2(v1116)){
                assert("Tensor range check" && 0 <= v1114 && v1114 < 1l);
                assert("Tensor range check" && 0 <= v1116 && v1116 < 4l);
                int v1118;
                v1118 = 4l * v1114;
                int v1119;
                v1119 = v1118 + v1116;
                float v1120;
                v1120 = v1084[v1119];
                float v1121;
                v1121 = v1069[v1119];
                bool v1122;
                v1122 = v1121 > 0.0f;
                assert("Tensor range check" && 0 <= v1114 && v1114 < 1l);
                assert("Tensor range check" && 0 <= v1116 && v1116 < 4l);
                v1112[v1119] = v1120;
                v1113[v1119] = v1122;
                v1116 += 1l ;
            }
            v1114 += 1l ;
        }
        float v1123; bool v1124;
        Tuple1 tmp35 = Tuple1{-1.0f / 0.0f, false};
        v1123 = tmp35.v0; v1124 = tmp35.v1;
        int v1125;
        v1125 = 0l;
        while (while_method_1(v1125)){
            int v1127;
            v1127 = 0l;
            while (while_method_2(v1127)){
                assert("Tensor range check" && 0 <= v1125 && v1125 < 1l);
                assert("Tensor range check" && 0 <= v1127 && v1127 < 4l);
                int v1129;
                v1129 = 4l * v1125;
                int v1130;
                v1130 = v1129 + v1127;
                float v1131;
                v1131 = v1112[v1130];
                bool v1132;
                v1132 = v1113[v1130];
                float v1139; bool v1140;
                if (v1124){
                    if (v1132){
                        bool v1133;
                        v1133 = v1123 >= v1131;
                        float v1134;
                        if (v1133){
                            v1134 = v1123;
                        } else {
                            v1134 = v1131;
                        }
                        v1139 = v1134; v1140 = true;
                    } else {
                        v1139 = v1123; v1140 = v1124;
                    }
                } else {
                    if (v1132){
                        v1139 = v1131; v1140 = v1132;
                    } else {
                        v1139 = v1123; v1140 = v1124;
                    }
                }
                v1123 = v1139;
                v1124 = v1140;
                v1127 += 1l ;
            }
            v1125 += 1l ;
        }
        auto v1141 = cooperative_groups::coalesced_threads();
        int v1142;
        v1142 = threadIdx.x;
        auto v1143 = cooperative_groups::labeled_partition(v1141,v1142);
        Closure3 v1144{};
        float v1145; bool v1146;
        Tuple1 tmp36 = cooperative_groups::reduce(v1143, Tuple1{v1123, v1124}, v1144);
        v1145 = tmp36.v0; v1146 = tmp36.v1;
        bool v1147;
        v1147 = v1146 == false;
        if (v1147){
            assert("The local reduce must be true." && v1146);
        } else {
        }
        float v1149[4l];
        int v1150[4l];
        int v1151;
        v1151 = 0l;
        while (while_method_1(v1151)){
            int v1153;
            v1153 = 0l;
            while (while_method_2(v1153)){
                assert("Tensor range check" && 0 <= v1151 && v1151 < 1l);
                assert("Tensor range check" && 0 <= v1153 && v1153 < 4l);
                int v1155;
                v1155 = 4l * v1151;
                int v1156;
                v1156 = v1155 + v1153;
                int v1157;
                v1157 = v967[v1156];
                float v1158;
                v1158 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v1151 && v1151 < 1l);
                assert("Tensor range check" && 0 <= v1153 && v1153 < 4l);
                v1149[v1156] = v1158;
                v1150[v1156] = v1157;
                v1153 += 1l ;
            }
            v1151 += 1l ;
        }
        float v1159; int v1160;
        Tuple2 tmp37 = Tuple2{0.0f, 2147483647l};
        v1159 = tmp37.v0; v1160 = tmp37.v1;
        int v1161;
        v1161 = 0l;
        while (while_method_1(v1161)){
            int v1163;
            v1163 = 0l;
            while (while_method_2(v1163)){
                assert("Tensor range check" && 0 <= v1161 && v1161 < 1l);
                assert("Tensor range check" && 0 <= v1163 && v1163 < 4l);
                int v1165;
                v1165 = 4l * v1161;
                int v1166;
                v1166 = v1165 + v1163;
                float v1167;
                v1167 = v1149[v1166];
                int v1168;
                v1168 = v1150[v1166];
                bool v1169;
                v1169 = v1160 < v1168;
                float v1170; int v1171;
                if (v1169){
                    v1170 = v1159; v1171 = v1160;
                } else {
                    v1170 = v1167; v1171 = v1168;
                }
                v1159 = v1170;
                v1160 = v1171;
                v1163 += 1l ;
            }
            v1161 += 1l ;
        }
        auto v1172 = cooperative_groups::coalesced_threads();
        int v1173;
        v1173 = threadIdx.x;
        auto v1174 = cooperative_groups::labeled_partition(v1172,v1173);
        Closure4 v1175{};
        float v1176; int v1177;
        Tuple2 tmp38 = cooperative_groups::reduce(v1174, Tuple2{v1159, v1160}, v1175);
        v1176 = tmp38.v0; v1177 = tmp38.v1;
        float v1178;
        v1178 = v1145 * v1176;
        int v1179[4l];
        bool v1180[4l];
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
                float v1187;
                v1187 = v1112[v1186];
                bool v1188;
                v1188 = v1113[v1186];
                int v1189;
                v1189 = v967[v1186];
                int v1192; bool v1193;
                if (v1188){
                    float v1190;
                    v1190 = v1187 - v1178;
                    bool v1191;
                    v1191 = v1190 >= 0.0f;
                    v1192 = v1189; v1193 = v1191;
                } else {
                    v1192 = 2147483647l; v1193 = false;
                }
                assert("Tensor range check" && 0 <= v1181 && v1181 < 1l);
                assert("Tensor range check" && 0 <= v1183 && v1183 < 4l);
                v1179[v1186] = v1192;
                v1180[v1186] = v1193;
                v1183 += 1l ;
            }
            v1181 += 1l ;
        }
        int v1194; bool v1195;
        Tuple3 tmp39 = Tuple3{2147483647l, false};
        v1194 = tmp39.v0; v1195 = tmp39.v1;
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
                int v1202;
                v1202 = v1179[v1201];
                bool v1203;
                v1203 = v1180[v1201];
                int v1210; bool v1211;
                if (v1195){
                    if (v1203){
                        bool v1204;
                        v1204 = v1194 < v1202;
                        int v1205;
                        if (v1204){
                            v1205 = v1194;
                        } else {
                            v1205 = v1202;
                        }
                        v1210 = v1205; v1211 = true;
                    } else {
                        v1210 = v1194; v1211 = v1195;
                    }
                } else {
                    if (v1203){
                        v1210 = v1202; v1211 = v1203;
                    } else {
                        v1210 = v1194; v1211 = v1195;
                    }
                }
                v1194 = v1210;
                v1195 = v1211;
                v1198 += 1l ;
            }
            v1196 += 1l ;
        }
        auto v1212 = cooperative_groups::coalesced_threads();
        int v1213;
        v1213 = threadIdx.x;
        auto v1214 = cooperative_groups::labeled_partition(v1212,v1213);
        Closure5 v1215{};
        int v1216; bool v1217;
        Tuple3 tmp40 = cooperative_groups::reduce(v1214, Tuple3{v1194, v1195}, v1215);
        v1216 = tmp40.v0; v1217 = tmp40.v1;
        bool v1218;
        v1218 = v1217 == false;
        if (v1218){
            assert("The local reduce must be true." && v1217);
        } else {
        }
        float v1220; int v1221;
        Tuple2 tmp41 = Tuple2{0.0f, 2147483647l};
        v1220 = tmp41.v0; v1221 = tmp41.v1;
        int v1222;
        v1222 = 0l;
        while (while_method_1(v1222)){
            int v1224;
            v1224 = 0l;
            while (while_method_2(v1224)){
                assert("Tensor range check" && 0 <= v1222 && v1222 < 1l);
                assert("Tensor range check" && 0 <= v1224 && v1224 < 4l);
                int v1226;
                v1226 = 4l * v1222;
                int v1227;
                v1227 = v1226 + v1224;
                float v1228;
                v1228 = v1069[v1227];
                int v1229;
                v1229 = v967[v1227];
                bool v1230;
                v1230 = v1221 == v1216;
                float v1234; int v1235;
                if (v1230){
                    v1234 = v1220; v1235 = v1221;
                } else {
                    bool v1231;
                    v1231 = v1229 == v1216;
                    if (v1231){
                        v1234 = v1228; v1235 = v1229;
                    } else {
                        v1234 = v1220; v1235 = v1221;
                    }
                }
                v1220 = v1234;
                v1221 = v1235;
                v1224 += 1l ;
            }
            v1222 += 1l ;
        }
        auto v1236 = cooperative_groups::coalesced_threads();
        int v1237;
        v1237 = threadIdx.x;
        auto v1238 = cooperative_groups::labeled_partition(v1236,v1237);
        Closure6 v1239{v1216};
        float v1240; int v1241;
        Tuple2 tmp42 = cooperative_groups::reduce(v1238, Tuple2{v1220, v1221}, v1239);
        v1240 = tmp42.v0; v1241 = tmp42.v1;
        bool v1242;
        v1242 = v1241 == 2147483647l;
        bool v1243;
        v1243 = v1242 != true;
        bool v1244;
        v1244 = v1243 == false;
        if (v1244){
            assert("Expected a valid action id in get_action." && v1243);
        } else {
        }
        assert("Tensor range check" && 0 <= v961 && v961 < 1l);
        v948[v963] = v1216;
        v949[v963] = v1240;
        v961 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1246;
    v1246 = threadIdx.x;
    assert("Tensor range check" && 0 <= v1246 && v1246 < 32l);
    int v1247;
    v1247 = v948[v1246];
    float v1248;
    v1248 = v949[v1246];
    v946[0l] = Tuple0{v1247, v1248};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1249; float v1250;
    Tuple0 tmp43 = v946[0l];
    v1249 = tmp43.v0; v1250 = tmp43.v1;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v17, v18, v19, v20, v945, v944, v1250, v1249);
    int v1251;
    v1251 = 457l;
    int v1252;
    v1252 = 0l;
    Tuple0 v1253[1l];
    __shared__ float * v1254[32l];
    __shared__ int v1255[32l];
    __shared__ float v1256[32l];
    int v1257;
    v1257 = threadIdx.x;
    float * v1258;
    v1258 = v8+1828l;
    assert("Tensor range check" && 0 <= v1257 && v1257 < 32l);
    v1254[v1257] = v1258;
    asm("barrier.cta.sync %0;" :: "r"(0l));
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
    v1264 = v1260 % 1l;
    bool v1265;
    v1265 = v1260 < 32l;
    bool v1266;
    v1266 = v1265 == false;
    if (v1266){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1265);
    } else {
    }
    assert("Tensor range check" && 0 <= v1260 && v1260 < 32l);
    assert("Tensor range check" && 0 <= v1260 && v1260 < 32l);
    int v1268;
    v1268 = 0l;
    while (while_method_1(v1268)){
        assert("Tensor range check" && 0 <= v1268 && v1268 < 1l);
        int v1270;
        v1270 = v1268 + v1260;
        float * v1271;
        v1271 = v1254[v1270];
        assert("Tensor range check" && 0 <= v1264 && v1264 < 1l);
        int v1272;
        v1272 = 4l * v1264;
        float v1273[4l];
        int v1274[4l];
        int v1275;
        v1275 = 0l;
        while (while_method_1(v1275)){
            assert("Tensor range check" && 0 <= v1275 && v1275 < 1l);
            int v1277;
            v1277 = 4l * v1275;
            assert("Tensor range check" && 0 <= v1275 && v1275 < 1l);
            int v1278;
            v1278 = v1277 + v1272;
            int4* v1279;
            v1279 = reinterpret_cast<int4*>(v1271 + v1278);
            int4* v1280;
            v1280 = reinterpret_cast<int4*>(v1273 + v1277);
            assert("Pointer alignment check" && (unsigned long long)(v1279) % 4l == 0 && (unsigned long long)(v1280) % 4l == 0);
            *v1280 = *v1279;
            v1275 += 1l ;
        }
        int v1281;
        v1281 = 0l;
        while (while_method_1(v1281)){
            int v1283;
            v1283 = 0l;
            while (while_method_2(v1283)){
                bool v1285;
                v1285 = 0l <= v1283;
                bool v1287;
                if (v1285){
                    bool v1286;
                    v1286 = v1283 < 4l;
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
                bool v1290;
                v1290 = 0l <= v1264;
                bool v1292;
                if (v1290){
                    bool v1291;
                    v1291 = v1264 < 1l;
                    v1292 = v1291;
                } else {
                    v1292 = false;
                }
                bool v1293;
                v1293 = v1292 == false;
                if (v1293){
                    assert("The indices should be inside the range of the dimension." && v1292);
                } else {
                }
                int v1295;
                v1295 = v1264 * 4l;
                int v1296;
                v1296 = v1283 + v1295;
                bool v1297;
                v1297 = 0l <= v1281;
                bool v1299;
                if (v1297){
                    bool v1298;
                    v1298 = v1281 < 1l;
                    v1299 = v1298;
                } else {
                    v1299 = false;
                }
                bool v1300;
                v1300 = v1299 == false;
                if (v1300){
                    assert("The indices should be inside the range of the dimension." && v1299);
                } else {
                }
                int v1302;
                v1302 = v1281 * 4l;
                int v1303;
                v1303 = v1296 + v1302;
                assert("Tensor range check" && 0 <= v1281 && v1281 < 1l);
                assert("Tensor range check" && 0 <= v1283 && v1283 < 4l);
                int v1304;
                v1304 = 4l * v1281;
                int v1305;
                v1305 = v1304 + v1283;
                v1274[v1305] = v1303;
                v1283 += 1l ;
            }
            v1281 += 1l ;
        }
        bool v1306;
        v1306 = 0l <= v1268;
        bool v1308;
        if (v1306){
            bool v1307;
            v1307 = v1268 < 1l;
            v1308 = v1307;
        } else {
            v1308 = false;
        }
        bool v1309;
        v1309 = v1308 == false;
        if (v1309){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1308);
        } else {
        }
        bool v1311;
        v1311 = v1261 && v1265;
        bool v1312;
        v1312 = v1311 == false;
        if (v1312){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1311);
        } else {
        }
        int v1314;
        v1314 = v1260 + v1268;
        bool v1315[4l];
        int v1316;
        v1316 = 0l;
        while (while_method_1(v1316)){
            int v1318;
            v1318 = 0l;
            while (while_method_2(v1318)){
                assert("Tensor range check" && 0 <= v1316 && v1316 < 1l);
                assert("Tensor range check" && 0 <= v1318 && v1318 < 4l);
                int v1320;
                v1320 = 4l * v1316;
                int v1321;
                v1321 = v1320 + v1318;
                float v1322;
                v1322 = v1273[v1321];
                int v1323;
                v1323 = v1274[v1321];
                bool v1324;
                v1324 = v1323 < 3l;
                assert("Tensor range check" && 0 <= v1316 && v1316 < 1l);
                assert("Tensor range check" && 0 <= v1318 && v1318 < 4l);
                v1315[v1321] = v1324;
                v1318 += 1l ;
            }
            v1316 += 1l ;
        }
        float v1325[4l];
        int v1326;
        v1326 = 0l;
        while (while_method_1(v1326)){
            int v1328;
            v1328 = 0l;
            while (while_method_2(v1328)){
                assert("Tensor range check" && 0 <= v1326 && v1326 < 1l);
                assert("Tensor range check" && 0 <= v1328 && v1328 < 4l);
                int v1330;
                v1330 = 4l * v1326;
                int v1331;
                v1331 = v1330 + v1328;
                float v1332;
                v1332 = v1273[v1331];
                bool v1333;
                v1333 = v1315[v1331];
                float v1336;
                if (v1333){
                    bool v1334;
                    v1334 = 0.0f >= v1332;
                    if (v1334){
                        v1336 = 0.0f;
                    } else {
                        v1336 = v1332;
                    }
                } else {
                    v1336 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1326 && v1326 < 1l);
                assert("Tensor range check" && 0 <= v1328 && v1328 < 4l);
                v1325[v1331] = v1336;
                v1328 += 1l ;
            }
            v1326 += 1l ;
        }
        float v1337;
        v1337 = 0.0f;
        int v1338;
        v1338 = 0l;
        while (while_method_1(v1338)){
            int v1340;
            v1340 = 0l;
            while (while_method_2(v1340)){
                assert("Tensor range check" && 0 <= v1338 && v1338 < 1l);
                assert("Tensor range check" && 0 <= v1340 && v1340 < 4l);
                int v1342;
                v1342 = 4l * v1338;
                int v1343;
                v1343 = v1342 + v1340;
                float v1344;
                v1344 = v1325[v1343];
                float v1345;
                v1345 = v1337 + v1344;
                v1337 = v1345;
                v1340 += 1l ;
            }
            v1338 += 1l ;
        }
        auto v1346 = cooperative_groups::coalesced_threads();
        int v1347;
        v1347 = threadIdx.x;
        auto v1348 = cooperative_groups::labeled_partition(v1346,v1347);
        Closure0 v1349{};
        float v1350;
        v1350 = cooperative_groups::reduce(v1348, v1337, v1349);
        int v1351[4l];
        int v1352;
        v1352 = 0l;
        while (while_method_1(v1352)){
            int v1354;
            v1354 = 0l;
            while (while_method_2(v1354)){
                assert("Tensor range check" && 0 <= v1352 && v1352 < 1l);
                assert("Tensor range check" && 0 <= v1354 && v1354 < 4l);
                int v1356;
                v1356 = 4l * v1352;
                int v1357;
                v1357 = v1356 + v1354;
                bool v1358;
                v1358 = v1315[v1357];
                int v1359;
                if (v1358){
                    v1359 = 1l;
                } else {
                    v1359 = 0l;
                }
                assert("Tensor range check" && 0 <= v1352 && v1352 < 1l);
                assert("Tensor range check" && 0 <= v1354 && v1354 < 4l);
                v1351[v1357] = v1359;
                v1354 += 1l ;
            }
            v1352 += 1l ;
        }
        int v1360;
        v1360 = 0l;
        int v1361;
        v1361 = 0l;
        while (while_method_1(v1361)){
            int v1363;
            v1363 = 0l;
            while (while_method_2(v1363)){
                assert("Tensor range check" && 0 <= v1361 && v1361 < 1l);
                assert("Tensor range check" && 0 <= v1363 && v1363 < 4l);
                int v1365;
                v1365 = 4l * v1361;
                int v1366;
                v1366 = v1365 + v1363;
                int v1367;
                v1367 = v1351[v1366];
                int v1368;
                v1368 = v1360 + v1367;
                v1360 = v1368;
                v1363 += 1l ;
            }
            v1361 += 1l ;
        }
        auto v1369 = cooperative_groups::coalesced_threads();
        int v1370;
        v1370 = threadIdx.x;
        auto v1371 = cooperative_groups::labeled_partition(v1369,v1370);
        Closure1 v1372{};
        int v1373;
        v1373 = cooperative_groups::reduce(v1371, v1360, v1372);
        float v1374;
        v1374 = (float)v1373;
        float v1375;
        v1375 = 1.0f / v1374;
        float v1376[4l];
        int v1377;
        v1377 = 0l;
        while (while_method_1(v1377)){
            int v1379;
            v1379 = 0l;
            while (while_method_2(v1379)){
                assert("Tensor range check" && 0 <= v1377 && v1377 < 1l);
                assert("Tensor range check" && 0 <= v1379 && v1379 < 4l);
                int v1381;
                v1381 = 4l * v1377;
                int v1382;
                v1382 = v1381 + v1379;
                float v1383;
                v1383 = v1325[v1382];
                bool v1384;
                v1384 = v1315[v1382];
                bool v1385;
                v1385 = v1384 == false;
                float v1390;
                if (v1385){
                    v1390 = 0.0f;
                } else {
                    bool v1386;
                    v1386 = v1350 == 0.0f;
                    bool v1387;
                    v1387 = v1386 != true;
                    if (v1387){
                        float v1388;
                        v1388 = v1383 / v1350;
                        v1390 = v1388;
                    } else {
                        v1390 = v1375;
                    }
                }
                assert("Tensor range check" && 0 <= v1377 && v1377 < 1l);
                assert("Tensor range check" && 0 <= v1379 && v1379 < 4l);
                v1376[v1382] = v1390;
                v1379 += 1l ;
            }
            v1377 += 1l ;
        }
        float v1391[4l];
        float v1392;
        v1392 = 0.0f;
        int v1393;
        v1393 = 0l;
        while (while_method_1(v1393)){
            assert("Tensor range check" && 0 <= v1393 && v1393 < 1l);
            int v1395;
            v1395 = 4l * v1393;
            assert("Tensor range check" && 0 <= v1393 && v1393 < 1l);
            int v1396; float v1397;
            Tuple0 tmp44 = Tuple0{0l, 0.0f};
            v1396 = tmp44.v0; v1397 = tmp44.v1;
            while (while_method_2(v1396)){
                assert("Tensor range check" && 0 <= v1396 && v1396 < 4l);
                int v1399;
                v1399 = v1396 + v1395;
                float v1400;
                v1400 = v1376[v1399];
                float v1401;
                v1401 = v1397 + v1400;
                v1397 = v1401;
                v1396 += 1l ;
            }
            auto v1402 = cooperative_groups::coalesced_threads();
            int v1403;
            v1403 = threadIdx.x;
            auto v1404 = cooperative_groups::labeled_partition(v1402,v1403);
            Closure2 v1405{};
            float v1406;
            v1406 = cooperative_groups::inclusive_scan(v1404, v1397, v1405);
            float v1407;
            v1407 = v1404.shfl_up(v1406,1);
            bool v1408;
            v1408 = v1404.thread_rank() == 0;
            float v1409;
            if (v1408){
                v1409 = 0.0f;
            } else {
                v1409 = v1407;
            }
            float v1410;
            v1410 = v1404.shfl(v1406,v1404.num_threads()-1);
            float v1411;
            v1411 = v1392 + v1409;
            int v1412; float v1413;
            Tuple0 tmp45 = Tuple0{0l, v1411};
            v1412 = tmp45.v0; v1413 = tmp45.v1;
            while (while_method_2(v1412)){
                assert("Tensor range check" && 0 <= v1412 && v1412 < 4l);
                int v1415;
                v1415 = v1412 + v1395;
                float v1416;
                v1416 = v1376[v1415];
                float v1417;
                v1417 = v1413 + v1416;
                assert("Tensor range check" && 0 <= v1412 && v1412 < 4l);
                v1391[v1415] = v1417;
                v1413 = v1417;
                v1412 += 1l ;
            }
            float v1418;
            v1418 = v1392 + v1410;
            v1392 = v1418;
            v1393 += 1l ;
        }
        float v1419[4l];
        bool v1420[4l];
        int v1421;
        v1421 = 0l;
        while (while_method_1(v1421)){
            int v1423;
            v1423 = 0l;
            while (while_method_2(v1423)){
                assert("Tensor range check" && 0 <= v1421 && v1421 < 1l);
                assert("Tensor range check" && 0 <= v1423 && v1423 < 4l);
                int v1425;
                v1425 = 4l * v1421;
                int v1426;
                v1426 = v1425 + v1423;
                float v1427;
                v1427 = v1391[v1426];
                float v1428;
                v1428 = v1376[v1426];
                bool v1429;
                v1429 = v1428 > 0.0f;
                assert("Tensor range check" && 0 <= v1421 && v1421 < 1l);
                assert("Tensor range check" && 0 <= v1423 && v1423 < 4l);
                v1419[v1426] = v1427;
                v1420[v1426] = v1429;
                v1423 += 1l ;
            }
            v1421 += 1l ;
        }
        float v1430; bool v1431;
        Tuple1 tmp46 = Tuple1{-1.0f / 0.0f, false};
        v1430 = tmp46.v0; v1431 = tmp46.v1;
        int v1432;
        v1432 = 0l;
        while (while_method_1(v1432)){
            int v1434;
            v1434 = 0l;
            while (while_method_2(v1434)){
                assert("Tensor range check" && 0 <= v1432 && v1432 < 1l);
                assert("Tensor range check" && 0 <= v1434 && v1434 < 4l);
                int v1436;
                v1436 = 4l * v1432;
                int v1437;
                v1437 = v1436 + v1434;
                float v1438;
                v1438 = v1419[v1437];
                bool v1439;
                v1439 = v1420[v1437];
                float v1446; bool v1447;
                if (v1431){
                    if (v1439){
                        bool v1440;
                        v1440 = v1430 >= v1438;
                        float v1441;
                        if (v1440){
                            v1441 = v1430;
                        } else {
                            v1441 = v1438;
                        }
                        v1446 = v1441; v1447 = true;
                    } else {
                        v1446 = v1430; v1447 = v1431;
                    }
                } else {
                    if (v1439){
                        v1446 = v1438; v1447 = v1439;
                    } else {
                        v1446 = v1430; v1447 = v1431;
                    }
                }
                v1430 = v1446;
                v1431 = v1447;
                v1434 += 1l ;
            }
            v1432 += 1l ;
        }
        auto v1448 = cooperative_groups::coalesced_threads();
        int v1449;
        v1449 = threadIdx.x;
        auto v1450 = cooperative_groups::labeled_partition(v1448,v1449);
        Closure3 v1451{};
        float v1452; bool v1453;
        Tuple1 tmp47 = cooperative_groups::reduce(v1450, Tuple1{v1430, v1431}, v1451);
        v1452 = tmp47.v0; v1453 = tmp47.v1;
        bool v1454;
        v1454 = v1453 == false;
        if (v1454){
            assert("The local reduce must be true." && v1453);
        } else {
        }
        float v1456[4l];
        int v1457[4l];
        int v1458;
        v1458 = 0l;
        while (while_method_1(v1458)){
            int v1460;
            v1460 = 0l;
            while (while_method_2(v1460)){
                assert("Tensor range check" && 0 <= v1458 && v1458 < 1l);
                assert("Tensor range check" && 0 <= v1460 && v1460 < 4l);
                int v1462;
                v1462 = 4l * v1458;
                int v1463;
                v1463 = v1462 + v1460;
                int v1464;
                v1464 = v1274[v1463];
                float v1465;
                v1465 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v1458 && v1458 < 1l);
                assert("Tensor range check" && 0 <= v1460 && v1460 < 4l);
                v1456[v1463] = v1465;
                v1457[v1463] = v1464;
                v1460 += 1l ;
            }
            v1458 += 1l ;
        }
        float v1466; int v1467;
        Tuple2 tmp48 = Tuple2{0.0f, 2147483647l};
        v1466 = tmp48.v0; v1467 = tmp48.v1;
        int v1468;
        v1468 = 0l;
        while (while_method_1(v1468)){
            int v1470;
            v1470 = 0l;
            while (while_method_2(v1470)){
                assert("Tensor range check" && 0 <= v1468 && v1468 < 1l);
                assert("Tensor range check" && 0 <= v1470 && v1470 < 4l);
                int v1472;
                v1472 = 4l * v1468;
                int v1473;
                v1473 = v1472 + v1470;
                float v1474;
                v1474 = v1456[v1473];
                int v1475;
                v1475 = v1457[v1473];
                bool v1476;
                v1476 = v1467 < v1475;
                float v1477; int v1478;
                if (v1476){
                    v1477 = v1466; v1478 = v1467;
                } else {
                    v1477 = v1474; v1478 = v1475;
                }
                v1466 = v1477;
                v1467 = v1478;
                v1470 += 1l ;
            }
            v1468 += 1l ;
        }
        auto v1479 = cooperative_groups::coalesced_threads();
        int v1480;
        v1480 = threadIdx.x;
        auto v1481 = cooperative_groups::labeled_partition(v1479,v1480);
        Closure4 v1482{};
        float v1483; int v1484;
        Tuple2 tmp49 = cooperative_groups::reduce(v1481, Tuple2{v1466, v1467}, v1482);
        v1483 = tmp49.v0; v1484 = tmp49.v1;
        float v1485;
        v1485 = v1452 * v1483;
        int v1486[4l];
        bool v1487[4l];
        int v1488;
        v1488 = 0l;
        while (while_method_1(v1488)){
            int v1490;
            v1490 = 0l;
            while (while_method_2(v1490)){
                assert("Tensor range check" && 0 <= v1488 && v1488 < 1l);
                assert("Tensor range check" && 0 <= v1490 && v1490 < 4l);
                int v1492;
                v1492 = 4l * v1488;
                int v1493;
                v1493 = v1492 + v1490;
                float v1494;
                v1494 = v1419[v1493];
                bool v1495;
                v1495 = v1420[v1493];
                int v1496;
                v1496 = v1274[v1493];
                int v1499; bool v1500;
                if (v1495){
                    float v1497;
                    v1497 = v1494 - v1485;
                    bool v1498;
                    v1498 = v1497 >= 0.0f;
                    v1499 = v1496; v1500 = v1498;
                } else {
                    v1499 = 2147483647l; v1500 = false;
                }
                assert("Tensor range check" && 0 <= v1488 && v1488 < 1l);
                assert("Tensor range check" && 0 <= v1490 && v1490 < 4l);
                v1486[v1493] = v1499;
                v1487[v1493] = v1500;
                v1490 += 1l ;
            }
            v1488 += 1l ;
        }
        int v1501; bool v1502;
        Tuple3 tmp50 = Tuple3{2147483647l, false};
        v1501 = tmp50.v0; v1502 = tmp50.v1;
        int v1503;
        v1503 = 0l;
        while (while_method_1(v1503)){
            int v1505;
            v1505 = 0l;
            while (while_method_2(v1505)){
                assert("Tensor range check" && 0 <= v1503 && v1503 < 1l);
                assert("Tensor range check" && 0 <= v1505 && v1505 < 4l);
                int v1507;
                v1507 = 4l * v1503;
                int v1508;
                v1508 = v1507 + v1505;
                int v1509;
                v1509 = v1486[v1508];
                bool v1510;
                v1510 = v1487[v1508];
                int v1517; bool v1518;
                if (v1502){
                    if (v1510){
                        bool v1511;
                        v1511 = v1501 < v1509;
                        int v1512;
                        if (v1511){
                            v1512 = v1501;
                        } else {
                            v1512 = v1509;
                        }
                        v1517 = v1512; v1518 = true;
                    } else {
                        v1517 = v1501; v1518 = v1502;
                    }
                } else {
                    if (v1510){
                        v1517 = v1509; v1518 = v1510;
                    } else {
                        v1517 = v1501; v1518 = v1502;
                    }
                }
                v1501 = v1517;
                v1502 = v1518;
                v1505 += 1l ;
            }
            v1503 += 1l ;
        }
        auto v1519 = cooperative_groups::coalesced_threads();
        int v1520;
        v1520 = threadIdx.x;
        auto v1521 = cooperative_groups::labeled_partition(v1519,v1520);
        Closure5 v1522{};
        int v1523; bool v1524;
        Tuple3 tmp51 = cooperative_groups::reduce(v1521, Tuple3{v1501, v1502}, v1522);
        v1523 = tmp51.v0; v1524 = tmp51.v1;
        bool v1525;
        v1525 = v1524 == false;
        if (v1525){
            assert("The local reduce must be true." && v1524);
        } else {
        }
        float v1527; int v1528;
        Tuple2 tmp52 = Tuple2{0.0f, 2147483647l};
        v1527 = tmp52.v0; v1528 = tmp52.v1;
        int v1529;
        v1529 = 0l;
        while (while_method_1(v1529)){
            int v1531;
            v1531 = 0l;
            while (while_method_2(v1531)){
                assert("Tensor range check" && 0 <= v1529 && v1529 < 1l);
                assert("Tensor range check" && 0 <= v1531 && v1531 < 4l);
                int v1533;
                v1533 = 4l * v1529;
                int v1534;
                v1534 = v1533 + v1531;
                float v1535;
                v1535 = v1376[v1534];
                int v1536;
                v1536 = v1274[v1534];
                bool v1537;
                v1537 = v1528 == v1523;
                float v1541; int v1542;
                if (v1537){
                    v1541 = v1527; v1542 = v1528;
                } else {
                    bool v1538;
                    v1538 = v1536 == v1523;
                    if (v1538){
                        v1541 = v1535; v1542 = v1536;
                    } else {
                        v1541 = v1527; v1542 = v1528;
                    }
                }
                v1527 = v1541;
                v1528 = v1542;
                v1531 += 1l ;
            }
            v1529 += 1l ;
        }
        auto v1543 = cooperative_groups::coalesced_threads();
        int v1544;
        v1544 = threadIdx.x;
        auto v1545 = cooperative_groups::labeled_partition(v1543,v1544);
        Closure6 v1546{v1523};
        float v1547; int v1548;
        Tuple2 tmp53 = cooperative_groups::reduce(v1545, Tuple2{v1527, v1528}, v1546);
        v1547 = tmp53.v0; v1548 = tmp53.v1;
        bool v1549;
        v1549 = v1548 == 2147483647l;
        bool v1550;
        v1550 = v1549 != true;
        bool v1551;
        v1551 = v1550 == false;
        if (v1551){
            assert("Expected a valid action id in get_action." && v1550);
        } else {
        }
        assert("Tensor range check" && 0 <= v1268 && v1268 < 1l);
        v1255[v1270] = v1523;
        v1256[v1270] = v1547;
        v1268 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1553;
    v1553 = threadIdx.x;
    assert("Tensor range check" && 0 <= v1553 && v1553 < 32l);
    int v1554;
    v1554 = v1255[v1553];
    float v1555;
    v1555 = v1256[v1553];
    v1253[0l] = Tuple0{v1554, v1555};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1556; float v1557;
    Tuple0 tmp54 = v1253[0l];
    v1556 = tmp54.v0; v1557 = tmp54.v1;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v17, v18, v19, v20, v1252, v1251, v1557, v1556);
    int v1558;
    v1558 = 3447l;
    int v1559;
    v1559 = 1l;
    Tuple0 v1560[1l];
    __shared__ float * v1561[32l];
    __shared__ int v1562[32l];
    __shared__ float v1563[32l];
    int v1564;
    v1564 = threadIdx.x;
    float * v1565;
    v1565 = v8+13788l;
    assert("Tensor range check" && 0 <= v1564 && v1564 < 32l);
    v1561[v1564] = v1565;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1567;
    v1567 = threadIdx.x;
    bool v1568;
    v1568 = 0l <= v1567;
    bool v1569;
    v1569 = v1568 == false;
    if (v1569){
        assert("The index needs to be zero or positive." && v1568);
    } else {
    }
    int v1571;
    v1571 = v1567 % 1l;
    bool v1572;
    v1572 = v1567 < 32l;
    bool v1573;
    v1573 = v1572 == false;
    if (v1573){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1572);
    } else {
    }
    assert("Tensor range check" && 0 <= v1567 && v1567 < 32l);
    assert("Tensor range check" && 0 <= v1567 && v1567 < 32l);
    int v1575;
    v1575 = 0l;
    while (while_method_1(v1575)){
        assert("Tensor range check" && 0 <= v1575 && v1575 < 1l);
        int v1577;
        v1577 = v1575 + v1567;
        float * v1578;
        v1578 = v1561[v1577];
        assert("Tensor range check" && 0 <= v1571 && v1571 < 1l);
        int v1579;
        v1579 = 4l * v1571;
        float v1580[4l];
        int v1581[4l];
        int v1582;
        v1582 = 0l;
        while (while_method_1(v1582)){
            assert("Tensor range check" && 0 <= v1582 && v1582 < 1l);
            int v1584;
            v1584 = 4l * v1582;
            assert("Tensor range check" && 0 <= v1582 && v1582 < 1l);
            int v1585;
            v1585 = v1584 + v1579;
            int4* v1586;
            v1586 = reinterpret_cast<int4*>(v1578 + v1585);
            int4* v1587;
            v1587 = reinterpret_cast<int4*>(v1580 + v1584);
            assert("Pointer alignment check" && (unsigned long long)(v1586) % 4l == 0 && (unsigned long long)(v1587) % 4l == 0);
            *v1587 = *v1586;
            v1582 += 1l ;
        }
        int v1588;
        v1588 = 0l;
        while (while_method_1(v1588)){
            int v1590;
            v1590 = 0l;
            while (while_method_2(v1590)){
                bool v1592;
                v1592 = 0l <= v1590;
                bool v1594;
                if (v1592){
                    bool v1593;
                    v1593 = v1590 < 4l;
                    v1594 = v1593;
                } else {
                    v1594 = false;
                }
                bool v1595;
                v1595 = v1594 == false;
                if (v1595){
                    assert("The indices should be inside the range of the dimension." && v1594);
                } else {
                }
                bool v1597;
                v1597 = 0l <= v1571;
                bool v1599;
                if (v1597){
                    bool v1598;
                    v1598 = v1571 < 1l;
                    v1599 = v1598;
                } else {
                    v1599 = false;
                }
                bool v1600;
                v1600 = v1599 == false;
                if (v1600){
                    assert("The indices should be inside the range of the dimension." && v1599);
                } else {
                }
                int v1602;
                v1602 = v1571 * 4l;
                int v1603;
                v1603 = v1590 + v1602;
                bool v1604;
                v1604 = 0l <= v1588;
                bool v1606;
                if (v1604){
                    bool v1605;
                    v1605 = v1588 < 1l;
                    v1606 = v1605;
                } else {
                    v1606 = false;
                }
                bool v1607;
                v1607 = v1606 == false;
                if (v1607){
                    assert("The indices should be inside the range of the dimension." && v1606);
                } else {
                }
                int v1609;
                v1609 = v1588 * 4l;
                int v1610;
                v1610 = v1603 + v1609;
                assert("Tensor range check" && 0 <= v1588 && v1588 < 1l);
                assert("Tensor range check" && 0 <= v1590 && v1590 < 4l);
                int v1611;
                v1611 = 4l * v1588;
                int v1612;
                v1612 = v1611 + v1590;
                v1581[v1612] = v1610;
                v1590 += 1l ;
            }
            v1588 += 1l ;
        }
        bool v1613;
        v1613 = 0l <= v1575;
        bool v1615;
        if (v1613){
            bool v1614;
            v1614 = v1575 < 1l;
            v1615 = v1614;
        } else {
            v1615 = false;
        }
        bool v1616;
        v1616 = v1615 == false;
        if (v1616){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1615);
        } else {
        }
        bool v1618;
        v1618 = v1568 && v1572;
        bool v1619;
        v1619 = v1618 == false;
        if (v1619){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1618);
        } else {
        }
        int v1621;
        v1621 = v1567 + v1575;
        bool v1622[4l];
        int v1623;
        v1623 = 0l;
        while (while_method_1(v1623)){
            int v1625;
            v1625 = 0l;
            while (while_method_2(v1625)){
                assert("Tensor range check" && 0 <= v1623 && v1623 < 1l);
                assert("Tensor range check" && 0 <= v1625 && v1625 < 4l);
                int v1627;
                v1627 = 4l * v1623;
                int v1628;
                v1628 = v1627 + v1625;
                float v1629;
                v1629 = v1580[v1628];
                int v1630;
                v1630 = v1581[v1628];
                bool v1631;
                v1631 = v1630 < 3l;
                assert("Tensor range check" && 0 <= v1623 && v1623 < 1l);
                assert("Tensor range check" && 0 <= v1625 && v1625 < 4l);
                v1622[v1628] = v1631;
                v1625 += 1l ;
            }
            v1623 += 1l ;
        }
        float v1632[4l];
        int v1633;
        v1633 = 0l;
        while (while_method_1(v1633)){
            int v1635;
            v1635 = 0l;
            while (while_method_2(v1635)){
                assert("Tensor range check" && 0 <= v1633 && v1633 < 1l);
                assert("Tensor range check" && 0 <= v1635 && v1635 < 4l);
                int v1637;
                v1637 = 4l * v1633;
                int v1638;
                v1638 = v1637 + v1635;
                float v1639;
                v1639 = v1580[v1638];
                bool v1640;
                v1640 = v1622[v1638];
                float v1643;
                if (v1640){
                    bool v1641;
                    v1641 = 0.0f >= v1639;
                    if (v1641){
                        v1643 = 0.0f;
                    } else {
                        v1643 = v1639;
                    }
                } else {
                    v1643 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1633 && v1633 < 1l);
                assert("Tensor range check" && 0 <= v1635 && v1635 < 4l);
                v1632[v1638] = v1643;
                v1635 += 1l ;
            }
            v1633 += 1l ;
        }
        float v1644;
        v1644 = 0.0f;
        int v1645;
        v1645 = 0l;
        while (while_method_1(v1645)){
            int v1647;
            v1647 = 0l;
            while (while_method_2(v1647)){
                assert("Tensor range check" && 0 <= v1645 && v1645 < 1l);
                assert("Tensor range check" && 0 <= v1647 && v1647 < 4l);
                int v1649;
                v1649 = 4l * v1645;
                int v1650;
                v1650 = v1649 + v1647;
                float v1651;
                v1651 = v1632[v1650];
                float v1652;
                v1652 = v1644 + v1651;
                v1644 = v1652;
                v1647 += 1l ;
            }
            v1645 += 1l ;
        }
        auto v1653 = cooperative_groups::coalesced_threads();
        int v1654;
        v1654 = threadIdx.x;
        auto v1655 = cooperative_groups::labeled_partition(v1653,v1654);
        Closure0 v1656{};
        float v1657;
        v1657 = cooperative_groups::reduce(v1655, v1644, v1656);
        int v1658[4l];
        int v1659;
        v1659 = 0l;
        while (while_method_1(v1659)){
            int v1661;
            v1661 = 0l;
            while (while_method_2(v1661)){
                assert("Tensor range check" && 0 <= v1659 && v1659 < 1l);
                assert("Tensor range check" && 0 <= v1661 && v1661 < 4l);
                int v1663;
                v1663 = 4l * v1659;
                int v1664;
                v1664 = v1663 + v1661;
                bool v1665;
                v1665 = v1622[v1664];
                int v1666;
                if (v1665){
                    v1666 = 1l;
                } else {
                    v1666 = 0l;
                }
                assert("Tensor range check" && 0 <= v1659 && v1659 < 1l);
                assert("Tensor range check" && 0 <= v1661 && v1661 < 4l);
                v1658[v1664] = v1666;
                v1661 += 1l ;
            }
            v1659 += 1l ;
        }
        int v1667;
        v1667 = 0l;
        int v1668;
        v1668 = 0l;
        while (while_method_1(v1668)){
            int v1670;
            v1670 = 0l;
            while (while_method_2(v1670)){
                assert("Tensor range check" && 0 <= v1668 && v1668 < 1l);
                assert("Tensor range check" && 0 <= v1670 && v1670 < 4l);
                int v1672;
                v1672 = 4l * v1668;
                int v1673;
                v1673 = v1672 + v1670;
                int v1674;
                v1674 = v1658[v1673];
                int v1675;
                v1675 = v1667 + v1674;
                v1667 = v1675;
                v1670 += 1l ;
            }
            v1668 += 1l ;
        }
        auto v1676 = cooperative_groups::coalesced_threads();
        int v1677;
        v1677 = threadIdx.x;
        auto v1678 = cooperative_groups::labeled_partition(v1676,v1677);
        Closure1 v1679{};
        int v1680;
        v1680 = cooperative_groups::reduce(v1678, v1667, v1679);
        float v1681;
        v1681 = (float)v1680;
        float v1682;
        v1682 = 1.0f / v1681;
        float v1683[4l];
        int v1684;
        v1684 = 0l;
        while (while_method_1(v1684)){
            int v1686;
            v1686 = 0l;
            while (while_method_2(v1686)){
                assert("Tensor range check" && 0 <= v1684 && v1684 < 1l);
                assert("Tensor range check" && 0 <= v1686 && v1686 < 4l);
                int v1688;
                v1688 = 4l * v1684;
                int v1689;
                v1689 = v1688 + v1686;
                float v1690;
                v1690 = v1632[v1689];
                bool v1691;
                v1691 = v1622[v1689];
                bool v1692;
                v1692 = v1691 == false;
                float v1697;
                if (v1692){
                    v1697 = 0.0f;
                } else {
                    bool v1693;
                    v1693 = v1657 == 0.0f;
                    bool v1694;
                    v1694 = v1693 != true;
                    if (v1694){
                        float v1695;
                        v1695 = v1690 / v1657;
                        v1697 = v1695;
                    } else {
                        v1697 = v1682;
                    }
                }
                assert("Tensor range check" && 0 <= v1684 && v1684 < 1l);
                assert("Tensor range check" && 0 <= v1686 && v1686 < 4l);
                v1683[v1689] = v1697;
                v1686 += 1l ;
            }
            v1684 += 1l ;
        }
        float v1698[4l];
        float v1699;
        v1699 = 0.0f;
        int v1700;
        v1700 = 0l;
        while (while_method_1(v1700)){
            assert("Tensor range check" && 0 <= v1700 && v1700 < 1l);
            int v1702;
            v1702 = 4l * v1700;
            assert("Tensor range check" && 0 <= v1700 && v1700 < 1l);
            int v1703; float v1704;
            Tuple0 tmp55 = Tuple0{0l, 0.0f};
            v1703 = tmp55.v0; v1704 = tmp55.v1;
            while (while_method_2(v1703)){
                assert("Tensor range check" && 0 <= v1703 && v1703 < 4l);
                int v1706;
                v1706 = v1703 + v1702;
                float v1707;
                v1707 = v1683[v1706];
                float v1708;
                v1708 = v1704 + v1707;
                v1704 = v1708;
                v1703 += 1l ;
            }
            auto v1709 = cooperative_groups::coalesced_threads();
            int v1710;
            v1710 = threadIdx.x;
            auto v1711 = cooperative_groups::labeled_partition(v1709,v1710);
            Closure2 v1712{};
            float v1713;
            v1713 = cooperative_groups::inclusive_scan(v1711, v1704, v1712);
            float v1714;
            v1714 = v1711.shfl_up(v1713,1);
            bool v1715;
            v1715 = v1711.thread_rank() == 0;
            float v1716;
            if (v1715){
                v1716 = 0.0f;
            } else {
                v1716 = v1714;
            }
            float v1717;
            v1717 = v1711.shfl(v1713,v1711.num_threads()-1);
            float v1718;
            v1718 = v1699 + v1716;
            int v1719; float v1720;
            Tuple0 tmp56 = Tuple0{0l, v1718};
            v1719 = tmp56.v0; v1720 = tmp56.v1;
            while (while_method_2(v1719)){
                assert("Tensor range check" && 0 <= v1719 && v1719 < 4l);
                int v1722;
                v1722 = v1719 + v1702;
                float v1723;
                v1723 = v1683[v1722];
                float v1724;
                v1724 = v1720 + v1723;
                assert("Tensor range check" && 0 <= v1719 && v1719 < 4l);
                v1698[v1722] = v1724;
                v1720 = v1724;
                v1719 += 1l ;
            }
            float v1725;
            v1725 = v1699 + v1717;
            v1699 = v1725;
            v1700 += 1l ;
        }
        float v1726[4l];
        bool v1727[4l];
        int v1728;
        v1728 = 0l;
        while (while_method_1(v1728)){
            int v1730;
            v1730 = 0l;
            while (while_method_2(v1730)){
                assert("Tensor range check" && 0 <= v1728 && v1728 < 1l);
                assert("Tensor range check" && 0 <= v1730 && v1730 < 4l);
                int v1732;
                v1732 = 4l * v1728;
                int v1733;
                v1733 = v1732 + v1730;
                float v1734;
                v1734 = v1698[v1733];
                float v1735;
                v1735 = v1683[v1733];
                bool v1736;
                v1736 = v1735 > 0.0f;
                assert("Tensor range check" && 0 <= v1728 && v1728 < 1l);
                assert("Tensor range check" && 0 <= v1730 && v1730 < 4l);
                v1726[v1733] = v1734;
                v1727[v1733] = v1736;
                v1730 += 1l ;
            }
            v1728 += 1l ;
        }
        float v1737; bool v1738;
        Tuple1 tmp57 = Tuple1{-1.0f / 0.0f, false};
        v1737 = tmp57.v0; v1738 = tmp57.v1;
        int v1739;
        v1739 = 0l;
        while (while_method_1(v1739)){
            int v1741;
            v1741 = 0l;
            while (while_method_2(v1741)){
                assert("Tensor range check" && 0 <= v1739 && v1739 < 1l);
                assert("Tensor range check" && 0 <= v1741 && v1741 < 4l);
                int v1743;
                v1743 = 4l * v1739;
                int v1744;
                v1744 = v1743 + v1741;
                float v1745;
                v1745 = v1726[v1744];
                bool v1746;
                v1746 = v1727[v1744];
                float v1753; bool v1754;
                if (v1738){
                    if (v1746){
                        bool v1747;
                        v1747 = v1737 >= v1745;
                        float v1748;
                        if (v1747){
                            v1748 = v1737;
                        } else {
                            v1748 = v1745;
                        }
                        v1753 = v1748; v1754 = true;
                    } else {
                        v1753 = v1737; v1754 = v1738;
                    }
                } else {
                    if (v1746){
                        v1753 = v1745; v1754 = v1746;
                    } else {
                        v1753 = v1737; v1754 = v1738;
                    }
                }
                v1737 = v1753;
                v1738 = v1754;
                v1741 += 1l ;
            }
            v1739 += 1l ;
        }
        auto v1755 = cooperative_groups::coalesced_threads();
        int v1756;
        v1756 = threadIdx.x;
        auto v1757 = cooperative_groups::labeled_partition(v1755,v1756);
        Closure3 v1758{};
        float v1759; bool v1760;
        Tuple1 tmp58 = cooperative_groups::reduce(v1757, Tuple1{v1737, v1738}, v1758);
        v1759 = tmp58.v0; v1760 = tmp58.v1;
        bool v1761;
        v1761 = v1760 == false;
        if (v1761){
            assert("The local reduce must be true." && v1760);
        } else {
        }
        float v1763[4l];
        int v1764[4l];
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
                int v1771;
                v1771 = v1581[v1770];
                float v1772;
                v1772 = curand_uniform(&v16);
                assert("Tensor range check" && 0 <= v1765 && v1765 < 1l);
                assert("Tensor range check" && 0 <= v1767 && v1767 < 4l);
                v1763[v1770] = v1772;
                v1764[v1770] = v1771;
                v1767 += 1l ;
            }
            v1765 += 1l ;
        }
        float v1773; int v1774;
        Tuple2 tmp59 = Tuple2{0.0f, 2147483647l};
        v1773 = tmp59.v0; v1774 = tmp59.v1;
        int v1775;
        v1775 = 0l;
        while (while_method_1(v1775)){
            int v1777;
            v1777 = 0l;
            while (while_method_2(v1777)){
                assert("Tensor range check" && 0 <= v1775 && v1775 < 1l);
                assert("Tensor range check" && 0 <= v1777 && v1777 < 4l);
                int v1779;
                v1779 = 4l * v1775;
                int v1780;
                v1780 = v1779 + v1777;
                float v1781;
                v1781 = v1763[v1780];
                int v1782;
                v1782 = v1764[v1780];
                bool v1783;
                v1783 = v1774 < v1782;
                float v1784; int v1785;
                if (v1783){
                    v1784 = v1773; v1785 = v1774;
                } else {
                    v1784 = v1781; v1785 = v1782;
                }
                v1773 = v1784;
                v1774 = v1785;
                v1777 += 1l ;
            }
            v1775 += 1l ;
        }
        auto v1786 = cooperative_groups::coalesced_threads();
        int v1787;
        v1787 = threadIdx.x;
        auto v1788 = cooperative_groups::labeled_partition(v1786,v1787);
        Closure4 v1789{};
        float v1790; int v1791;
        Tuple2 tmp60 = cooperative_groups::reduce(v1788, Tuple2{v1773, v1774}, v1789);
        v1790 = tmp60.v0; v1791 = tmp60.v1;
        float v1792;
        v1792 = v1759 * v1790;
        int v1793[4l];
        bool v1794[4l];
        int v1795;
        v1795 = 0l;
        while (while_method_1(v1795)){
            int v1797;
            v1797 = 0l;
            while (while_method_2(v1797)){
                assert("Tensor range check" && 0 <= v1795 && v1795 < 1l);
                assert("Tensor range check" && 0 <= v1797 && v1797 < 4l);
                int v1799;
                v1799 = 4l * v1795;
                int v1800;
                v1800 = v1799 + v1797;
                float v1801;
                v1801 = v1726[v1800];
                bool v1802;
                v1802 = v1727[v1800];
                int v1803;
                v1803 = v1581[v1800];
                int v1806; bool v1807;
                if (v1802){
                    float v1804;
                    v1804 = v1801 - v1792;
                    bool v1805;
                    v1805 = v1804 >= 0.0f;
                    v1806 = v1803; v1807 = v1805;
                } else {
                    v1806 = 2147483647l; v1807 = false;
                }
                assert("Tensor range check" && 0 <= v1795 && v1795 < 1l);
                assert("Tensor range check" && 0 <= v1797 && v1797 < 4l);
                v1793[v1800] = v1806;
                v1794[v1800] = v1807;
                v1797 += 1l ;
            }
            v1795 += 1l ;
        }
        int v1808; bool v1809;
        Tuple3 tmp61 = Tuple3{2147483647l, false};
        v1808 = tmp61.v0; v1809 = tmp61.v1;
        int v1810;
        v1810 = 0l;
        while (while_method_1(v1810)){
            int v1812;
            v1812 = 0l;
            while (while_method_2(v1812)){
                assert("Tensor range check" && 0 <= v1810 && v1810 < 1l);
                assert("Tensor range check" && 0 <= v1812 && v1812 < 4l);
                int v1814;
                v1814 = 4l * v1810;
                int v1815;
                v1815 = v1814 + v1812;
                int v1816;
                v1816 = v1793[v1815];
                bool v1817;
                v1817 = v1794[v1815];
                int v1824; bool v1825;
                if (v1809){
                    if (v1817){
                        bool v1818;
                        v1818 = v1808 < v1816;
                        int v1819;
                        if (v1818){
                            v1819 = v1808;
                        } else {
                            v1819 = v1816;
                        }
                        v1824 = v1819; v1825 = true;
                    } else {
                        v1824 = v1808; v1825 = v1809;
                    }
                } else {
                    if (v1817){
                        v1824 = v1816; v1825 = v1817;
                    } else {
                        v1824 = v1808; v1825 = v1809;
                    }
                }
                v1808 = v1824;
                v1809 = v1825;
                v1812 += 1l ;
            }
            v1810 += 1l ;
        }
        auto v1826 = cooperative_groups::coalesced_threads();
        int v1827;
        v1827 = threadIdx.x;
        auto v1828 = cooperative_groups::labeled_partition(v1826,v1827);
        Closure5 v1829{};
        int v1830; bool v1831;
        Tuple3 tmp62 = cooperative_groups::reduce(v1828, Tuple3{v1808, v1809}, v1829);
        v1830 = tmp62.v0; v1831 = tmp62.v1;
        bool v1832;
        v1832 = v1831 == false;
        if (v1832){
            assert("The local reduce must be true." && v1831);
        } else {
        }
        float v1834; int v1835;
        Tuple2 tmp63 = Tuple2{0.0f, 2147483647l};
        v1834 = tmp63.v0; v1835 = tmp63.v1;
        int v1836;
        v1836 = 0l;
        while (while_method_1(v1836)){
            int v1838;
            v1838 = 0l;
            while (while_method_2(v1838)){
                assert("Tensor range check" && 0 <= v1836 && v1836 < 1l);
                assert("Tensor range check" && 0 <= v1838 && v1838 < 4l);
                int v1840;
                v1840 = 4l * v1836;
                int v1841;
                v1841 = v1840 + v1838;
                float v1842;
                v1842 = v1683[v1841];
                int v1843;
                v1843 = v1581[v1841];
                bool v1844;
                v1844 = v1835 == v1830;
                float v1848; int v1849;
                if (v1844){
                    v1848 = v1834; v1849 = v1835;
                } else {
                    bool v1845;
                    v1845 = v1843 == v1830;
                    if (v1845){
                        v1848 = v1842; v1849 = v1843;
                    } else {
                        v1848 = v1834; v1849 = v1835;
                    }
                }
                v1834 = v1848;
                v1835 = v1849;
                v1838 += 1l ;
            }
            v1836 += 1l ;
        }
        auto v1850 = cooperative_groups::coalesced_threads();
        int v1851;
        v1851 = threadIdx.x;
        auto v1852 = cooperative_groups::labeled_partition(v1850,v1851);
        Closure6 v1853{v1830};
        float v1854; int v1855;
        Tuple2 tmp64 = cooperative_groups::reduce(v1852, Tuple2{v1834, v1835}, v1853);
        v1854 = tmp64.v0; v1855 = tmp64.v1;
        bool v1856;
        v1856 = v1855 == 2147483647l;
        bool v1857;
        v1857 = v1856 != true;
        bool v1858;
        v1858 = v1857 == false;
        if (v1858){
            assert("Expected a valid action id in get_action." && v1857);
        } else {
        }
        assert("Tensor range check" && 0 <= v1575 && v1575 < 1l);
        v1562[v1577] = v1830;
        v1563[v1577] = v1854;
        v1575 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1860;
    v1860 = threadIdx.x;
    assert("Tensor range check" && 0 <= v1860 && v1860 < 32l);
    int v1861;
    v1861 = v1562[v1860];
    float v1862;
    v1862 = v1563[v1860];
    v1560[0l] = Tuple0{v1861, v1862};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1863; float v1864;
    Tuple0 tmp65 = v1560[0l];
    v1863 = tmp65.v0; v1864 = tmp65.v1;
    push__0(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v17, v18, v19, v20, v1559, v1558, v1864, v1863);
    int v1865 = v18;
    int v1866; float v1867;
    Tuple0 tmp66 = Tuple0{v1865, -13.0f};
    v1866 = tmp66.v0; v1867 = tmp66.v1;
    while (while_method_3(v1866)){
        v1866 -= 1l ;
        assert("Tensor range check" && 0 <= v1866 && v1866 < 16l);
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
        int v1869;
        v1869 = 32l * v1866;
        int v1870;
        v1870 = v1869 + v17;
        int v1871;
        v1871 = v0[v1870];
        float v1872;
        v1872 = v1[v1870];
        int v1873;
        v1873 = v2[v1870];
        int v1874;
        v1874 = v3[v1870];
        assert("Tensor range check" && 0 <= v1874 && v1874 < 4096l);
        int v1875;
        v1875 = 4l * v1874;
        assert("Tensor range check" && 0 <= v1866 && v1866 < 16l);
        int v1876;
        v1876 = 64l * v1866;
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
        int v1877;
        v1877 = 2l * v17;
        int v1878;
        v1878 = v1877 + v1876;
        assert("Tensor range check" && 0 <= v1866 && v1866 < 16l);
        int v1879;
        v1879 = 128l * v1866;
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
        int v1880;
        v1880 = 4l * v17;
        int v1881;
        v1881 = v1880 + v1879;
        assert("Tensor range check" && 0 <= v1866 && v1866 < 16l);
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
        v7[v1870] = v1867;
        float v1882[1l];
        __shared__ Tuple4 v1883[32l];
        __shared__ float * v1884[32l];
        __shared__ float v1885[32l];
        int v1886;
        v1886 = threadIdx.x;
        float * v1887;
        v1887 = v9+v1875;
        float * v1889;
        v1889 = v11+v1875;
        float * v1891;
        v1891 = v12+v1875;
        assert("Tensor range check" && 0 <= v1886 && v1886 < 32l);
        v1883[v1886] = Tuple4{v1887, v1889, v1891};
        int v1893;
        v1893 = threadIdx.x;
        float * v1894;
        v1894 = v6+v1881;
        assert("Tensor range check" && 0 <= v1893 && v1893 < 32l);
        v1884[v1893] = v1894;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v1896;
        v1896 = threadIdx.x;
        bool v1897;
        v1897 = 0l <= v1896;
        bool v1898;
        v1898 = v1897 == false;
        if (v1898){
            assert("The index needs to be zero or positive." && v1897);
        } else {
        }
        int v1900;
        v1900 = v1896 % 1l;
        bool v1901;
        v1901 = v1896 < 32l;
        bool v1902;
        v1902 = v1901 == false;
        if (v1902){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1901);
        } else {
        }
        assert("Tensor range check" && 0 <= v1896 && v1896 < 32l);
        assert("Tensor range check" && 0 <= v1896 && v1896 < 32l);
        assert("Tensor range check" && 0 <= v1896 && v1896 < 32l);
        int v1904;
        v1904 = 0l;
        while (while_method_1(v1904)){
            assert("Tensor range check" && 0 <= v1904 && v1904 < 1l);
            int v1906;
            v1906 = v1904 + v1896;
            float * v1907; float * v1908; float * v1909;
            Tuple4 tmp67 = v1883[v1906];
            v1907 = tmp67.v0; v1908 = tmp67.v1; v1909 = tmp67.v2;
            assert("Tensor range check" && 0 <= v1904 && v1904 < 1l);
            float * v1910;
            v1910 = v1884[v1906];
            assert("Tensor range check" && 0 <= v1900 && v1900 < 1l);
            int v1911;
            v1911 = 4l * v1900;
            float v1912[4l];
            float v1913[4l];
            float v1914[4l];
            int v1915[4l];
            int v1916;
            v1916 = 0l;
            while (while_method_1(v1916)){
                assert("Tensor range check" && 0 <= v1916 && v1916 < 1l);
                int v1918;
                v1918 = 4l * v1916;
                assert("Tensor range check" && 0 <= v1916 && v1916 < 1l);
                int v1919;
                v1919 = v1918 + v1911;
                int4* v1920;
                v1920 = reinterpret_cast<int4*>(v1907 + v1919);
                int4* v1921;
                v1921 = reinterpret_cast<int4*>(v1912 + v1918);
                assert("Pointer alignment check" && (unsigned long long)(v1920) % 4l == 0 && (unsigned long long)(v1921) % 4l == 0);
                *v1921 = *v1920;
                int4* v1922;
                v1922 = reinterpret_cast<int4*>(v1908 + v1919);
                int4* v1923;
                v1923 = reinterpret_cast<int4*>(v1913 + v1918);
                assert("Pointer alignment check" && (unsigned long long)(v1922) % 4l == 0 && (unsigned long long)(v1923) % 4l == 0);
                *v1923 = *v1922;
                int4* v1924;
                v1924 = reinterpret_cast<int4*>(v1909 + v1919);
                int4* v1925;
                v1925 = reinterpret_cast<int4*>(v1914 + v1918);
                assert("Pointer alignment check" && (unsigned long long)(v1924) % 4l == 0 && (unsigned long long)(v1925) % 4l == 0);
                *v1925 = *v1924;
                v1916 += 1l ;
            }
            int v1926;
            v1926 = 0l;
            while (while_method_1(v1926)){
                int v1928;
                v1928 = 0l;
                while (while_method_2(v1928)){
                    bool v1930;
                    v1930 = 0l <= v1928;
                    bool v1932;
                    if (v1930){
                        bool v1931;
                        v1931 = v1928 < 4l;
                        v1932 = v1931;
                    } else {
                        v1932 = false;
                    }
                    bool v1933;
                    v1933 = v1932 == false;
                    if (v1933){
                        assert("The indices should be inside the range of the dimension." && v1932);
                    } else {
                    }
                    bool v1935;
                    v1935 = 0l <= v1900;
                    bool v1937;
                    if (v1935){
                        bool v1936;
                        v1936 = v1900 < 1l;
                        v1937 = v1936;
                    } else {
                        v1937 = false;
                    }
                    bool v1938;
                    v1938 = v1937 == false;
                    if (v1938){
                        assert("The indices should be inside the range of the dimension." && v1937);
                    } else {
                    }
                    int v1940;
                    v1940 = v1900 * 4l;
                    int v1941;
                    v1941 = v1928 + v1940;
                    bool v1942;
                    v1942 = 0l <= v1926;
                    bool v1944;
                    if (v1942){
                        bool v1943;
                        v1943 = v1926 < 1l;
                        v1944 = v1943;
                    } else {
                        v1944 = false;
                    }
                    bool v1945;
                    v1945 = v1944 == false;
                    if (v1945){
                        assert("The indices should be inside the range of the dimension." && v1944);
                    } else {
                    }
                    int v1947;
                    v1947 = v1926 * 4l;
                    int v1948;
                    v1948 = v1941 + v1947;
                    assert("Tensor range check" && 0 <= v1926 && v1926 < 1l);
                    assert("Tensor range check" && 0 <= v1928 && v1928 < 4l);
                    int v1949;
                    v1949 = 4l * v1926;
                    int v1950;
                    v1950 = v1949 + v1928;
                    v1915[v1950] = v1948;
                    v1928 += 1l ;
                }
                v1926 += 1l ;
            }
            bool v1951;
            v1951 = 0l <= v1904;
            bool v1953;
            if (v1951){
                bool v1952;
                v1952 = v1904 < 1l;
                v1953 = v1952;
            } else {
                v1953 = false;
            }
            bool v1954;
            v1954 = v1953 == false;
            if (v1954){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1953);
            } else {
            }
            bool v1956;
            v1956 = v1897 && v1901;
            bool v1957;
            v1957 = v1956 == false;
            if (v1957){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1956);
            } else {
            }
            int v1959;
            v1959 = v1896 + v1904;
            float v1960[4l];
            int v1961;
            v1961 = 0l;
            while (while_method_1(v1961)){
                int v1963;
                v1963 = 0l;
                while (while_method_2(v1963)){
                    assert("Tensor range check" && 0 <= v1961 && v1961 < 1l);
                    assert("Tensor range check" && 0 <= v1963 && v1963 < 4l);
                    int v1965;
                    v1965 = 4l * v1961;
                    int v1966;
                    v1966 = v1965 + v1963;
                    float v1967;
                    v1967 = v1913[v1966];
                    float v1968;
                    v1968 = v1914[v1966];
                    bool v1969;
                    v1969 = v1968 == 0.0f;
                    bool v1970;
                    v1970 = v1969 != true;
                    float v1972;
                    if (v1970){
                        float v1971;
                        v1971 = v1967 / v1968;
                        v1972 = v1971;
                    } else {
                        v1972 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v1961 && v1961 < 1l);
                    assert("Tensor range check" && 0 <= v1963 && v1963 < 4l);
                    v1960[v1966] = v1972;
                    v1963 += 1l ;
                }
                v1961 += 1l ;
            }
            bool v1973[4l];
            int v1974;
            v1974 = 0l;
            while (while_method_1(v1974)){
                int v1976;
                v1976 = 0l;
                while (while_method_2(v1976)){
                    assert("Tensor range check" && 0 <= v1974 && v1974 < 1l);
                    assert("Tensor range check" && 0 <= v1976 && v1976 < 4l);
                    int v1978;
                    v1978 = 4l * v1974;
                    int v1979;
                    v1979 = v1978 + v1976;
                    float v1980;
                    v1980 = v1912[v1979];
                    int v1981;
                    v1981 = v1915[v1979];
                    bool v1982;
                    v1982 = v1981 < 3l;
                    assert("Tensor range check" && 0 <= v1974 && v1974 < 1l);
                    assert("Tensor range check" && 0 <= v1976 && v1976 < 4l);
                    v1973[v1979] = v1982;
                    v1976 += 1l ;
                }
                v1974 += 1l ;
            }
            float v1983[4l];
            int v1984;
            v1984 = 0l;
            while (while_method_1(v1984)){
                int v1986;
                v1986 = 0l;
                while (while_method_2(v1986)){
                    assert("Tensor range check" && 0 <= v1984 && v1984 < 1l);
                    assert("Tensor range check" && 0 <= v1986 && v1986 < 4l);
                    int v1988;
                    v1988 = 4l * v1984;
                    int v1989;
                    v1989 = v1988 + v1986;
                    float v1990;
                    v1990 = v1912[v1989];
                    bool v1991;
                    v1991 = v1973[v1989];
                    float v1994;
                    if (v1991){
                        bool v1992;
                        v1992 = 0.0f >= v1990;
                        if (v1992){
                            v1994 = 0.0f;
                        } else {
                            v1994 = v1990;
                        }
                    } else {
                        v1994 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v1984 && v1984 < 1l);
                    assert("Tensor range check" && 0 <= v1986 && v1986 < 4l);
                    v1983[v1989] = v1994;
                    v1986 += 1l ;
                }
                v1984 += 1l ;
            }
            float v1995;
            v1995 = 0.0f;
            int v1996;
            v1996 = 0l;
            while (while_method_1(v1996)){
                int v1998;
                v1998 = 0l;
                while (while_method_2(v1998)){
                    assert("Tensor range check" && 0 <= v1996 && v1996 < 1l);
                    assert("Tensor range check" && 0 <= v1998 && v1998 < 4l);
                    int v2000;
                    v2000 = 4l * v1996;
                    int v2001;
                    v2001 = v2000 + v1998;
                    float v2002;
                    v2002 = v1983[v2001];
                    float v2003;
                    v2003 = v1995 + v2002;
                    v1995 = v2003;
                    v1998 += 1l ;
                }
                v1996 += 1l ;
            }
            auto v2004 = cooperative_groups::coalesced_threads();
            int v2005;
            v2005 = threadIdx.x;
            auto v2006 = cooperative_groups::labeled_partition(v2004,v2005);
            Closure0 v2007{};
            float v2008;
            v2008 = cooperative_groups::reduce(v2006, v1995, v2007);
            int v2009[4l];
            int v2010;
            v2010 = 0l;
            while (while_method_1(v2010)){
                int v2012;
                v2012 = 0l;
                while (while_method_2(v2012)){
                    assert("Tensor range check" && 0 <= v2010 && v2010 < 1l);
                    assert("Tensor range check" && 0 <= v2012 && v2012 < 4l);
                    int v2014;
                    v2014 = 4l * v2010;
                    int v2015;
                    v2015 = v2014 + v2012;
                    bool v2016;
                    v2016 = v1973[v2015];
                    int v2017;
                    if (v2016){
                        v2017 = 1l;
                    } else {
                        v2017 = 0l;
                    }
                    assert("Tensor range check" && 0 <= v2010 && v2010 < 1l);
                    assert("Tensor range check" && 0 <= v2012 && v2012 < 4l);
                    v2009[v2015] = v2017;
                    v2012 += 1l ;
                }
                v2010 += 1l ;
            }
            int v2018;
            v2018 = 0l;
            int v2019;
            v2019 = 0l;
            while (while_method_1(v2019)){
                int v2021;
                v2021 = 0l;
                while (while_method_2(v2021)){
                    assert("Tensor range check" && 0 <= v2019 && v2019 < 1l);
                    assert("Tensor range check" && 0 <= v2021 && v2021 < 4l);
                    int v2023;
                    v2023 = 4l * v2019;
                    int v2024;
                    v2024 = v2023 + v2021;
                    int v2025;
                    v2025 = v2009[v2024];
                    int v2026;
                    v2026 = v2018 + v2025;
                    v2018 = v2026;
                    v2021 += 1l ;
                }
                v2019 += 1l ;
            }
            auto v2027 = cooperative_groups::coalesced_threads();
            int v2028;
            v2028 = threadIdx.x;
            auto v2029 = cooperative_groups::labeled_partition(v2027,v2028);
            Closure1 v2030{};
            int v2031;
            v2031 = cooperative_groups::reduce(v2029, v2018, v2030);
            float v2032;
            v2032 = (float)v2031;
            float v2033;
            v2033 = 1.0f / v2032;
            float v2034[4l];
            int v2035;
            v2035 = 0l;
            while (while_method_1(v2035)){
                int v2037;
                v2037 = 0l;
                while (while_method_2(v2037)){
                    assert("Tensor range check" && 0 <= v2035 && v2035 < 1l);
                    assert("Tensor range check" && 0 <= v2037 && v2037 < 4l);
                    int v2039;
                    v2039 = 4l * v2035;
                    int v2040;
                    v2040 = v2039 + v2037;
                    float v2041;
                    v2041 = v1983[v2040];
                    bool v2042;
                    v2042 = v1973[v2040];
                    bool v2043;
                    v2043 = v2042 == false;
                    float v2048;
                    if (v2043){
                        v2048 = 0.0f;
                    } else {
                        bool v2044;
                        v2044 = v2008 == 0.0f;
                        bool v2045;
                        v2045 = v2044 != true;
                        if (v2045){
                            float v2046;
                            v2046 = v2041 / v2008;
                            v2048 = v2046;
                        } else {
                            v2048 = v2033;
                        }
                    }
                    assert("Tensor range check" && 0 <= v2035 && v2035 < 1l);
                    assert("Tensor range check" && 0 <= v2037 && v2037 < 4l);
                    v2034[v2040] = v2048;
                    v2037 += 1l ;
                }
                v2035 += 1l ;
            }
            float v2049[4l];
            int v2050;
            v2050 = 0l;
            while (while_method_1(v2050)){
                int v2052;
                v2052 = 0l;
                while (while_method_2(v2052)){
                    assert("Tensor range check" && 0 <= v2050 && v2050 < 1l);
                    assert("Tensor range check" && 0 <= v2052 && v2052 < 4l);
                    int v2054;
                    v2054 = 4l * v2050;
                    int v2055;
                    v2055 = v2054 + v2052;
                    float v2056;
                    v2056 = v1960[v2055];
                    int v2057;
                    v2057 = v1915[v2055];
                    bool v2058;
                    v2058 = v1871 == v2057;
                    float v2061;
                    if (v2058){
                        float v2059;
                        v2059 = v1867 - v2056;
                        float v2060;
                        v2060 = v2059 / v1872;
                        v2061 = v2060;
                    } else {
                        v2061 = 0.0f;
                    }
                    float v2062;
                    v2062 = v2061 + v2056;
                    assert("Tensor range check" && 0 <= v2050 && v2050 < 1l);
                    assert("Tensor range check" && 0 <= v2052 && v2052 < 4l);
                    v2049[v2055] = v2062;
                    v2052 += 1l ;
                }
                v2050 += 1l ;
            }
            float v2063[4l];
            int v2064;
            v2064 = 0l;
            while (while_method_1(v2064)){
                int v2066;
                v2066 = 0l;
                while (while_method_2(v2066)){
                    assert("Tensor range check" && 0 <= v2064 && v2064 < 1l);
                    assert("Tensor range check" && 0 <= v2066 && v2066 < 4l);
                    int v2068;
                    v2068 = 4l * v2064;
                    int v2069;
                    v2069 = v2068 + v2066;
                    float v2070;
                    v2070 = v2034[v2069];
                    float v2071;
                    v2071 = v2049[v2069];
                    float v2072;
                    v2072 = v2070 * v2071;
                    assert("Tensor range check" && 0 <= v2064 && v2064 < 1l);
                    assert("Tensor range check" && 0 <= v2066 && v2066 < 4l);
                    v2063[v2069] = v2072;
                    v2066 += 1l ;
                }
                v2064 += 1l ;
            }
            float v2073;
            v2073 = 0.0f;
            int v2074;
            v2074 = 0l;
            while (while_method_1(v2074)){
                int v2076;
                v2076 = 0l;
                while (while_method_2(v2076)){
                    assert("Tensor range check" && 0 <= v2074 && v2074 < 1l);
                    assert("Tensor range check" && 0 <= v2076 && v2076 < 4l);
                    int v2078;
                    v2078 = 4l * v2074;
                    int v2079;
                    v2079 = v2078 + v2076;
                    float v2080;
                    v2080 = v2063[v2079];
                    float v2081;
                    v2081 = v2073 + v2080;
                    v2073 = v2081;
                    v2076 += 1l ;
                }
                v2074 += 1l ;
            }
            auto v2082 = cooperative_groups::coalesced_threads();
            int v2083;
            v2083 = threadIdx.x;
            auto v2084 = cooperative_groups::labeled_partition(v2082,v2083);
            float v2085;
            v2085 = cooperative_groups::reduce(v2084, v2073, v2007);
            float v2086[4l];
            int v2087;
            v2087 = 0l;
            while (while_method_1(v2087)){
                int v2089;
                v2089 = 0l;
                while (while_method_2(v2089)){
                    assert("Tensor range check" && 0 <= v2087 && v2087 < 1l);
                    assert("Tensor range check" && 0 <= v2089 && v2089 < 4l);
                    int v2091;
                    v2091 = 4l * v2087;
                    int v2092;
                    v2092 = v2091 + v2089;
                    float v2093;
                    v2093 = v2049[v2092];
                    double v2094[2l];
                    int v2095;
                    v2095 = 0l;
                    while (while_method_0(v2095)){
                        assert("Tensor range check" && 0 <= v2095 && v2095 < 2l);
                        int v2097;
                        v2097 = v2095 + v1878;
                        double v2098;
                        v2098 = v4[v2097];
                        bool v2099;
                        v2099 = v1873 == v2095;
                        double v2100;
                        if (v2099){
                            v2100 = 0.0;
                        } else {
                            v2100 = v2098;
                        }
                        assert("Tensor range check" && 0 <= v2095 && v2095 < 2l);
                        v2094[v2095] = v2100;
                        v2095 += 1l ;
                    }
                    double v2101;
                    v2101 = 0.0;
                    int v2102;
                    v2102 = 0l;
                    while (while_method_0(v2102)){
                        assert("Tensor range check" && 0 <= v2102 && v2102 < 2l);
                        double v2104;
                        v2104 = v2094[v2102];
                        double v2105;
                        v2105 = v2101 + v2104;
                        v2101 = v2105;
                        v2102 += 1l ;
                    }
                    double v2106;
                    v2106 = 0.0;
                    int v2107;
                    v2107 = 0l;
                    while (while_method_0(v2107)){
                        assert("Tensor range check" && 0 <= v2107 && v2107 < 2l);
                        int v2109;
                        v2109 = v2107 + v1878;
                        double v2110;
                        v2110 = v5[v2109];
                        double v2111;
                        v2111 = v2106 + v2110;
                        v2106 = v2111;
                        v2107 += 1l ;
                    }
                    double v2112;
                    v2112 = v2101 - v2106;
                    double v2113;
                    v2113 = exp(v2112);
                    float v2114;
                    v2114 = (float)v2113;
                    float v2115;
                    v2115 = v2093 - v2085;
                    float v2116;
                    v2116 = v2114 * v2115;
                    assert("Tensor range check" && 0 <= v2087 && v2087 < 1l);
                    assert("Tensor range check" && 0 <= v2089 && v2089 < 4l);
                    v2086[v2092] = v2116;
                    v2089 += 1l ;
                }
                v2087 += 1l ;
            }
            assert("Tensor range check" && 0 <= v1904 && v1904 < 1l);
            v1885[v1906] = v2085;
            assert("Tensor range check" && 0 <= v1900 && v1900 < 1l);
            int v2117;
            v2117 = 0l;
            while (while_method_1(v2117)){
                assert("Tensor range check" && 0 <= v2117 && v2117 < 1l);
                int v2119;
                v2119 = 4l * v2117;
                int v2120;
                v2120 = v2119 + v1911;
                assert("Tensor range check" && 0 <= v2117 && v2117 < 1l);
                int4* v2121;
                v2121 = reinterpret_cast<int4*>(v2086 + v2119);
                int4* v2122;
                v2122 = reinterpret_cast<int4*>(v1910 + v2120);
                assert("Pointer alignment check" && (unsigned long long)(v2121) % 4l == 0 && (unsigned long long)(v2122) % 4l == 0);
                *v2122 = *v2121;
                v2117 += 1l ;
            }
            v1904 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v2123;
        v2123 = threadIdx.x;
        assert("Tensor range check" && 0 <= v2123 && v2123 < 32l);
        float v2124;
        v2124 = v1885[v2123];
        v1882[0l] = v2124;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        float v2125;
        v2125 = v1882[0l];
        v1867 = v2125;
    }
    cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v2126 = console_lock;
    auto v2127 = cooperative_groups::coalesced_threads();
    v2126.acquire();
    printf("{%s = %f}\n","fin_reward", v1867);
    v2126.release();
    v2127.sync() ;
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
