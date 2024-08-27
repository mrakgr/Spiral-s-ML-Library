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
__device__ Tuple0 method_1(float * v0, float * v1, float * v2, float * v3, float * v4, int v5, int v6);
__device__ void push_0(float * v0, float * v1, float * v2, float * v3, float * v4, int * v5, float * v6, int * v7, int * v8, double * v9, double * v10, float * v11, float * v12, float * v13, int v14, int & v15, double * v16, double * v17, int v18, int v19, int v20);
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
__device__ Tuple0 method_1(float * v0, float * v1, float * v2, float * v3, float * v4, int v5, int v6){
    assert("Tensor range check" && 0 <= v6 && v6 < 4l);
    int v7;
    v7 = 16384l * v6;
    assert("Tensor range check" && 0 <= v5 && v5 < 4096l);
    int v8;
    v8 = 4l * v5;
    int v9;
    v9 = v8 + v7;
    float * v10;
    v10 = v0+v9;
    float * v12;
    v12 = v1+v9;
    __shared__ float * v14[32l];
    __shared__ float * v15[32l];
    /* void shared array create v16 */;
    __shared__ float v17[32l];
    __shared__ float v18[32l];
    __shared__ int v19[32l];
    int v20;
    v20 = threadIdx.x;
    bool v21;
    v21 = v20 < 32l;
    if (v21){
        assert("Tensor range check" && 0 <= v20 && v20 < 32l);
        v14[v20] = v10;
        v15[v20] = v12;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v22;
    v22 = 0l <= v20;
    bool v23;
    v23 = v22 == false;
    if (v23){
        assert("The index needs to be zero or positive." && v22);
    } else {
    }
    int v25;
    v25 = v20 % 1l;
    bool v26;
    v26 = v21 == false;
    if (v26){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v21);
    } else {
    }
    assert("Tensor range check" && 0 <= v20 && v20 < 32l);
    int v28;
    v28 = 0l;
    while (while_method_1(v28)){
        bool v30;
        v30 = v22 && v21;
        bool v31;
        v31 = v30 == false;
        if (v31){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v30);
        } else {
        }
        bool v33;
        v33 = 0l <= v28;
        bool v35;
        if (v33){
            bool v34;
            v34 = v28 < 1l;
            v35 = v34;
        } else {
            v35 = false;
        }
        bool v36;
        v36 = v35 == false;
        if (v36){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v35);
        } else {
        }
        int v38;
        v38 = v28 * 32l;
        int v39;
        v39 = v38 + v20;
        assert("Tensor range check" && 0 <= v28 && v28 < 1l);
        int v40;
        v40 = 32l * v28;
        int v41;
        v41 = v40 + v20;
        float * v42;
        v42 = v14[v41];
        float * v43;
        v43 = v15[v41];
        /* void array index */;
        assert("Tensor range check" && 0 <= v25 && v25 < 1l);
        int v44;
        v44 = 4l * v25;
        float v45[4l];
        float v46[4l];
        int v47[4l];
        int v48;
        v48 = 0l;
        while (while_method_1(v48)){
            assert("Tensor range check" && 0 <= v48 && v48 < 1l);
            int v50;
            v50 = 4l * v48;
            assert("Tensor range check" && 0 <= v48 && v48 < 1l);
            int v51;
            v51 = v50 + v44;
            int4* v52;
            v52 = reinterpret_cast<int4*>(v42 + v51);
            int4* v53;
            v53 = reinterpret_cast<int4*>(v45 + v50);
            assert("Pointer alignment check" && (unsigned long long)(v52) % 4l == 0 && (unsigned long long)(v53) % 4l == 0);
            *v53 = *v52;
            int4* v54;
            v54 = reinterpret_cast<int4*>(v43 + v51);
            int4* v55;
            v55 = reinterpret_cast<int4*>(v46 + v50);
            assert("Pointer alignment check" && (unsigned long long)(v54) % 4l == 0 && (unsigned long long)(v55) % 4l == 0);
            *v55 = *v54;
            v48 += 1l ;
        }
        int v56;
        v56 = 0l;
        while (while_method_1(v56)){
            int v58;
            v58 = 0l;
            while (while_method_2(v58)){
                bool v60;
                v60 = 0l <= v58;
                bool v62;
                if (v60){
                    bool v61;
                    v61 = v58 < 4l;
                    v62 = v61;
                } else {
                    v62 = false;
                }
                bool v63;
                v63 = v62 == false;
                if (v63){
                    assert("The indices should be inside the range of the dimension." && v62);
                } else {
                }
                bool v65;
                v65 = 0l <= v25;
                bool v67;
                if (v65){
                    bool v66;
                    v66 = v25 < 1l;
                    v67 = v66;
                } else {
                    v67 = false;
                }
                bool v68;
                v68 = v67 == false;
                if (v68){
                    assert("The indices should be inside the range of the dimension." && v67);
                } else {
                }
                int v70;
                v70 = v25 * 4l;
                int v71;
                v71 = v58 + v70;
                bool v72;
                v72 = 0l <= v56;
                bool v74;
                if (v72){
                    bool v73;
                    v73 = v56 < 1l;
                    v74 = v73;
                } else {
                    v74 = false;
                }
                bool v75;
                v75 = v74 == false;
                if (v75){
                    assert("The indices should be inside the range of the dimension." && v74);
                } else {
                }
                int v77;
                v77 = v56 * 4l;
                int v78;
                v78 = v71 + v77;
                assert("Tensor range check" && 0 <= v56 && v56 < 1l);
                assert("Tensor range check" && 0 <= v58 && v58 < 4l);
                int v79;
                v79 = 4l * v56;
                int v80;
                v80 = v79 + v58;
                v47[v80] = v78;
                v58 += 1l ;
            }
            v56 += 1l ;
        }
        unsigned long long v81;
        v81 = clock64();
        int v82;
        v82 = threadIdx.x;
        unsigned long long v83;
        v83 = (unsigned long long)v82;
        curandStatePhilox4_32_10_t v84;
        curand_init(v81,v83,0ull,&v84);
        bool v85[4l];
        int v86;
        v86 = 0l;
        while (while_method_1(v86)){
            int v88;
            v88 = 0l;
            while (while_method_2(v88)){
                assert("Tensor range check" && 0 <= v86 && v86 < 1l);
                assert("Tensor range check" && 0 <= v88 && v88 < 4l);
                int v90;
                v90 = 4l * v86;
                int v91;
                v91 = v90 + v88;
                float v92;
                v92 = v45[v91];
                int v93;
                v93 = v47[v91];
                bool v94;
                v94 = v93 < 3l;
                assert("Tensor range check" && 0 <= v86 && v86 < 1l);
                assert("Tensor range check" && 0 <= v88 && v88 < 4l);
                v85[v91] = v94;
                v88 += 1l ;
            }
            v86 += 1l ;
        }
        float v95[4l];
        int v96;
        v96 = 0l;
        while (while_method_1(v96)){
            int v98;
            v98 = 0l;
            while (while_method_2(v98)){
                assert("Tensor range check" && 0 <= v96 && v96 < 1l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                int v100;
                v100 = 4l * v96;
                int v101;
                v101 = v100 + v98;
                float v102;
                v102 = v45[v101];
                bool v103;
                v103 = v85[v101];
                float v106;
                if (v103){
                    bool v104;
                    v104 = 0.0f >= v102;
                    if (v104){
                        v106 = 0.0f;
                    } else {
                        v106 = v102;
                    }
                } else {
                    v106 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v96 && v96 < 1l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                v95[v101] = v106;
                v98 += 1l ;
            }
            v96 += 1l ;
        }
        float v107;
        v107 = 0.0f;
        int v108;
        v108 = 0l;
        while (while_method_1(v108)){
            int v110;
            v110 = 0l;
            while (while_method_2(v110)){
                assert("Tensor range check" && 0 <= v108 && v108 < 1l);
                assert("Tensor range check" && 0 <= v110 && v110 < 4l);
                int v112;
                v112 = 4l * v108;
                int v113;
                v113 = v112 + v110;
                float v114;
                v114 = v95[v113];
                float v115;
                v115 = v107 + v114;
                v107 = v115;
                v110 += 1l ;
            }
            v108 += 1l ;
        }
        auto v116 = cooperative_groups::coalesced_threads();
        int v117;
        v117 = threadIdx.x;
        auto v118 = cooperative_groups::labeled_partition(v116,v117);
        Closure0 v119{};
        float v120;
        v120 = cooperative_groups::reduce(v118, v107, v119);
        int v121[4l];
        int v122;
        v122 = 0l;
        while (while_method_1(v122)){
            int v124;
            v124 = 0l;
            while (while_method_2(v124)){
                assert("Tensor range check" && 0 <= v122 && v122 < 1l);
                assert("Tensor range check" && 0 <= v124 && v124 < 4l);
                int v126;
                v126 = 4l * v122;
                int v127;
                v127 = v126 + v124;
                bool v128;
                v128 = v85[v127];
                int v129;
                if (v128){
                    v129 = 1l;
                } else {
                    v129 = 0l;
                }
                assert("Tensor range check" && 0 <= v122 && v122 < 1l);
                assert("Tensor range check" && 0 <= v124 && v124 < 4l);
                v121[v127] = v129;
                v124 += 1l ;
            }
            v122 += 1l ;
        }
        int v130;
        v130 = 0l;
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
                int v137;
                v137 = v121[v136];
                int v138;
                v138 = v130 + v137;
                v130 = v138;
                v133 += 1l ;
            }
            v131 += 1l ;
        }
        auto v139 = cooperative_groups::coalesced_threads();
        int v140;
        v140 = threadIdx.x;
        auto v141 = cooperative_groups::labeled_partition(v139,v140);
        Closure1 v142{};
        int v143;
        v143 = cooperative_groups::reduce(v141, v130, v142);
        float v144;
        v144 = (float)v143;
        float v145;
        v145 = 1.0f / v144;
        float v146[4l];
        int v147;
        v147 = 0l;
        while (while_method_1(v147)){
            int v149;
            v149 = 0l;
            while (while_method_2(v149)){
                assert("Tensor range check" && 0 <= v147 && v147 < 1l);
                assert("Tensor range check" && 0 <= v149 && v149 < 4l);
                int v151;
                v151 = 4l * v147;
                int v152;
                v152 = v151 + v149;
                float v153;
                v153 = v95[v152];
                bool v154;
                v154 = v85[v152];
                bool v155;
                v155 = v154 == false;
                float v160;
                if (v155){
                    v160 = 0.0f;
                } else {
                    bool v156;
                    v156 = v120 == 0.0f;
                    bool v157;
                    v157 = v156 != true;
                    if (v157){
                        float v158;
                        v158 = v153 / v120;
                        v160 = v158;
                    } else {
                        v160 = v145;
                    }
                }
                assert("Tensor range check" && 0 <= v147 && v147 < 1l);
                assert("Tensor range check" && 0 <= v149 && v149 < 4l);
                v146[v152] = v160;
                v149 += 1l ;
            }
            v147 += 1l ;
        }
        float v161[4l];
        float v162;
        v162 = 0.0f;
        int v163;
        v163 = 0l;
        while (while_method_1(v163)){
            assert("Tensor range check" && 0 <= v163 && v163 < 1l);
            int v165;
            v165 = 4l * v163;
            assert("Tensor range check" && 0 <= v163 && v163 < 1l);
            int v166; float v167;
            Tuple1 tmp0 = Tuple1{0l, 0.0f};
            v166 = tmp0.v0; v167 = tmp0.v1;
            while (while_method_2(v166)){
                assert("Tensor range check" && 0 <= v166 && v166 < 4l);
                int v169;
                v169 = v166 + v165;
                float v170;
                v170 = v146[v169];
                float v171;
                v171 = v167 + v170;
                v167 = v171;
                v166 += 1l ;
            }
            auto v172 = cooperative_groups::coalesced_threads();
            int v173;
            v173 = threadIdx.x;
            auto v174 = cooperative_groups::labeled_partition(v172,v173);
            Closure2 v175{};
            float v176;
            v176 = cooperative_groups::inclusive_scan(v174, v167, v175);
            float v177;
            v177 = v174.shfl_up(v176,1);
            bool v178;
            v178 = v174.thread_rank() == 0;
            float v179;
            if (v178){
                v179 = 0.0f;
            } else {
                v179 = v177;
            }
            float v180;
            v180 = v174.shfl(v176,v174.num_threads()-1);
            float v181;
            v181 = v162 + v179;
            int v182; float v183;
            Tuple1 tmp1 = Tuple1{0l, v181};
            v182 = tmp1.v0; v183 = tmp1.v1;
            while (while_method_2(v182)){
                assert("Tensor range check" && 0 <= v182 && v182 < 4l);
                int v185;
                v185 = v182 + v165;
                float v186;
                v186 = v146[v185];
                float v187;
                v187 = v183 + v186;
                assert("Tensor range check" && 0 <= v182 && v182 < 4l);
                v161[v185] = v187;
                v183 = v187;
                v182 += 1l ;
            }
            float v188;
            v188 = v162 + v180;
            v162 = v188;
            v163 += 1l ;
        }
        float v189[4l];
        bool v190[4l];
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
                v197 = v161[v196];
                float v198;
                v198 = v146[v196];
                bool v199;
                v199 = v198 > 0.0f;
                assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                assert("Tensor range check" && 0 <= v193 && v193 < 4l);
                v189[v196] = v197;
                v190[v196] = v199;
                v193 += 1l ;
            }
            v191 += 1l ;
        }
        float v200; bool v201;
        Tuple2 tmp2 = Tuple2{-1.0f / 0.0f, false};
        v200 = tmp2.v0; v201 = tmp2.v1;
        int v202;
        v202 = 0l;
        while (while_method_1(v202)){
            int v204;
            v204 = 0l;
            while (while_method_2(v204)){
                assert("Tensor range check" && 0 <= v202 && v202 < 1l);
                assert("Tensor range check" && 0 <= v204 && v204 < 4l);
                int v206;
                v206 = 4l * v202;
                int v207;
                v207 = v206 + v204;
                float v208;
                v208 = v189[v207];
                bool v209;
                v209 = v190[v207];
                float v216; bool v217;
                if (v201){
                    if (v209){
                        bool v210;
                        v210 = v200 >= v208;
                        float v211;
                        if (v210){
                            v211 = v200;
                        } else {
                            v211 = v208;
                        }
                        v216 = v211; v217 = true;
                    } else {
                        v216 = v200; v217 = v201;
                    }
                } else {
                    if (v209){
                        v216 = v208; v217 = v209;
                    } else {
                        v216 = v200; v217 = v201;
                    }
                }
                v200 = v216;
                v201 = v217;
                v204 += 1l ;
            }
            v202 += 1l ;
        }
        auto v218 = cooperative_groups::coalesced_threads();
        int v219;
        v219 = threadIdx.x;
        auto v220 = cooperative_groups::labeled_partition(v218,v219);
        Closure3 v221{};
        float v222; bool v223;
        Tuple2 tmp3 = cooperative_groups::reduce(v220, Tuple2{v200, v201}, v221);
        v222 = tmp3.v0; v223 = tmp3.v1;
        bool v224;
        v224 = v223 == false;
        if (v224){
            assert("The local reduce must be true." && v223);
        } else {
        }
        float v226[4l];
        int v227[4l];
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
                v234 = v47[v233];
                float v235;
                v235 = curand_uniform(&v84);
                assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                assert("Tensor range check" && 0 <= v230 && v230 < 4l);
                v226[v233] = v235;
                v227[v233] = v234;
                v230 += 1l ;
            }
            v228 += 1l ;
        }
        float v236; int v237;
        Tuple3 tmp4 = Tuple3{0.0f, 2147483647l};
        v236 = tmp4.v0; v237 = tmp4.v1;
        int v238;
        v238 = 0l;
        while (while_method_1(v238)){
            int v240;
            v240 = 0l;
            while (while_method_2(v240)){
                assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                assert("Tensor range check" && 0 <= v240 && v240 < 4l);
                int v242;
                v242 = 4l * v238;
                int v243;
                v243 = v242 + v240;
                float v244;
                v244 = v226[v243];
                int v245;
                v245 = v227[v243];
                bool v246;
                v246 = v237 < v245;
                float v247; int v248;
                if (v246){
                    v247 = v236; v248 = v237;
                } else {
                    v247 = v244; v248 = v245;
                }
                v236 = v247;
                v237 = v248;
                v240 += 1l ;
            }
            v238 += 1l ;
        }
        auto v249 = cooperative_groups::coalesced_threads();
        int v250;
        v250 = threadIdx.x;
        auto v251 = cooperative_groups::labeled_partition(v249,v250);
        Closure4 v252{};
        float v253; int v254;
        Tuple3 tmp5 = cooperative_groups::reduce(v251, Tuple3{v236, v237}, v252);
        v253 = tmp5.v0; v254 = tmp5.v1;
        float v255;
        v255 = v222 * v253;
        int v256[4l];
        bool v257[4l];
        int v258;
        v258 = 0l;
        while (while_method_1(v258)){
            int v260;
            v260 = 0l;
            while (while_method_2(v260)){
                assert("Tensor range check" && 0 <= v258 && v258 < 1l);
                assert("Tensor range check" && 0 <= v260 && v260 < 4l);
                int v262;
                v262 = 4l * v258;
                int v263;
                v263 = v262 + v260;
                float v264;
                v264 = v189[v263];
                bool v265;
                v265 = v190[v263];
                int v266;
                v266 = v47[v263];
                int v269; bool v270;
                if (v265){
                    float v267;
                    v267 = v264 - v255;
                    bool v268;
                    v268 = v267 >= 0.0f;
                    v269 = v266; v270 = v268;
                } else {
                    v269 = 2147483647l; v270 = false;
                }
                assert("Tensor range check" && 0 <= v258 && v258 < 1l);
                assert("Tensor range check" && 0 <= v260 && v260 < 4l);
                v256[v263] = v269;
                v257[v263] = v270;
                v260 += 1l ;
            }
            v258 += 1l ;
        }
        int v271; bool v272;
        Tuple4 tmp6 = Tuple4{2147483647l, false};
        v271 = tmp6.v0; v272 = tmp6.v1;
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
                int v279;
                v279 = v256[v278];
                bool v280;
                v280 = v257[v278];
                int v287; bool v288;
                if (v272){
                    if (v280){
                        bool v281;
                        v281 = v271 < v279;
                        int v282;
                        if (v281){
                            v282 = v271;
                        } else {
                            v282 = v279;
                        }
                        v287 = v282; v288 = true;
                    } else {
                        v287 = v271; v288 = v272;
                    }
                } else {
                    if (v280){
                        v287 = v279; v288 = v280;
                    } else {
                        v287 = v271; v288 = v272;
                    }
                }
                v271 = v287;
                v272 = v288;
                v275 += 1l ;
            }
            v273 += 1l ;
        }
        auto v289 = cooperative_groups::coalesced_threads();
        int v290;
        v290 = threadIdx.x;
        auto v291 = cooperative_groups::labeled_partition(v289,v290);
        Closure5 v292{};
        int v293; bool v294;
        Tuple4 tmp7 = cooperative_groups::reduce(v291, Tuple4{v271, v272}, v292);
        v293 = tmp7.v0; v294 = tmp7.v1;
        bool v295;
        v295 = v294 == false;
        if (v295){
            assert("The local reduce must be true." && v294);
        } else {
        }
        bool v297[4l];
        int v298;
        v298 = 0l;
        while (while_method_1(v298)){
            int v300;
            v300 = 0l;
            while (while_method_2(v300)){
                assert("Tensor range check" && 0 <= v298 && v298 < 1l);
                assert("Tensor range check" && 0 <= v300 && v300 < 4l);
                int v302;
                v302 = 4l * v298;
                int v303;
                v303 = v302 + v300;
                float v304;
                v304 = v46[v303];
                int v305;
                v305 = v47[v303];
                bool v306;
                v306 = v305 < 3l;
                assert("Tensor range check" && 0 <= v298 && v298 < 1l);
                assert("Tensor range check" && 0 <= v300 && v300 < 4l);
                v297[v303] = v306;
                v300 += 1l ;
            }
            v298 += 1l ;
        }
        float v307[4l];
        int v308;
        v308 = 0l;
        while (while_method_1(v308)){
            int v310;
            v310 = 0l;
            while (while_method_2(v310)){
                assert("Tensor range check" && 0 <= v308 && v308 < 1l);
                assert("Tensor range check" && 0 <= v310 && v310 < 4l);
                int v312;
                v312 = 4l * v308;
                int v313;
                v313 = v312 + v310;
                float v314;
                v314 = v46[v313];
                bool v315;
                v315 = v297[v313];
                float v318;
                if (v315){
                    bool v316;
                    v316 = 0.0f >= v314;
                    if (v316){
                        v318 = 0.0f;
                    } else {
                        v318 = v314;
                    }
                } else {
                    v318 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v308 && v308 < 1l);
                assert("Tensor range check" && 0 <= v310 && v310 < 4l);
                v307[v313] = v318;
                v310 += 1l ;
            }
            v308 += 1l ;
        }
        float v319;
        v319 = 0.0f;
        int v320;
        v320 = 0l;
        while (while_method_1(v320)){
            int v322;
            v322 = 0l;
            while (while_method_2(v322)){
                assert("Tensor range check" && 0 <= v320 && v320 < 1l);
                assert("Tensor range check" && 0 <= v322 && v322 < 4l);
                int v324;
                v324 = 4l * v320;
                int v325;
                v325 = v324 + v322;
                float v326;
                v326 = v307[v325];
                float v327;
                v327 = v319 + v326;
                v319 = v327;
                v322 += 1l ;
            }
            v320 += 1l ;
        }
        auto v328 = cooperative_groups::coalesced_threads();
        int v329;
        v329 = threadIdx.x;
        auto v330 = cooperative_groups::labeled_partition(v328,v329);
        float v331;
        v331 = cooperative_groups::reduce(v330, v319, v119);
        int v332[4l];
        int v333;
        v333 = 0l;
        while (while_method_1(v333)){
            int v335;
            v335 = 0l;
            while (while_method_2(v335)){
                assert("Tensor range check" && 0 <= v333 && v333 < 1l);
                assert("Tensor range check" && 0 <= v335 && v335 < 4l);
                int v337;
                v337 = 4l * v333;
                int v338;
                v338 = v337 + v335;
                bool v339;
                v339 = v297[v338];
                int v340;
                if (v339){
                    v340 = 1l;
                } else {
                    v340 = 0l;
                }
                assert("Tensor range check" && 0 <= v333 && v333 < 1l);
                assert("Tensor range check" && 0 <= v335 && v335 < 4l);
                v332[v338] = v340;
                v335 += 1l ;
            }
            v333 += 1l ;
        }
        int v341;
        v341 = 0l;
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
                int v348;
                v348 = v332[v347];
                int v349;
                v349 = v341 + v348;
                v341 = v349;
                v344 += 1l ;
            }
            v342 += 1l ;
        }
        auto v350 = cooperative_groups::coalesced_threads();
        int v351;
        v351 = threadIdx.x;
        auto v352 = cooperative_groups::labeled_partition(v350,v351);
        int v353;
        v353 = cooperative_groups::reduce(v352, v341, v142);
        float v354;
        v354 = (float)v353;
        float v355;
        v355 = 1.0f / v354;
        float v356[4l];
        int v357;
        v357 = 0l;
        while (while_method_1(v357)){
            int v359;
            v359 = 0l;
            while (while_method_2(v359)){
                assert("Tensor range check" && 0 <= v357 && v357 < 1l);
                assert("Tensor range check" && 0 <= v359 && v359 < 4l);
                int v361;
                v361 = 4l * v357;
                int v362;
                v362 = v361 + v359;
                float v363;
                v363 = v307[v362];
                bool v364;
                v364 = v297[v362];
                bool v365;
                v365 = v364 == false;
                float v370;
                if (v365){
                    v370 = 0.0f;
                } else {
                    bool v366;
                    v366 = v331 == 0.0f;
                    bool v367;
                    v367 = v366 != true;
                    if (v367){
                        float v368;
                        v368 = v363 / v331;
                        v370 = v368;
                    } else {
                        v370 = v355;
                    }
                }
                assert("Tensor range check" && 0 <= v357 && v357 < 1l);
                assert("Tensor range check" && 0 <= v359 && v359 < 4l);
                v356[v362] = v370;
                v359 += 1l ;
            }
            v357 += 1l ;
        }
        float v371; int v372;
        Tuple3 tmp8 = Tuple3{0.0f, 2147483647l};
        v371 = tmp8.v0; v372 = tmp8.v1;
        int v373;
        v373 = 0l;
        while (while_method_1(v373)){
            int v375;
            v375 = 0l;
            while (while_method_2(v375)){
                assert("Tensor range check" && 0 <= v373 && v373 < 1l);
                assert("Tensor range check" && 0 <= v375 && v375 < 4l);
                int v377;
                v377 = 4l * v373;
                int v378;
                v378 = v377 + v375;
                float v379;
                v379 = v146[v378];
                int v380;
                v380 = v47[v378];
                bool v381;
                v381 = v372 == v293;
                float v385; int v386;
                if (v381){
                    v385 = v371; v386 = v372;
                } else {
                    bool v382;
                    v382 = v380 == v293;
                    if (v382){
                        v385 = v379; v386 = v380;
                    } else {
                        v385 = v371; v386 = v372;
                    }
                }
                v371 = v385;
                v372 = v386;
                v375 += 1l ;
            }
            v373 += 1l ;
        }
        auto v387 = cooperative_groups::coalesced_threads();
        int v388;
        v388 = threadIdx.x;
        auto v389 = cooperative_groups::labeled_partition(v387,v388);
        Closure6 v390{v293};
        float v391; int v392;
        Tuple3 tmp9 = cooperative_groups::reduce(v389, Tuple3{v371, v372}, v390);
        v391 = tmp9.v0; v392 = tmp9.v1;
        bool v393;
        v393 = v392 == 2147483647l;
        bool v394;
        v394 = v393 != true;
        bool v395;
        v395 = v394 == false;
        if (v395){
            assert("Expected a valid action id in get_action." && v394);
        } else {
        }
        float v397; int v398;
        Tuple3 tmp10 = Tuple3{0.0f, 2147483647l};
        v397 = tmp10.v0; v398 = tmp10.v1;
        int v399;
        v399 = 0l;
        while (while_method_1(v399)){
            int v401;
            v401 = 0l;
            while (while_method_2(v401)){
                assert("Tensor range check" && 0 <= v399 && v399 < 1l);
                assert("Tensor range check" && 0 <= v401 && v401 < 4l);
                int v403;
                v403 = 4l * v399;
                int v404;
                v404 = v403 + v401;
                float v405;
                v405 = v356[v404];
                int v406;
                v406 = v47[v404];
                bool v407;
                v407 = v398 == v293;
                float v411; int v412;
                if (v407){
                    v411 = v397; v412 = v398;
                } else {
                    bool v408;
                    v408 = v406 == v293;
                    if (v408){
                        v411 = v405; v412 = v406;
                    } else {
                        v411 = v397; v412 = v398;
                    }
                }
                v397 = v411;
                v398 = v412;
                v401 += 1l ;
            }
            v399 += 1l ;
        }
        auto v413 = cooperative_groups::coalesced_threads();
        int v414;
        v414 = threadIdx.x;
        auto v415 = cooperative_groups::labeled_partition(v413,v414);
        float v416; int v417;
        Tuple3 tmp11 = cooperative_groups::reduce(v415, Tuple3{v397, v398}, v390);
        v416 = tmp11.v0; v417 = tmp11.v1;
        bool v418;
        v418 = v417 == 2147483647l;
        bool v419;
        v419 = v418 != true;
        bool v420;
        v420 = v419 == false;
        if (v420){
            assert("Expected a valid action id in get_action." && v419);
        } else {
        }
        int v422;
        v422 = 0l;
        while (while_method_1(v422)){
            assert("Tensor range check" && 0 <= v422 && v422 < 1l);
            assert("Tensor range check" && 0 <= v422 && v422 < 1l);
            v422 += 1l ;
        }
        assert("Tensor range check" && 0 <= v39 && v39 < 32l);
        v17[v39] = v416;
        v18[v39] = v391;
        v19[v39] = v293;
        v28 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float v431; float v432; int v433;
    if (v21){
        assert("Tensor range check" && 0 <= v20 && v20 < 32l);
        float v424;
        v424 = v17[v20];
        float v425;
        v425 = v18[v20];
        int v426;
        v426 = v19[v20];
        v431 = v424; v432 = v425; v433 = v426;
    } else {
        Tuple0 v427[1l];
        float v428; float v429; int v430;
        Tuple0 tmp12 = v427[0l];
        v428 = tmp12.v0; v429 = tmp12.v1; v430 = tmp12.v2;
        v431 = v428; v432 = v429; v433 = v430;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return Tuple0{v431, v432, v433};
}
__device__ void push_0(float * v0, float * v1, float * v2, float * v3, float * v4, int * v5, float * v6, int * v7, int * v8, double * v9, double * v10, float * v11, float * v12, float * v13, int v14, int & v15, double * v16, double * v17, int v18, int v19, int v20){
    float v21; float v22; int v23;
    Tuple0 tmp13 = method_1(v0, v1, v2, v3, v4, v19, v20);
    v21 = tmp13.v0; v22 = tmp13.v1; v23 = tmp13.v2;
    int v24 = v15;
    int v25;
    v25 = v24 + 1l;
    v15 = v25;
    assert("Tensor range check" && 0 <= v20 && v20 < 4l);
    assert("Tensor range check" && 0 <= v24 && v24 < 16l);
    assert("Tensor range check" && 0 <= v14 && v14 < 32l);
    int v26;
    v26 = 32l * v24;
    int v27;
    v27 = v26 + v14;
    int v28;
    v28 = 512l * v20;
    int v29;
    v29 = v28 + v27;
    v5[v29] = v23;
    v6[v29] = v22;
    v7[v29] = v18;
    v8[v29] = v19;
    double v30;
    v30 = (double)v22;
    double v31;
    v31 = log(v30);
    double v32;
    v32 = (double)v21;
    double v33;
    v33 = log(v32);
    assert("Tensor range check" && 0 <= v18 && v18 < 2l);
    double v34;
    v34 = v16[v18];
    double v35;
    v35 = v17[v18];
    double v36;
    v36 = v33 + v34;
    double v37;
    v37 = v31 + v35;
    assert("Tensor range check" && 0 <= v18 && v18 < 2l);
    v16[v18] = v36;
    v17[v18] = v37;
    assert("Tensor range check" && 0 <= v20 && v20 < 4l);
    int v38;
    v38 = 1024l * v20;
    assert("Tensor range check" && 0 <= v24 && v24 < 16l);
    int v39;
    v39 = 64l * v24;
    int v40;
    v40 = v39 + v38;
    assert("Tensor range check" && 0 <= v14 && v14 < 32l);
    int v41;
    v41 = 2l * v14;
    int v42;
    v42 = v41 + v40;
    int v43;
    v43 = 0l;
    while (while_method_0(v43)){
        assert("Tensor range check" && 0 <= v43 && v43 < 2l);
        double v45;
        v45 = v16[v43];
        double v46;
        v46 = v17[v43];
        assert("Tensor range check" && 0 <= v43 && v43 < 2l);
        int v47;
        v47 = v43 + v42;
        v9[v47] = v45;
        v10[v47] = v46;
        v43 += 1l ;
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
    v4 = reinterpret_cast<float *>(&v1[262144ull]);
    float * v6;
    v6 = reinterpret_cast<float *>(&v1[524288ull]);
    float * v8;
    v8 = reinterpret_cast<float *>(&v1[786432ull]);
    float * v10;
    v10 = reinterpret_cast<float *>(&v1[1048576ull]);
    int * v12;
    v12 = reinterpret_cast<int *>(&v0[0ull]);
    float * v14;
    v14 = reinterpret_cast<float *>(&v0[8192ull]);
    int * v16;
    v16 = reinterpret_cast<int *>(&v0[16384ull]);
    int * v18;
    v18 = reinterpret_cast<int *>(&v0[24576ull]);
    double * v20;
    v20 = reinterpret_cast<double *>(&v0[32768ull]);
    double * v22;
    v22 = reinterpret_cast<double *>(&v0[65536ull]);
    float * v24;
    v24 = reinterpret_cast<float *>(&v0[98304ull]);
    float * v26;
    v26 = reinterpret_cast<float *>(&v0[131072ull]);
    float * v28;
    v28 = reinterpret_cast<float *>(&v0[139264ull]);
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
    v36 = 0l;
    int v37;
    v37 = 235l;
    int v38;
    v38 = 0l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v38, v37, v36);
    int v39;
    v39 = 0l;
    int v40;
    v40 = 212l;
    int v41;
    v41 = 1l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v41, v40, v39);
    int v42;
    v42 = 0l;
    int v43;
    v43 = 790l;
    int v44;
    v44 = 0l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v44, v43, v42);
    int v45;
    v45 = 0l;
    int v46;
    v46 = 343l;
    int v47;
    v47 = 1l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v47, v46, v45);
    int v48;
    v48 = 0l;
    int v49;
    v49 = 457l;
    int v50;
    v50 = 0l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v50, v49, v48);
    int v51;
    v51 = 0l;
    int v52;
    v52 = 3447l;
    int v53;
    v53 = 1l;
    push_0(v2, v4, v6, v8, v10, v12, v14, v16, v18, v20, v22, v24, v26, v28, v30, v31, v32, v33, v53, v52, v51);
    int v54 = v31;
    int v55;
    v55 = threadIdx.x;
    float v56[2l];
    v56[0l] = 13.0f;
    v56[1l] = -13.0f;
    int v57;
    v57 = v54;
    while (while_method_3(v57)){
        v57 -= 1l ;
        assert("Tensor range check" && 0 <= v57 && v57 < 16l);
        assert("Tensor range check" && 0 <= v55 && v55 < 32l);
        int v59;
        v59 = 32l * v57;
        int v60;
        v60 = v59 + v55;
        int v61;
        v61 = v12[v60];
        float v62;
        v62 = v14[v60];
        int v63;
        v63 = v16[v60];
        int v64;
        v64 = v18[v60];
        assert("Tensor range check" && 0 <= v63 && v63 < 2l);
        float v65;
        v65 = v56[v63];
        assert("Tensor range check" && 0 <= v64 && v64 < 4096l);
        int v66;
        v66 = 4l * v64;
        assert("Tensor range check" && 0 <= v57 && v57 < 16l);
        int v67;
        v67 = 128l * v57;
        assert("Tensor range check" && 0 <= v55 && v55 < 32l);
        int v68;
        v68 = 4l * v55;
        int v69;
        v69 = v68 + v67;
        assert("Tensor range check" && 0 <= v57 && v57 < 16l);
        int v70;
        v70 = 64l * v57;
        double * v71;
        v71 = v20+v70;
        double * v73;
        v73 = v22+v70;
        assert("Tensor range check" && 0 <= v55 && v55 < 32l);
        int v75;
        v75 = 2l * v55;
        double v76[2l];
        int v77;
        v77 = 0l;
        while (while_method_0(v77)){
            assert("Tensor range check" && 0 <= v77 && v77 < 2l);
            int v79;
            v79 = v77 + v75;
            double v80;
            v80 = v71[v79];
            bool v81;
            v81 = v63 == v77;
            double v82;
            if (v81){
                v82 = 0.0;
            } else {
                v82 = v80;
            }
            assert("Tensor range check" && 0 <= v77 && v77 < 2l);
            v76[v77] = v82;
            v77 += 1l ;
        }
        double v83;
        v83 = 0.0;
        int v84;
        v84 = 0l;
        while (while_method_0(v84)){
            assert("Tensor range check" && 0 <= v84 && v84 < 2l);
            double v86;
            v86 = v76[v84];
            double v87;
            v87 = v83 + v86;
            v83 = v87;
            v84 += 1l ;
        }
        double v88;
        v88 = 0.0;
        int v89;
        v89 = 0l;
        while (while_method_0(v89)){
            assert("Tensor range check" && 0 <= v89 && v89 < 2l);
            int v91;
            v91 = v89 + v75;
            double v92;
            v92 = v73[v91];
            double v93;
            v93 = v88 + v92;
            v88 = v93;
            v89 += 1l ;
        }
        double v94;
        v94 = v83 - v88;
        double v95;
        v95 = exp(v94);
        float v96;
        v96 = (float)v95;
        float v97;
        v97 = v65 * v96;
        assert("Tensor range check" && 0 <= v57 && v57 < 16l);
        assert("Tensor range check" && 0 <= v55 && v55 < 32l);
        v26[v60] = v97;
        v28[v60] = v96;
        float * v98;
        v98 = v4+v66;
        float * v100;
        v100 = v8+v66;
        float * v102;
        v102 = v10+v66;
        float * v104;
        v104 = v24+v69;
        __shared__ float v106[32l];
        __shared__ int v107[32l];
        __shared__ float v108[32l];
        __shared__ int v109[32l];
        __shared__ double * v110[32l];
        __shared__ double * v111[32l];
        __shared__ float * v112[32l];
        __shared__ float * v113[32l];
        __shared__ float * v114[32l];
        __shared__ float * v115[32l];
        /* void shared array create v116 */;
        __shared__ float v117[32l];
        int v118;
        v118 = threadIdx.x;
        bool v119;
        v119 = v118 < 32l;
        if (v119){
            assert("Tensor range check" && 0 <= v118 && v118 < 32l);
            v106[v118] = v62;
            v107[v118] = v61;
            v108[v118] = v65;
            v109[v118] = v63;
            v110[v118] = v71;
            v111[v118] = v73;
            v112[v118] = v98;
            v113[v118] = v100;
            v114[v118] = v102;
            v115[v118] = v104;
            /* void array set */;
        } else {
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        bool v120;
        v120 = 0l <= v118;
        bool v121;
        v121 = v120 == false;
        if (v121){
            assert("The index needs to be zero or positive." && v120);
        } else {
        }
        int v123;
        v123 = v118 % 1l;
        bool v124;
        v124 = v119 == false;
        if (v124){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v119);
        } else {
        }
        assert("Tensor range check" && 0 <= v118 && v118 < 32l);
        int v126;
        v126 = 0l;
        while (while_method_1(v126)){
            bool v128;
            v128 = v120 && v119;
            bool v129;
            v129 = v128 == false;
            if (v129){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v128);
            } else {
            }
            bool v131;
            v131 = 0l <= v126;
            bool v133;
            if (v131){
                bool v132;
                v132 = v126 < 1l;
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
            v136 = v126 * 32l;
            int v137;
            v137 = v136 + v118;
            assert("Tensor range check" && 0 <= v126 && v126 < 1l);
            int v138;
            v138 = 32l * v126;
            int v139;
            v139 = v138 + v118;
            float v140;
            v140 = v106[v139];
            int v141;
            v141 = v107[v139];
            float v142;
            v142 = v108[v139];
            int v143;
            v143 = v109[v139];
            double * v144;
            v144 = v110[v139];
            double * v145;
            v145 = v111[v139];
            float * v146;
            v146 = v112[v139];
            float * v147;
            v147 = v113[v139];
            float * v148;
            v148 = v114[v139];
            float * v149;
            v149 = v115[v139];
            /* void array index */;
            assert("Tensor range check" && 0 <= v123 && v123 < 1l);
            int v150;
            v150 = 4l * v123;
            float v151[4l];
            float v152[4l];
            float v153[4l];
            int v154[4l];
            int v155;
            v155 = 0l;
            while (while_method_1(v155)){
                assert("Tensor range check" && 0 <= v155 && v155 < 1l);
                int v157;
                v157 = 4l * v155;
                assert("Tensor range check" && 0 <= v155 && v155 < 1l);
                int v158;
                v158 = v157 + v150;
                int4* v159;
                v159 = reinterpret_cast<int4*>(v146 + v158);
                int4* v160;
                v160 = reinterpret_cast<int4*>(v151 + v157);
                assert("Pointer alignment check" && (unsigned long long)(v159) % 4l == 0 && (unsigned long long)(v160) % 4l == 0);
                *v160 = *v159;
                int4* v161;
                v161 = reinterpret_cast<int4*>(v147 + v158);
                int4* v162;
                v162 = reinterpret_cast<int4*>(v152 + v157);
                assert("Pointer alignment check" && (unsigned long long)(v161) % 4l == 0 && (unsigned long long)(v162) % 4l == 0);
                *v162 = *v161;
                int4* v163;
                v163 = reinterpret_cast<int4*>(v148 + v158);
                int4* v164;
                v164 = reinterpret_cast<int4*>(v153 + v157);
                assert("Pointer alignment check" && (unsigned long long)(v163) % 4l == 0 && (unsigned long long)(v164) % 4l == 0);
                *v164 = *v163;
                v155 += 1l ;
            }
            int v165;
            v165 = 0l;
            while (while_method_1(v165)){
                int v167;
                v167 = 0l;
                while (while_method_2(v167)){
                    bool v169;
                    v169 = 0l <= v167;
                    bool v171;
                    if (v169){
                        bool v170;
                        v170 = v167 < 4l;
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
                    bool v174;
                    v174 = 0l <= v123;
                    bool v176;
                    if (v174){
                        bool v175;
                        v175 = v123 < 1l;
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
                    int v179;
                    v179 = v123 * 4l;
                    int v180;
                    v180 = v167 + v179;
                    bool v181;
                    v181 = 0l <= v165;
                    bool v183;
                    if (v181){
                        bool v182;
                        v182 = v165 < 1l;
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
                    v186 = v165 * 4l;
                    int v187;
                    v187 = v180 + v186;
                    assert("Tensor range check" && 0 <= v165 && v165 < 1l);
                    assert("Tensor range check" && 0 <= v167 && v167 < 4l);
                    int v188;
                    v188 = 4l * v165;
                    int v189;
                    v189 = v188 + v167;
                    v154[v189] = v187;
                    v167 += 1l ;
                }
                v165 += 1l ;
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
                    v197 = v152[v196];
                    float v198;
                    v198 = v153[v196];
                    bool v199;
                    v199 = v198 == 0.0f;
                    bool v200;
                    v200 = v199 != true;
                    float v202;
                    if (v200){
                        float v201;
                        v201 = v197 / v198;
                        v202 = v201;
                    } else {
                        v202 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                    assert("Tensor range check" && 0 <= v193 && v193 < 4l);
                    v190[v196] = v202;
                    v193 += 1l ;
                }
                v191 += 1l ;
            }
            bool v203[4l];
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
                    v210 = v151[v209];
                    int v211;
                    v211 = v154[v209];
                    bool v212;
                    v212 = v211 < 3l;
                    assert("Tensor range check" && 0 <= v204 && v204 < 1l);
                    assert("Tensor range check" && 0 <= v206 && v206 < 4l);
                    v203[v209] = v212;
                    v206 += 1l ;
                }
                v204 += 1l ;
            }
            float v213[4l];
            int v214;
            v214 = 0l;
            while (while_method_1(v214)){
                int v216;
                v216 = 0l;
                while (while_method_2(v216)){
                    assert("Tensor range check" && 0 <= v214 && v214 < 1l);
                    assert("Tensor range check" && 0 <= v216 && v216 < 4l);
                    int v218;
                    v218 = 4l * v214;
                    int v219;
                    v219 = v218 + v216;
                    float v220;
                    v220 = v151[v219];
                    bool v221;
                    v221 = v203[v219];
                    float v224;
                    if (v221){
                        bool v222;
                        v222 = 0.0f >= v220;
                        if (v222){
                            v224 = 0.0f;
                        } else {
                            v224 = v220;
                        }
                    } else {
                        v224 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v214 && v214 < 1l);
                    assert("Tensor range check" && 0 <= v216 && v216 < 4l);
                    v213[v219] = v224;
                    v216 += 1l ;
                }
                v214 += 1l ;
            }
            float v225;
            v225 = 0.0f;
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
                    float v232;
                    v232 = v213[v231];
                    float v233;
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
            Closure0 v237{};
            float v238;
            v238 = cooperative_groups::reduce(v236, v225, v237);
            int v239[4l];
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
                    bool v246;
                    v246 = v203[v245];
                    int v247;
                    if (v246){
                        v247 = 1l;
                    } else {
                        v247 = 0l;
                    }
                    assert("Tensor range check" && 0 <= v240 && v240 < 1l);
                    assert("Tensor range check" && 0 <= v242 && v242 < 4l);
                    v239[v245] = v247;
                    v242 += 1l ;
                }
                v240 += 1l ;
            }
            int v248;
            v248 = 0l;
            int v249;
            v249 = 0l;
            while (while_method_1(v249)){
                int v251;
                v251 = 0l;
                while (while_method_2(v251)){
                    assert("Tensor range check" && 0 <= v249 && v249 < 1l);
                    assert("Tensor range check" && 0 <= v251 && v251 < 4l);
                    int v253;
                    v253 = 4l * v249;
                    int v254;
                    v254 = v253 + v251;
                    int v255;
                    v255 = v239[v254];
                    int v256;
                    v256 = v248 + v255;
                    v248 = v256;
                    v251 += 1l ;
                }
                v249 += 1l ;
            }
            auto v257 = cooperative_groups::coalesced_threads();
            int v258;
            v258 = threadIdx.x;
            auto v259 = cooperative_groups::labeled_partition(v257,v258);
            Closure1 v260{};
            int v261;
            v261 = cooperative_groups::reduce(v259, v248, v260);
            float v262;
            v262 = (float)v261;
            float v263;
            v263 = 1.0f / v262;
            float v264[4l];
            int v265;
            v265 = 0l;
            while (while_method_1(v265)){
                int v267;
                v267 = 0l;
                while (while_method_2(v267)){
                    assert("Tensor range check" && 0 <= v265 && v265 < 1l);
                    assert("Tensor range check" && 0 <= v267 && v267 < 4l);
                    int v269;
                    v269 = 4l * v265;
                    int v270;
                    v270 = v269 + v267;
                    float v271;
                    v271 = v213[v270];
                    bool v272;
                    v272 = v203[v270];
                    bool v273;
                    v273 = v272 == false;
                    float v278;
                    if (v273){
                        v278 = 0.0f;
                    } else {
                        bool v274;
                        v274 = v238 == 0.0f;
                        bool v275;
                        v275 = v274 != true;
                        if (v275){
                            float v276;
                            v276 = v271 / v238;
                            v278 = v276;
                        } else {
                            v278 = v263;
                        }
                    }
                    assert("Tensor range check" && 0 <= v265 && v265 < 1l);
                    assert("Tensor range check" && 0 <= v267 && v267 < 4l);
                    v264[v270] = v278;
                    v267 += 1l ;
                }
                v265 += 1l ;
            }
            float v279[4l];
            int v280;
            v280 = 0l;
            while (while_method_1(v280)){
                int v282;
                v282 = 0l;
                while (while_method_2(v282)){
                    assert("Tensor range check" && 0 <= v280 && v280 < 1l);
                    assert("Tensor range check" && 0 <= v282 && v282 < 4l);
                    int v284;
                    v284 = 4l * v280;
                    int v285;
                    v285 = v284 + v282;
                    float v286;
                    v286 = v190[v285];
                    int v287;
                    v287 = v154[v285];
                    bool v288;
                    v288 = v141 == v287;
                    float v291;
                    if (v288){
                        float v289;
                        v289 = v142 - v286;
                        float v290;
                        v290 = v289 / v140;
                        v291 = v290;
                    } else {
                        v291 = 0.0f;
                    }
                    float v292;
                    v292 = v291 + v286;
                    assert("Tensor range check" && 0 <= v280 && v280 < 1l);
                    assert("Tensor range check" && 0 <= v282 && v282 < 4l);
                    v279[v285] = v292;
                    v282 += 1l ;
                }
                v280 += 1l ;
            }
            float v293[4l];
            int v294;
            v294 = 0l;
            while (while_method_1(v294)){
                int v296;
                v296 = 0l;
                while (while_method_2(v296)){
                    assert("Tensor range check" && 0 <= v294 && v294 < 1l);
                    assert("Tensor range check" && 0 <= v296 && v296 < 4l);
                    int v298;
                    v298 = 4l * v294;
                    int v299;
                    v299 = v298 + v296;
                    float v300;
                    v300 = v264[v299];
                    float v301;
                    v301 = v279[v299];
                    float v302;
                    v302 = v300 * v301;
                    assert("Tensor range check" && 0 <= v294 && v294 < 1l);
                    assert("Tensor range check" && 0 <= v296 && v296 < 4l);
                    v293[v299] = v302;
                    v296 += 1l ;
                }
                v294 += 1l ;
            }
            float v303;
            v303 = 0.0f;
            int v304;
            v304 = 0l;
            while (while_method_1(v304)){
                int v306;
                v306 = 0l;
                while (while_method_2(v306)){
                    assert("Tensor range check" && 0 <= v304 && v304 < 1l);
                    assert("Tensor range check" && 0 <= v306 && v306 < 4l);
                    int v308;
                    v308 = 4l * v304;
                    int v309;
                    v309 = v308 + v306;
                    float v310;
                    v310 = v293[v309];
                    float v311;
                    v311 = v303 + v310;
                    v303 = v311;
                    v306 += 1l ;
                }
                v304 += 1l ;
            }
            auto v312 = cooperative_groups::coalesced_threads();
            int v313;
            v313 = threadIdx.x;
            auto v314 = cooperative_groups::labeled_partition(v312,v313);
            float v315;
            v315 = cooperative_groups::reduce(v314, v303, v237);
            assert("Tensor range check" && 0 <= v137 && v137 < 32l);
            int v316;
            v316 = 2l * v137;
            double v317[2l];
            int v318;
            v318 = 0l;
            while (while_method_0(v318)){
                assert("Tensor range check" && 0 <= v318 && v318 < 2l);
                int v320;
                v320 = v318 + v316;
                double v321;
                v321 = v144[v320];
                bool v322;
                v322 = v143 == v318;
                double v323;
                if (v322){
                    v323 = 0.0;
                } else {
                    v323 = v321;
                }
                assert("Tensor range check" && 0 <= v318 && v318 < 2l);
                v317[v318] = v323;
                v318 += 1l ;
            }
            double v324;
            v324 = 0.0;
            int v325;
            v325 = 0l;
            while (while_method_0(v325)){
                assert("Tensor range check" && 0 <= v325 && v325 < 2l);
                double v327;
                v327 = v317[v325];
                double v328;
                v328 = v324 + v327;
                v324 = v328;
                v325 += 1l ;
            }
            double v329;
            v329 = 0.0;
            int v330;
            v330 = 0l;
            while (while_method_0(v330)){
                assert("Tensor range check" && 0 <= v330 && v330 < 2l);
                int v332;
                v332 = v330 + v316;
                double v333;
                v333 = v145[v332];
                double v334;
                v334 = v329 + v333;
                v329 = v334;
                v330 += 1l ;
            }
            double v335;
            v335 = v324 - v329;
            double v336;
            v336 = exp(v335);
            float v337;
            v337 = (float)v336;
            float v338[4l];
            int v339;
            v339 = 0l;
            while (while_method_1(v339)){
                int v341;
                v341 = 0l;
                while (while_method_2(v341)){
                    assert("Tensor range check" && 0 <= v339 && v339 < 1l);
                    assert("Tensor range check" && 0 <= v341 && v341 < 4l);
                    int v343;
                    v343 = 4l * v339;
                    int v344;
                    v344 = v343 + v341;
                    float v345;
                    v345 = v279[v344];
                    float v346;
                    v346 = v345 - v315;
                    float v347;
                    v347 = v337 * v346;
                    assert("Tensor range check" && 0 <= v339 && v339 < 1l);
                    assert("Tensor range check" && 0 <= v341 && v341 < 4l);
                    v338[v344] = v347;
                    v341 += 1l ;
                }
                v339 += 1l ;
            }
            int v348;
            v348 = 0l;
            while (while_method_1(v348)){
                assert("Tensor range check" && 0 <= v348 && v348 < 1l);
                int v350;
                v350 = 4l * v348;
                int v351;
                v351 = v350 + v150;
                assert("Tensor range check" && 0 <= v348 && v348 < 1l);
                int4* v352;
                v352 = reinterpret_cast<int4*>(v338 + v350);
                int4* v353;
                v353 = reinterpret_cast<int4*>(v149 + v351);
                assert("Pointer alignment check" && (unsigned long long)(v352) % 4l == 0 && (unsigned long long)(v353) % 4l == 0);
                *v353 = *v352;
                v348 += 1l ;
            }
            assert("Tensor range check" && 0 <= v137 && v137 < 32l);
            v117[v137] = v315;
            v126 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        float v357;
        if (v119){
            assert("Tensor range check" && 0 <= v118 && v118 < 32l);
            float v354;
            v354 = v117[v118];
            v357 = v354;
        } else {
            float v355[1l];
            float v356;
            v356 = v355[0l];
            v357 = v356;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        assert("Tensor range check" && 0 <= v63 && v63 < 2l);
        v56[v63] = v357;
        cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v358 = console_lock;
        auto v359 = cooperative_groups::coalesced_threads();
        v358.acquire();
        int v360;
        v360 = 0l;
        printf("{%s = %c","rewards", '[');
        int v361;
        v361 = 0l;
        while (while_method_0(v361)){
            int v363;
            v363 = v360;
            bool v364;
            v364 = v363 >= 100l;
            if (v364){
                printf("%s"," ...");
                break;
            } else {
            }
            bool v365;
            v365 = v361 == 0l;
            bool v366;
            v366 = v365 != true;
            if (v366){
                printf("%s","; ");
            } else {
            }
            int v367;
            v367 = v360 + 1l;
            v360 = v367;
            float v368;
            v368 = v56[v361];
            printf("%f",v368);
            v361 += 1l ;
        }
        printf("%c",']');
        printf("}\n");
        v358.release();
        v359.sync() ;
    }
    int v387 = v31;
    int v388;
    v388 = threadIdx.x;
    int v389;
    v389 = v387;
    while (while_method_3(v389)){
        v389 -= 1l ;
        assert("Tensor range check" && 0 <= v389 && v389 < 16l);
        assert("Tensor range check" && 0 <= v388 && v388 < 32l);
        int v391;
        v391 = 32l * v389;
        int v392;
        v392 = v391 + v388;
        int v393;
        v393 = v12[v392];
        float v394;
        v394 = v14[v392];
        int v395;
        v395 = v16[v392];
        int v396;
        v396 = v18[v392];
        assert("Tensor range check" && 0 <= v389 && v389 < 16l);
        assert("Tensor range check" && 0 <= v388 && v388 < 32l);
        float v397;
        v397 = v26[v392];
        float v398;
        v398 = v28[v392];
        assert("Tensor range check" && 0 <= v396 && v396 < 4096l);
        int v399;
        v399 = 4l * v396;
        float * v400;
        v400 = v2+v399;
        float * v402;
        v402 = v4+v399;
        float * v404;
        v404 = v6+v399;
        float * v406;
        v406 = v8+v399;
        float * v408;
        v408 = v10+v399;
        assert("Tensor range check" && 0 <= v389 && v389 < 16l);
        int v410;
        v410 = 128l * v389;
        assert("Tensor range check" && 0 <= v388 && v388 < 32l);
        int v411;
        v411 = 4l * v388;
        int v412;
        v412 = v411 + v410;
        assert("Tensor range check" && 0 <= v393 && v393 < 4l);
        float * v413;
        v413 = v406+v393;
        float * v415;
        v415 = v408+v393;
        float v417;
        v417 = atomicAdd(v413,v397);
        float v418;
        v418 = atomicAdd(v415,v398);
        float * v419;
        v419 = v24+v412;
        __shared__ float * v421[32l];
        __shared__ float * v422[32l];
        /* void shared array create v423 */;
        /* void shared array create v424 */;
        int v425;
        v425 = threadIdx.x;
        bool v426;
        v426 = v425 < 32l;
        if (v426){
            assert("Tensor range check" && 0 <= v425 && v425 < 32l);
            v421[v425] = v404;
            v422[v425] = v419;
            /* void array set */;
        } else {
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        bool v427;
        v427 = 0l <= v425;
        bool v428;
        v428 = v427 == false;
        if (v428){
            assert("The index needs to be zero or positive." && v427);
        } else {
        }
        int v430;
        v430 = v425 % 1l;
        bool v431;
        v431 = v426 == false;
        if (v431){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v426);
        } else {
        }
        assert("Tensor range check" && 0 <= v425 && v425 < 32l);
        int v433;
        v433 = 0l;
        while (while_method_1(v433)){
            bool v435;
            v435 = v427 && v426;
            bool v436;
            v436 = v435 == false;
            if (v436){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v435);
            } else {
            }
            bool v438;
            v438 = 0l <= v433;
            bool v440;
            if (v438){
                bool v439;
                v439 = v433 < 1l;
                v440 = v439;
            } else {
                v440 = false;
            }
            bool v441;
            v441 = v440 == false;
            if (v441){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v440);
            } else {
            }
            int v443;
            v443 = v433 * 32l;
            int v444;
            v444 = v443 + v425;
            assert("Tensor range check" && 0 <= v433 && v433 < 1l);
            int v445;
            v445 = 32l * v433;
            int v446;
            v446 = v445 + v425;
            float * v447;
            v447 = v421[v446];
            float * v448;
            v448 = v422[v446];
            /* void array index */;
            assert("Tensor range check" && 0 <= v430 && v430 < 1l);
            int v449;
            v449 = 4l * v430;
            float v450[4l];
            int v451[4l];
            int v452;
            v452 = 0l;
            while (while_method_1(v452)){
                assert("Tensor range check" && 0 <= v452 && v452 < 1l);
                int v454;
                v454 = 4l * v452;
                assert("Tensor range check" && 0 <= v452 && v452 < 1l);
                int v455;
                v455 = v454 + v449;
                int4* v456;
                v456 = reinterpret_cast<int4*>(v448 + v455);
                int4* v457;
                v457 = reinterpret_cast<int4*>(v450 + v454);
                assert("Pointer alignment check" && (unsigned long long)(v456) % 4l == 0 && (unsigned long long)(v457) % 4l == 0);
                *v457 = *v456;
                v452 += 1l ;
            }
            int v458;
            v458 = 0l;
            while (while_method_1(v458)){
                int v460;
                v460 = 0l;
                while (while_method_2(v460)){
                    bool v462;
                    v462 = 0l <= v460;
                    bool v464;
                    if (v462){
                        bool v463;
                        v463 = v460 < 4l;
                        v464 = v463;
                    } else {
                        v464 = false;
                    }
                    bool v465;
                    v465 = v464 == false;
                    if (v465){
                        assert("The indices should be inside the range of the dimension." && v464);
                    } else {
                    }
                    bool v467;
                    v467 = 0l <= v430;
                    bool v469;
                    if (v467){
                        bool v468;
                        v468 = v430 < 1l;
                        v469 = v468;
                    } else {
                        v469 = false;
                    }
                    bool v470;
                    v470 = v469 == false;
                    if (v470){
                        assert("The indices should be inside the range of the dimension." && v469);
                    } else {
                    }
                    int v472;
                    v472 = v430 * 4l;
                    int v473;
                    v473 = v460 + v472;
                    bool v474;
                    v474 = 0l <= v458;
                    bool v476;
                    if (v474){
                        bool v475;
                        v475 = v458 < 1l;
                        v476 = v475;
                    } else {
                        v476 = false;
                    }
                    bool v477;
                    v477 = v476 == false;
                    if (v477){
                        assert("The indices should be inside the range of the dimension." && v476);
                    } else {
                    }
                    int v479;
                    v479 = v458 * 4l;
                    int v480;
                    v480 = v473 + v479;
                    assert("Tensor range check" && 0 <= v458 && v458 < 1l);
                    assert("Tensor range check" && 0 <= v460 && v460 < 4l);
                    int v481;
                    v481 = 4l * v458;
                    int v482;
                    v482 = v481 + v460;
                    v451[v482] = v480;
                    v460 += 1l ;
                }
                v458 += 1l ;
            }
            int v483;
            v483 = 0l;
            while (while_method_1(v483)){
                int v485;
                v485 = 0l;
                while (while_method_2(v485)){
                    assert("Tensor range check" && 0 <= v483 && v483 < 1l);
                    assert("Tensor range check" && 0 <= v485 && v485 < 4l);
                    int v487;
                    v487 = 4l * v483;
                    int v488;
                    v488 = v487 + v485;
                    float v489;
                    v489 = v450[v488];
                    int v490;
                    v490 = v451[v488];
                    assert("Tensor range check" && 0 <= v490 && v490 < 4l);
                    float * v491;
                    v491 = v447+v490;
                    float v493;
                    v493 = atomicAdd(v491,v489);
                    v485 += 1l ;
                }
                v483 += 1l ;
            }
            int v494;
            v494 = 0l;
            while (while_method_1(v494)){
                assert("Tensor range check" && 0 <= v494 && v494 < 1l);
                assert("Tensor range check" && 0 <= v494 && v494 < 1l);
                v494 += 1l ;
            }
            assert("Tensor range check" && 0 <= v444 && v444 < 32l);
            /* void array set */;
            v433 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        if (v426){
            assert("Tensor range check" && 0 <= v425 && v425 < 32l);
            /* void array index */;
        } else {
            /* void array create */
            /* void array index */;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
    }
    int v497;
    v497 = threadIdx.x;
    bool v498;
    v498 = 0l <= v497;
    bool v499;
    v499 = v498 == false;
    if (v499){
        assert("The index needs to be zero or positive." && v498);
    } else {
    }
    int v501;
    v501 = v497 % 1l;
    int v502;
    v502 = v497 % 32l;
    int v503;
    v503 = v497 / 32l;
    bool v504;
    v504 = v503 < 1l;
    bool v505;
    v505 = v504 == false;
    if (v505){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v504);
    } else {
    }
    assert("Tensor range check" && 0 <= v503 && v503 < 1l);
    assert("Tensor range check" && 0 <= v502 && v502 < 32l);
    assert("Tensor range check" && 0 <= v501 && v501 < 1l);
    int v507;
    v507 = 4l * v501;
    int v508;
    v508 = 4l * v502;
    int v509;
    v509 = v508 + v507;
    int v510;
    v510 = 16384l * v503;
    int v511;
    v511 = v510 + v509;
    assert("Tensor range check" && 0 <= v503 && v503 < 1l);
    assert("Tensor range check" && 0 <= v502 && v502 < 32l);
    assert("Tensor range check" && 0 <= v501 && v501 < 1l);
    int v512;
    v512 = 0l;
    while (while_method_2(v512)){
        int v514;
        v514 = 0l;
        while (while_method_4(v514)){
            assert("Tensor range check" && 0 <= v512 && v512 < 4l);
            assert("Tensor range check" && 0 <= v514 && v514 < 128l);
            int v516;
            v516 = 128l * v514;
            int v517;
            v517 = v516 + v511;
            int v518;
            v518 = 16384l * v512;
            int v519;
            v519 = v518 + v517;
            float v520[4l];
            float v521[4l];
            float v522[4l];
            float v523[4l];
            float v524[4l];
            int v525[4l];
            int v526;
            v526 = 0l;
            while (while_method_1(v526)){
                assert("Tensor range check" && 0 <= v526 && v526 < 1l);
                int v528;
                v528 = 4l * v526;
                assert("Tensor range check" && 0 <= v526 && v526 < 1l);
                int v529;
                v529 = v528 + v519;
                int4* v530;
                v530 = reinterpret_cast<int4*>(v2 + v529);
                int4* v531;
                v531 = reinterpret_cast<int4*>(v520 + v528);
                assert("Pointer alignment check" && (unsigned long long)(v530) % 4l == 0 && (unsigned long long)(v531) % 4l == 0);
                *v531 = *v530;
                int4* v532;
                v532 = reinterpret_cast<int4*>(v4 + v529);
                int4* v533;
                v533 = reinterpret_cast<int4*>(v521 + v528);
                assert("Pointer alignment check" && (unsigned long long)(v532) % 4l == 0 && (unsigned long long)(v533) % 4l == 0);
                *v533 = *v532;
                int4* v534;
                v534 = reinterpret_cast<int4*>(v6 + v529);
                int4* v535;
                v535 = reinterpret_cast<int4*>(v522 + v528);
                assert("Pointer alignment check" && (unsigned long long)(v534) % 4l == 0 && (unsigned long long)(v535) % 4l == 0);
                *v535 = *v534;
                int4* v536;
                v536 = reinterpret_cast<int4*>(v8 + v529);
                int4* v537;
                v537 = reinterpret_cast<int4*>(v523 + v528);
                assert("Pointer alignment check" && (unsigned long long)(v536) % 4l == 0 && (unsigned long long)(v537) % 4l == 0);
                *v537 = *v536;
                int4* v538;
                v538 = reinterpret_cast<int4*>(v10 + v529);
                int4* v539;
                v539 = reinterpret_cast<int4*>(v524 + v528);
                assert("Pointer alignment check" && (unsigned long long)(v538) % 4l == 0 && (unsigned long long)(v539) % 4l == 0);
                *v539 = *v538;
                v526 += 1l ;
            }
            int v540;
            v540 = 0l;
            while (while_method_1(v540)){
                int v542;
                v542 = 0l;
                while (while_method_2(v542)){
                    bool v544;
                    v544 = 0l <= v542;
                    bool v546;
                    if (v544){
                        bool v545;
                        v545 = v542 < 4l;
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
                    bool v549;
                    v549 = 0l <= v501;
                    bool v551;
                    if (v549){
                        bool v550;
                        v550 = v501 < 1l;
                        v551 = v550;
                    } else {
                        v551 = false;
                    }
                    bool v552;
                    v552 = v551 == false;
                    if (v552){
                        assert("The indices should be inside the range of the dimension." && v551);
                    } else {
                    }
                    int v554;
                    v554 = v501 * 4l;
                    int v555;
                    v555 = v542 + v554;
                    bool v556;
                    v556 = 0l <= v540;
                    bool v558;
                    if (v556){
                        bool v557;
                        v557 = v540 < 1l;
                        v558 = v557;
                    } else {
                        v558 = false;
                    }
                    bool v559;
                    v559 = v558 == false;
                    if (v559){
                        assert("The indices should be inside the range of the dimension." && v558);
                    } else {
                    }
                    int v561;
                    v561 = v540 * 4l;
                    int v562;
                    v562 = v555 + v561;
                    assert("Tensor range check" && 0 <= v540 && v540 < 1l);
                    assert("Tensor range check" && 0 <= v542 && v542 < 4l);
                    int v563;
                    v563 = 4l * v540;
                    int v564;
                    v564 = v563 + v542;
                    v525[v564] = v562;
                    v542 += 1l ;
                }
                v540 += 1l ;
            }
            bool v565;
            v565 = 0l <= v503;
            bool v566;
            v566 = v565 && v504;
            bool v567;
            v567 = v566 == false;
            if (v567){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v566);
            } else {
            }
            bool v569;
            v569 = 0l <= v502;
            bool v571;
            if (v569){
                bool v570;
                v570 = v502 < 32l;
                v571 = v570;
            } else {
                v571 = false;
            }
            bool v572;
            v572 = v571 == false;
            if (v572){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v571);
            } else {
            }
            bool v574;
            v574 = 0l <= v512;
            bool v576;
            if (v574){
                bool v575;
                v575 = v512 < 4l;
                v576 = v575;
            } else {
                v576 = false;
            }
            bool v577;
            v577 = v576 == false;
            if (v577){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v576);
            } else {
            }
            bool v579;
            v579 = 0l <= v514;
            bool v581;
            if (v579){
                bool v580;
                v580 = v514 < 128l;
                v581 = v580;
            } else {
                v581 = false;
            }
            bool v582;
            v582 = v581 == false;
            if (v582){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v581);
            } else {
            }
            int v584;
            v584 = v514 * 32l;
            int v585;
            v585 = v512 + v503;
            int v586;
            v586 = v584 + v502;
            bool v587[4l];
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
                    v594 = v522[v593];
                    bool v595;
                    v595 = v594 == 0.0f;
                    bool v596;
                    v596 = v595 != true;
                    assert("Tensor range check" && 0 <= v588 && v588 < 1l);
                    assert("Tensor range check" && 0 <= v590 && v590 < 4l);
                    v587[v593] = v596;
                    v590 += 1l ;
                }
                v588 += 1l ;
            }
            bool v597;
            v597 = false;
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
                    bool v604;
                    v604 = v587[v603];
                    bool v605;
                    v605 = v597 || v604;
                    v597 = v605;
                    v600 += 1l ;
                }
                v598 += 1l ;
            }
            auto v606 = cooperative_groups::coalesced_threads();
            int v607;
            v607 = threadIdx.x;
            auto v608 = cooperative_groups::labeled_partition(v606,v607);
            Closure7 v609{};
            bool v610;
            v610 = cooperative_groups::reduce(v608, v597, v609);
            if (v610){
                float v611[4l];
                int v612;
                v612 = 0l;
                while (while_method_1(v612)){
                    int v614;
                    v614 = 0l;
                    while (while_method_2(v614)){
                        assert("Tensor range check" && 0 <= v612 && v612 < 1l);
                        assert("Tensor range check" && 0 <= v614 && v614 < 4l);
                        int v616;
                        v616 = 4l * v612;
                        int v617;
                        v617 = v616 + v614;
                        float v618;
                        v618 = v521[v617];
                        float v619;
                        v619 = v522[v617];
                        float v620;
                        v620 = v618 + v619;
                        bool v621;
                        v621 = 0.0f >= v620;
                        float v622;
                        if (v621){
                            v622 = 0.0f;
                        } else {
                            v622 = v620;
                        }
                        assert("Tensor range check" && 0 <= v612 && v612 < 1l);
                        assert("Tensor range check" && 0 <= v614 && v614 < 4l);
                        v611[v617] = v622;
                        v614 += 1l ;
                    }
                    v612 += 1l ;
                }
                float v623[4l];
                int v624;
                v624 = 0l;
                while (while_method_1(v624)){
                    int v626;
                    v626 = 0l;
                    while (while_method_2(v626)){
                        assert("Tensor range check" && 0 <= v624 && v624 < 1l);
                        assert("Tensor range check" && 0 <= v626 && v626 < 4l);
                        int v628;
                        v628 = 4l * v624;
                        int v629;
                        v629 = v628 + v626;
                        float v630;
                        v630 = v611[v629];
                        bool v631;
                        v631 = 0.0f >= v630;
                        float v632;
                        if (v631){
                            v632 = 0.0f;
                        } else {
                            v632 = v630;
                        }
                        assert("Tensor range check" && 0 <= v624 && v624 < 1l);
                        assert("Tensor range check" && 0 <= v626 && v626 < 4l);
                        v623[v629] = v632;
                        v626 += 1l ;
                    }
                    v624 += 1l ;
                }
                float v633;
                v633 = 0.0f;
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
                        v640 = v623[v639];
                        float v641;
                        v641 = v633 + v640;
                        v633 = v641;
                        v636 += 1l ;
                    }
                    v634 += 1l ;
                }
                auto v642 = cooperative_groups::coalesced_threads();
                int v643;
                v643 = threadIdx.x;
                auto v644 = cooperative_groups::labeled_partition(v642,v643);
                Closure0 v645{};
                float v646;
                v646 = cooperative_groups::reduce(v644, v633, v645);
                float v647[4l];
                int v648;
                v648 = 0l;
                while (while_method_1(v648)){
                    int v650;
                    v650 = 0l;
                    while (while_method_2(v650)){
                        assert("Tensor range check" && 0 <= v648 && v648 < 1l);
                        assert("Tensor range check" && 0 <= v650 && v650 < 4l);
                        int v652;
                        v652 = 4l * v648;
                        int v653;
                        v653 = v652 + v650;
                        float v654;
                        v654 = v623[v653];
                        bool v655;
                        v655 = v646 == 0.0f;
                        bool v656;
                        v656 = v655 != true;
                        float v658;
                        if (v656){
                            float v657;
                            v657 = v654 / v646;
                            v658 = v657;
                        } else {
                            v658 = 0.25f;
                        }
                        assert("Tensor range check" && 0 <= v648 && v648 < 1l);
                        assert("Tensor range check" && 0 <= v650 && v650 < 4l);
                        v647[v653] = v658;
                        v650 += 1l ;
                    }
                    v648 += 1l ;
                }
                float v659[4l];
                int v660;
                v660 = 0l;
                while (while_method_1(v660)){
                    int v662;
                    v662 = 0l;
                    while (while_method_2(v662)){
                        assert("Tensor range check" && 0 <= v660 && v660 < 1l);
                        assert("Tensor range check" && 0 <= v662 && v662 < 4l);
                        int v664;
                        v664 = 4l * v660;
                        int v665;
                        v665 = v664 + v662;
                        float v666;
                        v666 = v520[v665];
                        float v667;
                        v667 = v647[v665];
                        float v668;
                        v668 = v666 + v667;
                        assert("Tensor range check" && 0 <= v660 && v660 < 1l);
                        assert("Tensor range check" && 0 <= v662 && v662 < 4l);
                        v659[v665] = v668;
                        v662 += 1l ;
                    }
                    v660 += 1l ;
                }
                float v669[4l];
                int v670;
                v670 = 0l;
                while (while_method_1(v670)){
                    int v672;
                    v672 = 0l;
                    while (while_method_2(v672)){
                        assert("Tensor range check" && 0 <= v670 && v670 < 1l);
                        assert("Tensor range check" && 0 <= v672 && v672 < 4l);
                        int v674;
                        v674 = 4l * v670;
                        int v675;
                        v675 = v674 + v672;
                        float v676;
                        v676 = v659[v675];
                        float v677;
                        v677 = -v676;
                        bool v678;
                        v678 = v676 >= v677;
                        float v679;
                        if (v678){
                            v679 = v676;
                        } else {
                            v679 = v677;
                        }
                        assert("Tensor range check" && 0 <= v670 && v670 < 1l);
                        assert("Tensor range check" && 0 <= v672 && v672 < 4l);
                        v669[v675] = v679;
                        v672 += 1l ;
                    }
                    v670 += 1l ;
                }
                float v680;
                v680 = 0.0f;
                int v681;
                v681 = 0l;
                while (while_method_1(v681)){
                    int v683;
                    v683 = 0l;
                    while (while_method_2(v683)){
                        assert("Tensor range check" && 0 <= v681 && v681 < 1l);
                        assert("Tensor range check" && 0 <= v683 && v683 < 4l);
                        int v685;
                        v685 = 4l * v681;
                        int v686;
                        v686 = v685 + v683;
                        float v687;
                        v687 = v669[v686];
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
                auto v691 = cooperative_groups::labeled_partition(v689,v690);
                float v692;
                v692 = cooperative_groups::reduce(v691, v680, v645);
                bool v693;
                v693 = v692 > 100.0f;
                float v695;
                if (v693){
                    float v694;
                    v694 = 100.0f / v692;
                    v695 = v694;
                } else {
                    v695 = 1.0f;
                }
                float v696[4l];
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
                        float v703;
                        v703 = v669[v702];
                        float v704;
                        v704 = v695 * v703;
                        assert("Tensor range check" && 0 <= v697 && v697 < 1l);
                        assert("Tensor range check" && 0 <= v699 && v699 < 4l);
                        v696[v702] = v704;
                        v699 += 1l ;
                    }
                    v697 += 1l ;
                }
                int v705;
                v705 = 0l;
                while (while_method_1(v705)){
                    int v707;
                    v707 = 0l;
                    while (while_method_2(v707)){
                        assert("Tensor range check" && 0 <= v705 && v705 < 1l);
                        assert("Tensor range check" && 0 <= v707 && v707 < 4l);
                        int v709;
                        v709 = 4l * v705;
                        int v710;
                        v710 = v709 + v707;
                        float v711;
                        v711 = v611[v710];
                        float v712;
                        v712 = v696[v710];
                        float v713;
                        v713 = v523[v710];
                        float v714;
                        v714 = v524[v710];
                        assert("Tensor range check" && 0 <= v705 && v705 < 1l);
                        assert("Tensor range check" && 0 <= v707 && v707 < 4l);
                        v520[v710] = v712;
                        v521[v710] = v711;
                        v522[v710] = 0.0f;
                        v523[v710] = v713;
                        v524[v710] = v714;
                        v707 += 1l ;
                    }
                    v705 += 1l ;
                }
            } else {
            }
            assert("Tensor range check" && 0 <= v512 && v512 < 4l);
            assert("Tensor range check" && 0 <= v514 && v514 < 128l);
            int v715;
            v715 = 0l;
            while (while_method_1(v715)){
                assert("Tensor range check" && 0 <= v715 && v715 < 1l);
                int v717;
                v717 = 4l * v715;
                int v718;
                v718 = v717 + v519;
                assert("Tensor range check" && 0 <= v715 && v715 < 1l);
                int4* v719;
                v719 = reinterpret_cast<int4*>(v520 + v717);
                int4* v720;
                v720 = reinterpret_cast<int4*>(v2 + v718);
                assert("Pointer alignment check" && (unsigned long long)(v719) % 4l == 0 && (unsigned long long)(v720) % 4l == 0);
                *v720 = *v719;
                int4* v721;
                v721 = reinterpret_cast<int4*>(v521 + v717);
                int4* v722;
                v722 = reinterpret_cast<int4*>(v4 + v718);
                assert("Pointer alignment check" && (unsigned long long)(v721) % 4l == 0 && (unsigned long long)(v722) % 4l == 0);
                *v722 = *v721;
                int4* v723;
                v723 = reinterpret_cast<int4*>(v522 + v717);
                int4* v724;
                v724 = reinterpret_cast<int4*>(v6 + v718);
                assert("Pointer alignment check" && (unsigned long long)(v723) % 4l == 0 && (unsigned long long)(v724) % 4l == 0);
                *v724 = *v723;
                int4* v725;
                v725 = reinterpret_cast<int4*>(v523 + v717);
                int4* v726;
                v726 = reinterpret_cast<int4*>(v8 + v718);
                assert("Pointer alignment check" && (unsigned long long)(v725) % 4l == 0 && (unsigned long long)(v726) % 4l == 0);
                *v726 = *v725;
                int4* v727;
                v727 = reinterpret_cast<int4*>(v524 + v717);
                int4* v728;
                v728 = reinterpret_cast<int4*>(v10 + v718);
                assert("Pointer alignment check" && (unsigned long long)(v727) % 4l == 0 && (unsigned long long)(v728) % 4l == 0);
                *v728 = *v727;
                v715 += 1l ;
            }
            v514 += 1l ;
        }
        v512 += 1l ;
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
    v0 = cp.empty(1310720,dtype=cp.uint8)
    v1 = cp.empty(147456,dtype=cp.uint8)
    v3 = v0[0:0+4*65536].view(cp.float32)
    v5 = v0[262144:262144+4*65536].view(cp.float32)
    v7 = v0[524288:524288+4*65536].view(cp.float32)
    v9 = v0[786432:786432+4*65536].view(cp.float32)
    v11 = v0[1048576:1048576+4*65536].view(cp.float32)
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
    v12 = 0
    v13 = raw_module.get_function(f"entry{v12}")
    del v12
    v13.max_dynamic_shared_size_bytes = 0 
    v13((1,),(32,),(v1, v0),shared_mem=0)
    del v0, v1, v13
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
