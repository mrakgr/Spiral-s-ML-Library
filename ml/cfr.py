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
    assert("Tensor range check" && 0 <= v20 && v20 < 32l);
    v14[v20] = v10;
    v15[v20] = v12;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v21;
    v21 = threadIdx.x;
    bool v22;
    v22 = 0l <= v21;
    bool v23;
    v23 = v22 == false;
    if (v23){
        assert("The index needs to be zero or positive." && v22);
    } else {
    }
    int v25;
    v25 = v21 % 1l;
    bool v26;
    v26 = v21 < 32l;
    bool v27;
    v27 = v26 == false;
    if (v27){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v26);
    } else {
    }
    assert("Tensor range check" && 0 <= v21 && v21 < 32l);
    int v29;
    v29 = 0l;
    while (while_method_1(v29)){
        bool v31;
        v31 = v22 && v26;
        bool v32;
        v32 = v31 == false;
        if (v32){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v31);
        } else {
        }
        bool v34;
        v34 = 0l <= v29;
        bool v36;
        if (v34){
            bool v35;
            v35 = v29 < 1l;
            v36 = v35;
        } else {
            v36 = false;
        }
        bool v37;
        v37 = v36 == false;
        if (v37){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v36);
        } else {
        }
        int v39;
        v39 = v29 * 32l;
        int v40;
        v40 = v39 + v21;
        assert("Tensor range check" && 0 <= v29 && v29 < 1l);
        int v41;
        v41 = 32l * v29;
        int v42;
        v42 = v41 + v21;
        float * v43;
        v43 = v14[v42];
        float * v44;
        v44 = v15[v42];
        /* void array index */;
        assert("Tensor range check" && 0 <= v25 && v25 < 1l);
        int v45;
        v45 = 4l * v25;
        float v46[4l];
        float v47[4l];
        int v48[4l];
        int v49;
        v49 = 0l;
        while (while_method_1(v49)){
            assert("Tensor range check" && 0 <= v49 && v49 < 1l);
            int v51;
            v51 = 4l * v49;
            assert("Tensor range check" && 0 <= v49 && v49 < 1l);
            int v52;
            v52 = v51 + v45;
            int4* v53;
            v53 = reinterpret_cast<int4*>(v43 + v52);
            int4* v54;
            v54 = reinterpret_cast<int4*>(v46 + v51);
            assert("Pointer alignment check" && (unsigned long long)(v53) % 4l == 0 && (unsigned long long)(v54) % 4l == 0);
            *v54 = *v53;
            int4* v55;
            v55 = reinterpret_cast<int4*>(v44 + v52);
            int4* v56;
            v56 = reinterpret_cast<int4*>(v47 + v51);
            assert("Pointer alignment check" && (unsigned long long)(v55) % 4l == 0 && (unsigned long long)(v56) % 4l == 0);
            *v56 = *v55;
            v49 += 1l ;
        }
        int v57;
        v57 = 0l;
        while (while_method_1(v57)){
            int v59;
            v59 = 0l;
            while (while_method_2(v59)){
                bool v61;
                v61 = 0l <= v59;
                bool v63;
                if (v61){
                    bool v62;
                    v62 = v59 < 4l;
                    v63 = v62;
                } else {
                    v63 = false;
                }
                bool v64;
                v64 = v63 == false;
                if (v64){
                    assert("The indices should be inside the range of the dimension." && v63);
                } else {
                }
                bool v66;
                v66 = 0l <= v25;
                bool v68;
                if (v66){
                    bool v67;
                    v67 = v25 < 1l;
                    v68 = v67;
                } else {
                    v68 = false;
                }
                bool v69;
                v69 = v68 == false;
                if (v69){
                    assert("The indices should be inside the range of the dimension." && v68);
                } else {
                }
                int v71;
                v71 = v25 * 4l;
                int v72;
                v72 = v59 + v71;
                bool v73;
                v73 = 0l <= v57;
                bool v75;
                if (v73){
                    bool v74;
                    v74 = v57 < 1l;
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
                v78 = v57 * 4l;
                int v79;
                v79 = v72 + v78;
                assert("Tensor range check" && 0 <= v57 && v57 < 1l);
                assert("Tensor range check" && 0 <= v59 && v59 < 4l);
                int v80;
                v80 = 4l * v57;
                int v81;
                v81 = v80 + v59;
                v48[v81] = v79;
                v59 += 1l ;
            }
            v57 += 1l ;
        }
        unsigned long long v82;
        v82 = clock64();
        int v83;
        v83 = threadIdx.x;
        unsigned long long v84;
        v84 = (unsigned long long)v83;
        curandStatePhilox4_32_10_t v85;
        curand_init(v82,v84,0ull,&v85);
        bool v86[4l];
        int v87;
        v87 = 0l;
        while (while_method_1(v87)){
            int v89;
            v89 = 0l;
            while (while_method_2(v89)){
                assert("Tensor range check" && 0 <= v87 && v87 < 1l);
                assert("Tensor range check" && 0 <= v89 && v89 < 4l);
                int v91;
                v91 = 4l * v87;
                int v92;
                v92 = v91 + v89;
                float v93;
                v93 = v46[v92];
                int v94;
                v94 = v48[v92];
                bool v95;
                v95 = v94 < 3l;
                assert("Tensor range check" && 0 <= v87 && v87 < 1l);
                assert("Tensor range check" && 0 <= v89 && v89 < 4l);
                v86[v92] = v95;
                v89 += 1l ;
            }
            v87 += 1l ;
        }
        float v96[4l];
        int v97;
        v97 = 0l;
        while (while_method_1(v97)){
            int v99;
            v99 = 0l;
            while (while_method_2(v99)){
                assert("Tensor range check" && 0 <= v97 && v97 < 1l);
                assert("Tensor range check" && 0 <= v99 && v99 < 4l);
                int v101;
                v101 = 4l * v97;
                int v102;
                v102 = v101 + v99;
                float v103;
                v103 = v46[v102];
                bool v104;
                v104 = v86[v102];
                float v107;
                if (v104){
                    bool v105;
                    v105 = 0.0f >= v103;
                    if (v105){
                        v107 = 0.0f;
                    } else {
                        v107 = v103;
                    }
                } else {
                    v107 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v97 && v97 < 1l);
                assert("Tensor range check" && 0 <= v99 && v99 < 4l);
                v96[v102] = v107;
                v99 += 1l ;
            }
            v97 += 1l ;
        }
        float v108;
        v108 = 0.0f;
        int v109;
        v109 = 0l;
        while (while_method_1(v109)){
            int v111;
            v111 = 0l;
            while (while_method_2(v111)){
                assert("Tensor range check" && 0 <= v109 && v109 < 1l);
                assert("Tensor range check" && 0 <= v111 && v111 < 4l);
                int v113;
                v113 = 4l * v109;
                int v114;
                v114 = v113 + v111;
                float v115;
                v115 = v96[v114];
                float v116;
                v116 = v108 + v115;
                v108 = v116;
                v111 += 1l ;
            }
            v109 += 1l ;
        }
        auto v117 = cooperative_groups::coalesced_threads();
        int v118;
        v118 = threadIdx.x;
        auto v119 = cooperative_groups::labeled_partition(v117,v118);
        Closure0 v120{};
        float v121;
        v121 = cooperative_groups::reduce(v119, v108, v120);
        int v122[4l];
        int v123;
        v123 = 0l;
        while (while_method_1(v123)){
            int v125;
            v125 = 0l;
            while (while_method_2(v125)){
                assert("Tensor range check" && 0 <= v123 && v123 < 1l);
                assert("Tensor range check" && 0 <= v125 && v125 < 4l);
                int v127;
                v127 = 4l * v123;
                int v128;
                v128 = v127 + v125;
                bool v129;
                v129 = v86[v128];
                int v130;
                if (v129){
                    v130 = 1l;
                } else {
                    v130 = 0l;
                }
                assert("Tensor range check" && 0 <= v123 && v123 < 1l);
                assert("Tensor range check" && 0 <= v125 && v125 < 4l);
                v122[v128] = v130;
                v125 += 1l ;
            }
            v123 += 1l ;
        }
        int v131;
        v131 = 0l;
        int v132;
        v132 = 0l;
        while (while_method_1(v132)){
            int v134;
            v134 = 0l;
            while (while_method_2(v134)){
                assert("Tensor range check" && 0 <= v132 && v132 < 1l);
                assert("Tensor range check" && 0 <= v134 && v134 < 4l);
                int v136;
                v136 = 4l * v132;
                int v137;
                v137 = v136 + v134;
                int v138;
                v138 = v122[v137];
                int v139;
                v139 = v131 + v138;
                v131 = v139;
                v134 += 1l ;
            }
            v132 += 1l ;
        }
        auto v140 = cooperative_groups::coalesced_threads();
        int v141;
        v141 = threadIdx.x;
        auto v142 = cooperative_groups::labeled_partition(v140,v141);
        Closure1 v143{};
        int v144;
        v144 = cooperative_groups::reduce(v142, v131, v143);
        float v145;
        v145 = (float)v144;
        float v146;
        v146 = 1.0f / v145;
        float v147[4l];
        int v148;
        v148 = 0l;
        while (while_method_1(v148)){
            int v150;
            v150 = 0l;
            while (while_method_2(v150)){
                assert("Tensor range check" && 0 <= v148 && v148 < 1l);
                assert("Tensor range check" && 0 <= v150 && v150 < 4l);
                int v152;
                v152 = 4l * v148;
                int v153;
                v153 = v152 + v150;
                float v154;
                v154 = v96[v153];
                bool v155;
                v155 = v86[v153];
                bool v156;
                v156 = v155 == false;
                float v161;
                if (v156){
                    v161 = 0.0f;
                } else {
                    bool v157;
                    v157 = v121 == 0.0f;
                    bool v158;
                    v158 = v157 != true;
                    if (v158){
                        float v159;
                        v159 = v154 / v121;
                        v161 = v159;
                    } else {
                        v161 = v146;
                    }
                }
                assert("Tensor range check" && 0 <= v148 && v148 < 1l);
                assert("Tensor range check" && 0 <= v150 && v150 < 4l);
                v147[v153] = v161;
                v150 += 1l ;
            }
            v148 += 1l ;
        }
        float v162[4l];
        float v163;
        v163 = 0.0f;
        int v164;
        v164 = 0l;
        while (while_method_1(v164)){
            assert("Tensor range check" && 0 <= v164 && v164 < 1l);
            int v166;
            v166 = 4l * v164;
            assert("Tensor range check" && 0 <= v164 && v164 < 1l);
            int v167; float v168;
            Tuple1 tmp0 = Tuple1{0l, 0.0f};
            v167 = tmp0.v0; v168 = tmp0.v1;
            while (while_method_2(v167)){
                assert("Tensor range check" && 0 <= v167 && v167 < 4l);
                int v170;
                v170 = v167 + v166;
                float v171;
                v171 = v147[v170];
                float v172;
                v172 = v168 + v171;
                v168 = v172;
                v167 += 1l ;
            }
            auto v173 = cooperative_groups::coalesced_threads();
            int v174;
            v174 = threadIdx.x;
            auto v175 = cooperative_groups::labeled_partition(v173,v174);
            Closure2 v176{};
            float v177;
            v177 = cooperative_groups::inclusive_scan(v175, v168, v176);
            float v178;
            v178 = v175.shfl_up(v177,1);
            bool v179;
            v179 = v175.thread_rank() == 0;
            float v180;
            if (v179){
                v180 = 0.0f;
            } else {
                v180 = v178;
            }
            float v181;
            v181 = v175.shfl(v177,v175.num_threads()-1);
            float v182;
            v182 = v163 + v180;
            int v183; float v184;
            Tuple1 tmp1 = Tuple1{0l, v182};
            v183 = tmp1.v0; v184 = tmp1.v1;
            while (while_method_2(v183)){
                assert("Tensor range check" && 0 <= v183 && v183 < 4l);
                int v186;
                v186 = v183 + v166;
                float v187;
                v187 = v147[v186];
                float v188;
                v188 = v184 + v187;
                assert("Tensor range check" && 0 <= v183 && v183 < 4l);
                v162[v186] = v188;
                v184 = v188;
                v183 += 1l ;
            }
            float v189;
            v189 = v163 + v181;
            v163 = v189;
            v164 += 1l ;
        }
        float v190[4l];
        bool v191[4l];
        int v192;
        v192 = 0l;
        while (while_method_1(v192)){
            int v194;
            v194 = 0l;
            while (while_method_2(v194)){
                assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                assert("Tensor range check" && 0 <= v194 && v194 < 4l);
                int v196;
                v196 = 4l * v192;
                int v197;
                v197 = v196 + v194;
                float v198;
                v198 = v162[v197];
                float v199;
                v199 = v147[v197];
                bool v200;
                v200 = v199 > 0.0f;
                assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                assert("Tensor range check" && 0 <= v194 && v194 < 4l);
                v190[v197] = v198;
                v191[v197] = v200;
                v194 += 1l ;
            }
            v192 += 1l ;
        }
        float v201; bool v202;
        Tuple2 tmp2 = Tuple2{-1.0f / 0.0f, false};
        v201 = tmp2.v0; v202 = tmp2.v1;
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
                bool v210;
                v210 = v191[v208];
                float v217; bool v218;
                if (v202){
                    if (v210){
                        bool v211;
                        v211 = v201 >= v209;
                        float v212;
                        if (v211){
                            v212 = v201;
                        } else {
                            v212 = v209;
                        }
                        v217 = v212; v218 = true;
                    } else {
                        v217 = v201; v218 = v202;
                    }
                } else {
                    if (v210){
                        v217 = v209; v218 = v210;
                    } else {
                        v217 = v201; v218 = v202;
                    }
                }
                v201 = v217;
                v202 = v218;
                v205 += 1l ;
            }
            v203 += 1l ;
        }
        auto v219 = cooperative_groups::coalesced_threads();
        int v220;
        v220 = threadIdx.x;
        auto v221 = cooperative_groups::labeled_partition(v219,v220);
        Closure3 v222{};
        float v223; bool v224;
        Tuple2 tmp3 = cooperative_groups::reduce(v221, Tuple2{v201, v202}, v222);
        v223 = tmp3.v0; v224 = tmp3.v1;
        bool v225;
        v225 = v224 == false;
        if (v225){
            assert("The local reduce must be true." && v224);
        } else {
        }
        float v227[4l];
        int v228[4l];
        int v229;
        v229 = 0l;
        while (while_method_1(v229)){
            int v231;
            v231 = 0l;
            while (while_method_2(v231)){
                assert("Tensor range check" && 0 <= v229 && v229 < 1l);
                assert("Tensor range check" && 0 <= v231 && v231 < 4l);
                int v233;
                v233 = 4l * v229;
                int v234;
                v234 = v233 + v231;
                int v235;
                v235 = v48[v234];
                float v236;
                v236 = curand_uniform(&v85);
                assert("Tensor range check" && 0 <= v229 && v229 < 1l);
                assert("Tensor range check" && 0 <= v231 && v231 < 4l);
                v227[v234] = v236;
                v228[v234] = v235;
                v231 += 1l ;
            }
            v229 += 1l ;
        }
        float v237; int v238;
        Tuple3 tmp4 = Tuple3{0.0f, 2147483647l};
        v237 = tmp4.v0; v238 = tmp4.v1;
        int v239;
        v239 = 0l;
        while (while_method_1(v239)){
            int v241;
            v241 = 0l;
            while (while_method_2(v241)){
                assert("Tensor range check" && 0 <= v239 && v239 < 1l);
                assert("Tensor range check" && 0 <= v241 && v241 < 4l);
                int v243;
                v243 = 4l * v239;
                int v244;
                v244 = v243 + v241;
                float v245;
                v245 = v227[v244];
                int v246;
                v246 = v228[v244];
                bool v247;
                v247 = v238 < v246;
                float v248; int v249;
                if (v247){
                    v248 = v237; v249 = v238;
                } else {
                    v248 = v245; v249 = v246;
                }
                v237 = v248;
                v238 = v249;
                v241 += 1l ;
            }
            v239 += 1l ;
        }
        auto v250 = cooperative_groups::coalesced_threads();
        int v251;
        v251 = threadIdx.x;
        auto v252 = cooperative_groups::labeled_partition(v250,v251);
        Closure4 v253{};
        float v254; int v255;
        Tuple3 tmp5 = cooperative_groups::reduce(v252, Tuple3{v237, v238}, v253);
        v254 = tmp5.v0; v255 = tmp5.v1;
        float v256;
        v256 = v223 * v254;
        int v257[4l];
        bool v258[4l];
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
                v265 = v190[v264];
                bool v266;
                v266 = v191[v264];
                int v267;
                v267 = v48[v264];
                int v270; bool v271;
                if (v266){
                    float v268;
                    v268 = v265 - v256;
                    bool v269;
                    v269 = v268 >= 0.0f;
                    v270 = v267; v271 = v269;
                } else {
                    v270 = 2147483647l; v271 = false;
                }
                assert("Tensor range check" && 0 <= v259 && v259 < 1l);
                assert("Tensor range check" && 0 <= v261 && v261 < 4l);
                v257[v264] = v270;
                v258[v264] = v271;
                v261 += 1l ;
            }
            v259 += 1l ;
        }
        int v272; bool v273;
        Tuple4 tmp6 = Tuple4{2147483647l, false};
        v272 = tmp6.v0; v273 = tmp6.v1;
        int v274;
        v274 = 0l;
        while (while_method_1(v274)){
            int v276;
            v276 = 0l;
            while (while_method_2(v276)){
                assert("Tensor range check" && 0 <= v274 && v274 < 1l);
                assert("Tensor range check" && 0 <= v276 && v276 < 4l);
                int v278;
                v278 = 4l * v274;
                int v279;
                v279 = v278 + v276;
                int v280;
                v280 = v257[v279];
                bool v281;
                v281 = v258[v279];
                int v288; bool v289;
                if (v273){
                    if (v281){
                        bool v282;
                        v282 = v272 < v280;
                        int v283;
                        if (v282){
                            v283 = v272;
                        } else {
                            v283 = v280;
                        }
                        v288 = v283; v289 = true;
                    } else {
                        v288 = v272; v289 = v273;
                    }
                } else {
                    if (v281){
                        v288 = v280; v289 = v281;
                    } else {
                        v288 = v272; v289 = v273;
                    }
                }
                v272 = v288;
                v273 = v289;
                v276 += 1l ;
            }
            v274 += 1l ;
        }
        auto v290 = cooperative_groups::coalesced_threads();
        int v291;
        v291 = threadIdx.x;
        auto v292 = cooperative_groups::labeled_partition(v290,v291);
        Closure5 v293{};
        int v294; bool v295;
        Tuple4 tmp7 = cooperative_groups::reduce(v292, Tuple4{v272, v273}, v293);
        v294 = tmp7.v0; v295 = tmp7.v1;
        bool v296;
        v296 = v295 == false;
        if (v296){
            assert("The local reduce must be true." && v295);
        } else {
        }
        bool v298[4l];
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
                v305 = v47[v304];
                int v306;
                v306 = v48[v304];
                bool v307;
                v307 = v306 < 3l;
                assert("Tensor range check" && 0 <= v299 && v299 < 1l);
                assert("Tensor range check" && 0 <= v301 && v301 < 4l);
                v298[v304] = v307;
                v301 += 1l ;
            }
            v299 += 1l ;
        }
        float v308[4l];
        int v309;
        v309 = 0l;
        while (while_method_1(v309)){
            int v311;
            v311 = 0l;
            while (while_method_2(v311)){
                assert("Tensor range check" && 0 <= v309 && v309 < 1l);
                assert("Tensor range check" && 0 <= v311 && v311 < 4l);
                int v313;
                v313 = 4l * v309;
                int v314;
                v314 = v313 + v311;
                float v315;
                v315 = v47[v314];
                bool v316;
                v316 = v298[v314];
                float v319;
                if (v316){
                    bool v317;
                    v317 = 0.0f >= v315;
                    if (v317){
                        v319 = 0.0f;
                    } else {
                        v319 = v315;
                    }
                } else {
                    v319 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v309 && v309 < 1l);
                assert("Tensor range check" && 0 <= v311 && v311 < 4l);
                v308[v314] = v319;
                v311 += 1l ;
            }
            v309 += 1l ;
        }
        float v320;
        v320 = 0.0f;
        int v321;
        v321 = 0l;
        while (while_method_1(v321)){
            int v323;
            v323 = 0l;
            while (while_method_2(v323)){
                assert("Tensor range check" && 0 <= v321 && v321 < 1l);
                assert("Tensor range check" && 0 <= v323 && v323 < 4l);
                int v325;
                v325 = 4l * v321;
                int v326;
                v326 = v325 + v323;
                float v327;
                v327 = v308[v326];
                float v328;
                v328 = v320 + v327;
                v320 = v328;
                v323 += 1l ;
            }
            v321 += 1l ;
        }
        auto v329 = cooperative_groups::coalesced_threads();
        int v330;
        v330 = threadIdx.x;
        auto v331 = cooperative_groups::labeled_partition(v329,v330);
        float v332;
        v332 = cooperative_groups::reduce(v331, v320, v120);
        int v333[4l];
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
                bool v340;
                v340 = v298[v339];
                int v341;
                if (v340){
                    v341 = 1l;
                } else {
                    v341 = 0l;
                }
                assert("Tensor range check" && 0 <= v334 && v334 < 1l);
                assert("Tensor range check" && 0 <= v336 && v336 < 4l);
                v333[v339] = v341;
                v336 += 1l ;
            }
            v334 += 1l ;
        }
        int v342;
        v342 = 0l;
        int v343;
        v343 = 0l;
        while (while_method_1(v343)){
            int v345;
            v345 = 0l;
            while (while_method_2(v345)){
                assert("Tensor range check" && 0 <= v343 && v343 < 1l);
                assert("Tensor range check" && 0 <= v345 && v345 < 4l);
                int v347;
                v347 = 4l * v343;
                int v348;
                v348 = v347 + v345;
                int v349;
                v349 = v333[v348];
                int v350;
                v350 = v342 + v349;
                v342 = v350;
                v345 += 1l ;
            }
            v343 += 1l ;
        }
        auto v351 = cooperative_groups::coalesced_threads();
        int v352;
        v352 = threadIdx.x;
        auto v353 = cooperative_groups::labeled_partition(v351,v352);
        int v354;
        v354 = cooperative_groups::reduce(v353, v342, v143);
        float v355;
        v355 = (float)v354;
        float v356;
        v356 = 1.0f / v355;
        float v357[4l];
        int v358;
        v358 = 0l;
        while (while_method_1(v358)){
            int v360;
            v360 = 0l;
            while (while_method_2(v360)){
                assert("Tensor range check" && 0 <= v358 && v358 < 1l);
                assert("Tensor range check" && 0 <= v360 && v360 < 4l);
                int v362;
                v362 = 4l * v358;
                int v363;
                v363 = v362 + v360;
                float v364;
                v364 = v308[v363];
                bool v365;
                v365 = v298[v363];
                bool v366;
                v366 = v365 == false;
                float v371;
                if (v366){
                    v371 = 0.0f;
                } else {
                    bool v367;
                    v367 = v332 == 0.0f;
                    bool v368;
                    v368 = v367 != true;
                    if (v368){
                        float v369;
                        v369 = v364 / v332;
                        v371 = v369;
                    } else {
                        v371 = v356;
                    }
                }
                assert("Tensor range check" && 0 <= v358 && v358 < 1l);
                assert("Tensor range check" && 0 <= v360 && v360 < 4l);
                v357[v363] = v371;
                v360 += 1l ;
            }
            v358 += 1l ;
        }
        float v372; int v373;
        Tuple3 tmp8 = Tuple3{0.0f, 2147483647l};
        v372 = tmp8.v0; v373 = tmp8.v1;
        int v374;
        v374 = 0l;
        while (while_method_1(v374)){
            int v376;
            v376 = 0l;
            while (while_method_2(v376)){
                assert("Tensor range check" && 0 <= v374 && v374 < 1l);
                assert("Tensor range check" && 0 <= v376 && v376 < 4l);
                int v378;
                v378 = 4l * v374;
                int v379;
                v379 = v378 + v376;
                float v380;
                v380 = v147[v379];
                int v381;
                v381 = v48[v379];
                bool v382;
                v382 = v373 == v294;
                float v386; int v387;
                if (v382){
                    v386 = v372; v387 = v373;
                } else {
                    bool v383;
                    v383 = v381 == v294;
                    if (v383){
                        v386 = v380; v387 = v381;
                    } else {
                        v386 = v372; v387 = v373;
                    }
                }
                v372 = v386;
                v373 = v387;
                v376 += 1l ;
            }
            v374 += 1l ;
        }
        auto v388 = cooperative_groups::coalesced_threads();
        int v389;
        v389 = threadIdx.x;
        auto v390 = cooperative_groups::labeled_partition(v388,v389);
        Closure6 v391{v294};
        float v392; int v393;
        Tuple3 tmp9 = cooperative_groups::reduce(v390, Tuple3{v372, v373}, v391);
        v392 = tmp9.v0; v393 = tmp9.v1;
        bool v394;
        v394 = v393 == 2147483647l;
        bool v395;
        v395 = v394 != true;
        bool v396;
        v396 = v395 == false;
        if (v396){
            assert("Expected a valid action id in get_action." && v395);
        } else {
        }
        float v398; int v399;
        Tuple3 tmp10 = Tuple3{0.0f, 2147483647l};
        v398 = tmp10.v0; v399 = tmp10.v1;
        int v400;
        v400 = 0l;
        while (while_method_1(v400)){
            int v402;
            v402 = 0l;
            while (while_method_2(v402)){
                assert("Tensor range check" && 0 <= v400 && v400 < 1l);
                assert("Tensor range check" && 0 <= v402 && v402 < 4l);
                int v404;
                v404 = 4l * v400;
                int v405;
                v405 = v404 + v402;
                float v406;
                v406 = v357[v405];
                int v407;
                v407 = v48[v405];
                bool v408;
                v408 = v399 == v294;
                float v412; int v413;
                if (v408){
                    v412 = v398; v413 = v399;
                } else {
                    bool v409;
                    v409 = v407 == v294;
                    if (v409){
                        v412 = v406; v413 = v407;
                    } else {
                        v412 = v398; v413 = v399;
                    }
                }
                v398 = v412;
                v399 = v413;
                v402 += 1l ;
            }
            v400 += 1l ;
        }
        auto v414 = cooperative_groups::coalesced_threads();
        int v415;
        v415 = threadIdx.x;
        auto v416 = cooperative_groups::labeled_partition(v414,v415);
        float v417; int v418;
        Tuple3 tmp11 = cooperative_groups::reduce(v416, Tuple3{v398, v399}, v391);
        v417 = tmp11.v0; v418 = tmp11.v1;
        bool v419;
        v419 = v418 == 2147483647l;
        bool v420;
        v420 = v419 != true;
        bool v421;
        v421 = v420 == false;
        if (v421){
            assert("Expected a valid action id in get_action." && v420);
        } else {
        }
        int v423;
        v423 = 0l;
        while (while_method_1(v423)){
            assert("Tensor range check" && 0 <= v423 && v423 < 1l);
            assert("Tensor range check" && 0 <= v423 && v423 < 1l);
            v423 += 1l ;
        }
        assert("Tensor range check" && 0 <= v40 && v40 < 32l);
        v17[v40] = v417;
        v18[v40] = v392;
        v19[v40] = v294;
        v29 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v425;
    v425 = threadIdx.x;
    assert("Tensor range check" && 0 <= v425 && v425 < 32l);
    float v426;
    v426 = v17[v425];
    float v427;
    v427 = v18[v425];
    int v428;
    v428 = v19[v425];
    return Tuple0{v426, v427, v428};
}
__device__ void push_0(float * v0, float * v1, float * v2, float * v3, float * v4, int * v5, float * v6, int * v7, int * v8, double * v9, double * v10, float * v11, float * v12, float * v13, int v14, int & v15, double * v16, double * v17, int v18, int v19, int v20){
    float v21; float v22; int v23;
    Tuple0 tmp12 = method_1(v0, v1, v2, v3, v4, v19, v20);
    v21 = tmp12.v0; v22 = tmp12.v1; v23 = tmp12.v2;
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
    v12 = reinterpret_cast<int *>(&v1[1310720ull]);
    float * v14;
    v14 = reinterpret_cast<float *>(&v1[1318912ull]);
    int * v16;
    v16 = reinterpret_cast<int *>(&v1[1327104ull]);
    int * v18;
    v18 = reinterpret_cast<int *>(&v1[1335296ull]);
    double * v20;
    v20 = reinterpret_cast<double *>(&v1[1343488ull]);
    double * v22;
    v22 = reinterpret_cast<double *>(&v1[1376256ull]);
    float * v24;
    v24 = reinterpret_cast<float *>(&v1[1409024ull]);
    float * v26;
    v26 = reinterpret_cast<float *>(&v1[1441792ull]);
    float * v28;
    v28 = reinterpret_cast<float *>(&v1[1449984ull]);
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
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v119;
        v119 = threadIdx.x;
        bool v120;
        v120 = 0l <= v119;
        bool v121;
        v121 = v120 == false;
        if (v121){
            assert("The index needs to be zero or positive." && v120);
        } else {
        }
        int v123;
        v123 = v119 % 1l;
        bool v124;
        v124 = v119 < 32l;
        bool v125;
        v125 = v124 == false;
        if (v125){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v124);
        } else {
        }
        assert("Tensor range check" && 0 <= v119 && v119 < 32l);
        int v127;
        v127 = 0l;
        while (while_method_1(v127)){
            bool v129;
            v129 = v120 && v124;
            bool v130;
            v130 = v129 == false;
            if (v130){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v129);
            } else {
            }
            bool v132;
            v132 = 0l <= v127;
            bool v134;
            if (v132){
                bool v133;
                v133 = v127 < 1l;
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
            v137 = v127 * 32l;
            int v138;
            v138 = v137 + v119;
            assert("Tensor range check" && 0 <= v127 && v127 < 1l);
            int v139;
            v139 = 32l * v127;
            int v140;
            v140 = v139 + v119;
            float v141;
            v141 = v106[v140];
            int v142;
            v142 = v107[v140];
            float v143;
            v143 = v108[v140];
            int v144;
            v144 = v109[v140];
            double * v145;
            v145 = v110[v140];
            double * v146;
            v146 = v111[v140];
            float * v147;
            v147 = v112[v140];
            float * v148;
            v148 = v113[v140];
            float * v149;
            v149 = v114[v140];
            float * v150;
            v150 = v115[v140];
            /* void array index */;
            assert("Tensor range check" && 0 <= v123 && v123 < 1l);
            int v151;
            v151 = 4l * v123;
            float v152[4l];
            float v153[4l];
            float v154[4l];
            int v155[4l];
            int v156;
            v156 = 0l;
            while (while_method_1(v156)){
                assert("Tensor range check" && 0 <= v156 && v156 < 1l);
                int v158;
                v158 = 4l * v156;
                assert("Tensor range check" && 0 <= v156 && v156 < 1l);
                int v159;
                v159 = v158 + v151;
                int4* v160;
                v160 = reinterpret_cast<int4*>(v147 + v159);
                int4* v161;
                v161 = reinterpret_cast<int4*>(v152 + v158);
                assert("Pointer alignment check" && (unsigned long long)(v160) % 4l == 0 && (unsigned long long)(v161) % 4l == 0);
                *v161 = *v160;
                int4* v162;
                v162 = reinterpret_cast<int4*>(v148 + v159);
                int4* v163;
                v163 = reinterpret_cast<int4*>(v153 + v158);
                assert("Pointer alignment check" && (unsigned long long)(v162) % 4l == 0 && (unsigned long long)(v163) % 4l == 0);
                *v163 = *v162;
                int4* v164;
                v164 = reinterpret_cast<int4*>(v149 + v159);
                int4* v165;
                v165 = reinterpret_cast<int4*>(v154 + v158);
                assert("Pointer alignment check" && (unsigned long long)(v164) % 4l == 0 && (unsigned long long)(v165) % 4l == 0);
                *v165 = *v164;
                v156 += 1l ;
            }
            int v166;
            v166 = 0l;
            while (while_method_1(v166)){
                int v168;
                v168 = 0l;
                while (while_method_2(v168)){
                    bool v170;
                    v170 = 0l <= v168;
                    bool v172;
                    if (v170){
                        bool v171;
                        v171 = v168 < 4l;
                        v172 = v171;
                    } else {
                        v172 = false;
                    }
                    bool v173;
                    v173 = v172 == false;
                    if (v173){
                        assert("The indices should be inside the range of the dimension." && v172);
                    } else {
                    }
                    bool v175;
                    v175 = 0l <= v123;
                    bool v177;
                    if (v175){
                        bool v176;
                        v176 = v123 < 1l;
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
                    int v180;
                    v180 = v123 * 4l;
                    int v181;
                    v181 = v168 + v180;
                    bool v182;
                    v182 = 0l <= v166;
                    bool v184;
                    if (v182){
                        bool v183;
                        v183 = v166 < 1l;
                        v184 = v183;
                    } else {
                        v184 = false;
                    }
                    bool v185;
                    v185 = v184 == false;
                    if (v185){
                        assert("The indices should be inside the range of the dimension." && v184);
                    } else {
                    }
                    int v187;
                    v187 = v166 * 4l;
                    int v188;
                    v188 = v181 + v187;
                    assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                    assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                    int v189;
                    v189 = 4l * v166;
                    int v190;
                    v190 = v189 + v168;
                    v155[v190] = v188;
                    v168 += 1l ;
                }
                v166 += 1l ;
            }
            float v191[4l];
            int v192;
            v192 = 0l;
            while (while_method_1(v192)){
                int v194;
                v194 = 0l;
                while (while_method_2(v194)){
                    assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                    assert("Tensor range check" && 0 <= v194 && v194 < 4l);
                    int v196;
                    v196 = 4l * v192;
                    int v197;
                    v197 = v196 + v194;
                    float v198;
                    v198 = v153[v197];
                    float v199;
                    v199 = v154[v197];
                    bool v200;
                    v200 = v199 == 0.0f;
                    bool v201;
                    v201 = v200 != true;
                    float v203;
                    if (v201){
                        float v202;
                        v202 = v198 / v199;
                        v203 = v202;
                    } else {
                        v203 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v192 && v192 < 1l);
                    assert("Tensor range check" && 0 <= v194 && v194 < 4l);
                    v191[v197] = v203;
                    v194 += 1l ;
                }
                v192 += 1l ;
            }
            bool v204[4l];
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
                    v211 = v152[v210];
                    int v212;
                    v212 = v155[v210];
                    bool v213;
                    v213 = v212 < 3l;
                    assert("Tensor range check" && 0 <= v205 && v205 < 1l);
                    assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                    v204[v210] = v213;
                    v207 += 1l ;
                }
                v205 += 1l ;
            }
            float v214[4l];
            int v215;
            v215 = 0l;
            while (while_method_1(v215)){
                int v217;
                v217 = 0l;
                while (while_method_2(v217)){
                    assert("Tensor range check" && 0 <= v215 && v215 < 1l);
                    assert("Tensor range check" && 0 <= v217 && v217 < 4l);
                    int v219;
                    v219 = 4l * v215;
                    int v220;
                    v220 = v219 + v217;
                    float v221;
                    v221 = v152[v220];
                    bool v222;
                    v222 = v204[v220];
                    float v225;
                    if (v222){
                        bool v223;
                        v223 = 0.0f >= v221;
                        if (v223){
                            v225 = 0.0f;
                        } else {
                            v225 = v221;
                        }
                    } else {
                        v225 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v215 && v215 < 1l);
                    assert("Tensor range check" && 0 <= v217 && v217 < 4l);
                    v214[v220] = v225;
                    v217 += 1l ;
                }
                v215 += 1l ;
            }
            float v226;
            v226 = 0.0f;
            int v227;
            v227 = 0l;
            while (while_method_1(v227)){
                int v229;
                v229 = 0l;
                while (while_method_2(v229)){
                    assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                    assert("Tensor range check" && 0 <= v229 && v229 < 4l);
                    int v231;
                    v231 = 4l * v227;
                    int v232;
                    v232 = v231 + v229;
                    float v233;
                    v233 = v214[v232];
                    float v234;
                    v234 = v226 + v233;
                    v226 = v234;
                    v229 += 1l ;
                }
                v227 += 1l ;
            }
            auto v235 = cooperative_groups::coalesced_threads();
            int v236;
            v236 = threadIdx.x;
            auto v237 = cooperative_groups::labeled_partition(v235,v236);
            Closure0 v238{};
            float v239;
            v239 = cooperative_groups::reduce(v237, v226, v238);
            int v240[4l];
            int v241;
            v241 = 0l;
            while (while_method_1(v241)){
                int v243;
                v243 = 0l;
                while (while_method_2(v243)){
                    assert("Tensor range check" && 0 <= v241 && v241 < 1l);
                    assert("Tensor range check" && 0 <= v243 && v243 < 4l);
                    int v245;
                    v245 = 4l * v241;
                    int v246;
                    v246 = v245 + v243;
                    bool v247;
                    v247 = v204[v246];
                    int v248;
                    if (v247){
                        v248 = 1l;
                    } else {
                        v248 = 0l;
                    }
                    assert("Tensor range check" && 0 <= v241 && v241 < 1l);
                    assert("Tensor range check" && 0 <= v243 && v243 < 4l);
                    v240[v246] = v248;
                    v243 += 1l ;
                }
                v241 += 1l ;
            }
            int v249;
            v249 = 0l;
            int v250;
            v250 = 0l;
            while (while_method_1(v250)){
                int v252;
                v252 = 0l;
                while (while_method_2(v252)){
                    assert("Tensor range check" && 0 <= v250 && v250 < 1l);
                    assert("Tensor range check" && 0 <= v252 && v252 < 4l);
                    int v254;
                    v254 = 4l * v250;
                    int v255;
                    v255 = v254 + v252;
                    int v256;
                    v256 = v240[v255];
                    int v257;
                    v257 = v249 + v256;
                    v249 = v257;
                    v252 += 1l ;
                }
                v250 += 1l ;
            }
            auto v258 = cooperative_groups::coalesced_threads();
            int v259;
            v259 = threadIdx.x;
            auto v260 = cooperative_groups::labeled_partition(v258,v259);
            Closure1 v261{};
            int v262;
            v262 = cooperative_groups::reduce(v260, v249, v261);
            float v263;
            v263 = (float)v262;
            float v264;
            v264 = 1.0f / v263;
            float v265[4l];
            int v266;
            v266 = 0l;
            while (while_method_1(v266)){
                int v268;
                v268 = 0l;
                while (while_method_2(v268)){
                    assert("Tensor range check" && 0 <= v266 && v266 < 1l);
                    assert("Tensor range check" && 0 <= v268 && v268 < 4l);
                    int v270;
                    v270 = 4l * v266;
                    int v271;
                    v271 = v270 + v268;
                    float v272;
                    v272 = v214[v271];
                    bool v273;
                    v273 = v204[v271];
                    bool v274;
                    v274 = v273 == false;
                    float v279;
                    if (v274){
                        v279 = 0.0f;
                    } else {
                        bool v275;
                        v275 = v239 == 0.0f;
                        bool v276;
                        v276 = v275 != true;
                        if (v276){
                            float v277;
                            v277 = v272 / v239;
                            v279 = v277;
                        } else {
                            v279 = v264;
                        }
                    }
                    assert("Tensor range check" && 0 <= v266 && v266 < 1l);
                    assert("Tensor range check" && 0 <= v268 && v268 < 4l);
                    v265[v271] = v279;
                    v268 += 1l ;
                }
                v266 += 1l ;
            }
            float v280[4l];
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
                    v287 = v191[v286];
                    int v288;
                    v288 = v155[v286];
                    bool v289;
                    v289 = v142 == v288;
                    float v292;
                    if (v289){
                        float v290;
                        v290 = v143 - v287;
                        float v291;
                        v291 = v290 / v141;
                        v292 = v291;
                    } else {
                        v292 = 0.0f;
                    }
                    float v293;
                    v293 = v292 + v287;
                    assert("Tensor range check" && 0 <= v281 && v281 < 1l);
                    assert("Tensor range check" && 0 <= v283 && v283 < 4l);
                    v280[v286] = v293;
                    v283 += 1l ;
                }
                v281 += 1l ;
            }
            float v294[4l];
            int v295;
            v295 = 0l;
            while (while_method_1(v295)){
                int v297;
                v297 = 0l;
                while (while_method_2(v297)){
                    assert("Tensor range check" && 0 <= v295 && v295 < 1l);
                    assert("Tensor range check" && 0 <= v297 && v297 < 4l);
                    int v299;
                    v299 = 4l * v295;
                    int v300;
                    v300 = v299 + v297;
                    float v301;
                    v301 = v265[v300];
                    float v302;
                    v302 = v280[v300];
                    float v303;
                    v303 = v301 * v302;
                    assert("Tensor range check" && 0 <= v295 && v295 < 1l);
                    assert("Tensor range check" && 0 <= v297 && v297 < 4l);
                    v294[v300] = v303;
                    v297 += 1l ;
                }
                v295 += 1l ;
            }
            float v304;
            v304 = 0.0f;
            int v305;
            v305 = 0l;
            while (while_method_1(v305)){
                int v307;
                v307 = 0l;
                while (while_method_2(v307)){
                    assert("Tensor range check" && 0 <= v305 && v305 < 1l);
                    assert("Tensor range check" && 0 <= v307 && v307 < 4l);
                    int v309;
                    v309 = 4l * v305;
                    int v310;
                    v310 = v309 + v307;
                    float v311;
                    v311 = v294[v310];
                    float v312;
                    v312 = v304 + v311;
                    v304 = v312;
                    v307 += 1l ;
                }
                v305 += 1l ;
            }
            auto v313 = cooperative_groups::coalesced_threads();
            int v314;
            v314 = threadIdx.x;
            auto v315 = cooperative_groups::labeled_partition(v313,v314);
            float v316;
            v316 = cooperative_groups::reduce(v315, v304, v238);
            assert("Tensor range check" && 0 <= v138 && v138 < 32l);
            int v317;
            v317 = 2l * v138;
            double v318[2l];
            int v319;
            v319 = 0l;
            while (while_method_0(v319)){
                assert("Tensor range check" && 0 <= v319 && v319 < 2l);
                int v321;
                v321 = v319 + v317;
                double v322;
                v322 = v145[v321];
                bool v323;
                v323 = v144 == v319;
                double v324;
                if (v323){
                    v324 = 0.0;
                } else {
                    v324 = v322;
                }
                assert("Tensor range check" && 0 <= v319 && v319 < 2l);
                v318[v319] = v324;
                v319 += 1l ;
            }
            double v325;
            v325 = 0.0;
            int v326;
            v326 = 0l;
            while (while_method_0(v326)){
                assert("Tensor range check" && 0 <= v326 && v326 < 2l);
                double v328;
                v328 = v318[v326];
                double v329;
                v329 = v325 + v328;
                v325 = v329;
                v326 += 1l ;
            }
            double v330;
            v330 = 0.0;
            int v331;
            v331 = 0l;
            while (while_method_0(v331)){
                assert("Tensor range check" && 0 <= v331 && v331 < 2l);
                int v333;
                v333 = v331 + v317;
                double v334;
                v334 = v146[v333];
                double v335;
                v335 = v330 + v334;
                v330 = v335;
                v331 += 1l ;
            }
            double v336;
            v336 = v325 - v330;
            double v337;
            v337 = exp(v336);
            float v338;
            v338 = (float)v337;
            float v339[4l];
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
                    float v346;
                    v346 = v280[v345];
                    float v347;
                    v347 = v346 - v316;
                    float v348;
                    v348 = v338 * v347;
                    assert("Tensor range check" && 0 <= v340 && v340 < 1l);
                    assert("Tensor range check" && 0 <= v342 && v342 < 4l);
                    v339[v345] = v348;
                    v342 += 1l ;
                }
                v340 += 1l ;
            }
            int v349;
            v349 = 0l;
            while (while_method_1(v349)){
                assert("Tensor range check" && 0 <= v349 && v349 < 1l);
                int v351;
                v351 = 4l * v349;
                int v352;
                v352 = v351 + v151;
                assert("Tensor range check" && 0 <= v349 && v349 < 1l);
                int4* v353;
                v353 = reinterpret_cast<int4*>(v339 + v351);
                int4* v354;
                v354 = reinterpret_cast<int4*>(v150 + v352);
                assert("Pointer alignment check" && (unsigned long long)(v353) % 4l == 0 && (unsigned long long)(v354) % 4l == 0);
                *v354 = *v353;
                v349 += 1l ;
            }
            assert("Tensor range check" && 0 <= v138 && v138 < 32l);
            v117[v138] = v316;
            v127 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v355;
        v355 = threadIdx.x;
        assert("Tensor range check" && 0 <= v355 && v355 < 32l);
        float v356;
        v356 = v117[v355];
        assert("Tensor range check" && 0 <= v63 && v63 < 2l);
        v56[v63] = v356;
        cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v357 = console_lock;
        auto v358 = cooperative_groups::coalesced_threads();
        v357.acquire();
        int v359;
        v359 = 0l;
        printf("{%s = %c","rewards", '[');
        int v360;
        v360 = 0l;
        while (while_method_0(v360)){
            int v362;
            v362 = v359;
            bool v363;
            v363 = v362 >= 100l;
            if (v363){
                printf("%s"," ...");
                break;
            } else {
            }
            bool v364;
            v364 = v360 == 0l;
            bool v365;
            v365 = v364 != true;
            if (v365){
                printf("%s","; ");
            } else {
            }
            int v366;
            v366 = v359 + 1l;
            v359 = v366;
            float v367;
            v367 = v56[v360];
            printf("%f",v367);
            v360 += 1l ;
        }
        printf("%c",']');
        printf("}\n");
        v357.release();
        v358.sync() ;
    }
    int v386 = v31;
    int v387;
    v387 = threadIdx.x;
    int v388;
    v388 = v386;
    while (while_method_3(v388)){
        v388 -= 1l ;
        assert("Tensor range check" && 0 <= v388 && v388 < 16l);
        assert("Tensor range check" && 0 <= v387 && v387 < 32l);
        int v390;
        v390 = 32l * v388;
        int v391;
        v391 = v390 + v387;
        int v392;
        v392 = v12[v391];
        float v393;
        v393 = v14[v391];
        int v394;
        v394 = v16[v391];
        int v395;
        v395 = v18[v391];
        assert("Tensor range check" && 0 <= v388 && v388 < 16l);
        assert("Tensor range check" && 0 <= v387 && v387 < 32l);
        float v396;
        v396 = v26[v391];
        float v397;
        v397 = v28[v391];
        assert("Tensor range check" && 0 <= v395 && v395 < 4096l);
        int v398;
        v398 = 4l * v395;
        float * v399;
        v399 = v2+v398;
        float * v401;
        v401 = v4+v398;
        float * v403;
        v403 = v6+v398;
        float * v405;
        v405 = v8+v398;
        float * v407;
        v407 = v10+v398;
        assert("Tensor range check" && 0 <= v388 && v388 < 16l);
        int v409;
        v409 = 128l * v388;
        assert("Tensor range check" && 0 <= v387 && v387 < 32l);
        int v410;
        v410 = 4l * v387;
        int v411;
        v411 = v410 + v409;
        assert("Tensor range check" && 0 <= v392 && v392 < 4l);
        float * v412;
        v412 = v405+v392;
        float * v414;
        v414 = v407+v392;
        float v416;
        v416 = atomicAdd(v412,v396);
        float v417;
        v417 = atomicAdd(v414,v397);
        float * v418;
        v418 = v24+v411;
        __shared__ float * v420[32l];
        __shared__ float * v421[32l];
        /* void shared array create v422 */;
        /* void shared array create v423 */;
        int v424;
        v424 = threadIdx.x;
        assert("Tensor range check" && 0 <= v424 && v424 < 32l);
        v420[v424] = v403;
        v421[v424] = v418;
        /* void array set */;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        int v425;
        v425 = threadIdx.x;
        bool v426;
        v426 = 0l <= v425;
        bool v427;
        v427 = v426 == false;
        if (v427){
            assert("The index needs to be zero or positive." && v426);
        } else {
        }
        int v429;
        v429 = v425 % 1l;
        bool v430;
        v430 = v425 < 32l;
        bool v431;
        v431 = v430 == false;
        if (v431){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v430);
        } else {
        }
        assert("Tensor range check" && 0 <= v425 && v425 < 32l);
        int v433;
        v433 = 0l;
        while (while_method_1(v433)){
            bool v435;
            v435 = v426 && v430;
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
            v447 = v420[v446];
            float * v448;
            v448 = v421[v446];
            /* void array index */;
            assert("Tensor range check" && 0 <= v429 && v429 < 1l);
            int v449;
            v449 = 4l * v429;
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
                    v467 = 0l <= v429;
                    bool v469;
                    if (v467){
                        bool v468;
                        v468 = v429 < 1l;
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
                    v472 = v429 * 4l;
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
        int v496;
        v496 = threadIdx.x;
        assert("Tensor range check" && 0 <= v496 && v496 < 32l);
        /* void array index */;
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
    v0 = cp.empty(1458176,dtype=cp.uint8)
    v1 = cp.empty(0,dtype=cp.uint8)
    v3 = v0[0:0+4*65536].view(cp.float32)
    v5 = v0[262144:262144+4*65536].view(cp.float32)
    v7 = v0[524288:524288+4*65536].view(cp.float32)
    v9 = v0[786432:786432+4*65536].view(cp.float32)
    v11 = v0[1048576:1048576+4*65536].view(cp.float32)
    v13 = v0[1310720:1310720+4*2048].view(cp.int32)
    v15 = v0[1318912:1318912+4*2048].view(cp.float32)
    v17 = v0[1327104:1327104+4*2048].view(cp.int32)
    v19 = v0[1335296:1335296+4*2048].view(cp.int32)
    v21 = v0[1343488:1343488+8*4096].view(cp.float64)
    v23 = v0[1376256:1376256+8*4096].view(cp.float64)
    v25 = v0[1409024:1409024+4*8192].view(cp.float32)
    v27 = v0[1441792:1441792+4*2048].view(cp.float32)
    v29 = v0[1449984:1449984+4*2048].view(cp.float32)
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
