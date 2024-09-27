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
    v1 = v0 < 2048;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 64;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 32;
    return v1;
}
extern "C" __global__ void entry0(int * v0, float * v1, float * v2, int * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int * v9, int * v10, int * v11, int * v12, int * v13, float * v14, int * v15, float * v16, int * v17) {
    float v18;
    v18 = 0.0f;
    int v19;
    v19 = threadIdx.x;
    int v20;
    v20 = v19;
    while (while_method_0(v20)){
        bool v22;
        v22 = 0 <= v20;
        bool v23;
        v23 = v22 == false;
        if (v23){
            assert("The index needs to be zero or positive." && v22);
        } else {
        }
        int v25;
        v25 = v20 % 32;
        int v26;
        v26 = v20 / 32;
        bool v27;
        v27 = v26 < 64;
        bool v28;
        v28 = v27 == false;
        if (v28){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v27);
        } else {
        }
        assert("Tensor range check" && 0 <= v26 && v26 < 64);
        assert("Tensor range check" && 0 <= v25 && v25 < 32);
        int v30;
        v30 = 4 * v25;
        int v31;
        v31 = 128 * v26;
        int v32;
        v32 = v31 + v30;
        float v33[4];
        int4* v34;
        v34 = reinterpret_cast<int4*>(v1 + v32);
        int4* v35;
        v35 = reinterpret_cast<int4*>(v33 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v34) % 16 == 0 && reinterpret_cast<unsigned long long>(v35) % 16 == 0);
        *v35 = *v34;
        int v36; float v37;
        Tuple0 tmp0 = Tuple0{0, v18};
        v36 = tmp0.v0; v37 = tmp0.v1;
        while (while_method_1(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 4);
            float v39;
            v39 = v33[v36];
            float v40;
            v40 = v37 + v39;
            v37 = v40;
            v36 += 1 ;
        }
        v18 = v37;
        v20 += 256 ;
    }
    auto v41 = cooperative_groups::coalesced_threads();
    Closure0 v42{};
    float v43;
    v43 = cooperative_groups::reduce(v41, v18, v42);
    int v44;
    v44 = threadIdx.x;
    int v45;
    v45 = v44 / 32;
    extern __shared__ unsigned char v46[];
    float * v47;
    v47 = reinterpret_cast<float *>(&v46[0ull]);
    assert("Tensor range check" && 0 <= v45 && v45 < 8);
    v47[v45] = v43;
    __syncthreads();
    int v49;
    v49 = threadIdx.x;
    int v50;
    v50 = v49 % 32;
    bool v51;
    v51 = v45 == 0;
    bool v53;
    if (v51){
        bool v52;
        v52 = v50 < 8;
        v53 = v52;
    } else {
        v53 = false;
    }
    if (v53){
        auto v54 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v50 && v50 < 8);
        float v55;
        v55 = v47[v50];
        float v56;
        v56 = cooperative_groups::reduce(v54, v55, v42);
        v2[0] = v56;
    } else {
    }
    __syncthreads();
    int v57;
    v57 = threadIdx.x;
    bool v58;
    v58 = 0 <= v57;
    bool v59;
    v59 = v58 == false;
    if (v59){
        assert("The index needs to be zero or positive." && v58);
    } else {
    }
    int v61;
    v61 = v57 % 32;
    int v62;
    v62 = v57 / 32;
    bool v63;
    v63 = v62 < 8;
    bool v64;
    v64 = v63 == false;
    if (v64){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v63);
    } else {
    }
    assert("Tensor range check" && 0 <= v62 && v62 < 8);
    assert("Tensor range check" && 0 <= v61 && v61 < 32);
    int v66;
    v66 = 4 * v61;
    int v67;
    v67 = 128 * v62;
    int v68;
    v68 = v67 + v66;
    assert("Tensor range check" && 0 <= v62 && v62 < 8);
    assert("Tensor range check" && 0 <= v61 && v61 < 32);
    int v69;
    v69 = 0;
    while (while_method_2(v69)){
        assert("Tensor range check" && 0 <= v69 && v69 < 8);
        int v71;
        v71 = 1024 * v69;
        int v72;
        v72 = v71 + v68;
        int v73[4];
        int v74[4];
        int v75;
        v75 = 0;
        while (while_method_3(v75)){
            assert("Tensor range check" && 0 <= v75 && v75 < 1);
            int v77;
            v77 = 4 * v75;
            assert("Tensor range check" && 0 <= v75 && v75 < 1);
            int v78;
            v78 = 128 * v75;
            int v79;
            v79 = v78 + v72;
            int4* v80;
            v80 = reinterpret_cast<int4*>(v0 + v79);
            int4* v81;
            v81 = reinterpret_cast<int4*>(v73 + v77);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v80) % 16 == 0 && reinterpret_cast<unsigned long long>(v81) % 16 == 0);
            *v81 = *v80;
            v75 += 1 ;
        }
        int v82;
        v82 = 0;
        while (while_method_3(v82)){
            int v84;
            v84 = 0;
            while (while_method_1(v84)){
                bool v86;
                v86 = 0 <= v84;
                bool v88;
                if (v86){
                    bool v87;
                    v87 = v84 < 4;
                    v88 = v87;
                } else {
                    v88 = false;
                }
                bool v89;
                v89 = v88 == false;
                if (v89){
                    assert("The indices should be inside the range of the dimension." && v88);
                } else {
                }
                bool v91;
                v91 = 0 <= v61;
                bool v93;
                if (v91){
                    bool v92;
                    v92 = v61 < 32;
                    v93 = v92;
                } else {
                    v93 = false;
                }
                bool v94;
                v94 = v93 == false;
                if (v94){
                    assert("The indices should be inside the range of the dimension." && v93);
                } else {
                }
                int v96;
                v96 = v61 * 4;
                int v97;
                v97 = v84 + v96;
                bool v98;
                v98 = 0 <= v82;
                bool v100;
                if (v98){
                    bool v99;
                    v99 = v82 < 1;
                    v100 = v99;
                } else {
                    v100 = false;
                }
                bool v101;
                v101 = v100 == false;
                if (v101){
                    assert("The indices should be inside the range of the dimension." && v100);
                } else {
                }
                int v103;
                v103 = v82 * 128;
                int v104;
                v104 = v97 + v103;
                assert("Tensor range check" && 0 <= v82 && v82 < 1);
                assert("Tensor range check" && 0 <= v84 && v84 < 4);
                int v105;
                v105 = 4 * v82;
                int v106;
                v106 = v105 + v84;
                v74[v106] = v104;
                v84 += 1 ;
            }
            v82 += 1 ;
        }
        bool v107;
        v107 = 0 <= v62;
        bool v108;
        v108 = v107 && v63;
        bool v109;
        v109 = v108 == false;
        if (v109){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v108);
        } else {
        }
        bool v111;
        v111 = 0 <= v69;
        bool v113;
        if (v111){
            bool v112;
            v112 = v69 < 8;
            v113 = v112;
        } else {
            v113 = false;
        }
        bool v114;
        v114 = v113 == false;
        if (v114){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v113);
        } else {
        }
        int v116;
        v116 = v69 * 8;
        int v117;
        v117 = v116 + v62;
        assert("Tensor range check" && 0 <= v69 && v69 < 8);
        int v118;
        v118 = 0;
        while (while_method_3(v118)){
            assert("Tensor range check" && 0 <= v118 && v118 < 1);
            int v120;
            v120 = 128 * v118;
            int v121;
            v121 = v120 + v72;
            assert("Tensor range check" && 0 <= v118 && v118 < 1);
            int v122;
            v122 = 4 * v118;
            int4* v123;
            v123 = reinterpret_cast<int4*>(v73 + v122);
            int4* v124;
            v124 = reinterpret_cast<int4*>(v3 + v121);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v123) % 16 == 0 && reinterpret_cast<unsigned long long>(v124) % 16 == 0);
            *v124 = *v123;
            v118 += 1 ;
        }
        v69 += 1 ;
    }
    __syncthreads();
    int v125;
    v125 = threadIdx.x;
    bool v126;
    v126 = 0 <= v125;
    bool v127;
    v127 = v126 == false;
    if (v127){
        assert("The index needs to be zero or positive." && v126);
    } else {
    }
    int v129;
    v129 = v125 % 32;
    int v130;
    v130 = v125 / 32;
    bool v131;
    v131 = v130 < 8;
    bool v132;
    v132 = v131 == false;
    if (v132){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v131);
    } else {
    }
    assert("Tensor range check" && 0 <= v130 && v130 < 8);
    assert("Tensor range check" && 0 <= v129 && v129 < 32);
    int v134;
    v134 = 4 * v129;
    int v135;
    v135 = 128 * v130;
    int v136;
    v136 = v135 + v134;
    assert("Tensor range check" && 0 <= v130 && v130 < 8);
    assert("Tensor range check" && 0 <= v129 && v129 < 32);
    int v137;
    v137 = 0;
    while (while_method_2(v137)){
        assert("Tensor range check" && 0 <= v137 && v137 < 8);
        int v139;
        v139 = 1024 * v137;
        int v140;
        v140 = v139 + v136;
        float v141[4];
        int v142[4];
        int v143;
        v143 = 0;
        while (while_method_3(v143)){
            assert("Tensor range check" && 0 <= v143 && v143 < 1);
            int v145;
            v145 = 4 * v143;
            assert("Tensor range check" && 0 <= v143 && v143 < 1);
            int v146;
            v146 = 128 * v143;
            int v147;
            v147 = v146 + v140;
            int4* v148;
            v148 = reinterpret_cast<int4*>(v1 + v147);
            int4* v149;
            v149 = reinterpret_cast<int4*>(v141 + v145);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v148) % 16 == 0 && reinterpret_cast<unsigned long long>(v149) % 16 == 0);
            *v149 = *v148;
            v143 += 1 ;
        }
        int v150;
        v150 = 0;
        while (while_method_3(v150)){
            int v152;
            v152 = 0;
            while (while_method_1(v152)){
                bool v154;
                v154 = 0 <= v152;
                bool v156;
                if (v154){
                    bool v155;
                    v155 = v152 < 4;
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
                bool v159;
                v159 = 0 <= v129;
                bool v161;
                if (v159){
                    bool v160;
                    v160 = v129 < 32;
                    v161 = v160;
                } else {
                    v161 = false;
                }
                bool v162;
                v162 = v161 == false;
                if (v162){
                    assert("The indices should be inside the range of the dimension." && v161);
                } else {
                }
                int v164;
                v164 = v129 * 4;
                int v165;
                v165 = v152 + v164;
                bool v166;
                v166 = 0 <= v150;
                bool v168;
                if (v166){
                    bool v167;
                    v167 = v150 < 1;
                    v168 = v167;
                } else {
                    v168 = false;
                }
                bool v169;
                v169 = v168 == false;
                if (v169){
                    assert("The indices should be inside the range of the dimension." && v168);
                } else {
                }
                int v171;
                v171 = v150 * 128;
                int v172;
                v172 = v165 + v171;
                assert("Tensor range check" && 0 <= v150 && v150 < 1);
                assert("Tensor range check" && 0 <= v152 && v152 < 4);
                int v173;
                v173 = 4 * v150;
                int v174;
                v174 = v173 + v152;
                v142[v174] = v172;
                v152 += 1 ;
            }
            v150 += 1 ;
        }
        bool v175;
        v175 = 0 <= v130;
        bool v176;
        v176 = v175 && v131;
        bool v177;
        v177 = v176 == false;
        if (v177){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v176);
        } else {
        }
        bool v179;
        v179 = 0 <= v137;
        bool v181;
        if (v179){
            bool v180;
            v180 = v137 < 8;
            v181 = v180;
        } else {
            v181 = false;
        }
        bool v182;
        v182 = v181 == false;
        if (v182){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v181);
        } else {
        }
        int v184;
        v184 = v137 * 8;
        int v185;
        v185 = v184 + v130;
        int v186[4];
        int v187[4];
        int v188;
        v188 = 0;
        while (while_method_3(v188)){
            int v190;
            v190 = 0;
            while (while_method_1(v190)){
                assert("Tensor range check" && 0 <= v188 && v188 < 1);
                assert("Tensor range check" && 0 <= v190 && v190 < 4);
                int v192;
                v192 = 4 * v188;
                int v193;
                v193 = v192 + v190;
                int v194;
                v194 = v142[v193];
                assert("Tensor range check" && 0 <= v188 && v188 < 1);
                assert("Tensor range check" && 0 <= v190 && v190 < 4);
                v186[v193] = v185;
                v187[v193] = v194;
                v190 += 1 ;
            }
            v188 += 1 ;
        }
        assert("Tensor range check" && 0 <= v137 && v137 < 8);
        int v195;
        v195 = 0;
        while (while_method_3(v195)){
            assert("Tensor range check" && 0 <= v195 && v195 < 1);
            int v197;
            v197 = 128 * v195;
            int v198;
            v198 = v197 + v140;
            assert("Tensor range check" && 0 <= v195 && v195 < 1);
            int v199;
            v199 = 4 * v195;
            int4* v200;
            v200 = reinterpret_cast<int4*>(v186 + v199);
            int4* v201;
            v201 = reinterpret_cast<int4*>(v10 + v198);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v200) % 16 == 0 && reinterpret_cast<unsigned long long>(v201) % 16 == 0);
            *v201 = *v200;
            int4* v202;
            v202 = reinterpret_cast<int4*>(v187 + v199);
            int4* v203;
            v203 = reinterpret_cast<int4*>(v11 + v198);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v202) % 16 == 0 && reinterpret_cast<unsigned long long>(v203) % 16 == 0);
            *v203 = *v202;
            v195 += 1 ;
        }
        v137 += 1 ;
    }
    __syncthreads();
    int v204;
    v204 = threadIdx.x;
    bool v205;
    v205 = 0 <= v204;
    bool v206;
    v206 = v205 == false;
    if (v206){
        assert("The index needs to be zero or positive." && v205);
    } else {
    }
    int v208;
    v208 = v204 % 32;
    int v209;
    v209 = v204 / 32;
    bool v210;
    v210 = v209 < 8;
    bool v211;
    v211 = v210 == false;
    if (v211){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v210);
    } else {
    }
    assert("Tensor range check" && 0 <= v209 && v209 < 8);
    assert("Tensor range check" && 0 <= v208 && v208 < 32);
    int v213;
    v213 = 4 * v208;
    int v214;
    v214 = 128 * v209;
    int v215;
    v215 = v214 + v213;
    assert("Tensor range check" && 0 <= v209 && v209 < 8);
    int v216;
    v216 = 0;
    while (while_method_2(v216)){
        assert("Tensor range check" && 0 <= v216 && v216 < 8);
        int v218;
        v218 = 1024 * v216;
        int v219;
        v219 = v218 + v215;
        float v220[4];
        int v221[4];
        int v222;
        v222 = 0;
        while (while_method_3(v222)){
            assert("Tensor range check" && 0 <= v222 && v222 < 1);
            int v224;
            v224 = 4 * v222;
            assert("Tensor range check" && 0 <= v222 && v222 < 1);
            int v225;
            v225 = 128 * v222;
            int v226;
            v226 = v225 + v219;
            int4* v227;
            v227 = reinterpret_cast<int4*>(v1 + v226);
            int4* v228;
            v228 = reinterpret_cast<int4*>(v220 + v224);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v227) % 16 == 0 && reinterpret_cast<unsigned long long>(v228) % 16 == 0);
            *v228 = *v227;
            v222 += 1 ;
        }
        int v229;
        v229 = 0;
        while (while_method_3(v229)){
            int v231;
            v231 = 0;
            while (while_method_1(v231)){
                bool v233;
                v233 = 0 <= v231;
                bool v235;
                if (v233){
                    bool v234;
                    v234 = v231 < 4;
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
                v238 = 0 <= v208;
                bool v240;
                if (v238){
                    bool v239;
                    v239 = v208 < 32;
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
                v243 = v208 * 4;
                int v244;
                v244 = v231 + v243;
                bool v245;
                v245 = 0 <= v229;
                bool v247;
                if (v245){
                    bool v246;
                    v246 = v229 < 1;
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
                v250 = v229 * 128;
                int v251;
                v251 = v244 + v250;
                assert("Tensor range check" && 0 <= v229 && v229 < 1);
                assert("Tensor range check" && 0 <= v231 && v231 < 4);
                int v252;
                v252 = 4 * v229;
                int v253;
                v253 = v252 + v231;
                v221[v253] = v251;
                v231 += 1 ;
            }
            v229 += 1 ;
        }
        bool v254;
        v254 = 0 <= v209;
        bool v255;
        v255 = v254 && v210;
        bool v256;
        v256 = v255 == false;
        if (v256){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v255);
        } else {
        }
        bool v258;
        v258 = 0 <= v216;
        bool v260;
        if (v258){
            bool v259;
            v259 = v216 < 8;
            v260 = v259;
        } else {
            v260 = false;
        }
        bool v261;
        v261 = v260 == false;
        if (v261){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v260);
        } else {
        }
        int v263;
        v263 = v216 * 8;
        int v264;
        v264 = v263 + v209;
        assert("Tensor range check" && 0 <= v216 && v216 < 8);
        int v265;
        v265 = 8 * v216;
        int v266;
        v266 = v265 + v209;
        v12[v266] = v264;
        v216 += 1 ;
    }
    __syncthreads();
    int v267;
    v267 = threadIdx.x;
    bool v268;
    v268 = 0 <= v267;
    bool v269;
    v269 = v268 == false;
    if (v269){
        assert("The index needs to be zero or positive." && v268);
    } else {
    }
    int v271;
    v271 = v267 % 32;
    int v272;
    v272 = v267 / 32;
    bool v273;
    v273 = v272 < 8;
    bool v274;
    v274 = v273 == false;
    if (v274){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v273);
    } else {
    }
    assert("Tensor range check" && 0 <= v272 && v272 < 8);
    assert("Tensor range check" && 0 <= v271 && v271 < 32);
    int v276;
    v276 = 4 * v271;
    int v277;
    v277 = 128 * v272;
    int v278;
    v278 = v277 + v276;
    assert("Tensor range check" && 0 <= v272 && v272 < 8);
    assert("Tensor range check" && 0 <= v271 && v271 < 32);
    int v279;
    v279 = 0;
    while (while_method_2(v279)){
        assert("Tensor range check" && 0 <= v279 && v279 < 8);
        int v281;
        v281 = 1024 * v279;
        int v282;
        v282 = v281 + v278;
        float v283[4];
        int v284[4];
        int v285;
        v285 = 0;
        while (while_method_3(v285)){
            assert("Tensor range check" && 0 <= v285 && v285 < 1);
            int v287;
            v287 = 4 * v285;
            assert("Tensor range check" && 0 <= v285 && v285 < 1);
            int v288;
            v288 = 128 * v285;
            int v289;
            v289 = v288 + v282;
            int4* v290;
            v290 = reinterpret_cast<int4*>(v1 + v289);
            int4* v291;
            v291 = reinterpret_cast<int4*>(v283 + v287);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v290) % 16 == 0 && reinterpret_cast<unsigned long long>(v291) % 16 == 0);
            *v291 = *v290;
            v285 += 1 ;
        }
        int v292;
        v292 = 0;
        while (while_method_3(v292)){
            int v294;
            v294 = 0;
            while (while_method_1(v294)){
                bool v296;
                v296 = 0 <= v294;
                bool v298;
                if (v296){
                    bool v297;
                    v297 = v294 < 4;
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
                v301 = 0 <= v271;
                bool v303;
                if (v301){
                    bool v302;
                    v302 = v271 < 32;
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
                v306 = v271 * 4;
                int v307;
                v307 = v294 + v306;
                bool v308;
                v308 = 0 <= v292;
                bool v310;
                if (v308){
                    bool v309;
                    v309 = v292 < 1;
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
                v313 = v292 * 128;
                int v314;
                v314 = v307 + v313;
                assert("Tensor range check" && 0 <= v292 && v292 < 1);
                assert("Tensor range check" && 0 <= v294 && v294 < 4);
                int v315;
                v315 = 4 * v292;
                int v316;
                v316 = v315 + v294;
                v284[v316] = v314;
                v294 += 1 ;
            }
            v292 += 1 ;
        }
        bool v317;
        v317 = 0 <= v272;
        bool v318;
        v318 = v317 && v273;
        bool v319;
        v319 = v318 == false;
        if (v319){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v318);
        } else {
        }
        bool v321;
        v321 = 0 <= v279;
        bool v323;
        if (v321){
            bool v322;
            v322 = v279 < 8;
            v323 = v322;
        } else {
            v323 = false;
        }
        bool v324;
        v324 = v323 == false;
        if (v324){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v323);
        } else {
        }
        int v326;
        v326 = v279 * 8;
        int v327;
        v327 = v326 + v272;
        float v328;
        v328 = 0.0f;
        int v329;
        v329 = 0;
        while (while_method_3(v329)){
            int v331;
            v331 = 0;
            while (while_method_1(v331)){
                assert("Tensor range check" && 0 <= v329 && v329 < 1);
                assert("Tensor range check" && 0 <= v331 && v331 < 4);
                int v333;
                v333 = 4 * v329;
                int v334;
                v334 = v333 + v331;
                float v335;
                v335 = v283[v334];
                float v336;
                v336 = v328 + v335;
                v328 = v336;
                v331 += 1 ;
            }
            v329 += 1 ;
        }
        auto v337 = cooperative_groups::coalesced_threads();
        int v338;
        v338 = threadIdx.x;
        int v339;
        v339 = v338 / 32;
        auto v340 = cooperative_groups::labeled_partition(v337,v339);
        float v341;
        v341 = cooperative_groups::reduce(v340, v328, v42);
        float v342;
        v342 = v341 / 128.0f;
        float v343[4];
        int v344;
        v344 = 0;
        while (while_method_3(v344)){
            int v346;
            v346 = 0;
            while (while_method_1(v346)){
                assert("Tensor range check" && 0 <= v344 && v344 < 1);
                assert("Tensor range check" && 0 <= v346 && v346 < 4);
                int v348;
                v348 = 4 * v344;
                int v349;
                v349 = v348 + v346;
                float v350;
                v350 = v283[v349];
                float v351;
                v351 = v350 - v342;
                float v352;
                v352 = exp(v351);
                assert("Tensor range check" && 0 <= v344 && v344 < 1);
                assert("Tensor range check" && 0 <= v346 && v346 < 4);
                v343[v349] = v352;
                v346 += 1 ;
            }
            v344 += 1 ;
        }
        float v353;
        v353 = 0.0f;
        int v354;
        v354 = 0;
        while (while_method_3(v354)){
            int v356;
            v356 = 0;
            while (while_method_1(v356)){
                assert("Tensor range check" && 0 <= v354 && v354 < 1);
                assert("Tensor range check" && 0 <= v356 && v356 < 4);
                int v358;
                v358 = 4 * v354;
                int v359;
                v359 = v358 + v356;
                float v360;
                v360 = v343[v359];
                float v361;
                v361 = v353 + v360;
                v353 = v361;
                v356 += 1 ;
            }
            v354 += 1 ;
        }
        auto v362 = cooperative_groups::coalesced_threads();
        int v363;
        v363 = threadIdx.x;
        int v364;
        v364 = v363 / 32;
        auto v365 = cooperative_groups::labeled_partition(v362,v364);
        float v366;
        v366 = cooperative_groups::reduce(v365, v353, v42);
        float v367[4];
        int v368;
        v368 = 0;
        while (while_method_3(v368)){
            int v370;
            v370 = 0;
            while (while_method_1(v370)){
                assert("Tensor range check" && 0 <= v368 && v368 < 1);
                assert("Tensor range check" && 0 <= v370 && v370 < 4);
                int v372;
                v372 = 4 * v368;
                int v373;
                v373 = v372 + v370;
                float v374;
                v374 = v343[v373];
                float v375;
                v375 = v374 / v366;
                assert("Tensor range check" && 0 <= v368 && v368 < 1);
                assert("Tensor range check" && 0 <= v370 && v370 < 4);
                v367[v373] = v375;
                v370 += 1 ;
            }
            v368 += 1 ;
        }
        assert("Tensor range check" && 0 <= v279 && v279 < 8);
        int v376;
        v376 = 0;
        while (while_method_3(v376)){
            assert("Tensor range check" && 0 <= v376 && v376 < 1);
            int v378;
            v378 = 128 * v376;
            int v379;
            v379 = v378 + v282;
            assert("Tensor range check" && 0 <= v376 && v376 < 1);
            int v380;
            v380 = 4 * v376;
            int4* v381;
            v381 = reinterpret_cast<int4*>(v367 + v380);
            int4* v382;
            v382 = reinterpret_cast<int4*>(v4 + v379);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v381) % 16 == 0 && reinterpret_cast<unsigned long long>(v382) % 16 == 0);
            *v382 = *v381;
            v376 += 1 ;
        }
        v279 += 1 ;
    }
    __syncthreads();
    int v383;
    v383 = threadIdx.x;
    bool v384;
    v384 = 0 <= v383;
    bool v385;
    v385 = v384 == false;
    if (v385){
        assert("The index needs to be zero or positive." && v384);
    } else {
    }
    int v387;
    v387 = v383 % 32;
    int v388;
    v388 = v383 / 32;
    bool v389;
    v389 = v388 < 8;
    bool v390;
    v390 = v389 == false;
    if (v390){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v389);
    } else {
    }
    assert("Tensor range check" && 0 <= v388 && v388 < 8);
    assert("Tensor range check" && 0 <= v387 && v387 < 32);
    int v392;
    v392 = 4 * v387;
    int v393;
    v393 = 128 * v388;
    int v394;
    v394 = v393 + v392;
    assert("Tensor range check" && 0 <= v388 && v388 < 8);
    assert("Tensor range check" && 0 <= v387 && v387 < 32);
    int v395;
    v395 = 0;
    while (while_method_2(v395)){
        assert("Tensor range check" && 0 <= v395 && v395 < 8);
        int v397;
        v397 = 1024 * v395;
        int v398;
        v398 = v397 + v394;
        float v399[4];
        int v400[4];
        int v401;
        v401 = 0;
        while (while_method_3(v401)){
            assert("Tensor range check" && 0 <= v401 && v401 < 1);
            int v403;
            v403 = 4 * v401;
            assert("Tensor range check" && 0 <= v401 && v401 < 1);
            int v404;
            v404 = 128 * v401;
            int v405;
            v405 = v404 + v398;
            int4* v406;
            v406 = reinterpret_cast<int4*>(v1 + v405);
            int4* v407;
            v407 = reinterpret_cast<int4*>(v399 + v403);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v406) % 16 == 0 && reinterpret_cast<unsigned long long>(v407) % 16 == 0);
            *v407 = *v406;
            v401 += 1 ;
        }
        int v408;
        v408 = 0;
        while (while_method_3(v408)){
            int v410;
            v410 = 0;
            while (while_method_1(v410)){
                bool v412;
                v412 = 0 <= v410;
                bool v414;
                if (v412){
                    bool v413;
                    v413 = v410 < 4;
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
                v417 = 0 <= v387;
                bool v419;
                if (v417){
                    bool v418;
                    v418 = v387 < 32;
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
                v422 = v387 * 4;
                int v423;
                v423 = v410 + v422;
                bool v424;
                v424 = 0 <= v408;
                bool v426;
                if (v424){
                    bool v425;
                    v425 = v408 < 1;
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
                v429 = v408 * 128;
                int v430;
                v430 = v423 + v429;
                assert("Tensor range check" && 0 <= v408 && v408 < 1);
                assert("Tensor range check" && 0 <= v410 && v410 < 4);
                int v431;
                v431 = 4 * v408;
                int v432;
                v432 = v431 + v410;
                v400[v432] = v430;
                v410 += 1 ;
            }
            v408 += 1 ;
        }
        bool v433;
        v433 = 0 <= v388;
        bool v434;
        v434 = v433 && v389;
        bool v435;
        v435 = v434 == false;
        if (v435){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v434);
        } else {
        }
        bool v437;
        v437 = 0 <= v395;
        bool v439;
        if (v437){
            bool v438;
            v438 = v395 < 8;
            v439 = v438;
        } else {
            v439 = false;
        }
        bool v440;
        v440 = v439 == false;
        if (v440){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v439);
        } else {
        }
        int v442;
        v442 = v395 * 8;
        int v443;
        v443 = v442 + v388;
        float v444[4];
        int v445;
        v445 = 0;
        while (while_method_3(v445)){
            int v447;
            v447 = 0;
            while (while_method_1(v447)){
                assert("Tensor range check" && 0 <= v445 && v445 < 1);
                assert("Tensor range check" && 0 <= v447 && v447 < 4);
                int v449;
                v449 = 4 * v445;
                int v450;
                v450 = v449 + v447;
                float v451;
                v451 = v399[v450];
                float v452;
                v452 = v451 * v451;
                assert("Tensor range check" && 0 <= v445 && v445 < 1);
                assert("Tensor range check" && 0 <= v447 && v447 < 4);
                v444[v450] = v452;
                v447 += 1 ;
            }
            v445 += 1 ;
        }
        float v453;
        v453 = 0.0f;
        int v454;
        v454 = 0;
        while (while_method_3(v454)){
            int v456;
            v456 = 0;
            while (while_method_1(v456)){
                assert("Tensor range check" && 0 <= v454 && v454 < 1);
                assert("Tensor range check" && 0 <= v456 && v456 < 4);
                int v458;
                v458 = 4 * v454;
                int v459;
                v459 = v458 + v456;
                float v460;
                v460 = v444[v459];
                float v461;
                v461 = v453 + v460;
                v453 = v461;
                v456 += 1 ;
            }
            v454 += 1 ;
        }
        auto v462 = cooperative_groups::coalesced_threads();
        int v463;
        v463 = threadIdx.x;
        int v464;
        v464 = v463 / 32;
        auto v465 = cooperative_groups::labeled_partition(v462,v464);
        float v466;
        v466 = cooperative_groups::reduce(v465, v453, v42);
        float v467[4];
        int v468;
        v468 = 0;
        while (while_method_3(v468)){
            int v470;
            v470 = 0;
            while (while_method_1(v470)){
                assert("Tensor range check" && 0 <= v468 && v468 < 1);
                assert("Tensor range check" && 0 <= v470 && v470 < 4);
                int v472;
                v472 = 4 * v468;
                int v473;
                v473 = v472 + v470;
                float v474;
                v474 = v399[v473];
                bool v475;
                v475 = v466 == 0.0f;
                bool v476;
                v476 = v475 != true;
                float v478;
                if (v476){
                    float v477;
                    v477 = v474 / v466;
                    v478 = v477;
                } else {
                    v478 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v468 && v468 < 1);
                assert("Tensor range check" && 0 <= v470 && v470 < 4);
                v467[v473] = v478;
                v470 += 1 ;
            }
            v468 += 1 ;
        }
        assert("Tensor range check" && 0 <= v395 && v395 < 8);
        int v479;
        v479 = 0;
        while (while_method_3(v479)){
            assert("Tensor range check" && 0 <= v479 && v479 < 1);
            int v481;
            v481 = 128 * v479;
            int v482;
            v482 = v481 + v398;
            assert("Tensor range check" && 0 <= v479 && v479 < 1);
            int v483;
            v483 = 4 * v479;
            int4* v484;
            v484 = reinterpret_cast<int4*>(v467 + v483);
            int4* v485;
            v485 = reinterpret_cast<int4*>(v8 + v482);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v484) % 16 == 0 && reinterpret_cast<unsigned long long>(v485) % 16 == 0);
            *v485 = *v484;
            v479 += 1 ;
        }
        v395 += 1 ;
    }
    __syncthreads();
    int v486;
    v486 = threadIdx.x;
    bool v487;
    v487 = 0 <= v486;
    bool v488;
    v488 = v487 == false;
    if (v488){
        assert("The index needs to be zero or positive." && v487);
    } else {
    }
    int v490;
    v490 = v486 % 32;
    int v491;
    v491 = v486 / 32;
    bool v492;
    v492 = v491 < 8;
    bool v493;
    v493 = v492 == false;
    if (v493){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v492);
    } else {
    }
    assert("Tensor range check" && 0 <= v491 && v491 < 8);
    assert("Tensor range check" && 0 <= v490 && v490 < 32);
    int v495;
    v495 = 4 * v490;
    int v496;
    v496 = 128 * v491;
    int v497;
    v497 = v496 + v495;
    assert("Tensor range check" && 0 <= v491 && v491 < 8);
    int v498;
    v498 = 0;
    while (while_method_2(v498)){
        assert("Tensor range check" && 0 <= v498 && v498 < 8);
        int v500;
        v500 = 1024 * v498;
        int v501;
        v501 = v500 + v497;
        float v502[4];
        int v503[4];
        int v504;
        v504 = 0;
        while (while_method_3(v504)){
            assert("Tensor range check" && 0 <= v504 && v504 < 1);
            int v506;
            v506 = 4 * v504;
            assert("Tensor range check" && 0 <= v504 && v504 < 1);
            int v507;
            v507 = 128 * v504;
            int v508;
            v508 = v507 + v501;
            int4* v509;
            v509 = reinterpret_cast<int4*>(v1 + v508);
            int4* v510;
            v510 = reinterpret_cast<int4*>(v502 + v506);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v509) % 16 == 0 && reinterpret_cast<unsigned long long>(v510) % 16 == 0);
            *v510 = *v509;
            v504 += 1 ;
        }
        int v511;
        v511 = 0;
        while (while_method_3(v511)){
            int v513;
            v513 = 0;
            while (while_method_1(v513)){
                bool v515;
                v515 = 0 <= v513;
                bool v517;
                if (v515){
                    bool v516;
                    v516 = v513 < 4;
                    v517 = v516;
                } else {
                    v517 = false;
                }
                bool v518;
                v518 = v517 == false;
                if (v518){
                    assert("The indices should be inside the range of the dimension." && v517);
                } else {
                }
                bool v520;
                v520 = 0 <= v490;
                bool v522;
                if (v520){
                    bool v521;
                    v521 = v490 < 32;
                    v522 = v521;
                } else {
                    v522 = false;
                }
                bool v523;
                v523 = v522 == false;
                if (v523){
                    assert("The indices should be inside the range of the dimension." && v522);
                } else {
                }
                int v525;
                v525 = v490 * 4;
                int v526;
                v526 = v513 + v525;
                bool v527;
                v527 = 0 <= v511;
                bool v529;
                if (v527){
                    bool v528;
                    v528 = v511 < 1;
                    v529 = v528;
                } else {
                    v529 = false;
                }
                bool v530;
                v530 = v529 == false;
                if (v530){
                    assert("The indices should be inside the range of the dimension." && v529);
                } else {
                }
                int v532;
                v532 = v511 * 128;
                int v533;
                v533 = v526 + v532;
                assert("Tensor range check" && 0 <= v511 && v511 < 1);
                assert("Tensor range check" && 0 <= v513 && v513 < 4);
                int v534;
                v534 = 4 * v511;
                int v535;
                v535 = v534 + v513;
                v503[v535] = v533;
                v513 += 1 ;
            }
            v511 += 1 ;
        }
        bool v536;
        v536 = 0 <= v491;
        bool v537;
        v537 = v536 && v492;
        bool v538;
        v538 = v537 == false;
        if (v538){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v537);
        } else {
        }
        bool v540;
        v540 = 0 <= v498;
        bool v542;
        if (v540){
            bool v541;
            v541 = v498 < 8;
            v542 = v541;
        } else {
            v542 = false;
        }
        bool v543;
        v543 = v542 == false;
        if (v543){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v542);
        } else {
        }
        int v545;
        v545 = v498 * 8;
        int v546;
        v546 = v545 + v491;
        float v547; int v548;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0};
        v547 = tmp1.v0; v548 = tmp1.v1;
        int v549;
        v549 = 0;
        while (while_method_3(v549)){
            int v551;
            v551 = 0;
            while (while_method_1(v551)){
                assert("Tensor range check" && 0 <= v549 && v549 < 1);
                assert("Tensor range check" && 0 <= v551 && v551 < 4);
                int v553;
                v553 = 4 * v549;
                int v554;
                v554 = v553 + v551;
                float v555;
                v555 = v502[v554];
                int v556;
                v556 = v503[v554];
                bool v557;
                v557 = v547 > v555;
                float v558; int v559;
                if (v557){
                    v558 = v547; v559 = v548;
                } else {
                    v558 = v555; v559 = v556;
                }
                v547 = v558;
                v548 = v559;
                v551 += 1 ;
            }
            v549 += 1 ;
        }
        auto v560 = cooperative_groups::coalesced_threads();
        int v561;
        v561 = threadIdx.x;
        int v562;
        v562 = v561 / 32;
        auto v563 = cooperative_groups::labeled_partition(v560,v562);
        Closure1 v564{};
        float v565; int v566;
        Tuple1 tmp2 = cooperative_groups::reduce(v563, Tuple1{v547, v548}, v564);
        v565 = tmp2.v0; v566 = tmp2.v1;
        assert("Tensor range check" && 0 <= v498 && v498 < 8);
        int v567;
        v567 = 8 * v498;
        int v568;
        v568 = v567 + v491;
        v9[v568] = v566;
        v498 += 1 ;
    }
    __syncthreads();
    int v569;
    v569 = threadIdx.x;
    bool v570;
    v570 = 0 <= v569;
    bool v571;
    v571 = v570 == false;
    if (v571){
        assert("The index needs to be zero or positive." && v570);
    } else {
    }
    int v573;
    v573 = v569 % 32;
    int v574;
    v574 = v569 / 32;
    bool v575;
    v575 = v574 < 8;
    bool v576;
    v576 = v575 == false;
    if (v576){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v575);
    } else {
    }
    assert("Tensor range check" && 0 <= v574 && v574 < 8);
    assert("Tensor range check" && 0 <= v573 && v573 < 32);
    int v578;
    v578 = 4 * v573;
    int v579;
    v579 = 128 * v574;
    int v580;
    v580 = v579 + v578;
    assert("Tensor range check" && 0 <= v574 && v574 < 8);
    assert("Tensor range check" && 0 <= v573 && v573 < 32);
    int v581;
    v581 = 0;
    while (while_method_2(v581)){
        assert("Tensor range check" && 0 <= v581 && v581 < 8);
        int v583;
        v583 = 1024 * v581;
        int v584;
        v584 = v583 + v580;
        float v585[4];
        int v586[4];
        int v587;
        v587 = 0;
        while (while_method_3(v587)){
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v589;
            v589 = 4 * v587;
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v590;
            v590 = 128 * v587;
            int v591;
            v591 = v590 + v584;
            int4* v592;
            v592 = reinterpret_cast<int4*>(v1 + v591);
            int4* v593;
            v593 = reinterpret_cast<int4*>(v585 + v589);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v592) % 16 == 0 && reinterpret_cast<unsigned long long>(v593) % 16 == 0);
            *v593 = *v592;
            v587 += 1 ;
        }
        int v594;
        v594 = 0;
        while (while_method_3(v594)){
            int v596;
            v596 = 0;
            while (while_method_1(v596)){
                bool v598;
                v598 = 0 <= v596;
                bool v600;
                if (v598){
                    bool v599;
                    v599 = v596 < 4;
                    v600 = v599;
                } else {
                    v600 = false;
                }
                bool v601;
                v601 = v600 == false;
                if (v601){
                    assert("The indices should be inside the range of the dimension." && v600);
                } else {
                }
                bool v603;
                v603 = 0 <= v573;
                bool v605;
                if (v603){
                    bool v604;
                    v604 = v573 < 32;
                    v605 = v604;
                } else {
                    v605 = false;
                }
                bool v606;
                v606 = v605 == false;
                if (v606){
                    assert("The indices should be inside the range of the dimension." && v605);
                } else {
                }
                int v608;
                v608 = v573 * 4;
                int v609;
                v609 = v596 + v608;
                bool v610;
                v610 = 0 <= v594;
                bool v612;
                if (v610){
                    bool v611;
                    v611 = v594 < 1;
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
                int v615;
                v615 = v594 * 128;
                int v616;
                v616 = v609 + v615;
                assert("Tensor range check" && 0 <= v594 && v594 < 1);
                assert("Tensor range check" && 0 <= v596 && v596 < 4);
                int v617;
                v617 = 4 * v594;
                int v618;
                v618 = v617 + v596;
                v586[v618] = v616;
                v596 += 1 ;
            }
            v594 += 1 ;
        }
        bool v619;
        v619 = 0 <= v574;
        bool v620;
        v620 = v619 && v575;
        bool v621;
        v621 = v620 == false;
        if (v621){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v620);
        } else {
        }
        bool v623;
        v623 = 0 <= v581;
        bool v625;
        if (v623){
            bool v624;
            v624 = v581 < 8;
            v625 = v624;
        } else {
            v625 = false;
        }
        bool v626;
        v626 = v625 == false;
        if (v626){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v625);
        } else {
        }
        int v628;
        v628 = v581 * 8;
        int v629;
        v629 = v628 + v574;
        float v630;
        v630 = 0.0f;
        int v631;
        v631 = 0;
        while (while_method_3(v631)){
            int v633;
            v633 = 0;
            while (while_method_1(v633)){
                assert("Tensor range check" && 0 <= v631 && v631 < 1);
                assert("Tensor range check" && 0 <= v633 && v633 < 4);
                int v635;
                v635 = 4 * v631;
                int v636;
                v636 = v635 + v633;
                float v637;
                v637 = v585[v636];
                float v638;
                v638 = v630 + v637;
                v630 = v638;
                v633 += 1 ;
            }
            v631 += 1 ;
        }
        auto v639 = cooperative_groups::coalesced_threads();
        int v640;
        v640 = threadIdx.x;
        int v641;
        v641 = v640 / 32;
        auto v642 = cooperative_groups::labeled_partition(v639,v641);
        float v643;
        v643 = cooperative_groups::reduce(v642, v630, v42);
        float v644;
        v644 = v643 / 128.0f;
        float v645[4];
        int v646;
        v646 = 0;
        while (while_method_3(v646)){
            int v648;
            v648 = 0;
            while (while_method_1(v648)){
                assert("Tensor range check" && 0 <= v646 && v646 < 1);
                assert("Tensor range check" && 0 <= v648 && v648 < 4);
                int v650;
                v650 = 4 * v646;
                int v651;
                v651 = v650 + v648;
                float v652;
                v652 = v585[v651];
                float v653;
                v653 = v652 - v644;
                float v654;
                v654 = exp(v653);
                assert("Tensor range check" && 0 <= v646 && v646 < 1);
                assert("Tensor range check" && 0 <= v648 && v648 < 4);
                v645[v651] = v654;
                v648 += 1 ;
            }
            v646 += 1 ;
        }
        float v655;
        v655 = 0.0f;
        int v656;
        v656 = 0;
        while (while_method_3(v656)){
            int v658;
            v658 = 0;
            while (while_method_1(v658)){
                assert("Tensor range check" && 0 <= v656 && v656 < 1);
                assert("Tensor range check" && 0 <= v658 && v658 < 4);
                int v660;
                v660 = 4 * v656;
                int v661;
                v661 = v660 + v658;
                float v662;
                v662 = v645[v661];
                float v663;
                v663 = v655 + v662;
                v655 = v663;
                v658 += 1 ;
            }
            v656 += 1 ;
        }
        auto v664 = cooperative_groups::coalesced_threads();
        int v665;
        v665 = threadIdx.x;
        int v666;
        v666 = v665 / 32;
        auto v667 = cooperative_groups::labeled_partition(v664,v666);
        float v668;
        v668 = cooperative_groups::reduce(v667, v655, v42);
        float v669[4];
        int v670;
        v670 = 0;
        while (while_method_3(v670)){
            int v672;
            v672 = 0;
            while (while_method_1(v672)){
                assert("Tensor range check" && 0 <= v670 && v670 < 1);
                assert("Tensor range check" && 0 <= v672 && v672 < 4);
                int v674;
                v674 = 4 * v670;
                int v675;
                v675 = v674 + v672;
                float v676;
                v676 = v645[v675];
                float v677;
                v677 = v676 / v668;
                assert("Tensor range check" && 0 <= v670 && v670 < 1);
                assert("Tensor range check" && 0 <= v672 && v672 < 4);
                v669[v675] = v677;
                v672 += 1 ;
            }
            v670 += 1 ;
        }
        float v678[4];
        float v679;
        v679 = 0.0f;
        int v680;
        v680 = 0;
        while (while_method_3(v680)){
            assert("Tensor range check" && 0 <= v680 && v680 < 1);
            int v682;
            v682 = 4 * v680;
            assert("Tensor range check" && 0 <= v680 && v680 < 1);
            int v683; float v684;
            Tuple0 tmp3 = Tuple0{0, 0.0f};
            v683 = tmp3.v0; v684 = tmp3.v1;
            while (while_method_1(v683)){
                assert("Tensor range check" && 0 <= v683 && v683 < 4);
                int v686;
                v686 = v683 + v682;
                float v687;
                v687 = v669[v686];
                float v688;
                v688 = v684 + v687;
                v684 = v688;
                v683 += 1 ;
            }
            auto v689 = cooperative_groups::coalesced_threads();
            int v690;
            v690 = threadIdx.x;
            int v691;
            v691 = v690 / 32;
            auto v692 = cooperative_groups::labeled_partition(v689,v691);
            Closure2 v693{};
            float v694;
            v694 = cooperative_groups::inclusive_scan(v692, v684, v693);
            float v695;
            v695 = v692.shfl_up(v694,1);
            bool v696;
            v696 = v692.thread_rank() == 0;
            float v697;
            if (v696){
                v697 = 0.0f;
            } else {
                v697 = v695;
            }
            float v698;
            v698 = v692.shfl(v694,v692.num_threads()-1);
            float v699;
            v699 = v679 + v697;
            int v700; float v701;
            Tuple0 tmp4 = Tuple0{0, v699};
            v700 = tmp4.v0; v701 = tmp4.v1;
            while (while_method_1(v700)){
                assert("Tensor range check" && 0 <= v700 && v700 < 4);
                int v703;
                v703 = v700 + v682;
                float v704;
                v704 = v669[v703];
                float v705;
                v705 = v701 + v704;
                assert("Tensor range check" && 0 <= v700 && v700 < 4);
                v678[v703] = v705;
                v701 = v705;
                v700 += 1 ;
            }
            float v706;
            v706 = v679 + v698;
            v679 = v706;
            v680 += 1 ;
        }
        assert("Tensor range check" && 0 <= v581 && v581 < 8);
        int v707;
        v707 = 0;
        while (while_method_3(v707)){
            assert("Tensor range check" && 0 <= v707 && v707 < 1);
            int v709;
            v709 = 128 * v707;
            int v710;
            v710 = v709 + v584;
            assert("Tensor range check" && 0 <= v707 && v707 < 1);
            int v711;
            v711 = 4 * v707;
            int4* v712;
            v712 = reinterpret_cast<int4*>(v669 + v711);
            int4* v713;
            v713 = reinterpret_cast<int4*>(v6 + v710);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v712) % 16 == 0 && reinterpret_cast<unsigned long long>(v713) % 16 == 0);
            *v713 = *v712;
            int4* v714;
            v714 = reinterpret_cast<int4*>(v678 + v711);
            int4* v715;
            v715 = reinterpret_cast<int4*>(v7 + v710);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v714) % 16 == 0 && reinterpret_cast<unsigned long long>(v715) % 16 == 0);
            *v715 = *v714;
            v707 += 1 ;
        }
        v581 += 1 ;
    }
    __syncthreads();
    int v716;
    v716 = threadIdx.x;
    bool v717;
    v717 = 0 <= v716;
    bool v718;
    v718 = v717 == false;
    if (v718){
        assert("The index needs to be zero or positive." && v717);
    } else {
    }
    int v720;
    v720 = v716 % 32;
    int v721;
    v721 = v716 / 32;
    bool v722;
    v722 = v721 < 8;
    bool v723;
    v723 = v722 == false;
    if (v723){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v722);
    } else {
    }
    assert("Tensor range check" && 0 <= v721 && v721 < 8);
    assert("Tensor range check" && 0 <= v720 && v720 < 32);
    int v725;
    v725 = 4 * v720;
    int v726;
    v726 = 128 * v721;
    int v727;
    v727 = v726 + v725;
    assert("Tensor range check" && 0 <= v721 && v721 < 8);
    assert("Tensor range check" && 0 <= v720 && v720 < 32);
    int v728;
    v728 = 0;
    while (while_method_2(v728)){
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v730;
        v730 = 1024 * v728;
        int v731;
        v731 = v730 + v727;
        int v732[4];
        int v733[4];
        int v734;
        v734 = 0;
        while (while_method_3(v734)){
            assert("Tensor range check" && 0 <= v734 && v734 < 1);
            int v736;
            v736 = 4 * v734;
            assert("Tensor range check" && 0 <= v734 && v734 < 1);
            int v737;
            v737 = 128 * v734;
            int v738;
            v738 = v737 + v731;
            int4* v739;
            v739 = reinterpret_cast<int4*>(v0 + v738);
            int4* v740;
            v740 = reinterpret_cast<int4*>(v732 + v736);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v739) % 16 == 0 && reinterpret_cast<unsigned long long>(v740) % 16 == 0);
            *v740 = *v739;
            v734 += 1 ;
        }
        int v741;
        v741 = 0;
        while (while_method_3(v741)){
            int v743;
            v743 = 0;
            while (while_method_1(v743)){
                bool v745;
                v745 = 0 <= v743;
                bool v747;
                if (v745){
                    bool v746;
                    v746 = v743 < 4;
                    v747 = v746;
                } else {
                    v747 = false;
                }
                bool v748;
                v748 = v747 == false;
                if (v748){
                    assert("The indices should be inside the range of the dimension." && v747);
                } else {
                }
                bool v750;
                v750 = 0 <= v720;
                bool v752;
                if (v750){
                    bool v751;
                    v751 = v720 < 32;
                    v752 = v751;
                } else {
                    v752 = false;
                }
                bool v753;
                v753 = v752 == false;
                if (v753){
                    assert("The indices should be inside the range of the dimension." && v752);
                } else {
                }
                int v755;
                v755 = v720 * 4;
                int v756;
                v756 = v743 + v755;
                bool v757;
                v757 = 0 <= v741;
                bool v759;
                if (v757){
                    bool v758;
                    v758 = v741 < 1;
                    v759 = v758;
                } else {
                    v759 = false;
                }
                bool v760;
                v760 = v759 == false;
                if (v760){
                    assert("The indices should be inside the range of the dimension." && v759);
                } else {
                }
                int v762;
                v762 = v741 * 128;
                int v763;
                v763 = v756 + v762;
                assert("Tensor range check" && 0 <= v741 && v741 < 1);
                assert("Tensor range check" && 0 <= v743 && v743 < 4);
                int v764;
                v764 = 4 * v741;
                int v765;
                v765 = v764 + v743;
                v733[v765] = v763;
                v743 += 1 ;
            }
            v741 += 1 ;
        }
        bool v766;
        v766 = 0 <= v721;
        bool v767;
        v767 = v766 && v722;
        bool v768;
        v768 = v767 == false;
        if (v768){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v767);
        } else {
        }
        bool v770;
        v770 = 0 <= v728;
        bool v772;
        if (v770){
            bool v771;
            v771 = v728 < 8;
            v772 = v771;
        } else {
            v772 = false;
        }
        bool v773;
        v773 = v772 == false;
        if (v773){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v772);
        } else {
        }
        int v775;
        v775 = v728 * 8;
        int v776;
        v776 = v775 + v721;
        int v777[4];
        int v778;
        v778 = 0;
        int v779;
        v779 = 0;
        while (while_method_3(v779)){
            assert("Tensor range check" && 0 <= v779 && v779 < 1);
            int v781;
            v781 = 4 * v779;
            assert("Tensor range check" && 0 <= v779 && v779 < 1);
            int v782; int v783;
            Tuple2 tmp5 = Tuple2{0, 0};
            v782 = tmp5.v0; v783 = tmp5.v1;
            while (while_method_1(v782)){
                assert("Tensor range check" && 0 <= v782 && v782 < 4);
                int v785;
                v785 = v782 + v781;
                int v786;
                v786 = v732[v785];
                int v787;
                v787 = v783 + v786;
                v783 = v787;
                v782 += 1 ;
            }
            auto v788 = cooperative_groups::coalesced_threads();
            int v789;
            v789 = threadIdx.x;
            int v790;
            v790 = v789 / 32;
            auto v791 = cooperative_groups::labeled_partition(v788,v790);
            Closure3 v792{};
            int v793;
            v793 = cooperative_groups::inclusive_scan(v791, v783, v792);
            int v794;
            v794 = v791.shfl_up(v793,1);
            bool v795;
            v795 = v791.thread_rank() == 0;
            int v796;
            if (v795){
                v796 = 0;
            } else {
                v796 = v794;
            }
            int v797;
            v797 = v791.shfl(v793,v791.num_threads()-1);
            int v798;
            v798 = v778 + v796;
            int v799; int v800;
            Tuple2 tmp6 = Tuple2{0, v798};
            v799 = tmp6.v0; v800 = tmp6.v1;
            while (while_method_1(v799)){
                assert("Tensor range check" && 0 <= v799 && v799 < 4);
                int v802;
                v802 = v799 + v781;
                int v803;
                v803 = v732[v802];
                assert("Tensor range check" && 0 <= v799 && v799 < 4);
                v777[v802] = v800;
                int v804;
                v804 = v800 + v803;
                v800 = v804;
                v799 += 1 ;
            }
            int v805;
            v805 = v778 + v797;
            v778 = v805;
            v779 += 1 ;
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v806;
        v806 = 0;
        while (while_method_3(v806)){
            assert("Tensor range check" && 0 <= v806 && v806 < 1);
            int v808;
            v808 = 128 * v806;
            int v809;
            v809 = v808 + v731;
            assert("Tensor range check" && 0 <= v806 && v806 < 1);
            int v810;
            v810 = 4 * v806;
            int4* v811;
            v811 = reinterpret_cast<int4*>(v777 + v810);
            int4* v812;
            v812 = reinterpret_cast<int4*>(v13 + v809);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v811) % 16 == 0 && reinterpret_cast<unsigned long long>(v812) % 16 == 0);
            *v812 = *v811;
            v806 += 1 ;
        }
        v728 += 1 ;
    }
    __syncthreads();
    int v813;
    v813 = threadIdx.x;
    bool v814;
    v814 = 0 <= v813;
    bool v815;
    v815 = v814 == false;
    if (v815){
        assert("The index needs to be zero or positive." && v814);
    } else {
    }
    int v817;
    v817 = v813 % 32;
    int v818;
    v818 = v813 / 32;
    bool v819;
    v819 = v818 < 8;
    bool v820;
    v820 = v819 == false;
    if (v820){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v819);
    } else {
    }
    assert("Tensor range check" && 0 <= v818 && v818 < 8);
    assert("Tensor range check" && 0 <= v817 && v817 < 32);
    int v822;
    v822 = 4 * v817;
    int v823;
    v823 = 128 * v818;
    int v824;
    v824 = v823 + v822;
    assert("Tensor range check" && 0 <= v818 && v818 < 8);
    assert("Tensor range check" && 0 <= v817 && v817 < 32);
    int v825;
    v825 = 0;
    while (while_method_2(v825)){
        assert("Tensor range check" && 0 <= v825 && v825 < 8);
        int v827;
        v827 = 1024 * v825;
        int v828;
        v828 = v827 + v824;
        float v829[4];
        int v830[4];
        int v831;
        v831 = 0;
        while (while_method_3(v831)){
            assert("Tensor range check" && 0 <= v831 && v831 < 1);
            int v833;
            v833 = 4 * v831;
            assert("Tensor range check" && 0 <= v831 && v831 < 1);
            int v834;
            v834 = 128 * v831;
            int v835;
            v835 = v834 + v828;
            int4* v836;
            v836 = reinterpret_cast<int4*>(v1 + v835);
            int4* v837;
            v837 = reinterpret_cast<int4*>(v829 + v833);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v836) % 16 == 0 && reinterpret_cast<unsigned long long>(v837) % 16 == 0);
            *v837 = *v836;
            v831 += 1 ;
        }
        int v838;
        v838 = 0;
        while (while_method_3(v838)){
            int v840;
            v840 = 0;
            while (while_method_1(v840)){
                bool v842;
                v842 = 0 <= v840;
                bool v844;
                if (v842){
                    bool v843;
                    v843 = v840 < 4;
                    v844 = v843;
                } else {
                    v844 = false;
                }
                bool v845;
                v845 = v844 == false;
                if (v845){
                    assert("The indices should be inside the range of the dimension." && v844);
                } else {
                }
                bool v847;
                v847 = 0 <= v817;
                bool v849;
                if (v847){
                    bool v848;
                    v848 = v817 < 32;
                    v849 = v848;
                } else {
                    v849 = false;
                }
                bool v850;
                v850 = v849 == false;
                if (v850){
                    assert("The indices should be inside the range of the dimension." && v849);
                } else {
                }
                int v852;
                v852 = v817 * 4;
                int v853;
                v853 = v840 + v852;
                bool v854;
                v854 = 0 <= v838;
                bool v856;
                if (v854){
                    bool v855;
                    v855 = v838 < 1;
                    v856 = v855;
                } else {
                    v856 = false;
                }
                bool v857;
                v857 = v856 == false;
                if (v857){
                    assert("The indices should be inside the range of the dimension." && v856);
                } else {
                }
                int v859;
                v859 = v838 * 128;
                int v860;
                v860 = v853 + v859;
                assert("Tensor range check" && 0 <= v838 && v838 < 1);
                assert("Tensor range check" && 0 <= v840 && v840 < 4);
                int v861;
                v861 = 4 * v838;
                int v862;
                v862 = v861 + v840;
                v830[v862] = v860;
                v840 += 1 ;
            }
            v838 += 1 ;
        }
        bool v863;
        v863 = 0 <= v818;
        bool v864;
        v864 = v863 && v819;
        bool v865;
        v865 = v864 == false;
        if (v865){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v864);
        } else {
        }
        bool v867;
        v867 = 0 <= v825;
        bool v869;
        if (v867){
            bool v868;
            v868 = v825 < 8;
            v869 = v868;
        } else {
            v869 = false;
        }
        bool v870;
        v870 = v869 == false;
        if (v870){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v869);
        } else {
        }
        int v872;
        v872 = v825 * 8;
        int v873;
        v873 = v872 + v818;
        bool v874[4];
        int v875;
        v875 = 0;
        while (while_method_3(v875)){
            int v877;
            v877 = 0;
            while (while_method_1(v877)){
                assert("Tensor range check" && 0 <= v875 && v875 < 1);
                assert("Tensor range check" && 0 <= v877 && v877 < 4);
                int v879;
                v879 = 4 * v875;
                int v880;
                v880 = v879 + v877;
                float v881;
                v881 = v829[v880];
                int v882;
                v882 = v830[v880];
                bool v883;
                v883 = v882 < 4;
                assert("Tensor range check" && 0 <= v875 && v875 < 1);
                assert("Tensor range check" && 0 <= v877 && v877 < 4);
                v874[v880] = v883;
                v877 += 1 ;
            }
            v875 += 1 ;
        }
        int v884[4];
        int v885;
        v885 = 0;
        while (while_method_3(v885)){
            int v887;
            v887 = 0;
            while (while_method_1(v887)){
                assert("Tensor range check" && 0 <= v885 && v885 < 1);
                assert("Tensor range check" && 0 <= v887 && v887 < 4);
                int v889;
                v889 = 4 * v885;
                int v890;
                v890 = v889 + v887;
                bool v891;
                v891 = v874[v890];
                int v892;
                if (v891){
                    v892 = 1;
                } else {
                    v892 = 0;
                }
                assert("Tensor range check" && 0 <= v885 && v885 < 1);
                assert("Tensor range check" && 0 <= v887 && v887 < 4);
                v884[v890] = v892;
                v887 += 1 ;
            }
            v885 += 1 ;
        }
        int v893;
        v893 = 0;
        int v894;
        v894 = 0;
        while (while_method_3(v894)){
            int v896;
            v896 = 0;
            while (while_method_1(v896)){
                assert("Tensor range check" && 0 <= v894 && v894 < 1);
                assert("Tensor range check" && 0 <= v896 && v896 < 4);
                int v898;
                v898 = 4 * v894;
                int v899;
                v899 = v898 + v896;
                int v900;
                v900 = v884[v899];
                int v901;
                v901 = v893 + v900;
                v893 = v901;
                v896 += 1 ;
            }
            v894 += 1 ;
        }
        auto v902 = cooperative_groups::coalesced_threads();
        int v903;
        v903 = threadIdx.x;
        int v904;
        v904 = v903 / 32;
        auto v905 = cooperative_groups::labeled_partition(v902,v904);
        Closure4 v906{};
        int v907;
        v907 = cooperative_groups::reduce(v905, v893, v906);
        float v908[4];
        int v909;
        v909 = 0;
        while (while_method_3(v909)){
            int v911;
            v911 = 0;
            while (while_method_1(v911)){
                assert("Tensor range check" && 0 <= v909 && v909 < 1);
                assert("Tensor range check" && 0 <= v911 && v911 < 4);
                int v913;
                v913 = 4 * v909;
                int v914;
                v914 = v913 + v911;
                float v915;
                v915 = v829[v914];
                bool v916;
                v916 = v874[v914];
                float v917;
                if (v916){
                    v917 = v915;
                } else {
                    v917 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v909 && v909 < 1);
                assert("Tensor range check" && 0 <= v911 && v911 < 4);
                v908[v914] = v917;
                v911 += 1 ;
            }
            v909 += 1 ;
        }
        float v918;
        v918 = 0.0f;
        int v919;
        v919 = 0;
        while (while_method_3(v919)){
            int v921;
            v921 = 0;
            while (while_method_1(v921)){
                assert("Tensor range check" && 0 <= v919 && v919 < 1);
                assert("Tensor range check" && 0 <= v921 && v921 < 4);
                int v923;
                v923 = 4 * v919;
                int v924;
                v924 = v923 + v921;
                float v925;
                v925 = v908[v924];
                float v926;
                v926 = v918 + v925;
                v918 = v926;
                v921 += 1 ;
            }
            v919 += 1 ;
        }
        auto v927 = cooperative_groups::coalesced_threads();
        int v928;
        v928 = threadIdx.x;
        int v929;
        v929 = v928 / 32;
        auto v930 = cooperative_groups::labeled_partition(v927,v929);
        float v931;
        v931 = cooperative_groups::reduce(v930, v918, v42);
        float v932;
        v932 = (float)v907;
        float v933;
        v933 = v931 / v932;
        float v934[4];
        int v935;
        v935 = 0;
        while (while_method_3(v935)){
            int v937;
            v937 = 0;
            while (while_method_1(v937)){
                assert("Tensor range check" && 0 <= v935 && v935 < 1);
                assert("Tensor range check" && 0 <= v937 && v937 < 4);
                int v939;
                v939 = 4 * v935;
                int v940;
                v940 = v939 + v937;
                float v941;
                v941 = v829[v940];
                bool v942;
                v942 = v874[v940];
                float v943;
                if (v942){
                    v943 = v941;
                } else {
                    v943 = -1.0f / 0.0f;
                }
                float v944;
                v944 = v943 - v933;
                float v945;
                v945 = exp(v944);
                assert("Tensor range check" && 0 <= v935 && v935 < 1);
                assert("Tensor range check" && 0 <= v937 && v937 < 4);
                v934[v940] = v945;
                v937 += 1 ;
            }
            v935 += 1 ;
        }
        float v946;
        v946 = 0.0f;
        int v947;
        v947 = 0;
        while (while_method_3(v947)){
            int v949;
            v949 = 0;
            while (while_method_1(v949)){
                assert("Tensor range check" && 0 <= v947 && v947 < 1);
                assert("Tensor range check" && 0 <= v949 && v949 < 4);
                int v951;
                v951 = 4 * v947;
                int v952;
                v952 = v951 + v949;
                float v953;
                v953 = v934[v952];
                float v954;
                v954 = v946 + v953;
                v946 = v954;
                v949 += 1 ;
            }
            v947 += 1 ;
        }
        auto v955 = cooperative_groups::coalesced_threads();
        int v956;
        v956 = threadIdx.x;
        int v957;
        v957 = v956 / 32;
        auto v958 = cooperative_groups::labeled_partition(v955,v957);
        float v959;
        v959 = cooperative_groups::reduce(v958, v946, v42);
        float v960[4];
        int v961;
        v961 = 0;
        while (while_method_3(v961)){
            int v963;
            v963 = 0;
            while (while_method_1(v963)){
                assert("Tensor range check" && 0 <= v961 && v961 < 1);
                assert("Tensor range check" && 0 <= v963 && v963 < 4);
                int v965;
                v965 = 4 * v961;
                int v966;
                v966 = v965 + v963;
                float v967;
                v967 = v934[v966];
                float v968;
                v968 = v967 / v959;
                assert("Tensor range check" && 0 <= v961 && v961 < 1);
                assert("Tensor range check" && 0 <= v963 && v963 < 4);
                v960[v966] = v968;
                v963 += 1 ;
            }
            v961 += 1 ;
        }
        assert("Tensor range check" && 0 <= v825 && v825 < 8);
        int v969;
        v969 = 0;
        while (while_method_3(v969)){
            assert("Tensor range check" && 0 <= v969 && v969 < 1);
            int v971;
            v971 = 128 * v969;
            int v972;
            v972 = v971 + v828;
            assert("Tensor range check" && 0 <= v969 && v969 < 1);
            int v973;
            v973 = 4 * v969;
            int4* v974;
            v974 = reinterpret_cast<int4*>(v960 + v973);
            int4* v975;
            v975 = reinterpret_cast<int4*>(v5 + v972);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v974) % 16 == 0 && reinterpret_cast<unsigned long long>(v975) % 16 == 0);
            *v975 = *v974;
            v969 += 1 ;
        }
        v825 += 1 ;
    }
    __syncthreads();
    int v976;
    v976 = threadIdx.x;
    int v977;
    v977 = blockIdx.x;
    int v978;
    v978 = v977 * 256;
    int v979;
    v979 = v976 + v978;
    unsigned long long v980;
    v980 = (unsigned long long)v979;
    curandStatePhilox4_32_10_t v981;
    curand_init(12344321ull,v980,0ull,&v981);
    int v982;
    v982 = threadIdx.x;
    bool v983;
    v983 = 0 <= v982;
    bool v984;
    v984 = v983 == false;
    if (v984){
        assert("The index needs to be zero or positive." && v983);
    } else {
    }
    int v986;
    v986 = v982 % 32;
    int v987;
    v987 = v982 / 32;
    bool v988;
    v988 = v987 < 8;
    bool v989;
    v989 = v988 == false;
    if (v989){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v988);
    } else {
    }
    assert("Tensor range check" && 0 <= v987 && v987 < 8);
    assert("Tensor range check" && 0 <= v986 && v986 < 32);
    int v991;
    v991 = 4 * v986;
    int v992;
    v992 = 128 * v987;
    int v993;
    v993 = v992 + v991;
    assert("Tensor range check" && 0 <= v987 && v987 < 8);
    assert("Tensor range check" && 0 <= v986 && v986 < 32);
    assert("Tensor range check" && 0 <= v987 && v987 < 8);
    int v994;
    v994 = 0;
    while (while_method_2(v994)){
        assert("Tensor range check" && 0 <= v994 && v994 < 8);
        int v996;
        v996 = 1024 * v994;
        int v997;
        v997 = v996 + v993;
        float v998[4];
        int v999[4];
        int v1000;
        v1000 = 0;
        while (while_method_3(v1000)){
            assert("Tensor range check" && 0 <= v1000 && v1000 < 1);
            int v1002;
            v1002 = 4 * v1000;
            assert("Tensor range check" && 0 <= v1000 && v1000 < 1);
            int v1003;
            v1003 = 128 * v1000;
            int v1004;
            v1004 = v1003 + v997;
            int4* v1005;
            v1005 = reinterpret_cast<int4*>(v1 + v1004);
            int4* v1006;
            v1006 = reinterpret_cast<int4*>(v998 + v1002);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1005) % 16 == 0 && reinterpret_cast<unsigned long long>(v1006) % 16 == 0);
            *v1006 = *v1005;
            v1000 += 1 ;
        }
        int v1007;
        v1007 = 0;
        while (while_method_3(v1007)){
            int v1009;
            v1009 = 0;
            while (while_method_1(v1009)){
                bool v1011;
                v1011 = 0 <= v1009;
                bool v1013;
                if (v1011){
                    bool v1012;
                    v1012 = v1009 < 4;
                    v1013 = v1012;
                } else {
                    v1013 = false;
                }
                bool v1014;
                v1014 = v1013 == false;
                if (v1014){
                    assert("The indices should be inside the range of the dimension." && v1013);
                } else {
                }
                bool v1016;
                v1016 = 0 <= v986;
                bool v1018;
                if (v1016){
                    bool v1017;
                    v1017 = v986 < 32;
                    v1018 = v1017;
                } else {
                    v1018 = false;
                }
                bool v1019;
                v1019 = v1018 == false;
                if (v1019){
                    assert("The indices should be inside the range of the dimension." && v1018);
                } else {
                }
                int v1021;
                v1021 = v986 * 4;
                int v1022;
                v1022 = v1009 + v1021;
                bool v1023;
                v1023 = 0 <= v1007;
                bool v1025;
                if (v1023){
                    bool v1024;
                    v1024 = v1007 < 1;
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
                int v1028;
                v1028 = v1007 * 128;
                int v1029;
                v1029 = v1022 + v1028;
                assert("Tensor range check" && 0 <= v1007 && v1007 < 1);
                assert("Tensor range check" && 0 <= v1009 && v1009 < 4);
                int v1030;
                v1030 = 4 * v1007;
                int v1031;
                v1031 = v1030 + v1009;
                v999[v1031] = v1029;
                v1009 += 1 ;
            }
            v1007 += 1 ;
        }
        bool v1032;
        v1032 = 0 <= v987;
        bool v1033;
        v1033 = v1032 && v988;
        bool v1034;
        v1034 = v1033 == false;
        if (v1034){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1033);
        } else {
        }
        bool v1036;
        v1036 = 0 <= v994;
        bool v1038;
        if (v1036){
            bool v1037;
            v1037 = v994 < 8;
            v1038 = v1037;
        } else {
            v1038 = false;
        }
        bool v1039;
        v1039 = v1038 == false;
        if (v1039){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1038);
        } else {
        }
        int v1041;
        v1041 = v994 * 8;
        int v1042;
        v1042 = v1041 + v987;
        float v1043;
        v1043 = 0.0f;
        int v1044;
        v1044 = 0;
        while (while_method_3(v1044)){
            int v1046;
            v1046 = 0;
            while (while_method_1(v1046)){
                assert("Tensor range check" && 0 <= v1044 && v1044 < 1);
                assert("Tensor range check" && 0 <= v1046 && v1046 < 4);
                int v1048;
                v1048 = 4 * v1044;
                int v1049;
                v1049 = v1048 + v1046;
                float v1050;
                v1050 = v998[v1049];
                float v1051;
                v1051 = v1043 + v1050;
                v1043 = v1051;
                v1046 += 1 ;
            }
            v1044 += 1 ;
        }
        auto v1052 = cooperative_groups::coalesced_threads();
        int v1053;
        v1053 = threadIdx.x;
        int v1054;
        v1054 = v1053 / 32;
        auto v1055 = cooperative_groups::labeled_partition(v1052,v1054);
        float v1056;
        v1056 = cooperative_groups::reduce(v1055, v1043, v42);
        float v1057;
        v1057 = v1056 / 128.0f;
        float v1058[4];
        int v1059;
        v1059 = 0;
        while (while_method_3(v1059)){
            int v1061;
            v1061 = 0;
            while (while_method_1(v1061)){
                assert("Tensor range check" && 0 <= v1059 && v1059 < 1);
                assert("Tensor range check" && 0 <= v1061 && v1061 < 4);
                int v1063;
                v1063 = 4 * v1059;
                int v1064;
                v1064 = v1063 + v1061;
                float v1065;
                v1065 = v998[v1064];
                float v1066;
                v1066 = v1065 - v1057;
                float v1067;
                v1067 = exp(v1066);
                assert("Tensor range check" && 0 <= v1059 && v1059 < 1);
                assert("Tensor range check" && 0 <= v1061 && v1061 < 4);
                v1058[v1064] = v1067;
                v1061 += 1 ;
            }
            v1059 += 1 ;
        }
        float v1068;
        v1068 = 0.0f;
        int v1069;
        v1069 = 0;
        while (while_method_3(v1069)){
            int v1071;
            v1071 = 0;
            while (while_method_1(v1071)){
                assert("Tensor range check" && 0 <= v1069 && v1069 < 1);
                assert("Tensor range check" && 0 <= v1071 && v1071 < 4);
                int v1073;
                v1073 = 4 * v1069;
                int v1074;
                v1074 = v1073 + v1071;
                float v1075;
                v1075 = v1058[v1074];
                float v1076;
                v1076 = v1068 + v1075;
                v1068 = v1076;
                v1071 += 1 ;
            }
            v1069 += 1 ;
        }
        auto v1077 = cooperative_groups::coalesced_threads();
        int v1078;
        v1078 = threadIdx.x;
        int v1079;
        v1079 = v1078 / 32;
        auto v1080 = cooperative_groups::labeled_partition(v1077,v1079);
        float v1081;
        v1081 = cooperative_groups::reduce(v1080, v1068, v42);
        float v1082[4];
        int v1083;
        v1083 = 0;
        while (while_method_3(v1083)){
            int v1085;
            v1085 = 0;
            while (while_method_1(v1085)){
                assert("Tensor range check" && 0 <= v1083 && v1083 < 1);
                assert("Tensor range check" && 0 <= v1085 && v1085 < 4);
                int v1087;
                v1087 = 4 * v1083;
                int v1088;
                v1088 = v1087 + v1085;
                float v1089;
                v1089 = v1058[v1088];
                float v1090;
                v1090 = v1089 / v1081;
                assert("Tensor range check" && 0 <= v1083 && v1083 < 1);
                assert("Tensor range check" && 0 <= v1085 && v1085 < 4);
                v1082[v1088] = v1090;
                v1085 += 1 ;
            }
            v1083 += 1 ;
        }
        float v1091[4];
        float v1092;
        v1092 = 0.0f;
        int v1093;
        v1093 = 0;
        while (while_method_3(v1093)){
            assert("Tensor range check" && 0 <= v1093 && v1093 < 1);
            int v1095;
            v1095 = 4 * v1093;
            assert("Tensor range check" && 0 <= v1093 && v1093 < 1);
            int v1096; float v1097;
            Tuple0 tmp7 = Tuple0{0, 0.0f};
            v1096 = tmp7.v0; v1097 = tmp7.v1;
            while (while_method_1(v1096)){
                assert("Tensor range check" && 0 <= v1096 && v1096 < 4);
                int v1099;
                v1099 = v1096 + v1095;
                float v1100;
                v1100 = v1082[v1099];
                float v1101;
                v1101 = v1097 + v1100;
                v1097 = v1101;
                v1096 += 1 ;
            }
            auto v1102 = cooperative_groups::coalesced_threads();
            int v1103;
            v1103 = threadIdx.x;
            int v1104;
            v1104 = v1103 / 32;
            auto v1105 = cooperative_groups::labeled_partition(v1102,v1104);
            Closure2 v1106{};
            float v1107;
            v1107 = cooperative_groups::inclusive_scan(v1105, v1097, v1106);
            float v1108;
            v1108 = v1105.shfl_up(v1107,1);
            bool v1109;
            v1109 = v1105.thread_rank() == 0;
            float v1110;
            if (v1109){
                v1110 = 0.0f;
            } else {
                v1110 = v1108;
            }
            float v1111;
            v1111 = v1105.shfl(v1107,v1105.num_threads()-1);
            float v1112;
            v1112 = v1092 + v1110;
            int v1113; float v1114;
            Tuple0 tmp8 = Tuple0{0, v1112};
            v1113 = tmp8.v0; v1114 = tmp8.v1;
            while (while_method_1(v1113)){
                assert("Tensor range check" && 0 <= v1113 && v1113 < 4);
                int v1116;
                v1116 = v1113 + v1095;
                float v1117;
                v1117 = v1082[v1116];
                float v1118;
                v1118 = v1114 + v1117;
                assert("Tensor range check" && 0 <= v1113 && v1113 < 4);
                v1091[v1116] = v1118;
                v1114 = v1118;
                v1113 += 1 ;
            }
            float v1119;
            v1119 = v1092 + v1111;
            v1092 = v1119;
            v1093 += 1 ;
        }
        float v1120[4];
        bool v1121[4];
        int v1122;
        v1122 = 0;
        while (while_method_3(v1122)){
            int v1124;
            v1124 = 0;
            while (while_method_1(v1124)){
                assert("Tensor range check" && 0 <= v1122 && v1122 < 1);
                assert("Tensor range check" && 0 <= v1124 && v1124 < 4);
                int v1126;
                v1126 = 4 * v1122;
                int v1127;
                v1127 = v1126 + v1124;
                float v1128;
                v1128 = v1091[v1127];
                float v1129;
                v1129 = v1082[v1127];
                bool v1130;
                v1130 = v1129 > 0.0f;
                assert("Tensor range check" && 0 <= v1122 && v1122 < 1);
                assert("Tensor range check" && 0 <= v1124 && v1124 < 4);
                v1120[v1127] = v1128;
                v1121[v1127] = v1130;
                v1124 += 1 ;
            }
            v1122 += 1 ;
        }
        float v1131; bool v1132;
        Tuple3 tmp9 = Tuple3{-1.0f / 0.0f, false};
        v1131 = tmp9.v0; v1132 = tmp9.v1;
        int v1133;
        v1133 = 0;
        while (while_method_3(v1133)){
            int v1135;
            v1135 = 0;
            while (while_method_1(v1135)){
                assert("Tensor range check" && 0 <= v1133 && v1133 < 1);
                assert("Tensor range check" && 0 <= v1135 && v1135 < 4);
                int v1137;
                v1137 = 4 * v1133;
                int v1138;
                v1138 = v1137 + v1135;
                float v1139;
                v1139 = v1120[v1138];
                bool v1140;
                v1140 = v1121[v1138];
                float v1147; bool v1148;
                if (v1132){
                    if (v1140){
                        bool v1141;
                        v1141 = v1131 >= v1139;
                        float v1142;
                        if (v1141){
                            v1142 = v1131;
                        } else {
                            v1142 = v1139;
                        }
                        v1147 = v1142; v1148 = true;
                    } else {
                        v1147 = v1131; v1148 = v1132;
                    }
                } else {
                    if (v1140){
                        v1147 = v1139; v1148 = v1140;
                    } else {
                        v1147 = v1131; v1148 = v1132;
                    }
                }
                v1131 = v1147;
                v1132 = v1148;
                v1135 += 1 ;
            }
            v1133 += 1 ;
        }
        auto v1149 = cooperative_groups::coalesced_threads();
        int v1150;
        v1150 = threadIdx.x;
        int v1151;
        v1151 = v1150 / 32;
        auto v1152 = cooperative_groups::labeled_partition(v1149,v1151);
        Closure5 v1153{};
        float v1154; bool v1155;
        Tuple3 tmp10 = cooperative_groups::reduce(v1152, Tuple3{v1131, v1132}, v1153);
        v1154 = tmp10.v0; v1155 = tmp10.v1;
        bool v1156;
        v1156 = v1155 == false;
        if (v1156){
            assert("The local reduce must be true." && v1155);
        } else {
        }
        float v1158[4];
        int v1159[4];
        int v1160;
        v1160 = 0;
        while (while_method_3(v1160)){
            int v1162;
            v1162 = 0;
            while (while_method_1(v1162)){
                assert("Tensor range check" && 0 <= v1160 && v1160 < 1);
                assert("Tensor range check" && 0 <= v1162 && v1162 < 4);
                int v1164;
                v1164 = 4 * v1160;
                int v1165;
                v1165 = v1164 + v1162;
                int v1166;
                v1166 = v999[v1165];
                float v1167;
                v1167 = curand_uniform(&v981);
                assert("Tensor range check" && 0 <= v1160 && v1160 < 1);
                assert("Tensor range check" && 0 <= v1162 && v1162 < 4);
                v1158[v1165] = v1167;
                v1159[v1165] = v1166;
                v1162 += 1 ;
            }
            v1160 += 1 ;
        }
        float v1168; int v1169;
        Tuple1 tmp11 = Tuple1{0.0f, 2147483647};
        v1168 = tmp11.v0; v1169 = tmp11.v1;
        int v1170;
        v1170 = 0;
        while (while_method_3(v1170)){
            int v1172;
            v1172 = 0;
            while (while_method_1(v1172)){
                assert("Tensor range check" && 0 <= v1170 && v1170 < 1);
                assert("Tensor range check" && 0 <= v1172 && v1172 < 4);
                int v1174;
                v1174 = 4 * v1170;
                int v1175;
                v1175 = v1174 + v1172;
                float v1176;
                v1176 = v1158[v1175];
                int v1177;
                v1177 = v1159[v1175];
                bool v1178;
                v1178 = v1169 < v1177;
                float v1179; int v1180;
                if (v1178){
                    v1179 = v1168; v1180 = v1169;
                } else {
                    v1179 = v1176; v1180 = v1177;
                }
                v1168 = v1179;
                v1169 = v1180;
                v1172 += 1 ;
            }
            v1170 += 1 ;
        }
        auto v1181 = cooperative_groups::coalesced_threads();
        int v1182;
        v1182 = threadIdx.x;
        int v1183;
        v1183 = v1182 / 32;
        auto v1184 = cooperative_groups::labeled_partition(v1181,v1183);
        Closure6 v1185{};
        float v1186; int v1187;
        Tuple1 tmp12 = cooperative_groups::reduce(v1184, Tuple1{v1168, v1169}, v1185);
        v1186 = tmp12.v0; v1187 = tmp12.v1;
        float v1188;
        v1188 = v1154 * v1186;
        int v1189[4];
        bool v1190[4];
        int v1191;
        v1191 = 0;
        while (while_method_3(v1191)){
            int v1193;
            v1193 = 0;
            while (while_method_1(v1193)){
                assert("Tensor range check" && 0 <= v1191 && v1191 < 1);
                assert("Tensor range check" && 0 <= v1193 && v1193 < 4);
                int v1195;
                v1195 = 4 * v1191;
                int v1196;
                v1196 = v1195 + v1193;
                float v1197;
                v1197 = v1120[v1196];
                bool v1198;
                v1198 = v1121[v1196];
                int v1199;
                v1199 = v999[v1196];
                int v1202; bool v1203;
                if (v1198){
                    float v1200;
                    v1200 = v1197 - v1188;
                    bool v1201;
                    v1201 = v1200 >= 0.0f;
                    v1202 = v1199; v1203 = v1201;
                } else {
                    v1202 = 2147483647; v1203 = false;
                }
                assert("Tensor range check" && 0 <= v1191 && v1191 < 1);
                assert("Tensor range check" && 0 <= v1193 && v1193 < 4);
                v1189[v1196] = v1202;
                v1190[v1196] = v1203;
                v1193 += 1 ;
            }
            v1191 += 1 ;
        }
        int v1204; bool v1205;
        Tuple4 tmp13 = Tuple4{2147483647, false};
        v1204 = tmp13.v0; v1205 = tmp13.v1;
        int v1206;
        v1206 = 0;
        while (while_method_3(v1206)){
            int v1208;
            v1208 = 0;
            while (while_method_1(v1208)){
                assert("Tensor range check" && 0 <= v1206 && v1206 < 1);
                assert("Tensor range check" && 0 <= v1208 && v1208 < 4);
                int v1210;
                v1210 = 4 * v1206;
                int v1211;
                v1211 = v1210 + v1208;
                int v1212;
                v1212 = v1189[v1211];
                bool v1213;
                v1213 = v1190[v1211];
                int v1220; bool v1221;
                if (v1205){
                    if (v1213){
                        bool v1214;
                        v1214 = v1204 < v1212;
                        int v1215;
                        if (v1214){
                            v1215 = v1204;
                        } else {
                            v1215 = v1212;
                        }
                        v1220 = v1215; v1221 = true;
                    } else {
                        v1220 = v1204; v1221 = v1205;
                    }
                } else {
                    if (v1213){
                        v1220 = v1212; v1221 = v1213;
                    } else {
                        v1220 = v1204; v1221 = v1205;
                    }
                }
                v1204 = v1220;
                v1205 = v1221;
                v1208 += 1 ;
            }
            v1206 += 1 ;
        }
        auto v1222 = cooperative_groups::coalesced_threads();
        int v1223;
        v1223 = threadIdx.x;
        int v1224;
        v1224 = v1223 / 32;
        auto v1225 = cooperative_groups::labeled_partition(v1222,v1224);
        Closure7 v1226{};
        int v1227; bool v1228;
        Tuple4 tmp14 = cooperative_groups::reduce(v1225, Tuple4{v1204, v1205}, v1226);
        v1227 = tmp14.v0; v1228 = tmp14.v1;
        bool v1229;
        v1229 = v1228 == false;
        if (v1229){
            assert("The local reduce must be true." && v1228);
        } else {
        }
        assert("Tensor range check" && 0 <= v994 && v994 < 8);
        int v1231;
        v1231 = 0;
        while (while_method_3(v1231)){
            assert("Tensor range check" && 0 <= v1231 && v1231 < 1);
            int v1233;
            v1233 = 128 * v1231;
            int v1234;
            v1234 = v1233 + v997;
            assert("Tensor range check" && 0 <= v1231 && v1231 < 1);
            int v1235;
            v1235 = 4 * v1231;
            int4* v1236;
            v1236 = reinterpret_cast<int4*>(v1082 + v1235);
            int4* v1237;
            v1237 = reinterpret_cast<int4*>(v14 + v1234);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1236) % 16 == 0 && reinterpret_cast<unsigned long long>(v1237) % 16 == 0);
            *v1237 = *v1236;
            v1231 += 1 ;
        }
        assert("Tensor range check" && 0 <= v994 && v994 < 8);
        int v1238;
        v1238 = 8 * v994;
        int v1239;
        v1239 = v1238 + v987;
        v15[v1239] = v1227;
        v994 += 1 ;
    }
    __syncthreads();
    int v1240;
    v1240 = threadIdx.x;
    int v1241;
    v1241 = blockIdx.x;
    int v1242;
    v1242 = v1241 * 256;
    int v1243;
    v1243 = v1240 + v1242;
    unsigned long long v1244;
    v1244 = (unsigned long long)v1243;
    curandStatePhilox4_32_10_t v1245;
    curand_init(12344321ull,v1244,0ull,&v1245);
    int v1246;
    v1246 = threadIdx.x;
    bool v1247;
    v1247 = 0 <= v1246;
    bool v1248;
    v1248 = v1247 == false;
    if (v1248){
        assert("The index needs to be zero or positive." && v1247);
    } else {
    }
    int v1250;
    v1250 = v1246 % 32;
    int v1251;
    v1251 = v1246 / 32;
    bool v1252;
    v1252 = v1251 < 8;
    bool v1253;
    v1253 = v1252 == false;
    if (v1253){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1252);
    } else {
    }
    assert("Tensor range check" && 0 <= v1251 && v1251 < 8);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 32);
    int v1255;
    v1255 = 4 * v1250;
    int v1256;
    v1256 = 128 * v1251;
    int v1257;
    v1257 = v1256 + v1255;
    assert("Tensor range check" && 0 <= v1251 && v1251 < 8);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 32);
    assert("Tensor range check" && 0 <= v1251 && v1251 < 8);
    int v1258;
    v1258 = 0;
    while (while_method_2(v1258)){
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1260;
        v1260 = 1024 * v1258;
        int v1261;
        v1261 = v1260 + v1257;
        float v1262[4];
        int v1263[4];
        int v1264;
        v1264 = 0;
        while (while_method_3(v1264)){
            assert("Tensor range check" && 0 <= v1264 && v1264 < 1);
            int v1266;
            v1266 = 4 * v1264;
            assert("Tensor range check" && 0 <= v1264 && v1264 < 1);
            int v1267;
            v1267 = 128 * v1264;
            int v1268;
            v1268 = v1267 + v1261;
            int4* v1269;
            v1269 = reinterpret_cast<int4*>(v1 + v1268);
            int4* v1270;
            v1270 = reinterpret_cast<int4*>(v1262 + v1266);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1269) % 16 == 0 && reinterpret_cast<unsigned long long>(v1270) % 16 == 0);
            *v1270 = *v1269;
            v1264 += 1 ;
        }
        int v1271;
        v1271 = 0;
        while (while_method_3(v1271)){
            int v1273;
            v1273 = 0;
            while (while_method_1(v1273)){
                bool v1275;
                v1275 = 0 <= v1273;
                bool v1277;
                if (v1275){
                    bool v1276;
                    v1276 = v1273 < 4;
                    v1277 = v1276;
                } else {
                    v1277 = false;
                }
                bool v1278;
                v1278 = v1277 == false;
                if (v1278){
                    assert("The indices should be inside the range of the dimension." && v1277);
                } else {
                }
                bool v1280;
                v1280 = 0 <= v1250;
                bool v1282;
                if (v1280){
                    bool v1281;
                    v1281 = v1250 < 32;
                    v1282 = v1281;
                } else {
                    v1282 = false;
                }
                bool v1283;
                v1283 = v1282 == false;
                if (v1283){
                    assert("The indices should be inside the range of the dimension." && v1282);
                } else {
                }
                int v1285;
                v1285 = v1250 * 4;
                int v1286;
                v1286 = v1273 + v1285;
                bool v1287;
                v1287 = 0 <= v1271;
                bool v1289;
                if (v1287){
                    bool v1288;
                    v1288 = v1271 < 1;
                    v1289 = v1288;
                } else {
                    v1289 = false;
                }
                bool v1290;
                v1290 = v1289 == false;
                if (v1290){
                    assert("The indices should be inside the range of the dimension." && v1289);
                } else {
                }
                int v1292;
                v1292 = v1271 * 128;
                int v1293;
                v1293 = v1286 + v1292;
                assert("Tensor range check" && 0 <= v1271 && v1271 < 1);
                assert("Tensor range check" && 0 <= v1273 && v1273 < 4);
                int v1294;
                v1294 = 4 * v1271;
                int v1295;
                v1295 = v1294 + v1273;
                v1263[v1295] = v1293;
                v1273 += 1 ;
            }
            v1271 += 1 ;
        }
        bool v1296;
        v1296 = 0 <= v1251;
        bool v1297;
        v1297 = v1296 && v1252;
        bool v1298;
        v1298 = v1297 == false;
        if (v1298){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1297);
        } else {
        }
        bool v1300;
        v1300 = 0 <= v1258;
        bool v1302;
        if (v1300){
            bool v1301;
            v1301 = v1258 < 8;
            v1302 = v1301;
        } else {
            v1302 = false;
        }
        bool v1303;
        v1303 = v1302 == false;
        if (v1303){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1302);
        } else {
        }
        int v1305;
        v1305 = v1258 * 8;
        int v1306;
        v1306 = v1305 + v1251;
        bool v1307[4];
        int v1308;
        v1308 = 0;
        while (while_method_3(v1308)){
            int v1310;
            v1310 = 0;
            while (while_method_1(v1310)){
                assert("Tensor range check" && 0 <= v1308 && v1308 < 1);
                assert("Tensor range check" && 0 <= v1310 && v1310 < 4);
                int v1312;
                v1312 = 4 * v1308;
                int v1313;
                v1313 = v1312 + v1310;
                float v1314;
                v1314 = v1262[v1313];
                int v1315;
                v1315 = v1263[v1313];
                bool v1316;
                v1316 = v1315 < 11;
                assert("Tensor range check" && 0 <= v1308 && v1308 < 1);
                assert("Tensor range check" && 0 <= v1310 && v1310 < 4);
                v1307[v1313] = v1316;
                v1310 += 1 ;
            }
            v1308 += 1 ;
        }
        int v1317[4];
        int v1318;
        v1318 = 0;
        while (while_method_3(v1318)){
            int v1320;
            v1320 = 0;
            while (while_method_1(v1320)){
                assert("Tensor range check" && 0 <= v1318 && v1318 < 1);
                assert("Tensor range check" && 0 <= v1320 && v1320 < 4);
                int v1322;
                v1322 = 4 * v1318;
                int v1323;
                v1323 = v1322 + v1320;
                bool v1324;
                v1324 = v1307[v1323];
                int v1325;
                if (v1324){
                    v1325 = 1;
                } else {
                    v1325 = 0;
                }
                assert("Tensor range check" && 0 <= v1318 && v1318 < 1);
                assert("Tensor range check" && 0 <= v1320 && v1320 < 4);
                v1317[v1323] = v1325;
                v1320 += 1 ;
            }
            v1318 += 1 ;
        }
        int v1326;
        v1326 = 0;
        int v1327;
        v1327 = 0;
        while (while_method_3(v1327)){
            int v1329;
            v1329 = 0;
            while (while_method_1(v1329)){
                assert("Tensor range check" && 0 <= v1327 && v1327 < 1);
                assert("Tensor range check" && 0 <= v1329 && v1329 < 4);
                int v1331;
                v1331 = 4 * v1327;
                int v1332;
                v1332 = v1331 + v1329;
                int v1333;
                v1333 = v1317[v1332];
                int v1334;
                v1334 = v1326 + v1333;
                v1326 = v1334;
                v1329 += 1 ;
            }
            v1327 += 1 ;
        }
        auto v1335 = cooperative_groups::coalesced_threads();
        int v1336;
        v1336 = threadIdx.x;
        int v1337;
        v1337 = v1336 / 32;
        auto v1338 = cooperative_groups::labeled_partition(v1335,v1337);
        Closure4 v1339{};
        int v1340;
        v1340 = cooperative_groups::reduce(v1338, v1326, v1339);
        float v1341[4];
        int v1342;
        v1342 = 0;
        while (while_method_3(v1342)){
            int v1344;
            v1344 = 0;
            while (while_method_1(v1344)){
                assert("Tensor range check" && 0 <= v1342 && v1342 < 1);
                assert("Tensor range check" && 0 <= v1344 && v1344 < 4);
                int v1346;
                v1346 = 4 * v1342;
                int v1347;
                v1347 = v1346 + v1344;
                float v1348;
                v1348 = v1262[v1347];
                bool v1349;
                v1349 = v1307[v1347];
                float v1350;
                if (v1349){
                    v1350 = v1348;
                } else {
                    v1350 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1342 && v1342 < 1);
                assert("Tensor range check" && 0 <= v1344 && v1344 < 4);
                v1341[v1347] = v1350;
                v1344 += 1 ;
            }
            v1342 += 1 ;
        }
        float v1351;
        v1351 = 0.0f;
        int v1352;
        v1352 = 0;
        while (while_method_3(v1352)){
            int v1354;
            v1354 = 0;
            while (while_method_1(v1354)){
                assert("Tensor range check" && 0 <= v1352 && v1352 < 1);
                assert("Tensor range check" && 0 <= v1354 && v1354 < 4);
                int v1356;
                v1356 = 4 * v1352;
                int v1357;
                v1357 = v1356 + v1354;
                float v1358;
                v1358 = v1341[v1357];
                float v1359;
                v1359 = v1351 + v1358;
                v1351 = v1359;
                v1354 += 1 ;
            }
            v1352 += 1 ;
        }
        auto v1360 = cooperative_groups::coalesced_threads();
        int v1361;
        v1361 = threadIdx.x;
        int v1362;
        v1362 = v1361 / 32;
        auto v1363 = cooperative_groups::labeled_partition(v1360,v1362);
        float v1364;
        v1364 = cooperative_groups::reduce(v1363, v1351, v42);
        float v1365;
        v1365 = (float)v1340;
        float v1366;
        v1366 = v1364 / v1365;
        float v1367[4];
        int v1368;
        v1368 = 0;
        while (while_method_3(v1368)){
            int v1370;
            v1370 = 0;
            while (while_method_1(v1370)){
                assert("Tensor range check" && 0 <= v1368 && v1368 < 1);
                assert("Tensor range check" && 0 <= v1370 && v1370 < 4);
                int v1372;
                v1372 = 4 * v1368;
                int v1373;
                v1373 = v1372 + v1370;
                float v1374;
                v1374 = v1262[v1373];
                bool v1375;
                v1375 = v1307[v1373];
                float v1376;
                if (v1375){
                    v1376 = v1374;
                } else {
                    v1376 = -1.0f / 0.0f;
                }
                float v1377;
                v1377 = v1376 - v1366;
                float v1378;
                v1378 = exp(v1377);
                assert("Tensor range check" && 0 <= v1368 && v1368 < 1);
                assert("Tensor range check" && 0 <= v1370 && v1370 < 4);
                v1367[v1373] = v1378;
                v1370 += 1 ;
            }
            v1368 += 1 ;
        }
        float v1379;
        v1379 = 0.0f;
        int v1380;
        v1380 = 0;
        while (while_method_3(v1380)){
            int v1382;
            v1382 = 0;
            while (while_method_1(v1382)){
                assert("Tensor range check" && 0 <= v1380 && v1380 < 1);
                assert("Tensor range check" && 0 <= v1382 && v1382 < 4);
                int v1384;
                v1384 = 4 * v1380;
                int v1385;
                v1385 = v1384 + v1382;
                float v1386;
                v1386 = v1367[v1385];
                float v1387;
                v1387 = v1379 + v1386;
                v1379 = v1387;
                v1382 += 1 ;
            }
            v1380 += 1 ;
        }
        auto v1388 = cooperative_groups::coalesced_threads();
        int v1389;
        v1389 = threadIdx.x;
        int v1390;
        v1390 = v1389 / 32;
        auto v1391 = cooperative_groups::labeled_partition(v1388,v1390);
        float v1392;
        v1392 = cooperative_groups::reduce(v1391, v1379, v42);
        float v1393[4];
        int v1394;
        v1394 = 0;
        while (while_method_3(v1394)){
            int v1396;
            v1396 = 0;
            while (while_method_1(v1396)){
                assert("Tensor range check" && 0 <= v1394 && v1394 < 1);
                assert("Tensor range check" && 0 <= v1396 && v1396 < 4);
                int v1398;
                v1398 = 4 * v1394;
                int v1399;
                v1399 = v1398 + v1396;
                float v1400;
                v1400 = v1367[v1399];
                float v1401;
                v1401 = v1400 / v1392;
                assert("Tensor range check" && 0 <= v1394 && v1394 < 1);
                assert("Tensor range check" && 0 <= v1396 && v1396 < 4);
                v1393[v1399] = v1401;
                v1396 += 1 ;
            }
            v1394 += 1 ;
        }
        float v1402[4];
        float v1403;
        v1403 = 0.0f;
        int v1404;
        v1404 = 0;
        while (while_method_3(v1404)){
            assert("Tensor range check" && 0 <= v1404 && v1404 < 1);
            int v1406;
            v1406 = 4 * v1404;
            assert("Tensor range check" && 0 <= v1404 && v1404 < 1);
            int v1407; float v1408;
            Tuple0 tmp15 = Tuple0{0, 0.0f};
            v1407 = tmp15.v0; v1408 = tmp15.v1;
            while (while_method_1(v1407)){
                assert("Tensor range check" && 0 <= v1407 && v1407 < 4);
                int v1410;
                v1410 = v1407 + v1406;
                float v1411;
                v1411 = v1393[v1410];
                float v1412;
                v1412 = v1408 + v1411;
                v1408 = v1412;
                v1407 += 1 ;
            }
            auto v1413 = cooperative_groups::coalesced_threads();
            int v1414;
            v1414 = threadIdx.x;
            int v1415;
            v1415 = v1414 / 32;
            auto v1416 = cooperative_groups::labeled_partition(v1413,v1415);
            Closure2 v1417{};
            float v1418;
            v1418 = cooperative_groups::inclusive_scan(v1416, v1408, v1417);
            float v1419;
            v1419 = v1416.shfl_up(v1418,1);
            bool v1420;
            v1420 = v1416.thread_rank() == 0;
            float v1421;
            if (v1420){
                v1421 = 0.0f;
            } else {
                v1421 = v1419;
            }
            float v1422;
            v1422 = v1416.shfl(v1418,v1416.num_threads()-1);
            float v1423;
            v1423 = v1403 + v1421;
            int v1424; float v1425;
            Tuple0 tmp16 = Tuple0{0, v1423};
            v1424 = tmp16.v0; v1425 = tmp16.v1;
            while (while_method_1(v1424)){
                assert("Tensor range check" && 0 <= v1424 && v1424 < 4);
                int v1427;
                v1427 = v1424 + v1406;
                float v1428;
                v1428 = v1393[v1427];
                float v1429;
                v1429 = v1425 + v1428;
                assert("Tensor range check" && 0 <= v1424 && v1424 < 4);
                v1402[v1427] = v1429;
                v1425 = v1429;
                v1424 += 1 ;
            }
            float v1430;
            v1430 = v1403 + v1422;
            v1403 = v1430;
            v1404 += 1 ;
        }
        float v1431[4];
        bool v1432[4];
        int v1433;
        v1433 = 0;
        while (while_method_3(v1433)){
            int v1435;
            v1435 = 0;
            while (while_method_1(v1435)){
                assert("Tensor range check" && 0 <= v1433 && v1433 < 1);
                assert("Tensor range check" && 0 <= v1435 && v1435 < 4);
                int v1437;
                v1437 = 4 * v1433;
                int v1438;
                v1438 = v1437 + v1435;
                float v1439;
                v1439 = v1402[v1438];
                float v1440;
                v1440 = v1393[v1438];
                bool v1441;
                v1441 = v1440 > 0.0f;
                assert("Tensor range check" && 0 <= v1433 && v1433 < 1);
                assert("Tensor range check" && 0 <= v1435 && v1435 < 4);
                v1431[v1438] = v1439;
                v1432[v1438] = v1441;
                v1435 += 1 ;
            }
            v1433 += 1 ;
        }
        float v1442; bool v1443;
        Tuple3 tmp17 = Tuple3{-1.0f / 0.0f, false};
        v1442 = tmp17.v0; v1443 = tmp17.v1;
        int v1444;
        v1444 = 0;
        while (while_method_3(v1444)){
            int v1446;
            v1446 = 0;
            while (while_method_1(v1446)){
                assert("Tensor range check" && 0 <= v1444 && v1444 < 1);
                assert("Tensor range check" && 0 <= v1446 && v1446 < 4);
                int v1448;
                v1448 = 4 * v1444;
                int v1449;
                v1449 = v1448 + v1446;
                float v1450;
                v1450 = v1431[v1449];
                bool v1451;
                v1451 = v1432[v1449];
                float v1458; bool v1459;
                if (v1443){
                    if (v1451){
                        bool v1452;
                        v1452 = v1442 >= v1450;
                        float v1453;
                        if (v1452){
                            v1453 = v1442;
                        } else {
                            v1453 = v1450;
                        }
                        v1458 = v1453; v1459 = true;
                    } else {
                        v1458 = v1442; v1459 = v1443;
                    }
                } else {
                    if (v1451){
                        v1458 = v1450; v1459 = v1451;
                    } else {
                        v1458 = v1442; v1459 = v1443;
                    }
                }
                v1442 = v1458;
                v1443 = v1459;
                v1446 += 1 ;
            }
            v1444 += 1 ;
        }
        auto v1460 = cooperative_groups::coalesced_threads();
        int v1461;
        v1461 = threadIdx.x;
        int v1462;
        v1462 = v1461 / 32;
        auto v1463 = cooperative_groups::labeled_partition(v1460,v1462);
        Closure5 v1464{};
        float v1465; bool v1466;
        Tuple3 tmp18 = cooperative_groups::reduce(v1463, Tuple3{v1442, v1443}, v1464);
        v1465 = tmp18.v0; v1466 = tmp18.v1;
        bool v1467;
        v1467 = v1466 == false;
        if (v1467){
            assert("The local reduce must be true." && v1466);
        } else {
        }
        float v1469[4];
        int v1470[4];
        int v1471;
        v1471 = 0;
        while (while_method_3(v1471)){
            int v1473;
            v1473 = 0;
            while (while_method_1(v1473)){
                assert("Tensor range check" && 0 <= v1471 && v1471 < 1);
                assert("Tensor range check" && 0 <= v1473 && v1473 < 4);
                int v1475;
                v1475 = 4 * v1471;
                int v1476;
                v1476 = v1475 + v1473;
                int v1477;
                v1477 = v1263[v1476];
                float v1478;
                v1478 = curand_uniform(&v1245);
                assert("Tensor range check" && 0 <= v1471 && v1471 < 1);
                assert("Tensor range check" && 0 <= v1473 && v1473 < 4);
                v1469[v1476] = v1478;
                v1470[v1476] = v1477;
                v1473 += 1 ;
            }
            v1471 += 1 ;
        }
        float v1479; int v1480;
        Tuple1 tmp19 = Tuple1{0.0f, 2147483647};
        v1479 = tmp19.v0; v1480 = tmp19.v1;
        int v1481;
        v1481 = 0;
        while (while_method_3(v1481)){
            int v1483;
            v1483 = 0;
            while (while_method_1(v1483)){
                assert("Tensor range check" && 0 <= v1481 && v1481 < 1);
                assert("Tensor range check" && 0 <= v1483 && v1483 < 4);
                int v1485;
                v1485 = 4 * v1481;
                int v1486;
                v1486 = v1485 + v1483;
                float v1487;
                v1487 = v1469[v1486];
                int v1488;
                v1488 = v1470[v1486];
                bool v1489;
                v1489 = v1480 < v1488;
                float v1490; int v1491;
                if (v1489){
                    v1490 = v1479; v1491 = v1480;
                } else {
                    v1490 = v1487; v1491 = v1488;
                }
                v1479 = v1490;
                v1480 = v1491;
                v1483 += 1 ;
            }
            v1481 += 1 ;
        }
        auto v1492 = cooperative_groups::coalesced_threads();
        int v1493;
        v1493 = threadIdx.x;
        int v1494;
        v1494 = v1493 / 32;
        auto v1495 = cooperative_groups::labeled_partition(v1492,v1494);
        Closure6 v1496{};
        float v1497; int v1498;
        Tuple1 tmp20 = cooperative_groups::reduce(v1495, Tuple1{v1479, v1480}, v1496);
        v1497 = tmp20.v0; v1498 = tmp20.v1;
        float v1499;
        v1499 = v1465 * v1497;
        int v1500[4];
        bool v1501[4];
        int v1502;
        v1502 = 0;
        while (while_method_3(v1502)){
            int v1504;
            v1504 = 0;
            while (while_method_1(v1504)){
                assert("Tensor range check" && 0 <= v1502 && v1502 < 1);
                assert("Tensor range check" && 0 <= v1504 && v1504 < 4);
                int v1506;
                v1506 = 4 * v1502;
                int v1507;
                v1507 = v1506 + v1504;
                float v1508;
                v1508 = v1431[v1507];
                bool v1509;
                v1509 = v1432[v1507];
                int v1510;
                v1510 = v1263[v1507];
                int v1513; bool v1514;
                if (v1509){
                    float v1511;
                    v1511 = v1508 - v1499;
                    bool v1512;
                    v1512 = v1511 >= 0.0f;
                    v1513 = v1510; v1514 = v1512;
                } else {
                    v1513 = 2147483647; v1514 = false;
                }
                assert("Tensor range check" && 0 <= v1502 && v1502 < 1);
                assert("Tensor range check" && 0 <= v1504 && v1504 < 4);
                v1500[v1507] = v1513;
                v1501[v1507] = v1514;
                v1504 += 1 ;
            }
            v1502 += 1 ;
        }
        int v1515; bool v1516;
        Tuple4 tmp21 = Tuple4{2147483647, false};
        v1515 = tmp21.v0; v1516 = tmp21.v1;
        int v1517;
        v1517 = 0;
        while (while_method_3(v1517)){
            int v1519;
            v1519 = 0;
            while (while_method_1(v1519)){
                assert("Tensor range check" && 0 <= v1517 && v1517 < 1);
                assert("Tensor range check" && 0 <= v1519 && v1519 < 4);
                int v1521;
                v1521 = 4 * v1517;
                int v1522;
                v1522 = v1521 + v1519;
                int v1523;
                v1523 = v1500[v1522];
                bool v1524;
                v1524 = v1501[v1522];
                int v1531; bool v1532;
                if (v1516){
                    if (v1524){
                        bool v1525;
                        v1525 = v1515 < v1523;
                        int v1526;
                        if (v1525){
                            v1526 = v1515;
                        } else {
                            v1526 = v1523;
                        }
                        v1531 = v1526; v1532 = true;
                    } else {
                        v1531 = v1515; v1532 = v1516;
                    }
                } else {
                    if (v1524){
                        v1531 = v1523; v1532 = v1524;
                    } else {
                        v1531 = v1515; v1532 = v1516;
                    }
                }
                v1515 = v1531;
                v1516 = v1532;
                v1519 += 1 ;
            }
            v1517 += 1 ;
        }
        auto v1533 = cooperative_groups::coalesced_threads();
        int v1534;
        v1534 = threadIdx.x;
        int v1535;
        v1535 = v1534 / 32;
        auto v1536 = cooperative_groups::labeled_partition(v1533,v1535);
        Closure7 v1537{};
        int v1538; bool v1539;
        Tuple4 tmp22 = cooperative_groups::reduce(v1536, Tuple4{v1515, v1516}, v1537);
        v1538 = tmp22.v0; v1539 = tmp22.v1;
        bool v1540;
        v1540 = v1539 == false;
        if (v1540){
            assert("The local reduce must be true." && v1539);
        } else {
        }
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1542;
        v1542 = 0;
        while (while_method_3(v1542)){
            assert("Tensor range check" && 0 <= v1542 && v1542 < 1);
            int v1544;
            v1544 = 128 * v1542;
            int v1545;
            v1545 = v1544 + v1261;
            assert("Tensor range check" && 0 <= v1542 && v1542 < 1);
            int v1546;
            v1546 = 4 * v1542;
            int4* v1547;
            v1547 = reinterpret_cast<int4*>(v1393 + v1546);
            int4* v1548;
            v1548 = reinterpret_cast<int4*>(v16 + v1545);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1547) % 16 == 0 && reinterpret_cast<unsigned long long>(v1548) % 16 == 0);
            *v1548 = *v1547;
            v1542 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1549;
        v1549 = 8 * v1258;
        int v1550;
        v1550 = v1549 + v1251;
        v17[v1550] = v1538;
        v1258 += 1 ;
    }
    __syncthreads();
    return ;
}
extern "C" __global__ void entry1(int * v0, float * v1, float * v2, int * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int * v9, int * v10, int * v11, int * v12, int * v13, float * v14, int * v15, float * v16, int * v17) {
    float v18;
    v18 = 0.0f;
    int v19;
    v19 = threadIdx.x;
    int v20;
    v20 = v19;
    while (while_method_0(v20)){
        bool v22;
        v22 = 0 <= v20;
        bool v23;
        v23 = v22 == false;
        if (v23){
            assert("The index needs to be zero or positive." && v22);
        } else {
        }
        int v25;
        v25 = v20 % 16;
        int v26;
        v26 = v20 / 16;
        bool v27;
        v27 = v26 < 128;
        bool v28;
        v28 = v27 == false;
        if (v28){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v27);
        } else {
        }
        assert("Tensor range check" && 0 <= v26 && v26 < 128);
        assert("Tensor range check" && 0 <= v25 && v25 < 16);
        int v30;
        v30 = 4 * v25;
        int v31;
        v31 = 64 * v26;
        int v32;
        v32 = v31 + v30;
        float v33[4];
        int4* v34;
        v34 = reinterpret_cast<int4*>(v1 + v32);
        int4* v35;
        v35 = reinterpret_cast<int4*>(v33 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v34) % 16 == 0 && reinterpret_cast<unsigned long long>(v35) % 16 == 0);
        *v35 = *v34;
        int v36; float v37;
        Tuple0 tmp23 = Tuple0{0, v18};
        v36 = tmp23.v0; v37 = tmp23.v1;
        while (while_method_1(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 4);
            float v39;
            v39 = v33[v36];
            float v40;
            v40 = v37 + v39;
            v37 = v40;
            v36 += 1 ;
        }
        v18 = v37;
        v20 += 256 ;
    }
    auto v41 = cooperative_groups::coalesced_threads();
    Closure0 v42{};
    float v43;
    v43 = cooperative_groups::reduce(v41, v18, v42);
    int v44;
    v44 = threadIdx.x;
    int v45;
    v45 = v44 / 32;
    extern __shared__ unsigned char v46[];
    float * v47;
    v47 = reinterpret_cast<float *>(&v46[0ull]);
    assert("Tensor range check" && 0 <= v45 && v45 < 8);
    v47[v45] = v43;
    __syncthreads();
    int v49;
    v49 = threadIdx.x;
    int v50;
    v50 = v49 % 32;
    bool v51;
    v51 = v45 == 0;
    bool v53;
    if (v51){
        bool v52;
        v52 = v50 < 8;
        v53 = v52;
    } else {
        v53 = false;
    }
    if (v53){
        auto v54 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v50 && v50 < 8);
        float v55;
        v55 = v47[v50];
        float v56;
        v56 = cooperative_groups::reduce(v54, v55, v42);
        v2[0] = v56;
    } else {
    }
    __syncthreads();
    int v57;
    v57 = threadIdx.x;
    bool v58;
    v58 = 0 <= v57;
    bool v59;
    v59 = v58 == false;
    if (v59){
        assert("The index needs to be zero or positive." && v58);
    } else {
    }
    int v61;
    v61 = v57 % 16;
    int v62;
    v62 = v57 / 16;
    bool v63;
    v63 = v62 < 16;
    bool v64;
    v64 = v63 == false;
    if (v64){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v63);
    } else {
    }
    assert("Tensor range check" && 0 <= v62 && v62 < 16);
    assert("Tensor range check" && 0 <= v61 && v61 < 16);
    int v66;
    v66 = 4 * v61;
    int v67;
    v67 = 64 * v62;
    int v68;
    v68 = v67 + v66;
    assert("Tensor range check" && 0 <= v62 && v62 < 16);
    assert("Tensor range check" && 0 <= v61 && v61 < 16);
    int v69;
    v69 = 0;
    while (while_method_2(v69)){
        assert("Tensor range check" && 0 <= v69 && v69 < 8);
        int v71;
        v71 = 1024 * v69;
        int v72;
        v72 = v71 + v68;
        int v73[4];
        int v74[4];
        int v75;
        v75 = 0;
        while (while_method_3(v75)){
            assert("Tensor range check" && 0 <= v75 && v75 < 1);
            int v77;
            v77 = 4 * v75;
            assert("Tensor range check" && 0 <= v75 && v75 < 1);
            int v78;
            v78 = 64 * v75;
            int v79;
            v79 = v78 + v72;
            int4* v80;
            v80 = reinterpret_cast<int4*>(v0 + v79);
            int4* v81;
            v81 = reinterpret_cast<int4*>(v73 + v77);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v80) % 16 == 0 && reinterpret_cast<unsigned long long>(v81) % 16 == 0);
            *v81 = *v80;
            v75 += 1 ;
        }
        int v82;
        v82 = 0;
        while (while_method_3(v82)){
            int v84;
            v84 = 0;
            while (while_method_1(v84)){
                bool v86;
                v86 = 0 <= v84;
                bool v88;
                if (v86){
                    bool v87;
                    v87 = v84 < 4;
                    v88 = v87;
                } else {
                    v88 = false;
                }
                bool v89;
                v89 = v88 == false;
                if (v89){
                    assert("The indices should be inside the range of the dimension." && v88);
                } else {
                }
                bool v91;
                v91 = 0 <= v61;
                bool v93;
                if (v91){
                    bool v92;
                    v92 = v61 < 16;
                    v93 = v92;
                } else {
                    v93 = false;
                }
                bool v94;
                v94 = v93 == false;
                if (v94){
                    assert("The indices should be inside the range of the dimension." && v93);
                } else {
                }
                int v96;
                v96 = v61 * 4;
                int v97;
                v97 = v84 + v96;
                bool v98;
                v98 = 0 <= v82;
                bool v100;
                if (v98){
                    bool v99;
                    v99 = v82 < 1;
                    v100 = v99;
                } else {
                    v100 = false;
                }
                bool v101;
                v101 = v100 == false;
                if (v101){
                    assert("The indices should be inside the range of the dimension." && v100);
                } else {
                }
                int v103;
                v103 = v82 * 64;
                int v104;
                v104 = v97 + v103;
                assert("Tensor range check" && 0 <= v82 && v82 < 1);
                assert("Tensor range check" && 0 <= v84 && v84 < 4);
                int v105;
                v105 = 4 * v82;
                int v106;
                v106 = v105 + v84;
                v74[v106] = v104;
                v84 += 1 ;
            }
            v82 += 1 ;
        }
        bool v107;
        v107 = 0 <= v62;
        bool v108;
        v108 = v107 && v63;
        bool v109;
        v109 = v108 == false;
        if (v109){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v108);
        } else {
        }
        bool v111;
        v111 = 0 <= v69;
        bool v113;
        if (v111){
            bool v112;
            v112 = v69 < 8;
            v113 = v112;
        } else {
            v113 = false;
        }
        bool v114;
        v114 = v113 == false;
        if (v114){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v113);
        } else {
        }
        int v116;
        v116 = v69 * 16;
        int v117;
        v117 = v116 + v62;
        assert("Tensor range check" && 0 <= v69 && v69 < 8);
        int v118;
        v118 = 0;
        while (while_method_3(v118)){
            assert("Tensor range check" && 0 <= v118 && v118 < 1);
            int v120;
            v120 = 64 * v118;
            int v121;
            v121 = v120 + v72;
            assert("Tensor range check" && 0 <= v118 && v118 < 1);
            int v122;
            v122 = 4 * v118;
            int4* v123;
            v123 = reinterpret_cast<int4*>(v73 + v122);
            int4* v124;
            v124 = reinterpret_cast<int4*>(v3 + v121);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v123) % 16 == 0 && reinterpret_cast<unsigned long long>(v124) % 16 == 0);
            *v124 = *v123;
            v118 += 1 ;
        }
        v69 += 1 ;
    }
    __syncthreads();
    int v125;
    v125 = threadIdx.x;
    bool v126;
    v126 = 0 <= v125;
    bool v127;
    v127 = v126 == false;
    if (v127){
        assert("The index needs to be zero or positive." && v126);
    } else {
    }
    int v129;
    v129 = v125 % 16;
    int v130;
    v130 = v125 / 16;
    bool v131;
    v131 = v130 < 16;
    bool v132;
    v132 = v131 == false;
    if (v132){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v131);
    } else {
    }
    assert("Tensor range check" && 0 <= v130 && v130 < 16);
    assert("Tensor range check" && 0 <= v129 && v129 < 16);
    int v134;
    v134 = 4 * v129;
    int v135;
    v135 = 64 * v130;
    int v136;
    v136 = v135 + v134;
    assert("Tensor range check" && 0 <= v130 && v130 < 16);
    assert("Tensor range check" && 0 <= v129 && v129 < 16);
    int v137;
    v137 = 0;
    while (while_method_2(v137)){
        assert("Tensor range check" && 0 <= v137 && v137 < 8);
        int v139;
        v139 = 1024 * v137;
        int v140;
        v140 = v139 + v136;
        float v141[4];
        int v142[4];
        int v143;
        v143 = 0;
        while (while_method_3(v143)){
            assert("Tensor range check" && 0 <= v143 && v143 < 1);
            int v145;
            v145 = 4 * v143;
            assert("Tensor range check" && 0 <= v143 && v143 < 1);
            int v146;
            v146 = 64 * v143;
            int v147;
            v147 = v146 + v140;
            int4* v148;
            v148 = reinterpret_cast<int4*>(v1 + v147);
            int4* v149;
            v149 = reinterpret_cast<int4*>(v141 + v145);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v148) % 16 == 0 && reinterpret_cast<unsigned long long>(v149) % 16 == 0);
            *v149 = *v148;
            v143 += 1 ;
        }
        int v150;
        v150 = 0;
        while (while_method_3(v150)){
            int v152;
            v152 = 0;
            while (while_method_1(v152)){
                bool v154;
                v154 = 0 <= v152;
                bool v156;
                if (v154){
                    bool v155;
                    v155 = v152 < 4;
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
                bool v159;
                v159 = 0 <= v129;
                bool v161;
                if (v159){
                    bool v160;
                    v160 = v129 < 16;
                    v161 = v160;
                } else {
                    v161 = false;
                }
                bool v162;
                v162 = v161 == false;
                if (v162){
                    assert("The indices should be inside the range of the dimension." && v161);
                } else {
                }
                int v164;
                v164 = v129 * 4;
                int v165;
                v165 = v152 + v164;
                bool v166;
                v166 = 0 <= v150;
                bool v168;
                if (v166){
                    bool v167;
                    v167 = v150 < 1;
                    v168 = v167;
                } else {
                    v168 = false;
                }
                bool v169;
                v169 = v168 == false;
                if (v169){
                    assert("The indices should be inside the range of the dimension." && v168);
                } else {
                }
                int v171;
                v171 = v150 * 64;
                int v172;
                v172 = v165 + v171;
                assert("Tensor range check" && 0 <= v150 && v150 < 1);
                assert("Tensor range check" && 0 <= v152 && v152 < 4);
                int v173;
                v173 = 4 * v150;
                int v174;
                v174 = v173 + v152;
                v142[v174] = v172;
                v152 += 1 ;
            }
            v150 += 1 ;
        }
        bool v175;
        v175 = 0 <= v130;
        bool v176;
        v176 = v175 && v131;
        bool v177;
        v177 = v176 == false;
        if (v177){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v176);
        } else {
        }
        bool v179;
        v179 = 0 <= v137;
        bool v181;
        if (v179){
            bool v180;
            v180 = v137 < 8;
            v181 = v180;
        } else {
            v181 = false;
        }
        bool v182;
        v182 = v181 == false;
        if (v182){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v181);
        } else {
        }
        int v184;
        v184 = v137 * 16;
        int v185;
        v185 = v184 + v130;
        int v186[4];
        int v187[4];
        int v188;
        v188 = 0;
        while (while_method_3(v188)){
            int v190;
            v190 = 0;
            while (while_method_1(v190)){
                assert("Tensor range check" && 0 <= v188 && v188 < 1);
                assert("Tensor range check" && 0 <= v190 && v190 < 4);
                int v192;
                v192 = 4 * v188;
                int v193;
                v193 = v192 + v190;
                int v194;
                v194 = v142[v193];
                assert("Tensor range check" && 0 <= v188 && v188 < 1);
                assert("Tensor range check" && 0 <= v190 && v190 < 4);
                v186[v193] = v185;
                v187[v193] = v194;
                v190 += 1 ;
            }
            v188 += 1 ;
        }
        assert("Tensor range check" && 0 <= v137 && v137 < 8);
        int v195;
        v195 = 0;
        while (while_method_3(v195)){
            assert("Tensor range check" && 0 <= v195 && v195 < 1);
            int v197;
            v197 = 64 * v195;
            int v198;
            v198 = v197 + v140;
            assert("Tensor range check" && 0 <= v195 && v195 < 1);
            int v199;
            v199 = 4 * v195;
            int4* v200;
            v200 = reinterpret_cast<int4*>(v186 + v199);
            int4* v201;
            v201 = reinterpret_cast<int4*>(v10 + v198);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v200) % 16 == 0 && reinterpret_cast<unsigned long long>(v201) % 16 == 0);
            *v201 = *v200;
            int4* v202;
            v202 = reinterpret_cast<int4*>(v187 + v199);
            int4* v203;
            v203 = reinterpret_cast<int4*>(v11 + v198);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v202) % 16 == 0 && reinterpret_cast<unsigned long long>(v203) % 16 == 0);
            *v203 = *v202;
            v195 += 1 ;
        }
        v137 += 1 ;
    }
    __syncthreads();
    int v204;
    v204 = threadIdx.x;
    bool v205;
    v205 = 0 <= v204;
    bool v206;
    v206 = v205 == false;
    if (v206){
        assert("The index needs to be zero or positive." && v205);
    } else {
    }
    int v208;
    v208 = v204 % 16;
    int v209;
    v209 = v204 / 16;
    bool v210;
    v210 = v209 < 16;
    bool v211;
    v211 = v210 == false;
    if (v211){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v210);
    } else {
    }
    assert("Tensor range check" && 0 <= v209 && v209 < 16);
    assert("Tensor range check" && 0 <= v208 && v208 < 16);
    int v213;
    v213 = 4 * v208;
    int v214;
    v214 = 64 * v209;
    int v215;
    v215 = v214 + v213;
    assert("Tensor range check" && 0 <= v209 && v209 < 16);
    int v216;
    v216 = 0;
    while (while_method_2(v216)){
        assert("Tensor range check" && 0 <= v216 && v216 < 8);
        int v218;
        v218 = 1024 * v216;
        int v219;
        v219 = v218 + v215;
        float v220[4];
        int v221[4];
        int v222;
        v222 = 0;
        while (while_method_3(v222)){
            assert("Tensor range check" && 0 <= v222 && v222 < 1);
            int v224;
            v224 = 4 * v222;
            assert("Tensor range check" && 0 <= v222 && v222 < 1);
            int v225;
            v225 = 64 * v222;
            int v226;
            v226 = v225 + v219;
            int4* v227;
            v227 = reinterpret_cast<int4*>(v1 + v226);
            int4* v228;
            v228 = reinterpret_cast<int4*>(v220 + v224);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v227) % 16 == 0 && reinterpret_cast<unsigned long long>(v228) % 16 == 0);
            *v228 = *v227;
            v222 += 1 ;
        }
        int v229;
        v229 = 0;
        while (while_method_3(v229)){
            int v231;
            v231 = 0;
            while (while_method_1(v231)){
                bool v233;
                v233 = 0 <= v231;
                bool v235;
                if (v233){
                    bool v234;
                    v234 = v231 < 4;
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
                v238 = 0 <= v208;
                bool v240;
                if (v238){
                    bool v239;
                    v239 = v208 < 16;
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
                v243 = v208 * 4;
                int v244;
                v244 = v231 + v243;
                bool v245;
                v245 = 0 <= v229;
                bool v247;
                if (v245){
                    bool v246;
                    v246 = v229 < 1;
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
                v250 = v229 * 64;
                int v251;
                v251 = v244 + v250;
                assert("Tensor range check" && 0 <= v229 && v229 < 1);
                assert("Tensor range check" && 0 <= v231 && v231 < 4);
                int v252;
                v252 = 4 * v229;
                int v253;
                v253 = v252 + v231;
                v221[v253] = v251;
                v231 += 1 ;
            }
            v229 += 1 ;
        }
        bool v254;
        v254 = 0 <= v209;
        bool v255;
        v255 = v254 && v210;
        bool v256;
        v256 = v255 == false;
        if (v256){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v255);
        } else {
        }
        bool v258;
        v258 = 0 <= v216;
        bool v260;
        if (v258){
            bool v259;
            v259 = v216 < 8;
            v260 = v259;
        } else {
            v260 = false;
        }
        bool v261;
        v261 = v260 == false;
        if (v261){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v260);
        } else {
        }
        int v263;
        v263 = v216 * 16;
        int v264;
        v264 = v263 + v209;
        assert("Tensor range check" && 0 <= v216 && v216 < 8);
        int v265;
        v265 = 16 * v216;
        int v266;
        v266 = v265 + v209;
        v12[v266] = v264;
        v216 += 1 ;
    }
    __syncthreads();
    int v267;
    v267 = threadIdx.x;
    bool v268;
    v268 = 0 <= v267;
    bool v269;
    v269 = v268 == false;
    if (v269){
        assert("The index needs to be zero or positive." && v268);
    } else {
    }
    int v271;
    v271 = v267 % 16;
    int v272;
    v272 = v267 / 16;
    bool v273;
    v273 = v272 < 16;
    bool v274;
    v274 = v273 == false;
    if (v274){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v273);
    } else {
    }
    assert("Tensor range check" && 0 <= v272 && v272 < 16);
    assert("Tensor range check" && 0 <= v271 && v271 < 16);
    int v276;
    v276 = 4 * v271;
    int v277;
    v277 = 64 * v272;
    int v278;
    v278 = v277 + v276;
    assert("Tensor range check" && 0 <= v272 && v272 < 16);
    assert("Tensor range check" && 0 <= v271 && v271 < 16);
    int v279;
    v279 = 0;
    while (while_method_2(v279)){
        assert("Tensor range check" && 0 <= v279 && v279 < 8);
        int v281;
        v281 = 1024 * v279;
        int v282;
        v282 = v281 + v278;
        float v283[4];
        int v284[4];
        int v285;
        v285 = 0;
        while (while_method_3(v285)){
            assert("Tensor range check" && 0 <= v285 && v285 < 1);
            int v287;
            v287 = 4 * v285;
            assert("Tensor range check" && 0 <= v285 && v285 < 1);
            int v288;
            v288 = 64 * v285;
            int v289;
            v289 = v288 + v282;
            int4* v290;
            v290 = reinterpret_cast<int4*>(v1 + v289);
            int4* v291;
            v291 = reinterpret_cast<int4*>(v283 + v287);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v290) % 16 == 0 && reinterpret_cast<unsigned long long>(v291) % 16 == 0);
            *v291 = *v290;
            v285 += 1 ;
        }
        int v292;
        v292 = 0;
        while (while_method_3(v292)){
            int v294;
            v294 = 0;
            while (while_method_1(v294)){
                bool v296;
                v296 = 0 <= v294;
                bool v298;
                if (v296){
                    bool v297;
                    v297 = v294 < 4;
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
                v301 = 0 <= v271;
                bool v303;
                if (v301){
                    bool v302;
                    v302 = v271 < 16;
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
                v306 = v271 * 4;
                int v307;
                v307 = v294 + v306;
                bool v308;
                v308 = 0 <= v292;
                bool v310;
                if (v308){
                    bool v309;
                    v309 = v292 < 1;
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
                v313 = v292 * 64;
                int v314;
                v314 = v307 + v313;
                assert("Tensor range check" && 0 <= v292 && v292 < 1);
                assert("Tensor range check" && 0 <= v294 && v294 < 4);
                int v315;
                v315 = 4 * v292;
                int v316;
                v316 = v315 + v294;
                v284[v316] = v314;
                v294 += 1 ;
            }
            v292 += 1 ;
        }
        bool v317;
        v317 = 0 <= v272;
        bool v318;
        v318 = v317 && v273;
        bool v319;
        v319 = v318 == false;
        if (v319){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v318);
        } else {
        }
        bool v321;
        v321 = 0 <= v279;
        bool v323;
        if (v321){
            bool v322;
            v322 = v279 < 8;
            v323 = v322;
        } else {
            v323 = false;
        }
        bool v324;
        v324 = v323 == false;
        if (v324){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v323);
        } else {
        }
        int v326;
        v326 = v279 * 16;
        int v327;
        v327 = v326 + v272;
        float v328;
        v328 = 0.0f;
        int v329;
        v329 = 0;
        while (while_method_3(v329)){
            int v331;
            v331 = 0;
            while (while_method_1(v331)){
                assert("Tensor range check" && 0 <= v329 && v329 < 1);
                assert("Tensor range check" && 0 <= v331 && v331 < 4);
                int v333;
                v333 = 4 * v329;
                int v334;
                v334 = v333 + v331;
                float v335;
                v335 = v283[v334];
                float v336;
                v336 = v328 + v335;
                v328 = v336;
                v331 += 1 ;
            }
            v329 += 1 ;
        }
        auto v337 = cooperative_groups::coalesced_threads();
        int v338;
        v338 = threadIdx.x;
        int v339;
        v339 = v338 / 16;
        auto v340 = cooperative_groups::labeled_partition(v337,v339);
        float v341;
        v341 = cooperative_groups::reduce(v340, v328, v42);
        float v342;
        v342 = v341 / 64.0f;
        float v343[4];
        int v344;
        v344 = 0;
        while (while_method_3(v344)){
            int v346;
            v346 = 0;
            while (while_method_1(v346)){
                assert("Tensor range check" && 0 <= v344 && v344 < 1);
                assert("Tensor range check" && 0 <= v346 && v346 < 4);
                int v348;
                v348 = 4 * v344;
                int v349;
                v349 = v348 + v346;
                float v350;
                v350 = v283[v349];
                float v351;
                v351 = v350 - v342;
                float v352;
                v352 = exp(v351);
                assert("Tensor range check" && 0 <= v344 && v344 < 1);
                assert("Tensor range check" && 0 <= v346 && v346 < 4);
                v343[v349] = v352;
                v346 += 1 ;
            }
            v344 += 1 ;
        }
        float v353;
        v353 = 0.0f;
        int v354;
        v354 = 0;
        while (while_method_3(v354)){
            int v356;
            v356 = 0;
            while (while_method_1(v356)){
                assert("Tensor range check" && 0 <= v354 && v354 < 1);
                assert("Tensor range check" && 0 <= v356 && v356 < 4);
                int v358;
                v358 = 4 * v354;
                int v359;
                v359 = v358 + v356;
                float v360;
                v360 = v343[v359];
                float v361;
                v361 = v353 + v360;
                v353 = v361;
                v356 += 1 ;
            }
            v354 += 1 ;
        }
        auto v362 = cooperative_groups::coalesced_threads();
        int v363;
        v363 = threadIdx.x;
        int v364;
        v364 = v363 / 16;
        auto v365 = cooperative_groups::labeled_partition(v362,v364);
        float v366;
        v366 = cooperative_groups::reduce(v365, v353, v42);
        float v367[4];
        int v368;
        v368 = 0;
        while (while_method_3(v368)){
            int v370;
            v370 = 0;
            while (while_method_1(v370)){
                assert("Tensor range check" && 0 <= v368 && v368 < 1);
                assert("Tensor range check" && 0 <= v370 && v370 < 4);
                int v372;
                v372 = 4 * v368;
                int v373;
                v373 = v372 + v370;
                float v374;
                v374 = v343[v373];
                float v375;
                v375 = v374 / v366;
                assert("Tensor range check" && 0 <= v368 && v368 < 1);
                assert("Tensor range check" && 0 <= v370 && v370 < 4);
                v367[v373] = v375;
                v370 += 1 ;
            }
            v368 += 1 ;
        }
        assert("Tensor range check" && 0 <= v279 && v279 < 8);
        int v376;
        v376 = 0;
        while (while_method_3(v376)){
            assert("Tensor range check" && 0 <= v376 && v376 < 1);
            int v378;
            v378 = 64 * v376;
            int v379;
            v379 = v378 + v282;
            assert("Tensor range check" && 0 <= v376 && v376 < 1);
            int v380;
            v380 = 4 * v376;
            int4* v381;
            v381 = reinterpret_cast<int4*>(v367 + v380);
            int4* v382;
            v382 = reinterpret_cast<int4*>(v4 + v379);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v381) % 16 == 0 && reinterpret_cast<unsigned long long>(v382) % 16 == 0);
            *v382 = *v381;
            v376 += 1 ;
        }
        v279 += 1 ;
    }
    __syncthreads();
    int v383;
    v383 = threadIdx.x;
    bool v384;
    v384 = 0 <= v383;
    bool v385;
    v385 = v384 == false;
    if (v385){
        assert("The index needs to be zero or positive." && v384);
    } else {
    }
    int v387;
    v387 = v383 % 16;
    int v388;
    v388 = v383 / 16;
    bool v389;
    v389 = v388 < 16;
    bool v390;
    v390 = v389 == false;
    if (v390){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v389);
    } else {
    }
    assert("Tensor range check" && 0 <= v388 && v388 < 16);
    assert("Tensor range check" && 0 <= v387 && v387 < 16);
    int v392;
    v392 = 4 * v387;
    int v393;
    v393 = 64 * v388;
    int v394;
    v394 = v393 + v392;
    assert("Tensor range check" && 0 <= v388 && v388 < 16);
    assert("Tensor range check" && 0 <= v387 && v387 < 16);
    int v395;
    v395 = 0;
    while (while_method_2(v395)){
        assert("Tensor range check" && 0 <= v395 && v395 < 8);
        int v397;
        v397 = 1024 * v395;
        int v398;
        v398 = v397 + v394;
        float v399[4];
        int v400[4];
        int v401;
        v401 = 0;
        while (while_method_3(v401)){
            assert("Tensor range check" && 0 <= v401 && v401 < 1);
            int v403;
            v403 = 4 * v401;
            assert("Tensor range check" && 0 <= v401 && v401 < 1);
            int v404;
            v404 = 64 * v401;
            int v405;
            v405 = v404 + v398;
            int4* v406;
            v406 = reinterpret_cast<int4*>(v1 + v405);
            int4* v407;
            v407 = reinterpret_cast<int4*>(v399 + v403);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v406) % 16 == 0 && reinterpret_cast<unsigned long long>(v407) % 16 == 0);
            *v407 = *v406;
            v401 += 1 ;
        }
        int v408;
        v408 = 0;
        while (while_method_3(v408)){
            int v410;
            v410 = 0;
            while (while_method_1(v410)){
                bool v412;
                v412 = 0 <= v410;
                bool v414;
                if (v412){
                    bool v413;
                    v413 = v410 < 4;
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
                v417 = 0 <= v387;
                bool v419;
                if (v417){
                    bool v418;
                    v418 = v387 < 16;
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
                v422 = v387 * 4;
                int v423;
                v423 = v410 + v422;
                bool v424;
                v424 = 0 <= v408;
                bool v426;
                if (v424){
                    bool v425;
                    v425 = v408 < 1;
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
                v429 = v408 * 64;
                int v430;
                v430 = v423 + v429;
                assert("Tensor range check" && 0 <= v408 && v408 < 1);
                assert("Tensor range check" && 0 <= v410 && v410 < 4);
                int v431;
                v431 = 4 * v408;
                int v432;
                v432 = v431 + v410;
                v400[v432] = v430;
                v410 += 1 ;
            }
            v408 += 1 ;
        }
        bool v433;
        v433 = 0 <= v388;
        bool v434;
        v434 = v433 && v389;
        bool v435;
        v435 = v434 == false;
        if (v435){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v434);
        } else {
        }
        bool v437;
        v437 = 0 <= v395;
        bool v439;
        if (v437){
            bool v438;
            v438 = v395 < 8;
            v439 = v438;
        } else {
            v439 = false;
        }
        bool v440;
        v440 = v439 == false;
        if (v440){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v439);
        } else {
        }
        int v442;
        v442 = v395 * 16;
        int v443;
        v443 = v442 + v388;
        float v444[4];
        int v445;
        v445 = 0;
        while (while_method_3(v445)){
            int v447;
            v447 = 0;
            while (while_method_1(v447)){
                assert("Tensor range check" && 0 <= v445 && v445 < 1);
                assert("Tensor range check" && 0 <= v447 && v447 < 4);
                int v449;
                v449 = 4 * v445;
                int v450;
                v450 = v449 + v447;
                float v451;
                v451 = v399[v450];
                float v452;
                v452 = v451 * v451;
                assert("Tensor range check" && 0 <= v445 && v445 < 1);
                assert("Tensor range check" && 0 <= v447 && v447 < 4);
                v444[v450] = v452;
                v447 += 1 ;
            }
            v445 += 1 ;
        }
        float v453;
        v453 = 0.0f;
        int v454;
        v454 = 0;
        while (while_method_3(v454)){
            int v456;
            v456 = 0;
            while (while_method_1(v456)){
                assert("Tensor range check" && 0 <= v454 && v454 < 1);
                assert("Tensor range check" && 0 <= v456 && v456 < 4);
                int v458;
                v458 = 4 * v454;
                int v459;
                v459 = v458 + v456;
                float v460;
                v460 = v444[v459];
                float v461;
                v461 = v453 + v460;
                v453 = v461;
                v456 += 1 ;
            }
            v454 += 1 ;
        }
        auto v462 = cooperative_groups::coalesced_threads();
        int v463;
        v463 = threadIdx.x;
        int v464;
        v464 = v463 / 16;
        auto v465 = cooperative_groups::labeled_partition(v462,v464);
        float v466;
        v466 = cooperative_groups::reduce(v465, v453, v42);
        float v467[4];
        int v468;
        v468 = 0;
        while (while_method_3(v468)){
            int v470;
            v470 = 0;
            while (while_method_1(v470)){
                assert("Tensor range check" && 0 <= v468 && v468 < 1);
                assert("Tensor range check" && 0 <= v470 && v470 < 4);
                int v472;
                v472 = 4 * v468;
                int v473;
                v473 = v472 + v470;
                float v474;
                v474 = v399[v473];
                bool v475;
                v475 = v466 == 0.0f;
                bool v476;
                v476 = v475 != true;
                float v478;
                if (v476){
                    float v477;
                    v477 = v474 / v466;
                    v478 = v477;
                } else {
                    v478 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v468 && v468 < 1);
                assert("Tensor range check" && 0 <= v470 && v470 < 4);
                v467[v473] = v478;
                v470 += 1 ;
            }
            v468 += 1 ;
        }
        assert("Tensor range check" && 0 <= v395 && v395 < 8);
        int v479;
        v479 = 0;
        while (while_method_3(v479)){
            assert("Tensor range check" && 0 <= v479 && v479 < 1);
            int v481;
            v481 = 64 * v479;
            int v482;
            v482 = v481 + v398;
            assert("Tensor range check" && 0 <= v479 && v479 < 1);
            int v483;
            v483 = 4 * v479;
            int4* v484;
            v484 = reinterpret_cast<int4*>(v467 + v483);
            int4* v485;
            v485 = reinterpret_cast<int4*>(v8 + v482);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v484) % 16 == 0 && reinterpret_cast<unsigned long long>(v485) % 16 == 0);
            *v485 = *v484;
            v479 += 1 ;
        }
        v395 += 1 ;
    }
    __syncthreads();
    int v486;
    v486 = threadIdx.x;
    bool v487;
    v487 = 0 <= v486;
    bool v488;
    v488 = v487 == false;
    if (v488){
        assert("The index needs to be zero or positive." && v487);
    } else {
    }
    int v490;
    v490 = v486 % 16;
    int v491;
    v491 = v486 / 16;
    bool v492;
    v492 = v491 < 16;
    bool v493;
    v493 = v492 == false;
    if (v493){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v492);
    } else {
    }
    assert("Tensor range check" && 0 <= v491 && v491 < 16);
    assert("Tensor range check" && 0 <= v490 && v490 < 16);
    int v495;
    v495 = 4 * v490;
    int v496;
    v496 = 64 * v491;
    int v497;
    v497 = v496 + v495;
    assert("Tensor range check" && 0 <= v491 && v491 < 16);
    int v498;
    v498 = 0;
    while (while_method_2(v498)){
        assert("Tensor range check" && 0 <= v498 && v498 < 8);
        int v500;
        v500 = 1024 * v498;
        int v501;
        v501 = v500 + v497;
        float v502[4];
        int v503[4];
        int v504;
        v504 = 0;
        while (while_method_3(v504)){
            assert("Tensor range check" && 0 <= v504 && v504 < 1);
            int v506;
            v506 = 4 * v504;
            assert("Tensor range check" && 0 <= v504 && v504 < 1);
            int v507;
            v507 = 64 * v504;
            int v508;
            v508 = v507 + v501;
            int4* v509;
            v509 = reinterpret_cast<int4*>(v1 + v508);
            int4* v510;
            v510 = reinterpret_cast<int4*>(v502 + v506);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v509) % 16 == 0 && reinterpret_cast<unsigned long long>(v510) % 16 == 0);
            *v510 = *v509;
            v504 += 1 ;
        }
        int v511;
        v511 = 0;
        while (while_method_3(v511)){
            int v513;
            v513 = 0;
            while (while_method_1(v513)){
                bool v515;
                v515 = 0 <= v513;
                bool v517;
                if (v515){
                    bool v516;
                    v516 = v513 < 4;
                    v517 = v516;
                } else {
                    v517 = false;
                }
                bool v518;
                v518 = v517 == false;
                if (v518){
                    assert("The indices should be inside the range of the dimension." && v517);
                } else {
                }
                bool v520;
                v520 = 0 <= v490;
                bool v522;
                if (v520){
                    bool v521;
                    v521 = v490 < 16;
                    v522 = v521;
                } else {
                    v522 = false;
                }
                bool v523;
                v523 = v522 == false;
                if (v523){
                    assert("The indices should be inside the range of the dimension." && v522);
                } else {
                }
                int v525;
                v525 = v490 * 4;
                int v526;
                v526 = v513 + v525;
                bool v527;
                v527 = 0 <= v511;
                bool v529;
                if (v527){
                    bool v528;
                    v528 = v511 < 1;
                    v529 = v528;
                } else {
                    v529 = false;
                }
                bool v530;
                v530 = v529 == false;
                if (v530){
                    assert("The indices should be inside the range of the dimension." && v529);
                } else {
                }
                int v532;
                v532 = v511 * 64;
                int v533;
                v533 = v526 + v532;
                assert("Tensor range check" && 0 <= v511 && v511 < 1);
                assert("Tensor range check" && 0 <= v513 && v513 < 4);
                int v534;
                v534 = 4 * v511;
                int v535;
                v535 = v534 + v513;
                v503[v535] = v533;
                v513 += 1 ;
            }
            v511 += 1 ;
        }
        bool v536;
        v536 = 0 <= v491;
        bool v537;
        v537 = v536 && v492;
        bool v538;
        v538 = v537 == false;
        if (v538){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v537);
        } else {
        }
        bool v540;
        v540 = 0 <= v498;
        bool v542;
        if (v540){
            bool v541;
            v541 = v498 < 8;
            v542 = v541;
        } else {
            v542 = false;
        }
        bool v543;
        v543 = v542 == false;
        if (v543){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v542);
        } else {
        }
        int v545;
        v545 = v498 * 16;
        int v546;
        v546 = v545 + v491;
        float v547; int v548;
        Tuple1 tmp24 = Tuple1{-1.0f / 0.0f, 0};
        v547 = tmp24.v0; v548 = tmp24.v1;
        int v549;
        v549 = 0;
        while (while_method_3(v549)){
            int v551;
            v551 = 0;
            while (while_method_1(v551)){
                assert("Tensor range check" && 0 <= v549 && v549 < 1);
                assert("Tensor range check" && 0 <= v551 && v551 < 4);
                int v553;
                v553 = 4 * v549;
                int v554;
                v554 = v553 + v551;
                float v555;
                v555 = v502[v554];
                int v556;
                v556 = v503[v554];
                bool v557;
                v557 = v547 > v555;
                float v558; int v559;
                if (v557){
                    v558 = v547; v559 = v548;
                } else {
                    v558 = v555; v559 = v556;
                }
                v547 = v558;
                v548 = v559;
                v551 += 1 ;
            }
            v549 += 1 ;
        }
        auto v560 = cooperative_groups::coalesced_threads();
        int v561;
        v561 = threadIdx.x;
        int v562;
        v562 = v561 / 16;
        auto v563 = cooperative_groups::labeled_partition(v560,v562);
        Closure1 v564{};
        float v565; int v566;
        Tuple1 tmp25 = cooperative_groups::reduce(v563, Tuple1{v547, v548}, v564);
        v565 = tmp25.v0; v566 = tmp25.v1;
        assert("Tensor range check" && 0 <= v498 && v498 < 8);
        int v567;
        v567 = 16 * v498;
        int v568;
        v568 = v567 + v491;
        v9[v568] = v566;
        v498 += 1 ;
    }
    __syncthreads();
    int v569;
    v569 = threadIdx.x;
    bool v570;
    v570 = 0 <= v569;
    bool v571;
    v571 = v570 == false;
    if (v571){
        assert("The index needs to be zero or positive." && v570);
    } else {
    }
    int v573;
    v573 = v569 % 16;
    int v574;
    v574 = v569 / 16;
    bool v575;
    v575 = v574 < 16;
    bool v576;
    v576 = v575 == false;
    if (v576){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v575);
    } else {
    }
    assert("Tensor range check" && 0 <= v574 && v574 < 16);
    assert("Tensor range check" && 0 <= v573 && v573 < 16);
    int v578;
    v578 = 4 * v573;
    int v579;
    v579 = 64 * v574;
    int v580;
    v580 = v579 + v578;
    assert("Tensor range check" && 0 <= v574 && v574 < 16);
    assert("Tensor range check" && 0 <= v573 && v573 < 16);
    int v581;
    v581 = 0;
    while (while_method_2(v581)){
        assert("Tensor range check" && 0 <= v581 && v581 < 8);
        int v583;
        v583 = 1024 * v581;
        int v584;
        v584 = v583 + v580;
        float v585[4];
        int v586[4];
        int v587;
        v587 = 0;
        while (while_method_3(v587)){
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v589;
            v589 = 4 * v587;
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v590;
            v590 = 64 * v587;
            int v591;
            v591 = v590 + v584;
            int4* v592;
            v592 = reinterpret_cast<int4*>(v1 + v591);
            int4* v593;
            v593 = reinterpret_cast<int4*>(v585 + v589);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v592) % 16 == 0 && reinterpret_cast<unsigned long long>(v593) % 16 == 0);
            *v593 = *v592;
            v587 += 1 ;
        }
        int v594;
        v594 = 0;
        while (while_method_3(v594)){
            int v596;
            v596 = 0;
            while (while_method_1(v596)){
                bool v598;
                v598 = 0 <= v596;
                bool v600;
                if (v598){
                    bool v599;
                    v599 = v596 < 4;
                    v600 = v599;
                } else {
                    v600 = false;
                }
                bool v601;
                v601 = v600 == false;
                if (v601){
                    assert("The indices should be inside the range of the dimension." && v600);
                } else {
                }
                bool v603;
                v603 = 0 <= v573;
                bool v605;
                if (v603){
                    bool v604;
                    v604 = v573 < 16;
                    v605 = v604;
                } else {
                    v605 = false;
                }
                bool v606;
                v606 = v605 == false;
                if (v606){
                    assert("The indices should be inside the range of the dimension." && v605);
                } else {
                }
                int v608;
                v608 = v573 * 4;
                int v609;
                v609 = v596 + v608;
                bool v610;
                v610 = 0 <= v594;
                bool v612;
                if (v610){
                    bool v611;
                    v611 = v594 < 1;
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
                int v615;
                v615 = v594 * 64;
                int v616;
                v616 = v609 + v615;
                assert("Tensor range check" && 0 <= v594 && v594 < 1);
                assert("Tensor range check" && 0 <= v596 && v596 < 4);
                int v617;
                v617 = 4 * v594;
                int v618;
                v618 = v617 + v596;
                v586[v618] = v616;
                v596 += 1 ;
            }
            v594 += 1 ;
        }
        bool v619;
        v619 = 0 <= v574;
        bool v620;
        v620 = v619 && v575;
        bool v621;
        v621 = v620 == false;
        if (v621){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v620);
        } else {
        }
        bool v623;
        v623 = 0 <= v581;
        bool v625;
        if (v623){
            bool v624;
            v624 = v581 < 8;
            v625 = v624;
        } else {
            v625 = false;
        }
        bool v626;
        v626 = v625 == false;
        if (v626){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v625);
        } else {
        }
        int v628;
        v628 = v581 * 16;
        int v629;
        v629 = v628 + v574;
        float v630;
        v630 = 0.0f;
        int v631;
        v631 = 0;
        while (while_method_3(v631)){
            int v633;
            v633 = 0;
            while (while_method_1(v633)){
                assert("Tensor range check" && 0 <= v631 && v631 < 1);
                assert("Tensor range check" && 0 <= v633 && v633 < 4);
                int v635;
                v635 = 4 * v631;
                int v636;
                v636 = v635 + v633;
                float v637;
                v637 = v585[v636];
                float v638;
                v638 = v630 + v637;
                v630 = v638;
                v633 += 1 ;
            }
            v631 += 1 ;
        }
        auto v639 = cooperative_groups::coalesced_threads();
        int v640;
        v640 = threadIdx.x;
        int v641;
        v641 = v640 / 16;
        auto v642 = cooperative_groups::labeled_partition(v639,v641);
        float v643;
        v643 = cooperative_groups::reduce(v642, v630, v42);
        float v644;
        v644 = v643 / 64.0f;
        float v645[4];
        int v646;
        v646 = 0;
        while (while_method_3(v646)){
            int v648;
            v648 = 0;
            while (while_method_1(v648)){
                assert("Tensor range check" && 0 <= v646 && v646 < 1);
                assert("Tensor range check" && 0 <= v648 && v648 < 4);
                int v650;
                v650 = 4 * v646;
                int v651;
                v651 = v650 + v648;
                float v652;
                v652 = v585[v651];
                float v653;
                v653 = v652 - v644;
                float v654;
                v654 = exp(v653);
                assert("Tensor range check" && 0 <= v646 && v646 < 1);
                assert("Tensor range check" && 0 <= v648 && v648 < 4);
                v645[v651] = v654;
                v648 += 1 ;
            }
            v646 += 1 ;
        }
        float v655;
        v655 = 0.0f;
        int v656;
        v656 = 0;
        while (while_method_3(v656)){
            int v658;
            v658 = 0;
            while (while_method_1(v658)){
                assert("Tensor range check" && 0 <= v656 && v656 < 1);
                assert("Tensor range check" && 0 <= v658 && v658 < 4);
                int v660;
                v660 = 4 * v656;
                int v661;
                v661 = v660 + v658;
                float v662;
                v662 = v645[v661];
                float v663;
                v663 = v655 + v662;
                v655 = v663;
                v658 += 1 ;
            }
            v656 += 1 ;
        }
        auto v664 = cooperative_groups::coalesced_threads();
        int v665;
        v665 = threadIdx.x;
        int v666;
        v666 = v665 / 16;
        auto v667 = cooperative_groups::labeled_partition(v664,v666);
        float v668;
        v668 = cooperative_groups::reduce(v667, v655, v42);
        float v669[4];
        int v670;
        v670 = 0;
        while (while_method_3(v670)){
            int v672;
            v672 = 0;
            while (while_method_1(v672)){
                assert("Tensor range check" && 0 <= v670 && v670 < 1);
                assert("Tensor range check" && 0 <= v672 && v672 < 4);
                int v674;
                v674 = 4 * v670;
                int v675;
                v675 = v674 + v672;
                float v676;
                v676 = v645[v675];
                float v677;
                v677 = v676 / v668;
                assert("Tensor range check" && 0 <= v670 && v670 < 1);
                assert("Tensor range check" && 0 <= v672 && v672 < 4);
                v669[v675] = v677;
                v672 += 1 ;
            }
            v670 += 1 ;
        }
        float v678[4];
        float v679;
        v679 = 0.0f;
        int v680;
        v680 = 0;
        while (while_method_3(v680)){
            assert("Tensor range check" && 0 <= v680 && v680 < 1);
            int v682;
            v682 = 4 * v680;
            assert("Tensor range check" && 0 <= v680 && v680 < 1);
            int v683; float v684;
            Tuple0 tmp26 = Tuple0{0, 0.0f};
            v683 = tmp26.v0; v684 = tmp26.v1;
            while (while_method_1(v683)){
                assert("Tensor range check" && 0 <= v683 && v683 < 4);
                int v686;
                v686 = v683 + v682;
                float v687;
                v687 = v669[v686];
                float v688;
                v688 = v684 + v687;
                v684 = v688;
                v683 += 1 ;
            }
            auto v689 = cooperative_groups::coalesced_threads();
            int v690;
            v690 = threadIdx.x;
            int v691;
            v691 = v690 / 16;
            auto v692 = cooperative_groups::labeled_partition(v689,v691);
            Closure2 v693{};
            float v694;
            v694 = cooperative_groups::inclusive_scan(v692, v684, v693);
            float v695;
            v695 = v692.shfl_up(v694,1);
            bool v696;
            v696 = v692.thread_rank() == 0;
            float v697;
            if (v696){
                v697 = 0.0f;
            } else {
                v697 = v695;
            }
            float v698;
            v698 = v692.shfl(v694,v692.num_threads()-1);
            float v699;
            v699 = v679 + v697;
            int v700; float v701;
            Tuple0 tmp27 = Tuple0{0, v699};
            v700 = tmp27.v0; v701 = tmp27.v1;
            while (while_method_1(v700)){
                assert("Tensor range check" && 0 <= v700 && v700 < 4);
                int v703;
                v703 = v700 + v682;
                float v704;
                v704 = v669[v703];
                float v705;
                v705 = v701 + v704;
                assert("Tensor range check" && 0 <= v700 && v700 < 4);
                v678[v703] = v705;
                v701 = v705;
                v700 += 1 ;
            }
            float v706;
            v706 = v679 + v698;
            v679 = v706;
            v680 += 1 ;
        }
        assert("Tensor range check" && 0 <= v581 && v581 < 8);
        int v707;
        v707 = 0;
        while (while_method_3(v707)){
            assert("Tensor range check" && 0 <= v707 && v707 < 1);
            int v709;
            v709 = 64 * v707;
            int v710;
            v710 = v709 + v584;
            assert("Tensor range check" && 0 <= v707 && v707 < 1);
            int v711;
            v711 = 4 * v707;
            int4* v712;
            v712 = reinterpret_cast<int4*>(v669 + v711);
            int4* v713;
            v713 = reinterpret_cast<int4*>(v6 + v710);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v712) % 16 == 0 && reinterpret_cast<unsigned long long>(v713) % 16 == 0);
            *v713 = *v712;
            int4* v714;
            v714 = reinterpret_cast<int4*>(v678 + v711);
            int4* v715;
            v715 = reinterpret_cast<int4*>(v7 + v710);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v714) % 16 == 0 && reinterpret_cast<unsigned long long>(v715) % 16 == 0);
            *v715 = *v714;
            v707 += 1 ;
        }
        v581 += 1 ;
    }
    __syncthreads();
    int v716;
    v716 = threadIdx.x;
    bool v717;
    v717 = 0 <= v716;
    bool v718;
    v718 = v717 == false;
    if (v718){
        assert("The index needs to be zero or positive." && v717);
    } else {
    }
    int v720;
    v720 = v716 % 16;
    int v721;
    v721 = v716 / 16;
    bool v722;
    v722 = v721 < 16;
    bool v723;
    v723 = v722 == false;
    if (v723){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v722);
    } else {
    }
    assert("Tensor range check" && 0 <= v721 && v721 < 16);
    assert("Tensor range check" && 0 <= v720 && v720 < 16);
    int v725;
    v725 = 4 * v720;
    int v726;
    v726 = 64 * v721;
    int v727;
    v727 = v726 + v725;
    assert("Tensor range check" && 0 <= v721 && v721 < 16);
    assert("Tensor range check" && 0 <= v720 && v720 < 16);
    int v728;
    v728 = 0;
    while (while_method_2(v728)){
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v730;
        v730 = 1024 * v728;
        int v731;
        v731 = v730 + v727;
        int v732[4];
        int v733[4];
        int v734;
        v734 = 0;
        while (while_method_3(v734)){
            assert("Tensor range check" && 0 <= v734 && v734 < 1);
            int v736;
            v736 = 4 * v734;
            assert("Tensor range check" && 0 <= v734 && v734 < 1);
            int v737;
            v737 = 64 * v734;
            int v738;
            v738 = v737 + v731;
            int4* v739;
            v739 = reinterpret_cast<int4*>(v0 + v738);
            int4* v740;
            v740 = reinterpret_cast<int4*>(v732 + v736);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v739) % 16 == 0 && reinterpret_cast<unsigned long long>(v740) % 16 == 0);
            *v740 = *v739;
            v734 += 1 ;
        }
        int v741;
        v741 = 0;
        while (while_method_3(v741)){
            int v743;
            v743 = 0;
            while (while_method_1(v743)){
                bool v745;
                v745 = 0 <= v743;
                bool v747;
                if (v745){
                    bool v746;
                    v746 = v743 < 4;
                    v747 = v746;
                } else {
                    v747 = false;
                }
                bool v748;
                v748 = v747 == false;
                if (v748){
                    assert("The indices should be inside the range of the dimension." && v747);
                } else {
                }
                bool v750;
                v750 = 0 <= v720;
                bool v752;
                if (v750){
                    bool v751;
                    v751 = v720 < 16;
                    v752 = v751;
                } else {
                    v752 = false;
                }
                bool v753;
                v753 = v752 == false;
                if (v753){
                    assert("The indices should be inside the range of the dimension." && v752);
                } else {
                }
                int v755;
                v755 = v720 * 4;
                int v756;
                v756 = v743 + v755;
                bool v757;
                v757 = 0 <= v741;
                bool v759;
                if (v757){
                    bool v758;
                    v758 = v741 < 1;
                    v759 = v758;
                } else {
                    v759 = false;
                }
                bool v760;
                v760 = v759 == false;
                if (v760){
                    assert("The indices should be inside the range of the dimension." && v759);
                } else {
                }
                int v762;
                v762 = v741 * 64;
                int v763;
                v763 = v756 + v762;
                assert("Tensor range check" && 0 <= v741 && v741 < 1);
                assert("Tensor range check" && 0 <= v743 && v743 < 4);
                int v764;
                v764 = 4 * v741;
                int v765;
                v765 = v764 + v743;
                v733[v765] = v763;
                v743 += 1 ;
            }
            v741 += 1 ;
        }
        bool v766;
        v766 = 0 <= v721;
        bool v767;
        v767 = v766 && v722;
        bool v768;
        v768 = v767 == false;
        if (v768){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v767);
        } else {
        }
        bool v770;
        v770 = 0 <= v728;
        bool v772;
        if (v770){
            bool v771;
            v771 = v728 < 8;
            v772 = v771;
        } else {
            v772 = false;
        }
        bool v773;
        v773 = v772 == false;
        if (v773){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v772);
        } else {
        }
        int v775;
        v775 = v728 * 16;
        int v776;
        v776 = v775 + v721;
        int v777[4];
        int v778;
        v778 = 0;
        int v779;
        v779 = 0;
        while (while_method_3(v779)){
            assert("Tensor range check" && 0 <= v779 && v779 < 1);
            int v781;
            v781 = 4 * v779;
            assert("Tensor range check" && 0 <= v779 && v779 < 1);
            int v782; int v783;
            Tuple2 tmp28 = Tuple2{0, 0};
            v782 = tmp28.v0; v783 = tmp28.v1;
            while (while_method_1(v782)){
                assert("Tensor range check" && 0 <= v782 && v782 < 4);
                int v785;
                v785 = v782 + v781;
                int v786;
                v786 = v732[v785];
                int v787;
                v787 = v783 + v786;
                v783 = v787;
                v782 += 1 ;
            }
            auto v788 = cooperative_groups::coalesced_threads();
            int v789;
            v789 = threadIdx.x;
            int v790;
            v790 = v789 / 16;
            auto v791 = cooperative_groups::labeled_partition(v788,v790);
            Closure3 v792{};
            int v793;
            v793 = cooperative_groups::inclusive_scan(v791, v783, v792);
            int v794;
            v794 = v791.shfl_up(v793,1);
            bool v795;
            v795 = v791.thread_rank() == 0;
            int v796;
            if (v795){
                v796 = 0;
            } else {
                v796 = v794;
            }
            int v797;
            v797 = v791.shfl(v793,v791.num_threads()-1);
            int v798;
            v798 = v778 + v796;
            int v799; int v800;
            Tuple2 tmp29 = Tuple2{0, v798};
            v799 = tmp29.v0; v800 = tmp29.v1;
            while (while_method_1(v799)){
                assert("Tensor range check" && 0 <= v799 && v799 < 4);
                int v802;
                v802 = v799 + v781;
                int v803;
                v803 = v732[v802];
                assert("Tensor range check" && 0 <= v799 && v799 < 4);
                v777[v802] = v800;
                int v804;
                v804 = v800 + v803;
                v800 = v804;
                v799 += 1 ;
            }
            int v805;
            v805 = v778 + v797;
            v778 = v805;
            v779 += 1 ;
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v806;
        v806 = 0;
        while (while_method_3(v806)){
            assert("Tensor range check" && 0 <= v806 && v806 < 1);
            int v808;
            v808 = 64 * v806;
            int v809;
            v809 = v808 + v731;
            assert("Tensor range check" && 0 <= v806 && v806 < 1);
            int v810;
            v810 = 4 * v806;
            int4* v811;
            v811 = reinterpret_cast<int4*>(v777 + v810);
            int4* v812;
            v812 = reinterpret_cast<int4*>(v13 + v809);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v811) % 16 == 0 && reinterpret_cast<unsigned long long>(v812) % 16 == 0);
            *v812 = *v811;
            v806 += 1 ;
        }
        v728 += 1 ;
    }
    __syncthreads();
    int v813;
    v813 = threadIdx.x;
    bool v814;
    v814 = 0 <= v813;
    bool v815;
    v815 = v814 == false;
    if (v815){
        assert("The index needs to be zero or positive." && v814);
    } else {
    }
    int v817;
    v817 = v813 % 16;
    int v818;
    v818 = v813 / 16;
    bool v819;
    v819 = v818 < 16;
    bool v820;
    v820 = v819 == false;
    if (v820){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v819);
    } else {
    }
    assert("Tensor range check" && 0 <= v818 && v818 < 16);
    assert("Tensor range check" && 0 <= v817 && v817 < 16);
    int v822;
    v822 = 4 * v817;
    int v823;
    v823 = 64 * v818;
    int v824;
    v824 = v823 + v822;
    assert("Tensor range check" && 0 <= v818 && v818 < 16);
    assert("Tensor range check" && 0 <= v817 && v817 < 16);
    int v825;
    v825 = 0;
    while (while_method_2(v825)){
        assert("Tensor range check" && 0 <= v825 && v825 < 8);
        int v827;
        v827 = 1024 * v825;
        int v828;
        v828 = v827 + v824;
        float v829[4];
        int v830[4];
        int v831;
        v831 = 0;
        while (while_method_3(v831)){
            assert("Tensor range check" && 0 <= v831 && v831 < 1);
            int v833;
            v833 = 4 * v831;
            assert("Tensor range check" && 0 <= v831 && v831 < 1);
            int v834;
            v834 = 64 * v831;
            int v835;
            v835 = v834 + v828;
            int4* v836;
            v836 = reinterpret_cast<int4*>(v1 + v835);
            int4* v837;
            v837 = reinterpret_cast<int4*>(v829 + v833);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v836) % 16 == 0 && reinterpret_cast<unsigned long long>(v837) % 16 == 0);
            *v837 = *v836;
            v831 += 1 ;
        }
        int v838;
        v838 = 0;
        while (while_method_3(v838)){
            int v840;
            v840 = 0;
            while (while_method_1(v840)){
                bool v842;
                v842 = 0 <= v840;
                bool v844;
                if (v842){
                    bool v843;
                    v843 = v840 < 4;
                    v844 = v843;
                } else {
                    v844 = false;
                }
                bool v845;
                v845 = v844 == false;
                if (v845){
                    assert("The indices should be inside the range of the dimension." && v844);
                } else {
                }
                bool v847;
                v847 = 0 <= v817;
                bool v849;
                if (v847){
                    bool v848;
                    v848 = v817 < 16;
                    v849 = v848;
                } else {
                    v849 = false;
                }
                bool v850;
                v850 = v849 == false;
                if (v850){
                    assert("The indices should be inside the range of the dimension." && v849);
                } else {
                }
                int v852;
                v852 = v817 * 4;
                int v853;
                v853 = v840 + v852;
                bool v854;
                v854 = 0 <= v838;
                bool v856;
                if (v854){
                    bool v855;
                    v855 = v838 < 1;
                    v856 = v855;
                } else {
                    v856 = false;
                }
                bool v857;
                v857 = v856 == false;
                if (v857){
                    assert("The indices should be inside the range of the dimension." && v856);
                } else {
                }
                int v859;
                v859 = v838 * 64;
                int v860;
                v860 = v853 + v859;
                assert("Tensor range check" && 0 <= v838 && v838 < 1);
                assert("Tensor range check" && 0 <= v840 && v840 < 4);
                int v861;
                v861 = 4 * v838;
                int v862;
                v862 = v861 + v840;
                v830[v862] = v860;
                v840 += 1 ;
            }
            v838 += 1 ;
        }
        bool v863;
        v863 = 0 <= v818;
        bool v864;
        v864 = v863 && v819;
        bool v865;
        v865 = v864 == false;
        if (v865){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v864);
        } else {
        }
        bool v867;
        v867 = 0 <= v825;
        bool v869;
        if (v867){
            bool v868;
            v868 = v825 < 8;
            v869 = v868;
        } else {
            v869 = false;
        }
        bool v870;
        v870 = v869 == false;
        if (v870){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v869);
        } else {
        }
        int v872;
        v872 = v825 * 16;
        int v873;
        v873 = v872 + v818;
        bool v874[4];
        int v875;
        v875 = 0;
        while (while_method_3(v875)){
            int v877;
            v877 = 0;
            while (while_method_1(v877)){
                assert("Tensor range check" && 0 <= v875 && v875 < 1);
                assert("Tensor range check" && 0 <= v877 && v877 < 4);
                int v879;
                v879 = 4 * v875;
                int v880;
                v880 = v879 + v877;
                float v881;
                v881 = v829[v880];
                int v882;
                v882 = v830[v880];
                bool v883;
                v883 = v882 < 4;
                assert("Tensor range check" && 0 <= v875 && v875 < 1);
                assert("Tensor range check" && 0 <= v877 && v877 < 4);
                v874[v880] = v883;
                v877 += 1 ;
            }
            v875 += 1 ;
        }
        int v884[4];
        int v885;
        v885 = 0;
        while (while_method_3(v885)){
            int v887;
            v887 = 0;
            while (while_method_1(v887)){
                assert("Tensor range check" && 0 <= v885 && v885 < 1);
                assert("Tensor range check" && 0 <= v887 && v887 < 4);
                int v889;
                v889 = 4 * v885;
                int v890;
                v890 = v889 + v887;
                bool v891;
                v891 = v874[v890];
                int v892;
                if (v891){
                    v892 = 1;
                } else {
                    v892 = 0;
                }
                assert("Tensor range check" && 0 <= v885 && v885 < 1);
                assert("Tensor range check" && 0 <= v887 && v887 < 4);
                v884[v890] = v892;
                v887 += 1 ;
            }
            v885 += 1 ;
        }
        int v893;
        v893 = 0;
        int v894;
        v894 = 0;
        while (while_method_3(v894)){
            int v896;
            v896 = 0;
            while (while_method_1(v896)){
                assert("Tensor range check" && 0 <= v894 && v894 < 1);
                assert("Tensor range check" && 0 <= v896 && v896 < 4);
                int v898;
                v898 = 4 * v894;
                int v899;
                v899 = v898 + v896;
                int v900;
                v900 = v884[v899];
                int v901;
                v901 = v893 + v900;
                v893 = v901;
                v896 += 1 ;
            }
            v894 += 1 ;
        }
        auto v902 = cooperative_groups::coalesced_threads();
        int v903;
        v903 = threadIdx.x;
        int v904;
        v904 = v903 / 16;
        auto v905 = cooperative_groups::labeled_partition(v902,v904);
        Closure4 v906{};
        int v907;
        v907 = cooperative_groups::reduce(v905, v893, v906);
        float v908[4];
        int v909;
        v909 = 0;
        while (while_method_3(v909)){
            int v911;
            v911 = 0;
            while (while_method_1(v911)){
                assert("Tensor range check" && 0 <= v909 && v909 < 1);
                assert("Tensor range check" && 0 <= v911 && v911 < 4);
                int v913;
                v913 = 4 * v909;
                int v914;
                v914 = v913 + v911;
                float v915;
                v915 = v829[v914];
                bool v916;
                v916 = v874[v914];
                float v917;
                if (v916){
                    v917 = v915;
                } else {
                    v917 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v909 && v909 < 1);
                assert("Tensor range check" && 0 <= v911 && v911 < 4);
                v908[v914] = v917;
                v911 += 1 ;
            }
            v909 += 1 ;
        }
        float v918;
        v918 = 0.0f;
        int v919;
        v919 = 0;
        while (while_method_3(v919)){
            int v921;
            v921 = 0;
            while (while_method_1(v921)){
                assert("Tensor range check" && 0 <= v919 && v919 < 1);
                assert("Tensor range check" && 0 <= v921 && v921 < 4);
                int v923;
                v923 = 4 * v919;
                int v924;
                v924 = v923 + v921;
                float v925;
                v925 = v908[v924];
                float v926;
                v926 = v918 + v925;
                v918 = v926;
                v921 += 1 ;
            }
            v919 += 1 ;
        }
        auto v927 = cooperative_groups::coalesced_threads();
        int v928;
        v928 = threadIdx.x;
        int v929;
        v929 = v928 / 16;
        auto v930 = cooperative_groups::labeled_partition(v927,v929);
        float v931;
        v931 = cooperative_groups::reduce(v930, v918, v42);
        float v932;
        v932 = (float)v907;
        float v933;
        v933 = v931 / v932;
        float v934[4];
        int v935;
        v935 = 0;
        while (while_method_3(v935)){
            int v937;
            v937 = 0;
            while (while_method_1(v937)){
                assert("Tensor range check" && 0 <= v935 && v935 < 1);
                assert("Tensor range check" && 0 <= v937 && v937 < 4);
                int v939;
                v939 = 4 * v935;
                int v940;
                v940 = v939 + v937;
                float v941;
                v941 = v829[v940];
                bool v942;
                v942 = v874[v940];
                float v943;
                if (v942){
                    v943 = v941;
                } else {
                    v943 = -1.0f / 0.0f;
                }
                float v944;
                v944 = v943 - v933;
                float v945;
                v945 = exp(v944);
                assert("Tensor range check" && 0 <= v935 && v935 < 1);
                assert("Tensor range check" && 0 <= v937 && v937 < 4);
                v934[v940] = v945;
                v937 += 1 ;
            }
            v935 += 1 ;
        }
        float v946;
        v946 = 0.0f;
        int v947;
        v947 = 0;
        while (while_method_3(v947)){
            int v949;
            v949 = 0;
            while (while_method_1(v949)){
                assert("Tensor range check" && 0 <= v947 && v947 < 1);
                assert("Tensor range check" && 0 <= v949 && v949 < 4);
                int v951;
                v951 = 4 * v947;
                int v952;
                v952 = v951 + v949;
                float v953;
                v953 = v934[v952];
                float v954;
                v954 = v946 + v953;
                v946 = v954;
                v949 += 1 ;
            }
            v947 += 1 ;
        }
        auto v955 = cooperative_groups::coalesced_threads();
        int v956;
        v956 = threadIdx.x;
        int v957;
        v957 = v956 / 16;
        auto v958 = cooperative_groups::labeled_partition(v955,v957);
        float v959;
        v959 = cooperative_groups::reduce(v958, v946, v42);
        float v960[4];
        int v961;
        v961 = 0;
        while (while_method_3(v961)){
            int v963;
            v963 = 0;
            while (while_method_1(v963)){
                assert("Tensor range check" && 0 <= v961 && v961 < 1);
                assert("Tensor range check" && 0 <= v963 && v963 < 4);
                int v965;
                v965 = 4 * v961;
                int v966;
                v966 = v965 + v963;
                float v967;
                v967 = v934[v966];
                float v968;
                v968 = v967 / v959;
                assert("Tensor range check" && 0 <= v961 && v961 < 1);
                assert("Tensor range check" && 0 <= v963 && v963 < 4);
                v960[v966] = v968;
                v963 += 1 ;
            }
            v961 += 1 ;
        }
        assert("Tensor range check" && 0 <= v825 && v825 < 8);
        int v969;
        v969 = 0;
        while (while_method_3(v969)){
            assert("Tensor range check" && 0 <= v969 && v969 < 1);
            int v971;
            v971 = 64 * v969;
            int v972;
            v972 = v971 + v828;
            assert("Tensor range check" && 0 <= v969 && v969 < 1);
            int v973;
            v973 = 4 * v969;
            int4* v974;
            v974 = reinterpret_cast<int4*>(v960 + v973);
            int4* v975;
            v975 = reinterpret_cast<int4*>(v5 + v972);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v974) % 16 == 0 && reinterpret_cast<unsigned long long>(v975) % 16 == 0);
            *v975 = *v974;
            v969 += 1 ;
        }
        v825 += 1 ;
    }
    __syncthreads();
    int v976;
    v976 = threadIdx.x;
    int v977;
    v977 = blockIdx.x;
    int v978;
    v978 = v977 * 256;
    int v979;
    v979 = v976 + v978;
    unsigned long long v980;
    v980 = (unsigned long long)v979;
    curandStatePhilox4_32_10_t v981;
    curand_init(12344321ull,v980,0ull,&v981);
    int v982;
    v982 = threadIdx.x;
    bool v983;
    v983 = 0 <= v982;
    bool v984;
    v984 = v983 == false;
    if (v984){
        assert("The index needs to be zero or positive." && v983);
    } else {
    }
    int v986;
    v986 = v982 % 16;
    int v987;
    v987 = v982 / 16;
    bool v988;
    v988 = v987 < 16;
    bool v989;
    v989 = v988 == false;
    if (v989){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v988);
    } else {
    }
    assert("Tensor range check" && 0 <= v987 && v987 < 16);
    assert("Tensor range check" && 0 <= v986 && v986 < 16);
    int v991;
    v991 = 4 * v986;
    int v992;
    v992 = 64 * v987;
    int v993;
    v993 = v992 + v991;
    assert("Tensor range check" && 0 <= v987 && v987 < 16);
    assert("Tensor range check" && 0 <= v986 && v986 < 16);
    assert("Tensor range check" && 0 <= v987 && v987 < 16);
    int v994;
    v994 = 0;
    while (while_method_2(v994)){
        assert("Tensor range check" && 0 <= v994 && v994 < 8);
        int v996;
        v996 = 1024 * v994;
        int v997;
        v997 = v996 + v993;
        float v998[4];
        int v999[4];
        int v1000;
        v1000 = 0;
        while (while_method_3(v1000)){
            assert("Tensor range check" && 0 <= v1000 && v1000 < 1);
            int v1002;
            v1002 = 4 * v1000;
            assert("Tensor range check" && 0 <= v1000 && v1000 < 1);
            int v1003;
            v1003 = 64 * v1000;
            int v1004;
            v1004 = v1003 + v997;
            int4* v1005;
            v1005 = reinterpret_cast<int4*>(v1 + v1004);
            int4* v1006;
            v1006 = reinterpret_cast<int4*>(v998 + v1002);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1005) % 16 == 0 && reinterpret_cast<unsigned long long>(v1006) % 16 == 0);
            *v1006 = *v1005;
            v1000 += 1 ;
        }
        int v1007;
        v1007 = 0;
        while (while_method_3(v1007)){
            int v1009;
            v1009 = 0;
            while (while_method_1(v1009)){
                bool v1011;
                v1011 = 0 <= v1009;
                bool v1013;
                if (v1011){
                    bool v1012;
                    v1012 = v1009 < 4;
                    v1013 = v1012;
                } else {
                    v1013 = false;
                }
                bool v1014;
                v1014 = v1013 == false;
                if (v1014){
                    assert("The indices should be inside the range of the dimension." && v1013);
                } else {
                }
                bool v1016;
                v1016 = 0 <= v986;
                bool v1018;
                if (v1016){
                    bool v1017;
                    v1017 = v986 < 16;
                    v1018 = v1017;
                } else {
                    v1018 = false;
                }
                bool v1019;
                v1019 = v1018 == false;
                if (v1019){
                    assert("The indices should be inside the range of the dimension." && v1018);
                } else {
                }
                int v1021;
                v1021 = v986 * 4;
                int v1022;
                v1022 = v1009 + v1021;
                bool v1023;
                v1023 = 0 <= v1007;
                bool v1025;
                if (v1023){
                    bool v1024;
                    v1024 = v1007 < 1;
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
                int v1028;
                v1028 = v1007 * 64;
                int v1029;
                v1029 = v1022 + v1028;
                assert("Tensor range check" && 0 <= v1007 && v1007 < 1);
                assert("Tensor range check" && 0 <= v1009 && v1009 < 4);
                int v1030;
                v1030 = 4 * v1007;
                int v1031;
                v1031 = v1030 + v1009;
                v999[v1031] = v1029;
                v1009 += 1 ;
            }
            v1007 += 1 ;
        }
        bool v1032;
        v1032 = 0 <= v987;
        bool v1033;
        v1033 = v1032 && v988;
        bool v1034;
        v1034 = v1033 == false;
        if (v1034){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1033);
        } else {
        }
        bool v1036;
        v1036 = 0 <= v994;
        bool v1038;
        if (v1036){
            bool v1037;
            v1037 = v994 < 8;
            v1038 = v1037;
        } else {
            v1038 = false;
        }
        bool v1039;
        v1039 = v1038 == false;
        if (v1039){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1038);
        } else {
        }
        int v1041;
        v1041 = v994 * 16;
        int v1042;
        v1042 = v1041 + v987;
        float v1043;
        v1043 = 0.0f;
        int v1044;
        v1044 = 0;
        while (while_method_3(v1044)){
            int v1046;
            v1046 = 0;
            while (while_method_1(v1046)){
                assert("Tensor range check" && 0 <= v1044 && v1044 < 1);
                assert("Tensor range check" && 0 <= v1046 && v1046 < 4);
                int v1048;
                v1048 = 4 * v1044;
                int v1049;
                v1049 = v1048 + v1046;
                float v1050;
                v1050 = v998[v1049];
                float v1051;
                v1051 = v1043 + v1050;
                v1043 = v1051;
                v1046 += 1 ;
            }
            v1044 += 1 ;
        }
        auto v1052 = cooperative_groups::coalesced_threads();
        int v1053;
        v1053 = threadIdx.x;
        int v1054;
        v1054 = v1053 / 16;
        auto v1055 = cooperative_groups::labeled_partition(v1052,v1054);
        float v1056;
        v1056 = cooperative_groups::reduce(v1055, v1043, v42);
        float v1057;
        v1057 = v1056 / 64.0f;
        float v1058[4];
        int v1059;
        v1059 = 0;
        while (while_method_3(v1059)){
            int v1061;
            v1061 = 0;
            while (while_method_1(v1061)){
                assert("Tensor range check" && 0 <= v1059 && v1059 < 1);
                assert("Tensor range check" && 0 <= v1061 && v1061 < 4);
                int v1063;
                v1063 = 4 * v1059;
                int v1064;
                v1064 = v1063 + v1061;
                float v1065;
                v1065 = v998[v1064];
                float v1066;
                v1066 = v1065 - v1057;
                float v1067;
                v1067 = exp(v1066);
                assert("Tensor range check" && 0 <= v1059 && v1059 < 1);
                assert("Tensor range check" && 0 <= v1061 && v1061 < 4);
                v1058[v1064] = v1067;
                v1061 += 1 ;
            }
            v1059 += 1 ;
        }
        float v1068;
        v1068 = 0.0f;
        int v1069;
        v1069 = 0;
        while (while_method_3(v1069)){
            int v1071;
            v1071 = 0;
            while (while_method_1(v1071)){
                assert("Tensor range check" && 0 <= v1069 && v1069 < 1);
                assert("Tensor range check" && 0 <= v1071 && v1071 < 4);
                int v1073;
                v1073 = 4 * v1069;
                int v1074;
                v1074 = v1073 + v1071;
                float v1075;
                v1075 = v1058[v1074];
                float v1076;
                v1076 = v1068 + v1075;
                v1068 = v1076;
                v1071 += 1 ;
            }
            v1069 += 1 ;
        }
        auto v1077 = cooperative_groups::coalesced_threads();
        int v1078;
        v1078 = threadIdx.x;
        int v1079;
        v1079 = v1078 / 16;
        auto v1080 = cooperative_groups::labeled_partition(v1077,v1079);
        float v1081;
        v1081 = cooperative_groups::reduce(v1080, v1068, v42);
        float v1082[4];
        int v1083;
        v1083 = 0;
        while (while_method_3(v1083)){
            int v1085;
            v1085 = 0;
            while (while_method_1(v1085)){
                assert("Tensor range check" && 0 <= v1083 && v1083 < 1);
                assert("Tensor range check" && 0 <= v1085 && v1085 < 4);
                int v1087;
                v1087 = 4 * v1083;
                int v1088;
                v1088 = v1087 + v1085;
                float v1089;
                v1089 = v1058[v1088];
                float v1090;
                v1090 = v1089 / v1081;
                assert("Tensor range check" && 0 <= v1083 && v1083 < 1);
                assert("Tensor range check" && 0 <= v1085 && v1085 < 4);
                v1082[v1088] = v1090;
                v1085 += 1 ;
            }
            v1083 += 1 ;
        }
        float v1091[4];
        float v1092;
        v1092 = 0.0f;
        int v1093;
        v1093 = 0;
        while (while_method_3(v1093)){
            assert("Tensor range check" && 0 <= v1093 && v1093 < 1);
            int v1095;
            v1095 = 4 * v1093;
            assert("Tensor range check" && 0 <= v1093 && v1093 < 1);
            int v1096; float v1097;
            Tuple0 tmp30 = Tuple0{0, 0.0f};
            v1096 = tmp30.v0; v1097 = tmp30.v1;
            while (while_method_1(v1096)){
                assert("Tensor range check" && 0 <= v1096 && v1096 < 4);
                int v1099;
                v1099 = v1096 + v1095;
                float v1100;
                v1100 = v1082[v1099];
                float v1101;
                v1101 = v1097 + v1100;
                v1097 = v1101;
                v1096 += 1 ;
            }
            auto v1102 = cooperative_groups::coalesced_threads();
            int v1103;
            v1103 = threadIdx.x;
            int v1104;
            v1104 = v1103 / 16;
            auto v1105 = cooperative_groups::labeled_partition(v1102,v1104);
            Closure2 v1106{};
            float v1107;
            v1107 = cooperative_groups::inclusive_scan(v1105, v1097, v1106);
            float v1108;
            v1108 = v1105.shfl_up(v1107,1);
            bool v1109;
            v1109 = v1105.thread_rank() == 0;
            float v1110;
            if (v1109){
                v1110 = 0.0f;
            } else {
                v1110 = v1108;
            }
            float v1111;
            v1111 = v1105.shfl(v1107,v1105.num_threads()-1);
            float v1112;
            v1112 = v1092 + v1110;
            int v1113; float v1114;
            Tuple0 tmp31 = Tuple0{0, v1112};
            v1113 = tmp31.v0; v1114 = tmp31.v1;
            while (while_method_1(v1113)){
                assert("Tensor range check" && 0 <= v1113 && v1113 < 4);
                int v1116;
                v1116 = v1113 + v1095;
                float v1117;
                v1117 = v1082[v1116];
                float v1118;
                v1118 = v1114 + v1117;
                assert("Tensor range check" && 0 <= v1113 && v1113 < 4);
                v1091[v1116] = v1118;
                v1114 = v1118;
                v1113 += 1 ;
            }
            float v1119;
            v1119 = v1092 + v1111;
            v1092 = v1119;
            v1093 += 1 ;
        }
        float v1120[4];
        bool v1121[4];
        int v1122;
        v1122 = 0;
        while (while_method_3(v1122)){
            int v1124;
            v1124 = 0;
            while (while_method_1(v1124)){
                assert("Tensor range check" && 0 <= v1122 && v1122 < 1);
                assert("Tensor range check" && 0 <= v1124 && v1124 < 4);
                int v1126;
                v1126 = 4 * v1122;
                int v1127;
                v1127 = v1126 + v1124;
                float v1128;
                v1128 = v1091[v1127];
                float v1129;
                v1129 = v1082[v1127];
                bool v1130;
                v1130 = v1129 > 0.0f;
                assert("Tensor range check" && 0 <= v1122 && v1122 < 1);
                assert("Tensor range check" && 0 <= v1124 && v1124 < 4);
                v1120[v1127] = v1128;
                v1121[v1127] = v1130;
                v1124 += 1 ;
            }
            v1122 += 1 ;
        }
        float v1131; bool v1132;
        Tuple3 tmp32 = Tuple3{-1.0f / 0.0f, false};
        v1131 = tmp32.v0; v1132 = tmp32.v1;
        int v1133;
        v1133 = 0;
        while (while_method_3(v1133)){
            int v1135;
            v1135 = 0;
            while (while_method_1(v1135)){
                assert("Tensor range check" && 0 <= v1133 && v1133 < 1);
                assert("Tensor range check" && 0 <= v1135 && v1135 < 4);
                int v1137;
                v1137 = 4 * v1133;
                int v1138;
                v1138 = v1137 + v1135;
                float v1139;
                v1139 = v1120[v1138];
                bool v1140;
                v1140 = v1121[v1138];
                float v1147; bool v1148;
                if (v1132){
                    if (v1140){
                        bool v1141;
                        v1141 = v1131 >= v1139;
                        float v1142;
                        if (v1141){
                            v1142 = v1131;
                        } else {
                            v1142 = v1139;
                        }
                        v1147 = v1142; v1148 = true;
                    } else {
                        v1147 = v1131; v1148 = v1132;
                    }
                } else {
                    if (v1140){
                        v1147 = v1139; v1148 = v1140;
                    } else {
                        v1147 = v1131; v1148 = v1132;
                    }
                }
                v1131 = v1147;
                v1132 = v1148;
                v1135 += 1 ;
            }
            v1133 += 1 ;
        }
        auto v1149 = cooperative_groups::coalesced_threads();
        int v1150;
        v1150 = threadIdx.x;
        int v1151;
        v1151 = v1150 / 16;
        auto v1152 = cooperative_groups::labeled_partition(v1149,v1151);
        Closure5 v1153{};
        float v1154; bool v1155;
        Tuple3 tmp33 = cooperative_groups::reduce(v1152, Tuple3{v1131, v1132}, v1153);
        v1154 = tmp33.v0; v1155 = tmp33.v1;
        bool v1156;
        v1156 = v1155 == false;
        if (v1156){
            assert("The local reduce must be true." && v1155);
        } else {
        }
        float v1158[4];
        int v1159[4];
        int v1160;
        v1160 = 0;
        while (while_method_3(v1160)){
            int v1162;
            v1162 = 0;
            while (while_method_1(v1162)){
                assert("Tensor range check" && 0 <= v1160 && v1160 < 1);
                assert("Tensor range check" && 0 <= v1162 && v1162 < 4);
                int v1164;
                v1164 = 4 * v1160;
                int v1165;
                v1165 = v1164 + v1162;
                int v1166;
                v1166 = v999[v1165];
                float v1167;
                v1167 = curand_uniform(&v981);
                assert("Tensor range check" && 0 <= v1160 && v1160 < 1);
                assert("Tensor range check" && 0 <= v1162 && v1162 < 4);
                v1158[v1165] = v1167;
                v1159[v1165] = v1166;
                v1162 += 1 ;
            }
            v1160 += 1 ;
        }
        float v1168; int v1169;
        Tuple1 tmp34 = Tuple1{0.0f, 2147483647};
        v1168 = tmp34.v0; v1169 = tmp34.v1;
        int v1170;
        v1170 = 0;
        while (while_method_3(v1170)){
            int v1172;
            v1172 = 0;
            while (while_method_1(v1172)){
                assert("Tensor range check" && 0 <= v1170 && v1170 < 1);
                assert("Tensor range check" && 0 <= v1172 && v1172 < 4);
                int v1174;
                v1174 = 4 * v1170;
                int v1175;
                v1175 = v1174 + v1172;
                float v1176;
                v1176 = v1158[v1175];
                int v1177;
                v1177 = v1159[v1175];
                bool v1178;
                v1178 = v1169 < v1177;
                float v1179; int v1180;
                if (v1178){
                    v1179 = v1168; v1180 = v1169;
                } else {
                    v1179 = v1176; v1180 = v1177;
                }
                v1168 = v1179;
                v1169 = v1180;
                v1172 += 1 ;
            }
            v1170 += 1 ;
        }
        auto v1181 = cooperative_groups::coalesced_threads();
        int v1182;
        v1182 = threadIdx.x;
        int v1183;
        v1183 = v1182 / 16;
        auto v1184 = cooperative_groups::labeled_partition(v1181,v1183);
        Closure6 v1185{};
        float v1186; int v1187;
        Tuple1 tmp35 = cooperative_groups::reduce(v1184, Tuple1{v1168, v1169}, v1185);
        v1186 = tmp35.v0; v1187 = tmp35.v1;
        float v1188;
        v1188 = v1154 * v1186;
        int v1189[4];
        bool v1190[4];
        int v1191;
        v1191 = 0;
        while (while_method_3(v1191)){
            int v1193;
            v1193 = 0;
            while (while_method_1(v1193)){
                assert("Tensor range check" && 0 <= v1191 && v1191 < 1);
                assert("Tensor range check" && 0 <= v1193 && v1193 < 4);
                int v1195;
                v1195 = 4 * v1191;
                int v1196;
                v1196 = v1195 + v1193;
                float v1197;
                v1197 = v1120[v1196];
                bool v1198;
                v1198 = v1121[v1196];
                int v1199;
                v1199 = v999[v1196];
                int v1202; bool v1203;
                if (v1198){
                    float v1200;
                    v1200 = v1197 - v1188;
                    bool v1201;
                    v1201 = v1200 >= 0.0f;
                    v1202 = v1199; v1203 = v1201;
                } else {
                    v1202 = 2147483647; v1203 = false;
                }
                assert("Tensor range check" && 0 <= v1191 && v1191 < 1);
                assert("Tensor range check" && 0 <= v1193 && v1193 < 4);
                v1189[v1196] = v1202;
                v1190[v1196] = v1203;
                v1193 += 1 ;
            }
            v1191 += 1 ;
        }
        int v1204; bool v1205;
        Tuple4 tmp36 = Tuple4{2147483647, false};
        v1204 = tmp36.v0; v1205 = tmp36.v1;
        int v1206;
        v1206 = 0;
        while (while_method_3(v1206)){
            int v1208;
            v1208 = 0;
            while (while_method_1(v1208)){
                assert("Tensor range check" && 0 <= v1206 && v1206 < 1);
                assert("Tensor range check" && 0 <= v1208 && v1208 < 4);
                int v1210;
                v1210 = 4 * v1206;
                int v1211;
                v1211 = v1210 + v1208;
                int v1212;
                v1212 = v1189[v1211];
                bool v1213;
                v1213 = v1190[v1211];
                int v1220; bool v1221;
                if (v1205){
                    if (v1213){
                        bool v1214;
                        v1214 = v1204 < v1212;
                        int v1215;
                        if (v1214){
                            v1215 = v1204;
                        } else {
                            v1215 = v1212;
                        }
                        v1220 = v1215; v1221 = true;
                    } else {
                        v1220 = v1204; v1221 = v1205;
                    }
                } else {
                    if (v1213){
                        v1220 = v1212; v1221 = v1213;
                    } else {
                        v1220 = v1204; v1221 = v1205;
                    }
                }
                v1204 = v1220;
                v1205 = v1221;
                v1208 += 1 ;
            }
            v1206 += 1 ;
        }
        auto v1222 = cooperative_groups::coalesced_threads();
        int v1223;
        v1223 = threadIdx.x;
        int v1224;
        v1224 = v1223 / 16;
        auto v1225 = cooperative_groups::labeled_partition(v1222,v1224);
        Closure7 v1226{};
        int v1227; bool v1228;
        Tuple4 tmp37 = cooperative_groups::reduce(v1225, Tuple4{v1204, v1205}, v1226);
        v1227 = tmp37.v0; v1228 = tmp37.v1;
        bool v1229;
        v1229 = v1228 == false;
        if (v1229){
            assert("The local reduce must be true." && v1228);
        } else {
        }
        assert("Tensor range check" && 0 <= v994 && v994 < 8);
        int v1231;
        v1231 = 0;
        while (while_method_3(v1231)){
            assert("Tensor range check" && 0 <= v1231 && v1231 < 1);
            int v1233;
            v1233 = 64 * v1231;
            int v1234;
            v1234 = v1233 + v997;
            assert("Tensor range check" && 0 <= v1231 && v1231 < 1);
            int v1235;
            v1235 = 4 * v1231;
            int4* v1236;
            v1236 = reinterpret_cast<int4*>(v1082 + v1235);
            int4* v1237;
            v1237 = reinterpret_cast<int4*>(v14 + v1234);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1236) % 16 == 0 && reinterpret_cast<unsigned long long>(v1237) % 16 == 0);
            *v1237 = *v1236;
            v1231 += 1 ;
        }
        assert("Tensor range check" && 0 <= v994 && v994 < 8);
        int v1238;
        v1238 = 16 * v994;
        int v1239;
        v1239 = v1238 + v987;
        v15[v1239] = v1227;
        v994 += 1 ;
    }
    __syncthreads();
    int v1240;
    v1240 = threadIdx.x;
    int v1241;
    v1241 = blockIdx.x;
    int v1242;
    v1242 = v1241 * 256;
    int v1243;
    v1243 = v1240 + v1242;
    unsigned long long v1244;
    v1244 = (unsigned long long)v1243;
    curandStatePhilox4_32_10_t v1245;
    curand_init(12344321ull,v1244,0ull,&v1245);
    int v1246;
    v1246 = threadIdx.x;
    bool v1247;
    v1247 = 0 <= v1246;
    bool v1248;
    v1248 = v1247 == false;
    if (v1248){
        assert("The index needs to be zero or positive." && v1247);
    } else {
    }
    int v1250;
    v1250 = v1246 % 16;
    int v1251;
    v1251 = v1246 / 16;
    bool v1252;
    v1252 = v1251 < 16;
    bool v1253;
    v1253 = v1252 == false;
    if (v1253){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1252);
    } else {
    }
    assert("Tensor range check" && 0 <= v1251 && v1251 < 16);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 16);
    int v1255;
    v1255 = 4 * v1250;
    int v1256;
    v1256 = 64 * v1251;
    int v1257;
    v1257 = v1256 + v1255;
    assert("Tensor range check" && 0 <= v1251 && v1251 < 16);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 16);
    assert("Tensor range check" && 0 <= v1251 && v1251 < 16);
    int v1258;
    v1258 = 0;
    while (while_method_2(v1258)){
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1260;
        v1260 = 1024 * v1258;
        int v1261;
        v1261 = v1260 + v1257;
        float v1262[4];
        int v1263[4];
        int v1264;
        v1264 = 0;
        while (while_method_3(v1264)){
            assert("Tensor range check" && 0 <= v1264 && v1264 < 1);
            int v1266;
            v1266 = 4 * v1264;
            assert("Tensor range check" && 0 <= v1264 && v1264 < 1);
            int v1267;
            v1267 = 64 * v1264;
            int v1268;
            v1268 = v1267 + v1261;
            int4* v1269;
            v1269 = reinterpret_cast<int4*>(v1 + v1268);
            int4* v1270;
            v1270 = reinterpret_cast<int4*>(v1262 + v1266);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1269) % 16 == 0 && reinterpret_cast<unsigned long long>(v1270) % 16 == 0);
            *v1270 = *v1269;
            v1264 += 1 ;
        }
        int v1271;
        v1271 = 0;
        while (while_method_3(v1271)){
            int v1273;
            v1273 = 0;
            while (while_method_1(v1273)){
                bool v1275;
                v1275 = 0 <= v1273;
                bool v1277;
                if (v1275){
                    bool v1276;
                    v1276 = v1273 < 4;
                    v1277 = v1276;
                } else {
                    v1277 = false;
                }
                bool v1278;
                v1278 = v1277 == false;
                if (v1278){
                    assert("The indices should be inside the range of the dimension." && v1277);
                } else {
                }
                bool v1280;
                v1280 = 0 <= v1250;
                bool v1282;
                if (v1280){
                    bool v1281;
                    v1281 = v1250 < 16;
                    v1282 = v1281;
                } else {
                    v1282 = false;
                }
                bool v1283;
                v1283 = v1282 == false;
                if (v1283){
                    assert("The indices should be inside the range of the dimension." && v1282);
                } else {
                }
                int v1285;
                v1285 = v1250 * 4;
                int v1286;
                v1286 = v1273 + v1285;
                bool v1287;
                v1287 = 0 <= v1271;
                bool v1289;
                if (v1287){
                    bool v1288;
                    v1288 = v1271 < 1;
                    v1289 = v1288;
                } else {
                    v1289 = false;
                }
                bool v1290;
                v1290 = v1289 == false;
                if (v1290){
                    assert("The indices should be inside the range of the dimension." && v1289);
                } else {
                }
                int v1292;
                v1292 = v1271 * 64;
                int v1293;
                v1293 = v1286 + v1292;
                assert("Tensor range check" && 0 <= v1271 && v1271 < 1);
                assert("Tensor range check" && 0 <= v1273 && v1273 < 4);
                int v1294;
                v1294 = 4 * v1271;
                int v1295;
                v1295 = v1294 + v1273;
                v1263[v1295] = v1293;
                v1273 += 1 ;
            }
            v1271 += 1 ;
        }
        bool v1296;
        v1296 = 0 <= v1251;
        bool v1297;
        v1297 = v1296 && v1252;
        bool v1298;
        v1298 = v1297 == false;
        if (v1298){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1297);
        } else {
        }
        bool v1300;
        v1300 = 0 <= v1258;
        bool v1302;
        if (v1300){
            bool v1301;
            v1301 = v1258 < 8;
            v1302 = v1301;
        } else {
            v1302 = false;
        }
        bool v1303;
        v1303 = v1302 == false;
        if (v1303){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1302);
        } else {
        }
        int v1305;
        v1305 = v1258 * 16;
        int v1306;
        v1306 = v1305 + v1251;
        bool v1307[4];
        int v1308;
        v1308 = 0;
        while (while_method_3(v1308)){
            int v1310;
            v1310 = 0;
            while (while_method_1(v1310)){
                assert("Tensor range check" && 0 <= v1308 && v1308 < 1);
                assert("Tensor range check" && 0 <= v1310 && v1310 < 4);
                int v1312;
                v1312 = 4 * v1308;
                int v1313;
                v1313 = v1312 + v1310;
                float v1314;
                v1314 = v1262[v1313];
                int v1315;
                v1315 = v1263[v1313];
                bool v1316;
                v1316 = v1315 < 11;
                assert("Tensor range check" && 0 <= v1308 && v1308 < 1);
                assert("Tensor range check" && 0 <= v1310 && v1310 < 4);
                v1307[v1313] = v1316;
                v1310 += 1 ;
            }
            v1308 += 1 ;
        }
        int v1317[4];
        int v1318;
        v1318 = 0;
        while (while_method_3(v1318)){
            int v1320;
            v1320 = 0;
            while (while_method_1(v1320)){
                assert("Tensor range check" && 0 <= v1318 && v1318 < 1);
                assert("Tensor range check" && 0 <= v1320 && v1320 < 4);
                int v1322;
                v1322 = 4 * v1318;
                int v1323;
                v1323 = v1322 + v1320;
                bool v1324;
                v1324 = v1307[v1323];
                int v1325;
                if (v1324){
                    v1325 = 1;
                } else {
                    v1325 = 0;
                }
                assert("Tensor range check" && 0 <= v1318 && v1318 < 1);
                assert("Tensor range check" && 0 <= v1320 && v1320 < 4);
                v1317[v1323] = v1325;
                v1320 += 1 ;
            }
            v1318 += 1 ;
        }
        int v1326;
        v1326 = 0;
        int v1327;
        v1327 = 0;
        while (while_method_3(v1327)){
            int v1329;
            v1329 = 0;
            while (while_method_1(v1329)){
                assert("Tensor range check" && 0 <= v1327 && v1327 < 1);
                assert("Tensor range check" && 0 <= v1329 && v1329 < 4);
                int v1331;
                v1331 = 4 * v1327;
                int v1332;
                v1332 = v1331 + v1329;
                int v1333;
                v1333 = v1317[v1332];
                int v1334;
                v1334 = v1326 + v1333;
                v1326 = v1334;
                v1329 += 1 ;
            }
            v1327 += 1 ;
        }
        auto v1335 = cooperative_groups::coalesced_threads();
        int v1336;
        v1336 = threadIdx.x;
        int v1337;
        v1337 = v1336 / 16;
        auto v1338 = cooperative_groups::labeled_partition(v1335,v1337);
        Closure4 v1339{};
        int v1340;
        v1340 = cooperative_groups::reduce(v1338, v1326, v1339);
        float v1341[4];
        int v1342;
        v1342 = 0;
        while (while_method_3(v1342)){
            int v1344;
            v1344 = 0;
            while (while_method_1(v1344)){
                assert("Tensor range check" && 0 <= v1342 && v1342 < 1);
                assert("Tensor range check" && 0 <= v1344 && v1344 < 4);
                int v1346;
                v1346 = 4 * v1342;
                int v1347;
                v1347 = v1346 + v1344;
                float v1348;
                v1348 = v1262[v1347];
                bool v1349;
                v1349 = v1307[v1347];
                float v1350;
                if (v1349){
                    v1350 = v1348;
                } else {
                    v1350 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1342 && v1342 < 1);
                assert("Tensor range check" && 0 <= v1344 && v1344 < 4);
                v1341[v1347] = v1350;
                v1344 += 1 ;
            }
            v1342 += 1 ;
        }
        float v1351;
        v1351 = 0.0f;
        int v1352;
        v1352 = 0;
        while (while_method_3(v1352)){
            int v1354;
            v1354 = 0;
            while (while_method_1(v1354)){
                assert("Tensor range check" && 0 <= v1352 && v1352 < 1);
                assert("Tensor range check" && 0 <= v1354 && v1354 < 4);
                int v1356;
                v1356 = 4 * v1352;
                int v1357;
                v1357 = v1356 + v1354;
                float v1358;
                v1358 = v1341[v1357];
                float v1359;
                v1359 = v1351 + v1358;
                v1351 = v1359;
                v1354 += 1 ;
            }
            v1352 += 1 ;
        }
        auto v1360 = cooperative_groups::coalesced_threads();
        int v1361;
        v1361 = threadIdx.x;
        int v1362;
        v1362 = v1361 / 16;
        auto v1363 = cooperative_groups::labeled_partition(v1360,v1362);
        float v1364;
        v1364 = cooperative_groups::reduce(v1363, v1351, v42);
        float v1365;
        v1365 = (float)v1340;
        float v1366;
        v1366 = v1364 / v1365;
        float v1367[4];
        int v1368;
        v1368 = 0;
        while (while_method_3(v1368)){
            int v1370;
            v1370 = 0;
            while (while_method_1(v1370)){
                assert("Tensor range check" && 0 <= v1368 && v1368 < 1);
                assert("Tensor range check" && 0 <= v1370 && v1370 < 4);
                int v1372;
                v1372 = 4 * v1368;
                int v1373;
                v1373 = v1372 + v1370;
                float v1374;
                v1374 = v1262[v1373];
                bool v1375;
                v1375 = v1307[v1373];
                float v1376;
                if (v1375){
                    v1376 = v1374;
                } else {
                    v1376 = -1.0f / 0.0f;
                }
                float v1377;
                v1377 = v1376 - v1366;
                float v1378;
                v1378 = exp(v1377);
                assert("Tensor range check" && 0 <= v1368 && v1368 < 1);
                assert("Tensor range check" && 0 <= v1370 && v1370 < 4);
                v1367[v1373] = v1378;
                v1370 += 1 ;
            }
            v1368 += 1 ;
        }
        float v1379;
        v1379 = 0.0f;
        int v1380;
        v1380 = 0;
        while (while_method_3(v1380)){
            int v1382;
            v1382 = 0;
            while (while_method_1(v1382)){
                assert("Tensor range check" && 0 <= v1380 && v1380 < 1);
                assert("Tensor range check" && 0 <= v1382 && v1382 < 4);
                int v1384;
                v1384 = 4 * v1380;
                int v1385;
                v1385 = v1384 + v1382;
                float v1386;
                v1386 = v1367[v1385];
                float v1387;
                v1387 = v1379 + v1386;
                v1379 = v1387;
                v1382 += 1 ;
            }
            v1380 += 1 ;
        }
        auto v1388 = cooperative_groups::coalesced_threads();
        int v1389;
        v1389 = threadIdx.x;
        int v1390;
        v1390 = v1389 / 16;
        auto v1391 = cooperative_groups::labeled_partition(v1388,v1390);
        float v1392;
        v1392 = cooperative_groups::reduce(v1391, v1379, v42);
        float v1393[4];
        int v1394;
        v1394 = 0;
        while (while_method_3(v1394)){
            int v1396;
            v1396 = 0;
            while (while_method_1(v1396)){
                assert("Tensor range check" && 0 <= v1394 && v1394 < 1);
                assert("Tensor range check" && 0 <= v1396 && v1396 < 4);
                int v1398;
                v1398 = 4 * v1394;
                int v1399;
                v1399 = v1398 + v1396;
                float v1400;
                v1400 = v1367[v1399];
                float v1401;
                v1401 = v1400 / v1392;
                assert("Tensor range check" && 0 <= v1394 && v1394 < 1);
                assert("Tensor range check" && 0 <= v1396 && v1396 < 4);
                v1393[v1399] = v1401;
                v1396 += 1 ;
            }
            v1394 += 1 ;
        }
        float v1402[4];
        float v1403;
        v1403 = 0.0f;
        int v1404;
        v1404 = 0;
        while (while_method_3(v1404)){
            assert("Tensor range check" && 0 <= v1404 && v1404 < 1);
            int v1406;
            v1406 = 4 * v1404;
            assert("Tensor range check" && 0 <= v1404 && v1404 < 1);
            int v1407; float v1408;
            Tuple0 tmp38 = Tuple0{0, 0.0f};
            v1407 = tmp38.v0; v1408 = tmp38.v1;
            while (while_method_1(v1407)){
                assert("Tensor range check" && 0 <= v1407 && v1407 < 4);
                int v1410;
                v1410 = v1407 + v1406;
                float v1411;
                v1411 = v1393[v1410];
                float v1412;
                v1412 = v1408 + v1411;
                v1408 = v1412;
                v1407 += 1 ;
            }
            auto v1413 = cooperative_groups::coalesced_threads();
            int v1414;
            v1414 = threadIdx.x;
            int v1415;
            v1415 = v1414 / 16;
            auto v1416 = cooperative_groups::labeled_partition(v1413,v1415);
            Closure2 v1417{};
            float v1418;
            v1418 = cooperative_groups::inclusive_scan(v1416, v1408, v1417);
            float v1419;
            v1419 = v1416.shfl_up(v1418,1);
            bool v1420;
            v1420 = v1416.thread_rank() == 0;
            float v1421;
            if (v1420){
                v1421 = 0.0f;
            } else {
                v1421 = v1419;
            }
            float v1422;
            v1422 = v1416.shfl(v1418,v1416.num_threads()-1);
            float v1423;
            v1423 = v1403 + v1421;
            int v1424; float v1425;
            Tuple0 tmp39 = Tuple0{0, v1423};
            v1424 = tmp39.v0; v1425 = tmp39.v1;
            while (while_method_1(v1424)){
                assert("Tensor range check" && 0 <= v1424 && v1424 < 4);
                int v1427;
                v1427 = v1424 + v1406;
                float v1428;
                v1428 = v1393[v1427];
                float v1429;
                v1429 = v1425 + v1428;
                assert("Tensor range check" && 0 <= v1424 && v1424 < 4);
                v1402[v1427] = v1429;
                v1425 = v1429;
                v1424 += 1 ;
            }
            float v1430;
            v1430 = v1403 + v1422;
            v1403 = v1430;
            v1404 += 1 ;
        }
        float v1431[4];
        bool v1432[4];
        int v1433;
        v1433 = 0;
        while (while_method_3(v1433)){
            int v1435;
            v1435 = 0;
            while (while_method_1(v1435)){
                assert("Tensor range check" && 0 <= v1433 && v1433 < 1);
                assert("Tensor range check" && 0 <= v1435 && v1435 < 4);
                int v1437;
                v1437 = 4 * v1433;
                int v1438;
                v1438 = v1437 + v1435;
                float v1439;
                v1439 = v1402[v1438];
                float v1440;
                v1440 = v1393[v1438];
                bool v1441;
                v1441 = v1440 > 0.0f;
                assert("Tensor range check" && 0 <= v1433 && v1433 < 1);
                assert("Tensor range check" && 0 <= v1435 && v1435 < 4);
                v1431[v1438] = v1439;
                v1432[v1438] = v1441;
                v1435 += 1 ;
            }
            v1433 += 1 ;
        }
        float v1442; bool v1443;
        Tuple3 tmp40 = Tuple3{-1.0f / 0.0f, false};
        v1442 = tmp40.v0; v1443 = tmp40.v1;
        int v1444;
        v1444 = 0;
        while (while_method_3(v1444)){
            int v1446;
            v1446 = 0;
            while (while_method_1(v1446)){
                assert("Tensor range check" && 0 <= v1444 && v1444 < 1);
                assert("Tensor range check" && 0 <= v1446 && v1446 < 4);
                int v1448;
                v1448 = 4 * v1444;
                int v1449;
                v1449 = v1448 + v1446;
                float v1450;
                v1450 = v1431[v1449];
                bool v1451;
                v1451 = v1432[v1449];
                float v1458; bool v1459;
                if (v1443){
                    if (v1451){
                        bool v1452;
                        v1452 = v1442 >= v1450;
                        float v1453;
                        if (v1452){
                            v1453 = v1442;
                        } else {
                            v1453 = v1450;
                        }
                        v1458 = v1453; v1459 = true;
                    } else {
                        v1458 = v1442; v1459 = v1443;
                    }
                } else {
                    if (v1451){
                        v1458 = v1450; v1459 = v1451;
                    } else {
                        v1458 = v1442; v1459 = v1443;
                    }
                }
                v1442 = v1458;
                v1443 = v1459;
                v1446 += 1 ;
            }
            v1444 += 1 ;
        }
        auto v1460 = cooperative_groups::coalesced_threads();
        int v1461;
        v1461 = threadIdx.x;
        int v1462;
        v1462 = v1461 / 16;
        auto v1463 = cooperative_groups::labeled_partition(v1460,v1462);
        Closure5 v1464{};
        float v1465; bool v1466;
        Tuple3 tmp41 = cooperative_groups::reduce(v1463, Tuple3{v1442, v1443}, v1464);
        v1465 = tmp41.v0; v1466 = tmp41.v1;
        bool v1467;
        v1467 = v1466 == false;
        if (v1467){
            assert("The local reduce must be true." && v1466);
        } else {
        }
        float v1469[4];
        int v1470[4];
        int v1471;
        v1471 = 0;
        while (while_method_3(v1471)){
            int v1473;
            v1473 = 0;
            while (while_method_1(v1473)){
                assert("Tensor range check" && 0 <= v1471 && v1471 < 1);
                assert("Tensor range check" && 0 <= v1473 && v1473 < 4);
                int v1475;
                v1475 = 4 * v1471;
                int v1476;
                v1476 = v1475 + v1473;
                int v1477;
                v1477 = v1263[v1476];
                float v1478;
                v1478 = curand_uniform(&v1245);
                assert("Tensor range check" && 0 <= v1471 && v1471 < 1);
                assert("Tensor range check" && 0 <= v1473 && v1473 < 4);
                v1469[v1476] = v1478;
                v1470[v1476] = v1477;
                v1473 += 1 ;
            }
            v1471 += 1 ;
        }
        float v1479; int v1480;
        Tuple1 tmp42 = Tuple1{0.0f, 2147483647};
        v1479 = tmp42.v0; v1480 = tmp42.v1;
        int v1481;
        v1481 = 0;
        while (while_method_3(v1481)){
            int v1483;
            v1483 = 0;
            while (while_method_1(v1483)){
                assert("Tensor range check" && 0 <= v1481 && v1481 < 1);
                assert("Tensor range check" && 0 <= v1483 && v1483 < 4);
                int v1485;
                v1485 = 4 * v1481;
                int v1486;
                v1486 = v1485 + v1483;
                float v1487;
                v1487 = v1469[v1486];
                int v1488;
                v1488 = v1470[v1486];
                bool v1489;
                v1489 = v1480 < v1488;
                float v1490; int v1491;
                if (v1489){
                    v1490 = v1479; v1491 = v1480;
                } else {
                    v1490 = v1487; v1491 = v1488;
                }
                v1479 = v1490;
                v1480 = v1491;
                v1483 += 1 ;
            }
            v1481 += 1 ;
        }
        auto v1492 = cooperative_groups::coalesced_threads();
        int v1493;
        v1493 = threadIdx.x;
        int v1494;
        v1494 = v1493 / 16;
        auto v1495 = cooperative_groups::labeled_partition(v1492,v1494);
        Closure6 v1496{};
        float v1497; int v1498;
        Tuple1 tmp43 = cooperative_groups::reduce(v1495, Tuple1{v1479, v1480}, v1496);
        v1497 = tmp43.v0; v1498 = tmp43.v1;
        float v1499;
        v1499 = v1465 * v1497;
        int v1500[4];
        bool v1501[4];
        int v1502;
        v1502 = 0;
        while (while_method_3(v1502)){
            int v1504;
            v1504 = 0;
            while (while_method_1(v1504)){
                assert("Tensor range check" && 0 <= v1502 && v1502 < 1);
                assert("Tensor range check" && 0 <= v1504 && v1504 < 4);
                int v1506;
                v1506 = 4 * v1502;
                int v1507;
                v1507 = v1506 + v1504;
                float v1508;
                v1508 = v1431[v1507];
                bool v1509;
                v1509 = v1432[v1507];
                int v1510;
                v1510 = v1263[v1507];
                int v1513; bool v1514;
                if (v1509){
                    float v1511;
                    v1511 = v1508 - v1499;
                    bool v1512;
                    v1512 = v1511 >= 0.0f;
                    v1513 = v1510; v1514 = v1512;
                } else {
                    v1513 = 2147483647; v1514 = false;
                }
                assert("Tensor range check" && 0 <= v1502 && v1502 < 1);
                assert("Tensor range check" && 0 <= v1504 && v1504 < 4);
                v1500[v1507] = v1513;
                v1501[v1507] = v1514;
                v1504 += 1 ;
            }
            v1502 += 1 ;
        }
        int v1515; bool v1516;
        Tuple4 tmp44 = Tuple4{2147483647, false};
        v1515 = tmp44.v0; v1516 = tmp44.v1;
        int v1517;
        v1517 = 0;
        while (while_method_3(v1517)){
            int v1519;
            v1519 = 0;
            while (while_method_1(v1519)){
                assert("Tensor range check" && 0 <= v1517 && v1517 < 1);
                assert("Tensor range check" && 0 <= v1519 && v1519 < 4);
                int v1521;
                v1521 = 4 * v1517;
                int v1522;
                v1522 = v1521 + v1519;
                int v1523;
                v1523 = v1500[v1522];
                bool v1524;
                v1524 = v1501[v1522];
                int v1531; bool v1532;
                if (v1516){
                    if (v1524){
                        bool v1525;
                        v1525 = v1515 < v1523;
                        int v1526;
                        if (v1525){
                            v1526 = v1515;
                        } else {
                            v1526 = v1523;
                        }
                        v1531 = v1526; v1532 = true;
                    } else {
                        v1531 = v1515; v1532 = v1516;
                    }
                } else {
                    if (v1524){
                        v1531 = v1523; v1532 = v1524;
                    } else {
                        v1531 = v1515; v1532 = v1516;
                    }
                }
                v1515 = v1531;
                v1516 = v1532;
                v1519 += 1 ;
            }
            v1517 += 1 ;
        }
        auto v1533 = cooperative_groups::coalesced_threads();
        int v1534;
        v1534 = threadIdx.x;
        int v1535;
        v1535 = v1534 / 16;
        auto v1536 = cooperative_groups::labeled_partition(v1533,v1535);
        Closure7 v1537{};
        int v1538; bool v1539;
        Tuple4 tmp45 = cooperative_groups::reduce(v1536, Tuple4{v1515, v1516}, v1537);
        v1538 = tmp45.v0; v1539 = tmp45.v1;
        bool v1540;
        v1540 = v1539 == false;
        if (v1540){
            assert("The local reduce must be true." && v1539);
        } else {
        }
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1542;
        v1542 = 0;
        while (while_method_3(v1542)){
            assert("Tensor range check" && 0 <= v1542 && v1542 < 1);
            int v1544;
            v1544 = 64 * v1542;
            int v1545;
            v1545 = v1544 + v1261;
            assert("Tensor range check" && 0 <= v1542 && v1542 < 1);
            int v1546;
            v1546 = 4 * v1542;
            int4* v1547;
            v1547 = reinterpret_cast<int4*>(v1393 + v1546);
            int4* v1548;
            v1548 = reinterpret_cast<int4*>(v16 + v1545);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1547) % 16 == 0 && reinterpret_cast<unsigned long long>(v1548) % 16 == 0);
            *v1548 = *v1547;
            v1542 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1549;
        v1549 = 16 * v1258;
        int v1550;
        v1550 = v1549 + v1251;
        v17[v1550] = v1538;
        v1258 += 1 ;
    }
    __syncthreads();
    return ;
}
extern "C" __global__ void entry2(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 256);
    int v9;
    v9 = 16 * v8;
    int v10;
    v10 = threadIdx.x;
    assert("Tensor range check" && 0 <= v10 && v10 < 256);
    int v11;
    v11 = 16 * v10;
    int v12;
    v12 = threadIdx.x;
    assert("Tensor range check" && 0 <= v12 && v12 < 256);
    int v13;
    v13 = 16 * v12;
    int v14;
    v14 = threadIdx.x;
    assert("Tensor range check" && 0 <= v14 && v14 < 256);
    int v15;
    v15 = 16 * v14;
    int v16;
    v16 = threadIdx.x;
    assert("Tensor range check" && 0 <= v16 && v16 < 256);
    int v17;
    v17 = 16 * v16;
    float * v18;
    v18 = v1+v9;
    int * v20;
    v20 = v2+v15;
    int * v22;
    v22 = v3+v15;
    int v24;
    v24 = sizeof(float *);
    unsigned long long v25;
    v25 = (unsigned long long)v24;
    unsigned long long v26;
    v26 = 256ull * v25;
    unsigned long long v27;
    v27 = v26 + 16ull;
    unsigned long long v28;
    v28 = v27 - 1ull;
    unsigned long long v29;
    v29 = v28 % 16ull;
    unsigned long long v30;
    v30 = v28 - v29;
    int v31;
    v31 = sizeof(int *);
    unsigned long long v32;
    v32 = (unsigned long long)v31;
    unsigned long long v33;
    v33 = 256ull * v32;
    unsigned long long v34;
    v34 = v30 + v33;
    unsigned long long v35;
    v35 = v34 + 16ull;
    unsigned long long v36;
    v36 = v35 - 1ull;
    unsigned long long v37;
    v37 = v36 % 16ull;
    unsigned long long v38;
    v38 = v36 - v37;
    unsigned long long v39;
    v39 = v38 + v33;
    bool v40;
    v40 = v39 <= 98304ull;
    bool v41;
    v41 = v40 == false;
    if (v41){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v40);
    } else {
    }
    extern __shared__ unsigned char v43[];
    bool v44;
    v44 = v39 <= v39;
    bool v45;
    v45 = v44 == false;
    if (v45){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v44);
    } else {
    }
    float * * v47;
    v47 = reinterpret_cast<float * *>(&v43[0ull]);
    int * * v49;
    v49 = reinterpret_cast<int * *>(&v43[v30]);
    int * * v51;
    v51 = reinterpret_cast<int * *>(&v43[v38]);
    int v53;
    v53 = threadIdx.x;
    assert("Tensor range check" && 0 <= v53 && v53 < 256);
    v47[v53] = v18;
    v49[v53] = v20;
    v51[v53] = v22;
    __syncthreads();
    bool v54;
    v54 = 0 <= v53;
    bool v55;
    v55 = v54 == false;
    if (v55){
        assert("The index needs to be zero or positive." && v54);
    } else {
    }
    int v57;
    v57 = v53 % 4;
    int v58;
    v58 = v53 / 4;
    bool v59;
    v59 = v58 < 64;
    bool v60;
    v60 = v59 == false;
    if (v60){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v59);
    } else {
    }
    assert("Tensor range check" && 0 <= v58 && v58 < 64);
    int v62;
    v62 = 0;
    while (while_method_1(v62)){
        bool v64;
        v64 = 0 <= v58;
        bool v65;
        v65 = v64 && v59;
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v65);
        } else {
        }
        bool v68;
        v68 = 0 <= v62;
        bool v70;
        if (v68){
            bool v69;
            v69 = v62 < 4;
            v70 = v69;
        } else {
            v70 = false;
        }
        bool v71;
        v71 = v70 == false;
        if (v71){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v70);
        } else {
        }
        int v73;
        v73 = v62 * 64;
        int v74;
        v74 = v73 + v58;
        assert("Tensor range check" && 0 <= v62 && v62 < 4);
        int v75;
        v75 = 64 * v62;
        int v76;
        v76 = v75 + v58;
        float * v77;
        v77 = v47[v76];
        int * v78;
        v78 = v49[v76];
        int * v79;
        v79 = v51[v76];
        int v80;
        v80 = blockIdx.x;
        int v81;
        v81 = v80 * 256;
        int v82;
        v82 = v81 + v74;
        assert("Tensor range check" && 0 <= v57 && v57 < 4);
        int v83;
        v83 = 4 * v57;
        float v84[4];
        int v85[4];
        int v86;
        v86 = 0;
        while (while_method_3(v86)){
            assert("Tensor range check" && 0 <= v86 && v86 < 1);
            int v88;
            v88 = 4 * v86;
            assert("Tensor range check" && 0 <= v86 && v86 < 1);
            int v89;
            v89 = 16 * v86;
            int v90;
            v90 = v89 + v83;
            int4* v91;
            v91 = reinterpret_cast<int4*>(v77 + v90);
            int4* v92;
            v92 = reinterpret_cast<int4*>(v84 + v88);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v91) % 16 == 0 && reinterpret_cast<unsigned long long>(v92) % 16 == 0);
            *v92 = *v91;
            v86 += 1 ;
        }
        int v93;
        v93 = 0;
        while (while_method_3(v93)){
            int v95;
            v95 = 0;
            while (while_method_1(v95)){
                bool v97;
                v97 = 0 <= v95;
                bool v99;
                if (v97){
                    bool v98;
                    v98 = v95 < 4;
                    v99 = v98;
                } else {
                    v99 = false;
                }
                bool v100;
                v100 = v99 == false;
                if (v100){
                    assert("The indices should be inside the range of the dimension." && v99);
                } else {
                }
                bool v102;
                v102 = 0 <= v57;
                bool v104;
                if (v102){
                    bool v103;
                    v103 = v57 < 4;
                    v104 = v103;
                } else {
                    v104 = false;
                }
                bool v105;
                v105 = v104 == false;
                if (v105){
                    assert("The indices should be inside the range of the dimension." && v104);
                } else {
                }
                int v107;
                v107 = v57 * 4;
                int v108;
                v108 = v95 + v107;
                bool v109;
                v109 = 0 <= v93;
                bool v111;
                if (v109){
                    bool v110;
                    v110 = v93 < 1;
                    v111 = v110;
                } else {
                    v111 = false;
                }
                bool v112;
                v112 = v111 == false;
                if (v112){
                    assert("The indices should be inside the range of the dimension." && v111);
                } else {
                }
                int v114;
                v114 = v93 * 16;
                int v115;
                v115 = v108 + v114;
                assert("Tensor range check" && 0 <= v93 && v93 < 1);
                assert("Tensor range check" && 0 <= v95 && v95 < 4);
                int v116;
                v116 = 4 * v93;
                int v117;
                v117 = v116 + v95;
                v85[v117] = v115;
                v95 += 1 ;
            }
            v93 += 1 ;
        }
        int v118[4];
        int v119[4];
        int v120;
        v120 = 0;
        while (while_method_3(v120)){
            int v122;
            v122 = 0;
            while (while_method_1(v122)){
                assert("Tensor range check" && 0 <= v120 && v120 < 1);
                assert("Tensor range check" && 0 <= v122 && v122 < 4);
                int v124;
                v124 = 4 * v120;
                int v125;
                v125 = v124 + v122;
                int v126;
                v126 = v85[v125];
                assert("Tensor range check" && 0 <= v120 && v120 < 1);
                assert("Tensor range check" && 0 <= v122 && v122 < 4);
                v118[v125] = v82;
                v119[v125] = v126;
                v122 += 1 ;
            }
            v120 += 1 ;
        }
        int v127;
        v127 = 0;
        while (while_method_3(v127)){
            assert("Tensor range check" && 0 <= v127 && v127 < 1);
            int v129;
            v129 = 16 * v127;
            int v130;
            v130 = v129 + v83;
            assert("Tensor range check" && 0 <= v127 && v127 < 1);
            int v131;
            v131 = 4 * v127;
            int4* v132;
            v132 = reinterpret_cast<int4*>(v118 + v131);
            int4* v133;
            v133 = reinterpret_cast<int4*>(v78 + v130);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v132) % 16 == 0 && reinterpret_cast<unsigned long long>(v133) % 16 == 0);
            *v133 = *v132;
            int4* v134;
            v134 = reinterpret_cast<int4*>(v119 + v131);
            int4* v135;
            v135 = reinterpret_cast<int4*>(v79 + v130);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v134) % 16 == 0 && reinterpret_cast<unsigned long long>(v135) % 16 == 0);
            *v135 = *v134;
            v127 += 1 ;
        }
        assert("Tensor range check" && 0 <= v74 && v74 < 256);
        v62 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v53 && v53 < 256);
    __syncthreads();
    float * v136;
    v136 = v1+v9;
    unsigned long long v138;
    v138 = v30 + 1024ull;
    bool v139;
    v139 = v138 <= 98304ull;
    bool v140;
    v140 = v139 == false;
    if (v140){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v139);
    } else {
    }
    extern __shared__ unsigned char v142[];
    bool v143;
    v143 = v138 <= v138;
    bool v144;
    v144 = v143 == false;
    if (v144){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v143);
    } else {
    }
    float * * v146;
    v146 = reinterpret_cast<float * *>(&v142[0ull]);
    int * v148;
    v148 = reinterpret_cast<int *>(&v142[v30]);
    int v150;
    v150 = threadIdx.x;
    assert("Tensor range check" && 0 <= v150 && v150 < 256);
    v146[v150] = v136;
    __syncthreads();
    bool v151;
    v151 = 0 <= v150;
    bool v152;
    v152 = v151 == false;
    if (v152){
        assert("The index needs to be zero or positive." && v151);
    } else {
    }
    int v154;
    v154 = v150 % 4;
    int v155;
    v155 = v150 / 4;
    bool v156;
    v156 = v155 < 64;
    bool v157;
    v157 = v156 == false;
    if (v157){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v156);
    } else {
    }
    assert("Tensor range check" && 0 <= v155 && v155 < 64);
    int v159;
    v159 = 0;
    while (while_method_1(v159)){
        bool v161;
        v161 = 0 <= v155;
        bool v162;
        v162 = v161 && v156;
        bool v163;
        v163 = v162 == false;
        if (v163){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v162);
        } else {
        }
        bool v165;
        v165 = 0 <= v159;
        bool v167;
        if (v165){
            bool v166;
            v166 = v159 < 4;
            v167 = v166;
        } else {
            v167 = false;
        }
        bool v168;
        v168 = v167 == false;
        if (v168){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v167);
        } else {
        }
        int v170;
        v170 = v159 * 64;
        int v171;
        v171 = v170 + v155;
        assert("Tensor range check" && 0 <= v159 && v159 < 4);
        int v172;
        v172 = 64 * v159;
        int v173;
        v173 = v172 + v155;
        float * v174;
        v174 = v146[v173];
        int v175;
        v175 = blockIdx.x;
        int v176;
        v176 = v175 * 256;
        int v177;
        v177 = v176 + v171;
        assert("Tensor range check" && 0 <= v154 && v154 < 4);
        int v178;
        v178 = 4 * v154;
        float v179[4];
        int v180[4];
        int v181;
        v181 = 0;
        while (while_method_3(v181)){
            assert("Tensor range check" && 0 <= v181 && v181 < 1);
            int v183;
            v183 = 4 * v181;
            assert("Tensor range check" && 0 <= v181 && v181 < 1);
            int v184;
            v184 = 16 * v181;
            int v185;
            v185 = v184 + v178;
            int4* v186;
            v186 = reinterpret_cast<int4*>(v174 + v185);
            int4* v187;
            v187 = reinterpret_cast<int4*>(v179 + v183);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v186) % 16 == 0 && reinterpret_cast<unsigned long long>(v187) % 16 == 0);
            *v187 = *v186;
            v181 += 1 ;
        }
        int v188;
        v188 = 0;
        while (while_method_3(v188)){
            int v190;
            v190 = 0;
            while (while_method_1(v190)){
                bool v192;
                v192 = 0 <= v190;
                bool v194;
                if (v192){
                    bool v193;
                    v193 = v190 < 4;
                    v194 = v193;
                } else {
                    v194 = false;
                }
                bool v195;
                v195 = v194 == false;
                if (v195){
                    assert("The indices should be inside the range of the dimension." && v194);
                } else {
                }
                bool v197;
                v197 = 0 <= v154;
                bool v199;
                if (v197){
                    bool v198;
                    v198 = v154 < 4;
                    v199 = v198;
                } else {
                    v199 = false;
                }
                bool v200;
                v200 = v199 == false;
                if (v200){
                    assert("The indices should be inside the range of the dimension." && v199);
                } else {
                }
                int v202;
                v202 = v154 * 4;
                int v203;
                v203 = v190 + v202;
                bool v204;
                v204 = 0 <= v188;
                bool v206;
                if (v204){
                    bool v205;
                    v205 = v188 < 1;
                    v206 = v205;
                } else {
                    v206 = false;
                }
                bool v207;
                v207 = v206 == false;
                if (v207){
                    assert("The indices should be inside the range of the dimension." && v206);
                } else {
                }
                int v209;
                v209 = v188 * 16;
                int v210;
                v210 = v203 + v209;
                assert("Tensor range check" && 0 <= v188 && v188 < 1);
                assert("Tensor range check" && 0 <= v190 && v190 < 4);
                int v211;
                v211 = 4 * v188;
                int v212;
                v212 = v211 + v190;
                v180[v212] = v210;
                v190 += 1 ;
            }
            v188 += 1 ;
        }
        int v213;
        v213 = 0;
        while (while_method_3(v213)){
            assert("Tensor range check" && 0 <= v213 && v213 < 1);
            assert("Tensor range check" && 0 <= v213 && v213 < 1);
            v213 += 1 ;
        }
        assert("Tensor range check" && 0 <= v171 && v171 < 256);
        v148[v171] = v177;
        v159 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v150 && v150 < 256);
    int v215;
    v215 = v148[v150];
    __syncthreads();
    int v216;
    v216 = threadIdx.x;
    assert("Tensor range check" && 0 <= v216 && v216 < 256);
    v4[v216] = v215;
    float * v217;
    v217 = v1+v9;
    float * v219;
    v219 = v6+v17;
    unsigned long long v221;
    v221 = v30 + v26;
    bool v222;
    v222 = v221 <= 98304ull;
    bool v223;
    v223 = v222 == false;
    if (v223){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v222);
    } else {
    }
    extern __shared__ unsigned char v225[];
    bool v226;
    v226 = v221 <= v221;
    bool v227;
    v227 = v226 == false;
    if (v227){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v226);
    } else {
    }
    float * * v229;
    v229 = reinterpret_cast<float * *>(&v225[0ull]);
    float * * v231;
    v231 = reinterpret_cast<float * *>(&v225[v30]);
    int v233;
    v233 = threadIdx.x;
    assert("Tensor range check" && 0 <= v233 && v233 < 256);
    v229[v233] = v217;
    v231[v233] = v219;
    __syncthreads();
    bool v234;
    v234 = 0 <= v233;
    bool v235;
    v235 = v234 == false;
    if (v235){
        assert("The index needs to be zero or positive." && v234);
    } else {
    }
    int v237;
    v237 = v233 % 4;
    int v238;
    v238 = v233 / 4;
    bool v239;
    v239 = v238 < 64;
    bool v240;
    v240 = v239 == false;
    if (v240){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v239);
    } else {
    }
    assert("Tensor range check" && 0 <= v238 && v238 < 64);
    int v242;
    v242 = 0;
    while (while_method_1(v242)){
        bool v244;
        v244 = 0 <= v238;
        bool v245;
        v245 = v244 && v239;
        bool v246;
        v246 = v245 == false;
        if (v246){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v245);
        } else {
        }
        bool v248;
        v248 = 0 <= v242;
        bool v250;
        if (v248){
            bool v249;
            v249 = v242 < 4;
            v250 = v249;
        } else {
            v250 = false;
        }
        bool v251;
        v251 = v250 == false;
        if (v251){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v250);
        } else {
        }
        int v253;
        v253 = v242 * 64;
        int v254;
        v254 = v253 + v238;
        assert("Tensor range check" && 0 <= v242 && v242 < 4);
        int v255;
        v255 = 64 * v242;
        int v256;
        v256 = v255 + v238;
        float * v257;
        v257 = v229[v256];
        float * v258;
        v258 = v231[v256];
        int v259;
        v259 = blockIdx.x;
        int v260;
        v260 = v259 * 256;
        int v261;
        v261 = v260 + v254;
        assert("Tensor range check" && 0 <= v237 && v237 < 4);
        int v262;
        v262 = 4 * v237;
        float v263[4];
        int v264[4];
        int v265;
        v265 = 0;
        while (while_method_3(v265)){
            assert("Tensor range check" && 0 <= v265 && v265 < 1);
            int v267;
            v267 = 4 * v265;
            assert("Tensor range check" && 0 <= v265 && v265 < 1);
            int v268;
            v268 = 16 * v265;
            int v269;
            v269 = v268 + v262;
            int4* v270;
            v270 = reinterpret_cast<int4*>(v257 + v269);
            int4* v271;
            v271 = reinterpret_cast<int4*>(v263 + v267);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v270) % 16 == 0 && reinterpret_cast<unsigned long long>(v271) % 16 == 0);
            *v271 = *v270;
            v265 += 1 ;
        }
        int v272;
        v272 = 0;
        while (while_method_3(v272)){
            int v274;
            v274 = 0;
            while (while_method_1(v274)){
                bool v276;
                v276 = 0 <= v274;
                bool v278;
                if (v276){
                    bool v277;
                    v277 = v274 < 4;
                    v278 = v277;
                } else {
                    v278 = false;
                }
                bool v279;
                v279 = v278 == false;
                if (v279){
                    assert("The indices should be inside the range of the dimension." && v278);
                } else {
                }
                bool v281;
                v281 = 0 <= v237;
                bool v283;
                if (v281){
                    bool v282;
                    v282 = v237 < 4;
                    v283 = v282;
                } else {
                    v283 = false;
                }
                bool v284;
                v284 = v283 == false;
                if (v284){
                    assert("The indices should be inside the range of the dimension." && v283);
                } else {
                }
                int v286;
                v286 = v237 * 4;
                int v287;
                v287 = v274 + v286;
                bool v288;
                v288 = 0 <= v272;
                bool v290;
                if (v288){
                    bool v289;
                    v289 = v272 < 1;
                    v290 = v289;
                } else {
                    v290 = false;
                }
                bool v291;
                v291 = v290 == false;
                if (v291){
                    assert("The indices should be inside the range of the dimension." && v290);
                } else {
                }
                int v293;
                v293 = v272 * 16;
                int v294;
                v294 = v287 + v293;
                assert("Tensor range check" && 0 <= v272 && v272 < 1);
                assert("Tensor range check" && 0 <= v274 && v274 < 4);
                int v295;
                v295 = 4 * v272;
                int v296;
                v296 = v295 + v274;
                v264[v296] = v294;
                v274 += 1 ;
            }
            v272 += 1 ;
        }
        int v297;
        v297 = 0;
        while (while_method_3(v297)){
            assert("Tensor range check" && 0 <= v297 && v297 < 1);
            int v299;
            v299 = 16 * v297;
            int v300;
            v300 = v299 + v262;
            assert("Tensor range check" && 0 <= v297 && v297 < 1);
            int v301;
            v301 = 4 * v297;
            int4* v302;
            v302 = reinterpret_cast<int4*>(v263 + v301);
            int4* v303;
            v303 = reinterpret_cast<int4*>(v258 + v300);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v302) % 16 == 0 && reinterpret_cast<unsigned long long>(v303) % 16 == 0);
            *v303 = *v302;
            v297 += 1 ;
        }
        assert("Tensor range check" && 0 <= v254 && v254 < 256);
        v242 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v233 && v233 < 256);
    __syncthreads();
    float * v304;
    v304 = v1+v9;
    float * v306;
    v306 = v7+v13;
    if (v223){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v222);
    } else {
    }
    extern __shared__ unsigned char v309[];
    if (v227){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v226);
    } else {
    }
    float * * v311;
    v311 = reinterpret_cast<float * *>(&v309[0ull]);
    float * * v313;
    v313 = reinterpret_cast<float * *>(&v309[v30]);
    int v315;
    v315 = threadIdx.x;
    assert("Tensor range check" && 0 <= v315 && v315 < 256);
    v311[v315] = v304;
    v313[v315] = v306;
    __syncthreads();
    bool v316;
    v316 = 0 <= v315;
    bool v317;
    v317 = v316 == false;
    if (v317){
        assert("The index needs to be zero or positive." && v316);
    } else {
    }
    int v319;
    v319 = v315 % 4;
    int v320;
    v320 = v315 / 4;
    bool v321;
    v321 = v320 < 64;
    bool v322;
    v322 = v321 == false;
    if (v322){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v321);
    } else {
    }
    assert("Tensor range check" && 0 <= v320 && v320 < 64);
    int v324;
    v324 = 0;
    while (while_method_1(v324)){
        bool v326;
        v326 = 0 <= v320;
        bool v327;
        v327 = v326 && v321;
        bool v328;
        v328 = v327 == false;
        if (v328){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v327);
        } else {
        }
        bool v330;
        v330 = 0 <= v324;
        bool v332;
        if (v330){
            bool v331;
            v331 = v324 < 4;
            v332 = v331;
        } else {
            v332 = false;
        }
        bool v333;
        v333 = v332 == false;
        if (v333){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v332);
        } else {
        }
        int v335;
        v335 = v324 * 64;
        int v336;
        v336 = v335 + v320;
        assert("Tensor range check" && 0 <= v324 && v324 < 4);
        int v337;
        v337 = 64 * v324;
        int v338;
        v338 = v337 + v320;
        float * v339;
        v339 = v311[v338];
        float * v340;
        v340 = v313[v338];
        int v341;
        v341 = blockIdx.x;
        int v342;
        v342 = v341 * 256;
        int v343;
        v343 = v342 + v336;
        assert("Tensor range check" && 0 <= v319 && v319 < 4);
        int v344;
        v344 = 4 * v319;
        float v345[4];
        int v346[4];
        int v347;
        v347 = 0;
        while (while_method_3(v347)){
            assert("Tensor range check" && 0 <= v347 && v347 < 1);
            int v349;
            v349 = 4 * v347;
            assert("Tensor range check" && 0 <= v347 && v347 < 1);
            int v350;
            v350 = 16 * v347;
            int v351;
            v351 = v350 + v344;
            int4* v352;
            v352 = reinterpret_cast<int4*>(v339 + v351);
            int4* v353;
            v353 = reinterpret_cast<int4*>(v345 + v349);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v352) % 16 == 0 && reinterpret_cast<unsigned long long>(v353) % 16 == 0);
            *v353 = *v352;
            v347 += 1 ;
        }
        int v354;
        v354 = 0;
        while (while_method_3(v354)){
            int v356;
            v356 = 0;
            while (while_method_1(v356)){
                bool v358;
                v358 = 0 <= v356;
                bool v360;
                if (v358){
                    bool v359;
                    v359 = v356 < 4;
                    v360 = v359;
                } else {
                    v360 = false;
                }
                bool v361;
                v361 = v360 == false;
                if (v361){
                    assert("The indices should be inside the range of the dimension." && v360);
                } else {
                }
                bool v363;
                v363 = 0 <= v319;
                bool v365;
                if (v363){
                    bool v364;
                    v364 = v319 < 4;
                    v365 = v364;
                } else {
                    v365 = false;
                }
                bool v366;
                v366 = v365 == false;
                if (v366){
                    assert("The indices should be inside the range of the dimension." && v365);
                } else {
                }
                int v368;
                v368 = v319 * 4;
                int v369;
                v369 = v356 + v368;
                bool v370;
                v370 = 0 <= v354;
                bool v372;
                if (v370){
                    bool v371;
                    v371 = v354 < 1;
                    v372 = v371;
                } else {
                    v372 = false;
                }
                bool v373;
                v373 = v372 == false;
                if (v373){
                    assert("The indices should be inside the range of the dimension." && v372);
                } else {
                }
                int v375;
                v375 = v354 * 16;
                int v376;
                v376 = v369 + v375;
                assert("Tensor range check" && 0 <= v354 && v354 < 1);
                assert("Tensor range check" && 0 <= v356 && v356 < 4);
                int v377;
                v377 = 4 * v354;
                int v378;
                v378 = v377 + v356;
                v346[v378] = v376;
                v356 += 1 ;
            }
            v354 += 1 ;
        }
        bool v379[4];
        int v380;
        v380 = 0;
        while (while_method_3(v380)){
            int v382;
            v382 = 0;
            while (while_method_1(v382)){
                assert("Tensor range check" && 0 <= v380 && v380 < 1);
                assert("Tensor range check" && 0 <= v382 && v382 < 4);
                int v384;
                v384 = 4 * v380;
                int v385;
                v385 = v384 + v382;
                float v386;
                v386 = v345[v385];
                int v387;
                v387 = v346[v385];
                bool v388;
                v388 = v387 < 3;
                assert("Tensor range check" && 0 <= v380 && v380 < 1);
                assert("Tensor range check" && 0 <= v382 && v382 < 4);
                v379[v385] = v388;
                v382 += 1 ;
            }
            v380 += 1 ;
        }
        float v389[4];
        int v390;
        v390 = 0;
        while (while_method_3(v390)){
            int v392;
            v392 = 0;
            while (while_method_1(v392)){
                assert("Tensor range check" && 0 <= v390 && v390 < 1);
                assert("Tensor range check" && 0 <= v392 && v392 < 4);
                int v394;
                v394 = 4 * v390;
                int v395;
                v395 = v394 + v392;
                float v396;
                v396 = v345[v395];
                bool v397;
                v397 = v379[v395];
                float v400;
                if (v397){
                    bool v398;
                    v398 = 0.0f >= v396;
                    if (v398){
                        v400 = 0.0f;
                    } else {
                        v400 = v396;
                    }
                } else {
                    v400 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v390 && v390 < 1);
                assert("Tensor range check" && 0 <= v392 && v392 < 4);
                v389[v395] = v400;
                v392 += 1 ;
            }
            v390 += 1 ;
        }
        float v401;
        v401 = 0.0f;
        int v402;
        v402 = 0;
        while (while_method_3(v402)){
            int v404;
            v404 = 0;
            while (while_method_1(v404)){
                assert("Tensor range check" && 0 <= v402 && v402 < 1);
                assert("Tensor range check" && 0 <= v404 && v404 < 4);
                int v406;
                v406 = 4 * v402;
                int v407;
                v407 = v406 + v404;
                float v408;
                v408 = v389[v407];
                float v409;
                v409 = v401 + v408;
                v401 = v409;
                v404 += 1 ;
            }
            v402 += 1 ;
        }
        auto v410 = cooperative_groups::coalesced_threads();
        int v411;
        v411 = threadIdx.x;
        int v412;
        v412 = v411 / 4;
        auto v413 = cooperative_groups::labeled_partition(v410,v412);
        Closure0 v414{};
        float v415;
        v415 = cooperative_groups::reduce(v413, v401, v414);
        int v416[4];
        int v417;
        v417 = 0;
        while (while_method_3(v417)){
            int v419;
            v419 = 0;
            while (while_method_1(v419)){
                assert("Tensor range check" && 0 <= v417 && v417 < 1);
                assert("Tensor range check" && 0 <= v419 && v419 < 4);
                int v421;
                v421 = 4 * v417;
                int v422;
                v422 = v421 + v419;
                bool v423;
                v423 = v379[v422];
                int v424;
                if (v423){
                    v424 = 1;
                } else {
                    v424 = 0;
                }
                assert("Tensor range check" && 0 <= v417 && v417 < 1);
                assert("Tensor range check" && 0 <= v419 && v419 < 4);
                v416[v422] = v424;
                v419 += 1 ;
            }
            v417 += 1 ;
        }
        int v425;
        v425 = 0;
        int v426;
        v426 = 0;
        while (while_method_3(v426)){
            int v428;
            v428 = 0;
            while (while_method_1(v428)){
                assert("Tensor range check" && 0 <= v426 && v426 < 1);
                assert("Tensor range check" && 0 <= v428 && v428 < 4);
                int v430;
                v430 = 4 * v426;
                int v431;
                v431 = v430 + v428;
                int v432;
                v432 = v416[v431];
                int v433;
                v433 = v425 + v432;
                v425 = v433;
                v428 += 1 ;
            }
            v426 += 1 ;
        }
        auto v434 = cooperative_groups::coalesced_threads();
        int v435;
        v435 = threadIdx.x;
        int v436;
        v436 = v435 / 4;
        auto v437 = cooperative_groups::labeled_partition(v434,v436);
        Closure4 v438{};
        int v439;
        v439 = cooperative_groups::reduce(v437, v425, v438);
        float v440;
        v440 = (float)v439;
        float v441;
        v441 = 1.0f / v440;
        float v442[4];
        int v443;
        v443 = 0;
        while (while_method_3(v443)){
            int v445;
            v445 = 0;
            while (while_method_1(v445)){
                assert("Tensor range check" && 0 <= v443 && v443 < 1);
                assert("Tensor range check" && 0 <= v445 && v445 < 4);
                int v447;
                v447 = 4 * v443;
                int v448;
                v448 = v447 + v445;
                float v449;
                v449 = v389[v448];
                bool v450;
                v450 = v379[v448];
                bool v451;
                v451 = v450 == false;
                float v456;
                if (v451){
                    v456 = 0.0f;
                } else {
                    bool v452;
                    v452 = v415 == 0.0f;
                    bool v453;
                    v453 = v452 != true;
                    if (v453){
                        float v454;
                        v454 = v449 / v415;
                        v456 = v454;
                    } else {
                        v456 = v441;
                    }
                }
                assert("Tensor range check" && 0 <= v443 && v443 < 1);
                assert("Tensor range check" && 0 <= v445 && v445 < 4);
                v442[v448] = v456;
                v445 += 1 ;
            }
            v443 += 1 ;
        }
        int v457;
        v457 = 0;
        while (while_method_3(v457)){
            assert("Tensor range check" && 0 <= v457 && v457 < 1);
            int v459;
            v459 = 16 * v457;
            int v460;
            v460 = v459 + v344;
            assert("Tensor range check" && 0 <= v457 && v457 < 1);
            int v461;
            v461 = 4 * v457;
            int4* v462;
            v462 = reinterpret_cast<int4*>(v442 + v461);
            int4* v463;
            v463 = reinterpret_cast<int4*>(v340 + v460);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v462) % 16 == 0 && reinterpret_cast<unsigned long long>(v463) % 16 == 0);
            *v463 = *v462;
            v457 += 1 ;
        }
        assert("Tensor range check" && 0 <= v336 && v336 < 256);
        v324 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v315 && v315 < 256);
    __syncthreads();
    int v464;
    v464 = threadIdx.x;
    int v465;
    v465 = blockIdx.x;
    int v466;
    v466 = v465 * 256;
    int v467;
    v467 = v464 + v466;
    unsigned long long v468;
    v468 = (unsigned long long)v467;
    curandStatePhilox4_32_10_t v469;
    curand_init(12344321ull,v468,0ull,&v469);
    float * v470;
    v470 = v1+v9;
    if (v140){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v139);
    } else {
    }
    extern __shared__ unsigned char v473[];
    if (v144){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v143);
    } else {
    }
    float * * v475;
    v475 = reinterpret_cast<float * *>(&v473[0ull]);
    int * v477;
    v477 = reinterpret_cast<int *>(&v473[v30]);
    int v479;
    v479 = threadIdx.x;
    assert("Tensor range check" && 0 <= v479 && v479 < 256);
    v475[v479] = v470;
    __syncthreads();
    bool v480;
    v480 = 0 <= v479;
    bool v481;
    v481 = v480 == false;
    if (v481){
        assert("The index needs to be zero or positive." && v480);
    } else {
    }
    int v483;
    v483 = v479 % 4;
    int v484;
    v484 = v479 / 4;
    bool v485;
    v485 = v484 < 64;
    bool v486;
    v486 = v485 == false;
    if (v486){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v485);
    } else {
    }
    assert("Tensor range check" && 0 <= v484 && v484 < 64);
    int v488;
    v488 = 0;
    while (while_method_1(v488)){
        bool v490;
        v490 = 0 <= v484;
        bool v491;
        v491 = v490 && v485;
        bool v492;
        v492 = v491 == false;
        if (v492){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v491);
        } else {
        }
        bool v494;
        v494 = 0 <= v488;
        bool v496;
        if (v494){
            bool v495;
            v495 = v488 < 4;
            v496 = v495;
        } else {
            v496 = false;
        }
        bool v497;
        v497 = v496 == false;
        if (v497){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v496);
        } else {
        }
        int v499;
        v499 = v488 * 64;
        int v500;
        v500 = v499 + v484;
        assert("Tensor range check" && 0 <= v488 && v488 < 4);
        int v501;
        v501 = 64 * v488;
        int v502;
        v502 = v501 + v484;
        float * v503;
        v503 = v475[v502];
        int v504;
        v504 = blockIdx.x;
        int v505;
        v505 = v504 * 256;
        int v506;
        v506 = v505 + v500;
        assert("Tensor range check" && 0 <= v483 && v483 < 4);
        int v507;
        v507 = 4 * v483;
        float v508[4];
        int v509[4];
        int v510;
        v510 = 0;
        while (while_method_3(v510)){
            assert("Tensor range check" && 0 <= v510 && v510 < 1);
            int v512;
            v512 = 4 * v510;
            assert("Tensor range check" && 0 <= v510 && v510 < 1);
            int v513;
            v513 = 16 * v510;
            int v514;
            v514 = v513 + v507;
            int4* v515;
            v515 = reinterpret_cast<int4*>(v503 + v514);
            int4* v516;
            v516 = reinterpret_cast<int4*>(v508 + v512);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v515) % 16 == 0 && reinterpret_cast<unsigned long long>(v516) % 16 == 0);
            *v516 = *v515;
            v510 += 1 ;
        }
        int v517;
        v517 = 0;
        while (while_method_3(v517)){
            int v519;
            v519 = 0;
            while (while_method_1(v519)){
                bool v521;
                v521 = 0 <= v519;
                bool v523;
                if (v521){
                    bool v522;
                    v522 = v519 < 4;
                    v523 = v522;
                } else {
                    v523 = false;
                }
                bool v524;
                v524 = v523 == false;
                if (v524){
                    assert("The indices should be inside the range of the dimension." && v523);
                } else {
                }
                bool v526;
                v526 = 0 <= v483;
                bool v528;
                if (v526){
                    bool v527;
                    v527 = v483 < 4;
                    v528 = v527;
                } else {
                    v528 = false;
                }
                bool v529;
                v529 = v528 == false;
                if (v529){
                    assert("The indices should be inside the range of the dimension." && v528);
                } else {
                }
                int v531;
                v531 = v483 * 4;
                int v532;
                v532 = v519 + v531;
                bool v533;
                v533 = 0 <= v517;
                bool v535;
                if (v533){
                    bool v534;
                    v534 = v517 < 1;
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
                v538 = v517 * 16;
                int v539;
                v539 = v532 + v538;
                assert("Tensor range check" && 0 <= v517 && v517 < 1);
                assert("Tensor range check" && 0 <= v519 && v519 < 4);
                int v540;
                v540 = 4 * v517;
                int v541;
                v541 = v540 + v519;
                v509[v541] = v539;
                v519 += 1 ;
            }
            v517 += 1 ;
        }
        bool v542[4];
        int v543;
        v543 = 0;
        while (while_method_3(v543)){
            int v545;
            v545 = 0;
            while (while_method_1(v545)){
                assert("Tensor range check" && 0 <= v543 && v543 < 1);
                assert("Tensor range check" && 0 <= v545 && v545 < 4);
                int v547;
                v547 = 4 * v543;
                int v548;
                v548 = v547 + v545;
                float v549;
                v549 = v508[v548];
                int v550;
                v550 = v509[v548];
                bool v551;
                v551 = v550 < 3;
                assert("Tensor range check" && 0 <= v543 && v543 < 1);
                assert("Tensor range check" && 0 <= v545 && v545 < 4);
                v542[v548] = v551;
                v545 += 1 ;
            }
            v543 += 1 ;
        }
        int v552[4];
        int v553;
        v553 = 0;
        while (while_method_3(v553)){
            int v555;
            v555 = 0;
            while (while_method_1(v555)){
                assert("Tensor range check" && 0 <= v553 && v553 < 1);
                assert("Tensor range check" && 0 <= v555 && v555 < 4);
                int v557;
                v557 = 4 * v553;
                int v558;
                v558 = v557 + v555;
                bool v559;
                v559 = v542[v558];
                int v560;
                if (v559){
                    v560 = 1;
                } else {
                    v560 = 0;
                }
                assert("Tensor range check" && 0 <= v553 && v553 < 1);
                assert("Tensor range check" && 0 <= v555 && v555 < 4);
                v552[v558] = v560;
                v555 += 1 ;
            }
            v553 += 1 ;
        }
        int v561;
        v561 = 0;
        int v562;
        v562 = 0;
        while (while_method_3(v562)){
            int v564;
            v564 = 0;
            while (while_method_1(v564)){
                assert("Tensor range check" && 0 <= v562 && v562 < 1);
                assert("Tensor range check" && 0 <= v564 && v564 < 4);
                int v566;
                v566 = 4 * v562;
                int v567;
                v567 = v566 + v564;
                int v568;
                v568 = v552[v567];
                int v569;
                v569 = v561 + v568;
                v561 = v569;
                v564 += 1 ;
            }
            v562 += 1 ;
        }
        auto v570 = cooperative_groups::coalesced_threads();
        int v571;
        v571 = threadIdx.x;
        int v572;
        v572 = v571 / 4;
        auto v573 = cooperative_groups::labeled_partition(v570,v572);
        Closure4 v574{};
        int v575;
        v575 = cooperative_groups::reduce(v573, v561, v574);
        float v576[4];
        int v577;
        v577 = 0;
        while (while_method_3(v577)){
            int v579;
            v579 = 0;
            while (while_method_1(v579)){
                assert("Tensor range check" && 0 <= v577 && v577 < 1);
                assert("Tensor range check" && 0 <= v579 && v579 < 4);
                int v581;
                v581 = 4 * v577;
                int v582;
                v582 = v581 + v579;
                float v583;
                v583 = v508[v582];
                bool v584;
                v584 = v542[v582];
                float v585;
                if (v584){
                    v585 = v583;
                } else {
                    v585 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v577 && v577 < 1);
                assert("Tensor range check" && 0 <= v579 && v579 < 4);
                v576[v582] = v585;
                v579 += 1 ;
            }
            v577 += 1 ;
        }
        float v586;
        v586 = 0.0f;
        int v587;
        v587 = 0;
        while (while_method_3(v587)){
            int v589;
            v589 = 0;
            while (while_method_1(v589)){
                assert("Tensor range check" && 0 <= v587 && v587 < 1);
                assert("Tensor range check" && 0 <= v589 && v589 < 4);
                int v591;
                v591 = 4 * v587;
                int v592;
                v592 = v591 + v589;
                float v593;
                v593 = v576[v592];
                float v594;
                v594 = v586 + v593;
                v586 = v594;
                v589 += 1 ;
            }
            v587 += 1 ;
        }
        auto v595 = cooperative_groups::coalesced_threads();
        int v596;
        v596 = threadIdx.x;
        int v597;
        v597 = v596 / 4;
        auto v598 = cooperative_groups::labeled_partition(v595,v597);
        Closure0 v599{};
        float v600;
        v600 = cooperative_groups::reduce(v598, v586, v599);
        float v601;
        v601 = (float)v575;
        float v602;
        v602 = v600 / v601;
        float v603[4];
        int v604;
        v604 = 0;
        while (while_method_3(v604)){
            int v606;
            v606 = 0;
            while (while_method_1(v606)){
                assert("Tensor range check" && 0 <= v604 && v604 < 1);
                assert("Tensor range check" && 0 <= v606 && v606 < 4);
                int v608;
                v608 = 4 * v604;
                int v609;
                v609 = v608 + v606;
                float v610;
                v610 = v508[v609];
                bool v611;
                v611 = v542[v609];
                float v612;
                if (v611){
                    v612 = v610;
                } else {
                    v612 = -1.0f / 0.0f;
                }
                float v613;
                v613 = v612 - v602;
                float v614;
                v614 = exp(v613);
                assert("Tensor range check" && 0 <= v604 && v604 < 1);
                assert("Tensor range check" && 0 <= v606 && v606 < 4);
                v603[v609] = v614;
                v606 += 1 ;
            }
            v604 += 1 ;
        }
        float v615;
        v615 = 0.0f;
        int v616;
        v616 = 0;
        while (while_method_3(v616)){
            int v618;
            v618 = 0;
            while (while_method_1(v618)){
                assert("Tensor range check" && 0 <= v616 && v616 < 1);
                assert("Tensor range check" && 0 <= v618 && v618 < 4);
                int v620;
                v620 = 4 * v616;
                int v621;
                v621 = v620 + v618;
                float v622;
                v622 = v603[v621];
                float v623;
                v623 = v615 + v622;
                v615 = v623;
                v618 += 1 ;
            }
            v616 += 1 ;
        }
        auto v624 = cooperative_groups::coalesced_threads();
        int v625;
        v625 = threadIdx.x;
        int v626;
        v626 = v625 / 4;
        auto v627 = cooperative_groups::labeled_partition(v624,v626);
        float v628;
        v628 = cooperative_groups::reduce(v627, v615, v599);
        float v629[4];
        int v630;
        v630 = 0;
        while (while_method_3(v630)){
            int v632;
            v632 = 0;
            while (while_method_1(v632)){
                assert("Tensor range check" && 0 <= v630 && v630 < 1);
                assert("Tensor range check" && 0 <= v632 && v632 < 4);
                int v634;
                v634 = 4 * v630;
                int v635;
                v635 = v634 + v632;
                float v636;
                v636 = v603[v635];
                float v637;
                v637 = v636 / v628;
                assert("Tensor range check" && 0 <= v630 && v630 < 1);
                assert("Tensor range check" && 0 <= v632 && v632 < 4);
                v629[v635] = v637;
                v632 += 1 ;
            }
            v630 += 1 ;
        }
        float v638[4];
        float v639;
        v639 = 0.0f;
        int v640;
        v640 = 0;
        while (while_method_3(v640)){
            assert("Tensor range check" && 0 <= v640 && v640 < 1);
            int v642;
            v642 = 4 * v640;
            assert("Tensor range check" && 0 <= v640 && v640 < 1);
            int v643; float v644;
            Tuple0 tmp46 = Tuple0{0, 0.0f};
            v643 = tmp46.v0; v644 = tmp46.v1;
            while (while_method_1(v643)){
                assert("Tensor range check" && 0 <= v643 && v643 < 4);
                int v646;
                v646 = v643 + v642;
                float v647;
                v647 = v629[v646];
                float v648;
                v648 = v644 + v647;
                v644 = v648;
                v643 += 1 ;
            }
            auto v649 = cooperative_groups::coalesced_threads();
            int v650;
            v650 = threadIdx.x;
            int v651;
            v651 = v650 / 4;
            auto v652 = cooperative_groups::labeled_partition(v649,v651);
            Closure2 v653{};
            float v654;
            v654 = cooperative_groups::inclusive_scan(v652, v644, v653);
            float v655;
            v655 = v652.shfl_up(v654,1);
            bool v656;
            v656 = v652.thread_rank() == 0;
            float v657;
            if (v656){
                v657 = 0.0f;
            } else {
                v657 = v655;
            }
            float v658;
            v658 = v652.shfl(v654,v652.num_threads()-1);
            float v659;
            v659 = v639 + v657;
            int v660; float v661;
            Tuple0 tmp47 = Tuple0{0, v659};
            v660 = tmp47.v0; v661 = tmp47.v1;
            while (while_method_1(v660)){
                assert("Tensor range check" && 0 <= v660 && v660 < 4);
                int v663;
                v663 = v660 + v642;
                float v664;
                v664 = v629[v663];
                float v665;
                v665 = v661 + v664;
                assert("Tensor range check" && 0 <= v660 && v660 < 4);
                v638[v663] = v665;
                v661 = v665;
                v660 += 1 ;
            }
            float v666;
            v666 = v639 + v658;
            v639 = v666;
            v640 += 1 ;
        }
        float v667[4];
        bool v668[4];
        int v669;
        v669 = 0;
        while (while_method_3(v669)){
            int v671;
            v671 = 0;
            while (while_method_1(v671)){
                assert("Tensor range check" && 0 <= v669 && v669 < 1);
                assert("Tensor range check" && 0 <= v671 && v671 < 4);
                int v673;
                v673 = 4 * v669;
                int v674;
                v674 = v673 + v671;
                float v675;
                v675 = v638[v674];
                float v676;
                v676 = v629[v674];
                bool v677;
                v677 = v676 > 0.0f;
                assert("Tensor range check" && 0 <= v669 && v669 < 1);
                assert("Tensor range check" && 0 <= v671 && v671 < 4);
                v667[v674] = v675;
                v668[v674] = v677;
                v671 += 1 ;
            }
            v669 += 1 ;
        }
        float v678; bool v679;
        Tuple3 tmp48 = Tuple3{-1.0f / 0.0f, false};
        v678 = tmp48.v0; v679 = tmp48.v1;
        int v680;
        v680 = 0;
        while (while_method_3(v680)){
            int v682;
            v682 = 0;
            while (while_method_1(v682)){
                assert("Tensor range check" && 0 <= v680 && v680 < 1);
                assert("Tensor range check" && 0 <= v682 && v682 < 4);
                int v684;
                v684 = 4 * v680;
                int v685;
                v685 = v684 + v682;
                float v686;
                v686 = v667[v685];
                bool v687;
                v687 = v668[v685];
                float v694; bool v695;
                if (v679){
                    if (v687){
                        bool v688;
                        v688 = v678 >= v686;
                        float v689;
                        if (v688){
                            v689 = v678;
                        } else {
                            v689 = v686;
                        }
                        v694 = v689; v695 = true;
                    } else {
                        v694 = v678; v695 = v679;
                    }
                } else {
                    if (v687){
                        v694 = v686; v695 = v687;
                    } else {
                        v694 = v678; v695 = v679;
                    }
                }
                v678 = v694;
                v679 = v695;
                v682 += 1 ;
            }
            v680 += 1 ;
        }
        auto v696 = cooperative_groups::coalesced_threads();
        int v697;
        v697 = threadIdx.x;
        int v698;
        v698 = v697 / 4;
        auto v699 = cooperative_groups::labeled_partition(v696,v698);
        Closure5 v700{};
        float v701; bool v702;
        Tuple3 tmp49 = cooperative_groups::reduce(v699, Tuple3{v678, v679}, v700);
        v701 = tmp49.v0; v702 = tmp49.v1;
        bool v703;
        v703 = v702 == false;
        if (v703){
            assert("The local reduce must be true." && v702);
        } else {
        }
        float v705[4];
        int v706[4];
        int v707;
        v707 = 0;
        while (while_method_3(v707)){
            int v709;
            v709 = 0;
            while (while_method_1(v709)){
                assert("Tensor range check" && 0 <= v707 && v707 < 1);
                assert("Tensor range check" && 0 <= v709 && v709 < 4);
                int v711;
                v711 = 4 * v707;
                int v712;
                v712 = v711 + v709;
                int v713;
                v713 = v509[v712];
                float v714;
                v714 = curand_uniform(&v469);
                assert("Tensor range check" && 0 <= v707 && v707 < 1);
                assert("Tensor range check" && 0 <= v709 && v709 < 4);
                v705[v712] = v714;
                v706[v712] = v713;
                v709 += 1 ;
            }
            v707 += 1 ;
        }
        float v715; int v716;
        Tuple1 tmp50 = Tuple1{0.0f, 2147483647};
        v715 = tmp50.v0; v716 = tmp50.v1;
        int v717;
        v717 = 0;
        while (while_method_3(v717)){
            int v719;
            v719 = 0;
            while (while_method_1(v719)){
                assert("Tensor range check" && 0 <= v717 && v717 < 1);
                assert("Tensor range check" && 0 <= v719 && v719 < 4);
                int v721;
                v721 = 4 * v717;
                int v722;
                v722 = v721 + v719;
                float v723;
                v723 = v705[v722];
                int v724;
                v724 = v706[v722];
                bool v725;
                v725 = v716 < v724;
                float v726; int v727;
                if (v725){
                    v726 = v715; v727 = v716;
                } else {
                    v726 = v723; v727 = v724;
                }
                v715 = v726;
                v716 = v727;
                v719 += 1 ;
            }
            v717 += 1 ;
        }
        auto v728 = cooperative_groups::coalesced_threads();
        int v729;
        v729 = threadIdx.x;
        int v730;
        v730 = v729 / 4;
        auto v731 = cooperative_groups::labeled_partition(v728,v730);
        Closure6 v732{};
        float v733; int v734;
        Tuple1 tmp51 = cooperative_groups::reduce(v731, Tuple1{v715, v716}, v732);
        v733 = tmp51.v0; v734 = tmp51.v1;
        float v735;
        v735 = v701 * v733;
        int v736[4];
        bool v737[4];
        int v738;
        v738 = 0;
        while (while_method_3(v738)){
            int v740;
            v740 = 0;
            while (while_method_1(v740)){
                assert("Tensor range check" && 0 <= v738 && v738 < 1);
                assert("Tensor range check" && 0 <= v740 && v740 < 4);
                int v742;
                v742 = 4 * v738;
                int v743;
                v743 = v742 + v740;
                float v744;
                v744 = v667[v743];
                bool v745;
                v745 = v668[v743];
                int v746;
                v746 = v509[v743];
                int v749; bool v750;
                if (v745){
                    float v747;
                    v747 = v744 - v735;
                    bool v748;
                    v748 = v747 >= 0.0f;
                    v749 = v746; v750 = v748;
                } else {
                    v749 = 2147483647; v750 = false;
                }
                assert("Tensor range check" && 0 <= v738 && v738 < 1);
                assert("Tensor range check" && 0 <= v740 && v740 < 4);
                v736[v743] = v749;
                v737[v743] = v750;
                v740 += 1 ;
            }
            v738 += 1 ;
        }
        int v751; bool v752;
        Tuple4 tmp52 = Tuple4{2147483647, false};
        v751 = tmp52.v0; v752 = tmp52.v1;
        int v753;
        v753 = 0;
        while (while_method_3(v753)){
            int v755;
            v755 = 0;
            while (while_method_1(v755)){
                assert("Tensor range check" && 0 <= v753 && v753 < 1);
                assert("Tensor range check" && 0 <= v755 && v755 < 4);
                int v757;
                v757 = 4 * v753;
                int v758;
                v758 = v757 + v755;
                int v759;
                v759 = v736[v758];
                bool v760;
                v760 = v737[v758];
                int v767; bool v768;
                if (v752){
                    if (v760){
                        bool v761;
                        v761 = v751 < v759;
                        int v762;
                        if (v761){
                            v762 = v751;
                        } else {
                            v762 = v759;
                        }
                        v767 = v762; v768 = true;
                    } else {
                        v767 = v751; v768 = v752;
                    }
                } else {
                    if (v760){
                        v767 = v759; v768 = v760;
                    } else {
                        v767 = v751; v768 = v752;
                    }
                }
                v751 = v767;
                v752 = v768;
                v755 += 1 ;
            }
            v753 += 1 ;
        }
        auto v769 = cooperative_groups::coalesced_threads();
        int v770;
        v770 = threadIdx.x;
        int v771;
        v771 = v770 / 4;
        auto v772 = cooperative_groups::labeled_partition(v769,v771);
        Closure7 v773{};
        int v774; bool v775;
        Tuple4 tmp53 = cooperative_groups::reduce(v772, Tuple4{v751, v752}, v773);
        v774 = tmp53.v0; v775 = tmp53.v1;
        bool v776;
        v776 = v775 == false;
        if (v776){
            assert("The local reduce must be true." && v775);
        } else {
        }
        int v778;
        v778 = 0;
        while (while_method_3(v778)){
            assert("Tensor range check" && 0 <= v778 && v778 < 1);
            assert("Tensor range check" && 0 <= v778 && v778 < 1);
            v778 += 1 ;
        }
        assert("Tensor range check" && 0 <= v500 && v500 < 256);
        v477[v500] = v774;
        v488 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v479 && v479 < 256);
    int v780;
    v780 = v477[v479];
    __syncthreads();
    int v781;
    v781 = threadIdx.x;
    assert("Tensor range check" && 0 <= v781 && v781 < 256);
    v5[v781] = v780;
    return ;
}
extern "C" __global__ void entry3(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 256);
    int v9;
    v9 = 256 * v8;
    int v10;
    v10 = threadIdx.x;
    assert("Tensor range check" && 0 <= v10 && v10 < 256);
    int v11;
    v11 = 256 * v10;
    int v12;
    v12 = threadIdx.x;
    assert("Tensor range check" && 0 <= v12 && v12 < 256);
    int v13;
    v13 = 256 * v12;
    int v14;
    v14 = threadIdx.x;
    assert("Tensor range check" && 0 <= v14 && v14 < 256);
    int v15;
    v15 = 256 * v14;
    int v16;
    v16 = threadIdx.x;
    assert("Tensor range check" && 0 <= v16 && v16 < 256);
    int v17;
    v17 = 256 * v16;
    float * v18;
    v18 = v1+v9;
    int * v20;
    v20 = v2+v15;
    int * v22;
    v22 = v3+v15;
    int v24;
    v24 = sizeof(float *);
    unsigned long long v25;
    v25 = (unsigned long long)v24;
    unsigned long long v26;
    v26 = 256ull * v25;
    unsigned long long v27;
    v27 = v26 + 16ull;
    unsigned long long v28;
    v28 = v27 - 1ull;
    unsigned long long v29;
    v29 = v28 % 16ull;
    unsigned long long v30;
    v30 = v28 - v29;
    int v31;
    v31 = sizeof(int *);
    unsigned long long v32;
    v32 = (unsigned long long)v31;
    unsigned long long v33;
    v33 = 256ull * v32;
    unsigned long long v34;
    v34 = v30 + v33;
    unsigned long long v35;
    v35 = v34 + 16ull;
    unsigned long long v36;
    v36 = v35 - 1ull;
    unsigned long long v37;
    v37 = v36 % 16ull;
    unsigned long long v38;
    v38 = v36 - v37;
    unsigned long long v39;
    v39 = v38 + v33;
    bool v40;
    v40 = v39 <= 98304ull;
    bool v41;
    v41 = v40 == false;
    if (v41){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v40);
    } else {
    }
    extern __shared__ unsigned char v43[];
    bool v44;
    v44 = v39 <= v39;
    bool v45;
    v45 = v44 == false;
    if (v45){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v44);
    } else {
    }
    float * * v47;
    v47 = reinterpret_cast<float * *>(&v43[0ull]);
    int * * v49;
    v49 = reinterpret_cast<int * *>(&v43[v30]);
    int * * v51;
    v51 = reinterpret_cast<int * *>(&v43[v38]);
    int v53;
    v53 = threadIdx.x;
    assert("Tensor range check" && 0 <= v53 && v53 < 256);
    v47[v53] = v18;
    v49[v53] = v20;
    v51[v53] = v22;
    __syncthreads();
    bool v54;
    v54 = 0 <= v53;
    bool v55;
    v55 = v54 == false;
    if (v55){
        assert("The index needs to be zero or positive." && v54);
    } else {
    }
    int v57;
    v57 = v53 % 64;
    int v58;
    v58 = v53 / 64;
    bool v59;
    v59 = v58 < 4;
    bool v60;
    v60 = v59 == false;
    if (v60){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v59);
    } else {
    }
    assert("Tensor range check" && 0 <= v58 && v58 < 4);
    int v62;
    v62 = 0;
    while (while_method_4(v62)){
        bool v64;
        v64 = 0 <= v58;
        bool v65;
        v65 = v64 && v59;
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v65);
        } else {
        }
        bool v68;
        v68 = 0 <= v62;
        bool v70;
        if (v68){
            bool v69;
            v69 = v62 < 64;
            v70 = v69;
        } else {
            v70 = false;
        }
        bool v71;
        v71 = v70 == false;
        if (v71){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v70);
        } else {
        }
        int v73;
        v73 = v62 * 4;
        int v74;
        v74 = v73 + v58;
        assert("Tensor range check" && 0 <= v62 && v62 < 64);
        int v75;
        v75 = 4 * v62;
        int v76;
        v76 = v75 + v58;
        float * v77;
        v77 = v47[v76];
        int * v78;
        v78 = v49[v76];
        int * v79;
        v79 = v51[v76];
        int v80;
        v80 = blockIdx.x;
        int v81;
        v81 = v80 * 256;
        int v82;
        v82 = v81 + v74;
        assert("Tensor range check" && 0 <= v57 && v57 < 64);
        int v83;
        v83 = 4 * v57;
        float v84[4];
        int v85[4];
        int v86;
        v86 = 0;
        while (while_method_3(v86)){
            assert("Tensor range check" && 0 <= v86 && v86 < 1);
            int v88;
            v88 = 4 * v86;
            assert("Tensor range check" && 0 <= v86 && v86 < 1);
            int v89;
            v89 = 256 * v86;
            int v90;
            v90 = v89 + v83;
            int4* v91;
            v91 = reinterpret_cast<int4*>(v77 + v90);
            int4* v92;
            v92 = reinterpret_cast<int4*>(v84 + v88);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v91) % 16 == 0 && reinterpret_cast<unsigned long long>(v92) % 16 == 0);
            *v92 = *v91;
            v86 += 1 ;
        }
        int v93;
        v93 = 0;
        while (while_method_3(v93)){
            int v95;
            v95 = 0;
            while (while_method_1(v95)){
                bool v97;
                v97 = 0 <= v95;
                bool v99;
                if (v97){
                    bool v98;
                    v98 = v95 < 4;
                    v99 = v98;
                } else {
                    v99 = false;
                }
                bool v100;
                v100 = v99 == false;
                if (v100){
                    assert("The indices should be inside the range of the dimension." && v99);
                } else {
                }
                bool v102;
                v102 = 0 <= v57;
                bool v104;
                if (v102){
                    bool v103;
                    v103 = v57 < 64;
                    v104 = v103;
                } else {
                    v104 = false;
                }
                bool v105;
                v105 = v104 == false;
                if (v105){
                    assert("The indices should be inside the range of the dimension." && v104);
                } else {
                }
                int v107;
                v107 = v57 * 4;
                int v108;
                v108 = v95 + v107;
                bool v109;
                v109 = 0 <= v93;
                bool v111;
                if (v109){
                    bool v110;
                    v110 = v93 < 1;
                    v111 = v110;
                } else {
                    v111 = false;
                }
                bool v112;
                v112 = v111 == false;
                if (v112){
                    assert("The indices should be inside the range of the dimension." && v111);
                } else {
                }
                int v114;
                v114 = v93 * 256;
                int v115;
                v115 = v108 + v114;
                assert("Tensor range check" && 0 <= v93 && v93 < 1);
                assert("Tensor range check" && 0 <= v95 && v95 < 4);
                int v116;
                v116 = 4 * v93;
                int v117;
                v117 = v116 + v95;
                v85[v117] = v115;
                v95 += 1 ;
            }
            v93 += 1 ;
        }
        int v118[4];
        int v119[4];
        int v120;
        v120 = 0;
        while (while_method_3(v120)){
            int v122;
            v122 = 0;
            while (while_method_1(v122)){
                assert("Tensor range check" && 0 <= v120 && v120 < 1);
                assert("Tensor range check" && 0 <= v122 && v122 < 4);
                int v124;
                v124 = 4 * v120;
                int v125;
                v125 = v124 + v122;
                int v126;
                v126 = v85[v125];
                assert("Tensor range check" && 0 <= v120 && v120 < 1);
                assert("Tensor range check" && 0 <= v122 && v122 < 4);
                v118[v125] = v82;
                v119[v125] = v126;
                v122 += 1 ;
            }
            v120 += 1 ;
        }
        int v127;
        v127 = 0;
        while (while_method_3(v127)){
            assert("Tensor range check" && 0 <= v127 && v127 < 1);
            int v129;
            v129 = 256 * v127;
            int v130;
            v130 = v129 + v83;
            assert("Tensor range check" && 0 <= v127 && v127 < 1);
            int v131;
            v131 = 4 * v127;
            int4* v132;
            v132 = reinterpret_cast<int4*>(v118 + v131);
            int4* v133;
            v133 = reinterpret_cast<int4*>(v78 + v130);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v132) % 16 == 0 && reinterpret_cast<unsigned long long>(v133) % 16 == 0);
            *v133 = *v132;
            int4* v134;
            v134 = reinterpret_cast<int4*>(v119 + v131);
            int4* v135;
            v135 = reinterpret_cast<int4*>(v79 + v130);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v134) % 16 == 0 && reinterpret_cast<unsigned long long>(v135) % 16 == 0);
            *v135 = *v134;
            v127 += 1 ;
        }
        assert("Tensor range check" && 0 <= v74 && v74 < 256);
        v62 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v53 && v53 < 256);
    __syncthreads();
    float * v136;
    v136 = v1+v9;
    unsigned long long v138;
    v138 = v30 + 1024ull;
    bool v139;
    v139 = v138 <= 98304ull;
    bool v140;
    v140 = v139 == false;
    if (v140){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v139);
    } else {
    }
    extern __shared__ unsigned char v142[];
    bool v143;
    v143 = v138 <= v138;
    bool v144;
    v144 = v143 == false;
    if (v144){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v143);
    } else {
    }
    float * * v146;
    v146 = reinterpret_cast<float * *>(&v142[0ull]);
    int * v148;
    v148 = reinterpret_cast<int *>(&v142[v30]);
    int v150;
    v150 = threadIdx.x;
    assert("Tensor range check" && 0 <= v150 && v150 < 256);
    v146[v150] = v136;
    __syncthreads();
    bool v151;
    v151 = 0 <= v150;
    bool v152;
    v152 = v151 == false;
    if (v152){
        assert("The index needs to be zero or positive." && v151);
    } else {
    }
    int v154;
    v154 = v150 % 64;
    int v155;
    v155 = v150 / 64;
    bool v156;
    v156 = v155 < 4;
    bool v157;
    v157 = v156 == false;
    if (v157){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v156);
    } else {
    }
    assert("Tensor range check" && 0 <= v155 && v155 < 4);
    int v159;
    v159 = 0;
    while (while_method_4(v159)){
        bool v161;
        v161 = 0 <= v155;
        bool v162;
        v162 = v161 && v156;
        bool v163;
        v163 = v162 == false;
        if (v163){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v162);
        } else {
        }
        bool v165;
        v165 = 0 <= v159;
        bool v167;
        if (v165){
            bool v166;
            v166 = v159 < 64;
            v167 = v166;
        } else {
            v167 = false;
        }
        bool v168;
        v168 = v167 == false;
        if (v168){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v167);
        } else {
        }
        int v170;
        v170 = v159 * 4;
        int v171;
        v171 = v170 + v155;
        assert("Tensor range check" && 0 <= v159 && v159 < 64);
        int v172;
        v172 = 4 * v159;
        int v173;
        v173 = v172 + v155;
        float * v174;
        v174 = v146[v173];
        int v175;
        v175 = blockIdx.x;
        int v176;
        v176 = v175 * 256;
        int v177;
        v177 = v176 + v171;
        assert("Tensor range check" && 0 <= v154 && v154 < 64);
        int v178;
        v178 = 4 * v154;
        float v179[4];
        int v180[4];
        int v181;
        v181 = 0;
        while (while_method_3(v181)){
            assert("Tensor range check" && 0 <= v181 && v181 < 1);
            int v183;
            v183 = 4 * v181;
            assert("Tensor range check" && 0 <= v181 && v181 < 1);
            int v184;
            v184 = 256 * v181;
            int v185;
            v185 = v184 + v178;
            int4* v186;
            v186 = reinterpret_cast<int4*>(v174 + v185);
            int4* v187;
            v187 = reinterpret_cast<int4*>(v179 + v183);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v186) % 16 == 0 && reinterpret_cast<unsigned long long>(v187) % 16 == 0);
            *v187 = *v186;
            v181 += 1 ;
        }
        int v188;
        v188 = 0;
        while (while_method_3(v188)){
            int v190;
            v190 = 0;
            while (while_method_1(v190)){
                bool v192;
                v192 = 0 <= v190;
                bool v194;
                if (v192){
                    bool v193;
                    v193 = v190 < 4;
                    v194 = v193;
                } else {
                    v194 = false;
                }
                bool v195;
                v195 = v194 == false;
                if (v195){
                    assert("The indices should be inside the range of the dimension." && v194);
                } else {
                }
                bool v197;
                v197 = 0 <= v154;
                bool v199;
                if (v197){
                    bool v198;
                    v198 = v154 < 64;
                    v199 = v198;
                } else {
                    v199 = false;
                }
                bool v200;
                v200 = v199 == false;
                if (v200){
                    assert("The indices should be inside the range of the dimension." && v199);
                } else {
                }
                int v202;
                v202 = v154 * 4;
                int v203;
                v203 = v190 + v202;
                bool v204;
                v204 = 0 <= v188;
                bool v206;
                if (v204){
                    bool v205;
                    v205 = v188 < 1;
                    v206 = v205;
                } else {
                    v206 = false;
                }
                bool v207;
                v207 = v206 == false;
                if (v207){
                    assert("The indices should be inside the range of the dimension." && v206);
                } else {
                }
                int v209;
                v209 = v188 * 256;
                int v210;
                v210 = v203 + v209;
                assert("Tensor range check" && 0 <= v188 && v188 < 1);
                assert("Tensor range check" && 0 <= v190 && v190 < 4);
                int v211;
                v211 = 4 * v188;
                int v212;
                v212 = v211 + v190;
                v180[v212] = v210;
                v190 += 1 ;
            }
            v188 += 1 ;
        }
        int v213;
        v213 = 0;
        while (while_method_3(v213)){
            assert("Tensor range check" && 0 <= v213 && v213 < 1);
            assert("Tensor range check" && 0 <= v213 && v213 < 1);
            v213 += 1 ;
        }
        assert("Tensor range check" && 0 <= v171 && v171 < 256);
        v148[v171] = v177;
        v159 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v150 && v150 < 256);
    int v215;
    v215 = v148[v150];
    __syncthreads();
    int v216;
    v216 = threadIdx.x;
    assert("Tensor range check" && 0 <= v216 && v216 < 256);
    v4[v216] = v215;
    float * v217;
    v217 = v1+v9;
    float * v219;
    v219 = v6+v17;
    unsigned long long v221;
    v221 = v30 + v26;
    bool v222;
    v222 = v221 <= 98304ull;
    bool v223;
    v223 = v222 == false;
    if (v223){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v222);
    } else {
    }
    extern __shared__ unsigned char v225[];
    bool v226;
    v226 = v221 <= v221;
    bool v227;
    v227 = v226 == false;
    if (v227){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v226);
    } else {
    }
    float * * v229;
    v229 = reinterpret_cast<float * *>(&v225[0ull]);
    float * * v231;
    v231 = reinterpret_cast<float * *>(&v225[v30]);
    int v233;
    v233 = threadIdx.x;
    assert("Tensor range check" && 0 <= v233 && v233 < 256);
    v229[v233] = v217;
    v231[v233] = v219;
    __syncthreads();
    bool v234;
    v234 = 0 <= v233;
    bool v235;
    v235 = v234 == false;
    if (v235){
        assert("The index needs to be zero or positive." && v234);
    } else {
    }
    int v237;
    v237 = v233 % 64;
    int v238;
    v238 = v233 / 64;
    bool v239;
    v239 = v238 < 4;
    bool v240;
    v240 = v239 == false;
    if (v240){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v239);
    } else {
    }
    assert("Tensor range check" && 0 <= v238 && v238 < 4);
    int v242;
    v242 = 0;
    while (while_method_4(v242)){
        bool v244;
        v244 = 0 <= v238;
        bool v245;
        v245 = v244 && v239;
        bool v246;
        v246 = v245 == false;
        if (v246){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v245);
        } else {
        }
        bool v248;
        v248 = 0 <= v242;
        bool v250;
        if (v248){
            bool v249;
            v249 = v242 < 64;
            v250 = v249;
        } else {
            v250 = false;
        }
        bool v251;
        v251 = v250 == false;
        if (v251){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v250);
        } else {
        }
        int v253;
        v253 = v242 * 4;
        int v254;
        v254 = v253 + v238;
        assert("Tensor range check" && 0 <= v242 && v242 < 64);
        int v255;
        v255 = 4 * v242;
        int v256;
        v256 = v255 + v238;
        float * v257;
        v257 = v229[v256];
        float * v258;
        v258 = v231[v256];
        int v259;
        v259 = blockIdx.x;
        int v260;
        v260 = v259 * 256;
        int v261;
        v261 = v260 + v254;
        assert("Tensor range check" && 0 <= v237 && v237 < 64);
        int v262;
        v262 = 4 * v237;
        float v263[4];
        int v264[4];
        int v265;
        v265 = 0;
        while (while_method_3(v265)){
            assert("Tensor range check" && 0 <= v265 && v265 < 1);
            int v267;
            v267 = 4 * v265;
            assert("Tensor range check" && 0 <= v265 && v265 < 1);
            int v268;
            v268 = 256 * v265;
            int v269;
            v269 = v268 + v262;
            int4* v270;
            v270 = reinterpret_cast<int4*>(v257 + v269);
            int4* v271;
            v271 = reinterpret_cast<int4*>(v263 + v267);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v270) % 16 == 0 && reinterpret_cast<unsigned long long>(v271) % 16 == 0);
            *v271 = *v270;
            v265 += 1 ;
        }
        int v272;
        v272 = 0;
        while (while_method_3(v272)){
            int v274;
            v274 = 0;
            while (while_method_1(v274)){
                bool v276;
                v276 = 0 <= v274;
                bool v278;
                if (v276){
                    bool v277;
                    v277 = v274 < 4;
                    v278 = v277;
                } else {
                    v278 = false;
                }
                bool v279;
                v279 = v278 == false;
                if (v279){
                    assert("The indices should be inside the range of the dimension." && v278);
                } else {
                }
                bool v281;
                v281 = 0 <= v237;
                bool v283;
                if (v281){
                    bool v282;
                    v282 = v237 < 64;
                    v283 = v282;
                } else {
                    v283 = false;
                }
                bool v284;
                v284 = v283 == false;
                if (v284){
                    assert("The indices should be inside the range of the dimension." && v283);
                } else {
                }
                int v286;
                v286 = v237 * 4;
                int v287;
                v287 = v274 + v286;
                bool v288;
                v288 = 0 <= v272;
                bool v290;
                if (v288){
                    bool v289;
                    v289 = v272 < 1;
                    v290 = v289;
                } else {
                    v290 = false;
                }
                bool v291;
                v291 = v290 == false;
                if (v291){
                    assert("The indices should be inside the range of the dimension." && v290);
                } else {
                }
                int v293;
                v293 = v272 * 256;
                int v294;
                v294 = v287 + v293;
                assert("Tensor range check" && 0 <= v272 && v272 < 1);
                assert("Tensor range check" && 0 <= v274 && v274 < 4);
                int v295;
                v295 = 4 * v272;
                int v296;
                v296 = v295 + v274;
                v264[v296] = v294;
                v274 += 1 ;
            }
            v272 += 1 ;
        }
        int v297;
        v297 = 0;
        while (while_method_3(v297)){
            assert("Tensor range check" && 0 <= v297 && v297 < 1);
            int v299;
            v299 = 256 * v297;
            int v300;
            v300 = v299 + v262;
            assert("Tensor range check" && 0 <= v297 && v297 < 1);
            int v301;
            v301 = 4 * v297;
            int4* v302;
            v302 = reinterpret_cast<int4*>(v263 + v301);
            int4* v303;
            v303 = reinterpret_cast<int4*>(v258 + v300);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v302) % 16 == 0 && reinterpret_cast<unsigned long long>(v303) % 16 == 0);
            *v303 = *v302;
            v297 += 1 ;
        }
        assert("Tensor range check" && 0 <= v254 && v254 < 256);
        v242 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v233 && v233 < 256);
    __syncthreads();
    float * v304;
    v304 = v1+v9;
    float * v306;
    v306 = v7+v13;
    if (v223){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v222);
    } else {
    }
    extern __shared__ unsigned char v309[];
    if (v227){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v226);
    } else {
    }
    float * * v311;
    v311 = reinterpret_cast<float * *>(&v309[0ull]);
    float * * v313;
    v313 = reinterpret_cast<float * *>(&v309[v30]);
    int v315;
    v315 = threadIdx.x;
    assert("Tensor range check" && 0 <= v315 && v315 < 256);
    v311[v315] = v304;
    v313[v315] = v306;
    __syncthreads();
    bool v316;
    v316 = 0 <= v315;
    bool v317;
    v317 = v316 == false;
    if (v317){
        assert("The index needs to be zero or positive." && v316);
    } else {
    }
    int v319;
    v319 = v315 % 64;
    int v320;
    v320 = v315 / 64;
    bool v321;
    v321 = v320 < 4;
    bool v322;
    v322 = v321 == false;
    if (v322){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v321);
    } else {
    }
    assert("Tensor range check" && 0 <= v320 && v320 < 4);
    int v324;
    v324 = 0;
    while (while_method_4(v324)){
        bool v326;
        v326 = 0 <= v320;
        bool v327;
        v327 = v326 && v321;
        bool v328;
        v328 = v327 == false;
        if (v328){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v327);
        } else {
        }
        bool v330;
        v330 = 0 <= v324;
        bool v332;
        if (v330){
            bool v331;
            v331 = v324 < 64;
            v332 = v331;
        } else {
            v332 = false;
        }
        bool v333;
        v333 = v332 == false;
        if (v333){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v332);
        } else {
        }
        int v335;
        v335 = v324 * 4;
        int v336;
        v336 = v335 + v320;
        assert("Tensor range check" && 0 <= v324 && v324 < 64);
        int v337;
        v337 = 4 * v324;
        int v338;
        v338 = v337 + v320;
        float * v339;
        v339 = v311[v338];
        float * v340;
        v340 = v313[v338];
        int v341;
        v341 = blockIdx.x;
        int v342;
        v342 = v341 * 256;
        int v343;
        v343 = v342 + v336;
        assert("Tensor range check" && 0 <= v319 && v319 < 64);
        int v344;
        v344 = 4 * v319;
        float v345[4];
        int v346[4];
        int v347;
        v347 = 0;
        while (while_method_3(v347)){
            assert("Tensor range check" && 0 <= v347 && v347 < 1);
            int v349;
            v349 = 4 * v347;
            assert("Tensor range check" && 0 <= v347 && v347 < 1);
            int v350;
            v350 = 256 * v347;
            int v351;
            v351 = v350 + v344;
            int4* v352;
            v352 = reinterpret_cast<int4*>(v339 + v351);
            int4* v353;
            v353 = reinterpret_cast<int4*>(v345 + v349);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v352) % 16 == 0 && reinterpret_cast<unsigned long long>(v353) % 16 == 0);
            *v353 = *v352;
            v347 += 1 ;
        }
        int v354;
        v354 = 0;
        while (while_method_3(v354)){
            int v356;
            v356 = 0;
            while (while_method_1(v356)){
                bool v358;
                v358 = 0 <= v356;
                bool v360;
                if (v358){
                    bool v359;
                    v359 = v356 < 4;
                    v360 = v359;
                } else {
                    v360 = false;
                }
                bool v361;
                v361 = v360 == false;
                if (v361){
                    assert("The indices should be inside the range of the dimension." && v360);
                } else {
                }
                bool v363;
                v363 = 0 <= v319;
                bool v365;
                if (v363){
                    bool v364;
                    v364 = v319 < 64;
                    v365 = v364;
                } else {
                    v365 = false;
                }
                bool v366;
                v366 = v365 == false;
                if (v366){
                    assert("The indices should be inside the range of the dimension." && v365);
                } else {
                }
                int v368;
                v368 = v319 * 4;
                int v369;
                v369 = v356 + v368;
                bool v370;
                v370 = 0 <= v354;
                bool v372;
                if (v370){
                    bool v371;
                    v371 = v354 < 1;
                    v372 = v371;
                } else {
                    v372 = false;
                }
                bool v373;
                v373 = v372 == false;
                if (v373){
                    assert("The indices should be inside the range of the dimension." && v372);
                } else {
                }
                int v375;
                v375 = v354 * 256;
                int v376;
                v376 = v369 + v375;
                assert("Tensor range check" && 0 <= v354 && v354 < 1);
                assert("Tensor range check" && 0 <= v356 && v356 < 4);
                int v377;
                v377 = 4 * v354;
                int v378;
                v378 = v377 + v356;
                v346[v378] = v376;
                v356 += 1 ;
            }
            v354 += 1 ;
        }
        bool v379[4];
        int v380;
        v380 = 0;
        while (while_method_3(v380)){
            int v382;
            v382 = 0;
            while (while_method_1(v382)){
                assert("Tensor range check" && 0 <= v380 && v380 < 1);
                assert("Tensor range check" && 0 <= v382 && v382 < 4);
                int v384;
                v384 = 4 * v380;
                int v385;
                v385 = v384 + v382;
                float v386;
                v386 = v345[v385];
                int v387;
                v387 = v346[v385];
                bool v388;
                v388 = v387 < 3;
                assert("Tensor range check" && 0 <= v380 && v380 < 1);
                assert("Tensor range check" && 0 <= v382 && v382 < 4);
                v379[v385] = v388;
                v382 += 1 ;
            }
            v380 += 1 ;
        }
        float v389[4];
        int v390;
        v390 = 0;
        while (while_method_3(v390)){
            int v392;
            v392 = 0;
            while (while_method_1(v392)){
                assert("Tensor range check" && 0 <= v390 && v390 < 1);
                assert("Tensor range check" && 0 <= v392 && v392 < 4);
                int v394;
                v394 = 4 * v390;
                int v395;
                v395 = v394 + v392;
                float v396;
                v396 = v345[v395];
                bool v397;
                v397 = v379[v395];
                float v400;
                if (v397){
                    bool v398;
                    v398 = 0.0f >= v396;
                    if (v398){
                        v400 = 0.0f;
                    } else {
                        v400 = v396;
                    }
                } else {
                    v400 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v390 && v390 < 1);
                assert("Tensor range check" && 0 <= v392 && v392 < 4);
                v389[v395] = v400;
                v392 += 1 ;
            }
            v390 += 1 ;
        }
        float v401;
        v401 = 0.0f;
        int v402;
        v402 = 0;
        while (while_method_3(v402)){
            int v404;
            v404 = 0;
            while (while_method_1(v404)){
                assert("Tensor range check" && 0 <= v402 && v402 < 1);
                assert("Tensor range check" && 0 <= v404 && v404 < 4);
                int v406;
                v406 = 4 * v402;
                int v407;
                v407 = v406 + v404;
                float v408;
                v408 = v389[v407];
                float v409;
                v409 = v401 + v408;
                v401 = v409;
                v404 += 1 ;
            }
            v402 += 1 ;
        }
        auto v410 = cooperative_groups::coalesced_threads();
        Closure0 v411{};
        float v412;
        v412 = cooperative_groups::reduce(v410, v401, v411);
        int v413;
        v413 = threadIdx.x;
        int v414;
        v414 = v413 / 32;
        unsigned long long v415;
        v415 = v221 + 16ull;
        unsigned long long v416;
        v416 = v415 - 1ull;
        unsigned long long v417;
        v417 = v416 % 16ull;
        unsigned long long v418;
        v418 = v416 - v417;
        unsigned long long v419;
        v419 = v418 + 32ull;
        bool v420;
        v420 = v419 <= 98304ull;
        bool v421;
        v421 = v420 == false;
        if (v421){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v420);
        } else {
        }
        extern __shared__ unsigned char v423[];
        bool v424;
        v424 = v419 <= v419;
        bool v425;
        v425 = v424 == false;
        if (v425){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v424);
        } else {
        }
        float * v427;
        v427 = reinterpret_cast<float *>(&v423[v418]);
        bool v429;
        v429 = 0 <= v414;
        bool v430;
        v430 = v429 == false;
        if (v430){
            assert("The index needs to be zero or positive." && v429);
        } else {
        }
        int v432;
        v432 = v414 % 2;
        int v433;
        v433 = v414 / 2;
        bool v434;
        v434 = v433 < 4;
        bool v435;
        v435 = v434 == false;
        if (v435){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v434);
        } else {
        }
        assert("Tensor range check" && 0 <= v433 && v433 < 4);
        assert("Tensor range check" && 0 <= v432 && v432 < 2);
        int v437;
        v437 = 2 * v433;
        int v438;
        v438 = v437 + v432;
        v427[v438] = v412;
        int v439;
        v439 = v433 + 1;
        bool v440;
        v440 = v439 < 16;
        bool v441;
        v441 = v440 == false;
        if (v441){
            assert("The barrier_id has to be less than 16." && v440);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v439), "r"(64));
        int v443;
        v443 = threadIdx.x;
        int v444;
        v444 = v443 % 32;
        bool v445;
        v445 = v444 < 2;
        float v448;
        if (v445){
            assert("Tensor range check" && 0 <= v433 && v433 < 4);
            assert("Tensor range check" && 0 <= v444 && v444 < 2);
            int v446;
            v446 = v437 + v444;
            float v447;
            v447 = v427[v446];
            v448 = v447;
        } else {
            v448 = 0.0f;
        }
        __syncthreads();
        float v449;
        v449 = cooperative_groups::reduce(v410, v448, v411);
        int v450[4];
        int v451;
        v451 = 0;
        while (while_method_3(v451)){
            int v453;
            v453 = 0;
            while (while_method_1(v453)){
                assert("Tensor range check" && 0 <= v451 && v451 < 1);
                assert("Tensor range check" && 0 <= v453 && v453 < 4);
                int v455;
                v455 = 4 * v451;
                int v456;
                v456 = v455 + v453;
                bool v457;
                v457 = v379[v456];
                int v458;
                if (v457){
                    v458 = 1;
                } else {
                    v458 = 0;
                }
                assert("Tensor range check" && 0 <= v451 && v451 < 1);
                assert("Tensor range check" && 0 <= v453 && v453 < 4);
                v450[v456] = v458;
                v453 += 1 ;
            }
            v451 += 1 ;
        }
        int v459;
        v459 = 0;
        int v460;
        v460 = 0;
        while (while_method_3(v460)){
            int v462;
            v462 = 0;
            while (while_method_1(v462)){
                assert("Tensor range check" && 0 <= v460 && v460 < 1);
                assert("Tensor range check" && 0 <= v462 && v462 < 4);
                int v464;
                v464 = 4 * v460;
                int v465;
                v465 = v464 + v462;
                int v466;
                v466 = v450[v465];
                int v467;
                v467 = v459 + v466;
                v459 = v467;
                v462 += 1 ;
            }
            v460 += 1 ;
        }
        auto v468 = cooperative_groups::coalesced_threads();
        Closure4 v469{};
        int v470;
        v470 = cooperative_groups::reduce(v468, v459, v469);
        int v471;
        v471 = threadIdx.x;
        int v472;
        v472 = v471 / 32;
        if (v421){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v420);
        } else {
        }
        extern __shared__ unsigned char v474[];
        if (v425){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v424);
        } else {
        }
        int * v476;
        v476 = reinterpret_cast<int *>(&v474[v418]);
        bool v478;
        v478 = 0 <= v472;
        bool v479;
        v479 = v478 == false;
        if (v479){
            assert("The index needs to be zero or positive." && v478);
        } else {
        }
        int v481;
        v481 = v472 % 2;
        int v482;
        v482 = v472 / 2;
        bool v483;
        v483 = v482 < 4;
        bool v484;
        v484 = v483 == false;
        if (v484){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v483);
        } else {
        }
        assert("Tensor range check" && 0 <= v482 && v482 < 4);
        assert("Tensor range check" && 0 <= v481 && v481 < 2);
        int v486;
        v486 = 2 * v482;
        int v487;
        v487 = v486 + v481;
        v476[v487] = v470;
        int v488;
        v488 = v482 + 1;
        bool v489;
        v489 = v488 < 16;
        bool v490;
        v490 = v489 == false;
        if (v490){
            assert("The barrier_id has to be less than 16." && v489);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v488), "r"(64));
        int v492;
        v492 = threadIdx.x;
        int v493;
        v493 = v492 % 32;
        bool v494;
        v494 = v493 < 2;
        int v497;
        if (v494){
            assert("Tensor range check" && 0 <= v482 && v482 < 4);
            assert("Tensor range check" && 0 <= v493 && v493 < 2);
            int v495;
            v495 = v486 + v493;
            int v496;
            v496 = v476[v495];
            v497 = v496;
        } else {
            v497 = 0;
        }
        __syncthreads();
        int v498;
        v498 = cooperative_groups::reduce(v468, v497, v469);
        float v499;
        v499 = (float)v498;
        float v500;
        v500 = 1.0f / v499;
        float v501[4];
        int v502;
        v502 = 0;
        while (while_method_3(v502)){
            int v504;
            v504 = 0;
            while (while_method_1(v504)){
                assert("Tensor range check" && 0 <= v502 && v502 < 1);
                assert("Tensor range check" && 0 <= v504 && v504 < 4);
                int v506;
                v506 = 4 * v502;
                int v507;
                v507 = v506 + v504;
                float v508;
                v508 = v389[v507];
                bool v509;
                v509 = v379[v507];
                bool v510;
                v510 = v509 == false;
                float v515;
                if (v510){
                    v515 = 0.0f;
                } else {
                    bool v511;
                    v511 = v449 == 0.0f;
                    bool v512;
                    v512 = v511 != true;
                    if (v512){
                        float v513;
                        v513 = v508 / v449;
                        v515 = v513;
                    } else {
                        v515 = v500;
                    }
                }
                assert("Tensor range check" && 0 <= v502 && v502 < 1);
                assert("Tensor range check" && 0 <= v504 && v504 < 4);
                v501[v507] = v515;
                v504 += 1 ;
            }
            v502 += 1 ;
        }
        int v516;
        v516 = 0;
        while (while_method_3(v516)){
            assert("Tensor range check" && 0 <= v516 && v516 < 1);
            int v518;
            v518 = 256 * v516;
            int v519;
            v519 = v518 + v344;
            assert("Tensor range check" && 0 <= v516 && v516 < 1);
            int v520;
            v520 = 4 * v516;
            int4* v521;
            v521 = reinterpret_cast<int4*>(v501 + v520);
            int4* v522;
            v522 = reinterpret_cast<int4*>(v340 + v519);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v521) % 16 == 0 && reinterpret_cast<unsigned long long>(v522) % 16 == 0);
            *v522 = *v521;
            v516 += 1 ;
        }
        assert("Tensor range check" && 0 <= v336 && v336 < 256);
        v324 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v315 && v315 < 256);
    __syncthreads();
    int v523;
    v523 = threadIdx.x;
    int v524;
    v524 = blockIdx.x;
    int v525;
    v525 = v524 * 256;
    int v526;
    v526 = v523 + v525;
    unsigned long long v527;
    v527 = (unsigned long long)v526;
    curandStatePhilox4_32_10_t v528;
    curand_init(12344321ull,v527,0ull,&v528);
    float * v529;
    v529 = v1+v9;
    if (v140){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v139);
    } else {
    }
    extern __shared__ unsigned char v532[];
    if (v144){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v143);
    } else {
    }
    float * * v534;
    v534 = reinterpret_cast<float * *>(&v532[0ull]);
    int * v536;
    v536 = reinterpret_cast<int *>(&v532[v30]);
    int v538;
    v538 = threadIdx.x;
    assert("Tensor range check" && 0 <= v538 && v538 < 256);
    v534[v538] = v529;
    __syncthreads();
    bool v539;
    v539 = 0 <= v538;
    bool v540;
    v540 = v539 == false;
    if (v540){
        assert("The index needs to be zero or positive." && v539);
    } else {
    }
    int v542;
    v542 = v538 % 64;
    int v543;
    v543 = v538 / 64;
    bool v544;
    v544 = v543 < 4;
    bool v545;
    v545 = v544 == false;
    if (v545){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v544);
    } else {
    }
    assert("Tensor range check" && 0 <= v543 && v543 < 4);
    int v547;
    v547 = 0;
    while (while_method_4(v547)){
        bool v549;
        v549 = 0 <= v543;
        bool v550;
        v550 = v549 && v544;
        bool v551;
        v551 = v550 == false;
        if (v551){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v550);
        } else {
        }
        bool v553;
        v553 = 0 <= v547;
        bool v555;
        if (v553){
            bool v554;
            v554 = v547 < 64;
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
        v558 = v547 * 4;
        int v559;
        v559 = v558 + v543;
        assert("Tensor range check" && 0 <= v547 && v547 < 64);
        int v560;
        v560 = 4 * v547;
        int v561;
        v561 = v560 + v543;
        float * v562;
        v562 = v534[v561];
        int v563;
        v563 = blockIdx.x;
        int v564;
        v564 = v563 * 256;
        int v565;
        v565 = v564 + v559;
        assert("Tensor range check" && 0 <= v542 && v542 < 64);
        int v566;
        v566 = 4 * v542;
        float v567[4];
        int v568[4];
        int v569;
        v569 = 0;
        while (while_method_3(v569)){
            assert("Tensor range check" && 0 <= v569 && v569 < 1);
            int v571;
            v571 = 4 * v569;
            assert("Tensor range check" && 0 <= v569 && v569 < 1);
            int v572;
            v572 = 256 * v569;
            int v573;
            v573 = v572 + v566;
            int4* v574;
            v574 = reinterpret_cast<int4*>(v562 + v573);
            int4* v575;
            v575 = reinterpret_cast<int4*>(v567 + v571);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v574) % 16 == 0 && reinterpret_cast<unsigned long long>(v575) % 16 == 0);
            *v575 = *v574;
            v569 += 1 ;
        }
        int v576;
        v576 = 0;
        while (while_method_3(v576)){
            int v578;
            v578 = 0;
            while (while_method_1(v578)){
                bool v580;
                v580 = 0 <= v578;
                bool v582;
                if (v580){
                    bool v581;
                    v581 = v578 < 4;
                    v582 = v581;
                } else {
                    v582 = false;
                }
                bool v583;
                v583 = v582 == false;
                if (v583){
                    assert("The indices should be inside the range of the dimension." && v582);
                } else {
                }
                bool v585;
                v585 = 0 <= v542;
                bool v587;
                if (v585){
                    bool v586;
                    v586 = v542 < 64;
                    v587 = v586;
                } else {
                    v587 = false;
                }
                bool v588;
                v588 = v587 == false;
                if (v588){
                    assert("The indices should be inside the range of the dimension." && v587);
                } else {
                }
                int v590;
                v590 = v542 * 4;
                int v591;
                v591 = v578 + v590;
                bool v592;
                v592 = 0 <= v576;
                bool v594;
                if (v592){
                    bool v593;
                    v593 = v576 < 1;
                    v594 = v593;
                } else {
                    v594 = false;
                }
                bool v595;
                v595 = v594 == false;
                if (v595){
                    assert("The indices should be inside the range of the dimension." && v594);
                } else {
                }
                int v597;
                v597 = v576 * 256;
                int v598;
                v598 = v591 + v597;
                assert("Tensor range check" && 0 <= v576 && v576 < 1);
                assert("Tensor range check" && 0 <= v578 && v578 < 4);
                int v599;
                v599 = 4 * v576;
                int v600;
                v600 = v599 + v578;
                v568[v600] = v598;
                v578 += 1 ;
            }
            v576 += 1 ;
        }
        bool v601[4];
        int v602;
        v602 = 0;
        while (while_method_3(v602)){
            int v604;
            v604 = 0;
            while (while_method_1(v604)){
                assert("Tensor range check" && 0 <= v602 && v602 < 1);
                assert("Tensor range check" && 0 <= v604 && v604 < 4);
                int v606;
                v606 = 4 * v602;
                int v607;
                v607 = v606 + v604;
                float v608;
                v608 = v567[v607];
                int v609;
                v609 = v568[v607];
                bool v610;
                v610 = v609 < 3;
                assert("Tensor range check" && 0 <= v602 && v602 < 1);
                assert("Tensor range check" && 0 <= v604 && v604 < 4);
                v601[v607] = v610;
                v604 += 1 ;
            }
            v602 += 1 ;
        }
        int v611[4];
        int v612;
        v612 = 0;
        while (while_method_3(v612)){
            int v614;
            v614 = 0;
            while (while_method_1(v614)){
                assert("Tensor range check" && 0 <= v612 && v612 < 1);
                assert("Tensor range check" && 0 <= v614 && v614 < 4);
                int v616;
                v616 = 4 * v612;
                int v617;
                v617 = v616 + v614;
                bool v618;
                v618 = v601[v617];
                int v619;
                if (v618){
                    v619 = 1;
                } else {
                    v619 = 0;
                }
                assert("Tensor range check" && 0 <= v612 && v612 < 1);
                assert("Tensor range check" && 0 <= v614 && v614 < 4);
                v611[v617] = v619;
                v614 += 1 ;
            }
            v612 += 1 ;
        }
        int v620;
        v620 = 0;
        int v621;
        v621 = 0;
        while (while_method_3(v621)){
            int v623;
            v623 = 0;
            while (while_method_1(v623)){
                assert("Tensor range check" && 0 <= v621 && v621 < 1);
                assert("Tensor range check" && 0 <= v623 && v623 < 4);
                int v625;
                v625 = 4 * v621;
                int v626;
                v626 = v625 + v623;
                int v627;
                v627 = v611[v626];
                int v628;
                v628 = v620 + v627;
                v620 = v628;
                v623 += 1 ;
            }
            v621 += 1 ;
        }
        auto v629 = cooperative_groups::coalesced_threads();
        Closure4 v630{};
        int v631;
        v631 = cooperative_groups::reduce(v629, v620, v630);
        int v632;
        v632 = threadIdx.x;
        int v633;
        v633 = v632 / 32;
        unsigned long long v634;
        v634 = v138 + 16ull;
        unsigned long long v635;
        v635 = v634 - 1ull;
        unsigned long long v636;
        v636 = v635 % 16ull;
        unsigned long long v637;
        v637 = v635 - v636;
        unsigned long long v638;
        v638 = v637 + 32ull;
        bool v639;
        v639 = v638 <= 98304ull;
        bool v640;
        v640 = v639 == false;
        if (v640){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v639);
        } else {
        }
        extern __shared__ unsigned char v642[];
        bool v643;
        v643 = v638 <= v638;
        bool v644;
        v644 = v643 == false;
        if (v644){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v643);
        } else {
        }
        int * v646;
        v646 = reinterpret_cast<int *>(&v642[v637]);
        bool v648;
        v648 = 0 <= v633;
        bool v649;
        v649 = v648 == false;
        if (v649){
            assert("The index needs to be zero or positive." && v648);
        } else {
        }
        int v651;
        v651 = v633 % 2;
        int v652;
        v652 = v633 / 2;
        bool v653;
        v653 = v652 < 4;
        bool v654;
        v654 = v653 == false;
        if (v654){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v653);
        } else {
        }
        assert("Tensor range check" && 0 <= v652 && v652 < 4);
        assert("Tensor range check" && 0 <= v651 && v651 < 2);
        int v656;
        v656 = 2 * v652;
        int v657;
        v657 = v656 + v651;
        v646[v657] = v631;
        int v658;
        v658 = v652 + 1;
        bool v659;
        v659 = v658 < 16;
        bool v660;
        v660 = v659 == false;
        if (v660){
            assert("The barrier_id has to be less than 16." && v659);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v658), "r"(64));
        int v662;
        v662 = threadIdx.x;
        int v663;
        v663 = v662 % 32;
        bool v664;
        v664 = v663 < 2;
        int v667;
        if (v664){
            assert("Tensor range check" && 0 <= v652 && v652 < 4);
            assert("Tensor range check" && 0 <= v663 && v663 < 2);
            int v665;
            v665 = v656 + v663;
            int v666;
            v666 = v646[v665];
            v667 = v666;
        } else {
            v667 = 0;
        }
        __syncthreads();
        int v668;
        v668 = cooperative_groups::reduce(v629, v667, v630);
        float v669[4];
        int v670;
        v670 = 0;
        while (while_method_3(v670)){
            int v672;
            v672 = 0;
            while (while_method_1(v672)){
                assert("Tensor range check" && 0 <= v670 && v670 < 1);
                assert("Tensor range check" && 0 <= v672 && v672 < 4);
                int v674;
                v674 = 4 * v670;
                int v675;
                v675 = v674 + v672;
                float v676;
                v676 = v567[v675];
                bool v677;
                v677 = v601[v675];
                float v678;
                if (v677){
                    v678 = v676;
                } else {
                    v678 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v670 && v670 < 1);
                assert("Tensor range check" && 0 <= v672 && v672 < 4);
                v669[v675] = v678;
                v672 += 1 ;
            }
            v670 += 1 ;
        }
        float v679;
        v679 = 0.0f;
        int v680;
        v680 = 0;
        while (while_method_3(v680)){
            int v682;
            v682 = 0;
            while (while_method_1(v682)){
                assert("Tensor range check" && 0 <= v680 && v680 < 1);
                assert("Tensor range check" && 0 <= v682 && v682 < 4);
                int v684;
                v684 = 4 * v680;
                int v685;
                v685 = v684 + v682;
                float v686;
                v686 = v669[v685];
                float v687;
                v687 = v679 + v686;
                v679 = v687;
                v682 += 1 ;
            }
            v680 += 1 ;
        }
        auto v688 = cooperative_groups::coalesced_threads();
        Closure0 v689{};
        float v690;
        v690 = cooperative_groups::reduce(v688, v679, v689);
        int v691;
        v691 = threadIdx.x;
        int v692;
        v692 = v691 / 32;
        if (v640){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v639);
        } else {
        }
        extern __shared__ unsigned char v694[];
        if (v644){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v643);
        } else {
        }
        float * v696;
        v696 = reinterpret_cast<float *>(&v694[v637]);
        bool v698;
        v698 = 0 <= v692;
        bool v699;
        v699 = v698 == false;
        if (v699){
            assert("The index needs to be zero or positive." && v698);
        } else {
        }
        int v701;
        v701 = v692 % 2;
        int v702;
        v702 = v692 / 2;
        bool v703;
        v703 = v702 < 4;
        bool v704;
        v704 = v703 == false;
        if (v704){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v703);
        } else {
        }
        assert("Tensor range check" && 0 <= v702 && v702 < 4);
        assert("Tensor range check" && 0 <= v701 && v701 < 2);
        int v706;
        v706 = 2 * v702;
        int v707;
        v707 = v706 + v701;
        v696[v707] = v690;
        int v708;
        v708 = v702 + 1;
        bool v709;
        v709 = v708 < 16;
        bool v710;
        v710 = v709 == false;
        if (v710){
            assert("The barrier_id has to be less than 16." && v709);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v708), "r"(64));
        int v712;
        v712 = threadIdx.x;
        int v713;
        v713 = v712 % 32;
        bool v714;
        v714 = v713 < 2;
        float v717;
        if (v714){
            assert("Tensor range check" && 0 <= v702 && v702 < 4);
            assert("Tensor range check" && 0 <= v713 && v713 < 2);
            int v715;
            v715 = v706 + v713;
            float v716;
            v716 = v696[v715];
            v717 = v716;
        } else {
            v717 = 0.0f;
        }
        __syncthreads();
        float v718;
        v718 = cooperative_groups::reduce(v688, v717, v689);
        float v719;
        v719 = (float)v668;
        float v720;
        v720 = v718 / v719;
        float v721[4];
        int v722;
        v722 = 0;
        while (while_method_3(v722)){
            int v724;
            v724 = 0;
            while (while_method_1(v724)){
                assert("Tensor range check" && 0 <= v722 && v722 < 1);
                assert("Tensor range check" && 0 <= v724 && v724 < 4);
                int v726;
                v726 = 4 * v722;
                int v727;
                v727 = v726 + v724;
                float v728;
                v728 = v567[v727];
                bool v729;
                v729 = v601[v727];
                float v730;
                if (v729){
                    v730 = v728;
                } else {
                    v730 = -1.0f / 0.0f;
                }
                float v731;
                v731 = v730 - v720;
                float v732;
                v732 = exp(v731);
                assert("Tensor range check" && 0 <= v722 && v722 < 1);
                assert("Tensor range check" && 0 <= v724 && v724 < 4);
                v721[v727] = v732;
                v724 += 1 ;
            }
            v722 += 1 ;
        }
        float v733;
        v733 = 0.0f;
        int v734;
        v734 = 0;
        while (while_method_3(v734)){
            int v736;
            v736 = 0;
            while (while_method_1(v736)){
                assert("Tensor range check" && 0 <= v734 && v734 < 1);
                assert("Tensor range check" && 0 <= v736 && v736 < 4);
                int v738;
                v738 = 4 * v734;
                int v739;
                v739 = v738 + v736;
                float v740;
                v740 = v721[v739];
                float v741;
                v741 = v733 + v740;
                v733 = v741;
                v736 += 1 ;
            }
            v734 += 1 ;
        }
        auto v742 = cooperative_groups::coalesced_threads();
        float v743;
        v743 = cooperative_groups::reduce(v742, v733, v689);
        int v744;
        v744 = threadIdx.x;
        int v745;
        v745 = v744 / 32;
        if (v640){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v639);
        } else {
        }
        extern __shared__ unsigned char v747[];
        if (v644){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v643);
        } else {
        }
        float * v749;
        v749 = reinterpret_cast<float *>(&v747[v637]);
        bool v751;
        v751 = 0 <= v745;
        bool v752;
        v752 = v751 == false;
        if (v752){
            assert("The index needs to be zero or positive." && v751);
        } else {
        }
        int v754;
        v754 = v745 % 2;
        int v755;
        v755 = v745 / 2;
        bool v756;
        v756 = v755 < 4;
        bool v757;
        v757 = v756 == false;
        if (v757){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v756);
        } else {
        }
        assert("Tensor range check" && 0 <= v755 && v755 < 4);
        assert("Tensor range check" && 0 <= v754 && v754 < 2);
        int v759;
        v759 = 2 * v755;
        int v760;
        v760 = v759 + v754;
        v749[v760] = v743;
        int v761;
        v761 = v755 + 1;
        bool v762;
        v762 = v761 < 16;
        bool v763;
        v763 = v762 == false;
        if (v763){
            assert("The barrier_id has to be less than 16." && v762);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v761), "r"(64));
        int v765;
        v765 = threadIdx.x;
        int v766;
        v766 = v765 % 32;
        bool v767;
        v767 = v766 < 2;
        float v770;
        if (v767){
            assert("Tensor range check" && 0 <= v755 && v755 < 4);
            assert("Tensor range check" && 0 <= v766 && v766 < 2);
            int v768;
            v768 = v759 + v766;
            float v769;
            v769 = v749[v768];
            v770 = v769;
        } else {
            v770 = 0.0f;
        }
        __syncthreads();
        float v771;
        v771 = cooperative_groups::reduce(v742, v770, v689);
        float v772[4];
        int v773;
        v773 = 0;
        while (while_method_3(v773)){
            int v775;
            v775 = 0;
            while (while_method_1(v775)){
                assert("Tensor range check" && 0 <= v773 && v773 < 1);
                assert("Tensor range check" && 0 <= v775 && v775 < 4);
                int v777;
                v777 = 4 * v773;
                int v778;
                v778 = v777 + v775;
                float v779;
                v779 = v721[v778];
                float v780;
                v780 = v779 / v771;
                assert("Tensor range check" && 0 <= v773 && v773 < 1);
                assert("Tensor range check" && 0 <= v775 && v775 < 4);
                v772[v778] = v780;
                v775 += 1 ;
            }
            v773 += 1 ;
        }
        float v781[4];
        float v782;
        v782 = 0.0f;
        int v783;
        v783 = 0;
        while (while_method_3(v783)){
            assert("Tensor range check" && 0 <= v783 && v783 < 1);
            int v785;
            v785 = 4 * v783;
            assert("Tensor range check" && 0 <= v783 && v783 < 1);
            int v786; float v787;
            Tuple0 tmp54 = Tuple0{0, 0.0f};
            v786 = tmp54.v0; v787 = tmp54.v1;
            while (while_method_1(v786)){
                assert("Tensor range check" && 0 <= v786 && v786 < 4);
                int v789;
                v789 = v786 + v785;
                float v790;
                v790 = v772[v789];
                float v791;
                v791 = v787 + v790;
                v787 = v791;
                v786 += 1 ;
            }
            auto v792 = cooperative_groups::coalesced_threads();
            int v793;
            v793 = threadIdx.x;
            int v794;
            v794 = v793 / 32;
            if (v640){
                assert("The dynamic shared memory is insufficient to allocate the tensor." && v639);
            } else {
            }
            extern __shared__ unsigned char v796[];
            if (v644){
                assert("The length of the partition has to be less than or equal to the length of the base array." && v643);
            } else {
            }
            float * v798;
            v798 = reinterpret_cast<float *>(&v796[v637]);
            Closure2 v800{};
            float v801;
            v801 = cooperative_groups::inclusive_scan(v792, v787, v800);
            float v802;
            v802 = v792.shfl_up(v801,1);
            bool v803;
            v803 = v792.thread_rank() == 0;
            float v804;
            if (v803){
                v804 = 0.0f;
            } else {
                v804 = v802;
            }
            float v805;
            v805 = v792.shfl(v801,v792.num_threads()-1);
            bool v806;
            v806 = 0 <= v794;
            bool v807;
            v807 = v806 == false;
            if (v807){
                assert("The index needs to be zero or positive." && v806);
            } else {
            }
            int v809;
            v809 = v794 % 2;
            int v810;
            v810 = v794 / 2;
            bool v811;
            v811 = v810 < 4;
            bool v812;
            v812 = v811 == false;
            if (v812){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v811);
            } else {
            }
            assert("Tensor range check" && 0 <= v810 && v810 < 4);
            assert("Tensor range check" && 0 <= v809 && v809 < 2);
            int v814;
            v814 = 2 * v810;
            int v815;
            v815 = v814 + v809;
            v798[v815] = v805;
            int v816;
            v816 = v810 + 1;
            bool v817;
            v817 = v816 < 16;
            bool v818;
            v818 = v817 == false;
            if (v818){
                assert("The barrier_id has to be less than 16." && v817);
            } else {
            }
            asm("barrier.cta.sync %0, %1;" :: "r"(v816), "r"(64));
            int v820;
            v820 = threadIdx.x;
            int v821;
            v821 = v820 % 32;
            bool v822;
            v822 = v821 < 2;
            float v825;
            if (v822){
                assert("Tensor range check" && 0 <= v810 && v810 < 4);
                assert("Tensor range check" && 0 <= v821 && v821 < 2);
                int v823;
                v823 = v814 + v821;
                float v824;
                v824 = v798[v823];
                v825 = v824;
            } else {
                v825 = 0.0f;
            }
            __syncthreads();
            float v826;
            v826 = cooperative_groups::inclusive_scan(v792, v825, v800);
            float v827;
            v827 = v792.shfl_up(v826,1);
            bool v828;
            v828 = v792.thread_rank() == 0;
            float v829;
            if (v828){
                v829 = 0.0f;
            } else {
                v829 = v827;
            }
            float v830;
            v830 = v792.shfl(v826,v792.num_threads()-1);
            float v831;
            v831 = v792.shfl(v829,v809);
            float v832;
            v832 = v831 + v804;
            float v833;
            v833 = v782 + v832;
            int v834; float v835;
            Tuple0 tmp55 = Tuple0{0, v833};
            v834 = tmp55.v0; v835 = tmp55.v1;
            while (while_method_1(v834)){
                assert("Tensor range check" && 0 <= v834 && v834 < 4);
                int v837;
                v837 = v834 + v785;
                float v838;
                v838 = v772[v837];
                float v839;
                v839 = v835 + v838;
                assert("Tensor range check" && 0 <= v834 && v834 < 4);
                v781[v837] = v839;
                v835 = v839;
                v834 += 1 ;
            }
            float v840;
            v840 = v782 + v830;
            v782 = v840;
            v783 += 1 ;
        }
        float v841[4];
        bool v842[4];
        int v843;
        v843 = 0;
        while (while_method_3(v843)){
            int v845;
            v845 = 0;
            while (while_method_1(v845)){
                assert("Tensor range check" && 0 <= v843 && v843 < 1);
                assert("Tensor range check" && 0 <= v845 && v845 < 4);
                int v847;
                v847 = 4 * v843;
                int v848;
                v848 = v847 + v845;
                float v849;
                v849 = v781[v848];
                float v850;
                v850 = v772[v848];
                bool v851;
                v851 = v850 > 0.0f;
                assert("Tensor range check" && 0 <= v843 && v843 < 1);
                assert("Tensor range check" && 0 <= v845 && v845 < 4);
                v841[v848] = v849;
                v842[v848] = v851;
                v845 += 1 ;
            }
            v843 += 1 ;
        }
        float v852; bool v853;
        Tuple3 tmp56 = Tuple3{-1.0f / 0.0f, false};
        v852 = tmp56.v0; v853 = tmp56.v1;
        int v854;
        v854 = 0;
        while (while_method_3(v854)){
            int v856;
            v856 = 0;
            while (while_method_1(v856)){
                assert("Tensor range check" && 0 <= v854 && v854 < 1);
                assert("Tensor range check" && 0 <= v856 && v856 < 4);
                int v858;
                v858 = 4 * v854;
                int v859;
                v859 = v858 + v856;
                float v860;
                v860 = v841[v859];
                bool v861;
                v861 = v842[v859];
                float v868; bool v869;
                if (v853){
                    if (v861){
                        bool v862;
                        v862 = v852 >= v860;
                        float v863;
                        if (v862){
                            v863 = v852;
                        } else {
                            v863 = v860;
                        }
                        v868 = v863; v869 = true;
                    } else {
                        v868 = v852; v869 = v853;
                    }
                } else {
                    if (v861){
                        v868 = v860; v869 = v861;
                    } else {
                        v868 = v852; v869 = v853;
                    }
                }
                v852 = v868;
                v853 = v869;
                v856 += 1 ;
            }
            v854 += 1 ;
        }
        auto v870 = cooperative_groups::coalesced_threads();
        Closure5 v871{};
        float v872; bool v873;
        Tuple3 tmp57 = cooperative_groups::reduce(v870, Tuple3{v852, v853}, v871);
        v872 = tmp57.v0; v873 = tmp57.v1;
        int v874;
        v874 = threadIdx.x;
        int v875;
        v875 = v874 / 32;
        unsigned long long v876;
        v876 = v638 + 16ull;
        unsigned long long v877;
        v877 = v876 - 1ull;
        unsigned long long v878;
        v878 = v877 % 16ull;
        unsigned long long v879;
        v879 = v877 - v878;
        unsigned long long v880;
        v880 = v879 + 8ull;
        bool v881;
        v881 = v880 <= 98304ull;
        bool v882;
        v882 = v881 == false;
        if (v882){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v881);
        } else {
        }
        extern __shared__ unsigned char v884[];
        bool v885;
        v885 = v880 <= v880;
        bool v886;
        v886 = v885 == false;
        if (v886){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v885);
        } else {
        }
        float * v888;
        v888 = reinterpret_cast<float *>(&v884[v637]);
        bool * v890;
        v890 = reinterpret_cast<bool *>(&v884[v879]);
        bool v892;
        v892 = 0 <= v875;
        bool v893;
        v893 = v892 == false;
        if (v893){
            assert("The index needs to be zero or positive." && v892);
        } else {
        }
        int v895;
        v895 = v875 % 2;
        int v896;
        v896 = v875 / 2;
        bool v897;
        v897 = v896 < 4;
        bool v898;
        v898 = v897 == false;
        if (v898){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v897);
        } else {
        }
        assert("Tensor range check" && 0 <= v896 && v896 < 4);
        assert("Tensor range check" && 0 <= v895 && v895 < 2);
        int v900;
        v900 = 2 * v896;
        int v901;
        v901 = v900 + v895;
        v888[v901] = v872;
        v890[v901] = v873;
        int v902;
        v902 = v896 + 1;
        bool v903;
        v903 = v902 < 16;
        bool v904;
        v904 = v903 == false;
        if (v904){
            assert("The barrier_id has to be less than 16." && v903);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v902), "r"(64));
        int v906;
        v906 = threadIdx.x;
        int v907;
        v907 = v906 % 32;
        bool v908;
        v908 = v907 < 2;
        float v912; bool v913;
        if (v908){
            assert("Tensor range check" && 0 <= v896 && v896 < 4);
            assert("Tensor range check" && 0 <= v907 && v907 < 2);
            int v909;
            v909 = v900 + v907;
            float v910;
            v910 = v888[v909];
            bool v911;
            v911 = v890[v909];
            v912 = v910; v913 = v911;
        } else {
            v912 = -1.0f / 0.0f; v913 = false;
        }
        __syncthreads();
        float v914; bool v915;
        Tuple3 tmp58 = cooperative_groups::reduce(v870, Tuple3{v912, v913}, v871);
        v914 = tmp58.v0; v915 = tmp58.v1;
        bool v916;
        v916 = v915 == false;
        if (v916){
            assert("The local reduce must be true." && v915);
        } else {
        }
        float v918[4];
        int v919[4];
        int v920;
        v920 = 0;
        while (while_method_3(v920)){
            int v922;
            v922 = 0;
            while (while_method_1(v922)){
                assert("Tensor range check" && 0 <= v920 && v920 < 1);
                assert("Tensor range check" && 0 <= v922 && v922 < 4);
                int v924;
                v924 = 4 * v920;
                int v925;
                v925 = v924 + v922;
                int v926;
                v926 = v568[v925];
                float v927;
                v927 = curand_uniform(&v528);
                assert("Tensor range check" && 0 <= v920 && v920 < 1);
                assert("Tensor range check" && 0 <= v922 && v922 < 4);
                v918[v925] = v927;
                v919[v925] = v926;
                v922 += 1 ;
            }
            v920 += 1 ;
        }
        float v928; int v929;
        Tuple1 tmp59 = Tuple1{0.0f, 2147483647};
        v928 = tmp59.v0; v929 = tmp59.v1;
        int v930;
        v930 = 0;
        while (while_method_3(v930)){
            int v932;
            v932 = 0;
            while (while_method_1(v932)){
                assert("Tensor range check" && 0 <= v930 && v930 < 1);
                assert("Tensor range check" && 0 <= v932 && v932 < 4);
                int v934;
                v934 = 4 * v930;
                int v935;
                v935 = v934 + v932;
                float v936;
                v936 = v918[v935];
                int v937;
                v937 = v919[v935];
                bool v938;
                v938 = v929 < v937;
                float v939; int v940;
                if (v938){
                    v939 = v928; v940 = v929;
                } else {
                    v939 = v936; v940 = v937;
                }
                v928 = v939;
                v929 = v940;
                v932 += 1 ;
            }
            v930 += 1 ;
        }
        auto v941 = cooperative_groups::coalesced_threads();
        Closure6 v942{};
        float v943; int v944;
        Tuple1 tmp60 = cooperative_groups::reduce(v941, Tuple1{v928, v929}, v942);
        v943 = tmp60.v0; v944 = tmp60.v1;
        int v945;
        v945 = threadIdx.x;
        int v946;
        v946 = v945 / 32;
        unsigned long long v947;
        v947 = v879 + 32ull;
        bool v948;
        v948 = v947 <= 98304ull;
        bool v949;
        v949 = v948 == false;
        if (v949){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v948);
        } else {
        }
        extern __shared__ unsigned char v951[];
        bool v952;
        v952 = v947 <= v947;
        bool v953;
        v953 = v952 == false;
        if (v953){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v952);
        } else {
        }
        float * v955;
        v955 = reinterpret_cast<float *>(&v951[v637]);
        int * v957;
        v957 = reinterpret_cast<int *>(&v951[v879]);
        bool v959;
        v959 = 0 <= v946;
        bool v960;
        v960 = v959 == false;
        if (v960){
            assert("The index needs to be zero or positive." && v959);
        } else {
        }
        int v962;
        v962 = v946 % 2;
        int v963;
        v963 = v946 / 2;
        bool v964;
        v964 = v963 < 4;
        bool v965;
        v965 = v964 == false;
        if (v965){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v964);
        } else {
        }
        assert("Tensor range check" && 0 <= v963 && v963 < 4);
        assert("Tensor range check" && 0 <= v962 && v962 < 2);
        int v967;
        v967 = 2 * v963;
        int v968;
        v968 = v967 + v962;
        v955[v968] = v943;
        v957[v968] = v944;
        int v969;
        v969 = v963 + 1;
        bool v970;
        v970 = v969 < 16;
        bool v971;
        v971 = v970 == false;
        if (v971){
            assert("The barrier_id has to be less than 16." && v970);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v969), "r"(64));
        int v973;
        v973 = threadIdx.x;
        int v974;
        v974 = v973 % 32;
        bool v975;
        v975 = v974 < 2;
        float v979; int v980;
        if (v975){
            assert("Tensor range check" && 0 <= v963 && v963 < 4);
            assert("Tensor range check" && 0 <= v974 && v974 < 2);
            int v976;
            v976 = v967 + v974;
            float v977;
            v977 = v955[v976];
            int v978;
            v978 = v957[v976];
            v979 = v977; v980 = v978;
        } else {
            v979 = 0.0f; v980 = 2147483647;
        }
        __syncthreads();
        float v981; int v982;
        Tuple1 tmp61 = cooperative_groups::reduce(v941, Tuple1{v979, v980}, v942);
        v981 = tmp61.v0; v982 = tmp61.v1;
        float v983;
        v983 = v914 * v981;
        int v984[4];
        bool v985[4];
        int v986;
        v986 = 0;
        while (while_method_3(v986)){
            int v988;
            v988 = 0;
            while (while_method_1(v988)){
                assert("Tensor range check" && 0 <= v986 && v986 < 1);
                assert("Tensor range check" && 0 <= v988 && v988 < 4);
                int v990;
                v990 = 4 * v986;
                int v991;
                v991 = v990 + v988;
                float v992;
                v992 = v841[v991];
                bool v993;
                v993 = v842[v991];
                int v994;
                v994 = v568[v991];
                int v997; bool v998;
                if (v993){
                    float v995;
                    v995 = v992 - v983;
                    bool v996;
                    v996 = v995 >= 0.0f;
                    v997 = v994; v998 = v996;
                } else {
                    v997 = 2147483647; v998 = false;
                }
                assert("Tensor range check" && 0 <= v986 && v986 < 1);
                assert("Tensor range check" && 0 <= v988 && v988 < 4);
                v984[v991] = v997;
                v985[v991] = v998;
                v988 += 1 ;
            }
            v986 += 1 ;
        }
        int v999; bool v1000;
        Tuple4 tmp62 = Tuple4{2147483647, false};
        v999 = tmp62.v0; v1000 = tmp62.v1;
        int v1001;
        v1001 = 0;
        while (while_method_3(v1001)){
            int v1003;
            v1003 = 0;
            while (while_method_1(v1003)){
                assert("Tensor range check" && 0 <= v1001 && v1001 < 1);
                assert("Tensor range check" && 0 <= v1003 && v1003 < 4);
                int v1005;
                v1005 = 4 * v1001;
                int v1006;
                v1006 = v1005 + v1003;
                int v1007;
                v1007 = v984[v1006];
                bool v1008;
                v1008 = v985[v1006];
                int v1015; bool v1016;
                if (v1000){
                    if (v1008){
                        bool v1009;
                        v1009 = v999 < v1007;
                        int v1010;
                        if (v1009){
                            v1010 = v999;
                        } else {
                            v1010 = v1007;
                        }
                        v1015 = v1010; v1016 = true;
                    } else {
                        v1015 = v999; v1016 = v1000;
                    }
                } else {
                    if (v1008){
                        v1015 = v1007; v1016 = v1008;
                    } else {
                        v1015 = v999; v1016 = v1000;
                    }
                }
                v999 = v1015;
                v1000 = v1016;
                v1003 += 1 ;
            }
            v1001 += 1 ;
        }
        auto v1017 = cooperative_groups::coalesced_threads();
        Closure7 v1018{};
        int v1019; bool v1020;
        Tuple4 tmp63 = cooperative_groups::reduce(v1017, Tuple4{v999, v1000}, v1018);
        v1019 = tmp63.v0; v1020 = tmp63.v1;
        int v1021;
        v1021 = threadIdx.x;
        int v1022;
        v1022 = v1021 / 32;
        if (v882){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v881);
        } else {
        }
        extern __shared__ unsigned char v1024[];
        if (v886){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v885);
        } else {
        }
        int * v1026;
        v1026 = reinterpret_cast<int *>(&v1024[v637]);
        bool * v1028;
        v1028 = reinterpret_cast<bool *>(&v1024[v879]);
        bool v1030;
        v1030 = 0 <= v1022;
        bool v1031;
        v1031 = v1030 == false;
        if (v1031){
            assert("The index needs to be zero or positive." && v1030);
        } else {
        }
        int v1033;
        v1033 = v1022 % 2;
        int v1034;
        v1034 = v1022 / 2;
        bool v1035;
        v1035 = v1034 < 4;
        bool v1036;
        v1036 = v1035 == false;
        if (v1036){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1035);
        } else {
        }
        assert("Tensor range check" && 0 <= v1034 && v1034 < 4);
        assert("Tensor range check" && 0 <= v1033 && v1033 < 2);
        int v1038;
        v1038 = 2 * v1034;
        int v1039;
        v1039 = v1038 + v1033;
        v1026[v1039] = v1019;
        v1028[v1039] = v1020;
        int v1040;
        v1040 = v1034 + 1;
        bool v1041;
        v1041 = v1040 < 16;
        bool v1042;
        v1042 = v1041 == false;
        if (v1042){
            assert("The barrier_id has to be less than 16." && v1041);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v1040), "r"(64));
        int v1044;
        v1044 = threadIdx.x;
        int v1045;
        v1045 = v1044 % 32;
        bool v1046;
        v1046 = v1045 < 2;
        int v1050; bool v1051;
        if (v1046){
            assert("Tensor range check" && 0 <= v1034 && v1034 < 4);
            assert("Tensor range check" && 0 <= v1045 && v1045 < 2);
            int v1047;
            v1047 = v1038 + v1045;
            int v1048;
            v1048 = v1026[v1047];
            bool v1049;
            v1049 = v1028[v1047];
            v1050 = v1048; v1051 = v1049;
        } else {
            v1050 = 2147483647; v1051 = false;
        }
        __syncthreads();
        int v1052; bool v1053;
        Tuple4 tmp64 = cooperative_groups::reduce(v1017, Tuple4{v1050, v1051}, v1018);
        v1052 = tmp64.v0; v1053 = tmp64.v1;
        bool v1054;
        v1054 = v1053 == false;
        if (v1054){
            assert("The local reduce must be true." && v1053);
        } else {
        }
        int v1056;
        v1056 = 0;
        while (while_method_3(v1056)){
            assert("Tensor range check" && 0 <= v1056 && v1056 < 1);
            assert("Tensor range check" && 0 <= v1056 && v1056 < 1);
            v1056 += 1 ;
        }
        assert("Tensor range check" && 0 <= v559 && v559 < 256);
        v536[v559] = v1052;
        v547 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v538 && v538 < 256);
    int v1058;
    v1058 = v536[v538];
    __syncthreads();
    int v1059;
    v1059 = threadIdx.x;
    assert("Tensor range check" && 0 <= v1059 && v1059 < 256);
    v5[v1059] = v1058;
    return ;
}
extern "C" __global__ void entry4(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14, float * v15, int * v16) {
    auto v17 = cooperative_groups::this_grid();
    int v18;
    v18 = threadIdx.x;
    bool v19;
    v19 = 0 <= v18;
    bool v20;
    v20 = v19 == false;
    if (v20){
        assert("The index needs to be zero or positive." && v19);
    } else {
    }
    int v22;
    v22 = v18 % 16;
    int v23;
    v23 = v18 / 16;
    bool v24;
    v24 = v23 < 16;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
    } else {
    }
    assert("Tensor range check" && 0 <= v23 && v23 < 16);
    assert("Tensor range check" && 0 <= v22 && v22 < 16);
    int v27;
    v27 = 4 * v22;
    int v28;
    v28 = 64 * v23;
    int v29;
    v29 = v28 + v27;
    assert("Tensor range check" && 0 <= v23 && v23 < 16);
    assert("Tensor range check" && 0 <= v22 && v22 < 16);
    int v30;
    v30 = blockIdx.x;
    int v31;
    v31 = v30;
    while (while_method_2(v31)){
        bool v33;
        v33 = 0 <= v31;
        bool v34;
        v34 = v33 == false;
        if (v34){
            assert("The index needs to be zero or positive." && v33);
        } else {
        }
        bool v36;
        v36 = v31 < 8;
        bool v37;
        v37 = v36 == false;
        if (v37){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v36);
        } else {
        }
        assert("Tensor range check" && 0 <= v31 && v31 < 8);
        int v39;
        v39 = 1024 * v31;
        int v40;
        v40 = v39 + v29;
        int v41[4];
        int v42[4];
        int v43;
        v43 = 0;
        while (while_method_3(v43)){
            assert("Tensor range check" && 0 <= v43 && v43 < 1);
            int v45;
            v45 = 4 * v43;
            assert("Tensor range check" && 0 <= v43 && v43 < 1);
            int v46;
            v46 = 64 * v43;
            int v47;
            v47 = v46 + v40;
            int4* v48;
            v48 = reinterpret_cast<int4*>(v0 + v47);
            int4* v49;
            v49 = reinterpret_cast<int4*>(v41 + v45);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v48) % 16 == 0 && reinterpret_cast<unsigned long long>(v49) % 16 == 0);
            *v49 = *v48;
            v43 += 1 ;
        }
        int v50;
        v50 = 0;
        while (while_method_3(v50)){
            int v52;
            v52 = 0;
            while (while_method_1(v52)){
                bool v54;
                v54 = 0 <= v52;
                bool v56;
                if (v54){
                    bool v55;
                    v55 = v52 < 4;
                    v56 = v55;
                } else {
                    v56 = false;
                }
                bool v57;
                v57 = v56 == false;
                if (v57){
                    assert("The indices should be inside the range of the dimension." && v56);
                } else {
                }
                bool v59;
                v59 = 0 <= v22;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v22 < 16;
                    v61 = v60;
                } else {
                    v61 = false;
                }
                bool v62;
                v62 = v61 == false;
                if (v62){
                    assert("The indices should be inside the range of the dimension." && v61);
                } else {
                }
                int v64;
                v64 = v22 * 4;
                int v65;
                v65 = v52 + v64;
                bool v66;
                v66 = 0 <= v50;
                bool v68;
                if (v66){
                    bool v67;
                    v67 = v50 < 1;
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
                v71 = v50 * 64;
                int v72;
                v72 = v65 + v71;
                assert("Tensor range check" && 0 <= v50 && v50 < 1);
                assert("Tensor range check" && 0 <= v52 && v52 < 4);
                int v73;
                v73 = 4 * v50;
                int v74;
                v74 = v73 + v52;
                v42[v74] = v72;
                v52 += 1 ;
            }
            v50 += 1 ;
        }
        bool v75;
        v75 = 0 <= v23;
        bool v76;
        v76 = v75 && v24;
        bool v77;
        v77 = v76 == false;
        if (v77){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v76);
        } else {
        }
        bool v79;
        v79 = v33 && v36;
        bool v80;
        v80 = v79 == false;
        if (v80){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v79);
        } else {
        }
        int v82;
        v82 = v31 * 16;
        int v83;
        v83 = v82 + v23;
        assert("Tensor range check" && 0 <= v31 && v31 < 8);
        int v84;
        v84 = 0;
        while (while_method_3(v84)){
            assert("Tensor range check" && 0 <= v84 && v84 < 1);
            int v86;
            v86 = 64 * v84;
            int v87;
            v87 = v86 + v40;
            assert("Tensor range check" && 0 <= v84 && v84 < 1);
            int v88;
            v88 = 4 * v84;
            int4* v89;
            v89 = reinterpret_cast<int4*>(v41 + v88);
            int4* v90;
            v90 = reinterpret_cast<int4*>(v2 + v87);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v89) % 16 == 0 && reinterpret_cast<unsigned long long>(v90) % 16 == 0);
            *v90 = *v89;
            v84 += 1 ;
        }
        v31 += 24 ;
    }
    v17.sync() ;
    int v91;
    v91 = threadIdx.x;
    bool v92;
    v92 = 0 <= v91;
    bool v93;
    v93 = v92 == false;
    if (v93){
        assert("The index needs to be zero or positive." && v92);
    } else {
    }
    int v95;
    v95 = v91 % 16;
    int v96;
    v96 = v91 / 16;
    bool v97;
    v97 = v96 < 16;
    bool v98;
    v98 = v97 == false;
    if (v98){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v97);
    } else {
    }
    assert("Tensor range check" && 0 <= v96 && v96 < 16);
    assert("Tensor range check" && 0 <= v95 && v95 < 16);
    int v100;
    v100 = 4 * v95;
    int v101;
    v101 = 64 * v96;
    int v102;
    v102 = v101 + v100;
    assert("Tensor range check" && 0 <= v96 && v96 < 16);
    assert("Tensor range check" && 0 <= v95 && v95 < 16);
    int v103;
    v103 = blockIdx.x;
    int v104;
    v104 = v103;
    while (while_method_2(v104)){
        bool v106;
        v106 = 0 <= v104;
        bool v107;
        v107 = v106 == false;
        if (v107){
            assert("The index needs to be zero or positive." && v106);
        } else {
        }
        bool v109;
        v109 = v104 < 8;
        bool v110;
        v110 = v109 == false;
        if (v110){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v109);
        } else {
        }
        assert("Tensor range check" && 0 <= v104 && v104 < 8);
        int v112;
        v112 = 1024 * v104;
        int v113;
        v113 = v112 + v102;
        float v114[4];
        int v115[4];
        int v116;
        v116 = 0;
        while (while_method_3(v116)){
            assert("Tensor range check" && 0 <= v116 && v116 < 1);
            int v118;
            v118 = 4 * v116;
            assert("Tensor range check" && 0 <= v116 && v116 < 1);
            int v119;
            v119 = 64 * v116;
            int v120;
            v120 = v119 + v113;
            int4* v121;
            v121 = reinterpret_cast<int4*>(v1 + v120);
            int4* v122;
            v122 = reinterpret_cast<int4*>(v114 + v118);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v121) % 16 == 0 && reinterpret_cast<unsigned long long>(v122) % 16 == 0);
            *v122 = *v121;
            v116 += 1 ;
        }
        int v123;
        v123 = 0;
        while (while_method_3(v123)){
            int v125;
            v125 = 0;
            while (while_method_1(v125)){
                bool v127;
                v127 = 0 <= v125;
                bool v129;
                if (v127){
                    bool v128;
                    v128 = v125 < 4;
                    v129 = v128;
                } else {
                    v129 = false;
                }
                bool v130;
                v130 = v129 == false;
                if (v130){
                    assert("The indices should be inside the range of the dimension." && v129);
                } else {
                }
                bool v132;
                v132 = 0 <= v95;
                bool v134;
                if (v132){
                    bool v133;
                    v133 = v95 < 16;
                    v134 = v133;
                } else {
                    v134 = false;
                }
                bool v135;
                v135 = v134 == false;
                if (v135){
                    assert("The indices should be inside the range of the dimension." && v134);
                } else {
                }
                int v137;
                v137 = v95 * 4;
                int v138;
                v138 = v125 + v137;
                bool v139;
                v139 = 0 <= v123;
                bool v141;
                if (v139){
                    bool v140;
                    v140 = v123 < 1;
                    v141 = v140;
                } else {
                    v141 = false;
                }
                bool v142;
                v142 = v141 == false;
                if (v142){
                    assert("The indices should be inside the range of the dimension." && v141);
                } else {
                }
                int v144;
                v144 = v123 * 64;
                int v145;
                v145 = v138 + v144;
                assert("Tensor range check" && 0 <= v123 && v123 < 1);
                assert("Tensor range check" && 0 <= v125 && v125 < 4);
                int v146;
                v146 = 4 * v123;
                int v147;
                v147 = v146 + v125;
                v115[v147] = v145;
                v125 += 1 ;
            }
            v123 += 1 ;
        }
        bool v148;
        v148 = 0 <= v96;
        bool v149;
        v149 = v148 && v97;
        bool v150;
        v150 = v149 == false;
        if (v150){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v149);
        } else {
        }
        bool v152;
        v152 = v106 && v109;
        bool v153;
        v153 = v152 == false;
        if (v153){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v152);
        } else {
        }
        int v155;
        v155 = v104 * 16;
        int v156;
        v156 = v155 + v96;
        int v157[4];
        int v158[4];
        int v159;
        v159 = 0;
        while (while_method_3(v159)){
            int v161;
            v161 = 0;
            while (while_method_1(v161)){
                assert("Tensor range check" && 0 <= v159 && v159 < 1);
                assert("Tensor range check" && 0 <= v161 && v161 < 4);
                int v163;
                v163 = 4 * v159;
                int v164;
                v164 = v163 + v161;
                int v165;
                v165 = v115[v164];
                assert("Tensor range check" && 0 <= v159 && v159 < 1);
                assert("Tensor range check" && 0 <= v161 && v161 < 4);
                v157[v164] = v156;
                v158[v164] = v165;
                v161 += 1 ;
            }
            v159 += 1 ;
        }
        assert("Tensor range check" && 0 <= v104 && v104 < 8);
        int v166;
        v166 = 0;
        while (while_method_3(v166)){
            assert("Tensor range check" && 0 <= v166 && v166 < 1);
            int v168;
            v168 = 64 * v166;
            int v169;
            v169 = v168 + v113;
            assert("Tensor range check" && 0 <= v166 && v166 < 1);
            int v170;
            v170 = 4 * v166;
            int4* v171;
            v171 = reinterpret_cast<int4*>(v157 + v170);
            int4* v172;
            v172 = reinterpret_cast<int4*>(v9 + v169);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v171) % 16 == 0 && reinterpret_cast<unsigned long long>(v172) % 16 == 0);
            *v172 = *v171;
            int4* v173;
            v173 = reinterpret_cast<int4*>(v158 + v170);
            int4* v174;
            v174 = reinterpret_cast<int4*>(v10 + v169);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v173) % 16 == 0 && reinterpret_cast<unsigned long long>(v174) % 16 == 0);
            *v174 = *v173;
            v166 += 1 ;
        }
        v104 += 24 ;
    }
    v17.sync() ;
    int v175;
    v175 = threadIdx.x;
    bool v176;
    v176 = 0 <= v175;
    bool v177;
    v177 = v176 == false;
    if (v177){
        assert("The index needs to be zero or positive." && v176);
    } else {
    }
    int v179;
    v179 = v175 % 16;
    int v180;
    v180 = v175 / 16;
    bool v181;
    v181 = v180 < 16;
    bool v182;
    v182 = v181 == false;
    if (v182){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v181);
    } else {
    }
    assert("Tensor range check" && 0 <= v180 && v180 < 16);
    assert("Tensor range check" && 0 <= v179 && v179 < 16);
    int v184;
    v184 = 4 * v179;
    int v185;
    v185 = 64 * v180;
    int v186;
    v186 = v185 + v184;
    assert("Tensor range check" && 0 <= v180 && v180 < 16);
    int v187;
    v187 = blockIdx.x;
    int v188;
    v188 = v187;
    while (while_method_2(v188)){
        bool v190;
        v190 = 0 <= v188;
        bool v191;
        v191 = v190 == false;
        if (v191){
            assert("The index needs to be zero or positive." && v190);
        } else {
        }
        bool v193;
        v193 = v188 < 8;
        bool v194;
        v194 = v193 == false;
        if (v194){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v193);
        } else {
        }
        assert("Tensor range check" && 0 <= v188 && v188 < 8);
        int v196;
        v196 = 1024 * v188;
        int v197;
        v197 = v196 + v186;
        float v198[4];
        int v199[4];
        int v200;
        v200 = 0;
        while (while_method_3(v200)){
            assert("Tensor range check" && 0 <= v200 && v200 < 1);
            int v202;
            v202 = 4 * v200;
            assert("Tensor range check" && 0 <= v200 && v200 < 1);
            int v203;
            v203 = 64 * v200;
            int v204;
            v204 = v203 + v197;
            int4* v205;
            v205 = reinterpret_cast<int4*>(v1 + v204);
            int4* v206;
            v206 = reinterpret_cast<int4*>(v198 + v202);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v205) % 16 == 0 && reinterpret_cast<unsigned long long>(v206) % 16 == 0);
            *v206 = *v205;
            v200 += 1 ;
        }
        int v207;
        v207 = 0;
        while (while_method_3(v207)){
            int v209;
            v209 = 0;
            while (while_method_1(v209)){
                bool v211;
                v211 = 0 <= v209;
                bool v213;
                if (v211){
                    bool v212;
                    v212 = v209 < 4;
                    v213 = v212;
                } else {
                    v213 = false;
                }
                bool v214;
                v214 = v213 == false;
                if (v214){
                    assert("The indices should be inside the range of the dimension." && v213);
                } else {
                }
                bool v216;
                v216 = 0 <= v179;
                bool v218;
                if (v216){
                    bool v217;
                    v217 = v179 < 16;
                    v218 = v217;
                } else {
                    v218 = false;
                }
                bool v219;
                v219 = v218 == false;
                if (v219){
                    assert("The indices should be inside the range of the dimension." && v218);
                } else {
                }
                int v221;
                v221 = v179 * 4;
                int v222;
                v222 = v209 + v221;
                bool v223;
                v223 = 0 <= v207;
                bool v225;
                if (v223){
                    bool v224;
                    v224 = v207 < 1;
                    v225 = v224;
                } else {
                    v225 = false;
                }
                bool v226;
                v226 = v225 == false;
                if (v226){
                    assert("The indices should be inside the range of the dimension." && v225);
                } else {
                }
                int v228;
                v228 = v207 * 64;
                int v229;
                v229 = v222 + v228;
                assert("Tensor range check" && 0 <= v207 && v207 < 1);
                assert("Tensor range check" && 0 <= v209 && v209 < 4);
                int v230;
                v230 = 4 * v207;
                int v231;
                v231 = v230 + v209;
                v199[v231] = v229;
                v209 += 1 ;
            }
            v207 += 1 ;
        }
        bool v232;
        v232 = 0 <= v180;
        bool v233;
        v233 = v232 && v181;
        bool v234;
        v234 = v233 == false;
        if (v234){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v233);
        } else {
        }
        bool v236;
        v236 = v190 && v193;
        bool v237;
        v237 = v236 == false;
        if (v237){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v236);
        } else {
        }
        int v239;
        v239 = v188 * 16;
        int v240;
        v240 = v239 + v180;
        assert("Tensor range check" && 0 <= v188 && v188 < 8);
        int v241;
        v241 = 16 * v188;
        int v242;
        v242 = v241 + v180;
        v11[v242] = v240;
        v188 += 24 ;
    }
    v17.sync() ;
    int v243;
    v243 = threadIdx.x;
    bool v244;
    v244 = 0 <= v243;
    bool v245;
    v245 = v244 == false;
    if (v245){
        assert("The index needs to be zero or positive." && v244);
    } else {
    }
    int v247;
    v247 = v243 % 16;
    int v248;
    v248 = v243 / 16;
    bool v249;
    v249 = v248 < 16;
    bool v250;
    v250 = v249 == false;
    if (v250){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v249);
    } else {
    }
    assert("Tensor range check" && 0 <= v248 && v248 < 16);
    assert("Tensor range check" && 0 <= v247 && v247 < 16);
    int v252;
    v252 = 4 * v247;
    int v253;
    v253 = 64 * v248;
    int v254;
    v254 = v253 + v252;
    assert("Tensor range check" && 0 <= v248 && v248 < 16);
    assert("Tensor range check" && 0 <= v247 && v247 < 16);
    int v255;
    v255 = blockIdx.x;
    int v256;
    v256 = v255;
    while (while_method_2(v256)){
        bool v258;
        v258 = 0 <= v256;
        bool v259;
        v259 = v258 == false;
        if (v259){
            assert("The index needs to be zero or positive." && v258);
        } else {
        }
        bool v261;
        v261 = v256 < 8;
        bool v262;
        v262 = v261 == false;
        if (v262){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v261);
        } else {
        }
        assert("Tensor range check" && 0 <= v256 && v256 < 8);
        int v264;
        v264 = 1024 * v256;
        int v265;
        v265 = v264 + v254;
        float v266[4];
        int v267[4];
        int v268;
        v268 = 0;
        while (while_method_3(v268)){
            assert("Tensor range check" && 0 <= v268 && v268 < 1);
            int v270;
            v270 = 4 * v268;
            assert("Tensor range check" && 0 <= v268 && v268 < 1);
            int v271;
            v271 = 64 * v268;
            int v272;
            v272 = v271 + v265;
            int4* v273;
            v273 = reinterpret_cast<int4*>(v1 + v272);
            int4* v274;
            v274 = reinterpret_cast<int4*>(v266 + v270);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v273) % 16 == 0 && reinterpret_cast<unsigned long long>(v274) % 16 == 0);
            *v274 = *v273;
            v268 += 1 ;
        }
        int v275;
        v275 = 0;
        while (while_method_3(v275)){
            int v277;
            v277 = 0;
            while (while_method_1(v277)){
                bool v279;
                v279 = 0 <= v277;
                bool v281;
                if (v279){
                    bool v280;
                    v280 = v277 < 4;
                    v281 = v280;
                } else {
                    v281 = false;
                }
                bool v282;
                v282 = v281 == false;
                if (v282){
                    assert("The indices should be inside the range of the dimension." && v281);
                } else {
                }
                bool v284;
                v284 = 0 <= v247;
                bool v286;
                if (v284){
                    bool v285;
                    v285 = v247 < 16;
                    v286 = v285;
                } else {
                    v286 = false;
                }
                bool v287;
                v287 = v286 == false;
                if (v287){
                    assert("The indices should be inside the range of the dimension." && v286);
                } else {
                }
                int v289;
                v289 = v247 * 4;
                int v290;
                v290 = v277 + v289;
                bool v291;
                v291 = 0 <= v275;
                bool v293;
                if (v291){
                    bool v292;
                    v292 = v275 < 1;
                    v293 = v292;
                } else {
                    v293 = false;
                }
                bool v294;
                v294 = v293 == false;
                if (v294){
                    assert("The indices should be inside the range of the dimension." && v293);
                } else {
                }
                int v296;
                v296 = v275 * 64;
                int v297;
                v297 = v290 + v296;
                assert("Tensor range check" && 0 <= v275 && v275 < 1);
                assert("Tensor range check" && 0 <= v277 && v277 < 4);
                int v298;
                v298 = 4 * v275;
                int v299;
                v299 = v298 + v277;
                v267[v299] = v297;
                v277 += 1 ;
            }
            v275 += 1 ;
        }
        bool v300;
        v300 = 0 <= v248;
        bool v301;
        v301 = v300 && v249;
        bool v302;
        v302 = v301 == false;
        if (v302){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v301);
        } else {
        }
        bool v304;
        v304 = v258 && v261;
        bool v305;
        v305 = v304 == false;
        if (v305){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v304);
        } else {
        }
        int v307;
        v307 = v256 * 16;
        int v308;
        v308 = v307 + v248;
        float v309;
        v309 = 0.0f;
        int v310;
        v310 = 0;
        while (while_method_3(v310)){
            int v312;
            v312 = 0;
            while (while_method_1(v312)){
                assert("Tensor range check" && 0 <= v310 && v310 < 1);
                assert("Tensor range check" && 0 <= v312 && v312 < 4);
                int v314;
                v314 = 4 * v310;
                int v315;
                v315 = v314 + v312;
                float v316;
                v316 = v266[v315];
                float v317;
                v317 = v309 + v316;
                v309 = v317;
                v312 += 1 ;
            }
            v310 += 1 ;
        }
        auto v318 = cooperative_groups::coalesced_threads();
        int v319;
        v319 = threadIdx.x;
        int v320;
        v320 = v319 / 16;
        auto v321 = cooperative_groups::labeled_partition(v318,v320);
        Closure0 v322{};
        float v323;
        v323 = cooperative_groups::reduce(v321, v309, v322);
        float v324;
        v324 = v323 / 64.0f;
        float v325[4];
        int v326;
        v326 = 0;
        while (while_method_3(v326)){
            int v328;
            v328 = 0;
            while (while_method_1(v328)){
                assert("Tensor range check" && 0 <= v326 && v326 < 1);
                assert("Tensor range check" && 0 <= v328 && v328 < 4);
                int v330;
                v330 = 4 * v326;
                int v331;
                v331 = v330 + v328;
                float v332;
                v332 = v266[v331];
                float v333;
                v333 = v332 - v324;
                float v334;
                v334 = exp(v333);
                assert("Tensor range check" && 0 <= v326 && v326 < 1);
                assert("Tensor range check" && 0 <= v328 && v328 < 4);
                v325[v331] = v334;
                v328 += 1 ;
            }
            v326 += 1 ;
        }
        float v335;
        v335 = 0.0f;
        int v336;
        v336 = 0;
        while (while_method_3(v336)){
            int v338;
            v338 = 0;
            while (while_method_1(v338)){
                assert("Tensor range check" && 0 <= v336 && v336 < 1);
                assert("Tensor range check" && 0 <= v338 && v338 < 4);
                int v340;
                v340 = 4 * v336;
                int v341;
                v341 = v340 + v338;
                float v342;
                v342 = v325[v341];
                float v343;
                v343 = v335 + v342;
                v335 = v343;
                v338 += 1 ;
            }
            v336 += 1 ;
        }
        auto v344 = cooperative_groups::coalesced_threads();
        int v345;
        v345 = threadIdx.x;
        int v346;
        v346 = v345 / 16;
        auto v347 = cooperative_groups::labeled_partition(v344,v346);
        float v348;
        v348 = cooperative_groups::reduce(v347, v335, v322);
        float v349[4];
        int v350;
        v350 = 0;
        while (while_method_3(v350)){
            int v352;
            v352 = 0;
            while (while_method_1(v352)){
                assert("Tensor range check" && 0 <= v350 && v350 < 1);
                assert("Tensor range check" && 0 <= v352 && v352 < 4);
                int v354;
                v354 = 4 * v350;
                int v355;
                v355 = v354 + v352;
                float v356;
                v356 = v325[v355];
                float v357;
                v357 = v356 / v348;
                assert("Tensor range check" && 0 <= v350 && v350 < 1);
                assert("Tensor range check" && 0 <= v352 && v352 < 4);
                v349[v355] = v357;
                v352 += 1 ;
            }
            v350 += 1 ;
        }
        assert("Tensor range check" && 0 <= v256 && v256 < 8);
        int v358;
        v358 = 0;
        while (while_method_3(v358)){
            assert("Tensor range check" && 0 <= v358 && v358 < 1);
            int v360;
            v360 = 64 * v358;
            int v361;
            v361 = v360 + v265;
            assert("Tensor range check" && 0 <= v358 && v358 < 1);
            int v362;
            v362 = 4 * v358;
            int4* v363;
            v363 = reinterpret_cast<int4*>(v349 + v362);
            int4* v364;
            v364 = reinterpret_cast<int4*>(v3 + v361);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v363) % 16 == 0 && reinterpret_cast<unsigned long long>(v364) % 16 == 0);
            *v364 = *v363;
            v358 += 1 ;
        }
        v256 += 24 ;
    }
    v17.sync() ;
    int v365;
    v365 = threadIdx.x;
    bool v366;
    v366 = 0 <= v365;
    bool v367;
    v367 = v366 == false;
    if (v367){
        assert("The index needs to be zero or positive." && v366);
    } else {
    }
    int v369;
    v369 = v365 % 16;
    int v370;
    v370 = v365 / 16;
    bool v371;
    v371 = v370 < 16;
    bool v372;
    v372 = v371 == false;
    if (v372){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v371);
    } else {
    }
    assert("Tensor range check" && 0 <= v370 && v370 < 16);
    assert("Tensor range check" && 0 <= v369 && v369 < 16);
    int v374;
    v374 = 4 * v369;
    int v375;
    v375 = 64 * v370;
    int v376;
    v376 = v375 + v374;
    assert("Tensor range check" && 0 <= v370 && v370 < 16);
    assert("Tensor range check" && 0 <= v369 && v369 < 16);
    int v377;
    v377 = blockIdx.x;
    int v378;
    v378 = v377;
    while (while_method_2(v378)){
        bool v380;
        v380 = 0 <= v378;
        bool v381;
        v381 = v380 == false;
        if (v381){
            assert("The index needs to be zero or positive." && v380);
        } else {
        }
        bool v383;
        v383 = v378 < 8;
        bool v384;
        v384 = v383 == false;
        if (v384){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v383);
        } else {
        }
        assert("Tensor range check" && 0 <= v378 && v378 < 8);
        int v386;
        v386 = 1024 * v378;
        int v387;
        v387 = v386 + v376;
        float v388[4];
        int v389[4];
        int v390;
        v390 = 0;
        while (while_method_3(v390)){
            assert("Tensor range check" && 0 <= v390 && v390 < 1);
            int v392;
            v392 = 4 * v390;
            assert("Tensor range check" && 0 <= v390 && v390 < 1);
            int v393;
            v393 = 64 * v390;
            int v394;
            v394 = v393 + v387;
            int4* v395;
            v395 = reinterpret_cast<int4*>(v1 + v394);
            int4* v396;
            v396 = reinterpret_cast<int4*>(v388 + v392);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v395) % 16 == 0 && reinterpret_cast<unsigned long long>(v396) % 16 == 0);
            *v396 = *v395;
            v390 += 1 ;
        }
        int v397;
        v397 = 0;
        while (while_method_3(v397)){
            int v399;
            v399 = 0;
            while (while_method_1(v399)){
                bool v401;
                v401 = 0 <= v399;
                bool v403;
                if (v401){
                    bool v402;
                    v402 = v399 < 4;
                    v403 = v402;
                } else {
                    v403 = false;
                }
                bool v404;
                v404 = v403 == false;
                if (v404){
                    assert("The indices should be inside the range of the dimension." && v403);
                } else {
                }
                bool v406;
                v406 = 0 <= v369;
                bool v408;
                if (v406){
                    bool v407;
                    v407 = v369 < 16;
                    v408 = v407;
                } else {
                    v408 = false;
                }
                bool v409;
                v409 = v408 == false;
                if (v409){
                    assert("The indices should be inside the range of the dimension." && v408);
                } else {
                }
                int v411;
                v411 = v369 * 4;
                int v412;
                v412 = v399 + v411;
                bool v413;
                v413 = 0 <= v397;
                bool v415;
                if (v413){
                    bool v414;
                    v414 = v397 < 1;
                    v415 = v414;
                } else {
                    v415 = false;
                }
                bool v416;
                v416 = v415 == false;
                if (v416){
                    assert("The indices should be inside the range of the dimension." && v415);
                } else {
                }
                int v418;
                v418 = v397 * 64;
                int v419;
                v419 = v412 + v418;
                assert("Tensor range check" && 0 <= v397 && v397 < 1);
                assert("Tensor range check" && 0 <= v399 && v399 < 4);
                int v420;
                v420 = 4 * v397;
                int v421;
                v421 = v420 + v399;
                v389[v421] = v419;
                v399 += 1 ;
            }
            v397 += 1 ;
        }
        bool v422;
        v422 = 0 <= v370;
        bool v423;
        v423 = v422 && v371;
        bool v424;
        v424 = v423 == false;
        if (v424){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v423);
        } else {
        }
        bool v426;
        v426 = v380 && v383;
        bool v427;
        v427 = v426 == false;
        if (v427){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v426);
        } else {
        }
        int v429;
        v429 = v378 * 16;
        int v430;
        v430 = v429 + v370;
        float v431[4];
        int v432;
        v432 = 0;
        while (while_method_3(v432)){
            int v434;
            v434 = 0;
            while (while_method_1(v434)){
                assert("Tensor range check" && 0 <= v432 && v432 < 1);
                assert("Tensor range check" && 0 <= v434 && v434 < 4);
                int v436;
                v436 = 4 * v432;
                int v437;
                v437 = v436 + v434;
                float v438;
                v438 = v388[v437];
                float v439;
                v439 = v438 * v438;
                assert("Tensor range check" && 0 <= v432 && v432 < 1);
                assert("Tensor range check" && 0 <= v434 && v434 < 4);
                v431[v437] = v439;
                v434 += 1 ;
            }
            v432 += 1 ;
        }
        float v440;
        v440 = 0.0f;
        int v441;
        v441 = 0;
        while (while_method_3(v441)){
            int v443;
            v443 = 0;
            while (while_method_1(v443)){
                assert("Tensor range check" && 0 <= v441 && v441 < 1);
                assert("Tensor range check" && 0 <= v443 && v443 < 4);
                int v445;
                v445 = 4 * v441;
                int v446;
                v446 = v445 + v443;
                float v447;
                v447 = v431[v446];
                float v448;
                v448 = v440 + v447;
                v440 = v448;
                v443 += 1 ;
            }
            v441 += 1 ;
        }
        auto v449 = cooperative_groups::coalesced_threads();
        int v450;
        v450 = threadIdx.x;
        int v451;
        v451 = v450 / 16;
        auto v452 = cooperative_groups::labeled_partition(v449,v451);
        Closure0 v453{};
        float v454;
        v454 = cooperative_groups::reduce(v452, v440, v453);
        float v455[4];
        int v456;
        v456 = 0;
        while (while_method_3(v456)){
            int v458;
            v458 = 0;
            while (while_method_1(v458)){
                assert("Tensor range check" && 0 <= v456 && v456 < 1);
                assert("Tensor range check" && 0 <= v458 && v458 < 4);
                int v460;
                v460 = 4 * v456;
                int v461;
                v461 = v460 + v458;
                float v462;
                v462 = v388[v461];
                bool v463;
                v463 = v454 == 0.0f;
                bool v464;
                v464 = v463 != true;
                float v466;
                if (v464){
                    float v465;
                    v465 = v462 / v454;
                    v466 = v465;
                } else {
                    v466 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v456 && v456 < 1);
                assert("Tensor range check" && 0 <= v458 && v458 < 4);
                v455[v461] = v466;
                v458 += 1 ;
            }
            v456 += 1 ;
        }
        assert("Tensor range check" && 0 <= v378 && v378 < 8);
        int v467;
        v467 = 0;
        while (while_method_3(v467)){
            assert("Tensor range check" && 0 <= v467 && v467 < 1);
            int v469;
            v469 = 64 * v467;
            int v470;
            v470 = v469 + v387;
            assert("Tensor range check" && 0 <= v467 && v467 < 1);
            int v471;
            v471 = 4 * v467;
            int4* v472;
            v472 = reinterpret_cast<int4*>(v455 + v471);
            int4* v473;
            v473 = reinterpret_cast<int4*>(v7 + v470);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v472) % 16 == 0 && reinterpret_cast<unsigned long long>(v473) % 16 == 0);
            *v473 = *v472;
            v467 += 1 ;
        }
        v378 += 24 ;
    }
    v17.sync() ;
    int v474;
    v474 = threadIdx.x;
    bool v475;
    v475 = 0 <= v474;
    bool v476;
    v476 = v475 == false;
    if (v476){
        assert("The index needs to be zero or positive." && v475);
    } else {
    }
    int v478;
    v478 = v474 % 16;
    int v479;
    v479 = v474 / 16;
    bool v480;
    v480 = v479 < 16;
    bool v481;
    v481 = v480 == false;
    if (v481){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v480);
    } else {
    }
    assert("Tensor range check" && 0 <= v479 && v479 < 16);
    assert("Tensor range check" && 0 <= v478 && v478 < 16);
    int v483;
    v483 = 4 * v478;
    int v484;
    v484 = 64 * v479;
    int v485;
    v485 = v484 + v483;
    assert("Tensor range check" && 0 <= v479 && v479 < 16);
    int v486;
    v486 = blockIdx.x;
    int v487;
    v487 = v486;
    while (while_method_2(v487)){
        bool v489;
        v489 = 0 <= v487;
        bool v490;
        v490 = v489 == false;
        if (v490){
            assert("The index needs to be zero or positive." && v489);
        } else {
        }
        bool v492;
        v492 = v487 < 8;
        bool v493;
        v493 = v492 == false;
        if (v493){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v492);
        } else {
        }
        assert("Tensor range check" && 0 <= v487 && v487 < 8);
        int v495;
        v495 = 1024 * v487;
        int v496;
        v496 = v495 + v485;
        float v497[4];
        int v498[4];
        int v499;
        v499 = 0;
        while (while_method_3(v499)){
            assert("Tensor range check" && 0 <= v499 && v499 < 1);
            int v501;
            v501 = 4 * v499;
            assert("Tensor range check" && 0 <= v499 && v499 < 1);
            int v502;
            v502 = 64 * v499;
            int v503;
            v503 = v502 + v496;
            int4* v504;
            v504 = reinterpret_cast<int4*>(v1 + v503);
            int4* v505;
            v505 = reinterpret_cast<int4*>(v497 + v501);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v504) % 16 == 0 && reinterpret_cast<unsigned long long>(v505) % 16 == 0);
            *v505 = *v504;
            v499 += 1 ;
        }
        int v506;
        v506 = 0;
        while (while_method_3(v506)){
            int v508;
            v508 = 0;
            while (while_method_1(v508)){
                bool v510;
                v510 = 0 <= v508;
                bool v512;
                if (v510){
                    bool v511;
                    v511 = v508 < 4;
                    v512 = v511;
                } else {
                    v512 = false;
                }
                bool v513;
                v513 = v512 == false;
                if (v513){
                    assert("The indices should be inside the range of the dimension." && v512);
                } else {
                }
                bool v515;
                v515 = 0 <= v478;
                bool v517;
                if (v515){
                    bool v516;
                    v516 = v478 < 16;
                    v517 = v516;
                } else {
                    v517 = false;
                }
                bool v518;
                v518 = v517 == false;
                if (v518){
                    assert("The indices should be inside the range of the dimension." && v517);
                } else {
                }
                int v520;
                v520 = v478 * 4;
                int v521;
                v521 = v508 + v520;
                bool v522;
                v522 = 0 <= v506;
                bool v524;
                if (v522){
                    bool v523;
                    v523 = v506 < 1;
                    v524 = v523;
                } else {
                    v524 = false;
                }
                bool v525;
                v525 = v524 == false;
                if (v525){
                    assert("The indices should be inside the range of the dimension." && v524);
                } else {
                }
                int v527;
                v527 = v506 * 64;
                int v528;
                v528 = v521 + v527;
                assert("Tensor range check" && 0 <= v506 && v506 < 1);
                assert("Tensor range check" && 0 <= v508 && v508 < 4);
                int v529;
                v529 = 4 * v506;
                int v530;
                v530 = v529 + v508;
                v498[v530] = v528;
                v508 += 1 ;
            }
            v506 += 1 ;
        }
        bool v531;
        v531 = 0 <= v479;
        bool v532;
        v532 = v531 && v480;
        bool v533;
        v533 = v532 == false;
        if (v533){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v532);
        } else {
        }
        bool v535;
        v535 = v489 && v492;
        bool v536;
        v536 = v535 == false;
        if (v536){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v535);
        } else {
        }
        int v538;
        v538 = v487 * 16;
        int v539;
        v539 = v538 + v479;
        float v540; int v541;
        Tuple1 tmp65 = Tuple1{-1.0f / 0.0f, 0};
        v540 = tmp65.v0; v541 = tmp65.v1;
        int v542;
        v542 = 0;
        while (while_method_3(v542)){
            int v544;
            v544 = 0;
            while (while_method_1(v544)){
                assert("Tensor range check" && 0 <= v542 && v542 < 1);
                assert("Tensor range check" && 0 <= v544 && v544 < 4);
                int v546;
                v546 = 4 * v542;
                int v547;
                v547 = v546 + v544;
                float v548;
                v548 = v497[v547];
                int v549;
                v549 = v498[v547];
                bool v550;
                v550 = v540 > v548;
                float v551; int v552;
                if (v550){
                    v551 = v540; v552 = v541;
                } else {
                    v551 = v548; v552 = v549;
                }
                v540 = v551;
                v541 = v552;
                v544 += 1 ;
            }
            v542 += 1 ;
        }
        auto v553 = cooperative_groups::coalesced_threads();
        int v554;
        v554 = threadIdx.x;
        int v555;
        v555 = v554 / 16;
        auto v556 = cooperative_groups::labeled_partition(v553,v555);
        Closure1 v557{};
        float v558; int v559;
        Tuple1 tmp66 = cooperative_groups::reduce(v556, Tuple1{v540, v541}, v557);
        v558 = tmp66.v0; v559 = tmp66.v1;
        assert("Tensor range check" && 0 <= v487 && v487 < 8);
        int v560;
        v560 = 16 * v487;
        int v561;
        v561 = v560 + v479;
        v8[v561] = v559;
        v487 += 24 ;
    }
    v17.sync() ;
    int v562;
    v562 = threadIdx.x;
    bool v563;
    v563 = 0 <= v562;
    bool v564;
    v564 = v563 == false;
    if (v564){
        assert("The index needs to be zero or positive." && v563);
    } else {
    }
    int v566;
    v566 = v562 % 16;
    int v567;
    v567 = v562 / 16;
    bool v568;
    v568 = v567 < 16;
    bool v569;
    v569 = v568 == false;
    if (v569){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v568);
    } else {
    }
    assert("Tensor range check" && 0 <= v567 && v567 < 16);
    assert("Tensor range check" && 0 <= v566 && v566 < 16);
    int v571;
    v571 = 4 * v566;
    int v572;
    v572 = 64 * v567;
    int v573;
    v573 = v572 + v571;
    assert("Tensor range check" && 0 <= v567 && v567 < 16);
    assert("Tensor range check" && 0 <= v566 && v566 < 16);
    int v574;
    v574 = blockIdx.x;
    int v575;
    v575 = v574;
    while (while_method_2(v575)){
        bool v577;
        v577 = 0 <= v575;
        bool v578;
        v578 = v577 == false;
        if (v578){
            assert("The index needs to be zero or positive." && v577);
        } else {
        }
        bool v580;
        v580 = v575 < 8;
        bool v581;
        v581 = v580 == false;
        if (v581){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v580);
        } else {
        }
        assert("Tensor range check" && 0 <= v575 && v575 < 8);
        int v583;
        v583 = 1024 * v575;
        int v584;
        v584 = v583 + v573;
        float v585[4];
        int v586[4];
        int v587;
        v587 = 0;
        while (while_method_3(v587)){
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v589;
            v589 = 4 * v587;
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v590;
            v590 = 64 * v587;
            int v591;
            v591 = v590 + v584;
            int4* v592;
            v592 = reinterpret_cast<int4*>(v1 + v591);
            int4* v593;
            v593 = reinterpret_cast<int4*>(v585 + v589);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v592) % 16 == 0 && reinterpret_cast<unsigned long long>(v593) % 16 == 0);
            *v593 = *v592;
            v587 += 1 ;
        }
        int v594;
        v594 = 0;
        while (while_method_3(v594)){
            int v596;
            v596 = 0;
            while (while_method_1(v596)){
                bool v598;
                v598 = 0 <= v596;
                bool v600;
                if (v598){
                    bool v599;
                    v599 = v596 < 4;
                    v600 = v599;
                } else {
                    v600 = false;
                }
                bool v601;
                v601 = v600 == false;
                if (v601){
                    assert("The indices should be inside the range of the dimension." && v600);
                } else {
                }
                bool v603;
                v603 = 0 <= v566;
                bool v605;
                if (v603){
                    bool v604;
                    v604 = v566 < 16;
                    v605 = v604;
                } else {
                    v605 = false;
                }
                bool v606;
                v606 = v605 == false;
                if (v606){
                    assert("The indices should be inside the range of the dimension." && v605);
                } else {
                }
                int v608;
                v608 = v566 * 4;
                int v609;
                v609 = v596 + v608;
                bool v610;
                v610 = 0 <= v594;
                bool v612;
                if (v610){
                    bool v611;
                    v611 = v594 < 1;
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
                int v615;
                v615 = v594 * 64;
                int v616;
                v616 = v609 + v615;
                assert("Tensor range check" && 0 <= v594 && v594 < 1);
                assert("Tensor range check" && 0 <= v596 && v596 < 4);
                int v617;
                v617 = 4 * v594;
                int v618;
                v618 = v617 + v596;
                v586[v618] = v616;
                v596 += 1 ;
            }
            v594 += 1 ;
        }
        bool v619;
        v619 = 0 <= v567;
        bool v620;
        v620 = v619 && v568;
        bool v621;
        v621 = v620 == false;
        if (v621){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v620);
        } else {
        }
        bool v623;
        v623 = v577 && v580;
        bool v624;
        v624 = v623 == false;
        if (v624){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v623);
        } else {
        }
        int v626;
        v626 = v575 * 16;
        int v627;
        v627 = v626 + v567;
        float v628;
        v628 = 0.0f;
        int v629;
        v629 = 0;
        while (while_method_3(v629)){
            int v631;
            v631 = 0;
            while (while_method_1(v631)){
                assert("Tensor range check" && 0 <= v629 && v629 < 1);
                assert("Tensor range check" && 0 <= v631 && v631 < 4);
                int v633;
                v633 = 4 * v629;
                int v634;
                v634 = v633 + v631;
                float v635;
                v635 = v585[v634];
                float v636;
                v636 = v628 + v635;
                v628 = v636;
                v631 += 1 ;
            }
            v629 += 1 ;
        }
        auto v637 = cooperative_groups::coalesced_threads();
        int v638;
        v638 = threadIdx.x;
        int v639;
        v639 = v638 / 16;
        auto v640 = cooperative_groups::labeled_partition(v637,v639);
        Closure0 v641{};
        float v642;
        v642 = cooperative_groups::reduce(v640, v628, v641);
        float v643;
        v643 = v642 / 64.0f;
        float v644[4];
        int v645;
        v645 = 0;
        while (while_method_3(v645)){
            int v647;
            v647 = 0;
            while (while_method_1(v647)){
                assert("Tensor range check" && 0 <= v645 && v645 < 1);
                assert("Tensor range check" && 0 <= v647 && v647 < 4);
                int v649;
                v649 = 4 * v645;
                int v650;
                v650 = v649 + v647;
                float v651;
                v651 = v585[v650];
                float v652;
                v652 = v651 - v643;
                float v653;
                v653 = exp(v652);
                assert("Tensor range check" && 0 <= v645 && v645 < 1);
                assert("Tensor range check" && 0 <= v647 && v647 < 4);
                v644[v650] = v653;
                v647 += 1 ;
            }
            v645 += 1 ;
        }
        float v654;
        v654 = 0.0f;
        int v655;
        v655 = 0;
        while (while_method_3(v655)){
            int v657;
            v657 = 0;
            while (while_method_1(v657)){
                assert("Tensor range check" && 0 <= v655 && v655 < 1);
                assert("Tensor range check" && 0 <= v657 && v657 < 4);
                int v659;
                v659 = 4 * v655;
                int v660;
                v660 = v659 + v657;
                float v661;
                v661 = v644[v660];
                float v662;
                v662 = v654 + v661;
                v654 = v662;
                v657 += 1 ;
            }
            v655 += 1 ;
        }
        auto v663 = cooperative_groups::coalesced_threads();
        int v664;
        v664 = threadIdx.x;
        int v665;
        v665 = v664 / 16;
        auto v666 = cooperative_groups::labeled_partition(v663,v665);
        float v667;
        v667 = cooperative_groups::reduce(v666, v654, v641);
        float v668[4];
        int v669;
        v669 = 0;
        while (while_method_3(v669)){
            int v671;
            v671 = 0;
            while (while_method_1(v671)){
                assert("Tensor range check" && 0 <= v669 && v669 < 1);
                assert("Tensor range check" && 0 <= v671 && v671 < 4);
                int v673;
                v673 = 4 * v669;
                int v674;
                v674 = v673 + v671;
                float v675;
                v675 = v644[v674];
                float v676;
                v676 = v675 / v667;
                assert("Tensor range check" && 0 <= v669 && v669 < 1);
                assert("Tensor range check" && 0 <= v671 && v671 < 4);
                v668[v674] = v676;
                v671 += 1 ;
            }
            v669 += 1 ;
        }
        float v677[4];
        float v678;
        v678 = 0.0f;
        int v679;
        v679 = 0;
        while (while_method_3(v679)){
            assert("Tensor range check" && 0 <= v679 && v679 < 1);
            int v681;
            v681 = 4 * v679;
            assert("Tensor range check" && 0 <= v679 && v679 < 1);
            int v682; float v683;
            Tuple0 tmp67 = Tuple0{0, 0.0f};
            v682 = tmp67.v0; v683 = tmp67.v1;
            while (while_method_1(v682)){
                assert("Tensor range check" && 0 <= v682 && v682 < 4);
                int v685;
                v685 = v682 + v681;
                float v686;
                v686 = v668[v685];
                float v687;
                v687 = v683 + v686;
                v683 = v687;
                v682 += 1 ;
            }
            auto v688 = cooperative_groups::coalesced_threads();
            int v689;
            v689 = threadIdx.x;
            int v690;
            v690 = v689 / 16;
            auto v691 = cooperative_groups::labeled_partition(v688,v690);
            Closure2 v692{};
            float v693;
            v693 = cooperative_groups::inclusive_scan(v691, v683, v692);
            float v694;
            v694 = v691.shfl_up(v693,1);
            bool v695;
            v695 = v691.thread_rank() == 0;
            float v696;
            if (v695){
                v696 = 0.0f;
            } else {
                v696 = v694;
            }
            float v697;
            v697 = v691.shfl(v693,v691.num_threads()-1);
            float v698;
            v698 = v678 + v696;
            int v699; float v700;
            Tuple0 tmp68 = Tuple0{0, v698};
            v699 = tmp68.v0; v700 = tmp68.v1;
            while (while_method_1(v699)){
                assert("Tensor range check" && 0 <= v699 && v699 < 4);
                int v702;
                v702 = v699 + v681;
                float v703;
                v703 = v668[v702];
                float v704;
                v704 = v700 + v703;
                assert("Tensor range check" && 0 <= v699 && v699 < 4);
                v677[v702] = v704;
                v700 = v704;
                v699 += 1 ;
            }
            float v705;
            v705 = v678 + v697;
            v678 = v705;
            v679 += 1 ;
        }
        assert("Tensor range check" && 0 <= v575 && v575 < 8);
        int v706;
        v706 = 0;
        while (while_method_3(v706)){
            assert("Tensor range check" && 0 <= v706 && v706 < 1);
            int v708;
            v708 = 64 * v706;
            int v709;
            v709 = v708 + v584;
            assert("Tensor range check" && 0 <= v706 && v706 < 1);
            int v710;
            v710 = 4 * v706;
            int4* v711;
            v711 = reinterpret_cast<int4*>(v668 + v710);
            int4* v712;
            v712 = reinterpret_cast<int4*>(v5 + v709);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v711) % 16 == 0 && reinterpret_cast<unsigned long long>(v712) % 16 == 0);
            *v712 = *v711;
            int4* v713;
            v713 = reinterpret_cast<int4*>(v677 + v710);
            int4* v714;
            v714 = reinterpret_cast<int4*>(v6 + v709);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v713) % 16 == 0 && reinterpret_cast<unsigned long long>(v714) % 16 == 0);
            *v714 = *v713;
            v706 += 1 ;
        }
        v575 += 24 ;
    }
    v17.sync() ;
    int v715;
    v715 = threadIdx.x;
    bool v716;
    v716 = 0 <= v715;
    bool v717;
    v717 = v716 == false;
    if (v717){
        assert("The index needs to be zero or positive." && v716);
    } else {
    }
    int v719;
    v719 = v715 % 16;
    int v720;
    v720 = v715 / 16;
    bool v721;
    v721 = v720 < 16;
    bool v722;
    v722 = v721 == false;
    if (v722){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v721);
    } else {
    }
    assert("Tensor range check" && 0 <= v720 && v720 < 16);
    assert("Tensor range check" && 0 <= v719 && v719 < 16);
    int v724;
    v724 = 4 * v719;
    int v725;
    v725 = 64 * v720;
    int v726;
    v726 = v725 + v724;
    assert("Tensor range check" && 0 <= v720 && v720 < 16);
    assert("Tensor range check" && 0 <= v719 && v719 < 16);
    int v727;
    v727 = blockIdx.x;
    int v728;
    v728 = v727;
    while (while_method_2(v728)){
        bool v730;
        v730 = 0 <= v728;
        bool v731;
        v731 = v730 == false;
        if (v731){
            assert("The index needs to be zero or positive." && v730);
        } else {
        }
        bool v733;
        v733 = v728 < 8;
        bool v734;
        v734 = v733 == false;
        if (v734){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v733);
        } else {
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v736;
        v736 = 1024 * v728;
        int v737;
        v737 = v736 + v726;
        int v738[4];
        int v739[4];
        int v740;
        v740 = 0;
        while (while_method_3(v740)){
            assert("Tensor range check" && 0 <= v740 && v740 < 1);
            int v742;
            v742 = 4 * v740;
            assert("Tensor range check" && 0 <= v740 && v740 < 1);
            int v743;
            v743 = 64 * v740;
            int v744;
            v744 = v743 + v737;
            int4* v745;
            v745 = reinterpret_cast<int4*>(v0 + v744);
            int4* v746;
            v746 = reinterpret_cast<int4*>(v738 + v742);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v745) % 16 == 0 && reinterpret_cast<unsigned long long>(v746) % 16 == 0);
            *v746 = *v745;
            v740 += 1 ;
        }
        int v747;
        v747 = 0;
        while (while_method_3(v747)){
            int v749;
            v749 = 0;
            while (while_method_1(v749)){
                bool v751;
                v751 = 0 <= v749;
                bool v753;
                if (v751){
                    bool v752;
                    v752 = v749 < 4;
                    v753 = v752;
                } else {
                    v753 = false;
                }
                bool v754;
                v754 = v753 == false;
                if (v754){
                    assert("The indices should be inside the range of the dimension." && v753);
                } else {
                }
                bool v756;
                v756 = 0 <= v719;
                bool v758;
                if (v756){
                    bool v757;
                    v757 = v719 < 16;
                    v758 = v757;
                } else {
                    v758 = false;
                }
                bool v759;
                v759 = v758 == false;
                if (v759){
                    assert("The indices should be inside the range of the dimension." && v758);
                } else {
                }
                int v761;
                v761 = v719 * 4;
                int v762;
                v762 = v749 + v761;
                bool v763;
                v763 = 0 <= v747;
                bool v765;
                if (v763){
                    bool v764;
                    v764 = v747 < 1;
                    v765 = v764;
                } else {
                    v765 = false;
                }
                bool v766;
                v766 = v765 == false;
                if (v766){
                    assert("The indices should be inside the range of the dimension." && v765);
                } else {
                }
                int v768;
                v768 = v747 * 64;
                int v769;
                v769 = v762 + v768;
                assert("Tensor range check" && 0 <= v747 && v747 < 1);
                assert("Tensor range check" && 0 <= v749 && v749 < 4);
                int v770;
                v770 = 4 * v747;
                int v771;
                v771 = v770 + v749;
                v739[v771] = v769;
                v749 += 1 ;
            }
            v747 += 1 ;
        }
        bool v772;
        v772 = 0 <= v720;
        bool v773;
        v773 = v772 && v721;
        bool v774;
        v774 = v773 == false;
        if (v774){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v773);
        } else {
        }
        bool v776;
        v776 = v730 && v733;
        bool v777;
        v777 = v776 == false;
        if (v777){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v776);
        } else {
        }
        int v779;
        v779 = v728 * 16;
        int v780;
        v780 = v779 + v720;
        int v781[4];
        int v782;
        v782 = 0;
        int v783;
        v783 = 0;
        while (while_method_3(v783)){
            assert("Tensor range check" && 0 <= v783 && v783 < 1);
            int v785;
            v785 = 4 * v783;
            assert("Tensor range check" && 0 <= v783 && v783 < 1);
            int v786; int v787;
            Tuple2 tmp69 = Tuple2{0, 0};
            v786 = tmp69.v0; v787 = tmp69.v1;
            while (while_method_1(v786)){
                assert("Tensor range check" && 0 <= v786 && v786 < 4);
                int v789;
                v789 = v786 + v785;
                int v790;
                v790 = v738[v789];
                int v791;
                v791 = v787 + v790;
                v787 = v791;
                v786 += 1 ;
            }
            auto v792 = cooperative_groups::coalesced_threads();
            int v793;
            v793 = threadIdx.x;
            int v794;
            v794 = v793 / 16;
            auto v795 = cooperative_groups::labeled_partition(v792,v794);
            Closure3 v796{};
            int v797;
            v797 = cooperative_groups::inclusive_scan(v795, v787, v796);
            int v798;
            v798 = v795.shfl_up(v797,1);
            bool v799;
            v799 = v795.thread_rank() == 0;
            int v800;
            if (v799){
                v800 = 0;
            } else {
                v800 = v798;
            }
            int v801;
            v801 = v795.shfl(v797,v795.num_threads()-1);
            int v802;
            v802 = v782 + v800;
            int v803; int v804;
            Tuple2 tmp70 = Tuple2{0, v802};
            v803 = tmp70.v0; v804 = tmp70.v1;
            while (while_method_1(v803)){
                assert("Tensor range check" && 0 <= v803 && v803 < 4);
                int v806;
                v806 = v803 + v785;
                int v807;
                v807 = v738[v806];
                assert("Tensor range check" && 0 <= v803 && v803 < 4);
                v781[v806] = v804;
                int v808;
                v808 = v804 + v807;
                v804 = v808;
                v803 += 1 ;
            }
            int v809;
            v809 = v782 + v801;
            v782 = v809;
            v783 += 1 ;
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v810;
        v810 = 0;
        while (while_method_3(v810)){
            assert("Tensor range check" && 0 <= v810 && v810 < 1);
            int v812;
            v812 = 64 * v810;
            int v813;
            v813 = v812 + v737;
            assert("Tensor range check" && 0 <= v810 && v810 < 1);
            int v814;
            v814 = 4 * v810;
            int4* v815;
            v815 = reinterpret_cast<int4*>(v781 + v814);
            int4* v816;
            v816 = reinterpret_cast<int4*>(v12 + v813);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v815) % 16 == 0 && reinterpret_cast<unsigned long long>(v816) % 16 == 0);
            *v816 = *v815;
            v810 += 1 ;
        }
        v728 += 24 ;
    }
    v17.sync() ;
    int v817;
    v817 = threadIdx.x;
    bool v818;
    v818 = 0 <= v817;
    bool v819;
    v819 = v818 == false;
    if (v819){
        assert("The index needs to be zero or positive." && v818);
    } else {
    }
    int v821;
    v821 = v817 % 16;
    int v822;
    v822 = v817 / 16;
    bool v823;
    v823 = v822 < 16;
    bool v824;
    v824 = v823 == false;
    if (v824){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v823);
    } else {
    }
    assert("Tensor range check" && 0 <= v822 && v822 < 16);
    assert("Tensor range check" && 0 <= v821 && v821 < 16);
    int v826;
    v826 = 4 * v821;
    int v827;
    v827 = 64 * v822;
    int v828;
    v828 = v827 + v826;
    assert("Tensor range check" && 0 <= v822 && v822 < 16);
    assert("Tensor range check" && 0 <= v821 && v821 < 16);
    int v829;
    v829 = blockIdx.x;
    int v830;
    v830 = v829;
    while (while_method_2(v830)){
        bool v832;
        v832 = 0 <= v830;
        bool v833;
        v833 = v832 == false;
        if (v833){
            assert("The index needs to be zero or positive." && v832);
        } else {
        }
        bool v835;
        v835 = v830 < 8;
        bool v836;
        v836 = v835 == false;
        if (v836){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v835);
        } else {
        }
        assert("Tensor range check" && 0 <= v830 && v830 < 8);
        int v838;
        v838 = 1024 * v830;
        int v839;
        v839 = v838 + v828;
        float v840[4];
        int v841[4];
        int v842;
        v842 = 0;
        while (while_method_3(v842)){
            assert("Tensor range check" && 0 <= v842 && v842 < 1);
            int v844;
            v844 = 4 * v842;
            assert("Tensor range check" && 0 <= v842 && v842 < 1);
            int v845;
            v845 = 64 * v842;
            int v846;
            v846 = v845 + v839;
            int4* v847;
            v847 = reinterpret_cast<int4*>(v1 + v846);
            int4* v848;
            v848 = reinterpret_cast<int4*>(v840 + v844);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v847) % 16 == 0 && reinterpret_cast<unsigned long long>(v848) % 16 == 0);
            *v848 = *v847;
            v842 += 1 ;
        }
        int v849;
        v849 = 0;
        while (while_method_3(v849)){
            int v851;
            v851 = 0;
            while (while_method_1(v851)){
                bool v853;
                v853 = 0 <= v851;
                bool v855;
                if (v853){
                    bool v854;
                    v854 = v851 < 4;
                    v855 = v854;
                } else {
                    v855 = false;
                }
                bool v856;
                v856 = v855 == false;
                if (v856){
                    assert("The indices should be inside the range of the dimension." && v855);
                } else {
                }
                bool v858;
                v858 = 0 <= v821;
                bool v860;
                if (v858){
                    bool v859;
                    v859 = v821 < 16;
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
                int v863;
                v863 = v821 * 4;
                int v864;
                v864 = v851 + v863;
                bool v865;
                v865 = 0 <= v849;
                bool v867;
                if (v865){
                    bool v866;
                    v866 = v849 < 1;
                    v867 = v866;
                } else {
                    v867 = false;
                }
                bool v868;
                v868 = v867 == false;
                if (v868){
                    assert("The indices should be inside the range of the dimension." && v867);
                } else {
                }
                int v870;
                v870 = v849 * 64;
                int v871;
                v871 = v864 + v870;
                assert("Tensor range check" && 0 <= v849 && v849 < 1);
                assert("Tensor range check" && 0 <= v851 && v851 < 4);
                int v872;
                v872 = 4 * v849;
                int v873;
                v873 = v872 + v851;
                v841[v873] = v871;
                v851 += 1 ;
            }
            v849 += 1 ;
        }
        bool v874;
        v874 = 0 <= v822;
        bool v875;
        v875 = v874 && v823;
        bool v876;
        v876 = v875 == false;
        if (v876){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v875);
        } else {
        }
        bool v878;
        v878 = v832 && v835;
        bool v879;
        v879 = v878 == false;
        if (v879){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v878);
        } else {
        }
        int v881;
        v881 = v830 * 16;
        int v882;
        v882 = v881 + v822;
        bool v883[4];
        int v884;
        v884 = 0;
        while (while_method_3(v884)){
            int v886;
            v886 = 0;
            while (while_method_1(v886)){
                assert("Tensor range check" && 0 <= v884 && v884 < 1);
                assert("Tensor range check" && 0 <= v886 && v886 < 4);
                int v888;
                v888 = 4 * v884;
                int v889;
                v889 = v888 + v886;
                float v890;
                v890 = v840[v889];
                int v891;
                v891 = v841[v889];
                bool v892;
                v892 = v891 < 4;
                assert("Tensor range check" && 0 <= v884 && v884 < 1);
                assert("Tensor range check" && 0 <= v886 && v886 < 4);
                v883[v889] = v892;
                v886 += 1 ;
            }
            v884 += 1 ;
        }
        int v893[4];
        int v894;
        v894 = 0;
        while (while_method_3(v894)){
            int v896;
            v896 = 0;
            while (while_method_1(v896)){
                assert("Tensor range check" && 0 <= v894 && v894 < 1);
                assert("Tensor range check" && 0 <= v896 && v896 < 4);
                int v898;
                v898 = 4 * v894;
                int v899;
                v899 = v898 + v896;
                bool v900;
                v900 = v883[v899];
                int v901;
                if (v900){
                    v901 = 1;
                } else {
                    v901 = 0;
                }
                assert("Tensor range check" && 0 <= v894 && v894 < 1);
                assert("Tensor range check" && 0 <= v896 && v896 < 4);
                v893[v899] = v901;
                v896 += 1 ;
            }
            v894 += 1 ;
        }
        int v902;
        v902 = 0;
        int v903;
        v903 = 0;
        while (while_method_3(v903)){
            int v905;
            v905 = 0;
            while (while_method_1(v905)){
                assert("Tensor range check" && 0 <= v903 && v903 < 1);
                assert("Tensor range check" && 0 <= v905 && v905 < 4);
                int v907;
                v907 = 4 * v903;
                int v908;
                v908 = v907 + v905;
                int v909;
                v909 = v893[v908];
                int v910;
                v910 = v902 + v909;
                v902 = v910;
                v905 += 1 ;
            }
            v903 += 1 ;
        }
        auto v911 = cooperative_groups::coalesced_threads();
        int v912;
        v912 = threadIdx.x;
        int v913;
        v913 = v912 / 16;
        auto v914 = cooperative_groups::labeled_partition(v911,v913);
        Closure4 v915{};
        int v916;
        v916 = cooperative_groups::reduce(v914, v902, v915);
        float v917[4];
        int v918;
        v918 = 0;
        while (while_method_3(v918)){
            int v920;
            v920 = 0;
            while (while_method_1(v920)){
                assert("Tensor range check" && 0 <= v918 && v918 < 1);
                assert("Tensor range check" && 0 <= v920 && v920 < 4);
                int v922;
                v922 = 4 * v918;
                int v923;
                v923 = v922 + v920;
                float v924;
                v924 = v840[v923];
                bool v925;
                v925 = v883[v923];
                float v926;
                if (v925){
                    v926 = v924;
                } else {
                    v926 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v918 && v918 < 1);
                assert("Tensor range check" && 0 <= v920 && v920 < 4);
                v917[v923] = v926;
                v920 += 1 ;
            }
            v918 += 1 ;
        }
        float v927;
        v927 = 0.0f;
        int v928;
        v928 = 0;
        while (while_method_3(v928)){
            int v930;
            v930 = 0;
            while (while_method_1(v930)){
                assert("Tensor range check" && 0 <= v928 && v928 < 1);
                assert("Tensor range check" && 0 <= v930 && v930 < 4);
                int v932;
                v932 = 4 * v928;
                int v933;
                v933 = v932 + v930;
                float v934;
                v934 = v917[v933];
                float v935;
                v935 = v927 + v934;
                v927 = v935;
                v930 += 1 ;
            }
            v928 += 1 ;
        }
        auto v936 = cooperative_groups::coalesced_threads();
        int v937;
        v937 = threadIdx.x;
        int v938;
        v938 = v937 / 16;
        auto v939 = cooperative_groups::labeled_partition(v936,v938);
        Closure0 v940{};
        float v941;
        v941 = cooperative_groups::reduce(v939, v927, v940);
        float v942;
        v942 = (float)v916;
        float v943;
        v943 = v941 / v942;
        float v944[4];
        int v945;
        v945 = 0;
        while (while_method_3(v945)){
            int v947;
            v947 = 0;
            while (while_method_1(v947)){
                assert("Tensor range check" && 0 <= v945 && v945 < 1);
                assert("Tensor range check" && 0 <= v947 && v947 < 4);
                int v949;
                v949 = 4 * v945;
                int v950;
                v950 = v949 + v947;
                float v951;
                v951 = v840[v950];
                bool v952;
                v952 = v883[v950];
                float v953;
                if (v952){
                    v953 = v951;
                } else {
                    v953 = -1.0f / 0.0f;
                }
                float v954;
                v954 = v953 - v943;
                float v955;
                v955 = exp(v954);
                assert("Tensor range check" && 0 <= v945 && v945 < 1);
                assert("Tensor range check" && 0 <= v947 && v947 < 4);
                v944[v950] = v955;
                v947 += 1 ;
            }
            v945 += 1 ;
        }
        float v956;
        v956 = 0.0f;
        int v957;
        v957 = 0;
        while (while_method_3(v957)){
            int v959;
            v959 = 0;
            while (while_method_1(v959)){
                assert("Tensor range check" && 0 <= v957 && v957 < 1);
                assert("Tensor range check" && 0 <= v959 && v959 < 4);
                int v961;
                v961 = 4 * v957;
                int v962;
                v962 = v961 + v959;
                float v963;
                v963 = v944[v962];
                float v964;
                v964 = v956 + v963;
                v956 = v964;
                v959 += 1 ;
            }
            v957 += 1 ;
        }
        auto v965 = cooperative_groups::coalesced_threads();
        int v966;
        v966 = threadIdx.x;
        int v967;
        v967 = v966 / 16;
        auto v968 = cooperative_groups::labeled_partition(v965,v967);
        float v969;
        v969 = cooperative_groups::reduce(v968, v956, v940);
        float v970[4];
        int v971;
        v971 = 0;
        while (while_method_3(v971)){
            int v973;
            v973 = 0;
            while (while_method_1(v973)){
                assert("Tensor range check" && 0 <= v971 && v971 < 1);
                assert("Tensor range check" && 0 <= v973 && v973 < 4);
                int v975;
                v975 = 4 * v971;
                int v976;
                v976 = v975 + v973;
                float v977;
                v977 = v944[v976];
                float v978;
                v978 = v977 / v969;
                assert("Tensor range check" && 0 <= v971 && v971 < 1);
                assert("Tensor range check" && 0 <= v973 && v973 < 4);
                v970[v976] = v978;
                v973 += 1 ;
            }
            v971 += 1 ;
        }
        assert("Tensor range check" && 0 <= v830 && v830 < 8);
        int v979;
        v979 = 0;
        while (while_method_3(v979)){
            assert("Tensor range check" && 0 <= v979 && v979 < 1);
            int v981;
            v981 = 64 * v979;
            int v982;
            v982 = v981 + v839;
            assert("Tensor range check" && 0 <= v979 && v979 < 1);
            int v983;
            v983 = 4 * v979;
            int4* v984;
            v984 = reinterpret_cast<int4*>(v970 + v983);
            int4* v985;
            v985 = reinterpret_cast<int4*>(v4 + v982);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v984) % 16 == 0 && reinterpret_cast<unsigned long long>(v985) % 16 == 0);
            *v985 = *v984;
            v979 += 1 ;
        }
        v830 += 24 ;
    }
    v17.sync() ;
    int v986;
    v986 = threadIdx.x;
    int v987;
    v987 = blockIdx.x;
    int v988;
    v988 = v987 * 256;
    int v989;
    v989 = v986 + v988;
    unsigned long long v990;
    v990 = (unsigned long long)v989;
    curandStatePhilox4_32_10_t v991;
    curand_init(12344321ull,v990,0ull,&v991);
    int v992;
    v992 = threadIdx.x;
    bool v993;
    v993 = 0 <= v992;
    bool v994;
    v994 = v993 == false;
    if (v994){
        assert("The index needs to be zero or positive." && v993);
    } else {
    }
    int v996;
    v996 = v992 % 16;
    int v997;
    v997 = v992 / 16;
    bool v998;
    v998 = v997 < 16;
    bool v999;
    v999 = v998 == false;
    if (v999){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v998);
    } else {
    }
    assert("Tensor range check" && 0 <= v997 && v997 < 16);
    assert("Tensor range check" && 0 <= v996 && v996 < 16);
    int v1001;
    v1001 = 4 * v996;
    int v1002;
    v1002 = 64 * v997;
    int v1003;
    v1003 = v1002 + v1001;
    assert("Tensor range check" && 0 <= v997 && v997 < 16);
    assert("Tensor range check" && 0 <= v996 && v996 < 16);
    assert("Tensor range check" && 0 <= v997 && v997 < 16);
    int v1004;
    v1004 = blockIdx.x;
    int v1005;
    v1005 = v1004;
    while (while_method_2(v1005)){
        bool v1007;
        v1007 = 0 <= v1005;
        bool v1008;
        v1008 = v1007 == false;
        if (v1008){
            assert("The index needs to be zero or positive." && v1007);
        } else {
        }
        bool v1010;
        v1010 = v1005 < 8;
        bool v1011;
        v1011 = v1010 == false;
        if (v1011){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1010);
        } else {
        }
        assert("Tensor range check" && 0 <= v1005 && v1005 < 8);
        int v1013;
        v1013 = 1024 * v1005;
        int v1014;
        v1014 = v1013 + v1003;
        float v1015[4];
        int v1016[4];
        int v1017;
        v1017 = 0;
        while (while_method_3(v1017)){
            assert("Tensor range check" && 0 <= v1017 && v1017 < 1);
            int v1019;
            v1019 = 4 * v1017;
            assert("Tensor range check" && 0 <= v1017 && v1017 < 1);
            int v1020;
            v1020 = 64 * v1017;
            int v1021;
            v1021 = v1020 + v1014;
            int4* v1022;
            v1022 = reinterpret_cast<int4*>(v1 + v1021);
            int4* v1023;
            v1023 = reinterpret_cast<int4*>(v1015 + v1019);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1022) % 16 == 0 && reinterpret_cast<unsigned long long>(v1023) % 16 == 0);
            *v1023 = *v1022;
            v1017 += 1 ;
        }
        int v1024;
        v1024 = 0;
        while (while_method_3(v1024)){
            int v1026;
            v1026 = 0;
            while (while_method_1(v1026)){
                bool v1028;
                v1028 = 0 <= v1026;
                bool v1030;
                if (v1028){
                    bool v1029;
                    v1029 = v1026 < 4;
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
                v1033 = 0 <= v996;
                bool v1035;
                if (v1033){
                    bool v1034;
                    v1034 = v996 < 16;
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
                v1038 = v996 * 4;
                int v1039;
                v1039 = v1026 + v1038;
                bool v1040;
                v1040 = 0 <= v1024;
                bool v1042;
                if (v1040){
                    bool v1041;
                    v1041 = v1024 < 1;
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
                v1045 = v1024 * 64;
                int v1046;
                v1046 = v1039 + v1045;
                assert("Tensor range check" && 0 <= v1024 && v1024 < 1);
                assert("Tensor range check" && 0 <= v1026 && v1026 < 4);
                int v1047;
                v1047 = 4 * v1024;
                int v1048;
                v1048 = v1047 + v1026;
                v1016[v1048] = v1046;
                v1026 += 1 ;
            }
            v1024 += 1 ;
        }
        bool v1049;
        v1049 = 0 <= v997;
        bool v1050;
        v1050 = v1049 && v998;
        bool v1051;
        v1051 = v1050 == false;
        if (v1051){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1050);
        } else {
        }
        bool v1053;
        v1053 = v1007 && v1010;
        bool v1054;
        v1054 = v1053 == false;
        if (v1054){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1053);
        } else {
        }
        int v1056;
        v1056 = v1005 * 16;
        int v1057;
        v1057 = v1056 + v997;
        float v1058;
        v1058 = 0.0f;
        int v1059;
        v1059 = 0;
        while (while_method_3(v1059)){
            int v1061;
            v1061 = 0;
            while (while_method_1(v1061)){
                assert("Tensor range check" && 0 <= v1059 && v1059 < 1);
                assert("Tensor range check" && 0 <= v1061 && v1061 < 4);
                int v1063;
                v1063 = 4 * v1059;
                int v1064;
                v1064 = v1063 + v1061;
                float v1065;
                v1065 = v1015[v1064];
                float v1066;
                v1066 = v1058 + v1065;
                v1058 = v1066;
                v1061 += 1 ;
            }
            v1059 += 1 ;
        }
        auto v1067 = cooperative_groups::coalesced_threads();
        int v1068;
        v1068 = threadIdx.x;
        int v1069;
        v1069 = v1068 / 16;
        auto v1070 = cooperative_groups::labeled_partition(v1067,v1069);
        Closure0 v1071{};
        float v1072;
        v1072 = cooperative_groups::reduce(v1070, v1058, v1071);
        float v1073;
        v1073 = v1072 / 64.0f;
        float v1074[4];
        int v1075;
        v1075 = 0;
        while (while_method_3(v1075)){
            int v1077;
            v1077 = 0;
            while (while_method_1(v1077)){
                assert("Tensor range check" && 0 <= v1075 && v1075 < 1);
                assert("Tensor range check" && 0 <= v1077 && v1077 < 4);
                int v1079;
                v1079 = 4 * v1075;
                int v1080;
                v1080 = v1079 + v1077;
                float v1081;
                v1081 = v1015[v1080];
                float v1082;
                v1082 = v1081 - v1073;
                float v1083;
                v1083 = exp(v1082);
                assert("Tensor range check" && 0 <= v1075 && v1075 < 1);
                assert("Tensor range check" && 0 <= v1077 && v1077 < 4);
                v1074[v1080] = v1083;
                v1077 += 1 ;
            }
            v1075 += 1 ;
        }
        float v1084;
        v1084 = 0.0f;
        int v1085;
        v1085 = 0;
        while (while_method_3(v1085)){
            int v1087;
            v1087 = 0;
            while (while_method_1(v1087)){
                assert("Tensor range check" && 0 <= v1085 && v1085 < 1);
                assert("Tensor range check" && 0 <= v1087 && v1087 < 4);
                int v1089;
                v1089 = 4 * v1085;
                int v1090;
                v1090 = v1089 + v1087;
                float v1091;
                v1091 = v1074[v1090];
                float v1092;
                v1092 = v1084 + v1091;
                v1084 = v1092;
                v1087 += 1 ;
            }
            v1085 += 1 ;
        }
        auto v1093 = cooperative_groups::coalesced_threads();
        int v1094;
        v1094 = threadIdx.x;
        int v1095;
        v1095 = v1094 / 16;
        auto v1096 = cooperative_groups::labeled_partition(v1093,v1095);
        float v1097;
        v1097 = cooperative_groups::reduce(v1096, v1084, v1071);
        float v1098[4];
        int v1099;
        v1099 = 0;
        while (while_method_3(v1099)){
            int v1101;
            v1101 = 0;
            while (while_method_1(v1101)){
                assert("Tensor range check" && 0 <= v1099 && v1099 < 1);
                assert("Tensor range check" && 0 <= v1101 && v1101 < 4);
                int v1103;
                v1103 = 4 * v1099;
                int v1104;
                v1104 = v1103 + v1101;
                float v1105;
                v1105 = v1074[v1104];
                float v1106;
                v1106 = v1105 / v1097;
                assert("Tensor range check" && 0 <= v1099 && v1099 < 1);
                assert("Tensor range check" && 0 <= v1101 && v1101 < 4);
                v1098[v1104] = v1106;
                v1101 += 1 ;
            }
            v1099 += 1 ;
        }
        float v1107[4];
        float v1108;
        v1108 = 0.0f;
        int v1109;
        v1109 = 0;
        while (while_method_3(v1109)){
            assert("Tensor range check" && 0 <= v1109 && v1109 < 1);
            int v1111;
            v1111 = 4 * v1109;
            assert("Tensor range check" && 0 <= v1109 && v1109 < 1);
            int v1112; float v1113;
            Tuple0 tmp71 = Tuple0{0, 0.0f};
            v1112 = tmp71.v0; v1113 = tmp71.v1;
            while (while_method_1(v1112)){
                assert("Tensor range check" && 0 <= v1112 && v1112 < 4);
                int v1115;
                v1115 = v1112 + v1111;
                float v1116;
                v1116 = v1098[v1115];
                float v1117;
                v1117 = v1113 + v1116;
                v1113 = v1117;
                v1112 += 1 ;
            }
            auto v1118 = cooperative_groups::coalesced_threads();
            int v1119;
            v1119 = threadIdx.x;
            int v1120;
            v1120 = v1119 / 16;
            auto v1121 = cooperative_groups::labeled_partition(v1118,v1120);
            Closure2 v1122{};
            float v1123;
            v1123 = cooperative_groups::inclusive_scan(v1121, v1113, v1122);
            float v1124;
            v1124 = v1121.shfl_up(v1123,1);
            bool v1125;
            v1125 = v1121.thread_rank() == 0;
            float v1126;
            if (v1125){
                v1126 = 0.0f;
            } else {
                v1126 = v1124;
            }
            float v1127;
            v1127 = v1121.shfl(v1123,v1121.num_threads()-1);
            float v1128;
            v1128 = v1108 + v1126;
            int v1129; float v1130;
            Tuple0 tmp72 = Tuple0{0, v1128};
            v1129 = tmp72.v0; v1130 = tmp72.v1;
            while (while_method_1(v1129)){
                assert("Tensor range check" && 0 <= v1129 && v1129 < 4);
                int v1132;
                v1132 = v1129 + v1111;
                float v1133;
                v1133 = v1098[v1132];
                float v1134;
                v1134 = v1130 + v1133;
                assert("Tensor range check" && 0 <= v1129 && v1129 < 4);
                v1107[v1132] = v1134;
                v1130 = v1134;
                v1129 += 1 ;
            }
            float v1135;
            v1135 = v1108 + v1127;
            v1108 = v1135;
            v1109 += 1 ;
        }
        float v1136[4];
        bool v1137[4];
        int v1138;
        v1138 = 0;
        while (while_method_3(v1138)){
            int v1140;
            v1140 = 0;
            while (while_method_1(v1140)){
                assert("Tensor range check" && 0 <= v1138 && v1138 < 1);
                assert("Tensor range check" && 0 <= v1140 && v1140 < 4);
                int v1142;
                v1142 = 4 * v1138;
                int v1143;
                v1143 = v1142 + v1140;
                float v1144;
                v1144 = v1107[v1143];
                float v1145;
                v1145 = v1098[v1143];
                bool v1146;
                v1146 = v1145 > 0.0f;
                assert("Tensor range check" && 0 <= v1138 && v1138 < 1);
                assert("Tensor range check" && 0 <= v1140 && v1140 < 4);
                v1136[v1143] = v1144;
                v1137[v1143] = v1146;
                v1140 += 1 ;
            }
            v1138 += 1 ;
        }
        float v1147; bool v1148;
        Tuple3 tmp73 = Tuple3{-1.0f / 0.0f, false};
        v1147 = tmp73.v0; v1148 = tmp73.v1;
        int v1149;
        v1149 = 0;
        while (while_method_3(v1149)){
            int v1151;
            v1151 = 0;
            while (while_method_1(v1151)){
                assert("Tensor range check" && 0 <= v1149 && v1149 < 1);
                assert("Tensor range check" && 0 <= v1151 && v1151 < 4);
                int v1153;
                v1153 = 4 * v1149;
                int v1154;
                v1154 = v1153 + v1151;
                float v1155;
                v1155 = v1136[v1154];
                bool v1156;
                v1156 = v1137[v1154];
                float v1163; bool v1164;
                if (v1148){
                    if (v1156){
                        bool v1157;
                        v1157 = v1147 >= v1155;
                        float v1158;
                        if (v1157){
                            v1158 = v1147;
                        } else {
                            v1158 = v1155;
                        }
                        v1163 = v1158; v1164 = true;
                    } else {
                        v1163 = v1147; v1164 = v1148;
                    }
                } else {
                    if (v1156){
                        v1163 = v1155; v1164 = v1156;
                    } else {
                        v1163 = v1147; v1164 = v1148;
                    }
                }
                v1147 = v1163;
                v1148 = v1164;
                v1151 += 1 ;
            }
            v1149 += 1 ;
        }
        auto v1165 = cooperative_groups::coalesced_threads();
        int v1166;
        v1166 = threadIdx.x;
        int v1167;
        v1167 = v1166 / 16;
        auto v1168 = cooperative_groups::labeled_partition(v1165,v1167);
        Closure5 v1169{};
        float v1170; bool v1171;
        Tuple3 tmp74 = cooperative_groups::reduce(v1168, Tuple3{v1147, v1148}, v1169);
        v1170 = tmp74.v0; v1171 = tmp74.v1;
        bool v1172;
        v1172 = v1171 == false;
        if (v1172){
            assert("The local reduce must be true." && v1171);
        } else {
        }
        float v1174[4];
        int v1175[4];
        int v1176;
        v1176 = 0;
        while (while_method_3(v1176)){
            int v1178;
            v1178 = 0;
            while (while_method_1(v1178)){
                assert("Tensor range check" && 0 <= v1176 && v1176 < 1);
                assert("Tensor range check" && 0 <= v1178 && v1178 < 4);
                int v1180;
                v1180 = 4 * v1176;
                int v1181;
                v1181 = v1180 + v1178;
                int v1182;
                v1182 = v1016[v1181];
                float v1183;
                v1183 = curand_uniform(&v991);
                assert("Tensor range check" && 0 <= v1176 && v1176 < 1);
                assert("Tensor range check" && 0 <= v1178 && v1178 < 4);
                v1174[v1181] = v1183;
                v1175[v1181] = v1182;
                v1178 += 1 ;
            }
            v1176 += 1 ;
        }
        float v1184; int v1185;
        Tuple1 tmp75 = Tuple1{0.0f, 2147483647};
        v1184 = tmp75.v0; v1185 = tmp75.v1;
        int v1186;
        v1186 = 0;
        while (while_method_3(v1186)){
            int v1188;
            v1188 = 0;
            while (while_method_1(v1188)){
                assert("Tensor range check" && 0 <= v1186 && v1186 < 1);
                assert("Tensor range check" && 0 <= v1188 && v1188 < 4);
                int v1190;
                v1190 = 4 * v1186;
                int v1191;
                v1191 = v1190 + v1188;
                float v1192;
                v1192 = v1174[v1191];
                int v1193;
                v1193 = v1175[v1191];
                bool v1194;
                v1194 = v1185 < v1193;
                float v1195; int v1196;
                if (v1194){
                    v1195 = v1184; v1196 = v1185;
                } else {
                    v1195 = v1192; v1196 = v1193;
                }
                v1184 = v1195;
                v1185 = v1196;
                v1188 += 1 ;
            }
            v1186 += 1 ;
        }
        auto v1197 = cooperative_groups::coalesced_threads();
        int v1198;
        v1198 = threadIdx.x;
        int v1199;
        v1199 = v1198 / 16;
        auto v1200 = cooperative_groups::labeled_partition(v1197,v1199);
        Closure6 v1201{};
        float v1202; int v1203;
        Tuple1 tmp76 = cooperative_groups::reduce(v1200, Tuple1{v1184, v1185}, v1201);
        v1202 = tmp76.v0; v1203 = tmp76.v1;
        float v1204;
        v1204 = v1170 * v1202;
        int v1205[4];
        bool v1206[4];
        int v1207;
        v1207 = 0;
        while (while_method_3(v1207)){
            int v1209;
            v1209 = 0;
            while (while_method_1(v1209)){
                assert("Tensor range check" && 0 <= v1207 && v1207 < 1);
                assert("Tensor range check" && 0 <= v1209 && v1209 < 4);
                int v1211;
                v1211 = 4 * v1207;
                int v1212;
                v1212 = v1211 + v1209;
                float v1213;
                v1213 = v1136[v1212];
                bool v1214;
                v1214 = v1137[v1212];
                int v1215;
                v1215 = v1016[v1212];
                int v1218; bool v1219;
                if (v1214){
                    float v1216;
                    v1216 = v1213 - v1204;
                    bool v1217;
                    v1217 = v1216 >= 0.0f;
                    v1218 = v1215; v1219 = v1217;
                } else {
                    v1218 = 2147483647; v1219 = false;
                }
                assert("Tensor range check" && 0 <= v1207 && v1207 < 1);
                assert("Tensor range check" && 0 <= v1209 && v1209 < 4);
                v1205[v1212] = v1218;
                v1206[v1212] = v1219;
                v1209 += 1 ;
            }
            v1207 += 1 ;
        }
        int v1220; bool v1221;
        Tuple4 tmp77 = Tuple4{2147483647, false};
        v1220 = tmp77.v0; v1221 = tmp77.v1;
        int v1222;
        v1222 = 0;
        while (while_method_3(v1222)){
            int v1224;
            v1224 = 0;
            while (while_method_1(v1224)){
                assert("Tensor range check" && 0 <= v1222 && v1222 < 1);
                assert("Tensor range check" && 0 <= v1224 && v1224 < 4);
                int v1226;
                v1226 = 4 * v1222;
                int v1227;
                v1227 = v1226 + v1224;
                int v1228;
                v1228 = v1205[v1227];
                bool v1229;
                v1229 = v1206[v1227];
                int v1236; bool v1237;
                if (v1221){
                    if (v1229){
                        bool v1230;
                        v1230 = v1220 < v1228;
                        int v1231;
                        if (v1230){
                            v1231 = v1220;
                        } else {
                            v1231 = v1228;
                        }
                        v1236 = v1231; v1237 = true;
                    } else {
                        v1236 = v1220; v1237 = v1221;
                    }
                } else {
                    if (v1229){
                        v1236 = v1228; v1237 = v1229;
                    } else {
                        v1236 = v1220; v1237 = v1221;
                    }
                }
                v1220 = v1236;
                v1221 = v1237;
                v1224 += 1 ;
            }
            v1222 += 1 ;
        }
        auto v1238 = cooperative_groups::coalesced_threads();
        int v1239;
        v1239 = threadIdx.x;
        int v1240;
        v1240 = v1239 / 16;
        auto v1241 = cooperative_groups::labeled_partition(v1238,v1240);
        Closure7 v1242{};
        int v1243; bool v1244;
        Tuple4 tmp78 = cooperative_groups::reduce(v1241, Tuple4{v1220, v1221}, v1242);
        v1243 = tmp78.v0; v1244 = tmp78.v1;
        bool v1245;
        v1245 = v1244 == false;
        if (v1245){
            assert("The local reduce must be true." && v1244);
        } else {
        }
        assert("Tensor range check" && 0 <= v1005 && v1005 < 8);
        int v1247;
        v1247 = 0;
        while (while_method_3(v1247)){
            assert("Tensor range check" && 0 <= v1247 && v1247 < 1);
            int v1249;
            v1249 = 64 * v1247;
            int v1250;
            v1250 = v1249 + v1014;
            assert("Tensor range check" && 0 <= v1247 && v1247 < 1);
            int v1251;
            v1251 = 4 * v1247;
            int4* v1252;
            v1252 = reinterpret_cast<int4*>(v1098 + v1251);
            int4* v1253;
            v1253 = reinterpret_cast<int4*>(v13 + v1250);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1252) % 16 == 0 && reinterpret_cast<unsigned long long>(v1253) % 16 == 0);
            *v1253 = *v1252;
            v1247 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1005 && v1005 < 8);
        int v1254;
        v1254 = 16 * v1005;
        int v1255;
        v1255 = v1254 + v997;
        v14[v1255] = v1243;
        v1005 += 24 ;
    }
    v17.sync() ;
    int v1256;
    v1256 = threadIdx.x;
    int v1257;
    v1257 = blockIdx.x;
    int v1258;
    v1258 = v1257 * 256;
    int v1259;
    v1259 = v1256 + v1258;
    unsigned long long v1260;
    v1260 = (unsigned long long)v1259;
    curandStatePhilox4_32_10_t v1261;
    curand_init(12344321ull,v1260,0ull,&v1261);
    int v1262;
    v1262 = threadIdx.x;
    bool v1263;
    v1263 = 0 <= v1262;
    bool v1264;
    v1264 = v1263 == false;
    if (v1264){
        assert("The index needs to be zero or positive." && v1263);
    } else {
    }
    int v1266;
    v1266 = v1262 % 16;
    int v1267;
    v1267 = v1262 / 16;
    bool v1268;
    v1268 = v1267 < 16;
    bool v1269;
    v1269 = v1268 == false;
    if (v1269){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1268);
    } else {
    }
    assert("Tensor range check" && 0 <= v1267 && v1267 < 16);
    assert("Tensor range check" && 0 <= v1266 && v1266 < 16);
    int v1271;
    v1271 = 4 * v1266;
    int v1272;
    v1272 = 64 * v1267;
    int v1273;
    v1273 = v1272 + v1271;
    assert("Tensor range check" && 0 <= v1267 && v1267 < 16);
    assert("Tensor range check" && 0 <= v1266 && v1266 < 16);
    assert("Tensor range check" && 0 <= v1267 && v1267 < 16);
    int v1274;
    v1274 = blockIdx.x;
    int v1275;
    v1275 = v1274;
    while (while_method_2(v1275)){
        bool v1277;
        v1277 = 0 <= v1275;
        bool v1278;
        v1278 = v1277 == false;
        if (v1278){
            assert("The index needs to be zero or positive." && v1277);
        } else {
        }
        bool v1280;
        v1280 = v1275 < 8;
        bool v1281;
        v1281 = v1280 == false;
        if (v1281){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1280);
        } else {
        }
        assert("Tensor range check" && 0 <= v1275 && v1275 < 8);
        int v1283;
        v1283 = 1024 * v1275;
        int v1284;
        v1284 = v1283 + v1273;
        float v1285[4];
        int v1286[4];
        int v1287;
        v1287 = 0;
        while (while_method_3(v1287)){
            assert("Tensor range check" && 0 <= v1287 && v1287 < 1);
            int v1289;
            v1289 = 4 * v1287;
            assert("Tensor range check" && 0 <= v1287 && v1287 < 1);
            int v1290;
            v1290 = 64 * v1287;
            int v1291;
            v1291 = v1290 + v1284;
            int4* v1292;
            v1292 = reinterpret_cast<int4*>(v1 + v1291);
            int4* v1293;
            v1293 = reinterpret_cast<int4*>(v1285 + v1289);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1292) % 16 == 0 && reinterpret_cast<unsigned long long>(v1293) % 16 == 0);
            *v1293 = *v1292;
            v1287 += 1 ;
        }
        int v1294;
        v1294 = 0;
        while (while_method_3(v1294)){
            int v1296;
            v1296 = 0;
            while (while_method_1(v1296)){
                bool v1298;
                v1298 = 0 <= v1296;
                bool v1300;
                if (v1298){
                    bool v1299;
                    v1299 = v1296 < 4;
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
                v1303 = 0 <= v1266;
                bool v1305;
                if (v1303){
                    bool v1304;
                    v1304 = v1266 < 16;
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
                v1308 = v1266 * 4;
                int v1309;
                v1309 = v1296 + v1308;
                bool v1310;
                v1310 = 0 <= v1294;
                bool v1312;
                if (v1310){
                    bool v1311;
                    v1311 = v1294 < 1;
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
                v1315 = v1294 * 64;
                int v1316;
                v1316 = v1309 + v1315;
                assert("Tensor range check" && 0 <= v1294 && v1294 < 1);
                assert("Tensor range check" && 0 <= v1296 && v1296 < 4);
                int v1317;
                v1317 = 4 * v1294;
                int v1318;
                v1318 = v1317 + v1296;
                v1286[v1318] = v1316;
                v1296 += 1 ;
            }
            v1294 += 1 ;
        }
        bool v1319;
        v1319 = 0 <= v1267;
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
        v1326 = v1275 * 16;
        int v1327;
        v1327 = v1326 + v1267;
        bool v1328[4];
        int v1329;
        v1329 = 0;
        while (while_method_3(v1329)){
            int v1331;
            v1331 = 0;
            while (while_method_1(v1331)){
                assert("Tensor range check" && 0 <= v1329 && v1329 < 1);
                assert("Tensor range check" && 0 <= v1331 && v1331 < 4);
                int v1333;
                v1333 = 4 * v1329;
                int v1334;
                v1334 = v1333 + v1331;
                float v1335;
                v1335 = v1285[v1334];
                int v1336;
                v1336 = v1286[v1334];
                bool v1337;
                v1337 = v1336 < 11;
                assert("Tensor range check" && 0 <= v1329 && v1329 < 1);
                assert("Tensor range check" && 0 <= v1331 && v1331 < 4);
                v1328[v1334] = v1337;
                v1331 += 1 ;
            }
            v1329 += 1 ;
        }
        int v1338[4];
        int v1339;
        v1339 = 0;
        while (while_method_3(v1339)){
            int v1341;
            v1341 = 0;
            while (while_method_1(v1341)){
                assert("Tensor range check" && 0 <= v1339 && v1339 < 1);
                assert("Tensor range check" && 0 <= v1341 && v1341 < 4);
                int v1343;
                v1343 = 4 * v1339;
                int v1344;
                v1344 = v1343 + v1341;
                bool v1345;
                v1345 = v1328[v1344];
                int v1346;
                if (v1345){
                    v1346 = 1;
                } else {
                    v1346 = 0;
                }
                assert("Tensor range check" && 0 <= v1339 && v1339 < 1);
                assert("Tensor range check" && 0 <= v1341 && v1341 < 4);
                v1338[v1344] = v1346;
                v1341 += 1 ;
            }
            v1339 += 1 ;
        }
        int v1347;
        v1347 = 0;
        int v1348;
        v1348 = 0;
        while (while_method_3(v1348)){
            int v1350;
            v1350 = 0;
            while (while_method_1(v1350)){
                assert("Tensor range check" && 0 <= v1348 && v1348 < 1);
                assert("Tensor range check" && 0 <= v1350 && v1350 < 4);
                int v1352;
                v1352 = 4 * v1348;
                int v1353;
                v1353 = v1352 + v1350;
                int v1354;
                v1354 = v1338[v1353];
                int v1355;
                v1355 = v1347 + v1354;
                v1347 = v1355;
                v1350 += 1 ;
            }
            v1348 += 1 ;
        }
        auto v1356 = cooperative_groups::coalesced_threads();
        int v1357;
        v1357 = threadIdx.x;
        int v1358;
        v1358 = v1357 / 16;
        auto v1359 = cooperative_groups::labeled_partition(v1356,v1358);
        Closure4 v1360{};
        int v1361;
        v1361 = cooperative_groups::reduce(v1359, v1347, v1360);
        float v1362[4];
        int v1363;
        v1363 = 0;
        while (while_method_3(v1363)){
            int v1365;
            v1365 = 0;
            while (while_method_1(v1365)){
                assert("Tensor range check" && 0 <= v1363 && v1363 < 1);
                assert("Tensor range check" && 0 <= v1365 && v1365 < 4);
                int v1367;
                v1367 = 4 * v1363;
                int v1368;
                v1368 = v1367 + v1365;
                float v1369;
                v1369 = v1285[v1368];
                bool v1370;
                v1370 = v1328[v1368];
                float v1371;
                if (v1370){
                    v1371 = v1369;
                } else {
                    v1371 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1363 && v1363 < 1);
                assert("Tensor range check" && 0 <= v1365 && v1365 < 4);
                v1362[v1368] = v1371;
                v1365 += 1 ;
            }
            v1363 += 1 ;
        }
        float v1372;
        v1372 = 0.0f;
        int v1373;
        v1373 = 0;
        while (while_method_3(v1373)){
            int v1375;
            v1375 = 0;
            while (while_method_1(v1375)){
                assert("Tensor range check" && 0 <= v1373 && v1373 < 1);
                assert("Tensor range check" && 0 <= v1375 && v1375 < 4);
                int v1377;
                v1377 = 4 * v1373;
                int v1378;
                v1378 = v1377 + v1375;
                float v1379;
                v1379 = v1362[v1378];
                float v1380;
                v1380 = v1372 + v1379;
                v1372 = v1380;
                v1375 += 1 ;
            }
            v1373 += 1 ;
        }
        auto v1381 = cooperative_groups::coalesced_threads();
        int v1382;
        v1382 = threadIdx.x;
        int v1383;
        v1383 = v1382 / 16;
        auto v1384 = cooperative_groups::labeled_partition(v1381,v1383);
        Closure0 v1385{};
        float v1386;
        v1386 = cooperative_groups::reduce(v1384, v1372, v1385);
        float v1387;
        v1387 = (float)v1361;
        float v1388;
        v1388 = v1386 / v1387;
        float v1389[4];
        int v1390;
        v1390 = 0;
        while (while_method_3(v1390)){
            int v1392;
            v1392 = 0;
            while (while_method_1(v1392)){
                assert("Tensor range check" && 0 <= v1390 && v1390 < 1);
                assert("Tensor range check" && 0 <= v1392 && v1392 < 4);
                int v1394;
                v1394 = 4 * v1390;
                int v1395;
                v1395 = v1394 + v1392;
                float v1396;
                v1396 = v1285[v1395];
                bool v1397;
                v1397 = v1328[v1395];
                float v1398;
                if (v1397){
                    v1398 = v1396;
                } else {
                    v1398 = -1.0f / 0.0f;
                }
                float v1399;
                v1399 = v1398 - v1388;
                float v1400;
                v1400 = exp(v1399);
                assert("Tensor range check" && 0 <= v1390 && v1390 < 1);
                assert("Tensor range check" && 0 <= v1392 && v1392 < 4);
                v1389[v1395] = v1400;
                v1392 += 1 ;
            }
            v1390 += 1 ;
        }
        float v1401;
        v1401 = 0.0f;
        int v1402;
        v1402 = 0;
        while (while_method_3(v1402)){
            int v1404;
            v1404 = 0;
            while (while_method_1(v1404)){
                assert("Tensor range check" && 0 <= v1402 && v1402 < 1);
                assert("Tensor range check" && 0 <= v1404 && v1404 < 4);
                int v1406;
                v1406 = 4 * v1402;
                int v1407;
                v1407 = v1406 + v1404;
                float v1408;
                v1408 = v1389[v1407];
                float v1409;
                v1409 = v1401 + v1408;
                v1401 = v1409;
                v1404 += 1 ;
            }
            v1402 += 1 ;
        }
        auto v1410 = cooperative_groups::coalesced_threads();
        int v1411;
        v1411 = threadIdx.x;
        int v1412;
        v1412 = v1411 / 16;
        auto v1413 = cooperative_groups::labeled_partition(v1410,v1412);
        float v1414;
        v1414 = cooperative_groups::reduce(v1413, v1401, v1385);
        float v1415[4];
        int v1416;
        v1416 = 0;
        while (while_method_3(v1416)){
            int v1418;
            v1418 = 0;
            while (while_method_1(v1418)){
                assert("Tensor range check" && 0 <= v1416 && v1416 < 1);
                assert("Tensor range check" && 0 <= v1418 && v1418 < 4);
                int v1420;
                v1420 = 4 * v1416;
                int v1421;
                v1421 = v1420 + v1418;
                float v1422;
                v1422 = v1389[v1421];
                float v1423;
                v1423 = v1422 / v1414;
                assert("Tensor range check" && 0 <= v1416 && v1416 < 1);
                assert("Tensor range check" && 0 <= v1418 && v1418 < 4);
                v1415[v1421] = v1423;
                v1418 += 1 ;
            }
            v1416 += 1 ;
        }
        float v1424[4];
        float v1425;
        v1425 = 0.0f;
        int v1426;
        v1426 = 0;
        while (while_method_3(v1426)){
            assert("Tensor range check" && 0 <= v1426 && v1426 < 1);
            int v1428;
            v1428 = 4 * v1426;
            assert("Tensor range check" && 0 <= v1426 && v1426 < 1);
            int v1429; float v1430;
            Tuple0 tmp79 = Tuple0{0, 0.0f};
            v1429 = tmp79.v0; v1430 = tmp79.v1;
            while (while_method_1(v1429)){
                assert("Tensor range check" && 0 <= v1429 && v1429 < 4);
                int v1432;
                v1432 = v1429 + v1428;
                float v1433;
                v1433 = v1415[v1432];
                float v1434;
                v1434 = v1430 + v1433;
                v1430 = v1434;
                v1429 += 1 ;
            }
            auto v1435 = cooperative_groups::coalesced_threads();
            int v1436;
            v1436 = threadIdx.x;
            int v1437;
            v1437 = v1436 / 16;
            auto v1438 = cooperative_groups::labeled_partition(v1435,v1437);
            Closure2 v1439{};
            float v1440;
            v1440 = cooperative_groups::inclusive_scan(v1438, v1430, v1439);
            float v1441;
            v1441 = v1438.shfl_up(v1440,1);
            bool v1442;
            v1442 = v1438.thread_rank() == 0;
            float v1443;
            if (v1442){
                v1443 = 0.0f;
            } else {
                v1443 = v1441;
            }
            float v1444;
            v1444 = v1438.shfl(v1440,v1438.num_threads()-1);
            float v1445;
            v1445 = v1425 + v1443;
            int v1446; float v1447;
            Tuple0 tmp80 = Tuple0{0, v1445};
            v1446 = tmp80.v0; v1447 = tmp80.v1;
            while (while_method_1(v1446)){
                assert("Tensor range check" && 0 <= v1446 && v1446 < 4);
                int v1449;
                v1449 = v1446 + v1428;
                float v1450;
                v1450 = v1415[v1449];
                float v1451;
                v1451 = v1447 + v1450;
                assert("Tensor range check" && 0 <= v1446 && v1446 < 4);
                v1424[v1449] = v1451;
                v1447 = v1451;
                v1446 += 1 ;
            }
            float v1452;
            v1452 = v1425 + v1444;
            v1425 = v1452;
            v1426 += 1 ;
        }
        float v1453[4];
        bool v1454[4];
        int v1455;
        v1455 = 0;
        while (while_method_3(v1455)){
            int v1457;
            v1457 = 0;
            while (while_method_1(v1457)){
                assert("Tensor range check" && 0 <= v1455 && v1455 < 1);
                assert("Tensor range check" && 0 <= v1457 && v1457 < 4);
                int v1459;
                v1459 = 4 * v1455;
                int v1460;
                v1460 = v1459 + v1457;
                float v1461;
                v1461 = v1424[v1460];
                float v1462;
                v1462 = v1415[v1460];
                bool v1463;
                v1463 = v1462 > 0.0f;
                assert("Tensor range check" && 0 <= v1455 && v1455 < 1);
                assert("Tensor range check" && 0 <= v1457 && v1457 < 4);
                v1453[v1460] = v1461;
                v1454[v1460] = v1463;
                v1457 += 1 ;
            }
            v1455 += 1 ;
        }
        float v1464; bool v1465;
        Tuple3 tmp81 = Tuple3{-1.0f / 0.0f, false};
        v1464 = tmp81.v0; v1465 = tmp81.v1;
        int v1466;
        v1466 = 0;
        while (while_method_3(v1466)){
            int v1468;
            v1468 = 0;
            while (while_method_1(v1468)){
                assert("Tensor range check" && 0 <= v1466 && v1466 < 1);
                assert("Tensor range check" && 0 <= v1468 && v1468 < 4);
                int v1470;
                v1470 = 4 * v1466;
                int v1471;
                v1471 = v1470 + v1468;
                float v1472;
                v1472 = v1453[v1471];
                bool v1473;
                v1473 = v1454[v1471];
                float v1480; bool v1481;
                if (v1465){
                    if (v1473){
                        bool v1474;
                        v1474 = v1464 >= v1472;
                        float v1475;
                        if (v1474){
                            v1475 = v1464;
                        } else {
                            v1475 = v1472;
                        }
                        v1480 = v1475; v1481 = true;
                    } else {
                        v1480 = v1464; v1481 = v1465;
                    }
                } else {
                    if (v1473){
                        v1480 = v1472; v1481 = v1473;
                    } else {
                        v1480 = v1464; v1481 = v1465;
                    }
                }
                v1464 = v1480;
                v1465 = v1481;
                v1468 += 1 ;
            }
            v1466 += 1 ;
        }
        auto v1482 = cooperative_groups::coalesced_threads();
        int v1483;
        v1483 = threadIdx.x;
        int v1484;
        v1484 = v1483 / 16;
        auto v1485 = cooperative_groups::labeled_partition(v1482,v1484);
        Closure5 v1486{};
        float v1487; bool v1488;
        Tuple3 tmp82 = cooperative_groups::reduce(v1485, Tuple3{v1464, v1465}, v1486);
        v1487 = tmp82.v0; v1488 = tmp82.v1;
        bool v1489;
        v1489 = v1488 == false;
        if (v1489){
            assert("The local reduce must be true." && v1488);
        } else {
        }
        float v1491[4];
        int v1492[4];
        int v1493;
        v1493 = 0;
        while (while_method_3(v1493)){
            int v1495;
            v1495 = 0;
            while (while_method_1(v1495)){
                assert("Tensor range check" && 0 <= v1493 && v1493 < 1);
                assert("Tensor range check" && 0 <= v1495 && v1495 < 4);
                int v1497;
                v1497 = 4 * v1493;
                int v1498;
                v1498 = v1497 + v1495;
                int v1499;
                v1499 = v1286[v1498];
                float v1500;
                v1500 = curand_uniform(&v1261);
                assert("Tensor range check" && 0 <= v1493 && v1493 < 1);
                assert("Tensor range check" && 0 <= v1495 && v1495 < 4);
                v1491[v1498] = v1500;
                v1492[v1498] = v1499;
                v1495 += 1 ;
            }
            v1493 += 1 ;
        }
        float v1501; int v1502;
        Tuple1 tmp83 = Tuple1{0.0f, 2147483647};
        v1501 = tmp83.v0; v1502 = tmp83.v1;
        int v1503;
        v1503 = 0;
        while (while_method_3(v1503)){
            int v1505;
            v1505 = 0;
            while (while_method_1(v1505)){
                assert("Tensor range check" && 0 <= v1503 && v1503 < 1);
                assert("Tensor range check" && 0 <= v1505 && v1505 < 4);
                int v1507;
                v1507 = 4 * v1503;
                int v1508;
                v1508 = v1507 + v1505;
                float v1509;
                v1509 = v1491[v1508];
                int v1510;
                v1510 = v1492[v1508];
                bool v1511;
                v1511 = v1502 < v1510;
                float v1512; int v1513;
                if (v1511){
                    v1512 = v1501; v1513 = v1502;
                } else {
                    v1512 = v1509; v1513 = v1510;
                }
                v1501 = v1512;
                v1502 = v1513;
                v1505 += 1 ;
            }
            v1503 += 1 ;
        }
        auto v1514 = cooperative_groups::coalesced_threads();
        int v1515;
        v1515 = threadIdx.x;
        int v1516;
        v1516 = v1515 / 16;
        auto v1517 = cooperative_groups::labeled_partition(v1514,v1516);
        Closure6 v1518{};
        float v1519; int v1520;
        Tuple1 tmp84 = cooperative_groups::reduce(v1517, Tuple1{v1501, v1502}, v1518);
        v1519 = tmp84.v0; v1520 = tmp84.v1;
        float v1521;
        v1521 = v1487 * v1519;
        int v1522[4];
        bool v1523[4];
        int v1524;
        v1524 = 0;
        while (while_method_3(v1524)){
            int v1526;
            v1526 = 0;
            while (while_method_1(v1526)){
                assert("Tensor range check" && 0 <= v1524 && v1524 < 1);
                assert("Tensor range check" && 0 <= v1526 && v1526 < 4);
                int v1528;
                v1528 = 4 * v1524;
                int v1529;
                v1529 = v1528 + v1526;
                float v1530;
                v1530 = v1453[v1529];
                bool v1531;
                v1531 = v1454[v1529];
                int v1532;
                v1532 = v1286[v1529];
                int v1535; bool v1536;
                if (v1531){
                    float v1533;
                    v1533 = v1530 - v1521;
                    bool v1534;
                    v1534 = v1533 >= 0.0f;
                    v1535 = v1532; v1536 = v1534;
                } else {
                    v1535 = 2147483647; v1536 = false;
                }
                assert("Tensor range check" && 0 <= v1524 && v1524 < 1);
                assert("Tensor range check" && 0 <= v1526 && v1526 < 4);
                v1522[v1529] = v1535;
                v1523[v1529] = v1536;
                v1526 += 1 ;
            }
            v1524 += 1 ;
        }
        int v1537; bool v1538;
        Tuple4 tmp85 = Tuple4{2147483647, false};
        v1537 = tmp85.v0; v1538 = tmp85.v1;
        int v1539;
        v1539 = 0;
        while (while_method_3(v1539)){
            int v1541;
            v1541 = 0;
            while (while_method_1(v1541)){
                assert("Tensor range check" && 0 <= v1539 && v1539 < 1);
                assert("Tensor range check" && 0 <= v1541 && v1541 < 4);
                int v1543;
                v1543 = 4 * v1539;
                int v1544;
                v1544 = v1543 + v1541;
                int v1545;
                v1545 = v1522[v1544];
                bool v1546;
                v1546 = v1523[v1544];
                int v1553; bool v1554;
                if (v1538){
                    if (v1546){
                        bool v1547;
                        v1547 = v1537 < v1545;
                        int v1548;
                        if (v1547){
                            v1548 = v1537;
                        } else {
                            v1548 = v1545;
                        }
                        v1553 = v1548; v1554 = true;
                    } else {
                        v1553 = v1537; v1554 = v1538;
                    }
                } else {
                    if (v1546){
                        v1553 = v1545; v1554 = v1546;
                    } else {
                        v1553 = v1537; v1554 = v1538;
                    }
                }
                v1537 = v1553;
                v1538 = v1554;
                v1541 += 1 ;
            }
            v1539 += 1 ;
        }
        auto v1555 = cooperative_groups::coalesced_threads();
        int v1556;
        v1556 = threadIdx.x;
        int v1557;
        v1557 = v1556 / 16;
        auto v1558 = cooperative_groups::labeled_partition(v1555,v1557);
        Closure7 v1559{};
        int v1560; bool v1561;
        Tuple4 tmp86 = cooperative_groups::reduce(v1558, Tuple4{v1537, v1538}, v1559);
        v1560 = tmp86.v0; v1561 = tmp86.v1;
        bool v1562;
        v1562 = v1561 == false;
        if (v1562){
            assert("The local reduce must be true." && v1561);
        } else {
        }
        assert("Tensor range check" && 0 <= v1275 && v1275 < 8);
        int v1564;
        v1564 = 0;
        while (while_method_3(v1564)){
            assert("Tensor range check" && 0 <= v1564 && v1564 < 1);
            int v1566;
            v1566 = 64 * v1564;
            int v1567;
            v1567 = v1566 + v1284;
            assert("Tensor range check" && 0 <= v1564 && v1564 < 1);
            int v1568;
            v1568 = 4 * v1564;
            int4* v1569;
            v1569 = reinterpret_cast<int4*>(v1415 + v1568);
            int4* v1570;
            v1570 = reinterpret_cast<int4*>(v15 + v1567);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1569) % 16 == 0 && reinterpret_cast<unsigned long long>(v1570) % 16 == 0);
            *v1570 = *v1569;
            v1564 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1275 && v1275 < 8);
        int v1571;
        v1571 = 16 * v1275;
        int v1572;
        v1572 = v1571 + v1267;
        v16[v1572] = v1560;
        v1275 += 24 ;
    }
    v17.sync() ;
    return ;
}
extern "C" __global__ void entry5(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14, float * v15, int * v16) {
    auto v17 = cooperative_groups::this_grid();
    int v18;
    v18 = threadIdx.x;
    bool v19;
    v19 = 0 <= v18;
    bool v20;
    v20 = v19 == false;
    if (v20){
        assert("The index needs to be zero or positive." && v19);
    } else {
    }
    int v22;
    v22 = v18 % 32;
    int v23;
    v23 = v18 / 32;
    bool v24;
    v24 = v23 < 8;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
    } else {
    }
    assert("Tensor range check" && 0 <= v23 && v23 < 8);
    assert("Tensor range check" && 0 <= v22 && v22 < 32);
    int v27;
    v27 = 4 * v22;
    int v28;
    v28 = 128 * v23;
    int v29;
    v29 = v28 + v27;
    assert("Tensor range check" && 0 <= v23 && v23 < 8);
    assert("Tensor range check" && 0 <= v22 && v22 < 32);
    int v30;
    v30 = blockIdx.x;
    int v31;
    v31 = v30;
    while (while_method_2(v31)){
        bool v33;
        v33 = 0 <= v31;
        bool v34;
        v34 = v33 == false;
        if (v34){
            assert("The index needs to be zero or positive." && v33);
        } else {
        }
        bool v36;
        v36 = v31 < 8;
        bool v37;
        v37 = v36 == false;
        if (v37){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v36);
        } else {
        }
        assert("Tensor range check" && 0 <= v31 && v31 < 8);
        int v39;
        v39 = 1024 * v31;
        int v40;
        v40 = v39 + v29;
        int v41[4];
        int v42[4];
        int v43;
        v43 = 0;
        while (while_method_3(v43)){
            assert("Tensor range check" && 0 <= v43 && v43 < 1);
            int v45;
            v45 = 4 * v43;
            assert("Tensor range check" && 0 <= v43 && v43 < 1);
            int v46;
            v46 = 128 * v43;
            int v47;
            v47 = v46 + v40;
            int4* v48;
            v48 = reinterpret_cast<int4*>(v0 + v47);
            int4* v49;
            v49 = reinterpret_cast<int4*>(v41 + v45);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v48) % 16 == 0 && reinterpret_cast<unsigned long long>(v49) % 16 == 0);
            *v49 = *v48;
            v43 += 1 ;
        }
        int v50;
        v50 = 0;
        while (while_method_3(v50)){
            int v52;
            v52 = 0;
            while (while_method_1(v52)){
                bool v54;
                v54 = 0 <= v52;
                bool v56;
                if (v54){
                    bool v55;
                    v55 = v52 < 4;
                    v56 = v55;
                } else {
                    v56 = false;
                }
                bool v57;
                v57 = v56 == false;
                if (v57){
                    assert("The indices should be inside the range of the dimension." && v56);
                } else {
                }
                bool v59;
                v59 = 0 <= v22;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v22 < 32;
                    v61 = v60;
                } else {
                    v61 = false;
                }
                bool v62;
                v62 = v61 == false;
                if (v62){
                    assert("The indices should be inside the range of the dimension." && v61);
                } else {
                }
                int v64;
                v64 = v22 * 4;
                int v65;
                v65 = v52 + v64;
                bool v66;
                v66 = 0 <= v50;
                bool v68;
                if (v66){
                    bool v67;
                    v67 = v50 < 1;
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
                v71 = v50 * 128;
                int v72;
                v72 = v65 + v71;
                assert("Tensor range check" && 0 <= v50 && v50 < 1);
                assert("Tensor range check" && 0 <= v52 && v52 < 4);
                int v73;
                v73 = 4 * v50;
                int v74;
                v74 = v73 + v52;
                v42[v74] = v72;
                v52 += 1 ;
            }
            v50 += 1 ;
        }
        bool v75;
        v75 = 0 <= v23;
        bool v76;
        v76 = v75 && v24;
        bool v77;
        v77 = v76 == false;
        if (v77){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v76);
        } else {
        }
        bool v79;
        v79 = v33 && v36;
        bool v80;
        v80 = v79 == false;
        if (v80){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v79);
        } else {
        }
        int v82;
        v82 = v31 * 8;
        int v83;
        v83 = v82 + v23;
        assert("Tensor range check" && 0 <= v31 && v31 < 8);
        int v84;
        v84 = 0;
        while (while_method_3(v84)){
            assert("Tensor range check" && 0 <= v84 && v84 < 1);
            int v86;
            v86 = 128 * v84;
            int v87;
            v87 = v86 + v40;
            assert("Tensor range check" && 0 <= v84 && v84 < 1);
            int v88;
            v88 = 4 * v84;
            int4* v89;
            v89 = reinterpret_cast<int4*>(v41 + v88);
            int4* v90;
            v90 = reinterpret_cast<int4*>(v2 + v87);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v89) % 16 == 0 && reinterpret_cast<unsigned long long>(v90) % 16 == 0);
            *v90 = *v89;
            v84 += 1 ;
        }
        v31 += 24 ;
    }
    v17.sync() ;
    int v91;
    v91 = threadIdx.x;
    bool v92;
    v92 = 0 <= v91;
    bool v93;
    v93 = v92 == false;
    if (v93){
        assert("The index needs to be zero or positive." && v92);
    } else {
    }
    int v95;
    v95 = v91 % 32;
    int v96;
    v96 = v91 / 32;
    bool v97;
    v97 = v96 < 8;
    bool v98;
    v98 = v97 == false;
    if (v98){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v97);
    } else {
    }
    assert("Tensor range check" && 0 <= v96 && v96 < 8);
    assert("Tensor range check" && 0 <= v95 && v95 < 32);
    int v100;
    v100 = 4 * v95;
    int v101;
    v101 = 128 * v96;
    int v102;
    v102 = v101 + v100;
    assert("Tensor range check" && 0 <= v96 && v96 < 8);
    assert("Tensor range check" && 0 <= v95 && v95 < 32);
    int v103;
    v103 = blockIdx.x;
    int v104;
    v104 = v103;
    while (while_method_2(v104)){
        bool v106;
        v106 = 0 <= v104;
        bool v107;
        v107 = v106 == false;
        if (v107){
            assert("The index needs to be zero or positive." && v106);
        } else {
        }
        bool v109;
        v109 = v104 < 8;
        bool v110;
        v110 = v109 == false;
        if (v110){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v109);
        } else {
        }
        assert("Tensor range check" && 0 <= v104 && v104 < 8);
        int v112;
        v112 = 1024 * v104;
        int v113;
        v113 = v112 + v102;
        float v114[4];
        int v115[4];
        int v116;
        v116 = 0;
        while (while_method_3(v116)){
            assert("Tensor range check" && 0 <= v116 && v116 < 1);
            int v118;
            v118 = 4 * v116;
            assert("Tensor range check" && 0 <= v116 && v116 < 1);
            int v119;
            v119 = 128 * v116;
            int v120;
            v120 = v119 + v113;
            int4* v121;
            v121 = reinterpret_cast<int4*>(v1 + v120);
            int4* v122;
            v122 = reinterpret_cast<int4*>(v114 + v118);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v121) % 16 == 0 && reinterpret_cast<unsigned long long>(v122) % 16 == 0);
            *v122 = *v121;
            v116 += 1 ;
        }
        int v123;
        v123 = 0;
        while (while_method_3(v123)){
            int v125;
            v125 = 0;
            while (while_method_1(v125)){
                bool v127;
                v127 = 0 <= v125;
                bool v129;
                if (v127){
                    bool v128;
                    v128 = v125 < 4;
                    v129 = v128;
                } else {
                    v129 = false;
                }
                bool v130;
                v130 = v129 == false;
                if (v130){
                    assert("The indices should be inside the range of the dimension." && v129);
                } else {
                }
                bool v132;
                v132 = 0 <= v95;
                bool v134;
                if (v132){
                    bool v133;
                    v133 = v95 < 32;
                    v134 = v133;
                } else {
                    v134 = false;
                }
                bool v135;
                v135 = v134 == false;
                if (v135){
                    assert("The indices should be inside the range of the dimension." && v134);
                } else {
                }
                int v137;
                v137 = v95 * 4;
                int v138;
                v138 = v125 + v137;
                bool v139;
                v139 = 0 <= v123;
                bool v141;
                if (v139){
                    bool v140;
                    v140 = v123 < 1;
                    v141 = v140;
                } else {
                    v141 = false;
                }
                bool v142;
                v142 = v141 == false;
                if (v142){
                    assert("The indices should be inside the range of the dimension." && v141);
                } else {
                }
                int v144;
                v144 = v123 * 128;
                int v145;
                v145 = v138 + v144;
                assert("Tensor range check" && 0 <= v123 && v123 < 1);
                assert("Tensor range check" && 0 <= v125 && v125 < 4);
                int v146;
                v146 = 4 * v123;
                int v147;
                v147 = v146 + v125;
                v115[v147] = v145;
                v125 += 1 ;
            }
            v123 += 1 ;
        }
        bool v148;
        v148 = 0 <= v96;
        bool v149;
        v149 = v148 && v97;
        bool v150;
        v150 = v149 == false;
        if (v150){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v149);
        } else {
        }
        bool v152;
        v152 = v106 && v109;
        bool v153;
        v153 = v152 == false;
        if (v153){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v152);
        } else {
        }
        int v155;
        v155 = v104 * 8;
        int v156;
        v156 = v155 + v96;
        int v157[4];
        int v158[4];
        int v159;
        v159 = 0;
        while (while_method_3(v159)){
            int v161;
            v161 = 0;
            while (while_method_1(v161)){
                assert("Tensor range check" && 0 <= v159 && v159 < 1);
                assert("Tensor range check" && 0 <= v161 && v161 < 4);
                int v163;
                v163 = 4 * v159;
                int v164;
                v164 = v163 + v161;
                int v165;
                v165 = v115[v164];
                assert("Tensor range check" && 0 <= v159 && v159 < 1);
                assert("Tensor range check" && 0 <= v161 && v161 < 4);
                v157[v164] = v156;
                v158[v164] = v165;
                v161 += 1 ;
            }
            v159 += 1 ;
        }
        assert("Tensor range check" && 0 <= v104 && v104 < 8);
        int v166;
        v166 = 0;
        while (while_method_3(v166)){
            assert("Tensor range check" && 0 <= v166 && v166 < 1);
            int v168;
            v168 = 128 * v166;
            int v169;
            v169 = v168 + v113;
            assert("Tensor range check" && 0 <= v166 && v166 < 1);
            int v170;
            v170 = 4 * v166;
            int4* v171;
            v171 = reinterpret_cast<int4*>(v157 + v170);
            int4* v172;
            v172 = reinterpret_cast<int4*>(v9 + v169);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v171) % 16 == 0 && reinterpret_cast<unsigned long long>(v172) % 16 == 0);
            *v172 = *v171;
            int4* v173;
            v173 = reinterpret_cast<int4*>(v158 + v170);
            int4* v174;
            v174 = reinterpret_cast<int4*>(v10 + v169);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v173) % 16 == 0 && reinterpret_cast<unsigned long long>(v174) % 16 == 0);
            *v174 = *v173;
            v166 += 1 ;
        }
        v104 += 24 ;
    }
    v17.sync() ;
    int v175;
    v175 = threadIdx.x;
    bool v176;
    v176 = 0 <= v175;
    bool v177;
    v177 = v176 == false;
    if (v177){
        assert("The index needs to be zero or positive." && v176);
    } else {
    }
    int v179;
    v179 = v175 % 32;
    int v180;
    v180 = v175 / 32;
    bool v181;
    v181 = v180 < 8;
    bool v182;
    v182 = v181 == false;
    if (v182){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v181);
    } else {
    }
    assert("Tensor range check" && 0 <= v180 && v180 < 8);
    assert("Tensor range check" && 0 <= v179 && v179 < 32);
    int v184;
    v184 = 4 * v179;
    int v185;
    v185 = 128 * v180;
    int v186;
    v186 = v185 + v184;
    assert("Tensor range check" && 0 <= v180 && v180 < 8);
    int v187;
    v187 = blockIdx.x;
    int v188;
    v188 = v187;
    while (while_method_2(v188)){
        bool v190;
        v190 = 0 <= v188;
        bool v191;
        v191 = v190 == false;
        if (v191){
            assert("The index needs to be zero or positive." && v190);
        } else {
        }
        bool v193;
        v193 = v188 < 8;
        bool v194;
        v194 = v193 == false;
        if (v194){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v193);
        } else {
        }
        assert("Tensor range check" && 0 <= v188 && v188 < 8);
        int v196;
        v196 = 1024 * v188;
        int v197;
        v197 = v196 + v186;
        float v198[4];
        int v199[4];
        int v200;
        v200 = 0;
        while (while_method_3(v200)){
            assert("Tensor range check" && 0 <= v200 && v200 < 1);
            int v202;
            v202 = 4 * v200;
            assert("Tensor range check" && 0 <= v200 && v200 < 1);
            int v203;
            v203 = 128 * v200;
            int v204;
            v204 = v203 + v197;
            int4* v205;
            v205 = reinterpret_cast<int4*>(v1 + v204);
            int4* v206;
            v206 = reinterpret_cast<int4*>(v198 + v202);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v205) % 16 == 0 && reinterpret_cast<unsigned long long>(v206) % 16 == 0);
            *v206 = *v205;
            v200 += 1 ;
        }
        int v207;
        v207 = 0;
        while (while_method_3(v207)){
            int v209;
            v209 = 0;
            while (while_method_1(v209)){
                bool v211;
                v211 = 0 <= v209;
                bool v213;
                if (v211){
                    bool v212;
                    v212 = v209 < 4;
                    v213 = v212;
                } else {
                    v213 = false;
                }
                bool v214;
                v214 = v213 == false;
                if (v214){
                    assert("The indices should be inside the range of the dimension." && v213);
                } else {
                }
                bool v216;
                v216 = 0 <= v179;
                bool v218;
                if (v216){
                    bool v217;
                    v217 = v179 < 32;
                    v218 = v217;
                } else {
                    v218 = false;
                }
                bool v219;
                v219 = v218 == false;
                if (v219){
                    assert("The indices should be inside the range of the dimension." && v218);
                } else {
                }
                int v221;
                v221 = v179 * 4;
                int v222;
                v222 = v209 + v221;
                bool v223;
                v223 = 0 <= v207;
                bool v225;
                if (v223){
                    bool v224;
                    v224 = v207 < 1;
                    v225 = v224;
                } else {
                    v225 = false;
                }
                bool v226;
                v226 = v225 == false;
                if (v226){
                    assert("The indices should be inside the range of the dimension." && v225);
                } else {
                }
                int v228;
                v228 = v207 * 128;
                int v229;
                v229 = v222 + v228;
                assert("Tensor range check" && 0 <= v207 && v207 < 1);
                assert("Tensor range check" && 0 <= v209 && v209 < 4);
                int v230;
                v230 = 4 * v207;
                int v231;
                v231 = v230 + v209;
                v199[v231] = v229;
                v209 += 1 ;
            }
            v207 += 1 ;
        }
        bool v232;
        v232 = 0 <= v180;
        bool v233;
        v233 = v232 && v181;
        bool v234;
        v234 = v233 == false;
        if (v234){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v233);
        } else {
        }
        bool v236;
        v236 = v190 && v193;
        bool v237;
        v237 = v236 == false;
        if (v237){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v236);
        } else {
        }
        int v239;
        v239 = v188 * 8;
        int v240;
        v240 = v239 + v180;
        assert("Tensor range check" && 0 <= v188 && v188 < 8);
        int v241;
        v241 = 8 * v188;
        int v242;
        v242 = v241 + v180;
        v11[v242] = v240;
        v188 += 24 ;
    }
    v17.sync() ;
    int v243;
    v243 = threadIdx.x;
    bool v244;
    v244 = 0 <= v243;
    bool v245;
    v245 = v244 == false;
    if (v245){
        assert("The index needs to be zero or positive." && v244);
    } else {
    }
    int v247;
    v247 = v243 % 32;
    int v248;
    v248 = v243 / 32;
    bool v249;
    v249 = v248 < 8;
    bool v250;
    v250 = v249 == false;
    if (v250){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v249);
    } else {
    }
    assert("Tensor range check" && 0 <= v248 && v248 < 8);
    assert("Tensor range check" && 0 <= v247 && v247 < 32);
    int v252;
    v252 = 4 * v247;
    int v253;
    v253 = 128 * v248;
    int v254;
    v254 = v253 + v252;
    assert("Tensor range check" && 0 <= v248 && v248 < 8);
    assert("Tensor range check" && 0 <= v247 && v247 < 32);
    int v255;
    v255 = blockIdx.x;
    int v256;
    v256 = v255;
    while (while_method_2(v256)){
        bool v258;
        v258 = 0 <= v256;
        bool v259;
        v259 = v258 == false;
        if (v259){
            assert("The index needs to be zero or positive." && v258);
        } else {
        }
        bool v261;
        v261 = v256 < 8;
        bool v262;
        v262 = v261 == false;
        if (v262){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v261);
        } else {
        }
        assert("Tensor range check" && 0 <= v256 && v256 < 8);
        int v264;
        v264 = 1024 * v256;
        int v265;
        v265 = v264 + v254;
        float v266[4];
        int v267[4];
        int v268;
        v268 = 0;
        while (while_method_3(v268)){
            assert("Tensor range check" && 0 <= v268 && v268 < 1);
            int v270;
            v270 = 4 * v268;
            assert("Tensor range check" && 0 <= v268 && v268 < 1);
            int v271;
            v271 = 128 * v268;
            int v272;
            v272 = v271 + v265;
            int4* v273;
            v273 = reinterpret_cast<int4*>(v1 + v272);
            int4* v274;
            v274 = reinterpret_cast<int4*>(v266 + v270);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v273) % 16 == 0 && reinterpret_cast<unsigned long long>(v274) % 16 == 0);
            *v274 = *v273;
            v268 += 1 ;
        }
        int v275;
        v275 = 0;
        while (while_method_3(v275)){
            int v277;
            v277 = 0;
            while (while_method_1(v277)){
                bool v279;
                v279 = 0 <= v277;
                bool v281;
                if (v279){
                    bool v280;
                    v280 = v277 < 4;
                    v281 = v280;
                } else {
                    v281 = false;
                }
                bool v282;
                v282 = v281 == false;
                if (v282){
                    assert("The indices should be inside the range of the dimension." && v281);
                } else {
                }
                bool v284;
                v284 = 0 <= v247;
                bool v286;
                if (v284){
                    bool v285;
                    v285 = v247 < 32;
                    v286 = v285;
                } else {
                    v286 = false;
                }
                bool v287;
                v287 = v286 == false;
                if (v287){
                    assert("The indices should be inside the range of the dimension." && v286);
                } else {
                }
                int v289;
                v289 = v247 * 4;
                int v290;
                v290 = v277 + v289;
                bool v291;
                v291 = 0 <= v275;
                bool v293;
                if (v291){
                    bool v292;
                    v292 = v275 < 1;
                    v293 = v292;
                } else {
                    v293 = false;
                }
                bool v294;
                v294 = v293 == false;
                if (v294){
                    assert("The indices should be inside the range of the dimension." && v293);
                } else {
                }
                int v296;
                v296 = v275 * 128;
                int v297;
                v297 = v290 + v296;
                assert("Tensor range check" && 0 <= v275 && v275 < 1);
                assert("Tensor range check" && 0 <= v277 && v277 < 4);
                int v298;
                v298 = 4 * v275;
                int v299;
                v299 = v298 + v277;
                v267[v299] = v297;
                v277 += 1 ;
            }
            v275 += 1 ;
        }
        bool v300;
        v300 = 0 <= v248;
        bool v301;
        v301 = v300 && v249;
        bool v302;
        v302 = v301 == false;
        if (v302){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v301);
        } else {
        }
        bool v304;
        v304 = v258 && v261;
        bool v305;
        v305 = v304 == false;
        if (v305){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v304);
        } else {
        }
        int v307;
        v307 = v256 * 8;
        int v308;
        v308 = v307 + v248;
        float v309;
        v309 = 0.0f;
        int v310;
        v310 = 0;
        while (while_method_3(v310)){
            int v312;
            v312 = 0;
            while (while_method_1(v312)){
                assert("Tensor range check" && 0 <= v310 && v310 < 1);
                assert("Tensor range check" && 0 <= v312 && v312 < 4);
                int v314;
                v314 = 4 * v310;
                int v315;
                v315 = v314 + v312;
                float v316;
                v316 = v266[v315];
                float v317;
                v317 = v309 + v316;
                v309 = v317;
                v312 += 1 ;
            }
            v310 += 1 ;
        }
        auto v318 = cooperative_groups::coalesced_threads();
        int v319;
        v319 = threadIdx.x;
        int v320;
        v320 = v319 / 32;
        auto v321 = cooperative_groups::labeled_partition(v318,v320);
        Closure0 v322{};
        float v323;
        v323 = cooperative_groups::reduce(v321, v309, v322);
        float v324;
        v324 = v323 / 128.0f;
        float v325[4];
        int v326;
        v326 = 0;
        while (while_method_3(v326)){
            int v328;
            v328 = 0;
            while (while_method_1(v328)){
                assert("Tensor range check" && 0 <= v326 && v326 < 1);
                assert("Tensor range check" && 0 <= v328 && v328 < 4);
                int v330;
                v330 = 4 * v326;
                int v331;
                v331 = v330 + v328;
                float v332;
                v332 = v266[v331];
                float v333;
                v333 = v332 - v324;
                float v334;
                v334 = exp(v333);
                assert("Tensor range check" && 0 <= v326 && v326 < 1);
                assert("Tensor range check" && 0 <= v328 && v328 < 4);
                v325[v331] = v334;
                v328 += 1 ;
            }
            v326 += 1 ;
        }
        float v335;
        v335 = 0.0f;
        int v336;
        v336 = 0;
        while (while_method_3(v336)){
            int v338;
            v338 = 0;
            while (while_method_1(v338)){
                assert("Tensor range check" && 0 <= v336 && v336 < 1);
                assert("Tensor range check" && 0 <= v338 && v338 < 4);
                int v340;
                v340 = 4 * v336;
                int v341;
                v341 = v340 + v338;
                float v342;
                v342 = v325[v341];
                float v343;
                v343 = v335 + v342;
                v335 = v343;
                v338 += 1 ;
            }
            v336 += 1 ;
        }
        auto v344 = cooperative_groups::coalesced_threads();
        int v345;
        v345 = threadIdx.x;
        int v346;
        v346 = v345 / 32;
        auto v347 = cooperative_groups::labeled_partition(v344,v346);
        float v348;
        v348 = cooperative_groups::reduce(v347, v335, v322);
        float v349[4];
        int v350;
        v350 = 0;
        while (while_method_3(v350)){
            int v352;
            v352 = 0;
            while (while_method_1(v352)){
                assert("Tensor range check" && 0 <= v350 && v350 < 1);
                assert("Tensor range check" && 0 <= v352 && v352 < 4);
                int v354;
                v354 = 4 * v350;
                int v355;
                v355 = v354 + v352;
                float v356;
                v356 = v325[v355];
                float v357;
                v357 = v356 / v348;
                assert("Tensor range check" && 0 <= v350 && v350 < 1);
                assert("Tensor range check" && 0 <= v352 && v352 < 4);
                v349[v355] = v357;
                v352 += 1 ;
            }
            v350 += 1 ;
        }
        assert("Tensor range check" && 0 <= v256 && v256 < 8);
        int v358;
        v358 = 0;
        while (while_method_3(v358)){
            assert("Tensor range check" && 0 <= v358 && v358 < 1);
            int v360;
            v360 = 128 * v358;
            int v361;
            v361 = v360 + v265;
            assert("Tensor range check" && 0 <= v358 && v358 < 1);
            int v362;
            v362 = 4 * v358;
            int4* v363;
            v363 = reinterpret_cast<int4*>(v349 + v362);
            int4* v364;
            v364 = reinterpret_cast<int4*>(v3 + v361);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v363) % 16 == 0 && reinterpret_cast<unsigned long long>(v364) % 16 == 0);
            *v364 = *v363;
            v358 += 1 ;
        }
        v256 += 24 ;
    }
    v17.sync() ;
    int v365;
    v365 = threadIdx.x;
    bool v366;
    v366 = 0 <= v365;
    bool v367;
    v367 = v366 == false;
    if (v367){
        assert("The index needs to be zero or positive." && v366);
    } else {
    }
    int v369;
    v369 = v365 % 32;
    int v370;
    v370 = v365 / 32;
    bool v371;
    v371 = v370 < 8;
    bool v372;
    v372 = v371 == false;
    if (v372){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v371);
    } else {
    }
    assert("Tensor range check" && 0 <= v370 && v370 < 8);
    assert("Tensor range check" && 0 <= v369 && v369 < 32);
    int v374;
    v374 = 4 * v369;
    int v375;
    v375 = 128 * v370;
    int v376;
    v376 = v375 + v374;
    assert("Tensor range check" && 0 <= v370 && v370 < 8);
    assert("Tensor range check" && 0 <= v369 && v369 < 32);
    int v377;
    v377 = blockIdx.x;
    int v378;
    v378 = v377;
    while (while_method_2(v378)){
        bool v380;
        v380 = 0 <= v378;
        bool v381;
        v381 = v380 == false;
        if (v381){
            assert("The index needs to be zero or positive." && v380);
        } else {
        }
        bool v383;
        v383 = v378 < 8;
        bool v384;
        v384 = v383 == false;
        if (v384){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v383);
        } else {
        }
        assert("Tensor range check" && 0 <= v378 && v378 < 8);
        int v386;
        v386 = 1024 * v378;
        int v387;
        v387 = v386 + v376;
        float v388[4];
        int v389[4];
        int v390;
        v390 = 0;
        while (while_method_3(v390)){
            assert("Tensor range check" && 0 <= v390 && v390 < 1);
            int v392;
            v392 = 4 * v390;
            assert("Tensor range check" && 0 <= v390 && v390 < 1);
            int v393;
            v393 = 128 * v390;
            int v394;
            v394 = v393 + v387;
            int4* v395;
            v395 = reinterpret_cast<int4*>(v1 + v394);
            int4* v396;
            v396 = reinterpret_cast<int4*>(v388 + v392);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v395) % 16 == 0 && reinterpret_cast<unsigned long long>(v396) % 16 == 0);
            *v396 = *v395;
            v390 += 1 ;
        }
        int v397;
        v397 = 0;
        while (while_method_3(v397)){
            int v399;
            v399 = 0;
            while (while_method_1(v399)){
                bool v401;
                v401 = 0 <= v399;
                bool v403;
                if (v401){
                    bool v402;
                    v402 = v399 < 4;
                    v403 = v402;
                } else {
                    v403 = false;
                }
                bool v404;
                v404 = v403 == false;
                if (v404){
                    assert("The indices should be inside the range of the dimension." && v403);
                } else {
                }
                bool v406;
                v406 = 0 <= v369;
                bool v408;
                if (v406){
                    bool v407;
                    v407 = v369 < 32;
                    v408 = v407;
                } else {
                    v408 = false;
                }
                bool v409;
                v409 = v408 == false;
                if (v409){
                    assert("The indices should be inside the range of the dimension." && v408);
                } else {
                }
                int v411;
                v411 = v369 * 4;
                int v412;
                v412 = v399 + v411;
                bool v413;
                v413 = 0 <= v397;
                bool v415;
                if (v413){
                    bool v414;
                    v414 = v397 < 1;
                    v415 = v414;
                } else {
                    v415 = false;
                }
                bool v416;
                v416 = v415 == false;
                if (v416){
                    assert("The indices should be inside the range of the dimension." && v415);
                } else {
                }
                int v418;
                v418 = v397 * 128;
                int v419;
                v419 = v412 + v418;
                assert("Tensor range check" && 0 <= v397 && v397 < 1);
                assert("Tensor range check" && 0 <= v399 && v399 < 4);
                int v420;
                v420 = 4 * v397;
                int v421;
                v421 = v420 + v399;
                v389[v421] = v419;
                v399 += 1 ;
            }
            v397 += 1 ;
        }
        bool v422;
        v422 = 0 <= v370;
        bool v423;
        v423 = v422 && v371;
        bool v424;
        v424 = v423 == false;
        if (v424){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v423);
        } else {
        }
        bool v426;
        v426 = v380 && v383;
        bool v427;
        v427 = v426 == false;
        if (v427){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v426);
        } else {
        }
        int v429;
        v429 = v378 * 8;
        int v430;
        v430 = v429 + v370;
        float v431[4];
        int v432;
        v432 = 0;
        while (while_method_3(v432)){
            int v434;
            v434 = 0;
            while (while_method_1(v434)){
                assert("Tensor range check" && 0 <= v432 && v432 < 1);
                assert("Tensor range check" && 0 <= v434 && v434 < 4);
                int v436;
                v436 = 4 * v432;
                int v437;
                v437 = v436 + v434;
                float v438;
                v438 = v388[v437];
                float v439;
                v439 = v438 * v438;
                assert("Tensor range check" && 0 <= v432 && v432 < 1);
                assert("Tensor range check" && 0 <= v434 && v434 < 4);
                v431[v437] = v439;
                v434 += 1 ;
            }
            v432 += 1 ;
        }
        float v440;
        v440 = 0.0f;
        int v441;
        v441 = 0;
        while (while_method_3(v441)){
            int v443;
            v443 = 0;
            while (while_method_1(v443)){
                assert("Tensor range check" && 0 <= v441 && v441 < 1);
                assert("Tensor range check" && 0 <= v443 && v443 < 4);
                int v445;
                v445 = 4 * v441;
                int v446;
                v446 = v445 + v443;
                float v447;
                v447 = v431[v446];
                float v448;
                v448 = v440 + v447;
                v440 = v448;
                v443 += 1 ;
            }
            v441 += 1 ;
        }
        auto v449 = cooperative_groups::coalesced_threads();
        int v450;
        v450 = threadIdx.x;
        int v451;
        v451 = v450 / 32;
        auto v452 = cooperative_groups::labeled_partition(v449,v451);
        Closure0 v453{};
        float v454;
        v454 = cooperative_groups::reduce(v452, v440, v453);
        float v455[4];
        int v456;
        v456 = 0;
        while (while_method_3(v456)){
            int v458;
            v458 = 0;
            while (while_method_1(v458)){
                assert("Tensor range check" && 0 <= v456 && v456 < 1);
                assert("Tensor range check" && 0 <= v458 && v458 < 4);
                int v460;
                v460 = 4 * v456;
                int v461;
                v461 = v460 + v458;
                float v462;
                v462 = v388[v461];
                bool v463;
                v463 = v454 == 0.0f;
                bool v464;
                v464 = v463 != true;
                float v466;
                if (v464){
                    float v465;
                    v465 = v462 / v454;
                    v466 = v465;
                } else {
                    v466 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v456 && v456 < 1);
                assert("Tensor range check" && 0 <= v458 && v458 < 4);
                v455[v461] = v466;
                v458 += 1 ;
            }
            v456 += 1 ;
        }
        assert("Tensor range check" && 0 <= v378 && v378 < 8);
        int v467;
        v467 = 0;
        while (while_method_3(v467)){
            assert("Tensor range check" && 0 <= v467 && v467 < 1);
            int v469;
            v469 = 128 * v467;
            int v470;
            v470 = v469 + v387;
            assert("Tensor range check" && 0 <= v467 && v467 < 1);
            int v471;
            v471 = 4 * v467;
            int4* v472;
            v472 = reinterpret_cast<int4*>(v455 + v471);
            int4* v473;
            v473 = reinterpret_cast<int4*>(v7 + v470);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v472) % 16 == 0 && reinterpret_cast<unsigned long long>(v473) % 16 == 0);
            *v473 = *v472;
            v467 += 1 ;
        }
        v378 += 24 ;
    }
    v17.sync() ;
    int v474;
    v474 = threadIdx.x;
    bool v475;
    v475 = 0 <= v474;
    bool v476;
    v476 = v475 == false;
    if (v476){
        assert("The index needs to be zero or positive." && v475);
    } else {
    }
    int v478;
    v478 = v474 % 32;
    int v479;
    v479 = v474 / 32;
    bool v480;
    v480 = v479 < 8;
    bool v481;
    v481 = v480 == false;
    if (v481){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v480);
    } else {
    }
    assert("Tensor range check" && 0 <= v479 && v479 < 8);
    assert("Tensor range check" && 0 <= v478 && v478 < 32);
    int v483;
    v483 = 4 * v478;
    int v484;
    v484 = 128 * v479;
    int v485;
    v485 = v484 + v483;
    assert("Tensor range check" && 0 <= v479 && v479 < 8);
    int v486;
    v486 = blockIdx.x;
    int v487;
    v487 = v486;
    while (while_method_2(v487)){
        bool v489;
        v489 = 0 <= v487;
        bool v490;
        v490 = v489 == false;
        if (v490){
            assert("The index needs to be zero or positive." && v489);
        } else {
        }
        bool v492;
        v492 = v487 < 8;
        bool v493;
        v493 = v492 == false;
        if (v493){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v492);
        } else {
        }
        assert("Tensor range check" && 0 <= v487 && v487 < 8);
        int v495;
        v495 = 1024 * v487;
        int v496;
        v496 = v495 + v485;
        float v497[4];
        int v498[4];
        int v499;
        v499 = 0;
        while (while_method_3(v499)){
            assert("Tensor range check" && 0 <= v499 && v499 < 1);
            int v501;
            v501 = 4 * v499;
            assert("Tensor range check" && 0 <= v499 && v499 < 1);
            int v502;
            v502 = 128 * v499;
            int v503;
            v503 = v502 + v496;
            int4* v504;
            v504 = reinterpret_cast<int4*>(v1 + v503);
            int4* v505;
            v505 = reinterpret_cast<int4*>(v497 + v501);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v504) % 16 == 0 && reinterpret_cast<unsigned long long>(v505) % 16 == 0);
            *v505 = *v504;
            v499 += 1 ;
        }
        int v506;
        v506 = 0;
        while (while_method_3(v506)){
            int v508;
            v508 = 0;
            while (while_method_1(v508)){
                bool v510;
                v510 = 0 <= v508;
                bool v512;
                if (v510){
                    bool v511;
                    v511 = v508 < 4;
                    v512 = v511;
                } else {
                    v512 = false;
                }
                bool v513;
                v513 = v512 == false;
                if (v513){
                    assert("The indices should be inside the range of the dimension." && v512);
                } else {
                }
                bool v515;
                v515 = 0 <= v478;
                bool v517;
                if (v515){
                    bool v516;
                    v516 = v478 < 32;
                    v517 = v516;
                } else {
                    v517 = false;
                }
                bool v518;
                v518 = v517 == false;
                if (v518){
                    assert("The indices should be inside the range of the dimension." && v517);
                } else {
                }
                int v520;
                v520 = v478 * 4;
                int v521;
                v521 = v508 + v520;
                bool v522;
                v522 = 0 <= v506;
                bool v524;
                if (v522){
                    bool v523;
                    v523 = v506 < 1;
                    v524 = v523;
                } else {
                    v524 = false;
                }
                bool v525;
                v525 = v524 == false;
                if (v525){
                    assert("The indices should be inside the range of the dimension." && v524);
                } else {
                }
                int v527;
                v527 = v506 * 128;
                int v528;
                v528 = v521 + v527;
                assert("Tensor range check" && 0 <= v506 && v506 < 1);
                assert("Tensor range check" && 0 <= v508 && v508 < 4);
                int v529;
                v529 = 4 * v506;
                int v530;
                v530 = v529 + v508;
                v498[v530] = v528;
                v508 += 1 ;
            }
            v506 += 1 ;
        }
        bool v531;
        v531 = 0 <= v479;
        bool v532;
        v532 = v531 && v480;
        bool v533;
        v533 = v532 == false;
        if (v533){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v532);
        } else {
        }
        bool v535;
        v535 = v489 && v492;
        bool v536;
        v536 = v535 == false;
        if (v536){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v535);
        } else {
        }
        int v538;
        v538 = v487 * 8;
        int v539;
        v539 = v538 + v479;
        float v540; int v541;
        Tuple1 tmp87 = Tuple1{-1.0f / 0.0f, 0};
        v540 = tmp87.v0; v541 = tmp87.v1;
        int v542;
        v542 = 0;
        while (while_method_3(v542)){
            int v544;
            v544 = 0;
            while (while_method_1(v544)){
                assert("Tensor range check" && 0 <= v542 && v542 < 1);
                assert("Tensor range check" && 0 <= v544 && v544 < 4);
                int v546;
                v546 = 4 * v542;
                int v547;
                v547 = v546 + v544;
                float v548;
                v548 = v497[v547];
                int v549;
                v549 = v498[v547];
                bool v550;
                v550 = v540 > v548;
                float v551; int v552;
                if (v550){
                    v551 = v540; v552 = v541;
                } else {
                    v551 = v548; v552 = v549;
                }
                v540 = v551;
                v541 = v552;
                v544 += 1 ;
            }
            v542 += 1 ;
        }
        auto v553 = cooperative_groups::coalesced_threads();
        int v554;
        v554 = threadIdx.x;
        int v555;
        v555 = v554 / 32;
        auto v556 = cooperative_groups::labeled_partition(v553,v555);
        Closure1 v557{};
        float v558; int v559;
        Tuple1 tmp88 = cooperative_groups::reduce(v556, Tuple1{v540, v541}, v557);
        v558 = tmp88.v0; v559 = tmp88.v1;
        assert("Tensor range check" && 0 <= v487 && v487 < 8);
        int v560;
        v560 = 8 * v487;
        int v561;
        v561 = v560 + v479;
        v8[v561] = v559;
        v487 += 24 ;
    }
    v17.sync() ;
    int v562;
    v562 = threadIdx.x;
    bool v563;
    v563 = 0 <= v562;
    bool v564;
    v564 = v563 == false;
    if (v564){
        assert("The index needs to be zero or positive." && v563);
    } else {
    }
    int v566;
    v566 = v562 % 32;
    int v567;
    v567 = v562 / 32;
    bool v568;
    v568 = v567 < 8;
    bool v569;
    v569 = v568 == false;
    if (v569){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v568);
    } else {
    }
    assert("Tensor range check" && 0 <= v567 && v567 < 8);
    assert("Tensor range check" && 0 <= v566 && v566 < 32);
    int v571;
    v571 = 4 * v566;
    int v572;
    v572 = 128 * v567;
    int v573;
    v573 = v572 + v571;
    assert("Tensor range check" && 0 <= v567 && v567 < 8);
    assert("Tensor range check" && 0 <= v566 && v566 < 32);
    int v574;
    v574 = blockIdx.x;
    int v575;
    v575 = v574;
    while (while_method_2(v575)){
        bool v577;
        v577 = 0 <= v575;
        bool v578;
        v578 = v577 == false;
        if (v578){
            assert("The index needs to be zero or positive." && v577);
        } else {
        }
        bool v580;
        v580 = v575 < 8;
        bool v581;
        v581 = v580 == false;
        if (v581){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v580);
        } else {
        }
        assert("Tensor range check" && 0 <= v575 && v575 < 8);
        int v583;
        v583 = 1024 * v575;
        int v584;
        v584 = v583 + v573;
        float v585[4];
        int v586[4];
        int v587;
        v587 = 0;
        while (while_method_3(v587)){
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v589;
            v589 = 4 * v587;
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v590;
            v590 = 128 * v587;
            int v591;
            v591 = v590 + v584;
            int4* v592;
            v592 = reinterpret_cast<int4*>(v1 + v591);
            int4* v593;
            v593 = reinterpret_cast<int4*>(v585 + v589);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v592) % 16 == 0 && reinterpret_cast<unsigned long long>(v593) % 16 == 0);
            *v593 = *v592;
            v587 += 1 ;
        }
        int v594;
        v594 = 0;
        while (while_method_3(v594)){
            int v596;
            v596 = 0;
            while (while_method_1(v596)){
                bool v598;
                v598 = 0 <= v596;
                bool v600;
                if (v598){
                    bool v599;
                    v599 = v596 < 4;
                    v600 = v599;
                } else {
                    v600 = false;
                }
                bool v601;
                v601 = v600 == false;
                if (v601){
                    assert("The indices should be inside the range of the dimension." && v600);
                } else {
                }
                bool v603;
                v603 = 0 <= v566;
                bool v605;
                if (v603){
                    bool v604;
                    v604 = v566 < 32;
                    v605 = v604;
                } else {
                    v605 = false;
                }
                bool v606;
                v606 = v605 == false;
                if (v606){
                    assert("The indices should be inside the range of the dimension." && v605);
                } else {
                }
                int v608;
                v608 = v566 * 4;
                int v609;
                v609 = v596 + v608;
                bool v610;
                v610 = 0 <= v594;
                bool v612;
                if (v610){
                    bool v611;
                    v611 = v594 < 1;
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
                int v615;
                v615 = v594 * 128;
                int v616;
                v616 = v609 + v615;
                assert("Tensor range check" && 0 <= v594 && v594 < 1);
                assert("Tensor range check" && 0 <= v596 && v596 < 4);
                int v617;
                v617 = 4 * v594;
                int v618;
                v618 = v617 + v596;
                v586[v618] = v616;
                v596 += 1 ;
            }
            v594 += 1 ;
        }
        bool v619;
        v619 = 0 <= v567;
        bool v620;
        v620 = v619 && v568;
        bool v621;
        v621 = v620 == false;
        if (v621){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v620);
        } else {
        }
        bool v623;
        v623 = v577 && v580;
        bool v624;
        v624 = v623 == false;
        if (v624){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v623);
        } else {
        }
        int v626;
        v626 = v575 * 8;
        int v627;
        v627 = v626 + v567;
        float v628;
        v628 = 0.0f;
        int v629;
        v629 = 0;
        while (while_method_3(v629)){
            int v631;
            v631 = 0;
            while (while_method_1(v631)){
                assert("Tensor range check" && 0 <= v629 && v629 < 1);
                assert("Tensor range check" && 0 <= v631 && v631 < 4);
                int v633;
                v633 = 4 * v629;
                int v634;
                v634 = v633 + v631;
                float v635;
                v635 = v585[v634];
                float v636;
                v636 = v628 + v635;
                v628 = v636;
                v631 += 1 ;
            }
            v629 += 1 ;
        }
        auto v637 = cooperative_groups::coalesced_threads();
        int v638;
        v638 = threadIdx.x;
        int v639;
        v639 = v638 / 32;
        auto v640 = cooperative_groups::labeled_partition(v637,v639);
        Closure0 v641{};
        float v642;
        v642 = cooperative_groups::reduce(v640, v628, v641);
        float v643;
        v643 = v642 / 128.0f;
        float v644[4];
        int v645;
        v645 = 0;
        while (while_method_3(v645)){
            int v647;
            v647 = 0;
            while (while_method_1(v647)){
                assert("Tensor range check" && 0 <= v645 && v645 < 1);
                assert("Tensor range check" && 0 <= v647 && v647 < 4);
                int v649;
                v649 = 4 * v645;
                int v650;
                v650 = v649 + v647;
                float v651;
                v651 = v585[v650];
                float v652;
                v652 = v651 - v643;
                float v653;
                v653 = exp(v652);
                assert("Tensor range check" && 0 <= v645 && v645 < 1);
                assert("Tensor range check" && 0 <= v647 && v647 < 4);
                v644[v650] = v653;
                v647 += 1 ;
            }
            v645 += 1 ;
        }
        float v654;
        v654 = 0.0f;
        int v655;
        v655 = 0;
        while (while_method_3(v655)){
            int v657;
            v657 = 0;
            while (while_method_1(v657)){
                assert("Tensor range check" && 0 <= v655 && v655 < 1);
                assert("Tensor range check" && 0 <= v657 && v657 < 4);
                int v659;
                v659 = 4 * v655;
                int v660;
                v660 = v659 + v657;
                float v661;
                v661 = v644[v660];
                float v662;
                v662 = v654 + v661;
                v654 = v662;
                v657 += 1 ;
            }
            v655 += 1 ;
        }
        auto v663 = cooperative_groups::coalesced_threads();
        int v664;
        v664 = threadIdx.x;
        int v665;
        v665 = v664 / 32;
        auto v666 = cooperative_groups::labeled_partition(v663,v665);
        float v667;
        v667 = cooperative_groups::reduce(v666, v654, v641);
        float v668[4];
        int v669;
        v669 = 0;
        while (while_method_3(v669)){
            int v671;
            v671 = 0;
            while (while_method_1(v671)){
                assert("Tensor range check" && 0 <= v669 && v669 < 1);
                assert("Tensor range check" && 0 <= v671 && v671 < 4);
                int v673;
                v673 = 4 * v669;
                int v674;
                v674 = v673 + v671;
                float v675;
                v675 = v644[v674];
                float v676;
                v676 = v675 / v667;
                assert("Tensor range check" && 0 <= v669 && v669 < 1);
                assert("Tensor range check" && 0 <= v671 && v671 < 4);
                v668[v674] = v676;
                v671 += 1 ;
            }
            v669 += 1 ;
        }
        float v677[4];
        float v678;
        v678 = 0.0f;
        int v679;
        v679 = 0;
        while (while_method_3(v679)){
            assert("Tensor range check" && 0 <= v679 && v679 < 1);
            int v681;
            v681 = 4 * v679;
            assert("Tensor range check" && 0 <= v679 && v679 < 1);
            int v682; float v683;
            Tuple0 tmp89 = Tuple0{0, 0.0f};
            v682 = tmp89.v0; v683 = tmp89.v1;
            while (while_method_1(v682)){
                assert("Tensor range check" && 0 <= v682 && v682 < 4);
                int v685;
                v685 = v682 + v681;
                float v686;
                v686 = v668[v685];
                float v687;
                v687 = v683 + v686;
                v683 = v687;
                v682 += 1 ;
            }
            auto v688 = cooperative_groups::coalesced_threads();
            int v689;
            v689 = threadIdx.x;
            int v690;
            v690 = v689 / 32;
            auto v691 = cooperative_groups::labeled_partition(v688,v690);
            Closure2 v692{};
            float v693;
            v693 = cooperative_groups::inclusive_scan(v691, v683, v692);
            float v694;
            v694 = v691.shfl_up(v693,1);
            bool v695;
            v695 = v691.thread_rank() == 0;
            float v696;
            if (v695){
                v696 = 0.0f;
            } else {
                v696 = v694;
            }
            float v697;
            v697 = v691.shfl(v693,v691.num_threads()-1);
            float v698;
            v698 = v678 + v696;
            int v699; float v700;
            Tuple0 tmp90 = Tuple0{0, v698};
            v699 = tmp90.v0; v700 = tmp90.v1;
            while (while_method_1(v699)){
                assert("Tensor range check" && 0 <= v699 && v699 < 4);
                int v702;
                v702 = v699 + v681;
                float v703;
                v703 = v668[v702];
                float v704;
                v704 = v700 + v703;
                assert("Tensor range check" && 0 <= v699 && v699 < 4);
                v677[v702] = v704;
                v700 = v704;
                v699 += 1 ;
            }
            float v705;
            v705 = v678 + v697;
            v678 = v705;
            v679 += 1 ;
        }
        assert("Tensor range check" && 0 <= v575 && v575 < 8);
        int v706;
        v706 = 0;
        while (while_method_3(v706)){
            assert("Tensor range check" && 0 <= v706 && v706 < 1);
            int v708;
            v708 = 128 * v706;
            int v709;
            v709 = v708 + v584;
            assert("Tensor range check" && 0 <= v706 && v706 < 1);
            int v710;
            v710 = 4 * v706;
            int4* v711;
            v711 = reinterpret_cast<int4*>(v668 + v710);
            int4* v712;
            v712 = reinterpret_cast<int4*>(v5 + v709);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v711) % 16 == 0 && reinterpret_cast<unsigned long long>(v712) % 16 == 0);
            *v712 = *v711;
            int4* v713;
            v713 = reinterpret_cast<int4*>(v677 + v710);
            int4* v714;
            v714 = reinterpret_cast<int4*>(v6 + v709);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v713) % 16 == 0 && reinterpret_cast<unsigned long long>(v714) % 16 == 0);
            *v714 = *v713;
            v706 += 1 ;
        }
        v575 += 24 ;
    }
    v17.sync() ;
    int v715;
    v715 = threadIdx.x;
    bool v716;
    v716 = 0 <= v715;
    bool v717;
    v717 = v716 == false;
    if (v717){
        assert("The index needs to be zero or positive." && v716);
    } else {
    }
    int v719;
    v719 = v715 % 32;
    int v720;
    v720 = v715 / 32;
    bool v721;
    v721 = v720 < 8;
    bool v722;
    v722 = v721 == false;
    if (v722){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v721);
    } else {
    }
    assert("Tensor range check" && 0 <= v720 && v720 < 8);
    assert("Tensor range check" && 0 <= v719 && v719 < 32);
    int v724;
    v724 = 4 * v719;
    int v725;
    v725 = 128 * v720;
    int v726;
    v726 = v725 + v724;
    assert("Tensor range check" && 0 <= v720 && v720 < 8);
    assert("Tensor range check" && 0 <= v719 && v719 < 32);
    int v727;
    v727 = blockIdx.x;
    int v728;
    v728 = v727;
    while (while_method_2(v728)){
        bool v730;
        v730 = 0 <= v728;
        bool v731;
        v731 = v730 == false;
        if (v731){
            assert("The index needs to be zero or positive." && v730);
        } else {
        }
        bool v733;
        v733 = v728 < 8;
        bool v734;
        v734 = v733 == false;
        if (v734){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v733);
        } else {
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v736;
        v736 = 1024 * v728;
        int v737;
        v737 = v736 + v726;
        int v738[4];
        int v739[4];
        int v740;
        v740 = 0;
        while (while_method_3(v740)){
            assert("Tensor range check" && 0 <= v740 && v740 < 1);
            int v742;
            v742 = 4 * v740;
            assert("Tensor range check" && 0 <= v740 && v740 < 1);
            int v743;
            v743 = 128 * v740;
            int v744;
            v744 = v743 + v737;
            int4* v745;
            v745 = reinterpret_cast<int4*>(v0 + v744);
            int4* v746;
            v746 = reinterpret_cast<int4*>(v738 + v742);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v745) % 16 == 0 && reinterpret_cast<unsigned long long>(v746) % 16 == 0);
            *v746 = *v745;
            v740 += 1 ;
        }
        int v747;
        v747 = 0;
        while (while_method_3(v747)){
            int v749;
            v749 = 0;
            while (while_method_1(v749)){
                bool v751;
                v751 = 0 <= v749;
                bool v753;
                if (v751){
                    bool v752;
                    v752 = v749 < 4;
                    v753 = v752;
                } else {
                    v753 = false;
                }
                bool v754;
                v754 = v753 == false;
                if (v754){
                    assert("The indices should be inside the range of the dimension." && v753);
                } else {
                }
                bool v756;
                v756 = 0 <= v719;
                bool v758;
                if (v756){
                    bool v757;
                    v757 = v719 < 32;
                    v758 = v757;
                } else {
                    v758 = false;
                }
                bool v759;
                v759 = v758 == false;
                if (v759){
                    assert("The indices should be inside the range of the dimension." && v758);
                } else {
                }
                int v761;
                v761 = v719 * 4;
                int v762;
                v762 = v749 + v761;
                bool v763;
                v763 = 0 <= v747;
                bool v765;
                if (v763){
                    bool v764;
                    v764 = v747 < 1;
                    v765 = v764;
                } else {
                    v765 = false;
                }
                bool v766;
                v766 = v765 == false;
                if (v766){
                    assert("The indices should be inside the range of the dimension." && v765);
                } else {
                }
                int v768;
                v768 = v747 * 128;
                int v769;
                v769 = v762 + v768;
                assert("Tensor range check" && 0 <= v747 && v747 < 1);
                assert("Tensor range check" && 0 <= v749 && v749 < 4);
                int v770;
                v770 = 4 * v747;
                int v771;
                v771 = v770 + v749;
                v739[v771] = v769;
                v749 += 1 ;
            }
            v747 += 1 ;
        }
        bool v772;
        v772 = 0 <= v720;
        bool v773;
        v773 = v772 && v721;
        bool v774;
        v774 = v773 == false;
        if (v774){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v773);
        } else {
        }
        bool v776;
        v776 = v730 && v733;
        bool v777;
        v777 = v776 == false;
        if (v777){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v776);
        } else {
        }
        int v779;
        v779 = v728 * 8;
        int v780;
        v780 = v779 + v720;
        int v781[4];
        int v782;
        v782 = 0;
        int v783;
        v783 = 0;
        while (while_method_3(v783)){
            assert("Tensor range check" && 0 <= v783 && v783 < 1);
            int v785;
            v785 = 4 * v783;
            assert("Tensor range check" && 0 <= v783 && v783 < 1);
            int v786; int v787;
            Tuple2 tmp91 = Tuple2{0, 0};
            v786 = tmp91.v0; v787 = tmp91.v1;
            while (while_method_1(v786)){
                assert("Tensor range check" && 0 <= v786 && v786 < 4);
                int v789;
                v789 = v786 + v785;
                int v790;
                v790 = v738[v789];
                int v791;
                v791 = v787 + v790;
                v787 = v791;
                v786 += 1 ;
            }
            auto v792 = cooperative_groups::coalesced_threads();
            int v793;
            v793 = threadIdx.x;
            int v794;
            v794 = v793 / 32;
            auto v795 = cooperative_groups::labeled_partition(v792,v794);
            Closure3 v796{};
            int v797;
            v797 = cooperative_groups::inclusive_scan(v795, v787, v796);
            int v798;
            v798 = v795.shfl_up(v797,1);
            bool v799;
            v799 = v795.thread_rank() == 0;
            int v800;
            if (v799){
                v800 = 0;
            } else {
                v800 = v798;
            }
            int v801;
            v801 = v795.shfl(v797,v795.num_threads()-1);
            int v802;
            v802 = v782 + v800;
            int v803; int v804;
            Tuple2 tmp92 = Tuple2{0, v802};
            v803 = tmp92.v0; v804 = tmp92.v1;
            while (while_method_1(v803)){
                assert("Tensor range check" && 0 <= v803 && v803 < 4);
                int v806;
                v806 = v803 + v785;
                int v807;
                v807 = v738[v806];
                assert("Tensor range check" && 0 <= v803 && v803 < 4);
                v781[v806] = v804;
                int v808;
                v808 = v804 + v807;
                v804 = v808;
                v803 += 1 ;
            }
            int v809;
            v809 = v782 + v801;
            v782 = v809;
            v783 += 1 ;
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v810;
        v810 = 0;
        while (while_method_3(v810)){
            assert("Tensor range check" && 0 <= v810 && v810 < 1);
            int v812;
            v812 = 128 * v810;
            int v813;
            v813 = v812 + v737;
            assert("Tensor range check" && 0 <= v810 && v810 < 1);
            int v814;
            v814 = 4 * v810;
            int4* v815;
            v815 = reinterpret_cast<int4*>(v781 + v814);
            int4* v816;
            v816 = reinterpret_cast<int4*>(v12 + v813);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v815) % 16 == 0 && reinterpret_cast<unsigned long long>(v816) % 16 == 0);
            *v816 = *v815;
            v810 += 1 ;
        }
        v728 += 24 ;
    }
    v17.sync() ;
    int v817;
    v817 = threadIdx.x;
    bool v818;
    v818 = 0 <= v817;
    bool v819;
    v819 = v818 == false;
    if (v819){
        assert("The index needs to be zero or positive." && v818);
    } else {
    }
    int v821;
    v821 = v817 % 32;
    int v822;
    v822 = v817 / 32;
    bool v823;
    v823 = v822 < 8;
    bool v824;
    v824 = v823 == false;
    if (v824){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v823);
    } else {
    }
    assert("Tensor range check" && 0 <= v822 && v822 < 8);
    assert("Tensor range check" && 0 <= v821 && v821 < 32);
    int v826;
    v826 = 4 * v821;
    int v827;
    v827 = 128 * v822;
    int v828;
    v828 = v827 + v826;
    assert("Tensor range check" && 0 <= v822 && v822 < 8);
    assert("Tensor range check" && 0 <= v821 && v821 < 32);
    int v829;
    v829 = blockIdx.x;
    int v830;
    v830 = v829;
    while (while_method_2(v830)){
        bool v832;
        v832 = 0 <= v830;
        bool v833;
        v833 = v832 == false;
        if (v833){
            assert("The index needs to be zero or positive." && v832);
        } else {
        }
        bool v835;
        v835 = v830 < 8;
        bool v836;
        v836 = v835 == false;
        if (v836){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v835);
        } else {
        }
        assert("Tensor range check" && 0 <= v830 && v830 < 8);
        int v838;
        v838 = 1024 * v830;
        int v839;
        v839 = v838 + v828;
        float v840[4];
        int v841[4];
        int v842;
        v842 = 0;
        while (while_method_3(v842)){
            assert("Tensor range check" && 0 <= v842 && v842 < 1);
            int v844;
            v844 = 4 * v842;
            assert("Tensor range check" && 0 <= v842 && v842 < 1);
            int v845;
            v845 = 128 * v842;
            int v846;
            v846 = v845 + v839;
            int4* v847;
            v847 = reinterpret_cast<int4*>(v1 + v846);
            int4* v848;
            v848 = reinterpret_cast<int4*>(v840 + v844);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v847) % 16 == 0 && reinterpret_cast<unsigned long long>(v848) % 16 == 0);
            *v848 = *v847;
            v842 += 1 ;
        }
        int v849;
        v849 = 0;
        while (while_method_3(v849)){
            int v851;
            v851 = 0;
            while (while_method_1(v851)){
                bool v853;
                v853 = 0 <= v851;
                bool v855;
                if (v853){
                    bool v854;
                    v854 = v851 < 4;
                    v855 = v854;
                } else {
                    v855 = false;
                }
                bool v856;
                v856 = v855 == false;
                if (v856){
                    assert("The indices should be inside the range of the dimension." && v855);
                } else {
                }
                bool v858;
                v858 = 0 <= v821;
                bool v860;
                if (v858){
                    bool v859;
                    v859 = v821 < 32;
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
                int v863;
                v863 = v821 * 4;
                int v864;
                v864 = v851 + v863;
                bool v865;
                v865 = 0 <= v849;
                bool v867;
                if (v865){
                    bool v866;
                    v866 = v849 < 1;
                    v867 = v866;
                } else {
                    v867 = false;
                }
                bool v868;
                v868 = v867 == false;
                if (v868){
                    assert("The indices should be inside the range of the dimension." && v867);
                } else {
                }
                int v870;
                v870 = v849 * 128;
                int v871;
                v871 = v864 + v870;
                assert("Tensor range check" && 0 <= v849 && v849 < 1);
                assert("Tensor range check" && 0 <= v851 && v851 < 4);
                int v872;
                v872 = 4 * v849;
                int v873;
                v873 = v872 + v851;
                v841[v873] = v871;
                v851 += 1 ;
            }
            v849 += 1 ;
        }
        bool v874;
        v874 = 0 <= v822;
        bool v875;
        v875 = v874 && v823;
        bool v876;
        v876 = v875 == false;
        if (v876){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v875);
        } else {
        }
        bool v878;
        v878 = v832 && v835;
        bool v879;
        v879 = v878 == false;
        if (v879){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v878);
        } else {
        }
        int v881;
        v881 = v830 * 8;
        int v882;
        v882 = v881 + v822;
        bool v883[4];
        int v884;
        v884 = 0;
        while (while_method_3(v884)){
            int v886;
            v886 = 0;
            while (while_method_1(v886)){
                assert("Tensor range check" && 0 <= v884 && v884 < 1);
                assert("Tensor range check" && 0 <= v886 && v886 < 4);
                int v888;
                v888 = 4 * v884;
                int v889;
                v889 = v888 + v886;
                float v890;
                v890 = v840[v889];
                int v891;
                v891 = v841[v889];
                bool v892;
                v892 = v891 < 4;
                assert("Tensor range check" && 0 <= v884 && v884 < 1);
                assert("Tensor range check" && 0 <= v886 && v886 < 4);
                v883[v889] = v892;
                v886 += 1 ;
            }
            v884 += 1 ;
        }
        int v893[4];
        int v894;
        v894 = 0;
        while (while_method_3(v894)){
            int v896;
            v896 = 0;
            while (while_method_1(v896)){
                assert("Tensor range check" && 0 <= v894 && v894 < 1);
                assert("Tensor range check" && 0 <= v896 && v896 < 4);
                int v898;
                v898 = 4 * v894;
                int v899;
                v899 = v898 + v896;
                bool v900;
                v900 = v883[v899];
                int v901;
                if (v900){
                    v901 = 1;
                } else {
                    v901 = 0;
                }
                assert("Tensor range check" && 0 <= v894 && v894 < 1);
                assert("Tensor range check" && 0 <= v896 && v896 < 4);
                v893[v899] = v901;
                v896 += 1 ;
            }
            v894 += 1 ;
        }
        int v902;
        v902 = 0;
        int v903;
        v903 = 0;
        while (while_method_3(v903)){
            int v905;
            v905 = 0;
            while (while_method_1(v905)){
                assert("Tensor range check" && 0 <= v903 && v903 < 1);
                assert("Tensor range check" && 0 <= v905 && v905 < 4);
                int v907;
                v907 = 4 * v903;
                int v908;
                v908 = v907 + v905;
                int v909;
                v909 = v893[v908];
                int v910;
                v910 = v902 + v909;
                v902 = v910;
                v905 += 1 ;
            }
            v903 += 1 ;
        }
        auto v911 = cooperative_groups::coalesced_threads();
        int v912;
        v912 = threadIdx.x;
        int v913;
        v913 = v912 / 32;
        auto v914 = cooperative_groups::labeled_partition(v911,v913);
        Closure4 v915{};
        int v916;
        v916 = cooperative_groups::reduce(v914, v902, v915);
        float v917[4];
        int v918;
        v918 = 0;
        while (while_method_3(v918)){
            int v920;
            v920 = 0;
            while (while_method_1(v920)){
                assert("Tensor range check" && 0 <= v918 && v918 < 1);
                assert("Tensor range check" && 0 <= v920 && v920 < 4);
                int v922;
                v922 = 4 * v918;
                int v923;
                v923 = v922 + v920;
                float v924;
                v924 = v840[v923];
                bool v925;
                v925 = v883[v923];
                float v926;
                if (v925){
                    v926 = v924;
                } else {
                    v926 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v918 && v918 < 1);
                assert("Tensor range check" && 0 <= v920 && v920 < 4);
                v917[v923] = v926;
                v920 += 1 ;
            }
            v918 += 1 ;
        }
        float v927;
        v927 = 0.0f;
        int v928;
        v928 = 0;
        while (while_method_3(v928)){
            int v930;
            v930 = 0;
            while (while_method_1(v930)){
                assert("Tensor range check" && 0 <= v928 && v928 < 1);
                assert("Tensor range check" && 0 <= v930 && v930 < 4);
                int v932;
                v932 = 4 * v928;
                int v933;
                v933 = v932 + v930;
                float v934;
                v934 = v917[v933];
                float v935;
                v935 = v927 + v934;
                v927 = v935;
                v930 += 1 ;
            }
            v928 += 1 ;
        }
        auto v936 = cooperative_groups::coalesced_threads();
        int v937;
        v937 = threadIdx.x;
        int v938;
        v938 = v937 / 32;
        auto v939 = cooperative_groups::labeled_partition(v936,v938);
        Closure0 v940{};
        float v941;
        v941 = cooperative_groups::reduce(v939, v927, v940);
        float v942;
        v942 = (float)v916;
        float v943;
        v943 = v941 / v942;
        float v944[4];
        int v945;
        v945 = 0;
        while (while_method_3(v945)){
            int v947;
            v947 = 0;
            while (while_method_1(v947)){
                assert("Tensor range check" && 0 <= v945 && v945 < 1);
                assert("Tensor range check" && 0 <= v947 && v947 < 4);
                int v949;
                v949 = 4 * v945;
                int v950;
                v950 = v949 + v947;
                float v951;
                v951 = v840[v950];
                bool v952;
                v952 = v883[v950];
                float v953;
                if (v952){
                    v953 = v951;
                } else {
                    v953 = -1.0f / 0.0f;
                }
                float v954;
                v954 = v953 - v943;
                float v955;
                v955 = exp(v954);
                assert("Tensor range check" && 0 <= v945 && v945 < 1);
                assert("Tensor range check" && 0 <= v947 && v947 < 4);
                v944[v950] = v955;
                v947 += 1 ;
            }
            v945 += 1 ;
        }
        float v956;
        v956 = 0.0f;
        int v957;
        v957 = 0;
        while (while_method_3(v957)){
            int v959;
            v959 = 0;
            while (while_method_1(v959)){
                assert("Tensor range check" && 0 <= v957 && v957 < 1);
                assert("Tensor range check" && 0 <= v959 && v959 < 4);
                int v961;
                v961 = 4 * v957;
                int v962;
                v962 = v961 + v959;
                float v963;
                v963 = v944[v962];
                float v964;
                v964 = v956 + v963;
                v956 = v964;
                v959 += 1 ;
            }
            v957 += 1 ;
        }
        auto v965 = cooperative_groups::coalesced_threads();
        int v966;
        v966 = threadIdx.x;
        int v967;
        v967 = v966 / 32;
        auto v968 = cooperative_groups::labeled_partition(v965,v967);
        float v969;
        v969 = cooperative_groups::reduce(v968, v956, v940);
        float v970[4];
        int v971;
        v971 = 0;
        while (while_method_3(v971)){
            int v973;
            v973 = 0;
            while (while_method_1(v973)){
                assert("Tensor range check" && 0 <= v971 && v971 < 1);
                assert("Tensor range check" && 0 <= v973 && v973 < 4);
                int v975;
                v975 = 4 * v971;
                int v976;
                v976 = v975 + v973;
                float v977;
                v977 = v944[v976];
                float v978;
                v978 = v977 / v969;
                assert("Tensor range check" && 0 <= v971 && v971 < 1);
                assert("Tensor range check" && 0 <= v973 && v973 < 4);
                v970[v976] = v978;
                v973 += 1 ;
            }
            v971 += 1 ;
        }
        assert("Tensor range check" && 0 <= v830 && v830 < 8);
        int v979;
        v979 = 0;
        while (while_method_3(v979)){
            assert("Tensor range check" && 0 <= v979 && v979 < 1);
            int v981;
            v981 = 128 * v979;
            int v982;
            v982 = v981 + v839;
            assert("Tensor range check" && 0 <= v979 && v979 < 1);
            int v983;
            v983 = 4 * v979;
            int4* v984;
            v984 = reinterpret_cast<int4*>(v970 + v983);
            int4* v985;
            v985 = reinterpret_cast<int4*>(v4 + v982);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v984) % 16 == 0 && reinterpret_cast<unsigned long long>(v985) % 16 == 0);
            *v985 = *v984;
            v979 += 1 ;
        }
        v830 += 24 ;
    }
    v17.sync() ;
    int v986;
    v986 = threadIdx.x;
    int v987;
    v987 = blockIdx.x;
    int v988;
    v988 = v987 * 256;
    int v989;
    v989 = v986 + v988;
    unsigned long long v990;
    v990 = (unsigned long long)v989;
    curandStatePhilox4_32_10_t v991;
    curand_init(12344321ull,v990,0ull,&v991);
    int v992;
    v992 = threadIdx.x;
    bool v993;
    v993 = 0 <= v992;
    bool v994;
    v994 = v993 == false;
    if (v994){
        assert("The index needs to be zero or positive." && v993);
    } else {
    }
    int v996;
    v996 = v992 % 32;
    int v997;
    v997 = v992 / 32;
    bool v998;
    v998 = v997 < 8;
    bool v999;
    v999 = v998 == false;
    if (v999){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v998);
    } else {
    }
    assert("Tensor range check" && 0 <= v997 && v997 < 8);
    assert("Tensor range check" && 0 <= v996 && v996 < 32);
    int v1001;
    v1001 = 4 * v996;
    int v1002;
    v1002 = 128 * v997;
    int v1003;
    v1003 = v1002 + v1001;
    assert("Tensor range check" && 0 <= v997 && v997 < 8);
    assert("Tensor range check" && 0 <= v996 && v996 < 32);
    assert("Tensor range check" && 0 <= v997 && v997 < 8);
    int v1004;
    v1004 = blockIdx.x;
    int v1005;
    v1005 = v1004;
    while (while_method_2(v1005)){
        bool v1007;
        v1007 = 0 <= v1005;
        bool v1008;
        v1008 = v1007 == false;
        if (v1008){
            assert("The index needs to be zero or positive." && v1007);
        } else {
        }
        bool v1010;
        v1010 = v1005 < 8;
        bool v1011;
        v1011 = v1010 == false;
        if (v1011){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1010);
        } else {
        }
        assert("Tensor range check" && 0 <= v1005 && v1005 < 8);
        int v1013;
        v1013 = 1024 * v1005;
        int v1014;
        v1014 = v1013 + v1003;
        float v1015[4];
        int v1016[4];
        int v1017;
        v1017 = 0;
        while (while_method_3(v1017)){
            assert("Tensor range check" && 0 <= v1017 && v1017 < 1);
            int v1019;
            v1019 = 4 * v1017;
            assert("Tensor range check" && 0 <= v1017 && v1017 < 1);
            int v1020;
            v1020 = 128 * v1017;
            int v1021;
            v1021 = v1020 + v1014;
            int4* v1022;
            v1022 = reinterpret_cast<int4*>(v1 + v1021);
            int4* v1023;
            v1023 = reinterpret_cast<int4*>(v1015 + v1019);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1022) % 16 == 0 && reinterpret_cast<unsigned long long>(v1023) % 16 == 0);
            *v1023 = *v1022;
            v1017 += 1 ;
        }
        int v1024;
        v1024 = 0;
        while (while_method_3(v1024)){
            int v1026;
            v1026 = 0;
            while (while_method_1(v1026)){
                bool v1028;
                v1028 = 0 <= v1026;
                bool v1030;
                if (v1028){
                    bool v1029;
                    v1029 = v1026 < 4;
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
                v1033 = 0 <= v996;
                bool v1035;
                if (v1033){
                    bool v1034;
                    v1034 = v996 < 32;
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
                v1038 = v996 * 4;
                int v1039;
                v1039 = v1026 + v1038;
                bool v1040;
                v1040 = 0 <= v1024;
                bool v1042;
                if (v1040){
                    bool v1041;
                    v1041 = v1024 < 1;
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
                v1045 = v1024 * 128;
                int v1046;
                v1046 = v1039 + v1045;
                assert("Tensor range check" && 0 <= v1024 && v1024 < 1);
                assert("Tensor range check" && 0 <= v1026 && v1026 < 4);
                int v1047;
                v1047 = 4 * v1024;
                int v1048;
                v1048 = v1047 + v1026;
                v1016[v1048] = v1046;
                v1026 += 1 ;
            }
            v1024 += 1 ;
        }
        bool v1049;
        v1049 = 0 <= v997;
        bool v1050;
        v1050 = v1049 && v998;
        bool v1051;
        v1051 = v1050 == false;
        if (v1051){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1050);
        } else {
        }
        bool v1053;
        v1053 = v1007 && v1010;
        bool v1054;
        v1054 = v1053 == false;
        if (v1054){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1053);
        } else {
        }
        int v1056;
        v1056 = v1005 * 8;
        int v1057;
        v1057 = v1056 + v997;
        float v1058;
        v1058 = 0.0f;
        int v1059;
        v1059 = 0;
        while (while_method_3(v1059)){
            int v1061;
            v1061 = 0;
            while (while_method_1(v1061)){
                assert("Tensor range check" && 0 <= v1059 && v1059 < 1);
                assert("Tensor range check" && 0 <= v1061 && v1061 < 4);
                int v1063;
                v1063 = 4 * v1059;
                int v1064;
                v1064 = v1063 + v1061;
                float v1065;
                v1065 = v1015[v1064];
                float v1066;
                v1066 = v1058 + v1065;
                v1058 = v1066;
                v1061 += 1 ;
            }
            v1059 += 1 ;
        }
        auto v1067 = cooperative_groups::coalesced_threads();
        int v1068;
        v1068 = threadIdx.x;
        int v1069;
        v1069 = v1068 / 32;
        auto v1070 = cooperative_groups::labeled_partition(v1067,v1069);
        Closure0 v1071{};
        float v1072;
        v1072 = cooperative_groups::reduce(v1070, v1058, v1071);
        float v1073;
        v1073 = v1072 / 128.0f;
        float v1074[4];
        int v1075;
        v1075 = 0;
        while (while_method_3(v1075)){
            int v1077;
            v1077 = 0;
            while (while_method_1(v1077)){
                assert("Tensor range check" && 0 <= v1075 && v1075 < 1);
                assert("Tensor range check" && 0 <= v1077 && v1077 < 4);
                int v1079;
                v1079 = 4 * v1075;
                int v1080;
                v1080 = v1079 + v1077;
                float v1081;
                v1081 = v1015[v1080];
                float v1082;
                v1082 = v1081 - v1073;
                float v1083;
                v1083 = exp(v1082);
                assert("Tensor range check" && 0 <= v1075 && v1075 < 1);
                assert("Tensor range check" && 0 <= v1077 && v1077 < 4);
                v1074[v1080] = v1083;
                v1077 += 1 ;
            }
            v1075 += 1 ;
        }
        float v1084;
        v1084 = 0.0f;
        int v1085;
        v1085 = 0;
        while (while_method_3(v1085)){
            int v1087;
            v1087 = 0;
            while (while_method_1(v1087)){
                assert("Tensor range check" && 0 <= v1085 && v1085 < 1);
                assert("Tensor range check" && 0 <= v1087 && v1087 < 4);
                int v1089;
                v1089 = 4 * v1085;
                int v1090;
                v1090 = v1089 + v1087;
                float v1091;
                v1091 = v1074[v1090];
                float v1092;
                v1092 = v1084 + v1091;
                v1084 = v1092;
                v1087 += 1 ;
            }
            v1085 += 1 ;
        }
        auto v1093 = cooperative_groups::coalesced_threads();
        int v1094;
        v1094 = threadIdx.x;
        int v1095;
        v1095 = v1094 / 32;
        auto v1096 = cooperative_groups::labeled_partition(v1093,v1095);
        float v1097;
        v1097 = cooperative_groups::reduce(v1096, v1084, v1071);
        float v1098[4];
        int v1099;
        v1099 = 0;
        while (while_method_3(v1099)){
            int v1101;
            v1101 = 0;
            while (while_method_1(v1101)){
                assert("Tensor range check" && 0 <= v1099 && v1099 < 1);
                assert("Tensor range check" && 0 <= v1101 && v1101 < 4);
                int v1103;
                v1103 = 4 * v1099;
                int v1104;
                v1104 = v1103 + v1101;
                float v1105;
                v1105 = v1074[v1104];
                float v1106;
                v1106 = v1105 / v1097;
                assert("Tensor range check" && 0 <= v1099 && v1099 < 1);
                assert("Tensor range check" && 0 <= v1101 && v1101 < 4);
                v1098[v1104] = v1106;
                v1101 += 1 ;
            }
            v1099 += 1 ;
        }
        float v1107[4];
        float v1108;
        v1108 = 0.0f;
        int v1109;
        v1109 = 0;
        while (while_method_3(v1109)){
            assert("Tensor range check" && 0 <= v1109 && v1109 < 1);
            int v1111;
            v1111 = 4 * v1109;
            assert("Tensor range check" && 0 <= v1109 && v1109 < 1);
            int v1112; float v1113;
            Tuple0 tmp93 = Tuple0{0, 0.0f};
            v1112 = tmp93.v0; v1113 = tmp93.v1;
            while (while_method_1(v1112)){
                assert("Tensor range check" && 0 <= v1112 && v1112 < 4);
                int v1115;
                v1115 = v1112 + v1111;
                float v1116;
                v1116 = v1098[v1115];
                float v1117;
                v1117 = v1113 + v1116;
                v1113 = v1117;
                v1112 += 1 ;
            }
            auto v1118 = cooperative_groups::coalesced_threads();
            int v1119;
            v1119 = threadIdx.x;
            int v1120;
            v1120 = v1119 / 32;
            auto v1121 = cooperative_groups::labeled_partition(v1118,v1120);
            Closure2 v1122{};
            float v1123;
            v1123 = cooperative_groups::inclusive_scan(v1121, v1113, v1122);
            float v1124;
            v1124 = v1121.shfl_up(v1123,1);
            bool v1125;
            v1125 = v1121.thread_rank() == 0;
            float v1126;
            if (v1125){
                v1126 = 0.0f;
            } else {
                v1126 = v1124;
            }
            float v1127;
            v1127 = v1121.shfl(v1123,v1121.num_threads()-1);
            float v1128;
            v1128 = v1108 + v1126;
            int v1129; float v1130;
            Tuple0 tmp94 = Tuple0{0, v1128};
            v1129 = tmp94.v0; v1130 = tmp94.v1;
            while (while_method_1(v1129)){
                assert("Tensor range check" && 0 <= v1129 && v1129 < 4);
                int v1132;
                v1132 = v1129 + v1111;
                float v1133;
                v1133 = v1098[v1132];
                float v1134;
                v1134 = v1130 + v1133;
                assert("Tensor range check" && 0 <= v1129 && v1129 < 4);
                v1107[v1132] = v1134;
                v1130 = v1134;
                v1129 += 1 ;
            }
            float v1135;
            v1135 = v1108 + v1127;
            v1108 = v1135;
            v1109 += 1 ;
        }
        float v1136[4];
        bool v1137[4];
        int v1138;
        v1138 = 0;
        while (while_method_3(v1138)){
            int v1140;
            v1140 = 0;
            while (while_method_1(v1140)){
                assert("Tensor range check" && 0 <= v1138 && v1138 < 1);
                assert("Tensor range check" && 0 <= v1140 && v1140 < 4);
                int v1142;
                v1142 = 4 * v1138;
                int v1143;
                v1143 = v1142 + v1140;
                float v1144;
                v1144 = v1107[v1143];
                float v1145;
                v1145 = v1098[v1143];
                bool v1146;
                v1146 = v1145 > 0.0f;
                assert("Tensor range check" && 0 <= v1138 && v1138 < 1);
                assert("Tensor range check" && 0 <= v1140 && v1140 < 4);
                v1136[v1143] = v1144;
                v1137[v1143] = v1146;
                v1140 += 1 ;
            }
            v1138 += 1 ;
        }
        float v1147; bool v1148;
        Tuple3 tmp95 = Tuple3{-1.0f / 0.0f, false};
        v1147 = tmp95.v0; v1148 = tmp95.v1;
        int v1149;
        v1149 = 0;
        while (while_method_3(v1149)){
            int v1151;
            v1151 = 0;
            while (while_method_1(v1151)){
                assert("Tensor range check" && 0 <= v1149 && v1149 < 1);
                assert("Tensor range check" && 0 <= v1151 && v1151 < 4);
                int v1153;
                v1153 = 4 * v1149;
                int v1154;
                v1154 = v1153 + v1151;
                float v1155;
                v1155 = v1136[v1154];
                bool v1156;
                v1156 = v1137[v1154];
                float v1163; bool v1164;
                if (v1148){
                    if (v1156){
                        bool v1157;
                        v1157 = v1147 >= v1155;
                        float v1158;
                        if (v1157){
                            v1158 = v1147;
                        } else {
                            v1158 = v1155;
                        }
                        v1163 = v1158; v1164 = true;
                    } else {
                        v1163 = v1147; v1164 = v1148;
                    }
                } else {
                    if (v1156){
                        v1163 = v1155; v1164 = v1156;
                    } else {
                        v1163 = v1147; v1164 = v1148;
                    }
                }
                v1147 = v1163;
                v1148 = v1164;
                v1151 += 1 ;
            }
            v1149 += 1 ;
        }
        auto v1165 = cooperative_groups::coalesced_threads();
        int v1166;
        v1166 = threadIdx.x;
        int v1167;
        v1167 = v1166 / 32;
        auto v1168 = cooperative_groups::labeled_partition(v1165,v1167);
        Closure5 v1169{};
        float v1170; bool v1171;
        Tuple3 tmp96 = cooperative_groups::reduce(v1168, Tuple3{v1147, v1148}, v1169);
        v1170 = tmp96.v0; v1171 = tmp96.v1;
        bool v1172;
        v1172 = v1171 == false;
        if (v1172){
            assert("The local reduce must be true." && v1171);
        } else {
        }
        float v1174[4];
        int v1175[4];
        int v1176;
        v1176 = 0;
        while (while_method_3(v1176)){
            int v1178;
            v1178 = 0;
            while (while_method_1(v1178)){
                assert("Tensor range check" && 0 <= v1176 && v1176 < 1);
                assert("Tensor range check" && 0 <= v1178 && v1178 < 4);
                int v1180;
                v1180 = 4 * v1176;
                int v1181;
                v1181 = v1180 + v1178;
                int v1182;
                v1182 = v1016[v1181];
                float v1183;
                v1183 = curand_uniform(&v991);
                assert("Tensor range check" && 0 <= v1176 && v1176 < 1);
                assert("Tensor range check" && 0 <= v1178 && v1178 < 4);
                v1174[v1181] = v1183;
                v1175[v1181] = v1182;
                v1178 += 1 ;
            }
            v1176 += 1 ;
        }
        float v1184; int v1185;
        Tuple1 tmp97 = Tuple1{0.0f, 2147483647};
        v1184 = tmp97.v0; v1185 = tmp97.v1;
        int v1186;
        v1186 = 0;
        while (while_method_3(v1186)){
            int v1188;
            v1188 = 0;
            while (while_method_1(v1188)){
                assert("Tensor range check" && 0 <= v1186 && v1186 < 1);
                assert("Tensor range check" && 0 <= v1188 && v1188 < 4);
                int v1190;
                v1190 = 4 * v1186;
                int v1191;
                v1191 = v1190 + v1188;
                float v1192;
                v1192 = v1174[v1191];
                int v1193;
                v1193 = v1175[v1191];
                bool v1194;
                v1194 = v1185 < v1193;
                float v1195; int v1196;
                if (v1194){
                    v1195 = v1184; v1196 = v1185;
                } else {
                    v1195 = v1192; v1196 = v1193;
                }
                v1184 = v1195;
                v1185 = v1196;
                v1188 += 1 ;
            }
            v1186 += 1 ;
        }
        auto v1197 = cooperative_groups::coalesced_threads();
        int v1198;
        v1198 = threadIdx.x;
        int v1199;
        v1199 = v1198 / 32;
        auto v1200 = cooperative_groups::labeled_partition(v1197,v1199);
        Closure6 v1201{};
        float v1202; int v1203;
        Tuple1 tmp98 = cooperative_groups::reduce(v1200, Tuple1{v1184, v1185}, v1201);
        v1202 = tmp98.v0; v1203 = tmp98.v1;
        float v1204;
        v1204 = v1170 * v1202;
        int v1205[4];
        bool v1206[4];
        int v1207;
        v1207 = 0;
        while (while_method_3(v1207)){
            int v1209;
            v1209 = 0;
            while (while_method_1(v1209)){
                assert("Tensor range check" && 0 <= v1207 && v1207 < 1);
                assert("Tensor range check" && 0 <= v1209 && v1209 < 4);
                int v1211;
                v1211 = 4 * v1207;
                int v1212;
                v1212 = v1211 + v1209;
                float v1213;
                v1213 = v1136[v1212];
                bool v1214;
                v1214 = v1137[v1212];
                int v1215;
                v1215 = v1016[v1212];
                int v1218; bool v1219;
                if (v1214){
                    float v1216;
                    v1216 = v1213 - v1204;
                    bool v1217;
                    v1217 = v1216 >= 0.0f;
                    v1218 = v1215; v1219 = v1217;
                } else {
                    v1218 = 2147483647; v1219 = false;
                }
                assert("Tensor range check" && 0 <= v1207 && v1207 < 1);
                assert("Tensor range check" && 0 <= v1209 && v1209 < 4);
                v1205[v1212] = v1218;
                v1206[v1212] = v1219;
                v1209 += 1 ;
            }
            v1207 += 1 ;
        }
        int v1220; bool v1221;
        Tuple4 tmp99 = Tuple4{2147483647, false};
        v1220 = tmp99.v0; v1221 = tmp99.v1;
        int v1222;
        v1222 = 0;
        while (while_method_3(v1222)){
            int v1224;
            v1224 = 0;
            while (while_method_1(v1224)){
                assert("Tensor range check" && 0 <= v1222 && v1222 < 1);
                assert("Tensor range check" && 0 <= v1224 && v1224 < 4);
                int v1226;
                v1226 = 4 * v1222;
                int v1227;
                v1227 = v1226 + v1224;
                int v1228;
                v1228 = v1205[v1227];
                bool v1229;
                v1229 = v1206[v1227];
                int v1236; bool v1237;
                if (v1221){
                    if (v1229){
                        bool v1230;
                        v1230 = v1220 < v1228;
                        int v1231;
                        if (v1230){
                            v1231 = v1220;
                        } else {
                            v1231 = v1228;
                        }
                        v1236 = v1231; v1237 = true;
                    } else {
                        v1236 = v1220; v1237 = v1221;
                    }
                } else {
                    if (v1229){
                        v1236 = v1228; v1237 = v1229;
                    } else {
                        v1236 = v1220; v1237 = v1221;
                    }
                }
                v1220 = v1236;
                v1221 = v1237;
                v1224 += 1 ;
            }
            v1222 += 1 ;
        }
        auto v1238 = cooperative_groups::coalesced_threads();
        int v1239;
        v1239 = threadIdx.x;
        int v1240;
        v1240 = v1239 / 32;
        auto v1241 = cooperative_groups::labeled_partition(v1238,v1240);
        Closure7 v1242{};
        int v1243; bool v1244;
        Tuple4 tmp100 = cooperative_groups::reduce(v1241, Tuple4{v1220, v1221}, v1242);
        v1243 = tmp100.v0; v1244 = tmp100.v1;
        bool v1245;
        v1245 = v1244 == false;
        if (v1245){
            assert("The local reduce must be true." && v1244);
        } else {
        }
        assert("Tensor range check" && 0 <= v1005 && v1005 < 8);
        int v1247;
        v1247 = 0;
        while (while_method_3(v1247)){
            assert("Tensor range check" && 0 <= v1247 && v1247 < 1);
            int v1249;
            v1249 = 128 * v1247;
            int v1250;
            v1250 = v1249 + v1014;
            assert("Tensor range check" && 0 <= v1247 && v1247 < 1);
            int v1251;
            v1251 = 4 * v1247;
            int4* v1252;
            v1252 = reinterpret_cast<int4*>(v1098 + v1251);
            int4* v1253;
            v1253 = reinterpret_cast<int4*>(v13 + v1250);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1252) % 16 == 0 && reinterpret_cast<unsigned long long>(v1253) % 16 == 0);
            *v1253 = *v1252;
            v1247 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1005 && v1005 < 8);
        int v1254;
        v1254 = 8 * v1005;
        int v1255;
        v1255 = v1254 + v997;
        v14[v1255] = v1243;
        v1005 += 24 ;
    }
    v17.sync() ;
    int v1256;
    v1256 = threadIdx.x;
    int v1257;
    v1257 = blockIdx.x;
    int v1258;
    v1258 = v1257 * 256;
    int v1259;
    v1259 = v1256 + v1258;
    unsigned long long v1260;
    v1260 = (unsigned long long)v1259;
    curandStatePhilox4_32_10_t v1261;
    curand_init(12344321ull,v1260,0ull,&v1261);
    int v1262;
    v1262 = threadIdx.x;
    bool v1263;
    v1263 = 0 <= v1262;
    bool v1264;
    v1264 = v1263 == false;
    if (v1264){
        assert("The index needs to be zero or positive." && v1263);
    } else {
    }
    int v1266;
    v1266 = v1262 % 32;
    int v1267;
    v1267 = v1262 / 32;
    bool v1268;
    v1268 = v1267 < 8;
    bool v1269;
    v1269 = v1268 == false;
    if (v1269){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1268);
    } else {
    }
    assert("Tensor range check" && 0 <= v1267 && v1267 < 8);
    assert("Tensor range check" && 0 <= v1266 && v1266 < 32);
    int v1271;
    v1271 = 4 * v1266;
    int v1272;
    v1272 = 128 * v1267;
    int v1273;
    v1273 = v1272 + v1271;
    assert("Tensor range check" && 0 <= v1267 && v1267 < 8);
    assert("Tensor range check" && 0 <= v1266 && v1266 < 32);
    assert("Tensor range check" && 0 <= v1267 && v1267 < 8);
    int v1274;
    v1274 = blockIdx.x;
    int v1275;
    v1275 = v1274;
    while (while_method_2(v1275)){
        bool v1277;
        v1277 = 0 <= v1275;
        bool v1278;
        v1278 = v1277 == false;
        if (v1278){
            assert("The index needs to be zero or positive." && v1277);
        } else {
        }
        bool v1280;
        v1280 = v1275 < 8;
        bool v1281;
        v1281 = v1280 == false;
        if (v1281){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1280);
        } else {
        }
        assert("Tensor range check" && 0 <= v1275 && v1275 < 8);
        int v1283;
        v1283 = 1024 * v1275;
        int v1284;
        v1284 = v1283 + v1273;
        float v1285[4];
        int v1286[4];
        int v1287;
        v1287 = 0;
        while (while_method_3(v1287)){
            assert("Tensor range check" && 0 <= v1287 && v1287 < 1);
            int v1289;
            v1289 = 4 * v1287;
            assert("Tensor range check" && 0 <= v1287 && v1287 < 1);
            int v1290;
            v1290 = 128 * v1287;
            int v1291;
            v1291 = v1290 + v1284;
            int4* v1292;
            v1292 = reinterpret_cast<int4*>(v1 + v1291);
            int4* v1293;
            v1293 = reinterpret_cast<int4*>(v1285 + v1289);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1292) % 16 == 0 && reinterpret_cast<unsigned long long>(v1293) % 16 == 0);
            *v1293 = *v1292;
            v1287 += 1 ;
        }
        int v1294;
        v1294 = 0;
        while (while_method_3(v1294)){
            int v1296;
            v1296 = 0;
            while (while_method_1(v1296)){
                bool v1298;
                v1298 = 0 <= v1296;
                bool v1300;
                if (v1298){
                    bool v1299;
                    v1299 = v1296 < 4;
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
                v1303 = 0 <= v1266;
                bool v1305;
                if (v1303){
                    bool v1304;
                    v1304 = v1266 < 32;
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
                v1308 = v1266 * 4;
                int v1309;
                v1309 = v1296 + v1308;
                bool v1310;
                v1310 = 0 <= v1294;
                bool v1312;
                if (v1310){
                    bool v1311;
                    v1311 = v1294 < 1;
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
                v1315 = v1294 * 128;
                int v1316;
                v1316 = v1309 + v1315;
                assert("Tensor range check" && 0 <= v1294 && v1294 < 1);
                assert("Tensor range check" && 0 <= v1296 && v1296 < 4);
                int v1317;
                v1317 = 4 * v1294;
                int v1318;
                v1318 = v1317 + v1296;
                v1286[v1318] = v1316;
                v1296 += 1 ;
            }
            v1294 += 1 ;
        }
        bool v1319;
        v1319 = 0 <= v1267;
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
        v1326 = v1275 * 8;
        int v1327;
        v1327 = v1326 + v1267;
        bool v1328[4];
        int v1329;
        v1329 = 0;
        while (while_method_3(v1329)){
            int v1331;
            v1331 = 0;
            while (while_method_1(v1331)){
                assert("Tensor range check" && 0 <= v1329 && v1329 < 1);
                assert("Tensor range check" && 0 <= v1331 && v1331 < 4);
                int v1333;
                v1333 = 4 * v1329;
                int v1334;
                v1334 = v1333 + v1331;
                float v1335;
                v1335 = v1285[v1334];
                int v1336;
                v1336 = v1286[v1334];
                bool v1337;
                v1337 = v1336 < 11;
                assert("Tensor range check" && 0 <= v1329 && v1329 < 1);
                assert("Tensor range check" && 0 <= v1331 && v1331 < 4);
                v1328[v1334] = v1337;
                v1331 += 1 ;
            }
            v1329 += 1 ;
        }
        int v1338[4];
        int v1339;
        v1339 = 0;
        while (while_method_3(v1339)){
            int v1341;
            v1341 = 0;
            while (while_method_1(v1341)){
                assert("Tensor range check" && 0 <= v1339 && v1339 < 1);
                assert("Tensor range check" && 0 <= v1341 && v1341 < 4);
                int v1343;
                v1343 = 4 * v1339;
                int v1344;
                v1344 = v1343 + v1341;
                bool v1345;
                v1345 = v1328[v1344];
                int v1346;
                if (v1345){
                    v1346 = 1;
                } else {
                    v1346 = 0;
                }
                assert("Tensor range check" && 0 <= v1339 && v1339 < 1);
                assert("Tensor range check" && 0 <= v1341 && v1341 < 4);
                v1338[v1344] = v1346;
                v1341 += 1 ;
            }
            v1339 += 1 ;
        }
        int v1347;
        v1347 = 0;
        int v1348;
        v1348 = 0;
        while (while_method_3(v1348)){
            int v1350;
            v1350 = 0;
            while (while_method_1(v1350)){
                assert("Tensor range check" && 0 <= v1348 && v1348 < 1);
                assert("Tensor range check" && 0 <= v1350 && v1350 < 4);
                int v1352;
                v1352 = 4 * v1348;
                int v1353;
                v1353 = v1352 + v1350;
                int v1354;
                v1354 = v1338[v1353];
                int v1355;
                v1355 = v1347 + v1354;
                v1347 = v1355;
                v1350 += 1 ;
            }
            v1348 += 1 ;
        }
        auto v1356 = cooperative_groups::coalesced_threads();
        int v1357;
        v1357 = threadIdx.x;
        int v1358;
        v1358 = v1357 / 32;
        auto v1359 = cooperative_groups::labeled_partition(v1356,v1358);
        Closure4 v1360{};
        int v1361;
        v1361 = cooperative_groups::reduce(v1359, v1347, v1360);
        float v1362[4];
        int v1363;
        v1363 = 0;
        while (while_method_3(v1363)){
            int v1365;
            v1365 = 0;
            while (while_method_1(v1365)){
                assert("Tensor range check" && 0 <= v1363 && v1363 < 1);
                assert("Tensor range check" && 0 <= v1365 && v1365 < 4);
                int v1367;
                v1367 = 4 * v1363;
                int v1368;
                v1368 = v1367 + v1365;
                float v1369;
                v1369 = v1285[v1368];
                bool v1370;
                v1370 = v1328[v1368];
                float v1371;
                if (v1370){
                    v1371 = v1369;
                } else {
                    v1371 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1363 && v1363 < 1);
                assert("Tensor range check" && 0 <= v1365 && v1365 < 4);
                v1362[v1368] = v1371;
                v1365 += 1 ;
            }
            v1363 += 1 ;
        }
        float v1372;
        v1372 = 0.0f;
        int v1373;
        v1373 = 0;
        while (while_method_3(v1373)){
            int v1375;
            v1375 = 0;
            while (while_method_1(v1375)){
                assert("Tensor range check" && 0 <= v1373 && v1373 < 1);
                assert("Tensor range check" && 0 <= v1375 && v1375 < 4);
                int v1377;
                v1377 = 4 * v1373;
                int v1378;
                v1378 = v1377 + v1375;
                float v1379;
                v1379 = v1362[v1378];
                float v1380;
                v1380 = v1372 + v1379;
                v1372 = v1380;
                v1375 += 1 ;
            }
            v1373 += 1 ;
        }
        auto v1381 = cooperative_groups::coalesced_threads();
        int v1382;
        v1382 = threadIdx.x;
        int v1383;
        v1383 = v1382 / 32;
        auto v1384 = cooperative_groups::labeled_partition(v1381,v1383);
        Closure0 v1385{};
        float v1386;
        v1386 = cooperative_groups::reduce(v1384, v1372, v1385);
        float v1387;
        v1387 = (float)v1361;
        float v1388;
        v1388 = v1386 / v1387;
        float v1389[4];
        int v1390;
        v1390 = 0;
        while (while_method_3(v1390)){
            int v1392;
            v1392 = 0;
            while (while_method_1(v1392)){
                assert("Tensor range check" && 0 <= v1390 && v1390 < 1);
                assert("Tensor range check" && 0 <= v1392 && v1392 < 4);
                int v1394;
                v1394 = 4 * v1390;
                int v1395;
                v1395 = v1394 + v1392;
                float v1396;
                v1396 = v1285[v1395];
                bool v1397;
                v1397 = v1328[v1395];
                float v1398;
                if (v1397){
                    v1398 = v1396;
                } else {
                    v1398 = -1.0f / 0.0f;
                }
                float v1399;
                v1399 = v1398 - v1388;
                float v1400;
                v1400 = exp(v1399);
                assert("Tensor range check" && 0 <= v1390 && v1390 < 1);
                assert("Tensor range check" && 0 <= v1392 && v1392 < 4);
                v1389[v1395] = v1400;
                v1392 += 1 ;
            }
            v1390 += 1 ;
        }
        float v1401;
        v1401 = 0.0f;
        int v1402;
        v1402 = 0;
        while (while_method_3(v1402)){
            int v1404;
            v1404 = 0;
            while (while_method_1(v1404)){
                assert("Tensor range check" && 0 <= v1402 && v1402 < 1);
                assert("Tensor range check" && 0 <= v1404 && v1404 < 4);
                int v1406;
                v1406 = 4 * v1402;
                int v1407;
                v1407 = v1406 + v1404;
                float v1408;
                v1408 = v1389[v1407];
                float v1409;
                v1409 = v1401 + v1408;
                v1401 = v1409;
                v1404 += 1 ;
            }
            v1402 += 1 ;
        }
        auto v1410 = cooperative_groups::coalesced_threads();
        int v1411;
        v1411 = threadIdx.x;
        int v1412;
        v1412 = v1411 / 32;
        auto v1413 = cooperative_groups::labeled_partition(v1410,v1412);
        float v1414;
        v1414 = cooperative_groups::reduce(v1413, v1401, v1385);
        float v1415[4];
        int v1416;
        v1416 = 0;
        while (while_method_3(v1416)){
            int v1418;
            v1418 = 0;
            while (while_method_1(v1418)){
                assert("Tensor range check" && 0 <= v1416 && v1416 < 1);
                assert("Tensor range check" && 0 <= v1418 && v1418 < 4);
                int v1420;
                v1420 = 4 * v1416;
                int v1421;
                v1421 = v1420 + v1418;
                float v1422;
                v1422 = v1389[v1421];
                float v1423;
                v1423 = v1422 / v1414;
                assert("Tensor range check" && 0 <= v1416 && v1416 < 1);
                assert("Tensor range check" && 0 <= v1418 && v1418 < 4);
                v1415[v1421] = v1423;
                v1418 += 1 ;
            }
            v1416 += 1 ;
        }
        float v1424[4];
        float v1425;
        v1425 = 0.0f;
        int v1426;
        v1426 = 0;
        while (while_method_3(v1426)){
            assert("Tensor range check" && 0 <= v1426 && v1426 < 1);
            int v1428;
            v1428 = 4 * v1426;
            assert("Tensor range check" && 0 <= v1426 && v1426 < 1);
            int v1429; float v1430;
            Tuple0 tmp101 = Tuple0{0, 0.0f};
            v1429 = tmp101.v0; v1430 = tmp101.v1;
            while (while_method_1(v1429)){
                assert("Tensor range check" && 0 <= v1429 && v1429 < 4);
                int v1432;
                v1432 = v1429 + v1428;
                float v1433;
                v1433 = v1415[v1432];
                float v1434;
                v1434 = v1430 + v1433;
                v1430 = v1434;
                v1429 += 1 ;
            }
            auto v1435 = cooperative_groups::coalesced_threads();
            int v1436;
            v1436 = threadIdx.x;
            int v1437;
            v1437 = v1436 / 32;
            auto v1438 = cooperative_groups::labeled_partition(v1435,v1437);
            Closure2 v1439{};
            float v1440;
            v1440 = cooperative_groups::inclusive_scan(v1438, v1430, v1439);
            float v1441;
            v1441 = v1438.shfl_up(v1440,1);
            bool v1442;
            v1442 = v1438.thread_rank() == 0;
            float v1443;
            if (v1442){
                v1443 = 0.0f;
            } else {
                v1443 = v1441;
            }
            float v1444;
            v1444 = v1438.shfl(v1440,v1438.num_threads()-1);
            float v1445;
            v1445 = v1425 + v1443;
            int v1446; float v1447;
            Tuple0 tmp102 = Tuple0{0, v1445};
            v1446 = tmp102.v0; v1447 = tmp102.v1;
            while (while_method_1(v1446)){
                assert("Tensor range check" && 0 <= v1446 && v1446 < 4);
                int v1449;
                v1449 = v1446 + v1428;
                float v1450;
                v1450 = v1415[v1449];
                float v1451;
                v1451 = v1447 + v1450;
                assert("Tensor range check" && 0 <= v1446 && v1446 < 4);
                v1424[v1449] = v1451;
                v1447 = v1451;
                v1446 += 1 ;
            }
            float v1452;
            v1452 = v1425 + v1444;
            v1425 = v1452;
            v1426 += 1 ;
        }
        float v1453[4];
        bool v1454[4];
        int v1455;
        v1455 = 0;
        while (while_method_3(v1455)){
            int v1457;
            v1457 = 0;
            while (while_method_1(v1457)){
                assert("Tensor range check" && 0 <= v1455 && v1455 < 1);
                assert("Tensor range check" && 0 <= v1457 && v1457 < 4);
                int v1459;
                v1459 = 4 * v1455;
                int v1460;
                v1460 = v1459 + v1457;
                float v1461;
                v1461 = v1424[v1460];
                float v1462;
                v1462 = v1415[v1460];
                bool v1463;
                v1463 = v1462 > 0.0f;
                assert("Tensor range check" && 0 <= v1455 && v1455 < 1);
                assert("Tensor range check" && 0 <= v1457 && v1457 < 4);
                v1453[v1460] = v1461;
                v1454[v1460] = v1463;
                v1457 += 1 ;
            }
            v1455 += 1 ;
        }
        float v1464; bool v1465;
        Tuple3 tmp103 = Tuple3{-1.0f / 0.0f, false};
        v1464 = tmp103.v0; v1465 = tmp103.v1;
        int v1466;
        v1466 = 0;
        while (while_method_3(v1466)){
            int v1468;
            v1468 = 0;
            while (while_method_1(v1468)){
                assert("Tensor range check" && 0 <= v1466 && v1466 < 1);
                assert("Tensor range check" && 0 <= v1468 && v1468 < 4);
                int v1470;
                v1470 = 4 * v1466;
                int v1471;
                v1471 = v1470 + v1468;
                float v1472;
                v1472 = v1453[v1471];
                bool v1473;
                v1473 = v1454[v1471];
                float v1480; bool v1481;
                if (v1465){
                    if (v1473){
                        bool v1474;
                        v1474 = v1464 >= v1472;
                        float v1475;
                        if (v1474){
                            v1475 = v1464;
                        } else {
                            v1475 = v1472;
                        }
                        v1480 = v1475; v1481 = true;
                    } else {
                        v1480 = v1464; v1481 = v1465;
                    }
                } else {
                    if (v1473){
                        v1480 = v1472; v1481 = v1473;
                    } else {
                        v1480 = v1464; v1481 = v1465;
                    }
                }
                v1464 = v1480;
                v1465 = v1481;
                v1468 += 1 ;
            }
            v1466 += 1 ;
        }
        auto v1482 = cooperative_groups::coalesced_threads();
        int v1483;
        v1483 = threadIdx.x;
        int v1484;
        v1484 = v1483 / 32;
        auto v1485 = cooperative_groups::labeled_partition(v1482,v1484);
        Closure5 v1486{};
        float v1487; bool v1488;
        Tuple3 tmp104 = cooperative_groups::reduce(v1485, Tuple3{v1464, v1465}, v1486);
        v1487 = tmp104.v0; v1488 = tmp104.v1;
        bool v1489;
        v1489 = v1488 == false;
        if (v1489){
            assert("The local reduce must be true." && v1488);
        } else {
        }
        float v1491[4];
        int v1492[4];
        int v1493;
        v1493 = 0;
        while (while_method_3(v1493)){
            int v1495;
            v1495 = 0;
            while (while_method_1(v1495)){
                assert("Tensor range check" && 0 <= v1493 && v1493 < 1);
                assert("Tensor range check" && 0 <= v1495 && v1495 < 4);
                int v1497;
                v1497 = 4 * v1493;
                int v1498;
                v1498 = v1497 + v1495;
                int v1499;
                v1499 = v1286[v1498];
                float v1500;
                v1500 = curand_uniform(&v1261);
                assert("Tensor range check" && 0 <= v1493 && v1493 < 1);
                assert("Tensor range check" && 0 <= v1495 && v1495 < 4);
                v1491[v1498] = v1500;
                v1492[v1498] = v1499;
                v1495 += 1 ;
            }
            v1493 += 1 ;
        }
        float v1501; int v1502;
        Tuple1 tmp105 = Tuple1{0.0f, 2147483647};
        v1501 = tmp105.v0; v1502 = tmp105.v1;
        int v1503;
        v1503 = 0;
        while (while_method_3(v1503)){
            int v1505;
            v1505 = 0;
            while (while_method_1(v1505)){
                assert("Tensor range check" && 0 <= v1503 && v1503 < 1);
                assert("Tensor range check" && 0 <= v1505 && v1505 < 4);
                int v1507;
                v1507 = 4 * v1503;
                int v1508;
                v1508 = v1507 + v1505;
                float v1509;
                v1509 = v1491[v1508];
                int v1510;
                v1510 = v1492[v1508];
                bool v1511;
                v1511 = v1502 < v1510;
                float v1512; int v1513;
                if (v1511){
                    v1512 = v1501; v1513 = v1502;
                } else {
                    v1512 = v1509; v1513 = v1510;
                }
                v1501 = v1512;
                v1502 = v1513;
                v1505 += 1 ;
            }
            v1503 += 1 ;
        }
        auto v1514 = cooperative_groups::coalesced_threads();
        int v1515;
        v1515 = threadIdx.x;
        int v1516;
        v1516 = v1515 / 32;
        auto v1517 = cooperative_groups::labeled_partition(v1514,v1516);
        Closure6 v1518{};
        float v1519; int v1520;
        Tuple1 tmp106 = cooperative_groups::reduce(v1517, Tuple1{v1501, v1502}, v1518);
        v1519 = tmp106.v0; v1520 = tmp106.v1;
        float v1521;
        v1521 = v1487 * v1519;
        int v1522[4];
        bool v1523[4];
        int v1524;
        v1524 = 0;
        while (while_method_3(v1524)){
            int v1526;
            v1526 = 0;
            while (while_method_1(v1526)){
                assert("Tensor range check" && 0 <= v1524 && v1524 < 1);
                assert("Tensor range check" && 0 <= v1526 && v1526 < 4);
                int v1528;
                v1528 = 4 * v1524;
                int v1529;
                v1529 = v1528 + v1526;
                float v1530;
                v1530 = v1453[v1529];
                bool v1531;
                v1531 = v1454[v1529];
                int v1532;
                v1532 = v1286[v1529];
                int v1535; bool v1536;
                if (v1531){
                    float v1533;
                    v1533 = v1530 - v1521;
                    bool v1534;
                    v1534 = v1533 >= 0.0f;
                    v1535 = v1532; v1536 = v1534;
                } else {
                    v1535 = 2147483647; v1536 = false;
                }
                assert("Tensor range check" && 0 <= v1524 && v1524 < 1);
                assert("Tensor range check" && 0 <= v1526 && v1526 < 4);
                v1522[v1529] = v1535;
                v1523[v1529] = v1536;
                v1526 += 1 ;
            }
            v1524 += 1 ;
        }
        int v1537; bool v1538;
        Tuple4 tmp107 = Tuple4{2147483647, false};
        v1537 = tmp107.v0; v1538 = tmp107.v1;
        int v1539;
        v1539 = 0;
        while (while_method_3(v1539)){
            int v1541;
            v1541 = 0;
            while (while_method_1(v1541)){
                assert("Tensor range check" && 0 <= v1539 && v1539 < 1);
                assert("Tensor range check" && 0 <= v1541 && v1541 < 4);
                int v1543;
                v1543 = 4 * v1539;
                int v1544;
                v1544 = v1543 + v1541;
                int v1545;
                v1545 = v1522[v1544];
                bool v1546;
                v1546 = v1523[v1544];
                int v1553; bool v1554;
                if (v1538){
                    if (v1546){
                        bool v1547;
                        v1547 = v1537 < v1545;
                        int v1548;
                        if (v1547){
                            v1548 = v1537;
                        } else {
                            v1548 = v1545;
                        }
                        v1553 = v1548; v1554 = true;
                    } else {
                        v1553 = v1537; v1554 = v1538;
                    }
                } else {
                    if (v1546){
                        v1553 = v1545; v1554 = v1546;
                    } else {
                        v1553 = v1537; v1554 = v1538;
                    }
                }
                v1537 = v1553;
                v1538 = v1554;
                v1541 += 1 ;
            }
            v1539 += 1 ;
        }
        auto v1555 = cooperative_groups::coalesced_threads();
        int v1556;
        v1556 = threadIdx.x;
        int v1557;
        v1557 = v1556 / 32;
        auto v1558 = cooperative_groups::labeled_partition(v1555,v1557);
        Closure7 v1559{};
        int v1560; bool v1561;
        Tuple4 tmp108 = cooperative_groups::reduce(v1558, Tuple4{v1537, v1538}, v1559);
        v1560 = tmp108.v0; v1561 = tmp108.v1;
        bool v1562;
        v1562 = v1561 == false;
        if (v1562){
            assert("The local reduce must be true." && v1561);
        } else {
        }
        assert("Tensor range check" && 0 <= v1275 && v1275 < 8);
        int v1564;
        v1564 = 0;
        while (while_method_3(v1564)){
            assert("Tensor range check" && 0 <= v1564 && v1564 < 1);
            int v1566;
            v1566 = 128 * v1564;
            int v1567;
            v1567 = v1566 + v1284;
            assert("Tensor range check" && 0 <= v1564 && v1564 < 1);
            int v1568;
            v1568 = 4 * v1564;
            int4* v1569;
            v1569 = reinterpret_cast<int4*>(v1415 + v1568);
            int4* v1570;
            v1570 = reinterpret_cast<int4*>(v15 + v1567);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1569) % 16 == 0 && reinterpret_cast<unsigned long long>(v1570) % 16 == 0);
            *v1570 = *v1569;
            v1564 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1275 && v1275 < 8);
        int v1571;
        v1571 = 8 * v1275;
        int v1572;
        v1572 = v1571 + v1267;
        v16[v1572] = v1560;
        v1275 += 24 ;
    }
    v17.sync() ;
    return ;
}
extern "C" __global__ void entry6(int * v0, int * v1) {
    auto v2 = cooperative_groups::this_grid();
    extern __shared__ unsigned char v3[];
    int * v4;
    v4 = reinterpret_cast<int *>(&v3[0ull]);
    int v6;
    v6 = blockIdx.x;
    int v7;
    v7 = v6;
    while (while_method_5(v7)){
        bool v9;
        v9 = 0 <= v7;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The index needs to be zero or positive." && v9);
        } else {
        }
        int v12;
        v12 = v7 % 1;
        bool v13;
        v13 = v7 < 2;
        bool v14;
        v14 = v13 == false;
        if (v14){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v13);
        } else {
        }
        assert("Tensor range check" && 0 <= v7 && v7 < 2);
        assert("Tensor range check" && 0 <= v12 && v12 < 1);
        assert("Tensor range check" && 0 <= v12 && v12 < 1);
        int v16;
        v16 = 8 * v12;
        int v17;
        v17 = 32 * v12;
        int v18;
        v18 = v17 + v16;
        int v19;
        v19 = 32 * v7;
        int v20;
        v20 = v19 + v18;
        int v21;
        v21 = 4 * v12;
        int v22;
        v22 = v21 + v17;
        int v23;
        v23 = v19 + v22;
        int v24;
        v24 = threadIdx.x;
        int v25;
        v25 = v24;
        while (while_method_6(v25)){
            bool v27;
            v27 = 0 <= v25;
            bool v28;
            v28 = v27 == false;
            if (v28){
                assert("The index needs to be zero or positive." && v27);
            } else {
            }
            int v30;
            v30 = v25 % 8;
            int v31;
            v31 = v25 / 8;
            bool v32;
            v32 = v31 < 4;
            bool v33;
            v33 = v32 == false;
            if (v33){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v32);
            } else {
            }
            assert("Tensor range check" && 0 <= v31 && v31 < 4);
            assert("Tensor range check" && 0 <= v30 && v30 < 8);
            int v35;
            v35 = v30 + v20;
            int v36;
            v36 = 8 * v31;
            int v37;
            v37 = v36 + v35;
            int v38;
            v38 = v0[v37];
            assert("Tensor range check" && 0 <= v31 && v31 < 4);
            assert("Tensor range check" && 0 <= v30 && v30 < 8);
            int v39;
            v39 = 129 * v31;
            int v40;
            v40 = v39 + v30;
            v4[v40] = v38;
            v25 += 256 ;
        }
        int v41;
        v41 = threadIdx.x;
        int v42;
        v42 = v41;
        while (while_method_2(v42)){
            bool v44;
            v44 = 0 <= v42;
            bool v45;
            v45 = v44 == false;
            if (v45){
                assert("The index needs to be zero or positive." && v44);
            } else {
            }
            int v47;
            v47 = v42 % 1;
            bool v48;
            v48 = v42 < 8;
            bool v49;
            v49 = v48 == false;
            if (v49){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v48);
            } else {
            }
            assert("Tensor range check" && 0 <= v42 && v42 < 8);
            assert("Tensor range check" && 0 <= v47 && v47 < 1);
            int v51;
            v51 = 4 * v47;
            int v52;
            v52 = v51 + v23;
            int v53;
            v53 = 4 * v42;
            int v54;
            v54 = v53 + v52;
            int v55;
            v55 = 516 * v47;
            int v56;
            v56 = v42 + v55;
            int v57[4];
            int v58;
            v58 = 0;
            while (while_method_1(v58)){
                assert("Tensor range check" && 0 <= v58 && v58 < 4);
                int v60;
                v60 = 129 * v58;
                int v61;
                v61 = v60 + v56;
                int v62;
                v62 = v4[v61];
                assert("Tensor range check" && 0 <= v58 && v58 < 4);
                v57[v58] = v62;
                v58 += 1 ;
            }
            int4* v63;
            v63 = reinterpret_cast<int4*>(v57 + 0);
            int4* v64;
            v64 = reinterpret_cast<int4*>(v1 + v54);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v63) % 16 == 0 && reinterpret_cast<unsigned long long>(v64) % 16 == 0);
            *v64 = *v63;
            v42 += 256 ;
        }
        __syncthreads();
        v7 += 24 ;
    }
    v2.sync() ;
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
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

import sys
import pathlib
options = []
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=255')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : cp.ndarray, v9 : cp.ndarray, v10 : cp.ndarray, v11 : cp.ndarray, v12 : cp.ndarray, v13 : cp.ndarray, v14 : cp.ndarray, v15 : cp.ndarray, v16 : cp.ndarray, v17 : cp.ndarray) -> None:
    v18 = "test_text_outputs/primitives/"
    v19 = "test2/a/"
    v20 = "kernel_params.txt"
    v21 = pathlib.Path(v18,v19,v20)
    del v18, v19, v20
    v21.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v21),'w')
    del v21
    v22 = cp.cuda.Device().attributes['MultiProcessorCount']
    v23 = v22 == 24
    del v22
    v24 = v23 == False
    if v24:
        v25 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v23, v25
        del v25
    else:
        pass
    del v23, v24
    v26 = 0
    v27 = raw_module.get_function(f"entry{v26}")
    del v26
    v27.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v27((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v27
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method2(v0 : i32) -> bool:
    v1 = v0 < 64
    del v0
    return v1
def method3(v0 : i32) -> bool:
    v1 = v0 < 128
    del v0
    return v1
def method1(v0 : cp.ndarray) -> None:
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method4(v0 : cp.ndarray) -> None:
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
        while method3(v40):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method6(v0 : i32) -> bool:
    v1 = v0 < 1
    del v0
    return v1
def method5(v0 : cp.ndarray) -> None:
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
    while method6(v23):
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
    print(v34.format(),end="")
    del v34
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method7(v0 : cp.ndarray) -> None:
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method8(v0 : cp.ndarray) -> None:
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method9(v0 : cp.ndarray) -> None:
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method10(v0 : cp.ndarray) -> None:
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method11(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
        while method3(v43):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method12(v0 : cp.ndarray) -> None:
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
        while method3(v40):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method13(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
        while method3(v43):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method14(v0 : cp.ndarray) -> None:
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method15(v0 : cp.ndarray) -> None:
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
        while method3(v40):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method16(v0 : cp.ndarray) -> None:
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method17(v0 : cp.ndarray) -> None:
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method18(v0 : cp.ndarray) -> None:
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method19(v0 : cp.ndarray) -> None:
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method20(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : cp.ndarray, v9 : cp.ndarray, v10 : cp.ndarray, v11 : cp.ndarray, v12 : cp.ndarray, v13 : cp.ndarray, v14 : cp.ndarray, v15 : cp.ndarray, v16 : cp.ndarray, v17 : cp.ndarray) -> None:
    v18 = "test_text_outputs/primitives/"
    v19 = "test2/b/"
    v20 = "kernel_params.txt"
    v21 = pathlib.Path(v18,v19,v20)
    del v18, v19, v20
    v21.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v21),'w')
    del v21
    v22 = cp.cuda.Device().attributes['MultiProcessorCount']
    v23 = v22 == 24
    del v22
    v24 = v23 == False
    if v24:
        v25 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v23, v25
        del v25
    else:
        pass
    del v23, v24
    v26 = 1
    v27 = raw_module.get_function(f"entry{v26}")
    del v26
    v27.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v27((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v27
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method21(v0 : cp.ndarray) -> None:
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
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method22(v0 : cp.ndarray) -> None:
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
    while method3(v32):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method23(v0 : cp.ndarray) -> None:
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
    while method6(v23):
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
    print(v34.format(),end="")
    del v34
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method24(v0 : cp.ndarray) -> None:
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
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method25(v0 : cp.ndarray) -> None:
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
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method26(v0 : cp.ndarray) -> None:
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
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method27(v0 : cp.ndarray) -> None:
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
    while method3(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method28(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method3(v35):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method29(v0 : cp.ndarray) -> None:
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
    while method3(v32):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method30(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method3(v35):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method31(v0 : cp.ndarray) -> None:
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
    while method3(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method32(v0 : cp.ndarray) -> None:
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
    while method3(v32):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method33(v0 : cp.ndarray) -> None:
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
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method34(v0 : cp.ndarray) -> None:
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
    while method3(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method35(v0 : cp.ndarray) -> None:
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
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method36(v0 : cp.ndarray) -> None:
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
    while method3(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method37(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> None:
    v8 = "test_text_outputs/primitives/"
    v9 = "test3/a"
    v10 = "kernel_params.txt"
    v11 = pathlib.Path(v8,v9,v10)
    del v8, v9, v10
    v11.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v11),'w')
    del v11
    v12 = cp.cuda.Device().attributes['MultiProcessorCount']
    v13 = v12 == 24
    del v12
    v14 = v13 == False
    if v14:
        v15 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v13, v15
        del v15
    else:
        pass
    del v13, v14
    v16 = 2
    v17 = raw_module.get_function(f"entry{v16}")
    del v16
    v17.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v17((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v17
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method39(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method40(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def method38(v0 : cp.ndarray) -> None:
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
    while method39(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method41(v0 : cp.ndarray) -> None:
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
    while method39(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method42(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method39(v35):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method43(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "output_indices_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method39(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method44(v0 : cp.ndarray) -> None:
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
    while method39(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method45(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method39(v35):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method46(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> None:
    v8 = "test_text_outputs/primitives/"
    v9 = "test3/b"
    v10 = "kernel_params.txt"
    v11 = pathlib.Path(v8,v9,v10)
    del v8, v9, v10
    v11.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v11),'w')
    del v11
    v12 = cp.cuda.Device().attributes['MultiProcessorCount']
    v13 = v12 == 24
    del v12
    v14 = v13 == False
    if v14:
        v15 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v13, v15
        del v15
    else:
        pass
    del v13, v14
    v16 = 3
    v17 = raw_module.get_function(f"entry{v16}")
    del v16
    v17.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v17((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v17
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method47(v0 : cp.ndarray) -> None:
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
    while method39(v33):
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
        while method39(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method48(v0 : cp.ndarray) -> None:
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
    while method39(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method49(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method39(v35):
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
        while method39(v43):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method50(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "output_indices_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method39(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method51(v0 : cp.ndarray) -> None:
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
    while method39(v33):
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
        while method39(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method52(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method39(v35):
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
        while method39(v43):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method53(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : cp.ndarray, v9 : cp.ndarray, v10 : cp.ndarray, v11 : cp.ndarray, v12 : cp.ndarray, v13 : cp.ndarray, v14 : cp.ndarray, v15 : cp.ndarray, v16 : cp.ndarray) -> None:
    v17 = "test_text_outputs/primitives/"
    v18 = "test4/b/"
    v19 = "kernel_params.txt"
    v20 = pathlib.Path(v17,v18,v19)
    del v17, v18, v19
    v20.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v20),'w')
    del v20
    v21 = cp.cuda.Device().attributes['MultiProcessorCount']
    v22 = v21 == 24
    del v21
    v23 = v22 == False
    if v23:
        v24 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v22, v24
        del v24
    else:
        pass
    del v22, v23
    v25 = 4
    v26 = raw_module.get_function(f"entry{v25}")
    del v25
    v26.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v26((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v26
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method54(v0 : cp.ndarray) -> None:
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
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method55(v0 : cp.ndarray) -> None:
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
    while method3(v32):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method56(v0 : cp.ndarray) -> None:
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
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method57(v0 : cp.ndarray) -> None:
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
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method58(v0 : cp.ndarray) -> None:
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
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method59(v0 : cp.ndarray) -> None:
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
    while method3(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method60(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method3(v35):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method61(v0 : cp.ndarray) -> None:
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
    while method3(v32):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method62(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method3(v35):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method63(v0 : cp.ndarray) -> None:
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
    while method3(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method64(v0 : cp.ndarray) -> None:
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
    while method3(v32):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method65(v0 : cp.ndarray) -> None:
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
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method66(v0 : cp.ndarray) -> None:
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
    while method3(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method67(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_softmax''.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method3(v33):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method68(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_sampling'.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method3(v22):
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method69(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : cp.ndarray, v9 : cp.ndarray, v10 : cp.ndarray, v11 : cp.ndarray, v12 : cp.ndarray, v13 : cp.ndarray, v14 : cp.ndarray, v15 : cp.ndarray, v16 : cp.ndarray) -> None:
    v17 = "test_text_outputs/primitives/"
    v18 = "test4/a/"
    v19 = "kernel_params.txt"
    v20 = pathlib.Path(v17,v18,v19)
    del v17, v18, v19
    v20.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v20),'w')
    del v20
    v21 = cp.cuda.Device().attributes['MultiProcessorCount']
    v22 = v21 == 24
    del v21
    v23 = v22 == False
    if v23:
        v24 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v22, v24
        del v24
    else:
        pass
    del v22, v23
    v25 = 5
    v26 = raw_module.get_function(f"entry{v25}")
    del v25
    v26.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v26((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v26
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method70(v0 : cp.ndarray) -> None:
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method71(v0 : cp.ndarray) -> None:
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
        while method3(v40):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method72(v0 : cp.ndarray) -> None:
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method73(v0 : cp.ndarray) -> None:
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method74(v0 : cp.ndarray) -> None:
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method75(v0 : cp.ndarray) -> None:
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method76(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
        while method3(v43):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method77(v0 : cp.ndarray) -> None:
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
        while method3(v40):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method78(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
        while method3(v43):
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
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method79(v0 : cp.ndarray) -> None:
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method80(v0 : cp.ndarray) -> None:
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
        while method3(v40):
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
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method81(v0 : cp.ndarray) -> None:
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method82(v0 : cp.ndarray) -> None:
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method83(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_softmax''.txt"
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
        while method3(v41):
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
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method84(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_sampling'.txt"
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
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method86(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method87(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method88(v0 : i32) -> bool:
    v1 = v0 < 8
    del v0
    return v1
def method85() -> None:
    v0 = "test_text_outputs/primitives/"
    v1 = "test5"
    v2 = "transpose.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.arange(0,64,1,dtype=cp.int32) # type: ignore
    v5 = v4.size
    v6 = 64 == v5
    del v5
    v7 = v6 == False
    if v7:
        v8 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v6, v8
        del v8
    else:
        pass
    del v6, v7
    v9 = cp.empty(64,dtype=cp.int32)
    v37 = 0
    v38 = "{}"
    print(v38.format('['),end="")
    v39 = 0
    while method86(v39):
        v41 = v37
        v42 = v41 >= 100
        del v41
        if v42:
            v43 = " ..."
            print(v38.format(v43),end="")
            del v43
            break
        else:
            pass
        del v42
        v44 = v39 == 0
        v45 = v44 != True
        del v44
        if v45:
            v46 = "; "
            print(v38.format(v46),end="")
            del v46
        else:
            pass
        del v45
        print(v38.format('['),end="")
        v47 = 0
        while method87(v47):
            v49 = v37
            v50 = v49 >= 100
            del v49
            if v50:
                v51 = " ..."
                print(v38.format(v51),end="")
                del v51
                break
            else:
                pass
            del v50
            v52 = v47 == 0
            v53 = v52 != True
            del v52
            if v53:
                v54 = "; "
                print(v38.format(v54),end="")
                del v54
            else:
                pass
            del v53
            print(v38.format('['),end="")
            v55 = 0
            while method88(v55):
                v57 = v37
                v58 = v57 >= 100
                del v57
                if v58:
                    v59 = " ..."
                    print(v38.format(v59),end="")
                    del v59
                    break
                else:
                    pass
                del v58
                v60 = v55 == 0
                v61 = v60 != True
                del v60
                if v61:
                    v62 = "; "
                    print(v38.format(v62),end="")
                    del v62
                else:
                    pass
                del v61
                v63 = v37 + 1
                v37 = v63
                del v63
                v64 = v39 * 32
                v65 = v47 * 8
                v66 = v64 + v65
                del v64, v65
                v67 = v66 + v55
                del v66
                v68 = v4[v67].item()
                del v67
                print(v38.format(v68),end="")
                del v68
                v55 += 1 
            del v55
            print(v38.format(']'),end="")
            v47 += 1 
        del v47
        print(v38.format(']'),end="")
        v39 += 1 
    del v37, v39
    print(v38.format(']'),end="")
    v69 = "\n"
    print(v69.format(),end="")
    v70 = cp.cuda.Device().attributes['MultiProcessorCount']
    v71 = v70 == 24
    del v70
    v72 = v71 == False
    if v72:
        v73 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v71, v73
        del v73
    else:
        pass
    del v71, v72
    v74 = 6
    v75 = raw_module.get_function(f"entry{v74}")
    del v74
    v75.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v75((24,),(256,),(v4, v9),shared_mem=98304)
    del v4, v75
    v103 = 0
    print(v38.format('['),end="")
    v104 = 0
    while method86(v104):
        v106 = v103
        v107 = v106 >= 100
        del v106
        if v107:
            v108 = " ..."
            print(v38.format(v108),end="")
            del v108
            break
        else:
            pass
        del v107
        v109 = v104 == 0
        v110 = v109 != True
        del v109
        if v110:
            v111 = "; "
            print(v38.format(v111),end="")
            del v111
        else:
            pass
        del v110
        print(v38.format('['),end="")
        v112 = 0
        while method88(v112):
            v114 = v103
            v115 = v114 >= 100
            del v114
            if v115:
                v116 = " ..."
                print(v38.format(v116),end="")
                del v116
                break
            else:
                pass
            del v115
            v117 = v112 == 0
            v118 = v117 != True
            del v117
            if v118:
                v119 = "; "
                print(v38.format(v119),end="")
                del v119
            else:
                pass
            del v118
            print(v38.format('['),end="")
            v120 = 0
            while method87(v120):
                v122 = v103
                v123 = v122 >= 100
                del v122
                if v123:
                    v124 = " ..."
                    print(v38.format(v124),end="")
                    del v124
                    break
                else:
                    pass
                del v123
                v125 = v120 == 0
                v126 = v125 != True
                del v125
                if v126:
                    v127 = "; "
                    print(v38.format(v127),end="")
                    del v127
                else:
                    pass
                del v126
                v128 = v103 + 1
                v103 = v128
                del v128
                v129 = v104 * 32
                v130 = v112 * 4
                v131 = v129 + v130
                del v129, v130
                v132 = v131 + v120
                del v131
                v133 = v9[v132].item()
                del v132
                print(v38.format(v133),end="")
                del v133
                v120 += 1 
            del v120
            print(v38.format(']'),end="")
            v112 += 1 
        del v112
        print(v38.format(']'),end="")
        v104 += 1 
    del v9, v103, v104
    print(v38.format(']'),end="")
    del v38
    print(v69.format(),end="")
    del v69
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
    v20 = cp.empty(8192,dtype=cp.float32)
    v21 = cp.empty(64,dtype=cp.int32)
    method0(v0, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21)
    method1(v5)
    del v5
    method4(v0)
    del v0
    method5(v6)
    del v6
    method7(v8)
    del v8
    method8(v9)
    del v9
    method9(v12)
    del v12
    method10(v13)
    del v13
    method11(v10, v11)
    del v10, v11
    method12(v7)
    del v7
    method13(v14, v15)
    del v14, v15
    method14(v16)
    del v16
    method15(v17)
    del v17
    method16(v18)
    del v18
    method17(v19)
    del v19
    method18(v20)
    del v20
    method19(v21)
    del v21
    cp.random.seed(12344321)
    v22 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v23 = v22.size
    v24 = 8192 == v23
    del v23
    v25 = v24 == False
    if v25:
        v26 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v24, v26
        del v26
    else:
        pass
    del v24, v25
    v27 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v28 = cp.empty(1,dtype=cp.float32)
    v29 = cp.empty(8192,dtype=cp.int32)
    v30 = cp.empty(8192,dtype=cp.float32)
    v31 = cp.empty(8192,dtype=cp.float32)
    v32 = cp.empty(8192,dtype=cp.float32)
    v33 = cp.empty(8192,dtype=cp.float32)
    v34 = cp.empty(8192,dtype=cp.float32)
    v35 = cp.empty(128,dtype=cp.int32)
    v36 = cp.empty(8192,dtype=cp.int32)
    v37 = cp.empty(8192,dtype=cp.int32)
    v38 = cp.empty(128,dtype=cp.int32)
    v39 = cp.empty(8192,dtype=cp.int32)
    v40 = cp.empty(8192,dtype=cp.float32)
    v41 = cp.empty(128,dtype=cp.int32)
    v42 = cp.empty(8192,dtype=cp.float32)
    v43 = cp.empty(128,dtype=cp.int32)
    method20(v22, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43)
    method21(v27)
    del v27
    method22(v22)
    del v22
    method23(v28)
    del v28
    method24(v30)
    del v30
    method25(v31)
    del v31
    method26(v34)
    del v34
    method27(v35)
    del v35
    method28(v32, v33)
    del v32, v33
    method29(v29)
    del v29
    method30(v36, v37)
    del v36, v37
    method31(v38)
    del v38
    method32(v39)
    del v39
    method33(v40)
    del v40
    method34(v41)
    del v41
    method35(v42)
    del v42
    method36(v43)
    del v43
    cp.random.seed(12344321)
    v44 = cp.arange(0,4096,1,dtype=cp.float32) # type: ignore
    v45 = v44.size
    v46 = 4096 == v45
    del v45
    v47 = v46 == False
    if v47:
        v48 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v46, v48
        del v48
    else:
        pass
    del v46, v47
    v49 = cp.random.normal(0.0,1.0,4096,dtype=cp.float32) # type: ignore
    v50 = cp.empty(4096,dtype=cp.int32)
    v51 = cp.empty(4096,dtype=cp.int32)
    v52 = cp.empty(256,dtype=cp.int32)
    v53 = cp.empty(256,dtype=cp.int32)
    v54 = cp.empty(4096,dtype=cp.float32)
    v55 = cp.empty(4096,dtype=cp.float32)
    method37(v44, v49, v50, v51, v52, v53, v54, v55)
    method38(v44)
    del v44
    method41(v53)
    del v53
    method42(v50, v51)
    del v50, v51
    method43(v52)
    del v52
    method44(v55)
    del v55
    method45(v49, v54)
    del v49, v54
    cp.random.seed(12344321)
    v56 = cp.arange(0,65536,1,dtype=cp.float32) # type: ignore
    v57 = v56.size
    v58 = 65536 == v57
    del v57
    v59 = v58 == False
    if v59:
        v60 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v58, v60
        del v60
    else:
        pass
    del v58, v59
    v61 = cp.random.normal(0.0,1.0,65536,dtype=cp.float32) # type: ignore
    v62 = cp.empty(65536,dtype=cp.int32)
    v63 = cp.empty(65536,dtype=cp.int32)
    v64 = cp.empty(256,dtype=cp.int32)
    v65 = cp.empty(256,dtype=cp.int32)
    v66 = cp.empty(65536,dtype=cp.float32)
    v67 = cp.empty(65536,dtype=cp.float32)
    method46(v56, v61, v62, v63, v64, v65, v66, v67)
    method47(v56)
    del v56
    method48(v65)
    del v65
    method49(v62, v63)
    del v62, v63
    method50(v64)
    del v64
    method51(v67)
    del v67
    method52(v61, v66)
    del v61, v66
    cp.random.seed(12344321)
    v68 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v69 = v68.size
    v70 = 8192 == v69
    del v69
    v71 = v70 == False
    if v71:
        v72 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v70, v72
        del v72
    else:
        pass
    del v70, v71
    v73 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v74 = cp.empty(8192,dtype=cp.int32)
    v75 = cp.empty(8192,dtype=cp.float32)
    v76 = cp.empty(8192,dtype=cp.float32)
    v77 = cp.empty(8192,dtype=cp.float32)
    v78 = cp.empty(8192,dtype=cp.float32)
    v79 = cp.empty(8192,dtype=cp.float32)
    v80 = cp.empty(128,dtype=cp.int32)
    v81 = cp.empty(8192,dtype=cp.int32)
    v82 = cp.empty(8192,dtype=cp.int32)
    v83 = cp.empty(128,dtype=cp.int32)
    v84 = cp.empty(8192,dtype=cp.int32)
    v85 = cp.empty(8192,dtype=cp.float32)
    v86 = cp.empty(128,dtype=cp.int32)
    v87 = cp.empty(8192,dtype=cp.float32)
    v88 = cp.empty(128,dtype=cp.int32)
    method53(v68, v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88)
    method54(v73)
    del v73
    method55(v68)
    del v68
    method56(v75)
    del v75
    method57(v76)
    del v76
    method58(v79)
    del v79
    method59(v80)
    del v80
    method60(v77, v78)
    del v77, v78
    method61(v74)
    del v74
    method62(v81, v82)
    del v81, v82
    method63(v83)
    del v83
    method64(v84)
    del v84
    method65(v85)
    del v85
    method66(v86)
    del v86
    method67(v87)
    del v87
    method68(v88)
    del v88
    cp.random.seed(12344321)
    v89 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v90 = v89.size
    v91 = 8192 == v90
    del v90
    v92 = v91 == False
    if v92:
        v93 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v91, v93
        del v93
    else:
        pass
    del v91, v92
    v94 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v95 = cp.empty(8192,dtype=cp.int32)
    v96 = cp.empty(8192,dtype=cp.float32)
    v97 = cp.empty(8192,dtype=cp.float32)
    v98 = cp.empty(8192,dtype=cp.float32)
    v99 = cp.empty(8192,dtype=cp.float32)
    v100 = cp.empty(8192,dtype=cp.float32)
    v101 = cp.empty(64,dtype=cp.int32)
    v102 = cp.empty(8192,dtype=cp.int32)
    v103 = cp.empty(8192,dtype=cp.int32)
    v104 = cp.empty(64,dtype=cp.int32)
    v105 = cp.empty(8192,dtype=cp.int32)
    v106 = cp.empty(8192,dtype=cp.float32)
    v107 = cp.empty(64,dtype=cp.int32)
    v108 = cp.empty(8192,dtype=cp.float32)
    v109 = cp.empty(64,dtype=cp.int32)
    method69(v89, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107, v108, v109)
    method70(v94)
    del v94
    method71(v89)
    del v89
    method72(v96)
    del v96
    method73(v97)
    del v97
    method74(v100)
    del v100
    method75(v101)
    del v101
    method76(v98, v99)
    del v98, v99
    method77(v95)
    del v95
    method78(v102, v103)
    del v102, v103
    method79(v104)
    del v104
    method80(v105)
    del v105
    method81(v106)
    del v106
    method82(v107)
    del v107
    method83(v108)
    del v108
    method84(v109)
    del v109
    return method85()

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
