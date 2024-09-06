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
extern "C" __global__ void entry0(int * v0, float * v1, float * v2, int * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int * v9, int * v10, int * v11, int * v12, int * v13, float * v14, int * v15, float * v16, int * v17) {
    float v18;
    v18 = 0.0f;
    int v19;
    v19 = threadIdx.x;
    int v20;
    v20 = v19;
    while (while_method_0(v20)){
        bool v22;
        v22 = 0l <= v20;
        bool v23;
        v23 = v22 == false;
        if (v23){
            assert("The index needs to be zero or positive." && v22);
        } else {
        }
        int v25;
        v25 = v20 % 32l;
        int v26;
        v26 = v20 / 32l;
        bool v27;
        v27 = v26 < 64l;
        bool v28;
        v28 = v27 == false;
        if (v28){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v27);
        } else {
        }
        assert("Tensor range check" && 0 <= v26 && v26 < 64l);
        assert("Tensor range check" && 0 <= v25 && v25 < 32l);
        int v30;
        v30 = 4l * v25;
        int v31;
        v31 = 128l * v26;
        int v32;
        v32 = v31 + v30;
        float v33[4l];
        int4* v34;
        v34 = reinterpret_cast<int4*>(v1 + v32);
        int4* v35;
        v35 = reinterpret_cast<int4*>(v33 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v34) % 4l == 0 && (unsigned long long)(v35) % 4l == 0);
        *v35 = *v34;
        int v36; float v37;
        Tuple0 tmp0 = Tuple0{0l, v18};
        v36 = tmp0.v0; v37 = tmp0.v1;
        while (while_method_1(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 4l);
            float v39;
            v39 = v33[v36];
            float v40;
            v40 = v37 + v39;
            v37 = v40;
            v36 += 1l ;
        }
        v18 = v37;
        v20 += 32l ;
    }
    auto v41 = cooperative_groups::coalesced_threads();
    Closure0 v42{};
    float v43;
    v43 = cooperative_groups::reduce(v41, v18, v42);
    int v44;
    v44 = threadIdx.x;
    int v45;
    v45 = v44 / 32l;
    extern __shared__ unsigned char v46[];
    float * v47;
    v47 = reinterpret_cast<float *>(&v46[0ull]);
    assert("Tensor range check" && 0 <= v45 && v45 < 1l);
    v47[v45] = v43;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v49;
    v49 = threadIdx.x;
    int v50;
    v50 = v49 % 32l;
    bool v51;
    v51 = v45 == 0l;
    bool v53;
    if (v51){
        bool v52;
        v52 = v50 < 1l;
        v53 = v52;
    } else {
        v53 = false;
    }
    if (v53){
        auto v54 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v50 && v50 < 1l);
        float v55;
        v55 = v47[v50];
        float v56;
        v56 = cooperative_groups::reduce(v54, v55, v42);
        v2[0l] = v56;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v57;
    v57 = threadIdx.x;
    bool v58;
    v58 = 0l <= v57;
    bool v59;
    v59 = v58 == false;
    if (v59){
        assert("The index needs to be zero or positive." && v58);
    } else {
    }
    int v61;
    v61 = v57 % 32l;
    int v62;
    v62 = v57 / 32l;
    bool v63;
    v63 = v62 < 1l;
    bool v64;
    v64 = v63 == false;
    if (v64){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v63);
    } else {
    }
    assert("Tensor range check" && 0 <= v62 && v62 < 1l);
    assert("Tensor range check" && 0 <= v61 && v61 < 32l);
    int v66;
    v66 = 4l * v61;
    int v67;
    v67 = 128l * v62;
    int v68;
    v68 = v67 + v66;
    assert("Tensor range check" && 0 <= v62 && v62 < 1l);
    assert("Tensor range check" && 0 <= v61 && v61 < 32l);
    int v69;
    v69 = 0l;
    while (while_method_2(v69)){
        assert("Tensor range check" && 0 <= v69 && v69 < 64l);
        int v71;
        v71 = 128l * v69;
        int v72;
        v72 = v71 + v68;
        int v73[4l];
        int v74[4l];
        int v75;
        v75 = 0l;
        while (while_method_3(v75)){
            assert("Tensor range check" && 0 <= v75 && v75 < 1l);
            int v77;
            v77 = 4l * v75;
            assert("Tensor range check" && 0 <= v75 && v75 < 1l);
            int v78;
            v78 = 128l * v75;
            int v79;
            v79 = v78 + v72;
            int4* v80;
            v80 = reinterpret_cast<int4*>(v0 + v79);
            int4* v81;
            v81 = reinterpret_cast<int4*>(v73 + v77);
            assert("Pointer alignment check" && (unsigned long long)(v80) % 4l == 0 && (unsigned long long)(v81) % 4l == 0);
            *v81 = *v80;
            v75 += 1l ;
        }
        int v82;
        v82 = 0l;
        while (while_method_3(v82)){
            int v84;
            v84 = 0l;
            while (while_method_1(v84)){
                bool v86;
                v86 = 0l <= v84;
                bool v88;
                if (v86){
                    bool v87;
                    v87 = v84 < 4l;
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
                v91 = 0l <= v61;
                bool v93;
                if (v91){
                    bool v92;
                    v92 = v61 < 32l;
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
                v96 = v61 * 4l;
                int v97;
                v97 = v84 + v96;
                bool v98;
                v98 = 0l <= v82;
                bool v100;
                if (v98){
                    bool v99;
                    v99 = v82 < 1l;
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
                v103 = v82 * 128l;
                int v104;
                v104 = v97 + v103;
                assert("Tensor range check" && 0 <= v82 && v82 < 1l);
                assert("Tensor range check" && 0 <= v84 && v84 < 4l);
                int v105;
                v105 = 4l * v82;
                int v106;
                v106 = v105 + v84;
                v74[v106] = v104;
                v84 += 1l ;
            }
            v82 += 1l ;
        }
        bool v107;
        v107 = 0l <= v62;
        bool v108;
        v108 = v107 && v63;
        bool v109;
        v109 = v108 == false;
        if (v109){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v108);
        } else {
        }
        bool v111;
        v111 = 0l <= v69;
        bool v113;
        if (v111){
            bool v112;
            v112 = v69 < 64l;
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
        v116 = v69 + v62;
        assert("Tensor range check" && 0 <= v69 && v69 < 64l);
        int v117;
        v117 = 0l;
        while (while_method_3(v117)){
            assert("Tensor range check" && 0 <= v117 && v117 < 1l);
            int v119;
            v119 = 128l * v117;
            int v120;
            v120 = v119 + v72;
            assert("Tensor range check" && 0 <= v117 && v117 < 1l);
            int v121;
            v121 = 4l * v117;
            int4* v122;
            v122 = reinterpret_cast<int4*>(v73 + v121);
            int4* v123;
            v123 = reinterpret_cast<int4*>(v3 + v120);
            assert("Pointer alignment check" && (unsigned long long)(v122) % 4l == 0 && (unsigned long long)(v123) % 4l == 0);
            *v123 = *v122;
            v117 += 1l ;
        }
        v69 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v124;
    v124 = threadIdx.x;
    bool v125;
    v125 = 0l <= v124;
    bool v126;
    v126 = v125 == false;
    if (v126){
        assert("The index needs to be zero or positive." && v125);
    } else {
    }
    int v128;
    v128 = v124 % 32l;
    int v129;
    v129 = v124 / 32l;
    bool v130;
    v130 = v129 < 1l;
    bool v131;
    v131 = v130 == false;
    if (v131){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v130);
    } else {
    }
    assert("Tensor range check" && 0 <= v129 && v129 < 1l);
    assert("Tensor range check" && 0 <= v128 && v128 < 32l);
    int v133;
    v133 = 4l * v128;
    int v134;
    v134 = 128l * v129;
    int v135;
    v135 = v134 + v133;
    assert("Tensor range check" && 0 <= v129 && v129 < 1l);
    assert("Tensor range check" && 0 <= v128 && v128 < 32l);
    int v136;
    v136 = 0l;
    while (while_method_2(v136)){
        assert("Tensor range check" && 0 <= v136 && v136 < 64l);
        int v138;
        v138 = 128l * v136;
        int v139;
        v139 = v138 + v135;
        float v140[4l];
        int v141[4l];
        int v142;
        v142 = 0l;
        while (while_method_3(v142)){
            assert("Tensor range check" && 0 <= v142 && v142 < 1l);
            int v144;
            v144 = 4l * v142;
            assert("Tensor range check" && 0 <= v142 && v142 < 1l);
            int v145;
            v145 = 128l * v142;
            int v146;
            v146 = v145 + v139;
            int4* v147;
            v147 = reinterpret_cast<int4*>(v1 + v146);
            int4* v148;
            v148 = reinterpret_cast<int4*>(v140 + v144);
            assert("Pointer alignment check" && (unsigned long long)(v147) % 4l == 0 && (unsigned long long)(v148) % 4l == 0);
            *v148 = *v147;
            v142 += 1l ;
        }
        int v149;
        v149 = 0l;
        while (while_method_3(v149)){
            int v151;
            v151 = 0l;
            while (while_method_1(v151)){
                bool v153;
                v153 = 0l <= v151;
                bool v155;
                if (v153){
                    bool v154;
                    v154 = v151 < 4l;
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
                bool v158;
                v158 = 0l <= v128;
                bool v160;
                if (v158){
                    bool v159;
                    v159 = v128 < 32l;
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
                v163 = v128 * 4l;
                int v164;
                v164 = v151 + v163;
                bool v165;
                v165 = 0l <= v149;
                bool v167;
                if (v165){
                    bool v166;
                    v166 = v149 < 1l;
                    v167 = v166;
                } else {
                    v167 = false;
                }
                bool v168;
                v168 = v167 == false;
                if (v168){
                    assert("The indices should be inside the range of the dimension." && v167);
                } else {
                }
                int v170;
                v170 = v149 * 128l;
                int v171;
                v171 = v164 + v170;
                assert("Tensor range check" && 0 <= v149 && v149 < 1l);
                assert("Tensor range check" && 0 <= v151 && v151 < 4l);
                int v172;
                v172 = 4l * v149;
                int v173;
                v173 = v172 + v151;
                v141[v173] = v171;
                v151 += 1l ;
            }
            v149 += 1l ;
        }
        bool v174;
        v174 = 0l <= v129;
        bool v175;
        v175 = v174 && v130;
        bool v176;
        v176 = v175 == false;
        if (v176){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v175);
        } else {
        }
        bool v178;
        v178 = 0l <= v136;
        bool v180;
        if (v178){
            bool v179;
            v179 = v136 < 64l;
            v180 = v179;
        } else {
            v180 = false;
        }
        bool v181;
        v181 = v180 == false;
        if (v181){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v180);
        } else {
        }
        int v183;
        v183 = v136 + v129;
        int v184[4l];
        int v185[4l];
        int v186;
        v186 = 0l;
        while (while_method_3(v186)){
            int v188;
            v188 = 0l;
            while (while_method_1(v188)){
                assert("Tensor range check" && 0 <= v186 && v186 < 1l);
                assert("Tensor range check" && 0 <= v188 && v188 < 4l);
                int v190;
                v190 = 4l * v186;
                int v191;
                v191 = v190 + v188;
                int v192;
                v192 = v141[v191];
                assert("Tensor range check" && 0 <= v186 && v186 < 1l);
                assert("Tensor range check" && 0 <= v188 && v188 < 4l);
                v184[v191] = v183;
                v185[v191] = v192;
                v188 += 1l ;
            }
            v186 += 1l ;
        }
        assert("Tensor range check" && 0 <= v136 && v136 < 64l);
        int v193;
        v193 = 0l;
        while (while_method_3(v193)){
            assert("Tensor range check" && 0 <= v193 && v193 < 1l);
            int v195;
            v195 = 128l * v193;
            int v196;
            v196 = v195 + v139;
            assert("Tensor range check" && 0 <= v193 && v193 < 1l);
            int v197;
            v197 = 4l * v193;
            int4* v198;
            v198 = reinterpret_cast<int4*>(v184 + v197);
            int4* v199;
            v199 = reinterpret_cast<int4*>(v10 + v196);
            assert("Pointer alignment check" && (unsigned long long)(v198) % 4l == 0 && (unsigned long long)(v199) % 4l == 0);
            *v199 = *v198;
            int4* v200;
            v200 = reinterpret_cast<int4*>(v185 + v197);
            int4* v201;
            v201 = reinterpret_cast<int4*>(v11 + v196);
            assert("Pointer alignment check" && (unsigned long long)(v200) % 4l == 0 && (unsigned long long)(v201) % 4l == 0);
            *v201 = *v200;
            v193 += 1l ;
        }
        v136 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v202;
    v202 = threadIdx.x;
    bool v203;
    v203 = 0l <= v202;
    bool v204;
    v204 = v203 == false;
    if (v204){
        assert("The index needs to be zero or positive." && v203);
    } else {
    }
    int v206;
    v206 = v202 % 32l;
    int v207;
    v207 = v202 / 32l;
    bool v208;
    v208 = v207 < 1l;
    bool v209;
    v209 = v208 == false;
    if (v209){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v208);
    } else {
    }
    assert("Tensor range check" && 0 <= v207 && v207 < 1l);
    assert("Tensor range check" && 0 <= v206 && v206 < 32l);
    int v211;
    v211 = 4l * v206;
    int v212;
    v212 = 128l * v207;
    int v213;
    v213 = v212 + v211;
    assert("Tensor range check" && 0 <= v207 && v207 < 1l);
    int v214;
    v214 = 0l;
    while (while_method_2(v214)){
        assert("Tensor range check" && 0 <= v214 && v214 < 64l);
        int v216;
        v216 = 128l * v214;
        int v217;
        v217 = v216 + v213;
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
                v236 = 0l <= v206;
                bool v238;
                if (v236){
                    bool v237;
                    v237 = v206 < 32l;
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
                v241 = v206 * 4l;
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
        v252 = 0l <= v207;
        bool v253;
        v253 = v252 && v208;
        bool v254;
        v254 = v253 == false;
        if (v254){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v253);
        } else {
        }
        bool v256;
        v256 = 0l <= v214;
        bool v258;
        if (v256){
            bool v257;
            v257 = v214 < 64l;
            v258 = v257;
        } else {
            v258 = false;
        }
        bool v259;
        v259 = v258 == false;
        if (v259){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v258);
        } else {
        }
        int v261;
        v261 = v214 + v207;
        assert("Tensor range check" && 0 <= v214 && v214 < 64l);
        v12[v261] = v261;
        v214 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v262;
    v262 = threadIdx.x;
    bool v263;
    v263 = 0l <= v262;
    bool v264;
    v264 = v263 == false;
    if (v264){
        assert("The index needs to be zero or positive." && v263);
    } else {
    }
    int v266;
    v266 = v262 % 32l;
    int v267;
    v267 = v262 / 32l;
    bool v268;
    v268 = v267 < 1l;
    bool v269;
    v269 = v268 == false;
    if (v269){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v268);
    } else {
    }
    assert("Tensor range check" && 0 <= v267 && v267 < 1l);
    assert("Tensor range check" && 0 <= v266 && v266 < 32l);
    int v271;
    v271 = 4l * v266;
    int v272;
    v272 = 128l * v267;
    int v273;
    v273 = v272 + v271;
    assert("Tensor range check" && 0 <= v267 && v267 < 1l);
    assert("Tensor range check" && 0 <= v266 && v266 < 32l);
    int v274;
    v274 = 0l;
    while (while_method_2(v274)){
        assert("Tensor range check" && 0 <= v274 && v274 < 64l);
        int v276;
        v276 = 128l * v274;
        int v277;
        v277 = v276 + v273;
        float v278[4l];
        int v279[4l];
        int v280;
        v280 = 0l;
        while (while_method_3(v280)){
            assert("Tensor range check" && 0 <= v280 && v280 < 1l);
            int v282;
            v282 = 4l * v280;
            assert("Tensor range check" && 0 <= v280 && v280 < 1l);
            int v283;
            v283 = 128l * v280;
            int v284;
            v284 = v283 + v277;
            int4* v285;
            v285 = reinterpret_cast<int4*>(v1 + v284);
            int4* v286;
            v286 = reinterpret_cast<int4*>(v278 + v282);
            assert("Pointer alignment check" && (unsigned long long)(v285) % 4l == 0 && (unsigned long long)(v286) % 4l == 0);
            *v286 = *v285;
            v280 += 1l ;
        }
        int v287;
        v287 = 0l;
        while (while_method_3(v287)){
            int v289;
            v289 = 0l;
            while (while_method_1(v289)){
                bool v291;
                v291 = 0l <= v289;
                bool v293;
                if (v291){
                    bool v292;
                    v292 = v289 < 4l;
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
                bool v296;
                v296 = 0l <= v266;
                bool v298;
                if (v296){
                    bool v297;
                    v297 = v266 < 32l;
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
                int v301;
                v301 = v266 * 4l;
                int v302;
                v302 = v289 + v301;
                bool v303;
                v303 = 0l <= v287;
                bool v305;
                if (v303){
                    bool v304;
                    v304 = v287 < 1l;
                    v305 = v304;
                } else {
                    v305 = false;
                }
                bool v306;
                v306 = v305 == false;
                if (v306){
                    assert("The indices should be inside the range of the dimension." && v305);
                } else {
                }
                int v308;
                v308 = v287 * 128l;
                int v309;
                v309 = v302 + v308;
                assert("Tensor range check" && 0 <= v287 && v287 < 1l);
                assert("Tensor range check" && 0 <= v289 && v289 < 4l);
                int v310;
                v310 = 4l * v287;
                int v311;
                v311 = v310 + v289;
                v279[v311] = v309;
                v289 += 1l ;
            }
            v287 += 1l ;
        }
        bool v312;
        v312 = 0l <= v267;
        bool v313;
        v313 = v312 && v268;
        bool v314;
        v314 = v313 == false;
        if (v314){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v313);
        } else {
        }
        bool v316;
        v316 = 0l <= v274;
        bool v318;
        if (v316){
            bool v317;
            v317 = v274 < 64l;
            v318 = v317;
        } else {
            v318 = false;
        }
        bool v319;
        v319 = v318 == false;
        if (v319){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v318);
        } else {
        }
        int v321;
        v321 = v274 + v267;
        float v322;
        v322 = 0.0f;
        int v323;
        v323 = 0l;
        while (while_method_3(v323)){
            int v325;
            v325 = 0l;
            while (while_method_1(v325)){
                assert("Tensor range check" && 0 <= v323 && v323 < 1l);
                assert("Tensor range check" && 0 <= v325 && v325 < 4l);
                int v327;
                v327 = 4l * v323;
                int v328;
                v328 = v327 + v325;
                float v329;
                v329 = v278[v328];
                float v330;
                v330 = v322 + v329;
                v322 = v330;
                v325 += 1l ;
            }
            v323 += 1l ;
        }
        auto v331 = cooperative_groups::coalesced_threads();
        int v332;
        v332 = threadIdx.x;
        int v333;
        v333 = v332 / 32l;
        auto v334 = cooperative_groups::labeled_partition(v331,v333);
        float v335;
        v335 = cooperative_groups::reduce(v334, v322, v42);
        float v336;
        v336 = v335 / 128.0f;
        float v337[4l];
        int v338;
        v338 = 0l;
        while (while_method_3(v338)){
            int v340;
            v340 = 0l;
            while (while_method_1(v340)){
                assert("Tensor range check" && 0 <= v338 && v338 < 1l);
                assert("Tensor range check" && 0 <= v340 && v340 < 4l);
                int v342;
                v342 = 4l * v338;
                int v343;
                v343 = v342 + v340;
                float v344;
                v344 = v278[v343];
                float v345;
                v345 = v344 - v336;
                float v346;
                v346 = exp(v345);
                assert("Tensor range check" && 0 <= v338 && v338 < 1l);
                assert("Tensor range check" && 0 <= v340 && v340 < 4l);
                v337[v343] = v346;
                v340 += 1l ;
            }
            v338 += 1l ;
        }
        float v347;
        v347 = 0.0f;
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
                v354 = v337[v353];
                float v355;
                v355 = v347 + v354;
                v347 = v355;
                v350 += 1l ;
            }
            v348 += 1l ;
        }
        auto v356 = cooperative_groups::coalesced_threads();
        int v357;
        v357 = threadIdx.x;
        int v358;
        v358 = v357 / 32l;
        auto v359 = cooperative_groups::labeled_partition(v356,v358);
        float v360;
        v360 = cooperative_groups::reduce(v359, v347, v42);
        float v361[4l];
        int v362;
        v362 = 0l;
        while (while_method_3(v362)){
            int v364;
            v364 = 0l;
            while (while_method_1(v364)){
                assert("Tensor range check" && 0 <= v362 && v362 < 1l);
                assert("Tensor range check" && 0 <= v364 && v364 < 4l);
                int v366;
                v366 = 4l * v362;
                int v367;
                v367 = v366 + v364;
                float v368;
                v368 = v337[v367];
                float v369;
                v369 = v368 / v360;
                assert("Tensor range check" && 0 <= v362 && v362 < 1l);
                assert("Tensor range check" && 0 <= v364 && v364 < 4l);
                v361[v367] = v369;
                v364 += 1l ;
            }
            v362 += 1l ;
        }
        assert("Tensor range check" && 0 <= v274 && v274 < 64l);
        int v370;
        v370 = 0l;
        while (while_method_3(v370)){
            assert("Tensor range check" && 0 <= v370 && v370 < 1l);
            int v372;
            v372 = 128l * v370;
            int v373;
            v373 = v372 + v277;
            assert("Tensor range check" && 0 <= v370 && v370 < 1l);
            int v374;
            v374 = 4l * v370;
            int4* v375;
            v375 = reinterpret_cast<int4*>(v361 + v374);
            int4* v376;
            v376 = reinterpret_cast<int4*>(v4 + v373);
            assert("Pointer alignment check" && (unsigned long long)(v375) % 4l == 0 && (unsigned long long)(v376) % 4l == 0);
            *v376 = *v375;
            v370 += 1l ;
        }
        v274 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v377;
    v377 = threadIdx.x;
    bool v378;
    v378 = 0l <= v377;
    bool v379;
    v379 = v378 == false;
    if (v379){
        assert("The index needs to be zero or positive." && v378);
    } else {
    }
    int v381;
    v381 = v377 % 32l;
    int v382;
    v382 = v377 / 32l;
    bool v383;
    v383 = v382 < 1l;
    bool v384;
    v384 = v383 == false;
    if (v384){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v383);
    } else {
    }
    assert("Tensor range check" && 0 <= v382 && v382 < 1l);
    assert("Tensor range check" && 0 <= v381 && v381 < 32l);
    int v386;
    v386 = 4l * v381;
    int v387;
    v387 = 128l * v382;
    int v388;
    v388 = v387 + v386;
    assert("Tensor range check" && 0 <= v382 && v382 < 1l);
    assert("Tensor range check" && 0 <= v381 && v381 < 32l);
    int v389;
    v389 = 0l;
    while (while_method_2(v389)){
        assert("Tensor range check" && 0 <= v389 && v389 < 64l);
        int v391;
        v391 = 128l * v389;
        int v392;
        v392 = v391 + v388;
        float v393[4l];
        int v394[4l];
        int v395;
        v395 = 0l;
        while (while_method_3(v395)){
            assert("Tensor range check" && 0 <= v395 && v395 < 1l);
            int v397;
            v397 = 4l * v395;
            assert("Tensor range check" && 0 <= v395 && v395 < 1l);
            int v398;
            v398 = 128l * v395;
            int v399;
            v399 = v398 + v392;
            int4* v400;
            v400 = reinterpret_cast<int4*>(v1 + v399);
            int4* v401;
            v401 = reinterpret_cast<int4*>(v393 + v397);
            assert("Pointer alignment check" && (unsigned long long)(v400) % 4l == 0 && (unsigned long long)(v401) % 4l == 0);
            *v401 = *v400;
            v395 += 1l ;
        }
        int v402;
        v402 = 0l;
        while (while_method_3(v402)){
            int v404;
            v404 = 0l;
            while (while_method_1(v404)){
                bool v406;
                v406 = 0l <= v404;
                bool v408;
                if (v406){
                    bool v407;
                    v407 = v404 < 4l;
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
                bool v411;
                v411 = 0l <= v381;
                bool v413;
                if (v411){
                    bool v412;
                    v412 = v381 < 32l;
                    v413 = v412;
                } else {
                    v413 = false;
                }
                bool v414;
                v414 = v413 == false;
                if (v414){
                    assert("The indices should be inside the range of the dimension." && v413);
                } else {
                }
                int v416;
                v416 = v381 * 4l;
                int v417;
                v417 = v404 + v416;
                bool v418;
                v418 = 0l <= v402;
                bool v420;
                if (v418){
                    bool v419;
                    v419 = v402 < 1l;
                    v420 = v419;
                } else {
                    v420 = false;
                }
                bool v421;
                v421 = v420 == false;
                if (v421){
                    assert("The indices should be inside the range of the dimension." && v420);
                } else {
                }
                int v423;
                v423 = v402 * 128l;
                int v424;
                v424 = v417 + v423;
                assert("Tensor range check" && 0 <= v402 && v402 < 1l);
                assert("Tensor range check" && 0 <= v404 && v404 < 4l);
                int v425;
                v425 = 4l * v402;
                int v426;
                v426 = v425 + v404;
                v394[v426] = v424;
                v404 += 1l ;
            }
            v402 += 1l ;
        }
        bool v427;
        v427 = 0l <= v382;
        bool v428;
        v428 = v427 && v383;
        bool v429;
        v429 = v428 == false;
        if (v429){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v428);
        } else {
        }
        bool v431;
        v431 = 0l <= v389;
        bool v433;
        if (v431){
            bool v432;
            v432 = v389 < 64l;
            v433 = v432;
        } else {
            v433 = false;
        }
        bool v434;
        v434 = v433 == false;
        if (v434){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v433);
        } else {
        }
        int v436;
        v436 = v389 + v382;
        float v437[4l];
        int v438;
        v438 = 0l;
        while (while_method_3(v438)){
            int v440;
            v440 = 0l;
            while (while_method_1(v440)){
                assert("Tensor range check" && 0 <= v438 && v438 < 1l);
                assert("Tensor range check" && 0 <= v440 && v440 < 4l);
                int v442;
                v442 = 4l * v438;
                int v443;
                v443 = v442 + v440;
                float v444;
                v444 = v393[v443];
                float v445;
                v445 = v444 * v444;
                assert("Tensor range check" && 0 <= v438 && v438 < 1l);
                assert("Tensor range check" && 0 <= v440 && v440 < 4l);
                v437[v443] = v445;
                v440 += 1l ;
            }
            v438 += 1l ;
        }
        float v446;
        v446 = 0.0f;
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
                v453 = v437[v452];
                float v454;
                v454 = v446 + v453;
                v446 = v454;
                v449 += 1l ;
            }
            v447 += 1l ;
        }
        auto v455 = cooperative_groups::coalesced_threads();
        int v456;
        v456 = threadIdx.x;
        int v457;
        v457 = v456 / 32l;
        auto v458 = cooperative_groups::labeled_partition(v455,v457);
        float v459;
        v459 = cooperative_groups::reduce(v458, v446, v42);
        float v460[4l];
        int v461;
        v461 = 0l;
        while (while_method_3(v461)){
            int v463;
            v463 = 0l;
            while (while_method_1(v463)){
                assert("Tensor range check" && 0 <= v461 && v461 < 1l);
                assert("Tensor range check" && 0 <= v463 && v463 < 4l);
                int v465;
                v465 = 4l * v461;
                int v466;
                v466 = v465 + v463;
                float v467;
                v467 = v393[v466];
                bool v468;
                v468 = v459 == 0.0f;
                bool v469;
                v469 = v468 != true;
                float v471;
                if (v469){
                    float v470;
                    v470 = v467 / v459;
                    v471 = v470;
                } else {
                    v471 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v461 && v461 < 1l);
                assert("Tensor range check" && 0 <= v463 && v463 < 4l);
                v460[v466] = v471;
                v463 += 1l ;
            }
            v461 += 1l ;
        }
        assert("Tensor range check" && 0 <= v389 && v389 < 64l);
        int v472;
        v472 = 0l;
        while (while_method_3(v472)){
            assert("Tensor range check" && 0 <= v472 && v472 < 1l);
            int v474;
            v474 = 128l * v472;
            int v475;
            v475 = v474 + v392;
            assert("Tensor range check" && 0 <= v472 && v472 < 1l);
            int v476;
            v476 = 4l * v472;
            int4* v477;
            v477 = reinterpret_cast<int4*>(v460 + v476);
            int4* v478;
            v478 = reinterpret_cast<int4*>(v8 + v475);
            assert("Pointer alignment check" && (unsigned long long)(v477) % 4l == 0 && (unsigned long long)(v478) % 4l == 0);
            *v478 = *v477;
            v472 += 1l ;
        }
        v389 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v479;
    v479 = threadIdx.x;
    bool v480;
    v480 = 0l <= v479;
    bool v481;
    v481 = v480 == false;
    if (v481){
        assert("The index needs to be zero or positive." && v480);
    } else {
    }
    int v483;
    v483 = v479 % 32l;
    int v484;
    v484 = v479 / 32l;
    bool v485;
    v485 = v484 < 1l;
    bool v486;
    v486 = v485 == false;
    if (v486){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v485);
    } else {
    }
    assert("Tensor range check" && 0 <= v484 && v484 < 1l);
    assert("Tensor range check" && 0 <= v483 && v483 < 32l);
    int v488;
    v488 = 4l * v483;
    int v489;
    v489 = 128l * v484;
    int v490;
    v490 = v489 + v488;
    assert("Tensor range check" && 0 <= v484 && v484 < 1l);
    int v491;
    v491 = 0l;
    while (while_method_2(v491)){
        assert("Tensor range check" && 0 <= v491 && v491 < 64l);
        int v493;
        v493 = 128l * v491;
        int v494;
        v494 = v493 + v490;
        float v495[4l];
        int v496[4l];
        int v497;
        v497 = 0l;
        while (while_method_3(v497)){
            assert("Tensor range check" && 0 <= v497 && v497 < 1l);
            int v499;
            v499 = 4l * v497;
            assert("Tensor range check" && 0 <= v497 && v497 < 1l);
            int v500;
            v500 = 128l * v497;
            int v501;
            v501 = v500 + v494;
            int4* v502;
            v502 = reinterpret_cast<int4*>(v1 + v501);
            int4* v503;
            v503 = reinterpret_cast<int4*>(v495 + v499);
            assert("Pointer alignment check" && (unsigned long long)(v502) % 4l == 0 && (unsigned long long)(v503) % 4l == 0);
            *v503 = *v502;
            v497 += 1l ;
        }
        int v504;
        v504 = 0l;
        while (while_method_3(v504)){
            int v506;
            v506 = 0l;
            while (while_method_1(v506)){
                bool v508;
                v508 = 0l <= v506;
                bool v510;
                if (v508){
                    bool v509;
                    v509 = v506 < 4l;
                    v510 = v509;
                } else {
                    v510 = false;
                }
                bool v511;
                v511 = v510 == false;
                if (v511){
                    assert("The indices should be inside the range of the dimension." && v510);
                } else {
                }
                bool v513;
                v513 = 0l <= v483;
                bool v515;
                if (v513){
                    bool v514;
                    v514 = v483 < 32l;
                    v515 = v514;
                } else {
                    v515 = false;
                }
                bool v516;
                v516 = v515 == false;
                if (v516){
                    assert("The indices should be inside the range of the dimension." && v515);
                } else {
                }
                int v518;
                v518 = v483 * 4l;
                int v519;
                v519 = v506 + v518;
                bool v520;
                v520 = 0l <= v504;
                bool v522;
                if (v520){
                    bool v521;
                    v521 = v504 < 1l;
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
                v525 = v504 * 128l;
                int v526;
                v526 = v519 + v525;
                assert("Tensor range check" && 0 <= v504 && v504 < 1l);
                assert("Tensor range check" && 0 <= v506 && v506 < 4l);
                int v527;
                v527 = 4l * v504;
                int v528;
                v528 = v527 + v506;
                v496[v528] = v526;
                v506 += 1l ;
            }
            v504 += 1l ;
        }
        bool v529;
        v529 = 0l <= v484;
        bool v530;
        v530 = v529 && v485;
        bool v531;
        v531 = v530 == false;
        if (v531){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v530);
        } else {
        }
        bool v533;
        v533 = 0l <= v491;
        bool v535;
        if (v533){
            bool v534;
            v534 = v491 < 64l;
            v535 = v534;
        } else {
            v535 = false;
        }
        bool v536;
        v536 = v535 == false;
        if (v536){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v535);
        } else {
        }
        int v538;
        v538 = v491 + v484;
        float v539; int v540;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0l};
        v539 = tmp1.v0; v540 = tmp1.v1;
        int v541;
        v541 = 0l;
        while (while_method_3(v541)){
            int v543;
            v543 = 0l;
            while (while_method_1(v543)){
                assert("Tensor range check" && 0 <= v541 && v541 < 1l);
                assert("Tensor range check" && 0 <= v543 && v543 < 4l);
                int v545;
                v545 = 4l * v541;
                int v546;
                v546 = v545 + v543;
                float v547;
                v547 = v495[v546];
                int v548;
                v548 = v496[v546];
                bool v549;
                v549 = v539 > v547;
                float v550; int v551;
                if (v549){
                    v550 = v539; v551 = v540;
                } else {
                    v550 = v547; v551 = v548;
                }
                v539 = v550;
                v540 = v551;
                v543 += 1l ;
            }
            v541 += 1l ;
        }
        auto v552 = cooperative_groups::coalesced_threads();
        int v553;
        v553 = threadIdx.x;
        int v554;
        v554 = v553 / 32l;
        auto v555 = cooperative_groups::labeled_partition(v552,v554);
        Closure1 v556{};
        float v557; int v558;
        Tuple1 tmp2 = cooperative_groups::reduce(v555, Tuple1{v539, v540}, v556);
        v557 = tmp2.v0; v558 = tmp2.v1;
        assert("Tensor range check" && 0 <= v491 && v491 < 64l);
        v9[v538] = v558;
        v491 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v559;
    v559 = threadIdx.x;
    bool v560;
    v560 = 0l <= v559;
    bool v561;
    v561 = v560 == false;
    if (v561){
        assert("The index needs to be zero or positive." && v560);
    } else {
    }
    int v563;
    v563 = v559 % 32l;
    int v564;
    v564 = v559 / 32l;
    bool v565;
    v565 = v564 < 1l;
    bool v566;
    v566 = v565 == false;
    if (v566){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v565);
    } else {
    }
    assert("Tensor range check" && 0 <= v564 && v564 < 1l);
    assert("Tensor range check" && 0 <= v563 && v563 < 32l);
    int v568;
    v568 = 4l * v563;
    int v569;
    v569 = 128l * v564;
    int v570;
    v570 = v569 + v568;
    assert("Tensor range check" && 0 <= v564 && v564 < 1l);
    assert("Tensor range check" && 0 <= v563 && v563 < 32l);
    int v571;
    v571 = 0l;
    while (while_method_2(v571)){
        assert("Tensor range check" && 0 <= v571 && v571 < 64l);
        int v573;
        v573 = 128l * v571;
        int v574;
        v574 = v573 + v570;
        float v575[4l];
        int v576[4l];
        int v577;
        v577 = 0l;
        while (while_method_3(v577)){
            assert("Tensor range check" && 0 <= v577 && v577 < 1l);
            int v579;
            v579 = 4l * v577;
            assert("Tensor range check" && 0 <= v577 && v577 < 1l);
            int v580;
            v580 = 128l * v577;
            int v581;
            v581 = v580 + v574;
            int4* v582;
            v582 = reinterpret_cast<int4*>(v1 + v581);
            int4* v583;
            v583 = reinterpret_cast<int4*>(v575 + v579);
            assert("Pointer alignment check" && (unsigned long long)(v582) % 4l == 0 && (unsigned long long)(v583) % 4l == 0);
            *v583 = *v582;
            v577 += 1l ;
        }
        int v584;
        v584 = 0l;
        while (while_method_3(v584)){
            int v586;
            v586 = 0l;
            while (while_method_1(v586)){
                bool v588;
                v588 = 0l <= v586;
                bool v590;
                if (v588){
                    bool v589;
                    v589 = v586 < 4l;
                    v590 = v589;
                } else {
                    v590 = false;
                }
                bool v591;
                v591 = v590 == false;
                if (v591){
                    assert("The indices should be inside the range of the dimension." && v590);
                } else {
                }
                bool v593;
                v593 = 0l <= v563;
                bool v595;
                if (v593){
                    bool v594;
                    v594 = v563 < 32l;
                    v595 = v594;
                } else {
                    v595 = false;
                }
                bool v596;
                v596 = v595 == false;
                if (v596){
                    assert("The indices should be inside the range of the dimension." && v595);
                } else {
                }
                int v598;
                v598 = v563 * 4l;
                int v599;
                v599 = v586 + v598;
                bool v600;
                v600 = 0l <= v584;
                bool v602;
                if (v600){
                    bool v601;
                    v601 = v584 < 1l;
                    v602 = v601;
                } else {
                    v602 = false;
                }
                bool v603;
                v603 = v602 == false;
                if (v603){
                    assert("The indices should be inside the range of the dimension." && v602);
                } else {
                }
                int v605;
                v605 = v584 * 128l;
                int v606;
                v606 = v599 + v605;
                assert("Tensor range check" && 0 <= v584 && v584 < 1l);
                assert("Tensor range check" && 0 <= v586 && v586 < 4l);
                int v607;
                v607 = 4l * v584;
                int v608;
                v608 = v607 + v586;
                v576[v608] = v606;
                v586 += 1l ;
            }
            v584 += 1l ;
        }
        bool v609;
        v609 = 0l <= v564;
        bool v610;
        v610 = v609 && v565;
        bool v611;
        v611 = v610 == false;
        if (v611){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v610);
        } else {
        }
        bool v613;
        v613 = 0l <= v571;
        bool v615;
        if (v613){
            bool v614;
            v614 = v571 < 64l;
            v615 = v614;
        } else {
            v615 = false;
        }
        bool v616;
        v616 = v615 == false;
        if (v616){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v615);
        } else {
        }
        int v618;
        v618 = v571 + v564;
        float v619;
        v619 = 0.0f;
        int v620;
        v620 = 0l;
        while (while_method_3(v620)){
            int v622;
            v622 = 0l;
            while (while_method_1(v622)){
                assert("Tensor range check" && 0 <= v620 && v620 < 1l);
                assert("Tensor range check" && 0 <= v622 && v622 < 4l);
                int v624;
                v624 = 4l * v620;
                int v625;
                v625 = v624 + v622;
                float v626;
                v626 = v575[v625];
                float v627;
                v627 = v619 + v626;
                v619 = v627;
                v622 += 1l ;
            }
            v620 += 1l ;
        }
        auto v628 = cooperative_groups::coalesced_threads();
        int v629;
        v629 = threadIdx.x;
        int v630;
        v630 = v629 / 32l;
        auto v631 = cooperative_groups::labeled_partition(v628,v630);
        float v632;
        v632 = cooperative_groups::reduce(v631, v619, v42);
        float v633;
        v633 = v632 / 128.0f;
        float v634[4l];
        int v635;
        v635 = 0l;
        while (while_method_3(v635)){
            int v637;
            v637 = 0l;
            while (while_method_1(v637)){
                assert("Tensor range check" && 0 <= v635 && v635 < 1l);
                assert("Tensor range check" && 0 <= v637 && v637 < 4l);
                int v639;
                v639 = 4l * v635;
                int v640;
                v640 = v639 + v637;
                float v641;
                v641 = v575[v640];
                float v642;
                v642 = v641 - v633;
                float v643;
                v643 = exp(v642);
                assert("Tensor range check" && 0 <= v635 && v635 < 1l);
                assert("Tensor range check" && 0 <= v637 && v637 < 4l);
                v634[v640] = v643;
                v637 += 1l ;
            }
            v635 += 1l ;
        }
        float v644;
        v644 = 0.0f;
        int v645;
        v645 = 0l;
        while (while_method_3(v645)){
            int v647;
            v647 = 0l;
            while (while_method_1(v647)){
                assert("Tensor range check" && 0 <= v645 && v645 < 1l);
                assert("Tensor range check" && 0 <= v647 && v647 < 4l);
                int v649;
                v649 = 4l * v645;
                int v650;
                v650 = v649 + v647;
                float v651;
                v651 = v634[v650];
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
        v657 = cooperative_groups::reduce(v656, v644, v42);
        float v658[4l];
        int v659;
        v659 = 0l;
        while (while_method_3(v659)){
            int v661;
            v661 = 0l;
            while (while_method_1(v661)){
                assert("Tensor range check" && 0 <= v659 && v659 < 1l);
                assert("Tensor range check" && 0 <= v661 && v661 < 4l);
                int v663;
                v663 = 4l * v659;
                int v664;
                v664 = v663 + v661;
                float v665;
                v665 = v634[v664];
                float v666;
                v666 = v665 / v657;
                assert("Tensor range check" && 0 <= v659 && v659 < 1l);
                assert("Tensor range check" && 0 <= v661 && v661 < 4l);
                v658[v664] = v666;
                v661 += 1l ;
            }
            v659 += 1l ;
        }
        float v667[4l];
        float v668;
        v668 = 0.0f;
        int v669;
        v669 = 0l;
        while (while_method_3(v669)){
            assert("Tensor range check" && 0 <= v669 && v669 < 1l);
            int v671;
            v671 = 4l * v669;
            assert("Tensor range check" && 0 <= v669 && v669 < 1l);
            int v672; float v673;
            Tuple0 tmp3 = Tuple0{0l, 0.0f};
            v672 = tmp3.v0; v673 = tmp3.v1;
            while (while_method_1(v672)){
                assert("Tensor range check" && 0 <= v672 && v672 < 4l);
                int v675;
                v675 = v672 + v671;
                float v676;
                v676 = v658[v675];
                float v677;
                v677 = v673 + v676;
                v673 = v677;
                v672 += 1l ;
            }
            auto v678 = cooperative_groups::coalesced_threads();
            int v679;
            v679 = threadIdx.x;
            int v680;
            v680 = v679 / 32l;
            auto v681 = cooperative_groups::labeled_partition(v678,v680);
            Closure2 v682{};
            float v683;
            v683 = cooperative_groups::inclusive_scan(v681, v673, v682);
            float v684;
            v684 = v681.shfl_up(v683,1);
            bool v685;
            v685 = v681.thread_rank() == 0;
            float v686;
            if (v685){
                v686 = 0.0f;
            } else {
                v686 = v684;
            }
            float v687;
            v687 = v681.shfl(v683,v681.num_threads()-1);
            float v688;
            v688 = v668 + v686;
            int v689; float v690;
            Tuple0 tmp4 = Tuple0{0l, v688};
            v689 = tmp4.v0; v690 = tmp4.v1;
            while (while_method_1(v689)){
                assert("Tensor range check" && 0 <= v689 && v689 < 4l);
                int v692;
                v692 = v689 + v671;
                float v693;
                v693 = v658[v692];
                float v694;
                v694 = v690 + v693;
                assert("Tensor range check" && 0 <= v689 && v689 < 4l);
                v667[v692] = v694;
                v690 = v694;
                v689 += 1l ;
            }
            float v695;
            v695 = v668 + v687;
            v668 = v695;
            v669 += 1l ;
        }
        assert("Tensor range check" && 0 <= v571 && v571 < 64l);
        int v696;
        v696 = 0l;
        while (while_method_3(v696)){
            assert("Tensor range check" && 0 <= v696 && v696 < 1l);
            int v698;
            v698 = 128l * v696;
            int v699;
            v699 = v698 + v574;
            assert("Tensor range check" && 0 <= v696 && v696 < 1l);
            int v700;
            v700 = 4l * v696;
            int4* v701;
            v701 = reinterpret_cast<int4*>(v658 + v700);
            int4* v702;
            v702 = reinterpret_cast<int4*>(v6 + v699);
            assert("Pointer alignment check" && (unsigned long long)(v701) % 4l == 0 && (unsigned long long)(v702) % 4l == 0);
            *v702 = *v701;
            int4* v703;
            v703 = reinterpret_cast<int4*>(v667 + v700);
            int4* v704;
            v704 = reinterpret_cast<int4*>(v7 + v699);
            assert("Pointer alignment check" && (unsigned long long)(v703) % 4l == 0 && (unsigned long long)(v704) % 4l == 0);
            *v704 = *v703;
            v696 += 1l ;
        }
        v571 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v705;
    v705 = threadIdx.x;
    bool v706;
    v706 = 0l <= v705;
    bool v707;
    v707 = v706 == false;
    if (v707){
        assert("The index needs to be zero or positive." && v706);
    } else {
    }
    int v709;
    v709 = v705 % 32l;
    int v710;
    v710 = v705 / 32l;
    bool v711;
    v711 = v710 < 1l;
    bool v712;
    v712 = v711 == false;
    if (v712){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v711);
    } else {
    }
    assert("Tensor range check" && 0 <= v710 && v710 < 1l);
    assert("Tensor range check" && 0 <= v709 && v709 < 32l);
    int v714;
    v714 = 4l * v709;
    int v715;
    v715 = 128l * v710;
    int v716;
    v716 = v715 + v714;
    assert("Tensor range check" && 0 <= v710 && v710 < 1l);
    assert("Tensor range check" && 0 <= v709 && v709 < 32l);
    int v717;
    v717 = 0l;
    while (while_method_2(v717)){
        assert("Tensor range check" && 0 <= v717 && v717 < 64l);
        int v719;
        v719 = 128l * v717;
        int v720;
        v720 = v719 + v716;
        int v721[4l];
        int v722[4l];
        int v723;
        v723 = 0l;
        while (while_method_3(v723)){
            assert("Tensor range check" && 0 <= v723 && v723 < 1l);
            int v725;
            v725 = 4l * v723;
            assert("Tensor range check" && 0 <= v723 && v723 < 1l);
            int v726;
            v726 = 128l * v723;
            int v727;
            v727 = v726 + v720;
            int4* v728;
            v728 = reinterpret_cast<int4*>(v0 + v727);
            int4* v729;
            v729 = reinterpret_cast<int4*>(v721 + v725);
            assert("Pointer alignment check" && (unsigned long long)(v728) % 4l == 0 && (unsigned long long)(v729) % 4l == 0);
            *v729 = *v728;
            v723 += 1l ;
        }
        int v730;
        v730 = 0l;
        while (while_method_3(v730)){
            int v732;
            v732 = 0l;
            while (while_method_1(v732)){
                bool v734;
                v734 = 0l <= v732;
                bool v736;
                if (v734){
                    bool v735;
                    v735 = v732 < 4l;
                    v736 = v735;
                } else {
                    v736 = false;
                }
                bool v737;
                v737 = v736 == false;
                if (v737){
                    assert("The indices should be inside the range of the dimension." && v736);
                } else {
                }
                bool v739;
                v739 = 0l <= v709;
                bool v741;
                if (v739){
                    bool v740;
                    v740 = v709 < 32l;
                    v741 = v740;
                } else {
                    v741 = false;
                }
                bool v742;
                v742 = v741 == false;
                if (v742){
                    assert("The indices should be inside the range of the dimension." && v741);
                } else {
                }
                int v744;
                v744 = v709 * 4l;
                int v745;
                v745 = v732 + v744;
                bool v746;
                v746 = 0l <= v730;
                bool v748;
                if (v746){
                    bool v747;
                    v747 = v730 < 1l;
                    v748 = v747;
                } else {
                    v748 = false;
                }
                bool v749;
                v749 = v748 == false;
                if (v749){
                    assert("The indices should be inside the range of the dimension." && v748);
                } else {
                }
                int v751;
                v751 = v730 * 128l;
                int v752;
                v752 = v745 + v751;
                assert("Tensor range check" && 0 <= v730 && v730 < 1l);
                assert("Tensor range check" && 0 <= v732 && v732 < 4l);
                int v753;
                v753 = 4l * v730;
                int v754;
                v754 = v753 + v732;
                v722[v754] = v752;
                v732 += 1l ;
            }
            v730 += 1l ;
        }
        bool v755;
        v755 = 0l <= v710;
        bool v756;
        v756 = v755 && v711;
        bool v757;
        v757 = v756 == false;
        if (v757){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v756);
        } else {
        }
        bool v759;
        v759 = 0l <= v717;
        bool v761;
        if (v759){
            bool v760;
            v760 = v717 < 64l;
            v761 = v760;
        } else {
            v761 = false;
        }
        bool v762;
        v762 = v761 == false;
        if (v762){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v761);
        } else {
        }
        int v764;
        v764 = v717 + v710;
        int v765[4l];
        int v766;
        v766 = 0l;
        int v767;
        v767 = 0l;
        while (while_method_3(v767)){
            assert("Tensor range check" && 0 <= v767 && v767 < 1l);
            int v769;
            v769 = 4l * v767;
            assert("Tensor range check" && 0 <= v767 && v767 < 1l);
            int v770; int v771;
            Tuple2 tmp5 = Tuple2{0l, 0l};
            v770 = tmp5.v0; v771 = tmp5.v1;
            while (while_method_1(v770)){
                assert("Tensor range check" && 0 <= v770 && v770 < 4l);
                int v773;
                v773 = v770 + v769;
                int v774;
                v774 = v721[v773];
                int v775;
                v775 = v771 + v774;
                v771 = v775;
                v770 += 1l ;
            }
            auto v776 = cooperative_groups::coalesced_threads();
            int v777;
            v777 = threadIdx.x;
            int v778;
            v778 = v777 / 32l;
            auto v779 = cooperative_groups::labeled_partition(v776,v778);
            Closure3 v780{};
            int v781;
            v781 = cooperative_groups::inclusive_scan(v779, v771, v780);
            int v782;
            v782 = v779.shfl_up(v781,1);
            bool v783;
            v783 = v779.thread_rank() == 0;
            int v784;
            if (v783){
                v784 = 0l;
            } else {
                v784 = v782;
            }
            int v785;
            v785 = v779.shfl(v781,v779.num_threads()-1);
            int v786;
            v786 = v766 + v784;
            int v787; int v788;
            Tuple2 tmp6 = Tuple2{0l, v786};
            v787 = tmp6.v0; v788 = tmp6.v1;
            while (while_method_1(v787)){
                assert("Tensor range check" && 0 <= v787 && v787 < 4l);
                int v790;
                v790 = v787 + v769;
                int v791;
                v791 = v721[v790];
                assert("Tensor range check" && 0 <= v787 && v787 < 4l);
                v765[v790] = v788;
                int v792;
                v792 = v788 + v791;
                v788 = v792;
                v787 += 1l ;
            }
            int v793;
            v793 = v766 + v785;
            v766 = v793;
            v767 += 1l ;
        }
        assert("Tensor range check" && 0 <= v717 && v717 < 64l);
        int v794;
        v794 = 0l;
        while (while_method_3(v794)){
            assert("Tensor range check" && 0 <= v794 && v794 < 1l);
            int v796;
            v796 = 128l * v794;
            int v797;
            v797 = v796 + v720;
            assert("Tensor range check" && 0 <= v794 && v794 < 1l);
            int v798;
            v798 = 4l * v794;
            int4* v799;
            v799 = reinterpret_cast<int4*>(v765 + v798);
            int4* v800;
            v800 = reinterpret_cast<int4*>(v13 + v797);
            assert("Pointer alignment check" && (unsigned long long)(v799) % 4l == 0 && (unsigned long long)(v800) % 4l == 0);
            *v800 = *v799;
            v794 += 1l ;
        }
        v717 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v801;
    v801 = threadIdx.x;
    bool v802;
    v802 = 0l <= v801;
    bool v803;
    v803 = v802 == false;
    if (v803){
        assert("The index needs to be zero or positive." && v802);
    } else {
    }
    int v805;
    v805 = v801 % 32l;
    int v806;
    v806 = v801 / 32l;
    bool v807;
    v807 = v806 < 1l;
    bool v808;
    v808 = v807 == false;
    if (v808){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v807);
    } else {
    }
    assert("Tensor range check" && 0 <= v806 && v806 < 1l);
    assert("Tensor range check" && 0 <= v805 && v805 < 32l);
    int v810;
    v810 = 4l * v805;
    int v811;
    v811 = 128l * v806;
    int v812;
    v812 = v811 + v810;
    assert("Tensor range check" && 0 <= v806 && v806 < 1l);
    assert("Tensor range check" && 0 <= v805 && v805 < 32l);
    int v813;
    v813 = 0l;
    while (while_method_2(v813)){
        assert("Tensor range check" && 0 <= v813 && v813 < 64l);
        int v815;
        v815 = 128l * v813;
        int v816;
        v816 = v815 + v812;
        float v817[4l];
        int v818[4l];
        int v819;
        v819 = 0l;
        while (while_method_3(v819)){
            assert("Tensor range check" && 0 <= v819 && v819 < 1l);
            int v821;
            v821 = 4l * v819;
            assert("Tensor range check" && 0 <= v819 && v819 < 1l);
            int v822;
            v822 = 128l * v819;
            int v823;
            v823 = v822 + v816;
            int4* v824;
            v824 = reinterpret_cast<int4*>(v1 + v823);
            int4* v825;
            v825 = reinterpret_cast<int4*>(v817 + v821);
            assert("Pointer alignment check" && (unsigned long long)(v824) % 4l == 0 && (unsigned long long)(v825) % 4l == 0);
            *v825 = *v824;
            v819 += 1l ;
        }
        int v826;
        v826 = 0l;
        while (while_method_3(v826)){
            int v828;
            v828 = 0l;
            while (while_method_1(v828)){
                bool v830;
                v830 = 0l <= v828;
                bool v832;
                if (v830){
                    bool v831;
                    v831 = v828 < 4l;
                    v832 = v831;
                } else {
                    v832 = false;
                }
                bool v833;
                v833 = v832 == false;
                if (v833){
                    assert("The indices should be inside the range of the dimension." && v832);
                } else {
                }
                bool v835;
                v835 = 0l <= v805;
                bool v837;
                if (v835){
                    bool v836;
                    v836 = v805 < 32l;
                    v837 = v836;
                } else {
                    v837 = false;
                }
                bool v838;
                v838 = v837 == false;
                if (v838){
                    assert("The indices should be inside the range of the dimension." && v837);
                } else {
                }
                int v840;
                v840 = v805 * 4l;
                int v841;
                v841 = v828 + v840;
                bool v842;
                v842 = 0l <= v826;
                bool v844;
                if (v842){
                    bool v843;
                    v843 = v826 < 1l;
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
                int v847;
                v847 = v826 * 128l;
                int v848;
                v848 = v841 + v847;
                assert("Tensor range check" && 0 <= v826 && v826 < 1l);
                assert("Tensor range check" && 0 <= v828 && v828 < 4l);
                int v849;
                v849 = 4l * v826;
                int v850;
                v850 = v849 + v828;
                v818[v850] = v848;
                v828 += 1l ;
            }
            v826 += 1l ;
        }
        bool v851;
        v851 = 0l <= v806;
        bool v852;
        v852 = v851 && v807;
        bool v853;
        v853 = v852 == false;
        if (v853){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v852);
        } else {
        }
        bool v855;
        v855 = 0l <= v813;
        bool v857;
        if (v855){
            bool v856;
            v856 = v813 < 64l;
            v857 = v856;
        } else {
            v857 = false;
        }
        bool v858;
        v858 = v857 == false;
        if (v858){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v857);
        } else {
        }
        int v860;
        v860 = v813 + v806;
        bool v861[4l];
        int v862;
        v862 = 0l;
        while (while_method_3(v862)){
            int v864;
            v864 = 0l;
            while (while_method_1(v864)){
                assert("Tensor range check" && 0 <= v862 && v862 < 1l);
                assert("Tensor range check" && 0 <= v864 && v864 < 4l);
                int v866;
                v866 = 4l * v862;
                int v867;
                v867 = v866 + v864;
                float v868;
                v868 = v817[v867];
                int v869;
                v869 = v818[v867];
                bool v870;
                v870 = v869 < 4l;
                assert("Tensor range check" && 0 <= v862 && v862 < 1l);
                assert("Tensor range check" && 0 <= v864 && v864 < 4l);
                v861[v867] = v870;
                v864 += 1l ;
            }
            v862 += 1l ;
        }
        int v871[4l];
        int v872;
        v872 = 0l;
        while (while_method_3(v872)){
            int v874;
            v874 = 0l;
            while (while_method_1(v874)){
                assert("Tensor range check" && 0 <= v872 && v872 < 1l);
                assert("Tensor range check" && 0 <= v874 && v874 < 4l);
                int v876;
                v876 = 4l * v872;
                int v877;
                v877 = v876 + v874;
                bool v878;
                v878 = v861[v877];
                int v879;
                if (v878){
                    v879 = 1l;
                } else {
                    v879 = 0l;
                }
                assert("Tensor range check" && 0 <= v872 && v872 < 1l);
                assert("Tensor range check" && 0 <= v874 && v874 < 4l);
                v871[v877] = v879;
                v874 += 1l ;
            }
            v872 += 1l ;
        }
        int v880;
        v880 = 0l;
        int v881;
        v881 = 0l;
        while (while_method_3(v881)){
            int v883;
            v883 = 0l;
            while (while_method_1(v883)){
                assert("Tensor range check" && 0 <= v881 && v881 < 1l);
                assert("Tensor range check" && 0 <= v883 && v883 < 4l);
                int v885;
                v885 = 4l * v881;
                int v886;
                v886 = v885 + v883;
                int v887;
                v887 = v871[v886];
                int v888;
                v888 = v880 + v887;
                v880 = v888;
                v883 += 1l ;
            }
            v881 += 1l ;
        }
        auto v889 = cooperative_groups::coalesced_threads();
        int v890;
        v890 = threadIdx.x;
        int v891;
        v891 = v890 / 32l;
        auto v892 = cooperative_groups::labeled_partition(v889,v891);
        Closure4 v893{};
        int v894;
        v894 = cooperative_groups::reduce(v892, v880, v893);
        float v895[4l];
        int v896;
        v896 = 0l;
        while (while_method_3(v896)){
            int v898;
            v898 = 0l;
            while (while_method_1(v898)){
                assert("Tensor range check" && 0 <= v896 && v896 < 1l);
                assert("Tensor range check" && 0 <= v898 && v898 < 4l);
                int v900;
                v900 = 4l * v896;
                int v901;
                v901 = v900 + v898;
                float v902;
                v902 = v817[v901];
                bool v903;
                v903 = v861[v901];
                float v904;
                if (v903){
                    v904 = v902;
                } else {
                    v904 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v896 && v896 < 1l);
                assert("Tensor range check" && 0 <= v898 && v898 < 4l);
                v895[v901] = v904;
                v898 += 1l ;
            }
            v896 += 1l ;
        }
        float v905;
        v905 = 0.0f;
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
                v912 = v895[v911];
                float v913;
                v913 = v905 + v912;
                v905 = v913;
                v908 += 1l ;
            }
            v906 += 1l ;
        }
        auto v914 = cooperative_groups::coalesced_threads();
        int v915;
        v915 = threadIdx.x;
        int v916;
        v916 = v915 / 32l;
        auto v917 = cooperative_groups::labeled_partition(v914,v916);
        float v918;
        v918 = cooperative_groups::reduce(v917, v905, v42);
        float v919;
        v919 = (float)v894;
        float v920;
        v920 = v918 / v919;
        float v921[4l];
        int v922;
        v922 = 0l;
        while (while_method_3(v922)){
            int v924;
            v924 = 0l;
            while (while_method_1(v924)){
                assert("Tensor range check" && 0 <= v922 && v922 < 1l);
                assert("Tensor range check" && 0 <= v924 && v924 < 4l);
                int v926;
                v926 = 4l * v922;
                int v927;
                v927 = v926 + v924;
                float v928;
                v928 = v817[v927];
                bool v929;
                v929 = v861[v927];
                float v930;
                if (v929){
                    v930 = v928;
                } else {
                    v930 = -1.0f / 0.0f;
                }
                float v931;
                v931 = v930 - v920;
                float v932;
                v932 = exp(v931);
                assert("Tensor range check" && 0 <= v922 && v922 < 1l);
                assert("Tensor range check" && 0 <= v924 && v924 < 4l);
                v921[v927] = v932;
                v924 += 1l ;
            }
            v922 += 1l ;
        }
        float v933;
        v933 = 0.0f;
        int v934;
        v934 = 0l;
        while (while_method_3(v934)){
            int v936;
            v936 = 0l;
            while (while_method_1(v936)){
                assert("Tensor range check" && 0 <= v934 && v934 < 1l);
                assert("Tensor range check" && 0 <= v936 && v936 < 4l);
                int v938;
                v938 = 4l * v934;
                int v939;
                v939 = v938 + v936;
                float v940;
                v940 = v921[v939];
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
        v946 = cooperative_groups::reduce(v945, v933, v42);
        float v947[4l];
        int v948;
        v948 = 0l;
        while (while_method_3(v948)){
            int v950;
            v950 = 0l;
            while (while_method_1(v950)){
                assert("Tensor range check" && 0 <= v948 && v948 < 1l);
                assert("Tensor range check" && 0 <= v950 && v950 < 4l);
                int v952;
                v952 = 4l * v948;
                int v953;
                v953 = v952 + v950;
                float v954;
                v954 = v921[v953];
                float v955;
                v955 = v954 / v946;
                assert("Tensor range check" && 0 <= v948 && v948 < 1l);
                assert("Tensor range check" && 0 <= v950 && v950 < 4l);
                v947[v953] = v955;
                v950 += 1l ;
            }
            v948 += 1l ;
        }
        assert("Tensor range check" && 0 <= v813 && v813 < 64l);
        int v956;
        v956 = 0l;
        while (while_method_3(v956)){
            assert("Tensor range check" && 0 <= v956 && v956 < 1l);
            int v958;
            v958 = 128l * v956;
            int v959;
            v959 = v958 + v816;
            assert("Tensor range check" && 0 <= v956 && v956 < 1l);
            int v960;
            v960 = 4l * v956;
            int4* v961;
            v961 = reinterpret_cast<int4*>(v947 + v960);
            int4* v962;
            v962 = reinterpret_cast<int4*>(v5 + v959);
            assert("Pointer alignment check" && (unsigned long long)(v961) % 4l == 0 && (unsigned long long)(v962) % 4l == 0);
            *v962 = *v961;
            v956 += 1l ;
        }
        v813 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v963;
    v963 = threadIdx.x;
    int v964;
    v964 = blockIdx.x;
    int v965;
    v965 = v964 * 32l;
    int v966;
    v966 = v963 + v965;
    unsigned long long v967;
    v967 = (unsigned long long)v966;
    curandStatePhilox4_32_10_t v968;
    curand_init(12344321ull,v967,0ull,&v968);
    int v969;
    v969 = threadIdx.x;
    bool v970;
    v970 = 0l <= v969;
    bool v971;
    v971 = v970 == false;
    if (v971){
        assert("The index needs to be zero or positive." && v970);
    } else {
    }
    int v973;
    v973 = v969 % 32l;
    int v974;
    v974 = v969 / 32l;
    bool v975;
    v975 = v974 < 1l;
    bool v976;
    v976 = v975 == false;
    if (v976){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v975);
    } else {
    }
    assert("Tensor range check" && 0 <= v974 && v974 < 1l);
    assert("Tensor range check" && 0 <= v973 && v973 < 32l);
    int v978;
    v978 = 4l * v973;
    int v979;
    v979 = 128l * v974;
    int v980;
    v980 = v979 + v978;
    assert("Tensor range check" && 0 <= v974 && v974 < 1l);
    assert("Tensor range check" && 0 <= v973 && v973 < 32l);
    assert("Tensor range check" && 0 <= v974 && v974 < 1l);
    int v981;
    v981 = 0l;
    while (while_method_2(v981)){
        assert("Tensor range check" && 0 <= v981 && v981 < 64l);
        int v983;
        v983 = 128l * v981;
        int v984;
        v984 = v983 + v980;
        float v985[4l];
        int v986[4l];
        int v987;
        v987 = 0l;
        while (while_method_3(v987)){
            assert("Tensor range check" && 0 <= v987 && v987 < 1l);
            int v989;
            v989 = 4l * v987;
            assert("Tensor range check" && 0 <= v987 && v987 < 1l);
            int v990;
            v990 = 128l * v987;
            int v991;
            v991 = v990 + v984;
            int4* v992;
            v992 = reinterpret_cast<int4*>(v1 + v991);
            int4* v993;
            v993 = reinterpret_cast<int4*>(v985 + v989);
            assert("Pointer alignment check" && (unsigned long long)(v992) % 4l == 0 && (unsigned long long)(v993) % 4l == 0);
            *v993 = *v992;
            v987 += 1l ;
        }
        int v994;
        v994 = 0l;
        while (while_method_3(v994)){
            int v996;
            v996 = 0l;
            while (while_method_1(v996)){
                bool v998;
                v998 = 0l <= v996;
                bool v1000;
                if (v998){
                    bool v999;
                    v999 = v996 < 4l;
                    v1000 = v999;
                } else {
                    v1000 = false;
                }
                bool v1001;
                v1001 = v1000 == false;
                if (v1001){
                    assert("The indices should be inside the range of the dimension." && v1000);
                } else {
                }
                bool v1003;
                v1003 = 0l <= v973;
                bool v1005;
                if (v1003){
                    bool v1004;
                    v1004 = v973 < 32l;
                    v1005 = v1004;
                } else {
                    v1005 = false;
                }
                bool v1006;
                v1006 = v1005 == false;
                if (v1006){
                    assert("The indices should be inside the range of the dimension." && v1005);
                } else {
                }
                int v1008;
                v1008 = v973 * 4l;
                int v1009;
                v1009 = v996 + v1008;
                bool v1010;
                v1010 = 0l <= v994;
                bool v1012;
                if (v1010){
                    bool v1011;
                    v1011 = v994 < 1l;
                    v1012 = v1011;
                } else {
                    v1012 = false;
                }
                bool v1013;
                v1013 = v1012 == false;
                if (v1013){
                    assert("The indices should be inside the range of the dimension." && v1012);
                } else {
                }
                int v1015;
                v1015 = v994 * 128l;
                int v1016;
                v1016 = v1009 + v1015;
                assert("Tensor range check" && 0 <= v994 && v994 < 1l);
                assert("Tensor range check" && 0 <= v996 && v996 < 4l);
                int v1017;
                v1017 = 4l * v994;
                int v1018;
                v1018 = v1017 + v996;
                v986[v1018] = v1016;
                v996 += 1l ;
            }
            v994 += 1l ;
        }
        bool v1019;
        v1019 = 0l <= v974;
        bool v1020;
        v1020 = v1019 && v975;
        bool v1021;
        v1021 = v1020 == false;
        if (v1021){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1020);
        } else {
        }
        bool v1023;
        v1023 = 0l <= v981;
        bool v1025;
        if (v1023){
            bool v1024;
            v1024 = v981 < 64l;
            v1025 = v1024;
        } else {
            v1025 = false;
        }
        bool v1026;
        v1026 = v1025 == false;
        if (v1026){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1025);
        } else {
        }
        int v1028;
        v1028 = v981 + v974;
        float v1029;
        v1029 = 0.0f;
        int v1030;
        v1030 = 0l;
        while (while_method_3(v1030)){
            int v1032;
            v1032 = 0l;
            while (while_method_1(v1032)){
                assert("Tensor range check" && 0 <= v1030 && v1030 < 1l);
                assert("Tensor range check" && 0 <= v1032 && v1032 < 4l);
                int v1034;
                v1034 = 4l * v1030;
                int v1035;
                v1035 = v1034 + v1032;
                float v1036;
                v1036 = v985[v1035];
                float v1037;
                v1037 = v1029 + v1036;
                v1029 = v1037;
                v1032 += 1l ;
            }
            v1030 += 1l ;
        }
        auto v1038 = cooperative_groups::coalesced_threads();
        int v1039;
        v1039 = threadIdx.x;
        int v1040;
        v1040 = v1039 / 32l;
        auto v1041 = cooperative_groups::labeled_partition(v1038,v1040);
        float v1042;
        v1042 = cooperative_groups::reduce(v1041, v1029, v42);
        float v1043;
        v1043 = v1042 / 128.0f;
        float v1044[4l];
        int v1045;
        v1045 = 0l;
        while (while_method_3(v1045)){
            int v1047;
            v1047 = 0l;
            while (while_method_1(v1047)){
                assert("Tensor range check" && 0 <= v1045 && v1045 < 1l);
                assert("Tensor range check" && 0 <= v1047 && v1047 < 4l);
                int v1049;
                v1049 = 4l * v1045;
                int v1050;
                v1050 = v1049 + v1047;
                float v1051;
                v1051 = v985[v1050];
                float v1052;
                v1052 = v1051 - v1043;
                float v1053;
                v1053 = exp(v1052);
                assert("Tensor range check" && 0 <= v1045 && v1045 < 1l);
                assert("Tensor range check" && 0 <= v1047 && v1047 < 4l);
                v1044[v1050] = v1053;
                v1047 += 1l ;
            }
            v1045 += 1l ;
        }
        float v1054;
        v1054 = 0.0f;
        int v1055;
        v1055 = 0l;
        while (while_method_3(v1055)){
            int v1057;
            v1057 = 0l;
            while (while_method_1(v1057)){
                assert("Tensor range check" && 0 <= v1055 && v1055 < 1l);
                assert("Tensor range check" && 0 <= v1057 && v1057 < 4l);
                int v1059;
                v1059 = 4l * v1055;
                int v1060;
                v1060 = v1059 + v1057;
                float v1061;
                v1061 = v1044[v1060];
                float v1062;
                v1062 = v1054 + v1061;
                v1054 = v1062;
                v1057 += 1l ;
            }
            v1055 += 1l ;
        }
        auto v1063 = cooperative_groups::coalesced_threads();
        int v1064;
        v1064 = threadIdx.x;
        int v1065;
        v1065 = v1064 / 32l;
        auto v1066 = cooperative_groups::labeled_partition(v1063,v1065);
        float v1067;
        v1067 = cooperative_groups::reduce(v1066, v1054, v42);
        float v1068[4l];
        int v1069;
        v1069 = 0l;
        while (while_method_3(v1069)){
            int v1071;
            v1071 = 0l;
            while (while_method_1(v1071)){
                assert("Tensor range check" && 0 <= v1069 && v1069 < 1l);
                assert("Tensor range check" && 0 <= v1071 && v1071 < 4l);
                int v1073;
                v1073 = 4l * v1069;
                int v1074;
                v1074 = v1073 + v1071;
                float v1075;
                v1075 = v1044[v1074];
                float v1076;
                v1076 = v1075 / v1067;
                assert("Tensor range check" && 0 <= v1069 && v1069 < 1l);
                assert("Tensor range check" && 0 <= v1071 && v1071 < 4l);
                v1068[v1074] = v1076;
                v1071 += 1l ;
            }
            v1069 += 1l ;
        }
        float v1077[4l];
        float v1078;
        v1078 = 0.0f;
        int v1079;
        v1079 = 0l;
        while (while_method_3(v1079)){
            assert("Tensor range check" && 0 <= v1079 && v1079 < 1l);
            int v1081;
            v1081 = 4l * v1079;
            assert("Tensor range check" && 0 <= v1079 && v1079 < 1l);
            int v1082; float v1083;
            Tuple0 tmp7 = Tuple0{0l, 0.0f};
            v1082 = tmp7.v0; v1083 = tmp7.v1;
            while (while_method_1(v1082)){
                assert("Tensor range check" && 0 <= v1082 && v1082 < 4l);
                int v1085;
                v1085 = v1082 + v1081;
                float v1086;
                v1086 = v1068[v1085];
                float v1087;
                v1087 = v1083 + v1086;
                v1083 = v1087;
                v1082 += 1l ;
            }
            auto v1088 = cooperative_groups::coalesced_threads();
            int v1089;
            v1089 = threadIdx.x;
            int v1090;
            v1090 = v1089 / 32l;
            auto v1091 = cooperative_groups::labeled_partition(v1088,v1090);
            Closure2 v1092{};
            float v1093;
            v1093 = cooperative_groups::inclusive_scan(v1091, v1083, v1092);
            float v1094;
            v1094 = v1091.shfl_up(v1093,1);
            bool v1095;
            v1095 = v1091.thread_rank() == 0;
            float v1096;
            if (v1095){
                v1096 = 0.0f;
            } else {
                v1096 = v1094;
            }
            float v1097;
            v1097 = v1091.shfl(v1093,v1091.num_threads()-1);
            float v1098;
            v1098 = v1078 + v1096;
            int v1099; float v1100;
            Tuple0 tmp8 = Tuple0{0l, v1098};
            v1099 = tmp8.v0; v1100 = tmp8.v1;
            while (while_method_1(v1099)){
                assert("Tensor range check" && 0 <= v1099 && v1099 < 4l);
                int v1102;
                v1102 = v1099 + v1081;
                float v1103;
                v1103 = v1068[v1102];
                float v1104;
                v1104 = v1100 + v1103;
                assert("Tensor range check" && 0 <= v1099 && v1099 < 4l);
                v1077[v1102] = v1104;
                v1100 = v1104;
                v1099 += 1l ;
            }
            float v1105;
            v1105 = v1078 + v1097;
            v1078 = v1105;
            v1079 += 1l ;
        }
        float v1106[4l];
        bool v1107[4l];
        int v1108;
        v1108 = 0l;
        while (while_method_3(v1108)){
            int v1110;
            v1110 = 0l;
            while (while_method_1(v1110)){
                assert("Tensor range check" && 0 <= v1108 && v1108 < 1l);
                assert("Tensor range check" && 0 <= v1110 && v1110 < 4l);
                int v1112;
                v1112 = 4l * v1108;
                int v1113;
                v1113 = v1112 + v1110;
                float v1114;
                v1114 = v1077[v1113];
                float v1115;
                v1115 = v1068[v1113];
                bool v1116;
                v1116 = v1115 > 0.0f;
                assert("Tensor range check" && 0 <= v1108 && v1108 < 1l);
                assert("Tensor range check" && 0 <= v1110 && v1110 < 4l);
                v1106[v1113] = v1114;
                v1107[v1113] = v1116;
                v1110 += 1l ;
            }
            v1108 += 1l ;
        }
        float v1117; bool v1118;
        Tuple3 tmp9 = Tuple3{-1.0f / 0.0f, false};
        v1117 = tmp9.v0; v1118 = tmp9.v1;
        int v1119;
        v1119 = 0l;
        while (while_method_3(v1119)){
            int v1121;
            v1121 = 0l;
            while (while_method_1(v1121)){
                assert("Tensor range check" && 0 <= v1119 && v1119 < 1l);
                assert("Tensor range check" && 0 <= v1121 && v1121 < 4l);
                int v1123;
                v1123 = 4l * v1119;
                int v1124;
                v1124 = v1123 + v1121;
                float v1125;
                v1125 = v1106[v1124];
                bool v1126;
                v1126 = v1107[v1124];
                float v1133; bool v1134;
                if (v1118){
                    if (v1126){
                        bool v1127;
                        v1127 = v1117 >= v1125;
                        float v1128;
                        if (v1127){
                            v1128 = v1117;
                        } else {
                            v1128 = v1125;
                        }
                        v1133 = v1128; v1134 = true;
                    } else {
                        v1133 = v1117; v1134 = v1118;
                    }
                } else {
                    if (v1126){
                        v1133 = v1125; v1134 = v1126;
                    } else {
                        v1133 = v1117; v1134 = v1118;
                    }
                }
                v1117 = v1133;
                v1118 = v1134;
                v1121 += 1l ;
            }
            v1119 += 1l ;
        }
        auto v1135 = cooperative_groups::coalesced_threads();
        int v1136;
        v1136 = threadIdx.x;
        int v1137;
        v1137 = v1136 / 32l;
        auto v1138 = cooperative_groups::labeled_partition(v1135,v1137);
        Closure5 v1139{};
        float v1140; bool v1141;
        Tuple3 tmp10 = cooperative_groups::reduce(v1138, Tuple3{v1117, v1118}, v1139);
        v1140 = tmp10.v0; v1141 = tmp10.v1;
        bool v1142;
        v1142 = v1141 == false;
        if (v1142){
            assert("The local reduce must be true." && v1141);
        } else {
        }
        float v1144[4l];
        int v1145[4l];
        int v1146;
        v1146 = 0l;
        while (while_method_3(v1146)){
            int v1148;
            v1148 = 0l;
            while (while_method_1(v1148)){
                assert("Tensor range check" && 0 <= v1146 && v1146 < 1l);
                assert("Tensor range check" && 0 <= v1148 && v1148 < 4l);
                int v1150;
                v1150 = 4l * v1146;
                int v1151;
                v1151 = v1150 + v1148;
                int v1152;
                v1152 = v986[v1151];
                float v1153;
                v1153 = curand_uniform(&v968);
                assert("Tensor range check" && 0 <= v1146 && v1146 < 1l);
                assert("Tensor range check" && 0 <= v1148 && v1148 < 4l);
                v1144[v1151] = v1153;
                v1145[v1151] = v1152;
                v1148 += 1l ;
            }
            v1146 += 1l ;
        }
        float v1154; int v1155;
        Tuple1 tmp11 = Tuple1{0.0f, 2147483647l};
        v1154 = tmp11.v0; v1155 = tmp11.v1;
        int v1156;
        v1156 = 0l;
        while (while_method_3(v1156)){
            int v1158;
            v1158 = 0l;
            while (while_method_1(v1158)){
                assert("Tensor range check" && 0 <= v1156 && v1156 < 1l);
                assert("Tensor range check" && 0 <= v1158 && v1158 < 4l);
                int v1160;
                v1160 = 4l * v1156;
                int v1161;
                v1161 = v1160 + v1158;
                float v1162;
                v1162 = v1144[v1161];
                int v1163;
                v1163 = v1145[v1161];
                bool v1164;
                v1164 = v1155 < v1163;
                float v1165; int v1166;
                if (v1164){
                    v1165 = v1154; v1166 = v1155;
                } else {
                    v1165 = v1162; v1166 = v1163;
                }
                v1154 = v1165;
                v1155 = v1166;
                v1158 += 1l ;
            }
            v1156 += 1l ;
        }
        auto v1167 = cooperative_groups::coalesced_threads();
        int v1168;
        v1168 = threadIdx.x;
        int v1169;
        v1169 = v1168 / 32l;
        auto v1170 = cooperative_groups::labeled_partition(v1167,v1169);
        Closure6 v1171{};
        float v1172; int v1173;
        Tuple1 tmp12 = cooperative_groups::reduce(v1170, Tuple1{v1154, v1155}, v1171);
        v1172 = tmp12.v0; v1173 = tmp12.v1;
        float v1174;
        v1174 = v1140 * v1172;
        int v1175[4l];
        bool v1176[4l];
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
                float v1183;
                v1183 = v1106[v1182];
                bool v1184;
                v1184 = v1107[v1182];
                int v1185;
                v1185 = v986[v1182];
                int v1188; bool v1189;
                if (v1184){
                    float v1186;
                    v1186 = v1183 - v1174;
                    bool v1187;
                    v1187 = v1186 >= 0.0f;
                    v1188 = v1185; v1189 = v1187;
                } else {
                    v1188 = 2147483647l; v1189 = false;
                }
                assert("Tensor range check" && 0 <= v1177 && v1177 < 1l);
                assert("Tensor range check" && 0 <= v1179 && v1179 < 4l);
                v1175[v1182] = v1188;
                v1176[v1182] = v1189;
                v1179 += 1l ;
            }
            v1177 += 1l ;
        }
        int v1190; bool v1191;
        Tuple4 tmp13 = Tuple4{2147483647l, false};
        v1190 = tmp13.v0; v1191 = tmp13.v1;
        int v1192;
        v1192 = 0l;
        while (while_method_3(v1192)){
            int v1194;
            v1194 = 0l;
            while (while_method_1(v1194)){
                assert("Tensor range check" && 0 <= v1192 && v1192 < 1l);
                assert("Tensor range check" && 0 <= v1194 && v1194 < 4l);
                int v1196;
                v1196 = 4l * v1192;
                int v1197;
                v1197 = v1196 + v1194;
                int v1198;
                v1198 = v1175[v1197];
                bool v1199;
                v1199 = v1176[v1197];
                int v1206; bool v1207;
                if (v1191){
                    if (v1199){
                        bool v1200;
                        v1200 = v1190 < v1198;
                        int v1201;
                        if (v1200){
                            v1201 = v1190;
                        } else {
                            v1201 = v1198;
                        }
                        v1206 = v1201; v1207 = true;
                    } else {
                        v1206 = v1190; v1207 = v1191;
                    }
                } else {
                    if (v1199){
                        v1206 = v1198; v1207 = v1199;
                    } else {
                        v1206 = v1190; v1207 = v1191;
                    }
                }
                v1190 = v1206;
                v1191 = v1207;
                v1194 += 1l ;
            }
            v1192 += 1l ;
        }
        auto v1208 = cooperative_groups::coalesced_threads();
        int v1209;
        v1209 = threadIdx.x;
        int v1210;
        v1210 = v1209 / 32l;
        auto v1211 = cooperative_groups::labeled_partition(v1208,v1210);
        Closure7 v1212{};
        int v1213; bool v1214;
        Tuple4 tmp14 = cooperative_groups::reduce(v1211, Tuple4{v1190, v1191}, v1212);
        v1213 = tmp14.v0; v1214 = tmp14.v1;
        bool v1215;
        v1215 = v1214 == false;
        if (v1215){
            assert("The local reduce must be true." && v1214);
        } else {
        }
        assert("Tensor range check" && 0 <= v981 && v981 < 64l);
        int v1217;
        v1217 = 0l;
        while (while_method_3(v1217)){
            assert("Tensor range check" && 0 <= v1217 && v1217 < 1l);
            int v1219;
            v1219 = 128l * v1217;
            int v1220;
            v1220 = v1219 + v984;
            assert("Tensor range check" && 0 <= v1217 && v1217 < 1l);
            int v1221;
            v1221 = 4l * v1217;
            int4* v1222;
            v1222 = reinterpret_cast<int4*>(v1068 + v1221);
            int4* v1223;
            v1223 = reinterpret_cast<int4*>(v14 + v1220);
            assert("Pointer alignment check" && (unsigned long long)(v1222) % 4l == 0 && (unsigned long long)(v1223) % 4l == 0);
            *v1223 = *v1222;
            v1217 += 1l ;
        }
        assert("Tensor range check" && 0 <= v981 && v981 < 64l);
        v15[v1028] = v1213;
        v981 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1224;
    v1224 = threadIdx.x;
    int v1225;
    v1225 = blockIdx.x;
    int v1226;
    v1226 = v1225 * 32l;
    int v1227;
    v1227 = v1224 + v1226;
    unsigned long long v1228;
    v1228 = (unsigned long long)v1227;
    curandStatePhilox4_32_10_t v1229;
    curand_init(12344321ull,v1228,0ull,&v1229);
    int v1230;
    v1230 = threadIdx.x;
    bool v1231;
    v1231 = 0l <= v1230;
    bool v1232;
    v1232 = v1231 == false;
    if (v1232){
        assert("The index needs to be zero or positive." && v1231);
    } else {
    }
    int v1234;
    v1234 = v1230 % 32l;
    int v1235;
    v1235 = v1230 / 32l;
    bool v1236;
    v1236 = v1235 < 1l;
    bool v1237;
    v1237 = v1236 == false;
    if (v1237){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1236);
    } else {
    }
    assert("Tensor range check" && 0 <= v1235 && v1235 < 1l);
    assert("Tensor range check" && 0 <= v1234 && v1234 < 32l);
    int v1239;
    v1239 = 4l * v1234;
    int v1240;
    v1240 = 128l * v1235;
    int v1241;
    v1241 = v1240 + v1239;
    assert("Tensor range check" && 0 <= v1235 && v1235 < 1l);
    assert("Tensor range check" && 0 <= v1234 && v1234 < 32l);
    assert("Tensor range check" && 0 <= v1235 && v1235 < 1l);
    int v1242;
    v1242 = 0l;
    while (while_method_2(v1242)){
        assert("Tensor range check" && 0 <= v1242 && v1242 < 64l);
        int v1244;
        v1244 = 128l * v1242;
        int v1245;
        v1245 = v1244 + v1241;
        float v1246[4l];
        int v1247[4l];
        int v1248;
        v1248 = 0l;
        while (while_method_3(v1248)){
            assert("Tensor range check" && 0 <= v1248 && v1248 < 1l);
            int v1250;
            v1250 = 4l * v1248;
            assert("Tensor range check" && 0 <= v1248 && v1248 < 1l);
            int v1251;
            v1251 = 128l * v1248;
            int v1252;
            v1252 = v1251 + v1245;
            int4* v1253;
            v1253 = reinterpret_cast<int4*>(v1 + v1252);
            int4* v1254;
            v1254 = reinterpret_cast<int4*>(v1246 + v1250);
            assert("Pointer alignment check" && (unsigned long long)(v1253) % 4l == 0 && (unsigned long long)(v1254) % 4l == 0);
            *v1254 = *v1253;
            v1248 += 1l ;
        }
        int v1255;
        v1255 = 0l;
        while (while_method_3(v1255)){
            int v1257;
            v1257 = 0l;
            while (while_method_1(v1257)){
                bool v1259;
                v1259 = 0l <= v1257;
                bool v1261;
                if (v1259){
                    bool v1260;
                    v1260 = v1257 < 4l;
                    v1261 = v1260;
                } else {
                    v1261 = false;
                }
                bool v1262;
                v1262 = v1261 == false;
                if (v1262){
                    assert("The indices should be inside the range of the dimension." && v1261);
                } else {
                }
                bool v1264;
                v1264 = 0l <= v1234;
                bool v1266;
                if (v1264){
                    bool v1265;
                    v1265 = v1234 < 32l;
                    v1266 = v1265;
                } else {
                    v1266 = false;
                }
                bool v1267;
                v1267 = v1266 == false;
                if (v1267){
                    assert("The indices should be inside the range of the dimension." && v1266);
                } else {
                }
                int v1269;
                v1269 = v1234 * 4l;
                int v1270;
                v1270 = v1257 + v1269;
                bool v1271;
                v1271 = 0l <= v1255;
                bool v1273;
                if (v1271){
                    bool v1272;
                    v1272 = v1255 < 1l;
                    v1273 = v1272;
                } else {
                    v1273 = false;
                }
                bool v1274;
                v1274 = v1273 == false;
                if (v1274){
                    assert("The indices should be inside the range of the dimension." && v1273);
                } else {
                }
                int v1276;
                v1276 = v1255 * 128l;
                int v1277;
                v1277 = v1270 + v1276;
                assert("Tensor range check" && 0 <= v1255 && v1255 < 1l);
                assert("Tensor range check" && 0 <= v1257 && v1257 < 4l);
                int v1278;
                v1278 = 4l * v1255;
                int v1279;
                v1279 = v1278 + v1257;
                v1247[v1279] = v1277;
                v1257 += 1l ;
            }
            v1255 += 1l ;
        }
        bool v1280;
        v1280 = 0l <= v1235;
        bool v1281;
        v1281 = v1280 && v1236;
        bool v1282;
        v1282 = v1281 == false;
        if (v1282){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1281);
        } else {
        }
        bool v1284;
        v1284 = 0l <= v1242;
        bool v1286;
        if (v1284){
            bool v1285;
            v1285 = v1242 < 64l;
            v1286 = v1285;
        } else {
            v1286 = false;
        }
        bool v1287;
        v1287 = v1286 == false;
        if (v1287){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1286);
        } else {
        }
        int v1289;
        v1289 = v1242 + v1235;
        bool v1290[4l];
        int v1291;
        v1291 = 0l;
        while (while_method_3(v1291)){
            int v1293;
            v1293 = 0l;
            while (while_method_1(v1293)){
                assert("Tensor range check" && 0 <= v1291 && v1291 < 1l);
                assert("Tensor range check" && 0 <= v1293 && v1293 < 4l);
                int v1295;
                v1295 = 4l * v1291;
                int v1296;
                v1296 = v1295 + v1293;
                float v1297;
                v1297 = v1246[v1296];
                int v1298;
                v1298 = v1247[v1296];
                bool v1299;
                v1299 = v1298 < 11l;
                assert("Tensor range check" && 0 <= v1291 && v1291 < 1l);
                assert("Tensor range check" && 0 <= v1293 && v1293 < 4l);
                v1290[v1296] = v1299;
                v1293 += 1l ;
            }
            v1291 += 1l ;
        }
        int v1300[4l];
        int v1301;
        v1301 = 0l;
        while (while_method_3(v1301)){
            int v1303;
            v1303 = 0l;
            while (while_method_1(v1303)){
                assert("Tensor range check" && 0 <= v1301 && v1301 < 1l);
                assert("Tensor range check" && 0 <= v1303 && v1303 < 4l);
                int v1305;
                v1305 = 4l * v1301;
                int v1306;
                v1306 = v1305 + v1303;
                bool v1307;
                v1307 = v1290[v1306];
                int v1308;
                if (v1307){
                    v1308 = 1l;
                } else {
                    v1308 = 0l;
                }
                assert("Tensor range check" && 0 <= v1301 && v1301 < 1l);
                assert("Tensor range check" && 0 <= v1303 && v1303 < 4l);
                v1300[v1306] = v1308;
                v1303 += 1l ;
            }
            v1301 += 1l ;
        }
        int v1309;
        v1309 = 0l;
        int v1310;
        v1310 = 0l;
        while (while_method_3(v1310)){
            int v1312;
            v1312 = 0l;
            while (while_method_1(v1312)){
                assert("Tensor range check" && 0 <= v1310 && v1310 < 1l);
                assert("Tensor range check" && 0 <= v1312 && v1312 < 4l);
                int v1314;
                v1314 = 4l * v1310;
                int v1315;
                v1315 = v1314 + v1312;
                int v1316;
                v1316 = v1300[v1315];
                int v1317;
                v1317 = v1309 + v1316;
                v1309 = v1317;
                v1312 += 1l ;
            }
            v1310 += 1l ;
        }
        auto v1318 = cooperative_groups::coalesced_threads();
        int v1319;
        v1319 = threadIdx.x;
        int v1320;
        v1320 = v1319 / 32l;
        auto v1321 = cooperative_groups::labeled_partition(v1318,v1320);
        Closure4 v1322{};
        int v1323;
        v1323 = cooperative_groups::reduce(v1321, v1309, v1322);
        float v1324[4l];
        int v1325;
        v1325 = 0l;
        while (while_method_3(v1325)){
            int v1327;
            v1327 = 0l;
            while (while_method_1(v1327)){
                assert("Tensor range check" && 0 <= v1325 && v1325 < 1l);
                assert("Tensor range check" && 0 <= v1327 && v1327 < 4l);
                int v1329;
                v1329 = 4l * v1325;
                int v1330;
                v1330 = v1329 + v1327;
                float v1331;
                v1331 = v1246[v1330];
                bool v1332;
                v1332 = v1290[v1330];
                float v1333;
                if (v1332){
                    v1333 = v1331;
                } else {
                    v1333 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1325 && v1325 < 1l);
                assert("Tensor range check" && 0 <= v1327 && v1327 < 4l);
                v1324[v1330] = v1333;
                v1327 += 1l ;
            }
            v1325 += 1l ;
        }
        float v1334;
        v1334 = 0.0f;
        int v1335;
        v1335 = 0l;
        while (while_method_3(v1335)){
            int v1337;
            v1337 = 0l;
            while (while_method_1(v1337)){
                assert("Tensor range check" && 0 <= v1335 && v1335 < 1l);
                assert("Tensor range check" && 0 <= v1337 && v1337 < 4l);
                int v1339;
                v1339 = 4l * v1335;
                int v1340;
                v1340 = v1339 + v1337;
                float v1341;
                v1341 = v1324[v1340];
                float v1342;
                v1342 = v1334 + v1341;
                v1334 = v1342;
                v1337 += 1l ;
            }
            v1335 += 1l ;
        }
        auto v1343 = cooperative_groups::coalesced_threads();
        int v1344;
        v1344 = threadIdx.x;
        int v1345;
        v1345 = v1344 / 32l;
        auto v1346 = cooperative_groups::labeled_partition(v1343,v1345);
        float v1347;
        v1347 = cooperative_groups::reduce(v1346, v1334, v42);
        float v1348;
        v1348 = (float)v1323;
        float v1349;
        v1349 = v1347 / v1348;
        float v1350[4l];
        int v1351;
        v1351 = 0l;
        while (while_method_3(v1351)){
            int v1353;
            v1353 = 0l;
            while (while_method_1(v1353)){
                assert("Tensor range check" && 0 <= v1351 && v1351 < 1l);
                assert("Tensor range check" && 0 <= v1353 && v1353 < 4l);
                int v1355;
                v1355 = 4l * v1351;
                int v1356;
                v1356 = v1355 + v1353;
                float v1357;
                v1357 = v1246[v1356];
                bool v1358;
                v1358 = v1290[v1356];
                float v1359;
                if (v1358){
                    v1359 = v1357;
                } else {
                    v1359 = -1.0f / 0.0f;
                }
                float v1360;
                v1360 = v1359 - v1349;
                float v1361;
                v1361 = exp(v1360);
                assert("Tensor range check" && 0 <= v1351 && v1351 < 1l);
                assert("Tensor range check" && 0 <= v1353 && v1353 < 4l);
                v1350[v1356] = v1361;
                v1353 += 1l ;
            }
            v1351 += 1l ;
        }
        float v1362;
        v1362 = 0.0f;
        int v1363;
        v1363 = 0l;
        while (while_method_3(v1363)){
            int v1365;
            v1365 = 0l;
            while (while_method_1(v1365)){
                assert("Tensor range check" && 0 <= v1363 && v1363 < 1l);
                assert("Tensor range check" && 0 <= v1365 && v1365 < 4l);
                int v1367;
                v1367 = 4l * v1363;
                int v1368;
                v1368 = v1367 + v1365;
                float v1369;
                v1369 = v1350[v1368];
                float v1370;
                v1370 = v1362 + v1369;
                v1362 = v1370;
                v1365 += 1l ;
            }
            v1363 += 1l ;
        }
        auto v1371 = cooperative_groups::coalesced_threads();
        int v1372;
        v1372 = threadIdx.x;
        int v1373;
        v1373 = v1372 / 32l;
        auto v1374 = cooperative_groups::labeled_partition(v1371,v1373);
        float v1375;
        v1375 = cooperative_groups::reduce(v1374, v1362, v42);
        float v1376[4l];
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
                v1383 = v1350[v1382];
                float v1384;
                v1384 = v1383 / v1375;
                assert("Tensor range check" && 0 <= v1377 && v1377 < 1l);
                assert("Tensor range check" && 0 <= v1379 && v1379 < 4l);
                v1376[v1382] = v1384;
                v1379 += 1l ;
            }
            v1377 += 1l ;
        }
        float v1385[4l];
        float v1386;
        v1386 = 0.0f;
        int v1387;
        v1387 = 0l;
        while (while_method_3(v1387)){
            assert("Tensor range check" && 0 <= v1387 && v1387 < 1l);
            int v1389;
            v1389 = 4l * v1387;
            assert("Tensor range check" && 0 <= v1387 && v1387 < 1l);
            int v1390; float v1391;
            Tuple0 tmp15 = Tuple0{0l, 0.0f};
            v1390 = tmp15.v0; v1391 = tmp15.v1;
            while (while_method_1(v1390)){
                assert("Tensor range check" && 0 <= v1390 && v1390 < 4l);
                int v1393;
                v1393 = v1390 + v1389;
                float v1394;
                v1394 = v1376[v1393];
                float v1395;
                v1395 = v1391 + v1394;
                v1391 = v1395;
                v1390 += 1l ;
            }
            auto v1396 = cooperative_groups::coalesced_threads();
            int v1397;
            v1397 = threadIdx.x;
            int v1398;
            v1398 = v1397 / 32l;
            auto v1399 = cooperative_groups::labeled_partition(v1396,v1398);
            Closure2 v1400{};
            float v1401;
            v1401 = cooperative_groups::inclusive_scan(v1399, v1391, v1400);
            float v1402;
            v1402 = v1399.shfl_up(v1401,1);
            bool v1403;
            v1403 = v1399.thread_rank() == 0;
            float v1404;
            if (v1403){
                v1404 = 0.0f;
            } else {
                v1404 = v1402;
            }
            float v1405;
            v1405 = v1399.shfl(v1401,v1399.num_threads()-1);
            float v1406;
            v1406 = v1386 + v1404;
            int v1407; float v1408;
            Tuple0 tmp16 = Tuple0{0l, v1406};
            v1407 = tmp16.v0; v1408 = tmp16.v1;
            while (while_method_1(v1407)){
                assert("Tensor range check" && 0 <= v1407 && v1407 < 4l);
                int v1410;
                v1410 = v1407 + v1389;
                float v1411;
                v1411 = v1376[v1410];
                float v1412;
                v1412 = v1408 + v1411;
                assert("Tensor range check" && 0 <= v1407 && v1407 < 4l);
                v1385[v1410] = v1412;
                v1408 = v1412;
                v1407 += 1l ;
            }
            float v1413;
            v1413 = v1386 + v1405;
            v1386 = v1413;
            v1387 += 1l ;
        }
        float v1414[4l];
        bool v1415[4l];
        int v1416;
        v1416 = 0l;
        while (while_method_3(v1416)){
            int v1418;
            v1418 = 0l;
            while (while_method_1(v1418)){
                assert("Tensor range check" && 0 <= v1416 && v1416 < 1l);
                assert("Tensor range check" && 0 <= v1418 && v1418 < 4l);
                int v1420;
                v1420 = 4l * v1416;
                int v1421;
                v1421 = v1420 + v1418;
                float v1422;
                v1422 = v1385[v1421];
                float v1423;
                v1423 = v1376[v1421];
                bool v1424;
                v1424 = v1423 > 0.0f;
                assert("Tensor range check" && 0 <= v1416 && v1416 < 1l);
                assert("Tensor range check" && 0 <= v1418 && v1418 < 4l);
                v1414[v1421] = v1422;
                v1415[v1421] = v1424;
                v1418 += 1l ;
            }
            v1416 += 1l ;
        }
        float v1425; bool v1426;
        Tuple3 tmp17 = Tuple3{-1.0f / 0.0f, false};
        v1425 = tmp17.v0; v1426 = tmp17.v1;
        int v1427;
        v1427 = 0l;
        while (while_method_3(v1427)){
            int v1429;
            v1429 = 0l;
            while (while_method_1(v1429)){
                assert("Tensor range check" && 0 <= v1427 && v1427 < 1l);
                assert("Tensor range check" && 0 <= v1429 && v1429 < 4l);
                int v1431;
                v1431 = 4l * v1427;
                int v1432;
                v1432 = v1431 + v1429;
                float v1433;
                v1433 = v1414[v1432];
                bool v1434;
                v1434 = v1415[v1432];
                float v1441; bool v1442;
                if (v1426){
                    if (v1434){
                        bool v1435;
                        v1435 = v1425 >= v1433;
                        float v1436;
                        if (v1435){
                            v1436 = v1425;
                        } else {
                            v1436 = v1433;
                        }
                        v1441 = v1436; v1442 = true;
                    } else {
                        v1441 = v1425; v1442 = v1426;
                    }
                } else {
                    if (v1434){
                        v1441 = v1433; v1442 = v1434;
                    } else {
                        v1441 = v1425; v1442 = v1426;
                    }
                }
                v1425 = v1441;
                v1426 = v1442;
                v1429 += 1l ;
            }
            v1427 += 1l ;
        }
        auto v1443 = cooperative_groups::coalesced_threads();
        int v1444;
        v1444 = threadIdx.x;
        int v1445;
        v1445 = v1444 / 32l;
        auto v1446 = cooperative_groups::labeled_partition(v1443,v1445);
        Closure5 v1447{};
        float v1448; bool v1449;
        Tuple3 tmp18 = cooperative_groups::reduce(v1446, Tuple3{v1425, v1426}, v1447);
        v1448 = tmp18.v0; v1449 = tmp18.v1;
        bool v1450;
        v1450 = v1449 == false;
        if (v1450){
            assert("The local reduce must be true." && v1449);
        } else {
        }
        float v1452[4l];
        int v1453[4l];
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
                int v1460;
                v1460 = v1247[v1459];
                float v1461;
                v1461 = curand_uniform(&v1229);
                assert("Tensor range check" && 0 <= v1454 && v1454 < 1l);
                assert("Tensor range check" && 0 <= v1456 && v1456 < 4l);
                v1452[v1459] = v1461;
                v1453[v1459] = v1460;
                v1456 += 1l ;
            }
            v1454 += 1l ;
        }
        float v1462; int v1463;
        Tuple1 tmp19 = Tuple1{0.0f, 2147483647l};
        v1462 = tmp19.v0; v1463 = tmp19.v1;
        int v1464;
        v1464 = 0l;
        while (while_method_3(v1464)){
            int v1466;
            v1466 = 0l;
            while (while_method_1(v1466)){
                assert("Tensor range check" && 0 <= v1464 && v1464 < 1l);
                assert("Tensor range check" && 0 <= v1466 && v1466 < 4l);
                int v1468;
                v1468 = 4l * v1464;
                int v1469;
                v1469 = v1468 + v1466;
                float v1470;
                v1470 = v1452[v1469];
                int v1471;
                v1471 = v1453[v1469];
                bool v1472;
                v1472 = v1463 < v1471;
                float v1473; int v1474;
                if (v1472){
                    v1473 = v1462; v1474 = v1463;
                } else {
                    v1473 = v1470; v1474 = v1471;
                }
                v1462 = v1473;
                v1463 = v1474;
                v1466 += 1l ;
            }
            v1464 += 1l ;
        }
        auto v1475 = cooperative_groups::coalesced_threads();
        int v1476;
        v1476 = threadIdx.x;
        int v1477;
        v1477 = v1476 / 32l;
        auto v1478 = cooperative_groups::labeled_partition(v1475,v1477);
        Closure6 v1479{};
        float v1480; int v1481;
        Tuple1 tmp20 = cooperative_groups::reduce(v1478, Tuple1{v1462, v1463}, v1479);
        v1480 = tmp20.v0; v1481 = tmp20.v1;
        float v1482;
        v1482 = v1448 * v1480;
        int v1483[4l];
        bool v1484[4l];
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
                float v1491;
                v1491 = v1414[v1490];
                bool v1492;
                v1492 = v1415[v1490];
                int v1493;
                v1493 = v1247[v1490];
                int v1496; bool v1497;
                if (v1492){
                    float v1494;
                    v1494 = v1491 - v1482;
                    bool v1495;
                    v1495 = v1494 >= 0.0f;
                    v1496 = v1493; v1497 = v1495;
                } else {
                    v1496 = 2147483647l; v1497 = false;
                }
                assert("Tensor range check" && 0 <= v1485 && v1485 < 1l);
                assert("Tensor range check" && 0 <= v1487 && v1487 < 4l);
                v1483[v1490] = v1496;
                v1484[v1490] = v1497;
                v1487 += 1l ;
            }
            v1485 += 1l ;
        }
        int v1498; bool v1499;
        Tuple4 tmp21 = Tuple4{2147483647l, false};
        v1498 = tmp21.v0; v1499 = tmp21.v1;
        int v1500;
        v1500 = 0l;
        while (while_method_3(v1500)){
            int v1502;
            v1502 = 0l;
            while (while_method_1(v1502)){
                assert("Tensor range check" && 0 <= v1500 && v1500 < 1l);
                assert("Tensor range check" && 0 <= v1502 && v1502 < 4l);
                int v1504;
                v1504 = 4l * v1500;
                int v1505;
                v1505 = v1504 + v1502;
                int v1506;
                v1506 = v1483[v1505];
                bool v1507;
                v1507 = v1484[v1505];
                int v1514; bool v1515;
                if (v1499){
                    if (v1507){
                        bool v1508;
                        v1508 = v1498 < v1506;
                        int v1509;
                        if (v1508){
                            v1509 = v1498;
                        } else {
                            v1509 = v1506;
                        }
                        v1514 = v1509; v1515 = true;
                    } else {
                        v1514 = v1498; v1515 = v1499;
                    }
                } else {
                    if (v1507){
                        v1514 = v1506; v1515 = v1507;
                    } else {
                        v1514 = v1498; v1515 = v1499;
                    }
                }
                v1498 = v1514;
                v1499 = v1515;
                v1502 += 1l ;
            }
            v1500 += 1l ;
        }
        auto v1516 = cooperative_groups::coalesced_threads();
        int v1517;
        v1517 = threadIdx.x;
        int v1518;
        v1518 = v1517 / 32l;
        auto v1519 = cooperative_groups::labeled_partition(v1516,v1518);
        Closure7 v1520{};
        int v1521; bool v1522;
        Tuple4 tmp22 = cooperative_groups::reduce(v1519, Tuple4{v1498, v1499}, v1520);
        v1521 = tmp22.v0; v1522 = tmp22.v1;
        bool v1523;
        v1523 = v1522 == false;
        if (v1523){
            assert("The local reduce must be true." && v1522);
        } else {
        }
        assert("Tensor range check" && 0 <= v1242 && v1242 < 64l);
        int v1525;
        v1525 = 0l;
        while (while_method_3(v1525)){
            assert("Tensor range check" && 0 <= v1525 && v1525 < 1l);
            int v1527;
            v1527 = 128l * v1525;
            int v1528;
            v1528 = v1527 + v1245;
            assert("Tensor range check" && 0 <= v1525 && v1525 < 1l);
            int v1529;
            v1529 = 4l * v1525;
            int4* v1530;
            v1530 = reinterpret_cast<int4*>(v1376 + v1529);
            int4* v1531;
            v1531 = reinterpret_cast<int4*>(v16 + v1528);
            assert("Pointer alignment check" && (unsigned long long)(v1530) % 4l == 0 && (unsigned long long)(v1531) % 4l == 0);
            *v1531 = *v1530;
            v1525 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1242 && v1242 < 64l);
        v17[v1289] = v1521;
        v1242 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
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
        v22 = 0l <= v20;
        bool v23;
        v23 = v22 == false;
        if (v23){
            assert("The index needs to be zero or positive." && v22);
        } else {
        }
        int v25;
        v25 = v20 % 16l;
        int v26;
        v26 = v20 / 16l;
        bool v27;
        v27 = v26 < 128l;
        bool v28;
        v28 = v27 == false;
        if (v28){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v27);
        } else {
        }
        assert("Tensor range check" && 0 <= v26 && v26 < 128l);
        assert("Tensor range check" && 0 <= v25 && v25 < 16l);
        int v30;
        v30 = 4l * v25;
        int v31;
        v31 = 64l * v26;
        int v32;
        v32 = v31 + v30;
        float v33[4l];
        int4* v34;
        v34 = reinterpret_cast<int4*>(v1 + v32);
        int4* v35;
        v35 = reinterpret_cast<int4*>(v33 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v34) % 4l == 0 && (unsigned long long)(v35) % 4l == 0);
        *v35 = *v34;
        int v36; float v37;
        Tuple0 tmp23 = Tuple0{0l, v18};
        v36 = tmp23.v0; v37 = tmp23.v1;
        while (while_method_1(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 4l);
            float v39;
            v39 = v33[v36];
            float v40;
            v40 = v37 + v39;
            v37 = v40;
            v36 += 1l ;
        }
        v18 = v37;
        v20 += 32l ;
    }
    auto v41 = cooperative_groups::coalesced_threads();
    Closure0 v42{};
    float v43;
    v43 = cooperative_groups::reduce(v41, v18, v42);
    int v44;
    v44 = threadIdx.x;
    int v45;
    v45 = v44 / 32l;
    extern __shared__ unsigned char v46[];
    float * v47;
    v47 = reinterpret_cast<float *>(&v46[0ull]);
    assert("Tensor range check" && 0 <= v45 && v45 < 1l);
    v47[v45] = v43;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v49;
    v49 = threadIdx.x;
    int v50;
    v50 = v49 % 32l;
    bool v51;
    v51 = v45 == 0l;
    bool v53;
    if (v51){
        bool v52;
        v52 = v50 < 1l;
        v53 = v52;
    } else {
        v53 = false;
    }
    if (v53){
        auto v54 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v50 && v50 < 1l);
        float v55;
        v55 = v47[v50];
        float v56;
        v56 = cooperative_groups::reduce(v54, v55, v42);
        v2[0l] = v56;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v57;
    v57 = threadIdx.x;
    bool v58;
    v58 = 0l <= v57;
    bool v59;
    v59 = v58 == false;
    if (v59){
        assert("The index needs to be zero or positive." && v58);
    } else {
    }
    int v61;
    v61 = v57 % 16l;
    int v62;
    v62 = v57 / 16l;
    bool v63;
    v63 = v62 < 2l;
    bool v64;
    v64 = v63 == false;
    if (v64){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v63);
    } else {
    }
    assert("Tensor range check" && 0 <= v62 && v62 < 2l);
    assert("Tensor range check" && 0 <= v61 && v61 < 16l);
    int v66;
    v66 = 4l * v61;
    int v67;
    v67 = 64l * v62;
    int v68;
    v68 = v67 + v66;
    assert("Tensor range check" && 0 <= v62 && v62 < 2l);
    assert("Tensor range check" && 0 <= v61 && v61 < 16l);
    int v69;
    v69 = 0l;
    while (while_method_2(v69)){
        assert("Tensor range check" && 0 <= v69 && v69 < 64l);
        int v71;
        v71 = 128l * v69;
        int v72;
        v72 = v71 + v68;
        int v73[4l];
        int v74[4l];
        int v75;
        v75 = 0l;
        while (while_method_3(v75)){
            assert("Tensor range check" && 0 <= v75 && v75 < 1l);
            int v77;
            v77 = 4l * v75;
            assert("Tensor range check" && 0 <= v75 && v75 < 1l);
            int v78;
            v78 = 64l * v75;
            int v79;
            v79 = v78 + v72;
            int4* v80;
            v80 = reinterpret_cast<int4*>(v0 + v79);
            int4* v81;
            v81 = reinterpret_cast<int4*>(v73 + v77);
            assert("Pointer alignment check" && (unsigned long long)(v80) % 4l == 0 && (unsigned long long)(v81) % 4l == 0);
            *v81 = *v80;
            v75 += 1l ;
        }
        int v82;
        v82 = 0l;
        while (while_method_3(v82)){
            int v84;
            v84 = 0l;
            while (while_method_1(v84)){
                bool v86;
                v86 = 0l <= v84;
                bool v88;
                if (v86){
                    bool v87;
                    v87 = v84 < 4l;
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
                v91 = 0l <= v61;
                bool v93;
                if (v91){
                    bool v92;
                    v92 = v61 < 16l;
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
                v96 = v61 * 4l;
                int v97;
                v97 = v84 + v96;
                bool v98;
                v98 = 0l <= v82;
                bool v100;
                if (v98){
                    bool v99;
                    v99 = v82 < 1l;
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
                v103 = v82 * 64l;
                int v104;
                v104 = v97 + v103;
                assert("Tensor range check" && 0 <= v82 && v82 < 1l);
                assert("Tensor range check" && 0 <= v84 && v84 < 4l);
                int v105;
                v105 = 4l * v82;
                int v106;
                v106 = v105 + v84;
                v74[v106] = v104;
                v84 += 1l ;
            }
            v82 += 1l ;
        }
        bool v107;
        v107 = 0l <= v62;
        bool v108;
        v108 = v107 && v63;
        bool v109;
        v109 = v108 == false;
        if (v109){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v108);
        } else {
        }
        bool v111;
        v111 = 0l <= v69;
        bool v113;
        if (v111){
            bool v112;
            v112 = v69 < 64l;
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
        v116 = v69 * 2l;
        int v117;
        v117 = v116 + v62;
        assert("Tensor range check" && 0 <= v69 && v69 < 64l);
        int v118;
        v118 = 0l;
        while (while_method_3(v118)){
            assert("Tensor range check" && 0 <= v118 && v118 < 1l);
            int v120;
            v120 = 64l * v118;
            int v121;
            v121 = v120 + v72;
            assert("Tensor range check" && 0 <= v118 && v118 < 1l);
            int v122;
            v122 = 4l * v118;
            int4* v123;
            v123 = reinterpret_cast<int4*>(v73 + v122);
            int4* v124;
            v124 = reinterpret_cast<int4*>(v3 + v121);
            assert("Pointer alignment check" && (unsigned long long)(v123) % 4l == 0 && (unsigned long long)(v124) % 4l == 0);
            *v124 = *v123;
            v118 += 1l ;
        }
        v69 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v125;
    v125 = threadIdx.x;
    bool v126;
    v126 = 0l <= v125;
    bool v127;
    v127 = v126 == false;
    if (v127){
        assert("The index needs to be zero or positive." && v126);
    } else {
    }
    int v129;
    v129 = v125 % 16l;
    int v130;
    v130 = v125 / 16l;
    bool v131;
    v131 = v130 < 2l;
    bool v132;
    v132 = v131 == false;
    if (v132){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v131);
    } else {
    }
    assert("Tensor range check" && 0 <= v130 && v130 < 2l);
    assert("Tensor range check" && 0 <= v129 && v129 < 16l);
    int v134;
    v134 = 4l * v129;
    int v135;
    v135 = 64l * v130;
    int v136;
    v136 = v135 + v134;
    assert("Tensor range check" && 0 <= v130 && v130 < 2l);
    assert("Tensor range check" && 0 <= v129 && v129 < 16l);
    int v137;
    v137 = 0l;
    while (while_method_2(v137)){
        assert("Tensor range check" && 0 <= v137 && v137 < 64l);
        int v139;
        v139 = 128l * v137;
        int v140;
        v140 = v139 + v136;
        float v141[4l];
        int v142[4l];
        int v143;
        v143 = 0l;
        while (while_method_3(v143)){
            assert("Tensor range check" && 0 <= v143 && v143 < 1l);
            int v145;
            v145 = 4l * v143;
            assert("Tensor range check" && 0 <= v143 && v143 < 1l);
            int v146;
            v146 = 64l * v143;
            int v147;
            v147 = v146 + v140;
            int4* v148;
            v148 = reinterpret_cast<int4*>(v1 + v147);
            int4* v149;
            v149 = reinterpret_cast<int4*>(v141 + v145);
            assert("Pointer alignment check" && (unsigned long long)(v148) % 4l == 0 && (unsigned long long)(v149) % 4l == 0);
            *v149 = *v148;
            v143 += 1l ;
        }
        int v150;
        v150 = 0l;
        while (while_method_3(v150)){
            int v152;
            v152 = 0l;
            while (while_method_1(v152)){
                bool v154;
                v154 = 0l <= v152;
                bool v156;
                if (v154){
                    bool v155;
                    v155 = v152 < 4l;
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
                v159 = 0l <= v129;
                bool v161;
                if (v159){
                    bool v160;
                    v160 = v129 < 16l;
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
                v164 = v129 * 4l;
                int v165;
                v165 = v152 + v164;
                bool v166;
                v166 = 0l <= v150;
                bool v168;
                if (v166){
                    bool v167;
                    v167 = v150 < 1l;
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
                v171 = v150 * 64l;
                int v172;
                v172 = v165 + v171;
                assert("Tensor range check" && 0 <= v150 && v150 < 1l);
                assert("Tensor range check" && 0 <= v152 && v152 < 4l);
                int v173;
                v173 = 4l * v150;
                int v174;
                v174 = v173 + v152;
                v142[v174] = v172;
                v152 += 1l ;
            }
            v150 += 1l ;
        }
        bool v175;
        v175 = 0l <= v130;
        bool v176;
        v176 = v175 && v131;
        bool v177;
        v177 = v176 == false;
        if (v177){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v176);
        } else {
        }
        bool v179;
        v179 = 0l <= v137;
        bool v181;
        if (v179){
            bool v180;
            v180 = v137 < 64l;
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
        v184 = v137 * 2l;
        int v185;
        v185 = v184 + v130;
        int v186[4l];
        int v187[4l];
        int v188;
        v188 = 0l;
        while (while_method_3(v188)){
            int v190;
            v190 = 0l;
            while (while_method_1(v190)){
                assert("Tensor range check" && 0 <= v188 && v188 < 1l);
                assert("Tensor range check" && 0 <= v190 && v190 < 4l);
                int v192;
                v192 = 4l * v188;
                int v193;
                v193 = v192 + v190;
                int v194;
                v194 = v142[v193];
                assert("Tensor range check" && 0 <= v188 && v188 < 1l);
                assert("Tensor range check" && 0 <= v190 && v190 < 4l);
                v186[v193] = v185;
                v187[v193] = v194;
                v190 += 1l ;
            }
            v188 += 1l ;
        }
        assert("Tensor range check" && 0 <= v137 && v137 < 64l);
        int v195;
        v195 = 0l;
        while (while_method_3(v195)){
            assert("Tensor range check" && 0 <= v195 && v195 < 1l);
            int v197;
            v197 = 64l * v195;
            int v198;
            v198 = v197 + v140;
            assert("Tensor range check" && 0 <= v195 && v195 < 1l);
            int v199;
            v199 = 4l * v195;
            int4* v200;
            v200 = reinterpret_cast<int4*>(v186 + v199);
            int4* v201;
            v201 = reinterpret_cast<int4*>(v10 + v198);
            assert("Pointer alignment check" && (unsigned long long)(v200) % 4l == 0 && (unsigned long long)(v201) % 4l == 0);
            *v201 = *v200;
            int4* v202;
            v202 = reinterpret_cast<int4*>(v187 + v199);
            int4* v203;
            v203 = reinterpret_cast<int4*>(v11 + v198);
            assert("Pointer alignment check" && (unsigned long long)(v202) % 4l == 0 && (unsigned long long)(v203) % 4l == 0);
            *v203 = *v202;
            v195 += 1l ;
        }
        v137 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v204;
    v204 = threadIdx.x;
    bool v205;
    v205 = 0l <= v204;
    bool v206;
    v206 = v205 == false;
    if (v206){
        assert("The index needs to be zero or positive." && v205);
    } else {
    }
    int v208;
    v208 = v204 % 16l;
    int v209;
    v209 = v204 / 16l;
    bool v210;
    v210 = v209 < 2l;
    bool v211;
    v211 = v210 == false;
    if (v211){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v210);
    } else {
    }
    assert("Tensor range check" && 0 <= v209 && v209 < 2l);
    assert("Tensor range check" && 0 <= v208 && v208 < 16l);
    int v213;
    v213 = 4l * v208;
    int v214;
    v214 = 64l * v209;
    int v215;
    v215 = v214 + v213;
    assert("Tensor range check" && 0 <= v209 && v209 < 2l);
    int v216;
    v216 = 0l;
    while (while_method_2(v216)){
        assert("Tensor range check" && 0 <= v216 && v216 < 64l);
        int v218;
        v218 = 128l * v216;
        int v219;
        v219 = v218 + v215;
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
                v238 = 0l <= v208;
                bool v240;
                if (v238){
                    bool v239;
                    v239 = v208 < 16l;
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
                v243 = v208 * 4l;
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
        v254 = 0l <= v209;
        bool v255;
        v255 = v254 && v210;
        bool v256;
        v256 = v255 == false;
        if (v256){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v255);
        } else {
        }
        bool v258;
        v258 = 0l <= v216;
        bool v260;
        if (v258){
            bool v259;
            v259 = v216 < 64l;
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
        v263 = v216 * 2l;
        int v264;
        v264 = v263 + v209;
        assert("Tensor range check" && 0 <= v216 && v216 < 64l);
        int v265;
        v265 = 2l * v216;
        int v266;
        v266 = v265 + v209;
        v12[v266] = v264;
        v216 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v267;
    v267 = threadIdx.x;
    bool v268;
    v268 = 0l <= v267;
    bool v269;
    v269 = v268 == false;
    if (v269){
        assert("The index needs to be zero or positive." && v268);
    } else {
    }
    int v271;
    v271 = v267 % 16l;
    int v272;
    v272 = v267 / 16l;
    bool v273;
    v273 = v272 < 2l;
    bool v274;
    v274 = v273 == false;
    if (v274){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v273);
    } else {
    }
    assert("Tensor range check" && 0 <= v272 && v272 < 2l);
    assert("Tensor range check" && 0 <= v271 && v271 < 16l);
    int v276;
    v276 = 4l * v271;
    int v277;
    v277 = 64l * v272;
    int v278;
    v278 = v277 + v276;
    assert("Tensor range check" && 0 <= v272 && v272 < 2l);
    assert("Tensor range check" && 0 <= v271 && v271 < 16l);
    int v279;
    v279 = 0l;
    while (while_method_2(v279)){
        assert("Tensor range check" && 0 <= v279 && v279 < 64l);
        int v281;
        v281 = 128l * v279;
        int v282;
        v282 = v281 + v278;
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
            v288 = 64l * v285;
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
                v301 = 0l <= v271;
                bool v303;
                if (v301){
                    bool v302;
                    v302 = v271 < 16l;
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
                v306 = v271 * 4l;
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
                v313 = v292 * 64l;
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
        v317 = 0l <= v272;
        bool v318;
        v318 = v317 && v273;
        bool v319;
        v319 = v318 == false;
        if (v319){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v318);
        } else {
        }
        bool v321;
        v321 = 0l <= v279;
        bool v323;
        if (v321){
            bool v322;
            v322 = v279 < 64l;
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
        v326 = v279 * 2l;
        int v327;
        v327 = v326 + v272;
        float v328;
        v328 = 0.0f;
        int v329;
        v329 = 0l;
        while (while_method_3(v329)){
            int v331;
            v331 = 0l;
            while (while_method_1(v331)){
                assert("Tensor range check" && 0 <= v329 && v329 < 1l);
                assert("Tensor range check" && 0 <= v331 && v331 < 4l);
                int v333;
                v333 = 4l * v329;
                int v334;
                v334 = v333 + v331;
                float v335;
                v335 = v283[v334];
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
        int v339;
        v339 = v338 / 16l;
        auto v340 = cooperative_groups::labeled_partition(v337,v339);
        float v341;
        v341 = cooperative_groups::reduce(v340, v328, v42);
        float v342;
        v342 = v341 / 64.0f;
        float v343[4l];
        int v344;
        v344 = 0l;
        while (while_method_3(v344)){
            int v346;
            v346 = 0l;
            while (while_method_1(v346)){
                assert("Tensor range check" && 0 <= v344 && v344 < 1l);
                assert("Tensor range check" && 0 <= v346 && v346 < 4l);
                int v348;
                v348 = 4l * v344;
                int v349;
                v349 = v348 + v346;
                float v350;
                v350 = v283[v349];
                float v351;
                v351 = v350 - v342;
                float v352;
                v352 = exp(v351);
                assert("Tensor range check" && 0 <= v344 && v344 < 1l);
                assert("Tensor range check" && 0 <= v346 && v346 < 4l);
                v343[v349] = v352;
                v346 += 1l ;
            }
            v344 += 1l ;
        }
        float v353;
        v353 = 0.0f;
        int v354;
        v354 = 0l;
        while (while_method_3(v354)){
            int v356;
            v356 = 0l;
            while (while_method_1(v356)){
                assert("Tensor range check" && 0 <= v354 && v354 < 1l);
                assert("Tensor range check" && 0 <= v356 && v356 < 4l);
                int v358;
                v358 = 4l * v354;
                int v359;
                v359 = v358 + v356;
                float v360;
                v360 = v343[v359];
                float v361;
                v361 = v353 + v360;
                v353 = v361;
                v356 += 1l ;
            }
            v354 += 1l ;
        }
        auto v362 = cooperative_groups::coalesced_threads();
        int v363;
        v363 = threadIdx.x;
        int v364;
        v364 = v363 / 16l;
        auto v365 = cooperative_groups::labeled_partition(v362,v364);
        float v366;
        v366 = cooperative_groups::reduce(v365, v353, v42);
        float v367[4l];
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
                v374 = v343[v373];
                float v375;
                v375 = v374 / v366;
                assert("Tensor range check" && 0 <= v368 && v368 < 1l);
                assert("Tensor range check" && 0 <= v370 && v370 < 4l);
                v367[v373] = v375;
                v370 += 1l ;
            }
            v368 += 1l ;
        }
        assert("Tensor range check" && 0 <= v279 && v279 < 64l);
        int v376;
        v376 = 0l;
        while (while_method_3(v376)){
            assert("Tensor range check" && 0 <= v376 && v376 < 1l);
            int v378;
            v378 = 64l * v376;
            int v379;
            v379 = v378 + v282;
            assert("Tensor range check" && 0 <= v376 && v376 < 1l);
            int v380;
            v380 = 4l * v376;
            int4* v381;
            v381 = reinterpret_cast<int4*>(v367 + v380);
            int4* v382;
            v382 = reinterpret_cast<int4*>(v4 + v379);
            assert("Pointer alignment check" && (unsigned long long)(v381) % 4l == 0 && (unsigned long long)(v382) % 4l == 0);
            *v382 = *v381;
            v376 += 1l ;
        }
        v279 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v383;
    v383 = threadIdx.x;
    bool v384;
    v384 = 0l <= v383;
    bool v385;
    v385 = v384 == false;
    if (v385){
        assert("The index needs to be zero or positive." && v384);
    } else {
    }
    int v387;
    v387 = v383 % 16l;
    int v388;
    v388 = v383 / 16l;
    bool v389;
    v389 = v388 < 2l;
    bool v390;
    v390 = v389 == false;
    if (v390){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v389);
    } else {
    }
    assert("Tensor range check" && 0 <= v388 && v388 < 2l);
    assert("Tensor range check" && 0 <= v387 && v387 < 16l);
    int v392;
    v392 = 4l * v387;
    int v393;
    v393 = 64l * v388;
    int v394;
    v394 = v393 + v392;
    assert("Tensor range check" && 0 <= v388 && v388 < 2l);
    assert("Tensor range check" && 0 <= v387 && v387 < 16l);
    int v395;
    v395 = 0l;
    while (while_method_2(v395)){
        assert("Tensor range check" && 0 <= v395 && v395 < 64l);
        int v397;
        v397 = 128l * v395;
        int v398;
        v398 = v397 + v394;
        float v399[4l];
        int v400[4l];
        int v401;
        v401 = 0l;
        while (while_method_3(v401)){
            assert("Tensor range check" && 0 <= v401 && v401 < 1l);
            int v403;
            v403 = 4l * v401;
            assert("Tensor range check" && 0 <= v401 && v401 < 1l);
            int v404;
            v404 = 64l * v401;
            int v405;
            v405 = v404 + v398;
            int4* v406;
            v406 = reinterpret_cast<int4*>(v1 + v405);
            int4* v407;
            v407 = reinterpret_cast<int4*>(v399 + v403);
            assert("Pointer alignment check" && (unsigned long long)(v406) % 4l == 0 && (unsigned long long)(v407) % 4l == 0);
            *v407 = *v406;
            v401 += 1l ;
        }
        int v408;
        v408 = 0l;
        while (while_method_3(v408)){
            int v410;
            v410 = 0l;
            while (while_method_1(v410)){
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
                v417 = 0l <= v387;
                bool v419;
                if (v417){
                    bool v418;
                    v418 = v387 < 16l;
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
                v422 = v387 * 4l;
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
                v429 = v408 * 64l;
                int v430;
                v430 = v423 + v429;
                assert("Tensor range check" && 0 <= v408 && v408 < 1l);
                assert("Tensor range check" && 0 <= v410 && v410 < 4l);
                int v431;
                v431 = 4l * v408;
                int v432;
                v432 = v431 + v410;
                v400[v432] = v430;
                v410 += 1l ;
            }
            v408 += 1l ;
        }
        bool v433;
        v433 = 0l <= v388;
        bool v434;
        v434 = v433 && v389;
        bool v435;
        v435 = v434 == false;
        if (v435){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v434);
        } else {
        }
        bool v437;
        v437 = 0l <= v395;
        bool v439;
        if (v437){
            bool v438;
            v438 = v395 < 64l;
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
        v442 = v395 * 2l;
        int v443;
        v443 = v442 + v388;
        float v444[4l];
        int v445;
        v445 = 0l;
        while (while_method_3(v445)){
            int v447;
            v447 = 0l;
            while (while_method_1(v447)){
                assert("Tensor range check" && 0 <= v445 && v445 < 1l);
                assert("Tensor range check" && 0 <= v447 && v447 < 4l);
                int v449;
                v449 = 4l * v445;
                int v450;
                v450 = v449 + v447;
                float v451;
                v451 = v399[v450];
                float v452;
                v452 = v451 * v451;
                assert("Tensor range check" && 0 <= v445 && v445 < 1l);
                assert("Tensor range check" && 0 <= v447 && v447 < 4l);
                v444[v450] = v452;
                v447 += 1l ;
            }
            v445 += 1l ;
        }
        float v453;
        v453 = 0.0f;
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
                v460 = v444[v459];
                float v461;
                v461 = v453 + v460;
                v453 = v461;
                v456 += 1l ;
            }
            v454 += 1l ;
        }
        auto v462 = cooperative_groups::coalesced_threads();
        int v463;
        v463 = threadIdx.x;
        int v464;
        v464 = v463 / 16l;
        auto v465 = cooperative_groups::labeled_partition(v462,v464);
        float v466;
        v466 = cooperative_groups::reduce(v465, v453, v42);
        float v467[4l];
        int v468;
        v468 = 0l;
        while (while_method_3(v468)){
            int v470;
            v470 = 0l;
            while (while_method_1(v470)){
                assert("Tensor range check" && 0 <= v468 && v468 < 1l);
                assert("Tensor range check" && 0 <= v470 && v470 < 4l);
                int v472;
                v472 = 4l * v468;
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
                assert("Tensor range check" && 0 <= v468 && v468 < 1l);
                assert("Tensor range check" && 0 <= v470 && v470 < 4l);
                v467[v473] = v478;
                v470 += 1l ;
            }
            v468 += 1l ;
        }
        assert("Tensor range check" && 0 <= v395 && v395 < 64l);
        int v479;
        v479 = 0l;
        while (while_method_3(v479)){
            assert("Tensor range check" && 0 <= v479 && v479 < 1l);
            int v481;
            v481 = 64l * v479;
            int v482;
            v482 = v481 + v398;
            assert("Tensor range check" && 0 <= v479 && v479 < 1l);
            int v483;
            v483 = 4l * v479;
            int4* v484;
            v484 = reinterpret_cast<int4*>(v467 + v483);
            int4* v485;
            v485 = reinterpret_cast<int4*>(v8 + v482);
            assert("Pointer alignment check" && (unsigned long long)(v484) % 4l == 0 && (unsigned long long)(v485) % 4l == 0);
            *v485 = *v484;
            v479 += 1l ;
        }
        v395 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v486;
    v486 = threadIdx.x;
    bool v487;
    v487 = 0l <= v486;
    bool v488;
    v488 = v487 == false;
    if (v488){
        assert("The index needs to be zero or positive." && v487);
    } else {
    }
    int v490;
    v490 = v486 % 16l;
    int v491;
    v491 = v486 / 16l;
    bool v492;
    v492 = v491 < 2l;
    bool v493;
    v493 = v492 == false;
    if (v493){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v492);
    } else {
    }
    assert("Tensor range check" && 0 <= v491 && v491 < 2l);
    assert("Tensor range check" && 0 <= v490 && v490 < 16l);
    int v495;
    v495 = 4l * v490;
    int v496;
    v496 = 64l * v491;
    int v497;
    v497 = v496 + v495;
    assert("Tensor range check" && 0 <= v491 && v491 < 2l);
    int v498;
    v498 = 0l;
    while (while_method_2(v498)){
        assert("Tensor range check" && 0 <= v498 && v498 < 64l);
        int v500;
        v500 = 128l * v498;
        int v501;
        v501 = v500 + v497;
        float v502[4l];
        int v503[4l];
        int v504;
        v504 = 0l;
        while (while_method_3(v504)){
            assert("Tensor range check" && 0 <= v504 && v504 < 1l);
            int v506;
            v506 = 4l * v504;
            assert("Tensor range check" && 0 <= v504 && v504 < 1l);
            int v507;
            v507 = 64l * v504;
            int v508;
            v508 = v507 + v501;
            int4* v509;
            v509 = reinterpret_cast<int4*>(v1 + v508);
            int4* v510;
            v510 = reinterpret_cast<int4*>(v502 + v506);
            assert("Pointer alignment check" && (unsigned long long)(v509) % 4l == 0 && (unsigned long long)(v510) % 4l == 0);
            *v510 = *v509;
            v504 += 1l ;
        }
        int v511;
        v511 = 0l;
        while (while_method_3(v511)){
            int v513;
            v513 = 0l;
            while (while_method_1(v513)){
                bool v515;
                v515 = 0l <= v513;
                bool v517;
                if (v515){
                    bool v516;
                    v516 = v513 < 4l;
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
                v520 = 0l <= v490;
                bool v522;
                if (v520){
                    bool v521;
                    v521 = v490 < 16l;
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
                v525 = v490 * 4l;
                int v526;
                v526 = v513 + v525;
                bool v527;
                v527 = 0l <= v511;
                bool v529;
                if (v527){
                    bool v528;
                    v528 = v511 < 1l;
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
                v532 = v511 * 64l;
                int v533;
                v533 = v526 + v532;
                assert("Tensor range check" && 0 <= v511 && v511 < 1l);
                assert("Tensor range check" && 0 <= v513 && v513 < 4l);
                int v534;
                v534 = 4l * v511;
                int v535;
                v535 = v534 + v513;
                v503[v535] = v533;
                v513 += 1l ;
            }
            v511 += 1l ;
        }
        bool v536;
        v536 = 0l <= v491;
        bool v537;
        v537 = v536 && v492;
        bool v538;
        v538 = v537 == false;
        if (v538){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v537);
        } else {
        }
        bool v540;
        v540 = 0l <= v498;
        bool v542;
        if (v540){
            bool v541;
            v541 = v498 < 64l;
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
        v545 = v498 * 2l;
        int v546;
        v546 = v545 + v491;
        float v547; int v548;
        Tuple1 tmp24 = Tuple1{-1.0f / 0.0f, 0l};
        v547 = tmp24.v0; v548 = tmp24.v1;
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
                v551 += 1l ;
            }
            v549 += 1l ;
        }
        auto v560 = cooperative_groups::coalesced_threads();
        int v561;
        v561 = threadIdx.x;
        int v562;
        v562 = v561 / 16l;
        auto v563 = cooperative_groups::labeled_partition(v560,v562);
        Closure1 v564{};
        float v565; int v566;
        Tuple1 tmp25 = cooperative_groups::reduce(v563, Tuple1{v547, v548}, v564);
        v565 = tmp25.v0; v566 = tmp25.v1;
        assert("Tensor range check" && 0 <= v498 && v498 < 64l);
        int v567;
        v567 = 2l * v498;
        int v568;
        v568 = v567 + v491;
        v9[v568] = v566;
        v498 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v569;
    v569 = threadIdx.x;
    bool v570;
    v570 = 0l <= v569;
    bool v571;
    v571 = v570 == false;
    if (v571){
        assert("The index needs to be zero or positive." && v570);
    } else {
    }
    int v573;
    v573 = v569 % 16l;
    int v574;
    v574 = v569 / 16l;
    bool v575;
    v575 = v574 < 2l;
    bool v576;
    v576 = v575 == false;
    if (v576){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v575);
    } else {
    }
    assert("Tensor range check" && 0 <= v574 && v574 < 2l);
    assert("Tensor range check" && 0 <= v573 && v573 < 16l);
    int v578;
    v578 = 4l * v573;
    int v579;
    v579 = 64l * v574;
    int v580;
    v580 = v579 + v578;
    assert("Tensor range check" && 0 <= v574 && v574 < 2l);
    assert("Tensor range check" && 0 <= v573 && v573 < 16l);
    int v581;
    v581 = 0l;
    while (while_method_2(v581)){
        assert("Tensor range check" && 0 <= v581 && v581 < 64l);
        int v583;
        v583 = 128l * v581;
        int v584;
        v584 = v583 + v580;
        float v585[4l];
        int v586[4l];
        int v587;
        v587 = 0l;
        while (while_method_3(v587)){
            assert("Tensor range check" && 0 <= v587 && v587 < 1l);
            int v589;
            v589 = 4l * v587;
            assert("Tensor range check" && 0 <= v587 && v587 < 1l);
            int v590;
            v590 = 64l * v587;
            int v591;
            v591 = v590 + v584;
            int4* v592;
            v592 = reinterpret_cast<int4*>(v1 + v591);
            int4* v593;
            v593 = reinterpret_cast<int4*>(v585 + v589);
            assert("Pointer alignment check" && (unsigned long long)(v592) % 4l == 0 && (unsigned long long)(v593) % 4l == 0);
            *v593 = *v592;
            v587 += 1l ;
        }
        int v594;
        v594 = 0l;
        while (while_method_3(v594)){
            int v596;
            v596 = 0l;
            while (while_method_1(v596)){
                bool v598;
                v598 = 0l <= v596;
                bool v600;
                if (v598){
                    bool v599;
                    v599 = v596 < 4l;
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
                v603 = 0l <= v573;
                bool v605;
                if (v603){
                    bool v604;
                    v604 = v573 < 16l;
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
                v608 = v573 * 4l;
                int v609;
                v609 = v596 + v608;
                bool v610;
                v610 = 0l <= v594;
                bool v612;
                if (v610){
                    bool v611;
                    v611 = v594 < 1l;
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
                v615 = v594 * 64l;
                int v616;
                v616 = v609 + v615;
                assert("Tensor range check" && 0 <= v594 && v594 < 1l);
                assert("Tensor range check" && 0 <= v596 && v596 < 4l);
                int v617;
                v617 = 4l * v594;
                int v618;
                v618 = v617 + v596;
                v586[v618] = v616;
                v596 += 1l ;
            }
            v594 += 1l ;
        }
        bool v619;
        v619 = 0l <= v574;
        bool v620;
        v620 = v619 && v575;
        bool v621;
        v621 = v620 == false;
        if (v621){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v620);
        } else {
        }
        bool v623;
        v623 = 0l <= v581;
        bool v625;
        if (v623){
            bool v624;
            v624 = v581 < 64l;
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
        v628 = v581 * 2l;
        int v629;
        v629 = v628 + v574;
        float v630;
        v630 = 0.0f;
        int v631;
        v631 = 0l;
        while (while_method_3(v631)){
            int v633;
            v633 = 0l;
            while (while_method_1(v633)){
                assert("Tensor range check" && 0 <= v631 && v631 < 1l);
                assert("Tensor range check" && 0 <= v633 && v633 < 4l);
                int v635;
                v635 = 4l * v631;
                int v636;
                v636 = v635 + v633;
                float v637;
                v637 = v585[v636];
                float v638;
                v638 = v630 + v637;
                v630 = v638;
                v633 += 1l ;
            }
            v631 += 1l ;
        }
        auto v639 = cooperative_groups::coalesced_threads();
        int v640;
        v640 = threadIdx.x;
        int v641;
        v641 = v640 / 16l;
        auto v642 = cooperative_groups::labeled_partition(v639,v641);
        float v643;
        v643 = cooperative_groups::reduce(v642, v630, v42);
        float v644;
        v644 = v643 / 64.0f;
        float v645[4l];
        int v646;
        v646 = 0l;
        while (while_method_3(v646)){
            int v648;
            v648 = 0l;
            while (while_method_1(v648)){
                assert("Tensor range check" && 0 <= v646 && v646 < 1l);
                assert("Tensor range check" && 0 <= v648 && v648 < 4l);
                int v650;
                v650 = 4l * v646;
                int v651;
                v651 = v650 + v648;
                float v652;
                v652 = v585[v651];
                float v653;
                v653 = v652 - v644;
                float v654;
                v654 = exp(v653);
                assert("Tensor range check" && 0 <= v646 && v646 < 1l);
                assert("Tensor range check" && 0 <= v648 && v648 < 4l);
                v645[v651] = v654;
                v648 += 1l ;
            }
            v646 += 1l ;
        }
        float v655;
        v655 = 0.0f;
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
                v662 = v645[v661];
                float v663;
                v663 = v655 + v662;
                v655 = v663;
                v658 += 1l ;
            }
            v656 += 1l ;
        }
        auto v664 = cooperative_groups::coalesced_threads();
        int v665;
        v665 = threadIdx.x;
        int v666;
        v666 = v665 / 16l;
        auto v667 = cooperative_groups::labeled_partition(v664,v666);
        float v668;
        v668 = cooperative_groups::reduce(v667, v655, v42);
        float v669[4l];
        int v670;
        v670 = 0l;
        while (while_method_3(v670)){
            int v672;
            v672 = 0l;
            while (while_method_1(v672)){
                assert("Tensor range check" && 0 <= v670 && v670 < 1l);
                assert("Tensor range check" && 0 <= v672 && v672 < 4l);
                int v674;
                v674 = 4l * v670;
                int v675;
                v675 = v674 + v672;
                float v676;
                v676 = v645[v675];
                float v677;
                v677 = v676 / v668;
                assert("Tensor range check" && 0 <= v670 && v670 < 1l);
                assert("Tensor range check" && 0 <= v672 && v672 < 4l);
                v669[v675] = v677;
                v672 += 1l ;
            }
            v670 += 1l ;
        }
        float v678[4l];
        float v679;
        v679 = 0.0f;
        int v680;
        v680 = 0l;
        while (while_method_3(v680)){
            assert("Tensor range check" && 0 <= v680 && v680 < 1l);
            int v682;
            v682 = 4l * v680;
            assert("Tensor range check" && 0 <= v680 && v680 < 1l);
            int v683; float v684;
            Tuple0 tmp26 = Tuple0{0l, 0.0f};
            v683 = tmp26.v0; v684 = tmp26.v1;
            while (while_method_1(v683)){
                assert("Tensor range check" && 0 <= v683 && v683 < 4l);
                int v686;
                v686 = v683 + v682;
                float v687;
                v687 = v669[v686];
                float v688;
                v688 = v684 + v687;
                v684 = v688;
                v683 += 1l ;
            }
            auto v689 = cooperative_groups::coalesced_threads();
            int v690;
            v690 = threadIdx.x;
            int v691;
            v691 = v690 / 16l;
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
            Tuple0 tmp27 = Tuple0{0l, v699};
            v700 = tmp27.v0; v701 = tmp27.v1;
            while (while_method_1(v700)){
                assert("Tensor range check" && 0 <= v700 && v700 < 4l);
                int v703;
                v703 = v700 + v682;
                float v704;
                v704 = v669[v703];
                float v705;
                v705 = v701 + v704;
                assert("Tensor range check" && 0 <= v700 && v700 < 4l);
                v678[v703] = v705;
                v701 = v705;
                v700 += 1l ;
            }
            float v706;
            v706 = v679 + v698;
            v679 = v706;
            v680 += 1l ;
        }
        assert("Tensor range check" && 0 <= v581 && v581 < 64l);
        int v707;
        v707 = 0l;
        while (while_method_3(v707)){
            assert("Tensor range check" && 0 <= v707 && v707 < 1l);
            int v709;
            v709 = 64l * v707;
            int v710;
            v710 = v709 + v584;
            assert("Tensor range check" && 0 <= v707 && v707 < 1l);
            int v711;
            v711 = 4l * v707;
            int4* v712;
            v712 = reinterpret_cast<int4*>(v669 + v711);
            int4* v713;
            v713 = reinterpret_cast<int4*>(v6 + v710);
            assert("Pointer alignment check" && (unsigned long long)(v712) % 4l == 0 && (unsigned long long)(v713) % 4l == 0);
            *v713 = *v712;
            int4* v714;
            v714 = reinterpret_cast<int4*>(v678 + v711);
            int4* v715;
            v715 = reinterpret_cast<int4*>(v7 + v710);
            assert("Pointer alignment check" && (unsigned long long)(v714) % 4l == 0 && (unsigned long long)(v715) % 4l == 0);
            *v715 = *v714;
            v707 += 1l ;
        }
        v581 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v716;
    v716 = threadIdx.x;
    bool v717;
    v717 = 0l <= v716;
    bool v718;
    v718 = v717 == false;
    if (v718){
        assert("The index needs to be zero or positive." && v717);
    } else {
    }
    int v720;
    v720 = v716 % 16l;
    int v721;
    v721 = v716 / 16l;
    bool v722;
    v722 = v721 < 2l;
    bool v723;
    v723 = v722 == false;
    if (v723){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v722);
    } else {
    }
    assert("Tensor range check" && 0 <= v721 && v721 < 2l);
    assert("Tensor range check" && 0 <= v720 && v720 < 16l);
    int v725;
    v725 = 4l * v720;
    int v726;
    v726 = 64l * v721;
    int v727;
    v727 = v726 + v725;
    assert("Tensor range check" && 0 <= v721 && v721 < 2l);
    assert("Tensor range check" && 0 <= v720 && v720 < 16l);
    int v728;
    v728 = 0l;
    while (while_method_2(v728)){
        assert("Tensor range check" && 0 <= v728 && v728 < 64l);
        int v730;
        v730 = 128l * v728;
        int v731;
        v731 = v730 + v727;
        int v732[4l];
        int v733[4l];
        int v734;
        v734 = 0l;
        while (while_method_3(v734)){
            assert("Tensor range check" && 0 <= v734 && v734 < 1l);
            int v736;
            v736 = 4l * v734;
            assert("Tensor range check" && 0 <= v734 && v734 < 1l);
            int v737;
            v737 = 64l * v734;
            int v738;
            v738 = v737 + v731;
            int4* v739;
            v739 = reinterpret_cast<int4*>(v0 + v738);
            int4* v740;
            v740 = reinterpret_cast<int4*>(v732 + v736);
            assert("Pointer alignment check" && (unsigned long long)(v739) % 4l == 0 && (unsigned long long)(v740) % 4l == 0);
            *v740 = *v739;
            v734 += 1l ;
        }
        int v741;
        v741 = 0l;
        while (while_method_3(v741)){
            int v743;
            v743 = 0l;
            while (while_method_1(v743)){
                bool v745;
                v745 = 0l <= v743;
                bool v747;
                if (v745){
                    bool v746;
                    v746 = v743 < 4l;
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
                v750 = 0l <= v720;
                bool v752;
                if (v750){
                    bool v751;
                    v751 = v720 < 16l;
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
                v755 = v720 * 4l;
                int v756;
                v756 = v743 + v755;
                bool v757;
                v757 = 0l <= v741;
                bool v759;
                if (v757){
                    bool v758;
                    v758 = v741 < 1l;
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
                v762 = v741 * 64l;
                int v763;
                v763 = v756 + v762;
                assert("Tensor range check" && 0 <= v741 && v741 < 1l);
                assert("Tensor range check" && 0 <= v743 && v743 < 4l);
                int v764;
                v764 = 4l * v741;
                int v765;
                v765 = v764 + v743;
                v733[v765] = v763;
                v743 += 1l ;
            }
            v741 += 1l ;
        }
        bool v766;
        v766 = 0l <= v721;
        bool v767;
        v767 = v766 && v722;
        bool v768;
        v768 = v767 == false;
        if (v768){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v767);
        } else {
        }
        bool v770;
        v770 = 0l <= v728;
        bool v772;
        if (v770){
            bool v771;
            v771 = v728 < 64l;
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
        v775 = v728 * 2l;
        int v776;
        v776 = v775 + v721;
        int v777[4l];
        int v778;
        v778 = 0l;
        int v779;
        v779 = 0l;
        while (while_method_3(v779)){
            assert("Tensor range check" && 0 <= v779 && v779 < 1l);
            int v781;
            v781 = 4l * v779;
            assert("Tensor range check" && 0 <= v779 && v779 < 1l);
            int v782; int v783;
            Tuple2 tmp28 = Tuple2{0l, 0l};
            v782 = tmp28.v0; v783 = tmp28.v1;
            while (while_method_1(v782)){
                assert("Tensor range check" && 0 <= v782 && v782 < 4l);
                int v785;
                v785 = v782 + v781;
                int v786;
                v786 = v732[v785];
                int v787;
                v787 = v783 + v786;
                v783 = v787;
                v782 += 1l ;
            }
            auto v788 = cooperative_groups::coalesced_threads();
            int v789;
            v789 = threadIdx.x;
            int v790;
            v790 = v789 / 16l;
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
                v796 = 0l;
            } else {
                v796 = v794;
            }
            int v797;
            v797 = v791.shfl(v793,v791.num_threads()-1);
            int v798;
            v798 = v778 + v796;
            int v799; int v800;
            Tuple2 tmp29 = Tuple2{0l, v798};
            v799 = tmp29.v0; v800 = tmp29.v1;
            while (while_method_1(v799)){
                assert("Tensor range check" && 0 <= v799 && v799 < 4l);
                int v802;
                v802 = v799 + v781;
                int v803;
                v803 = v732[v802];
                assert("Tensor range check" && 0 <= v799 && v799 < 4l);
                v777[v802] = v800;
                int v804;
                v804 = v800 + v803;
                v800 = v804;
                v799 += 1l ;
            }
            int v805;
            v805 = v778 + v797;
            v778 = v805;
            v779 += 1l ;
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 64l);
        int v806;
        v806 = 0l;
        while (while_method_3(v806)){
            assert("Tensor range check" && 0 <= v806 && v806 < 1l);
            int v808;
            v808 = 64l * v806;
            int v809;
            v809 = v808 + v731;
            assert("Tensor range check" && 0 <= v806 && v806 < 1l);
            int v810;
            v810 = 4l * v806;
            int4* v811;
            v811 = reinterpret_cast<int4*>(v777 + v810);
            int4* v812;
            v812 = reinterpret_cast<int4*>(v13 + v809);
            assert("Pointer alignment check" && (unsigned long long)(v811) % 4l == 0 && (unsigned long long)(v812) % 4l == 0);
            *v812 = *v811;
            v806 += 1l ;
        }
        v728 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v813;
    v813 = threadIdx.x;
    bool v814;
    v814 = 0l <= v813;
    bool v815;
    v815 = v814 == false;
    if (v815){
        assert("The index needs to be zero or positive." && v814);
    } else {
    }
    int v817;
    v817 = v813 % 16l;
    int v818;
    v818 = v813 / 16l;
    bool v819;
    v819 = v818 < 2l;
    bool v820;
    v820 = v819 == false;
    if (v820){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v819);
    } else {
    }
    assert("Tensor range check" && 0 <= v818 && v818 < 2l);
    assert("Tensor range check" && 0 <= v817 && v817 < 16l);
    int v822;
    v822 = 4l * v817;
    int v823;
    v823 = 64l * v818;
    int v824;
    v824 = v823 + v822;
    assert("Tensor range check" && 0 <= v818 && v818 < 2l);
    assert("Tensor range check" && 0 <= v817 && v817 < 16l);
    int v825;
    v825 = 0l;
    while (while_method_2(v825)){
        assert("Tensor range check" && 0 <= v825 && v825 < 64l);
        int v827;
        v827 = 128l * v825;
        int v828;
        v828 = v827 + v824;
        float v829[4l];
        int v830[4l];
        int v831;
        v831 = 0l;
        while (while_method_3(v831)){
            assert("Tensor range check" && 0 <= v831 && v831 < 1l);
            int v833;
            v833 = 4l * v831;
            assert("Tensor range check" && 0 <= v831 && v831 < 1l);
            int v834;
            v834 = 64l * v831;
            int v835;
            v835 = v834 + v828;
            int4* v836;
            v836 = reinterpret_cast<int4*>(v1 + v835);
            int4* v837;
            v837 = reinterpret_cast<int4*>(v829 + v833);
            assert("Pointer alignment check" && (unsigned long long)(v836) % 4l == 0 && (unsigned long long)(v837) % 4l == 0);
            *v837 = *v836;
            v831 += 1l ;
        }
        int v838;
        v838 = 0l;
        while (while_method_3(v838)){
            int v840;
            v840 = 0l;
            while (while_method_1(v840)){
                bool v842;
                v842 = 0l <= v840;
                bool v844;
                if (v842){
                    bool v843;
                    v843 = v840 < 4l;
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
                v847 = 0l <= v817;
                bool v849;
                if (v847){
                    bool v848;
                    v848 = v817 < 16l;
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
                v852 = v817 * 4l;
                int v853;
                v853 = v840 + v852;
                bool v854;
                v854 = 0l <= v838;
                bool v856;
                if (v854){
                    bool v855;
                    v855 = v838 < 1l;
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
                v859 = v838 * 64l;
                int v860;
                v860 = v853 + v859;
                assert("Tensor range check" && 0 <= v838 && v838 < 1l);
                assert("Tensor range check" && 0 <= v840 && v840 < 4l);
                int v861;
                v861 = 4l * v838;
                int v862;
                v862 = v861 + v840;
                v830[v862] = v860;
                v840 += 1l ;
            }
            v838 += 1l ;
        }
        bool v863;
        v863 = 0l <= v818;
        bool v864;
        v864 = v863 && v819;
        bool v865;
        v865 = v864 == false;
        if (v865){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v864);
        } else {
        }
        bool v867;
        v867 = 0l <= v825;
        bool v869;
        if (v867){
            bool v868;
            v868 = v825 < 64l;
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
        v872 = v825 * 2l;
        int v873;
        v873 = v872 + v818;
        bool v874[4l];
        int v875;
        v875 = 0l;
        while (while_method_3(v875)){
            int v877;
            v877 = 0l;
            while (while_method_1(v877)){
                assert("Tensor range check" && 0 <= v875 && v875 < 1l);
                assert("Tensor range check" && 0 <= v877 && v877 < 4l);
                int v879;
                v879 = 4l * v875;
                int v880;
                v880 = v879 + v877;
                float v881;
                v881 = v829[v880];
                int v882;
                v882 = v830[v880];
                bool v883;
                v883 = v882 < 4l;
                assert("Tensor range check" && 0 <= v875 && v875 < 1l);
                assert("Tensor range check" && 0 <= v877 && v877 < 4l);
                v874[v880] = v883;
                v877 += 1l ;
            }
            v875 += 1l ;
        }
        int v884[4l];
        int v885;
        v885 = 0l;
        while (while_method_3(v885)){
            int v887;
            v887 = 0l;
            while (while_method_1(v887)){
                assert("Tensor range check" && 0 <= v885 && v885 < 1l);
                assert("Tensor range check" && 0 <= v887 && v887 < 4l);
                int v889;
                v889 = 4l * v885;
                int v890;
                v890 = v889 + v887;
                bool v891;
                v891 = v874[v890];
                int v892;
                if (v891){
                    v892 = 1l;
                } else {
                    v892 = 0l;
                }
                assert("Tensor range check" && 0 <= v885 && v885 < 1l);
                assert("Tensor range check" && 0 <= v887 && v887 < 4l);
                v884[v890] = v892;
                v887 += 1l ;
            }
            v885 += 1l ;
        }
        int v893;
        v893 = 0l;
        int v894;
        v894 = 0l;
        while (while_method_3(v894)){
            int v896;
            v896 = 0l;
            while (while_method_1(v896)){
                assert("Tensor range check" && 0 <= v894 && v894 < 1l);
                assert("Tensor range check" && 0 <= v896 && v896 < 4l);
                int v898;
                v898 = 4l * v894;
                int v899;
                v899 = v898 + v896;
                int v900;
                v900 = v884[v899];
                int v901;
                v901 = v893 + v900;
                v893 = v901;
                v896 += 1l ;
            }
            v894 += 1l ;
        }
        auto v902 = cooperative_groups::coalesced_threads();
        int v903;
        v903 = threadIdx.x;
        int v904;
        v904 = v903 / 16l;
        auto v905 = cooperative_groups::labeled_partition(v902,v904);
        Closure4 v906{};
        int v907;
        v907 = cooperative_groups::reduce(v905, v893, v906);
        float v908[4l];
        int v909;
        v909 = 0l;
        while (while_method_3(v909)){
            int v911;
            v911 = 0l;
            while (while_method_1(v911)){
                assert("Tensor range check" && 0 <= v909 && v909 < 1l);
                assert("Tensor range check" && 0 <= v911 && v911 < 4l);
                int v913;
                v913 = 4l * v909;
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
                assert("Tensor range check" && 0 <= v909 && v909 < 1l);
                assert("Tensor range check" && 0 <= v911 && v911 < 4l);
                v908[v914] = v917;
                v911 += 1l ;
            }
            v909 += 1l ;
        }
        float v918;
        v918 = 0.0f;
        int v919;
        v919 = 0l;
        while (while_method_3(v919)){
            int v921;
            v921 = 0l;
            while (while_method_1(v921)){
                assert("Tensor range check" && 0 <= v919 && v919 < 1l);
                assert("Tensor range check" && 0 <= v921 && v921 < 4l);
                int v923;
                v923 = 4l * v919;
                int v924;
                v924 = v923 + v921;
                float v925;
                v925 = v908[v924];
                float v926;
                v926 = v918 + v925;
                v918 = v926;
                v921 += 1l ;
            }
            v919 += 1l ;
        }
        auto v927 = cooperative_groups::coalesced_threads();
        int v928;
        v928 = threadIdx.x;
        int v929;
        v929 = v928 / 16l;
        auto v930 = cooperative_groups::labeled_partition(v927,v929);
        float v931;
        v931 = cooperative_groups::reduce(v930, v918, v42);
        float v932;
        v932 = (float)v907;
        float v933;
        v933 = v931 / v932;
        float v934[4l];
        int v935;
        v935 = 0l;
        while (while_method_3(v935)){
            int v937;
            v937 = 0l;
            while (while_method_1(v937)){
                assert("Tensor range check" && 0 <= v935 && v935 < 1l);
                assert("Tensor range check" && 0 <= v937 && v937 < 4l);
                int v939;
                v939 = 4l * v935;
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
        while (while_method_3(v947)){
            int v949;
            v949 = 0l;
            while (while_method_1(v949)){
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
        int v957;
        v957 = v956 / 16l;
        auto v958 = cooperative_groups::labeled_partition(v955,v957);
        float v959;
        v959 = cooperative_groups::reduce(v958, v946, v42);
        float v960[4l];
        int v961;
        v961 = 0l;
        while (while_method_3(v961)){
            int v963;
            v963 = 0l;
            while (while_method_1(v963)){
                assert("Tensor range check" && 0 <= v961 && v961 < 1l);
                assert("Tensor range check" && 0 <= v963 && v963 < 4l);
                int v965;
                v965 = 4l * v961;
                int v966;
                v966 = v965 + v963;
                float v967;
                v967 = v934[v966];
                float v968;
                v968 = v967 / v959;
                assert("Tensor range check" && 0 <= v961 && v961 < 1l);
                assert("Tensor range check" && 0 <= v963 && v963 < 4l);
                v960[v966] = v968;
                v963 += 1l ;
            }
            v961 += 1l ;
        }
        assert("Tensor range check" && 0 <= v825 && v825 < 64l);
        int v969;
        v969 = 0l;
        while (while_method_3(v969)){
            assert("Tensor range check" && 0 <= v969 && v969 < 1l);
            int v971;
            v971 = 64l * v969;
            int v972;
            v972 = v971 + v828;
            assert("Tensor range check" && 0 <= v969 && v969 < 1l);
            int v973;
            v973 = 4l * v969;
            int4* v974;
            v974 = reinterpret_cast<int4*>(v960 + v973);
            int4* v975;
            v975 = reinterpret_cast<int4*>(v5 + v972);
            assert("Pointer alignment check" && (unsigned long long)(v974) % 4l == 0 && (unsigned long long)(v975) % 4l == 0);
            *v975 = *v974;
            v969 += 1l ;
        }
        v825 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v976;
    v976 = threadIdx.x;
    int v977;
    v977 = blockIdx.x;
    int v978;
    v978 = v977 * 32l;
    int v979;
    v979 = v976 + v978;
    unsigned long long v980;
    v980 = (unsigned long long)v979;
    curandStatePhilox4_32_10_t v981;
    curand_init(12344321ull,v980,0ull,&v981);
    int v982;
    v982 = threadIdx.x;
    bool v983;
    v983 = 0l <= v982;
    bool v984;
    v984 = v983 == false;
    if (v984){
        assert("The index needs to be zero or positive." && v983);
    } else {
    }
    int v986;
    v986 = v982 % 16l;
    int v987;
    v987 = v982 / 16l;
    bool v988;
    v988 = v987 < 2l;
    bool v989;
    v989 = v988 == false;
    if (v989){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v988);
    } else {
    }
    assert("Tensor range check" && 0 <= v987 && v987 < 2l);
    assert("Tensor range check" && 0 <= v986 && v986 < 16l);
    int v991;
    v991 = 4l * v986;
    int v992;
    v992 = 64l * v987;
    int v993;
    v993 = v992 + v991;
    assert("Tensor range check" && 0 <= v987 && v987 < 2l);
    assert("Tensor range check" && 0 <= v986 && v986 < 16l);
    assert("Tensor range check" && 0 <= v987 && v987 < 2l);
    int v994;
    v994 = 0l;
    while (while_method_2(v994)){
        assert("Tensor range check" && 0 <= v994 && v994 < 64l);
        int v996;
        v996 = 128l * v994;
        int v997;
        v997 = v996 + v993;
        float v998[4l];
        int v999[4l];
        int v1000;
        v1000 = 0l;
        while (while_method_3(v1000)){
            assert("Tensor range check" && 0 <= v1000 && v1000 < 1l);
            int v1002;
            v1002 = 4l * v1000;
            assert("Tensor range check" && 0 <= v1000 && v1000 < 1l);
            int v1003;
            v1003 = 64l * v1000;
            int v1004;
            v1004 = v1003 + v997;
            int4* v1005;
            v1005 = reinterpret_cast<int4*>(v1 + v1004);
            int4* v1006;
            v1006 = reinterpret_cast<int4*>(v998 + v1002);
            assert("Pointer alignment check" && (unsigned long long)(v1005) % 4l == 0 && (unsigned long long)(v1006) % 4l == 0);
            *v1006 = *v1005;
            v1000 += 1l ;
        }
        int v1007;
        v1007 = 0l;
        while (while_method_3(v1007)){
            int v1009;
            v1009 = 0l;
            while (while_method_1(v1009)){
                bool v1011;
                v1011 = 0l <= v1009;
                bool v1013;
                if (v1011){
                    bool v1012;
                    v1012 = v1009 < 4l;
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
                v1016 = 0l <= v986;
                bool v1018;
                if (v1016){
                    bool v1017;
                    v1017 = v986 < 16l;
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
                v1021 = v986 * 4l;
                int v1022;
                v1022 = v1009 + v1021;
                bool v1023;
                v1023 = 0l <= v1007;
                bool v1025;
                if (v1023){
                    bool v1024;
                    v1024 = v1007 < 1l;
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
                v1028 = v1007 * 64l;
                int v1029;
                v1029 = v1022 + v1028;
                assert("Tensor range check" && 0 <= v1007 && v1007 < 1l);
                assert("Tensor range check" && 0 <= v1009 && v1009 < 4l);
                int v1030;
                v1030 = 4l * v1007;
                int v1031;
                v1031 = v1030 + v1009;
                v999[v1031] = v1029;
                v1009 += 1l ;
            }
            v1007 += 1l ;
        }
        bool v1032;
        v1032 = 0l <= v987;
        bool v1033;
        v1033 = v1032 && v988;
        bool v1034;
        v1034 = v1033 == false;
        if (v1034){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1033);
        } else {
        }
        bool v1036;
        v1036 = 0l <= v994;
        bool v1038;
        if (v1036){
            bool v1037;
            v1037 = v994 < 64l;
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
        v1041 = v994 * 2l;
        int v1042;
        v1042 = v1041 + v987;
        float v1043;
        v1043 = 0.0f;
        int v1044;
        v1044 = 0l;
        while (while_method_3(v1044)){
            int v1046;
            v1046 = 0l;
            while (while_method_1(v1046)){
                assert("Tensor range check" && 0 <= v1044 && v1044 < 1l);
                assert("Tensor range check" && 0 <= v1046 && v1046 < 4l);
                int v1048;
                v1048 = 4l * v1044;
                int v1049;
                v1049 = v1048 + v1046;
                float v1050;
                v1050 = v998[v1049];
                float v1051;
                v1051 = v1043 + v1050;
                v1043 = v1051;
                v1046 += 1l ;
            }
            v1044 += 1l ;
        }
        auto v1052 = cooperative_groups::coalesced_threads();
        int v1053;
        v1053 = threadIdx.x;
        int v1054;
        v1054 = v1053 / 16l;
        auto v1055 = cooperative_groups::labeled_partition(v1052,v1054);
        float v1056;
        v1056 = cooperative_groups::reduce(v1055, v1043, v42);
        float v1057;
        v1057 = v1056 / 64.0f;
        float v1058[4l];
        int v1059;
        v1059 = 0l;
        while (while_method_3(v1059)){
            int v1061;
            v1061 = 0l;
            while (while_method_1(v1061)){
                assert("Tensor range check" && 0 <= v1059 && v1059 < 1l);
                assert("Tensor range check" && 0 <= v1061 && v1061 < 4l);
                int v1063;
                v1063 = 4l * v1059;
                int v1064;
                v1064 = v1063 + v1061;
                float v1065;
                v1065 = v998[v1064];
                float v1066;
                v1066 = v1065 - v1057;
                float v1067;
                v1067 = exp(v1066);
                assert("Tensor range check" && 0 <= v1059 && v1059 < 1l);
                assert("Tensor range check" && 0 <= v1061 && v1061 < 4l);
                v1058[v1064] = v1067;
                v1061 += 1l ;
            }
            v1059 += 1l ;
        }
        float v1068;
        v1068 = 0.0f;
        int v1069;
        v1069 = 0l;
        while (while_method_3(v1069)){
            int v1071;
            v1071 = 0l;
            while (while_method_1(v1071)){
                assert("Tensor range check" && 0 <= v1069 && v1069 < 1l);
                assert("Tensor range check" && 0 <= v1071 && v1071 < 4l);
                int v1073;
                v1073 = 4l * v1069;
                int v1074;
                v1074 = v1073 + v1071;
                float v1075;
                v1075 = v1058[v1074];
                float v1076;
                v1076 = v1068 + v1075;
                v1068 = v1076;
                v1071 += 1l ;
            }
            v1069 += 1l ;
        }
        auto v1077 = cooperative_groups::coalesced_threads();
        int v1078;
        v1078 = threadIdx.x;
        int v1079;
        v1079 = v1078 / 16l;
        auto v1080 = cooperative_groups::labeled_partition(v1077,v1079);
        float v1081;
        v1081 = cooperative_groups::reduce(v1080, v1068, v42);
        float v1082[4l];
        int v1083;
        v1083 = 0l;
        while (while_method_3(v1083)){
            int v1085;
            v1085 = 0l;
            while (while_method_1(v1085)){
                assert("Tensor range check" && 0 <= v1083 && v1083 < 1l);
                assert("Tensor range check" && 0 <= v1085 && v1085 < 4l);
                int v1087;
                v1087 = 4l * v1083;
                int v1088;
                v1088 = v1087 + v1085;
                float v1089;
                v1089 = v1058[v1088];
                float v1090;
                v1090 = v1089 / v1081;
                assert("Tensor range check" && 0 <= v1083 && v1083 < 1l);
                assert("Tensor range check" && 0 <= v1085 && v1085 < 4l);
                v1082[v1088] = v1090;
                v1085 += 1l ;
            }
            v1083 += 1l ;
        }
        float v1091[4l];
        float v1092;
        v1092 = 0.0f;
        int v1093;
        v1093 = 0l;
        while (while_method_3(v1093)){
            assert("Tensor range check" && 0 <= v1093 && v1093 < 1l);
            int v1095;
            v1095 = 4l * v1093;
            assert("Tensor range check" && 0 <= v1093 && v1093 < 1l);
            int v1096; float v1097;
            Tuple0 tmp30 = Tuple0{0l, 0.0f};
            v1096 = tmp30.v0; v1097 = tmp30.v1;
            while (while_method_1(v1096)){
                assert("Tensor range check" && 0 <= v1096 && v1096 < 4l);
                int v1099;
                v1099 = v1096 + v1095;
                float v1100;
                v1100 = v1082[v1099];
                float v1101;
                v1101 = v1097 + v1100;
                v1097 = v1101;
                v1096 += 1l ;
            }
            auto v1102 = cooperative_groups::coalesced_threads();
            int v1103;
            v1103 = threadIdx.x;
            int v1104;
            v1104 = v1103 / 16l;
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
            Tuple0 tmp31 = Tuple0{0l, v1112};
            v1113 = tmp31.v0; v1114 = tmp31.v1;
            while (while_method_1(v1113)){
                assert("Tensor range check" && 0 <= v1113 && v1113 < 4l);
                int v1116;
                v1116 = v1113 + v1095;
                float v1117;
                v1117 = v1082[v1116];
                float v1118;
                v1118 = v1114 + v1117;
                assert("Tensor range check" && 0 <= v1113 && v1113 < 4l);
                v1091[v1116] = v1118;
                v1114 = v1118;
                v1113 += 1l ;
            }
            float v1119;
            v1119 = v1092 + v1111;
            v1092 = v1119;
            v1093 += 1l ;
        }
        float v1120[4l];
        bool v1121[4l];
        int v1122;
        v1122 = 0l;
        while (while_method_3(v1122)){
            int v1124;
            v1124 = 0l;
            while (while_method_1(v1124)){
                assert("Tensor range check" && 0 <= v1122 && v1122 < 1l);
                assert("Tensor range check" && 0 <= v1124 && v1124 < 4l);
                int v1126;
                v1126 = 4l * v1122;
                int v1127;
                v1127 = v1126 + v1124;
                float v1128;
                v1128 = v1091[v1127];
                float v1129;
                v1129 = v1082[v1127];
                bool v1130;
                v1130 = v1129 > 0.0f;
                assert("Tensor range check" && 0 <= v1122 && v1122 < 1l);
                assert("Tensor range check" && 0 <= v1124 && v1124 < 4l);
                v1120[v1127] = v1128;
                v1121[v1127] = v1130;
                v1124 += 1l ;
            }
            v1122 += 1l ;
        }
        float v1131; bool v1132;
        Tuple3 tmp32 = Tuple3{-1.0f / 0.0f, false};
        v1131 = tmp32.v0; v1132 = tmp32.v1;
        int v1133;
        v1133 = 0l;
        while (while_method_3(v1133)){
            int v1135;
            v1135 = 0l;
            while (while_method_1(v1135)){
                assert("Tensor range check" && 0 <= v1133 && v1133 < 1l);
                assert("Tensor range check" && 0 <= v1135 && v1135 < 4l);
                int v1137;
                v1137 = 4l * v1133;
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
                v1135 += 1l ;
            }
            v1133 += 1l ;
        }
        auto v1149 = cooperative_groups::coalesced_threads();
        int v1150;
        v1150 = threadIdx.x;
        int v1151;
        v1151 = v1150 / 16l;
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
        float v1158[4l];
        int v1159[4l];
        int v1160;
        v1160 = 0l;
        while (while_method_3(v1160)){
            int v1162;
            v1162 = 0l;
            while (while_method_1(v1162)){
                assert("Tensor range check" && 0 <= v1160 && v1160 < 1l);
                assert("Tensor range check" && 0 <= v1162 && v1162 < 4l);
                int v1164;
                v1164 = 4l * v1160;
                int v1165;
                v1165 = v1164 + v1162;
                int v1166;
                v1166 = v999[v1165];
                float v1167;
                v1167 = curand_uniform(&v981);
                assert("Tensor range check" && 0 <= v1160 && v1160 < 1l);
                assert("Tensor range check" && 0 <= v1162 && v1162 < 4l);
                v1158[v1165] = v1167;
                v1159[v1165] = v1166;
                v1162 += 1l ;
            }
            v1160 += 1l ;
        }
        float v1168; int v1169;
        Tuple1 tmp34 = Tuple1{0.0f, 2147483647l};
        v1168 = tmp34.v0; v1169 = tmp34.v1;
        int v1170;
        v1170 = 0l;
        while (while_method_3(v1170)){
            int v1172;
            v1172 = 0l;
            while (while_method_1(v1172)){
                assert("Tensor range check" && 0 <= v1170 && v1170 < 1l);
                assert("Tensor range check" && 0 <= v1172 && v1172 < 4l);
                int v1174;
                v1174 = 4l * v1170;
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
                v1172 += 1l ;
            }
            v1170 += 1l ;
        }
        auto v1181 = cooperative_groups::coalesced_threads();
        int v1182;
        v1182 = threadIdx.x;
        int v1183;
        v1183 = v1182 / 16l;
        auto v1184 = cooperative_groups::labeled_partition(v1181,v1183);
        Closure6 v1185{};
        float v1186; int v1187;
        Tuple1 tmp35 = cooperative_groups::reduce(v1184, Tuple1{v1168, v1169}, v1185);
        v1186 = tmp35.v0; v1187 = tmp35.v1;
        float v1188;
        v1188 = v1154 * v1186;
        int v1189[4l];
        bool v1190[4l];
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
                    v1202 = 2147483647l; v1203 = false;
                }
                assert("Tensor range check" && 0 <= v1191 && v1191 < 1l);
                assert("Tensor range check" && 0 <= v1193 && v1193 < 4l);
                v1189[v1196] = v1202;
                v1190[v1196] = v1203;
                v1193 += 1l ;
            }
            v1191 += 1l ;
        }
        int v1204; bool v1205;
        Tuple4 tmp36 = Tuple4{2147483647l, false};
        v1204 = tmp36.v0; v1205 = tmp36.v1;
        int v1206;
        v1206 = 0l;
        while (while_method_3(v1206)){
            int v1208;
            v1208 = 0l;
            while (while_method_1(v1208)){
                assert("Tensor range check" && 0 <= v1206 && v1206 < 1l);
                assert("Tensor range check" && 0 <= v1208 && v1208 < 4l);
                int v1210;
                v1210 = 4l * v1206;
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
                v1208 += 1l ;
            }
            v1206 += 1l ;
        }
        auto v1222 = cooperative_groups::coalesced_threads();
        int v1223;
        v1223 = threadIdx.x;
        int v1224;
        v1224 = v1223 / 16l;
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
        assert("Tensor range check" && 0 <= v994 && v994 < 64l);
        int v1231;
        v1231 = 0l;
        while (while_method_3(v1231)){
            assert("Tensor range check" && 0 <= v1231 && v1231 < 1l);
            int v1233;
            v1233 = 64l * v1231;
            int v1234;
            v1234 = v1233 + v997;
            assert("Tensor range check" && 0 <= v1231 && v1231 < 1l);
            int v1235;
            v1235 = 4l * v1231;
            int4* v1236;
            v1236 = reinterpret_cast<int4*>(v1082 + v1235);
            int4* v1237;
            v1237 = reinterpret_cast<int4*>(v14 + v1234);
            assert("Pointer alignment check" && (unsigned long long)(v1236) % 4l == 0 && (unsigned long long)(v1237) % 4l == 0);
            *v1237 = *v1236;
            v1231 += 1l ;
        }
        assert("Tensor range check" && 0 <= v994 && v994 < 64l);
        int v1238;
        v1238 = 2l * v994;
        int v1239;
        v1239 = v1238 + v987;
        v15[v1239] = v1227;
        v994 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1240;
    v1240 = threadIdx.x;
    int v1241;
    v1241 = blockIdx.x;
    int v1242;
    v1242 = v1241 * 32l;
    int v1243;
    v1243 = v1240 + v1242;
    unsigned long long v1244;
    v1244 = (unsigned long long)v1243;
    curandStatePhilox4_32_10_t v1245;
    curand_init(12344321ull,v1244,0ull,&v1245);
    int v1246;
    v1246 = threadIdx.x;
    bool v1247;
    v1247 = 0l <= v1246;
    bool v1248;
    v1248 = v1247 == false;
    if (v1248){
        assert("The index needs to be zero or positive." && v1247);
    } else {
    }
    int v1250;
    v1250 = v1246 % 16l;
    int v1251;
    v1251 = v1246 / 16l;
    bool v1252;
    v1252 = v1251 < 2l;
    bool v1253;
    v1253 = v1252 == false;
    if (v1253){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1252);
    } else {
    }
    assert("Tensor range check" && 0 <= v1251 && v1251 < 2l);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 16l);
    int v1255;
    v1255 = 4l * v1250;
    int v1256;
    v1256 = 64l * v1251;
    int v1257;
    v1257 = v1256 + v1255;
    assert("Tensor range check" && 0 <= v1251 && v1251 < 2l);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 16l);
    assert("Tensor range check" && 0 <= v1251 && v1251 < 2l);
    int v1258;
    v1258 = 0l;
    while (while_method_2(v1258)){
        assert("Tensor range check" && 0 <= v1258 && v1258 < 64l);
        int v1260;
        v1260 = 128l * v1258;
        int v1261;
        v1261 = v1260 + v1257;
        float v1262[4l];
        int v1263[4l];
        int v1264;
        v1264 = 0l;
        while (while_method_3(v1264)){
            assert("Tensor range check" && 0 <= v1264 && v1264 < 1l);
            int v1266;
            v1266 = 4l * v1264;
            assert("Tensor range check" && 0 <= v1264 && v1264 < 1l);
            int v1267;
            v1267 = 64l * v1264;
            int v1268;
            v1268 = v1267 + v1261;
            int4* v1269;
            v1269 = reinterpret_cast<int4*>(v1 + v1268);
            int4* v1270;
            v1270 = reinterpret_cast<int4*>(v1262 + v1266);
            assert("Pointer alignment check" && (unsigned long long)(v1269) % 4l == 0 && (unsigned long long)(v1270) % 4l == 0);
            *v1270 = *v1269;
            v1264 += 1l ;
        }
        int v1271;
        v1271 = 0l;
        while (while_method_3(v1271)){
            int v1273;
            v1273 = 0l;
            while (while_method_1(v1273)){
                bool v1275;
                v1275 = 0l <= v1273;
                bool v1277;
                if (v1275){
                    bool v1276;
                    v1276 = v1273 < 4l;
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
                v1280 = 0l <= v1250;
                bool v1282;
                if (v1280){
                    bool v1281;
                    v1281 = v1250 < 16l;
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
                v1285 = v1250 * 4l;
                int v1286;
                v1286 = v1273 + v1285;
                bool v1287;
                v1287 = 0l <= v1271;
                bool v1289;
                if (v1287){
                    bool v1288;
                    v1288 = v1271 < 1l;
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
                v1292 = v1271 * 64l;
                int v1293;
                v1293 = v1286 + v1292;
                assert("Tensor range check" && 0 <= v1271 && v1271 < 1l);
                assert("Tensor range check" && 0 <= v1273 && v1273 < 4l);
                int v1294;
                v1294 = 4l * v1271;
                int v1295;
                v1295 = v1294 + v1273;
                v1263[v1295] = v1293;
                v1273 += 1l ;
            }
            v1271 += 1l ;
        }
        bool v1296;
        v1296 = 0l <= v1251;
        bool v1297;
        v1297 = v1296 && v1252;
        bool v1298;
        v1298 = v1297 == false;
        if (v1298){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1297);
        } else {
        }
        bool v1300;
        v1300 = 0l <= v1258;
        bool v1302;
        if (v1300){
            bool v1301;
            v1301 = v1258 < 64l;
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
        v1305 = v1258 * 2l;
        int v1306;
        v1306 = v1305 + v1251;
        bool v1307[4l];
        int v1308;
        v1308 = 0l;
        while (while_method_3(v1308)){
            int v1310;
            v1310 = 0l;
            while (while_method_1(v1310)){
                assert("Tensor range check" && 0 <= v1308 && v1308 < 1l);
                assert("Tensor range check" && 0 <= v1310 && v1310 < 4l);
                int v1312;
                v1312 = 4l * v1308;
                int v1313;
                v1313 = v1312 + v1310;
                float v1314;
                v1314 = v1262[v1313];
                int v1315;
                v1315 = v1263[v1313];
                bool v1316;
                v1316 = v1315 < 11l;
                assert("Tensor range check" && 0 <= v1308 && v1308 < 1l);
                assert("Tensor range check" && 0 <= v1310 && v1310 < 4l);
                v1307[v1313] = v1316;
                v1310 += 1l ;
            }
            v1308 += 1l ;
        }
        int v1317[4l];
        int v1318;
        v1318 = 0l;
        while (while_method_3(v1318)){
            int v1320;
            v1320 = 0l;
            while (while_method_1(v1320)){
                assert("Tensor range check" && 0 <= v1318 && v1318 < 1l);
                assert("Tensor range check" && 0 <= v1320 && v1320 < 4l);
                int v1322;
                v1322 = 4l * v1318;
                int v1323;
                v1323 = v1322 + v1320;
                bool v1324;
                v1324 = v1307[v1323];
                int v1325;
                if (v1324){
                    v1325 = 1l;
                } else {
                    v1325 = 0l;
                }
                assert("Tensor range check" && 0 <= v1318 && v1318 < 1l);
                assert("Tensor range check" && 0 <= v1320 && v1320 < 4l);
                v1317[v1323] = v1325;
                v1320 += 1l ;
            }
            v1318 += 1l ;
        }
        int v1326;
        v1326 = 0l;
        int v1327;
        v1327 = 0l;
        while (while_method_3(v1327)){
            int v1329;
            v1329 = 0l;
            while (while_method_1(v1329)){
                assert("Tensor range check" && 0 <= v1327 && v1327 < 1l);
                assert("Tensor range check" && 0 <= v1329 && v1329 < 4l);
                int v1331;
                v1331 = 4l * v1327;
                int v1332;
                v1332 = v1331 + v1329;
                int v1333;
                v1333 = v1317[v1332];
                int v1334;
                v1334 = v1326 + v1333;
                v1326 = v1334;
                v1329 += 1l ;
            }
            v1327 += 1l ;
        }
        auto v1335 = cooperative_groups::coalesced_threads();
        int v1336;
        v1336 = threadIdx.x;
        int v1337;
        v1337 = v1336 / 16l;
        auto v1338 = cooperative_groups::labeled_partition(v1335,v1337);
        Closure4 v1339{};
        int v1340;
        v1340 = cooperative_groups::reduce(v1338, v1326, v1339);
        float v1341[4l];
        int v1342;
        v1342 = 0l;
        while (while_method_3(v1342)){
            int v1344;
            v1344 = 0l;
            while (while_method_1(v1344)){
                assert("Tensor range check" && 0 <= v1342 && v1342 < 1l);
                assert("Tensor range check" && 0 <= v1344 && v1344 < 4l);
                int v1346;
                v1346 = 4l * v1342;
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
                assert("Tensor range check" && 0 <= v1342 && v1342 < 1l);
                assert("Tensor range check" && 0 <= v1344 && v1344 < 4l);
                v1341[v1347] = v1350;
                v1344 += 1l ;
            }
            v1342 += 1l ;
        }
        float v1351;
        v1351 = 0.0f;
        int v1352;
        v1352 = 0l;
        while (while_method_3(v1352)){
            int v1354;
            v1354 = 0l;
            while (while_method_1(v1354)){
                assert("Tensor range check" && 0 <= v1352 && v1352 < 1l);
                assert("Tensor range check" && 0 <= v1354 && v1354 < 4l);
                int v1356;
                v1356 = 4l * v1352;
                int v1357;
                v1357 = v1356 + v1354;
                float v1358;
                v1358 = v1341[v1357];
                float v1359;
                v1359 = v1351 + v1358;
                v1351 = v1359;
                v1354 += 1l ;
            }
            v1352 += 1l ;
        }
        auto v1360 = cooperative_groups::coalesced_threads();
        int v1361;
        v1361 = threadIdx.x;
        int v1362;
        v1362 = v1361 / 16l;
        auto v1363 = cooperative_groups::labeled_partition(v1360,v1362);
        float v1364;
        v1364 = cooperative_groups::reduce(v1363, v1351, v42);
        float v1365;
        v1365 = (float)v1340;
        float v1366;
        v1366 = v1364 / v1365;
        float v1367[4l];
        int v1368;
        v1368 = 0l;
        while (while_method_3(v1368)){
            int v1370;
            v1370 = 0l;
            while (while_method_1(v1370)){
                assert("Tensor range check" && 0 <= v1368 && v1368 < 1l);
                assert("Tensor range check" && 0 <= v1370 && v1370 < 4l);
                int v1372;
                v1372 = 4l * v1368;
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
                assert("Tensor range check" && 0 <= v1368 && v1368 < 1l);
                assert("Tensor range check" && 0 <= v1370 && v1370 < 4l);
                v1367[v1373] = v1378;
                v1370 += 1l ;
            }
            v1368 += 1l ;
        }
        float v1379;
        v1379 = 0.0f;
        int v1380;
        v1380 = 0l;
        while (while_method_3(v1380)){
            int v1382;
            v1382 = 0l;
            while (while_method_1(v1382)){
                assert("Tensor range check" && 0 <= v1380 && v1380 < 1l);
                assert("Tensor range check" && 0 <= v1382 && v1382 < 4l);
                int v1384;
                v1384 = 4l * v1380;
                int v1385;
                v1385 = v1384 + v1382;
                float v1386;
                v1386 = v1367[v1385];
                float v1387;
                v1387 = v1379 + v1386;
                v1379 = v1387;
                v1382 += 1l ;
            }
            v1380 += 1l ;
        }
        auto v1388 = cooperative_groups::coalesced_threads();
        int v1389;
        v1389 = threadIdx.x;
        int v1390;
        v1390 = v1389 / 16l;
        auto v1391 = cooperative_groups::labeled_partition(v1388,v1390);
        float v1392;
        v1392 = cooperative_groups::reduce(v1391, v1379, v42);
        float v1393[4l];
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
                v1400 = v1367[v1399];
                float v1401;
                v1401 = v1400 / v1392;
                assert("Tensor range check" && 0 <= v1394 && v1394 < 1l);
                assert("Tensor range check" && 0 <= v1396 && v1396 < 4l);
                v1393[v1399] = v1401;
                v1396 += 1l ;
            }
            v1394 += 1l ;
        }
        float v1402[4l];
        float v1403;
        v1403 = 0.0f;
        int v1404;
        v1404 = 0l;
        while (while_method_3(v1404)){
            assert("Tensor range check" && 0 <= v1404 && v1404 < 1l);
            int v1406;
            v1406 = 4l * v1404;
            assert("Tensor range check" && 0 <= v1404 && v1404 < 1l);
            int v1407; float v1408;
            Tuple0 tmp38 = Tuple0{0l, 0.0f};
            v1407 = tmp38.v0; v1408 = tmp38.v1;
            while (while_method_1(v1407)){
                assert("Tensor range check" && 0 <= v1407 && v1407 < 4l);
                int v1410;
                v1410 = v1407 + v1406;
                float v1411;
                v1411 = v1393[v1410];
                float v1412;
                v1412 = v1408 + v1411;
                v1408 = v1412;
                v1407 += 1l ;
            }
            auto v1413 = cooperative_groups::coalesced_threads();
            int v1414;
            v1414 = threadIdx.x;
            int v1415;
            v1415 = v1414 / 16l;
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
            Tuple0 tmp39 = Tuple0{0l, v1423};
            v1424 = tmp39.v0; v1425 = tmp39.v1;
            while (while_method_1(v1424)){
                assert("Tensor range check" && 0 <= v1424 && v1424 < 4l);
                int v1427;
                v1427 = v1424 + v1406;
                float v1428;
                v1428 = v1393[v1427];
                float v1429;
                v1429 = v1425 + v1428;
                assert("Tensor range check" && 0 <= v1424 && v1424 < 4l);
                v1402[v1427] = v1429;
                v1425 = v1429;
                v1424 += 1l ;
            }
            float v1430;
            v1430 = v1403 + v1422;
            v1403 = v1430;
            v1404 += 1l ;
        }
        float v1431[4l];
        bool v1432[4l];
        int v1433;
        v1433 = 0l;
        while (while_method_3(v1433)){
            int v1435;
            v1435 = 0l;
            while (while_method_1(v1435)){
                assert("Tensor range check" && 0 <= v1433 && v1433 < 1l);
                assert("Tensor range check" && 0 <= v1435 && v1435 < 4l);
                int v1437;
                v1437 = 4l * v1433;
                int v1438;
                v1438 = v1437 + v1435;
                float v1439;
                v1439 = v1402[v1438];
                float v1440;
                v1440 = v1393[v1438];
                bool v1441;
                v1441 = v1440 > 0.0f;
                assert("Tensor range check" && 0 <= v1433 && v1433 < 1l);
                assert("Tensor range check" && 0 <= v1435 && v1435 < 4l);
                v1431[v1438] = v1439;
                v1432[v1438] = v1441;
                v1435 += 1l ;
            }
            v1433 += 1l ;
        }
        float v1442; bool v1443;
        Tuple3 tmp40 = Tuple3{-1.0f / 0.0f, false};
        v1442 = tmp40.v0; v1443 = tmp40.v1;
        int v1444;
        v1444 = 0l;
        while (while_method_3(v1444)){
            int v1446;
            v1446 = 0l;
            while (while_method_1(v1446)){
                assert("Tensor range check" && 0 <= v1444 && v1444 < 1l);
                assert("Tensor range check" && 0 <= v1446 && v1446 < 4l);
                int v1448;
                v1448 = 4l * v1444;
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
                v1446 += 1l ;
            }
            v1444 += 1l ;
        }
        auto v1460 = cooperative_groups::coalesced_threads();
        int v1461;
        v1461 = threadIdx.x;
        int v1462;
        v1462 = v1461 / 16l;
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
        float v1469[4l];
        int v1470[4l];
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
                int v1477;
                v1477 = v1263[v1476];
                float v1478;
                v1478 = curand_uniform(&v1245);
                assert("Tensor range check" && 0 <= v1471 && v1471 < 1l);
                assert("Tensor range check" && 0 <= v1473 && v1473 < 4l);
                v1469[v1476] = v1478;
                v1470[v1476] = v1477;
                v1473 += 1l ;
            }
            v1471 += 1l ;
        }
        float v1479; int v1480;
        Tuple1 tmp42 = Tuple1{0.0f, 2147483647l};
        v1479 = tmp42.v0; v1480 = tmp42.v1;
        int v1481;
        v1481 = 0l;
        while (while_method_3(v1481)){
            int v1483;
            v1483 = 0l;
            while (while_method_1(v1483)){
                assert("Tensor range check" && 0 <= v1481 && v1481 < 1l);
                assert("Tensor range check" && 0 <= v1483 && v1483 < 4l);
                int v1485;
                v1485 = 4l * v1481;
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
                v1483 += 1l ;
            }
            v1481 += 1l ;
        }
        auto v1492 = cooperative_groups::coalesced_threads();
        int v1493;
        v1493 = threadIdx.x;
        int v1494;
        v1494 = v1493 / 16l;
        auto v1495 = cooperative_groups::labeled_partition(v1492,v1494);
        Closure6 v1496{};
        float v1497; int v1498;
        Tuple1 tmp43 = cooperative_groups::reduce(v1495, Tuple1{v1479, v1480}, v1496);
        v1497 = tmp43.v0; v1498 = tmp43.v1;
        float v1499;
        v1499 = v1465 * v1497;
        int v1500[4l];
        bool v1501[4l];
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
                    v1513 = 2147483647l; v1514 = false;
                }
                assert("Tensor range check" && 0 <= v1502 && v1502 < 1l);
                assert("Tensor range check" && 0 <= v1504 && v1504 < 4l);
                v1500[v1507] = v1513;
                v1501[v1507] = v1514;
                v1504 += 1l ;
            }
            v1502 += 1l ;
        }
        int v1515; bool v1516;
        Tuple4 tmp44 = Tuple4{2147483647l, false};
        v1515 = tmp44.v0; v1516 = tmp44.v1;
        int v1517;
        v1517 = 0l;
        while (while_method_3(v1517)){
            int v1519;
            v1519 = 0l;
            while (while_method_1(v1519)){
                assert("Tensor range check" && 0 <= v1517 && v1517 < 1l);
                assert("Tensor range check" && 0 <= v1519 && v1519 < 4l);
                int v1521;
                v1521 = 4l * v1517;
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
                v1519 += 1l ;
            }
            v1517 += 1l ;
        }
        auto v1533 = cooperative_groups::coalesced_threads();
        int v1534;
        v1534 = threadIdx.x;
        int v1535;
        v1535 = v1534 / 16l;
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
        assert("Tensor range check" && 0 <= v1258 && v1258 < 64l);
        int v1542;
        v1542 = 0l;
        while (while_method_3(v1542)){
            assert("Tensor range check" && 0 <= v1542 && v1542 < 1l);
            int v1544;
            v1544 = 64l * v1542;
            int v1545;
            v1545 = v1544 + v1261;
            assert("Tensor range check" && 0 <= v1542 && v1542 < 1l);
            int v1546;
            v1546 = 4l * v1542;
            int4* v1547;
            v1547 = reinterpret_cast<int4*>(v1393 + v1546);
            int4* v1548;
            v1548 = reinterpret_cast<int4*>(v16 + v1545);
            assert("Pointer alignment check" && (unsigned long long)(v1547) % 4l == 0 && (unsigned long long)(v1548) % 4l == 0);
            *v1548 = *v1547;
            v1542 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1258 && v1258 < 64l);
        int v1549;
        v1549 = 2l * v1258;
        int v1550;
        v1550 = v1549 + v1251;
        v17[v1550] = v1538;
        v1258 += 1l ;
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
    int v24;
    v24 = sizeof(float *);
    unsigned long long v25;
    v25 = (unsigned long long)v24;
    unsigned long long v26;
    v26 = 32ull * v25;
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
    v33 = 32ull * v32;
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
    v40 = v39 <= 81920ull;
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
    assert("Tensor range check" && 0 <= v53 && v53 < 32l);
    v47[v53] = v18;
    v49[v53] = v20;
    v51[v53] = v22;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v54;
    v54 = 0l <= v53;
    bool v55;
    v55 = v54 == false;
    if (v55){
        assert("The index needs to be zero or positive." && v54);
    } else {
    }
    int v57;
    v57 = v53 % 4l;
    int v58;
    v58 = v53 / 4l;
    bool v59;
    v59 = v58 < 8l;
    bool v60;
    v60 = v59 == false;
    if (v60){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v59);
    } else {
    }
    assert("Tensor range check" && 0 <= v58 && v58 < 8l);
    int v62;
    v62 = 0l;
    while (while_method_1(v62)){
        bool v64;
        v64 = 0l <= v58;
        bool v65;
        v65 = v64 && v59;
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v65);
        } else {
        }
        bool v68;
        v68 = 0l <= v62;
        bool v70;
        if (v68){
            bool v69;
            v69 = v62 < 4l;
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
        v73 = v62 * 8l;
        int v74;
        v74 = v73 + v58;
        assert("Tensor range check" && 0 <= v62 && v62 < 4l);
        int v75;
        v75 = 8l * v62;
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
        v81 = v80 * 32l;
        int v82;
        v82 = v81 + v74;
        assert("Tensor range check" && 0 <= v57 && v57 < 4l);
        int v83;
        v83 = 4l * v57;
        float v84[4l];
        int v85[4l];
        int v86;
        v86 = 0l;
        while (while_method_3(v86)){
            assert("Tensor range check" && 0 <= v86 && v86 < 1l);
            int v88;
            v88 = 4l * v86;
            assert("Tensor range check" && 0 <= v86 && v86 < 1l);
            int v89;
            v89 = 16l * v86;
            int v90;
            v90 = v89 + v83;
            int4* v91;
            v91 = reinterpret_cast<int4*>(v77 + v90);
            int4* v92;
            v92 = reinterpret_cast<int4*>(v84 + v88);
            assert("Pointer alignment check" && (unsigned long long)(v91) % 4l == 0 && (unsigned long long)(v92) % 4l == 0);
            *v92 = *v91;
            v86 += 1l ;
        }
        int v93;
        v93 = 0l;
        while (while_method_3(v93)){
            int v95;
            v95 = 0l;
            while (while_method_1(v95)){
                bool v97;
                v97 = 0l <= v95;
                bool v99;
                if (v97){
                    bool v98;
                    v98 = v95 < 4l;
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
                v102 = 0l <= v57;
                bool v104;
                if (v102){
                    bool v103;
                    v103 = v57 < 4l;
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
                v107 = v57 * 4l;
                int v108;
                v108 = v95 + v107;
                bool v109;
                v109 = 0l <= v93;
                bool v111;
                if (v109){
                    bool v110;
                    v110 = v93 < 1l;
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
                v114 = v93 * 16l;
                int v115;
                v115 = v108 + v114;
                assert("Tensor range check" && 0 <= v93 && v93 < 1l);
                assert("Tensor range check" && 0 <= v95 && v95 < 4l);
                int v116;
                v116 = 4l * v93;
                int v117;
                v117 = v116 + v95;
                v85[v117] = v115;
                v95 += 1l ;
            }
            v93 += 1l ;
        }
        int v118[4l];
        int v119[4l];
        int v120;
        v120 = 0l;
        while (while_method_3(v120)){
            int v122;
            v122 = 0l;
            while (while_method_1(v122)){
                assert("Tensor range check" && 0 <= v120 && v120 < 1l);
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                int v124;
                v124 = 4l * v120;
                int v125;
                v125 = v124 + v122;
                int v126;
                v126 = v85[v125];
                assert("Tensor range check" && 0 <= v120 && v120 < 1l);
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                v118[v125] = v82;
                v119[v125] = v126;
                v122 += 1l ;
            }
            v120 += 1l ;
        }
        int v127;
        v127 = 0l;
        while (while_method_3(v127)){
            assert("Tensor range check" && 0 <= v127 && v127 < 1l);
            int v129;
            v129 = 16l * v127;
            int v130;
            v130 = v129 + v83;
            assert("Tensor range check" && 0 <= v127 && v127 < 1l);
            int v131;
            v131 = 4l * v127;
            int4* v132;
            v132 = reinterpret_cast<int4*>(v118 + v131);
            int4* v133;
            v133 = reinterpret_cast<int4*>(v78 + v130);
            assert("Pointer alignment check" && (unsigned long long)(v132) % 4l == 0 && (unsigned long long)(v133) % 4l == 0);
            *v133 = *v132;
            int4* v134;
            v134 = reinterpret_cast<int4*>(v119 + v131);
            int4* v135;
            v135 = reinterpret_cast<int4*>(v79 + v130);
            assert("Pointer alignment check" && (unsigned long long)(v134) % 4l == 0 && (unsigned long long)(v135) % 4l == 0);
            *v135 = *v134;
            v127 += 1l ;
        }
        assert("Tensor range check" && 0 <= v74 && v74 < 32l);
        v62 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v53 && v53 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v136;
    v136 = v1+v9;
    unsigned long long v138;
    v138 = v30 + 128ull;
    bool v139;
    v139 = v138 <= 81920ull;
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
    assert("Tensor range check" && 0 <= v150 && v150 < 32l);
    v146[v150] = v136;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v151;
    v151 = 0l <= v150;
    bool v152;
    v152 = v151 == false;
    if (v152){
        assert("The index needs to be zero or positive." && v151);
    } else {
    }
    int v154;
    v154 = v150 % 4l;
    int v155;
    v155 = v150 / 4l;
    bool v156;
    v156 = v155 < 8l;
    bool v157;
    v157 = v156 == false;
    if (v157){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v156);
    } else {
    }
    assert("Tensor range check" && 0 <= v155 && v155 < 8l);
    int v159;
    v159 = 0l;
    while (while_method_1(v159)){
        bool v161;
        v161 = 0l <= v155;
        bool v162;
        v162 = v161 && v156;
        bool v163;
        v163 = v162 == false;
        if (v163){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v162);
        } else {
        }
        bool v165;
        v165 = 0l <= v159;
        bool v167;
        if (v165){
            bool v166;
            v166 = v159 < 4l;
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
        v170 = v159 * 8l;
        int v171;
        v171 = v170 + v155;
        assert("Tensor range check" && 0 <= v159 && v159 < 4l);
        int v172;
        v172 = 8l * v159;
        int v173;
        v173 = v172 + v155;
        float * v174;
        v174 = v146[v173];
        int v175;
        v175 = blockIdx.x;
        int v176;
        v176 = v175 * 32l;
        int v177;
        v177 = v176 + v171;
        assert("Tensor range check" && 0 <= v154 && v154 < 4l);
        int v178;
        v178 = 4l * v154;
        float v179[4l];
        int v180[4l];
        int v181;
        v181 = 0l;
        while (while_method_3(v181)){
            assert("Tensor range check" && 0 <= v181 && v181 < 1l);
            int v183;
            v183 = 4l * v181;
            assert("Tensor range check" && 0 <= v181 && v181 < 1l);
            int v184;
            v184 = 16l * v181;
            int v185;
            v185 = v184 + v178;
            int4* v186;
            v186 = reinterpret_cast<int4*>(v174 + v185);
            int4* v187;
            v187 = reinterpret_cast<int4*>(v179 + v183);
            assert("Pointer alignment check" && (unsigned long long)(v186) % 4l == 0 && (unsigned long long)(v187) % 4l == 0);
            *v187 = *v186;
            v181 += 1l ;
        }
        int v188;
        v188 = 0l;
        while (while_method_3(v188)){
            int v190;
            v190 = 0l;
            while (while_method_1(v190)){
                bool v192;
                v192 = 0l <= v190;
                bool v194;
                if (v192){
                    bool v193;
                    v193 = v190 < 4l;
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
                v197 = 0l <= v154;
                bool v199;
                if (v197){
                    bool v198;
                    v198 = v154 < 4l;
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
                v202 = v154 * 4l;
                int v203;
                v203 = v190 + v202;
                bool v204;
                v204 = 0l <= v188;
                bool v206;
                if (v204){
                    bool v205;
                    v205 = v188 < 1l;
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
                v209 = v188 * 16l;
                int v210;
                v210 = v203 + v209;
                assert("Tensor range check" && 0 <= v188 && v188 < 1l);
                assert("Tensor range check" && 0 <= v190 && v190 < 4l);
                int v211;
                v211 = 4l * v188;
                int v212;
                v212 = v211 + v190;
                v180[v212] = v210;
                v190 += 1l ;
            }
            v188 += 1l ;
        }
        int v213;
        v213 = 0l;
        while (while_method_3(v213)){
            assert("Tensor range check" && 0 <= v213 && v213 < 1l);
            assert("Tensor range check" && 0 <= v213 && v213 < 1l);
            v213 += 1l ;
        }
        assert("Tensor range check" && 0 <= v171 && v171 < 32l);
        v148[v171] = v177;
        v159 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v150 && v150 < 32l);
    int v215;
    v215 = v148[v150];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v216;
    v216 = threadIdx.x;
    assert("Tensor range check" && 0 <= v216 && v216 < 32l);
    v4[v216] = v215;
    float * v217;
    v217 = v1+v9;
    float * v219;
    v219 = v6+v17;
    unsigned long long v221;
    v221 = v30 + v26;
    bool v222;
    v222 = v221 <= 81920ull;
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
    assert("Tensor range check" && 0 <= v233 && v233 < 32l);
    v229[v233] = v217;
    v231[v233] = v219;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v234;
    v234 = 0l <= v233;
    bool v235;
    v235 = v234 == false;
    if (v235){
        assert("The index needs to be zero or positive." && v234);
    } else {
    }
    int v237;
    v237 = v233 % 4l;
    int v238;
    v238 = v233 / 4l;
    bool v239;
    v239 = v238 < 8l;
    bool v240;
    v240 = v239 == false;
    if (v240){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v239);
    } else {
    }
    assert("Tensor range check" && 0 <= v238 && v238 < 8l);
    int v242;
    v242 = 0l;
    while (while_method_1(v242)){
        bool v244;
        v244 = 0l <= v238;
        bool v245;
        v245 = v244 && v239;
        bool v246;
        v246 = v245 == false;
        if (v246){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v245);
        } else {
        }
        bool v248;
        v248 = 0l <= v242;
        bool v250;
        if (v248){
            bool v249;
            v249 = v242 < 4l;
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
        v253 = v242 * 8l;
        int v254;
        v254 = v253 + v238;
        assert("Tensor range check" && 0 <= v242 && v242 < 4l);
        int v255;
        v255 = 8l * v242;
        int v256;
        v256 = v255 + v238;
        float * v257;
        v257 = v229[v256];
        float * v258;
        v258 = v231[v256];
        int v259;
        v259 = blockIdx.x;
        int v260;
        v260 = v259 * 32l;
        int v261;
        v261 = v260 + v254;
        assert("Tensor range check" && 0 <= v237 && v237 < 4l);
        int v262;
        v262 = 4l * v237;
        float v263[4l];
        int v264[4l];
        int v265;
        v265 = 0l;
        while (while_method_3(v265)){
            assert("Tensor range check" && 0 <= v265 && v265 < 1l);
            int v267;
            v267 = 4l * v265;
            assert("Tensor range check" && 0 <= v265 && v265 < 1l);
            int v268;
            v268 = 16l * v265;
            int v269;
            v269 = v268 + v262;
            int4* v270;
            v270 = reinterpret_cast<int4*>(v257 + v269);
            int4* v271;
            v271 = reinterpret_cast<int4*>(v263 + v267);
            assert("Pointer alignment check" && (unsigned long long)(v270) % 4l == 0 && (unsigned long long)(v271) % 4l == 0);
            *v271 = *v270;
            v265 += 1l ;
        }
        int v272;
        v272 = 0l;
        while (while_method_3(v272)){
            int v274;
            v274 = 0l;
            while (while_method_1(v274)){
                bool v276;
                v276 = 0l <= v274;
                bool v278;
                if (v276){
                    bool v277;
                    v277 = v274 < 4l;
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
                v281 = 0l <= v237;
                bool v283;
                if (v281){
                    bool v282;
                    v282 = v237 < 4l;
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
                v286 = v237 * 4l;
                int v287;
                v287 = v274 + v286;
                bool v288;
                v288 = 0l <= v272;
                bool v290;
                if (v288){
                    bool v289;
                    v289 = v272 < 1l;
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
                v293 = v272 * 16l;
                int v294;
                v294 = v287 + v293;
                assert("Tensor range check" && 0 <= v272 && v272 < 1l);
                assert("Tensor range check" && 0 <= v274 && v274 < 4l);
                int v295;
                v295 = 4l * v272;
                int v296;
                v296 = v295 + v274;
                v264[v296] = v294;
                v274 += 1l ;
            }
            v272 += 1l ;
        }
        int v297;
        v297 = 0l;
        while (while_method_3(v297)){
            assert("Tensor range check" && 0 <= v297 && v297 < 1l);
            int v299;
            v299 = 16l * v297;
            int v300;
            v300 = v299 + v262;
            assert("Tensor range check" && 0 <= v297 && v297 < 1l);
            int v301;
            v301 = 4l * v297;
            int4* v302;
            v302 = reinterpret_cast<int4*>(v263 + v301);
            int4* v303;
            v303 = reinterpret_cast<int4*>(v258 + v300);
            assert("Pointer alignment check" && (unsigned long long)(v302) % 4l == 0 && (unsigned long long)(v303) % 4l == 0);
            *v303 = *v302;
            v297 += 1l ;
        }
        assert("Tensor range check" && 0 <= v254 && v254 < 32l);
        v242 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v233 && v233 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
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
    assert("Tensor range check" && 0 <= v315 && v315 < 32l);
    v311[v315] = v304;
    v313[v315] = v306;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v316;
    v316 = 0l <= v315;
    bool v317;
    v317 = v316 == false;
    if (v317){
        assert("The index needs to be zero or positive." && v316);
    } else {
    }
    int v319;
    v319 = v315 % 4l;
    int v320;
    v320 = v315 / 4l;
    bool v321;
    v321 = v320 < 8l;
    bool v322;
    v322 = v321 == false;
    if (v322){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v321);
    } else {
    }
    assert("Tensor range check" && 0 <= v320 && v320 < 8l);
    int v324;
    v324 = 0l;
    while (while_method_1(v324)){
        bool v326;
        v326 = 0l <= v320;
        bool v327;
        v327 = v326 && v321;
        bool v328;
        v328 = v327 == false;
        if (v328){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v327);
        } else {
        }
        bool v330;
        v330 = 0l <= v324;
        bool v332;
        if (v330){
            bool v331;
            v331 = v324 < 4l;
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
        v335 = v324 * 8l;
        int v336;
        v336 = v335 + v320;
        assert("Tensor range check" && 0 <= v324 && v324 < 4l);
        int v337;
        v337 = 8l * v324;
        int v338;
        v338 = v337 + v320;
        float * v339;
        v339 = v311[v338];
        float * v340;
        v340 = v313[v338];
        int v341;
        v341 = blockIdx.x;
        int v342;
        v342 = v341 * 32l;
        int v343;
        v343 = v342 + v336;
        assert("Tensor range check" && 0 <= v319 && v319 < 4l);
        int v344;
        v344 = 4l * v319;
        float v345[4l];
        int v346[4l];
        int v347;
        v347 = 0l;
        while (while_method_3(v347)){
            assert("Tensor range check" && 0 <= v347 && v347 < 1l);
            int v349;
            v349 = 4l * v347;
            assert("Tensor range check" && 0 <= v347 && v347 < 1l);
            int v350;
            v350 = 16l * v347;
            int v351;
            v351 = v350 + v344;
            int4* v352;
            v352 = reinterpret_cast<int4*>(v339 + v351);
            int4* v353;
            v353 = reinterpret_cast<int4*>(v345 + v349);
            assert("Pointer alignment check" && (unsigned long long)(v352) % 4l == 0 && (unsigned long long)(v353) % 4l == 0);
            *v353 = *v352;
            v347 += 1l ;
        }
        int v354;
        v354 = 0l;
        while (while_method_3(v354)){
            int v356;
            v356 = 0l;
            while (while_method_1(v356)){
                bool v358;
                v358 = 0l <= v356;
                bool v360;
                if (v358){
                    bool v359;
                    v359 = v356 < 4l;
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
                v363 = 0l <= v319;
                bool v365;
                if (v363){
                    bool v364;
                    v364 = v319 < 4l;
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
                v368 = v319 * 4l;
                int v369;
                v369 = v356 + v368;
                bool v370;
                v370 = 0l <= v354;
                bool v372;
                if (v370){
                    bool v371;
                    v371 = v354 < 1l;
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
                v375 = v354 * 16l;
                int v376;
                v376 = v369 + v375;
                assert("Tensor range check" && 0 <= v354 && v354 < 1l);
                assert("Tensor range check" && 0 <= v356 && v356 < 4l);
                int v377;
                v377 = 4l * v354;
                int v378;
                v378 = v377 + v356;
                v346[v378] = v376;
                v356 += 1l ;
            }
            v354 += 1l ;
        }
        bool v379[4l];
        int v380;
        v380 = 0l;
        while (while_method_3(v380)){
            int v382;
            v382 = 0l;
            while (while_method_1(v382)){
                assert("Tensor range check" && 0 <= v380 && v380 < 1l);
                assert("Tensor range check" && 0 <= v382 && v382 < 4l);
                int v384;
                v384 = 4l * v380;
                int v385;
                v385 = v384 + v382;
                float v386;
                v386 = v345[v385];
                int v387;
                v387 = v346[v385];
                bool v388;
                v388 = v387 < 3l;
                assert("Tensor range check" && 0 <= v380 && v380 < 1l);
                assert("Tensor range check" && 0 <= v382 && v382 < 4l);
                v379[v385] = v388;
                v382 += 1l ;
            }
            v380 += 1l ;
        }
        float v389[4l];
        int v390;
        v390 = 0l;
        while (while_method_3(v390)){
            int v392;
            v392 = 0l;
            while (while_method_1(v392)){
                assert("Tensor range check" && 0 <= v390 && v390 < 1l);
                assert("Tensor range check" && 0 <= v392 && v392 < 4l);
                int v394;
                v394 = 4l * v390;
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
                assert("Tensor range check" && 0 <= v390 && v390 < 1l);
                assert("Tensor range check" && 0 <= v392 && v392 < 4l);
                v389[v395] = v400;
                v392 += 1l ;
            }
            v390 += 1l ;
        }
        float v401;
        v401 = 0.0f;
        int v402;
        v402 = 0l;
        while (while_method_3(v402)){
            int v404;
            v404 = 0l;
            while (while_method_1(v404)){
                assert("Tensor range check" && 0 <= v402 && v402 < 1l);
                assert("Tensor range check" && 0 <= v404 && v404 < 4l);
                int v406;
                v406 = 4l * v402;
                int v407;
                v407 = v406 + v404;
                float v408;
                v408 = v389[v407];
                float v409;
                v409 = v401 + v408;
                v401 = v409;
                v404 += 1l ;
            }
            v402 += 1l ;
        }
        auto v410 = cooperative_groups::coalesced_threads();
        int v411;
        v411 = threadIdx.x;
        int v412;
        v412 = v411 / 4l;
        auto v413 = cooperative_groups::labeled_partition(v410,v412);
        Closure0 v414{};
        float v415;
        v415 = cooperative_groups::reduce(v413, v401, v414);
        int v416[4l];
        int v417;
        v417 = 0l;
        while (while_method_3(v417)){
            int v419;
            v419 = 0l;
            while (while_method_1(v419)){
                assert("Tensor range check" && 0 <= v417 && v417 < 1l);
                assert("Tensor range check" && 0 <= v419 && v419 < 4l);
                int v421;
                v421 = 4l * v417;
                int v422;
                v422 = v421 + v419;
                bool v423;
                v423 = v379[v422];
                int v424;
                if (v423){
                    v424 = 1l;
                } else {
                    v424 = 0l;
                }
                assert("Tensor range check" && 0 <= v417 && v417 < 1l);
                assert("Tensor range check" && 0 <= v419 && v419 < 4l);
                v416[v422] = v424;
                v419 += 1l ;
            }
            v417 += 1l ;
        }
        int v425;
        v425 = 0l;
        int v426;
        v426 = 0l;
        while (while_method_3(v426)){
            int v428;
            v428 = 0l;
            while (while_method_1(v428)){
                assert("Tensor range check" && 0 <= v426 && v426 < 1l);
                assert("Tensor range check" && 0 <= v428 && v428 < 4l);
                int v430;
                v430 = 4l * v426;
                int v431;
                v431 = v430 + v428;
                int v432;
                v432 = v416[v431];
                int v433;
                v433 = v425 + v432;
                v425 = v433;
                v428 += 1l ;
            }
            v426 += 1l ;
        }
        auto v434 = cooperative_groups::coalesced_threads();
        int v435;
        v435 = threadIdx.x;
        int v436;
        v436 = v435 / 4l;
        auto v437 = cooperative_groups::labeled_partition(v434,v436);
        Closure4 v438{};
        int v439;
        v439 = cooperative_groups::reduce(v437, v425, v438);
        float v440;
        v440 = (float)v439;
        float v441;
        v441 = 1.0f / v440;
        float v442[4l];
        int v443;
        v443 = 0l;
        while (while_method_3(v443)){
            int v445;
            v445 = 0l;
            while (while_method_1(v445)){
                assert("Tensor range check" && 0 <= v443 && v443 < 1l);
                assert("Tensor range check" && 0 <= v445 && v445 < 4l);
                int v447;
                v447 = 4l * v443;
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
                assert("Tensor range check" && 0 <= v443 && v443 < 1l);
                assert("Tensor range check" && 0 <= v445 && v445 < 4l);
                v442[v448] = v456;
                v445 += 1l ;
            }
            v443 += 1l ;
        }
        int v457;
        v457 = 0l;
        while (while_method_3(v457)){
            assert("Tensor range check" && 0 <= v457 && v457 < 1l);
            int v459;
            v459 = 16l * v457;
            int v460;
            v460 = v459 + v344;
            assert("Tensor range check" && 0 <= v457 && v457 < 1l);
            int v461;
            v461 = 4l * v457;
            int4* v462;
            v462 = reinterpret_cast<int4*>(v442 + v461);
            int4* v463;
            v463 = reinterpret_cast<int4*>(v340 + v460);
            assert("Pointer alignment check" && (unsigned long long)(v462) % 4l == 0 && (unsigned long long)(v463) % 4l == 0);
            *v463 = *v462;
            v457 += 1l ;
        }
        assert("Tensor range check" && 0 <= v336 && v336 < 32l);
        v324 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v315 && v315 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v464;
    v464 = threadIdx.x;
    int v465;
    v465 = blockIdx.x;
    int v466;
    v466 = v465 * 32l;
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
    assert("Tensor range check" && 0 <= v479 && v479 < 32l);
    v475[v479] = v470;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v480;
    v480 = 0l <= v479;
    bool v481;
    v481 = v480 == false;
    if (v481){
        assert("The index needs to be zero or positive." && v480);
    } else {
    }
    int v483;
    v483 = v479 % 4l;
    int v484;
    v484 = v479 / 4l;
    bool v485;
    v485 = v484 < 8l;
    bool v486;
    v486 = v485 == false;
    if (v486){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v485);
    } else {
    }
    assert("Tensor range check" && 0 <= v484 && v484 < 8l);
    int v488;
    v488 = 0l;
    while (while_method_1(v488)){
        bool v490;
        v490 = 0l <= v484;
        bool v491;
        v491 = v490 && v485;
        bool v492;
        v492 = v491 == false;
        if (v492){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v491);
        } else {
        }
        bool v494;
        v494 = 0l <= v488;
        bool v496;
        if (v494){
            bool v495;
            v495 = v488 < 4l;
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
        v499 = v488 * 8l;
        int v500;
        v500 = v499 + v484;
        assert("Tensor range check" && 0 <= v488 && v488 < 4l);
        int v501;
        v501 = 8l * v488;
        int v502;
        v502 = v501 + v484;
        float * v503;
        v503 = v475[v502];
        int v504;
        v504 = blockIdx.x;
        int v505;
        v505 = v504 * 32l;
        int v506;
        v506 = v505 + v500;
        assert("Tensor range check" && 0 <= v483 && v483 < 4l);
        int v507;
        v507 = 4l * v483;
        float v508[4l];
        int v509[4l];
        int v510;
        v510 = 0l;
        while (while_method_3(v510)){
            assert("Tensor range check" && 0 <= v510 && v510 < 1l);
            int v512;
            v512 = 4l * v510;
            assert("Tensor range check" && 0 <= v510 && v510 < 1l);
            int v513;
            v513 = 16l * v510;
            int v514;
            v514 = v513 + v507;
            int4* v515;
            v515 = reinterpret_cast<int4*>(v503 + v514);
            int4* v516;
            v516 = reinterpret_cast<int4*>(v508 + v512);
            assert("Pointer alignment check" && (unsigned long long)(v515) % 4l == 0 && (unsigned long long)(v516) % 4l == 0);
            *v516 = *v515;
            v510 += 1l ;
        }
        int v517;
        v517 = 0l;
        while (while_method_3(v517)){
            int v519;
            v519 = 0l;
            while (while_method_1(v519)){
                bool v521;
                v521 = 0l <= v519;
                bool v523;
                if (v521){
                    bool v522;
                    v522 = v519 < 4l;
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
                v526 = 0l <= v483;
                bool v528;
                if (v526){
                    bool v527;
                    v527 = v483 < 4l;
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
                v531 = v483 * 4l;
                int v532;
                v532 = v519 + v531;
                bool v533;
                v533 = 0l <= v517;
                bool v535;
                if (v533){
                    bool v534;
                    v534 = v517 < 1l;
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
                v538 = v517 * 16l;
                int v539;
                v539 = v532 + v538;
                assert("Tensor range check" && 0 <= v517 && v517 < 1l);
                assert("Tensor range check" && 0 <= v519 && v519 < 4l);
                int v540;
                v540 = 4l * v517;
                int v541;
                v541 = v540 + v519;
                v509[v541] = v539;
                v519 += 1l ;
            }
            v517 += 1l ;
        }
        bool v542[4l];
        int v543;
        v543 = 0l;
        while (while_method_3(v543)){
            int v545;
            v545 = 0l;
            while (while_method_1(v545)){
                assert("Tensor range check" && 0 <= v543 && v543 < 1l);
                assert("Tensor range check" && 0 <= v545 && v545 < 4l);
                int v547;
                v547 = 4l * v543;
                int v548;
                v548 = v547 + v545;
                float v549;
                v549 = v508[v548];
                int v550;
                v550 = v509[v548];
                bool v551;
                v551 = v550 < 3l;
                assert("Tensor range check" && 0 <= v543 && v543 < 1l);
                assert("Tensor range check" && 0 <= v545 && v545 < 4l);
                v542[v548] = v551;
                v545 += 1l ;
            }
            v543 += 1l ;
        }
        int v552[4l];
        int v553;
        v553 = 0l;
        while (while_method_3(v553)){
            int v555;
            v555 = 0l;
            while (while_method_1(v555)){
                assert("Tensor range check" && 0 <= v553 && v553 < 1l);
                assert("Tensor range check" && 0 <= v555 && v555 < 4l);
                int v557;
                v557 = 4l * v553;
                int v558;
                v558 = v557 + v555;
                bool v559;
                v559 = v542[v558];
                int v560;
                if (v559){
                    v560 = 1l;
                } else {
                    v560 = 0l;
                }
                assert("Tensor range check" && 0 <= v553 && v553 < 1l);
                assert("Tensor range check" && 0 <= v555 && v555 < 4l);
                v552[v558] = v560;
                v555 += 1l ;
            }
            v553 += 1l ;
        }
        int v561;
        v561 = 0l;
        int v562;
        v562 = 0l;
        while (while_method_3(v562)){
            int v564;
            v564 = 0l;
            while (while_method_1(v564)){
                assert("Tensor range check" && 0 <= v562 && v562 < 1l);
                assert("Tensor range check" && 0 <= v564 && v564 < 4l);
                int v566;
                v566 = 4l * v562;
                int v567;
                v567 = v566 + v564;
                int v568;
                v568 = v552[v567];
                int v569;
                v569 = v561 + v568;
                v561 = v569;
                v564 += 1l ;
            }
            v562 += 1l ;
        }
        auto v570 = cooperative_groups::coalesced_threads();
        int v571;
        v571 = threadIdx.x;
        int v572;
        v572 = v571 / 4l;
        auto v573 = cooperative_groups::labeled_partition(v570,v572);
        Closure4 v574{};
        int v575;
        v575 = cooperative_groups::reduce(v573, v561, v574);
        float v576[4l];
        int v577;
        v577 = 0l;
        while (while_method_3(v577)){
            int v579;
            v579 = 0l;
            while (while_method_1(v579)){
                assert("Tensor range check" && 0 <= v577 && v577 < 1l);
                assert("Tensor range check" && 0 <= v579 && v579 < 4l);
                int v581;
                v581 = 4l * v577;
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
                assert("Tensor range check" && 0 <= v577 && v577 < 1l);
                assert("Tensor range check" && 0 <= v579 && v579 < 4l);
                v576[v582] = v585;
                v579 += 1l ;
            }
            v577 += 1l ;
        }
        float v586;
        v586 = 0.0f;
        int v587;
        v587 = 0l;
        while (while_method_3(v587)){
            int v589;
            v589 = 0l;
            while (while_method_1(v589)){
                assert("Tensor range check" && 0 <= v587 && v587 < 1l);
                assert("Tensor range check" && 0 <= v589 && v589 < 4l);
                int v591;
                v591 = 4l * v587;
                int v592;
                v592 = v591 + v589;
                float v593;
                v593 = v576[v592];
                float v594;
                v594 = v586 + v593;
                v586 = v594;
                v589 += 1l ;
            }
            v587 += 1l ;
        }
        auto v595 = cooperative_groups::coalesced_threads();
        int v596;
        v596 = threadIdx.x;
        int v597;
        v597 = v596 / 4l;
        auto v598 = cooperative_groups::labeled_partition(v595,v597);
        Closure0 v599{};
        float v600;
        v600 = cooperative_groups::reduce(v598, v586, v599);
        float v601;
        v601 = (float)v575;
        float v602;
        v602 = v600 / v601;
        float v603[4l];
        int v604;
        v604 = 0l;
        while (while_method_3(v604)){
            int v606;
            v606 = 0l;
            while (while_method_1(v606)){
                assert("Tensor range check" && 0 <= v604 && v604 < 1l);
                assert("Tensor range check" && 0 <= v606 && v606 < 4l);
                int v608;
                v608 = 4l * v604;
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
                assert("Tensor range check" && 0 <= v604 && v604 < 1l);
                assert("Tensor range check" && 0 <= v606 && v606 < 4l);
                v603[v609] = v614;
                v606 += 1l ;
            }
            v604 += 1l ;
        }
        float v615;
        v615 = 0.0f;
        int v616;
        v616 = 0l;
        while (while_method_3(v616)){
            int v618;
            v618 = 0l;
            while (while_method_1(v618)){
                assert("Tensor range check" && 0 <= v616 && v616 < 1l);
                assert("Tensor range check" && 0 <= v618 && v618 < 4l);
                int v620;
                v620 = 4l * v616;
                int v621;
                v621 = v620 + v618;
                float v622;
                v622 = v603[v621];
                float v623;
                v623 = v615 + v622;
                v615 = v623;
                v618 += 1l ;
            }
            v616 += 1l ;
        }
        auto v624 = cooperative_groups::coalesced_threads();
        int v625;
        v625 = threadIdx.x;
        int v626;
        v626 = v625 / 4l;
        auto v627 = cooperative_groups::labeled_partition(v624,v626);
        float v628;
        v628 = cooperative_groups::reduce(v627, v615, v599);
        float v629[4l];
        int v630;
        v630 = 0l;
        while (while_method_3(v630)){
            int v632;
            v632 = 0l;
            while (while_method_1(v632)){
                assert("Tensor range check" && 0 <= v630 && v630 < 1l);
                assert("Tensor range check" && 0 <= v632 && v632 < 4l);
                int v634;
                v634 = 4l * v630;
                int v635;
                v635 = v634 + v632;
                float v636;
                v636 = v603[v635];
                float v637;
                v637 = v636 / v628;
                assert("Tensor range check" && 0 <= v630 && v630 < 1l);
                assert("Tensor range check" && 0 <= v632 && v632 < 4l);
                v629[v635] = v637;
                v632 += 1l ;
            }
            v630 += 1l ;
        }
        float v638[4l];
        float v639;
        v639 = 0.0f;
        int v640;
        v640 = 0l;
        while (while_method_3(v640)){
            assert("Tensor range check" && 0 <= v640 && v640 < 1l);
            int v642;
            v642 = 4l * v640;
            assert("Tensor range check" && 0 <= v640 && v640 < 1l);
            int v643; float v644;
            Tuple0 tmp46 = Tuple0{0l, 0.0f};
            v643 = tmp46.v0; v644 = tmp46.v1;
            while (while_method_1(v643)){
                assert("Tensor range check" && 0 <= v643 && v643 < 4l);
                int v646;
                v646 = v643 + v642;
                float v647;
                v647 = v629[v646];
                float v648;
                v648 = v644 + v647;
                v644 = v648;
                v643 += 1l ;
            }
            auto v649 = cooperative_groups::coalesced_threads();
            int v650;
            v650 = threadIdx.x;
            int v651;
            v651 = v650 / 4l;
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
            Tuple0 tmp47 = Tuple0{0l, v659};
            v660 = tmp47.v0; v661 = tmp47.v1;
            while (while_method_1(v660)){
                assert("Tensor range check" && 0 <= v660 && v660 < 4l);
                int v663;
                v663 = v660 + v642;
                float v664;
                v664 = v629[v663];
                float v665;
                v665 = v661 + v664;
                assert("Tensor range check" && 0 <= v660 && v660 < 4l);
                v638[v663] = v665;
                v661 = v665;
                v660 += 1l ;
            }
            float v666;
            v666 = v639 + v658;
            v639 = v666;
            v640 += 1l ;
        }
        float v667[4l];
        bool v668[4l];
        int v669;
        v669 = 0l;
        while (while_method_3(v669)){
            int v671;
            v671 = 0l;
            while (while_method_1(v671)){
                assert("Tensor range check" && 0 <= v669 && v669 < 1l);
                assert("Tensor range check" && 0 <= v671 && v671 < 4l);
                int v673;
                v673 = 4l * v669;
                int v674;
                v674 = v673 + v671;
                float v675;
                v675 = v638[v674];
                float v676;
                v676 = v629[v674];
                bool v677;
                v677 = v676 > 0.0f;
                assert("Tensor range check" && 0 <= v669 && v669 < 1l);
                assert("Tensor range check" && 0 <= v671 && v671 < 4l);
                v667[v674] = v675;
                v668[v674] = v677;
                v671 += 1l ;
            }
            v669 += 1l ;
        }
        float v678; bool v679;
        Tuple3 tmp48 = Tuple3{-1.0f / 0.0f, false};
        v678 = tmp48.v0; v679 = tmp48.v1;
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
                v682 += 1l ;
            }
            v680 += 1l ;
        }
        auto v696 = cooperative_groups::coalesced_threads();
        int v697;
        v697 = threadIdx.x;
        int v698;
        v698 = v697 / 4l;
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
        float v705[4l];
        int v706[4l];
        int v707;
        v707 = 0l;
        while (while_method_3(v707)){
            int v709;
            v709 = 0l;
            while (while_method_1(v709)){
                assert("Tensor range check" && 0 <= v707 && v707 < 1l);
                assert("Tensor range check" && 0 <= v709 && v709 < 4l);
                int v711;
                v711 = 4l * v707;
                int v712;
                v712 = v711 + v709;
                int v713;
                v713 = v509[v712];
                float v714;
                v714 = curand_uniform(&v469);
                assert("Tensor range check" && 0 <= v707 && v707 < 1l);
                assert("Tensor range check" && 0 <= v709 && v709 < 4l);
                v705[v712] = v714;
                v706[v712] = v713;
                v709 += 1l ;
            }
            v707 += 1l ;
        }
        float v715; int v716;
        Tuple1 tmp50 = Tuple1{0.0f, 2147483647l};
        v715 = tmp50.v0; v716 = tmp50.v1;
        int v717;
        v717 = 0l;
        while (while_method_3(v717)){
            int v719;
            v719 = 0l;
            while (while_method_1(v719)){
                assert("Tensor range check" && 0 <= v717 && v717 < 1l);
                assert("Tensor range check" && 0 <= v719 && v719 < 4l);
                int v721;
                v721 = 4l * v717;
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
                v719 += 1l ;
            }
            v717 += 1l ;
        }
        auto v728 = cooperative_groups::coalesced_threads();
        int v729;
        v729 = threadIdx.x;
        int v730;
        v730 = v729 / 4l;
        auto v731 = cooperative_groups::labeled_partition(v728,v730);
        Closure6 v732{};
        float v733; int v734;
        Tuple1 tmp51 = cooperative_groups::reduce(v731, Tuple1{v715, v716}, v732);
        v733 = tmp51.v0; v734 = tmp51.v1;
        float v735;
        v735 = v701 * v733;
        int v736[4l];
        bool v737[4l];
        int v738;
        v738 = 0l;
        while (while_method_3(v738)){
            int v740;
            v740 = 0l;
            while (while_method_1(v740)){
                assert("Tensor range check" && 0 <= v738 && v738 < 1l);
                assert("Tensor range check" && 0 <= v740 && v740 < 4l);
                int v742;
                v742 = 4l * v738;
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
                    v749 = 2147483647l; v750 = false;
                }
                assert("Tensor range check" && 0 <= v738 && v738 < 1l);
                assert("Tensor range check" && 0 <= v740 && v740 < 4l);
                v736[v743] = v749;
                v737[v743] = v750;
                v740 += 1l ;
            }
            v738 += 1l ;
        }
        int v751; bool v752;
        Tuple4 tmp52 = Tuple4{2147483647l, false};
        v751 = tmp52.v0; v752 = tmp52.v1;
        int v753;
        v753 = 0l;
        while (while_method_3(v753)){
            int v755;
            v755 = 0l;
            while (while_method_1(v755)){
                assert("Tensor range check" && 0 <= v753 && v753 < 1l);
                assert("Tensor range check" && 0 <= v755 && v755 < 4l);
                int v757;
                v757 = 4l * v753;
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
                v755 += 1l ;
            }
            v753 += 1l ;
        }
        auto v769 = cooperative_groups::coalesced_threads();
        int v770;
        v770 = threadIdx.x;
        int v771;
        v771 = v770 / 4l;
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
        v778 = 0l;
        while (while_method_3(v778)){
            assert("Tensor range check" && 0 <= v778 && v778 < 1l);
            assert("Tensor range check" && 0 <= v778 && v778 < 1l);
            v778 += 1l ;
        }
        assert("Tensor range check" && 0 <= v500 && v500 < 32l);
        v477[v500] = v774;
        v488 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v479 && v479 < 32l);
    int v780;
    v780 = v477[v479];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v781;
    v781 = threadIdx.x;
    assert("Tensor range check" && 0 <= v781 && v781 < 32l);
    v5[v781] = v780;
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
    int v24;
    v24 = sizeof(float *);
    unsigned long long v25;
    v25 = (unsigned long long)v24;
    unsigned long long v26;
    v26 = 32ull * v25;
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
    v33 = 32ull * v32;
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
    v40 = v39 <= 81920ull;
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
    assert("Tensor range check" && 0 <= v53 && v53 < 32l);
    v47[v53] = v18;
    v49[v53] = v20;
    v51[v53] = v22;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v54;
    v54 = 0l <= v53;
    bool v55;
    v55 = v54 == false;
    if (v55){
        assert("The index needs to be zero or positive." && v54);
    } else {
    }
    int v57;
    v57 = v53 % 32l;
    int v58;
    v58 = v53 / 32l;
    bool v59;
    v59 = v58 < 1l;
    bool v60;
    v60 = v59 == false;
    if (v60){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v59);
    } else {
    }
    assert("Tensor range check" && 0 <= v58 && v58 < 1l);
    int v62;
    v62 = 0l;
    while (while_method_4(v62)){
        bool v64;
        v64 = 0l <= v58;
        bool v65;
        v65 = v64 && v59;
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v65);
        } else {
        }
        bool v68;
        v68 = 0l <= v62;
        bool v70;
        if (v68){
            bool v69;
            v69 = v62 < 32l;
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
        v73 = v62 + v58;
        assert("Tensor range check" && 0 <= v62 && v62 < 32l);
        float * v74;
        v74 = v47[v73];
        int * v75;
        v75 = v49[v73];
        int * v76;
        v76 = v51[v73];
        int v77;
        v77 = blockIdx.x;
        int v78;
        v78 = v77 * 32l;
        int v79;
        v79 = v78 + v73;
        assert("Tensor range check" && 0 <= v57 && v57 < 32l);
        int v80;
        v80 = 4l * v57;
        float v81[8l];
        int v82[8l];
        int v83;
        v83 = 0l;
        while (while_method_5(v83)){
            assert("Tensor range check" && 0 <= v83 && v83 < 2l);
            int v85;
            v85 = 4l * v83;
            assert("Tensor range check" && 0 <= v83 && v83 < 2l);
            int v86;
            v86 = 128l * v83;
            int v87;
            v87 = v86 + v80;
            int4* v88;
            v88 = reinterpret_cast<int4*>(v74 + v87);
            int4* v89;
            v89 = reinterpret_cast<int4*>(v81 + v85);
            assert("Pointer alignment check" && (unsigned long long)(v88) % 4l == 0 && (unsigned long long)(v89) % 4l == 0);
            *v89 = *v88;
            v83 += 1l ;
        }
        int v90;
        v90 = 0l;
        while (while_method_5(v90)){
            int v92;
            v92 = 0l;
            while (while_method_1(v92)){
                bool v94;
                v94 = 0l <= v92;
                bool v96;
                if (v94){
                    bool v95;
                    v95 = v92 < 4l;
                    v96 = v95;
                } else {
                    v96 = false;
                }
                bool v97;
                v97 = v96 == false;
                if (v97){
                    assert("The indices should be inside the range of the dimension." && v96);
                } else {
                }
                bool v99;
                v99 = 0l <= v57;
                bool v101;
                if (v99){
                    bool v100;
                    v100 = v57 < 32l;
                    v101 = v100;
                } else {
                    v101 = false;
                }
                bool v102;
                v102 = v101 == false;
                if (v102){
                    assert("The indices should be inside the range of the dimension." && v101);
                } else {
                }
                int v104;
                v104 = v57 * 4l;
                int v105;
                v105 = v92 + v104;
                bool v106;
                v106 = 0l <= v90;
                bool v108;
                if (v106){
                    bool v107;
                    v107 = v90 < 2l;
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
                int v111;
                v111 = v90 * 128l;
                int v112;
                v112 = v105 + v111;
                assert("Tensor range check" && 0 <= v90 && v90 < 2l);
                assert("Tensor range check" && 0 <= v92 && v92 < 4l);
                int v113;
                v113 = 4l * v90;
                int v114;
                v114 = v113 + v92;
                v82[v114] = v112;
                v92 += 1l ;
            }
            v90 += 1l ;
        }
        int v115[8l];
        int v116[8l];
        int v117;
        v117 = 0l;
        while (while_method_5(v117)){
            int v119;
            v119 = 0l;
            while (while_method_1(v119)){
                assert("Tensor range check" && 0 <= v117 && v117 < 2l);
                assert("Tensor range check" && 0 <= v119 && v119 < 4l);
                int v121;
                v121 = 4l * v117;
                int v122;
                v122 = v121 + v119;
                int v123;
                v123 = v82[v122];
                assert("Tensor range check" && 0 <= v117 && v117 < 2l);
                assert("Tensor range check" && 0 <= v119 && v119 < 4l);
                v115[v122] = v79;
                v116[v122] = v123;
                v119 += 1l ;
            }
            v117 += 1l ;
        }
        int v124;
        v124 = 0l;
        while (while_method_5(v124)){
            assert("Tensor range check" && 0 <= v124 && v124 < 2l);
            int v126;
            v126 = 128l * v124;
            int v127;
            v127 = v126 + v80;
            assert("Tensor range check" && 0 <= v124 && v124 < 2l);
            int v128;
            v128 = 4l * v124;
            int4* v129;
            v129 = reinterpret_cast<int4*>(v115 + v128);
            int4* v130;
            v130 = reinterpret_cast<int4*>(v75 + v127);
            assert("Pointer alignment check" && (unsigned long long)(v129) % 4l == 0 && (unsigned long long)(v130) % 4l == 0);
            *v130 = *v129;
            int4* v131;
            v131 = reinterpret_cast<int4*>(v116 + v128);
            int4* v132;
            v132 = reinterpret_cast<int4*>(v76 + v127);
            assert("Pointer alignment check" && (unsigned long long)(v131) % 4l == 0 && (unsigned long long)(v132) % 4l == 0);
            *v132 = *v131;
            v124 += 1l ;
        }
        assert("Tensor range check" && 0 <= v73 && v73 < 32l);
        v62 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v53 && v53 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v133;
    v133 = v1+v9;
    unsigned long long v135;
    v135 = v30 + 128ull;
    bool v136;
    v136 = v135 <= 81920ull;
    bool v137;
    v137 = v136 == false;
    if (v137){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v136);
    } else {
    }
    extern __shared__ unsigned char v139[];
    bool v140;
    v140 = v135 <= v135;
    bool v141;
    v141 = v140 == false;
    if (v141){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v140);
    } else {
    }
    float * * v143;
    v143 = reinterpret_cast<float * *>(&v139[0ull]);
    int * v145;
    v145 = reinterpret_cast<int *>(&v139[v30]);
    int v147;
    v147 = threadIdx.x;
    assert("Tensor range check" && 0 <= v147 && v147 < 32l);
    v143[v147] = v133;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v148;
    v148 = 0l <= v147;
    bool v149;
    v149 = v148 == false;
    if (v149){
        assert("The index needs to be zero or positive." && v148);
    } else {
    }
    int v151;
    v151 = v147 % 32l;
    int v152;
    v152 = v147 / 32l;
    bool v153;
    v153 = v152 < 1l;
    bool v154;
    v154 = v153 == false;
    if (v154){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v153);
    } else {
    }
    assert("Tensor range check" && 0 <= v152 && v152 < 1l);
    int v156;
    v156 = 0l;
    while (while_method_4(v156)){
        bool v158;
        v158 = 0l <= v152;
        bool v159;
        v159 = v158 && v153;
        bool v160;
        v160 = v159 == false;
        if (v160){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v159);
        } else {
        }
        bool v162;
        v162 = 0l <= v156;
        bool v164;
        if (v162){
            bool v163;
            v163 = v156 < 32l;
            v164 = v163;
        } else {
            v164 = false;
        }
        bool v165;
        v165 = v164 == false;
        if (v165){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v164);
        } else {
        }
        int v167;
        v167 = v156 + v152;
        assert("Tensor range check" && 0 <= v156 && v156 < 32l);
        float * v168;
        v168 = v143[v167];
        int v169;
        v169 = blockIdx.x;
        int v170;
        v170 = v169 * 32l;
        int v171;
        v171 = v170 + v167;
        assert("Tensor range check" && 0 <= v151 && v151 < 32l);
        int v172;
        v172 = 4l * v151;
        float v173[8l];
        int v174[8l];
        int v175;
        v175 = 0l;
        while (while_method_5(v175)){
            assert("Tensor range check" && 0 <= v175 && v175 < 2l);
            int v177;
            v177 = 4l * v175;
            assert("Tensor range check" && 0 <= v175 && v175 < 2l);
            int v178;
            v178 = 128l * v175;
            int v179;
            v179 = v178 + v172;
            int4* v180;
            v180 = reinterpret_cast<int4*>(v168 + v179);
            int4* v181;
            v181 = reinterpret_cast<int4*>(v173 + v177);
            assert("Pointer alignment check" && (unsigned long long)(v180) % 4l == 0 && (unsigned long long)(v181) % 4l == 0);
            *v181 = *v180;
            v175 += 1l ;
        }
        int v182;
        v182 = 0l;
        while (while_method_5(v182)){
            int v184;
            v184 = 0l;
            while (while_method_1(v184)){
                bool v186;
                v186 = 0l <= v184;
                bool v188;
                if (v186){
                    bool v187;
                    v187 = v184 < 4l;
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
                bool v191;
                v191 = 0l <= v151;
                bool v193;
                if (v191){
                    bool v192;
                    v192 = v151 < 32l;
                    v193 = v192;
                } else {
                    v193 = false;
                }
                bool v194;
                v194 = v193 == false;
                if (v194){
                    assert("The indices should be inside the range of the dimension." && v193);
                } else {
                }
                int v196;
                v196 = v151 * 4l;
                int v197;
                v197 = v184 + v196;
                bool v198;
                v198 = 0l <= v182;
                bool v200;
                if (v198){
                    bool v199;
                    v199 = v182 < 2l;
                    v200 = v199;
                } else {
                    v200 = false;
                }
                bool v201;
                v201 = v200 == false;
                if (v201){
                    assert("The indices should be inside the range of the dimension." && v200);
                } else {
                }
                int v203;
                v203 = v182 * 128l;
                int v204;
                v204 = v197 + v203;
                assert("Tensor range check" && 0 <= v182 && v182 < 2l);
                assert("Tensor range check" && 0 <= v184 && v184 < 4l);
                int v205;
                v205 = 4l * v182;
                int v206;
                v206 = v205 + v184;
                v174[v206] = v204;
                v184 += 1l ;
            }
            v182 += 1l ;
        }
        int v207;
        v207 = 0l;
        while (while_method_5(v207)){
            assert("Tensor range check" && 0 <= v207 && v207 < 2l);
            assert("Tensor range check" && 0 <= v207 && v207 < 2l);
            v207 += 1l ;
        }
        assert("Tensor range check" && 0 <= v167 && v167 < 32l);
        v145[v167] = v171;
        v156 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v147 && v147 < 32l);
    int v209;
    v209 = v145[v147];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v210;
    v210 = threadIdx.x;
    assert("Tensor range check" && 0 <= v210 && v210 < 32l);
    v4[v210] = v209;
    float * v211;
    v211 = v1+v9;
    float * v213;
    v213 = v6+v17;
    unsigned long long v215;
    v215 = v30 + v26;
    bool v216;
    v216 = v215 <= 81920ull;
    bool v217;
    v217 = v216 == false;
    if (v217){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v216);
    } else {
    }
    extern __shared__ unsigned char v219[];
    bool v220;
    v220 = v215 <= v215;
    bool v221;
    v221 = v220 == false;
    if (v221){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v220);
    } else {
    }
    float * * v223;
    v223 = reinterpret_cast<float * *>(&v219[0ull]);
    float * * v225;
    v225 = reinterpret_cast<float * *>(&v219[v30]);
    int v227;
    v227 = threadIdx.x;
    assert("Tensor range check" && 0 <= v227 && v227 < 32l);
    v223[v227] = v211;
    v225[v227] = v213;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v228;
    v228 = 0l <= v227;
    bool v229;
    v229 = v228 == false;
    if (v229){
        assert("The index needs to be zero or positive." && v228);
    } else {
    }
    int v231;
    v231 = v227 % 32l;
    int v232;
    v232 = v227 / 32l;
    bool v233;
    v233 = v232 < 1l;
    bool v234;
    v234 = v233 == false;
    if (v234){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v233);
    } else {
    }
    assert("Tensor range check" && 0 <= v232 && v232 < 1l);
    int v236;
    v236 = 0l;
    while (while_method_4(v236)){
        bool v238;
        v238 = 0l <= v232;
        bool v239;
        v239 = v238 && v233;
        bool v240;
        v240 = v239 == false;
        if (v240){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v239);
        } else {
        }
        bool v242;
        v242 = 0l <= v236;
        bool v244;
        if (v242){
            bool v243;
            v243 = v236 < 32l;
            v244 = v243;
        } else {
            v244 = false;
        }
        bool v245;
        v245 = v244 == false;
        if (v245){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v244);
        } else {
        }
        int v247;
        v247 = v236 + v232;
        assert("Tensor range check" && 0 <= v236 && v236 < 32l);
        float * v248;
        v248 = v223[v247];
        float * v249;
        v249 = v225[v247];
        int v250;
        v250 = blockIdx.x;
        int v251;
        v251 = v250 * 32l;
        int v252;
        v252 = v251 + v247;
        assert("Tensor range check" && 0 <= v231 && v231 < 32l);
        int v253;
        v253 = 4l * v231;
        float v254[8l];
        int v255[8l];
        int v256;
        v256 = 0l;
        while (while_method_5(v256)){
            assert("Tensor range check" && 0 <= v256 && v256 < 2l);
            int v258;
            v258 = 4l * v256;
            assert("Tensor range check" && 0 <= v256 && v256 < 2l);
            int v259;
            v259 = 128l * v256;
            int v260;
            v260 = v259 + v253;
            int4* v261;
            v261 = reinterpret_cast<int4*>(v248 + v260);
            int4* v262;
            v262 = reinterpret_cast<int4*>(v254 + v258);
            assert("Pointer alignment check" && (unsigned long long)(v261) % 4l == 0 && (unsigned long long)(v262) % 4l == 0);
            *v262 = *v261;
            v256 += 1l ;
        }
        int v263;
        v263 = 0l;
        while (while_method_5(v263)){
            int v265;
            v265 = 0l;
            while (while_method_1(v265)){
                bool v267;
                v267 = 0l <= v265;
                bool v269;
                if (v267){
                    bool v268;
                    v268 = v265 < 4l;
                    v269 = v268;
                } else {
                    v269 = false;
                }
                bool v270;
                v270 = v269 == false;
                if (v270){
                    assert("The indices should be inside the range of the dimension." && v269);
                } else {
                }
                bool v272;
                v272 = 0l <= v231;
                bool v274;
                if (v272){
                    bool v273;
                    v273 = v231 < 32l;
                    v274 = v273;
                } else {
                    v274 = false;
                }
                bool v275;
                v275 = v274 == false;
                if (v275){
                    assert("The indices should be inside the range of the dimension." && v274);
                } else {
                }
                int v277;
                v277 = v231 * 4l;
                int v278;
                v278 = v265 + v277;
                bool v279;
                v279 = 0l <= v263;
                bool v281;
                if (v279){
                    bool v280;
                    v280 = v263 < 2l;
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
                int v284;
                v284 = v263 * 128l;
                int v285;
                v285 = v278 + v284;
                assert("Tensor range check" && 0 <= v263 && v263 < 2l);
                assert("Tensor range check" && 0 <= v265 && v265 < 4l);
                int v286;
                v286 = 4l * v263;
                int v287;
                v287 = v286 + v265;
                v255[v287] = v285;
                v265 += 1l ;
            }
            v263 += 1l ;
        }
        int v288;
        v288 = 0l;
        while (while_method_5(v288)){
            assert("Tensor range check" && 0 <= v288 && v288 < 2l);
            int v290;
            v290 = 128l * v288;
            int v291;
            v291 = v290 + v253;
            assert("Tensor range check" && 0 <= v288 && v288 < 2l);
            int v292;
            v292 = 4l * v288;
            int4* v293;
            v293 = reinterpret_cast<int4*>(v254 + v292);
            int4* v294;
            v294 = reinterpret_cast<int4*>(v249 + v291);
            assert("Pointer alignment check" && (unsigned long long)(v293) % 4l == 0 && (unsigned long long)(v294) % 4l == 0);
            *v294 = *v293;
            v288 += 1l ;
        }
        assert("Tensor range check" && 0 <= v247 && v247 < 32l);
        v236 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v227 && v227 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v295;
    v295 = v1+v9;
    float * v297;
    v297 = v7+v13;
    if (v217){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v216);
    } else {
    }
    extern __shared__ unsigned char v300[];
    if (v221){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v220);
    } else {
    }
    float * * v302;
    v302 = reinterpret_cast<float * *>(&v300[0ull]);
    float * * v304;
    v304 = reinterpret_cast<float * *>(&v300[v30]);
    int v306;
    v306 = threadIdx.x;
    assert("Tensor range check" && 0 <= v306 && v306 < 32l);
    v302[v306] = v295;
    v304[v306] = v297;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v307;
    v307 = 0l <= v306;
    bool v308;
    v308 = v307 == false;
    if (v308){
        assert("The index needs to be zero or positive." && v307);
    } else {
    }
    int v310;
    v310 = v306 % 32l;
    int v311;
    v311 = v306 / 32l;
    bool v312;
    v312 = v311 < 1l;
    bool v313;
    v313 = v312 == false;
    if (v313){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v312);
    } else {
    }
    assert("Tensor range check" && 0 <= v311 && v311 < 1l);
    int v315;
    v315 = 0l;
    while (while_method_4(v315)){
        bool v317;
        v317 = 0l <= v311;
        bool v318;
        v318 = v317 && v312;
        bool v319;
        v319 = v318 == false;
        if (v319){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v318);
        } else {
        }
        bool v321;
        v321 = 0l <= v315;
        bool v323;
        if (v321){
            bool v322;
            v322 = v315 < 32l;
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
        v326 = v315 + v311;
        assert("Tensor range check" && 0 <= v315 && v315 < 32l);
        float * v327;
        v327 = v302[v326];
        float * v328;
        v328 = v304[v326];
        int v329;
        v329 = blockIdx.x;
        int v330;
        v330 = v329 * 32l;
        int v331;
        v331 = v330 + v326;
        assert("Tensor range check" && 0 <= v310 && v310 < 32l);
        int v332;
        v332 = 4l * v310;
        float v333[8l];
        int v334[8l];
        int v335;
        v335 = 0l;
        while (while_method_5(v335)){
            assert("Tensor range check" && 0 <= v335 && v335 < 2l);
            int v337;
            v337 = 4l * v335;
            assert("Tensor range check" && 0 <= v335 && v335 < 2l);
            int v338;
            v338 = 128l * v335;
            int v339;
            v339 = v338 + v332;
            int4* v340;
            v340 = reinterpret_cast<int4*>(v327 + v339);
            int4* v341;
            v341 = reinterpret_cast<int4*>(v333 + v337);
            assert("Pointer alignment check" && (unsigned long long)(v340) % 4l == 0 && (unsigned long long)(v341) % 4l == 0);
            *v341 = *v340;
            v335 += 1l ;
        }
        int v342;
        v342 = 0l;
        while (while_method_5(v342)){
            int v344;
            v344 = 0l;
            while (while_method_1(v344)){
                bool v346;
                v346 = 0l <= v344;
                bool v348;
                if (v346){
                    bool v347;
                    v347 = v344 < 4l;
                    v348 = v347;
                } else {
                    v348 = false;
                }
                bool v349;
                v349 = v348 == false;
                if (v349){
                    assert("The indices should be inside the range of the dimension." && v348);
                } else {
                }
                bool v351;
                v351 = 0l <= v310;
                bool v353;
                if (v351){
                    bool v352;
                    v352 = v310 < 32l;
                    v353 = v352;
                } else {
                    v353 = false;
                }
                bool v354;
                v354 = v353 == false;
                if (v354){
                    assert("The indices should be inside the range of the dimension." && v353);
                } else {
                }
                int v356;
                v356 = v310 * 4l;
                int v357;
                v357 = v344 + v356;
                bool v358;
                v358 = 0l <= v342;
                bool v360;
                if (v358){
                    bool v359;
                    v359 = v342 < 2l;
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
                int v363;
                v363 = v342 * 128l;
                int v364;
                v364 = v357 + v363;
                assert("Tensor range check" && 0 <= v342 && v342 < 2l);
                assert("Tensor range check" && 0 <= v344 && v344 < 4l);
                int v365;
                v365 = 4l * v342;
                int v366;
                v366 = v365 + v344;
                v334[v366] = v364;
                v344 += 1l ;
            }
            v342 += 1l ;
        }
        bool v367[8l];
        int v368;
        v368 = 0l;
        while (while_method_5(v368)){
            int v370;
            v370 = 0l;
            while (while_method_1(v370)){
                assert("Tensor range check" && 0 <= v368 && v368 < 2l);
                assert("Tensor range check" && 0 <= v370 && v370 < 4l);
                int v372;
                v372 = 4l * v368;
                int v373;
                v373 = v372 + v370;
                float v374;
                v374 = v333[v373];
                int v375;
                v375 = v334[v373];
                bool v376;
                v376 = v375 < 3l;
                assert("Tensor range check" && 0 <= v368 && v368 < 2l);
                assert("Tensor range check" && 0 <= v370 && v370 < 4l);
                v367[v373] = v376;
                v370 += 1l ;
            }
            v368 += 1l ;
        }
        float v377[8l];
        int v378;
        v378 = 0l;
        while (while_method_5(v378)){
            int v380;
            v380 = 0l;
            while (while_method_1(v380)){
                assert("Tensor range check" && 0 <= v378 && v378 < 2l);
                assert("Tensor range check" && 0 <= v380 && v380 < 4l);
                int v382;
                v382 = 4l * v378;
                int v383;
                v383 = v382 + v380;
                float v384;
                v384 = v333[v383];
                bool v385;
                v385 = v367[v383];
                float v388;
                if (v385){
                    bool v386;
                    v386 = 0.0f >= v384;
                    if (v386){
                        v388 = 0.0f;
                    } else {
                        v388 = v384;
                    }
                } else {
                    v388 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v378 && v378 < 2l);
                assert("Tensor range check" && 0 <= v380 && v380 < 4l);
                v377[v383] = v388;
                v380 += 1l ;
            }
            v378 += 1l ;
        }
        float v389;
        v389 = 0.0f;
        int v390;
        v390 = 0l;
        while (while_method_5(v390)){
            int v392;
            v392 = 0l;
            while (while_method_1(v392)){
                assert("Tensor range check" && 0 <= v390 && v390 < 2l);
                assert("Tensor range check" && 0 <= v392 && v392 < 4l);
                int v394;
                v394 = 4l * v390;
                int v395;
                v395 = v394 + v392;
                float v396;
                v396 = v377[v395];
                float v397;
                v397 = v389 + v396;
                v389 = v397;
                v392 += 1l ;
            }
            v390 += 1l ;
        }
        auto v398 = cooperative_groups::coalesced_threads();
        int v399;
        v399 = threadIdx.x;
        int v400;
        v400 = v399 / 32l;
        auto v401 = cooperative_groups::labeled_partition(v398,v400);
        Closure0 v402{};
        float v403;
        v403 = cooperative_groups::reduce(v401, v389, v402);
        int v404[8l];
        int v405;
        v405 = 0l;
        while (while_method_5(v405)){
            int v407;
            v407 = 0l;
            while (while_method_1(v407)){
                assert("Tensor range check" && 0 <= v405 && v405 < 2l);
                assert("Tensor range check" && 0 <= v407 && v407 < 4l);
                int v409;
                v409 = 4l * v405;
                int v410;
                v410 = v409 + v407;
                bool v411;
                v411 = v367[v410];
                int v412;
                if (v411){
                    v412 = 1l;
                } else {
                    v412 = 0l;
                }
                assert("Tensor range check" && 0 <= v405 && v405 < 2l);
                assert("Tensor range check" && 0 <= v407 && v407 < 4l);
                v404[v410] = v412;
                v407 += 1l ;
            }
            v405 += 1l ;
        }
        int v413;
        v413 = 0l;
        int v414;
        v414 = 0l;
        while (while_method_5(v414)){
            int v416;
            v416 = 0l;
            while (while_method_1(v416)){
                assert("Tensor range check" && 0 <= v414 && v414 < 2l);
                assert("Tensor range check" && 0 <= v416 && v416 < 4l);
                int v418;
                v418 = 4l * v414;
                int v419;
                v419 = v418 + v416;
                int v420;
                v420 = v404[v419];
                int v421;
                v421 = v413 + v420;
                v413 = v421;
                v416 += 1l ;
            }
            v414 += 1l ;
        }
        auto v422 = cooperative_groups::coalesced_threads();
        int v423;
        v423 = threadIdx.x;
        int v424;
        v424 = v423 / 32l;
        auto v425 = cooperative_groups::labeled_partition(v422,v424);
        Closure4 v426{};
        int v427;
        v427 = cooperative_groups::reduce(v425, v413, v426);
        float v428;
        v428 = (float)v427;
        float v429;
        v429 = 1.0f / v428;
        float v430[8l];
        int v431;
        v431 = 0l;
        while (while_method_5(v431)){
            int v433;
            v433 = 0l;
            while (while_method_1(v433)){
                assert("Tensor range check" && 0 <= v431 && v431 < 2l);
                assert("Tensor range check" && 0 <= v433 && v433 < 4l);
                int v435;
                v435 = 4l * v431;
                int v436;
                v436 = v435 + v433;
                float v437;
                v437 = v377[v436];
                bool v438;
                v438 = v367[v436];
                bool v439;
                v439 = v438 == false;
                float v444;
                if (v439){
                    v444 = 0.0f;
                } else {
                    bool v440;
                    v440 = v403 == 0.0f;
                    bool v441;
                    v441 = v440 != true;
                    if (v441){
                        float v442;
                        v442 = v437 / v403;
                        v444 = v442;
                    } else {
                        v444 = v429;
                    }
                }
                assert("Tensor range check" && 0 <= v431 && v431 < 2l);
                assert("Tensor range check" && 0 <= v433 && v433 < 4l);
                v430[v436] = v444;
                v433 += 1l ;
            }
            v431 += 1l ;
        }
        int v445;
        v445 = 0l;
        while (while_method_5(v445)){
            assert("Tensor range check" && 0 <= v445 && v445 < 2l);
            int v447;
            v447 = 128l * v445;
            int v448;
            v448 = v447 + v332;
            assert("Tensor range check" && 0 <= v445 && v445 < 2l);
            int v449;
            v449 = 4l * v445;
            int4* v450;
            v450 = reinterpret_cast<int4*>(v430 + v449);
            int4* v451;
            v451 = reinterpret_cast<int4*>(v328 + v448);
            assert("Pointer alignment check" && (unsigned long long)(v450) % 4l == 0 && (unsigned long long)(v451) % 4l == 0);
            *v451 = *v450;
            v445 += 1l ;
        }
        assert("Tensor range check" && 0 <= v326 && v326 < 32l);
        v315 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v306 && v306 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v452;
    v452 = threadIdx.x;
    int v453;
    v453 = blockIdx.x;
    int v454;
    v454 = v453 * 32l;
    int v455;
    v455 = v452 + v454;
    unsigned long long v456;
    v456 = (unsigned long long)v455;
    curandStatePhilox4_32_10_t v457;
    curand_init(12344321ull,v456,0ull,&v457);
    float * v458;
    v458 = v1+v9;
    if (v137){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v136);
    } else {
    }
    extern __shared__ unsigned char v461[];
    if (v141){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v140);
    } else {
    }
    float * * v463;
    v463 = reinterpret_cast<float * *>(&v461[0ull]);
    int * v465;
    v465 = reinterpret_cast<int *>(&v461[v30]);
    int v467;
    v467 = threadIdx.x;
    assert("Tensor range check" && 0 <= v467 && v467 < 32l);
    v463[v467] = v458;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v468;
    v468 = 0l <= v467;
    bool v469;
    v469 = v468 == false;
    if (v469){
        assert("The index needs to be zero or positive." && v468);
    } else {
    }
    int v471;
    v471 = v467 % 32l;
    int v472;
    v472 = v467 / 32l;
    bool v473;
    v473 = v472 < 1l;
    bool v474;
    v474 = v473 == false;
    if (v474){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v473);
    } else {
    }
    assert("Tensor range check" && 0 <= v472 && v472 < 1l);
    int v476;
    v476 = 0l;
    while (while_method_4(v476)){
        bool v478;
        v478 = 0l <= v472;
        bool v479;
        v479 = v478 && v473;
        bool v480;
        v480 = v479 == false;
        if (v480){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v479);
        } else {
        }
        bool v482;
        v482 = 0l <= v476;
        bool v484;
        if (v482){
            bool v483;
            v483 = v476 < 32l;
            v484 = v483;
        } else {
            v484 = false;
        }
        bool v485;
        v485 = v484 == false;
        if (v485){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v484);
        } else {
        }
        int v487;
        v487 = v476 + v472;
        assert("Tensor range check" && 0 <= v476 && v476 < 32l);
        float * v488;
        v488 = v463[v487];
        int v489;
        v489 = blockIdx.x;
        int v490;
        v490 = v489 * 32l;
        int v491;
        v491 = v490 + v487;
        assert("Tensor range check" && 0 <= v471 && v471 < 32l);
        int v492;
        v492 = 4l * v471;
        float v493[8l];
        int v494[8l];
        int v495;
        v495 = 0l;
        while (while_method_5(v495)){
            assert("Tensor range check" && 0 <= v495 && v495 < 2l);
            int v497;
            v497 = 4l * v495;
            assert("Tensor range check" && 0 <= v495 && v495 < 2l);
            int v498;
            v498 = 128l * v495;
            int v499;
            v499 = v498 + v492;
            int4* v500;
            v500 = reinterpret_cast<int4*>(v488 + v499);
            int4* v501;
            v501 = reinterpret_cast<int4*>(v493 + v497);
            assert("Pointer alignment check" && (unsigned long long)(v500) % 4l == 0 && (unsigned long long)(v501) % 4l == 0);
            *v501 = *v500;
            v495 += 1l ;
        }
        int v502;
        v502 = 0l;
        while (while_method_5(v502)){
            int v504;
            v504 = 0l;
            while (while_method_1(v504)){
                bool v506;
                v506 = 0l <= v504;
                bool v508;
                if (v506){
                    bool v507;
                    v507 = v504 < 4l;
                    v508 = v507;
                } else {
                    v508 = false;
                }
                bool v509;
                v509 = v508 == false;
                if (v509){
                    assert("The indices should be inside the range of the dimension." && v508);
                } else {
                }
                bool v511;
                v511 = 0l <= v471;
                bool v513;
                if (v511){
                    bool v512;
                    v512 = v471 < 32l;
                    v513 = v512;
                } else {
                    v513 = false;
                }
                bool v514;
                v514 = v513 == false;
                if (v514){
                    assert("The indices should be inside the range of the dimension." && v513);
                } else {
                }
                int v516;
                v516 = v471 * 4l;
                int v517;
                v517 = v504 + v516;
                bool v518;
                v518 = 0l <= v502;
                bool v520;
                if (v518){
                    bool v519;
                    v519 = v502 < 2l;
                    v520 = v519;
                } else {
                    v520 = false;
                }
                bool v521;
                v521 = v520 == false;
                if (v521){
                    assert("The indices should be inside the range of the dimension." && v520);
                } else {
                }
                int v523;
                v523 = v502 * 128l;
                int v524;
                v524 = v517 + v523;
                assert("Tensor range check" && 0 <= v502 && v502 < 2l);
                assert("Tensor range check" && 0 <= v504 && v504 < 4l);
                int v525;
                v525 = 4l * v502;
                int v526;
                v526 = v525 + v504;
                v494[v526] = v524;
                v504 += 1l ;
            }
            v502 += 1l ;
        }
        bool v527[8l];
        int v528;
        v528 = 0l;
        while (while_method_5(v528)){
            int v530;
            v530 = 0l;
            while (while_method_1(v530)){
                assert("Tensor range check" && 0 <= v528 && v528 < 2l);
                assert("Tensor range check" && 0 <= v530 && v530 < 4l);
                int v532;
                v532 = 4l * v528;
                int v533;
                v533 = v532 + v530;
                float v534;
                v534 = v493[v533];
                int v535;
                v535 = v494[v533];
                bool v536;
                v536 = v535 < 3l;
                assert("Tensor range check" && 0 <= v528 && v528 < 2l);
                assert("Tensor range check" && 0 <= v530 && v530 < 4l);
                v527[v533] = v536;
                v530 += 1l ;
            }
            v528 += 1l ;
        }
        int v537[8l];
        int v538;
        v538 = 0l;
        while (while_method_5(v538)){
            int v540;
            v540 = 0l;
            while (while_method_1(v540)){
                assert("Tensor range check" && 0 <= v538 && v538 < 2l);
                assert("Tensor range check" && 0 <= v540 && v540 < 4l);
                int v542;
                v542 = 4l * v538;
                int v543;
                v543 = v542 + v540;
                bool v544;
                v544 = v527[v543];
                int v545;
                if (v544){
                    v545 = 1l;
                } else {
                    v545 = 0l;
                }
                assert("Tensor range check" && 0 <= v538 && v538 < 2l);
                assert("Tensor range check" && 0 <= v540 && v540 < 4l);
                v537[v543] = v545;
                v540 += 1l ;
            }
            v538 += 1l ;
        }
        int v546;
        v546 = 0l;
        int v547;
        v547 = 0l;
        while (while_method_5(v547)){
            int v549;
            v549 = 0l;
            while (while_method_1(v549)){
                assert("Tensor range check" && 0 <= v547 && v547 < 2l);
                assert("Tensor range check" && 0 <= v549 && v549 < 4l);
                int v551;
                v551 = 4l * v547;
                int v552;
                v552 = v551 + v549;
                int v553;
                v553 = v537[v552];
                int v554;
                v554 = v546 + v553;
                v546 = v554;
                v549 += 1l ;
            }
            v547 += 1l ;
        }
        auto v555 = cooperative_groups::coalesced_threads();
        int v556;
        v556 = threadIdx.x;
        int v557;
        v557 = v556 / 32l;
        auto v558 = cooperative_groups::labeled_partition(v555,v557);
        Closure4 v559{};
        int v560;
        v560 = cooperative_groups::reduce(v558, v546, v559);
        float v561[8l];
        int v562;
        v562 = 0l;
        while (while_method_5(v562)){
            int v564;
            v564 = 0l;
            while (while_method_1(v564)){
                assert("Tensor range check" && 0 <= v562 && v562 < 2l);
                assert("Tensor range check" && 0 <= v564 && v564 < 4l);
                int v566;
                v566 = 4l * v562;
                int v567;
                v567 = v566 + v564;
                float v568;
                v568 = v493[v567];
                bool v569;
                v569 = v527[v567];
                float v570;
                if (v569){
                    v570 = v568;
                } else {
                    v570 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v562 && v562 < 2l);
                assert("Tensor range check" && 0 <= v564 && v564 < 4l);
                v561[v567] = v570;
                v564 += 1l ;
            }
            v562 += 1l ;
        }
        float v571;
        v571 = 0.0f;
        int v572;
        v572 = 0l;
        while (while_method_5(v572)){
            int v574;
            v574 = 0l;
            while (while_method_1(v574)){
                assert("Tensor range check" && 0 <= v572 && v572 < 2l);
                assert("Tensor range check" && 0 <= v574 && v574 < 4l);
                int v576;
                v576 = 4l * v572;
                int v577;
                v577 = v576 + v574;
                float v578;
                v578 = v561[v577];
                float v579;
                v579 = v571 + v578;
                v571 = v579;
                v574 += 1l ;
            }
            v572 += 1l ;
        }
        auto v580 = cooperative_groups::coalesced_threads();
        int v581;
        v581 = threadIdx.x;
        int v582;
        v582 = v581 / 32l;
        auto v583 = cooperative_groups::labeled_partition(v580,v582);
        Closure0 v584{};
        float v585;
        v585 = cooperative_groups::reduce(v583, v571, v584);
        float v586;
        v586 = (float)v560;
        float v587;
        v587 = v585 / v586;
        float v588[8l];
        int v589;
        v589 = 0l;
        while (while_method_5(v589)){
            int v591;
            v591 = 0l;
            while (while_method_1(v591)){
                assert("Tensor range check" && 0 <= v589 && v589 < 2l);
                assert("Tensor range check" && 0 <= v591 && v591 < 4l);
                int v593;
                v593 = 4l * v589;
                int v594;
                v594 = v593 + v591;
                float v595;
                v595 = v493[v594];
                bool v596;
                v596 = v527[v594];
                float v597;
                if (v596){
                    v597 = v595;
                } else {
                    v597 = -1.0f / 0.0f;
                }
                float v598;
                v598 = v597 - v587;
                float v599;
                v599 = exp(v598);
                assert("Tensor range check" && 0 <= v589 && v589 < 2l);
                assert("Tensor range check" && 0 <= v591 && v591 < 4l);
                v588[v594] = v599;
                v591 += 1l ;
            }
            v589 += 1l ;
        }
        float v600;
        v600 = 0.0f;
        int v601;
        v601 = 0l;
        while (while_method_5(v601)){
            int v603;
            v603 = 0l;
            while (while_method_1(v603)){
                assert("Tensor range check" && 0 <= v601 && v601 < 2l);
                assert("Tensor range check" && 0 <= v603 && v603 < 4l);
                int v605;
                v605 = 4l * v601;
                int v606;
                v606 = v605 + v603;
                float v607;
                v607 = v588[v606];
                float v608;
                v608 = v600 + v607;
                v600 = v608;
                v603 += 1l ;
            }
            v601 += 1l ;
        }
        auto v609 = cooperative_groups::coalesced_threads();
        int v610;
        v610 = threadIdx.x;
        int v611;
        v611 = v610 / 32l;
        auto v612 = cooperative_groups::labeled_partition(v609,v611);
        float v613;
        v613 = cooperative_groups::reduce(v612, v600, v584);
        float v614[8l];
        int v615;
        v615 = 0l;
        while (while_method_5(v615)){
            int v617;
            v617 = 0l;
            while (while_method_1(v617)){
                assert("Tensor range check" && 0 <= v615 && v615 < 2l);
                assert("Tensor range check" && 0 <= v617 && v617 < 4l);
                int v619;
                v619 = 4l * v615;
                int v620;
                v620 = v619 + v617;
                float v621;
                v621 = v588[v620];
                float v622;
                v622 = v621 / v613;
                assert("Tensor range check" && 0 <= v615 && v615 < 2l);
                assert("Tensor range check" && 0 <= v617 && v617 < 4l);
                v614[v620] = v622;
                v617 += 1l ;
            }
            v615 += 1l ;
        }
        float v623[8l];
        float v624;
        v624 = 0.0f;
        int v625;
        v625 = 0l;
        while (while_method_5(v625)){
            assert("Tensor range check" && 0 <= v625 && v625 < 2l);
            int v627;
            v627 = 4l * v625;
            assert("Tensor range check" && 0 <= v625 && v625 < 2l);
            int v628; float v629;
            Tuple0 tmp54 = Tuple0{0l, 0.0f};
            v628 = tmp54.v0; v629 = tmp54.v1;
            while (while_method_1(v628)){
                assert("Tensor range check" && 0 <= v628 && v628 < 4l);
                int v631;
                v631 = v628 + v627;
                float v632;
                v632 = v614[v631];
                float v633;
                v633 = v629 + v632;
                v629 = v633;
                v628 += 1l ;
            }
            auto v634 = cooperative_groups::coalesced_threads();
            int v635;
            v635 = threadIdx.x;
            int v636;
            v636 = v635 / 32l;
            auto v637 = cooperative_groups::labeled_partition(v634,v636);
            Closure2 v638{};
            float v639;
            v639 = cooperative_groups::inclusive_scan(v637, v629, v638);
            float v640;
            v640 = v637.shfl_up(v639,1);
            bool v641;
            v641 = v637.thread_rank() == 0;
            float v642;
            if (v641){
                v642 = 0.0f;
            } else {
                v642 = v640;
            }
            float v643;
            v643 = v637.shfl(v639,v637.num_threads()-1);
            float v644;
            v644 = v624 + v642;
            int v645; float v646;
            Tuple0 tmp55 = Tuple0{0l, v644};
            v645 = tmp55.v0; v646 = tmp55.v1;
            while (while_method_1(v645)){
                assert("Tensor range check" && 0 <= v645 && v645 < 4l);
                int v648;
                v648 = v645 + v627;
                float v649;
                v649 = v614[v648];
                float v650;
                v650 = v646 + v649;
                assert("Tensor range check" && 0 <= v645 && v645 < 4l);
                v623[v648] = v650;
                v646 = v650;
                v645 += 1l ;
            }
            float v651;
            v651 = v624 + v643;
            v624 = v651;
            v625 += 1l ;
        }
        float v652[8l];
        bool v653[8l];
        int v654;
        v654 = 0l;
        while (while_method_5(v654)){
            int v656;
            v656 = 0l;
            while (while_method_1(v656)){
                assert("Tensor range check" && 0 <= v654 && v654 < 2l);
                assert("Tensor range check" && 0 <= v656 && v656 < 4l);
                int v658;
                v658 = 4l * v654;
                int v659;
                v659 = v658 + v656;
                float v660;
                v660 = v623[v659];
                float v661;
                v661 = v614[v659];
                bool v662;
                v662 = v661 > 0.0f;
                assert("Tensor range check" && 0 <= v654 && v654 < 2l);
                assert("Tensor range check" && 0 <= v656 && v656 < 4l);
                v652[v659] = v660;
                v653[v659] = v662;
                v656 += 1l ;
            }
            v654 += 1l ;
        }
        float v663; bool v664;
        Tuple3 tmp56 = Tuple3{-1.0f / 0.0f, false};
        v663 = tmp56.v0; v664 = tmp56.v1;
        int v665;
        v665 = 0l;
        while (while_method_5(v665)){
            int v667;
            v667 = 0l;
            while (while_method_1(v667)){
                assert("Tensor range check" && 0 <= v665 && v665 < 2l);
                assert("Tensor range check" && 0 <= v667 && v667 < 4l);
                int v669;
                v669 = 4l * v665;
                int v670;
                v670 = v669 + v667;
                float v671;
                v671 = v652[v670];
                bool v672;
                v672 = v653[v670];
                float v679; bool v680;
                if (v664){
                    if (v672){
                        bool v673;
                        v673 = v663 >= v671;
                        float v674;
                        if (v673){
                            v674 = v663;
                        } else {
                            v674 = v671;
                        }
                        v679 = v674; v680 = true;
                    } else {
                        v679 = v663; v680 = v664;
                    }
                } else {
                    if (v672){
                        v679 = v671; v680 = v672;
                    } else {
                        v679 = v663; v680 = v664;
                    }
                }
                v663 = v679;
                v664 = v680;
                v667 += 1l ;
            }
            v665 += 1l ;
        }
        auto v681 = cooperative_groups::coalesced_threads();
        int v682;
        v682 = threadIdx.x;
        int v683;
        v683 = v682 / 32l;
        auto v684 = cooperative_groups::labeled_partition(v681,v683);
        Closure5 v685{};
        float v686; bool v687;
        Tuple3 tmp57 = cooperative_groups::reduce(v684, Tuple3{v663, v664}, v685);
        v686 = tmp57.v0; v687 = tmp57.v1;
        bool v688;
        v688 = v687 == false;
        if (v688){
            assert("The local reduce must be true." && v687);
        } else {
        }
        float v690[8l];
        int v691[8l];
        int v692;
        v692 = 0l;
        while (while_method_5(v692)){
            int v694;
            v694 = 0l;
            while (while_method_1(v694)){
                assert("Tensor range check" && 0 <= v692 && v692 < 2l);
                assert("Tensor range check" && 0 <= v694 && v694 < 4l);
                int v696;
                v696 = 4l * v692;
                int v697;
                v697 = v696 + v694;
                int v698;
                v698 = v494[v697];
                float v699;
                v699 = curand_uniform(&v457);
                assert("Tensor range check" && 0 <= v692 && v692 < 2l);
                assert("Tensor range check" && 0 <= v694 && v694 < 4l);
                v690[v697] = v699;
                v691[v697] = v698;
                v694 += 1l ;
            }
            v692 += 1l ;
        }
        float v700; int v701;
        Tuple1 tmp58 = Tuple1{0.0f, 2147483647l};
        v700 = tmp58.v0; v701 = tmp58.v1;
        int v702;
        v702 = 0l;
        while (while_method_5(v702)){
            int v704;
            v704 = 0l;
            while (while_method_1(v704)){
                assert("Tensor range check" && 0 <= v702 && v702 < 2l);
                assert("Tensor range check" && 0 <= v704 && v704 < 4l);
                int v706;
                v706 = 4l * v702;
                int v707;
                v707 = v706 + v704;
                float v708;
                v708 = v690[v707];
                int v709;
                v709 = v691[v707];
                bool v710;
                v710 = v701 < v709;
                float v711; int v712;
                if (v710){
                    v711 = v700; v712 = v701;
                } else {
                    v711 = v708; v712 = v709;
                }
                v700 = v711;
                v701 = v712;
                v704 += 1l ;
            }
            v702 += 1l ;
        }
        auto v713 = cooperative_groups::coalesced_threads();
        int v714;
        v714 = threadIdx.x;
        int v715;
        v715 = v714 / 32l;
        auto v716 = cooperative_groups::labeled_partition(v713,v715);
        Closure6 v717{};
        float v718; int v719;
        Tuple1 tmp59 = cooperative_groups::reduce(v716, Tuple1{v700, v701}, v717);
        v718 = tmp59.v0; v719 = tmp59.v1;
        float v720;
        v720 = v686 * v718;
        int v721[8l];
        bool v722[8l];
        int v723;
        v723 = 0l;
        while (while_method_5(v723)){
            int v725;
            v725 = 0l;
            while (while_method_1(v725)){
                assert("Tensor range check" && 0 <= v723 && v723 < 2l);
                assert("Tensor range check" && 0 <= v725 && v725 < 4l);
                int v727;
                v727 = 4l * v723;
                int v728;
                v728 = v727 + v725;
                float v729;
                v729 = v652[v728];
                bool v730;
                v730 = v653[v728];
                int v731;
                v731 = v494[v728];
                int v734; bool v735;
                if (v730){
                    float v732;
                    v732 = v729 - v720;
                    bool v733;
                    v733 = v732 >= 0.0f;
                    v734 = v731; v735 = v733;
                } else {
                    v734 = 2147483647l; v735 = false;
                }
                assert("Tensor range check" && 0 <= v723 && v723 < 2l);
                assert("Tensor range check" && 0 <= v725 && v725 < 4l);
                v721[v728] = v734;
                v722[v728] = v735;
                v725 += 1l ;
            }
            v723 += 1l ;
        }
        int v736; bool v737;
        Tuple4 tmp60 = Tuple4{2147483647l, false};
        v736 = tmp60.v0; v737 = tmp60.v1;
        int v738;
        v738 = 0l;
        while (while_method_5(v738)){
            int v740;
            v740 = 0l;
            while (while_method_1(v740)){
                assert("Tensor range check" && 0 <= v738 && v738 < 2l);
                assert("Tensor range check" && 0 <= v740 && v740 < 4l);
                int v742;
                v742 = 4l * v738;
                int v743;
                v743 = v742 + v740;
                int v744;
                v744 = v721[v743];
                bool v745;
                v745 = v722[v743];
                int v752; bool v753;
                if (v737){
                    if (v745){
                        bool v746;
                        v746 = v736 < v744;
                        int v747;
                        if (v746){
                            v747 = v736;
                        } else {
                            v747 = v744;
                        }
                        v752 = v747; v753 = true;
                    } else {
                        v752 = v736; v753 = v737;
                    }
                } else {
                    if (v745){
                        v752 = v744; v753 = v745;
                    } else {
                        v752 = v736; v753 = v737;
                    }
                }
                v736 = v752;
                v737 = v753;
                v740 += 1l ;
            }
            v738 += 1l ;
        }
        auto v754 = cooperative_groups::coalesced_threads();
        int v755;
        v755 = threadIdx.x;
        int v756;
        v756 = v755 / 32l;
        auto v757 = cooperative_groups::labeled_partition(v754,v756);
        Closure7 v758{};
        int v759; bool v760;
        Tuple4 tmp61 = cooperative_groups::reduce(v757, Tuple4{v736, v737}, v758);
        v759 = tmp61.v0; v760 = tmp61.v1;
        bool v761;
        v761 = v760 == false;
        if (v761){
            assert("The local reduce must be true." && v760);
        } else {
        }
        int v763;
        v763 = 0l;
        while (while_method_5(v763)){
            assert("Tensor range check" && 0 <= v763 && v763 < 2l);
            assert("Tensor range check" && 0 <= v763 && v763 < 2l);
            v763 += 1l ;
        }
        assert("Tensor range check" && 0 <= v487 && v487 < 32l);
        v465[v487] = v759;
        v476 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v467 && v467 < 32l);
    int v765;
    v765 = v465[v467];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v766;
    v766 = threadIdx.x;
    assert("Tensor range check" && 0 <= v766 && v766 < 32l);
    v5[v766] = v765;
    return ;
}
extern "C" __global__ void entry4(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14, float * v15, int * v16) {
    auto v17 = cooperative_groups::this_grid();
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
    v22 = v18 % 16l;
    int v23;
    v23 = v18 / 16l;
    bool v24;
    v24 = v23 < 2l;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
    } else {
    }
    assert("Tensor range check" && 0 <= v23 && v23 < 2l);
    assert("Tensor range check" && 0 <= v22 && v22 < 16l);
    int v27;
    v27 = 4l * v22;
    int v28;
    v28 = 64l * v23;
    int v29;
    v29 = v28 + v27;
    assert("Tensor range check" && 0 <= v23 && v23 < 2l);
    assert("Tensor range check" && 0 <= v22 && v22 < 16l);
    int v30;
    v30 = blockIdx.x;
    int v31;
    v31 = v30;
    while (while_method_2(v31)){
        bool v33;
        v33 = 0l <= v31;
        bool v34;
        v34 = v33 == false;
        if (v34){
            assert("The index needs to be zero or positive." && v33);
        } else {
        }
        bool v36;
        v36 = v31 < 64l;
        bool v37;
        v37 = v36 == false;
        if (v37){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v36);
        } else {
        }
        assert("Tensor range check" && 0 <= v31 && v31 < 64l);
        int v39;
        v39 = 128l * v31;
        int v40;
        v40 = v39 + v29;
        int v41[4l];
        int v42[4l];
        int v43;
        v43 = 0l;
        while (while_method_3(v43)){
            assert("Tensor range check" && 0 <= v43 && v43 < 1l);
            int v45;
            v45 = 4l * v43;
            assert("Tensor range check" && 0 <= v43 && v43 < 1l);
            int v46;
            v46 = 64l * v43;
            int v47;
            v47 = v46 + v40;
            int4* v48;
            v48 = reinterpret_cast<int4*>(v0 + v47);
            int4* v49;
            v49 = reinterpret_cast<int4*>(v41 + v45);
            assert("Pointer alignment check" && (unsigned long long)(v48) % 4l == 0 && (unsigned long long)(v49) % 4l == 0);
            *v49 = *v48;
            v43 += 1l ;
        }
        int v50;
        v50 = 0l;
        while (while_method_3(v50)){
            int v52;
            v52 = 0l;
            while (while_method_1(v52)){
                bool v54;
                v54 = 0l <= v52;
                bool v56;
                if (v54){
                    bool v55;
                    v55 = v52 < 4l;
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
                v59 = 0l <= v22;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v22 < 16l;
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
                v64 = v22 * 4l;
                int v65;
                v65 = v52 + v64;
                bool v66;
                v66 = 0l <= v50;
                bool v68;
                if (v66){
                    bool v67;
                    v67 = v50 < 1l;
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
                v71 = v50 * 64l;
                int v72;
                v72 = v65 + v71;
                assert("Tensor range check" && 0 <= v50 && v50 < 1l);
                assert("Tensor range check" && 0 <= v52 && v52 < 4l);
                int v73;
                v73 = 4l * v50;
                int v74;
                v74 = v73 + v52;
                v42[v74] = v72;
                v52 += 1l ;
            }
            v50 += 1l ;
        }
        bool v75;
        v75 = 0l <= v23;
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
        v82 = v31 * 2l;
        int v83;
        v83 = v82 + v23;
        assert("Tensor range check" && 0 <= v31 && v31 < 64l);
        int v84;
        v84 = 0l;
        while (while_method_3(v84)){
            assert("Tensor range check" && 0 <= v84 && v84 < 1l);
            int v86;
            v86 = 64l * v84;
            int v87;
            v87 = v86 + v40;
            assert("Tensor range check" && 0 <= v84 && v84 < 1l);
            int v88;
            v88 = 4l * v84;
            int4* v89;
            v89 = reinterpret_cast<int4*>(v41 + v88);
            int4* v90;
            v90 = reinterpret_cast<int4*>(v2 + v87);
            assert("Pointer alignment check" && (unsigned long long)(v89) % 4l == 0 && (unsigned long long)(v90) % 4l == 0);
            *v90 = *v89;
            v84 += 1l ;
        }
        v31 += 1l ;
    }
    v17.sync() ;
    int v91;
    v91 = threadIdx.x;
    bool v92;
    v92 = 0l <= v91;
    bool v93;
    v93 = v92 == false;
    if (v93){
        assert("The index needs to be zero or positive." && v92);
    } else {
    }
    int v95;
    v95 = v91 % 16l;
    int v96;
    v96 = v91 / 16l;
    bool v97;
    v97 = v96 < 2l;
    bool v98;
    v98 = v97 == false;
    if (v98){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v97);
    } else {
    }
    assert("Tensor range check" && 0 <= v96 && v96 < 2l);
    assert("Tensor range check" && 0 <= v95 && v95 < 16l);
    int v100;
    v100 = 4l * v95;
    int v101;
    v101 = 64l * v96;
    int v102;
    v102 = v101 + v100;
    assert("Tensor range check" && 0 <= v96 && v96 < 2l);
    assert("Tensor range check" && 0 <= v95 && v95 < 16l);
    int v103;
    v103 = blockIdx.x;
    int v104;
    v104 = v103;
    while (while_method_2(v104)){
        bool v106;
        v106 = 0l <= v104;
        bool v107;
        v107 = v106 == false;
        if (v107){
            assert("The index needs to be zero or positive." && v106);
        } else {
        }
        bool v109;
        v109 = v104 < 64l;
        bool v110;
        v110 = v109 == false;
        if (v110){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v109);
        } else {
        }
        assert("Tensor range check" && 0 <= v104 && v104 < 64l);
        int v112;
        v112 = 128l * v104;
        int v113;
        v113 = v112 + v102;
        float v114[4l];
        int v115[4l];
        int v116;
        v116 = 0l;
        while (while_method_3(v116)){
            assert("Tensor range check" && 0 <= v116 && v116 < 1l);
            int v118;
            v118 = 4l * v116;
            assert("Tensor range check" && 0 <= v116 && v116 < 1l);
            int v119;
            v119 = 64l * v116;
            int v120;
            v120 = v119 + v113;
            int4* v121;
            v121 = reinterpret_cast<int4*>(v1 + v120);
            int4* v122;
            v122 = reinterpret_cast<int4*>(v114 + v118);
            assert("Pointer alignment check" && (unsigned long long)(v121) % 4l == 0 && (unsigned long long)(v122) % 4l == 0);
            *v122 = *v121;
            v116 += 1l ;
        }
        int v123;
        v123 = 0l;
        while (while_method_3(v123)){
            int v125;
            v125 = 0l;
            while (while_method_1(v125)){
                bool v127;
                v127 = 0l <= v125;
                bool v129;
                if (v127){
                    bool v128;
                    v128 = v125 < 4l;
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
                v132 = 0l <= v95;
                bool v134;
                if (v132){
                    bool v133;
                    v133 = v95 < 16l;
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
                v137 = v95 * 4l;
                int v138;
                v138 = v125 + v137;
                bool v139;
                v139 = 0l <= v123;
                bool v141;
                if (v139){
                    bool v140;
                    v140 = v123 < 1l;
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
                v144 = v123 * 64l;
                int v145;
                v145 = v138 + v144;
                assert("Tensor range check" && 0 <= v123 && v123 < 1l);
                assert("Tensor range check" && 0 <= v125 && v125 < 4l);
                int v146;
                v146 = 4l * v123;
                int v147;
                v147 = v146 + v125;
                v115[v147] = v145;
                v125 += 1l ;
            }
            v123 += 1l ;
        }
        bool v148;
        v148 = 0l <= v96;
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
        v155 = v104 * 2l;
        int v156;
        v156 = v155 + v96;
        int v157[4l];
        int v158[4l];
        int v159;
        v159 = 0l;
        while (while_method_3(v159)){
            int v161;
            v161 = 0l;
            while (while_method_1(v161)){
                assert("Tensor range check" && 0 <= v159 && v159 < 1l);
                assert("Tensor range check" && 0 <= v161 && v161 < 4l);
                int v163;
                v163 = 4l * v159;
                int v164;
                v164 = v163 + v161;
                int v165;
                v165 = v115[v164];
                assert("Tensor range check" && 0 <= v159 && v159 < 1l);
                assert("Tensor range check" && 0 <= v161 && v161 < 4l);
                v157[v164] = v156;
                v158[v164] = v165;
                v161 += 1l ;
            }
            v159 += 1l ;
        }
        assert("Tensor range check" && 0 <= v104 && v104 < 64l);
        int v166;
        v166 = 0l;
        while (while_method_3(v166)){
            assert("Tensor range check" && 0 <= v166 && v166 < 1l);
            int v168;
            v168 = 64l * v166;
            int v169;
            v169 = v168 + v113;
            assert("Tensor range check" && 0 <= v166 && v166 < 1l);
            int v170;
            v170 = 4l * v166;
            int4* v171;
            v171 = reinterpret_cast<int4*>(v157 + v170);
            int4* v172;
            v172 = reinterpret_cast<int4*>(v9 + v169);
            assert("Pointer alignment check" && (unsigned long long)(v171) % 4l == 0 && (unsigned long long)(v172) % 4l == 0);
            *v172 = *v171;
            int4* v173;
            v173 = reinterpret_cast<int4*>(v158 + v170);
            int4* v174;
            v174 = reinterpret_cast<int4*>(v10 + v169);
            assert("Pointer alignment check" && (unsigned long long)(v173) % 4l == 0 && (unsigned long long)(v174) % 4l == 0);
            *v174 = *v173;
            v166 += 1l ;
        }
        v104 += 1l ;
    }
    v17.sync() ;
    int v175;
    v175 = threadIdx.x;
    bool v176;
    v176 = 0l <= v175;
    bool v177;
    v177 = v176 == false;
    if (v177){
        assert("The index needs to be zero or positive." && v176);
    } else {
    }
    int v179;
    v179 = v175 % 16l;
    int v180;
    v180 = v175 / 16l;
    bool v181;
    v181 = v180 < 2l;
    bool v182;
    v182 = v181 == false;
    if (v182){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v181);
    } else {
    }
    assert("Tensor range check" && 0 <= v180 && v180 < 2l);
    assert("Tensor range check" && 0 <= v179 && v179 < 16l);
    int v184;
    v184 = 4l * v179;
    int v185;
    v185 = 64l * v180;
    int v186;
    v186 = v185 + v184;
    assert("Tensor range check" && 0 <= v180 && v180 < 2l);
    int v187;
    v187 = blockIdx.x;
    int v188;
    v188 = v187;
    while (while_method_2(v188)){
        bool v190;
        v190 = 0l <= v188;
        bool v191;
        v191 = v190 == false;
        if (v191){
            assert("The index needs to be zero or positive." && v190);
        } else {
        }
        bool v193;
        v193 = v188 < 64l;
        bool v194;
        v194 = v193 == false;
        if (v194){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v193);
        } else {
        }
        assert("Tensor range check" && 0 <= v188 && v188 < 64l);
        int v196;
        v196 = 128l * v188;
        int v197;
        v197 = v196 + v186;
        float v198[4l];
        int v199[4l];
        int v200;
        v200 = 0l;
        while (while_method_3(v200)){
            assert("Tensor range check" && 0 <= v200 && v200 < 1l);
            int v202;
            v202 = 4l * v200;
            assert("Tensor range check" && 0 <= v200 && v200 < 1l);
            int v203;
            v203 = 64l * v200;
            int v204;
            v204 = v203 + v197;
            int4* v205;
            v205 = reinterpret_cast<int4*>(v1 + v204);
            int4* v206;
            v206 = reinterpret_cast<int4*>(v198 + v202);
            assert("Pointer alignment check" && (unsigned long long)(v205) % 4l == 0 && (unsigned long long)(v206) % 4l == 0);
            *v206 = *v205;
            v200 += 1l ;
        }
        int v207;
        v207 = 0l;
        while (while_method_3(v207)){
            int v209;
            v209 = 0l;
            while (while_method_1(v209)){
                bool v211;
                v211 = 0l <= v209;
                bool v213;
                if (v211){
                    bool v212;
                    v212 = v209 < 4l;
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
                v216 = 0l <= v179;
                bool v218;
                if (v216){
                    bool v217;
                    v217 = v179 < 16l;
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
                v221 = v179 * 4l;
                int v222;
                v222 = v209 + v221;
                bool v223;
                v223 = 0l <= v207;
                bool v225;
                if (v223){
                    bool v224;
                    v224 = v207 < 1l;
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
                v228 = v207 * 64l;
                int v229;
                v229 = v222 + v228;
                assert("Tensor range check" && 0 <= v207 && v207 < 1l);
                assert("Tensor range check" && 0 <= v209 && v209 < 4l);
                int v230;
                v230 = 4l * v207;
                int v231;
                v231 = v230 + v209;
                v199[v231] = v229;
                v209 += 1l ;
            }
            v207 += 1l ;
        }
        bool v232;
        v232 = 0l <= v180;
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
        v239 = v188 * 2l;
        int v240;
        v240 = v239 + v180;
        assert("Tensor range check" && 0 <= v188 && v188 < 64l);
        int v241;
        v241 = 2l * v188;
        int v242;
        v242 = v241 + v180;
        v11[v242] = v240;
        v188 += 1l ;
    }
    v17.sync() ;
    int v243;
    v243 = threadIdx.x;
    bool v244;
    v244 = 0l <= v243;
    bool v245;
    v245 = v244 == false;
    if (v245){
        assert("The index needs to be zero or positive." && v244);
    } else {
    }
    int v247;
    v247 = v243 % 16l;
    int v248;
    v248 = v243 / 16l;
    bool v249;
    v249 = v248 < 2l;
    bool v250;
    v250 = v249 == false;
    if (v250){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v249);
    } else {
    }
    assert("Tensor range check" && 0 <= v248 && v248 < 2l);
    assert("Tensor range check" && 0 <= v247 && v247 < 16l);
    int v252;
    v252 = 4l * v247;
    int v253;
    v253 = 64l * v248;
    int v254;
    v254 = v253 + v252;
    assert("Tensor range check" && 0 <= v248 && v248 < 2l);
    assert("Tensor range check" && 0 <= v247 && v247 < 16l);
    int v255;
    v255 = blockIdx.x;
    int v256;
    v256 = v255;
    while (while_method_2(v256)){
        bool v258;
        v258 = 0l <= v256;
        bool v259;
        v259 = v258 == false;
        if (v259){
            assert("The index needs to be zero or positive." && v258);
        } else {
        }
        bool v261;
        v261 = v256 < 64l;
        bool v262;
        v262 = v261 == false;
        if (v262){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v261);
        } else {
        }
        assert("Tensor range check" && 0 <= v256 && v256 < 64l);
        int v264;
        v264 = 128l * v256;
        int v265;
        v265 = v264 + v254;
        float v266[4l];
        int v267[4l];
        int v268;
        v268 = 0l;
        while (while_method_3(v268)){
            assert("Tensor range check" && 0 <= v268 && v268 < 1l);
            int v270;
            v270 = 4l * v268;
            assert("Tensor range check" && 0 <= v268 && v268 < 1l);
            int v271;
            v271 = 64l * v268;
            int v272;
            v272 = v271 + v265;
            int4* v273;
            v273 = reinterpret_cast<int4*>(v1 + v272);
            int4* v274;
            v274 = reinterpret_cast<int4*>(v266 + v270);
            assert("Pointer alignment check" && (unsigned long long)(v273) % 4l == 0 && (unsigned long long)(v274) % 4l == 0);
            *v274 = *v273;
            v268 += 1l ;
        }
        int v275;
        v275 = 0l;
        while (while_method_3(v275)){
            int v277;
            v277 = 0l;
            while (while_method_1(v277)){
                bool v279;
                v279 = 0l <= v277;
                bool v281;
                if (v279){
                    bool v280;
                    v280 = v277 < 4l;
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
                v284 = 0l <= v247;
                bool v286;
                if (v284){
                    bool v285;
                    v285 = v247 < 16l;
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
                v289 = v247 * 4l;
                int v290;
                v290 = v277 + v289;
                bool v291;
                v291 = 0l <= v275;
                bool v293;
                if (v291){
                    bool v292;
                    v292 = v275 < 1l;
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
                v296 = v275 * 64l;
                int v297;
                v297 = v290 + v296;
                assert("Tensor range check" && 0 <= v275 && v275 < 1l);
                assert("Tensor range check" && 0 <= v277 && v277 < 4l);
                int v298;
                v298 = 4l * v275;
                int v299;
                v299 = v298 + v277;
                v267[v299] = v297;
                v277 += 1l ;
            }
            v275 += 1l ;
        }
        bool v300;
        v300 = 0l <= v248;
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
        v307 = v256 * 2l;
        int v308;
        v308 = v307 + v248;
        float v309;
        v309 = 0.0f;
        int v310;
        v310 = 0l;
        while (while_method_3(v310)){
            int v312;
            v312 = 0l;
            while (while_method_1(v312)){
                assert("Tensor range check" && 0 <= v310 && v310 < 1l);
                assert("Tensor range check" && 0 <= v312 && v312 < 4l);
                int v314;
                v314 = 4l * v310;
                int v315;
                v315 = v314 + v312;
                float v316;
                v316 = v266[v315];
                float v317;
                v317 = v309 + v316;
                v309 = v317;
                v312 += 1l ;
            }
            v310 += 1l ;
        }
        auto v318 = cooperative_groups::coalesced_threads();
        int v319;
        v319 = threadIdx.x;
        int v320;
        v320 = v319 / 16l;
        auto v321 = cooperative_groups::labeled_partition(v318,v320);
        Closure0 v322{};
        float v323;
        v323 = cooperative_groups::reduce(v321, v309, v322);
        float v324;
        v324 = v323 / 64.0f;
        float v325[4l];
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
                v332 = v266[v331];
                float v333;
                v333 = v332 - v324;
                float v334;
                v334 = exp(v333);
                assert("Tensor range check" && 0 <= v326 && v326 < 1l);
                assert("Tensor range check" && 0 <= v328 && v328 < 4l);
                v325[v331] = v334;
                v328 += 1l ;
            }
            v326 += 1l ;
        }
        float v335;
        v335 = 0.0f;
        int v336;
        v336 = 0l;
        while (while_method_3(v336)){
            int v338;
            v338 = 0l;
            while (while_method_1(v338)){
                assert("Tensor range check" && 0 <= v336 && v336 < 1l);
                assert("Tensor range check" && 0 <= v338 && v338 < 4l);
                int v340;
                v340 = 4l * v336;
                int v341;
                v341 = v340 + v338;
                float v342;
                v342 = v325[v341];
                float v343;
                v343 = v335 + v342;
                v335 = v343;
                v338 += 1l ;
            }
            v336 += 1l ;
        }
        auto v344 = cooperative_groups::coalesced_threads();
        int v345;
        v345 = threadIdx.x;
        int v346;
        v346 = v345 / 16l;
        auto v347 = cooperative_groups::labeled_partition(v344,v346);
        float v348;
        v348 = cooperative_groups::reduce(v347, v335, v322);
        float v349[4l];
        int v350;
        v350 = 0l;
        while (while_method_3(v350)){
            int v352;
            v352 = 0l;
            while (while_method_1(v352)){
                assert("Tensor range check" && 0 <= v350 && v350 < 1l);
                assert("Tensor range check" && 0 <= v352 && v352 < 4l);
                int v354;
                v354 = 4l * v350;
                int v355;
                v355 = v354 + v352;
                float v356;
                v356 = v325[v355];
                float v357;
                v357 = v356 / v348;
                assert("Tensor range check" && 0 <= v350 && v350 < 1l);
                assert("Tensor range check" && 0 <= v352 && v352 < 4l);
                v349[v355] = v357;
                v352 += 1l ;
            }
            v350 += 1l ;
        }
        assert("Tensor range check" && 0 <= v256 && v256 < 64l);
        int v358;
        v358 = 0l;
        while (while_method_3(v358)){
            assert("Tensor range check" && 0 <= v358 && v358 < 1l);
            int v360;
            v360 = 64l * v358;
            int v361;
            v361 = v360 + v265;
            assert("Tensor range check" && 0 <= v358 && v358 < 1l);
            int v362;
            v362 = 4l * v358;
            int4* v363;
            v363 = reinterpret_cast<int4*>(v349 + v362);
            int4* v364;
            v364 = reinterpret_cast<int4*>(v3 + v361);
            assert("Pointer alignment check" && (unsigned long long)(v363) % 4l == 0 && (unsigned long long)(v364) % 4l == 0);
            *v364 = *v363;
            v358 += 1l ;
        }
        v256 += 1l ;
    }
    v17.sync() ;
    int v365;
    v365 = threadIdx.x;
    bool v366;
    v366 = 0l <= v365;
    bool v367;
    v367 = v366 == false;
    if (v367){
        assert("The index needs to be zero or positive." && v366);
    } else {
    }
    int v369;
    v369 = v365 % 16l;
    int v370;
    v370 = v365 / 16l;
    bool v371;
    v371 = v370 < 2l;
    bool v372;
    v372 = v371 == false;
    if (v372){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v371);
    } else {
    }
    assert("Tensor range check" && 0 <= v370 && v370 < 2l);
    assert("Tensor range check" && 0 <= v369 && v369 < 16l);
    int v374;
    v374 = 4l * v369;
    int v375;
    v375 = 64l * v370;
    int v376;
    v376 = v375 + v374;
    assert("Tensor range check" && 0 <= v370 && v370 < 2l);
    assert("Tensor range check" && 0 <= v369 && v369 < 16l);
    int v377;
    v377 = blockIdx.x;
    int v378;
    v378 = v377;
    while (while_method_2(v378)){
        bool v380;
        v380 = 0l <= v378;
        bool v381;
        v381 = v380 == false;
        if (v381){
            assert("The index needs to be zero or positive." && v380);
        } else {
        }
        bool v383;
        v383 = v378 < 64l;
        bool v384;
        v384 = v383 == false;
        if (v384){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v383);
        } else {
        }
        assert("Tensor range check" && 0 <= v378 && v378 < 64l);
        int v386;
        v386 = 128l * v378;
        int v387;
        v387 = v386 + v376;
        float v388[4l];
        int v389[4l];
        int v390;
        v390 = 0l;
        while (while_method_3(v390)){
            assert("Tensor range check" && 0 <= v390 && v390 < 1l);
            int v392;
            v392 = 4l * v390;
            assert("Tensor range check" && 0 <= v390 && v390 < 1l);
            int v393;
            v393 = 64l * v390;
            int v394;
            v394 = v393 + v387;
            int4* v395;
            v395 = reinterpret_cast<int4*>(v1 + v394);
            int4* v396;
            v396 = reinterpret_cast<int4*>(v388 + v392);
            assert("Pointer alignment check" && (unsigned long long)(v395) % 4l == 0 && (unsigned long long)(v396) % 4l == 0);
            *v396 = *v395;
            v390 += 1l ;
        }
        int v397;
        v397 = 0l;
        while (while_method_3(v397)){
            int v399;
            v399 = 0l;
            while (while_method_1(v399)){
                bool v401;
                v401 = 0l <= v399;
                bool v403;
                if (v401){
                    bool v402;
                    v402 = v399 < 4l;
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
                v406 = 0l <= v369;
                bool v408;
                if (v406){
                    bool v407;
                    v407 = v369 < 16l;
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
                v411 = v369 * 4l;
                int v412;
                v412 = v399 + v411;
                bool v413;
                v413 = 0l <= v397;
                bool v415;
                if (v413){
                    bool v414;
                    v414 = v397 < 1l;
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
                v418 = v397 * 64l;
                int v419;
                v419 = v412 + v418;
                assert("Tensor range check" && 0 <= v397 && v397 < 1l);
                assert("Tensor range check" && 0 <= v399 && v399 < 4l);
                int v420;
                v420 = 4l * v397;
                int v421;
                v421 = v420 + v399;
                v389[v421] = v419;
                v399 += 1l ;
            }
            v397 += 1l ;
        }
        bool v422;
        v422 = 0l <= v370;
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
        v429 = v378 * 2l;
        int v430;
        v430 = v429 + v370;
        float v431[4l];
        int v432;
        v432 = 0l;
        while (while_method_3(v432)){
            int v434;
            v434 = 0l;
            while (while_method_1(v434)){
                assert("Tensor range check" && 0 <= v432 && v432 < 1l);
                assert("Tensor range check" && 0 <= v434 && v434 < 4l);
                int v436;
                v436 = 4l * v432;
                int v437;
                v437 = v436 + v434;
                float v438;
                v438 = v388[v437];
                float v439;
                v439 = v438 * v438;
                assert("Tensor range check" && 0 <= v432 && v432 < 1l);
                assert("Tensor range check" && 0 <= v434 && v434 < 4l);
                v431[v437] = v439;
                v434 += 1l ;
            }
            v432 += 1l ;
        }
        float v440;
        v440 = 0.0f;
        int v441;
        v441 = 0l;
        while (while_method_3(v441)){
            int v443;
            v443 = 0l;
            while (while_method_1(v443)){
                assert("Tensor range check" && 0 <= v441 && v441 < 1l);
                assert("Tensor range check" && 0 <= v443 && v443 < 4l);
                int v445;
                v445 = 4l * v441;
                int v446;
                v446 = v445 + v443;
                float v447;
                v447 = v431[v446];
                float v448;
                v448 = v440 + v447;
                v440 = v448;
                v443 += 1l ;
            }
            v441 += 1l ;
        }
        auto v449 = cooperative_groups::coalesced_threads();
        int v450;
        v450 = threadIdx.x;
        int v451;
        v451 = v450 / 16l;
        auto v452 = cooperative_groups::labeled_partition(v449,v451);
        Closure0 v453{};
        float v454;
        v454 = cooperative_groups::reduce(v452, v440, v453);
        float v455[4l];
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
                assert("Tensor range check" && 0 <= v456 && v456 < 1l);
                assert("Tensor range check" && 0 <= v458 && v458 < 4l);
                v455[v461] = v466;
                v458 += 1l ;
            }
            v456 += 1l ;
        }
        assert("Tensor range check" && 0 <= v378 && v378 < 64l);
        int v467;
        v467 = 0l;
        while (while_method_3(v467)){
            assert("Tensor range check" && 0 <= v467 && v467 < 1l);
            int v469;
            v469 = 64l * v467;
            int v470;
            v470 = v469 + v387;
            assert("Tensor range check" && 0 <= v467 && v467 < 1l);
            int v471;
            v471 = 4l * v467;
            int4* v472;
            v472 = reinterpret_cast<int4*>(v455 + v471);
            int4* v473;
            v473 = reinterpret_cast<int4*>(v7 + v470);
            assert("Pointer alignment check" && (unsigned long long)(v472) % 4l == 0 && (unsigned long long)(v473) % 4l == 0);
            *v473 = *v472;
            v467 += 1l ;
        }
        v378 += 1l ;
    }
    v17.sync() ;
    int v474;
    v474 = threadIdx.x;
    bool v475;
    v475 = 0l <= v474;
    bool v476;
    v476 = v475 == false;
    if (v476){
        assert("The index needs to be zero or positive." && v475);
    } else {
    }
    int v478;
    v478 = v474 % 16l;
    int v479;
    v479 = v474 / 16l;
    bool v480;
    v480 = v479 < 2l;
    bool v481;
    v481 = v480 == false;
    if (v481){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v480);
    } else {
    }
    assert("Tensor range check" && 0 <= v479 && v479 < 2l);
    assert("Tensor range check" && 0 <= v478 && v478 < 16l);
    int v483;
    v483 = 4l * v478;
    int v484;
    v484 = 64l * v479;
    int v485;
    v485 = v484 + v483;
    assert("Tensor range check" && 0 <= v479 && v479 < 2l);
    int v486;
    v486 = blockIdx.x;
    int v487;
    v487 = v486;
    while (while_method_2(v487)){
        bool v489;
        v489 = 0l <= v487;
        bool v490;
        v490 = v489 == false;
        if (v490){
            assert("The index needs to be zero or positive." && v489);
        } else {
        }
        bool v492;
        v492 = v487 < 64l;
        bool v493;
        v493 = v492 == false;
        if (v493){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v492);
        } else {
        }
        assert("Tensor range check" && 0 <= v487 && v487 < 64l);
        int v495;
        v495 = 128l * v487;
        int v496;
        v496 = v495 + v485;
        float v497[4l];
        int v498[4l];
        int v499;
        v499 = 0l;
        while (while_method_3(v499)){
            assert("Tensor range check" && 0 <= v499 && v499 < 1l);
            int v501;
            v501 = 4l * v499;
            assert("Tensor range check" && 0 <= v499 && v499 < 1l);
            int v502;
            v502 = 64l * v499;
            int v503;
            v503 = v502 + v496;
            int4* v504;
            v504 = reinterpret_cast<int4*>(v1 + v503);
            int4* v505;
            v505 = reinterpret_cast<int4*>(v497 + v501);
            assert("Pointer alignment check" && (unsigned long long)(v504) % 4l == 0 && (unsigned long long)(v505) % 4l == 0);
            *v505 = *v504;
            v499 += 1l ;
        }
        int v506;
        v506 = 0l;
        while (while_method_3(v506)){
            int v508;
            v508 = 0l;
            while (while_method_1(v508)){
                bool v510;
                v510 = 0l <= v508;
                bool v512;
                if (v510){
                    bool v511;
                    v511 = v508 < 4l;
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
                v515 = 0l <= v478;
                bool v517;
                if (v515){
                    bool v516;
                    v516 = v478 < 16l;
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
                v520 = v478 * 4l;
                int v521;
                v521 = v508 + v520;
                bool v522;
                v522 = 0l <= v506;
                bool v524;
                if (v522){
                    bool v523;
                    v523 = v506 < 1l;
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
                v527 = v506 * 64l;
                int v528;
                v528 = v521 + v527;
                assert("Tensor range check" && 0 <= v506 && v506 < 1l);
                assert("Tensor range check" && 0 <= v508 && v508 < 4l);
                int v529;
                v529 = 4l * v506;
                int v530;
                v530 = v529 + v508;
                v498[v530] = v528;
                v508 += 1l ;
            }
            v506 += 1l ;
        }
        bool v531;
        v531 = 0l <= v479;
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
        v538 = v487 * 2l;
        int v539;
        v539 = v538 + v479;
        float v540; int v541;
        Tuple1 tmp62 = Tuple1{-1.0f / 0.0f, 0l};
        v540 = tmp62.v0; v541 = tmp62.v1;
        int v542;
        v542 = 0l;
        while (while_method_3(v542)){
            int v544;
            v544 = 0l;
            while (while_method_1(v544)){
                assert("Tensor range check" && 0 <= v542 && v542 < 1l);
                assert("Tensor range check" && 0 <= v544 && v544 < 4l);
                int v546;
                v546 = 4l * v542;
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
                v544 += 1l ;
            }
            v542 += 1l ;
        }
        auto v553 = cooperative_groups::coalesced_threads();
        int v554;
        v554 = threadIdx.x;
        int v555;
        v555 = v554 / 16l;
        auto v556 = cooperative_groups::labeled_partition(v553,v555);
        Closure1 v557{};
        float v558; int v559;
        Tuple1 tmp63 = cooperative_groups::reduce(v556, Tuple1{v540, v541}, v557);
        v558 = tmp63.v0; v559 = tmp63.v1;
        assert("Tensor range check" && 0 <= v487 && v487 < 64l);
        int v560;
        v560 = 2l * v487;
        int v561;
        v561 = v560 + v479;
        v8[v561] = v559;
        v487 += 1l ;
    }
    v17.sync() ;
    int v562;
    v562 = threadIdx.x;
    bool v563;
    v563 = 0l <= v562;
    bool v564;
    v564 = v563 == false;
    if (v564){
        assert("The index needs to be zero or positive." && v563);
    } else {
    }
    int v566;
    v566 = v562 % 16l;
    int v567;
    v567 = v562 / 16l;
    bool v568;
    v568 = v567 < 2l;
    bool v569;
    v569 = v568 == false;
    if (v569){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v568);
    } else {
    }
    assert("Tensor range check" && 0 <= v567 && v567 < 2l);
    assert("Tensor range check" && 0 <= v566 && v566 < 16l);
    int v571;
    v571 = 4l * v566;
    int v572;
    v572 = 64l * v567;
    int v573;
    v573 = v572 + v571;
    assert("Tensor range check" && 0 <= v567 && v567 < 2l);
    assert("Tensor range check" && 0 <= v566 && v566 < 16l);
    int v574;
    v574 = blockIdx.x;
    int v575;
    v575 = v574;
    while (while_method_2(v575)){
        bool v577;
        v577 = 0l <= v575;
        bool v578;
        v578 = v577 == false;
        if (v578){
            assert("The index needs to be zero or positive." && v577);
        } else {
        }
        bool v580;
        v580 = v575 < 64l;
        bool v581;
        v581 = v580 == false;
        if (v581){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v580);
        } else {
        }
        assert("Tensor range check" && 0 <= v575 && v575 < 64l);
        int v583;
        v583 = 128l * v575;
        int v584;
        v584 = v583 + v573;
        float v585[4l];
        int v586[4l];
        int v587;
        v587 = 0l;
        while (while_method_3(v587)){
            assert("Tensor range check" && 0 <= v587 && v587 < 1l);
            int v589;
            v589 = 4l * v587;
            assert("Tensor range check" && 0 <= v587 && v587 < 1l);
            int v590;
            v590 = 64l * v587;
            int v591;
            v591 = v590 + v584;
            int4* v592;
            v592 = reinterpret_cast<int4*>(v1 + v591);
            int4* v593;
            v593 = reinterpret_cast<int4*>(v585 + v589);
            assert("Pointer alignment check" && (unsigned long long)(v592) % 4l == 0 && (unsigned long long)(v593) % 4l == 0);
            *v593 = *v592;
            v587 += 1l ;
        }
        int v594;
        v594 = 0l;
        while (while_method_3(v594)){
            int v596;
            v596 = 0l;
            while (while_method_1(v596)){
                bool v598;
                v598 = 0l <= v596;
                bool v600;
                if (v598){
                    bool v599;
                    v599 = v596 < 4l;
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
                v603 = 0l <= v566;
                bool v605;
                if (v603){
                    bool v604;
                    v604 = v566 < 16l;
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
                v608 = v566 * 4l;
                int v609;
                v609 = v596 + v608;
                bool v610;
                v610 = 0l <= v594;
                bool v612;
                if (v610){
                    bool v611;
                    v611 = v594 < 1l;
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
                v615 = v594 * 64l;
                int v616;
                v616 = v609 + v615;
                assert("Tensor range check" && 0 <= v594 && v594 < 1l);
                assert("Tensor range check" && 0 <= v596 && v596 < 4l);
                int v617;
                v617 = 4l * v594;
                int v618;
                v618 = v617 + v596;
                v586[v618] = v616;
                v596 += 1l ;
            }
            v594 += 1l ;
        }
        bool v619;
        v619 = 0l <= v567;
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
        v626 = v575 * 2l;
        int v627;
        v627 = v626 + v567;
        float v628;
        v628 = 0.0f;
        int v629;
        v629 = 0l;
        while (while_method_3(v629)){
            int v631;
            v631 = 0l;
            while (while_method_1(v631)){
                assert("Tensor range check" && 0 <= v629 && v629 < 1l);
                assert("Tensor range check" && 0 <= v631 && v631 < 4l);
                int v633;
                v633 = 4l * v629;
                int v634;
                v634 = v633 + v631;
                float v635;
                v635 = v585[v634];
                float v636;
                v636 = v628 + v635;
                v628 = v636;
                v631 += 1l ;
            }
            v629 += 1l ;
        }
        auto v637 = cooperative_groups::coalesced_threads();
        int v638;
        v638 = threadIdx.x;
        int v639;
        v639 = v638 / 16l;
        auto v640 = cooperative_groups::labeled_partition(v637,v639);
        Closure0 v641{};
        float v642;
        v642 = cooperative_groups::reduce(v640, v628, v641);
        float v643;
        v643 = v642 / 64.0f;
        float v644[4l];
        int v645;
        v645 = 0l;
        while (while_method_3(v645)){
            int v647;
            v647 = 0l;
            while (while_method_1(v647)){
                assert("Tensor range check" && 0 <= v645 && v645 < 1l);
                assert("Tensor range check" && 0 <= v647 && v647 < 4l);
                int v649;
                v649 = 4l * v645;
                int v650;
                v650 = v649 + v647;
                float v651;
                v651 = v585[v650];
                float v652;
                v652 = v651 - v643;
                float v653;
                v653 = exp(v652);
                assert("Tensor range check" && 0 <= v645 && v645 < 1l);
                assert("Tensor range check" && 0 <= v647 && v647 < 4l);
                v644[v650] = v653;
                v647 += 1l ;
            }
            v645 += 1l ;
        }
        float v654;
        v654 = 0.0f;
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
                v661 = v644[v660];
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
        int v665;
        v665 = v664 / 16l;
        auto v666 = cooperative_groups::labeled_partition(v663,v665);
        float v667;
        v667 = cooperative_groups::reduce(v666, v654, v641);
        float v668[4l];
        int v669;
        v669 = 0l;
        while (while_method_3(v669)){
            int v671;
            v671 = 0l;
            while (while_method_1(v671)){
                assert("Tensor range check" && 0 <= v669 && v669 < 1l);
                assert("Tensor range check" && 0 <= v671 && v671 < 4l);
                int v673;
                v673 = 4l * v669;
                int v674;
                v674 = v673 + v671;
                float v675;
                v675 = v644[v674];
                float v676;
                v676 = v675 / v667;
                assert("Tensor range check" && 0 <= v669 && v669 < 1l);
                assert("Tensor range check" && 0 <= v671 && v671 < 4l);
                v668[v674] = v676;
                v671 += 1l ;
            }
            v669 += 1l ;
        }
        float v677[4l];
        float v678;
        v678 = 0.0f;
        int v679;
        v679 = 0l;
        while (while_method_3(v679)){
            assert("Tensor range check" && 0 <= v679 && v679 < 1l);
            int v681;
            v681 = 4l * v679;
            assert("Tensor range check" && 0 <= v679 && v679 < 1l);
            int v682; float v683;
            Tuple0 tmp64 = Tuple0{0l, 0.0f};
            v682 = tmp64.v0; v683 = tmp64.v1;
            while (while_method_1(v682)){
                assert("Tensor range check" && 0 <= v682 && v682 < 4l);
                int v685;
                v685 = v682 + v681;
                float v686;
                v686 = v668[v685];
                float v687;
                v687 = v683 + v686;
                v683 = v687;
                v682 += 1l ;
            }
            auto v688 = cooperative_groups::coalesced_threads();
            int v689;
            v689 = threadIdx.x;
            int v690;
            v690 = v689 / 16l;
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
            Tuple0 tmp65 = Tuple0{0l, v698};
            v699 = tmp65.v0; v700 = tmp65.v1;
            while (while_method_1(v699)){
                assert("Tensor range check" && 0 <= v699 && v699 < 4l);
                int v702;
                v702 = v699 + v681;
                float v703;
                v703 = v668[v702];
                float v704;
                v704 = v700 + v703;
                assert("Tensor range check" && 0 <= v699 && v699 < 4l);
                v677[v702] = v704;
                v700 = v704;
                v699 += 1l ;
            }
            float v705;
            v705 = v678 + v697;
            v678 = v705;
            v679 += 1l ;
        }
        assert("Tensor range check" && 0 <= v575 && v575 < 64l);
        int v706;
        v706 = 0l;
        while (while_method_3(v706)){
            assert("Tensor range check" && 0 <= v706 && v706 < 1l);
            int v708;
            v708 = 64l * v706;
            int v709;
            v709 = v708 + v584;
            assert("Tensor range check" && 0 <= v706 && v706 < 1l);
            int v710;
            v710 = 4l * v706;
            int4* v711;
            v711 = reinterpret_cast<int4*>(v668 + v710);
            int4* v712;
            v712 = reinterpret_cast<int4*>(v5 + v709);
            assert("Pointer alignment check" && (unsigned long long)(v711) % 4l == 0 && (unsigned long long)(v712) % 4l == 0);
            *v712 = *v711;
            int4* v713;
            v713 = reinterpret_cast<int4*>(v677 + v710);
            int4* v714;
            v714 = reinterpret_cast<int4*>(v6 + v709);
            assert("Pointer alignment check" && (unsigned long long)(v713) % 4l == 0 && (unsigned long long)(v714) % 4l == 0);
            *v714 = *v713;
            v706 += 1l ;
        }
        v575 += 1l ;
    }
    v17.sync() ;
    int v715;
    v715 = threadIdx.x;
    bool v716;
    v716 = 0l <= v715;
    bool v717;
    v717 = v716 == false;
    if (v717){
        assert("The index needs to be zero or positive." && v716);
    } else {
    }
    int v719;
    v719 = v715 % 16l;
    int v720;
    v720 = v715 / 16l;
    bool v721;
    v721 = v720 < 2l;
    bool v722;
    v722 = v721 == false;
    if (v722){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v721);
    } else {
    }
    assert("Tensor range check" && 0 <= v720 && v720 < 2l);
    assert("Tensor range check" && 0 <= v719 && v719 < 16l);
    int v724;
    v724 = 4l * v719;
    int v725;
    v725 = 64l * v720;
    int v726;
    v726 = v725 + v724;
    assert("Tensor range check" && 0 <= v720 && v720 < 2l);
    assert("Tensor range check" && 0 <= v719 && v719 < 16l);
    int v727;
    v727 = blockIdx.x;
    int v728;
    v728 = v727;
    while (while_method_2(v728)){
        bool v730;
        v730 = 0l <= v728;
        bool v731;
        v731 = v730 == false;
        if (v731){
            assert("The index needs to be zero or positive." && v730);
        } else {
        }
        bool v733;
        v733 = v728 < 64l;
        bool v734;
        v734 = v733 == false;
        if (v734){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v733);
        } else {
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 64l);
        int v736;
        v736 = 128l * v728;
        int v737;
        v737 = v736 + v726;
        int v738[4l];
        int v739[4l];
        int v740;
        v740 = 0l;
        while (while_method_3(v740)){
            assert("Tensor range check" && 0 <= v740 && v740 < 1l);
            int v742;
            v742 = 4l * v740;
            assert("Tensor range check" && 0 <= v740 && v740 < 1l);
            int v743;
            v743 = 64l * v740;
            int v744;
            v744 = v743 + v737;
            int4* v745;
            v745 = reinterpret_cast<int4*>(v0 + v744);
            int4* v746;
            v746 = reinterpret_cast<int4*>(v738 + v742);
            assert("Pointer alignment check" && (unsigned long long)(v745) % 4l == 0 && (unsigned long long)(v746) % 4l == 0);
            *v746 = *v745;
            v740 += 1l ;
        }
        int v747;
        v747 = 0l;
        while (while_method_3(v747)){
            int v749;
            v749 = 0l;
            while (while_method_1(v749)){
                bool v751;
                v751 = 0l <= v749;
                bool v753;
                if (v751){
                    bool v752;
                    v752 = v749 < 4l;
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
                v756 = 0l <= v719;
                bool v758;
                if (v756){
                    bool v757;
                    v757 = v719 < 16l;
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
                v761 = v719 * 4l;
                int v762;
                v762 = v749 + v761;
                bool v763;
                v763 = 0l <= v747;
                bool v765;
                if (v763){
                    bool v764;
                    v764 = v747 < 1l;
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
                v768 = v747 * 64l;
                int v769;
                v769 = v762 + v768;
                assert("Tensor range check" && 0 <= v747 && v747 < 1l);
                assert("Tensor range check" && 0 <= v749 && v749 < 4l);
                int v770;
                v770 = 4l * v747;
                int v771;
                v771 = v770 + v749;
                v739[v771] = v769;
                v749 += 1l ;
            }
            v747 += 1l ;
        }
        bool v772;
        v772 = 0l <= v720;
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
        v779 = v728 * 2l;
        int v780;
        v780 = v779 + v720;
        int v781[4l];
        int v782;
        v782 = 0l;
        int v783;
        v783 = 0l;
        while (while_method_3(v783)){
            assert("Tensor range check" && 0 <= v783 && v783 < 1l);
            int v785;
            v785 = 4l * v783;
            assert("Tensor range check" && 0 <= v783 && v783 < 1l);
            int v786; int v787;
            Tuple2 tmp66 = Tuple2{0l, 0l};
            v786 = tmp66.v0; v787 = tmp66.v1;
            while (while_method_1(v786)){
                assert("Tensor range check" && 0 <= v786 && v786 < 4l);
                int v789;
                v789 = v786 + v785;
                int v790;
                v790 = v738[v789];
                int v791;
                v791 = v787 + v790;
                v787 = v791;
                v786 += 1l ;
            }
            auto v792 = cooperative_groups::coalesced_threads();
            int v793;
            v793 = threadIdx.x;
            int v794;
            v794 = v793 / 16l;
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
                v800 = 0l;
            } else {
                v800 = v798;
            }
            int v801;
            v801 = v795.shfl(v797,v795.num_threads()-1);
            int v802;
            v802 = v782 + v800;
            int v803; int v804;
            Tuple2 tmp67 = Tuple2{0l, v802};
            v803 = tmp67.v0; v804 = tmp67.v1;
            while (while_method_1(v803)){
                assert("Tensor range check" && 0 <= v803 && v803 < 4l);
                int v806;
                v806 = v803 + v785;
                int v807;
                v807 = v738[v806];
                assert("Tensor range check" && 0 <= v803 && v803 < 4l);
                v781[v806] = v804;
                int v808;
                v808 = v804 + v807;
                v804 = v808;
                v803 += 1l ;
            }
            int v809;
            v809 = v782 + v801;
            v782 = v809;
            v783 += 1l ;
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 64l);
        int v810;
        v810 = 0l;
        while (while_method_3(v810)){
            assert("Tensor range check" && 0 <= v810 && v810 < 1l);
            int v812;
            v812 = 64l * v810;
            int v813;
            v813 = v812 + v737;
            assert("Tensor range check" && 0 <= v810 && v810 < 1l);
            int v814;
            v814 = 4l * v810;
            int4* v815;
            v815 = reinterpret_cast<int4*>(v781 + v814);
            int4* v816;
            v816 = reinterpret_cast<int4*>(v12 + v813);
            assert("Pointer alignment check" && (unsigned long long)(v815) % 4l == 0 && (unsigned long long)(v816) % 4l == 0);
            *v816 = *v815;
            v810 += 1l ;
        }
        v728 += 1l ;
    }
    v17.sync() ;
    int v817;
    v817 = threadIdx.x;
    bool v818;
    v818 = 0l <= v817;
    bool v819;
    v819 = v818 == false;
    if (v819){
        assert("The index needs to be zero or positive." && v818);
    } else {
    }
    int v821;
    v821 = v817 % 16l;
    int v822;
    v822 = v817 / 16l;
    bool v823;
    v823 = v822 < 2l;
    bool v824;
    v824 = v823 == false;
    if (v824){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v823);
    } else {
    }
    assert("Tensor range check" && 0 <= v822 && v822 < 2l);
    assert("Tensor range check" && 0 <= v821 && v821 < 16l);
    int v826;
    v826 = 4l * v821;
    int v827;
    v827 = 64l * v822;
    int v828;
    v828 = v827 + v826;
    assert("Tensor range check" && 0 <= v822 && v822 < 2l);
    assert("Tensor range check" && 0 <= v821 && v821 < 16l);
    int v829;
    v829 = blockIdx.x;
    int v830;
    v830 = v829;
    while (while_method_2(v830)){
        bool v832;
        v832 = 0l <= v830;
        bool v833;
        v833 = v832 == false;
        if (v833){
            assert("The index needs to be zero or positive." && v832);
        } else {
        }
        bool v835;
        v835 = v830 < 64l;
        bool v836;
        v836 = v835 == false;
        if (v836){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v835);
        } else {
        }
        assert("Tensor range check" && 0 <= v830 && v830 < 64l);
        int v838;
        v838 = 128l * v830;
        int v839;
        v839 = v838 + v828;
        float v840[4l];
        int v841[4l];
        int v842;
        v842 = 0l;
        while (while_method_3(v842)){
            assert("Tensor range check" && 0 <= v842 && v842 < 1l);
            int v844;
            v844 = 4l * v842;
            assert("Tensor range check" && 0 <= v842 && v842 < 1l);
            int v845;
            v845 = 64l * v842;
            int v846;
            v846 = v845 + v839;
            int4* v847;
            v847 = reinterpret_cast<int4*>(v1 + v846);
            int4* v848;
            v848 = reinterpret_cast<int4*>(v840 + v844);
            assert("Pointer alignment check" && (unsigned long long)(v847) % 4l == 0 && (unsigned long long)(v848) % 4l == 0);
            *v848 = *v847;
            v842 += 1l ;
        }
        int v849;
        v849 = 0l;
        while (while_method_3(v849)){
            int v851;
            v851 = 0l;
            while (while_method_1(v851)){
                bool v853;
                v853 = 0l <= v851;
                bool v855;
                if (v853){
                    bool v854;
                    v854 = v851 < 4l;
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
                v858 = 0l <= v821;
                bool v860;
                if (v858){
                    bool v859;
                    v859 = v821 < 16l;
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
                v863 = v821 * 4l;
                int v864;
                v864 = v851 + v863;
                bool v865;
                v865 = 0l <= v849;
                bool v867;
                if (v865){
                    bool v866;
                    v866 = v849 < 1l;
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
                v870 = v849 * 64l;
                int v871;
                v871 = v864 + v870;
                assert("Tensor range check" && 0 <= v849 && v849 < 1l);
                assert("Tensor range check" && 0 <= v851 && v851 < 4l);
                int v872;
                v872 = 4l * v849;
                int v873;
                v873 = v872 + v851;
                v841[v873] = v871;
                v851 += 1l ;
            }
            v849 += 1l ;
        }
        bool v874;
        v874 = 0l <= v822;
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
        v881 = v830 * 2l;
        int v882;
        v882 = v881 + v822;
        bool v883[4l];
        int v884;
        v884 = 0l;
        while (while_method_3(v884)){
            int v886;
            v886 = 0l;
            while (while_method_1(v886)){
                assert("Tensor range check" && 0 <= v884 && v884 < 1l);
                assert("Tensor range check" && 0 <= v886 && v886 < 4l);
                int v888;
                v888 = 4l * v884;
                int v889;
                v889 = v888 + v886;
                float v890;
                v890 = v840[v889];
                int v891;
                v891 = v841[v889];
                bool v892;
                v892 = v891 < 4l;
                assert("Tensor range check" && 0 <= v884 && v884 < 1l);
                assert("Tensor range check" && 0 <= v886 && v886 < 4l);
                v883[v889] = v892;
                v886 += 1l ;
            }
            v884 += 1l ;
        }
        int v893[4l];
        int v894;
        v894 = 0l;
        while (while_method_3(v894)){
            int v896;
            v896 = 0l;
            while (while_method_1(v896)){
                assert("Tensor range check" && 0 <= v894 && v894 < 1l);
                assert("Tensor range check" && 0 <= v896 && v896 < 4l);
                int v898;
                v898 = 4l * v894;
                int v899;
                v899 = v898 + v896;
                bool v900;
                v900 = v883[v899];
                int v901;
                if (v900){
                    v901 = 1l;
                } else {
                    v901 = 0l;
                }
                assert("Tensor range check" && 0 <= v894 && v894 < 1l);
                assert("Tensor range check" && 0 <= v896 && v896 < 4l);
                v893[v899] = v901;
                v896 += 1l ;
            }
            v894 += 1l ;
        }
        int v902;
        v902 = 0l;
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
                int v909;
                v909 = v893[v908];
                int v910;
                v910 = v902 + v909;
                v902 = v910;
                v905 += 1l ;
            }
            v903 += 1l ;
        }
        auto v911 = cooperative_groups::coalesced_threads();
        int v912;
        v912 = threadIdx.x;
        int v913;
        v913 = v912 / 16l;
        auto v914 = cooperative_groups::labeled_partition(v911,v913);
        Closure4 v915{};
        int v916;
        v916 = cooperative_groups::reduce(v914, v902, v915);
        float v917[4l];
        int v918;
        v918 = 0l;
        while (while_method_3(v918)){
            int v920;
            v920 = 0l;
            while (while_method_1(v920)){
                assert("Tensor range check" && 0 <= v918 && v918 < 1l);
                assert("Tensor range check" && 0 <= v920 && v920 < 4l);
                int v922;
                v922 = 4l * v918;
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
                assert("Tensor range check" && 0 <= v918 && v918 < 1l);
                assert("Tensor range check" && 0 <= v920 && v920 < 4l);
                v917[v923] = v926;
                v920 += 1l ;
            }
            v918 += 1l ;
        }
        float v927;
        v927 = 0.0f;
        int v928;
        v928 = 0l;
        while (while_method_3(v928)){
            int v930;
            v930 = 0l;
            while (while_method_1(v930)){
                assert("Tensor range check" && 0 <= v928 && v928 < 1l);
                assert("Tensor range check" && 0 <= v930 && v930 < 4l);
                int v932;
                v932 = 4l * v928;
                int v933;
                v933 = v932 + v930;
                float v934;
                v934 = v917[v933];
                float v935;
                v935 = v927 + v934;
                v927 = v935;
                v930 += 1l ;
            }
            v928 += 1l ;
        }
        auto v936 = cooperative_groups::coalesced_threads();
        int v937;
        v937 = threadIdx.x;
        int v938;
        v938 = v937 / 16l;
        auto v939 = cooperative_groups::labeled_partition(v936,v938);
        Closure0 v940{};
        float v941;
        v941 = cooperative_groups::reduce(v939, v927, v940);
        float v942;
        v942 = (float)v916;
        float v943;
        v943 = v941 / v942;
        float v944[4l];
        int v945;
        v945 = 0l;
        while (while_method_3(v945)){
            int v947;
            v947 = 0l;
            while (while_method_1(v947)){
                assert("Tensor range check" && 0 <= v945 && v945 < 1l);
                assert("Tensor range check" && 0 <= v947 && v947 < 4l);
                int v949;
                v949 = 4l * v945;
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
                assert("Tensor range check" && 0 <= v945 && v945 < 1l);
                assert("Tensor range check" && 0 <= v947 && v947 < 4l);
                v944[v950] = v955;
                v947 += 1l ;
            }
            v945 += 1l ;
        }
        float v956;
        v956 = 0.0f;
        int v957;
        v957 = 0l;
        while (while_method_3(v957)){
            int v959;
            v959 = 0l;
            while (while_method_1(v959)){
                assert("Tensor range check" && 0 <= v957 && v957 < 1l);
                assert("Tensor range check" && 0 <= v959 && v959 < 4l);
                int v961;
                v961 = 4l * v957;
                int v962;
                v962 = v961 + v959;
                float v963;
                v963 = v944[v962];
                float v964;
                v964 = v956 + v963;
                v956 = v964;
                v959 += 1l ;
            }
            v957 += 1l ;
        }
        auto v965 = cooperative_groups::coalesced_threads();
        int v966;
        v966 = threadIdx.x;
        int v967;
        v967 = v966 / 16l;
        auto v968 = cooperative_groups::labeled_partition(v965,v967);
        float v969;
        v969 = cooperative_groups::reduce(v968, v956, v940);
        float v970[4l];
        int v971;
        v971 = 0l;
        while (while_method_3(v971)){
            int v973;
            v973 = 0l;
            while (while_method_1(v973)){
                assert("Tensor range check" && 0 <= v971 && v971 < 1l);
                assert("Tensor range check" && 0 <= v973 && v973 < 4l);
                int v975;
                v975 = 4l * v971;
                int v976;
                v976 = v975 + v973;
                float v977;
                v977 = v944[v976];
                float v978;
                v978 = v977 / v969;
                assert("Tensor range check" && 0 <= v971 && v971 < 1l);
                assert("Tensor range check" && 0 <= v973 && v973 < 4l);
                v970[v976] = v978;
                v973 += 1l ;
            }
            v971 += 1l ;
        }
        assert("Tensor range check" && 0 <= v830 && v830 < 64l);
        int v979;
        v979 = 0l;
        while (while_method_3(v979)){
            assert("Tensor range check" && 0 <= v979 && v979 < 1l);
            int v981;
            v981 = 64l * v979;
            int v982;
            v982 = v981 + v839;
            assert("Tensor range check" && 0 <= v979 && v979 < 1l);
            int v983;
            v983 = 4l * v979;
            int4* v984;
            v984 = reinterpret_cast<int4*>(v970 + v983);
            int4* v985;
            v985 = reinterpret_cast<int4*>(v4 + v982);
            assert("Pointer alignment check" && (unsigned long long)(v984) % 4l == 0 && (unsigned long long)(v985) % 4l == 0);
            *v985 = *v984;
            v979 += 1l ;
        }
        v830 += 1l ;
    }
    v17.sync() ;
    int v986;
    v986 = threadIdx.x;
    int v987;
    v987 = blockIdx.x;
    int v988;
    v988 = v987 * 32l;
    int v989;
    v989 = v986 + v988;
    unsigned long long v990;
    v990 = (unsigned long long)v989;
    curandStatePhilox4_32_10_t v991;
    curand_init(12344321ull,v990,0ull,&v991);
    int v992;
    v992 = threadIdx.x;
    bool v993;
    v993 = 0l <= v992;
    bool v994;
    v994 = v993 == false;
    if (v994){
        assert("The index needs to be zero or positive." && v993);
    } else {
    }
    int v996;
    v996 = v992 % 16l;
    int v997;
    v997 = v992 / 16l;
    bool v998;
    v998 = v997 < 2l;
    bool v999;
    v999 = v998 == false;
    if (v999){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v998);
    } else {
    }
    assert("Tensor range check" && 0 <= v997 && v997 < 2l);
    assert("Tensor range check" && 0 <= v996 && v996 < 16l);
    int v1001;
    v1001 = 4l * v996;
    int v1002;
    v1002 = 64l * v997;
    int v1003;
    v1003 = v1002 + v1001;
    assert("Tensor range check" && 0 <= v997 && v997 < 2l);
    assert("Tensor range check" && 0 <= v996 && v996 < 16l);
    assert("Tensor range check" && 0 <= v997 && v997 < 2l);
    int v1004;
    v1004 = blockIdx.x;
    int v1005;
    v1005 = v1004;
    while (while_method_2(v1005)){
        bool v1007;
        v1007 = 0l <= v1005;
        bool v1008;
        v1008 = v1007 == false;
        if (v1008){
            assert("The index needs to be zero or positive." && v1007);
        } else {
        }
        bool v1010;
        v1010 = v1005 < 64l;
        bool v1011;
        v1011 = v1010 == false;
        if (v1011){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1010);
        } else {
        }
        assert("Tensor range check" && 0 <= v1005 && v1005 < 64l);
        int v1013;
        v1013 = 128l * v1005;
        int v1014;
        v1014 = v1013 + v1003;
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
                v1033 = 0l <= v996;
                bool v1035;
                if (v1033){
                    bool v1034;
                    v1034 = v996 < 16l;
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
                v1038 = v996 * 4l;
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
        v1049 = 0l <= v997;
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
        v1056 = v1005 * 2l;
        int v1057;
        v1057 = v1056 + v997;
        float v1058;
        v1058 = 0.0f;
        int v1059;
        v1059 = 0l;
        while (while_method_3(v1059)){
            int v1061;
            v1061 = 0l;
            while (while_method_1(v1061)){
                assert("Tensor range check" && 0 <= v1059 && v1059 < 1l);
                assert("Tensor range check" && 0 <= v1061 && v1061 < 4l);
                int v1063;
                v1063 = 4l * v1059;
                int v1064;
                v1064 = v1063 + v1061;
                float v1065;
                v1065 = v1015[v1064];
                float v1066;
                v1066 = v1058 + v1065;
                v1058 = v1066;
                v1061 += 1l ;
            }
            v1059 += 1l ;
        }
        auto v1067 = cooperative_groups::coalesced_threads();
        int v1068;
        v1068 = threadIdx.x;
        int v1069;
        v1069 = v1068 / 16l;
        auto v1070 = cooperative_groups::labeled_partition(v1067,v1069);
        Closure0 v1071{};
        float v1072;
        v1072 = cooperative_groups::reduce(v1070, v1058, v1071);
        float v1073;
        v1073 = v1072 / 64.0f;
        float v1074[4l];
        int v1075;
        v1075 = 0l;
        while (while_method_3(v1075)){
            int v1077;
            v1077 = 0l;
            while (while_method_1(v1077)){
                assert("Tensor range check" && 0 <= v1075 && v1075 < 1l);
                assert("Tensor range check" && 0 <= v1077 && v1077 < 4l);
                int v1079;
                v1079 = 4l * v1075;
                int v1080;
                v1080 = v1079 + v1077;
                float v1081;
                v1081 = v1015[v1080];
                float v1082;
                v1082 = v1081 - v1073;
                float v1083;
                v1083 = exp(v1082);
                assert("Tensor range check" && 0 <= v1075 && v1075 < 1l);
                assert("Tensor range check" && 0 <= v1077 && v1077 < 4l);
                v1074[v1080] = v1083;
                v1077 += 1l ;
            }
            v1075 += 1l ;
        }
        float v1084;
        v1084 = 0.0f;
        int v1085;
        v1085 = 0l;
        while (while_method_3(v1085)){
            int v1087;
            v1087 = 0l;
            while (while_method_1(v1087)){
                assert("Tensor range check" && 0 <= v1085 && v1085 < 1l);
                assert("Tensor range check" && 0 <= v1087 && v1087 < 4l);
                int v1089;
                v1089 = 4l * v1085;
                int v1090;
                v1090 = v1089 + v1087;
                float v1091;
                v1091 = v1074[v1090];
                float v1092;
                v1092 = v1084 + v1091;
                v1084 = v1092;
                v1087 += 1l ;
            }
            v1085 += 1l ;
        }
        auto v1093 = cooperative_groups::coalesced_threads();
        int v1094;
        v1094 = threadIdx.x;
        int v1095;
        v1095 = v1094 / 16l;
        auto v1096 = cooperative_groups::labeled_partition(v1093,v1095);
        float v1097;
        v1097 = cooperative_groups::reduce(v1096, v1084, v1071);
        float v1098[4l];
        int v1099;
        v1099 = 0l;
        while (while_method_3(v1099)){
            int v1101;
            v1101 = 0l;
            while (while_method_1(v1101)){
                assert("Tensor range check" && 0 <= v1099 && v1099 < 1l);
                assert("Tensor range check" && 0 <= v1101 && v1101 < 4l);
                int v1103;
                v1103 = 4l * v1099;
                int v1104;
                v1104 = v1103 + v1101;
                float v1105;
                v1105 = v1074[v1104];
                float v1106;
                v1106 = v1105 / v1097;
                assert("Tensor range check" && 0 <= v1099 && v1099 < 1l);
                assert("Tensor range check" && 0 <= v1101 && v1101 < 4l);
                v1098[v1104] = v1106;
                v1101 += 1l ;
            }
            v1099 += 1l ;
        }
        float v1107[4l];
        float v1108;
        v1108 = 0.0f;
        int v1109;
        v1109 = 0l;
        while (while_method_3(v1109)){
            assert("Tensor range check" && 0 <= v1109 && v1109 < 1l);
            int v1111;
            v1111 = 4l * v1109;
            assert("Tensor range check" && 0 <= v1109 && v1109 < 1l);
            int v1112; float v1113;
            Tuple0 tmp68 = Tuple0{0l, 0.0f};
            v1112 = tmp68.v0; v1113 = tmp68.v1;
            while (while_method_1(v1112)){
                assert("Tensor range check" && 0 <= v1112 && v1112 < 4l);
                int v1115;
                v1115 = v1112 + v1111;
                float v1116;
                v1116 = v1098[v1115];
                float v1117;
                v1117 = v1113 + v1116;
                v1113 = v1117;
                v1112 += 1l ;
            }
            auto v1118 = cooperative_groups::coalesced_threads();
            int v1119;
            v1119 = threadIdx.x;
            int v1120;
            v1120 = v1119 / 16l;
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
            Tuple0 tmp69 = Tuple0{0l, v1128};
            v1129 = tmp69.v0; v1130 = tmp69.v1;
            while (while_method_1(v1129)){
                assert("Tensor range check" && 0 <= v1129 && v1129 < 4l);
                int v1132;
                v1132 = v1129 + v1111;
                float v1133;
                v1133 = v1098[v1132];
                float v1134;
                v1134 = v1130 + v1133;
                assert("Tensor range check" && 0 <= v1129 && v1129 < 4l);
                v1107[v1132] = v1134;
                v1130 = v1134;
                v1129 += 1l ;
            }
            float v1135;
            v1135 = v1108 + v1127;
            v1108 = v1135;
            v1109 += 1l ;
        }
        float v1136[4l];
        bool v1137[4l];
        int v1138;
        v1138 = 0l;
        while (while_method_3(v1138)){
            int v1140;
            v1140 = 0l;
            while (while_method_1(v1140)){
                assert("Tensor range check" && 0 <= v1138 && v1138 < 1l);
                assert("Tensor range check" && 0 <= v1140 && v1140 < 4l);
                int v1142;
                v1142 = 4l * v1138;
                int v1143;
                v1143 = v1142 + v1140;
                float v1144;
                v1144 = v1107[v1143];
                float v1145;
                v1145 = v1098[v1143];
                bool v1146;
                v1146 = v1145 > 0.0f;
                assert("Tensor range check" && 0 <= v1138 && v1138 < 1l);
                assert("Tensor range check" && 0 <= v1140 && v1140 < 4l);
                v1136[v1143] = v1144;
                v1137[v1143] = v1146;
                v1140 += 1l ;
            }
            v1138 += 1l ;
        }
        float v1147; bool v1148;
        Tuple3 tmp70 = Tuple3{-1.0f / 0.0f, false};
        v1147 = tmp70.v0; v1148 = tmp70.v1;
        int v1149;
        v1149 = 0l;
        while (while_method_3(v1149)){
            int v1151;
            v1151 = 0l;
            while (while_method_1(v1151)){
                assert("Tensor range check" && 0 <= v1149 && v1149 < 1l);
                assert("Tensor range check" && 0 <= v1151 && v1151 < 4l);
                int v1153;
                v1153 = 4l * v1149;
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
                v1151 += 1l ;
            }
            v1149 += 1l ;
        }
        auto v1165 = cooperative_groups::coalesced_threads();
        int v1166;
        v1166 = threadIdx.x;
        int v1167;
        v1167 = v1166 / 16l;
        auto v1168 = cooperative_groups::labeled_partition(v1165,v1167);
        Closure5 v1169{};
        float v1170; bool v1171;
        Tuple3 tmp71 = cooperative_groups::reduce(v1168, Tuple3{v1147, v1148}, v1169);
        v1170 = tmp71.v0; v1171 = tmp71.v1;
        bool v1172;
        v1172 = v1171 == false;
        if (v1172){
            assert("The local reduce must be true." && v1171);
        } else {
        }
        float v1174[4l];
        int v1175[4l];
        int v1176;
        v1176 = 0l;
        while (while_method_3(v1176)){
            int v1178;
            v1178 = 0l;
            while (while_method_1(v1178)){
                assert("Tensor range check" && 0 <= v1176 && v1176 < 1l);
                assert("Tensor range check" && 0 <= v1178 && v1178 < 4l);
                int v1180;
                v1180 = 4l * v1176;
                int v1181;
                v1181 = v1180 + v1178;
                int v1182;
                v1182 = v1016[v1181];
                float v1183;
                v1183 = curand_uniform(&v991);
                assert("Tensor range check" && 0 <= v1176 && v1176 < 1l);
                assert("Tensor range check" && 0 <= v1178 && v1178 < 4l);
                v1174[v1181] = v1183;
                v1175[v1181] = v1182;
                v1178 += 1l ;
            }
            v1176 += 1l ;
        }
        float v1184; int v1185;
        Tuple1 tmp72 = Tuple1{0.0f, 2147483647l};
        v1184 = tmp72.v0; v1185 = tmp72.v1;
        int v1186;
        v1186 = 0l;
        while (while_method_3(v1186)){
            int v1188;
            v1188 = 0l;
            while (while_method_1(v1188)){
                assert("Tensor range check" && 0 <= v1186 && v1186 < 1l);
                assert("Tensor range check" && 0 <= v1188 && v1188 < 4l);
                int v1190;
                v1190 = 4l * v1186;
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
                v1188 += 1l ;
            }
            v1186 += 1l ;
        }
        auto v1197 = cooperative_groups::coalesced_threads();
        int v1198;
        v1198 = threadIdx.x;
        int v1199;
        v1199 = v1198 / 16l;
        auto v1200 = cooperative_groups::labeled_partition(v1197,v1199);
        Closure6 v1201{};
        float v1202; int v1203;
        Tuple1 tmp73 = cooperative_groups::reduce(v1200, Tuple1{v1184, v1185}, v1201);
        v1202 = tmp73.v0; v1203 = tmp73.v1;
        float v1204;
        v1204 = v1170 * v1202;
        int v1205[4l];
        bool v1206[4l];
        int v1207;
        v1207 = 0l;
        while (while_method_3(v1207)){
            int v1209;
            v1209 = 0l;
            while (while_method_1(v1209)){
                assert("Tensor range check" && 0 <= v1207 && v1207 < 1l);
                assert("Tensor range check" && 0 <= v1209 && v1209 < 4l);
                int v1211;
                v1211 = 4l * v1207;
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
                    v1218 = 2147483647l; v1219 = false;
                }
                assert("Tensor range check" && 0 <= v1207 && v1207 < 1l);
                assert("Tensor range check" && 0 <= v1209 && v1209 < 4l);
                v1205[v1212] = v1218;
                v1206[v1212] = v1219;
                v1209 += 1l ;
            }
            v1207 += 1l ;
        }
        int v1220; bool v1221;
        Tuple4 tmp74 = Tuple4{2147483647l, false};
        v1220 = tmp74.v0; v1221 = tmp74.v1;
        int v1222;
        v1222 = 0l;
        while (while_method_3(v1222)){
            int v1224;
            v1224 = 0l;
            while (while_method_1(v1224)){
                assert("Tensor range check" && 0 <= v1222 && v1222 < 1l);
                assert("Tensor range check" && 0 <= v1224 && v1224 < 4l);
                int v1226;
                v1226 = 4l * v1222;
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
                v1224 += 1l ;
            }
            v1222 += 1l ;
        }
        auto v1238 = cooperative_groups::coalesced_threads();
        int v1239;
        v1239 = threadIdx.x;
        int v1240;
        v1240 = v1239 / 16l;
        auto v1241 = cooperative_groups::labeled_partition(v1238,v1240);
        Closure7 v1242{};
        int v1243; bool v1244;
        Tuple4 tmp75 = cooperative_groups::reduce(v1241, Tuple4{v1220, v1221}, v1242);
        v1243 = tmp75.v0; v1244 = tmp75.v1;
        bool v1245;
        v1245 = v1244 == false;
        if (v1245){
            assert("The local reduce must be true." && v1244);
        } else {
        }
        assert("Tensor range check" && 0 <= v1005 && v1005 < 64l);
        int v1247;
        v1247 = 0l;
        while (while_method_3(v1247)){
            assert("Tensor range check" && 0 <= v1247 && v1247 < 1l);
            int v1249;
            v1249 = 64l * v1247;
            int v1250;
            v1250 = v1249 + v1014;
            assert("Tensor range check" && 0 <= v1247 && v1247 < 1l);
            int v1251;
            v1251 = 4l * v1247;
            int4* v1252;
            v1252 = reinterpret_cast<int4*>(v1098 + v1251);
            int4* v1253;
            v1253 = reinterpret_cast<int4*>(v13 + v1250);
            assert("Pointer alignment check" && (unsigned long long)(v1252) % 4l == 0 && (unsigned long long)(v1253) % 4l == 0);
            *v1253 = *v1252;
            v1247 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1005 && v1005 < 64l);
        int v1254;
        v1254 = 2l * v1005;
        int v1255;
        v1255 = v1254 + v997;
        v14[v1255] = v1243;
        v1005 += 1l ;
    }
    v17.sync() ;
    int v1256;
    v1256 = threadIdx.x;
    int v1257;
    v1257 = blockIdx.x;
    int v1258;
    v1258 = v1257 * 32l;
    int v1259;
    v1259 = v1256 + v1258;
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
    v1266 = v1262 % 16l;
    int v1267;
    v1267 = v1262 / 16l;
    bool v1268;
    v1268 = v1267 < 2l;
    bool v1269;
    v1269 = v1268 == false;
    if (v1269){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1268);
    } else {
    }
    assert("Tensor range check" && 0 <= v1267 && v1267 < 2l);
    assert("Tensor range check" && 0 <= v1266 && v1266 < 16l);
    int v1271;
    v1271 = 4l * v1266;
    int v1272;
    v1272 = 64l * v1267;
    int v1273;
    v1273 = v1272 + v1271;
    assert("Tensor range check" && 0 <= v1267 && v1267 < 2l);
    assert("Tensor range check" && 0 <= v1266 && v1266 < 16l);
    assert("Tensor range check" && 0 <= v1267 && v1267 < 2l);
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
            v1290 = 64l * v1287;
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
                    v1304 = v1266 < 16l;
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
                v1315 = v1294 * 64l;
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
        v1326 = v1275 * 2l;
        int v1327;
        v1327 = v1326 + v1267;
        bool v1328[4l];
        int v1329;
        v1329 = 0l;
        while (while_method_3(v1329)){
            int v1331;
            v1331 = 0l;
            while (while_method_1(v1331)){
                assert("Tensor range check" && 0 <= v1329 && v1329 < 1l);
                assert("Tensor range check" && 0 <= v1331 && v1331 < 4l);
                int v1333;
                v1333 = 4l * v1329;
                int v1334;
                v1334 = v1333 + v1331;
                float v1335;
                v1335 = v1285[v1334];
                int v1336;
                v1336 = v1286[v1334];
                bool v1337;
                v1337 = v1336 < 11l;
                assert("Tensor range check" && 0 <= v1329 && v1329 < 1l);
                assert("Tensor range check" && 0 <= v1331 && v1331 < 4l);
                v1328[v1334] = v1337;
                v1331 += 1l ;
            }
            v1329 += 1l ;
        }
        int v1338[4l];
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
                bool v1345;
                v1345 = v1328[v1344];
                int v1346;
                if (v1345){
                    v1346 = 1l;
                } else {
                    v1346 = 0l;
                }
                assert("Tensor range check" && 0 <= v1339 && v1339 < 1l);
                assert("Tensor range check" && 0 <= v1341 && v1341 < 4l);
                v1338[v1344] = v1346;
                v1341 += 1l ;
            }
            v1339 += 1l ;
        }
        int v1347;
        v1347 = 0l;
        int v1348;
        v1348 = 0l;
        while (while_method_3(v1348)){
            int v1350;
            v1350 = 0l;
            while (while_method_1(v1350)){
                assert("Tensor range check" && 0 <= v1348 && v1348 < 1l);
                assert("Tensor range check" && 0 <= v1350 && v1350 < 4l);
                int v1352;
                v1352 = 4l * v1348;
                int v1353;
                v1353 = v1352 + v1350;
                int v1354;
                v1354 = v1338[v1353];
                int v1355;
                v1355 = v1347 + v1354;
                v1347 = v1355;
                v1350 += 1l ;
            }
            v1348 += 1l ;
        }
        auto v1356 = cooperative_groups::coalesced_threads();
        int v1357;
        v1357 = threadIdx.x;
        int v1358;
        v1358 = v1357 / 16l;
        auto v1359 = cooperative_groups::labeled_partition(v1356,v1358);
        Closure4 v1360{};
        int v1361;
        v1361 = cooperative_groups::reduce(v1359, v1347, v1360);
        float v1362[4l];
        int v1363;
        v1363 = 0l;
        while (while_method_3(v1363)){
            int v1365;
            v1365 = 0l;
            while (while_method_1(v1365)){
                assert("Tensor range check" && 0 <= v1363 && v1363 < 1l);
                assert("Tensor range check" && 0 <= v1365 && v1365 < 4l);
                int v1367;
                v1367 = 4l * v1363;
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
                assert("Tensor range check" && 0 <= v1363 && v1363 < 1l);
                assert("Tensor range check" && 0 <= v1365 && v1365 < 4l);
                v1362[v1368] = v1371;
                v1365 += 1l ;
            }
            v1363 += 1l ;
        }
        float v1372;
        v1372 = 0.0f;
        int v1373;
        v1373 = 0l;
        while (while_method_3(v1373)){
            int v1375;
            v1375 = 0l;
            while (while_method_1(v1375)){
                assert("Tensor range check" && 0 <= v1373 && v1373 < 1l);
                assert("Tensor range check" && 0 <= v1375 && v1375 < 4l);
                int v1377;
                v1377 = 4l * v1373;
                int v1378;
                v1378 = v1377 + v1375;
                float v1379;
                v1379 = v1362[v1378];
                float v1380;
                v1380 = v1372 + v1379;
                v1372 = v1380;
                v1375 += 1l ;
            }
            v1373 += 1l ;
        }
        auto v1381 = cooperative_groups::coalesced_threads();
        int v1382;
        v1382 = threadIdx.x;
        int v1383;
        v1383 = v1382 / 16l;
        auto v1384 = cooperative_groups::labeled_partition(v1381,v1383);
        Closure0 v1385{};
        float v1386;
        v1386 = cooperative_groups::reduce(v1384, v1372, v1385);
        float v1387;
        v1387 = (float)v1361;
        float v1388;
        v1388 = v1386 / v1387;
        float v1389[4l];
        int v1390;
        v1390 = 0l;
        while (while_method_3(v1390)){
            int v1392;
            v1392 = 0l;
            while (while_method_1(v1392)){
                assert("Tensor range check" && 0 <= v1390 && v1390 < 1l);
                assert("Tensor range check" && 0 <= v1392 && v1392 < 4l);
                int v1394;
                v1394 = 4l * v1390;
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
                assert("Tensor range check" && 0 <= v1390 && v1390 < 1l);
                assert("Tensor range check" && 0 <= v1392 && v1392 < 4l);
                v1389[v1395] = v1400;
                v1392 += 1l ;
            }
            v1390 += 1l ;
        }
        float v1401;
        v1401 = 0.0f;
        int v1402;
        v1402 = 0l;
        while (while_method_3(v1402)){
            int v1404;
            v1404 = 0l;
            while (while_method_1(v1404)){
                assert("Tensor range check" && 0 <= v1402 && v1402 < 1l);
                assert("Tensor range check" && 0 <= v1404 && v1404 < 4l);
                int v1406;
                v1406 = 4l * v1402;
                int v1407;
                v1407 = v1406 + v1404;
                float v1408;
                v1408 = v1389[v1407];
                float v1409;
                v1409 = v1401 + v1408;
                v1401 = v1409;
                v1404 += 1l ;
            }
            v1402 += 1l ;
        }
        auto v1410 = cooperative_groups::coalesced_threads();
        int v1411;
        v1411 = threadIdx.x;
        int v1412;
        v1412 = v1411 / 16l;
        auto v1413 = cooperative_groups::labeled_partition(v1410,v1412);
        float v1414;
        v1414 = cooperative_groups::reduce(v1413, v1401, v1385);
        float v1415[4l];
        int v1416;
        v1416 = 0l;
        while (while_method_3(v1416)){
            int v1418;
            v1418 = 0l;
            while (while_method_1(v1418)){
                assert("Tensor range check" && 0 <= v1416 && v1416 < 1l);
                assert("Tensor range check" && 0 <= v1418 && v1418 < 4l);
                int v1420;
                v1420 = 4l * v1416;
                int v1421;
                v1421 = v1420 + v1418;
                float v1422;
                v1422 = v1389[v1421];
                float v1423;
                v1423 = v1422 / v1414;
                assert("Tensor range check" && 0 <= v1416 && v1416 < 1l);
                assert("Tensor range check" && 0 <= v1418 && v1418 < 4l);
                v1415[v1421] = v1423;
                v1418 += 1l ;
            }
            v1416 += 1l ;
        }
        float v1424[4l];
        float v1425;
        v1425 = 0.0f;
        int v1426;
        v1426 = 0l;
        while (while_method_3(v1426)){
            assert("Tensor range check" && 0 <= v1426 && v1426 < 1l);
            int v1428;
            v1428 = 4l * v1426;
            assert("Tensor range check" && 0 <= v1426 && v1426 < 1l);
            int v1429; float v1430;
            Tuple0 tmp76 = Tuple0{0l, 0.0f};
            v1429 = tmp76.v0; v1430 = tmp76.v1;
            while (while_method_1(v1429)){
                assert("Tensor range check" && 0 <= v1429 && v1429 < 4l);
                int v1432;
                v1432 = v1429 + v1428;
                float v1433;
                v1433 = v1415[v1432];
                float v1434;
                v1434 = v1430 + v1433;
                v1430 = v1434;
                v1429 += 1l ;
            }
            auto v1435 = cooperative_groups::coalesced_threads();
            int v1436;
            v1436 = threadIdx.x;
            int v1437;
            v1437 = v1436 / 16l;
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
            Tuple0 tmp77 = Tuple0{0l, v1445};
            v1446 = tmp77.v0; v1447 = tmp77.v1;
            while (while_method_1(v1446)){
                assert("Tensor range check" && 0 <= v1446 && v1446 < 4l);
                int v1449;
                v1449 = v1446 + v1428;
                float v1450;
                v1450 = v1415[v1449];
                float v1451;
                v1451 = v1447 + v1450;
                assert("Tensor range check" && 0 <= v1446 && v1446 < 4l);
                v1424[v1449] = v1451;
                v1447 = v1451;
                v1446 += 1l ;
            }
            float v1452;
            v1452 = v1425 + v1444;
            v1425 = v1452;
            v1426 += 1l ;
        }
        float v1453[4l];
        bool v1454[4l];
        int v1455;
        v1455 = 0l;
        while (while_method_3(v1455)){
            int v1457;
            v1457 = 0l;
            while (while_method_1(v1457)){
                assert("Tensor range check" && 0 <= v1455 && v1455 < 1l);
                assert("Tensor range check" && 0 <= v1457 && v1457 < 4l);
                int v1459;
                v1459 = 4l * v1455;
                int v1460;
                v1460 = v1459 + v1457;
                float v1461;
                v1461 = v1424[v1460];
                float v1462;
                v1462 = v1415[v1460];
                bool v1463;
                v1463 = v1462 > 0.0f;
                assert("Tensor range check" && 0 <= v1455 && v1455 < 1l);
                assert("Tensor range check" && 0 <= v1457 && v1457 < 4l);
                v1453[v1460] = v1461;
                v1454[v1460] = v1463;
                v1457 += 1l ;
            }
            v1455 += 1l ;
        }
        float v1464; bool v1465;
        Tuple3 tmp78 = Tuple3{-1.0f / 0.0f, false};
        v1464 = tmp78.v0; v1465 = tmp78.v1;
        int v1466;
        v1466 = 0l;
        while (while_method_3(v1466)){
            int v1468;
            v1468 = 0l;
            while (while_method_1(v1468)){
                assert("Tensor range check" && 0 <= v1466 && v1466 < 1l);
                assert("Tensor range check" && 0 <= v1468 && v1468 < 4l);
                int v1470;
                v1470 = 4l * v1466;
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
                v1468 += 1l ;
            }
            v1466 += 1l ;
        }
        auto v1482 = cooperative_groups::coalesced_threads();
        int v1483;
        v1483 = threadIdx.x;
        int v1484;
        v1484 = v1483 / 16l;
        auto v1485 = cooperative_groups::labeled_partition(v1482,v1484);
        Closure5 v1486{};
        float v1487; bool v1488;
        Tuple3 tmp79 = cooperative_groups::reduce(v1485, Tuple3{v1464, v1465}, v1486);
        v1487 = tmp79.v0; v1488 = tmp79.v1;
        bool v1489;
        v1489 = v1488 == false;
        if (v1489){
            assert("The local reduce must be true." && v1488);
        } else {
        }
        float v1491[4l];
        int v1492[4l];
        int v1493;
        v1493 = 0l;
        while (while_method_3(v1493)){
            int v1495;
            v1495 = 0l;
            while (while_method_1(v1495)){
                assert("Tensor range check" && 0 <= v1493 && v1493 < 1l);
                assert("Tensor range check" && 0 <= v1495 && v1495 < 4l);
                int v1497;
                v1497 = 4l * v1493;
                int v1498;
                v1498 = v1497 + v1495;
                int v1499;
                v1499 = v1286[v1498];
                float v1500;
                v1500 = curand_uniform(&v1261);
                assert("Tensor range check" && 0 <= v1493 && v1493 < 1l);
                assert("Tensor range check" && 0 <= v1495 && v1495 < 4l);
                v1491[v1498] = v1500;
                v1492[v1498] = v1499;
                v1495 += 1l ;
            }
            v1493 += 1l ;
        }
        float v1501; int v1502;
        Tuple1 tmp80 = Tuple1{0.0f, 2147483647l};
        v1501 = tmp80.v0; v1502 = tmp80.v1;
        int v1503;
        v1503 = 0l;
        while (while_method_3(v1503)){
            int v1505;
            v1505 = 0l;
            while (while_method_1(v1505)){
                assert("Tensor range check" && 0 <= v1503 && v1503 < 1l);
                assert("Tensor range check" && 0 <= v1505 && v1505 < 4l);
                int v1507;
                v1507 = 4l * v1503;
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
                v1505 += 1l ;
            }
            v1503 += 1l ;
        }
        auto v1514 = cooperative_groups::coalesced_threads();
        int v1515;
        v1515 = threadIdx.x;
        int v1516;
        v1516 = v1515 / 16l;
        auto v1517 = cooperative_groups::labeled_partition(v1514,v1516);
        Closure6 v1518{};
        float v1519; int v1520;
        Tuple1 tmp81 = cooperative_groups::reduce(v1517, Tuple1{v1501, v1502}, v1518);
        v1519 = tmp81.v0; v1520 = tmp81.v1;
        float v1521;
        v1521 = v1487 * v1519;
        int v1522[4l];
        bool v1523[4l];
        int v1524;
        v1524 = 0l;
        while (while_method_3(v1524)){
            int v1526;
            v1526 = 0l;
            while (while_method_1(v1526)){
                assert("Tensor range check" && 0 <= v1524 && v1524 < 1l);
                assert("Tensor range check" && 0 <= v1526 && v1526 < 4l);
                int v1528;
                v1528 = 4l * v1524;
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
                    v1535 = 2147483647l; v1536 = false;
                }
                assert("Tensor range check" && 0 <= v1524 && v1524 < 1l);
                assert("Tensor range check" && 0 <= v1526 && v1526 < 4l);
                v1522[v1529] = v1535;
                v1523[v1529] = v1536;
                v1526 += 1l ;
            }
            v1524 += 1l ;
        }
        int v1537; bool v1538;
        Tuple4 tmp82 = Tuple4{2147483647l, false};
        v1537 = tmp82.v0; v1538 = tmp82.v1;
        int v1539;
        v1539 = 0l;
        while (while_method_3(v1539)){
            int v1541;
            v1541 = 0l;
            while (while_method_1(v1541)){
                assert("Tensor range check" && 0 <= v1539 && v1539 < 1l);
                assert("Tensor range check" && 0 <= v1541 && v1541 < 4l);
                int v1543;
                v1543 = 4l * v1539;
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
                v1541 += 1l ;
            }
            v1539 += 1l ;
        }
        auto v1555 = cooperative_groups::coalesced_threads();
        int v1556;
        v1556 = threadIdx.x;
        int v1557;
        v1557 = v1556 / 16l;
        auto v1558 = cooperative_groups::labeled_partition(v1555,v1557);
        Closure7 v1559{};
        int v1560; bool v1561;
        Tuple4 tmp83 = cooperative_groups::reduce(v1558, Tuple4{v1537, v1538}, v1559);
        v1560 = tmp83.v0; v1561 = tmp83.v1;
        bool v1562;
        v1562 = v1561 == false;
        if (v1562){
            assert("The local reduce must be true." && v1561);
        } else {
        }
        assert("Tensor range check" && 0 <= v1275 && v1275 < 64l);
        int v1564;
        v1564 = 0l;
        while (while_method_3(v1564)){
            assert("Tensor range check" && 0 <= v1564 && v1564 < 1l);
            int v1566;
            v1566 = 64l * v1564;
            int v1567;
            v1567 = v1566 + v1284;
            assert("Tensor range check" && 0 <= v1564 && v1564 < 1l);
            int v1568;
            v1568 = 4l * v1564;
            int4* v1569;
            v1569 = reinterpret_cast<int4*>(v1415 + v1568);
            int4* v1570;
            v1570 = reinterpret_cast<int4*>(v15 + v1567);
            assert("Pointer alignment check" && (unsigned long long)(v1569) % 4l == 0 && (unsigned long long)(v1570) % 4l == 0);
            *v1570 = *v1569;
            v1564 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1275 && v1275 < 64l);
        int v1571;
        v1571 = 2l * v1275;
        int v1572;
        v1572 = v1571 + v1267;
        v16[v1572] = v1560;
        v1275 += 1l ;
    }
    v17.sync() ;
    return ;
}
extern "C" __global__ void entry5(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14, float * v15, int * v16) {
    auto v17 = cooperative_groups::this_grid();
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
    v22 = v18 % 32l;
    int v23;
    v23 = v18 / 32l;
    bool v24;
    v24 = v23 < 1l;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
    } else {
    }
    assert("Tensor range check" && 0 <= v23 && v23 < 1l);
    assert("Tensor range check" && 0 <= v22 && v22 < 32l);
    int v27;
    v27 = 4l * v22;
    int v28;
    v28 = 128l * v23;
    int v29;
    v29 = v28 + v27;
    assert("Tensor range check" && 0 <= v23 && v23 < 1l);
    assert("Tensor range check" && 0 <= v22 && v22 < 32l);
    int v30;
    v30 = blockIdx.x;
    int v31;
    v31 = v30;
    while (while_method_2(v31)){
        bool v33;
        v33 = 0l <= v31;
        bool v34;
        v34 = v33 == false;
        if (v34){
            assert("The index needs to be zero or positive." && v33);
        } else {
        }
        bool v36;
        v36 = v31 < 64l;
        bool v37;
        v37 = v36 == false;
        if (v37){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v36);
        } else {
        }
        assert("Tensor range check" && 0 <= v31 && v31 < 64l);
        int v39;
        v39 = 128l * v31;
        int v40;
        v40 = v39 + v29;
        int v41[4l];
        int v42[4l];
        int v43;
        v43 = 0l;
        while (while_method_3(v43)){
            assert("Tensor range check" && 0 <= v43 && v43 < 1l);
            int v45;
            v45 = 4l * v43;
            assert("Tensor range check" && 0 <= v43 && v43 < 1l);
            int v46;
            v46 = 128l * v43;
            int v47;
            v47 = v46 + v40;
            int4* v48;
            v48 = reinterpret_cast<int4*>(v0 + v47);
            int4* v49;
            v49 = reinterpret_cast<int4*>(v41 + v45);
            assert("Pointer alignment check" && (unsigned long long)(v48) % 4l == 0 && (unsigned long long)(v49) % 4l == 0);
            *v49 = *v48;
            v43 += 1l ;
        }
        int v50;
        v50 = 0l;
        while (while_method_3(v50)){
            int v52;
            v52 = 0l;
            while (while_method_1(v52)){
                bool v54;
                v54 = 0l <= v52;
                bool v56;
                if (v54){
                    bool v55;
                    v55 = v52 < 4l;
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
                v59 = 0l <= v22;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v22 < 32l;
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
                v64 = v22 * 4l;
                int v65;
                v65 = v52 + v64;
                bool v66;
                v66 = 0l <= v50;
                bool v68;
                if (v66){
                    bool v67;
                    v67 = v50 < 1l;
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
                v71 = v50 * 128l;
                int v72;
                v72 = v65 + v71;
                assert("Tensor range check" && 0 <= v50 && v50 < 1l);
                assert("Tensor range check" && 0 <= v52 && v52 < 4l);
                int v73;
                v73 = 4l * v50;
                int v74;
                v74 = v73 + v52;
                v42[v74] = v72;
                v52 += 1l ;
            }
            v50 += 1l ;
        }
        bool v75;
        v75 = 0l <= v23;
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
        v82 = v31 + v23;
        assert("Tensor range check" && 0 <= v31 && v31 < 64l);
        int v83;
        v83 = 0l;
        while (while_method_3(v83)){
            assert("Tensor range check" && 0 <= v83 && v83 < 1l);
            int v85;
            v85 = 128l * v83;
            int v86;
            v86 = v85 + v40;
            assert("Tensor range check" && 0 <= v83 && v83 < 1l);
            int v87;
            v87 = 4l * v83;
            int4* v88;
            v88 = reinterpret_cast<int4*>(v41 + v87);
            int4* v89;
            v89 = reinterpret_cast<int4*>(v2 + v86);
            assert("Pointer alignment check" && (unsigned long long)(v88) % 4l == 0 && (unsigned long long)(v89) % 4l == 0);
            *v89 = *v88;
            v83 += 1l ;
        }
        v31 += 1l ;
    }
    v17.sync() ;
    int v90;
    v90 = threadIdx.x;
    bool v91;
    v91 = 0l <= v90;
    bool v92;
    v92 = v91 == false;
    if (v92){
        assert("The index needs to be zero or positive." && v91);
    } else {
    }
    int v94;
    v94 = v90 % 32l;
    int v95;
    v95 = v90 / 32l;
    bool v96;
    v96 = v95 < 1l;
    bool v97;
    v97 = v96 == false;
    if (v97){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v96);
    } else {
    }
    assert("Tensor range check" && 0 <= v95 && v95 < 1l);
    assert("Tensor range check" && 0 <= v94 && v94 < 32l);
    int v99;
    v99 = 4l * v94;
    int v100;
    v100 = 128l * v95;
    int v101;
    v101 = v100 + v99;
    assert("Tensor range check" && 0 <= v95 && v95 < 1l);
    assert("Tensor range check" && 0 <= v94 && v94 < 32l);
    int v102;
    v102 = blockIdx.x;
    int v103;
    v103 = v102;
    while (while_method_2(v103)){
        bool v105;
        v105 = 0l <= v103;
        bool v106;
        v106 = v105 == false;
        if (v106){
            assert("The index needs to be zero or positive." && v105);
        } else {
        }
        bool v108;
        v108 = v103 < 64l;
        bool v109;
        v109 = v108 == false;
        if (v109){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v108);
        } else {
        }
        assert("Tensor range check" && 0 <= v103 && v103 < 64l);
        int v111;
        v111 = 128l * v103;
        int v112;
        v112 = v111 + v101;
        float v113[4l];
        int v114[4l];
        int v115;
        v115 = 0l;
        while (while_method_3(v115)){
            assert("Tensor range check" && 0 <= v115 && v115 < 1l);
            int v117;
            v117 = 4l * v115;
            assert("Tensor range check" && 0 <= v115 && v115 < 1l);
            int v118;
            v118 = 128l * v115;
            int v119;
            v119 = v118 + v112;
            int4* v120;
            v120 = reinterpret_cast<int4*>(v1 + v119);
            int4* v121;
            v121 = reinterpret_cast<int4*>(v113 + v117);
            assert("Pointer alignment check" && (unsigned long long)(v120) % 4l == 0 && (unsigned long long)(v121) % 4l == 0);
            *v121 = *v120;
            v115 += 1l ;
        }
        int v122;
        v122 = 0l;
        while (while_method_3(v122)){
            int v124;
            v124 = 0l;
            while (while_method_1(v124)){
                bool v126;
                v126 = 0l <= v124;
                bool v128;
                if (v126){
                    bool v127;
                    v127 = v124 < 4l;
                    v128 = v127;
                } else {
                    v128 = false;
                }
                bool v129;
                v129 = v128 == false;
                if (v129){
                    assert("The indices should be inside the range of the dimension." && v128);
                } else {
                }
                bool v131;
                v131 = 0l <= v94;
                bool v133;
                if (v131){
                    bool v132;
                    v132 = v94 < 32l;
                    v133 = v132;
                } else {
                    v133 = false;
                }
                bool v134;
                v134 = v133 == false;
                if (v134){
                    assert("The indices should be inside the range of the dimension." && v133);
                } else {
                }
                int v136;
                v136 = v94 * 4l;
                int v137;
                v137 = v124 + v136;
                bool v138;
                v138 = 0l <= v122;
                bool v140;
                if (v138){
                    bool v139;
                    v139 = v122 < 1l;
                    v140 = v139;
                } else {
                    v140 = false;
                }
                bool v141;
                v141 = v140 == false;
                if (v141){
                    assert("The indices should be inside the range of the dimension." && v140);
                } else {
                }
                int v143;
                v143 = v122 * 128l;
                int v144;
                v144 = v137 + v143;
                assert("Tensor range check" && 0 <= v122 && v122 < 1l);
                assert("Tensor range check" && 0 <= v124 && v124 < 4l);
                int v145;
                v145 = 4l * v122;
                int v146;
                v146 = v145 + v124;
                v114[v146] = v144;
                v124 += 1l ;
            }
            v122 += 1l ;
        }
        bool v147;
        v147 = 0l <= v95;
        bool v148;
        v148 = v147 && v96;
        bool v149;
        v149 = v148 == false;
        if (v149){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v148);
        } else {
        }
        bool v151;
        v151 = v105 && v108;
        bool v152;
        v152 = v151 == false;
        if (v152){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v151);
        } else {
        }
        int v154;
        v154 = v103 + v95;
        int v155[4l];
        int v156[4l];
        int v157;
        v157 = 0l;
        while (while_method_3(v157)){
            int v159;
            v159 = 0l;
            while (while_method_1(v159)){
                assert("Tensor range check" && 0 <= v157 && v157 < 1l);
                assert("Tensor range check" && 0 <= v159 && v159 < 4l);
                int v161;
                v161 = 4l * v157;
                int v162;
                v162 = v161 + v159;
                int v163;
                v163 = v114[v162];
                assert("Tensor range check" && 0 <= v157 && v157 < 1l);
                assert("Tensor range check" && 0 <= v159 && v159 < 4l);
                v155[v162] = v154;
                v156[v162] = v163;
                v159 += 1l ;
            }
            v157 += 1l ;
        }
        assert("Tensor range check" && 0 <= v103 && v103 < 64l);
        int v164;
        v164 = 0l;
        while (while_method_3(v164)){
            assert("Tensor range check" && 0 <= v164 && v164 < 1l);
            int v166;
            v166 = 128l * v164;
            int v167;
            v167 = v166 + v112;
            assert("Tensor range check" && 0 <= v164 && v164 < 1l);
            int v168;
            v168 = 4l * v164;
            int4* v169;
            v169 = reinterpret_cast<int4*>(v155 + v168);
            int4* v170;
            v170 = reinterpret_cast<int4*>(v9 + v167);
            assert("Pointer alignment check" && (unsigned long long)(v169) % 4l == 0 && (unsigned long long)(v170) % 4l == 0);
            *v170 = *v169;
            int4* v171;
            v171 = reinterpret_cast<int4*>(v156 + v168);
            int4* v172;
            v172 = reinterpret_cast<int4*>(v10 + v167);
            assert("Pointer alignment check" && (unsigned long long)(v171) % 4l == 0 && (unsigned long long)(v172) % 4l == 0);
            *v172 = *v171;
            v164 += 1l ;
        }
        v103 += 1l ;
    }
    v17.sync() ;
    int v173;
    v173 = threadIdx.x;
    bool v174;
    v174 = 0l <= v173;
    bool v175;
    v175 = v174 == false;
    if (v175){
        assert("The index needs to be zero or positive." && v174);
    } else {
    }
    int v177;
    v177 = v173 % 32l;
    int v178;
    v178 = v173 / 32l;
    bool v179;
    v179 = v178 < 1l;
    bool v180;
    v180 = v179 == false;
    if (v180){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v179);
    } else {
    }
    assert("Tensor range check" && 0 <= v178 && v178 < 1l);
    assert("Tensor range check" && 0 <= v177 && v177 < 32l);
    int v182;
    v182 = 4l * v177;
    int v183;
    v183 = 128l * v178;
    int v184;
    v184 = v183 + v182;
    assert("Tensor range check" && 0 <= v178 && v178 < 1l);
    int v185;
    v185 = blockIdx.x;
    int v186;
    v186 = v185;
    while (while_method_2(v186)){
        bool v188;
        v188 = 0l <= v186;
        bool v189;
        v189 = v188 == false;
        if (v189){
            assert("The index needs to be zero or positive." && v188);
        } else {
        }
        bool v191;
        v191 = v186 < 64l;
        bool v192;
        v192 = v191 == false;
        if (v192){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v191);
        } else {
        }
        assert("Tensor range check" && 0 <= v186 && v186 < 64l);
        int v194;
        v194 = 128l * v186;
        int v195;
        v195 = v194 + v184;
        float v196[4l];
        int v197[4l];
        int v198;
        v198 = 0l;
        while (while_method_3(v198)){
            assert("Tensor range check" && 0 <= v198 && v198 < 1l);
            int v200;
            v200 = 4l * v198;
            assert("Tensor range check" && 0 <= v198 && v198 < 1l);
            int v201;
            v201 = 128l * v198;
            int v202;
            v202 = v201 + v195;
            int4* v203;
            v203 = reinterpret_cast<int4*>(v1 + v202);
            int4* v204;
            v204 = reinterpret_cast<int4*>(v196 + v200);
            assert("Pointer alignment check" && (unsigned long long)(v203) % 4l == 0 && (unsigned long long)(v204) % 4l == 0);
            *v204 = *v203;
            v198 += 1l ;
        }
        int v205;
        v205 = 0l;
        while (while_method_3(v205)){
            int v207;
            v207 = 0l;
            while (while_method_1(v207)){
                bool v209;
                v209 = 0l <= v207;
                bool v211;
                if (v209){
                    bool v210;
                    v210 = v207 < 4l;
                    v211 = v210;
                } else {
                    v211 = false;
                }
                bool v212;
                v212 = v211 == false;
                if (v212){
                    assert("The indices should be inside the range of the dimension." && v211);
                } else {
                }
                bool v214;
                v214 = 0l <= v177;
                bool v216;
                if (v214){
                    bool v215;
                    v215 = v177 < 32l;
                    v216 = v215;
                } else {
                    v216 = false;
                }
                bool v217;
                v217 = v216 == false;
                if (v217){
                    assert("The indices should be inside the range of the dimension." && v216);
                } else {
                }
                int v219;
                v219 = v177 * 4l;
                int v220;
                v220 = v207 + v219;
                bool v221;
                v221 = 0l <= v205;
                bool v223;
                if (v221){
                    bool v222;
                    v222 = v205 < 1l;
                    v223 = v222;
                } else {
                    v223 = false;
                }
                bool v224;
                v224 = v223 == false;
                if (v224){
                    assert("The indices should be inside the range of the dimension." && v223);
                } else {
                }
                int v226;
                v226 = v205 * 128l;
                int v227;
                v227 = v220 + v226;
                assert("Tensor range check" && 0 <= v205 && v205 < 1l);
                assert("Tensor range check" && 0 <= v207 && v207 < 4l);
                int v228;
                v228 = 4l * v205;
                int v229;
                v229 = v228 + v207;
                v197[v229] = v227;
                v207 += 1l ;
            }
            v205 += 1l ;
        }
        bool v230;
        v230 = 0l <= v178;
        bool v231;
        v231 = v230 && v179;
        bool v232;
        v232 = v231 == false;
        if (v232){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v231);
        } else {
        }
        bool v234;
        v234 = v188 && v191;
        bool v235;
        v235 = v234 == false;
        if (v235){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v234);
        } else {
        }
        int v237;
        v237 = v186 + v178;
        assert("Tensor range check" && 0 <= v186 && v186 < 64l);
        v11[v237] = v237;
        v186 += 1l ;
    }
    v17.sync() ;
    int v238;
    v238 = threadIdx.x;
    bool v239;
    v239 = 0l <= v238;
    bool v240;
    v240 = v239 == false;
    if (v240){
        assert("The index needs to be zero or positive." && v239);
    } else {
    }
    int v242;
    v242 = v238 % 32l;
    int v243;
    v243 = v238 / 32l;
    bool v244;
    v244 = v243 < 1l;
    bool v245;
    v245 = v244 == false;
    if (v245){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v244);
    } else {
    }
    assert("Tensor range check" && 0 <= v243 && v243 < 1l);
    assert("Tensor range check" && 0 <= v242 && v242 < 32l);
    int v247;
    v247 = 4l * v242;
    int v248;
    v248 = 128l * v243;
    int v249;
    v249 = v248 + v247;
    assert("Tensor range check" && 0 <= v243 && v243 < 1l);
    assert("Tensor range check" && 0 <= v242 && v242 < 32l);
    int v250;
    v250 = blockIdx.x;
    int v251;
    v251 = v250;
    while (while_method_2(v251)){
        bool v253;
        v253 = 0l <= v251;
        bool v254;
        v254 = v253 == false;
        if (v254){
            assert("The index needs to be zero or positive." && v253);
        } else {
        }
        bool v256;
        v256 = v251 < 64l;
        bool v257;
        v257 = v256 == false;
        if (v257){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v256);
        } else {
        }
        assert("Tensor range check" && 0 <= v251 && v251 < 64l);
        int v259;
        v259 = 128l * v251;
        int v260;
        v260 = v259 + v249;
        float v261[4l];
        int v262[4l];
        int v263;
        v263 = 0l;
        while (while_method_3(v263)){
            assert("Tensor range check" && 0 <= v263 && v263 < 1l);
            int v265;
            v265 = 4l * v263;
            assert("Tensor range check" && 0 <= v263 && v263 < 1l);
            int v266;
            v266 = 128l * v263;
            int v267;
            v267 = v266 + v260;
            int4* v268;
            v268 = reinterpret_cast<int4*>(v1 + v267);
            int4* v269;
            v269 = reinterpret_cast<int4*>(v261 + v265);
            assert("Pointer alignment check" && (unsigned long long)(v268) % 4l == 0 && (unsigned long long)(v269) % 4l == 0);
            *v269 = *v268;
            v263 += 1l ;
        }
        int v270;
        v270 = 0l;
        while (while_method_3(v270)){
            int v272;
            v272 = 0l;
            while (while_method_1(v272)){
                bool v274;
                v274 = 0l <= v272;
                bool v276;
                if (v274){
                    bool v275;
                    v275 = v272 < 4l;
                    v276 = v275;
                } else {
                    v276 = false;
                }
                bool v277;
                v277 = v276 == false;
                if (v277){
                    assert("The indices should be inside the range of the dimension." && v276);
                } else {
                }
                bool v279;
                v279 = 0l <= v242;
                bool v281;
                if (v279){
                    bool v280;
                    v280 = v242 < 32l;
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
                int v284;
                v284 = v242 * 4l;
                int v285;
                v285 = v272 + v284;
                bool v286;
                v286 = 0l <= v270;
                bool v288;
                if (v286){
                    bool v287;
                    v287 = v270 < 1l;
                    v288 = v287;
                } else {
                    v288 = false;
                }
                bool v289;
                v289 = v288 == false;
                if (v289){
                    assert("The indices should be inside the range of the dimension." && v288);
                } else {
                }
                int v291;
                v291 = v270 * 128l;
                int v292;
                v292 = v285 + v291;
                assert("Tensor range check" && 0 <= v270 && v270 < 1l);
                assert("Tensor range check" && 0 <= v272 && v272 < 4l);
                int v293;
                v293 = 4l * v270;
                int v294;
                v294 = v293 + v272;
                v262[v294] = v292;
                v272 += 1l ;
            }
            v270 += 1l ;
        }
        bool v295;
        v295 = 0l <= v243;
        bool v296;
        v296 = v295 && v244;
        bool v297;
        v297 = v296 == false;
        if (v297){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v296);
        } else {
        }
        bool v299;
        v299 = v253 && v256;
        bool v300;
        v300 = v299 == false;
        if (v300){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v299);
        } else {
        }
        int v302;
        v302 = v251 + v243;
        float v303;
        v303 = 0.0f;
        int v304;
        v304 = 0l;
        while (while_method_3(v304)){
            int v306;
            v306 = 0l;
            while (while_method_1(v306)){
                assert("Tensor range check" && 0 <= v304 && v304 < 1l);
                assert("Tensor range check" && 0 <= v306 && v306 < 4l);
                int v308;
                v308 = 4l * v304;
                int v309;
                v309 = v308 + v306;
                float v310;
                v310 = v261[v309];
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
        int v314;
        v314 = v313 / 32l;
        auto v315 = cooperative_groups::labeled_partition(v312,v314);
        Closure0 v316{};
        float v317;
        v317 = cooperative_groups::reduce(v315, v303, v316);
        float v318;
        v318 = v317 / 128.0f;
        float v319[4l];
        int v320;
        v320 = 0l;
        while (while_method_3(v320)){
            int v322;
            v322 = 0l;
            while (while_method_1(v322)){
                assert("Tensor range check" && 0 <= v320 && v320 < 1l);
                assert("Tensor range check" && 0 <= v322 && v322 < 4l);
                int v324;
                v324 = 4l * v320;
                int v325;
                v325 = v324 + v322;
                float v326;
                v326 = v261[v325];
                float v327;
                v327 = v326 - v318;
                float v328;
                v328 = exp(v327);
                assert("Tensor range check" && 0 <= v320 && v320 < 1l);
                assert("Tensor range check" && 0 <= v322 && v322 < 4l);
                v319[v325] = v328;
                v322 += 1l ;
            }
            v320 += 1l ;
        }
        float v329;
        v329 = 0.0f;
        int v330;
        v330 = 0l;
        while (while_method_3(v330)){
            int v332;
            v332 = 0l;
            while (while_method_1(v332)){
                assert("Tensor range check" && 0 <= v330 && v330 < 1l);
                assert("Tensor range check" && 0 <= v332 && v332 < 4l);
                int v334;
                v334 = 4l * v330;
                int v335;
                v335 = v334 + v332;
                float v336;
                v336 = v319[v335];
                float v337;
                v337 = v329 + v336;
                v329 = v337;
                v332 += 1l ;
            }
            v330 += 1l ;
        }
        auto v338 = cooperative_groups::coalesced_threads();
        int v339;
        v339 = threadIdx.x;
        int v340;
        v340 = v339 / 32l;
        auto v341 = cooperative_groups::labeled_partition(v338,v340);
        float v342;
        v342 = cooperative_groups::reduce(v341, v329, v316);
        float v343[4l];
        int v344;
        v344 = 0l;
        while (while_method_3(v344)){
            int v346;
            v346 = 0l;
            while (while_method_1(v346)){
                assert("Tensor range check" && 0 <= v344 && v344 < 1l);
                assert("Tensor range check" && 0 <= v346 && v346 < 4l);
                int v348;
                v348 = 4l * v344;
                int v349;
                v349 = v348 + v346;
                float v350;
                v350 = v319[v349];
                float v351;
                v351 = v350 / v342;
                assert("Tensor range check" && 0 <= v344 && v344 < 1l);
                assert("Tensor range check" && 0 <= v346 && v346 < 4l);
                v343[v349] = v351;
                v346 += 1l ;
            }
            v344 += 1l ;
        }
        assert("Tensor range check" && 0 <= v251 && v251 < 64l);
        int v352;
        v352 = 0l;
        while (while_method_3(v352)){
            assert("Tensor range check" && 0 <= v352 && v352 < 1l);
            int v354;
            v354 = 128l * v352;
            int v355;
            v355 = v354 + v260;
            assert("Tensor range check" && 0 <= v352 && v352 < 1l);
            int v356;
            v356 = 4l * v352;
            int4* v357;
            v357 = reinterpret_cast<int4*>(v343 + v356);
            int4* v358;
            v358 = reinterpret_cast<int4*>(v3 + v355);
            assert("Pointer alignment check" && (unsigned long long)(v357) % 4l == 0 && (unsigned long long)(v358) % 4l == 0);
            *v358 = *v357;
            v352 += 1l ;
        }
        v251 += 1l ;
    }
    v17.sync() ;
    int v359;
    v359 = threadIdx.x;
    bool v360;
    v360 = 0l <= v359;
    bool v361;
    v361 = v360 == false;
    if (v361){
        assert("The index needs to be zero or positive." && v360);
    } else {
    }
    int v363;
    v363 = v359 % 32l;
    int v364;
    v364 = v359 / 32l;
    bool v365;
    v365 = v364 < 1l;
    bool v366;
    v366 = v365 == false;
    if (v366){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v365);
    } else {
    }
    assert("Tensor range check" && 0 <= v364 && v364 < 1l);
    assert("Tensor range check" && 0 <= v363 && v363 < 32l);
    int v368;
    v368 = 4l * v363;
    int v369;
    v369 = 128l * v364;
    int v370;
    v370 = v369 + v368;
    assert("Tensor range check" && 0 <= v364 && v364 < 1l);
    assert("Tensor range check" && 0 <= v363 && v363 < 32l);
    int v371;
    v371 = blockIdx.x;
    int v372;
    v372 = v371;
    while (while_method_2(v372)){
        bool v374;
        v374 = 0l <= v372;
        bool v375;
        v375 = v374 == false;
        if (v375){
            assert("The index needs to be zero or positive." && v374);
        } else {
        }
        bool v377;
        v377 = v372 < 64l;
        bool v378;
        v378 = v377 == false;
        if (v378){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v377);
        } else {
        }
        assert("Tensor range check" && 0 <= v372 && v372 < 64l);
        int v380;
        v380 = 128l * v372;
        int v381;
        v381 = v380 + v370;
        float v382[4l];
        int v383[4l];
        int v384;
        v384 = 0l;
        while (while_method_3(v384)){
            assert("Tensor range check" && 0 <= v384 && v384 < 1l);
            int v386;
            v386 = 4l * v384;
            assert("Tensor range check" && 0 <= v384 && v384 < 1l);
            int v387;
            v387 = 128l * v384;
            int v388;
            v388 = v387 + v381;
            int4* v389;
            v389 = reinterpret_cast<int4*>(v1 + v388);
            int4* v390;
            v390 = reinterpret_cast<int4*>(v382 + v386);
            assert("Pointer alignment check" && (unsigned long long)(v389) % 4l == 0 && (unsigned long long)(v390) % 4l == 0);
            *v390 = *v389;
            v384 += 1l ;
        }
        int v391;
        v391 = 0l;
        while (while_method_3(v391)){
            int v393;
            v393 = 0l;
            while (while_method_1(v393)){
                bool v395;
                v395 = 0l <= v393;
                bool v397;
                if (v395){
                    bool v396;
                    v396 = v393 < 4l;
                    v397 = v396;
                } else {
                    v397 = false;
                }
                bool v398;
                v398 = v397 == false;
                if (v398){
                    assert("The indices should be inside the range of the dimension." && v397);
                } else {
                }
                bool v400;
                v400 = 0l <= v363;
                bool v402;
                if (v400){
                    bool v401;
                    v401 = v363 < 32l;
                    v402 = v401;
                } else {
                    v402 = false;
                }
                bool v403;
                v403 = v402 == false;
                if (v403){
                    assert("The indices should be inside the range of the dimension." && v402);
                } else {
                }
                int v405;
                v405 = v363 * 4l;
                int v406;
                v406 = v393 + v405;
                bool v407;
                v407 = 0l <= v391;
                bool v409;
                if (v407){
                    bool v408;
                    v408 = v391 < 1l;
                    v409 = v408;
                } else {
                    v409 = false;
                }
                bool v410;
                v410 = v409 == false;
                if (v410){
                    assert("The indices should be inside the range of the dimension." && v409);
                } else {
                }
                int v412;
                v412 = v391 * 128l;
                int v413;
                v413 = v406 + v412;
                assert("Tensor range check" && 0 <= v391 && v391 < 1l);
                assert("Tensor range check" && 0 <= v393 && v393 < 4l);
                int v414;
                v414 = 4l * v391;
                int v415;
                v415 = v414 + v393;
                v383[v415] = v413;
                v393 += 1l ;
            }
            v391 += 1l ;
        }
        bool v416;
        v416 = 0l <= v364;
        bool v417;
        v417 = v416 && v365;
        bool v418;
        v418 = v417 == false;
        if (v418){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v417);
        } else {
        }
        bool v420;
        v420 = v374 && v377;
        bool v421;
        v421 = v420 == false;
        if (v421){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v420);
        } else {
        }
        int v423;
        v423 = v372 + v364;
        float v424[4l];
        int v425;
        v425 = 0l;
        while (while_method_3(v425)){
            int v427;
            v427 = 0l;
            while (while_method_1(v427)){
                assert("Tensor range check" && 0 <= v425 && v425 < 1l);
                assert("Tensor range check" && 0 <= v427 && v427 < 4l);
                int v429;
                v429 = 4l * v425;
                int v430;
                v430 = v429 + v427;
                float v431;
                v431 = v382[v430];
                float v432;
                v432 = v431 * v431;
                assert("Tensor range check" && 0 <= v425 && v425 < 1l);
                assert("Tensor range check" && 0 <= v427 && v427 < 4l);
                v424[v430] = v432;
                v427 += 1l ;
            }
            v425 += 1l ;
        }
        float v433;
        v433 = 0.0f;
        int v434;
        v434 = 0l;
        while (while_method_3(v434)){
            int v436;
            v436 = 0l;
            while (while_method_1(v436)){
                assert("Tensor range check" && 0 <= v434 && v434 < 1l);
                assert("Tensor range check" && 0 <= v436 && v436 < 4l);
                int v438;
                v438 = 4l * v434;
                int v439;
                v439 = v438 + v436;
                float v440;
                v440 = v424[v439];
                float v441;
                v441 = v433 + v440;
                v433 = v441;
                v436 += 1l ;
            }
            v434 += 1l ;
        }
        auto v442 = cooperative_groups::coalesced_threads();
        int v443;
        v443 = threadIdx.x;
        int v444;
        v444 = v443 / 32l;
        auto v445 = cooperative_groups::labeled_partition(v442,v444);
        Closure0 v446{};
        float v447;
        v447 = cooperative_groups::reduce(v445, v433, v446);
        float v448[4l];
        int v449;
        v449 = 0l;
        while (while_method_3(v449)){
            int v451;
            v451 = 0l;
            while (while_method_1(v451)){
                assert("Tensor range check" && 0 <= v449 && v449 < 1l);
                assert("Tensor range check" && 0 <= v451 && v451 < 4l);
                int v453;
                v453 = 4l * v449;
                int v454;
                v454 = v453 + v451;
                float v455;
                v455 = v382[v454];
                bool v456;
                v456 = v447 == 0.0f;
                bool v457;
                v457 = v456 != true;
                float v459;
                if (v457){
                    float v458;
                    v458 = v455 / v447;
                    v459 = v458;
                } else {
                    v459 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v449 && v449 < 1l);
                assert("Tensor range check" && 0 <= v451 && v451 < 4l);
                v448[v454] = v459;
                v451 += 1l ;
            }
            v449 += 1l ;
        }
        assert("Tensor range check" && 0 <= v372 && v372 < 64l);
        int v460;
        v460 = 0l;
        while (while_method_3(v460)){
            assert("Tensor range check" && 0 <= v460 && v460 < 1l);
            int v462;
            v462 = 128l * v460;
            int v463;
            v463 = v462 + v381;
            assert("Tensor range check" && 0 <= v460 && v460 < 1l);
            int v464;
            v464 = 4l * v460;
            int4* v465;
            v465 = reinterpret_cast<int4*>(v448 + v464);
            int4* v466;
            v466 = reinterpret_cast<int4*>(v7 + v463);
            assert("Pointer alignment check" && (unsigned long long)(v465) % 4l == 0 && (unsigned long long)(v466) % 4l == 0);
            *v466 = *v465;
            v460 += 1l ;
        }
        v372 += 1l ;
    }
    v17.sync() ;
    int v467;
    v467 = threadIdx.x;
    bool v468;
    v468 = 0l <= v467;
    bool v469;
    v469 = v468 == false;
    if (v469){
        assert("The index needs to be zero or positive." && v468);
    } else {
    }
    int v471;
    v471 = v467 % 32l;
    int v472;
    v472 = v467 / 32l;
    bool v473;
    v473 = v472 < 1l;
    bool v474;
    v474 = v473 == false;
    if (v474){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v473);
    } else {
    }
    assert("Tensor range check" && 0 <= v472 && v472 < 1l);
    assert("Tensor range check" && 0 <= v471 && v471 < 32l);
    int v476;
    v476 = 4l * v471;
    int v477;
    v477 = 128l * v472;
    int v478;
    v478 = v477 + v476;
    assert("Tensor range check" && 0 <= v472 && v472 < 1l);
    int v479;
    v479 = blockIdx.x;
    int v480;
    v480 = v479;
    while (while_method_2(v480)){
        bool v482;
        v482 = 0l <= v480;
        bool v483;
        v483 = v482 == false;
        if (v483){
            assert("The index needs to be zero or positive." && v482);
        } else {
        }
        bool v485;
        v485 = v480 < 64l;
        bool v486;
        v486 = v485 == false;
        if (v486){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v485);
        } else {
        }
        assert("Tensor range check" && 0 <= v480 && v480 < 64l);
        int v488;
        v488 = 128l * v480;
        int v489;
        v489 = v488 + v478;
        float v490[4l];
        int v491[4l];
        int v492;
        v492 = 0l;
        while (while_method_3(v492)){
            assert("Tensor range check" && 0 <= v492 && v492 < 1l);
            int v494;
            v494 = 4l * v492;
            assert("Tensor range check" && 0 <= v492 && v492 < 1l);
            int v495;
            v495 = 128l * v492;
            int v496;
            v496 = v495 + v489;
            int4* v497;
            v497 = reinterpret_cast<int4*>(v1 + v496);
            int4* v498;
            v498 = reinterpret_cast<int4*>(v490 + v494);
            assert("Pointer alignment check" && (unsigned long long)(v497) % 4l == 0 && (unsigned long long)(v498) % 4l == 0);
            *v498 = *v497;
            v492 += 1l ;
        }
        int v499;
        v499 = 0l;
        while (while_method_3(v499)){
            int v501;
            v501 = 0l;
            while (while_method_1(v501)){
                bool v503;
                v503 = 0l <= v501;
                bool v505;
                if (v503){
                    bool v504;
                    v504 = v501 < 4l;
                    v505 = v504;
                } else {
                    v505 = false;
                }
                bool v506;
                v506 = v505 == false;
                if (v506){
                    assert("The indices should be inside the range of the dimension." && v505);
                } else {
                }
                bool v508;
                v508 = 0l <= v471;
                bool v510;
                if (v508){
                    bool v509;
                    v509 = v471 < 32l;
                    v510 = v509;
                } else {
                    v510 = false;
                }
                bool v511;
                v511 = v510 == false;
                if (v511){
                    assert("The indices should be inside the range of the dimension." && v510);
                } else {
                }
                int v513;
                v513 = v471 * 4l;
                int v514;
                v514 = v501 + v513;
                bool v515;
                v515 = 0l <= v499;
                bool v517;
                if (v515){
                    bool v516;
                    v516 = v499 < 1l;
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
                v520 = v499 * 128l;
                int v521;
                v521 = v514 + v520;
                assert("Tensor range check" && 0 <= v499 && v499 < 1l);
                assert("Tensor range check" && 0 <= v501 && v501 < 4l);
                int v522;
                v522 = 4l * v499;
                int v523;
                v523 = v522 + v501;
                v491[v523] = v521;
                v501 += 1l ;
            }
            v499 += 1l ;
        }
        bool v524;
        v524 = 0l <= v472;
        bool v525;
        v525 = v524 && v473;
        bool v526;
        v526 = v525 == false;
        if (v526){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v525);
        } else {
        }
        bool v528;
        v528 = v482 && v485;
        bool v529;
        v529 = v528 == false;
        if (v529){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v528);
        } else {
        }
        int v531;
        v531 = v480 + v472;
        float v532; int v533;
        Tuple1 tmp84 = Tuple1{-1.0f / 0.0f, 0l};
        v532 = tmp84.v0; v533 = tmp84.v1;
        int v534;
        v534 = 0l;
        while (while_method_3(v534)){
            int v536;
            v536 = 0l;
            while (while_method_1(v536)){
                assert("Tensor range check" && 0 <= v534 && v534 < 1l);
                assert("Tensor range check" && 0 <= v536 && v536 < 4l);
                int v538;
                v538 = 4l * v534;
                int v539;
                v539 = v538 + v536;
                float v540;
                v540 = v490[v539];
                int v541;
                v541 = v491[v539];
                bool v542;
                v542 = v532 > v540;
                float v543; int v544;
                if (v542){
                    v543 = v532; v544 = v533;
                } else {
                    v543 = v540; v544 = v541;
                }
                v532 = v543;
                v533 = v544;
                v536 += 1l ;
            }
            v534 += 1l ;
        }
        auto v545 = cooperative_groups::coalesced_threads();
        int v546;
        v546 = threadIdx.x;
        int v547;
        v547 = v546 / 32l;
        auto v548 = cooperative_groups::labeled_partition(v545,v547);
        Closure1 v549{};
        float v550; int v551;
        Tuple1 tmp85 = cooperative_groups::reduce(v548, Tuple1{v532, v533}, v549);
        v550 = tmp85.v0; v551 = tmp85.v1;
        assert("Tensor range check" && 0 <= v480 && v480 < 64l);
        v8[v531] = v551;
        v480 += 1l ;
    }
    v17.sync() ;
    int v552;
    v552 = threadIdx.x;
    bool v553;
    v553 = 0l <= v552;
    bool v554;
    v554 = v553 == false;
    if (v554){
        assert("The index needs to be zero or positive." && v553);
    } else {
    }
    int v556;
    v556 = v552 % 32l;
    int v557;
    v557 = v552 / 32l;
    bool v558;
    v558 = v557 < 1l;
    bool v559;
    v559 = v558 == false;
    if (v559){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v558);
    } else {
    }
    assert("Tensor range check" && 0 <= v557 && v557 < 1l);
    assert("Tensor range check" && 0 <= v556 && v556 < 32l);
    int v561;
    v561 = 4l * v556;
    int v562;
    v562 = 128l * v557;
    int v563;
    v563 = v562 + v561;
    assert("Tensor range check" && 0 <= v557 && v557 < 1l);
    assert("Tensor range check" && 0 <= v556 && v556 < 32l);
    int v564;
    v564 = blockIdx.x;
    int v565;
    v565 = v564;
    while (while_method_2(v565)){
        bool v567;
        v567 = 0l <= v565;
        bool v568;
        v568 = v567 == false;
        if (v568){
            assert("The index needs to be zero or positive." && v567);
        } else {
        }
        bool v570;
        v570 = v565 < 64l;
        bool v571;
        v571 = v570 == false;
        if (v571){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v570);
        } else {
        }
        assert("Tensor range check" && 0 <= v565 && v565 < 64l);
        int v573;
        v573 = 128l * v565;
        int v574;
        v574 = v573 + v563;
        float v575[4l];
        int v576[4l];
        int v577;
        v577 = 0l;
        while (while_method_3(v577)){
            assert("Tensor range check" && 0 <= v577 && v577 < 1l);
            int v579;
            v579 = 4l * v577;
            assert("Tensor range check" && 0 <= v577 && v577 < 1l);
            int v580;
            v580 = 128l * v577;
            int v581;
            v581 = v580 + v574;
            int4* v582;
            v582 = reinterpret_cast<int4*>(v1 + v581);
            int4* v583;
            v583 = reinterpret_cast<int4*>(v575 + v579);
            assert("Pointer alignment check" && (unsigned long long)(v582) % 4l == 0 && (unsigned long long)(v583) % 4l == 0);
            *v583 = *v582;
            v577 += 1l ;
        }
        int v584;
        v584 = 0l;
        while (while_method_3(v584)){
            int v586;
            v586 = 0l;
            while (while_method_1(v586)){
                bool v588;
                v588 = 0l <= v586;
                bool v590;
                if (v588){
                    bool v589;
                    v589 = v586 < 4l;
                    v590 = v589;
                } else {
                    v590 = false;
                }
                bool v591;
                v591 = v590 == false;
                if (v591){
                    assert("The indices should be inside the range of the dimension." && v590);
                } else {
                }
                bool v593;
                v593 = 0l <= v556;
                bool v595;
                if (v593){
                    bool v594;
                    v594 = v556 < 32l;
                    v595 = v594;
                } else {
                    v595 = false;
                }
                bool v596;
                v596 = v595 == false;
                if (v596){
                    assert("The indices should be inside the range of the dimension." && v595);
                } else {
                }
                int v598;
                v598 = v556 * 4l;
                int v599;
                v599 = v586 + v598;
                bool v600;
                v600 = 0l <= v584;
                bool v602;
                if (v600){
                    bool v601;
                    v601 = v584 < 1l;
                    v602 = v601;
                } else {
                    v602 = false;
                }
                bool v603;
                v603 = v602 == false;
                if (v603){
                    assert("The indices should be inside the range of the dimension." && v602);
                } else {
                }
                int v605;
                v605 = v584 * 128l;
                int v606;
                v606 = v599 + v605;
                assert("Tensor range check" && 0 <= v584 && v584 < 1l);
                assert("Tensor range check" && 0 <= v586 && v586 < 4l);
                int v607;
                v607 = 4l * v584;
                int v608;
                v608 = v607 + v586;
                v576[v608] = v606;
                v586 += 1l ;
            }
            v584 += 1l ;
        }
        bool v609;
        v609 = 0l <= v557;
        bool v610;
        v610 = v609 && v558;
        bool v611;
        v611 = v610 == false;
        if (v611){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v610);
        } else {
        }
        bool v613;
        v613 = v567 && v570;
        bool v614;
        v614 = v613 == false;
        if (v614){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v613);
        } else {
        }
        int v616;
        v616 = v565 + v557;
        float v617;
        v617 = 0.0f;
        int v618;
        v618 = 0l;
        while (while_method_3(v618)){
            int v620;
            v620 = 0l;
            while (while_method_1(v620)){
                assert("Tensor range check" && 0 <= v618 && v618 < 1l);
                assert("Tensor range check" && 0 <= v620 && v620 < 4l);
                int v622;
                v622 = 4l * v618;
                int v623;
                v623 = v622 + v620;
                float v624;
                v624 = v575[v623];
                float v625;
                v625 = v617 + v624;
                v617 = v625;
                v620 += 1l ;
            }
            v618 += 1l ;
        }
        auto v626 = cooperative_groups::coalesced_threads();
        int v627;
        v627 = threadIdx.x;
        int v628;
        v628 = v627 / 32l;
        auto v629 = cooperative_groups::labeled_partition(v626,v628);
        Closure0 v630{};
        float v631;
        v631 = cooperative_groups::reduce(v629, v617, v630);
        float v632;
        v632 = v631 / 128.0f;
        float v633[4l];
        int v634;
        v634 = 0l;
        while (while_method_3(v634)){
            int v636;
            v636 = 0l;
            while (while_method_1(v636)){
                assert("Tensor range check" && 0 <= v634 && v634 < 1l);
                assert("Tensor range check" && 0 <= v636 && v636 < 4l);
                int v638;
                v638 = 4l * v634;
                int v639;
                v639 = v638 + v636;
                float v640;
                v640 = v575[v639];
                float v641;
                v641 = v640 - v632;
                float v642;
                v642 = exp(v641);
                assert("Tensor range check" && 0 <= v634 && v634 < 1l);
                assert("Tensor range check" && 0 <= v636 && v636 < 4l);
                v633[v639] = v642;
                v636 += 1l ;
            }
            v634 += 1l ;
        }
        float v643;
        v643 = 0.0f;
        int v644;
        v644 = 0l;
        while (while_method_3(v644)){
            int v646;
            v646 = 0l;
            while (while_method_1(v646)){
                assert("Tensor range check" && 0 <= v644 && v644 < 1l);
                assert("Tensor range check" && 0 <= v646 && v646 < 4l);
                int v648;
                v648 = 4l * v644;
                int v649;
                v649 = v648 + v646;
                float v650;
                v650 = v633[v649];
                float v651;
                v651 = v643 + v650;
                v643 = v651;
                v646 += 1l ;
            }
            v644 += 1l ;
        }
        auto v652 = cooperative_groups::coalesced_threads();
        int v653;
        v653 = threadIdx.x;
        int v654;
        v654 = v653 / 32l;
        auto v655 = cooperative_groups::labeled_partition(v652,v654);
        float v656;
        v656 = cooperative_groups::reduce(v655, v643, v630);
        float v657[4l];
        int v658;
        v658 = 0l;
        while (while_method_3(v658)){
            int v660;
            v660 = 0l;
            while (while_method_1(v660)){
                assert("Tensor range check" && 0 <= v658 && v658 < 1l);
                assert("Tensor range check" && 0 <= v660 && v660 < 4l);
                int v662;
                v662 = 4l * v658;
                int v663;
                v663 = v662 + v660;
                float v664;
                v664 = v633[v663];
                float v665;
                v665 = v664 / v656;
                assert("Tensor range check" && 0 <= v658 && v658 < 1l);
                assert("Tensor range check" && 0 <= v660 && v660 < 4l);
                v657[v663] = v665;
                v660 += 1l ;
            }
            v658 += 1l ;
        }
        float v666[4l];
        float v667;
        v667 = 0.0f;
        int v668;
        v668 = 0l;
        while (while_method_3(v668)){
            assert("Tensor range check" && 0 <= v668 && v668 < 1l);
            int v670;
            v670 = 4l * v668;
            assert("Tensor range check" && 0 <= v668 && v668 < 1l);
            int v671; float v672;
            Tuple0 tmp86 = Tuple0{0l, 0.0f};
            v671 = tmp86.v0; v672 = tmp86.v1;
            while (while_method_1(v671)){
                assert("Tensor range check" && 0 <= v671 && v671 < 4l);
                int v674;
                v674 = v671 + v670;
                float v675;
                v675 = v657[v674];
                float v676;
                v676 = v672 + v675;
                v672 = v676;
                v671 += 1l ;
            }
            auto v677 = cooperative_groups::coalesced_threads();
            int v678;
            v678 = threadIdx.x;
            int v679;
            v679 = v678 / 32l;
            auto v680 = cooperative_groups::labeled_partition(v677,v679);
            Closure2 v681{};
            float v682;
            v682 = cooperative_groups::inclusive_scan(v680, v672, v681);
            float v683;
            v683 = v680.shfl_up(v682,1);
            bool v684;
            v684 = v680.thread_rank() == 0;
            float v685;
            if (v684){
                v685 = 0.0f;
            } else {
                v685 = v683;
            }
            float v686;
            v686 = v680.shfl(v682,v680.num_threads()-1);
            float v687;
            v687 = v667 + v685;
            int v688; float v689;
            Tuple0 tmp87 = Tuple0{0l, v687};
            v688 = tmp87.v0; v689 = tmp87.v1;
            while (while_method_1(v688)){
                assert("Tensor range check" && 0 <= v688 && v688 < 4l);
                int v691;
                v691 = v688 + v670;
                float v692;
                v692 = v657[v691];
                float v693;
                v693 = v689 + v692;
                assert("Tensor range check" && 0 <= v688 && v688 < 4l);
                v666[v691] = v693;
                v689 = v693;
                v688 += 1l ;
            }
            float v694;
            v694 = v667 + v686;
            v667 = v694;
            v668 += 1l ;
        }
        assert("Tensor range check" && 0 <= v565 && v565 < 64l);
        int v695;
        v695 = 0l;
        while (while_method_3(v695)){
            assert("Tensor range check" && 0 <= v695 && v695 < 1l);
            int v697;
            v697 = 128l * v695;
            int v698;
            v698 = v697 + v574;
            assert("Tensor range check" && 0 <= v695 && v695 < 1l);
            int v699;
            v699 = 4l * v695;
            int4* v700;
            v700 = reinterpret_cast<int4*>(v657 + v699);
            int4* v701;
            v701 = reinterpret_cast<int4*>(v5 + v698);
            assert("Pointer alignment check" && (unsigned long long)(v700) % 4l == 0 && (unsigned long long)(v701) % 4l == 0);
            *v701 = *v700;
            int4* v702;
            v702 = reinterpret_cast<int4*>(v666 + v699);
            int4* v703;
            v703 = reinterpret_cast<int4*>(v6 + v698);
            assert("Pointer alignment check" && (unsigned long long)(v702) % 4l == 0 && (unsigned long long)(v703) % 4l == 0);
            *v703 = *v702;
            v695 += 1l ;
        }
        v565 += 1l ;
    }
    v17.sync() ;
    int v704;
    v704 = threadIdx.x;
    bool v705;
    v705 = 0l <= v704;
    bool v706;
    v706 = v705 == false;
    if (v706){
        assert("The index needs to be zero or positive." && v705);
    } else {
    }
    int v708;
    v708 = v704 % 32l;
    int v709;
    v709 = v704 / 32l;
    bool v710;
    v710 = v709 < 1l;
    bool v711;
    v711 = v710 == false;
    if (v711){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v710);
    } else {
    }
    assert("Tensor range check" && 0 <= v709 && v709 < 1l);
    assert("Tensor range check" && 0 <= v708 && v708 < 32l);
    int v713;
    v713 = 4l * v708;
    int v714;
    v714 = 128l * v709;
    int v715;
    v715 = v714 + v713;
    assert("Tensor range check" && 0 <= v709 && v709 < 1l);
    assert("Tensor range check" && 0 <= v708 && v708 < 32l);
    int v716;
    v716 = blockIdx.x;
    int v717;
    v717 = v716;
    while (while_method_2(v717)){
        bool v719;
        v719 = 0l <= v717;
        bool v720;
        v720 = v719 == false;
        if (v720){
            assert("The index needs to be zero or positive." && v719);
        } else {
        }
        bool v722;
        v722 = v717 < 64l;
        bool v723;
        v723 = v722 == false;
        if (v723){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v722);
        } else {
        }
        assert("Tensor range check" && 0 <= v717 && v717 < 64l);
        int v725;
        v725 = 128l * v717;
        int v726;
        v726 = v725 + v715;
        int v727[4l];
        int v728[4l];
        int v729;
        v729 = 0l;
        while (while_method_3(v729)){
            assert("Tensor range check" && 0 <= v729 && v729 < 1l);
            int v731;
            v731 = 4l * v729;
            assert("Tensor range check" && 0 <= v729 && v729 < 1l);
            int v732;
            v732 = 128l * v729;
            int v733;
            v733 = v732 + v726;
            int4* v734;
            v734 = reinterpret_cast<int4*>(v0 + v733);
            int4* v735;
            v735 = reinterpret_cast<int4*>(v727 + v731);
            assert("Pointer alignment check" && (unsigned long long)(v734) % 4l == 0 && (unsigned long long)(v735) % 4l == 0);
            *v735 = *v734;
            v729 += 1l ;
        }
        int v736;
        v736 = 0l;
        while (while_method_3(v736)){
            int v738;
            v738 = 0l;
            while (while_method_1(v738)){
                bool v740;
                v740 = 0l <= v738;
                bool v742;
                if (v740){
                    bool v741;
                    v741 = v738 < 4l;
                    v742 = v741;
                } else {
                    v742 = false;
                }
                bool v743;
                v743 = v742 == false;
                if (v743){
                    assert("The indices should be inside the range of the dimension." && v742);
                } else {
                }
                bool v745;
                v745 = 0l <= v708;
                bool v747;
                if (v745){
                    bool v746;
                    v746 = v708 < 32l;
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
                int v750;
                v750 = v708 * 4l;
                int v751;
                v751 = v738 + v750;
                bool v752;
                v752 = 0l <= v736;
                bool v754;
                if (v752){
                    bool v753;
                    v753 = v736 < 1l;
                    v754 = v753;
                } else {
                    v754 = false;
                }
                bool v755;
                v755 = v754 == false;
                if (v755){
                    assert("The indices should be inside the range of the dimension." && v754);
                } else {
                }
                int v757;
                v757 = v736 * 128l;
                int v758;
                v758 = v751 + v757;
                assert("Tensor range check" && 0 <= v736 && v736 < 1l);
                assert("Tensor range check" && 0 <= v738 && v738 < 4l);
                int v759;
                v759 = 4l * v736;
                int v760;
                v760 = v759 + v738;
                v728[v760] = v758;
                v738 += 1l ;
            }
            v736 += 1l ;
        }
        bool v761;
        v761 = 0l <= v709;
        bool v762;
        v762 = v761 && v710;
        bool v763;
        v763 = v762 == false;
        if (v763){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v762);
        } else {
        }
        bool v765;
        v765 = v719 && v722;
        bool v766;
        v766 = v765 == false;
        if (v766){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v765);
        } else {
        }
        int v768;
        v768 = v717 + v709;
        int v769[4l];
        int v770;
        v770 = 0l;
        int v771;
        v771 = 0l;
        while (while_method_3(v771)){
            assert("Tensor range check" && 0 <= v771 && v771 < 1l);
            int v773;
            v773 = 4l * v771;
            assert("Tensor range check" && 0 <= v771 && v771 < 1l);
            int v774; int v775;
            Tuple2 tmp88 = Tuple2{0l, 0l};
            v774 = tmp88.v0; v775 = tmp88.v1;
            while (while_method_1(v774)){
                assert("Tensor range check" && 0 <= v774 && v774 < 4l);
                int v777;
                v777 = v774 + v773;
                int v778;
                v778 = v727[v777];
                int v779;
                v779 = v775 + v778;
                v775 = v779;
                v774 += 1l ;
            }
            auto v780 = cooperative_groups::coalesced_threads();
            int v781;
            v781 = threadIdx.x;
            int v782;
            v782 = v781 / 32l;
            auto v783 = cooperative_groups::labeled_partition(v780,v782);
            Closure3 v784{};
            int v785;
            v785 = cooperative_groups::inclusive_scan(v783, v775, v784);
            int v786;
            v786 = v783.shfl_up(v785,1);
            bool v787;
            v787 = v783.thread_rank() == 0;
            int v788;
            if (v787){
                v788 = 0l;
            } else {
                v788 = v786;
            }
            int v789;
            v789 = v783.shfl(v785,v783.num_threads()-1);
            int v790;
            v790 = v770 + v788;
            int v791; int v792;
            Tuple2 tmp89 = Tuple2{0l, v790};
            v791 = tmp89.v0; v792 = tmp89.v1;
            while (while_method_1(v791)){
                assert("Tensor range check" && 0 <= v791 && v791 < 4l);
                int v794;
                v794 = v791 + v773;
                int v795;
                v795 = v727[v794];
                assert("Tensor range check" && 0 <= v791 && v791 < 4l);
                v769[v794] = v792;
                int v796;
                v796 = v792 + v795;
                v792 = v796;
                v791 += 1l ;
            }
            int v797;
            v797 = v770 + v789;
            v770 = v797;
            v771 += 1l ;
        }
        assert("Tensor range check" && 0 <= v717 && v717 < 64l);
        int v798;
        v798 = 0l;
        while (while_method_3(v798)){
            assert("Tensor range check" && 0 <= v798 && v798 < 1l);
            int v800;
            v800 = 128l * v798;
            int v801;
            v801 = v800 + v726;
            assert("Tensor range check" && 0 <= v798 && v798 < 1l);
            int v802;
            v802 = 4l * v798;
            int4* v803;
            v803 = reinterpret_cast<int4*>(v769 + v802);
            int4* v804;
            v804 = reinterpret_cast<int4*>(v12 + v801);
            assert("Pointer alignment check" && (unsigned long long)(v803) % 4l == 0 && (unsigned long long)(v804) % 4l == 0);
            *v804 = *v803;
            v798 += 1l ;
        }
        v717 += 1l ;
    }
    v17.sync() ;
    int v805;
    v805 = threadIdx.x;
    bool v806;
    v806 = 0l <= v805;
    bool v807;
    v807 = v806 == false;
    if (v807){
        assert("The index needs to be zero or positive." && v806);
    } else {
    }
    int v809;
    v809 = v805 % 32l;
    int v810;
    v810 = v805 / 32l;
    bool v811;
    v811 = v810 < 1l;
    bool v812;
    v812 = v811 == false;
    if (v812){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v811);
    } else {
    }
    assert("Tensor range check" && 0 <= v810 && v810 < 1l);
    assert("Tensor range check" && 0 <= v809 && v809 < 32l);
    int v814;
    v814 = 4l * v809;
    int v815;
    v815 = 128l * v810;
    int v816;
    v816 = v815 + v814;
    assert("Tensor range check" && 0 <= v810 && v810 < 1l);
    assert("Tensor range check" && 0 <= v809 && v809 < 32l);
    int v817;
    v817 = blockIdx.x;
    int v818;
    v818 = v817;
    while (while_method_2(v818)){
        bool v820;
        v820 = 0l <= v818;
        bool v821;
        v821 = v820 == false;
        if (v821){
            assert("The index needs to be zero or positive." && v820);
        } else {
        }
        bool v823;
        v823 = v818 < 64l;
        bool v824;
        v824 = v823 == false;
        if (v824){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v823);
        } else {
        }
        assert("Tensor range check" && 0 <= v818 && v818 < 64l);
        int v826;
        v826 = 128l * v818;
        int v827;
        v827 = v826 + v816;
        float v828[4l];
        int v829[4l];
        int v830;
        v830 = 0l;
        while (while_method_3(v830)){
            assert("Tensor range check" && 0 <= v830 && v830 < 1l);
            int v832;
            v832 = 4l * v830;
            assert("Tensor range check" && 0 <= v830 && v830 < 1l);
            int v833;
            v833 = 128l * v830;
            int v834;
            v834 = v833 + v827;
            int4* v835;
            v835 = reinterpret_cast<int4*>(v1 + v834);
            int4* v836;
            v836 = reinterpret_cast<int4*>(v828 + v832);
            assert("Pointer alignment check" && (unsigned long long)(v835) % 4l == 0 && (unsigned long long)(v836) % 4l == 0);
            *v836 = *v835;
            v830 += 1l ;
        }
        int v837;
        v837 = 0l;
        while (while_method_3(v837)){
            int v839;
            v839 = 0l;
            while (while_method_1(v839)){
                bool v841;
                v841 = 0l <= v839;
                bool v843;
                if (v841){
                    bool v842;
                    v842 = v839 < 4l;
                    v843 = v842;
                } else {
                    v843 = false;
                }
                bool v844;
                v844 = v843 == false;
                if (v844){
                    assert("The indices should be inside the range of the dimension." && v843);
                } else {
                }
                bool v846;
                v846 = 0l <= v809;
                bool v848;
                if (v846){
                    bool v847;
                    v847 = v809 < 32l;
                    v848 = v847;
                } else {
                    v848 = false;
                }
                bool v849;
                v849 = v848 == false;
                if (v849){
                    assert("The indices should be inside the range of the dimension." && v848);
                } else {
                }
                int v851;
                v851 = v809 * 4l;
                int v852;
                v852 = v839 + v851;
                bool v853;
                v853 = 0l <= v837;
                bool v855;
                if (v853){
                    bool v854;
                    v854 = v837 < 1l;
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
                int v858;
                v858 = v837 * 128l;
                int v859;
                v859 = v852 + v858;
                assert("Tensor range check" && 0 <= v837 && v837 < 1l);
                assert("Tensor range check" && 0 <= v839 && v839 < 4l);
                int v860;
                v860 = 4l * v837;
                int v861;
                v861 = v860 + v839;
                v829[v861] = v859;
                v839 += 1l ;
            }
            v837 += 1l ;
        }
        bool v862;
        v862 = 0l <= v810;
        bool v863;
        v863 = v862 && v811;
        bool v864;
        v864 = v863 == false;
        if (v864){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v863);
        } else {
        }
        bool v866;
        v866 = v820 && v823;
        bool v867;
        v867 = v866 == false;
        if (v867){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v866);
        } else {
        }
        int v869;
        v869 = v818 + v810;
        bool v870[4l];
        int v871;
        v871 = 0l;
        while (while_method_3(v871)){
            int v873;
            v873 = 0l;
            while (while_method_1(v873)){
                assert("Tensor range check" && 0 <= v871 && v871 < 1l);
                assert("Tensor range check" && 0 <= v873 && v873 < 4l);
                int v875;
                v875 = 4l * v871;
                int v876;
                v876 = v875 + v873;
                float v877;
                v877 = v828[v876];
                int v878;
                v878 = v829[v876];
                bool v879;
                v879 = v878 < 4l;
                assert("Tensor range check" && 0 <= v871 && v871 < 1l);
                assert("Tensor range check" && 0 <= v873 && v873 < 4l);
                v870[v876] = v879;
                v873 += 1l ;
            }
            v871 += 1l ;
        }
        int v880[4l];
        int v881;
        v881 = 0l;
        while (while_method_3(v881)){
            int v883;
            v883 = 0l;
            while (while_method_1(v883)){
                assert("Tensor range check" && 0 <= v881 && v881 < 1l);
                assert("Tensor range check" && 0 <= v883 && v883 < 4l);
                int v885;
                v885 = 4l * v881;
                int v886;
                v886 = v885 + v883;
                bool v887;
                v887 = v870[v886];
                int v888;
                if (v887){
                    v888 = 1l;
                } else {
                    v888 = 0l;
                }
                assert("Tensor range check" && 0 <= v881 && v881 < 1l);
                assert("Tensor range check" && 0 <= v883 && v883 < 4l);
                v880[v886] = v888;
                v883 += 1l ;
            }
            v881 += 1l ;
        }
        int v889;
        v889 = 0l;
        int v890;
        v890 = 0l;
        while (while_method_3(v890)){
            int v892;
            v892 = 0l;
            while (while_method_1(v892)){
                assert("Tensor range check" && 0 <= v890 && v890 < 1l);
                assert("Tensor range check" && 0 <= v892 && v892 < 4l);
                int v894;
                v894 = 4l * v890;
                int v895;
                v895 = v894 + v892;
                int v896;
                v896 = v880[v895];
                int v897;
                v897 = v889 + v896;
                v889 = v897;
                v892 += 1l ;
            }
            v890 += 1l ;
        }
        auto v898 = cooperative_groups::coalesced_threads();
        int v899;
        v899 = threadIdx.x;
        int v900;
        v900 = v899 / 32l;
        auto v901 = cooperative_groups::labeled_partition(v898,v900);
        Closure4 v902{};
        int v903;
        v903 = cooperative_groups::reduce(v901, v889, v902);
        float v904[4l];
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
                float v911;
                v911 = v828[v910];
                bool v912;
                v912 = v870[v910];
                float v913;
                if (v912){
                    v913 = v911;
                } else {
                    v913 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v905 && v905 < 1l);
                assert("Tensor range check" && 0 <= v907 && v907 < 4l);
                v904[v910] = v913;
                v907 += 1l ;
            }
            v905 += 1l ;
        }
        float v914;
        v914 = 0.0f;
        int v915;
        v915 = 0l;
        while (while_method_3(v915)){
            int v917;
            v917 = 0l;
            while (while_method_1(v917)){
                assert("Tensor range check" && 0 <= v915 && v915 < 1l);
                assert("Tensor range check" && 0 <= v917 && v917 < 4l);
                int v919;
                v919 = 4l * v915;
                int v920;
                v920 = v919 + v917;
                float v921;
                v921 = v904[v920];
                float v922;
                v922 = v914 + v921;
                v914 = v922;
                v917 += 1l ;
            }
            v915 += 1l ;
        }
        auto v923 = cooperative_groups::coalesced_threads();
        int v924;
        v924 = threadIdx.x;
        int v925;
        v925 = v924 / 32l;
        auto v926 = cooperative_groups::labeled_partition(v923,v925);
        Closure0 v927{};
        float v928;
        v928 = cooperative_groups::reduce(v926, v914, v927);
        float v929;
        v929 = (float)v903;
        float v930;
        v930 = v928 / v929;
        float v931[4l];
        int v932;
        v932 = 0l;
        while (while_method_3(v932)){
            int v934;
            v934 = 0l;
            while (while_method_1(v934)){
                assert("Tensor range check" && 0 <= v932 && v932 < 1l);
                assert("Tensor range check" && 0 <= v934 && v934 < 4l);
                int v936;
                v936 = 4l * v932;
                int v937;
                v937 = v936 + v934;
                float v938;
                v938 = v828[v937];
                bool v939;
                v939 = v870[v937];
                float v940;
                if (v939){
                    v940 = v938;
                } else {
                    v940 = -1.0f / 0.0f;
                }
                float v941;
                v941 = v940 - v930;
                float v942;
                v942 = exp(v941);
                assert("Tensor range check" && 0 <= v932 && v932 < 1l);
                assert("Tensor range check" && 0 <= v934 && v934 < 4l);
                v931[v937] = v942;
                v934 += 1l ;
            }
            v932 += 1l ;
        }
        float v943;
        v943 = 0.0f;
        int v944;
        v944 = 0l;
        while (while_method_3(v944)){
            int v946;
            v946 = 0l;
            while (while_method_1(v946)){
                assert("Tensor range check" && 0 <= v944 && v944 < 1l);
                assert("Tensor range check" && 0 <= v946 && v946 < 4l);
                int v948;
                v948 = 4l * v944;
                int v949;
                v949 = v948 + v946;
                float v950;
                v950 = v931[v949];
                float v951;
                v951 = v943 + v950;
                v943 = v951;
                v946 += 1l ;
            }
            v944 += 1l ;
        }
        auto v952 = cooperative_groups::coalesced_threads();
        int v953;
        v953 = threadIdx.x;
        int v954;
        v954 = v953 / 32l;
        auto v955 = cooperative_groups::labeled_partition(v952,v954);
        float v956;
        v956 = cooperative_groups::reduce(v955, v943, v927);
        float v957[4l];
        int v958;
        v958 = 0l;
        while (while_method_3(v958)){
            int v960;
            v960 = 0l;
            while (while_method_1(v960)){
                assert("Tensor range check" && 0 <= v958 && v958 < 1l);
                assert("Tensor range check" && 0 <= v960 && v960 < 4l);
                int v962;
                v962 = 4l * v958;
                int v963;
                v963 = v962 + v960;
                float v964;
                v964 = v931[v963];
                float v965;
                v965 = v964 / v956;
                assert("Tensor range check" && 0 <= v958 && v958 < 1l);
                assert("Tensor range check" && 0 <= v960 && v960 < 4l);
                v957[v963] = v965;
                v960 += 1l ;
            }
            v958 += 1l ;
        }
        assert("Tensor range check" && 0 <= v818 && v818 < 64l);
        int v966;
        v966 = 0l;
        while (while_method_3(v966)){
            assert("Tensor range check" && 0 <= v966 && v966 < 1l);
            int v968;
            v968 = 128l * v966;
            int v969;
            v969 = v968 + v827;
            assert("Tensor range check" && 0 <= v966 && v966 < 1l);
            int v970;
            v970 = 4l * v966;
            int4* v971;
            v971 = reinterpret_cast<int4*>(v957 + v970);
            int4* v972;
            v972 = reinterpret_cast<int4*>(v4 + v969);
            assert("Pointer alignment check" && (unsigned long long)(v971) % 4l == 0 && (unsigned long long)(v972) % 4l == 0);
            *v972 = *v971;
            v966 += 1l ;
        }
        v818 += 1l ;
    }
    v17.sync() ;
    int v973;
    v973 = threadIdx.x;
    int v974;
    v974 = blockIdx.x;
    int v975;
    v975 = v974 * 32l;
    int v976;
    v976 = v973 + v975;
    unsigned long long v977;
    v977 = (unsigned long long)v976;
    curandStatePhilox4_32_10_t v978;
    curand_init(12344321ull,v977,0ull,&v978);
    int v979;
    v979 = threadIdx.x;
    bool v980;
    v980 = 0l <= v979;
    bool v981;
    v981 = v980 == false;
    if (v981){
        assert("The index needs to be zero or positive." && v980);
    } else {
    }
    int v983;
    v983 = v979 % 32l;
    int v984;
    v984 = v979 / 32l;
    bool v985;
    v985 = v984 < 1l;
    bool v986;
    v986 = v985 == false;
    if (v986){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v985);
    } else {
    }
    assert("Tensor range check" && 0 <= v984 && v984 < 1l);
    assert("Tensor range check" && 0 <= v983 && v983 < 32l);
    int v988;
    v988 = 4l * v983;
    int v989;
    v989 = 128l * v984;
    int v990;
    v990 = v989 + v988;
    assert("Tensor range check" && 0 <= v984 && v984 < 1l);
    assert("Tensor range check" && 0 <= v983 && v983 < 32l);
    assert("Tensor range check" && 0 <= v984 && v984 < 1l);
    int v991;
    v991 = blockIdx.x;
    int v992;
    v992 = v991;
    while (while_method_2(v992)){
        bool v994;
        v994 = 0l <= v992;
        bool v995;
        v995 = v994 == false;
        if (v995){
            assert("The index needs to be zero or positive." && v994);
        } else {
        }
        bool v997;
        v997 = v992 < 64l;
        bool v998;
        v998 = v997 == false;
        if (v998){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v997);
        } else {
        }
        assert("Tensor range check" && 0 <= v992 && v992 < 64l);
        int v1000;
        v1000 = 128l * v992;
        int v1001;
        v1001 = v1000 + v990;
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
                v1020 = 0l <= v983;
                bool v1022;
                if (v1020){
                    bool v1021;
                    v1021 = v983 < 32l;
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
                v1025 = v983 * 4l;
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
        v1036 = 0l <= v984;
        bool v1037;
        v1037 = v1036 && v985;
        bool v1038;
        v1038 = v1037 == false;
        if (v1038){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1037);
        } else {
        }
        bool v1040;
        v1040 = v994 && v997;
        bool v1041;
        v1041 = v1040 == false;
        if (v1041){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1040);
        } else {
        }
        int v1043;
        v1043 = v992 + v984;
        float v1044;
        v1044 = 0.0f;
        int v1045;
        v1045 = 0l;
        while (while_method_3(v1045)){
            int v1047;
            v1047 = 0l;
            while (while_method_1(v1047)){
                assert("Tensor range check" && 0 <= v1045 && v1045 < 1l);
                assert("Tensor range check" && 0 <= v1047 && v1047 < 4l);
                int v1049;
                v1049 = 4l * v1045;
                int v1050;
                v1050 = v1049 + v1047;
                float v1051;
                v1051 = v1002[v1050];
                float v1052;
                v1052 = v1044 + v1051;
                v1044 = v1052;
                v1047 += 1l ;
            }
            v1045 += 1l ;
        }
        auto v1053 = cooperative_groups::coalesced_threads();
        int v1054;
        v1054 = threadIdx.x;
        int v1055;
        v1055 = v1054 / 32l;
        auto v1056 = cooperative_groups::labeled_partition(v1053,v1055);
        Closure0 v1057{};
        float v1058;
        v1058 = cooperative_groups::reduce(v1056, v1044, v1057);
        float v1059;
        v1059 = v1058 / 128.0f;
        float v1060[4l];
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
                v1067 = v1002[v1066];
                float v1068;
                v1068 = v1067 - v1059;
                float v1069;
                v1069 = exp(v1068);
                assert("Tensor range check" && 0 <= v1061 && v1061 < 1l);
                assert("Tensor range check" && 0 <= v1063 && v1063 < 4l);
                v1060[v1066] = v1069;
                v1063 += 1l ;
            }
            v1061 += 1l ;
        }
        float v1070;
        v1070 = 0.0f;
        int v1071;
        v1071 = 0l;
        while (while_method_3(v1071)){
            int v1073;
            v1073 = 0l;
            while (while_method_1(v1073)){
                assert("Tensor range check" && 0 <= v1071 && v1071 < 1l);
                assert("Tensor range check" && 0 <= v1073 && v1073 < 4l);
                int v1075;
                v1075 = 4l * v1071;
                int v1076;
                v1076 = v1075 + v1073;
                float v1077;
                v1077 = v1060[v1076];
                float v1078;
                v1078 = v1070 + v1077;
                v1070 = v1078;
                v1073 += 1l ;
            }
            v1071 += 1l ;
        }
        auto v1079 = cooperative_groups::coalesced_threads();
        int v1080;
        v1080 = threadIdx.x;
        int v1081;
        v1081 = v1080 / 32l;
        auto v1082 = cooperative_groups::labeled_partition(v1079,v1081);
        float v1083;
        v1083 = cooperative_groups::reduce(v1082, v1070, v1057);
        float v1084[4l];
        int v1085;
        v1085 = 0l;
        while (while_method_3(v1085)){
            int v1087;
            v1087 = 0l;
            while (while_method_1(v1087)){
                assert("Tensor range check" && 0 <= v1085 && v1085 < 1l);
                assert("Tensor range check" && 0 <= v1087 && v1087 < 4l);
                int v1089;
                v1089 = 4l * v1085;
                int v1090;
                v1090 = v1089 + v1087;
                float v1091;
                v1091 = v1060[v1090];
                float v1092;
                v1092 = v1091 / v1083;
                assert("Tensor range check" && 0 <= v1085 && v1085 < 1l);
                assert("Tensor range check" && 0 <= v1087 && v1087 < 4l);
                v1084[v1090] = v1092;
                v1087 += 1l ;
            }
            v1085 += 1l ;
        }
        float v1093[4l];
        float v1094;
        v1094 = 0.0f;
        int v1095;
        v1095 = 0l;
        while (while_method_3(v1095)){
            assert("Tensor range check" && 0 <= v1095 && v1095 < 1l);
            int v1097;
            v1097 = 4l * v1095;
            assert("Tensor range check" && 0 <= v1095 && v1095 < 1l);
            int v1098; float v1099;
            Tuple0 tmp90 = Tuple0{0l, 0.0f};
            v1098 = tmp90.v0; v1099 = tmp90.v1;
            while (while_method_1(v1098)){
                assert("Tensor range check" && 0 <= v1098 && v1098 < 4l);
                int v1101;
                v1101 = v1098 + v1097;
                float v1102;
                v1102 = v1084[v1101];
                float v1103;
                v1103 = v1099 + v1102;
                v1099 = v1103;
                v1098 += 1l ;
            }
            auto v1104 = cooperative_groups::coalesced_threads();
            int v1105;
            v1105 = threadIdx.x;
            int v1106;
            v1106 = v1105 / 32l;
            auto v1107 = cooperative_groups::labeled_partition(v1104,v1106);
            Closure2 v1108{};
            float v1109;
            v1109 = cooperative_groups::inclusive_scan(v1107, v1099, v1108);
            float v1110;
            v1110 = v1107.shfl_up(v1109,1);
            bool v1111;
            v1111 = v1107.thread_rank() == 0;
            float v1112;
            if (v1111){
                v1112 = 0.0f;
            } else {
                v1112 = v1110;
            }
            float v1113;
            v1113 = v1107.shfl(v1109,v1107.num_threads()-1);
            float v1114;
            v1114 = v1094 + v1112;
            int v1115; float v1116;
            Tuple0 tmp91 = Tuple0{0l, v1114};
            v1115 = tmp91.v0; v1116 = tmp91.v1;
            while (while_method_1(v1115)){
                assert("Tensor range check" && 0 <= v1115 && v1115 < 4l);
                int v1118;
                v1118 = v1115 + v1097;
                float v1119;
                v1119 = v1084[v1118];
                float v1120;
                v1120 = v1116 + v1119;
                assert("Tensor range check" && 0 <= v1115 && v1115 < 4l);
                v1093[v1118] = v1120;
                v1116 = v1120;
                v1115 += 1l ;
            }
            float v1121;
            v1121 = v1094 + v1113;
            v1094 = v1121;
            v1095 += 1l ;
        }
        float v1122[4l];
        bool v1123[4l];
        int v1124;
        v1124 = 0l;
        while (while_method_3(v1124)){
            int v1126;
            v1126 = 0l;
            while (while_method_1(v1126)){
                assert("Tensor range check" && 0 <= v1124 && v1124 < 1l);
                assert("Tensor range check" && 0 <= v1126 && v1126 < 4l);
                int v1128;
                v1128 = 4l * v1124;
                int v1129;
                v1129 = v1128 + v1126;
                float v1130;
                v1130 = v1093[v1129];
                float v1131;
                v1131 = v1084[v1129];
                bool v1132;
                v1132 = v1131 > 0.0f;
                assert("Tensor range check" && 0 <= v1124 && v1124 < 1l);
                assert("Tensor range check" && 0 <= v1126 && v1126 < 4l);
                v1122[v1129] = v1130;
                v1123[v1129] = v1132;
                v1126 += 1l ;
            }
            v1124 += 1l ;
        }
        float v1133; bool v1134;
        Tuple3 tmp92 = Tuple3{-1.0f / 0.0f, false};
        v1133 = tmp92.v0; v1134 = tmp92.v1;
        int v1135;
        v1135 = 0l;
        while (while_method_3(v1135)){
            int v1137;
            v1137 = 0l;
            while (while_method_1(v1137)){
                assert("Tensor range check" && 0 <= v1135 && v1135 < 1l);
                assert("Tensor range check" && 0 <= v1137 && v1137 < 4l);
                int v1139;
                v1139 = 4l * v1135;
                int v1140;
                v1140 = v1139 + v1137;
                float v1141;
                v1141 = v1122[v1140];
                bool v1142;
                v1142 = v1123[v1140];
                float v1149; bool v1150;
                if (v1134){
                    if (v1142){
                        bool v1143;
                        v1143 = v1133 >= v1141;
                        float v1144;
                        if (v1143){
                            v1144 = v1133;
                        } else {
                            v1144 = v1141;
                        }
                        v1149 = v1144; v1150 = true;
                    } else {
                        v1149 = v1133; v1150 = v1134;
                    }
                } else {
                    if (v1142){
                        v1149 = v1141; v1150 = v1142;
                    } else {
                        v1149 = v1133; v1150 = v1134;
                    }
                }
                v1133 = v1149;
                v1134 = v1150;
                v1137 += 1l ;
            }
            v1135 += 1l ;
        }
        auto v1151 = cooperative_groups::coalesced_threads();
        int v1152;
        v1152 = threadIdx.x;
        int v1153;
        v1153 = v1152 / 32l;
        auto v1154 = cooperative_groups::labeled_partition(v1151,v1153);
        Closure5 v1155{};
        float v1156; bool v1157;
        Tuple3 tmp93 = cooperative_groups::reduce(v1154, Tuple3{v1133, v1134}, v1155);
        v1156 = tmp93.v0; v1157 = tmp93.v1;
        bool v1158;
        v1158 = v1157 == false;
        if (v1158){
            assert("The local reduce must be true." && v1157);
        } else {
        }
        float v1160[4l];
        int v1161[4l];
        int v1162;
        v1162 = 0l;
        while (while_method_3(v1162)){
            int v1164;
            v1164 = 0l;
            while (while_method_1(v1164)){
                assert("Tensor range check" && 0 <= v1162 && v1162 < 1l);
                assert("Tensor range check" && 0 <= v1164 && v1164 < 4l);
                int v1166;
                v1166 = 4l * v1162;
                int v1167;
                v1167 = v1166 + v1164;
                int v1168;
                v1168 = v1003[v1167];
                float v1169;
                v1169 = curand_uniform(&v978);
                assert("Tensor range check" && 0 <= v1162 && v1162 < 1l);
                assert("Tensor range check" && 0 <= v1164 && v1164 < 4l);
                v1160[v1167] = v1169;
                v1161[v1167] = v1168;
                v1164 += 1l ;
            }
            v1162 += 1l ;
        }
        float v1170; int v1171;
        Tuple1 tmp94 = Tuple1{0.0f, 2147483647l};
        v1170 = tmp94.v0; v1171 = tmp94.v1;
        int v1172;
        v1172 = 0l;
        while (while_method_3(v1172)){
            int v1174;
            v1174 = 0l;
            while (while_method_1(v1174)){
                assert("Tensor range check" && 0 <= v1172 && v1172 < 1l);
                assert("Tensor range check" && 0 <= v1174 && v1174 < 4l);
                int v1176;
                v1176 = 4l * v1172;
                int v1177;
                v1177 = v1176 + v1174;
                float v1178;
                v1178 = v1160[v1177];
                int v1179;
                v1179 = v1161[v1177];
                bool v1180;
                v1180 = v1171 < v1179;
                float v1181; int v1182;
                if (v1180){
                    v1181 = v1170; v1182 = v1171;
                } else {
                    v1181 = v1178; v1182 = v1179;
                }
                v1170 = v1181;
                v1171 = v1182;
                v1174 += 1l ;
            }
            v1172 += 1l ;
        }
        auto v1183 = cooperative_groups::coalesced_threads();
        int v1184;
        v1184 = threadIdx.x;
        int v1185;
        v1185 = v1184 / 32l;
        auto v1186 = cooperative_groups::labeled_partition(v1183,v1185);
        Closure6 v1187{};
        float v1188; int v1189;
        Tuple1 tmp95 = cooperative_groups::reduce(v1186, Tuple1{v1170, v1171}, v1187);
        v1188 = tmp95.v0; v1189 = tmp95.v1;
        float v1190;
        v1190 = v1156 * v1188;
        int v1191[4l];
        bool v1192[4l];
        int v1193;
        v1193 = 0l;
        while (while_method_3(v1193)){
            int v1195;
            v1195 = 0l;
            while (while_method_1(v1195)){
                assert("Tensor range check" && 0 <= v1193 && v1193 < 1l);
                assert("Tensor range check" && 0 <= v1195 && v1195 < 4l);
                int v1197;
                v1197 = 4l * v1193;
                int v1198;
                v1198 = v1197 + v1195;
                float v1199;
                v1199 = v1122[v1198];
                bool v1200;
                v1200 = v1123[v1198];
                int v1201;
                v1201 = v1003[v1198];
                int v1204; bool v1205;
                if (v1200){
                    float v1202;
                    v1202 = v1199 - v1190;
                    bool v1203;
                    v1203 = v1202 >= 0.0f;
                    v1204 = v1201; v1205 = v1203;
                } else {
                    v1204 = 2147483647l; v1205 = false;
                }
                assert("Tensor range check" && 0 <= v1193 && v1193 < 1l);
                assert("Tensor range check" && 0 <= v1195 && v1195 < 4l);
                v1191[v1198] = v1204;
                v1192[v1198] = v1205;
                v1195 += 1l ;
            }
            v1193 += 1l ;
        }
        int v1206; bool v1207;
        Tuple4 tmp96 = Tuple4{2147483647l, false};
        v1206 = tmp96.v0; v1207 = tmp96.v1;
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
                int v1214;
                v1214 = v1191[v1213];
                bool v1215;
                v1215 = v1192[v1213];
                int v1222; bool v1223;
                if (v1207){
                    if (v1215){
                        bool v1216;
                        v1216 = v1206 < v1214;
                        int v1217;
                        if (v1216){
                            v1217 = v1206;
                        } else {
                            v1217 = v1214;
                        }
                        v1222 = v1217; v1223 = true;
                    } else {
                        v1222 = v1206; v1223 = v1207;
                    }
                } else {
                    if (v1215){
                        v1222 = v1214; v1223 = v1215;
                    } else {
                        v1222 = v1206; v1223 = v1207;
                    }
                }
                v1206 = v1222;
                v1207 = v1223;
                v1210 += 1l ;
            }
            v1208 += 1l ;
        }
        auto v1224 = cooperative_groups::coalesced_threads();
        int v1225;
        v1225 = threadIdx.x;
        int v1226;
        v1226 = v1225 / 32l;
        auto v1227 = cooperative_groups::labeled_partition(v1224,v1226);
        Closure7 v1228{};
        int v1229; bool v1230;
        Tuple4 tmp97 = cooperative_groups::reduce(v1227, Tuple4{v1206, v1207}, v1228);
        v1229 = tmp97.v0; v1230 = tmp97.v1;
        bool v1231;
        v1231 = v1230 == false;
        if (v1231){
            assert("The local reduce must be true." && v1230);
        } else {
        }
        assert("Tensor range check" && 0 <= v992 && v992 < 64l);
        int v1233;
        v1233 = 0l;
        while (while_method_3(v1233)){
            assert("Tensor range check" && 0 <= v1233 && v1233 < 1l);
            int v1235;
            v1235 = 128l * v1233;
            int v1236;
            v1236 = v1235 + v1001;
            assert("Tensor range check" && 0 <= v1233 && v1233 < 1l);
            int v1237;
            v1237 = 4l * v1233;
            int4* v1238;
            v1238 = reinterpret_cast<int4*>(v1084 + v1237);
            int4* v1239;
            v1239 = reinterpret_cast<int4*>(v13 + v1236);
            assert("Pointer alignment check" && (unsigned long long)(v1238) % 4l == 0 && (unsigned long long)(v1239) % 4l == 0);
            *v1239 = *v1238;
            v1233 += 1l ;
        }
        assert("Tensor range check" && 0 <= v992 && v992 < 64l);
        v14[v1043] = v1229;
        v992 += 1l ;
    }
    v17.sync() ;
    int v1240;
    v1240 = threadIdx.x;
    int v1241;
    v1241 = blockIdx.x;
    int v1242;
    v1242 = v1241 * 32l;
    int v1243;
    v1243 = v1240 + v1242;
    unsigned long long v1244;
    v1244 = (unsigned long long)v1243;
    curandStatePhilox4_32_10_t v1245;
    curand_init(12344321ull,v1244,0ull,&v1245);
    int v1246;
    v1246 = threadIdx.x;
    bool v1247;
    v1247 = 0l <= v1246;
    bool v1248;
    v1248 = v1247 == false;
    if (v1248){
        assert("The index needs to be zero or positive." && v1247);
    } else {
    }
    int v1250;
    v1250 = v1246 % 32l;
    int v1251;
    v1251 = v1246 / 32l;
    bool v1252;
    v1252 = v1251 < 1l;
    bool v1253;
    v1253 = v1252 == false;
    if (v1253){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1252);
    } else {
    }
    assert("Tensor range check" && 0 <= v1251 && v1251 < 1l);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 32l);
    int v1255;
    v1255 = 4l * v1250;
    int v1256;
    v1256 = 128l * v1251;
    int v1257;
    v1257 = v1256 + v1255;
    assert("Tensor range check" && 0 <= v1251 && v1251 < 1l);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 32l);
    assert("Tensor range check" && 0 <= v1251 && v1251 < 1l);
    int v1258;
    v1258 = blockIdx.x;
    int v1259;
    v1259 = v1258;
    while (while_method_2(v1259)){
        bool v1261;
        v1261 = 0l <= v1259;
        bool v1262;
        v1262 = v1261 == false;
        if (v1262){
            assert("The index needs to be zero or positive." && v1261);
        } else {
        }
        bool v1264;
        v1264 = v1259 < 64l;
        bool v1265;
        v1265 = v1264 == false;
        if (v1265){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1264);
        } else {
        }
        assert("Tensor range check" && 0 <= v1259 && v1259 < 64l);
        int v1267;
        v1267 = 128l * v1259;
        int v1268;
        v1268 = v1267 + v1257;
        float v1269[4l];
        int v1270[4l];
        int v1271;
        v1271 = 0l;
        while (while_method_3(v1271)){
            assert("Tensor range check" && 0 <= v1271 && v1271 < 1l);
            int v1273;
            v1273 = 4l * v1271;
            assert("Tensor range check" && 0 <= v1271 && v1271 < 1l);
            int v1274;
            v1274 = 128l * v1271;
            int v1275;
            v1275 = v1274 + v1268;
            int4* v1276;
            v1276 = reinterpret_cast<int4*>(v1 + v1275);
            int4* v1277;
            v1277 = reinterpret_cast<int4*>(v1269 + v1273);
            assert("Pointer alignment check" && (unsigned long long)(v1276) % 4l == 0 && (unsigned long long)(v1277) % 4l == 0);
            *v1277 = *v1276;
            v1271 += 1l ;
        }
        int v1278;
        v1278 = 0l;
        while (while_method_3(v1278)){
            int v1280;
            v1280 = 0l;
            while (while_method_1(v1280)){
                bool v1282;
                v1282 = 0l <= v1280;
                bool v1284;
                if (v1282){
                    bool v1283;
                    v1283 = v1280 < 4l;
                    v1284 = v1283;
                } else {
                    v1284 = false;
                }
                bool v1285;
                v1285 = v1284 == false;
                if (v1285){
                    assert("The indices should be inside the range of the dimension." && v1284);
                } else {
                }
                bool v1287;
                v1287 = 0l <= v1250;
                bool v1289;
                if (v1287){
                    bool v1288;
                    v1288 = v1250 < 32l;
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
                v1292 = v1250 * 4l;
                int v1293;
                v1293 = v1280 + v1292;
                bool v1294;
                v1294 = 0l <= v1278;
                bool v1296;
                if (v1294){
                    bool v1295;
                    v1295 = v1278 < 1l;
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
                v1299 = v1278 * 128l;
                int v1300;
                v1300 = v1293 + v1299;
                assert("Tensor range check" && 0 <= v1278 && v1278 < 1l);
                assert("Tensor range check" && 0 <= v1280 && v1280 < 4l);
                int v1301;
                v1301 = 4l * v1278;
                int v1302;
                v1302 = v1301 + v1280;
                v1270[v1302] = v1300;
                v1280 += 1l ;
            }
            v1278 += 1l ;
        }
        bool v1303;
        v1303 = 0l <= v1251;
        bool v1304;
        v1304 = v1303 && v1252;
        bool v1305;
        v1305 = v1304 == false;
        if (v1305){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1304);
        } else {
        }
        bool v1307;
        v1307 = v1261 && v1264;
        bool v1308;
        v1308 = v1307 == false;
        if (v1308){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1307);
        } else {
        }
        int v1310;
        v1310 = v1259 + v1251;
        bool v1311[4l];
        int v1312;
        v1312 = 0l;
        while (while_method_3(v1312)){
            int v1314;
            v1314 = 0l;
            while (while_method_1(v1314)){
                assert("Tensor range check" && 0 <= v1312 && v1312 < 1l);
                assert("Tensor range check" && 0 <= v1314 && v1314 < 4l);
                int v1316;
                v1316 = 4l * v1312;
                int v1317;
                v1317 = v1316 + v1314;
                float v1318;
                v1318 = v1269[v1317];
                int v1319;
                v1319 = v1270[v1317];
                bool v1320;
                v1320 = v1319 < 11l;
                assert("Tensor range check" && 0 <= v1312 && v1312 < 1l);
                assert("Tensor range check" && 0 <= v1314 && v1314 < 4l);
                v1311[v1317] = v1320;
                v1314 += 1l ;
            }
            v1312 += 1l ;
        }
        int v1321[4l];
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
                bool v1328;
                v1328 = v1311[v1327];
                int v1329;
                if (v1328){
                    v1329 = 1l;
                } else {
                    v1329 = 0l;
                }
                assert("Tensor range check" && 0 <= v1322 && v1322 < 1l);
                assert("Tensor range check" && 0 <= v1324 && v1324 < 4l);
                v1321[v1327] = v1329;
                v1324 += 1l ;
            }
            v1322 += 1l ;
        }
        int v1330;
        v1330 = 0l;
        int v1331;
        v1331 = 0l;
        while (while_method_3(v1331)){
            int v1333;
            v1333 = 0l;
            while (while_method_1(v1333)){
                assert("Tensor range check" && 0 <= v1331 && v1331 < 1l);
                assert("Tensor range check" && 0 <= v1333 && v1333 < 4l);
                int v1335;
                v1335 = 4l * v1331;
                int v1336;
                v1336 = v1335 + v1333;
                int v1337;
                v1337 = v1321[v1336];
                int v1338;
                v1338 = v1330 + v1337;
                v1330 = v1338;
                v1333 += 1l ;
            }
            v1331 += 1l ;
        }
        auto v1339 = cooperative_groups::coalesced_threads();
        int v1340;
        v1340 = threadIdx.x;
        int v1341;
        v1341 = v1340 / 32l;
        auto v1342 = cooperative_groups::labeled_partition(v1339,v1341);
        Closure4 v1343{};
        int v1344;
        v1344 = cooperative_groups::reduce(v1342, v1330, v1343);
        float v1345[4l];
        int v1346;
        v1346 = 0l;
        while (while_method_3(v1346)){
            int v1348;
            v1348 = 0l;
            while (while_method_1(v1348)){
                assert("Tensor range check" && 0 <= v1346 && v1346 < 1l);
                assert("Tensor range check" && 0 <= v1348 && v1348 < 4l);
                int v1350;
                v1350 = 4l * v1346;
                int v1351;
                v1351 = v1350 + v1348;
                float v1352;
                v1352 = v1269[v1351];
                bool v1353;
                v1353 = v1311[v1351];
                float v1354;
                if (v1353){
                    v1354 = v1352;
                } else {
                    v1354 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1346 && v1346 < 1l);
                assert("Tensor range check" && 0 <= v1348 && v1348 < 4l);
                v1345[v1351] = v1354;
                v1348 += 1l ;
            }
            v1346 += 1l ;
        }
        float v1355;
        v1355 = 0.0f;
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
                v1362 = v1345[v1361];
                float v1363;
                v1363 = v1355 + v1362;
                v1355 = v1363;
                v1358 += 1l ;
            }
            v1356 += 1l ;
        }
        auto v1364 = cooperative_groups::coalesced_threads();
        int v1365;
        v1365 = threadIdx.x;
        int v1366;
        v1366 = v1365 / 32l;
        auto v1367 = cooperative_groups::labeled_partition(v1364,v1366);
        Closure0 v1368{};
        float v1369;
        v1369 = cooperative_groups::reduce(v1367, v1355, v1368);
        float v1370;
        v1370 = (float)v1344;
        float v1371;
        v1371 = v1369 / v1370;
        float v1372[4l];
        int v1373;
        v1373 = 0l;
        while (while_method_3(v1373)){
            int v1375;
            v1375 = 0l;
            while (while_method_1(v1375)){
                assert("Tensor range check" && 0 <= v1373 && v1373 < 1l);
                assert("Tensor range check" && 0 <= v1375 && v1375 < 4l);
                int v1377;
                v1377 = 4l * v1373;
                int v1378;
                v1378 = v1377 + v1375;
                float v1379;
                v1379 = v1269[v1378];
                bool v1380;
                v1380 = v1311[v1378];
                float v1381;
                if (v1380){
                    v1381 = v1379;
                } else {
                    v1381 = -1.0f / 0.0f;
                }
                float v1382;
                v1382 = v1381 - v1371;
                float v1383;
                v1383 = exp(v1382);
                assert("Tensor range check" && 0 <= v1373 && v1373 < 1l);
                assert("Tensor range check" && 0 <= v1375 && v1375 < 4l);
                v1372[v1378] = v1383;
                v1375 += 1l ;
            }
            v1373 += 1l ;
        }
        float v1384;
        v1384 = 0.0f;
        int v1385;
        v1385 = 0l;
        while (while_method_3(v1385)){
            int v1387;
            v1387 = 0l;
            while (while_method_1(v1387)){
                assert("Tensor range check" && 0 <= v1385 && v1385 < 1l);
                assert("Tensor range check" && 0 <= v1387 && v1387 < 4l);
                int v1389;
                v1389 = 4l * v1385;
                int v1390;
                v1390 = v1389 + v1387;
                float v1391;
                v1391 = v1372[v1390];
                float v1392;
                v1392 = v1384 + v1391;
                v1384 = v1392;
                v1387 += 1l ;
            }
            v1385 += 1l ;
        }
        auto v1393 = cooperative_groups::coalesced_threads();
        int v1394;
        v1394 = threadIdx.x;
        int v1395;
        v1395 = v1394 / 32l;
        auto v1396 = cooperative_groups::labeled_partition(v1393,v1395);
        float v1397;
        v1397 = cooperative_groups::reduce(v1396, v1384, v1368);
        float v1398[4l];
        int v1399;
        v1399 = 0l;
        while (while_method_3(v1399)){
            int v1401;
            v1401 = 0l;
            while (while_method_1(v1401)){
                assert("Tensor range check" && 0 <= v1399 && v1399 < 1l);
                assert("Tensor range check" && 0 <= v1401 && v1401 < 4l);
                int v1403;
                v1403 = 4l * v1399;
                int v1404;
                v1404 = v1403 + v1401;
                float v1405;
                v1405 = v1372[v1404];
                float v1406;
                v1406 = v1405 / v1397;
                assert("Tensor range check" && 0 <= v1399 && v1399 < 1l);
                assert("Tensor range check" && 0 <= v1401 && v1401 < 4l);
                v1398[v1404] = v1406;
                v1401 += 1l ;
            }
            v1399 += 1l ;
        }
        float v1407[4l];
        float v1408;
        v1408 = 0.0f;
        int v1409;
        v1409 = 0l;
        while (while_method_3(v1409)){
            assert("Tensor range check" && 0 <= v1409 && v1409 < 1l);
            int v1411;
            v1411 = 4l * v1409;
            assert("Tensor range check" && 0 <= v1409 && v1409 < 1l);
            int v1412; float v1413;
            Tuple0 tmp98 = Tuple0{0l, 0.0f};
            v1412 = tmp98.v0; v1413 = tmp98.v1;
            while (while_method_1(v1412)){
                assert("Tensor range check" && 0 <= v1412 && v1412 < 4l);
                int v1415;
                v1415 = v1412 + v1411;
                float v1416;
                v1416 = v1398[v1415];
                float v1417;
                v1417 = v1413 + v1416;
                v1413 = v1417;
                v1412 += 1l ;
            }
            auto v1418 = cooperative_groups::coalesced_threads();
            int v1419;
            v1419 = threadIdx.x;
            int v1420;
            v1420 = v1419 / 32l;
            auto v1421 = cooperative_groups::labeled_partition(v1418,v1420);
            Closure2 v1422{};
            float v1423;
            v1423 = cooperative_groups::inclusive_scan(v1421, v1413, v1422);
            float v1424;
            v1424 = v1421.shfl_up(v1423,1);
            bool v1425;
            v1425 = v1421.thread_rank() == 0;
            float v1426;
            if (v1425){
                v1426 = 0.0f;
            } else {
                v1426 = v1424;
            }
            float v1427;
            v1427 = v1421.shfl(v1423,v1421.num_threads()-1);
            float v1428;
            v1428 = v1408 + v1426;
            int v1429; float v1430;
            Tuple0 tmp99 = Tuple0{0l, v1428};
            v1429 = tmp99.v0; v1430 = tmp99.v1;
            while (while_method_1(v1429)){
                assert("Tensor range check" && 0 <= v1429 && v1429 < 4l);
                int v1432;
                v1432 = v1429 + v1411;
                float v1433;
                v1433 = v1398[v1432];
                float v1434;
                v1434 = v1430 + v1433;
                assert("Tensor range check" && 0 <= v1429 && v1429 < 4l);
                v1407[v1432] = v1434;
                v1430 = v1434;
                v1429 += 1l ;
            }
            float v1435;
            v1435 = v1408 + v1427;
            v1408 = v1435;
            v1409 += 1l ;
        }
        float v1436[4l];
        bool v1437[4l];
        int v1438;
        v1438 = 0l;
        while (while_method_3(v1438)){
            int v1440;
            v1440 = 0l;
            while (while_method_1(v1440)){
                assert("Tensor range check" && 0 <= v1438 && v1438 < 1l);
                assert("Tensor range check" && 0 <= v1440 && v1440 < 4l);
                int v1442;
                v1442 = 4l * v1438;
                int v1443;
                v1443 = v1442 + v1440;
                float v1444;
                v1444 = v1407[v1443];
                float v1445;
                v1445 = v1398[v1443];
                bool v1446;
                v1446 = v1445 > 0.0f;
                assert("Tensor range check" && 0 <= v1438 && v1438 < 1l);
                assert("Tensor range check" && 0 <= v1440 && v1440 < 4l);
                v1436[v1443] = v1444;
                v1437[v1443] = v1446;
                v1440 += 1l ;
            }
            v1438 += 1l ;
        }
        float v1447; bool v1448;
        Tuple3 tmp100 = Tuple3{-1.0f / 0.0f, false};
        v1447 = tmp100.v0; v1448 = tmp100.v1;
        int v1449;
        v1449 = 0l;
        while (while_method_3(v1449)){
            int v1451;
            v1451 = 0l;
            while (while_method_1(v1451)){
                assert("Tensor range check" && 0 <= v1449 && v1449 < 1l);
                assert("Tensor range check" && 0 <= v1451 && v1451 < 4l);
                int v1453;
                v1453 = 4l * v1449;
                int v1454;
                v1454 = v1453 + v1451;
                float v1455;
                v1455 = v1436[v1454];
                bool v1456;
                v1456 = v1437[v1454];
                float v1463; bool v1464;
                if (v1448){
                    if (v1456){
                        bool v1457;
                        v1457 = v1447 >= v1455;
                        float v1458;
                        if (v1457){
                            v1458 = v1447;
                        } else {
                            v1458 = v1455;
                        }
                        v1463 = v1458; v1464 = true;
                    } else {
                        v1463 = v1447; v1464 = v1448;
                    }
                } else {
                    if (v1456){
                        v1463 = v1455; v1464 = v1456;
                    } else {
                        v1463 = v1447; v1464 = v1448;
                    }
                }
                v1447 = v1463;
                v1448 = v1464;
                v1451 += 1l ;
            }
            v1449 += 1l ;
        }
        auto v1465 = cooperative_groups::coalesced_threads();
        int v1466;
        v1466 = threadIdx.x;
        int v1467;
        v1467 = v1466 / 32l;
        auto v1468 = cooperative_groups::labeled_partition(v1465,v1467);
        Closure5 v1469{};
        float v1470; bool v1471;
        Tuple3 tmp101 = cooperative_groups::reduce(v1468, Tuple3{v1447, v1448}, v1469);
        v1470 = tmp101.v0; v1471 = tmp101.v1;
        bool v1472;
        v1472 = v1471 == false;
        if (v1472){
            assert("The local reduce must be true." && v1471);
        } else {
        }
        float v1474[4l];
        int v1475[4l];
        int v1476;
        v1476 = 0l;
        while (while_method_3(v1476)){
            int v1478;
            v1478 = 0l;
            while (while_method_1(v1478)){
                assert("Tensor range check" && 0 <= v1476 && v1476 < 1l);
                assert("Tensor range check" && 0 <= v1478 && v1478 < 4l);
                int v1480;
                v1480 = 4l * v1476;
                int v1481;
                v1481 = v1480 + v1478;
                int v1482;
                v1482 = v1270[v1481];
                float v1483;
                v1483 = curand_uniform(&v1245);
                assert("Tensor range check" && 0 <= v1476 && v1476 < 1l);
                assert("Tensor range check" && 0 <= v1478 && v1478 < 4l);
                v1474[v1481] = v1483;
                v1475[v1481] = v1482;
                v1478 += 1l ;
            }
            v1476 += 1l ;
        }
        float v1484; int v1485;
        Tuple1 tmp102 = Tuple1{0.0f, 2147483647l};
        v1484 = tmp102.v0; v1485 = tmp102.v1;
        int v1486;
        v1486 = 0l;
        while (while_method_3(v1486)){
            int v1488;
            v1488 = 0l;
            while (while_method_1(v1488)){
                assert("Tensor range check" && 0 <= v1486 && v1486 < 1l);
                assert("Tensor range check" && 0 <= v1488 && v1488 < 4l);
                int v1490;
                v1490 = 4l * v1486;
                int v1491;
                v1491 = v1490 + v1488;
                float v1492;
                v1492 = v1474[v1491];
                int v1493;
                v1493 = v1475[v1491];
                bool v1494;
                v1494 = v1485 < v1493;
                float v1495; int v1496;
                if (v1494){
                    v1495 = v1484; v1496 = v1485;
                } else {
                    v1495 = v1492; v1496 = v1493;
                }
                v1484 = v1495;
                v1485 = v1496;
                v1488 += 1l ;
            }
            v1486 += 1l ;
        }
        auto v1497 = cooperative_groups::coalesced_threads();
        int v1498;
        v1498 = threadIdx.x;
        int v1499;
        v1499 = v1498 / 32l;
        auto v1500 = cooperative_groups::labeled_partition(v1497,v1499);
        Closure6 v1501{};
        float v1502; int v1503;
        Tuple1 tmp103 = cooperative_groups::reduce(v1500, Tuple1{v1484, v1485}, v1501);
        v1502 = tmp103.v0; v1503 = tmp103.v1;
        float v1504;
        v1504 = v1470 * v1502;
        int v1505[4l];
        bool v1506[4l];
        int v1507;
        v1507 = 0l;
        while (while_method_3(v1507)){
            int v1509;
            v1509 = 0l;
            while (while_method_1(v1509)){
                assert("Tensor range check" && 0 <= v1507 && v1507 < 1l);
                assert("Tensor range check" && 0 <= v1509 && v1509 < 4l);
                int v1511;
                v1511 = 4l * v1507;
                int v1512;
                v1512 = v1511 + v1509;
                float v1513;
                v1513 = v1436[v1512];
                bool v1514;
                v1514 = v1437[v1512];
                int v1515;
                v1515 = v1270[v1512];
                int v1518; bool v1519;
                if (v1514){
                    float v1516;
                    v1516 = v1513 - v1504;
                    bool v1517;
                    v1517 = v1516 >= 0.0f;
                    v1518 = v1515; v1519 = v1517;
                } else {
                    v1518 = 2147483647l; v1519 = false;
                }
                assert("Tensor range check" && 0 <= v1507 && v1507 < 1l);
                assert("Tensor range check" && 0 <= v1509 && v1509 < 4l);
                v1505[v1512] = v1518;
                v1506[v1512] = v1519;
                v1509 += 1l ;
            }
            v1507 += 1l ;
        }
        int v1520; bool v1521;
        Tuple4 tmp104 = Tuple4{2147483647l, false};
        v1520 = tmp104.v0; v1521 = tmp104.v1;
        int v1522;
        v1522 = 0l;
        while (while_method_3(v1522)){
            int v1524;
            v1524 = 0l;
            while (while_method_1(v1524)){
                assert("Tensor range check" && 0 <= v1522 && v1522 < 1l);
                assert("Tensor range check" && 0 <= v1524 && v1524 < 4l);
                int v1526;
                v1526 = 4l * v1522;
                int v1527;
                v1527 = v1526 + v1524;
                int v1528;
                v1528 = v1505[v1527];
                bool v1529;
                v1529 = v1506[v1527];
                int v1536; bool v1537;
                if (v1521){
                    if (v1529){
                        bool v1530;
                        v1530 = v1520 < v1528;
                        int v1531;
                        if (v1530){
                            v1531 = v1520;
                        } else {
                            v1531 = v1528;
                        }
                        v1536 = v1531; v1537 = true;
                    } else {
                        v1536 = v1520; v1537 = v1521;
                    }
                } else {
                    if (v1529){
                        v1536 = v1528; v1537 = v1529;
                    } else {
                        v1536 = v1520; v1537 = v1521;
                    }
                }
                v1520 = v1536;
                v1521 = v1537;
                v1524 += 1l ;
            }
            v1522 += 1l ;
        }
        auto v1538 = cooperative_groups::coalesced_threads();
        int v1539;
        v1539 = threadIdx.x;
        int v1540;
        v1540 = v1539 / 32l;
        auto v1541 = cooperative_groups::labeled_partition(v1538,v1540);
        Closure7 v1542{};
        int v1543; bool v1544;
        Tuple4 tmp105 = cooperative_groups::reduce(v1541, Tuple4{v1520, v1521}, v1542);
        v1543 = tmp105.v0; v1544 = tmp105.v1;
        bool v1545;
        v1545 = v1544 == false;
        if (v1545){
            assert("The local reduce must be true." && v1544);
        } else {
        }
        assert("Tensor range check" && 0 <= v1259 && v1259 < 64l);
        int v1547;
        v1547 = 0l;
        while (while_method_3(v1547)){
            assert("Tensor range check" && 0 <= v1547 && v1547 < 1l);
            int v1549;
            v1549 = 128l * v1547;
            int v1550;
            v1550 = v1549 + v1268;
            assert("Tensor range check" && 0 <= v1547 && v1547 < 1l);
            int v1551;
            v1551 = 4l * v1547;
            int4* v1552;
            v1552 = reinterpret_cast<int4*>(v1398 + v1551);
            int4* v1553;
            v1553 = reinterpret_cast<int4*>(v15 + v1550);
            assert("Pointer alignment check" && (unsigned long long)(v1552) % 4l == 0 && (unsigned long long)(v1553) % 4l == 0);
            *v1553 = *v1552;
            v1547 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1259 && v1259 < 64l);
        v16[v1310] = v1543;
        v1259 += 1l ;
    }
    v17.sync() ;
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
def method18(v0 : cp.ndarray) -> None:
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
def method19(v0 : cp.ndarray) -> None:
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
def method20(v0 : cp.ndarray) -> None:
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
def method21(v0 : cp.ndarray) -> None:
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
def method22(v0 : cp.ndarray) -> None:
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
def method23(v0 : cp.ndarray) -> None:
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
def method24(v0 : cp.ndarray) -> None:
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
def method25(v0 : cp.ndarray) -> None:
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
def method26(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
def method27(v0 : cp.ndarray) -> None:
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
def method28(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
def method29(v0 : cp.ndarray) -> None:
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
def method30(v0 : cp.ndarray) -> None:
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
def method31(v0 : cp.ndarray) -> None:
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
def method32(v0 : cp.ndarray) -> None:
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
def method36(v0 : i32) -> bool:
    v1 = v0 < 32
    del v0
    return v1
def method37(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def method35(v0 : cp.ndarray) -> None:
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
    while method36(v33):
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
        while method37(v41):
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
def method38(v0 : cp.ndarray) -> None:
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
    while method36(v22):
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
def method39(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method36(v35):
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
        while method37(v43):
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
def method40(v0 : cp.ndarray) -> None:
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
    while method36(v22):
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
def method41(v0 : cp.ndarray) -> None:
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
    while method36(v33):
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
        while method37(v41):
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
def method42(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method36(v35):
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
        while method37(v43):
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
def method44(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method43(v0 : cp.ndarray) -> None:
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
    while method36(v33):
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
        while method44(v41):
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
def method45(v0 : cp.ndarray) -> None:
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
    while method36(v22):
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
def method46(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method36(v35):
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
        while method44(v43):
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
def method47(v0 : cp.ndarray) -> None:
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
    while method36(v22):
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
def method48(v0 : cp.ndarray) -> None:
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
    while method36(v33):
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
        while method44(v41):
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
def method49(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
    while method36(v35):
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
        while method44(v43):
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
def method50(v0 : cp.ndarray) -> None:
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
def method51(v0 : cp.ndarray) -> None:
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
def method52(v0 : cp.ndarray) -> None:
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
def method53(v0 : cp.ndarray) -> None:
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
def method54(v0 : cp.ndarray) -> None:
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
def method55(v0 : cp.ndarray) -> None:
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
def method56(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
def method57(v0 : cp.ndarray) -> None:
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
def method58(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
def method59(v0 : cp.ndarray) -> None:
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
def method60(v0 : cp.ndarray) -> None:
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
def method61(v0 : cp.ndarray) -> None:
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
def method62(v0 : cp.ndarray) -> None:
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
def method63(v0 : cp.ndarray) -> None:
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
def method64(v0 : cp.ndarray) -> None:
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
def method65(v0 : cp.ndarray) -> None:
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
def method66(v0 : cp.ndarray) -> None:
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
def method67(v0 : cp.ndarray) -> None:
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
def method68(v0 : cp.ndarray) -> None:
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
def method69(v0 : cp.ndarray) -> None:
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
def method70(v0 : cp.ndarray) -> None:
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
def method71(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
def method72(v0 : cp.ndarray) -> None:
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
def method73(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
def method74(v0 : cp.ndarray) -> None:
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
def method75(v0 : cp.ndarray) -> None:
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
def method76(v0 : cp.ndarray) -> None:
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
def method77(v0 : cp.ndarray) -> None:
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
def method78(v0 : cp.ndarray) -> None:
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
def method79(v0 : cp.ndarray) -> None:
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
    v20 = cp.empty(8192,dtype=cp.float32)
    v21 = cp.empty(64,dtype=cp.int32)
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
    v27.max_dynamic_shared_size_bytes = 81920 
    v27((1,),(32,),(v0, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21),shared_mem=81920)
    del v27
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
    method17(v20)
    del v20
    method18(v21)
    del v21
    cp.random.seed(12344321)
    v28 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v29 = v28.size
    v30 = 8192 == v29
    del v29
    v31 = v30 == False
    if v31:
        v32 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v30, v32
        del v32
    else:
        pass
    del v30, v31
    v33 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v34 = cp.empty(1,dtype=cp.float32)
    v35 = cp.empty(8192,dtype=cp.int32)
    v36 = cp.empty(8192,dtype=cp.float32)
    v37 = cp.empty(8192,dtype=cp.float32)
    v38 = cp.empty(8192,dtype=cp.float32)
    v39 = cp.empty(8192,dtype=cp.float32)
    v40 = cp.empty(8192,dtype=cp.float32)
    v41 = cp.empty(128,dtype=cp.int32)
    v42 = cp.empty(8192,dtype=cp.int32)
    v43 = cp.empty(8192,dtype=cp.int32)
    v44 = cp.empty(128,dtype=cp.int32)
    v45 = cp.empty(8192,dtype=cp.int32)
    v46 = cp.empty(8192,dtype=cp.float32)
    v47 = cp.empty(128,dtype=cp.int32)
    v48 = cp.empty(8192,dtype=cp.float32)
    v49 = cp.empty(128,dtype=cp.int32)
    v50 = cp.cuda.Device().attributes['MultiProcessorCount']
    v51 = v50 == 24
    del v50
    v52 = v51 == False
    if v52:
        v53 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v51, v53
        del v53
    else:
        pass
    del v51, v52
    v54 = 1
    v55 = raw_module.get_function(f"entry{v54}")
    del v54
    v55.max_dynamic_shared_size_bytes = 81920 
    v55((1,),(32,),(v28, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49),shared_mem=81920)
    del v55
    method19(v33)
    del v33
    method20(v28)
    del v28
    method21(v34)
    del v34
    method22(v36)
    del v36
    method23(v37)
    del v37
    method24(v40)
    del v40
    method25(v41)
    del v41
    method26(v38, v39)
    del v38, v39
    method27(v35)
    del v35
    method28(v42, v43)
    del v42, v43
    method29(v44)
    del v44
    method30(v45)
    del v45
    method31(v46)
    del v46
    method32(v47)
    del v47
    method33(v48)
    del v48
    method34(v49)
    del v49
    cp.random.seed(12344321)
    v56 = cp.arange(0,512,1,dtype=cp.float32) # type: ignore
    v57 = v56.size
    v58 = 512 == v57
    del v57
    v59 = v58 == False
    if v59:
        v60 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v58, v60
        del v60
    else:
        pass
    del v58, v59
    v61 = cp.random.normal(0.0,1.0,512,dtype=cp.float32) # type: ignore
    v62 = cp.empty(512,dtype=cp.int32)
    v63 = cp.empty(512,dtype=cp.int32)
    v64 = cp.empty(32,dtype=cp.int32)
    v65 = cp.empty(32,dtype=cp.int32)
    v66 = cp.empty(512,dtype=cp.float32)
    v67 = cp.empty(512,dtype=cp.float32)
    v68 = cp.cuda.Device().attributes['MultiProcessorCount']
    v69 = v68 == 24
    del v68
    v70 = v69 == False
    if v70:
        v71 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v69, v71
        del v71
    else:
        pass
    del v69, v70
    v72 = 2
    v73 = raw_module.get_function(f"entry{v72}")
    del v72
    v73.max_dynamic_shared_size_bytes = 81920 
    v73((1,),(32,),(v56, v61, v62, v63, v64, v65, v66, v67),shared_mem=81920)
    del v73
    method35(v56)
    del v56
    method38(v65)
    del v65
    method39(v62, v63)
    del v62, v63
    method40(v64)
    del v64
    method41(v67)
    del v67
    method42(v61, v66)
    del v61, v66
    cp.random.seed(12344321)
    v74 = cp.arange(0,8192,1,dtype=cp.float32) # type: ignore
    v75 = v74.size
    v76 = 8192 == v75
    del v75
    v77 = v76 == False
    if v77:
        v78 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v76, v78
        del v78
    else:
        pass
    del v76, v77
    v79 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v80 = cp.empty(8192,dtype=cp.int32)
    v81 = cp.empty(8192,dtype=cp.int32)
    v82 = cp.empty(32,dtype=cp.int32)
    v83 = cp.empty(32,dtype=cp.int32)
    v84 = cp.empty(8192,dtype=cp.float32)
    v85 = cp.empty(8192,dtype=cp.float32)
    v86 = cp.cuda.Device().attributes['MultiProcessorCount']
    v87 = v86 == 24
    del v86
    v88 = v87 == False
    if v88:
        v89 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v87, v89
        del v89
    else:
        pass
    del v87, v88
    v90 = 3
    v91 = raw_module.get_function(f"entry{v90}")
    del v90
    v91.max_dynamic_shared_size_bytes = 81920 
    v91((1,),(32,),(v74, v79, v80, v81, v82, v83, v84, v85),shared_mem=81920)
    del v91
    method43(v74)
    del v74
    method45(v83)
    del v83
    method46(v80, v81)
    del v80, v81
    method47(v82)
    del v82
    method48(v85)
    del v85
    method49(v79, v84)
    del v79, v84
    cp.random.seed(12344321)
    v92 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v93 = v92.size
    v94 = 8192 == v93
    del v93
    v95 = v94 == False
    if v95:
        v96 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v94, v96
        del v96
    else:
        pass
    del v94, v95
    v97 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v98 = cp.empty(8192,dtype=cp.int32)
    v99 = cp.empty(8192,dtype=cp.float32)
    v100 = cp.empty(8192,dtype=cp.float32)
    v101 = cp.empty(8192,dtype=cp.float32)
    v102 = cp.empty(8192,dtype=cp.float32)
    v103 = cp.empty(8192,dtype=cp.float32)
    v104 = cp.empty(128,dtype=cp.int32)
    v105 = cp.empty(8192,dtype=cp.int32)
    v106 = cp.empty(8192,dtype=cp.int32)
    v107 = cp.empty(128,dtype=cp.int32)
    v108 = cp.empty(8192,dtype=cp.int32)
    v109 = cp.empty(8192,dtype=cp.float32)
    v110 = cp.empty(128,dtype=cp.int32)
    v111 = cp.empty(8192,dtype=cp.float32)
    v112 = cp.empty(128,dtype=cp.int32)
    v113 = cp.cuda.Device().attributes['MultiProcessorCount']
    v114 = v113 == 24
    del v113
    v115 = v114 == False
    if v115:
        v116 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v114, v116
        del v116
    else:
        pass
    del v114, v115
    v117 = 4
    v118 = raw_module.get_function(f"entry{v117}")
    del v117
    v118.max_dynamic_shared_size_bytes = 81920 
    v118((1,),(32,),(v92, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107, v108, v109, v110, v111, v112),shared_mem=81920)
    del v118
    method50(v97)
    del v97
    method51(v92)
    del v92
    method52(v99)
    del v99
    method53(v100)
    del v100
    method54(v103)
    del v103
    method55(v104)
    del v104
    method56(v101, v102)
    del v101, v102
    method57(v98)
    del v98
    method58(v105, v106)
    del v105, v106
    method59(v107)
    del v107
    method60(v108)
    del v108
    method61(v109)
    del v109
    method62(v110)
    del v110
    method63(v111)
    del v111
    method64(v112)
    del v112
    cp.random.seed(12344321)
    v119 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v120 = v119.size
    v121 = 8192 == v120
    del v120
    v122 = v121 == False
    if v122:
        v123 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v121, v123
        del v123
    else:
        pass
    del v121, v122
    v124 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v125 = cp.empty(8192,dtype=cp.int32)
    v126 = cp.empty(8192,dtype=cp.float32)
    v127 = cp.empty(8192,dtype=cp.float32)
    v128 = cp.empty(8192,dtype=cp.float32)
    v129 = cp.empty(8192,dtype=cp.float32)
    v130 = cp.empty(8192,dtype=cp.float32)
    v131 = cp.empty(64,dtype=cp.int32)
    v132 = cp.empty(8192,dtype=cp.int32)
    v133 = cp.empty(8192,dtype=cp.int32)
    v134 = cp.empty(64,dtype=cp.int32)
    v135 = cp.empty(8192,dtype=cp.int32)
    v136 = cp.empty(8192,dtype=cp.float32)
    v137 = cp.empty(64,dtype=cp.int32)
    v138 = cp.empty(8192,dtype=cp.float32)
    v139 = cp.empty(64,dtype=cp.int32)
    v140 = cp.cuda.Device().attributes['MultiProcessorCount']
    v141 = v140 == 24
    del v140
    v142 = v141 == False
    if v142:
        v143 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v141, v143
        del v143
    else:
        pass
    del v141, v142
    v144 = 5
    v145 = raw_module.get_function(f"entry{v144}")
    del v144
    v145.max_dynamic_shared_size_bytes = 81920 
    v145((1,),(32,),(v119, v124, v125, v126, v127, v128, v129, v130, v131, v132, v133, v134, v135, v136, v137, v138, v139),shared_mem=81920)
    del v145
    method65(v124)
    del v124
    method66(v119)
    del v119
    method67(v126)
    del v126
    method68(v127)
    del v127
    method69(v130)
    del v130
    method70(v131)
    del v131
    method71(v128, v129)
    del v128, v129
    method72(v125)
    del v125
    method73(v132, v133)
    del v132, v133
    method74(v134)
    del v134
    method75(v135)
    del v135
    method76(v136)
    del v136
    method77(v137)
    del v137
    method78(v138)
    del v138
    return method79(v139)

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
