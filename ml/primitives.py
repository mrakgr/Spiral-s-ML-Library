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
    float v16;
    v16 = 0.0f;
    int v17;
    v17 = threadIdx.x;
    int v18;
    v18 = v17;
    while (while_method_0(v18)){
        bool v20;
        v20 = 0l <= v18;
        bool v21;
        v21 = v20 == false;
        if (v21){
            assert("The index needs to be zero or positive." && v20);
        } else {
        }
        int v23;
        v23 = v18 % 32l;
        int v24;
        v24 = v18 / 32l;
        bool v25;
        v25 = v24 < 64l;
        bool v26;
        v26 = v25 == false;
        if (v26){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v25);
        } else {
        }
        assert("Tensor range check" && 0 <= v24 && v24 < 64l);
        assert("Tensor range check" && 0 <= v23 && v23 < 32l);
        int v28;
        v28 = 4l * v23;
        int v29;
        v29 = 128l * v24;
        int v30;
        v30 = v29 + v28;
        float v31[4l];
        int4* v32;
        v32 = reinterpret_cast<int4*>(v1 + v30);
        int4* v33;
        v33 = reinterpret_cast<int4*>(v31 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v32) % 4l == 0 && (unsigned long long)(v33) % 4l == 0);
        *v33 = *v32;
        int v34; float v35;
        Tuple0 tmp0 = Tuple0{0l, v16};
        v34 = tmp0.v0; v35 = tmp0.v1;
        while (while_method_1(v34)){
            assert("Tensor range check" && 0 <= v34 && v34 < 4l);
            float v37;
            v37 = v31[v34];
            float v38;
            v38 = v35 + v37;
            v35 = v38;
            v34 += 1l ;
        }
        v16 = v35;
        v18 += 32l ;
    }
    auto v39 = cooperative_groups::coalesced_threads();
    Closure0 v40{};
    float v41;
    v41 = cooperative_groups::reduce(v39, v16, v40);
    int v42;
    v42 = threadIdx.x;
    int v43;
    v43 = v42 / 32l;
    extern __shared__ unsigned char v44[];
    float * v45;
    v45 = reinterpret_cast<float *>(&v44[0ull]);
    assert("Tensor range check" && 0 <= v43 && v43 < 1l);
    v45[v43] = v41;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v47;
    v47 = threadIdx.x;
    int v48;
    v48 = v47 % 32l;
    bool v49;
    v49 = v43 == 0l;
    bool v51;
    if (v49){
        bool v50;
        v50 = v48 < 1l;
        v51 = v50;
    } else {
        v51 = false;
    }
    if (v51){
        auto v52 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v48 && v48 < 1l);
        float v53;
        v53 = v45[v48];
        float v54;
        v54 = cooperative_groups::reduce(v52, v53, v40);
        v2[0l] = v54;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v55;
    v55 = threadIdx.x;
    bool v56;
    v56 = 0l <= v55;
    bool v57;
    v57 = v56 == false;
    if (v57){
        assert("The index needs to be zero or positive." && v56);
    } else {
    }
    int v59;
    v59 = v55 % 32l;
    int v60;
    v60 = v55 / 32l;
    bool v61;
    v61 = v60 < 1l;
    bool v62;
    v62 = v61 == false;
    if (v62){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v61);
    } else {
    }
    assert("Tensor range check" && 0 <= v60 && v60 < 1l);
    assert("Tensor range check" && 0 <= v59 && v59 < 32l);
    int v64;
    v64 = 4l * v59;
    int v65;
    v65 = 128l * v60;
    int v66;
    v66 = v65 + v64;
    assert("Tensor range check" && 0 <= v60 && v60 < 1l);
    assert("Tensor range check" && 0 <= v59 && v59 < 32l);
    int v67;
    v67 = 0l;
    while (while_method_2(v67)){
        assert("Tensor range check" && 0 <= v67 && v67 < 64l);
        int v69;
        v69 = 128l * v67;
        int v70;
        v70 = v69 + v66;
        int v71[4l];
        int v72[4l];
        int v73;
        v73 = 0l;
        while (while_method_3(v73)){
            assert("Tensor range check" && 0 <= v73 && v73 < 1l);
            int v75;
            v75 = 4l * v73;
            assert("Tensor range check" && 0 <= v73 && v73 < 1l);
            int v76;
            v76 = 128l * v73;
            int v77;
            v77 = v76 + v70;
            int4* v78;
            v78 = reinterpret_cast<int4*>(v0 + v77);
            int4* v79;
            v79 = reinterpret_cast<int4*>(v71 + v75);
            assert("Pointer alignment check" && (unsigned long long)(v78) % 4l == 0 && (unsigned long long)(v79) % 4l == 0);
            *v79 = *v78;
            v73 += 1l ;
        }
        int v80;
        v80 = 0l;
        while (while_method_3(v80)){
            int v82;
            v82 = 0l;
            while (while_method_1(v82)){
                bool v84;
                v84 = 0l <= v82;
                bool v86;
                if (v84){
                    bool v85;
                    v85 = v82 < 4l;
                    v86 = v85;
                } else {
                    v86 = false;
                }
                bool v87;
                v87 = v86 == false;
                if (v87){
                    assert("The indices should be inside the range of the dimension." && v86);
                } else {
                }
                bool v89;
                v89 = 0l <= v59;
                bool v91;
                if (v89){
                    bool v90;
                    v90 = v59 < 32l;
                    v91 = v90;
                } else {
                    v91 = false;
                }
                bool v92;
                v92 = v91 == false;
                if (v92){
                    assert("The indices should be inside the range of the dimension." && v91);
                } else {
                }
                int v94;
                v94 = v59 * 4l;
                int v95;
                v95 = v82 + v94;
                bool v96;
                v96 = 0l <= v80;
                bool v98;
                if (v96){
                    bool v97;
                    v97 = v80 < 1l;
                    v98 = v97;
                } else {
                    v98 = false;
                }
                bool v99;
                v99 = v98 == false;
                if (v99){
                    assert("The indices should be inside the range of the dimension." && v98);
                } else {
                }
                int v101;
                v101 = v80 * 128l;
                int v102;
                v102 = v95 + v101;
                assert("Tensor range check" && 0 <= v80 && v80 < 1l);
                assert("Tensor range check" && 0 <= v82 && v82 < 4l);
                int v103;
                v103 = 4l * v80;
                int v104;
                v104 = v103 + v82;
                v72[v104] = v102;
                v82 += 1l ;
            }
            v80 += 1l ;
        }
        bool v105;
        v105 = 0l <= v60;
        bool v106;
        v106 = v105 && v61;
        bool v107;
        v107 = v106 == false;
        if (v107){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v106);
        } else {
        }
        bool v109;
        v109 = 0l <= v67;
        bool v111;
        if (v109){
            bool v110;
            v110 = v67 < 64l;
            v111 = v110;
        } else {
            v111 = false;
        }
        bool v112;
        v112 = v111 == false;
        if (v112){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v111);
        } else {
        }
        int v114;
        v114 = v67 + v60;
        assert("Tensor range check" && 0 <= v67 && v67 < 64l);
        int v115;
        v115 = 0l;
        while (while_method_3(v115)){
            assert("Tensor range check" && 0 <= v115 && v115 < 1l);
            int v117;
            v117 = 128l * v115;
            int v118;
            v118 = v117 + v70;
            assert("Tensor range check" && 0 <= v115 && v115 < 1l);
            int v119;
            v119 = 4l * v115;
            int4* v120;
            v120 = reinterpret_cast<int4*>(v71 + v119);
            int4* v121;
            v121 = reinterpret_cast<int4*>(v3 + v118);
            assert("Pointer alignment check" && (unsigned long long)(v120) % 4l == 0 && (unsigned long long)(v121) % 4l == 0);
            *v121 = *v120;
            v115 += 1l ;
        }
        v67 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v122;
    v122 = threadIdx.x;
    bool v123;
    v123 = 0l <= v122;
    bool v124;
    v124 = v123 == false;
    if (v124){
        assert("The index needs to be zero or positive." && v123);
    } else {
    }
    int v126;
    v126 = v122 % 32l;
    int v127;
    v127 = v122 / 32l;
    bool v128;
    v128 = v127 < 1l;
    bool v129;
    v129 = v128 == false;
    if (v129){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v128);
    } else {
    }
    assert("Tensor range check" && 0 <= v127 && v127 < 1l);
    assert("Tensor range check" && 0 <= v126 && v126 < 32l);
    int v131;
    v131 = 4l * v126;
    int v132;
    v132 = 128l * v127;
    int v133;
    v133 = v132 + v131;
    assert("Tensor range check" && 0 <= v127 && v127 < 1l);
    assert("Tensor range check" && 0 <= v126 && v126 < 32l);
    int v134;
    v134 = 0l;
    while (while_method_2(v134)){
        assert("Tensor range check" && 0 <= v134 && v134 < 64l);
        int v136;
        v136 = 128l * v134;
        int v137;
        v137 = v136 + v133;
        float v138[4l];
        int v139[4l];
        int v140;
        v140 = 0l;
        while (while_method_3(v140)){
            assert("Tensor range check" && 0 <= v140 && v140 < 1l);
            int v142;
            v142 = 4l * v140;
            assert("Tensor range check" && 0 <= v140 && v140 < 1l);
            int v143;
            v143 = 128l * v140;
            int v144;
            v144 = v143 + v137;
            int4* v145;
            v145 = reinterpret_cast<int4*>(v1 + v144);
            int4* v146;
            v146 = reinterpret_cast<int4*>(v138 + v142);
            assert("Pointer alignment check" && (unsigned long long)(v145) % 4l == 0 && (unsigned long long)(v146) % 4l == 0);
            *v146 = *v145;
            v140 += 1l ;
        }
        int v147;
        v147 = 0l;
        while (while_method_3(v147)){
            int v149;
            v149 = 0l;
            while (while_method_1(v149)){
                bool v151;
                v151 = 0l <= v149;
                bool v153;
                if (v151){
                    bool v152;
                    v152 = v149 < 4l;
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
                bool v156;
                v156 = 0l <= v126;
                bool v158;
                if (v156){
                    bool v157;
                    v157 = v126 < 32l;
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
                int v161;
                v161 = v126 * 4l;
                int v162;
                v162 = v149 + v161;
                bool v163;
                v163 = 0l <= v147;
                bool v165;
                if (v163){
                    bool v164;
                    v164 = v147 < 1l;
                    v165 = v164;
                } else {
                    v165 = false;
                }
                bool v166;
                v166 = v165 == false;
                if (v166){
                    assert("The indices should be inside the range of the dimension." && v165);
                } else {
                }
                int v168;
                v168 = v147 * 128l;
                int v169;
                v169 = v162 + v168;
                assert("Tensor range check" && 0 <= v147 && v147 < 1l);
                assert("Tensor range check" && 0 <= v149 && v149 < 4l);
                int v170;
                v170 = 4l * v147;
                int v171;
                v171 = v170 + v149;
                v139[v171] = v169;
                v149 += 1l ;
            }
            v147 += 1l ;
        }
        bool v172;
        v172 = 0l <= v127;
        bool v173;
        v173 = v172 && v128;
        bool v174;
        v174 = v173 == false;
        if (v174){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v173);
        } else {
        }
        bool v176;
        v176 = 0l <= v134;
        bool v178;
        if (v176){
            bool v177;
            v177 = v134 < 64l;
            v178 = v177;
        } else {
            v178 = false;
        }
        bool v179;
        v179 = v178 == false;
        if (v179){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v178);
        } else {
        }
        int v181;
        v181 = v134 + v127;
        int v182[4l];
        int v183[4l];
        int v184;
        v184 = 0l;
        while (while_method_3(v184)){
            int v186;
            v186 = 0l;
            while (while_method_1(v186)){
                assert("Tensor range check" && 0 <= v184 && v184 < 1l);
                assert("Tensor range check" && 0 <= v186 && v186 < 4l);
                int v188;
                v188 = 4l * v184;
                int v189;
                v189 = v188 + v186;
                int v190;
                v190 = v139[v189];
                assert("Tensor range check" && 0 <= v184 && v184 < 1l);
                assert("Tensor range check" && 0 <= v186 && v186 < 4l);
                v182[v189] = v181;
                v183[v189] = v190;
                v186 += 1l ;
            }
            v184 += 1l ;
        }
        assert("Tensor range check" && 0 <= v134 && v134 < 64l);
        int v191;
        v191 = 0l;
        while (while_method_3(v191)){
            assert("Tensor range check" && 0 <= v191 && v191 < 1l);
            int v193;
            v193 = 128l * v191;
            int v194;
            v194 = v193 + v137;
            assert("Tensor range check" && 0 <= v191 && v191 < 1l);
            int v195;
            v195 = 4l * v191;
            int4* v196;
            v196 = reinterpret_cast<int4*>(v182 + v195);
            int4* v197;
            v197 = reinterpret_cast<int4*>(v10 + v194);
            assert("Pointer alignment check" && (unsigned long long)(v196) % 4l == 0 && (unsigned long long)(v197) % 4l == 0);
            *v197 = *v196;
            int4* v198;
            v198 = reinterpret_cast<int4*>(v183 + v195);
            int4* v199;
            v199 = reinterpret_cast<int4*>(v11 + v194);
            assert("Pointer alignment check" && (unsigned long long)(v198) % 4l == 0 && (unsigned long long)(v199) % 4l == 0);
            *v199 = *v198;
            v191 += 1l ;
        }
        v134 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v200;
    v200 = threadIdx.x;
    bool v201;
    v201 = 0l <= v200;
    bool v202;
    v202 = v201 == false;
    if (v202){
        assert("The index needs to be zero or positive." && v201);
    } else {
    }
    int v204;
    v204 = v200 % 32l;
    int v205;
    v205 = v200 / 32l;
    bool v206;
    v206 = v205 < 1l;
    bool v207;
    v207 = v206 == false;
    if (v207){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v206);
    } else {
    }
    assert("Tensor range check" && 0 <= v205 && v205 < 1l);
    assert("Tensor range check" && 0 <= v204 && v204 < 32l);
    int v209;
    v209 = 4l * v204;
    int v210;
    v210 = 128l * v205;
    int v211;
    v211 = v210 + v209;
    assert("Tensor range check" && 0 <= v205 && v205 < 1l);
    int v212;
    v212 = 0l;
    while (while_method_2(v212)){
        assert("Tensor range check" && 0 <= v212 && v212 < 64l);
        int v214;
        v214 = 128l * v212;
        int v215;
        v215 = v214 + v211;
        float v216[4l];
        int v217[4l];
        int v218;
        v218 = 0l;
        while (while_method_3(v218)){
            assert("Tensor range check" && 0 <= v218 && v218 < 1l);
            int v220;
            v220 = 4l * v218;
            assert("Tensor range check" && 0 <= v218 && v218 < 1l);
            int v221;
            v221 = 128l * v218;
            int v222;
            v222 = v221 + v215;
            int4* v223;
            v223 = reinterpret_cast<int4*>(v1 + v222);
            int4* v224;
            v224 = reinterpret_cast<int4*>(v216 + v220);
            assert("Pointer alignment check" && (unsigned long long)(v223) % 4l == 0 && (unsigned long long)(v224) % 4l == 0);
            *v224 = *v223;
            v218 += 1l ;
        }
        int v225;
        v225 = 0l;
        while (while_method_3(v225)){
            int v227;
            v227 = 0l;
            while (while_method_1(v227)){
                bool v229;
                v229 = 0l <= v227;
                bool v231;
                if (v229){
                    bool v230;
                    v230 = v227 < 4l;
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
                bool v234;
                v234 = 0l <= v204;
                bool v236;
                if (v234){
                    bool v235;
                    v235 = v204 < 32l;
                    v236 = v235;
                } else {
                    v236 = false;
                }
                bool v237;
                v237 = v236 == false;
                if (v237){
                    assert("The indices should be inside the range of the dimension." && v236);
                } else {
                }
                int v239;
                v239 = v204 * 4l;
                int v240;
                v240 = v227 + v239;
                bool v241;
                v241 = 0l <= v225;
                bool v243;
                if (v241){
                    bool v242;
                    v242 = v225 < 1l;
                    v243 = v242;
                } else {
                    v243 = false;
                }
                bool v244;
                v244 = v243 == false;
                if (v244){
                    assert("The indices should be inside the range of the dimension." && v243);
                } else {
                }
                int v246;
                v246 = v225 * 128l;
                int v247;
                v247 = v240 + v246;
                assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                assert("Tensor range check" && 0 <= v227 && v227 < 4l);
                int v248;
                v248 = 4l * v225;
                int v249;
                v249 = v248 + v227;
                v217[v249] = v247;
                v227 += 1l ;
            }
            v225 += 1l ;
        }
        bool v250;
        v250 = 0l <= v205;
        bool v251;
        v251 = v250 && v206;
        bool v252;
        v252 = v251 == false;
        if (v252){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v251);
        } else {
        }
        bool v254;
        v254 = 0l <= v212;
        bool v256;
        if (v254){
            bool v255;
            v255 = v212 < 64l;
            v256 = v255;
        } else {
            v256 = false;
        }
        bool v257;
        v257 = v256 == false;
        if (v257){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v256);
        } else {
        }
        int v259;
        v259 = v212 + v205;
        assert("Tensor range check" && 0 <= v212 && v212 < 64l);
        v12[v259] = v259;
        v212 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
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
    v272 = 0l;
    while (while_method_2(v272)){
        assert("Tensor range check" && 0 <= v272 && v272 < 64l);
        int v274;
        v274 = 128l * v272;
        int v275;
        v275 = v274 + v271;
        float v276[4l];
        int v277[4l];
        int v278;
        v278 = 0l;
        while (while_method_3(v278)){
            assert("Tensor range check" && 0 <= v278 && v278 < 1l);
            int v280;
            v280 = 4l * v278;
            assert("Tensor range check" && 0 <= v278 && v278 < 1l);
            int v281;
            v281 = 128l * v278;
            int v282;
            v282 = v281 + v275;
            int4* v283;
            v283 = reinterpret_cast<int4*>(v1 + v282);
            int4* v284;
            v284 = reinterpret_cast<int4*>(v276 + v280);
            assert("Pointer alignment check" && (unsigned long long)(v283) % 4l == 0 && (unsigned long long)(v284) % 4l == 0);
            *v284 = *v283;
            v278 += 1l ;
        }
        int v285;
        v285 = 0l;
        while (while_method_3(v285)){
            int v287;
            v287 = 0l;
            while (while_method_1(v287)){
                bool v289;
                v289 = 0l <= v287;
                bool v291;
                if (v289){
                    bool v290;
                    v290 = v287 < 4l;
                    v291 = v290;
                } else {
                    v291 = false;
                }
                bool v292;
                v292 = v291 == false;
                if (v292){
                    assert("The indices should be inside the range of the dimension." && v291);
                } else {
                }
                bool v294;
                v294 = 0l <= v264;
                bool v296;
                if (v294){
                    bool v295;
                    v295 = v264 < 32l;
                    v296 = v295;
                } else {
                    v296 = false;
                }
                bool v297;
                v297 = v296 == false;
                if (v297){
                    assert("The indices should be inside the range of the dimension." && v296);
                } else {
                }
                int v299;
                v299 = v264 * 4l;
                int v300;
                v300 = v287 + v299;
                bool v301;
                v301 = 0l <= v285;
                bool v303;
                if (v301){
                    bool v302;
                    v302 = v285 < 1l;
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
                v306 = v285 * 128l;
                int v307;
                v307 = v300 + v306;
                assert("Tensor range check" && 0 <= v285 && v285 < 1l);
                assert("Tensor range check" && 0 <= v287 && v287 < 4l);
                int v308;
                v308 = 4l * v285;
                int v309;
                v309 = v308 + v287;
                v277[v309] = v307;
                v287 += 1l ;
            }
            v285 += 1l ;
        }
        bool v310;
        v310 = 0l <= v265;
        bool v311;
        v311 = v310 && v266;
        bool v312;
        v312 = v311 == false;
        if (v312){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v311);
        } else {
        }
        bool v314;
        v314 = 0l <= v272;
        bool v316;
        if (v314){
            bool v315;
            v315 = v272 < 64l;
            v316 = v315;
        } else {
            v316 = false;
        }
        bool v317;
        v317 = v316 == false;
        if (v317){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v316);
        } else {
        }
        int v319;
        v319 = v272 + v265;
        float v320;
        v320 = 0.0f;
        int v321;
        v321 = 0l;
        while (while_method_3(v321)){
            int v323;
            v323 = 0l;
            while (while_method_1(v323)){
                assert("Tensor range check" && 0 <= v321 && v321 < 1l);
                assert("Tensor range check" && 0 <= v323 && v323 < 4l);
                int v325;
                v325 = 4l * v321;
                int v326;
                v326 = v325 + v323;
                float v327;
                v327 = v276[v326];
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
        int v331;
        v331 = v330 / 32l;
        auto v332 = cooperative_groups::labeled_partition(v329,v331);
        float v333;
        v333 = cooperative_groups::reduce(v332, v320, v40);
        float v334;
        v334 = v333 / 128.0f;
        float v335[4l];
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
                v342 = v276[v341];
                float v343;
                v343 = v342 - v334;
                float v344;
                v344 = exp(v343);
                assert("Tensor range check" && 0 <= v336 && v336 < 1l);
                assert("Tensor range check" && 0 <= v338 && v338 < 4l);
                v335[v341] = v344;
                v338 += 1l ;
            }
            v336 += 1l ;
        }
        float v345;
        v345 = 0.0f;
        int v346;
        v346 = 0l;
        while (while_method_3(v346)){
            int v348;
            v348 = 0l;
            while (while_method_1(v348)){
                assert("Tensor range check" && 0 <= v346 && v346 < 1l);
                assert("Tensor range check" && 0 <= v348 && v348 < 4l);
                int v350;
                v350 = 4l * v346;
                int v351;
                v351 = v350 + v348;
                float v352;
                v352 = v335[v351];
                float v353;
                v353 = v345 + v352;
                v345 = v353;
                v348 += 1l ;
            }
            v346 += 1l ;
        }
        auto v354 = cooperative_groups::coalesced_threads();
        int v355;
        v355 = threadIdx.x;
        int v356;
        v356 = v355 / 32l;
        auto v357 = cooperative_groups::labeled_partition(v354,v356);
        float v358;
        v358 = cooperative_groups::reduce(v357, v345, v40);
        float v359[4l];
        int v360;
        v360 = 0l;
        while (while_method_3(v360)){
            int v362;
            v362 = 0l;
            while (while_method_1(v362)){
                assert("Tensor range check" && 0 <= v360 && v360 < 1l);
                assert("Tensor range check" && 0 <= v362 && v362 < 4l);
                int v364;
                v364 = 4l * v360;
                int v365;
                v365 = v364 + v362;
                float v366;
                v366 = v335[v365];
                float v367;
                v367 = v366 / v358;
                assert("Tensor range check" && 0 <= v360 && v360 < 1l);
                assert("Tensor range check" && 0 <= v362 && v362 < 4l);
                v359[v365] = v367;
                v362 += 1l ;
            }
            v360 += 1l ;
        }
        assert("Tensor range check" && 0 <= v272 && v272 < 64l);
        int v368;
        v368 = 0l;
        while (while_method_3(v368)){
            assert("Tensor range check" && 0 <= v368 && v368 < 1l);
            int v370;
            v370 = 128l * v368;
            int v371;
            v371 = v370 + v275;
            assert("Tensor range check" && 0 <= v368 && v368 < 1l);
            int v372;
            v372 = 4l * v368;
            int4* v373;
            v373 = reinterpret_cast<int4*>(v359 + v372);
            int4* v374;
            v374 = reinterpret_cast<int4*>(v4 + v371);
            assert("Pointer alignment check" && (unsigned long long)(v373) % 4l == 0 && (unsigned long long)(v374) % 4l == 0);
            *v374 = *v373;
            v368 += 1l ;
        }
        v272 += 1l ;
    }
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
    v379 = v375 % 32l;
    int v380;
    v380 = v375 / 32l;
    bool v381;
    v381 = v380 < 1l;
    bool v382;
    v382 = v381 == false;
    if (v382){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v381);
    } else {
    }
    assert("Tensor range check" && 0 <= v380 && v380 < 1l);
    assert("Tensor range check" && 0 <= v379 && v379 < 32l);
    int v384;
    v384 = 4l * v379;
    int v385;
    v385 = 128l * v380;
    int v386;
    v386 = v385 + v384;
    assert("Tensor range check" && 0 <= v380 && v380 < 1l);
    assert("Tensor range check" && 0 <= v379 && v379 < 32l);
    int v387;
    v387 = 0l;
    while (while_method_2(v387)){
        assert("Tensor range check" && 0 <= v387 && v387 < 64l);
        int v389;
        v389 = 128l * v387;
        int v390;
        v390 = v389 + v386;
        float v391[4l];
        int v392[4l];
        int v393;
        v393 = 0l;
        while (while_method_3(v393)){
            assert("Tensor range check" && 0 <= v393 && v393 < 1l);
            int v395;
            v395 = 4l * v393;
            assert("Tensor range check" && 0 <= v393 && v393 < 1l);
            int v396;
            v396 = 128l * v393;
            int v397;
            v397 = v396 + v390;
            int4* v398;
            v398 = reinterpret_cast<int4*>(v1 + v397);
            int4* v399;
            v399 = reinterpret_cast<int4*>(v391 + v395);
            assert("Pointer alignment check" && (unsigned long long)(v398) % 4l == 0 && (unsigned long long)(v399) % 4l == 0);
            *v399 = *v398;
            v393 += 1l ;
        }
        int v400;
        v400 = 0l;
        while (while_method_3(v400)){
            int v402;
            v402 = 0l;
            while (while_method_1(v402)){
                bool v404;
                v404 = 0l <= v402;
                bool v406;
                if (v404){
                    bool v405;
                    v405 = v402 < 4l;
                    v406 = v405;
                } else {
                    v406 = false;
                }
                bool v407;
                v407 = v406 == false;
                if (v407){
                    assert("The indices should be inside the range of the dimension." && v406);
                } else {
                }
                bool v409;
                v409 = 0l <= v379;
                bool v411;
                if (v409){
                    bool v410;
                    v410 = v379 < 32l;
                    v411 = v410;
                } else {
                    v411 = false;
                }
                bool v412;
                v412 = v411 == false;
                if (v412){
                    assert("The indices should be inside the range of the dimension." && v411);
                } else {
                }
                int v414;
                v414 = v379 * 4l;
                int v415;
                v415 = v402 + v414;
                bool v416;
                v416 = 0l <= v400;
                bool v418;
                if (v416){
                    bool v417;
                    v417 = v400 < 1l;
                    v418 = v417;
                } else {
                    v418 = false;
                }
                bool v419;
                v419 = v418 == false;
                if (v419){
                    assert("The indices should be inside the range of the dimension." && v418);
                } else {
                }
                int v421;
                v421 = v400 * 128l;
                int v422;
                v422 = v415 + v421;
                assert("Tensor range check" && 0 <= v400 && v400 < 1l);
                assert("Tensor range check" && 0 <= v402 && v402 < 4l);
                int v423;
                v423 = 4l * v400;
                int v424;
                v424 = v423 + v402;
                v392[v424] = v422;
                v402 += 1l ;
            }
            v400 += 1l ;
        }
        bool v425;
        v425 = 0l <= v380;
        bool v426;
        v426 = v425 && v381;
        bool v427;
        v427 = v426 == false;
        if (v427){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v426);
        } else {
        }
        bool v429;
        v429 = 0l <= v387;
        bool v431;
        if (v429){
            bool v430;
            v430 = v387 < 64l;
            v431 = v430;
        } else {
            v431 = false;
        }
        bool v432;
        v432 = v431 == false;
        if (v432){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v431);
        } else {
        }
        int v434;
        v434 = v387 + v380;
        float v435[4l];
        int v436;
        v436 = 0l;
        while (while_method_3(v436)){
            int v438;
            v438 = 0l;
            while (while_method_1(v438)){
                assert("Tensor range check" && 0 <= v436 && v436 < 1l);
                assert("Tensor range check" && 0 <= v438 && v438 < 4l);
                int v440;
                v440 = 4l * v436;
                int v441;
                v441 = v440 + v438;
                float v442;
                v442 = v391[v441];
                float v443;
                v443 = v442 * v442;
                assert("Tensor range check" && 0 <= v436 && v436 < 1l);
                assert("Tensor range check" && 0 <= v438 && v438 < 4l);
                v435[v441] = v443;
                v438 += 1l ;
            }
            v436 += 1l ;
        }
        float v444;
        v444 = 0.0f;
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
                v451 = v435[v450];
                float v452;
                v452 = v444 + v451;
                v444 = v452;
                v447 += 1l ;
            }
            v445 += 1l ;
        }
        auto v453 = cooperative_groups::coalesced_threads();
        int v454;
        v454 = threadIdx.x;
        int v455;
        v455 = v454 / 32l;
        auto v456 = cooperative_groups::labeled_partition(v453,v455);
        float v457;
        v457 = cooperative_groups::reduce(v456, v444, v40);
        float v458[4l];
        int v459;
        v459 = 0l;
        while (while_method_3(v459)){
            int v461;
            v461 = 0l;
            while (while_method_1(v461)){
                assert("Tensor range check" && 0 <= v459 && v459 < 1l);
                assert("Tensor range check" && 0 <= v461 && v461 < 4l);
                int v463;
                v463 = 4l * v459;
                int v464;
                v464 = v463 + v461;
                float v465;
                v465 = v391[v464];
                bool v466;
                v466 = v457 == 0.0f;
                bool v467;
                v467 = v466 != true;
                float v469;
                if (v467){
                    float v468;
                    v468 = v465 / v457;
                    v469 = v468;
                } else {
                    v469 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v459 && v459 < 1l);
                assert("Tensor range check" && 0 <= v461 && v461 < 4l);
                v458[v464] = v469;
                v461 += 1l ;
            }
            v459 += 1l ;
        }
        assert("Tensor range check" && 0 <= v387 && v387 < 64l);
        int v470;
        v470 = 0l;
        while (while_method_3(v470)){
            assert("Tensor range check" && 0 <= v470 && v470 < 1l);
            int v472;
            v472 = 128l * v470;
            int v473;
            v473 = v472 + v390;
            assert("Tensor range check" && 0 <= v470 && v470 < 1l);
            int v474;
            v474 = 4l * v470;
            int4* v475;
            v475 = reinterpret_cast<int4*>(v458 + v474);
            int4* v476;
            v476 = reinterpret_cast<int4*>(v8 + v473);
            assert("Pointer alignment check" && (unsigned long long)(v475) % 4l == 0 && (unsigned long long)(v476) % 4l == 0);
            *v476 = *v475;
            v470 += 1l ;
        }
        v387 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v477;
    v477 = threadIdx.x;
    bool v478;
    v478 = 0l <= v477;
    bool v479;
    v479 = v478 == false;
    if (v479){
        assert("The index needs to be zero or positive." && v478);
    } else {
    }
    int v481;
    v481 = v477 % 32l;
    int v482;
    v482 = v477 / 32l;
    bool v483;
    v483 = v482 < 1l;
    bool v484;
    v484 = v483 == false;
    if (v484){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v483);
    } else {
    }
    assert("Tensor range check" && 0 <= v482 && v482 < 1l);
    assert("Tensor range check" && 0 <= v481 && v481 < 32l);
    int v486;
    v486 = 4l * v481;
    int v487;
    v487 = 128l * v482;
    int v488;
    v488 = v487 + v486;
    assert("Tensor range check" && 0 <= v482 && v482 < 1l);
    int v489;
    v489 = 0l;
    while (while_method_2(v489)){
        assert("Tensor range check" && 0 <= v489 && v489 < 64l);
        int v491;
        v491 = 128l * v489;
        int v492;
        v492 = v491 + v488;
        float v493[4l];
        int v494[4l];
        int v495;
        v495 = 0l;
        while (while_method_3(v495)){
            assert("Tensor range check" && 0 <= v495 && v495 < 1l);
            int v497;
            v497 = 4l * v495;
            assert("Tensor range check" && 0 <= v495 && v495 < 1l);
            int v498;
            v498 = 128l * v495;
            int v499;
            v499 = v498 + v492;
            int4* v500;
            v500 = reinterpret_cast<int4*>(v1 + v499);
            int4* v501;
            v501 = reinterpret_cast<int4*>(v493 + v497);
            assert("Pointer alignment check" && (unsigned long long)(v500) % 4l == 0 && (unsigned long long)(v501) % 4l == 0);
            *v501 = *v500;
            v495 += 1l ;
        }
        int v502;
        v502 = 0l;
        while (while_method_3(v502)){
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
                v511 = 0l <= v481;
                bool v513;
                if (v511){
                    bool v512;
                    v512 = v481 < 32l;
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
                v516 = v481 * 4l;
                int v517;
                v517 = v504 + v516;
                bool v518;
                v518 = 0l <= v502;
                bool v520;
                if (v518){
                    bool v519;
                    v519 = v502 < 1l;
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
                assert("Tensor range check" && 0 <= v502 && v502 < 1l);
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
        bool v527;
        v527 = 0l <= v482;
        bool v528;
        v528 = v527 && v483;
        bool v529;
        v529 = v528 == false;
        if (v529){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v528);
        } else {
        }
        bool v531;
        v531 = 0l <= v489;
        bool v533;
        if (v531){
            bool v532;
            v532 = v489 < 64l;
            v533 = v532;
        } else {
            v533 = false;
        }
        bool v534;
        v534 = v533 == false;
        if (v534){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v533);
        } else {
        }
        int v536;
        v536 = v489 + v482;
        float v537; int v538;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0l};
        v537 = tmp1.v0; v538 = tmp1.v1;
        int v539;
        v539 = 0l;
        while (while_method_3(v539)){
            int v541;
            v541 = 0l;
            while (while_method_1(v541)){
                assert("Tensor range check" && 0 <= v539 && v539 < 1l);
                assert("Tensor range check" && 0 <= v541 && v541 < 4l);
                int v543;
                v543 = 4l * v539;
                int v544;
                v544 = v543 + v541;
                float v545;
                v545 = v493[v544];
                int v546;
                v546 = v494[v544];
                bool v547;
                v547 = v537 > v545;
                float v548; int v549;
                if (v547){
                    v548 = v537; v549 = v538;
                } else {
                    v548 = v545; v549 = v546;
                }
                v537 = v548;
                v538 = v549;
                v541 += 1l ;
            }
            v539 += 1l ;
        }
        auto v550 = cooperative_groups::coalesced_threads();
        int v551;
        v551 = threadIdx.x;
        int v552;
        v552 = v551 / 32l;
        auto v553 = cooperative_groups::labeled_partition(v550,v552);
        Closure1 v554{};
        float v555; int v556;
        Tuple1 tmp2 = cooperative_groups::reduce(v553, Tuple1{v537, v538}, v554);
        v555 = tmp2.v0; v556 = tmp2.v1;
        assert("Tensor range check" && 0 <= v489 && v489 < 64l);
        v9[v536] = v556;
        v489 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v557;
    v557 = threadIdx.x;
    bool v558;
    v558 = 0l <= v557;
    bool v559;
    v559 = v558 == false;
    if (v559){
        assert("The index needs to be zero or positive." && v558);
    } else {
    }
    int v561;
    v561 = v557 % 32l;
    int v562;
    v562 = v557 / 32l;
    bool v563;
    v563 = v562 < 1l;
    bool v564;
    v564 = v563 == false;
    if (v564){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v563);
    } else {
    }
    assert("Tensor range check" && 0 <= v562 && v562 < 1l);
    assert("Tensor range check" && 0 <= v561 && v561 < 32l);
    int v566;
    v566 = 4l * v561;
    int v567;
    v567 = 128l * v562;
    int v568;
    v568 = v567 + v566;
    assert("Tensor range check" && 0 <= v562 && v562 < 1l);
    assert("Tensor range check" && 0 <= v561 && v561 < 32l);
    int v569;
    v569 = 0l;
    while (while_method_2(v569)){
        assert("Tensor range check" && 0 <= v569 && v569 < 64l);
        int v571;
        v571 = 128l * v569;
        int v572;
        v572 = v571 + v568;
        float v573[4l];
        int v574[4l];
        int v575;
        v575 = 0l;
        while (while_method_3(v575)){
            assert("Tensor range check" && 0 <= v575 && v575 < 1l);
            int v577;
            v577 = 4l * v575;
            assert("Tensor range check" && 0 <= v575 && v575 < 1l);
            int v578;
            v578 = 128l * v575;
            int v579;
            v579 = v578 + v572;
            int4* v580;
            v580 = reinterpret_cast<int4*>(v1 + v579);
            int4* v581;
            v581 = reinterpret_cast<int4*>(v573 + v577);
            assert("Pointer alignment check" && (unsigned long long)(v580) % 4l == 0 && (unsigned long long)(v581) % 4l == 0);
            *v581 = *v580;
            v575 += 1l ;
        }
        int v582;
        v582 = 0l;
        while (while_method_3(v582)){
            int v584;
            v584 = 0l;
            while (while_method_1(v584)){
                bool v586;
                v586 = 0l <= v584;
                bool v588;
                if (v586){
                    bool v587;
                    v587 = v584 < 4l;
                    v588 = v587;
                } else {
                    v588 = false;
                }
                bool v589;
                v589 = v588 == false;
                if (v589){
                    assert("The indices should be inside the range of the dimension." && v588);
                } else {
                }
                bool v591;
                v591 = 0l <= v561;
                bool v593;
                if (v591){
                    bool v592;
                    v592 = v561 < 32l;
                    v593 = v592;
                } else {
                    v593 = false;
                }
                bool v594;
                v594 = v593 == false;
                if (v594){
                    assert("The indices should be inside the range of the dimension." && v593);
                } else {
                }
                int v596;
                v596 = v561 * 4l;
                int v597;
                v597 = v584 + v596;
                bool v598;
                v598 = 0l <= v582;
                bool v600;
                if (v598){
                    bool v599;
                    v599 = v582 < 1l;
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
                int v603;
                v603 = v582 * 128l;
                int v604;
                v604 = v597 + v603;
                assert("Tensor range check" && 0 <= v582 && v582 < 1l);
                assert("Tensor range check" && 0 <= v584 && v584 < 4l);
                int v605;
                v605 = 4l * v582;
                int v606;
                v606 = v605 + v584;
                v574[v606] = v604;
                v584 += 1l ;
            }
            v582 += 1l ;
        }
        bool v607;
        v607 = 0l <= v562;
        bool v608;
        v608 = v607 && v563;
        bool v609;
        v609 = v608 == false;
        if (v609){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v608);
        } else {
        }
        bool v611;
        v611 = 0l <= v569;
        bool v613;
        if (v611){
            bool v612;
            v612 = v569 < 64l;
            v613 = v612;
        } else {
            v613 = false;
        }
        bool v614;
        v614 = v613 == false;
        if (v614){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v613);
        } else {
        }
        int v616;
        v616 = v569 + v562;
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
                v624 = v573[v623];
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
        float v630;
        v630 = cooperative_groups::reduce(v629, v617, v40);
        float v631;
        v631 = v630 / 128.0f;
        float v632[4l];
        int v633;
        v633 = 0l;
        while (while_method_3(v633)){
            int v635;
            v635 = 0l;
            while (while_method_1(v635)){
                assert("Tensor range check" && 0 <= v633 && v633 < 1l);
                assert("Tensor range check" && 0 <= v635 && v635 < 4l);
                int v637;
                v637 = 4l * v633;
                int v638;
                v638 = v637 + v635;
                float v639;
                v639 = v573[v638];
                float v640;
                v640 = v639 - v631;
                float v641;
                v641 = exp(v640);
                assert("Tensor range check" && 0 <= v633 && v633 < 1l);
                assert("Tensor range check" && 0 <= v635 && v635 < 4l);
                v632[v638] = v641;
                v635 += 1l ;
            }
            v633 += 1l ;
        }
        float v642;
        v642 = 0.0f;
        int v643;
        v643 = 0l;
        while (while_method_3(v643)){
            int v645;
            v645 = 0l;
            while (while_method_1(v645)){
                assert("Tensor range check" && 0 <= v643 && v643 < 1l);
                assert("Tensor range check" && 0 <= v645 && v645 < 4l);
                int v647;
                v647 = 4l * v643;
                int v648;
                v648 = v647 + v645;
                float v649;
                v649 = v632[v648];
                float v650;
                v650 = v642 + v649;
                v642 = v650;
                v645 += 1l ;
            }
            v643 += 1l ;
        }
        auto v651 = cooperative_groups::coalesced_threads();
        int v652;
        v652 = threadIdx.x;
        int v653;
        v653 = v652 / 32l;
        auto v654 = cooperative_groups::labeled_partition(v651,v653);
        float v655;
        v655 = cooperative_groups::reduce(v654, v642, v40);
        float v656[4l];
        int v657;
        v657 = 0l;
        while (while_method_3(v657)){
            int v659;
            v659 = 0l;
            while (while_method_1(v659)){
                assert("Tensor range check" && 0 <= v657 && v657 < 1l);
                assert("Tensor range check" && 0 <= v659 && v659 < 4l);
                int v661;
                v661 = 4l * v657;
                int v662;
                v662 = v661 + v659;
                float v663;
                v663 = v632[v662];
                float v664;
                v664 = v663 / v655;
                assert("Tensor range check" && 0 <= v657 && v657 < 1l);
                assert("Tensor range check" && 0 <= v659 && v659 < 4l);
                v656[v662] = v664;
                v659 += 1l ;
            }
            v657 += 1l ;
        }
        float v665[4l];
        float v666;
        v666 = 0.0f;
        int v667;
        v667 = 0l;
        while (while_method_3(v667)){
            assert("Tensor range check" && 0 <= v667 && v667 < 1l);
            int v669;
            v669 = 4l * v667;
            assert("Tensor range check" && 0 <= v667 && v667 < 1l);
            int v670; float v671;
            Tuple0 tmp3 = Tuple0{0l, 0.0f};
            v670 = tmp3.v0; v671 = tmp3.v1;
            while (while_method_1(v670)){
                assert("Tensor range check" && 0 <= v670 && v670 < 4l);
                int v673;
                v673 = v670 + v669;
                float v674;
                v674 = v656[v673];
                float v675;
                v675 = v671 + v674;
                v671 = v675;
                v670 += 1l ;
            }
            auto v676 = cooperative_groups::coalesced_threads();
            int v677;
            v677 = threadIdx.x;
            int v678;
            v678 = v677 / 32l;
            auto v679 = cooperative_groups::labeled_partition(v676,v678);
            Closure2 v680{};
            float v681;
            v681 = cooperative_groups::inclusive_scan(v679, v671, v680);
            float v682;
            v682 = v679.shfl_up(v681,1);
            bool v683;
            v683 = v679.thread_rank() == 0;
            float v684;
            if (v683){
                v684 = 0.0f;
            } else {
                v684 = v682;
            }
            float v685;
            v685 = v679.shfl(v681,v679.num_threads()-1);
            float v686;
            v686 = v666 + v684;
            int v687; float v688;
            Tuple0 tmp4 = Tuple0{0l, v686};
            v687 = tmp4.v0; v688 = tmp4.v1;
            while (while_method_1(v687)){
                assert("Tensor range check" && 0 <= v687 && v687 < 4l);
                int v690;
                v690 = v687 + v669;
                float v691;
                v691 = v656[v690];
                float v692;
                v692 = v688 + v691;
                assert("Tensor range check" && 0 <= v687 && v687 < 4l);
                v665[v690] = v692;
                v688 = v692;
                v687 += 1l ;
            }
            float v693;
            v693 = v666 + v685;
            v666 = v693;
            v667 += 1l ;
        }
        assert("Tensor range check" && 0 <= v569 && v569 < 64l);
        int v694;
        v694 = 0l;
        while (while_method_3(v694)){
            assert("Tensor range check" && 0 <= v694 && v694 < 1l);
            int v696;
            v696 = 128l * v694;
            int v697;
            v697 = v696 + v572;
            assert("Tensor range check" && 0 <= v694 && v694 < 1l);
            int v698;
            v698 = 4l * v694;
            int4* v699;
            v699 = reinterpret_cast<int4*>(v656 + v698);
            int4* v700;
            v700 = reinterpret_cast<int4*>(v6 + v697);
            assert("Pointer alignment check" && (unsigned long long)(v699) % 4l == 0 && (unsigned long long)(v700) % 4l == 0);
            *v700 = *v699;
            int4* v701;
            v701 = reinterpret_cast<int4*>(v665 + v698);
            int4* v702;
            v702 = reinterpret_cast<int4*>(v7 + v697);
            assert("Pointer alignment check" && (unsigned long long)(v701) % 4l == 0 && (unsigned long long)(v702) % 4l == 0);
            *v702 = *v701;
            v694 += 1l ;
        }
        v569 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v703;
    v703 = threadIdx.x;
    bool v704;
    v704 = 0l <= v703;
    bool v705;
    v705 = v704 == false;
    if (v705){
        assert("The index needs to be zero or positive." && v704);
    } else {
    }
    int v707;
    v707 = v703 % 32l;
    int v708;
    v708 = v703 / 32l;
    bool v709;
    v709 = v708 < 1l;
    bool v710;
    v710 = v709 == false;
    if (v710){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v709);
    } else {
    }
    assert("Tensor range check" && 0 <= v708 && v708 < 1l);
    assert("Tensor range check" && 0 <= v707 && v707 < 32l);
    int v712;
    v712 = 4l * v707;
    int v713;
    v713 = 128l * v708;
    int v714;
    v714 = v713 + v712;
    assert("Tensor range check" && 0 <= v708 && v708 < 1l);
    assert("Tensor range check" && 0 <= v707 && v707 < 32l);
    int v715;
    v715 = 0l;
    while (while_method_2(v715)){
        assert("Tensor range check" && 0 <= v715 && v715 < 64l);
        int v717;
        v717 = 128l * v715;
        int v718;
        v718 = v717 + v714;
        int v719[4l];
        int v720[4l];
        int v721;
        v721 = 0l;
        while (while_method_3(v721)){
            assert("Tensor range check" && 0 <= v721 && v721 < 1l);
            int v723;
            v723 = 4l * v721;
            assert("Tensor range check" && 0 <= v721 && v721 < 1l);
            int v724;
            v724 = 128l * v721;
            int v725;
            v725 = v724 + v718;
            int4* v726;
            v726 = reinterpret_cast<int4*>(v0 + v725);
            int4* v727;
            v727 = reinterpret_cast<int4*>(v719 + v723);
            assert("Pointer alignment check" && (unsigned long long)(v726) % 4l == 0 && (unsigned long long)(v727) % 4l == 0);
            *v727 = *v726;
            v721 += 1l ;
        }
        int v728;
        v728 = 0l;
        while (while_method_3(v728)){
            int v730;
            v730 = 0l;
            while (while_method_1(v730)){
                bool v732;
                v732 = 0l <= v730;
                bool v734;
                if (v732){
                    bool v733;
                    v733 = v730 < 4l;
                    v734 = v733;
                } else {
                    v734 = false;
                }
                bool v735;
                v735 = v734 == false;
                if (v735){
                    assert("The indices should be inside the range of the dimension." && v734);
                } else {
                }
                bool v737;
                v737 = 0l <= v707;
                bool v739;
                if (v737){
                    bool v738;
                    v738 = v707 < 32l;
                    v739 = v738;
                } else {
                    v739 = false;
                }
                bool v740;
                v740 = v739 == false;
                if (v740){
                    assert("The indices should be inside the range of the dimension." && v739);
                } else {
                }
                int v742;
                v742 = v707 * 4l;
                int v743;
                v743 = v730 + v742;
                bool v744;
                v744 = 0l <= v728;
                bool v746;
                if (v744){
                    bool v745;
                    v745 = v728 < 1l;
                    v746 = v745;
                } else {
                    v746 = false;
                }
                bool v747;
                v747 = v746 == false;
                if (v747){
                    assert("The indices should be inside the range of the dimension." && v746);
                } else {
                }
                int v749;
                v749 = v728 * 128l;
                int v750;
                v750 = v743 + v749;
                assert("Tensor range check" && 0 <= v728 && v728 < 1l);
                assert("Tensor range check" && 0 <= v730 && v730 < 4l);
                int v751;
                v751 = 4l * v728;
                int v752;
                v752 = v751 + v730;
                v720[v752] = v750;
                v730 += 1l ;
            }
            v728 += 1l ;
        }
        bool v753;
        v753 = 0l <= v708;
        bool v754;
        v754 = v753 && v709;
        bool v755;
        v755 = v754 == false;
        if (v755){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v754);
        } else {
        }
        bool v757;
        v757 = 0l <= v715;
        bool v759;
        if (v757){
            bool v758;
            v758 = v715 < 64l;
            v759 = v758;
        } else {
            v759 = false;
        }
        bool v760;
        v760 = v759 == false;
        if (v760){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v759);
        } else {
        }
        int v762;
        v762 = v715 + v708;
        int v763[4l];
        int v764;
        v764 = 0l;
        int v765;
        v765 = 0l;
        while (while_method_3(v765)){
            assert("Tensor range check" && 0 <= v765 && v765 < 1l);
            int v767;
            v767 = 4l * v765;
            assert("Tensor range check" && 0 <= v765 && v765 < 1l);
            int v768; int v769;
            Tuple2 tmp5 = Tuple2{0l, 0l};
            v768 = tmp5.v0; v769 = tmp5.v1;
            while (while_method_1(v768)){
                assert("Tensor range check" && 0 <= v768 && v768 < 4l);
                int v771;
                v771 = v768 + v767;
                int v772;
                v772 = v719[v771];
                int v773;
                v773 = v769 + v772;
                v769 = v773;
                v768 += 1l ;
            }
            auto v774 = cooperative_groups::coalesced_threads();
            int v775;
            v775 = threadIdx.x;
            int v776;
            v776 = v775 / 32l;
            auto v777 = cooperative_groups::labeled_partition(v774,v776);
            Closure3 v778{};
            int v779;
            v779 = cooperative_groups::inclusive_scan(v777, v769, v778);
            int v780;
            v780 = v777.shfl_up(v779,1);
            bool v781;
            v781 = v777.thread_rank() == 0;
            int v782;
            if (v781){
                v782 = 0l;
            } else {
                v782 = v780;
            }
            int v783;
            v783 = v777.shfl(v779,v777.num_threads()-1);
            int v784;
            v784 = v764 + v782;
            int v785; int v786;
            Tuple2 tmp6 = Tuple2{0l, v784};
            v785 = tmp6.v0; v786 = tmp6.v1;
            while (while_method_1(v785)){
                assert("Tensor range check" && 0 <= v785 && v785 < 4l);
                int v788;
                v788 = v785 + v767;
                int v789;
                v789 = v719[v788];
                assert("Tensor range check" && 0 <= v785 && v785 < 4l);
                v763[v788] = v786;
                int v790;
                v790 = v786 + v789;
                v786 = v790;
                v785 += 1l ;
            }
            int v791;
            v791 = v764 + v783;
            v764 = v791;
            v765 += 1l ;
        }
        assert("Tensor range check" && 0 <= v715 && v715 < 64l);
        int v792;
        v792 = 0l;
        while (while_method_3(v792)){
            assert("Tensor range check" && 0 <= v792 && v792 < 1l);
            int v794;
            v794 = 128l * v792;
            int v795;
            v795 = v794 + v718;
            assert("Tensor range check" && 0 <= v792 && v792 < 1l);
            int v796;
            v796 = 4l * v792;
            int4* v797;
            v797 = reinterpret_cast<int4*>(v763 + v796);
            int4* v798;
            v798 = reinterpret_cast<int4*>(v13 + v795);
            assert("Pointer alignment check" && (unsigned long long)(v797) % 4l == 0 && (unsigned long long)(v798) % 4l == 0);
            *v798 = *v797;
            v792 += 1l ;
        }
        v715 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v799;
    v799 = threadIdx.x;
    bool v800;
    v800 = 0l <= v799;
    bool v801;
    v801 = v800 == false;
    if (v801){
        assert("The index needs to be zero or positive." && v800);
    } else {
    }
    int v803;
    v803 = v799 % 32l;
    int v804;
    v804 = v799 / 32l;
    bool v805;
    v805 = v804 < 1l;
    bool v806;
    v806 = v805 == false;
    if (v806){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v805);
    } else {
    }
    assert("Tensor range check" && 0 <= v804 && v804 < 1l);
    assert("Tensor range check" && 0 <= v803 && v803 < 32l);
    int v808;
    v808 = 4l * v803;
    int v809;
    v809 = 128l * v804;
    int v810;
    v810 = v809 + v808;
    assert("Tensor range check" && 0 <= v804 && v804 < 1l);
    assert("Tensor range check" && 0 <= v803 && v803 < 32l);
    int v811;
    v811 = 0l;
    while (while_method_2(v811)){
        assert("Tensor range check" && 0 <= v811 && v811 < 64l);
        int v813;
        v813 = 128l * v811;
        int v814;
        v814 = v813 + v810;
        float v815[4l];
        int v816[4l];
        int v817;
        v817 = 0l;
        while (while_method_3(v817)){
            assert("Tensor range check" && 0 <= v817 && v817 < 1l);
            int v819;
            v819 = 4l * v817;
            assert("Tensor range check" && 0 <= v817 && v817 < 1l);
            int v820;
            v820 = 128l * v817;
            int v821;
            v821 = v820 + v814;
            int4* v822;
            v822 = reinterpret_cast<int4*>(v1 + v821);
            int4* v823;
            v823 = reinterpret_cast<int4*>(v815 + v819);
            assert("Pointer alignment check" && (unsigned long long)(v822) % 4l == 0 && (unsigned long long)(v823) % 4l == 0);
            *v823 = *v822;
            v817 += 1l ;
        }
        int v824;
        v824 = 0l;
        while (while_method_3(v824)){
            int v826;
            v826 = 0l;
            while (while_method_1(v826)){
                bool v828;
                v828 = 0l <= v826;
                bool v830;
                if (v828){
                    bool v829;
                    v829 = v826 < 4l;
                    v830 = v829;
                } else {
                    v830 = false;
                }
                bool v831;
                v831 = v830 == false;
                if (v831){
                    assert("The indices should be inside the range of the dimension." && v830);
                } else {
                }
                bool v833;
                v833 = 0l <= v803;
                bool v835;
                if (v833){
                    bool v834;
                    v834 = v803 < 32l;
                    v835 = v834;
                } else {
                    v835 = false;
                }
                bool v836;
                v836 = v835 == false;
                if (v836){
                    assert("The indices should be inside the range of the dimension." && v835);
                } else {
                }
                int v838;
                v838 = v803 * 4l;
                int v839;
                v839 = v826 + v838;
                bool v840;
                v840 = 0l <= v824;
                bool v842;
                if (v840){
                    bool v841;
                    v841 = v824 < 1l;
                    v842 = v841;
                } else {
                    v842 = false;
                }
                bool v843;
                v843 = v842 == false;
                if (v843){
                    assert("The indices should be inside the range of the dimension." && v842);
                } else {
                }
                int v845;
                v845 = v824 * 128l;
                int v846;
                v846 = v839 + v845;
                assert("Tensor range check" && 0 <= v824 && v824 < 1l);
                assert("Tensor range check" && 0 <= v826 && v826 < 4l);
                int v847;
                v847 = 4l * v824;
                int v848;
                v848 = v847 + v826;
                v816[v848] = v846;
                v826 += 1l ;
            }
            v824 += 1l ;
        }
        bool v849;
        v849 = 0l <= v804;
        bool v850;
        v850 = v849 && v805;
        bool v851;
        v851 = v850 == false;
        if (v851){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v850);
        } else {
        }
        bool v853;
        v853 = 0l <= v811;
        bool v855;
        if (v853){
            bool v854;
            v854 = v811 < 64l;
            v855 = v854;
        } else {
            v855 = false;
        }
        bool v856;
        v856 = v855 == false;
        if (v856){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v855);
        } else {
        }
        int v858;
        v858 = v811 + v804;
        bool v859[4l];
        int v860;
        v860 = 0l;
        while (while_method_3(v860)){
            int v862;
            v862 = 0l;
            while (while_method_1(v862)){
                assert("Tensor range check" && 0 <= v860 && v860 < 1l);
                assert("Tensor range check" && 0 <= v862 && v862 < 4l);
                int v864;
                v864 = 4l * v860;
                int v865;
                v865 = v864 + v862;
                float v866;
                v866 = v815[v865];
                int v867;
                v867 = v816[v865];
                bool v868;
                v868 = v867 < 4l;
                assert("Tensor range check" && 0 <= v860 && v860 < 1l);
                assert("Tensor range check" && 0 <= v862 && v862 < 4l);
                v859[v865] = v868;
                v862 += 1l ;
            }
            v860 += 1l ;
        }
        int v869[4l];
        int v870;
        v870 = 0l;
        while (while_method_3(v870)){
            int v872;
            v872 = 0l;
            while (while_method_1(v872)){
                assert("Tensor range check" && 0 <= v870 && v870 < 1l);
                assert("Tensor range check" && 0 <= v872 && v872 < 4l);
                int v874;
                v874 = 4l * v870;
                int v875;
                v875 = v874 + v872;
                bool v876;
                v876 = v859[v875];
                int v877;
                if (v876){
                    v877 = 1l;
                } else {
                    v877 = 0l;
                }
                assert("Tensor range check" && 0 <= v870 && v870 < 1l);
                assert("Tensor range check" && 0 <= v872 && v872 < 4l);
                v869[v875] = v877;
                v872 += 1l ;
            }
            v870 += 1l ;
        }
        int v878;
        v878 = 0l;
        int v879;
        v879 = 0l;
        while (while_method_3(v879)){
            int v881;
            v881 = 0l;
            while (while_method_1(v881)){
                assert("Tensor range check" && 0 <= v879 && v879 < 1l);
                assert("Tensor range check" && 0 <= v881 && v881 < 4l);
                int v883;
                v883 = 4l * v879;
                int v884;
                v884 = v883 + v881;
                int v885;
                v885 = v869[v884];
                int v886;
                v886 = v878 + v885;
                v878 = v886;
                v881 += 1l ;
            }
            v879 += 1l ;
        }
        auto v887 = cooperative_groups::coalesced_threads();
        int v888;
        v888 = threadIdx.x;
        int v889;
        v889 = v888 / 32l;
        auto v890 = cooperative_groups::labeled_partition(v887,v889);
        Closure4 v891{};
        int v892;
        v892 = cooperative_groups::reduce(v890, v878, v891);
        float v893[4l];
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
                float v900;
                v900 = v815[v899];
                bool v901;
                v901 = v859[v899];
                float v902;
                if (v901){
                    v902 = v900;
                } else {
                    v902 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v894 && v894 < 1l);
                assert("Tensor range check" && 0 <= v896 && v896 < 4l);
                v893[v899] = v902;
                v896 += 1l ;
            }
            v894 += 1l ;
        }
        float v903;
        v903 = 0.0f;
        int v904;
        v904 = 0l;
        while (while_method_3(v904)){
            int v906;
            v906 = 0l;
            while (while_method_1(v906)){
                assert("Tensor range check" && 0 <= v904 && v904 < 1l);
                assert("Tensor range check" && 0 <= v906 && v906 < 4l);
                int v908;
                v908 = 4l * v904;
                int v909;
                v909 = v908 + v906;
                float v910;
                v910 = v893[v909];
                float v911;
                v911 = v903 + v910;
                v903 = v911;
                v906 += 1l ;
            }
            v904 += 1l ;
        }
        auto v912 = cooperative_groups::coalesced_threads();
        int v913;
        v913 = threadIdx.x;
        int v914;
        v914 = v913 / 32l;
        auto v915 = cooperative_groups::labeled_partition(v912,v914);
        float v916;
        v916 = cooperative_groups::reduce(v915, v903, v40);
        float v917;
        v917 = (float)v892;
        float v918;
        v918 = v916 / v917;
        float v919[4l];
        int v920;
        v920 = 0l;
        while (while_method_3(v920)){
            int v922;
            v922 = 0l;
            while (while_method_1(v922)){
                assert("Tensor range check" && 0 <= v920 && v920 < 1l);
                assert("Tensor range check" && 0 <= v922 && v922 < 4l);
                int v924;
                v924 = 4l * v920;
                int v925;
                v925 = v924 + v922;
                float v926;
                v926 = v815[v925];
                bool v927;
                v927 = v859[v925];
                float v928;
                if (v927){
                    v928 = v926;
                } else {
                    v928 = -1.0f / 0.0f;
                }
                float v929;
                v929 = v928 - v918;
                float v930;
                v930 = exp(v929);
                assert("Tensor range check" && 0 <= v920 && v920 < 1l);
                assert("Tensor range check" && 0 <= v922 && v922 < 4l);
                v919[v925] = v930;
                v922 += 1l ;
            }
            v920 += 1l ;
        }
        float v931;
        v931 = 0.0f;
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
                v938 = v919[v937];
                float v939;
                v939 = v931 + v938;
                v931 = v939;
                v934 += 1l ;
            }
            v932 += 1l ;
        }
        auto v940 = cooperative_groups::coalesced_threads();
        int v941;
        v941 = threadIdx.x;
        int v942;
        v942 = v941 / 32l;
        auto v943 = cooperative_groups::labeled_partition(v940,v942);
        float v944;
        v944 = cooperative_groups::reduce(v943, v931, v40);
        float v945[4l];
        int v946;
        v946 = 0l;
        while (while_method_3(v946)){
            int v948;
            v948 = 0l;
            while (while_method_1(v948)){
                assert("Tensor range check" && 0 <= v946 && v946 < 1l);
                assert("Tensor range check" && 0 <= v948 && v948 < 4l);
                int v950;
                v950 = 4l * v946;
                int v951;
                v951 = v950 + v948;
                float v952;
                v952 = v919[v951];
                float v953;
                v953 = v952 / v944;
                assert("Tensor range check" && 0 <= v946 && v946 < 1l);
                assert("Tensor range check" && 0 <= v948 && v948 < 4l);
                v945[v951] = v953;
                v948 += 1l ;
            }
            v946 += 1l ;
        }
        assert("Tensor range check" && 0 <= v811 && v811 < 64l);
        int v954;
        v954 = 0l;
        while (while_method_3(v954)){
            assert("Tensor range check" && 0 <= v954 && v954 < 1l);
            int v956;
            v956 = 128l * v954;
            int v957;
            v957 = v956 + v814;
            assert("Tensor range check" && 0 <= v954 && v954 < 1l);
            int v958;
            v958 = 4l * v954;
            int4* v959;
            v959 = reinterpret_cast<int4*>(v945 + v958);
            int4* v960;
            v960 = reinterpret_cast<int4*>(v5 + v957);
            assert("Pointer alignment check" && (unsigned long long)(v959) % 4l == 0 && (unsigned long long)(v960) % 4l == 0);
            *v960 = *v959;
            v954 += 1l ;
        }
        v811 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v961;
    v961 = threadIdx.x;
    unsigned long long v962;
    v962 = (unsigned long long)v961;
    curandStatePhilox4_32_10_t v963;
    curand_init(12344321ull,v962,0ull,&v963);
    int v964;
    v964 = threadIdx.x;
    bool v965;
    v965 = 0l <= v964;
    bool v966;
    v966 = v965 == false;
    if (v966){
        assert("The index needs to be zero or positive." && v965);
    } else {
    }
    int v968;
    v968 = v964 % 32l;
    int v969;
    v969 = v964 / 32l;
    bool v970;
    v970 = v969 < 1l;
    bool v971;
    v971 = v970 == false;
    if (v971){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v970);
    } else {
    }
    assert("Tensor range check" && 0 <= v969 && v969 < 1l);
    assert("Tensor range check" && 0 <= v968 && v968 < 32l);
    int v973;
    v973 = 4l * v968;
    int v974;
    v974 = 128l * v969;
    int v975;
    v975 = v974 + v973;
    assert("Tensor range check" && 0 <= v969 && v969 < 1l);
    assert("Tensor range check" && 0 <= v968 && v968 < 32l);
    assert("Tensor range check" && 0 <= v969 && v969 < 1l);
    int v976;
    v976 = 0l;
    while (while_method_2(v976)){
        assert("Tensor range check" && 0 <= v976 && v976 < 64l);
        int v978;
        v978 = 128l * v976;
        int v979;
        v979 = v978 + v975;
        float v980[4l];
        int v981[4l];
        int v982;
        v982 = 0l;
        while (while_method_3(v982)){
            assert("Tensor range check" && 0 <= v982 && v982 < 1l);
            int v984;
            v984 = 4l * v982;
            assert("Tensor range check" && 0 <= v982 && v982 < 1l);
            int v985;
            v985 = 128l * v982;
            int v986;
            v986 = v985 + v979;
            int4* v987;
            v987 = reinterpret_cast<int4*>(v1 + v986);
            int4* v988;
            v988 = reinterpret_cast<int4*>(v980 + v984);
            assert("Pointer alignment check" && (unsigned long long)(v987) % 4l == 0 && (unsigned long long)(v988) % 4l == 0);
            *v988 = *v987;
            v982 += 1l ;
        }
        int v989;
        v989 = 0l;
        while (while_method_3(v989)){
            int v991;
            v991 = 0l;
            while (while_method_1(v991)){
                bool v993;
                v993 = 0l <= v991;
                bool v995;
                if (v993){
                    bool v994;
                    v994 = v991 < 4l;
                    v995 = v994;
                } else {
                    v995 = false;
                }
                bool v996;
                v996 = v995 == false;
                if (v996){
                    assert("The indices should be inside the range of the dimension." && v995);
                } else {
                }
                bool v998;
                v998 = 0l <= v968;
                bool v1000;
                if (v998){
                    bool v999;
                    v999 = v968 < 32l;
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
                int v1003;
                v1003 = v968 * 4l;
                int v1004;
                v1004 = v991 + v1003;
                bool v1005;
                v1005 = 0l <= v989;
                bool v1007;
                if (v1005){
                    bool v1006;
                    v1006 = v989 < 1l;
                    v1007 = v1006;
                } else {
                    v1007 = false;
                }
                bool v1008;
                v1008 = v1007 == false;
                if (v1008){
                    assert("The indices should be inside the range of the dimension." && v1007);
                } else {
                }
                int v1010;
                v1010 = v989 * 128l;
                int v1011;
                v1011 = v1004 + v1010;
                assert("Tensor range check" && 0 <= v989 && v989 < 1l);
                assert("Tensor range check" && 0 <= v991 && v991 < 4l);
                int v1012;
                v1012 = 4l * v989;
                int v1013;
                v1013 = v1012 + v991;
                v981[v1013] = v1011;
                v991 += 1l ;
            }
            v989 += 1l ;
        }
        bool v1014;
        v1014 = 0l <= v969;
        bool v1015;
        v1015 = v1014 && v970;
        bool v1016;
        v1016 = v1015 == false;
        if (v1016){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1015);
        } else {
        }
        bool v1018;
        v1018 = 0l <= v976;
        bool v1020;
        if (v1018){
            bool v1019;
            v1019 = v976 < 64l;
            v1020 = v1019;
        } else {
            v1020 = false;
        }
        bool v1021;
        v1021 = v1020 == false;
        if (v1021){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1020);
        } else {
        }
        int v1023;
        v1023 = v976 + v969;
        float v1024;
        v1024 = 0.0f;
        int v1025;
        v1025 = 0l;
        while (while_method_3(v1025)){
            int v1027;
            v1027 = 0l;
            while (while_method_1(v1027)){
                assert("Tensor range check" && 0 <= v1025 && v1025 < 1l);
                assert("Tensor range check" && 0 <= v1027 && v1027 < 4l);
                int v1029;
                v1029 = 4l * v1025;
                int v1030;
                v1030 = v1029 + v1027;
                float v1031;
                v1031 = v980[v1030];
                float v1032;
                v1032 = v1024 + v1031;
                v1024 = v1032;
                v1027 += 1l ;
            }
            v1025 += 1l ;
        }
        auto v1033 = cooperative_groups::coalesced_threads();
        int v1034;
        v1034 = threadIdx.x;
        int v1035;
        v1035 = v1034 / 32l;
        auto v1036 = cooperative_groups::labeled_partition(v1033,v1035);
        float v1037;
        v1037 = cooperative_groups::reduce(v1036, v1024, v40);
        float v1038;
        v1038 = v1037 / 128.0f;
        float v1039[4l];
        int v1040;
        v1040 = 0l;
        while (while_method_3(v1040)){
            int v1042;
            v1042 = 0l;
            while (while_method_1(v1042)){
                assert("Tensor range check" && 0 <= v1040 && v1040 < 1l);
                assert("Tensor range check" && 0 <= v1042 && v1042 < 4l);
                int v1044;
                v1044 = 4l * v1040;
                int v1045;
                v1045 = v1044 + v1042;
                float v1046;
                v1046 = v980[v1045];
                float v1047;
                v1047 = v1046 - v1038;
                float v1048;
                v1048 = exp(v1047);
                assert("Tensor range check" && 0 <= v1040 && v1040 < 1l);
                assert("Tensor range check" && 0 <= v1042 && v1042 < 4l);
                v1039[v1045] = v1048;
                v1042 += 1l ;
            }
            v1040 += 1l ;
        }
        float v1049;
        v1049 = 0.0f;
        int v1050;
        v1050 = 0l;
        while (while_method_3(v1050)){
            int v1052;
            v1052 = 0l;
            while (while_method_1(v1052)){
                assert("Tensor range check" && 0 <= v1050 && v1050 < 1l);
                assert("Tensor range check" && 0 <= v1052 && v1052 < 4l);
                int v1054;
                v1054 = 4l * v1050;
                int v1055;
                v1055 = v1054 + v1052;
                float v1056;
                v1056 = v1039[v1055];
                float v1057;
                v1057 = v1049 + v1056;
                v1049 = v1057;
                v1052 += 1l ;
            }
            v1050 += 1l ;
        }
        auto v1058 = cooperative_groups::coalesced_threads();
        int v1059;
        v1059 = threadIdx.x;
        int v1060;
        v1060 = v1059 / 32l;
        auto v1061 = cooperative_groups::labeled_partition(v1058,v1060);
        float v1062;
        v1062 = cooperative_groups::reduce(v1061, v1049, v40);
        float v1063[4l];
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
                v1070 = v1039[v1069];
                float v1071;
                v1071 = v1070 / v1062;
                assert("Tensor range check" && 0 <= v1064 && v1064 < 1l);
                assert("Tensor range check" && 0 <= v1066 && v1066 < 4l);
                v1063[v1069] = v1071;
                v1066 += 1l ;
            }
            v1064 += 1l ;
        }
        float v1072[4l];
        float v1073;
        v1073 = 0.0f;
        int v1074;
        v1074 = 0l;
        while (while_method_3(v1074)){
            assert("Tensor range check" && 0 <= v1074 && v1074 < 1l);
            int v1076;
            v1076 = 4l * v1074;
            assert("Tensor range check" && 0 <= v1074 && v1074 < 1l);
            int v1077; float v1078;
            Tuple0 tmp7 = Tuple0{0l, 0.0f};
            v1077 = tmp7.v0; v1078 = tmp7.v1;
            while (while_method_1(v1077)){
                assert("Tensor range check" && 0 <= v1077 && v1077 < 4l);
                int v1080;
                v1080 = v1077 + v1076;
                float v1081;
                v1081 = v1063[v1080];
                float v1082;
                v1082 = v1078 + v1081;
                v1078 = v1082;
                v1077 += 1l ;
            }
            auto v1083 = cooperative_groups::coalesced_threads();
            int v1084;
            v1084 = threadIdx.x;
            int v1085;
            v1085 = v1084 / 32l;
            auto v1086 = cooperative_groups::labeled_partition(v1083,v1085);
            Closure2 v1087{};
            float v1088;
            v1088 = cooperative_groups::inclusive_scan(v1086, v1078, v1087);
            float v1089;
            v1089 = v1086.shfl_up(v1088,1);
            bool v1090;
            v1090 = v1086.thread_rank() == 0;
            float v1091;
            if (v1090){
                v1091 = 0.0f;
            } else {
                v1091 = v1089;
            }
            float v1092;
            v1092 = v1086.shfl(v1088,v1086.num_threads()-1);
            float v1093;
            v1093 = v1073 + v1091;
            int v1094; float v1095;
            Tuple0 tmp8 = Tuple0{0l, v1093};
            v1094 = tmp8.v0; v1095 = tmp8.v1;
            while (while_method_1(v1094)){
                assert("Tensor range check" && 0 <= v1094 && v1094 < 4l);
                int v1097;
                v1097 = v1094 + v1076;
                float v1098;
                v1098 = v1063[v1097];
                float v1099;
                v1099 = v1095 + v1098;
                assert("Tensor range check" && 0 <= v1094 && v1094 < 4l);
                v1072[v1097] = v1099;
                v1095 = v1099;
                v1094 += 1l ;
            }
            float v1100;
            v1100 = v1073 + v1092;
            v1073 = v1100;
            v1074 += 1l ;
        }
        float v1101[4l];
        bool v1102[4l];
        int v1103;
        v1103 = 0l;
        while (while_method_3(v1103)){
            int v1105;
            v1105 = 0l;
            while (while_method_1(v1105)){
                assert("Tensor range check" && 0 <= v1103 && v1103 < 1l);
                assert("Tensor range check" && 0 <= v1105 && v1105 < 4l);
                int v1107;
                v1107 = 4l * v1103;
                int v1108;
                v1108 = v1107 + v1105;
                float v1109;
                v1109 = v1072[v1108];
                float v1110;
                v1110 = v1063[v1108];
                bool v1111;
                v1111 = v1110 > 0.0f;
                assert("Tensor range check" && 0 <= v1103 && v1103 < 1l);
                assert("Tensor range check" && 0 <= v1105 && v1105 < 4l);
                v1101[v1108] = v1109;
                v1102[v1108] = v1111;
                v1105 += 1l ;
            }
            v1103 += 1l ;
        }
        float v1112; bool v1113;
        Tuple3 tmp9 = Tuple3{-1.0f / 0.0f, false};
        v1112 = tmp9.v0; v1113 = tmp9.v1;
        int v1114;
        v1114 = 0l;
        while (while_method_3(v1114)){
            int v1116;
            v1116 = 0l;
            while (while_method_1(v1116)){
                assert("Tensor range check" && 0 <= v1114 && v1114 < 1l);
                assert("Tensor range check" && 0 <= v1116 && v1116 < 4l);
                int v1118;
                v1118 = 4l * v1114;
                int v1119;
                v1119 = v1118 + v1116;
                float v1120;
                v1120 = v1101[v1119];
                bool v1121;
                v1121 = v1102[v1119];
                float v1128; bool v1129;
                if (v1113){
                    if (v1121){
                        bool v1122;
                        v1122 = v1112 >= v1120;
                        float v1123;
                        if (v1122){
                            v1123 = v1112;
                        } else {
                            v1123 = v1120;
                        }
                        v1128 = v1123; v1129 = true;
                    } else {
                        v1128 = v1112; v1129 = v1113;
                    }
                } else {
                    if (v1121){
                        v1128 = v1120; v1129 = v1121;
                    } else {
                        v1128 = v1112; v1129 = v1113;
                    }
                }
                v1112 = v1128;
                v1113 = v1129;
                v1116 += 1l ;
            }
            v1114 += 1l ;
        }
        auto v1130 = cooperative_groups::coalesced_threads();
        int v1131;
        v1131 = threadIdx.x;
        int v1132;
        v1132 = v1131 / 32l;
        auto v1133 = cooperative_groups::labeled_partition(v1130,v1132);
        Closure5 v1134{};
        float v1135; bool v1136;
        Tuple3 tmp10 = cooperative_groups::reduce(v1133, Tuple3{v1112, v1113}, v1134);
        v1135 = tmp10.v0; v1136 = tmp10.v1;
        bool v1137;
        v1137 = v1136 == false;
        if (v1137){
            assert("The local reduce must be true." && v1136);
        } else {
        }
        float v1139[4l];
        int v1140[4l];
        int v1141;
        v1141 = 0l;
        while (while_method_3(v1141)){
            int v1143;
            v1143 = 0l;
            while (while_method_1(v1143)){
                assert("Tensor range check" && 0 <= v1141 && v1141 < 1l);
                assert("Tensor range check" && 0 <= v1143 && v1143 < 4l);
                int v1145;
                v1145 = 4l * v1141;
                int v1146;
                v1146 = v1145 + v1143;
                int v1147;
                v1147 = v981[v1146];
                float v1148;
                v1148 = curand_uniform(&v963);
                assert("Tensor range check" && 0 <= v1141 && v1141 < 1l);
                assert("Tensor range check" && 0 <= v1143 && v1143 < 4l);
                v1139[v1146] = v1148;
                v1140[v1146] = v1147;
                v1143 += 1l ;
            }
            v1141 += 1l ;
        }
        float v1149; int v1150;
        Tuple1 tmp11 = Tuple1{0.0f, 2147483647l};
        v1149 = tmp11.v0; v1150 = tmp11.v1;
        int v1151;
        v1151 = 0l;
        while (while_method_3(v1151)){
            int v1153;
            v1153 = 0l;
            while (while_method_1(v1153)){
                assert("Tensor range check" && 0 <= v1151 && v1151 < 1l);
                assert("Tensor range check" && 0 <= v1153 && v1153 < 4l);
                int v1155;
                v1155 = 4l * v1151;
                int v1156;
                v1156 = v1155 + v1153;
                float v1157;
                v1157 = v1139[v1156];
                int v1158;
                v1158 = v1140[v1156];
                bool v1159;
                v1159 = v1150 < v1158;
                float v1160; int v1161;
                if (v1159){
                    v1160 = v1149; v1161 = v1150;
                } else {
                    v1160 = v1157; v1161 = v1158;
                }
                v1149 = v1160;
                v1150 = v1161;
                v1153 += 1l ;
            }
            v1151 += 1l ;
        }
        auto v1162 = cooperative_groups::coalesced_threads();
        int v1163;
        v1163 = threadIdx.x;
        int v1164;
        v1164 = v1163 / 32l;
        auto v1165 = cooperative_groups::labeled_partition(v1162,v1164);
        Closure6 v1166{};
        float v1167; int v1168;
        Tuple1 tmp12 = cooperative_groups::reduce(v1165, Tuple1{v1149, v1150}, v1166);
        v1167 = tmp12.v0; v1168 = tmp12.v1;
        float v1169;
        v1169 = v1135 * v1167;
        int v1170[4l];
        bool v1171[4l];
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
                v1178 = v1101[v1177];
                bool v1179;
                v1179 = v1102[v1177];
                int v1180;
                v1180 = v981[v1177];
                int v1183; bool v1184;
                if (v1179){
                    float v1181;
                    v1181 = v1178 - v1169;
                    bool v1182;
                    v1182 = v1181 >= 0.0f;
                    v1183 = v1180; v1184 = v1182;
                } else {
                    v1183 = 2147483647l; v1184 = false;
                }
                assert("Tensor range check" && 0 <= v1172 && v1172 < 1l);
                assert("Tensor range check" && 0 <= v1174 && v1174 < 4l);
                v1170[v1177] = v1183;
                v1171[v1177] = v1184;
                v1174 += 1l ;
            }
            v1172 += 1l ;
        }
        int v1185; bool v1186;
        Tuple4 tmp13 = Tuple4{2147483647l, false};
        v1185 = tmp13.v0; v1186 = tmp13.v1;
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
                int v1193;
                v1193 = v1170[v1192];
                bool v1194;
                v1194 = v1171[v1192];
                int v1201; bool v1202;
                if (v1186){
                    if (v1194){
                        bool v1195;
                        v1195 = v1185 < v1193;
                        int v1196;
                        if (v1195){
                            v1196 = v1185;
                        } else {
                            v1196 = v1193;
                        }
                        v1201 = v1196; v1202 = true;
                    } else {
                        v1201 = v1185; v1202 = v1186;
                    }
                } else {
                    if (v1194){
                        v1201 = v1193; v1202 = v1194;
                    } else {
                        v1201 = v1185; v1202 = v1186;
                    }
                }
                v1185 = v1201;
                v1186 = v1202;
                v1189 += 1l ;
            }
            v1187 += 1l ;
        }
        auto v1203 = cooperative_groups::coalesced_threads();
        int v1204;
        v1204 = threadIdx.x;
        int v1205;
        v1205 = v1204 / 32l;
        auto v1206 = cooperative_groups::labeled_partition(v1203,v1205);
        Closure7 v1207{};
        int v1208; bool v1209;
        Tuple4 tmp14 = cooperative_groups::reduce(v1206, Tuple4{v1185, v1186}, v1207);
        v1208 = tmp14.v0; v1209 = tmp14.v1;
        bool v1210;
        v1210 = v1209 == false;
        if (v1210){
            assert("The local reduce must be true." && v1209);
        } else {
        }
        assert("Tensor range check" && 0 <= v976 && v976 < 64l);
        int v1212;
        v1212 = 0l;
        while (while_method_3(v1212)){
            assert("Tensor range check" && 0 <= v1212 && v1212 < 1l);
            int v1214;
            v1214 = 128l * v1212;
            int v1215;
            v1215 = v1214 + v979;
            assert("Tensor range check" && 0 <= v1212 && v1212 < 1l);
            int v1216;
            v1216 = 4l * v1212;
            int4* v1217;
            v1217 = reinterpret_cast<int4*>(v1063 + v1216);
            int4* v1218;
            v1218 = reinterpret_cast<int4*>(v14 + v1215);
            assert("Pointer alignment check" && (unsigned long long)(v1217) % 4l == 0 && (unsigned long long)(v1218) % 4l == 0);
            *v1218 = *v1217;
            v1212 += 1l ;
        }
        assert("Tensor range check" && 0 <= v976 && v976 < 64l);
        v15[v1023] = v1208;
        v976 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1219;
    v1219 = threadIdx.x;
    unsigned long long v1220;
    v1220 = (unsigned long long)v1219;
    curandStatePhilox4_32_10_t v1221;
    curand_init(12344321ull,v1220,0ull,&v1221);
    int v1222;
    v1222 = threadIdx.x;
    bool v1223;
    v1223 = 0l <= v1222;
    bool v1224;
    v1224 = v1223 == false;
    if (v1224){
        assert("The index needs to be zero or positive." && v1223);
    } else {
    }
    int v1226;
    v1226 = v1222 % 32l;
    int v1227;
    v1227 = v1222 / 32l;
    bool v1228;
    v1228 = v1227 < 1l;
    bool v1229;
    v1229 = v1228 == false;
    if (v1229){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1228);
    } else {
    }
    assert("Tensor range check" && 0 <= v1227 && v1227 < 1l);
    assert("Tensor range check" && 0 <= v1226 && v1226 < 32l);
    int v1231;
    v1231 = 4l * v1226;
    int v1232;
    v1232 = 128l * v1227;
    int v1233;
    v1233 = v1232 + v1231;
    assert("Tensor range check" && 0 <= v1227 && v1227 < 1l);
    assert("Tensor range check" && 0 <= v1226 && v1226 < 32l);
    assert("Tensor range check" && 0 <= v1227 && v1227 < 1l);
    int v1234;
    v1234 = 0l;
    while (while_method_2(v1234)){
        assert("Tensor range check" && 0 <= v1234 && v1234 < 64l);
        int v1236;
        v1236 = 128l * v1234;
        int v1237;
        v1237 = v1236 + v1233;
        float v1238[4l];
        int v1239[4l];
        int v1240;
        v1240 = 0l;
        while (while_method_3(v1240)){
            assert("Tensor range check" && 0 <= v1240 && v1240 < 1l);
            int v1242;
            v1242 = 4l * v1240;
            assert("Tensor range check" && 0 <= v1240 && v1240 < 1l);
            int v1243;
            v1243 = 128l * v1240;
            int v1244;
            v1244 = v1243 + v1237;
            int4* v1245;
            v1245 = reinterpret_cast<int4*>(v1 + v1244);
            int4* v1246;
            v1246 = reinterpret_cast<int4*>(v1238 + v1242);
            assert("Pointer alignment check" && (unsigned long long)(v1245) % 4l == 0 && (unsigned long long)(v1246) % 4l == 0);
            *v1246 = *v1245;
            v1240 += 1l ;
        }
        int v1247;
        v1247 = 0l;
        while (while_method_3(v1247)){
            int v1249;
            v1249 = 0l;
            while (while_method_1(v1249)){
                bool v1251;
                v1251 = 0l <= v1249;
                bool v1253;
                if (v1251){
                    bool v1252;
                    v1252 = v1249 < 4l;
                    v1253 = v1252;
                } else {
                    v1253 = false;
                }
                bool v1254;
                v1254 = v1253 == false;
                if (v1254){
                    assert("The indices should be inside the range of the dimension." && v1253);
                } else {
                }
                bool v1256;
                v1256 = 0l <= v1226;
                bool v1258;
                if (v1256){
                    bool v1257;
                    v1257 = v1226 < 32l;
                    v1258 = v1257;
                } else {
                    v1258 = false;
                }
                bool v1259;
                v1259 = v1258 == false;
                if (v1259){
                    assert("The indices should be inside the range of the dimension." && v1258);
                } else {
                }
                int v1261;
                v1261 = v1226 * 4l;
                int v1262;
                v1262 = v1249 + v1261;
                bool v1263;
                v1263 = 0l <= v1247;
                bool v1265;
                if (v1263){
                    bool v1264;
                    v1264 = v1247 < 1l;
                    v1265 = v1264;
                } else {
                    v1265 = false;
                }
                bool v1266;
                v1266 = v1265 == false;
                if (v1266){
                    assert("The indices should be inside the range of the dimension." && v1265);
                } else {
                }
                int v1268;
                v1268 = v1247 * 128l;
                int v1269;
                v1269 = v1262 + v1268;
                assert("Tensor range check" && 0 <= v1247 && v1247 < 1l);
                assert("Tensor range check" && 0 <= v1249 && v1249 < 4l);
                int v1270;
                v1270 = 4l * v1247;
                int v1271;
                v1271 = v1270 + v1249;
                v1239[v1271] = v1269;
                v1249 += 1l ;
            }
            v1247 += 1l ;
        }
        bool v1272;
        v1272 = 0l <= v1227;
        bool v1273;
        v1273 = v1272 && v1228;
        bool v1274;
        v1274 = v1273 == false;
        if (v1274){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1273);
        } else {
        }
        bool v1276;
        v1276 = 0l <= v1234;
        bool v1278;
        if (v1276){
            bool v1277;
            v1277 = v1234 < 64l;
            v1278 = v1277;
        } else {
            v1278 = false;
        }
        bool v1279;
        v1279 = v1278 == false;
        if (v1279){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1278);
        } else {
        }
        int v1281;
        v1281 = v1234 + v1227;
        bool v1282[4l];
        int v1283;
        v1283 = 0l;
        while (while_method_3(v1283)){
            int v1285;
            v1285 = 0l;
            while (while_method_1(v1285)){
                assert("Tensor range check" && 0 <= v1283 && v1283 < 1l);
                assert("Tensor range check" && 0 <= v1285 && v1285 < 4l);
                int v1287;
                v1287 = 4l * v1283;
                int v1288;
                v1288 = v1287 + v1285;
                float v1289;
                v1289 = v1238[v1288];
                int v1290;
                v1290 = v1239[v1288];
                bool v1291;
                v1291 = v1290 < 11l;
                assert("Tensor range check" && 0 <= v1283 && v1283 < 1l);
                assert("Tensor range check" && 0 <= v1285 && v1285 < 4l);
                v1282[v1288] = v1291;
                v1285 += 1l ;
            }
            v1283 += 1l ;
        }
        int v1292[4l];
        int v1293;
        v1293 = 0l;
        while (while_method_3(v1293)){
            int v1295;
            v1295 = 0l;
            while (while_method_1(v1295)){
                assert("Tensor range check" && 0 <= v1293 && v1293 < 1l);
                assert("Tensor range check" && 0 <= v1295 && v1295 < 4l);
                int v1297;
                v1297 = 4l * v1293;
                int v1298;
                v1298 = v1297 + v1295;
                bool v1299;
                v1299 = v1282[v1298];
                int v1300;
                if (v1299){
                    v1300 = 1l;
                } else {
                    v1300 = 0l;
                }
                assert("Tensor range check" && 0 <= v1293 && v1293 < 1l);
                assert("Tensor range check" && 0 <= v1295 && v1295 < 4l);
                v1292[v1298] = v1300;
                v1295 += 1l ;
            }
            v1293 += 1l ;
        }
        int v1301;
        v1301 = 0l;
        int v1302;
        v1302 = 0l;
        while (while_method_3(v1302)){
            int v1304;
            v1304 = 0l;
            while (while_method_1(v1304)){
                assert("Tensor range check" && 0 <= v1302 && v1302 < 1l);
                assert("Tensor range check" && 0 <= v1304 && v1304 < 4l);
                int v1306;
                v1306 = 4l * v1302;
                int v1307;
                v1307 = v1306 + v1304;
                int v1308;
                v1308 = v1292[v1307];
                int v1309;
                v1309 = v1301 + v1308;
                v1301 = v1309;
                v1304 += 1l ;
            }
            v1302 += 1l ;
        }
        auto v1310 = cooperative_groups::coalesced_threads();
        int v1311;
        v1311 = threadIdx.x;
        int v1312;
        v1312 = v1311 / 32l;
        auto v1313 = cooperative_groups::labeled_partition(v1310,v1312);
        Closure4 v1314{};
        int v1315;
        v1315 = cooperative_groups::reduce(v1313, v1301, v1314);
        float v1316[4l];
        int v1317;
        v1317 = 0l;
        while (while_method_3(v1317)){
            int v1319;
            v1319 = 0l;
            while (while_method_1(v1319)){
                assert("Tensor range check" && 0 <= v1317 && v1317 < 1l);
                assert("Tensor range check" && 0 <= v1319 && v1319 < 4l);
                int v1321;
                v1321 = 4l * v1317;
                int v1322;
                v1322 = v1321 + v1319;
                float v1323;
                v1323 = v1238[v1322];
                bool v1324;
                v1324 = v1282[v1322];
                float v1325;
                if (v1324){
                    v1325 = v1323;
                } else {
                    v1325 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1317 && v1317 < 1l);
                assert("Tensor range check" && 0 <= v1319 && v1319 < 4l);
                v1316[v1322] = v1325;
                v1319 += 1l ;
            }
            v1317 += 1l ;
        }
        float v1326;
        v1326 = 0.0f;
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
                float v1333;
                v1333 = v1316[v1332];
                float v1334;
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
        v1337 = v1336 / 32l;
        auto v1338 = cooperative_groups::labeled_partition(v1335,v1337);
        float v1339;
        v1339 = cooperative_groups::reduce(v1338, v1326, v40);
        float v1340;
        v1340 = (float)v1315;
        float v1341;
        v1341 = v1339 / v1340;
        float v1342[4l];
        int v1343;
        v1343 = 0l;
        while (while_method_3(v1343)){
            int v1345;
            v1345 = 0l;
            while (while_method_1(v1345)){
                assert("Tensor range check" && 0 <= v1343 && v1343 < 1l);
                assert("Tensor range check" && 0 <= v1345 && v1345 < 4l);
                int v1347;
                v1347 = 4l * v1343;
                int v1348;
                v1348 = v1347 + v1345;
                float v1349;
                v1349 = v1238[v1348];
                bool v1350;
                v1350 = v1282[v1348];
                float v1351;
                if (v1350){
                    v1351 = v1349;
                } else {
                    v1351 = -1.0f / 0.0f;
                }
                float v1352;
                v1352 = v1351 - v1341;
                float v1353;
                v1353 = exp(v1352);
                assert("Tensor range check" && 0 <= v1343 && v1343 < 1l);
                assert("Tensor range check" && 0 <= v1345 && v1345 < 4l);
                v1342[v1348] = v1353;
                v1345 += 1l ;
            }
            v1343 += 1l ;
        }
        float v1354;
        v1354 = 0.0f;
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
                float v1361;
                v1361 = v1342[v1360];
                float v1362;
                v1362 = v1354 + v1361;
                v1354 = v1362;
                v1357 += 1l ;
            }
            v1355 += 1l ;
        }
        auto v1363 = cooperative_groups::coalesced_threads();
        int v1364;
        v1364 = threadIdx.x;
        int v1365;
        v1365 = v1364 / 32l;
        auto v1366 = cooperative_groups::labeled_partition(v1363,v1365);
        float v1367;
        v1367 = cooperative_groups::reduce(v1366, v1354, v40);
        float v1368[4l];
        int v1369;
        v1369 = 0l;
        while (while_method_3(v1369)){
            int v1371;
            v1371 = 0l;
            while (while_method_1(v1371)){
                assert("Tensor range check" && 0 <= v1369 && v1369 < 1l);
                assert("Tensor range check" && 0 <= v1371 && v1371 < 4l);
                int v1373;
                v1373 = 4l * v1369;
                int v1374;
                v1374 = v1373 + v1371;
                float v1375;
                v1375 = v1342[v1374];
                float v1376;
                v1376 = v1375 / v1367;
                assert("Tensor range check" && 0 <= v1369 && v1369 < 1l);
                assert("Tensor range check" && 0 <= v1371 && v1371 < 4l);
                v1368[v1374] = v1376;
                v1371 += 1l ;
            }
            v1369 += 1l ;
        }
        float v1377[4l];
        float v1378;
        v1378 = 0.0f;
        int v1379;
        v1379 = 0l;
        while (while_method_3(v1379)){
            assert("Tensor range check" && 0 <= v1379 && v1379 < 1l);
            int v1381;
            v1381 = 4l * v1379;
            assert("Tensor range check" && 0 <= v1379 && v1379 < 1l);
            int v1382; float v1383;
            Tuple0 tmp15 = Tuple0{0l, 0.0f};
            v1382 = tmp15.v0; v1383 = tmp15.v1;
            while (while_method_1(v1382)){
                assert("Tensor range check" && 0 <= v1382 && v1382 < 4l);
                int v1385;
                v1385 = v1382 + v1381;
                float v1386;
                v1386 = v1368[v1385];
                float v1387;
                v1387 = v1383 + v1386;
                v1383 = v1387;
                v1382 += 1l ;
            }
            auto v1388 = cooperative_groups::coalesced_threads();
            int v1389;
            v1389 = threadIdx.x;
            int v1390;
            v1390 = v1389 / 32l;
            auto v1391 = cooperative_groups::labeled_partition(v1388,v1390);
            Closure2 v1392{};
            float v1393;
            v1393 = cooperative_groups::inclusive_scan(v1391, v1383, v1392);
            float v1394;
            v1394 = v1391.shfl_up(v1393,1);
            bool v1395;
            v1395 = v1391.thread_rank() == 0;
            float v1396;
            if (v1395){
                v1396 = 0.0f;
            } else {
                v1396 = v1394;
            }
            float v1397;
            v1397 = v1391.shfl(v1393,v1391.num_threads()-1);
            float v1398;
            v1398 = v1378 + v1396;
            int v1399; float v1400;
            Tuple0 tmp16 = Tuple0{0l, v1398};
            v1399 = tmp16.v0; v1400 = tmp16.v1;
            while (while_method_1(v1399)){
                assert("Tensor range check" && 0 <= v1399 && v1399 < 4l);
                int v1402;
                v1402 = v1399 + v1381;
                float v1403;
                v1403 = v1368[v1402];
                float v1404;
                v1404 = v1400 + v1403;
                assert("Tensor range check" && 0 <= v1399 && v1399 < 4l);
                v1377[v1402] = v1404;
                v1400 = v1404;
                v1399 += 1l ;
            }
            float v1405;
            v1405 = v1378 + v1397;
            v1378 = v1405;
            v1379 += 1l ;
        }
        float v1406[4l];
        bool v1407[4l];
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
                v1414 = v1377[v1413];
                float v1415;
                v1415 = v1368[v1413];
                bool v1416;
                v1416 = v1415 > 0.0f;
                assert("Tensor range check" && 0 <= v1408 && v1408 < 1l);
                assert("Tensor range check" && 0 <= v1410 && v1410 < 4l);
                v1406[v1413] = v1414;
                v1407[v1413] = v1416;
                v1410 += 1l ;
            }
            v1408 += 1l ;
        }
        float v1417; bool v1418;
        Tuple3 tmp17 = Tuple3{-1.0f / 0.0f, false};
        v1417 = tmp17.v0; v1418 = tmp17.v1;
        int v1419;
        v1419 = 0l;
        while (while_method_3(v1419)){
            int v1421;
            v1421 = 0l;
            while (while_method_1(v1421)){
                assert("Tensor range check" && 0 <= v1419 && v1419 < 1l);
                assert("Tensor range check" && 0 <= v1421 && v1421 < 4l);
                int v1423;
                v1423 = 4l * v1419;
                int v1424;
                v1424 = v1423 + v1421;
                float v1425;
                v1425 = v1406[v1424];
                bool v1426;
                v1426 = v1407[v1424];
                float v1433; bool v1434;
                if (v1418){
                    if (v1426){
                        bool v1427;
                        v1427 = v1417 >= v1425;
                        float v1428;
                        if (v1427){
                            v1428 = v1417;
                        } else {
                            v1428 = v1425;
                        }
                        v1433 = v1428; v1434 = true;
                    } else {
                        v1433 = v1417; v1434 = v1418;
                    }
                } else {
                    if (v1426){
                        v1433 = v1425; v1434 = v1426;
                    } else {
                        v1433 = v1417; v1434 = v1418;
                    }
                }
                v1417 = v1433;
                v1418 = v1434;
                v1421 += 1l ;
            }
            v1419 += 1l ;
        }
        auto v1435 = cooperative_groups::coalesced_threads();
        int v1436;
        v1436 = threadIdx.x;
        int v1437;
        v1437 = v1436 / 32l;
        auto v1438 = cooperative_groups::labeled_partition(v1435,v1437);
        Closure5 v1439{};
        float v1440; bool v1441;
        Tuple3 tmp18 = cooperative_groups::reduce(v1438, Tuple3{v1417, v1418}, v1439);
        v1440 = tmp18.v0; v1441 = tmp18.v1;
        bool v1442;
        v1442 = v1441 == false;
        if (v1442){
            assert("The local reduce must be true." && v1441);
        } else {
        }
        float v1444[4l];
        int v1445[4l];
        int v1446;
        v1446 = 0l;
        while (while_method_3(v1446)){
            int v1448;
            v1448 = 0l;
            while (while_method_1(v1448)){
                assert("Tensor range check" && 0 <= v1446 && v1446 < 1l);
                assert("Tensor range check" && 0 <= v1448 && v1448 < 4l);
                int v1450;
                v1450 = 4l * v1446;
                int v1451;
                v1451 = v1450 + v1448;
                int v1452;
                v1452 = v1239[v1451];
                float v1453;
                v1453 = curand_uniform(&v1221);
                assert("Tensor range check" && 0 <= v1446 && v1446 < 1l);
                assert("Tensor range check" && 0 <= v1448 && v1448 < 4l);
                v1444[v1451] = v1453;
                v1445[v1451] = v1452;
                v1448 += 1l ;
            }
            v1446 += 1l ;
        }
        float v1454; int v1455;
        Tuple1 tmp19 = Tuple1{0.0f, 2147483647l};
        v1454 = tmp19.v0; v1455 = tmp19.v1;
        int v1456;
        v1456 = 0l;
        while (while_method_3(v1456)){
            int v1458;
            v1458 = 0l;
            while (while_method_1(v1458)){
                assert("Tensor range check" && 0 <= v1456 && v1456 < 1l);
                assert("Tensor range check" && 0 <= v1458 && v1458 < 4l);
                int v1460;
                v1460 = 4l * v1456;
                int v1461;
                v1461 = v1460 + v1458;
                float v1462;
                v1462 = v1444[v1461];
                int v1463;
                v1463 = v1445[v1461];
                bool v1464;
                v1464 = v1455 < v1463;
                float v1465; int v1466;
                if (v1464){
                    v1465 = v1454; v1466 = v1455;
                } else {
                    v1465 = v1462; v1466 = v1463;
                }
                v1454 = v1465;
                v1455 = v1466;
                v1458 += 1l ;
            }
            v1456 += 1l ;
        }
        auto v1467 = cooperative_groups::coalesced_threads();
        int v1468;
        v1468 = threadIdx.x;
        int v1469;
        v1469 = v1468 / 32l;
        auto v1470 = cooperative_groups::labeled_partition(v1467,v1469);
        Closure6 v1471{};
        float v1472; int v1473;
        Tuple1 tmp20 = cooperative_groups::reduce(v1470, Tuple1{v1454, v1455}, v1471);
        v1472 = tmp20.v0; v1473 = tmp20.v1;
        float v1474;
        v1474 = v1440 * v1472;
        int v1475[4l];
        bool v1476[4l];
        int v1477;
        v1477 = 0l;
        while (while_method_3(v1477)){
            int v1479;
            v1479 = 0l;
            while (while_method_1(v1479)){
                assert("Tensor range check" && 0 <= v1477 && v1477 < 1l);
                assert("Tensor range check" && 0 <= v1479 && v1479 < 4l);
                int v1481;
                v1481 = 4l * v1477;
                int v1482;
                v1482 = v1481 + v1479;
                float v1483;
                v1483 = v1406[v1482];
                bool v1484;
                v1484 = v1407[v1482];
                int v1485;
                v1485 = v1239[v1482];
                int v1488; bool v1489;
                if (v1484){
                    float v1486;
                    v1486 = v1483 - v1474;
                    bool v1487;
                    v1487 = v1486 >= 0.0f;
                    v1488 = v1485; v1489 = v1487;
                } else {
                    v1488 = 2147483647l; v1489 = false;
                }
                assert("Tensor range check" && 0 <= v1477 && v1477 < 1l);
                assert("Tensor range check" && 0 <= v1479 && v1479 < 4l);
                v1475[v1482] = v1488;
                v1476[v1482] = v1489;
                v1479 += 1l ;
            }
            v1477 += 1l ;
        }
        int v1490; bool v1491;
        Tuple4 tmp21 = Tuple4{2147483647l, false};
        v1490 = tmp21.v0; v1491 = tmp21.v1;
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
                v1498 = v1475[v1497];
                bool v1499;
                v1499 = v1476[v1497];
                int v1506; bool v1507;
                if (v1491){
                    if (v1499){
                        bool v1500;
                        v1500 = v1490 < v1498;
                        int v1501;
                        if (v1500){
                            v1501 = v1490;
                        } else {
                            v1501 = v1498;
                        }
                        v1506 = v1501; v1507 = true;
                    } else {
                        v1506 = v1490; v1507 = v1491;
                    }
                } else {
                    if (v1499){
                        v1506 = v1498; v1507 = v1499;
                    } else {
                        v1506 = v1490; v1507 = v1491;
                    }
                }
                v1490 = v1506;
                v1491 = v1507;
                v1494 += 1l ;
            }
            v1492 += 1l ;
        }
        auto v1508 = cooperative_groups::coalesced_threads();
        int v1509;
        v1509 = threadIdx.x;
        int v1510;
        v1510 = v1509 / 32l;
        auto v1511 = cooperative_groups::labeled_partition(v1508,v1510);
        Closure7 v1512{};
        int v1513; bool v1514;
        Tuple4 tmp22 = cooperative_groups::reduce(v1511, Tuple4{v1490, v1491}, v1512);
        v1513 = tmp22.v0; v1514 = tmp22.v1;
        bool v1515;
        v1515 = v1514 == false;
        if (v1515){
            assert("The local reduce must be true." && v1514);
        } else {
        }
        assert("Tensor range check" && 0 <= v1234 && v1234 < 64l);
        int v1517;
        v1517 = 0l;
        while (while_method_3(v1517)){
            assert("Tensor range check" && 0 <= v1517 && v1517 < 1l);
            int v1519;
            v1519 = 128l * v1517;
            int v1520;
            v1520 = v1519 + v1237;
            assert("Tensor range check" && 0 <= v1517 && v1517 < 1l);
            int v1521;
            v1521 = 4l * v1517;
            int4* v1522;
            v1522 = reinterpret_cast<int4*>(v1368 + v1521);
            int4* v1523;
            v1523 = reinterpret_cast<int4*>(v14 + v1520);
            assert("Pointer alignment check" && (unsigned long long)(v1522) % 4l == 0 && (unsigned long long)(v1523) % 4l == 0);
            *v1523 = *v1522;
            v1517 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1234 && v1234 < 64l);
        v15[v1281] = v1513;
        v1234 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
extern "C" __global__ void entry1(int * v0, float * v1, float * v2, int * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int * v9, int * v10, int * v11, int * v12, int * v13, float * v14, int * v15) {
    float v16;
    v16 = 0.0f;
    int v17;
    v17 = threadIdx.x;
    int v18;
    v18 = v17;
    while (while_method_0(v18)){
        bool v20;
        v20 = 0l <= v18;
        bool v21;
        v21 = v20 == false;
        if (v21){
            assert("The index needs to be zero or positive." && v20);
        } else {
        }
        int v23;
        v23 = v18 % 16l;
        int v24;
        v24 = v18 / 16l;
        bool v25;
        v25 = v24 < 128l;
        bool v26;
        v26 = v25 == false;
        if (v26){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v25);
        } else {
        }
        assert("Tensor range check" && 0 <= v24 && v24 < 128l);
        assert("Tensor range check" && 0 <= v23 && v23 < 16l);
        int v28;
        v28 = 4l * v23;
        int v29;
        v29 = 64l * v24;
        int v30;
        v30 = v29 + v28;
        float v31[4l];
        int4* v32;
        v32 = reinterpret_cast<int4*>(v1 + v30);
        int4* v33;
        v33 = reinterpret_cast<int4*>(v31 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v32) % 4l == 0 && (unsigned long long)(v33) % 4l == 0);
        *v33 = *v32;
        int v34; float v35;
        Tuple0 tmp23 = Tuple0{0l, v16};
        v34 = tmp23.v0; v35 = tmp23.v1;
        while (while_method_1(v34)){
            assert("Tensor range check" && 0 <= v34 && v34 < 4l);
            float v37;
            v37 = v31[v34];
            float v38;
            v38 = v35 + v37;
            v35 = v38;
            v34 += 1l ;
        }
        v16 = v35;
        v18 += 32l ;
    }
    auto v39 = cooperative_groups::coalesced_threads();
    Closure0 v40{};
    float v41;
    v41 = cooperative_groups::reduce(v39, v16, v40);
    int v42;
    v42 = threadIdx.x;
    int v43;
    v43 = v42 / 32l;
    extern __shared__ unsigned char v44[];
    float * v45;
    v45 = reinterpret_cast<float *>(&v44[0ull]);
    assert("Tensor range check" && 0 <= v43 && v43 < 1l);
    v45[v43] = v41;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v47;
    v47 = threadIdx.x;
    int v48;
    v48 = v47 % 32l;
    bool v49;
    v49 = v43 == 0l;
    bool v51;
    if (v49){
        bool v50;
        v50 = v48 < 1l;
        v51 = v50;
    } else {
        v51 = false;
    }
    if (v51){
        auto v52 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v48 && v48 < 1l);
        float v53;
        v53 = v45[v48];
        float v54;
        v54 = cooperative_groups::reduce(v52, v53, v40);
        v2[0l] = v54;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v55;
    v55 = threadIdx.x;
    bool v56;
    v56 = 0l <= v55;
    bool v57;
    v57 = v56 == false;
    if (v57){
        assert("The index needs to be zero or positive." && v56);
    } else {
    }
    int v59;
    v59 = v55 % 16l;
    int v60;
    v60 = v55 / 16l;
    bool v61;
    v61 = v60 < 2l;
    bool v62;
    v62 = v61 == false;
    if (v62){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v61);
    } else {
    }
    assert("Tensor range check" && 0 <= v60 && v60 < 2l);
    assert("Tensor range check" && 0 <= v59 && v59 < 16l);
    int v64;
    v64 = 4l * v59;
    int v65;
    v65 = 64l * v60;
    int v66;
    v66 = v65 + v64;
    assert("Tensor range check" && 0 <= v60 && v60 < 2l);
    assert("Tensor range check" && 0 <= v59 && v59 < 16l);
    int v67;
    v67 = 0l;
    while (while_method_2(v67)){
        assert("Tensor range check" && 0 <= v67 && v67 < 64l);
        int v69;
        v69 = 128l * v67;
        int v70;
        v70 = v69 + v66;
        int v71[4l];
        int v72[4l];
        int v73;
        v73 = 0l;
        while (while_method_3(v73)){
            assert("Tensor range check" && 0 <= v73 && v73 < 1l);
            int v75;
            v75 = 4l * v73;
            assert("Tensor range check" && 0 <= v73 && v73 < 1l);
            int v76;
            v76 = 64l * v73;
            int v77;
            v77 = v76 + v70;
            int4* v78;
            v78 = reinterpret_cast<int4*>(v0 + v77);
            int4* v79;
            v79 = reinterpret_cast<int4*>(v71 + v75);
            assert("Pointer alignment check" && (unsigned long long)(v78) % 4l == 0 && (unsigned long long)(v79) % 4l == 0);
            *v79 = *v78;
            v73 += 1l ;
        }
        int v80;
        v80 = 0l;
        while (while_method_3(v80)){
            int v82;
            v82 = 0l;
            while (while_method_1(v82)){
                bool v84;
                v84 = 0l <= v82;
                bool v86;
                if (v84){
                    bool v85;
                    v85 = v82 < 4l;
                    v86 = v85;
                } else {
                    v86 = false;
                }
                bool v87;
                v87 = v86 == false;
                if (v87){
                    assert("The indices should be inside the range of the dimension." && v86);
                } else {
                }
                bool v89;
                v89 = 0l <= v59;
                bool v91;
                if (v89){
                    bool v90;
                    v90 = v59 < 16l;
                    v91 = v90;
                } else {
                    v91 = false;
                }
                bool v92;
                v92 = v91 == false;
                if (v92){
                    assert("The indices should be inside the range of the dimension." && v91);
                } else {
                }
                int v94;
                v94 = v59 * 4l;
                int v95;
                v95 = v82 + v94;
                bool v96;
                v96 = 0l <= v80;
                bool v98;
                if (v96){
                    bool v97;
                    v97 = v80 < 1l;
                    v98 = v97;
                } else {
                    v98 = false;
                }
                bool v99;
                v99 = v98 == false;
                if (v99){
                    assert("The indices should be inside the range of the dimension." && v98);
                } else {
                }
                int v101;
                v101 = v80 * 64l;
                int v102;
                v102 = v95 + v101;
                assert("Tensor range check" && 0 <= v80 && v80 < 1l);
                assert("Tensor range check" && 0 <= v82 && v82 < 4l);
                int v103;
                v103 = 4l * v80;
                int v104;
                v104 = v103 + v82;
                v72[v104] = v102;
                v82 += 1l ;
            }
            v80 += 1l ;
        }
        bool v105;
        v105 = 0l <= v60;
        bool v106;
        v106 = v105 && v61;
        bool v107;
        v107 = v106 == false;
        if (v107){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v106);
        } else {
        }
        bool v109;
        v109 = 0l <= v67;
        bool v111;
        if (v109){
            bool v110;
            v110 = v67 < 64l;
            v111 = v110;
        } else {
            v111 = false;
        }
        bool v112;
        v112 = v111 == false;
        if (v112){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v111);
        } else {
        }
        int v114;
        v114 = v67 * 2l;
        int v115;
        v115 = v114 + v60;
        assert("Tensor range check" && 0 <= v67 && v67 < 64l);
        int v116;
        v116 = 0l;
        while (while_method_3(v116)){
            assert("Tensor range check" && 0 <= v116 && v116 < 1l);
            int v118;
            v118 = 64l * v116;
            int v119;
            v119 = v118 + v70;
            assert("Tensor range check" && 0 <= v116 && v116 < 1l);
            int v120;
            v120 = 4l * v116;
            int4* v121;
            v121 = reinterpret_cast<int4*>(v71 + v120);
            int4* v122;
            v122 = reinterpret_cast<int4*>(v3 + v119);
            assert("Pointer alignment check" && (unsigned long long)(v121) % 4l == 0 && (unsigned long long)(v122) % 4l == 0);
            *v122 = *v121;
            v116 += 1l ;
        }
        v67 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v123;
    v123 = threadIdx.x;
    bool v124;
    v124 = 0l <= v123;
    bool v125;
    v125 = v124 == false;
    if (v125){
        assert("The index needs to be zero or positive." && v124);
    } else {
    }
    int v127;
    v127 = v123 % 16l;
    int v128;
    v128 = v123 / 16l;
    bool v129;
    v129 = v128 < 2l;
    bool v130;
    v130 = v129 == false;
    if (v130){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v129);
    } else {
    }
    assert("Tensor range check" && 0 <= v128 && v128 < 2l);
    assert("Tensor range check" && 0 <= v127 && v127 < 16l);
    int v132;
    v132 = 4l * v127;
    int v133;
    v133 = 64l * v128;
    int v134;
    v134 = v133 + v132;
    assert("Tensor range check" && 0 <= v128 && v128 < 2l);
    assert("Tensor range check" && 0 <= v127 && v127 < 16l);
    int v135;
    v135 = 0l;
    while (while_method_2(v135)){
        assert("Tensor range check" && 0 <= v135 && v135 < 64l);
        int v137;
        v137 = 128l * v135;
        int v138;
        v138 = v137 + v134;
        float v139[4l];
        int v140[4l];
        int v141;
        v141 = 0l;
        while (while_method_3(v141)){
            assert("Tensor range check" && 0 <= v141 && v141 < 1l);
            int v143;
            v143 = 4l * v141;
            assert("Tensor range check" && 0 <= v141 && v141 < 1l);
            int v144;
            v144 = 64l * v141;
            int v145;
            v145 = v144 + v138;
            int4* v146;
            v146 = reinterpret_cast<int4*>(v1 + v145);
            int4* v147;
            v147 = reinterpret_cast<int4*>(v139 + v143);
            assert("Pointer alignment check" && (unsigned long long)(v146) % 4l == 0 && (unsigned long long)(v147) % 4l == 0);
            *v147 = *v146;
            v141 += 1l ;
        }
        int v148;
        v148 = 0l;
        while (while_method_3(v148)){
            int v150;
            v150 = 0l;
            while (while_method_1(v150)){
                bool v152;
                v152 = 0l <= v150;
                bool v154;
                if (v152){
                    bool v153;
                    v153 = v150 < 4l;
                    v154 = v153;
                } else {
                    v154 = false;
                }
                bool v155;
                v155 = v154 == false;
                if (v155){
                    assert("The indices should be inside the range of the dimension." && v154);
                } else {
                }
                bool v157;
                v157 = 0l <= v127;
                bool v159;
                if (v157){
                    bool v158;
                    v158 = v127 < 16l;
                    v159 = v158;
                } else {
                    v159 = false;
                }
                bool v160;
                v160 = v159 == false;
                if (v160){
                    assert("The indices should be inside the range of the dimension." && v159);
                } else {
                }
                int v162;
                v162 = v127 * 4l;
                int v163;
                v163 = v150 + v162;
                bool v164;
                v164 = 0l <= v148;
                bool v166;
                if (v164){
                    bool v165;
                    v165 = v148 < 1l;
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
                int v169;
                v169 = v148 * 64l;
                int v170;
                v170 = v163 + v169;
                assert("Tensor range check" && 0 <= v148 && v148 < 1l);
                assert("Tensor range check" && 0 <= v150 && v150 < 4l);
                int v171;
                v171 = 4l * v148;
                int v172;
                v172 = v171 + v150;
                v140[v172] = v170;
                v150 += 1l ;
            }
            v148 += 1l ;
        }
        bool v173;
        v173 = 0l <= v128;
        bool v174;
        v174 = v173 && v129;
        bool v175;
        v175 = v174 == false;
        if (v175){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v174);
        } else {
        }
        bool v177;
        v177 = 0l <= v135;
        bool v179;
        if (v177){
            bool v178;
            v178 = v135 < 64l;
            v179 = v178;
        } else {
            v179 = false;
        }
        bool v180;
        v180 = v179 == false;
        if (v180){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v179);
        } else {
        }
        int v182;
        v182 = v135 * 2l;
        int v183;
        v183 = v182 + v128;
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
                v192 = v140[v191];
                assert("Tensor range check" && 0 <= v186 && v186 < 1l);
                assert("Tensor range check" && 0 <= v188 && v188 < 4l);
                v184[v191] = v183;
                v185[v191] = v192;
                v188 += 1l ;
            }
            v186 += 1l ;
        }
        assert("Tensor range check" && 0 <= v135 && v135 < 64l);
        int v193;
        v193 = 0l;
        while (while_method_3(v193)){
            assert("Tensor range check" && 0 <= v193 && v193 < 1l);
            int v195;
            v195 = 64l * v193;
            int v196;
            v196 = v195 + v138;
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
        v135 += 1l ;
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
    v206 = v202 % 16l;
    int v207;
    v207 = v202 / 16l;
    bool v208;
    v208 = v207 < 2l;
    bool v209;
    v209 = v208 == false;
    if (v209){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v208);
    } else {
    }
    assert("Tensor range check" && 0 <= v207 && v207 < 2l);
    assert("Tensor range check" && 0 <= v206 && v206 < 16l);
    int v211;
    v211 = 4l * v206;
    int v212;
    v212 = 64l * v207;
    int v213;
    v213 = v212 + v211;
    assert("Tensor range check" && 0 <= v207 && v207 < 2l);
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
            v223 = 64l * v220;
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
                    v237 = v206 < 16l;
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
                v248 = v227 * 64l;
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
        v261 = v214 * 2l;
        int v262;
        v262 = v261 + v207;
        assert("Tensor range check" && 0 <= v214 && v214 < 64l);
        int v263;
        v263 = 2l * v214;
        int v264;
        v264 = v263 + v207;
        v12[v264] = v262;
        v214 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
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
    v277 = 0l;
    while (while_method_2(v277)){
        assert("Tensor range check" && 0 <= v277 && v277 < 64l);
        int v279;
        v279 = 128l * v277;
        int v280;
        v280 = v279 + v276;
        float v281[4l];
        int v282[4l];
        int v283;
        v283 = 0l;
        while (while_method_3(v283)){
            assert("Tensor range check" && 0 <= v283 && v283 < 1l);
            int v285;
            v285 = 4l * v283;
            assert("Tensor range check" && 0 <= v283 && v283 < 1l);
            int v286;
            v286 = 64l * v283;
            int v287;
            v287 = v286 + v280;
            int4* v288;
            v288 = reinterpret_cast<int4*>(v1 + v287);
            int4* v289;
            v289 = reinterpret_cast<int4*>(v281 + v285);
            assert("Pointer alignment check" && (unsigned long long)(v288) % 4l == 0 && (unsigned long long)(v289) % 4l == 0);
            *v289 = *v288;
            v283 += 1l ;
        }
        int v290;
        v290 = 0l;
        while (while_method_3(v290)){
            int v292;
            v292 = 0l;
            while (while_method_1(v292)){
                bool v294;
                v294 = 0l <= v292;
                bool v296;
                if (v294){
                    bool v295;
                    v295 = v292 < 4l;
                    v296 = v295;
                } else {
                    v296 = false;
                }
                bool v297;
                v297 = v296 == false;
                if (v297){
                    assert("The indices should be inside the range of the dimension." && v296);
                } else {
                }
                bool v299;
                v299 = 0l <= v269;
                bool v301;
                if (v299){
                    bool v300;
                    v300 = v269 < 16l;
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
                int v304;
                v304 = v269 * 4l;
                int v305;
                v305 = v292 + v304;
                bool v306;
                v306 = 0l <= v290;
                bool v308;
                if (v306){
                    bool v307;
                    v307 = v290 < 1l;
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
                v311 = v290 * 64l;
                int v312;
                v312 = v305 + v311;
                assert("Tensor range check" && 0 <= v290 && v290 < 1l);
                assert("Tensor range check" && 0 <= v292 && v292 < 4l);
                int v313;
                v313 = 4l * v290;
                int v314;
                v314 = v313 + v292;
                v282[v314] = v312;
                v292 += 1l ;
            }
            v290 += 1l ;
        }
        bool v315;
        v315 = 0l <= v270;
        bool v316;
        v316 = v315 && v271;
        bool v317;
        v317 = v316 == false;
        if (v317){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v316);
        } else {
        }
        bool v319;
        v319 = 0l <= v277;
        bool v321;
        if (v319){
            bool v320;
            v320 = v277 < 64l;
            v321 = v320;
        } else {
            v321 = false;
        }
        bool v322;
        v322 = v321 == false;
        if (v322){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v321);
        } else {
        }
        int v324;
        v324 = v277 * 2l;
        int v325;
        v325 = v324 + v270;
        float v326;
        v326 = 0.0f;
        int v327;
        v327 = 0l;
        while (while_method_3(v327)){
            int v329;
            v329 = 0l;
            while (while_method_1(v329)){
                assert("Tensor range check" && 0 <= v327 && v327 < 1l);
                assert("Tensor range check" && 0 <= v329 && v329 < 4l);
                int v331;
                v331 = 4l * v327;
                int v332;
                v332 = v331 + v329;
                float v333;
                v333 = v281[v332];
                float v334;
                v334 = v326 + v333;
                v326 = v334;
                v329 += 1l ;
            }
            v327 += 1l ;
        }
        auto v335 = cooperative_groups::coalesced_threads();
        int v336;
        v336 = threadIdx.x;
        int v337;
        v337 = v336 / 16l;
        auto v338 = cooperative_groups::labeled_partition(v335,v337);
        float v339;
        v339 = cooperative_groups::reduce(v338, v326, v40);
        float v340;
        v340 = v339 / 64.0f;
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
                v348 = v281[v347];
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
        v362 = v361 / 16l;
        auto v363 = cooperative_groups::labeled_partition(v360,v362);
        float v364;
        v364 = cooperative_groups::reduce(v363, v351, v40);
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
        assert("Tensor range check" && 0 <= v277 && v277 < 64l);
        int v374;
        v374 = 0l;
        while (while_method_3(v374)){
            assert("Tensor range check" && 0 <= v374 && v374 < 1l);
            int v376;
            v376 = 64l * v374;
            int v377;
            v377 = v376 + v280;
            assert("Tensor range check" && 0 <= v374 && v374 < 1l);
            int v378;
            v378 = 4l * v374;
            int4* v379;
            v379 = reinterpret_cast<int4*>(v365 + v378);
            int4* v380;
            v380 = reinterpret_cast<int4*>(v4 + v377);
            assert("Pointer alignment check" && (unsigned long long)(v379) % 4l == 0 && (unsigned long long)(v380) % 4l == 0);
            *v380 = *v379;
            v374 += 1l ;
        }
        v277 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
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
    v385 = v381 % 16l;
    int v386;
    v386 = v381 / 16l;
    bool v387;
    v387 = v386 < 2l;
    bool v388;
    v388 = v387 == false;
    if (v388){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v387);
    } else {
    }
    assert("Tensor range check" && 0 <= v386 && v386 < 2l);
    assert("Tensor range check" && 0 <= v385 && v385 < 16l);
    int v390;
    v390 = 4l * v385;
    int v391;
    v391 = 64l * v386;
    int v392;
    v392 = v391 + v390;
    assert("Tensor range check" && 0 <= v386 && v386 < 2l);
    assert("Tensor range check" && 0 <= v385 && v385 < 16l);
    int v393;
    v393 = 0l;
    while (while_method_2(v393)){
        assert("Tensor range check" && 0 <= v393 && v393 < 64l);
        int v395;
        v395 = 128l * v393;
        int v396;
        v396 = v395 + v392;
        float v397[4l];
        int v398[4l];
        int v399;
        v399 = 0l;
        while (while_method_3(v399)){
            assert("Tensor range check" && 0 <= v399 && v399 < 1l);
            int v401;
            v401 = 4l * v399;
            assert("Tensor range check" && 0 <= v399 && v399 < 1l);
            int v402;
            v402 = 64l * v399;
            int v403;
            v403 = v402 + v396;
            int4* v404;
            v404 = reinterpret_cast<int4*>(v1 + v403);
            int4* v405;
            v405 = reinterpret_cast<int4*>(v397 + v401);
            assert("Pointer alignment check" && (unsigned long long)(v404) % 4l == 0 && (unsigned long long)(v405) % 4l == 0);
            *v405 = *v404;
            v399 += 1l ;
        }
        int v406;
        v406 = 0l;
        while (while_method_3(v406)){
            int v408;
            v408 = 0l;
            while (while_method_1(v408)){
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
                v415 = 0l <= v385;
                bool v417;
                if (v415){
                    bool v416;
                    v416 = v385 < 16l;
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
                v420 = v385 * 4l;
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
                v427 = v406 * 64l;
                int v428;
                v428 = v421 + v427;
                assert("Tensor range check" && 0 <= v406 && v406 < 1l);
                assert("Tensor range check" && 0 <= v408 && v408 < 4l);
                int v429;
                v429 = 4l * v406;
                int v430;
                v430 = v429 + v408;
                v398[v430] = v428;
                v408 += 1l ;
            }
            v406 += 1l ;
        }
        bool v431;
        v431 = 0l <= v386;
        bool v432;
        v432 = v431 && v387;
        bool v433;
        v433 = v432 == false;
        if (v433){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v432);
        } else {
        }
        bool v435;
        v435 = 0l <= v393;
        bool v437;
        if (v435){
            bool v436;
            v436 = v393 < 64l;
            v437 = v436;
        } else {
            v437 = false;
        }
        bool v438;
        v438 = v437 == false;
        if (v438){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v437);
        } else {
        }
        int v440;
        v440 = v393 * 2l;
        int v441;
        v441 = v440 + v386;
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
                v449 = v397[v448];
                float v450;
                v450 = v449 * v449;
                assert("Tensor range check" && 0 <= v443 && v443 < 1l);
                assert("Tensor range check" && 0 <= v445 && v445 < 4l);
                v442[v448] = v450;
                v445 += 1l ;
            }
            v443 += 1l ;
        }
        float v451;
        v451 = 0.0f;
        int v452;
        v452 = 0l;
        while (while_method_3(v452)){
            int v454;
            v454 = 0l;
            while (while_method_1(v454)){
                assert("Tensor range check" && 0 <= v452 && v452 < 1l);
                assert("Tensor range check" && 0 <= v454 && v454 < 4l);
                int v456;
                v456 = 4l * v452;
                int v457;
                v457 = v456 + v454;
                float v458;
                v458 = v442[v457];
                float v459;
                v459 = v451 + v458;
                v451 = v459;
                v454 += 1l ;
            }
            v452 += 1l ;
        }
        auto v460 = cooperative_groups::coalesced_threads();
        int v461;
        v461 = threadIdx.x;
        int v462;
        v462 = v461 / 16l;
        auto v463 = cooperative_groups::labeled_partition(v460,v462);
        float v464;
        v464 = cooperative_groups::reduce(v463, v451, v40);
        float v465[4l];
        int v466;
        v466 = 0l;
        while (while_method_3(v466)){
            int v468;
            v468 = 0l;
            while (while_method_1(v468)){
                assert("Tensor range check" && 0 <= v466 && v466 < 1l);
                assert("Tensor range check" && 0 <= v468 && v468 < 4l);
                int v470;
                v470 = 4l * v466;
                int v471;
                v471 = v470 + v468;
                float v472;
                v472 = v397[v471];
                bool v473;
                v473 = v464 == 0.0f;
                bool v474;
                v474 = v473 != true;
                float v476;
                if (v474){
                    float v475;
                    v475 = v472 / v464;
                    v476 = v475;
                } else {
                    v476 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v466 && v466 < 1l);
                assert("Tensor range check" && 0 <= v468 && v468 < 4l);
                v465[v471] = v476;
                v468 += 1l ;
            }
            v466 += 1l ;
        }
        assert("Tensor range check" && 0 <= v393 && v393 < 64l);
        int v477;
        v477 = 0l;
        while (while_method_3(v477)){
            assert("Tensor range check" && 0 <= v477 && v477 < 1l);
            int v479;
            v479 = 64l * v477;
            int v480;
            v480 = v479 + v396;
            assert("Tensor range check" && 0 <= v477 && v477 < 1l);
            int v481;
            v481 = 4l * v477;
            int4* v482;
            v482 = reinterpret_cast<int4*>(v465 + v481);
            int4* v483;
            v483 = reinterpret_cast<int4*>(v8 + v480);
            assert("Pointer alignment check" && (unsigned long long)(v482) % 4l == 0 && (unsigned long long)(v483) % 4l == 0);
            *v483 = *v482;
            v477 += 1l ;
        }
        v393 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v484;
    v484 = threadIdx.x;
    bool v485;
    v485 = 0l <= v484;
    bool v486;
    v486 = v485 == false;
    if (v486){
        assert("The index needs to be zero or positive." && v485);
    } else {
    }
    int v488;
    v488 = v484 % 16l;
    int v489;
    v489 = v484 / 16l;
    bool v490;
    v490 = v489 < 2l;
    bool v491;
    v491 = v490 == false;
    if (v491){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v490);
    } else {
    }
    assert("Tensor range check" && 0 <= v489 && v489 < 2l);
    assert("Tensor range check" && 0 <= v488 && v488 < 16l);
    int v493;
    v493 = 4l * v488;
    int v494;
    v494 = 64l * v489;
    int v495;
    v495 = v494 + v493;
    assert("Tensor range check" && 0 <= v489 && v489 < 2l);
    int v496;
    v496 = 0l;
    while (while_method_2(v496)){
        assert("Tensor range check" && 0 <= v496 && v496 < 64l);
        int v498;
        v498 = 128l * v496;
        int v499;
        v499 = v498 + v495;
        float v500[4l];
        int v501[4l];
        int v502;
        v502 = 0l;
        while (while_method_3(v502)){
            assert("Tensor range check" && 0 <= v502 && v502 < 1l);
            int v504;
            v504 = 4l * v502;
            assert("Tensor range check" && 0 <= v502 && v502 < 1l);
            int v505;
            v505 = 64l * v502;
            int v506;
            v506 = v505 + v499;
            int4* v507;
            v507 = reinterpret_cast<int4*>(v1 + v506);
            int4* v508;
            v508 = reinterpret_cast<int4*>(v500 + v504);
            assert("Pointer alignment check" && (unsigned long long)(v507) % 4l == 0 && (unsigned long long)(v508) % 4l == 0);
            *v508 = *v507;
            v502 += 1l ;
        }
        int v509;
        v509 = 0l;
        while (while_method_3(v509)){
            int v511;
            v511 = 0l;
            while (while_method_1(v511)){
                bool v513;
                v513 = 0l <= v511;
                bool v515;
                if (v513){
                    bool v514;
                    v514 = v511 < 4l;
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
                bool v518;
                v518 = 0l <= v488;
                bool v520;
                if (v518){
                    bool v519;
                    v519 = v488 < 16l;
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
                v523 = v488 * 4l;
                int v524;
                v524 = v511 + v523;
                bool v525;
                v525 = 0l <= v509;
                bool v527;
                if (v525){
                    bool v526;
                    v526 = v509 < 1l;
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
                int v530;
                v530 = v509 * 64l;
                int v531;
                v531 = v524 + v530;
                assert("Tensor range check" && 0 <= v509 && v509 < 1l);
                assert("Tensor range check" && 0 <= v511 && v511 < 4l);
                int v532;
                v532 = 4l * v509;
                int v533;
                v533 = v532 + v511;
                v501[v533] = v531;
                v511 += 1l ;
            }
            v509 += 1l ;
        }
        bool v534;
        v534 = 0l <= v489;
        bool v535;
        v535 = v534 && v490;
        bool v536;
        v536 = v535 == false;
        if (v536){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v535);
        } else {
        }
        bool v538;
        v538 = 0l <= v496;
        bool v540;
        if (v538){
            bool v539;
            v539 = v496 < 64l;
            v540 = v539;
        } else {
            v540 = false;
        }
        bool v541;
        v541 = v540 == false;
        if (v541){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v540);
        } else {
        }
        int v543;
        v543 = v496 * 2l;
        int v544;
        v544 = v543 + v489;
        float v545; int v546;
        Tuple1 tmp24 = Tuple1{-1.0f / 0.0f, 0l};
        v545 = tmp24.v0; v546 = tmp24.v1;
        int v547;
        v547 = 0l;
        while (while_method_3(v547)){
            int v549;
            v549 = 0l;
            while (while_method_1(v549)){
                assert("Tensor range check" && 0 <= v547 && v547 < 1l);
                assert("Tensor range check" && 0 <= v549 && v549 < 4l);
                int v551;
                v551 = 4l * v547;
                int v552;
                v552 = v551 + v549;
                float v553;
                v553 = v500[v552];
                int v554;
                v554 = v501[v552];
                bool v555;
                v555 = v545 > v553;
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
        int v560;
        v560 = v559 / 16l;
        auto v561 = cooperative_groups::labeled_partition(v558,v560);
        Closure1 v562{};
        float v563; int v564;
        Tuple1 tmp25 = cooperative_groups::reduce(v561, Tuple1{v545, v546}, v562);
        v563 = tmp25.v0; v564 = tmp25.v1;
        assert("Tensor range check" && 0 <= v496 && v496 < 64l);
        int v565;
        v565 = 2l * v496;
        int v566;
        v566 = v565 + v489;
        v9[v566] = v564;
        v496 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v567;
    v567 = threadIdx.x;
    bool v568;
    v568 = 0l <= v567;
    bool v569;
    v569 = v568 == false;
    if (v569){
        assert("The index needs to be zero or positive." && v568);
    } else {
    }
    int v571;
    v571 = v567 % 16l;
    int v572;
    v572 = v567 / 16l;
    bool v573;
    v573 = v572 < 2l;
    bool v574;
    v574 = v573 == false;
    if (v574){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v573);
    } else {
    }
    assert("Tensor range check" && 0 <= v572 && v572 < 2l);
    assert("Tensor range check" && 0 <= v571 && v571 < 16l);
    int v576;
    v576 = 4l * v571;
    int v577;
    v577 = 64l * v572;
    int v578;
    v578 = v577 + v576;
    assert("Tensor range check" && 0 <= v572 && v572 < 2l);
    assert("Tensor range check" && 0 <= v571 && v571 < 16l);
    int v579;
    v579 = 0l;
    while (while_method_2(v579)){
        assert("Tensor range check" && 0 <= v579 && v579 < 64l);
        int v581;
        v581 = 128l * v579;
        int v582;
        v582 = v581 + v578;
        float v583[4l];
        int v584[4l];
        int v585;
        v585 = 0l;
        while (while_method_3(v585)){
            assert("Tensor range check" && 0 <= v585 && v585 < 1l);
            int v587;
            v587 = 4l * v585;
            assert("Tensor range check" && 0 <= v585 && v585 < 1l);
            int v588;
            v588 = 64l * v585;
            int v589;
            v589 = v588 + v582;
            int4* v590;
            v590 = reinterpret_cast<int4*>(v1 + v589);
            int4* v591;
            v591 = reinterpret_cast<int4*>(v583 + v587);
            assert("Pointer alignment check" && (unsigned long long)(v590) % 4l == 0 && (unsigned long long)(v591) % 4l == 0);
            *v591 = *v590;
            v585 += 1l ;
        }
        int v592;
        v592 = 0l;
        while (while_method_3(v592)){
            int v594;
            v594 = 0l;
            while (while_method_1(v594)){
                bool v596;
                v596 = 0l <= v594;
                bool v598;
                if (v596){
                    bool v597;
                    v597 = v594 < 4l;
                    v598 = v597;
                } else {
                    v598 = false;
                }
                bool v599;
                v599 = v598 == false;
                if (v599){
                    assert("The indices should be inside the range of the dimension." && v598);
                } else {
                }
                bool v601;
                v601 = 0l <= v571;
                bool v603;
                if (v601){
                    bool v602;
                    v602 = v571 < 16l;
                    v603 = v602;
                } else {
                    v603 = false;
                }
                bool v604;
                v604 = v603 == false;
                if (v604){
                    assert("The indices should be inside the range of the dimension." && v603);
                } else {
                }
                int v606;
                v606 = v571 * 4l;
                int v607;
                v607 = v594 + v606;
                bool v608;
                v608 = 0l <= v592;
                bool v610;
                if (v608){
                    bool v609;
                    v609 = v592 < 1l;
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
                int v613;
                v613 = v592 * 64l;
                int v614;
                v614 = v607 + v613;
                assert("Tensor range check" && 0 <= v592 && v592 < 1l);
                assert("Tensor range check" && 0 <= v594 && v594 < 4l);
                int v615;
                v615 = 4l * v592;
                int v616;
                v616 = v615 + v594;
                v584[v616] = v614;
                v594 += 1l ;
            }
            v592 += 1l ;
        }
        bool v617;
        v617 = 0l <= v572;
        bool v618;
        v618 = v617 && v573;
        bool v619;
        v619 = v618 == false;
        if (v619){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v618);
        } else {
        }
        bool v621;
        v621 = 0l <= v579;
        bool v623;
        if (v621){
            bool v622;
            v622 = v579 < 64l;
            v623 = v622;
        } else {
            v623 = false;
        }
        bool v624;
        v624 = v623 == false;
        if (v624){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v623);
        } else {
        }
        int v626;
        v626 = v579 * 2l;
        int v627;
        v627 = v626 + v572;
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
                v635 = v583[v634];
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
        float v641;
        v641 = cooperative_groups::reduce(v640, v628, v40);
        float v642;
        v642 = v641 / 64.0f;
        float v643[4l];
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
                v650 = v583[v649];
                float v651;
                v651 = v650 - v642;
                float v652;
                v652 = exp(v651);
                assert("Tensor range check" && 0 <= v644 && v644 < 1l);
                assert("Tensor range check" && 0 <= v646 && v646 < 4l);
                v643[v649] = v652;
                v646 += 1l ;
            }
            v644 += 1l ;
        }
        float v653;
        v653 = 0.0f;
        int v654;
        v654 = 0l;
        while (while_method_3(v654)){
            int v656;
            v656 = 0l;
            while (while_method_1(v656)){
                assert("Tensor range check" && 0 <= v654 && v654 < 1l);
                assert("Tensor range check" && 0 <= v656 && v656 < 4l);
                int v658;
                v658 = 4l * v654;
                int v659;
                v659 = v658 + v656;
                float v660;
                v660 = v643[v659];
                float v661;
                v661 = v653 + v660;
                v653 = v661;
                v656 += 1l ;
            }
            v654 += 1l ;
        }
        auto v662 = cooperative_groups::coalesced_threads();
        int v663;
        v663 = threadIdx.x;
        int v664;
        v664 = v663 / 16l;
        auto v665 = cooperative_groups::labeled_partition(v662,v664);
        float v666;
        v666 = cooperative_groups::reduce(v665, v653, v40);
        float v667[4l];
        int v668;
        v668 = 0l;
        while (while_method_3(v668)){
            int v670;
            v670 = 0l;
            while (while_method_1(v670)){
                assert("Tensor range check" && 0 <= v668 && v668 < 1l);
                assert("Tensor range check" && 0 <= v670 && v670 < 4l);
                int v672;
                v672 = 4l * v668;
                int v673;
                v673 = v672 + v670;
                float v674;
                v674 = v643[v673];
                float v675;
                v675 = v674 / v666;
                assert("Tensor range check" && 0 <= v668 && v668 < 1l);
                assert("Tensor range check" && 0 <= v670 && v670 < 4l);
                v667[v673] = v675;
                v670 += 1l ;
            }
            v668 += 1l ;
        }
        float v676[4l];
        float v677;
        v677 = 0.0f;
        int v678;
        v678 = 0l;
        while (while_method_3(v678)){
            assert("Tensor range check" && 0 <= v678 && v678 < 1l);
            int v680;
            v680 = 4l * v678;
            assert("Tensor range check" && 0 <= v678 && v678 < 1l);
            int v681; float v682;
            Tuple0 tmp26 = Tuple0{0l, 0.0f};
            v681 = tmp26.v0; v682 = tmp26.v1;
            while (while_method_1(v681)){
                assert("Tensor range check" && 0 <= v681 && v681 < 4l);
                int v684;
                v684 = v681 + v680;
                float v685;
                v685 = v667[v684];
                float v686;
                v686 = v682 + v685;
                v682 = v686;
                v681 += 1l ;
            }
            auto v687 = cooperative_groups::coalesced_threads();
            int v688;
            v688 = threadIdx.x;
            int v689;
            v689 = v688 / 16l;
            auto v690 = cooperative_groups::labeled_partition(v687,v689);
            Closure2 v691{};
            float v692;
            v692 = cooperative_groups::inclusive_scan(v690, v682, v691);
            float v693;
            v693 = v690.shfl_up(v692,1);
            bool v694;
            v694 = v690.thread_rank() == 0;
            float v695;
            if (v694){
                v695 = 0.0f;
            } else {
                v695 = v693;
            }
            float v696;
            v696 = v690.shfl(v692,v690.num_threads()-1);
            float v697;
            v697 = v677 + v695;
            int v698; float v699;
            Tuple0 tmp27 = Tuple0{0l, v697};
            v698 = tmp27.v0; v699 = tmp27.v1;
            while (while_method_1(v698)){
                assert("Tensor range check" && 0 <= v698 && v698 < 4l);
                int v701;
                v701 = v698 + v680;
                float v702;
                v702 = v667[v701];
                float v703;
                v703 = v699 + v702;
                assert("Tensor range check" && 0 <= v698 && v698 < 4l);
                v676[v701] = v703;
                v699 = v703;
                v698 += 1l ;
            }
            float v704;
            v704 = v677 + v696;
            v677 = v704;
            v678 += 1l ;
        }
        assert("Tensor range check" && 0 <= v579 && v579 < 64l);
        int v705;
        v705 = 0l;
        while (while_method_3(v705)){
            assert("Tensor range check" && 0 <= v705 && v705 < 1l);
            int v707;
            v707 = 64l * v705;
            int v708;
            v708 = v707 + v582;
            assert("Tensor range check" && 0 <= v705 && v705 < 1l);
            int v709;
            v709 = 4l * v705;
            int4* v710;
            v710 = reinterpret_cast<int4*>(v667 + v709);
            int4* v711;
            v711 = reinterpret_cast<int4*>(v6 + v708);
            assert("Pointer alignment check" && (unsigned long long)(v710) % 4l == 0 && (unsigned long long)(v711) % 4l == 0);
            *v711 = *v710;
            int4* v712;
            v712 = reinterpret_cast<int4*>(v676 + v709);
            int4* v713;
            v713 = reinterpret_cast<int4*>(v7 + v708);
            assert("Pointer alignment check" && (unsigned long long)(v712) % 4l == 0 && (unsigned long long)(v713) % 4l == 0);
            *v713 = *v712;
            v705 += 1l ;
        }
        v579 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v714;
    v714 = threadIdx.x;
    bool v715;
    v715 = 0l <= v714;
    bool v716;
    v716 = v715 == false;
    if (v716){
        assert("The index needs to be zero or positive." && v715);
    } else {
    }
    int v718;
    v718 = v714 % 16l;
    int v719;
    v719 = v714 / 16l;
    bool v720;
    v720 = v719 < 2l;
    bool v721;
    v721 = v720 == false;
    if (v721){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v720);
    } else {
    }
    assert("Tensor range check" && 0 <= v719 && v719 < 2l);
    assert("Tensor range check" && 0 <= v718 && v718 < 16l);
    int v723;
    v723 = 4l * v718;
    int v724;
    v724 = 64l * v719;
    int v725;
    v725 = v724 + v723;
    assert("Tensor range check" && 0 <= v719 && v719 < 2l);
    assert("Tensor range check" && 0 <= v718 && v718 < 16l);
    int v726;
    v726 = 0l;
    while (while_method_2(v726)){
        assert("Tensor range check" && 0 <= v726 && v726 < 64l);
        int v728;
        v728 = 128l * v726;
        int v729;
        v729 = v728 + v725;
        int v730[4l];
        int v731[4l];
        int v732;
        v732 = 0l;
        while (while_method_3(v732)){
            assert("Tensor range check" && 0 <= v732 && v732 < 1l);
            int v734;
            v734 = 4l * v732;
            assert("Tensor range check" && 0 <= v732 && v732 < 1l);
            int v735;
            v735 = 64l * v732;
            int v736;
            v736 = v735 + v729;
            int4* v737;
            v737 = reinterpret_cast<int4*>(v0 + v736);
            int4* v738;
            v738 = reinterpret_cast<int4*>(v730 + v734);
            assert("Pointer alignment check" && (unsigned long long)(v737) % 4l == 0 && (unsigned long long)(v738) % 4l == 0);
            *v738 = *v737;
            v732 += 1l ;
        }
        int v739;
        v739 = 0l;
        while (while_method_3(v739)){
            int v741;
            v741 = 0l;
            while (while_method_1(v741)){
                bool v743;
                v743 = 0l <= v741;
                bool v745;
                if (v743){
                    bool v744;
                    v744 = v741 < 4l;
                    v745 = v744;
                } else {
                    v745 = false;
                }
                bool v746;
                v746 = v745 == false;
                if (v746){
                    assert("The indices should be inside the range of the dimension." && v745);
                } else {
                }
                bool v748;
                v748 = 0l <= v718;
                bool v750;
                if (v748){
                    bool v749;
                    v749 = v718 < 16l;
                    v750 = v749;
                } else {
                    v750 = false;
                }
                bool v751;
                v751 = v750 == false;
                if (v751){
                    assert("The indices should be inside the range of the dimension." && v750);
                } else {
                }
                int v753;
                v753 = v718 * 4l;
                int v754;
                v754 = v741 + v753;
                bool v755;
                v755 = 0l <= v739;
                bool v757;
                if (v755){
                    bool v756;
                    v756 = v739 < 1l;
                    v757 = v756;
                } else {
                    v757 = false;
                }
                bool v758;
                v758 = v757 == false;
                if (v758){
                    assert("The indices should be inside the range of the dimension." && v757);
                } else {
                }
                int v760;
                v760 = v739 * 64l;
                int v761;
                v761 = v754 + v760;
                assert("Tensor range check" && 0 <= v739 && v739 < 1l);
                assert("Tensor range check" && 0 <= v741 && v741 < 4l);
                int v762;
                v762 = 4l * v739;
                int v763;
                v763 = v762 + v741;
                v731[v763] = v761;
                v741 += 1l ;
            }
            v739 += 1l ;
        }
        bool v764;
        v764 = 0l <= v719;
        bool v765;
        v765 = v764 && v720;
        bool v766;
        v766 = v765 == false;
        if (v766){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v765);
        } else {
        }
        bool v768;
        v768 = 0l <= v726;
        bool v770;
        if (v768){
            bool v769;
            v769 = v726 < 64l;
            v770 = v769;
        } else {
            v770 = false;
        }
        bool v771;
        v771 = v770 == false;
        if (v771){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v770);
        } else {
        }
        int v773;
        v773 = v726 * 2l;
        int v774;
        v774 = v773 + v719;
        int v775[4l];
        int v776;
        v776 = 0l;
        int v777;
        v777 = 0l;
        while (while_method_3(v777)){
            assert("Tensor range check" && 0 <= v777 && v777 < 1l);
            int v779;
            v779 = 4l * v777;
            assert("Tensor range check" && 0 <= v777 && v777 < 1l);
            int v780; int v781;
            Tuple2 tmp28 = Tuple2{0l, 0l};
            v780 = tmp28.v0; v781 = tmp28.v1;
            while (while_method_1(v780)){
                assert("Tensor range check" && 0 <= v780 && v780 < 4l);
                int v783;
                v783 = v780 + v779;
                int v784;
                v784 = v730[v783];
                int v785;
                v785 = v781 + v784;
                v781 = v785;
                v780 += 1l ;
            }
            auto v786 = cooperative_groups::coalesced_threads();
            int v787;
            v787 = threadIdx.x;
            int v788;
            v788 = v787 / 16l;
            auto v789 = cooperative_groups::labeled_partition(v786,v788);
            Closure3 v790{};
            int v791;
            v791 = cooperative_groups::inclusive_scan(v789, v781, v790);
            int v792;
            v792 = v789.shfl_up(v791,1);
            bool v793;
            v793 = v789.thread_rank() == 0;
            int v794;
            if (v793){
                v794 = 0l;
            } else {
                v794 = v792;
            }
            int v795;
            v795 = v789.shfl(v791,v789.num_threads()-1);
            int v796;
            v796 = v776 + v794;
            int v797; int v798;
            Tuple2 tmp29 = Tuple2{0l, v796};
            v797 = tmp29.v0; v798 = tmp29.v1;
            while (while_method_1(v797)){
                assert("Tensor range check" && 0 <= v797 && v797 < 4l);
                int v800;
                v800 = v797 + v779;
                int v801;
                v801 = v730[v800];
                assert("Tensor range check" && 0 <= v797 && v797 < 4l);
                v775[v800] = v798;
                int v802;
                v802 = v798 + v801;
                v798 = v802;
                v797 += 1l ;
            }
            int v803;
            v803 = v776 + v795;
            v776 = v803;
            v777 += 1l ;
        }
        assert("Tensor range check" && 0 <= v726 && v726 < 64l);
        int v804;
        v804 = 0l;
        while (while_method_3(v804)){
            assert("Tensor range check" && 0 <= v804 && v804 < 1l);
            int v806;
            v806 = 64l * v804;
            int v807;
            v807 = v806 + v729;
            assert("Tensor range check" && 0 <= v804 && v804 < 1l);
            int v808;
            v808 = 4l * v804;
            int4* v809;
            v809 = reinterpret_cast<int4*>(v775 + v808);
            int4* v810;
            v810 = reinterpret_cast<int4*>(v13 + v807);
            assert("Pointer alignment check" && (unsigned long long)(v809) % 4l == 0 && (unsigned long long)(v810) % 4l == 0);
            *v810 = *v809;
            v804 += 1l ;
        }
        v726 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v811;
    v811 = threadIdx.x;
    bool v812;
    v812 = 0l <= v811;
    bool v813;
    v813 = v812 == false;
    if (v813){
        assert("The index needs to be zero or positive." && v812);
    } else {
    }
    int v815;
    v815 = v811 % 16l;
    int v816;
    v816 = v811 / 16l;
    bool v817;
    v817 = v816 < 2l;
    bool v818;
    v818 = v817 == false;
    if (v818){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v817);
    } else {
    }
    assert("Tensor range check" && 0 <= v816 && v816 < 2l);
    assert("Tensor range check" && 0 <= v815 && v815 < 16l);
    int v820;
    v820 = 4l * v815;
    int v821;
    v821 = 64l * v816;
    int v822;
    v822 = v821 + v820;
    assert("Tensor range check" && 0 <= v816 && v816 < 2l);
    assert("Tensor range check" && 0 <= v815 && v815 < 16l);
    int v823;
    v823 = 0l;
    while (while_method_2(v823)){
        assert("Tensor range check" && 0 <= v823 && v823 < 64l);
        int v825;
        v825 = 128l * v823;
        int v826;
        v826 = v825 + v822;
        float v827[4l];
        int v828[4l];
        int v829;
        v829 = 0l;
        while (while_method_3(v829)){
            assert("Tensor range check" && 0 <= v829 && v829 < 1l);
            int v831;
            v831 = 4l * v829;
            assert("Tensor range check" && 0 <= v829 && v829 < 1l);
            int v832;
            v832 = 64l * v829;
            int v833;
            v833 = v832 + v826;
            int4* v834;
            v834 = reinterpret_cast<int4*>(v1 + v833);
            int4* v835;
            v835 = reinterpret_cast<int4*>(v827 + v831);
            assert("Pointer alignment check" && (unsigned long long)(v834) % 4l == 0 && (unsigned long long)(v835) % 4l == 0);
            *v835 = *v834;
            v829 += 1l ;
        }
        int v836;
        v836 = 0l;
        while (while_method_3(v836)){
            int v838;
            v838 = 0l;
            while (while_method_1(v838)){
                bool v840;
                v840 = 0l <= v838;
                bool v842;
                if (v840){
                    bool v841;
                    v841 = v838 < 4l;
                    v842 = v841;
                } else {
                    v842 = false;
                }
                bool v843;
                v843 = v842 == false;
                if (v843){
                    assert("The indices should be inside the range of the dimension." && v842);
                } else {
                }
                bool v845;
                v845 = 0l <= v815;
                bool v847;
                if (v845){
                    bool v846;
                    v846 = v815 < 16l;
                    v847 = v846;
                } else {
                    v847 = false;
                }
                bool v848;
                v848 = v847 == false;
                if (v848){
                    assert("The indices should be inside the range of the dimension." && v847);
                } else {
                }
                int v850;
                v850 = v815 * 4l;
                int v851;
                v851 = v838 + v850;
                bool v852;
                v852 = 0l <= v836;
                bool v854;
                if (v852){
                    bool v853;
                    v853 = v836 < 1l;
                    v854 = v853;
                } else {
                    v854 = false;
                }
                bool v855;
                v855 = v854 == false;
                if (v855){
                    assert("The indices should be inside the range of the dimension." && v854);
                } else {
                }
                int v857;
                v857 = v836 * 64l;
                int v858;
                v858 = v851 + v857;
                assert("Tensor range check" && 0 <= v836 && v836 < 1l);
                assert("Tensor range check" && 0 <= v838 && v838 < 4l);
                int v859;
                v859 = 4l * v836;
                int v860;
                v860 = v859 + v838;
                v828[v860] = v858;
                v838 += 1l ;
            }
            v836 += 1l ;
        }
        bool v861;
        v861 = 0l <= v816;
        bool v862;
        v862 = v861 && v817;
        bool v863;
        v863 = v862 == false;
        if (v863){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v862);
        } else {
        }
        bool v865;
        v865 = 0l <= v823;
        bool v867;
        if (v865){
            bool v866;
            v866 = v823 < 64l;
            v867 = v866;
        } else {
            v867 = false;
        }
        bool v868;
        v868 = v867 == false;
        if (v868){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v867);
        } else {
        }
        int v870;
        v870 = v823 * 2l;
        int v871;
        v871 = v870 + v816;
        bool v872[4l];
        int v873;
        v873 = 0l;
        while (while_method_3(v873)){
            int v875;
            v875 = 0l;
            while (while_method_1(v875)){
                assert("Tensor range check" && 0 <= v873 && v873 < 1l);
                assert("Tensor range check" && 0 <= v875 && v875 < 4l);
                int v877;
                v877 = 4l * v873;
                int v878;
                v878 = v877 + v875;
                float v879;
                v879 = v827[v878];
                int v880;
                v880 = v828[v878];
                bool v881;
                v881 = v880 < 4l;
                assert("Tensor range check" && 0 <= v873 && v873 < 1l);
                assert("Tensor range check" && 0 <= v875 && v875 < 4l);
                v872[v878] = v881;
                v875 += 1l ;
            }
            v873 += 1l ;
        }
        int v882[4l];
        int v883;
        v883 = 0l;
        while (while_method_3(v883)){
            int v885;
            v885 = 0l;
            while (while_method_1(v885)){
                assert("Tensor range check" && 0 <= v883 && v883 < 1l);
                assert("Tensor range check" && 0 <= v885 && v885 < 4l);
                int v887;
                v887 = 4l * v883;
                int v888;
                v888 = v887 + v885;
                bool v889;
                v889 = v872[v888];
                int v890;
                if (v889){
                    v890 = 1l;
                } else {
                    v890 = 0l;
                }
                assert("Tensor range check" && 0 <= v883 && v883 < 1l);
                assert("Tensor range check" && 0 <= v885 && v885 < 4l);
                v882[v888] = v890;
                v885 += 1l ;
            }
            v883 += 1l ;
        }
        int v891;
        v891 = 0l;
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
                int v898;
                v898 = v882[v897];
                int v899;
                v899 = v891 + v898;
                v891 = v899;
                v894 += 1l ;
            }
            v892 += 1l ;
        }
        auto v900 = cooperative_groups::coalesced_threads();
        int v901;
        v901 = threadIdx.x;
        int v902;
        v902 = v901 / 16l;
        auto v903 = cooperative_groups::labeled_partition(v900,v902);
        Closure4 v904{};
        int v905;
        v905 = cooperative_groups::reduce(v903, v891, v904);
        float v906[4l];
        int v907;
        v907 = 0l;
        while (while_method_3(v907)){
            int v909;
            v909 = 0l;
            while (while_method_1(v909)){
                assert("Tensor range check" && 0 <= v907 && v907 < 1l);
                assert("Tensor range check" && 0 <= v909 && v909 < 4l);
                int v911;
                v911 = 4l * v907;
                int v912;
                v912 = v911 + v909;
                float v913;
                v913 = v827[v912];
                bool v914;
                v914 = v872[v912];
                float v915;
                if (v914){
                    v915 = v913;
                } else {
                    v915 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v907 && v907 < 1l);
                assert("Tensor range check" && 0 <= v909 && v909 < 4l);
                v906[v912] = v915;
                v909 += 1l ;
            }
            v907 += 1l ;
        }
        float v916;
        v916 = 0.0f;
        int v917;
        v917 = 0l;
        while (while_method_3(v917)){
            int v919;
            v919 = 0l;
            while (while_method_1(v919)){
                assert("Tensor range check" && 0 <= v917 && v917 < 1l);
                assert("Tensor range check" && 0 <= v919 && v919 < 4l);
                int v921;
                v921 = 4l * v917;
                int v922;
                v922 = v921 + v919;
                float v923;
                v923 = v906[v922];
                float v924;
                v924 = v916 + v923;
                v916 = v924;
                v919 += 1l ;
            }
            v917 += 1l ;
        }
        auto v925 = cooperative_groups::coalesced_threads();
        int v926;
        v926 = threadIdx.x;
        int v927;
        v927 = v926 / 16l;
        auto v928 = cooperative_groups::labeled_partition(v925,v927);
        float v929;
        v929 = cooperative_groups::reduce(v928, v916, v40);
        float v930;
        v930 = (float)v905;
        float v931;
        v931 = v929 / v930;
        float v932[4l];
        int v933;
        v933 = 0l;
        while (while_method_3(v933)){
            int v935;
            v935 = 0l;
            while (while_method_1(v935)){
                assert("Tensor range check" && 0 <= v933 && v933 < 1l);
                assert("Tensor range check" && 0 <= v935 && v935 < 4l);
                int v937;
                v937 = 4l * v933;
                int v938;
                v938 = v937 + v935;
                float v939;
                v939 = v827[v938];
                bool v940;
                v940 = v872[v938];
                float v941;
                if (v940){
                    v941 = v939;
                } else {
                    v941 = -1.0f / 0.0f;
                }
                float v942;
                v942 = v941 - v931;
                float v943;
                v943 = exp(v942);
                assert("Tensor range check" && 0 <= v933 && v933 < 1l);
                assert("Tensor range check" && 0 <= v935 && v935 < 4l);
                v932[v938] = v943;
                v935 += 1l ;
            }
            v933 += 1l ;
        }
        float v944;
        v944 = 0.0f;
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
                v951 = v932[v950];
                float v952;
                v952 = v944 + v951;
                v944 = v952;
                v947 += 1l ;
            }
            v945 += 1l ;
        }
        auto v953 = cooperative_groups::coalesced_threads();
        int v954;
        v954 = threadIdx.x;
        int v955;
        v955 = v954 / 16l;
        auto v956 = cooperative_groups::labeled_partition(v953,v955);
        float v957;
        v957 = cooperative_groups::reduce(v956, v944, v40);
        float v958[4l];
        int v959;
        v959 = 0l;
        while (while_method_3(v959)){
            int v961;
            v961 = 0l;
            while (while_method_1(v961)){
                assert("Tensor range check" && 0 <= v959 && v959 < 1l);
                assert("Tensor range check" && 0 <= v961 && v961 < 4l);
                int v963;
                v963 = 4l * v959;
                int v964;
                v964 = v963 + v961;
                float v965;
                v965 = v932[v964];
                float v966;
                v966 = v965 / v957;
                assert("Tensor range check" && 0 <= v959 && v959 < 1l);
                assert("Tensor range check" && 0 <= v961 && v961 < 4l);
                v958[v964] = v966;
                v961 += 1l ;
            }
            v959 += 1l ;
        }
        assert("Tensor range check" && 0 <= v823 && v823 < 64l);
        int v967;
        v967 = 0l;
        while (while_method_3(v967)){
            assert("Tensor range check" && 0 <= v967 && v967 < 1l);
            int v969;
            v969 = 64l * v967;
            int v970;
            v970 = v969 + v826;
            assert("Tensor range check" && 0 <= v967 && v967 < 1l);
            int v971;
            v971 = 4l * v967;
            int4* v972;
            v972 = reinterpret_cast<int4*>(v958 + v971);
            int4* v973;
            v973 = reinterpret_cast<int4*>(v5 + v970);
            assert("Pointer alignment check" && (unsigned long long)(v972) % 4l == 0 && (unsigned long long)(v973) % 4l == 0);
            *v973 = *v972;
            v967 += 1l ;
        }
        v823 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v974;
    v974 = threadIdx.x;
    unsigned long long v975;
    v975 = (unsigned long long)v974;
    curandStatePhilox4_32_10_t v976;
    curand_init(12344321ull,v975,0ull,&v976);
    int v977;
    v977 = threadIdx.x;
    bool v978;
    v978 = 0l <= v977;
    bool v979;
    v979 = v978 == false;
    if (v979){
        assert("The index needs to be zero or positive." && v978);
    } else {
    }
    int v981;
    v981 = v977 % 16l;
    int v982;
    v982 = v977 / 16l;
    bool v983;
    v983 = v982 < 2l;
    bool v984;
    v984 = v983 == false;
    if (v984){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v983);
    } else {
    }
    assert("Tensor range check" && 0 <= v982 && v982 < 2l);
    assert("Tensor range check" && 0 <= v981 && v981 < 16l);
    int v986;
    v986 = 4l * v981;
    int v987;
    v987 = 64l * v982;
    int v988;
    v988 = v987 + v986;
    assert("Tensor range check" && 0 <= v982 && v982 < 2l);
    assert("Tensor range check" && 0 <= v981 && v981 < 16l);
    assert("Tensor range check" && 0 <= v982 && v982 < 2l);
    int v989;
    v989 = 0l;
    while (while_method_2(v989)){
        assert("Tensor range check" && 0 <= v989 && v989 < 64l);
        int v991;
        v991 = 128l * v989;
        int v992;
        v992 = v991 + v988;
        float v993[4l];
        int v994[4l];
        int v995;
        v995 = 0l;
        while (while_method_3(v995)){
            assert("Tensor range check" && 0 <= v995 && v995 < 1l);
            int v997;
            v997 = 4l * v995;
            assert("Tensor range check" && 0 <= v995 && v995 < 1l);
            int v998;
            v998 = 64l * v995;
            int v999;
            v999 = v998 + v992;
            int4* v1000;
            v1000 = reinterpret_cast<int4*>(v1 + v999);
            int4* v1001;
            v1001 = reinterpret_cast<int4*>(v993 + v997);
            assert("Pointer alignment check" && (unsigned long long)(v1000) % 4l == 0 && (unsigned long long)(v1001) % 4l == 0);
            *v1001 = *v1000;
            v995 += 1l ;
        }
        int v1002;
        v1002 = 0l;
        while (while_method_3(v1002)){
            int v1004;
            v1004 = 0l;
            while (while_method_1(v1004)){
                bool v1006;
                v1006 = 0l <= v1004;
                bool v1008;
                if (v1006){
                    bool v1007;
                    v1007 = v1004 < 4l;
                    v1008 = v1007;
                } else {
                    v1008 = false;
                }
                bool v1009;
                v1009 = v1008 == false;
                if (v1009){
                    assert("The indices should be inside the range of the dimension." && v1008);
                } else {
                }
                bool v1011;
                v1011 = 0l <= v981;
                bool v1013;
                if (v1011){
                    bool v1012;
                    v1012 = v981 < 16l;
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
                int v1016;
                v1016 = v981 * 4l;
                int v1017;
                v1017 = v1004 + v1016;
                bool v1018;
                v1018 = 0l <= v1002;
                bool v1020;
                if (v1018){
                    bool v1019;
                    v1019 = v1002 < 1l;
                    v1020 = v1019;
                } else {
                    v1020 = false;
                }
                bool v1021;
                v1021 = v1020 == false;
                if (v1021){
                    assert("The indices should be inside the range of the dimension." && v1020);
                } else {
                }
                int v1023;
                v1023 = v1002 * 64l;
                int v1024;
                v1024 = v1017 + v1023;
                assert("Tensor range check" && 0 <= v1002 && v1002 < 1l);
                assert("Tensor range check" && 0 <= v1004 && v1004 < 4l);
                int v1025;
                v1025 = 4l * v1002;
                int v1026;
                v1026 = v1025 + v1004;
                v994[v1026] = v1024;
                v1004 += 1l ;
            }
            v1002 += 1l ;
        }
        bool v1027;
        v1027 = 0l <= v982;
        bool v1028;
        v1028 = v1027 && v983;
        bool v1029;
        v1029 = v1028 == false;
        if (v1029){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1028);
        } else {
        }
        bool v1031;
        v1031 = 0l <= v989;
        bool v1033;
        if (v1031){
            bool v1032;
            v1032 = v989 < 64l;
            v1033 = v1032;
        } else {
            v1033 = false;
        }
        bool v1034;
        v1034 = v1033 == false;
        if (v1034){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1033);
        } else {
        }
        int v1036;
        v1036 = v989 * 2l;
        int v1037;
        v1037 = v1036 + v982;
        float v1038;
        v1038 = 0.0f;
        int v1039;
        v1039 = 0l;
        while (while_method_3(v1039)){
            int v1041;
            v1041 = 0l;
            while (while_method_1(v1041)){
                assert("Tensor range check" && 0 <= v1039 && v1039 < 1l);
                assert("Tensor range check" && 0 <= v1041 && v1041 < 4l);
                int v1043;
                v1043 = 4l * v1039;
                int v1044;
                v1044 = v1043 + v1041;
                float v1045;
                v1045 = v993[v1044];
                float v1046;
                v1046 = v1038 + v1045;
                v1038 = v1046;
                v1041 += 1l ;
            }
            v1039 += 1l ;
        }
        auto v1047 = cooperative_groups::coalesced_threads();
        int v1048;
        v1048 = threadIdx.x;
        int v1049;
        v1049 = v1048 / 16l;
        auto v1050 = cooperative_groups::labeled_partition(v1047,v1049);
        float v1051;
        v1051 = cooperative_groups::reduce(v1050, v1038, v40);
        float v1052;
        v1052 = v1051 / 64.0f;
        float v1053[4l];
        int v1054;
        v1054 = 0l;
        while (while_method_3(v1054)){
            int v1056;
            v1056 = 0l;
            while (while_method_1(v1056)){
                assert("Tensor range check" && 0 <= v1054 && v1054 < 1l);
                assert("Tensor range check" && 0 <= v1056 && v1056 < 4l);
                int v1058;
                v1058 = 4l * v1054;
                int v1059;
                v1059 = v1058 + v1056;
                float v1060;
                v1060 = v993[v1059];
                float v1061;
                v1061 = v1060 - v1052;
                float v1062;
                v1062 = exp(v1061);
                assert("Tensor range check" && 0 <= v1054 && v1054 < 1l);
                assert("Tensor range check" && 0 <= v1056 && v1056 < 4l);
                v1053[v1059] = v1062;
                v1056 += 1l ;
            }
            v1054 += 1l ;
        }
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
                v1070 = v1053[v1069];
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
        v1074 = v1073 / 16l;
        auto v1075 = cooperative_groups::labeled_partition(v1072,v1074);
        float v1076;
        v1076 = cooperative_groups::reduce(v1075, v1063, v40);
        float v1077[4l];
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
                v1084 = v1053[v1083];
                float v1085;
                v1085 = v1084 / v1076;
                assert("Tensor range check" && 0 <= v1078 && v1078 < 1l);
                assert("Tensor range check" && 0 <= v1080 && v1080 < 4l);
                v1077[v1083] = v1085;
                v1080 += 1l ;
            }
            v1078 += 1l ;
        }
        float v1086[4l];
        float v1087;
        v1087 = 0.0f;
        int v1088;
        v1088 = 0l;
        while (while_method_3(v1088)){
            assert("Tensor range check" && 0 <= v1088 && v1088 < 1l);
            int v1090;
            v1090 = 4l * v1088;
            assert("Tensor range check" && 0 <= v1088 && v1088 < 1l);
            int v1091; float v1092;
            Tuple0 tmp30 = Tuple0{0l, 0.0f};
            v1091 = tmp30.v0; v1092 = tmp30.v1;
            while (while_method_1(v1091)){
                assert("Tensor range check" && 0 <= v1091 && v1091 < 4l);
                int v1094;
                v1094 = v1091 + v1090;
                float v1095;
                v1095 = v1077[v1094];
                float v1096;
                v1096 = v1092 + v1095;
                v1092 = v1096;
                v1091 += 1l ;
            }
            auto v1097 = cooperative_groups::coalesced_threads();
            int v1098;
            v1098 = threadIdx.x;
            int v1099;
            v1099 = v1098 / 16l;
            auto v1100 = cooperative_groups::labeled_partition(v1097,v1099);
            Closure2 v1101{};
            float v1102;
            v1102 = cooperative_groups::inclusive_scan(v1100, v1092, v1101);
            float v1103;
            v1103 = v1100.shfl_up(v1102,1);
            bool v1104;
            v1104 = v1100.thread_rank() == 0;
            float v1105;
            if (v1104){
                v1105 = 0.0f;
            } else {
                v1105 = v1103;
            }
            float v1106;
            v1106 = v1100.shfl(v1102,v1100.num_threads()-1);
            float v1107;
            v1107 = v1087 + v1105;
            int v1108; float v1109;
            Tuple0 tmp31 = Tuple0{0l, v1107};
            v1108 = tmp31.v0; v1109 = tmp31.v1;
            while (while_method_1(v1108)){
                assert("Tensor range check" && 0 <= v1108 && v1108 < 4l);
                int v1111;
                v1111 = v1108 + v1090;
                float v1112;
                v1112 = v1077[v1111];
                float v1113;
                v1113 = v1109 + v1112;
                assert("Tensor range check" && 0 <= v1108 && v1108 < 4l);
                v1086[v1111] = v1113;
                v1109 = v1113;
                v1108 += 1l ;
            }
            float v1114;
            v1114 = v1087 + v1106;
            v1087 = v1114;
            v1088 += 1l ;
        }
        float v1115[4l];
        bool v1116[4l];
        int v1117;
        v1117 = 0l;
        while (while_method_3(v1117)){
            int v1119;
            v1119 = 0l;
            while (while_method_1(v1119)){
                assert("Tensor range check" && 0 <= v1117 && v1117 < 1l);
                assert("Tensor range check" && 0 <= v1119 && v1119 < 4l);
                int v1121;
                v1121 = 4l * v1117;
                int v1122;
                v1122 = v1121 + v1119;
                float v1123;
                v1123 = v1086[v1122];
                float v1124;
                v1124 = v1077[v1122];
                bool v1125;
                v1125 = v1124 > 0.0f;
                assert("Tensor range check" && 0 <= v1117 && v1117 < 1l);
                assert("Tensor range check" && 0 <= v1119 && v1119 < 4l);
                v1115[v1122] = v1123;
                v1116[v1122] = v1125;
                v1119 += 1l ;
            }
            v1117 += 1l ;
        }
        float v1126; bool v1127;
        Tuple3 tmp32 = Tuple3{-1.0f / 0.0f, false};
        v1126 = tmp32.v0; v1127 = tmp32.v1;
        int v1128;
        v1128 = 0l;
        while (while_method_3(v1128)){
            int v1130;
            v1130 = 0l;
            while (while_method_1(v1130)){
                assert("Tensor range check" && 0 <= v1128 && v1128 < 1l);
                assert("Tensor range check" && 0 <= v1130 && v1130 < 4l);
                int v1132;
                v1132 = 4l * v1128;
                int v1133;
                v1133 = v1132 + v1130;
                float v1134;
                v1134 = v1115[v1133];
                bool v1135;
                v1135 = v1116[v1133];
                float v1142; bool v1143;
                if (v1127){
                    if (v1135){
                        bool v1136;
                        v1136 = v1126 >= v1134;
                        float v1137;
                        if (v1136){
                            v1137 = v1126;
                        } else {
                            v1137 = v1134;
                        }
                        v1142 = v1137; v1143 = true;
                    } else {
                        v1142 = v1126; v1143 = v1127;
                    }
                } else {
                    if (v1135){
                        v1142 = v1134; v1143 = v1135;
                    } else {
                        v1142 = v1126; v1143 = v1127;
                    }
                }
                v1126 = v1142;
                v1127 = v1143;
                v1130 += 1l ;
            }
            v1128 += 1l ;
        }
        auto v1144 = cooperative_groups::coalesced_threads();
        int v1145;
        v1145 = threadIdx.x;
        int v1146;
        v1146 = v1145 / 16l;
        auto v1147 = cooperative_groups::labeled_partition(v1144,v1146);
        Closure5 v1148{};
        float v1149; bool v1150;
        Tuple3 tmp33 = cooperative_groups::reduce(v1147, Tuple3{v1126, v1127}, v1148);
        v1149 = tmp33.v0; v1150 = tmp33.v1;
        bool v1151;
        v1151 = v1150 == false;
        if (v1151){
            assert("The local reduce must be true." && v1150);
        } else {
        }
        float v1153[4l];
        int v1154[4l];
        int v1155;
        v1155 = 0l;
        while (while_method_3(v1155)){
            int v1157;
            v1157 = 0l;
            while (while_method_1(v1157)){
                assert("Tensor range check" && 0 <= v1155 && v1155 < 1l);
                assert("Tensor range check" && 0 <= v1157 && v1157 < 4l);
                int v1159;
                v1159 = 4l * v1155;
                int v1160;
                v1160 = v1159 + v1157;
                int v1161;
                v1161 = v994[v1160];
                float v1162;
                v1162 = curand_uniform(&v976);
                assert("Tensor range check" && 0 <= v1155 && v1155 < 1l);
                assert("Tensor range check" && 0 <= v1157 && v1157 < 4l);
                v1153[v1160] = v1162;
                v1154[v1160] = v1161;
                v1157 += 1l ;
            }
            v1155 += 1l ;
        }
        float v1163; int v1164;
        Tuple1 tmp34 = Tuple1{0.0f, 2147483647l};
        v1163 = tmp34.v0; v1164 = tmp34.v1;
        int v1165;
        v1165 = 0l;
        while (while_method_3(v1165)){
            int v1167;
            v1167 = 0l;
            while (while_method_1(v1167)){
                assert("Tensor range check" && 0 <= v1165 && v1165 < 1l);
                assert("Tensor range check" && 0 <= v1167 && v1167 < 4l);
                int v1169;
                v1169 = 4l * v1165;
                int v1170;
                v1170 = v1169 + v1167;
                float v1171;
                v1171 = v1153[v1170];
                int v1172;
                v1172 = v1154[v1170];
                bool v1173;
                v1173 = v1164 < v1172;
                float v1174; int v1175;
                if (v1173){
                    v1174 = v1163; v1175 = v1164;
                } else {
                    v1174 = v1171; v1175 = v1172;
                }
                v1163 = v1174;
                v1164 = v1175;
                v1167 += 1l ;
            }
            v1165 += 1l ;
        }
        auto v1176 = cooperative_groups::coalesced_threads();
        int v1177;
        v1177 = threadIdx.x;
        int v1178;
        v1178 = v1177 / 16l;
        auto v1179 = cooperative_groups::labeled_partition(v1176,v1178);
        Closure6 v1180{};
        float v1181; int v1182;
        Tuple1 tmp35 = cooperative_groups::reduce(v1179, Tuple1{v1163, v1164}, v1180);
        v1181 = tmp35.v0; v1182 = tmp35.v1;
        float v1183;
        v1183 = v1149 * v1181;
        int v1184[4l];
        bool v1185[4l];
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
                v1192 = v1115[v1191];
                bool v1193;
                v1193 = v1116[v1191];
                int v1194;
                v1194 = v994[v1191];
                int v1197; bool v1198;
                if (v1193){
                    float v1195;
                    v1195 = v1192 - v1183;
                    bool v1196;
                    v1196 = v1195 >= 0.0f;
                    v1197 = v1194; v1198 = v1196;
                } else {
                    v1197 = 2147483647l; v1198 = false;
                }
                assert("Tensor range check" && 0 <= v1186 && v1186 < 1l);
                assert("Tensor range check" && 0 <= v1188 && v1188 < 4l);
                v1184[v1191] = v1197;
                v1185[v1191] = v1198;
                v1188 += 1l ;
            }
            v1186 += 1l ;
        }
        int v1199; bool v1200;
        Tuple4 tmp36 = Tuple4{2147483647l, false};
        v1199 = tmp36.v0; v1200 = tmp36.v1;
        int v1201;
        v1201 = 0l;
        while (while_method_3(v1201)){
            int v1203;
            v1203 = 0l;
            while (while_method_1(v1203)){
                assert("Tensor range check" && 0 <= v1201 && v1201 < 1l);
                assert("Tensor range check" && 0 <= v1203 && v1203 < 4l);
                int v1205;
                v1205 = 4l * v1201;
                int v1206;
                v1206 = v1205 + v1203;
                int v1207;
                v1207 = v1184[v1206];
                bool v1208;
                v1208 = v1185[v1206];
                int v1215; bool v1216;
                if (v1200){
                    if (v1208){
                        bool v1209;
                        v1209 = v1199 < v1207;
                        int v1210;
                        if (v1209){
                            v1210 = v1199;
                        } else {
                            v1210 = v1207;
                        }
                        v1215 = v1210; v1216 = true;
                    } else {
                        v1215 = v1199; v1216 = v1200;
                    }
                } else {
                    if (v1208){
                        v1215 = v1207; v1216 = v1208;
                    } else {
                        v1215 = v1199; v1216 = v1200;
                    }
                }
                v1199 = v1215;
                v1200 = v1216;
                v1203 += 1l ;
            }
            v1201 += 1l ;
        }
        auto v1217 = cooperative_groups::coalesced_threads();
        int v1218;
        v1218 = threadIdx.x;
        int v1219;
        v1219 = v1218 / 16l;
        auto v1220 = cooperative_groups::labeled_partition(v1217,v1219);
        Closure7 v1221{};
        int v1222; bool v1223;
        Tuple4 tmp37 = cooperative_groups::reduce(v1220, Tuple4{v1199, v1200}, v1221);
        v1222 = tmp37.v0; v1223 = tmp37.v1;
        bool v1224;
        v1224 = v1223 == false;
        if (v1224){
            assert("The local reduce must be true." && v1223);
        } else {
        }
        assert("Tensor range check" && 0 <= v989 && v989 < 64l);
        int v1226;
        v1226 = 0l;
        while (while_method_3(v1226)){
            assert("Tensor range check" && 0 <= v1226 && v1226 < 1l);
            int v1228;
            v1228 = 64l * v1226;
            int v1229;
            v1229 = v1228 + v992;
            assert("Tensor range check" && 0 <= v1226 && v1226 < 1l);
            int v1230;
            v1230 = 4l * v1226;
            int4* v1231;
            v1231 = reinterpret_cast<int4*>(v1077 + v1230);
            int4* v1232;
            v1232 = reinterpret_cast<int4*>(v14 + v1229);
            assert("Pointer alignment check" && (unsigned long long)(v1231) % 4l == 0 && (unsigned long long)(v1232) % 4l == 0);
            *v1232 = *v1231;
            v1226 += 1l ;
        }
        assert("Tensor range check" && 0 <= v989 && v989 < 64l);
        int v1233;
        v1233 = 2l * v989;
        int v1234;
        v1234 = v1233 + v982;
        v15[v1234] = v1222;
        v989 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1235;
    v1235 = threadIdx.x;
    unsigned long long v1236;
    v1236 = (unsigned long long)v1235;
    curandStatePhilox4_32_10_t v1237;
    curand_init(12344321ull,v1236,0ull,&v1237);
    int v1238;
    v1238 = threadIdx.x;
    bool v1239;
    v1239 = 0l <= v1238;
    bool v1240;
    v1240 = v1239 == false;
    if (v1240){
        assert("The index needs to be zero or positive." && v1239);
    } else {
    }
    int v1242;
    v1242 = v1238 % 16l;
    int v1243;
    v1243 = v1238 / 16l;
    bool v1244;
    v1244 = v1243 < 2l;
    bool v1245;
    v1245 = v1244 == false;
    if (v1245){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1244);
    } else {
    }
    assert("Tensor range check" && 0 <= v1243 && v1243 < 2l);
    assert("Tensor range check" && 0 <= v1242 && v1242 < 16l);
    int v1247;
    v1247 = 4l * v1242;
    int v1248;
    v1248 = 64l * v1243;
    int v1249;
    v1249 = v1248 + v1247;
    assert("Tensor range check" && 0 <= v1243 && v1243 < 2l);
    assert("Tensor range check" && 0 <= v1242 && v1242 < 16l);
    assert("Tensor range check" && 0 <= v1243 && v1243 < 2l);
    int v1250;
    v1250 = 0l;
    while (while_method_2(v1250)){
        assert("Tensor range check" && 0 <= v1250 && v1250 < 64l);
        int v1252;
        v1252 = 128l * v1250;
        int v1253;
        v1253 = v1252 + v1249;
        float v1254[4l];
        int v1255[4l];
        int v1256;
        v1256 = 0l;
        while (while_method_3(v1256)){
            assert("Tensor range check" && 0 <= v1256 && v1256 < 1l);
            int v1258;
            v1258 = 4l * v1256;
            assert("Tensor range check" && 0 <= v1256 && v1256 < 1l);
            int v1259;
            v1259 = 64l * v1256;
            int v1260;
            v1260 = v1259 + v1253;
            int4* v1261;
            v1261 = reinterpret_cast<int4*>(v1 + v1260);
            int4* v1262;
            v1262 = reinterpret_cast<int4*>(v1254 + v1258);
            assert("Pointer alignment check" && (unsigned long long)(v1261) % 4l == 0 && (unsigned long long)(v1262) % 4l == 0);
            *v1262 = *v1261;
            v1256 += 1l ;
        }
        int v1263;
        v1263 = 0l;
        while (while_method_3(v1263)){
            int v1265;
            v1265 = 0l;
            while (while_method_1(v1265)){
                bool v1267;
                v1267 = 0l <= v1265;
                bool v1269;
                if (v1267){
                    bool v1268;
                    v1268 = v1265 < 4l;
                    v1269 = v1268;
                } else {
                    v1269 = false;
                }
                bool v1270;
                v1270 = v1269 == false;
                if (v1270){
                    assert("The indices should be inside the range of the dimension." && v1269);
                } else {
                }
                bool v1272;
                v1272 = 0l <= v1242;
                bool v1274;
                if (v1272){
                    bool v1273;
                    v1273 = v1242 < 16l;
                    v1274 = v1273;
                } else {
                    v1274 = false;
                }
                bool v1275;
                v1275 = v1274 == false;
                if (v1275){
                    assert("The indices should be inside the range of the dimension." && v1274);
                } else {
                }
                int v1277;
                v1277 = v1242 * 4l;
                int v1278;
                v1278 = v1265 + v1277;
                bool v1279;
                v1279 = 0l <= v1263;
                bool v1281;
                if (v1279){
                    bool v1280;
                    v1280 = v1263 < 1l;
                    v1281 = v1280;
                } else {
                    v1281 = false;
                }
                bool v1282;
                v1282 = v1281 == false;
                if (v1282){
                    assert("The indices should be inside the range of the dimension." && v1281);
                } else {
                }
                int v1284;
                v1284 = v1263 * 64l;
                int v1285;
                v1285 = v1278 + v1284;
                assert("Tensor range check" && 0 <= v1263 && v1263 < 1l);
                assert("Tensor range check" && 0 <= v1265 && v1265 < 4l);
                int v1286;
                v1286 = 4l * v1263;
                int v1287;
                v1287 = v1286 + v1265;
                v1255[v1287] = v1285;
                v1265 += 1l ;
            }
            v1263 += 1l ;
        }
        bool v1288;
        v1288 = 0l <= v1243;
        bool v1289;
        v1289 = v1288 && v1244;
        bool v1290;
        v1290 = v1289 == false;
        if (v1290){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1289);
        } else {
        }
        bool v1292;
        v1292 = 0l <= v1250;
        bool v1294;
        if (v1292){
            bool v1293;
            v1293 = v1250 < 64l;
            v1294 = v1293;
        } else {
            v1294 = false;
        }
        bool v1295;
        v1295 = v1294 == false;
        if (v1295){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1294);
        } else {
        }
        int v1297;
        v1297 = v1250 * 2l;
        int v1298;
        v1298 = v1297 + v1243;
        bool v1299[4l];
        int v1300;
        v1300 = 0l;
        while (while_method_3(v1300)){
            int v1302;
            v1302 = 0l;
            while (while_method_1(v1302)){
                assert("Tensor range check" && 0 <= v1300 && v1300 < 1l);
                assert("Tensor range check" && 0 <= v1302 && v1302 < 4l);
                int v1304;
                v1304 = 4l * v1300;
                int v1305;
                v1305 = v1304 + v1302;
                float v1306;
                v1306 = v1254[v1305];
                int v1307;
                v1307 = v1255[v1305];
                bool v1308;
                v1308 = v1307 < 11l;
                assert("Tensor range check" && 0 <= v1300 && v1300 < 1l);
                assert("Tensor range check" && 0 <= v1302 && v1302 < 4l);
                v1299[v1305] = v1308;
                v1302 += 1l ;
            }
            v1300 += 1l ;
        }
        int v1309[4l];
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
                bool v1316;
                v1316 = v1299[v1315];
                int v1317;
                if (v1316){
                    v1317 = 1l;
                } else {
                    v1317 = 0l;
                }
                assert("Tensor range check" && 0 <= v1310 && v1310 < 1l);
                assert("Tensor range check" && 0 <= v1312 && v1312 < 4l);
                v1309[v1315] = v1317;
                v1312 += 1l ;
            }
            v1310 += 1l ;
        }
        int v1318;
        v1318 = 0l;
        int v1319;
        v1319 = 0l;
        while (while_method_3(v1319)){
            int v1321;
            v1321 = 0l;
            while (while_method_1(v1321)){
                assert("Tensor range check" && 0 <= v1319 && v1319 < 1l);
                assert("Tensor range check" && 0 <= v1321 && v1321 < 4l);
                int v1323;
                v1323 = 4l * v1319;
                int v1324;
                v1324 = v1323 + v1321;
                int v1325;
                v1325 = v1309[v1324];
                int v1326;
                v1326 = v1318 + v1325;
                v1318 = v1326;
                v1321 += 1l ;
            }
            v1319 += 1l ;
        }
        auto v1327 = cooperative_groups::coalesced_threads();
        int v1328;
        v1328 = threadIdx.x;
        int v1329;
        v1329 = v1328 / 16l;
        auto v1330 = cooperative_groups::labeled_partition(v1327,v1329);
        Closure4 v1331{};
        int v1332;
        v1332 = cooperative_groups::reduce(v1330, v1318, v1331);
        float v1333[4l];
        int v1334;
        v1334 = 0l;
        while (while_method_3(v1334)){
            int v1336;
            v1336 = 0l;
            while (while_method_1(v1336)){
                assert("Tensor range check" && 0 <= v1334 && v1334 < 1l);
                assert("Tensor range check" && 0 <= v1336 && v1336 < 4l);
                int v1338;
                v1338 = 4l * v1334;
                int v1339;
                v1339 = v1338 + v1336;
                float v1340;
                v1340 = v1254[v1339];
                bool v1341;
                v1341 = v1299[v1339];
                float v1342;
                if (v1341){
                    v1342 = v1340;
                } else {
                    v1342 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1334 && v1334 < 1l);
                assert("Tensor range check" && 0 <= v1336 && v1336 < 4l);
                v1333[v1339] = v1342;
                v1336 += 1l ;
            }
            v1334 += 1l ;
        }
        float v1343;
        v1343 = 0.0f;
        int v1344;
        v1344 = 0l;
        while (while_method_3(v1344)){
            int v1346;
            v1346 = 0l;
            while (while_method_1(v1346)){
                assert("Tensor range check" && 0 <= v1344 && v1344 < 1l);
                assert("Tensor range check" && 0 <= v1346 && v1346 < 4l);
                int v1348;
                v1348 = 4l * v1344;
                int v1349;
                v1349 = v1348 + v1346;
                float v1350;
                v1350 = v1333[v1349];
                float v1351;
                v1351 = v1343 + v1350;
                v1343 = v1351;
                v1346 += 1l ;
            }
            v1344 += 1l ;
        }
        auto v1352 = cooperative_groups::coalesced_threads();
        int v1353;
        v1353 = threadIdx.x;
        int v1354;
        v1354 = v1353 / 16l;
        auto v1355 = cooperative_groups::labeled_partition(v1352,v1354);
        float v1356;
        v1356 = cooperative_groups::reduce(v1355, v1343, v40);
        float v1357;
        v1357 = (float)v1332;
        float v1358;
        v1358 = v1356 / v1357;
        float v1359[4l];
        int v1360;
        v1360 = 0l;
        while (while_method_3(v1360)){
            int v1362;
            v1362 = 0l;
            while (while_method_1(v1362)){
                assert("Tensor range check" && 0 <= v1360 && v1360 < 1l);
                assert("Tensor range check" && 0 <= v1362 && v1362 < 4l);
                int v1364;
                v1364 = 4l * v1360;
                int v1365;
                v1365 = v1364 + v1362;
                float v1366;
                v1366 = v1254[v1365];
                bool v1367;
                v1367 = v1299[v1365];
                float v1368;
                if (v1367){
                    v1368 = v1366;
                } else {
                    v1368 = -1.0f / 0.0f;
                }
                float v1369;
                v1369 = v1368 - v1358;
                float v1370;
                v1370 = exp(v1369);
                assert("Tensor range check" && 0 <= v1360 && v1360 < 1l);
                assert("Tensor range check" && 0 <= v1362 && v1362 < 4l);
                v1359[v1365] = v1370;
                v1362 += 1l ;
            }
            v1360 += 1l ;
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
                v1378 = v1359[v1377];
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
        v1382 = v1381 / 16l;
        auto v1383 = cooperative_groups::labeled_partition(v1380,v1382);
        float v1384;
        v1384 = cooperative_groups::reduce(v1383, v1371, v40);
        float v1385[4l];
        int v1386;
        v1386 = 0l;
        while (while_method_3(v1386)){
            int v1388;
            v1388 = 0l;
            while (while_method_1(v1388)){
                assert("Tensor range check" && 0 <= v1386 && v1386 < 1l);
                assert("Tensor range check" && 0 <= v1388 && v1388 < 4l);
                int v1390;
                v1390 = 4l * v1386;
                int v1391;
                v1391 = v1390 + v1388;
                float v1392;
                v1392 = v1359[v1391];
                float v1393;
                v1393 = v1392 / v1384;
                assert("Tensor range check" && 0 <= v1386 && v1386 < 1l);
                assert("Tensor range check" && 0 <= v1388 && v1388 < 4l);
                v1385[v1391] = v1393;
                v1388 += 1l ;
            }
            v1386 += 1l ;
        }
        float v1394[4l];
        float v1395;
        v1395 = 0.0f;
        int v1396;
        v1396 = 0l;
        while (while_method_3(v1396)){
            assert("Tensor range check" && 0 <= v1396 && v1396 < 1l);
            int v1398;
            v1398 = 4l * v1396;
            assert("Tensor range check" && 0 <= v1396 && v1396 < 1l);
            int v1399; float v1400;
            Tuple0 tmp38 = Tuple0{0l, 0.0f};
            v1399 = tmp38.v0; v1400 = tmp38.v1;
            while (while_method_1(v1399)){
                assert("Tensor range check" && 0 <= v1399 && v1399 < 4l);
                int v1402;
                v1402 = v1399 + v1398;
                float v1403;
                v1403 = v1385[v1402];
                float v1404;
                v1404 = v1400 + v1403;
                v1400 = v1404;
                v1399 += 1l ;
            }
            auto v1405 = cooperative_groups::coalesced_threads();
            int v1406;
            v1406 = threadIdx.x;
            int v1407;
            v1407 = v1406 / 16l;
            auto v1408 = cooperative_groups::labeled_partition(v1405,v1407);
            Closure2 v1409{};
            float v1410;
            v1410 = cooperative_groups::inclusive_scan(v1408, v1400, v1409);
            float v1411;
            v1411 = v1408.shfl_up(v1410,1);
            bool v1412;
            v1412 = v1408.thread_rank() == 0;
            float v1413;
            if (v1412){
                v1413 = 0.0f;
            } else {
                v1413 = v1411;
            }
            float v1414;
            v1414 = v1408.shfl(v1410,v1408.num_threads()-1);
            float v1415;
            v1415 = v1395 + v1413;
            int v1416; float v1417;
            Tuple0 tmp39 = Tuple0{0l, v1415};
            v1416 = tmp39.v0; v1417 = tmp39.v1;
            while (while_method_1(v1416)){
                assert("Tensor range check" && 0 <= v1416 && v1416 < 4l);
                int v1419;
                v1419 = v1416 + v1398;
                float v1420;
                v1420 = v1385[v1419];
                float v1421;
                v1421 = v1417 + v1420;
                assert("Tensor range check" && 0 <= v1416 && v1416 < 4l);
                v1394[v1419] = v1421;
                v1417 = v1421;
                v1416 += 1l ;
            }
            float v1422;
            v1422 = v1395 + v1414;
            v1395 = v1422;
            v1396 += 1l ;
        }
        float v1423[4l];
        bool v1424[4l];
        int v1425;
        v1425 = 0l;
        while (while_method_3(v1425)){
            int v1427;
            v1427 = 0l;
            while (while_method_1(v1427)){
                assert("Tensor range check" && 0 <= v1425 && v1425 < 1l);
                assert("Tensor range check" && 0 <= v1427 && v1427 < 4l);
                int v1429;
                v1429 = 4l * v1425;
                int v1430;
                v1430 = v1429 + v1427;
                float v1431;
                v1431 = v1394[v1430];
                float v1432;
                v1432 = v1385[v1430];
                bool v1433;
                v1433 = v1432 > 0.0f;
                assert("Tensor range check" && 0 <= v1425 && v1425 < 1l);
                assert("Tensor range check" && 0 <= v1427 && v1427 < 4l);
                v1423[v1430] = v1431;
                v1424[v1430] = v1433;
                v1427 += 1l ;
            }
            v1425 += 1l ;
        }
        float v1434; bool v1435;
        Tuple3 tmp40 = Tuple3{-1.0f / 0.0f, false};
        v1434 = tmp40.v0; v1435 = tmp40.v1;
        int v1436;
        v1436 = 0l;
        while (while_method_3(v1436)){
            int v1438;
            v1438 = 0l;
            while (while_method_1(v1438)){
                assert("Tensor range check" && 0 <= v1436 && v1436 < 1l);
                assert("Tensor range check" && 0 <= v1438 && v1438 < 4l);
                int v1440;
                v1440 = 4l * v1436;
                int v1441;
                v1441 = v1440 + v1438;
                float v1442;
                v1442 = v1423[v1441];
                bool v1443;
                v1443 = v1424[v1441];
                float v1450; bool v1451;
                if (v1435){
                    if (v1443){
                        bool v1444;
                        v1444 = v1434 >= v1442;
                        float v1445;
                        if (v1444){
                            v1445 = v1434;
                        } else {
                            v1445 = v1442;
                        }
                        v1450 = v1445; v1451 = true;
                    } else {
                        v1450 = v1434; v1451 = v1435;
                    }
                } else {
                    if (v1443){
                        v1450 = v1442; v1451 = v1443;
                    } else {
                        v1450 = v1434; v1451 = v1435;
                    }
                }
                v1434 = v1450;
                v1435 = v1451;
                v1438 += 1l ;
            }
            v1436 += 1l ;
        }
        auto v1452 = cooperative_groups::coalesced_threads();
        int v1453;
        v1453 = threadIdx.x;
        int v1454;
        v1454 = v1453 / 16l;
        auto v1455 = cooperative_groups::labeled_partition(v1452,v1454);
        Closure5 v1456{};
        float v1457; bool v1458;
        Tuple3 tmp41 = cooperative_groups::reduce(v1455, Tuple3{v1434, v1435}, v1456);
        v1457 = tmp41.v0; v1458 = tmp41.v1;
        bool v1459;
        v1459 = v1458 == false;
        if (v1459){
            assert("The local reduce must be true." && v1458);
        } else {
        }
        float v1461[4l];
        int v1462[4l];
        int v1463;
        v1463 = 0l;
        while (while_method_3(v1463)){
            int v1465;
            v1465 = 0l;
            while (while_method_1(v1465)){
                assert("Tensor range check" && 0 <= v1463 && v1463 < 1l);
                assert("Tensor range check" && 0 <= v1465 && v1465 < 4l);
                int v1467;
                v1467 = 4l * v1463;
                int v1468;
                v1468 = v1467 + v1465;
                int v1469;
                v1469 = v1255[v1468];
                float v1470;
                v1470 = curand_uniform(&v1237);
                assert("Tensor range check" && 0 <= v1463 && v1463 < 1l);
                assert("Tensor range check" && 0 <= v1465 && v1465 < 4l);
                v1461[v1468] = v1470;
                v1462[v1468] = v1469;
                v1465 += 1l ;
            }
            v1463 += 1l ;
        }
        float v1471; int v1472;
        Tuple1 tmp42 = Tuple1{0.0f, 2147483647l};
        v1471 = tmp42.v0; v1472 = tmp42.v1;
        int v1473;
        v1473 = 0l;
        while (while_method_3(v1473)){
            int v1475;
            v1475 = 0l;
            while (while_method_1(v1475)){
                assert("Tensor range check" && 0 <= v1473 && v1473 < 1l);
                assert("Tensor range check" && 0 <= v1475 && v1475 < 4l);
                int v1477;
                v1477 = 4l * v1473;
                int v1478;
                v1478 = v1477 + v1475;
                float v1479;
                v1479 = v1461[v1478];
                int v1480;
                v1480 = v1462[v1478];
                bool v1481;
                v1481 = v1472 < v1480;
                float v1482; int v1483;
                if (v1481){
                    v1482 = v1471; v1483 = v1472;
                } else {
                    v1482 = v1479; v1483 = v1480;
                }
                v1471 = v1482;
                v1472 = v1483;
                v1475 += 1l ;
            }
            v1473 += 1l ;
        }
        auto v1484 = cooperative_groups::coalesced_threads();
        int v1485;
        v1485 = threadIdx.x;
        int v1486;
        v1486 = v1485 / 16l;
        auto v1487 = cooperative_groups::labeled_partition(v1484,v1486);
        Closure6 v1488{};
        float v1489; int v1490;
        Tuple1 tmp43 = cooperative_groups::reduce(v1487, Tuple1{v1471, v1472}, v1488);
        v1489 = tmp43.v0; v1490 = tmp43.v1;
        float v1491;
        v1491 = v1457 * v1489;
        int v1492[4l];
        bool v1493[4l];
        int v1494;
        v1494 = 0l;
        while (while_method_3(v1494)){
            int v1496;
            v1496 = 0l;
            while (while_method_1(v1496)){
                assert("Tensor range check" && 0 <= v1494 && v1494 < 1l);
                assert("Tensor range check" && 0 <= v1496 && v1496 < 4l);
                int v1498;
                v1498 = 4l * v1494;
                int v1499;
                v1499 = v1498 + v1496;
                float v1500;
                v1500 = v1423[v1499];
                bool v1501;
                v1501 = v1424[v1499];
                int v1502;
                v1502 = v1255[v1499];
                int v1505; bool v1506;
                if (v1501){
                    float v1503;
                    v1503 = v1500 - v1491;
                    bool v1504;
                    v1504 = v1503 >= 0.0f;
                    v1505 = v1502; v1506 = v1504;
                } else {
                    v1505 = 2147483647l; v1506 = false;
                }
                assert("Tensor range check" && 0 <= v1494 && v1494 < 1l);
                assert("Tensor range check" && 0 <= v1496 && v1496 < 4l);
                v1492[v1499] = v1505;
                v1493[v1499] = v1506;
                v1496 += 1l ;
            }
            v1494 += 1l ;
        }
        int v1507; bool v1508;
        Tuple4 tmp44 = Tuple4{2147483647l, false};
        v1507 = tmp44.v0; v1508 = tmp44.v1;
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
                v1515 = v1492[v1514];
                bool v1516;
                v1516 = v1493[v1514];
                int v1523; bool v1524;
                if (v1508){
                    if (v1516){
                        bool v1517;
                        v1517 = v1507 < v1515;
                        int v1518;
                        if (v1517){
                            v1518 = v1507;
                        } else {
                            v1518 = v1515;
                        }
                        v1523 = v1518; v1524 = true;
                    } else {
                        v1523 = v1507; v1524 = v1508;
                    }
                } else {
                    if (v1516){
                        v1523 = v1515; v1524 = v1516;
                    } else {
                        v1523 = v1507; v1524 = v1508;
                    }
                }
                v1507 = v1523;
                v1508 = v1524;
                v1511 += 1l ;
            }
            v1509 += 1l ;
        }
        auto v1525 = cooperative_groups::coalesced_threads();
        int v1526;
        v1526 = threadIdx.x;
        int v1527;
        v1527 = v1526 / 16l;
        auto v1528 = cooperative_groups::labeled_partition(v1525,v1527);
        Closure7 v1529{};
        int v1530; bool v1531;
        Tuple4 tmp45 = cooperative_groups::reduce(v1528, Tuple4{v1507, v1508}, v1529);
        v1530 = tmp45.v0; v1531 = tmp45.v1;
        bool v1532;
        v1532 = v1531 == false;
        if (v1532){
            assert("The local reduce must be true." && v1531);
        } else {
        }
        assert("Tensor range check" && 0 <= v1250 && v1250 < 64l);
        int v1534;
        v1534 = 0l;
        while (while_method_3(v1534)){
            assert("Tensor range check" && 0 <= v1534 && v1534 < 1l);
            int v1536;
            v1536 = 64l * v1534;
            int v1537;
            v1537 = v1536 + v1253;
            assert("Tensor range check" && 0 <= v1534 && v1534 < 1l);
            int v1538;
            v1538 = 4l * v1534;
            int4* v1539;
            v1539 = reinterpret_cast<int4*>(v1385 + v1538);
            int4* v1540;
            v1540 = reinterpret_cast<int4*>(v14 + v1537);
            assert("Pointer alignment check" && (unsigned long long)(v1539) % 4l == 0 && (unsigned long long)(v1540) % 4l == 0);
            *v1540 = *v1539;
            v1534 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1250 && v1250 < 64l);
        int v1541;
        v1541 = 2l * v1250;
        int v1542;
        v1542 = v1541 + v1243;
        v15[v1542] = v1530;
        v1250 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
extern "C" __global__ void entry2(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    unsigned long long v9;
    v9 = (unsigned long long)v8;
    curandStatePhilox4_32_10_t v10;
    curand_init(12344321ull,v9,0ull,&v10);
    int v11;
    v11 = threadIdx.x;
    assert("Tensor range check" && 0 <= v11 && v11 < 32l);
    int v12;
    v12 = 16l * v11;
    int v13;
    v13 = threadIdx.x;
    assert("Tensor range check" && 0 <= v13 && v13 < 32l);
    int v14;
    v14 = 16l * v13;
    int v15;
    v15 = threadIdx.x;
    assert("Tensor range check" && 0 <= v15 && v15 < 32l);
    int v16;
    v16 = 16l * v15;
    int v17;
    v17 = threadIdx.x;
    assert("Tensor range check" && 0 <= v17 && v17 < 32l);
    int v18;
    v18 = 16l * v17;
    int v19;
    v19 = threadIdx.x;
    assert("Tensor range check" && 0 <= v19 && v19 < 32l);
    int v20;
    v20 = 16l * v19;
    float * v21;
    v21 = v1+v12;
    int * v23;
    v23 = v2+v18;
    int * v25;
    v25 = v3+v18;
    int v27;
    v27 = sizeof(float *);
    unsigned long long v28;
    v28 = (unsigned long long)v27;
    unsigned long long v29;
    v29 = 32ull * v28;
    unsigned long long v30;
    v30 = v29 + 16ull;
    unsigned long long v31;
    v31 = v30 - 1ull;
    unsigned long long v32;
    v32 = v31 % 16ull;
    unsigned long long v33;
    v33 = v31 - v32;
    int v34;
    v34 = sizeof(int *);
    unsigned long long v35;
    v35 = (unsigned long long)v34;
    unsigned long long v36;
    v36 = 32ull * v35;
    unsigned long long v37;
    v37 = v33 + v36;
    unsigned long long v38;
    v38 = v37 + 16ull;
    unsigned long long v39;
    v39 = v38 - 1ull;
    unsigned long long v40;
    v40 = v39 % 16ull;
    unsigned long long v41;
    v41 = v39 - v40;
    unsigned long long v42;
    v42 = v41 + v36;
    bool v43;
    v43 = v42 <= 81920ull;
    bool v44;
    v44 = v43 == false;
    if (v44){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v43);
    } else {
    }
    extern __shared__ unsigned char v46[];
    bool v47;
    v47 = v42 <= v42;
    bool v48;
    v48 = v47 == false;
    if (v48){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v47);
    } else {
    }
    float * * v50;
    v50 = reinterpret_cast<float * *>(&v46[0ull]);
    int * * v52;
    v52 = reinterpret_cast<int * *>(&v46[v33]);
    int * * v54;
    v54 = reinterpret_cast<int * *>(&v46[v41]);
    int v56;
    v56 = threadIdx.x;
    assert("Tensor range check" && 0 <= v56 && v56 < 32l);
    v50[v56] = v21;
    v52[v56] = v23;
    v54[v56] = v25;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v57;
    v57 = 0l <= v56;
    bool v58;
    v58 = v57 == false;
    if (v58){
        assert("The index needs to be zero or positive." && v57);
    } else {
    }
    int v60;
    v60 = v56 % 4l;
    int v61;
    v61 = v56 / 4l;
    bool v62;
    v62 = v61 < 8l;
    bool v63;
    v63 = v62 == false;
    if (v63){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v62);
    } else {
    }
    assert("Tensor range check" && 0 <= v61 && v61 < 8l);
    int v65;
    v65 = 0l;
    while (while_method_1(v65)){
        bool v67;
        v67 = 0l <= v61;
        bool v68;
        v68 = v67 && v62;
        bool v69;
        v69 = v68 == false;
        if (v69){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v68);
        } else {
        }
        bool v71;
        v71 = 0l <= v65;
        bool v73;
        if (v71){
            bool v72;
            v72 = v65 < 4l;
            v73 = v72;
        } else {
            v73 = false;
        }
        bool v74;
        v74 = v73 == false;
        if (v74){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v73);
        } else {
        }
        int v76;
        v76 = v65 * 8l;
        int v77;
        v77 = v76 + v61;
        assert("Tensor range check" && 0 <= v65 && v65 < 4l);
        int v78;
        v78 = 8l * v65;
        int v79;
        v79 = v78 + v61;
        float * v80;
        v80 = v50[v79];
        int * v81;
        v81 = v52[v79];
        int * v82;
        v82 = v54[v79];
        int v83;
        v83 = blockIdx.x;
        int v84;
        v84 = v83 * 32l;
        int v85;
        v85 = v84 + v77;
        assert("Tensor range check" && 0 <= v60 && v60 < 4l);
        int v86;
        v86 = 4l * v60;
        float v87[4l];
        int v88[4l];
        int v89;
        v89 = 0l;
        while (while_method_3(v89)){
            assert("Tensor range check" && 0 <= v89 && v89 < 1l);
            int v91;
            v91 = 4l * v89;
            assert("Tensor range check" && 0 <= v89 && v89 < 1l);
            int v92;
            v92 = 16l * v89;
            int v93;
            v93 = v92 + v86;
            int4* v94;
            v94 = reinterpret_cast<int4*>(v80 + v93);
            int4* v95;
            v95 = reinterpret_cast<int4*>(v87 + v91);
            assert("Pointer alignment check" && (unsigned long long)(v94) % 4l == 0 && (unsigned long long)(v95) % 4l == 0);
            *v95 = *v94;
            v89 += 1l ;
        }
        int v96;
        v96 = 0l;
        while (while_method_3(v96)){
            int v98;
            v98 = 0l;
            while (while_method_1(v98)){
                bool v100;
                v100 = 0l <= v98;
                bool v102;
                if (v100){
                    bool v101;
                    v101 = v98 < 4l;
                    v102 = v101;
                } else {
                    v102 = false;
                }
                bool v103;
                v103 = v102 == false;
                if (v103){
                    assert("The indices should be inside the range of the dimension." && v102);
                } else {
                }
                bool v105;
                v105 = 0l <= v60;
                bool v107;
                if (v105){
                    bool v106;
                    v106 = v60 < 4l;
                    v107 = v106;
                } else {
                    v107 = false;
                }
                bool v108;
                v108 = v107 == false;
                if (v108){
                    assert("The indices should be inside the range of the dimension." && v107);
                } else {
                }
                int v110;
                v110 = v60 * 4l;
                int v111;
                v111 = v98 + v110;
                bool v112;
                v112 = 0l <= v96;
                bool v114;
                if (v112){
                    bool v113;
                    v113 = v96 < 1l;
                    v114 = v113;
                } else {
                    v114 = false;
                }
                bool v115;
                v115 = v114 == false;
                if (v115){
                    assert("The indices should be inside the range of the dimension." && v114);
                } else {
                }
                int v117;
                v117 = v96 * 16l;
                int v118;
                v118 = v111 + v117;
                assert("Tensor range check" && 0 <= v96 && v96 < 1l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                int v119;
                v119 = 4l * v96;
                int v120;
                v120 = v119 + v98;
                v88[v120] = v118;
                v98 += 1l ;
            }
            v96 += 1l ;
        }
        int v121[4l];
        int v122[4l];
        int v123;
        v123 = 0l;
        while (while_method_3(v123)){
            int v125;
            v125 = 0l;
            while (while_method_1(v125)){
                assert("Tensor range check" && 0 <= v123 && v123 < 1l);
                assert("Tensor range check" && 0 <= v125 && v125 < 4l);
                int v127;
                v127 = 4l * v123;
                int v128;
                v128 = v127 + v125;
                int v129;
                v129 = v88[v128];
                assert("Tensor range check" && 0 <= v123 && v123 < 1l);
                assert("Tensor range check" && 0 <= v125 && v125 < 4l);
                v121[v128] = v85;
                v122[v128] = v129;
                v125 += 1l ;
            }
            v123 += 1l ;
        }
        int v130;
        v130 = 0l;
        while (while_method_3(v130)){
            assert("Tensor range check" && 0 <= v130 && v130 < 1l);
            int v132;
            v132 = 16l * v130;
            int v133;
            v133 = v132 + v86;
            assert("Tensor range check" && 0 <= v130 && v130 < 1l);
            int v134;
            v134 = 4l * v130;
            int4* v135;
            v135 = reinterpret_cast<int4*>(v121 + v134);
            int4* v136;
            v136 = reinterpret_cast<int4*>(v81 + v133);
            assert("Pointer alignment check" && (unsigned long long)(v135) % 4l == 0 && (unsigned long long)(v136) % 4l == 0);
            *v136 = *v135;
            int4* v137;
            v137 = reinterpret_cast<int4*>(v122 + v134);
            int4* v138;
            v138 = reinterpret_cast<int4*>(v82 + v133);
            assert("Pointer alignment check" && (unsigned long long)(v137) % 4l == 0 && (unsigned long long)(v138) % 4l == 0);
            *v138 = *v137;
            v130 += 1l ;
        }
        assert("Tensor range check" && 0 <= v77 && v77 < 32l);
        v65 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v56 && v56 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v139;
    v139 = v1+v12;
    unsigned long long v141;
    v141 = v33 + 128ull;
    bool v142;
    v142 = v141 <= 81920ull;
    bool v143;
    v143 = v142 == false;
    if (v143){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v142);
    } else {
    }
    extern __shared__ unsigned char v145[];
    bool v146;
    v146 = v141 <= v141;
    bool v147;
    v147 = v146 == false;
    if (v147){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v146);
    } else {
    }
    float * * v149;
    v149 = reinterpret_cast<float * *>(&v145[0ull]);
    int * v151;
    v151 = reinterpret_cast<int *>(&v145[v33]);
    int v153;
    v153 = threadIdx.x;
    assert("Tensor range check" && 0 <= v153 && v153 < 32l);
    v149[v153] = v139;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v154;
    v154 = 0l <= v153;
    bool v155;
    v155 = v154 == false;
    if (v155){
        assert("The index needs to be zero or positive." && v154);
    } else {
    }
    int v157;
    v157 = v153 % 4l;
    int v158;
    v158 = v153 / 4l;
    bool v159;
    v159 = v158 < 8l;
    bool v160;
    v160 = v159 == false;
    if (v160){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v159);
    } else {
    }
    assert("Tensor range check" && 0 <= v158 && v158 < 8l);
    int v162;
    v162 = 0l;
    while (while_method_1(v162)){
        bool v164;
        v164 = 0l <= v158;
        bool v165;
        v165 = v164 && v159;
        bool v166;
        v166 = v165 == false;
        if (v166){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v165);
        } else {
        }
        bool v168;
        v168 = 0l <= v162;
        bool v170;
        if (v168){
            bool v169;
            v169 = v162 < 4l;
            v170 = v169;
        } else {
            v170 = false;
        }
        bool v171;
        v171 = v170 == false;
        if (v171){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v170);
        } else {
        }
        int v173;
        v173 = v162 * 8l;
        int v174;
        v174 = v173 + v158;
        assert("Tensor range check" && 0 <= v162 && v162 < 4l);
        int v175;
        v175 = 8l * v162;
        int v176;
        v176 = v175 + v158;
        float * v177;
        v177 = v149[v176];
        int v178;
        v178 = blockIdx.x;
        int v179;
        v179 = v178 * 32l;
        int v180;
        v180 = v179 + v174;
        assert("Tensor range check" && 0 <= v157 && v157 < 4l);
        int v181;
        v181 = 4l * v157;
        float v182[4l];
        int v183[4l];
        int v184;
        v184 = 0l;
        while (while_method_3(v184)){
            assert("Tensor range check" && 0 <= v184 && v184 < 1l);
            int v186;
            v186 = 4l * v184;
            assert("Tensor range check" && 0 <= v184 && v184 < 1l);
            int v187;
            v187 = 16l * v184;
            int v188;
            v188 = v187 + v181;
            int4* v189;
            v189 = reinterpret_cast<int4*>(v177 + v188);
            int4* v190;
            v190 = reinterpret_cast<int4*>(v182 + v186);
            assert("Pointer alignment check" && (unsigned long long)(v189) % 4l == 0 && (unsigned long long)(v190) % 4l == 0);
            *v190 = *v189;
            v184 += 1l ;
        }
        int v191;
        v191 = 0l;
        while (while_method_3(v191)){
            int v193;
            v193 = 0l;
            while (while_method_1(v193)){
                bool v195;
                v195 = 0l <= v193;
                bool v197;
                if (v195){
                    bool v196;
                    v196 = v193 < 4l;
                    v197 = v196;
                } else {
                    v197 = false;
                }
                bool v198;
                v198 = v197 == false;
                if (v198){
                    assert("The indices should be inside the range of the dimension." && v197);
                } else {
                }
                bool v200;
                v200 = 0l <= v157;
                bool v202;
                if (v200){
                    bool v201;
                    v201 = v157 < 4l;
                    v202 = v201;
                } else {
                    v202 = false;
                }
                bool v203;
                v203 = v202 == false;
                if (v203){
                    assert("The indices should be inside the range of the dimension." && v202);
                } else {
                }
                int v205;
                v205 = v157 * 4l;
                int v206;
                v206 = v193 + v205;
                bool v207;
                v207 = 0l <= v191;
                bool v209;
                if (v207){
                    bool v208;
                    v208 = v191 < 1l;
                    v209 = v208;
                } else {
                    v209 = false;
                }
                bool v210;
                v210 = v209 == false;
                if (v210){
                    assert("The indices should be inside the range of the dimension." && v209);
                } else {
                }
                int v212;
                v212 = v191 * 16l;
                int v213;
                v213 = v206 + v212;
                assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                assert("Tensor range check" && 0 <= v193 && v193 < 4l);
                int v214;
                v214 = 4l * v191;
                int v215;
                v215 = v214 + v193;
                v183[v215] = v213;
                v193 += 1l ;
            }
            v191 += 1l ;
        }
        int v216;
        v216 = 0l;
        while (while_method_3(v216)){
            assert("Tensor range check" && 0 <= v216 && v216 < 1l);
            assert("Tensor range check" && 0 <= v216 && v216 < 1l);
            v216 += 1l ;
        }
        assert("Tensor range check" && 0 <= v174 && v174 < 32l);
        v151[v174] = v180;
        v162 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v153 && v153 < 32l);
    int v218;
    v218 = v151[v153];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v219;
    v219 = threadIdx.x;
    assert("Tensor range check" && 0 <= v219 && v219 < 32l);
    v4[v219] = v218;
    float * v220;
    v220 = v1+v12;
    float * v222;
    v222 = v6+v20;
    unsigned long long v224;
    v224 = v33 + v29;
    bool v225;
    v225 = v224 <= 81920ull;
    bool v226;
    v226 = v225 == false;
    if (v226){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v225);
    } else {
    }
    extern __shared__ unsigned char v228[];
    bool v229;
    v229 = v224 <= v224;
    bool v230;
    v230 = v229 == false;
    if (v230){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v229);
    } else {
    }
    float * * v232;
    v232 = reinterpret_cast<float * *>(&v228[0ull]);
    float * * v234;
    v234 = reinterpret_cast<float * *>(&v228[v33]);
    int v236;
    v236 = threadIdx.x;
    assert("Tensor range check" && 0 <= v236 && v236 < 32l);
    v232[v236] = v220;
    v234[v236] = v222;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v237;
    v237 = 0l <= v236;
    bool v238;
    v238 = v237 == false;
    if (v238){
        assert("The index needs to be zero or positive." && v237);
    } else {
    }
    int v240;
    v240 = v236 % 4l;
    int v241;
    v241 = v236 / 4l;
    bool v242;
    v242 = v241 < 8l;
    bool v243;
    v243 = v242 == false;
    if (v243){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v242);
    } else {
    }
    assert("Tensor range check" && 0 <= v241 && v241 < 8l);
    int v245;
    v245 = 0l;
    while (while_method_1(v245)){
        bool v247;
        v247 = 0l <= v241;
        bool v248;
        v248 = v247 && v242;
        bool v249;
        v249 = v248 == false;
        if (v249){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v248);
        } else {
        }
        bool v251;
        v251 = 0l <= v245;
        bool v253;
        if (v251){
            bool v252;
            v252 = v245 < 4l;
            v253 = v252;
        } else {
            v253 = false;
        }
        bool v254;
        v254 = v253 == false;
        if (v254){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v253);
        } else {
        }
        int v256;
        v256 = v245 * 8l;
        int v257;
        v257 = v256 + v241;
        assert("Tensor range check" && 0 <= v245 && v245 < 4l);
        int v258;
        v258 = 8l * v245;
        int v259;
        v259 = v258 + v241;
        float * v260;
        v260 = v232[v259];
        float * v261;
        v261 = v234[v259];
        int v262;
        v262 = blockIdx.x;
        int v263;
        v263 = v262 * 32l;
        int v264;
        v264 = v263 + v257;
        assert("Tensor range check" && 0 <= v240 && v240 < 4l);
        int v265;
        v265 = 4l * v240;
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
            v271 = 16l * v268;
            int v272;
            v272 = v271 + v265;
            int4* v273;
            v273 = reinterpret_cast<int4*>(v260 + v272);
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
                v284 = 0l <= v240;
                bool v286;
                if (v284){
                    bool v285;
                    v285 = v240 < 4l;
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
                v289 = v240 * 4l;
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
                v296 = v275 * 16l;
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
        int v300;
        v300 = 0l;
        while (while_method_3(v300)){
            assert("Tensor range check" && 0 <= v300 && v300 < 1l);
            int v302;
            v302 = 16l * v300;
            int v303;
            v303 = v302 + v265;
            assert("Tensor range check" && 0 <= v300 && v300 < 1l);
            int v304;
            v304 = 4l * v300;
            int4* v305;
            v305 = reinterpret_cast<int4*>(v266 + v304);
            int4* v306;
            v306 = reinterpret_cast<int4*>(v261 + v303);
            assert("Pointer alignment check" && (unsigned long long)(v305) % 4l == 0 && (unsigned long long)(v306) % 4l == 0);
            *v306 = *v305;
            v300 += 1l ;
        }
        assert("Tensor range check" && 0 <= v257 && v257 < 32l);
        v245 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v236 && v236 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v307;
    v307 = v1+v12;
    float * v309;
    v309 = v7+v16;
    if (v226){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v225);
    } else {
    }
    extern __shared__ unsigned char v312[];
    if (v230){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v229);
    } else {
    }
    float * * v314;
    v314 = reinterpret_cast<float * *>(&v312[0ull]);
    float * * v316;
    v316 = reinterpret_cast<float * *>(&v312[v33]);
    int v318;
    v318 = threadIdx.x;
    assert("Tensor range check" && 0 <= v318 && v318 < 32l);
    v314[v318] = v307;
    v316[v318] = v309;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v319;
    v319 = 0l <= v318;
    bool v320;
    v320 = v319 == false;
    if (v320){
        assert("The index needs to be zero or positive." && v319);
    } else {
    }
    int v322;
    v322 = v318 % 4l;
    int v323;
    v323 = v318 / 4l;
    bool v324;
    v324 = v323 < 8l;
    bool v325;
    v325 = v324 == false;
    if (v325){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v324);
    } else {
    }
    assert("Tensor range check" && 0 <= v323 && v323 < 8l);
    int v327;
    v327 = 0l;
    while (while_method_1(v327)){
        bool v329;
        v329 = 0l <= v323;
        bool v330;
        v330 = v329 && v324;
        bool v331;
        v331 = v330 == false;
        if (v331){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v330);
        } else {
        }
        bool v333;
        v333 = 0l <= v327;
        bool v335;
        if (v333){
            bool v334;
            v334 = v327 < 4l;
            v335 = v334;
        } else {
            v335 = false;
        }
        bool v336;
        v336 = v335 == false;
        if (v336){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v335);
        } else {
        }
        int v338;
        v338 = v327 * 8l;
        int v339;
        v339 = v338 + v323;
        assert("Tensor range check" && 0 <= v327 && v327 < 4l);
        int v340;
        v340 = 8l * v327;
        int v341;
        v341 = v340 + v323;
        float * v342;
        v342 = v314[v341];
        float * v343;
        v343 = v316[v341];
        int v344;
        v344 = blockIdx.x;
        int v345;
        v345 = v344 * 32l;
        int v346;
        v346 = v345 + v339;
        assert("Tensor range check" && 0 <= v322 && v322 < 4l);
        int v347;
        v347 = 4l * v322;
        float v348[4l];
        int v349[4l];
        int v350;
        v350 = 0l;
        while (while_method_3(v350)){
            assert("Tensor range check" && 0 <= v350 && v350 < 1l);
            int v352;
            v352 = 4l * v350;
            assert("Tensor range check" && 0 <= v350 && v350 < 1l);
            int v353;
            v353 = 16l * v350;
            int v354;
            v354 = v353 + v347;
            int4* v355;
            v355 = reinterpret_cast<int4*>(v342 + v354);
            int4* v356;
            v356 = reinterpret_cast<int4*>(v348 + v352);
            assert("Pointer alignment check" && (unsigned long long)(v355) % 4l == 0 && (unsigned long long)(v356) % 4l == 0);
            *v356 = *v355;
            v350 += 1l ;
        }
        int v357;
        v357 = 0l;
        while (while_method_3(v357)){
            int v359;
            v359 = 0l;
            while (while_method_1(v359)){
                bool v361;
                v361 = 0l <= v359;
                bool v363;
                if (v361){
                    bool v362;
                    v362 = v359 < 4l;
                    v363 = v362;
                } else {
                    v363 = false;
                }
                bool v364;
                v364 = v363 == false;
                if (v364){
                    assert("The indices should be inside the range of the dimension." && v363);
                } else {
                }
                bool v366;
                v366 = 0l <= v322;
                bool v368;
                if (v366){
                    bool v367;
                    v367 = v322 < 4l;
                    v368 = v367;
                } else {
                    v368 = false;
                }
                bool v369;
                v369 = v368 == false;
                if (v369){
                    assert("The indices should be inside the range of the dimension." && v368);
                } else {
                }
                int v371;
                v371 = v322 * 4l;
                int v372;
                v372 = v359 + v371;
                bool v373;
                v373 = 0l <= v357;
                bool v375;
                if (v373){
                    bool v374;
                    v374 = v357 < 1l;
                    v375 = v374;
                } else {
                    v375 = false;
                }
                bool v376;
                v376 = v375 == false;
                if (v376){
                    assert("The indices should be inside the range of the dimension." && v375);
                } else {
                }
                int v378;
                v378 = v357 * 16l;
                int v379;
                v379 = v372 + v378;
                assert("Tensor range check" && 0 <= v357 && v357 < 1l);
                assert("Tensor range check" && 0 <= v359 && v359 < 4l);
                int v380;
                v380 = 4l * v357;
                int v381;
                v381 = v380 + v359;
                v349[v381] = v379;
                v359 += 1l ;
            }
            v357 += 1l ;
        }
        bool v382[4l];
        int v383;
        v383 = 0l;
        while (while_method_3(v383)){
            int v385;
            v385 = 0l;
            while (while_method_1(v385)){
                assert("Tensor range check" && 0 <= v383 && v383 < 1l);
                assert("Tensor range check" && 0 <= v385 && v385 < 4l);
                int v387;
                v387 = 4l * v383;
                int v388;
                v388 = v387 + v385;
                float v389;
                v389 = v348[v388];
                int v390;
                v390 = v349[v388];
                bool v391;
                v391 = v390 < 3l;
                assert("Tensor range check" && 0 <= v383 && v383 < 1l);
                assert("Tensor range check" && 0 <= v385 && v385 < 4l);
                v382[v388] = v391;
                v385 += 1l ;
            }
            v383 += 1l ;
        }
        float v392[4l];
        int v393;
        v393 = 0l;
        while (while_method_3(v393)){
            int v395;
            v395 = 0l;
            while (while_method_1(v395)){
                assert("Tensor range check" && 0 <= v393 && v393 < 1l);
                assert("Tensor range check" && 0 <= v395 && v395 < 4l);
                int v397;
                v397 = 4l * v393;
                int v398;
                v398 = v397 + v395;
                float v399;
                v399 = v348[v398];
                bool v400;
                v400 = v382[v398];
                float v403;
                if (v400){
                    bool v401;
                    v401 = 0.0f >= v399;
                    if (v401){
                        v403 = 0.0f;
                    } else {
                        v403 = v399;
                    }
                } else {
                    v403 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v393 && v393 < 1l);
                assert("Tensor range check" && 0 <= v395 && v395 < 4l);
                v392[v398] = v403;
                v395 += 1l ;
            }
            v393 += 1l ;
        }
        float v404;
        v404 = 0.0f;
        int v405;
        v405 = 0l;
        while (while_method_3(v405)){
            int v407;
            v407 = 0l;
            while (while_method_1(v407)){
                assert("Tensor range check" && 0 <= v405 && v405 < 1l);
                assert("Tensor range check" && 0 <= v407 && v407 < 4l);
                int v409;
                v409 = 4l * v405;
                int v410;
                v410 = v409 + v407;
                float v411;
                v411 = v392[v410];
                float v412;
                v412 = v404 + v411;
                v404 = v412;
                v407 += 1l ;
            }
            v405 += 1l ;
        }
        auto v413 = cooperative_groups::coalesced_threads();
        int v414;
        v414 = threadIdx.x;
        int v415;
        v415 = v414 / 4l;
        auto v416 = cooperative_groups::labeled_partition(v413,v415);
        Closure0 v417{};
        float v418;
        v418 = cooperative_groups::reduce(v416, v404, v417);
        int v419[4l];
        int v420;
        v420 = 0l;
        while (while_method_3(v420)){
            int v422;
            v422 = 0l;
            while (while_method_1(v422)){
                assert("Tensor range check" && 0 <= v420 && v420 < 1l);
                assert("Tensor range check" && 0 <= v422 && v422 < 4l);
                int v424;
                v424 = 4l * v420;
                int v425;
                v425 = v424 + v422;
                bool v426;
                v426 = v382[v425];
                int v427;
                if (v426){
                    v427 = 1l;
                } else {
                    v427 = 0l;
                }
                assert("Tensor range check" && 0 <= v420 && v420 < 1l);
                assert("Tensor range check" && 0 <= v422 && v422 < 4l);
                v419[v425] = v427;
                v422 += 1l ;
            }
            v420 += 1l ;
        }
        int v428;
        v428 = 0l;
        int v429;
        v429 = 0l;
        while (while_method_3(v429)){
            int v431;
            v431 = 0l;
            while (while_method_1(v431)){
                assert("Tensor range check" && 0 <= v429 && v429 < 1l);
                assert("Tensor range check" && 0 <= v431 && v431 < 4l);
                int v433;
                v433 = 4l * v429;
                int v434;
                v434 = v433 + v431;
                int v435;
                v435 = v419[v434];
                int v436;
                v436 = v428 + v435;
                v428 = v436;
                v431 += 1l ;
            }
            v429 += 1l ;
        }
        auto v437 = cooperative_groups::coalesced_threads();
        int v438;
        v438 = threadIdx.x;
        int v439;
        v439 = v438 / 4l;
        auto v440 = cooperative_groups::labeled_partition(v437,v439);
        Closure4 v441{};
        int v442;
        v442 = cooperative_groups::reduce(v440, v428, v441);
        float v443;
        v443 = (float)v442;
        float v444;
        v444 = 1.0f / v443;
        float v445[4l];
        int v446;
        v446 = 0l;
        while (while_method_3(v446)){
            int v448;
            v448 = 0l;
            while (while_method_1(v448)){
                assert("Tensor range check" && 0 <= v446 && v446 < 1l);
                assert("Tensor range check" && 0 <= v448 && v448 < 4l);
                int v450;
                v450 = 4l * v446;
                int v451;
                v451 = v450 + v448;
                float v452;
                v452 = v392[v451];
                bool v453;
                v453 = v382[v451];
                bool v454;
                v454 = v453 == false;
                float v459;
                if (v454){
                    v459 = 0.0f;
                } else {
                    bool v455;
                    v455 = v418 == 0.0f;
                    bool v456;
                    v456 = v455 != true;
                    if (v456){
                        float v457;
                        v457 = v452 / v418;
                        v459 = v457;
                    } else {
                        v459 = v444;
                    }
                }
                assert("Tensor range check" && 0 <= v446 && v446 < 1l);
                assert("Tensor range check" && 0 <= v448 && v448 < 4l);
                v445[v451] = v459;
                v448 += 1l ;
            }
            v446 += 1l ;
        }
        int v460;
        v460 = 0l;
        while (while_method_3(v460)){
            assert("Tensor range check" && 0 <= v460 && v460 < 1l);
            int v462;
            v462 = 16l * v460;
            int v463;
            v463 = v462 + v347;
            assert("Tensor range check" && 0 <= v460 && v460 < 1l);
            int v464;
            v464 = 4l * v460;
            int4* v465;
            v465 = reinterpret_cast<int4*>(v445 + v464);
            int4* v466;
            v466 = reinterpret_cast<int4*>(v343 + v463);
            assert("Pointer alignment check" && (unsigned long long)(v465) % 4l == 0 && (unsigned long long)(v466) % 4l == 0);
            *v466 = *v465;
            v460 += 1l ;
        }
        assert("Tensor range check" && 0 <= v339 && v339 < 32l);
        v327 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v318 && v318 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v467;
    v467 = v1+v12;
    if (v143){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v142);
    } else {
    }
    extern __shared__ unsigned char v470[];
    if (v147){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v146);
    } else {
    }
    float * * v472;
    v472 = reinterpret_cast<float * *>(&v470[0ull]);
    int * v474;
    v474 = reinterpret_cast<int *>(&v470[v33]);
    int v476;
    v476 = threadIdx.x;
    assert("Tensor range check" && 0 <= v476 && v476 < 32l);
    v472[v476] = v467;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v477;
    v477 = 0l <= v476;
    bool v478;
    v478 = v477 == false;
    if (v478){
        assert("The index needs to be zero or positive." && v477);
    } else {
    }
    int v480;
    v480 = v476 % 4l;
    int v481;
    v481 = v476 / 4l;
    bool v482;
    v482 = v481 < 8l;
    bool v483;
    v483 = v482 == false;
    if (v483){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v482);
    } else {
    }
    assert("Tensor range check" && 0 <= v481 && v481 < 8l);
    int v485;
    v485 = 0l;
    while (while_method_1(v485)){
        bool v487;
        v487 = 0l <= v481;
        bool v488;
        v488 = v487 && v482;
        bool v489;
        v489 = v488 == false;
        if (v489){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v488);
        } else {
        }
        bool v491;
        v491 = 0l <= v485;
        bool v493;
        if (v491){
            bool v492;
            v492 = v485 < 4l;
            v493 = v492;
        } else {
            v493 = false;
        }
        bool v494;
        v494 = v493 == false;
        if (v494){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v493);
        } else {
        }
        int v496;
        v496 = v485 * 8l;
        int v497;
        v497 = v496 + v481;
        assert("Tensor range check" && 0 <= v485 && v485 < 4l);
        int v498;
        v498 = 8l * v485;
        int v499;
        v499 = v498 + v481;
        float * v500;
        v500 = v472[v499];
        int v501;
        v501 = blockIdx.x;
        int v502;
        v502 = v501 * 32l;
        int v503;
        v503 = v502 + v497;
        assert("Tensor range check" && 0 <= v480 && v480 < 4l);
        int v504;
        v504 = 4l * v480;
        float v505[4l];
        int v506[4l];
        int v507;
        v507 = 0l;
        while (while_method_3(v507)){
            assert("Tensor range check" && 0 <= v507 && v507 < 1l);
            int v509;
            v509 = 4l * v507;
            assert("Tensor range check" && 0 <= v507 && v507 < 1l);
            int v510;
            v510 = 16l * v507;
            int v511;
            v511 = v510 + v504;
            int4* v512;
            v512 = reinterpret_cast<int4*>(v500 + v511);
            int4* v513;
            v513 = reinterpret_cast<int4*>(v505 + v509);
            assert("Pointer alignment check" && (unsigned long long)(v512) % 4l == 0 && (unsigned long long)(v513) % 4l == 0);
            *v513 = *v512;
            v507 += 1l ;
        }
        int v514;
        v514 = 0l;
        while (while_method_3(v514)){
            int v516;
            v516 = 0l;
            while (while_method_1(v516)){
                bool v518;
                v518 = 0l <= v516;
                bool v520;
                if (v518){
                    bool v519;
                    v519 = v516 < 4l;
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
                bool v523;
                v523 = 0l <= v480;
                bool v525;
                if (v523){
                    bool v524;
                    v524 = v480 < 4l;
                    v525 = v524;
                } else {
                    v525 = false;
                }
                bool v526;
                v526 = v525 == false;
                if (v526){
                    assert("The indices should be inside the range of the dimension." && v525);
                } else {
                }
                int v528;
                v528 = v480 * 4l;
                int v529;
                v529 = v516 + v528;
                bool v530;
                v530 = 0l <= v514;
                bool v532;
                if (v530){
                    bool v531;
                    v531 = v514 < 1l;
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
                v535 = v514 * 16l;
                int v536;
                v536 = v529 + v535;
                assert("Tensor range check" && 0 <= v514 && v514 < 1l);
                assert("Tensor range check" && 0 <= v516 && v516 < 4l);
                int v537;
                v537 = 4l * v514;
                int v538;
                v538 = v537 + v516;
                v506[v538] = v536;
                v516 += 1l ;
            }
            v514 += 1l ;
        }
        bool v539[4l];
        int v540;
        v540 = 0l;
        while (while_method_3(v540)){
            int v542;
            v542 = 0l;
            while (while_method_1(v542)){
                assert("Tensor range check" && 0 <= v540 && v540 < 1l);
                assert("Tensor range check" && 0 <= v542 && v542 < 4l);
                int v544;
                v544 = 4l * v540;
                int v545;
                v545 = v544 + v542;
                float v546;
                v546 = v505[v545];
                int v547;
                v547 = v506[v545];
                bool v548;
                v548 = v547 < 3l;
                assert("Tensor range check" && 0 <= v540 && v540 < 1l);
                assert("Tensor range check" && 0 <= v542 && v542 < 4l);
                v539[v545] = v548;
                v542 += 1l ;
            }
            v540 += 1l ;
        }
        int v549[4l];
        int v550;
        v550 = 0l;
        while (while_method_3(v550)){
            int v552;
            v552 = 0l;
            while (while_method_1(v552)){
                assert("Tensor range check" && 0 <= v550 && v550 < 1l);
                assert("Tensor range check" && 0 <= v552 && v552 < 4l);
                int v554;
                v554 = 4l * v550;
                int v555;
                v555 = v554 + v552;
                bool v556;
                v556 = v539[v555];
                int v557;
                if (v556){
                    v557 = 1l;
                } else {
                    v557 = 0l;
                }
                assert("Tensor range check" && 0 <= v550 && v550 < 1l);
                assert("Tensor range check" && 0 <= v552 && v552 < 4l);
                v549[v555] = v557;
                v552 += 1l ;
            }
            v550 += 1l ;
        }
        int v558;
        v558 = 0l;
        int v559;
        v559 = 0l;
        while (while_method_3(v559)){
            int v561;
            v561 = 0l;
            while (while_method_1(v561)){
                assert("Tensor range check" && 0 <= v559 && v559 < 1l);
                assert("Tensor range check" && 0 <= v561 && v561 < 4l);
                int v563;
                v563 = 4l * v559;
                int v564;
                v564 = v563 + v561;
                int v565;
                v565 = v549[v564];
                int v566;
                v566 = v558 + v565;
                v558 = v566;
                v561 += 1l ;
            }
            v559 += 1l ;
        }
        auto v567 = cooperative_groups::coalesced_threads();
        int v568;
        v568 = threadIdx.x;
        int v569;
        v569 = v568 / 4l;
        auto v570 = cooperative_groups::labeled_partition(v567,v569);
        Closure4 v571{};
        int v572;
        v572 = cooperative_groups::reduce(v570, v558, v571);
        float v573[4l];
        int v574;
        v574 = 0l;
        while (while_method_3(v574)){
            int v576;
            v576 = 0l;
            while (while_method_1(v576)){
                assert("Tensor range check" && 0 <= v574 && v574 < 1l);
                assert("Tensor range check" && 0 <= v576 && v576 < 4l);
                int v578;
                v578 = 4l * v574;
                int v579;
                v579 = v578 + v576;
                float v580;
                v580 = v505[v579];
                bool v581;
                v581 = v539[v579];
                float v582;
                if (v581){
                    v582 = v580;
                } else {
                    v582 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v574 && v574 < 1l);
                assert("Tensor range check" && 0 <= v576 && v576 < 4l);
                v573[v579] = v582;
                v576 += 1l ;
            }
            v574 += 1l ;
        }
        float v583;
        v583 = 0.0f;
        int v584;
        v584 = 0l;
        while (while_method_3(v584)){
            int v586;
            v586 = 0l;
            while (while_method_1(v586)){
                assert("Tensor range check" && 0 <= v584 && v584 < 1l);
                assert("Tensor range check" && 0 <= v586 && v586 < 4l);
                int v588;
                v588 = 4l * v584;
                int v589;
                v589 = v588 + v586;
                float v590;
                v590 = v573[v589];
                float v591;
                v591 = v583 + v590;
                v583 = v591;
                v586 += 1l ;
            }
            v584 += 1l ;
        }
        auto v592 = cooperative_groups::coalesced_threads();
        int v593;
        v593 = threadIdx.x;
        int v594;
        v594 = v593 / 4l;
        auto v595 = cooperative_groups::labeled_partition(v592,v594);
        Closure0 v596{};
        float v597;
        v597 = cooperative_groups::reduce(v595, v583, v596);
        float v598;
        v598 = (float)v572;
        float v599;
        v599 = v597 / v598;
        float v600[4l];
        int v601;
        v601 = 0l;
        while (while_method_3(v601)){
            int v603;
            v603 = 0l;
            while (while_method_1(v603)){
                assert("Tensor range check" && 0 <= v601 && v601 < 1l);
                assert("Tensor range check" && 0 <= v603 && v603 < 4l);
                int v605;
                v605 = 4l * v601;
                int v606;
                v606 = v605 + v603;
                float v607;
                v607 = v505[v606];
                bool v608;
                v608 = v539[v606];
                float v609;
                if (v608){
                    v609 = v607;
                } else {
                    v609 = -1.0f / 0.0f;
                }
                float v610;
                v610 = v609 - v599;
                float v611;
                v611 = exp(v610);
                assert("Tensor range check" && 0 <= v601 && v601 < 1l);
                assert("Tensor range check" && 0 <= v603 && v603 < 4l);
                v600[v606] = v611;
                v603 += 1l ;
            }
            v601 += 1l ;
        }
        float v612;
        v612 = 0.0f;
        int v613;
        v613 = 0l;
        while (while_method_3(v613)){
            int v615;
            v615 = 0l;
            while (while_method_1(v615)){
                assert("Tensor range check" && 0 <= v613 && v613 < 1l);
                assert("Tensor range check" && 0 <= v615 && v615 < 4l);
                int v617;
                v617 = 4l * v613;
                int v618;
                v618 = v617 + v615;
                float v619;
                v619 = v600[v618];
                float v620;
                v620 = v612 + v619;
                v612 = v620;
                v615 += 1l ;
            }
            v613 += 1l ;
        }
        auto v621 = cooperative_groups::coalesced_threads();
        int v622;
        v622 = threadIdx.x;
        int v623;
        v623 = v622 / 4l;
        auto v624 = cooperative_groups::labeled_partition(v621,v623);
        float v625;
        v625 = cooperative_groups::reduce(v624, v612, v596);
        float v626[4l];
        int v627;
        v627 = 0l;
        while (while_method_3(v627)){
            int v629;
            v629 = 0l;
            while (while_method_1(v629)){
                assert("Tensor range check" && 0 <= v627 && v627 < 1l);
                assert("Tensor range check" && 0 <= v629 && v629 < 4l);
                int v631;
                v631 = 4l * v627;
                int v632;
                v632 = v631 + v629;
                float v633;
                v633 = v600[v632];
                float v634;
                v634 = v633 / v625;
                assert("Tensor range check" && 0 <= v627 && v627 < 1l);
                assert("Tensor range check" && 0 <= v629 && v629 < 4l);
                v626[v632] = v634;
                v629 += 1l ;
            }
            v627 += 1l ;
        }
        float v635[4l];
        float v636;
        v636 = 0.0f;
        int v637;
        v637 = 0l;
        while (while_method_3(v637)){
            assert("Tensor range check" && 0 <= v637 && v637 < 1l);
            int v639;
            v639 = 4l * v637;
            assert("Tensor range check" && 0 <= v637 && v637 < 1l);
            int v640; float v641;
            Tuple0 tmp46 = Tuple0{0l, 0.0f};
            v640 = tmp46.v0; v641 = tmp46.v1;
            while (while_method_1(v640)){
                assert("Tensor range check" && 0 <= v640 && v640 < 4l);
                int v643;
                v643 = v640 + v639;
                float v644;
                v644 = v626[v643];
                float v645;
                v645 = v641 + v644;
                v641 = v645;
                v640 += 1l ;
            }
            auto v646 = cooperative_groups::coalesced_threads();
            int v647;
            v647 = threadIdx.x;
            int v648;
            v648 = v647 / 4l;
            auto v649 = cooperative_groups::labeled_partition(v646,v648);
            Closure2 v650{};
            float v651;
            v651 = cooperative_groups::inclusive_scan(v649, v641, v650);
            float v652;
            v652 = v649.shfl_up(v651,1);
            bool v653;
            v653 = v649.thread_rank() == 0;
            float v654;
            if (v653){
                v654 = 0.0f;
            } else {
                v654 = v652;
            }
            float v655;
            v655 = v649.shfl(v651,v649.num_threads()-1);
            float v656;
            v656 = v636 + v654;
            int v657; float v658;
            Tuple0 tmp47 = Tuple0{0l, v656};
            v657 = tmp47.v0; v658 = tmp47.v1;
            while (while_method_1(v657)){
                assert("Tensor range check" && 0 <= v657 && v657 < 4l);
                int v660;
                v660 = v657 + v639;
                float v661;
                v661 = v626[v660];
                float v662;
                v662 = v658 + v661;
                assert("Tensor range check" && 0 <= v657 && v657 < 4l);
                v635[v660] = v662;
                v658 = v662;
                v657 += 1l ;
            }
            float v663;
            v663 = v636 + v655;
            v636 = v663;
            v637 += 1l ;
        }
        float v664[4l];
        bool v665[4l];
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
                v672 = v635[v671];
                float v673;
                v673 = v626[v671];
                bool v674;
                v674 = v673 > 0.0f;
                assert("Tensor range check" && 0 <= v666 && v666 < 1l);
                assert("Tensor range check" && 0 <= v668 && v668 < 4l);
                v664[v671] = v672;
                v665[v671] = v674;
                v668 += 1l ;
            }
            v666 += 1l ;
        }
        float v675; bool v676;
        Tuple3 tmp48 = Tuple3{-1.0f / 0.0f, false};
        v675 = tmp48.v0; v676 = tmp48.v1;
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
                v683 = v664[v682];
                bool v684;
                v684 = v665[v682];
                float v691; bool v692;
                if (v676){
                    if (v684){
                        bool v685;
                        v685 = v675 >= v683;
                        float v686;
                        if (v685){
                            v686 = v675;
                        } else {
                            v686 = v683;
                        }
                        v691 = v686; v692 = true;
                    } else {
                        v691 = v675; v692 = v676;
                    }
                } else {
                    if (v684){
                        v691 = v683; v692 = v684;
                    } else {
                        v691 = v675; v692 = v676;
                    }
                }
                v675 = v691;
                v676 = v692;
                v679 += 1l ;
            }
            v677 += 1l ;
        }
        auto v693 = cooperative_groups::coalesced_threads();
        int v694;
        v694 = threadIdx.x;
        int v695;
        v695 = v694 / 4l;
        auto v696 = cooperative_groups::labeled_partition(v693,v695);
        Closure5 v697{};
        float v698; bool v699;
        Tuple3 tmp49 = cooperative_groups::reduce(v696, Tuple3{v675, v676}, v697);
        v698 = tmp49.v0; v699 = tmp49.v1;
        bool v700;
        v700 = v699 == false;
        if (v700){
            assert("The local reduce must be true." && v699);
        } else {
        }
        float v702[4l];
        int v703[4l];
        int v704;
        v704 = 0l;
        while (while_method_3(v704)){
            int v706;
            v706 = 0l;
            while (while_method_1(v706)){
                assert("Tensor range check" && 0 <= v704 && v704 < 1l);
                assert("Tensor range check" && 0 <= v706 && v706 < 4l);
                int v708;
                v708 = 4l * v704;
                int v709;
                v709 = v708 + v706;
                int v710;
                v710 = v506[v709];
                float v711;
                v711 = curand_uniform(&v10);
                assert("Tensor range check" && 0 <= v704 && v704 < 1l);
                assert("Tensor range check" && 0 <= v706 && v706 < 4l);
                v702[v709] = v711;
                v703[v709] = v710;
                v706 += 1l ;
            }
            v704 += 1l ;
        }
        float v712; int v713;
        Tuple1 tmp50 = Tuple1{0.0f, 2147483647l};
        v712 = tmp50.v0; v713 = tmp50.v1;
        int v714;
        v714 = 0l;
        while (while_method_3(v714)){
            int v716;
            v716 = 0l;
            while (while_method_1(v716)){
                assert("Tensor range check" && 0 <= v714 && v714 < 1l);
                assert("Tensor range check" && 0 <= v716 && v716 < 4l);
                int v718;
                v718 = 4l * v714;
                int v719;
                v719 = v718 + v716;
                float v720;
                v720 = v702[v719];
                int v721;
                v721 = v703[v719];
                bool v722;
                v722 = v713 < v721;
                float v723; int v724;
                if (v722){
                    v723 = v712; v724 = v713;
                } else {
                    v723 = v720; v724 = v721;
                }
                v712 = v723;
                v713 = v724;
                v716 += 1l ;
            }
            v714 += 1l ;
        }
        auto v725 = cooperative_groups::coalesced_threads();
        int v726;
        v726 = threadIdx.x;
        int v727;
        v727 = v726 / 4l;
        auto v728 = cooperative_groups::labeled_partition(v725,v727);
        Closure6 v729{};
        float v730; int v731;
        Tuple1 tmp51 = cooperative_groups::reduce(v728, Tuple1{v712, v713}, v729);
        v730 = tmp51.v0; v731 = tmp51.v1;
        float v732;
        v732 = v698 * v730;
        int v733[4l];
        bool v734[4l];
        int v735;
        v735 = 0l;
        while (while_method_3(v735)){
            int v737;
            v737 = 0l;
            while (while_method_1(v737)){
                assert("Tensor range check" && 0 <= v735 && v735 < 1l);
                assert("Tensor range check" && 0 <= v737 && v737 < 4l);
                int v739;
                v739 = 4l * v735;
                int v740;
                v740 = v739 + v737;
                float v741;
                v741 = v664[v740];
                bool v742;
                v742 = v665[v740];
                int v743;
                v743 = v506[v740];
                int v746; bool v747;
                if (v742){
                    float v744;
                    v744 = v741 - v732;
                    bool v745;
                    v745 = v744 >= 0.0f;
                    v746 = v743; v747 = v745;
                } else {
                    v746 = 2147483647l; v747 = false;
                }
                assert("Tensor range check" && 0 <= v735 && v735 < 1l);
                assert("Tensor range check" && 0 <= v737 && v737 < 4l);
                v733[v740] = v746;
                v734[v740] = v747;
                v737 += 1l ;
            }
            v735 += 1l ;
        }
        int v748; bool v749;
        Tuple4 tmp52 = Tuple4{2147483647l, false};
        v748 = tmp52.v0; v749 = tmp52.v1;
        int v750;
        v750 = 0l;
        while (while_method_3(v750)){
            int v752;
            v752 = 0l;
            while (while_method_1(v752)){
                assert("Tensor range check" && 0 <= v750 && v750 < 1l);
                assert("Tensor range check" && 0 <= v752 && v752 < 4l);
                int v754;
                v754 = 4l * v750;
                int v755;
                v755 = v754 + v752;
                int v756;
                v756 = v733[v755];
                bool v757;
                v757 = v734[v755];
                int v764; bool v765;
                if (v749){
                    if (v757){
                        bool v758;
                        v758 = v748 < v756;
                        int v759;
                        if (v758){
                            v759 = v748;
                        } else {
                            v759 = v756;
                        }
                        v764 = v759; v765 = true;
                    } else {
                        v764 = v748; v765 = v749;
                    }
                } else {
                    if (v757){
                        v764 = v756; v765 = v757;
                    } else {
                        v764 = v748; v765 = v749;
                    }
                }
                v748 = v764;
                v749 = v765;
                v752 += 1l ;
            }
            v750 += 1l ;
        }
        auto v766 = cooperative_groups::coalesced_threads();
        int v767;
        v767 = threadIdx.x;
        int v768;
        v768 = v767 / 4l;
        auto v769 = cooperative_groups::labeled_partition(v766,v768);
        Closure7 v770{};
        int v771; bool v772;
        Tuple4 tmp53 = cooperative_groups::reduce(v769, Tuple4{v748, v749}, v770);
        v771 = tmp53.v0; v772 = tmp53.v1;
        bool v773;
        v773 = v772 == false;
        if (v773){
            assert("The local reduce must be true." && v772);
        } else {
        }
        int v775;
        v775 = 0l;
        while (while_method_3(v775)){
            assert("Tensor range check" && 0 <= v775 && v775 < 1l);
            assert("Tensor range check" && 0 <= v775 && v775 < 1l);
            v775 += 1l ;
        }
        assert("Tensor range check" && 0 <= v497 && v497 < 32l);
        v474[v497] = v771;
        v485 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v476 && v476 < 32l);
    int v777;
    v777 = v474[v476];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v778;
    v778 = threadIdx.x;
    assert("Tensor range check" && 0 <= v778 && v778 < 32l);
    v5[v778] = v777;
    return ;
}
extern "C" __global__ void entry3(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    unsigned long long v9;
    v9 = (unsigned long long)v8;
    curandStatePhilox4_32_10_t v10;
    curand_init(12344321ull,v9,0ull,&v10);
    int v11;
    v11 = threadIdx.x;
    assert("Tensor range check" && 0 <= v11 && v11 < 32l);
    int v12;
    v12 = 256l * v11;
    int v13;
    v13 = threadIdx.x;
    assert("Tensor range check" && 0 <= v13 && v13 < 32l);
    int v14;
    v14 = 256l * v13;
    int v15;
    v15 = threadIdx.x;
    assert("Tensor range check" && 0 <= v15 && v15 < 32l);
    int v16;
    v16 = 256l * v15;
    int v17;
    v17 = threadIdx.x;
    assert("Tensor range check" && 0 <= v17 && v17 < 32l);
    int v18;
    v18 = 256l * v17;
    int v19;
    v19 = threadIdx.x;
    assert("Tensor range check" && 0 <= v19 && v19 < 32l);
    int v20;
    v20 = 256l * v19;
    float * v21;
    v21 = v1+v12;
    int * v23;
    v23 = v2+v18;
    int * v25;
    v25 = v3+v18;
    int v27;
    v27 = sizeof(float *);
    unsigned long long v28;
    v28 = (unsigned long long)v27;
    unsigned long long v29;
    v29 = 32ull * v28;
    unsigned long long v30;
    v30 = v29 + 16ull;
    unsigned long long v31;
    v31 = v30 - 1ull;
    unsigned long long v32;
    v32 = v31 % 16ull;
    unsigned long long v33;
    v33 = v31 - v32;
    int v34;
    v34 = sizeof(int *);
    unsigned long long v35;
    v35 = (unsigned long long)v34;
    unsigned long long v36;
    v36 = 32ull * v35;
    unsigned long long v37;
    v37 = v33 + v36;
    unsigned long long v38;
    v38 = v37 + 16ull;
    unsigned long long v39;
    v39 = v38 - 1ull;
    unsigned long long v40;
    v40 = v39 % 16ull;
    unsigned long long v41;
    v41 = v39 - v40;
    unsigned long long v42;
    v42 = v41 + v36;
    bool v43;
    v43 = v42 <= 81920ull;
    bool v44;
    v44 = v43 == false;
    if (v44){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v43);
    } else {
    }
    extern __shared__ unsigned char v46[];
    bool v47;
    v47 = v42 <= v42;
    bool v48;
    v48 = v47 == false;
    if (v48){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v47);
    } else {
    }
    float * * v50;
    v50 = reinterpret_cast<float * *>(&v46[0ull]);
    int * * v52;
    v52 = reinterpret_cast<int * *>(&v46[v33]);
    int * * v54;
    v54 = reinterpret_cast<int * *>(&v46[v41]);
    int v56;
    v56 = threadIdx.x;
    assert("Tensor range check" && 0 <= v56 && v56 < 32l);
    v50[v56] = v21;
    v52[v56] = v23;
    v54[v56] = v25;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v57;
    v57 = 0l <= v56;
    bool v58;
    v58 = v57 == false;
    if (v58){
        assert("The index needs to be zero or positive." && v57);
    } else {
    }
    int v60;
    v60 = v56 % 32l;
    int v61;
    v61 = v56 / 32l;
    bool v62;
    v62 = v61 < 1l;
    bool v63;
    v63 = v62 == false;
    if (v63){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v62);
    } else {
    }
    assert("Tensor range check" && 0 <= v61 && v61 < 1l);
    int v65;
    v65 = 0l;
    while (while_method_4(v65)){
        bool v67;
        v67 = 0l <= v61;
        bool v68;
        v68 = v67 && v62;
        bool v69;
        v69 = v68 == false;
        if (v69){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v68);
        } else {
        }
        bool v71;
        v71 = 0l <= v65;
        bool v73;
        if (v71){
            bool v72;
            v72 = v65 < 32l;
            v73 = v72;
        } else {
            v73 = false;
        }
        bool v74;
        v74 = v73 == false;
        if (v74){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v73);
        } else {
        }
        int v76;
        v76 = v65 + v61;
        assert("Tensor range check" && 0 <= v65 && v65 < 32l);
        float * v77;
        v77 = v50[v76];
        int * v78;
        v78 = v52[v76];
        int * v79;
        v79 = v54[v76];
        int v80;
        v80 = blockIdx.x;
        int v81;
        v81 = v80 * 32l;
        int v82;
        v82 = v81 + v76;
        assert("Tensor range check" && 0 <= v60 && v60 < 32l);
        int v83;
        v83 = 4l * v60;
        float v84[8l];
        int v85[8l];
        int v86;
        v86 = 0l;
        while (while_method_5(v86)){
            assert("Tensor range check" && 0 <= v86 && v86 < 2l);
            int v88;
            v88 = 4l * v86;
            assert("Tensor range check" && 0 <= v86 && v86 < 2l);
            int v89;
            v89 = 128l * v86;
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
        while (while_method_5(v93)){
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
                v102 = 0l <= v60;
                bool v104;
                if (v102){
                    bool v103;
                    v103 = v60 < 32l;
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
                v107 = v60 * 4l;
                int v108;
                v108 = v95 + v107;
                bool v109;
                v109 = 0l <= v93;
                bool v111;
                if (v109){
                    bool v110;
                    v110 = v93 < 2l;
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
                v114 = v93 * 128l;
                int v115;
                v115 = v108 + v114;
                assert("Tensor range check" && 0 <= v93 && v93 < 2l);
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
        int v118[8l];
        int v119[8l];
        int v120;
        v120 = 0l;
        while (while_method_5(v120)){
            int v122;
            v122 = 0l;
            while (while_method_1(v122)){
                assert("Tensor range check" && 0 <= v120 && v120 < 2l);
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                int v124;
                v124 = 4l * v120;
                int v125;
                v125 = v124 + v122;
                int v126;
                v126 = v85[v125];
                assert("Tensor range check" && 0 <= v120 && v120 < 2l);
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                v118[v125] = v82;
                v119[v125] = v126;
                v122 += 1l ;
            }
            v120 += 1l ;
        }
        int v127;
        v127 = 0l;
        while (while_method_5(v127)){
            assert("Tensor range check" && 0 <= v127 && v127 < 2l);
            int v129;
            v129 = 128l * v127;
            int v130;
            v130 = v129 + v83;
            assert("Tensor range check" && 0 <= v127 && v127 < 2l);
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
        assert("Tensor range check" && 0 <= v76 && v76 < 32l);
        v65 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v56 && v56 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v136;
    v136 = v1+v12;
    unsigned long long v138;
    v138 = v33 + 128ull;
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
    v148 = reinterpret_cast<int *>(&v142[v33]);
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
    v154 = v150 % 32l;
    int v155;
    v155 = v150 / 32l;
    bool v156;
    v156 = v155 < 1l;
    bool v157;
    v157 = v156 == false;
    if (v157){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v156);
    } else {
    }
    assert("Tensor range check" && 0 <= v155 && v155 < 1l);
    int v159;
    v159 = 0l;
    while (while_method_4(v159)){
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
            v166 = v159 < 32l;
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
        v170 = v159 + v155;
        assert("Tensor range check" && 0 <= v159 && v159 < 32l);
        float * v171;
        v171 = v146[v170];
        int v172;
        v172 = blockIdx.x;
        int v173;
        v173 = v172 * 32l;
        int v174;
        v174 = v173 + v170;
        assert("Tensor range check" && 0 <= v154 && v154 < 32l);
        int v175;
        v175 = 4l * v154;
        float v176[8l];
        int v177[8l];
        int v178;
        v178 = 0l;
        while (while_method_5(v178)){
            assert("Tensor range check" && 0 <= v178 && v178 < 2l);
            int v180;
            v180 = 4l * v178;
            assert("Tensor range check" && 0 <= v178 && v178 < 2l);
            int v181;
            v181 = 128l * v178;
            int v182;
            v182 = v181 + v175;
            int4* v183;
            v183 = reinterpret_cast<int4*>(v171 + v182);
            int4* v184;
            v184 = reinterpret_cast<int4*>(v176 + v180);
            assert("Pointer alignment check" && (unsigned long long)(v183) % 4l == 0 && (unsigned long long)(v184) % 4l == 0);
            *v184 = *v183;
            v178 += 1l ;
        }
        int v185;
        v185 = 0l;
        while (while_method_5(v185)){
            int v187;
            v187 = 0l;
            while (while_method_1(v187)){
                bool v189;
                v189 = 0l <= v187;
                bool v191;
                if (v189){
                    bool v190;
                    v190 = v187 < 4l;
                    v191 = v190;
                } else {
                    v191 = false;
                }
                bool v192;
                v192 = v191 == false;
                if (v192){
                    assert("The indices should be inside the range of the dimension." && v191);
                } else {
                }
                bool v194;
                v194 = 0l <= v154;
                bool v196;
                if (v194){
                    bool v195;
                    v195 = v154 < 32l;
                    v196 = v195;
                } else {
                    v196 = false;
                }
                bool v197;
                v197 = v196 == false;
                if (v197){
                    assert("The indices should be inside the range of the dimension." && v196);
                } else {
                }
                int v199;
                v199 = v154 * 4l;
                int v200;
                v200 = v187 + v199;
                bool v201;
                v201 = 0l <= v185;
                bool v203;
                if (v201){
                    bool v202;
                    v202 = v185 < 2l;
                    v203 = v202;
                } else {
                    v203 = false;
                }
                bool v204;
                v204 = v203 == false;
                if (v204){
                    assert("The indices should be inside the range of the dimension." && v203);
                } else {
                }
                int v206;
                v206 = v185 * 128l;
                int v207;
                v207 = v200 + v206;
                assert("Tensor range check" && 0 <= v185 && v185 < 2l);
                assert("Tensor range check" && 0 <= v187 && v187 < 4l);
                int v208;
                v208 = 4l * v185;
                int v209;
                v209 = v208 + v187;
                v177[v209] = v207;
                v187 += 1l ;
            }
            v185 += 1l ;
        }
        int v210;
        v210 = 0l;
        while (while_method_5(v210)){
            assert("Tensor range check" && 0 <= v210 && v210 < 2l);
            assert("Tensor range check" && 0 <= v210 && v210 < 2l);
            v210 += 1l ;
        }
        assert("Tensor range check" && 0 <= v170 && v170 < 32l);
        v148[v170] = v174;
        v159 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v150 && v150 < 32l);
    int v212;
    v212 = v148[v150];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v213;
    v213 = threadIdx.x;
    assert("Tensor range check" && 0 <= v213 && v213 < 32l);
    v4[v213] = v212;
    float * v214;
    v214 = v1+v12;
    float * v216;
    v216 = v6+v20;
    unsigned long long v218;
    v218 = v33 + v29;
    bool v219;
    v219 = v218 <= 81920ull;
    bool v220;
    v220 = v219 == false;
    if (v220){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v219);
    } else {
    }
    extern __shared__ unsigned char v222[];
    bool v223;
    v223 = v218 <= v218;
    bool v224;
    v224 = v223 == false;
    if (v224){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v223);
    } else {
    }
    float * * v226;
    v226 = reinterpret_cast<float * *>(&v222[0ull]);
    float * * v228;
    v228 = reinterpret_cast<float * *>(&v222[v33]);
    int v230;
    v230 = threadIdx.x;
    assert("Tensor range check" && 0 <= v230 && v230 < 32l);
    v226[v230] = v214;
    v228[v230] = v216;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v231;
    v231 = 0l <= v230;
    bool v232;
    v232 = v231 == false;
    if (v232){
        assert("The index needs to be zero or positive." && v231);
    } else {
    }
    int v234;
    v234 = v230 % 32l;
    int v235;
    v235 = v230 / 32l;
    bool v236;
    v236 = v235 < 1l;
    bool v237;
    v237 = v236 == false;
    if (v237){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v236);
    } else {
    }
    assert("Tensor range check" && 0 <= v235 && v235 < 1l);
    int v239;
    v239 = 0l;
    while (while_method_4(v239)){
        bool v241;
        v241 = 0l <= v235;
        bool v242;
        v242 = v241 && v236;
        bool v243;
        v243 = v242 == false;
        if (v243){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v242);
        } else {
        }
        bool v245;
        v245 = 0l <= v239;
        bool v247;
        if (v245){
            bool v246;
            v246 = v239 < 32l;
            v247 = v246;
        } else {
            v247 = false;
        }
        bool v248;
        v248 = v247 == false;
        if (v248){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v247);
        } else {
        }
        int v250;
        v250 = v239 + v235;
        assert("Tensor range check" && 0 <= v239 && v239 < 32l);
        float * v251;
        v251 = v226[v250];
        float * v252;
        v252 = v228[v250];
        int v253;
        v253 = blockIdx.x;
        int v254;
        v254 = v253 * 32l;
        int v255;
        v255 = v254 + v250;
        assert("Tensor range check" && 0 <= v234 && v234 < 32l);
        int v256;
        v256 = 4l * v234;
        float v257[8l];
        int v258[8l];
        int v259;
        v259 = 0l;
        while (while_method_5(v259)){
            assert("Tensor range check" && 0 <= v259 && v259 < 2l);
            int v261;
            v261 = 4l * v259;
            assert("Tensor range check" && 0 <= v259 && v259 < 2l);
            int v262;
            v262 = 128l * v259;
            int v263;
            v263 = v262 + v256;
            int4* v264;
            v264 = reinterpret_cast<int4*>(v251 + v263);
            int4* v265;
            v265 = reinterpret_cast<int4*>(v257 + v261);
            assert("Pointer alignment check" && (unsigned long long)(v264) % 4l == 0 && (unsigned long long)(v265) % 4l == 0);
            *v265 = *v264;
            v259 += 1l ;
        }
        int v266;
        v266 = 0l;
        while (while_method_5(v266)){
            int v268;
            v268 = 0l;
            while (while_method_1(v268)){
                bool v270;
                v270 = 0l <= v268;
                bool v272;
                if (v270){
                    bool v271;
                    v271 = v268 < 4l;
                    v272 = v271;
                } else {
                    v272 = false;
                }
                bool v273;
                v273 = v272 == false;
                if (v273){
                    assert("The indices should be inside the range of the dimension." && v272);
                } else {
                }
                bool v275;
                v275 = 0l <= v234;
                bool v277;
                if (v275){
                    bool v276;
                    v276 = v234 < 32l;
                    v277 = v276;
                } else {
                    v277 = false;
                }
                bool v278;
                v278 = v277 == false;
                if (v278){
                    assert("The indices should be inside the range of the dimension." && v277);
                } else {
                }
                int v280;
                v280 = v234 * 4l;
                int v281;
                v281 = v268 + v280;
                bool v282;
                v282 = 0l <= v266;
                bool v284;
                if (v282){
                    bool v283;
                    v283 = v266 < 2l;
                    v284 = v283;
                } else {
                    v284 = false;
                }
                bool v285;
                v285 = v284 == false;
                if (v285){
                    assert("The indices should be inside the range of the dimension." && v284);
                } else {
                }
                int v287;
                v287 = v266 * 128l;
                int v288;
                v288 = v281 + v287;
                assert("Tensor range check" && 0 <= v266 && v266 < 2l);
                assert("Tensor range check" && 0 <= v268 && v268 < 4l);
                int v289;
                v289 = 4l * v266;
                int v290;
                v290 = v289 + v268;
                v258[v290] = v288;
                v268 += 1l ;
            }
            v266 += 1l ;
        }
        int v291;
        v291 = 0l;
        while (while_method_5(v291)){
            assert("Tensor range check" && 0 <= v291 && v291 < 2l);
            int v293;
            v293 = 128l * v291;
            int v294;
            v294 = v293 + v256;
            assert("Tensor range check" && 0 <= v291 && v291 < 2l);
            int v295;
            v295 = 4l * v291;
            int4* v296;
            v296 = reinterpret_cast<int4*>(v257 + v295);
            int4* v297;
            v297 = reinterpret_cast<int4*>(v252 + v294);
            assert("Pointer alignment check" && (unsigned long long)(v296) % 4l == 0 && (unsigned long long)(v297) % 4l == 0);
            *v297 = *v296;
            v291 += 1l ;
        }
        assert("Tensor range check" && 0 <= v250 && v250 < 32l);
        v239 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v230 && v230 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v298;
    v298 = v1+v12;
    float * v300;
    v300 = v7+v16;
    if (v220){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v219);
    } else {
    }
    extern __shared__ unsigned char v303[];
    if (v224){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v223);
    } else {
    }
    float * * v305;
    v305 = reinterpret_cast<float * *>(&v303[0ull]);
    float * * v307;
    v307 = reinterpret_cast<float * *>(&v303[v33]);
    int v309;
    v309 = threadIdx.x;
    assert("Tensor range check" && 0 <= v309 && v309 < 32l);
    v305[v309] = v298;
    v307[v309] = v300;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v310;
    v310 = 0l <= v309;
    bool v311;
    v311 = v310 == false;
    if (v311){
        assert("The index needs to be zero or positive." && v310);
    } else {
    }
    int v313;
    v313 = v309 % 32l;
    int v314;
    v314 = v309 / 32l;
    bool v315;
    v315 = v314 < 1l;
    bool v316;
    v316 = v315 == false;
    if (v316){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v315);
    } else {
    }
    assert("Tensor range check" && 0 <= v314 && v314 < 1l);
    int v318;
    v318 = 0l;
    while (while_method_4(v318)){
        bool v320;
        v320 = 0l <= v314;
        bool v321;
        v321 = v320 && v315;
        bool v322;
        v322 = v321 == false;
        if (v322){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v321);
        } else {
        }
        bool v324;
        v324 = 0l <= v318;
        bool v326;
        if (v324){
            bool v325;
            v325 = v318 < 32l;
            v326 = v325;
        } else {
            v326 = false;
        }
        bool v327;
        v327 = v326 == false;
        if (v327){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v326);
        } else {
        }
        int v329;
        v329 = v318 + v314;
        assert("Tensor range check" && 0 <= v318 && v318 < 32l);
        float * v330;
        v330 = v305[v329];
        float * v331;
        v331 = v307[v329];
        int v332;
        v332 = blockIdx.x;
        int v333;
        v333 = v332 * 32l;
        int v334;
        v334 = v333 + v329;
        assert("Tensor range check" && 0 <= v313 && v313 < 32l);
        int v335;
        v335 = 4l * v313;
        float v336[8l];
        int v337[8l];
        int v338;
        v338 = 0l;
        while (while_method_5(v338)){
            assert("Tensor range check" && 0 <= v338 && v338 < 2l);
            int v340;
            v340 = 4l * v338;
            assert("Tensor range check" && 0 <= v338 && v338 < 2l);
            int v341;
            v341 = 128l * v338;
            int v342;
            v342 = v341 + v335;
            int4* v343;
            v343 = reinterpret_cast<int4*>(v330 + v342);
            int4* v344;
            v344 = reinterpret_cast<int4*>(v336 + v340);
            assert("Pointer alignment check" && (unsigned long long)(v343) % 4l == 0 && (unsigned long long)(v344) % 4l == 0);
            *v344 = *v343;
            v338 += 1l ;
        }
        int v345;
        v345 = 0l;
        while (while_method_5(v345)){
            int v347;
            v347 = 0l;
            while (while_method_1(v347)){
                bool v349;
                v349 = 0l <= v347;
                bool v351;
                if (v349){
                    bool v350;
                    v350 = v347 < 4l;
                    v351 = v350;
                } else {
                    v351 = false;
                }
                bool v352;
                v352 = v351 == false;
                if (v352){
                    assert("The indices should be inside the range of the dimension." && v351);
                } else {
                }
                bool v354;
                v354 = 0l <= v313;
                bool v356;
                if (v354){
                    bool v355;
                    v355 = v313 < 32l;
                    v356 = v355;
                } else {
                    v356 = false;
                }
                bool v357;
                v357 = v356 == false;
                if (v357){
                    assert("The indices should be inside the range of the dimension." && v356);
                } else {
                }
                int v359;
                v359 = v313 * 4l;
                int v360;
                v360 = v347 + v359;
                bool v361;
                v361 = 0l <= v345;
                bool v363;
                if (v361){
                    bool v362;
                    v362 = v345 < 2l;
                    v363 = v362;
                } else {
                    v363 = false;
                }
                bool v364;
                v364 = v363 == false;
                if (v364){
                    assert("The indices should be inside the range of the dimension." && v363);
                } else {
                }
                int v366;
                v366 = v345 * 128l;
                int v367;
                v367 = v360 + v366;
                assert("Tensor range check" && 0 <= v345 && v345 < 2l);
                assert("Tensor range check" && 0 <= v347 && v347 < 4l);
                int v368;
                v368 = 4l * v345;
                int v369;
                v369 = v368 + v347;
                v337[v369] = v367;
                v347 += 1l ;
            }
            v345 += 1l ;
        }
        bool v370[8l];
        int v371;
        v371 = 0l;
        while (while_method_5(v371)){
            int v373;
            v373 = 0l;
            while (while_method_1(v373)){
                assert("Tensor range check" && 0 <= v371 && v371 < 2l);
                assert("Tensor range check" && 0 <= v373 && v373 < 4l);
                int v375;
                v375 = 4l * v371;
                int v376;
                v376 = v375 + v373;
                float v377;
                v377 = v336[v376];
                int v378;
                v378 = v337[v376];
                bool v379;
                v379 = v378 < 3l;
                assert("Tensor range check" && 0 <= v371 && v371 < 2l);
                assert("Tensor range check" && 0 <= v373 && v373 < 4l);
                v370[v376] = v379;
                v373 += 1l ;
            }
            v371 += 1l ;
        }
        float v380[8l];
        int v381;
        v381 = 0l;
        while (while_method_5(v381)){
            int v383;
            v383 = 0l;
            while (while_method_1(v383)){
                assert("Tensor range check" && 0 <= v381 && v381 < 2l);
                assert("Tensor range check" && 0 <= v383 && v383 < 4l);
                int v385;
                v385 = 4l * v381;
                int v386;
                v386 = v385 + v383;
                float v387;
                v387 = v336[v386];
                bool v388;
                v388 = v370[v386];
                float v391;
                if (v388){
                    bool v389;
                    v389 = 0.0f >= v387;
                    if (v389){
                        v391 = 0.0f;
                    } else {
                        v391 = v387;
                    }
                } else {
                    v391 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v381 && v381 < 2l);
                assert("Tensor range check" && 0 <= v383 && v383 < 4l);
                v380[v386] = v391;
                v383 += 1l ;
            }
            v381 += 1l ;
        }
        float v392;
        v392 = 0.0f;
        int v393;
        v393 = 0l;
        while (while_method_5(v393)){
            int v395;
            v395 = 0l;
            while (while_method_1(v395)){
                assert("Tensor range check" && 0 <= v393 && v393 < 2l);
                assert("Tensor range check" && 0 <= v395 && v395 < 4l);
                int v397;
                v397 = 4l * v393;
                int v398;
                v398 = v397 + v395;
                float v399;
                v399 = v380[v398];
                float v400;
                v400 = v392 + v399;
                v392 = v400;
                v395 += 1l ;
            }
            v393 += 1l ;
        }
        auto v401 = cooperative_groups::coalesced_threads();
        int v402;
        v402 = threadIdx.x;
        int v403;
        v403 = v402 / 32l;
        auto v404 = cooperative_groups::labeled_partition(v401,v403);
        Closure0 v405{};
        float v406;
        v406 = cooperative_groups::reduce(v404, v392, v405);
        int v407[8l];
        int v408;
        v408 = 0l;
        while (while_method_5(v408)){
            int v410;
            v410 = 0l;
            while (while_method_1(v410)){
                assert("Tensor range check" && 0 <= v408 && v408 < 2l);
                assert("Tensor range check" && 0 <= v410 && v410 < 4l);
                int v412;
                v412 = 4l * v408;
                int v413;
                v413 = v412 + v410;
                bool v414;
                v414 = v370[v413];
                int v415;
                if (v414){
                    v415 = 1l;
                } else {
                    v415 = 0l;
                }
                assert("Tensor range check" && 0 <= v408 && v408 < 2l);
                assert("Tensor range check" && 0 <= v410 && v410 < 4l);
                v407[v413] = v415;
                v410 += 1l ;
            }
            v408 += 1l ;
        }
        int v416;
        v416 = 0l;
        int v417;
        v417 = 0l;
        while (while_method_5(v417)){
            int v419;
            v419 = 0l;
            while (while_method_1(v419)){
                assert("Tensor range check" && 0 <= v417 && v417 < 2l);
                assert("Tensor range check" && 0 <= v419 && v419 < 4l);
                int v421;
                v421 = 4l * v417;
                int v422;
                v422 = v421 + v419;
                int v423;
                v423 = v407[v422];
                int v424;
                v424 = v416 + v423;
                v416 = v424;
                v419 += 1l ;
            }
            v417 += 1l ;
        }
        auto v425 = cooperative_groups::coalesced_threads();
        int v426;
        v426 = threadIdx.x;
        int v427;
        v427 = v426 / 32l;
        auto v428 = cooperative_groups::labeled_partition(v425,v427);
        Closure4 v429{};
        int v430;
        v430 = cooperative_groups::reduce(v428, v416, v429);
        float v431;
        v431 = (float)v430;
        float v432;
        v432 = 1.0f / v431;
        float v433[8l];
        int v434;
        v434 = 0l;
        while (while_method_5(v434)){
            int v436;
            v436 = 0l;
            while (while_method_1(v436)){
                assert("Tensor range check" && 0 <= v434 && v434 < 2l);
                assert("Tensor range check" && 0 <= v436 && v436 < 4l);
                int v438;
                v438 = 4l * v434;
                int v439;
                v439 = v438 + v436;
                float v440;
                v440 = v380[v439];
                bool v441;
                v441 = v370[v439];
                bool v442;
                v442 = v441 == false;
                float v447;
                if (v442){
                    v447 = 0.0f;
                } else {
                    bool v443;
                    v443 = v406 == 0.0f;
                    bool v444;
                    v444 = v443 != true;
                    if (v444){
                        float v445;
                        v445 = v440 / v406;
                        v447 = v445;
                    } else {
                        v447 = v432;
                    }
                }
                assert("Tensor range check" && 0 <= v434 && v434 < 2l);
                assert("Tensor range check" && 0 <= v436 && v436 < 4l);
                v433[v439] = v447;
                v436 += 1l ;
            }
            v434 += 1l ;
        }
        int v448;
        v448 = 0l;
        while (while_method_5(v448)){
            assert("Tensor range check" && 0 <= v448 && v448 < 2l);
            int v450;
            v450 = 128l * v448;
            int v451;
            v451 = v450 + v335;
            assert("Tensor range check" && 0 <= v448 && v448 < 2l);
            int v452;
            v452 = 4l * v448;
            int4* v453;
            v453 = reinterpret_cast<int4*>(v433 + v452);
            int4* v454;
            v454 = reinterpret_cast<int4*>(v331 + v451);
            assert("Pointer alignment check" && (unsigned long long)(v453) % 4l == 0 && (unsigned long long)(v454) % 4l == 0);
            *v454 = *v453;
            v448 += 1l ;
        }
        assert("Tensor range check" && 0 <= v329 && v329 < 32l);
        v318 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v309 && v309 < 32l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v455;
    v455 = v1+v12;
    if (v140){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v139);
    } else {
    }
    extern __shared__ unsigned char v458[];
    if (v144){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v143);
    } else {
    }
    float * * v460;
    v460 = reinterpret_cast<float * *>(&v458[0ull]);
    int * v462;
    v462 = reinterpret_cast<int *>(&v458[v33]);
    int v464;
    v464 = threadIdx.x;
    assert("Tensor range check" && 0 <= v464 && v464 < 32l);
    v460[v464] = v455;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v465;
    v465 = 0l <= v464;
    bool v466;
    v466 = v465 == false;
    if (v466){
        assert("The index needs to be zero or positive." && v465);
    } else {
    }
    int v468;
    v468 = v464 % 32l;
    int v469;
    v469 = v464 / 32l;
    bool v470;
    v470 = v469 < 1l;
    bool v471;
    v471 = v470 == false;
    if (v471){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v470);
    } else {
    }
    assert("Tensor range check" && 0 <= v469 && v469 < 1l);
    int v473;
    v473 = 0l;
    while (while_method_4(v473)){
        bool v475;
        v475 = 0l <= v469;
        bool v476;
        v476 = v475 && v470;
        bool v477;
        v477 = v476 == false;
        if (v477){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v476);
        } else {
        }
        bool v479;
        v479 = 0l <= v473;
        bool v481;
        if (v479){
            bool v480;
            v480 = v473 < 32l;
            v481 = v480;
        } else {
            v481 = false;
        }
        bool v482;
        v482 = v481 == false;
        if (v482){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v481);
        } else {
        }
        int v484;
        v484 = v473 + v469;
        assert("Tensor range check" && 0 <= v473 && v473 < 32l);
        float * v485;
        v485 = v460[v484];
        int v486;
        v486 = blockIdx.x;
        int v487;
        v487 = v486 * 32l;
        int v488;
        v488 = v487 + v484;
        assert("Tensor range check" && 0 <= v468 && v468 < 32l);
        int v489;
        v489 = 4l * v468;
        float v490[8l];
        int v491[8l];
        int v492;
        v492 = 0l;
        while (while_method_5(v492)){
            assert("Tensor range check" && 0 <= v492 && v492 < 2l);
            int v494;
            v494 = 4l * v492;
            assert("Tensor range check" && 0 <= v492 && v492 < 2l);
            int v495;
            v495 = 128l * v492;
            int v496;
            v496 = v495 + v489;
            int4* v497;
            v497 = reinterpret_cast<int4*>(v485 + v496);
            int4* v498;
            v498 = reinterpret_cast<int4*>(v490 + v494);
            assert("Pointer alignment check" && (unsigned long long)(v497) % 4l == 0 && (unsigned long long)(v498) % 4l == 0);
            *v498 = *v497;
            v492 += 1l ;
        }
        int v499;
        v499 = 0l;
        while (while_method_5(v499)){
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
                v508 = 0l <= v468;
                bool v510;
                if (v508){
                    bool v509;
                    v509 = v468 < 32l;
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
                v513 = v468 * 4l;
                int v514;
                v514 = v501 + v513;
                bool v515;
                v515 = 0l <= v499;
                bool v517;
                if (v515){
                    bool v516;
                    v516 = v499 < 2l;
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
                assert("Tensor range check" && 0 <= v499 && v499 < 2l);
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
        bool v524[8l];
        int v525;
        v525 = 0l;
        while (while_method_5(v525)){
            int v527;
            v527 = 0l;
            while (while_method_1(v527)){
                assert("Tensor range check" && 0 <= v525 && v525 < 2l);
                assert("Tensor range check" && 0 <= v527 && v527 < 4l);
                int v529;
                v529 = 4l * v525;
                int v530;
                v530 = v529 + v527;
                float v531;
                v531 = v490[v530];
                int v532;
                v532 = v491[v530];
                bool v533;
                v533 = v532 < 3l;
                assert("Tensor range check" && 0 <= v525 && v525 < 2l);
                assert("Tensor range check" && 0 <= v527 && v527 < 4l);
                v524[v530] = v533;
                v527 += 1l ;
            }
            v525 += 1l ;
        }
        int v534[8l];
        int v535;
        v535 = 0l;
        while (while_method_5(v535)){
            int v537;
            v537 = 0l;
            while (while_method_1(v537)){
                assert("Tensor range check" && 0 <= v535 && v535 < 2l);
                assert("Tensor range check" && 0 <= v537 && v537 < 4l);
                int v539;
                v539 = 4l * v535;
                int v540;
                v540 = v539 + v537;
                bool v541;
                v541 = v524[v540];
                int v542;
                if (v541){
                    v542 = 1l;
                } else {
                    v542 = 0l;
                }
                assert("Tensor range check" && 0 <= v535 && v535 < 2l);
                assert("Tensor range check" && 0 <= v537 && v537 < 4l);
                v534[v540] = v542;
                v537 += 1l ;
            }
            v535 += 1l ;
        }
        int v543;
        v543 = 0l;
        int v544;
        v544 = 0l;
        while (while_method_5(v544)){
            int v546;
            v546 = 0l;
            while (while_method_1(v546)){
                assert("Tensor range check" && 0 <= v544 && v544 < 2l);
                assert("Tensor range check" && 0 <= v546 && v546 < 4l);
                int v548;
                v548 = 4l * v544;
                int v549;
                v549 = v548 + v546;
                int v550;
                v550 = v534[v549];
                int v551;
                v551 = v543 + v550;
                v543 = v551;
                v546 += 1l ;
            }
            v544 += 1l ;
        }
        auto v552 = cooperative_groups::coalesced_threads();
        int v553;
        v553 = threadIdx.x;
        int v554;
        v554 = v553 / 32l;
        auto v555 = cooperative_groups::labeled_partition(v552,v554);
        Closure4 v556{};
        int v557;
        v557 = cooperative_groups::reduce(v555, v543, v556);
        float v558[8l];
        int v559;
        v559 = 0l;
        while (while_method_5(v559)){
            int v561;
            v561 = 0l;
            while (while_method_1(v561)){
                assert("Tensor range check" && 0 <= v559 && v559 < 2l);
                assert("Tensor range check" && 0 <= v561 && v561 < 4l);
                int v563;
                v563 = 4l * v559;
                int v564;
                v564 = v563 + v561;
                float v565;
                v565 = v490[v564];
                bool v566;
                v566 = v524[v564];
                float v567;
                if (v566){
                    v567 = v565;
                } else {
                    v567 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v559 && v559 < 2l);
                assert("Tensor range check" && 0 <= v561 && v561 < 4l);
                v558[v564] = v567;
                v561 += 1l ;
            }
            v559 += 1l ;
        }
        float v568;
        v568 = 0.0f;
        int v569;
        v569 = 0l;
        while (while_method_5(v569)){
            int v571;
            v571 = 0l;
            while (while_method_1(v571)){
                assert("Tensor range check" && 0 <= v569 && v569 < 2l);
                assert("Tensor range check" && 0 <= v571 && v571 < 4l);
                int v573;
                v573 = 4l * v569;
                int v574;
                v574 = v573 + v571;
                float v575;
                v575 = v558[v574];
                float v576;
                v576 = v568 + v575;
                v568 = v576;
                v571 += 1l ;
            }
            v569 += 1l ;
        }
        auto v577 = cooperative_groups::coalesced_threads();
        int v578;
        v578 = threadIdx.x;
        int v579;
        v579 = v578 / 32l;
        auto v580 = cooperative_groups::labeled_partition(v577,v579);
        Closure0 v581{};
        float v582;
        v582 = cooperative_groups::reduce(v580, v568, v581);
        float v583;
        v583 = (float)v557;
        float v584;
        v584 = v582 / v583;
        float v585[8l];
        int v586;
        v586 = 0l;
        while (while_method_5(v586)){
            int v588;
            v588 = 0l;
            while (while_method_1(v588)){
                assert("Tensor range check" && 0 <= v586 && v586 < 2l);
                assert("Tensor range check" && 0 <= v588 && v588 < 4l);
                int v590;
                v590 = 4l * v586;
                int v591;
                v591 = v590 + v588;
                float v592;
                v592 = v490[v591];
                bool v593;
                v593 = v524[v591];
                float v594;
                if (v593){
                    v594 = v592;
                } else {
                    v594 = -1.0f / 0.0f;
                }
                float v595;
                v595 = v594 - v584;
                float v596;
                v596 = exp(v595);
                assert("Tensor range check" && 0 <= v586 && v586 < 2l);
                assert("Tensor range check" && 0 <= v588 && v588 < 4l);
                v585[v591] = v596;
                v588 += 1l ;
            }
            v586 += 1l ;
        }
        float v597;
        v597 = 0.0f;
        int v598;
        v598 = 0l;
        while (while_method_5(v598)){
            int v600;
            v600 = 0l;
            while (while_method_1(v600)){
                assert("Tensor range check" && 0 <= v598 && v598 < 2l);
                assert("Tensor range check" && 0 <= v600 && v600 < 4l);
                int v602;
                v602 = 4l * v598;
                int v603;
                v603 = v602 + v600;
                float v604;
                v604 = v585[v603];
                float v605;
                v605 = v597 + v604;
                v597 = v605;
                v600 += 1l ;
            }
            v598 += 1l ;
        }
        auto v606 = cooperative_groups::coalesced_threads();
        int v607;
        v607 = threadIdx.x;
        int v608;
        v608 = v607 / 32l;
        auto v609 = cooperative_groups::labeled_partition(v606,v608);
        float v610;
        v610 = cooperative_groups::reduce(v609, v597, v581);
        float v611[8l];
        int v612;
        v612 = 0l;
        while (while_method_5(v612)){
            int v614;
            v614 = 0l;
            while (while_method_1(v614)){
                assert("Tensor range check" && 0 <= v612 && v612 < 2l);
                assert("Tensor range check" && 0 <= v614 && v614 < 4l);
                int v616;
                v616 = 4l * v612;
                int v617;
                v617 = v616 + v614;
                float v618;
                v618 = v585[v617];
                float v619;
                v619 = v618 / v610;
                assert("Tensor range check" && 0 <= v612 && v612 < 2l);
                assert("Tensor range check" && 0 <= v614 && v614 < 4l);
                v611[v617] = v619;
                v614 += 1l ;
            }
            v612 += 1l ;
        }
        float v620[8l];
        float v621;
        v621 = 0.0f;
        int v622;
        v622 = 0l;
        while (while_method_5(v622)){
            assert("Tensor range check" && 0 <= v622 && v622 < 2l);
            int v624;
            v624 = 4l * v622;
            assert("Tensor range check" && 0 <= v622 && v622 < 2l);
            int v625; float v626;
            Tuple0 tmp54 = Tuple0{0l, 0.0f};
            v625 = tmp54.v0; v626 = tmp54.v1;
            while (while_method_1(v625)){
                assert("Tensor range check" && 0 <= v625 && v625 < 4l);
                int v628;
                v628 = v625 + v624;
                float v629;
                v629 = v611[v628];
                float v630;
                v630 = v626 + v629;
                v626 = v630;
                v625 += 1l ;
            }
            auto v631 = cooperative_groups::coalesced_threads();
            int v632;
            v632 = threadIdx.x;
            int v633;
            v633 = v632 / 32l;
            auto v634 = cooperative_groups::labeled_partition(v631,v633);
            Closure2 v635{};
            float v636;
            v636 = cooperative_groups::inclusive_scan(v634, v626, v635);
            float v637;
            v637 = v634.shfl_up(v636,1);
            bool v638;
            v638 = v634.thread_rank() == 0;
            float v639;
            if (v638){
                v639 = 0.0f;
            } else {
                v639 = v637;
            }
            float v640;
            v640 = v634.shfl(v636,v634.num_threads()-1);
            float v641;
            v641 = v621 + v639;
            int v642; float v643;
            Tuple0 tmp55 = Tuple0{0l, v641};
            v642 = tmp55.v0; v643 = tmp55.v1;
            while (while_method_1(v642)){
                assert("Tensor range check" && 0 <= v642 && v642 < 4l);
                int v645;
                v645 = v642 + v624;
                float v646;
                v646 = v611[v645];
                float v647;
                v647 = v643 + v646;
                assert("Tensor range check" && 0 <= v642 && v642 < 4l);
                v620[v645] = v647;
                v643 = v647;
                v642 += 1l ;
            }
            float v648;
            v648 = v621 + v640;
            v621 = v648;
            v622 += 1l ;
        }
        float v649[8l];
        bool v650[8l];
        int v651;
        v651 = 0l;
        while (while_method_5(v651)){
            int v653;
            v653 = 0l;
            while (while_method_1(v653)){
                assert("Tensor range check" && 0 <= v651 && v651 < 2l);
                assert("Tensor range check" && 0 <= v653 && v653 < 4l);
                int v655;
                v655 = 4l * v651;
                int v656;
                v656 = v655 + v653;
                float v657;
                v657 = v620[v656];
                float v658;
                v658 = v611[v656];
                bool v659;
                v659 = v658 > 0.0f;
                assert("Tensor range check" && 0 <= v651 && v651 < 2l);
                assert("Tensor range check" && 0 <= v653 && v653 < 4l);
                v649[v656] = v657;
                v650[v656] = v659;
                v653 += 1l ;
            }
            v651 += 1l ;
        }
        float v660; bool v661;
        Tuple3 tmp56 = Tuple3{-1.0f / 0.0f, false};
        v660 = tmp56.v0; v661 = tmp56.v1;
        int v662;
        v662 = 0l;
        while (while_method_5(v662)){
            int v664;
            v664 = 0l;
            while (while_method_1(v664)){
                assert("Tensor range check" && 0 <= v662 && v662 < 2l);
                assert("Tensor range check" && 0 <= v664 && v664 < 4l);
                int v666;
                v666 = 4l * v662;
                int v667;
                v667 = v666 + v664;
                float v668;
                v668 = v649[v667];
                bool v669;
                v669 = v650[v667];
                float v676; bool v677;
                if (v661){
                    if (v669){
                        bool v670;
                        v670 = v660 >= v668;
                        float v671;
                        if (v670){
                            v671 = v660;
                        } else {
                            v671 = v668;
                        }
                        v676 = v671; v677 = true;
                    } else {
                        v676 = v660; v677 = v661;
                    }
                } else {
                    if (v669){
                        v676 = v668; v677 = v669;
                    } else {
                        v676 = v660; v677 = v661;
                    }
                }
                v660 = v676;
                v661 = v677;
                v664 += 1l ;
            }
            v662 += 1l ;
        }
        auto v678 = cooperative_groups::coalesced_threads();
        int v679;
        v679 = threadIdx.x;
        int v680;
        v680 = v679 / 32l;
        auto v681 = cooperative_groups::labeled_partition(v678,v680);
        Closure5 v682{};
        float v683; bool v684;
        Tuple3 tmp57 = cooperative_groups::reduce(v681, Tuple3{v660, v661}, v682);
        v683 = tmp57.v0; v684 = tmp57.v1;
        bool v685;
        v685 = v684 == false;
        if (v685){
            assert("The local reduce must be true." && v684);
        } else {
        }
        float v687[8l];
        int v688[8l];
        int v689;
        v689 = 0l;
        while (while_method_5(v689)){
            int v691;
            v691 = 0l;
            while (while_method_1(v691)){
                assert("Tensor range check" && 0 <= v689 && v689 < 2l);
                assert("Tensor range check" && 0 <= v691 && v691 < 4l);
                int v693;
                v693 = 4l * v689;
                int v694;
                v694 = v693 + v691;
                int v695;
                v695 = v491[v694];
                float v696;
                v696 = curand_uniform(&v10);
                assert("Tensor range check" && 0 <= v689 && v689 < 2l);
                assert("Tensor range check" && 0 <= v691 && v691 < 4l);
                v687[v694] = v696;
                v688[v694] = v695;
                v691 += 1l ;
            }
            v689 += 1l ;
        }
        float v697; int v698;
        Tuple1 tmp58 = Tuple1{0.0f, 2147483647l};
        v697 = tmp58.v0; v698 = tmp58.v1;
        int v699;
        v699 = 0l;
        while (while_method_5(v699)){
            int v701;
            v701 = 0l;
            while (while_method_1(v701)){
                assert("Tensor range check" && 0 <= v699 && v699 < 2l);
                assert("Tensor range check" && 0 <= v701 && v701 < 4l);
                int v703;
                v703 = 4l * v699;
                int v704;
                v704 = v703 + v701;
                float v705;
                v705 = v687[v704];
                int v706;
                v706 = v688[v704];
                bool v707;
                v707 = v698 < v706;
                float v708; int v709;
                if (v707){
                    v708 = v697; v709 = v698;
                } else {
                    v708 = v705; v709 = v706;
                }
                v697 = v708;
                v698 = v709;
                v701 += 1l ;
            }
            v699 += 1l ;
        }
        auto v710 = cooperative_groups::coalesced_threads();
        int v711;
        v711 = threadIdx.x;
        int v712;
        v712 = v711 / 32l;
        auto v713 = cooperative_groups::labeled_partition(v710,v712);
        Closure6 v714{};
        float v715; int v716;
        Tuple1 tmp59 = cooperative_groups::reduce(v713, Tuple1{v697, v698}, v714);
        v715 = tmp59.v0; v716 = tmp59.v1;
        float v717;
        v717 = v683 * v715;
        int v718[8l];
        bool v719[8l];
        int v720;
        v720 = 0l;
        while (while_method_5(v720)){
            int v722;
            v722 = 0l;
            while (while_method_1(v722)){
                assert("Tensor range check" && 0 <= v720 && v720 < 2l);
                assert("Tensor range check" && 0 <= v722 && v722 < 4l);
                int v724;
                v724 = 4l * v720;
                int v725;
                v725 = v724 + v722;
                float v726;
                v726 = v649[v725];
                bool v727;
                v727 = v650[v725];
                int v728;
                v728 = v491[v725];
                int v731; bool v732;
                if (v727){
                    float v729;
                    v729 = v726 - v717;
                    bool v730;
                    v730 = v729 >= 0.0f;
                    v731 = v728; v732 = v730;
                } else {
                    v731 = 2147483647l; v732 = false;
                }
                assert("Tensor range check" && 0 <= v720 && v720 < 2l);
                assert("Tensor range check" && 0 <= v722 && v722 < 4l);
                v718[v725] = v731;
                v719[v725] = v732;
                v722 += 1l ;
            }
            v720 += 1l ;
        }
        int v733; bool v734;
        Tuple4 tmp60 = Tuple4{2147483647l, false};
        v733 = tmp60.v0; v734 = tmp60.v1;
        int v735;
        v735 = 0l;
        while (while_method_5(v735)){
            int v737;
            v737 = 0l;
            while (while_method_1(v737)){
                assert("Tensor range check" && 0 <= v735 && v735 < 2l);
                assert("Tensor range check" && 0 <= v737 && v737 < 4l);
                int v739;
                v739 = 4l * v735;
                int v740;
                v740 = v739 + v737;
                int v741;
                v741 = v718[v740];
                bool v742;
                v742 = v719[v740];
                int v749; bool v750;
                if (v734){
                    if (v742){
                        bool v743;
                        v743 = v733 < v741;
                        int v744;
                        if (v743){
                            v744 = v733;
                        } else {
                            v744 = v741;
                        }
                        v749 = v744; v750 = true;
                    } else {
                        v749 = v733; v750 = v734;
                    }
                } else {
                    if (v742){
                        v749 = v741; v750 = v742;
                    } else {
                        v749 = v733; v750 = v734;
                    }
                }
                v733 = v749;
                v734 = v750;
                v737 += 1l ;
            }
            v735 += 1l ;
        }
        auto v751 = cooperative_groups::coalesced_threads();
        int v752;
        v752 = threadIdx.x;
        int v753;
        v753 = v752 / 32l;
        auto v754 = cooperative_groups::labeled_partition(v751,v753);
        Closure7 v755{};
        int v756; bool v757;
        Tuple4 tmp61 = cooperative_groups::reduce(v754, Tuple4{v733, v734}, v755);
        v756 = tmp61.v0; v757 = tmp61.v1;
        bool v758;
        v758 = v757 == false;
        if (v758){
            assert("The local reduce must be true." && v757);
        } else {
        }
        int v760;
        v760 = 0l;
        while (while_method_5(v760)){
            assert("Tensor range check" && 0 <= v760 && v760 < 2l);
            assert("Tensor range check" && 0 <= v760 && v760 < 2l);
            v760 += 1l ;
        }
        assert("Tensor range check" && 0 <= v484 && v484 < 32l);
        v462[v484] = v756;
        v473 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v464 && v464 < 32l);
    int v762;
    v762 = v462[v464];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v763;
    v763 = threadIdx.x;
    assert("Tensor range check" && 0 <= v763 && v763 < 32l);
    v5[v763] = v762;
    return ;
}
extern "C" __global__ void entry4(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14) {
    auto v15 = cooperative_groups::this_grid();
    int v16;
    v16 = threadIdx.x;
    bool v17;
    v17 = 0l <= v16;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The index needs to be zero or positive." && v17);
    } else {
    }
    int v20;
    v20 = v16 % 16l;
    int v21;
    v21 = v16 / 16l;
    bool v22;
    v22 = v21 < 2l;
    bool v23;
    v23 = v22 == false;
    if (v23){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v22);
    } else {
    }
    assert("Tensor range check" && 0 <= v21 && v21 < 2l);
    assert("Tensor range check" && 0 <= v20 && v20 < 16l);
    int v25;
    v25 = 4l * v20;
    int v26;
    v26 = 64l * v21;
    int v27;
    v27 = v26 + v25;
    assert("Tensor range check" && 0 <= v21 && v21 < 2l);
    assert("Tensor range check" && 0 <= v20 && v20 < 16l);
    int v28;
    v28 = blockIdx.x;
    int v29;
    v29 = v28;
    while (while_method_2(v29)){
        bool v31;
        v31 = 0l <= v29;
        bool v32;
        v32 = v31 == false;
        if (v32){
            assert("The index needs to be zero or positive." && v31);
        } else {
        }
        bool v34;
        v34 = v29 < 64l;
        bool v35;
        v35 = v34 == false;
        if (v35){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v34);
        } else {
        }
        assert("Tensor range check" && 0 <= v29 && v29 < 64l);
        int v37;
        v37 = 128l * v29;
        int v38;
        v38 = v37 + v27;
        int v39[4l];
        int v40[4l];
        int v41;
        v41 = 0l;
        while (while_method_3(v41)){
            assert("Tensor range check" && 0 <= v41 && v41 < 1l);
            int v43;
            v43 = 4l * v41;
            assert("Tensor range check" && 0 <= v41 && v41 < 1l);
            int v44;
            v44 = 64l * v41;
            int v45;
            v45 = v44 + v38;
            int4* v46;
            v46 = reinterpret_cast<int4*>(v0 + v45);
            int4* v47;
            v47 = reinterpret_cast<int4*>(v39 + v43);
            assert("Pointer alignment check" && (unsigned long long)(v46) % 4l == 0 && (unsigned long long)(v47) % 4l == 0);
            *v47 = *v46;
            v41 += 1l ;
        }
        int v48;
        v48 = 0l;
        while (while_method_3(v48)){
            int v50;
            v50 = 0l;
            while (while_method_1(v50)){
                bool v52;
                v52 = 0l <= v50;
                bool v54;
                if (v52){
                    bool v53;
                    v53 = v50 < 4l;
                    v54 = v53;
                } else {
                    v54 = false;
                }
                bool v55;
                v55 = v54 == false;
                if (v55){
                    assert("The indices should be inside the range of the dimension." && v54);
                } else {
                }
                bool v57;
                v57 = 0l <= v20;
                bool v59;
                if (v57){
                    bool v58;
                    v58 = v20 < 16l;
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
                int v62;
                v62 = v20 * 4l;
                int v63;
                v63 = v50 + v62;
                bool v64;
                v64 = 0l <= v48;
                bool v66;
                if (v64){
                    bool v65;
                    v65 = v48 < 1l;
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
                int v69;
                v69 = v48 * 64l;
                int v70;
                v70 = v63 + v69;
                assert("Tensor range check" && 0 <= v48 && v48 < 1l);
                assert("Tensor range check" && 0 <= v50 && v50 < 4l);
                int v71;
                v71 = 4l * v48;
                int v72;
                v72 = v71 + v50;
                v40[v72] = v70;
                v50 += 1l ;
            }
            v48 += 1l ;
        }
        bool v73;
        v73 = 0l <= v21;
        bool v74;
        v74 = v73 && v22;
        bool v75;
        v75 = v74 == false;
        if (v75){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v74);
        } else {
        }
        bool v77;
        v77 = v31 && v34;
        bool v78;
        v78 = v77 == false;
        if (v78){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v77);
        } else {
        }
        int v80;
        v80 = v29 * 2l;
        int v81;
        v81 = v80 + v21;
        assert("Tensor range check" && 0 <= v29 && v29 < 64l);
        int v82;
        v82 = 0l;
        while (while_method_3(v82)){
            assert("Tensor range check" && 0 <= v82 && v82 < 1l);
            int v84;
            v84 = 64l * v82;
            int v85;
            v85 = v84 + v38;
            assert("Tensor range check" && 0 <= v82 && v82 < 1l);
            int v86;
            v86 = 4l * v82;
            int4* v87;
            v87 = reinterpret_cast<int4*>(v39 + v86);
            int4* v88;
            v88 = reinterpret_cast<int4*>(v2 + v85);
            assert("Pointer alignment check" && (unsigned long long)(v87) % 4l == 0 && (unsigned long long)(v88) % 4l == 0);
            *v88 = *v87;
            v82 += 1l ;
        }
        v29 += 1l ;
    }
    v15.sync() ;
    int v89;
    v89 = threadIdx.x;
    bool v90;
    v90 = 0l <= v89;
    bool v91;
    v91 = v90 == false;
    if (v91){
        assert("The index needs to be zero or positive." && v90);
    } else {
    }
    int v93;
    v93 = v89 % 16l;
    int v94;
    v94 = v89 / 16l;
    bool v95;
    v95 = v94 < 2l;
    bool v96;
    v96 = v95 == false;
    if (v96){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v95);
    } else {
    }
    assert("Tensor range check" && 0 <= v94 && v94 < 2l);
    assert("Tensor range check" && 0 <= v93 && v93 < 16l);
    int v98;
    v98 = 4l * v93;
    int v99;
    v99 = 64l * v94;
    int v100;
    v100 = v99 + v98;
    assert("Tensor range check" && 0 <= v94 && v94 < 2l);
    assert("Tensor range check" && 0 <= v93 && v93 < 16l);
    int v101;
    v101 = blockIdx.x;
    int v102;
    v102 = v101;
    while (while_method_2(v102)){
        bool v104;
        v104 = 0l <= v102;
        bool v105;
        v105 = v104 == false;
        if (v105){
            assert("The index needs to be zero or positive." && v104);
        } else {
        }
        bool v107;
        v107 = v102 < 64l;
        bool v108;
        v108 = v107 == false;
        if (v108){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v107);
        } else {
        }
        assert("Tensor range check" && 0 <= v102 && v102 < 64l);
        int v110;
        v110 = 128l * v102;
        int v111;
        v111 = v110 + v100;
        float v112[4l];
        int v113[4l];
        int v114;
        v114 = 0l;
        while (while_method_3(v114)){
            assert("Tensor range check" && 0 <= v114 && v114 < 1l);
            int v116;
            v116 = 4l * v114;
            assert("Tensor range check" && 0 <= v114 && v114 < 1l);
            int v117;
            v117 = 64l * v114;
            int v118;
            v118 = v117 + v111;
            int4* v119;
            v119 = reinterpret_cast<int4*>(v1 + v118);
            int4* v120;
            v120 = reinterpret_cast<int4*>(v112 + v116);
            assert("Pointer alignment check" && (unsigned long long)(v119) % 4l == 0 && (unsigned long long)(v120) % 4l == 0);
            *v120 = *v119;
            v114 += 1l ;
        }
        int v121;
        v121 = 0l;
        while (while_method_3(v121)){
            int v123;
            v123 = 0l;
            while (while_method_1(v123)){
                bool v125;
                v125 = 0l <= v123;
                bool v127;
                if (v125){
                    bool v126;
                    v126 = v123 < 4l;
                    v127 = v126;
                } else {
                    v127 = false;
                }
                bool v128;
                v128 = v127 == false;
                if (v128){
                    assert("The indices should be inside the range of the dimension." && v127);
                } else {
                }
                bool v130;
                v130 = 0l <= v93;
                bool v132;
                if (v130){
                    bool v131;
                    v131 = v93 < 16l;
                    v132 = v131;
                } else {
                    v132 = false;
                }
                bool v133;
                v133 = v132 == false;
                if (v133){
                    assert("The indices should be inside the range of the dimension." && v132);
                } else {
                }
                int v135;
                v135 = v93 * 4l;
                int v136;
                v136 = v123 + v135;
                bool v137;
                v137 = 0l <= v121;
                bool v139;
                if (v137){
                    bool v138;
                    v138 = v121 < 1l;
                    v139 = v138;
                } else {
                    v139 = false;
                }
                bool v140;
                v140 = v139 == false;
                if (v140){
                    assert("The indices should be inside the range of the dimension." && v139);
                } else {
                }
                int v142;
                v142 = v121 * 64l;
                int v143;
                v143 = v136 + v142;
                assert("Tensor range check" && 0 <= v121 && v121 < 1l);
                assert("Tensor range check" && 0 <= v123 && v123 < 4l);
                int v144;
                v144 = 4l * v121;
                int v145;
                v145 = v144 + v123;
                v113[v145] = v143;
                v123 += 1l ;
            }
            v121 += 1l ;
        }
        bool v146;
        v146 = 0l <= v94;
        bool v147;
        v147 = v146 && v95;
        bool v148;
        v148 = v147 == false;
        if (v148){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v147);
        } else {
        }
        bool v150;
        v150 = v104 && v107;
        bool v151;
        v151 = v150 == false;
        if (v151){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v150);
        } else {
        }
        int v153;
        v153 = v102 * 2l;
        int v154;
        v154 = v153 + v94;
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
                v163 = v113[v162];
                assert("Tensor range check" && 0 <= v157 && v157 < 1l);
                assert("Tensor range check" && 0 <= v159 && v159 < 4l);
                v155[v162] = v154;
                v156[v162] = v163;
                v159 += 1l ;
            }
            v157 += 1l ;
        }
        assert("Tensor range check" && 0 <= v102 && v102 < 64l);
        int v164;
        v164 = 0l;
        while (while_method_3(v164)){
            assert("Tensor range check" && 0 <= v164 && v164 < 1l);
            int v166;
            v166 = 64l * v164;
            int v167;
            v167 = v166 + v111;
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
        v102 += 1l ;
    }
    v15.sync() ;
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
    v177 = v173 % 16l;
    int v178;
    v178 = v173 / 16l;
    bool v179;
    v179 = v178 < 2l;
    bool v180;
    v180 = v179 == false;
    if (v180){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v179);
    } else {
    }
    assert("Tensor range check" && 0 <= v178 && v178 < 2l);
    assert("Tensor range check" && 0 <= v177 && v177 < 16l);
    int v182;
    v182 = 4l * v177;
    int v183;
    v183 = 64l * v178;
    int v184;
    v184 = v183 + v182;
    assert("Tensor range check" && 0 <= v178 && v178 < 2l);
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
            v201 = 64l * v198;
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
                    v215 = v177 < 16l;
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
                v226 = v205 * 64l;
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
        v237 = v186 * 2l;
        int v238;
        v238 = v237 + v178;
        assert("Tensor range check" && 0 <= v186 && v186 < 64l);
        int v239;
        v239 = 2l * v186;
        int v240;
        v240 = v239 + v178;
        v11[v240] = v238;
        v186 += 1l ;
    }
    v15.sync() ;
    int v241;
    v241 = threadIdx.x;
    bool v242;
    v242 = 0l <= v241;
    bool v243;
    v243 = v242 == false;
    if (v243){
        assert("The index needs to be zero or positive." && v242);
    } else {
    }
    int v245;
    v245 = v241 % 16l;
    int v246;
    v246 = v241 / 16l;
    bool v247;
    v247 = v246 < 2l;
    bool v248;
    v248 = v247 == false;
    if (v248){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v247);
    } else {
    }
    assert("Tensor range check" && 0 <= v246 && v246 < 2l);
    assert("Tensor range check" && 0 <= v245 && v245 < 16l);
    int v250;
    v250 = 4l * v245;
    int v251;
    v251 = 64l * v246;
    int v252;
    v252 = v251 + v250;
    assert("Tensor range check" && 0 <= v246 && v246 < 2l);
    assert("Tensor range check" && 0 <= v245 && v245 < 16l);
    int v253;
    v253 = blockIdx.x;
    int v254;
    v254 = v253;
    while (while_method_2(v254)){
        bool v256;
        v256 = 0l <= v254;
        bool v257;
        v257 = v256 == false;
        if (v257){
            assert("The index needs to be zero or positive." && v256);
        } else {
        }
        bool v259;
        v259 = v254 < 64l;
        bool v260;
        v260 = v259 == false;
        if (v260){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v259);
        } else {
        }
        assert("Tensor range check" && 0 <= v254 && v254 < 64l);
        int v262;
        v262 = 128l * v254;
        int v263;
        v263 = v262 + v252;
        float v264[4l];
        int v265[4l];
        int v266;
        v266 = 0l;
        while (while_method_3(v266)){
            assert("Tensor range check" && 0 <= v266 && v266 < 1l);
            int v268;
            v268 = 4l * v266;
            assert("Tensor range check" && 0 <= v266 && v266 < 1l);
            int v269;
            v269 = 64l * v266;
            int v270;
            v270 = v269 + v263;
            int4* v271;
            v271 = reinterpret_cast<int4*>(v1 + v270);
            int4* v272;
            v272 = reinterpret_cast<int4*>(v264 + v268);
            assert("Pointer alignment check" && (unsigned long long)(v271) % 4l == 0 && (unsigned long long)(v272) % 4l == 0);
            *v272 = *v271;
            v266 += 1l ;
        }
        int v273;
        v273 = 0l;
        while (while_method_3(v273)){
            int v275;
            v275 = 0l;
            while (while_method_1(v275)){
                bool v277;
                v277 = 0l <= v275;
                bool v279;
                if (v277){
                    bool v278;
                    v278 = v275 < 4l;
                    v279 = v278;
                } else {
                    v279 = false;
                }
                bool v280;
                v280 = v279 == false;
                if (v280){
                    assert("The indices should be inside the range of the dimension." && v279);
                } else {
                }
                bool v282;
                v282 = 0l <= v245;
                bool v284;
                if (v282){
                    bool v283;
                    v283 = v245 < 16l;
                    v284 = v283;
                } else {
                    v284 = false;
                }
                bool v285;
                v285 = v284 == false;
                if (v285){
                    assert("The indices should be inside the range of the dimension." && v284);
                } else {
                }
                int v287;
                v287 = v245 * 4l;
                int v288;
                v288 = v275 + v287;
                bool v289;
                v289 = 0l <= v273;
                bool v291;
                if (v289){
                    bool v290;
                    v290 = v273 < 1l;
                    v291 = v290;
                } else {
                    v291 = false;
                }
                bool v292;
                v292 = v291 == false;
                if (v292){
                    assert("The indices should be inside the range of the dimension." && v291);
                } else {
                }
                int v294;
                v294 = v273 * 64l;
                int v295;
                v295 = v288 + v294;
                assert("Tensor range check" && 0 <= v273 && v273 < 1l);
                assert("Tensor range check" && 0 <= v275 && v275 < 4l);
                int v296;
                v296 = 4l * v273;
                int v297;
                v297 = v296 + v275;
                v265[v297] = v295;
                v275 += 1l ;
            }
            v273 += 1l ;
        }
        bool v298;
        v298 = 0l <= v246;
        bool v299;
        v299 = v298 && v247;
        bool v300;
        v300 = v299 == false;
        if (v300){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v299);
        } else {
        }
        bool v302;
        v302 = v256 && v259;
        bool v303;
        v303 = v302 == false;
        if (v303){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v302);
        } else {
        }
        int v305;
        v305 = v254 * 2l;
        int v306;
        v306 = v305 + v246;
        float v307;
        v307 = 0.0f;
        int v308;
        v308 = 0l;
        while (while_method_3(v308)){
            int v310;
            v310 = 0l;
            while (while_method_1(v310)){
                assert("Tensor range check" && 0 <= v308 && v308 < 1l);
                assert("Tensor range check" && 0 <= v310 && v310 < 4l);
                int v312;
                v312 = 4l * v308;
                int v313;
                v313 = v312 + v310;
                float v314;
                v314 = v264[v313];
                float v315;
                v315 = v307 + v314;
                v307 = v315;
                v310 += 1l ;
            }
            v308 += 1l ;
        }
        auto v316 = cooperative_groups::coalesced_threads();
        int v317;
        v317 = threadIdx.x;
        int v318;
        v318 = v317 / 16l;
        auto v319 = cooperative_groups::labeled_partition(v316,v318);
        Closure0 v320{};
        float v321;
        v321 = cooperative_groups::reduce(v319, v307, v320);
        float v322;
        v322 = v321 / 64.0f;
        float v323[4l];
        int v324;
        v324 = 0l;
        while (while_method_3(v324)){
            int v326;
            v326 = 0l;
            while (while_method_1(v326)){
                assert("Tensor range check" && 0 <= v324 && v324 < 1l);
                assert("Tensor range check" && 0 <= v326 && v326 < 4l);
                int v328;
                v328 = 4l * v324;
                int v329;
                v329 = v328 + v326;
                float v330;
                v330 = v264[v329];
                float v331;
                v331 = v330 - v322;
                float v332;
                v332 = exp(v331);
                assert("Tensor range check" && 0 <= v324 && v324 < 1l);
                assert("Tensor range check" && 0 <= v326 && v326 < 4l);
                v323[v329] = v332;
                v326 += 1l ;
            }
            v324 += 1l ;
        }
        float v333;
        v333 = 0.0f;
        int v334;
        v334 = 0l;
        while (while_method_3(v334)){
            int v336;
            v336 = 0l;
            while (while_method_1(v336)){
                assert("Tensor range check" && 0 <= v334 && v334 < 1l);
                assert("Tensor range check" && 0 <= v336 && v336 < 4l);
                int v338;
                v338 = 4l * v334;
                int v339;
                v339 = v338 + v336;
                float v340;
                v340 = v323[v339];
                float v341;
                v341 = v333 + v340;
                v333 = v341;
                v336 += 1l ;
            }
            v334 += 1l ;
        }
        auto v342 = cooperative_groups::coalesced_threads();
        int v343;
        v343 = threadIdx.x;
        int v344;
        v344 = v343 / 16l;
        auto v345 = cooperative_groups::labeled_partition(v342,v344);
        float v346;
        v346 = cooperative_groups::reduce(v345, v333, v320);
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
                v354 = v323[v353];
                float v355;
                v355 = v354 / v346;
                assert("Tensor range check" && 0 <= v348 && v348 < 1l);
                assert("Tensor range check" && 0 <= v350 && v350 < 4l);
                v347[v353] = v355;
                v350 += 1l ;
            }
            v348 += 1l ;
        }
        assert("Tensor range check" && 0 <= v254 && v254 < 64l);
        int v356;
        v356 = 0l;
        while (while_method_3(v356)){
            assert("Tensor range check" && 0 <= v356 && v356 < 1l);
            int v358;
            v358 = 64l * v356;
            int v359;
            v359 = v358 + v263;
            assert("Tensor range check" && 0 <= v356 && v356 < 1l);
            int v360;
            v360 = 4l * v356;
            int4* v361;
            v361 = reinterpret_cast<int4*>(v347 + v360);
            int4* v362;
            v362 = reinterpret_cast<int4*>(v3 + v359);
            assert("Pointer alignment check" && (unsigned long long)(v361) % 4l == 0 && (unsigned long long)(v362) % 4l == 0);
            *v362 = *v361;
            v356 += 1l ;
        }
        v254 += 1l ;
    }
    v15.sync() ;
    int v363;
    v363 = threadIdx.x;
    bool v364;
    v364 = 0l <= v363;
    bool v365;
    v365 = v364 == false;
    if (v365){
        assert("The index needs to be zero or positive." && v364);
    } else {
    }
    int v367;
    v367 = v363 % 16l;
    int v368;
    v368 = v363 / 16l;
    bool v369;
    v369 = v368 < 2l;
    bool v370;
    v370 = v369 == false;
    if (v370){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v369);
    } else {
    }
    assert("Tensor range check" && 0 <= v368 && v368 < 2l);
    assert("Tensor range check" && 0 <= v367 && v367 < 16l);
    int v372;
    v372 = 4l * v367;
    int v373;
    v373 = 64l * v368;
    int v374;
    v374 = v373 + v372;
    assert("Tensor range check" && 0 <= v368 && v368 < 2l);
    assert("Tensor range check" && 0 <= v367 && v367 < 16l);
    int v375;
    v375 = blockIdx.x;
    int v376;
    v376 = v375;
    while (while_method_2(v376)){
        bool v378;
        v378 = 0l <= v376;
        bool v379;
        v379 = v378 == false;
        if (v379){
            assert("The index needs to be zero or positive." && v378);
        } else {
        }
        bool v381;
        v381 = v376 < 64l;
        bool v382;
        v382 = v381 == false;
        if (v382){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v381);
        } else {
        }
        assert("Tensor range check" && 0 <= v376 && v376 < 64l);
        int v384;
        v384 = 128l * v376;
        int v385;
        v385 = v384 + v374;
        float v386[4l];
        int v387[4l];
        int v388;
        v388 = 0l;
        while (while_method_3(v388)){
            assert("Tensor range check" && 0 <= v388 && v388 < 1l);
            int v390;
            v390 = 4l * v388;
            assert("Tensor range check" && 0 <= v388 && v388 < 1l);
            int v391;
            v391 = 64l * v388;
            int v392;
            v392 = v391 + v385;
            int4* v393;
            v393 = reinterpret_cast<int4*>(v1 + v392);
            int4* v394;
            v394 = reinterpret_cast<int4*>(v386 + v390);
            assert("Pointer alignment check" && (unsigned long long)(v393) % 4l == 0 && (unsigned long long)(v394) % 4l == 0);
            *v394 = *v393;
            v388 += 1l ;
        }
        int v395;
        v395 = 0l;
        while (while_method_3(v395)){
            int v397;
            v397 = 0l;
            while (while_method_1(v397)){
                bool v399;
                v399 = 0l <= v397;
                bool v401;
                if (v399){
                    bool v400;
                    v400 = v397 < 4l;
                    v401 = v400;
                } else {
                    v401 = false;
                }
                bool v402;
                v402 = v401 == false;
                if (v402){
                    assert("The indices should be inside the range of the dimension." && v401);
                } else {
                }
                bool v404;
                v404 = 0l <= v367;
                bool v406;
                if (v404){
                    bool v405;
                    v405 = v367 < 16l;
                    v406 = v405;
                } else {
                    v406 = false;
                }
                bool v407;
                v407 = v406 == false;
                if (v407){
                    assert("The indices should be inside the range of the dimension." && v406);
                } else {
                }
                int v409;
                v409 = v367 * 4l;
                int v410;
                v410 = v397 + v409;
                bool v411;
                v411 = 0l <= v395;
                bool v413;
                if (v411){
                    bool v412;
                    v412 = v395 < 1l;
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
                v416 = v395 * 64l;
                int v417;
                v417 = v410 + v416;
                assert("Tensor range check" && 0 <= v395 && v395 < 1l);
                assert("Tensor range check" && 0 <= v397 && v397 < 4l);
                int v418;
                v418 = 4l * v395;
                int v419;
                v419 = v418 + v397;
                v387[v419] = v417;
                v397 += 1l ;
            }
            v395 += 1l ;
        }
        bool v420;
        v420 = 0l <= v368;
        bool v421;
        v421 = v420 && v369;
        bool v422;
        v422 = v421 == false;
        if (v422){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v421);
        } else {
        }
        bool v424;
        v424 = v378 && v381;
        bool v425;
        v425 = v424 == false;
        if (v425){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v424);
        } else {
        }
        int v427;
        v427 = v376 * 2l;
        int v428;
        v428 = v427 + v368;
        float v429[4l];
        int v430;
        v430 = 0l;
        while (while_method_3(v430)){
            int v432;
            v432 = 0l;
            while (while_method_1(v432)){
                assert("Tensor range check" && 0 <= v430 && v430 < 1l);
                assert("Tensor range check" && 0 <= v432 && v432 < 4l);
                int v434;
                v434 = 4l * v430;
                int v435;
                v435 = v434 + v432;
                float v436;
                v436 = v386[v435];
                float v437;
                v437 = v436 * v436;
                assert("Tensor range check" && 0 <= v430 && v430 < 1l);
                assert("Tensor range check" && 0 <= v432 && v432 < 4l);
                v429[v435] = v437;
                v432 += 1l ;
            }
            v430 += 1l ;
        }
        float v438;
        v438 = 0.0f;
        int v439;
        v439 = 0l;
        while (while_method_3(v439)){
            int v441;
            v441 = 0l;
            while (while_method_1(v441)){
                assert("Tensor range check" && 0 <= v439 && v439 < 1l);
                assert("Tensor range check" && 0 <= v441 && v441 < 4l);
                int v443;
                v443 = 4l * v439;
                int v444;
                v444 = v443 + v441;
                float v445;
                v445 = v429[v444];
                float v446;
                v446 = v438 + v445;
                v438 = v446;
                v441 += 1l ;
            }
            v439 += 1l ;
        }
        auto v447 = cooperative_groups::coalesced_threads();
        int v448;
        v448 = threadIdx.x;
        int v449;
        v449 = v448 / 16l;
        auto v450 = cooperative_groups::labeled_partition(v447,v449);
        Closure0 v451{};
        float v452;
        v452 = cooperative_groups::reduce(v450, v438, v451);
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
                v460 = v386[v459];
                bool v461;
                v461 = v452 == 0.0f;
                bool v462;
                v462 = v461 != true;
                float v464;
                if (v462){
                    float v463;
                    v463 = v460 / v452;
                    v464 = v463;
                } else {
                    v464 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v454 && v454 < 1l);
                assert("Tensor range check" && 0 <= v456 && v456 < 4l);
                v453[v459] = v464;
                v456 += 1l ;
            }
            v454 += 1l ;
        }
        assert("Tensor range check" && 0 <= v376 && v376 < 64l);
        int v465;
        v465 = 0l;
        while (while_method_3(v465)){
            assert("Tensor range check" && 0 <= v465 && v465 < 1l);
            int v467;
            v467 = 64l * v465;
            int v468;
            v468 = v467 + v385;
            assert("Tensor range check" && 0 <= v465 && v465 < 1l);
            int v469;
            v469 = 4l * v465;
            int4* v470;
            v470 = reinterpret_cast<int4*>(v453 + v469);
            int4* v471;
            v471 = reinterpret_cast<int4*>(v7 + v468);
            assert("Pointer alignment check" && (unsigned long long)(v470) % 4l == 0 && (unsigned long long)(v471) % 4l == 0);
            *v471 = *v470;
            v465 += 1l ;
        }
        v376 += 1l ;
    }
    v15.sync() ;
    int v472;
    v472 = threadIdx.x;
    bool v473;
    v473 = 0l <= v472;
    bool v474;
    v474 = v473 == false;
    if (v474){
        assert("The index needs to be zero or positive." && v473);
    } else {
    }
    int v476;
    v476 = v472 % 16l;
    int v477;
    v477 = v472 / 16l;
    bool v478;
    v478 = v477 < 2l;
    bool v479;
    v479 = v478 == false;
    if (v479){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v478);
    } else {
    }
    assert("Tensor range check" && 0 <= v477 && v477 < 2l);
    assert("Tensor range check" && 0 <= v476 && v476 < 16l);
    int v481;
    v481 = 4l * v476;
    int v482;
    v482 = 64l * v477;
    int v483;
    v483 = v482 + v481;
    assert("Tensor range check" && 0 <= v477 && v477 < 2l);
    int v484;
    v484 = blockIdx.x;
    int v485;
    v485 = v484;
    while (while_method_2(v485)){
        bool v487;
        v487 = 0l <= v485;
        bool v488;
        v488 = v487 == false;
        if (v488){
            assert("The index needs to be zero or positive." && v487);
        } else {
        }
        bool v490;
        v490 = v485 < 64l;
        bool v491;
        v491 = v490 == false;
        if (v491){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v490);
        } else {
        }
        assert("Tensor range check" && 0 <= v485 && v485 < 64l);
        int v493;
        v493 = 128l * v485;
        int v494;
        v494 = v493 + v483;
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
            v500 = 64l * v497;
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
                v513 = 0l <= v476;
                bool v515;
                if (v513){
                    bool v514;
                    v514 = v476 < 16l;
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
                v518 = v476 * 4l;
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
                v525 = v504 * 64l;
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
        v529 = 0l <= v477;
        bool v530;
        v530 = v529 && v478;
        bool v531;
        v531 = v530 == false;
        if (v531){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v530);
        } else {
        }
        bool v533;
        v533 = v487 && v490;
        bool v534;
        v534 = v533 == false;
        if (v534){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v533);
        } else {
        }
        int v536;
        v536 = v485 * 2l;
        int v537;
        v537 = v536 + v477;
        float v538; int v539;
        Tuple1 tmp62 = Tuple1{-1.0f / 0.0f, 0l};
        v538 = tmp62.v0; v539 = tmp62.v1;
        int v540;
        v540 = 0l;
        while (while_method_3(v540)){
            int v542;
            v542 = 0l;
            while (while_method_1(v542)){
                assert("Tensor range check" && 0 <= v540 && v540 < 1l);
                assert("Tensor range check" && 0 <= v542 && v542 < 4l);
                int v544;
                v544 = 4l * v540;
                int v545;
                v545 = v544 + v542;
                float v546;
                v546 = v495[v545];
                int v547;
                v547 = v496[v545];
                bool v548;
                v548 = v538 > v546;
                float v549; int v550;
                if (v548){
                    v549 = v538; v550 = v539;
                } else {
                    v549 = v546; v550 = v547;
                }
                v538 = v549;
                v539 = v550;
                v542 += 1l ;
            }
            v540 += 1l ;
        }
        auto v551 = cooperative_groups::coalesced_threads();
        int v552;
        v552 = threadIdx.x;
        int v553;
        v553 = v552 / 16l;
        auto v554 = cooperative_groups::labeled_partition(v551,v553);
        Closure1 v555{};
        float v556; int v557;
        Tuple1 tmp63 = cooperative_groups::reduce(v554, Tuple1{v538, v539}, v555);
        v556 = tmp63.v0; v557 = tmp63.v1;
        assert("Tensor range check" && 0 <= v485 && v485 < 64l);
        int v558;
        v558 = 2l * v485;
        int v559;
        v559 = v558 + v477;
        v8[v559] = v557;
        v485 += 1l ;
    }
    v15.sync() ;
    int v560;
    v560 = threadIdx.x;
    bool v561;
    v561 = 0l <= v560;
    bool v562;
    v562 = v561 == false;
    if (v562){
        assert("The index needs to be zero or positive." && v561);
    } else {
    }
    int v564;
    v564 = v560 % 16l;
    int v565;
    v565 = v560 / 16l;
    bool v566;
    v566 = v565 < 2l;
    bool v567;
    v567 = v566 == false;
    if (v567){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v566);
    } else {
    }
    assert("Tensor range check" && 0 <= v565 && v565 < 2l);
    assert("Tensor range check" && 0 <= v564 && v564 < 16l);
    int v569;
    v569 = 4l * v564;
    int v570;
    v570 = 64l * v565;
    int v571;
    v571 = v570 + v569;
    assert("Tensor range check" && 0 <= v565 && v565 < 2l);
    assert("Tensor range check" && 0 <= v564 && v564 < 16l);
    int v572;
    v572 = blockIdx.x;
    int v573;
    v573 = v572;
    while (while_method_2(v573)){
        bool v575;
        v575 = 0l <= v573;
        bool v576;
        v576 = v575 == false;
        if (v576){
            assert("The index needs to be zero or positive." && v575);
        } else {
        }
        bool v578;
        v578 = v573 < 64l;
        bool v579;
        v579 = v578 == false;
        if (v579){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v578);
        } else {
        }
        assert("Tensor range check" && 0 <= v573 && v573 < 64l);
        int v581;
        v581 = 128l * v573;
        int v582;
        v582 = v581 + v571;
        float v583[4l];
        int v584[4l];
        int v585;
        v585 = 0l;
        while (while_method_3(v585)){
            assert("Tensor range check" && 0 <= v585 && v585 < 1l);
            int v587;
            v587 = 4l * v585;
            assert("Tensor range check" && 0 <= v585 && v585 < 1l);
            int v588;
            v588 = 64l * v585;
            int v589;
            v589 = v588 + v582;
            int4* v590;
            v590 = reinterpret_cast<int4*>(v1 + v589);
            int4* v591;
            v591 = reinterpret_cast<int4*>(v583 + v587);
            assert("Pointer alignment check" && (unsigned long long)(v590) % 4l == 0 && (unsigned long long)(v591) % 4l == 0);
            *v591 = *v590;
            v585 += 1l ;
        }
        int v592;
        v592 = 0l;
        while (while_method_3(v592)){
            int v594;
            v594 = 0l;
            while (while_method_1(v594)){
                bool v596;
                v596 = 0l <= v594;
                bool v598;
                if (v596){
                    bool v597;
                    v597 = v594 < 4l;
                    v598 = v597;
                } else {
                    v598 = false;
                }
                bool v599;
                v599 = v598 == false;
                if (v599){
                    assert("The indices should be inside the range of the dimension." && v598);
                } else {
                }
                bool v601;
                v601 = 0l <= v564;
                bool v603;
                if (v601){
                    bool v602;
                    v602 = v564 < 16l;
                    v603 = v602;
                } else {
                    v603 = false;
                }
                bool v604;
                v604 = v603 == false;
                if (v604){
                    assert("The indices should be inside the range of the dimension." && v603);
                } else {
                }
                int v606;
                v606 = v564 * 4l;
                int v607;
                v607 = v594 + v606;
                bool v608;
                v608 = 0l <= v592;
                bool v610;
                if (v608){
                    bool v609;
                    v609 = v592 < 1l;
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
                int v613;
                v613 = v592 * 64l;
                int v614;
                v614 = v607 + v613;
                assert("Tensor range check" && 0 <= v592 && v592 < 1l);
                assert("Tensor range check" && 0 <= v594 && v594 < 4l);
                int v615;
                v615 = 4l * v592;
                int v616;
                v616 = v615 + v594;
                v584[v616] = v614;
                v594 += 1l ;
            }
            v592 += 1l ;
        }
        bool v617;
        v617 = 0l <= v565;
        bool v618;
        v618 = v617 && v566;
        bool v619;
        v619 = v618 == false;
        if (v619){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v618);
        } else {
        }
        bool v621;
        v621 = v575 && v578;
        bool v622;
        v622 = v621 == false;
        if (v622){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v621);
        } else {
        }
        int v624;
        v624 = v573 * 2l;
        int v625;
        v625 = v624 + v565;
        float v626;
        v626 = 0.0f;
        int v627;
        v627 = 0l;
        while (while_method_3(v627)){
            int v629;
            v629 = 0l;
            while (while_method_1(v629)){
                assert("Tensor range check" && 0 <= v627 && v627 < 1l);
                assert("Tensor range check" && 0 <= v629 && v629 < 4l);
                int v631;
                v631 = 4l * v627;
                int v632;
                v632 = v631 + v629;
                float v633;
                v633 = v583[v632];
                float v634;
                v634 = v626 + v633;
                v626 = v634;
                v629 += 1l ;
            }
            v627 += 1l ;
        }
        auto v635 = cooperative_groups::coalesced_threads();
        int v636;
        v636 = threadIdx.x;
        int v637;
        v637 = v636 / 16l;
        auto v638 = cooperative_groups::labeled_partition(v635,v637);
        Closure0 v639{};
        float v640;
        v640 = cooperative_groups::reduce(v638, v626, v639);
        float v641;
        v641 = v640 / 64.0f;
        float v642[4l];
        int v643;
        v643 = 0l;
        while (while_method_3(v643)){
            int v645;
            v645 = 0l;
            while (while_method_1(v645)){
                assert("Tensor range check" && 0 <= v643 && v643 < 1l);
                assert("Tensor range check" && 0 <= v645 && v645 < 4l);
                int v647;
                v647 = 4l * v643;
                int v648;
                v648 = v647 + v645;
                float v649;
                v649 = v583[v648];
                float v650;
                v650 = v649 - v641;
                float v651;
                v651 = exp(v650);
                assert("Tensor range check" && 0 <= v643 && v643 < 1l);
                assert("Tensor range check" && 0 <= v645 && v645 < 4l);
                v642[v648] = v651;
                v645 += 1l ;
            }
            v643 += 1l ;
        }
        float v652;
        v652 = 0.0f;
        int v653;
        v653 = 0l;
        while (while_method_3(v653)){
            int v655;
            v655 = 0l;
            while (while_method_1(v655)){
                assert("Tensor range check" && 0 <= v653 && v653 < 1l);
                assert("Tensor range check" && 0 <= v655 && v655 < 4l);
                int v657;
                v657 = 4l * v653;
                int v658;
                v658 = v657 + v655;
                float v659;
                v659 = v642[v658];
                float v660;
                v660 = v652 + v659;
                v652 = v660;
                v655 += 1l ;
            }
            v653 += 1l ;
        }
        auto v661 = cooperative_groups::coalesced_threads();
        int v662;
        v662 = threadIdx.x;
        int v663;
        v663 = v662 / 16l;
        auto v664 = cooperative_groups::labeled_partition(v661,v663);
        float v665;
        v665 = cooperative_groups::reduce(v664, v652, v639);
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
                v673 = v642[v672];
                float v674;
                v674 = v673 / v665;
                assert("Tensor range check" && 0 <= v667 && v667 < 1l);
                assert("Tensor range check" && 0 <= v669 && v669 < 4l);
                v666[v672] = v674;
                v669 += 1l ;
            }
            v667 += 1l ;
        }
        float v675[4l];
        float v676;
        v676 = 0.0f;
        int v677;
        v677 = 0l;
        while (while_method_3(v677)){
            assert("Tensor range check" && 0 <= v677 && v677 < 1l);
            int v679;
            v679 = 4l * v677;
            assert("Tensor range check" && 0 <= v677 && v677 < 1l);
            int v680; float v681;
            Tuple0 tmp64 = Tuple0{0l, 0.0f};
            v680 = tmp64.v0; v681 = tmp64.v1;
            while (while_method_1(v680)){
                assert("Tensor range check" && 0 <= v680 && v680 < 4l);
                int v683;
                v683 = v680 + v679;
                float v684;
                v684 = v666[v683];
                float v685;
                v685 = v681 + v684;
                v681 = v685;
                v680 += 1l ;
            }
            auto v686 = cooperative_groups::coalesced_threads();
            int v687;
            v687 = threadIdx.x;
            int v688;
            v688 = v687 / 16l;
            auto v689 = cooperative_groups::labeled_partition(v686,v688);
            Closure2 v690{};
            float v691;
            v691 = cooperative_groups::inclusive_scan(v689, v681, v690);
            float v692;
            v692 = v689.shfl_up(v691,1);
            bool v693;
            v693 = v689.thread_rank() == 0;
            float v694;
            if (v693){
                v694 = 0.0f;
            } else {
                v694 = v692;
            }
            float v695;
            v695 = v689.shfl(v691,v689.num_threads()-1);
            float v696;
            v696 = v676 + v694;
            int v697; float v698;
            Tuple0 tmp65 = Tuple0{0l, v696};
            v697 = tmp65.v0; v698 = tmp65.v1;
            while (while_method_1(v697)){
                assert("Tensor range check" && 0 <= v697 && v697 < 4l);
                int v700;
                v700 = v697 + v679;
                float v701;
                v701 = v666[v700];
                float v702;
                v702 = v698 + v701;
                assert("Tensor range check" && 0 <= v697 && v697 < 4l);
                v675[v700] = v702;
                v698 = v702;
                v697 += 1l ;
            }
            float v703;
            v703 = v676 + v695;
            v676 = v703;
            v677 += 1l ;
        }
        assert("Tensor range check" && 0 <= v573 && v573 < 64l);
        int v704;
        v704 = 0l;
        while (while_method_3(v704)){
            assert("Tensor range check" && 0 <= v704 && v704 < 1l);
            int v706;
            v706 = 64l * v704;
            int v707;
            v707 = v706 + v582;
            assert("Tensor range check" && 0 <= v704 && v704 < 1l);
            int v708;
            v708 = 4l * v704;
            int4* v709;
            v709 = reinterpret_cast<int4*>(v666 + v708);
            int4* v710;
            v710 = reinterpret_cast<int4*>(v5 + v707);
            assert("Pointer alignment check" && (unsigned long long)(v709) % 4l == 0 && (unsigned long long)(v710) % 4l == 0);
            *v710 = *v709;
            int4* v711;
            v711 = reinterpret_cast<int4*>(v675 + v708);
            int4* v712;
            v712 = reinterpret_cast<int4*>(v6 + v707);
            assert("Pointer alignment check" && (unsigned long long)(v711) % 4l == 0 && (unsigned long long)(v712) % 4l == 0);
            *v712 = *v711;
            v704 += 1l ;
        }
        v573 += 1l ;
    }
    v15.sync() ;
    int v713;
    v713 = threadIdx.x;
    bool v714;
    v714 = 0l <= v713;
    bool v715;
    v715 = v714 == false;
    if (v715){
        assert("The index needs to be zero or positive." && v714);
    } else {
    }
    int v717;
    v717 = v713 % 16l;
    int v718;
    v718 = v713 / 16l;
    bool v719;
    v719 = v718 < 2l;
    bool v720;
    v720 = v719 == false;
    if (v720){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v719);
    } else {
    }
    assert("Tensor range check" && 0 <= v718 && v718 < 2l);
    assert("Tensor range check" && 0 <= v717 && v717 < 16l);
    int v722;
    v722 = 4l * v717;
    int v723;
    v723 = 64l * v718;
    int v724;
    v724 = v723 + v722;
    assert("Tensor range check" && 0 <= v718 && v718 < 2l);
    assert("Tensor range check" && 0 <= v717 && v717 < 16l);
    int v725;
    v725 = blockIdx.x;
    int v726;
    v726 = v725;
    while (while_method_2(v726)){
        bool v728;
        v728 = 0l <= v726;
        bool v729;
        v729 = v728 == false;
        if (v729){
            assert("The index needs to be zero or positive." && v728);
        } else {
        }
        bool v731;
        v731 = v726 < 64l;
        bool v732;
        v732 = v731 == false;
        if (v732){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v731);
        } else {
        }
        assert("Tensor range check" && 0 <= v726 && v726 < 64l);
        int v734;
        v734 = 128l * v726;
        int v735;
        v735 = v734 + v724;
        int v736[4l];
        int v737[4l];
        int v738;
        v738 = 0l;
        while (while_method_3(v738)){
            assert("Tensor range check" && 0 <= v738 && v738 < 1l);
            int v740;
            v740 = 4l * v738;
            assert("Tensor range check" && 0 <= v738 && v738 < 1l);
            int v741;
            v741 = 64l * v738;
            int v742;
            v742 = v741 + v735;
            int4* v743;
            v743 = reinterpret_cast<int4*>(v0 + v742);
            int4* v744;
            v744 = reinterpret_cast<int4*>(v736 + v740);
            assert("Pointer alignment check" && (unsigned long long)(v743) % 4l == 0 && (unsigned long long)(v744) % 4l == 0);
            *v744 = *v743;
            v738 += 1l ;
        }
        int v745;
        v745 = 0l;
        while (while_method_3(v745)){
            int v747;
            v747 = 0l;
            while (while_method_1(v747)){
                bool v749;
                v749 = 0l <= v747;
                bool v751;
                if (v749){
                    bool v750;
                    v750 = v747 < 4l;
                    v751 = v750;
                } else {
                    v751 = false;
                }
                bool v752;
                v752 = v751 == false;
                if (v752){
                    assert("The indices should be inside the range of the dimension." && v751);
                } else {
                }
                bool v754;
                v754 = 0l <= v717;
                bool v756;
                if (v754){
                    bool v755;
                    v755 = v717 < 16l;
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
                int v759;
                v759 = v717 * 4l;
                int v760;
                v760 = v747 + v759;
                bool v761;
                v761 = 0l <= v745;
                bool v763;
                if (v761){
                    bool v762;
                    v762 = v745 < 1l;
                    v763 = v762;
                } else {
                    v763 = false;
                }
                bool v764;
                v764 = v763 == false;
                if (v764){
                    assert("The indices should be inside the range of the dimension." && v763);
                } else {
                }
                int v766;
                v766 = v745 * 64l;
                int v767;
                v767 = v760 + v766;
                assert("Tensor range check" && 0 <= v745 && v745 < 1l);
                assert("Tensor range check" && 0 <= v747 && v747 < 4l);
                int v768;
                v768 = 4l * v745;
                int v769;
                v769 = v768 + v747;
                v737[v769] = v767;
                v747 += 1l ;
            }
            v745 += 1l ;
        }
        bool v770;
        v770 = 0l <= v718;
        bool v771;
        v771 = v770 && v719;
        bool v772;
        v772 = v771 == false;
        if (v772){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v771);
        } else {
        }
        bool v774;
        v774 = v728 && v731;
        bool v775;
        v775 = v774 == false;
        if (v775){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v774);
        } else {
        }
        int v777;
        v777 = v726 * 2l;
        int v778;
        v778 = v777 + v718;
        int v779[4l];
        int v780;
        v780 = 0l;
        int v781;
        v781 = 0l;
        while (while_method_3(v781)){
            assert("Tensor range check" && 0 <= v781 && v781 < 1l);
            int v783;
            v783 = 4l * v781;
            assert("Tensor range check" && 0 <= v781 && v781 < 1l);
            int v784; int v785;
            Tuple2 tmp66 = Tuple2{0l, 0l};
            v784 = tmp66.v0; v785 = tmp66.v1;
            while (while_method_1(v784)){
                assert("Tensor range check" && 0 <= v784 && v784 < 4l);
                int v787;
                v787 = v784 + v783;
                int v788;
                v788 = v736[v787];
                int v789;
                v789 = v785 + v788;
                v785 = v789;
                v784 += 1l ;
            }
            auto v790 = cooperative_groups::coalesced_threads();
            int v791;
            v791 = threadIdx.x;
            int v792;
            v792 = v791 / 16l;
            auto v793 = cooperative_groups::labeled_partition(v790,v792);
            Closure3 v794{};
            int v795;
            v795 = cooperative_groups::inclusive_scan(v793, v785, v794);
            int v796;
            v796 = v793.shfl_up(v795,1);
            bool v797;
            v797 = v793.thread_rank() == 0;
            int v798;
            if (v797){
                v798 = 0l;
            } else {
                v798 = v796;
            }
            int v799;
            v799 = v793.shfl(v795,v793.num_threads()-1);
            int v800;
            v800 = v780 + v798;
            int v801; int v802;
            Tuple2 tmp67 = Tuple2{0l, v800};
            v801 = tmp67.v0; v802 = tmp67.v1;
            while (while_method_1(v801)){
                assert("Tensor range check" && 0 <= v801 && v801 < 4l);
                int v804;
                v804 = v801 + v783;
                int v805;
                v805 = v736[v804];
                assert("Tensor range check" && 0 <= v801 && v801 < 4l);
                v779[v804] = v802;
                int v806;
                v806 = v802 + v805;
                v802 = v806;
                v801 += 1l ;
            }
            int v807;
            v807 = v780 + v799;
            v780 = v807;
            v781 += 1l ;
        }
        assert("Tensor range check" && 0 <= v726 && v726 < 64l);
        int v808;
        v808 = 0l;
        while (while_method_3(v808)){
            assert("Tensor range check" && 0 <= v808 && v808 < 1l);
            int v810;
            v810 = 64l * v808;
            int v811;
            v811 = v810 + v735;
            assert("Tensor range check" && 0 <= v808 && v808 < 1l);
            int v812;
            v812 = 4l * v808;
            int4* v813;
            v813 = reinterpret_cast<int4*>(v779 + v812);
            int4* v814;
            v814 = reinterpret_cast<int4*>(v12 + v811);
            assert("Pointer alignment check" && (unsigned long long)(v813) % 4l == 0 && (unsigned long long)(v814) % 4l == 0);
            *v814 = *v813;
            v808 += 1l ;
        }
        v726 += 1l ;
    }
    v15.sync() ;
    int v815;
    v815 = threadIdx.x;
    bool v816;
    v816 = 0l <= v815;
    bool v817;
    v817 = v816 == false;
    if (v817){
        assert("The index needs to be zero or positive." && v816);
    } else {
    }
    int v819;
    v819 = v815 % 16l;
    int v820;
    v820 = v815 / 16l;
    bool v821;
    v821 = v820 < 2l;
    bool v822;
    v822 = v821 == false;
    if (v822){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v821);
    } else {
    }
    assert("Tensor range check" && 0 <= v820 && v820 < 2l);
    assert("Tensor range check" && 0 <= v819 && v819 < 16l);
    int v824;
    v824 = 4l * v819;
    int v825;
    v825 = 64l * v820;
    int v826;
    v826 = v825 + v824;
    assert("Tensor range check" && 0 <= v820 && v820 < 2l);
    assert("Tensor range check" && 0 <= v819 && v819 < 16l);
    int v827;
    v827 = blockIdx.x;
    int v828;
    v828 = v827;
    while (while_method_2(v828)){
        bool v830;
        v830 = 0l <= v828;
        bool v831;
        v831 = v830 == false;
        if (v831){
            assert("The index needs to be zero or positive." && v830);
        } else {
        }
        bool v833;
        v833 = v828 < 64l;
        bool v834;
        v834 = v833 == false;
        if (v834){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v833);
        } else {
        }
        assert("Tensor range check" && 0 <= v828 && v828 < 64l);
        int v836;
        v836 = 128l * v828;
        int v837;
        v837 = v836 + v826;
        float v838[4l];
        int v839[4l];
        int v840;
        v840 = 0l;
        while (while_method_3(v840)){
            assert("Tensor range check" && 0 <= v840 && v840 < 1l);
            int v842;
            v842 = 4l * v840;
            assert("Tensor range check" && 0 <= v840 && v840 < 1l);
            int v843;
            v843 = 64l * v840;
            int v844;
            v844 = v843 + v837;
            int4* v845;
            v845 = reinterpret_cast<int4*>(v1 + v844);
            int4* v846;
            v846 = reinterpret_cast<int4*>(v838 + v842);
            assert("Pointer alignment check" && (unsigned long long)(v845) % 4l == 0 && (unsigned long long)(v846) % 4l == 0);
            *v846 = *v845;
            v840 += 1l ;
        }
        int v847;
        v847 = 0l;
        while (while_method_3(v847)){
            int v849;
            v849 = 0l;
            while (while_method_1(v849)){
                bool v851;
                v851 = 0l <= v849;
                bool v853;
                if (v851){
                    bool v852;
                    v852 = v849 < 4l;
                    v853 = v852;
                } else {
                    v853 = false;
                }
                bool v854;
                v854 = v853 == false;
                if (v854){
                    assert("The indices should be inside the range of the dimension." && v853);
                } else {
                }
                bool v856;
                v856 = 0l <= v819;
                bool v858;
                if (v856){
                    bool v857;
                    v857 = v819 < 16l;
                    v858 = v857;
                } else {
                    v858 = false;
                }
                bool v859;
                v859 = v858 == false;
                if (v859){
                    assert("The indices should be inside the range of the dimension." && v858);
                } else {
                }
                int v861;
                v861 = v819 * 4l;
                int v862;
                v862 = v849 + v861;
                bool v863;
                v863 = 0l <= v847;
                bool v865;
                if (v863){
                    bool v864;
                    v864 = v847 < 1l;
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
                int v868;
                v868 = v847 * 64l;
                int v869;
                v869 = v862 + v868;
                assert("Tensor range check" && 0 <= v847 && v847 < 1l);
                assert("Tensor range check" && 0 <= v849 && v849 < 4l);
                int v870;
                v870 = 4l * v847;
                int v871;
                v871 = v870 + v849;
                v839[v871] = v869;
                v849 += 1l ;
            }
            v847 += 1l ;
        }
        bool v872;
        v872 = 0l <= v820;
        bool v873;
        v873 = v872 && v821;
        bool v874;
        v874 = v873 == false;
        if (v874){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v873);
        } else {
        }
        bool v876;
        v876 = v830 && v833;
        bool v877;
        v877 = v876 == false;
        if (v877){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v876);
        } else {
        }
        int v879;
        v879 = v828 * 2l;
        int v880;
        v880 = v879 + v820;
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
                v888 = v838[v887];
                int v889;
                v889 = v839[v887];
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
        v911 = v910 / 16l;
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
                v922 = v838[v921];
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
        v936 = v935 / 16l;
        auto v937 = cooperative_groups::labeled_partition(v934,v936);
        Closure0 v938{};
        float v939;
        v939 = cooperative_groups::reduce(v937, v925, v938);
        float v940;
        v940 = (float)v914;
        float v941;
        v941 = v939 / v940;
        float v942[4l];
        int v943;
        v943 = 0l;
        while (while_method_3(v943)){
            int v945;
            v945 = 0l;
            while (while_method_1(v945)){
                assert("Tensor range check" && 0 <= v943 && v943 < 1l);
                assert("Tensor range check" && 0 <= v945 && v945 < 4l);
                int v947;
                v947 = 4l * v943;
                int v948;
                v948 = v947 + v945;
                float v949;
                v949 = v838[v948];
                bool v950;
                v950 = v881[v948];
                float v951;
                if (v950){
                    v951 = v949;
                } else {
                    v951 = -1.0f / 0.0f;
                }
                float v952;
                v952 = v951 - v941;
                float v953;
                v953 = exp(v952);
                assert("Tensor range check" && 0 <= v943 && v943 < 1l);
                assert("Tensor range check" && 0 <= v945 && v945 < 4l);
                v942[v948] = v953;
                v945 += 1l ;
            }
            v943 += 1l ;
        }
        float v954;
        v954 = 0.0f;
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
                v961 = v942[v960];
                float v962;
                v962 = v954 + v961;
                v954 = v962;
                v957 += 1l ;
            }
            v955 += 1l ;
        }
        auto v963 = cooperative_groups::coalesced_threads();
        int v964;
        v964 = threadIdx.x;
        int v965;
        v965 = v964 / 16l;
        auto v966 = cooperative_groups::labeled_partition(v963,v965);
        float v967;
        v967 = cooperative_groups::reduce(v966, v954, v938);
        float v968[4l];
        int v969;
        v969 = 0l;
        while (while_method_3(v969)){
            int v971;
            v971 = 0l;
            while (while_method_1(v971)){
                assert("Tensor range check" && 0 <= v969 && v969 < 1l);
                assert("Tensor range check" && 0 <= v971 && v971 < 4l);
                int v973;
                v973 = 4l * v969;
                int v974;
                v974 = v973 + v971;
                float v975;
                v975 = v942[v974];
                float v976;
                v976 = v975 / v967;
                assert("Tensor range check" && 0 <= v969 && v969 < 1l);
                assert("Tensor range check" && 0 <= v971 && v971 < 4l);
                v968[v974] = v976;
                v971 += 1l ;
            }
            v969 += 1l ;
        }
        assert("Tensor range check" && 0 <= v828 && v828 < 64l);
        int v977;
        v977 = 0l;
        while (while_method_3(v977)){
            assert("Tensor range check" && 0 <= v977 && v977 < 1l);
            int v979;
            v979 = 64l * v977;
            int v980;
            v980 = v979 + v837;
            assert("Tensor range check" && 0 <= v977 && v977 < 1l);
            int v981;
            v981 = 4l * v977;
            int4* v982;
            v982 = reinterpret_cast<int4*>(v968 + v981);
            int4* v983;
            v983 = reinterpret_cast<int4*>(v4 + v980);
            assert("Pointer alignment check" && (unsigned long long)(v982) % 4l == 0 && (unsigned long long)(v983) % 4l == 0);
            *v983 = *v982;
            v977 += 1l ;
        }
        v828 += 1l ;
    }
    v15.sync() ;
    int v984;
    v984 = threadIdx.x;
    unsigned long long v985;
    v985 = (unsigned long long)v984;
    curandStatePhilox4_32_10_t v986;
    curand_init(12344321ull,v985,0ull,&v986);
    int v987;
    v987 = threadIdx.x;
    bool v988;
    v988 = 0l <= v987;
    bool v989;
    v989 = v988 == false;
    if (v989){
        assert("The index needs to be zero or positive." && v988);
    } else {
    }
    int v991;
    v991 = v987 % 16l;
    int v992;
    v992 = v987 / 16l;
    bool v993;
    v993 = v992 < 2l;
    bool v994;
    v994 = v993 == false;
    if (v994){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v993);
    } else {
    }
    assert("Tensor range check" && 0 <= v992 && v992 < 2l);
    assert("Tensor range check" && 0 <= v991 && v991 < 16l);
    int v996;
    v996 = 4l * v991;
    int v997;
    v997 = 64l * v992;
    int v998;
    v998 = v997 + v996;
    assert("Tensor range check" && 0 <= v992 && v992 < 2l);
    assert("Tensor range check" && 0 <= v991 && v991 < 16l);
    assert("Tensor range check" && 0 <= v992 && v992 < 2l);
    int v999;
    v999 = blockIdx.x;
    int v1000;
    v1000 = v999;
    while (while_method_2(v1000)){
        bool v1002;
        v1002 = 0l <= v1000;
        bool v1003;
        v1003 = v1002 == false;
        if (v1003){
            assert("The index needs to be zero or positive." && v1002);
        } else {
        }
        bool v1005;
        v1005 = v1000 < 64l;
        bool v1006;
        v1006 = v1005 == false;
        if (v1006){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1005);
        } else {
        }
        assert("Tensor range check" && 0 <= v1000 && v1000 < 64l);
        int v1008;
        v1008 = 128l * v1000;
        int v1009;
        v1009 = v1008 + v998;
        float v1010[4l];
        int v1011[4l];
        int v1012;
        v1012 = 0l;
        while (while_method_3(v1012)){
            assert("Tensor range check" && 0 <= v1012 && v1012 < 1l);
            int v1014;
            v1014 = 4l * v1012;
            assert("Tensor range check" && 0 <= v1012 && v1012 < 1l);
            int v1015;
            v1015 = 64l * v1012;
            int v1016;
            v1016 = v1015 + v1009;
            int4* v1017;
            v1017 = reinterpret_cast<int4*>(v1 + v1016);
            int4* v1018;
            v1018 = reinterpret_cast<int4*>(v1010 + v1014);
            assert("Pointer alignment check" && (unsigned long long)(v1017) % 4l == 0 && (unsigned long long)(v1018) % 4l == 0);
            *v1018 = *v1017;
            v1012 += 1l ;
        }
        int v1019;
        v1019 = 0l;
        while (while_method_3(v1019)){
            int v1021;
            v1021 = 0l;
            while (while_method_1(v1021)){
                bool v1023;
                v1023 = 0l <= v1021;
                bool v1025;
                if (v1023){
                    bool v1024;
                    v1024 = v1021 < 4l;
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
                bool v1028;
                v1028 = 0l <= v991;
                bool v1030;
                if (v1028){
                    bool v1029;
                    v1029 = v991 < 16l;
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
                int v1033;
                v1033 = v991 * 4l;
                int v1034;
                v1034 = v1021 + v1033;
                bool v1035;
                v1035 = 0l <= v1019;
                bool v1037;
                if (v1035){
                    bool v1036;
                    v1036 = v1019 < 1l;
                    v1037 = v1036;
                } else {
                    v1037 = false;
                }
                bool v1038;
                v1038 = v1037 == false;
                if (v1038){
                    assert("The indices should be inside the range of the dimension." && v1037);
                } else {
                }
                int v1040;
                v1040 = v1019 * 64l;
                int v1041;
                v1041 = v1034 + v1040;
                assert("Tensor range check" && 0 <= v1019 && v1019 < 1l);
                assert("Tensor range check" && 0 <= v1021 && v1021 < 4l);
                int v1042;
                v1042 = 4l * v1019;
                int v1043;
                v1043 = v1042 + v1021;
                v1011[v1043] = v1041;
                v1021 += 1l ;
            }
            v1019 += 1l ;
        }
        bool v1044;
        v1044 = 0l <= v992;
        bool v1045;
        v1045 = v1044 && v993;
        bool v1046;
        v1046 = v1045 == false;
        if (v1046){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1045);
        } else {
        }
        bool v1048;
        v1048 = v1002 && v1005;
        bool v1049;
        v1049 = v1048 == false;
        if (v1049){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1048);
        } else {
        }
        int v1051;
        v1051 = v1000 * 2l;
        int v1052;
        v1052 = v1051 + v992;
        float v1053;
        v1053 = 0.0f;
        int v1054;
        v1054 = 0l;
        while (while_method_3(v1054)){
            int v1056;
            v1056 = 0l;
            while (while_method_1(v1056)){
                assert("Tensor range check" && 0 <= v1054 && v1054 < 1l);
                assert("Tensor range check" && 0 <= v1056 && v1056 < 4l);
                int v1058;
                v1058 = 4l * v1054;
                int v1059;
                v1059 = v1058 + v1056;
                float v1060;
                v1060 = v1010[v1059];
                float v1061;
                v1061 = v1053 + v1060;
                v1053 = v1061;
                v1056 += 1l ;
            }
            v1054 += 1l ;
        }
        auto v1062 = cooperative_groups::coalesced_threads();
        int v1063;
        v1063 = threadIdx.x;
        int v1064;
        v1064 = v1063 / 16l;
        auto v1065 = cooperative_groups::labeled_partition(v1062,v1064);
        Closure0 v1066{};
        float v1067;
        v1067 = cooperative_groups::reduce(v1065, v1053, v1066);
        float v1068;
        v1068 = v1067 / 64.0f;
        float v1069[4l];
        int v1070;
        v1070 = 0l;
        while (while_method_3(v1070)){
            int v1072;
            v1072 = 0l;
            while (while_method_1(v1072)){
                assert("Tensor range check" && 0 <= v1070 && v1070 < 1l);
                assert("Tensor range check" && 0 <= v1072 && v1072 < 4l);
                int v1074;
                v1074 = 4l * v1070;
                int v1075;
                v1075 = v1074 + v1072;
                float v1076;
                v1076 = v1010[v1075];
                float v1077;
                v1077 = v1076 - v1068;
                float v1078;
                v1078 = exp(v1077);
                assert("Tensor range check" && 0 <= v1070 && v1070 < 1l);
                assert("Tensor range check" && 0 <= v1072 && v1072 < 4l);
                v1069[v1075] = v1078;
                v1072 += 1l ;
            }
            v1070 += 1l ;
        }
        float v1079;
        v1079 = 0.0f;
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
                v1086 = v1069[v1085];
                float v1087;
                v1087 = v1079 + v1086;
                v1079 = v1087;
                v1082 += 1l ;
            }
            v1080 += 1l ;
        }
        auto v1088 = cooperative_groups::coalesced_threads();
        int v1089;
        v1089 = threadIdx.x;
        int v1090;
        v1090 = v1089 / 16l;
        auto v1091 = cooperative_groups::labeled_partition(v1088,v1090);
        float v1092;
        v1092 = cooperative_groups::reduce(v1091, v1079, v1066);
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
                v1100 = v1069[v1099];
                float v1101;
                v1101 = v1100 / v1092;
                assert("Tensor range check" && 0 <= v1094 && v1094 < 1l);
                assert("Tensor range check" && 0 <= v1096 && v1096 < 4l);
                v1093[v1099] = v1101;
                v1096 += 1l ;
            }
            v1094 += 1l ;
        }
        float v1102[4l];
        float v1103;
        v1103 = 0.0f;
        int v1104;
        v1104 = 0l;
        while (while_method_3(v1104)){
            assert("Tensor range check" && 0 <= v1104 && v1104 < 1l);
            int v1106;
            v1106 = 4l * v1104;
            assert("Tensor range check" && 0 <= v1104 && v1104 < 1l);
            int v1107; float v1108;
            Tuple0 tmp68 = Tuple0{0l, 0.0f};
            v1107 = tmp68.v0; v1108 = tmp68.v1;
            while (while_method_1(v1107)){
                assert("Tensor range check" && 0 <= v1107 && v1107 < 4l);
                int v1110;
                v1110 = v1107 + v1106;
                float v1111;
                v1111 = v1093[v1110];
                float v1112;
                v1112 = v1108 + v1111;
                v1108 = v1112;
                v1107 += 1l ;
            }
            auto v1113 = cooperative_groups::coalesced_threads();
            int v1114;
            v1114 = threadIdx.x;
            int v1115;
            v1115 = v1114 / 16l;
            auto v1116 = cooperative_groups::labeled_partition(v1113,v1115);
            Closure2 v1117{};
            float v1118;
            v1118 = cooperative_groups::inclusive_scan(v1116, v1108, v1117);
            float v1119;
            v1119 = v1116.shfl_up(v1118,1);
            bool v1120;
            v1120 = v1116.thread_rank() == 0;
            float v1121;
            if (v1120){
                v1121 = 0.0f;
            } else {
                v1121 = v1119;
            }
            float v1122;
            v1122 = v1116.shfl(v1118,v1116.num_threads()-1);
            float v1123;
            v1123 = v1103 + v1121;
            int v1124; float v1125;
            Tuple0 tmp69 = Tuple0{0l, v1123};
            v1124 = tmp69.v0; v1125 = tmp69.v1;
            while (while_method_1(v1124)){
                assert("Tensor range check" && 0 <= v1124 && v1124 < 4l);
                int v1127;
                v1127 = v1124 + v1106;
                float v1128;
                v1128 = v1093[v1127];
                float v1129;
                v1129 = v1125 + v1128;
                assert("Tensor range check" && 0 <= v1124 && v1124 < 4l);
                v1102[v1127] = v1129;
                v1125 = v1129;
                v1124 += 1l ;
            }
            float v1130;
            v1130 = v1103 + v1122;
            v1103 = v1130;
            v1104 += 1l ;
        }
        float v1131[4l];
        bool v1132[4l];
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
                v1139 = v1102[v1138];
                float v1140;
                v1140 = v1093[v1138];
                bool v1141;
                v1141 = v1140 > 0.0f;
                assert("Tensor range check" && 0 <= v1133 && v1133 < 1l);
                assert("Tensor range check" && 0 <= v1135 && v1135 < 4l);
                v1131[v1138] = v1139;
                v1132[v1138] = v1141;
                v1135 += 1l ;
            }
            v1133 += 1l ;
        }
        float v1142; bool v1143;
        Tuple3 tmp70 = Tuple3{-1.0f / 0.0f, false};
        v1142 = tmp70.v0; v1143 = tmp70.v1;
        int v1144;
        v1144 = 0l;
        while (while_method_3(v1144)){
            int v1146;
            v1146 = 0l;
            while (while_method_1(v1146)){
                assert("Tensor range check" && 0 <= v1144 && v1144 < 1l);
                assert("Tensor range check" && 0 <= v1146 && v1146 < 4l);
                int v1148;
                v1148 = 4l * v1144;
                int v1149;
                v1149 = v1148 + v1146;
                float v1150;
                v1150 = v1131[v1149];
                bool v1151;
                v1151 = v1132[v1149];
                float v1158; bool v1159;
                if (v1143){
                    if (v1151){
                        bool v1152;
                        v1152 = v1142 >= v1150;
                        float v1153;
                        if (v1152){
                            v1153 = v1142;
                        } else {
                            v1153 = v1150;
                        }
                        v1158 = v1153; v1159 = true;
                    } else {
                        v1158 = v1142; v1159 = v1143;
                    }
                } else {
                    if (v1151){
                        v1158 = v1150; v1159 = v1151;
                    } else {
                        v1158 = v1142; v1159 = v1143;
                    }
                }
                v1142 = v1158;
                v1143 = v1159;
                v1146 += 1l ;
            }
            v1144 += 1l ;
        }
        auto v1160 = cooperative_groups::coalesced_threads();
        int v1161;
        v1161 = threadIdx.x;
        int v1162;
        v1162 = v1161 / 16l;
        auto v1163 = cooperative_groups::labeled_partition(v1160,v1162);
        Closure5 v1164{};
        float v1165; bool v1166;
        Tuple3 tmp71 = cooperative_groups::reduce(v1163, Tuple3{v1142, v1143}, v1164);
        v1165 = tmp71.v0; v1166 = tmp71.v1;
        bool v1167;
        v1167 = v1166 == false;
        if (v1167){
            assert("The local reduce must be true." && v1166);
        } else {
        }
        float v1169[4l];
        int v1170[4l];
        int v1171;
        v1171 = 0l;
        while (while_method_3(v1171)){
            int v1173;
            v1173 = 0l;
            while (while_method_1(v1173)){
                assert("Tensor range check" && 0 <= v1171 && v1171 < 1l);
                assert("Tensor range check" && 0 <= v1173 && v1173 < 4l);
                int v1175;
                v1175 = 4l * v1171;
                int v1176;
                v1176 = v1175 + v1173;
                int v1177;
                v1177 = v1011[v1176];
                float v1178;
                v1178 = curand_uniform(&v986);
                assert("Tensor range check" && 0 <= v1171 && v1171 < 1l);
                assert("Tensor range check" && 0 <= v1173 && v1173 < 4l);
                v1169[v1176] = v1178;
                v1170[v1176] = v1177;
                v1173 += 1l ;
            }
            v1171 += 1l ;
        }
        float v1179; int v1180;
        Tuple1 tmp72 = Tuple1{0.0f, 2147483647l};
        v1179 = tmp72.v0; v1180 = tmp72.v1;
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
                float v1187;
                v1187 = v1169[v1186];
                int v1188;
                v1188 = v1170[v1186];
                bool v1189;
                v1189 = v1180 < v1188;
                float v1190; int v1191;
                if (v1189){
                    v1190 = v1179; v1191 = v1180;
                } else {
                    v1190 = v1187; v1191 = v1188;
                }
                v1179 = v1190;
                v1180 = v1191;
                v1183 += 1l ;
            }
            v1181 += 1l ;
        }
        auto v1192 = cooperative_groups::coalesced_threads();
        int v1193;
        v1193 = threadIdx.x;
        int v1194;
        v1194 = v1193 / 16l;
        auto v1195 = cooperative_groups::labeled_partition(v1192,v1194);
        Closure6 v1196{};
        float v1197; int v1198;
        Tuple1 tmp73 = cooperative_groups::reduce(v1195, Tuple1{v1179, v1180}, v1196);
        v1197 = tmp73.v0; v1198 = tmp73.v1;
        float v1199;
        v1199 = v1165 * v1197;
        int v1200[4l];
        bool v1201[4l];
        int v1202;
        v1202 = 0l;
        while (while_method_3(v1202)){
            int v1204;
            v1204 = 0l;
            while (while_method_1(v1204)){
                assert("Tensor range check" && 0 <= v1202 && v1202 < 1l);
                assert("Tensor range check" && 0 <= v1204 && v1204 < 4l);
                int v1206;
                v1206 = 4l * v1202;
                int v1207;
                v1207 = v1206 + v1204;
                float v1208;
                v1208 = v1131[v1207];
                bool v1209;
                v1209 = v1132[v1207];
                int v1210;
                v1210 = v1011[v1207];
                int v1213; bool v1214;
                if (v1209){
                    float v1211;
                    v1211 = v1208 - v1199;
                    bool v1212;
                    v1212 = v1211 >= 0.0f;
                    v1213 = v1210; v1214 = v1212;
                } else {
                    v1213 = 2147483647l; v1214 = false;
                }
                assert("Tensor range check" && 0 <= v1202 && v1202 < 1l);
                assert("Tensor range check" && 0 <= v1204 && v1204 < 4l);
                v1200[v1207] = v1213;
                v1201[v1207] = v1214;
                v1204 += 1l ;
            }
            v1202 += 1l ;
        }
        int v1215; bool v1216;
        Tuple4 tmp74 = Tuple4{2147483647l, false};
        v1215 = tmp74.v0; v1216 = tmp74.v1;
        int v1217;
        v1217 = 0l;
        while (while_method_3(v1217)){
            int v1219;
            v1219 = 0l;
            while (while_method_1(v1219)){
                assert("Tensor range check" && 0 <= v1217 && v1217 < 1l);
                assert("Tensor range check" && 0 <= v1219 && v1219 < 4l);
                int v1221;
                v1221 = 4l * v1217;
                int v1222;
                v1222 = v1221 + v1219;
                int v1223;
                v1223 = v1200[v1222];
                bool v1224;
                v1224 = v1201[v1222];
                int v1231; bool v1232;
                if (v1216){
                    if (v1224){
                        bool v1225;
                        v1225 = v1215 < v1223;
                        int v1226;
                        if (v1225){
                            v1226 = v1215;
                        } else {
                            v1226 = v1223;
                        }
                        v1231 = v1226; v1232 = true;
                    } else {
                        v1231 = v1215; v1232 = v1216;
                    }
                } else {
                    if (v1224){
                        v1231 = v1223; v1232 = v1224;
                    } else {
                        v1231 = v1215; v1232 = v1216;
                    }
                }
                v1215 = v1231;
                v1216 = v1232;
                v1219 += 1l ;
            }
            v1217 += 1l ;
        }
        auto v1233 = cooperative_groups::coalesced_threads();
        int v1234;
        v1234 = threadIdx.x;
        int v1235;
        v1235 = v1234 / 16l;
        auto v1236 = cooperative_groups::labeled_partition(v1233,v1235);
        Closure7 v1237{};
        int v1238; bool v1239;
        Tuple4 tmp75 = cooperative_groups::reduce(v1236, Tuple4{v1215, v1216}, v1237);
        v1238 = tmp75.v0; v1239 = tmp75.v1;
        bool v1240;
        v1240 = v1239 == false;
        if (v1240){
            assert("The local reduce must be true." && v1239);
        } else {
        }
        assert("Tensor range check" && 0 <= v1000 && v1000 < 64l);
        int v1242;
        v1242 = 0l;
        while (while_method_3(v1242)){
            assert("Tensor range check" && 0 <= v1242 && v1242 < 1l);
            int v1244;
            v1244 = 64l * v1242;
            int v1245;
            v1245 = v1244 + v1009;
            assert("Tensor range check" && 0 <= v1242 && v1242 < 1l);
            int v1246;
            v1246 = 4l * v1242;
            int4* v1247;
            v1247 = reinterpret_cast<int4*>(v1093 + v1246);
            int4* v1248;
            v1248 = reinterpret_cast<int4*>(v13 + v1245);
            assert("Pointer alignment check" && (unsigned long long)(v1247) % 4l == 0 && (unsigned long long)(v1248) % 4l == 0);
            *v1248 = *v1247;
            v1242 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1000 && v1000 < 64l);
        int v1249;
        v1249 = 2l * v1000;
        int v1250;
        v1250 = v1249 + v992;
        v14[v1250] = v1238;
        v1000 += 1l ;
    }
    v15.sync() ;
    int v1251;
    v1251 = threadIdx.x;
    unsigned long long v1252;
    v1252 = (unsigned long long)v1251;
    curandStatePhilox4_32_10_t v1253;
    curand_init(12344321ull,v1252,0ull,&v1253);
    int v1254;
    v1254 = threadIdx.x;
    bool v1255;
    v1255 = 0l <= v1254;
    bool v1256;
    v1256 = v1255 == false;
    if (v1256){
        assert("The index needs to be zero or positive." && v1255);
    } else {
    }
    int v1258;
    v1258 = v1254 % 16l;
    int v1259;
    v1259 = v1254 / 16l;
    bool v1260;
    v1260 = v1259 < 2l;
    bool v1261;
    v1261 = v1260 == false;
    if (v1261){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1260);
    } else {
    }
    assert("Tensor range check" && 0 <= v1259 && v1259 < 2l);
    assert("Tensor range check" && 0 <= v1258 && v1258 < 16l);
    int v1263;
    v1263 = 4l * v1258;
    int v1264;
    v1264 = 64l * v1259;
    int v1265;
    v1265 = v1264 + v1263;
    assert("Tensor range check" && 0 <= v1259 && v1259 < 2l);
    assert("Tensor range check" && 0 <= v1258 && v1258 < 16l);
    assert("Tensor range check" && 0 <= v1259 && v1259 < 2l);
    int v1266;
    v1266 = blockIdx.x;
    int v1267;
    v1267 = v1266;
    while (while_method_2(v1267)){
        bool v1269;
        v1269 = 0l <= v1267;
        bool v1270;
        v1270 = v1269 == false;
        if (v1270){
            assert("The index needs to be zero or positive." && v1269);
        } else {
        }
        bool v1272;
        v1272 = v1267 < 64l;
        bool v1273;
        v1273 = v1272 == false;
        if (v1273){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1272);
        } else {
        }
        assert("Tensor range check" && 0 <= v1267 && v1267 < 64l);
        int v1275;
        v1275 = 128l * v1267;
        int v1276;
        v1276 = v1275 + v1265;
        float v1277[4l];
        int v1278[4l];
        int v1279;
        v1279 = 0l;
        while (while_method_3(v1279)){
            assert("Tensor range check" && 0 <= v1279 && v1279 < 1l);
            int v1281;
            v1281 = 4l * v1279;
            assert("Tensor range check" && 0 <= v1279 && v1279 < 1l);
            int v1282;
            v1282 = 64l * v1279;
            int v1283;
            v1283 = v1282 + v1276;
            int4* v1284;
            v1284 = reinterpret_cast<int4*>(v1 + v1283);
            int4* v1285;
            v1285 = reinterpret_cast<int4*>(v1277 + v1281);
            assert("Pointer alignment check" && (unsigned long long)(v1284) % 4l == 0 && (unsigned long long)(v1285) % 4l == 0);
            *v1285 = *v1284;
            v1279 += 1l ;
        }
        int v1286;
        v1286 = 0l;
        while (while_method_3(v1286)){
            int v1288;
            v1288 = 0l;
            while (while_method_1(v1288)){
                bool v1290;
                v1290 = 0l <= v1288;
                bool v1292;
                if (v1290){
                    bool v1291;
                    v1291 = v1288 < 4l;
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
                bool v1295;
                v1295 = 0l <= v1258;
                bool v1297;
                if (v1295){
                    bool v1296;
                    v1296 = v1258 < 16l;
                    v1297 = v1296;
                } else {
                    v1297 = false;
                }
                bool v1298;
                v1298 = v1297 == false;
                if (v1298){
                    assert("The indices should be inside the range of the dimension." && v1297);
                } else {
                }
                int v1300;
                v1300 = v1258 * 4l;
                int v1301;
                v1301 = v1288 + v1300;
                bool v1302;
                v1302 = 0l <= v1286;
                bool v1304;
                if (v1302){
                    bool v1303;
                    v1303 = v1286 < 1l;
                    v1304 = v1303;
                } else {
                    v1304 = false;
                }
                bool v1305;
                v1305 = v1304 == false;
                if (v1305){
                    assert("The indices should be inside the range of the dimension." && v1304);
                } else {
                }
                int v1307;
                v1307 = v1286 * 64l;
                int v1308;
                v1308 = v1301 + v1307;
                assert("Tensor range check" && 0 <= v1286 && v1286 < 1l);
                assert("Tensor range check" && 0 <= v1288 && v1288 < 4l);
                int v1309;
                v1309 = 4l * v1286;
                int v1310;
                v1310 = v1309 + v1288;
                v1278[v1310] = v1308;
                v1288 += 1l ;
            }
            v1286 += 1l ;
        }
        bool v1311;
        v1311 = 0l <= v1259;
        bool v1312;
        v1312 = v1311 && v1260;
        bool v1313;
        v1313 = v1312 == false;
        if (v1313){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1312);
        } else {
        }
        bool v1315;
        v1315 = v1269 && v1272;
        bool v1316;
        v1316 = v1315 == false;
        if (v1316){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1315);
        } else {
        }
        int v1318;
        v1318 = v1267 * 2l;
        int v1319;
        v1319 = v1318 + v1259;
        bool v1320[4l];
        int v1321;
        v1321 = 0l;
        while (while_method_3(v1321)){
            int v1323;
            v1323 = 0l;
            while (while_method_1(v1323)){
                assert("Tensor range check" && 0 <= v1321 && v1321 < 1l);
                assert("Tensor range check" && 0 <= v1323 && v1323 < 4l);
                int v1325;
                v1325 = 4l * v1321;
                int v1326;
                v1326 = v1325 + v1323;
                float v1327;
                v1327 = v1277[v1326];
                int v1328;
                v1328 = v1278[v1326];
                bool v1329;
                v1329 = v1328 < 11l;
                assert("Tensor range check" && 0 <= v1321 && v1321 < 1l);
                assert("Tensor range check" && 0 <= v1323 && v1323 < 4l);
                v1320[v1326] = v1329;
                v1323 += 1l ;
            }
            v1321 += 1l ;
        }
        int v1330[4l];
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
                bool v1337;
                v1337 = v1320[v1336];
                int v1338;
                if (v1337){
                    v1338 = 1l;
                } else {
                    v1338 = 0l;
                }
                assert("Tensor range check" && 0 <= v1331 && v1331 < 1l);
                assert("Tensor range check" && 0 <= v1333 && v1333 < 4l);
                v1330[v1336] = v1338;
                v1333 += 1l ;
            }
            v1331 += 1l ;
        }
        int v1339;
        v1339 = 0l;
        int v1340;
        v1340 = 0l;
        while (while_method_3(v1340)){
            int v1342;
            v1342 = 0l;
            while (while_method_1(v1342)){
                assert("Tensor range check" && 0 <= v1340 && v1340 < 1l);
                assert("Tensor range check" && 0 <= v1342 && v1342 < 4l);
                int v1344;
                v1344 = 4l * v1340;
                int v1345;
                v1345 = v1344 + v1342;
                int v1346;
                v1346 = v1330[v1345];
                int v1347;
                v1347 = v1339 + v1346;
                v1339 = v1347;
                v1342 += 1l ;
            }
            v1340 += 1l ;
        }
        auto v1348 = cooperative_groups::coalesced_threads();
        int v1349;
        v1349 = threadIdx.x;
        int v1350;
        v1350 = v1349 / 16l;
        auto v1351 = cooperative_groups::labeled_partition(v1348,v1350);
        Closure4 v1352{};
        int v1353;
        v1353 = cooperative_groups::reduce(v1351, v1339, v1352);
        float v1354[4l];
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
                float v1361;
                v1361 = v1277[v1360];
                bool v1362;
                v1362 = v1320[v1360];
                float v1363;
                if (v1362){
                    v1363 = v1361;
                } else {
                    v1363 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1355 && v1355 < 1l);
                assert("Tensor range check" && 0 <= v1357 && v1357 < 4l);
                v1354[v1360] = v1363;
                v1357 += 1l ;
            }
            v1355 += 1l ;
        }
        float v1364;
        v1364 = 0.0f;
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
                v1371 = v1354[v1370];
                float v1372;
                v1372 = v1364 + v1371;
                v1364 = v1372;
                v1367 += 1l ;
            }
            v1365 += 1l ;
        }
        auto v1373 = cooperative_groups::coalesced_threads();
        int v1374;
        v1374 = threadIdx.x;
        int v1375;
        v1375 = v1374 / 16l;
        auto v1376 = cooperative_groups::labeled_partition(v1373,v1375);
        Closure0 v1377{};
        float v1378;
        v1378 = cooperative_groups::reduce(v1376, v1364, v1377);
        float v1379;
        v1379 = (float)v1353;
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
                v1388 = v1277[v1387];
                bool v1389;
                v1389 = v1320[v1387];
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
        v1406 = cooperative_groups::reduce(v1405, v1393, v1377);
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
            Tuple0 tmp76 = Tuple0{0l, 0.0f};
            v1421 = tmp76.v0; v1422 = tmp76.v1;
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
            Tuple0 tmp77 = Tuple0{0l, v1437};
            v1438 = tmp77.v0; v1439 = tmp77.v1;
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
        Tuple3 tmp78 = Tuple3{-1.0f / 0.0f, false};
        v1456 = tmp78.v0; v1457 = tmp78.v1;
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
        Tuple3 tmp79 = cooperative_groups::reduce(v1477, Tuple3{v1456, v1457}, v1478);
        v1479 = tmp79.v0; v1480 = tmp79.v1;
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
                v1491 = v1278[v1490];
                float v1492;
                v1492 = curand_uniform(&v1253);
                assert("Tensor range check" && 0 <= v1485 && v1485 < 1l);
                assert("Tensor range check" && 0 <= v1487 && v1487 < 4l);
                v1483[v1490] = v1492;
                v1484[v1490] = v1491;
                v1487 += 1l ;
            }
            v1485 += 1l ;
        }
        float v1493; int v1494;
        Tuple1 tmp80 = Tuple1{0.0f, 2147483647l};
        v1493 = tmp80.v0; v1494 = tmp80.v1;
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
        Tuple1 tmp81 = cooperative_groups::reduce(v1509, Tuple1{v1493, v1494}, v1510);
        v1511 = tmp81.v0; v1512 = tmp81.v1;
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
                v1524 = v1278[v1521];
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
        Tuple4 tmp82 = Tuple4{2147483647l, false};
        v1529 = tmp82.v0; v1530 = tmp82.v1;
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
        Tuple4 tmp83 = cooperative_groups::reduce(v1550, Tuple4{v1529, v1530}, v1551);
        v1552 = tmp83.v0; v1553 = tmp83.v1;
        bool v1554;
        v1554 = v1553 == false;
        if (v1554){
            assert("The local reduce must be true." && v1553);
        } else {
        }
        assert("Tensor range check" && 0 <= v1267 && v1267 < 64l);
        int v1556;
        v1556 = 0l;
        while (while_method_3(v1556)){
            assert("Tensor range check" && 0 <= v1556 && v1556 < 1l);
            int v1558;
            v1558 = 64l * v1556;
            int v1559;
            v1559 = v1558 + v1276;
            assert("Tensor range check" && 0 <= v1556 && v1556 < 1l);
            int v1560;
            v1560 = 4l * v1556;
            int4* v1561;
            v1561 = reinterpret_cast<int4*>(v1407 + v1560);
            int4* v1562;
            v1562 = reinterpret_cast<int4*>(v13 + v1559);
            assert("Pointer alignment check" && (unsigned long long)(v1561) % 4l == 0 && (unsigned long long)(v1562) % 4l == 0);
            *v1562 = *v1561;
            v1556 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1267 && v1267 < 64l);
        int v1563;
        v1563 = 2l * v1267;
        int v1564;
        v1564 = v1563 + v1259;
        v14[v1564] = v1552;
        v1267 += 1l ;
    }
    v15.sync() ;
    return ;
}
extern "C" __global__ void entry5(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14) {
    auto v15 = cooperative_groups::this_grid();
    int v16;
    v16 = threadIdx.x;
    bool v17;
    v17 = 0l <= v16;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The index needs to be zero or positive." && v17);
    } else {
    }
    int v20;
    v20 = v16 % 32l;
    int v21;
    v21 = v16 / 32l;
    bool v22;
    v22 = v21 < 1l;
    bool v23;
    v23 = v22 == false;
    if (v23){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v22);
    } else {
    }
    assert("Tensor range check" && 0 <= v21 && v21 < 1l);
    assert("Tensor range check" && 0 <= v20 && v20 < 32l);
    int v25;
    v25 = 4l * v20;
    int v26;
    v26 = 128l * v21;
    int v27;
    v27 = v26 + v25;
    assert("Tensor range check" && 0 <= v21 && v21 < 1l);
    assert("Tensor range check" && 0 <= v20 && v20 < 32l);
    int v28;
    v28 = blockIdx.x;
    int v29;
    v29 = v28;
    while (while_method_2(v29)){
        bool v31;
        v31 = 0l <= v29;
        bool v32;
        v32 = v31 == false;
        if (v32){
            assert("The index needs to be zero or positive." && v31);
        } else {
        }
        bool v34;
        v34 = v29 < 64l;
        bool v35;
        v35 = v34 == false;
        if (v35){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v34);
        } else {
        }
        assert("Tensor range check" && 0 <= v29 && v29 < 64l);
        int v37;
        v37 = 128l * v29;
        int v38;
        v38 = v37 + v27;
        int v39[4l];
        int v40[4l];
        int v41;
        v41 = 0l;
        while (while_method_3(v41)){
            assert("Tensor range check" && 0 <= v41 && v41 < 1l);
            int v43;
            v43 = 4l * v41;
            assert("Tensor range check" && 0 <= v41 && v41 < 1l);
            int v44;
            v44 = 128l * v41;
            int v45;
            v45 = v44 + v38;
            int4* v46;
            v46 = reinterpret_cast<int4*>(v0 + v45);
            int4* v47;
            v47 = reinterpret_cast<int4*>(v39 + v43);
            assert("Pointer alignment check" && (unsigned long long)(v46) % 4l == 0 && (unsigned long long)(v47) % 4l == 0);
            *v47 = *v46;
            v41 += 1l ;
        }
        int v48;
        v48 = 0l;
        while (while_method_3(v48)){
            int v50;
            v50 = 0l;
            while (while_method_1(v50)){
                bool v52;
                v52 = 0l <= v50;
                bool v54;
                if (v52){
                    bool v53;
                    v53 = v50 < 4l;
                    v54 = v53;
                } else {
                    v54 = false;
                }
                bool v55;
                v55 = v54 == false;
                if (v55){
                    assert("The indices should be inside the range of the dimension." && v54);
                } else {
                }
                bool v57;
                v57 = 0l <= v20;
                bool v59;
                if (v57){
                    bool v58;
                    v58 = v20 < 32l;
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
                int v62;
                v62 = v20 * 4l;
                int v63;
                v63 = v50 + v62;
                bool v64;
                v64 = 0l <= v48;
                bool v66;
                if (v64){
                    bool v65;
                    v65 = v48 < 1l;
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
                int v69;
                v69 = v48 * 128l;
                int v70;
                v70 = v63 + v69;
                assert("Tensor range check" && 0 <= v48 && v48 < 1l);
                assert("Tensor range check" && 0 <= v50 && v50 < 4l);
                int v71;
                v71 = 4l * v48;
                int v72;
                v72 = v71 + v50;
                v40[v72] = v70;
                v50 += 1l ;
            }
            v48 += 1l ;
        }
        bool v73;
        v73 = 0l <= v21;
        bool v74;
        v74 = v73 && v22;
        bool v75;
        v75 = v74 == false;
        if (v75){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v74);
        } else {
        }
        bool v77;
        v77 = v31 && v34;
        bool v78;
        v78 = v77 == false;
        if (v78){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v77);
        } else {
        }
        int v80;
        v80 = v29 + v21;
        assert("Tensor range check" && 0 <= v29 && v29 < 64l);
        int v81;
        v81 = 0l;
        while (while_method_3(v81)){
            assert("Tensor range check" && 0 <= v81 && v81 < 1l);
            int v83;
            v83 = 128l * v81;
            int v84;
            v84 = v83 + v38;
            assert("Tensor range check" && 0 <= v81 && v81 < 1l);
            int v85;
            v85 = 4l * v81;
            int4* v86;
            v86 = reinterpret_cast<int4*>(v39 + v85);
            int4* v87;
            v87 = reinterpret_cast<int4*>(v2 + v84);
            assert("Pointer alignment check" && (unsigned long long)(v86) % 4l == 0 && (unsigned long long)(v87) % 4l == 0);
            *v87 = *v86;
            v81 += 1l ;
        }
        v29 += 1l ;
    }
    v15.sync() ;
    int v88;
    v88 = threadIdx.x;
    bool v89;
    v89 = 0l <= v88;
    bool v90;
    v90 = v89 == false;
    if (v90){
        assert("The index needs to be zero or positive." && v89);
    } else {
    }
    int v92;
    v92 = v88 % 32l;
    int v93;
    v93 = v88 / 32l;
    bool v94;
    v94 = v93 < 1l;
    bool v95;
    v95 = v94 == false;
    if (v95){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v94);
    } else {
    }
    assert("Tensor range check" && 0 <= v93 && v93 < 1l);
    assert("Tensor range check" && 0 <= v92 && v92 < 32l);
    int v97;
    v97 = 4l * v92;
    int v98;
    v98 = 128l * v93;
    int v99;
    v99 = v98 + v97;
    assert("Tensor range check" && 0 <= v93 && v93 < 1l);
    assert("Tensor range check" && 0 <= v92 && v92 < 32l);
    int v100;
    v100 = blockIdx.x;
    int v101;
    v101 = v100;
    while (while_method_2(v101)){
        bool v103;
        v103 = 0l <= v101;
        bool v104;
        v104 = v103 == false;
        if (v104){
            assert("The index needs to be zero or positive." && v103);
        } else {
        }
        bool v106;
        v106 = v101 < 64l;
        bool v107;
        v107 = v106 == false;
        if (v107){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v106);
        } else {
        }
        assert("Tensor range check" && 0 <= v101 && v101 < 64l);
        int v109;
        v109 = 128l * v101;
        int v110;
        v110 = v109 + v99;
        float v111[4l];
        int v112[4l];
        int v113;
        v113 = 0l;
        while (while_method_3(v113)){
            assert("Tensor range check" && 0 <= v113 && v113 < 1l);
            int v115;
            v115 = 4l * v113;
            assert("Tensor range check" && 0 <= v113 && v113 < 1l);
            int v116;
            v116 = 128l * v113;
            int v117;
            v117 = v116 + v110;
            int4* v118;
            v118 = reinterpret_cast<int4*>(v1 + v117);
            int4* v119;
            v119 = reinterpret_cast<int4*>(v111 + v115);
            assert("Pointer alignment check" && (unsigned long long)(v118) % 4l == 0 && (unsigned long long)(v119) % 4l == 0);
            *v119 = *v118;
            v113 += 1l ;
        }
        int v120;
        v120 = 0l;
        while (while_method_3(v120)){
            int v122;
            v122 = 0l;
            while (while_method_1(v122)){
                bool v124;
                v124 = 0l <= v122;
                bool v126;
                if (v124){
                    bool v125;
                    v125 = v122 < 4l;
                    v126 = v125;
                } else {
                    v126 = false;
                }
                bool v127;
                v127 = v126 == false;
                if (v127){
                    assert("The indices should be inside the range of the dimension." && v126);
                } else {
                }
                bool v129;
                v129 = 0l <= v92;
                bool v131;
                if (v129){
                    bool v130;
                    v130 = v92 < 32l;
                    v131 = v130;
                } else {
                    v131 = false;
                }
                bool v132;
                v132 = v131 == false;
                if (v132){
                    assert("The indices should be inside the range of the dimension." && v131);
                } else {
                }
                int v134;
                v134 = v92 * 4l;
                int v135;
                v135 = v122 + v134;
                bool v136;
                v136 = 0l <= v120;
                bool v138;
                if (v136){
                    bool v137;
                    v137 = v120 < 1l;
                    v138 = v137;
                } else {
                    v138 = false;
                }
                bool v139;
                v139 = v138 == false;
                if (v139){
                    assert("The indices should be inside the range of the dimension." && v138);
                } else {
                }
                int v141;
                v141 = v120 * 128l;
                int v142;
                v142 = v135 + v141;
                assert("Tensor range check" && 0 <= v120 && v120 < 1l);
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                int v143;
                v143 = 4l * v120;
                int v144;
                v144 = v143 + v122;
                v112[v144] = v142;
                v122 += 1l ;
            }
            v120 += 1l ;
        }
        bool v145;
        v145 = 0l <= v93;
        bool v146;
        v146 = v145 && v94;
        bool v147;
        v147 = v146 == false;
        if (v147){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v146);
        } else {
        }
        bool v149;
        v149 = v103 && v106;
        bool v150;
        v150 = v149 == false;
        if (v150){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v149);
        } else {
        }
        int v152;
        v152 = v101 + v93;
        int v153[4l];
        int v154[4l];
        int v155;
        v155 = 0l;
        while (while_method_3(v155)){
            int v157;
            v157 = 0l;
            while (while_method_1(v157)){
                assert("Tensor range check" && 0 <= v155 && v155 < 1l);
                assert("Tensor range check" && 0 <= v157 && v157 < 4l);
                int v159;
                v159 = 4l * v155;
                int v160;
                v160 = v159 + v157;
                int v161;
                v161 = v112[v160];
                assert("Tensor range check" && 0 <= v155 && v155 < 1l);
                assert("Tensor range check" && 0 <= v157 && v157 < 4l);
                v153[v160] = v152;
                v154[v160] = v161;
                v157 += 1l ;
            }
            v155 += 1l ;
        }
        assert("Tensor range check" && 0 <= v101 && v101 < 64l);
        int v162;
        v162 = 0l;
        while (while_method_3(v162)){
            assert("Tensor range check" && 0 <= v162 && v162 < 1l);
            int v164;
            v164 = 128l * v162;
            int v165;
            v165 = v164 + v110;
            assert("Tensor range check" && 0 <= v162 && v162 < 1l);
            int v166;
            v166 = 4l * v162;
            int4* v167;
            v167 = reinterpret_cast<int4*>(v153 + v166);
            int4* v168;
            v168 = reinterpret_cast<int4*>(v9 + v165);
            assert("Pointer alignment check" && (unsigned long long)(v167) % 4l == 0 && (unsigned long long)(v168) % 4l == 0);
            *v168 = *v167;
            int4* v169;
            v169 = reinterpret_cast<int4*>(v154 + v166);
            int4* v170;
            v170 = reinterpret_cast<int4*>(v10 + v165);
            assert("Pointer alignment check" && (unsigned long long)(v169) % 4l == 0 && (unsigned long long)(v170) % 4l == 0);
            *v170 = *v169;
            v162 += 1l ;
        }
        v101 += 1l ;
    }
    v15.sync() ;
    int v171;
    v171 = threadIdx.x;
    bool v172;
    v172 = 0l <= v171;
    bool v173;
    v173 = v172 == false;
    if (v173){
        assert("The index needs to be zero or positive." && v172);
    } else {
    }
    int v175;
    v175 = v171 % 32l;
    int v176;
    v176 = v171 / 32l;
    bool v177;
    v177 = v176 < 1l;
    bool v178;
    v178 = v177 == false;
    if (v178){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v177);
    } else {
    }
    assert("Tensor range check" && 0 <= v176 && v176 < 1l);
    assert("Tensor range check" && 0 <= v175 && v175 < 32l);
    int v180;
    v180 = 4l * v175;
    int v181;
    v181 = 128l * v176;
    int v182;
    v182 = v181 + v180;
    assert("Tensor range check" && 0 <= v176 && v176 < 1l);
    int v183;
    v183 = blockIdx.x;
    int v184;
    v184 = v183;
    while (while_method_2(v184)){
        bool v186;
        v186 = 0l <= v184;
        bool v187;
        v187 = v186 == false;
        if (v187){
            assert("The index needs to be zero or positive." && v186);
        } else {
        }
        bool v189;
        v189 = v184 < 64l;
        bool v190;
        v190 = v189 == false;
        if (v190){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v189);
        } else {
        }
        assert("Tensor range check" && 0 <= v184 && v184 < 64l);
        int v192;
        v192 = 128l * v184;
        int v193;
        v193 = v192 + v182;
        float v194[4l];
        int v195[4l];
        int v196;
        v196 = 0l;
        while (while_method_3(v196)){
            assert("Tensor range check" && 0 <= v196 && v196 < 1l);
            int v198;
            v198 = 4l * v196;
            assert("Tensor range check" && 0 <= v196 && v196 < 1l);
            int v199;
            v199 = 128l * v196;
            int v200;
            v200 = v199 + v193;
            int4* v201;
            v201 = reinterpret_cast<int4*>(v1 + v200);
            int4* v202;
            v202 = reinterpret_cast<int4*>(v194 + v198);
            assert("Pointer alignment check" && (unsigned long long)(v201) % 4l == 0 && (unsigned long long)(v202) % 4l == 0);
            *v202 = *v201;
            v196 += 1l ;
        }
        int v203;
        v203 = 0l;
        while (while_method_3(v203)){
            int v205;
            v205 = 0l;
            while (while_method_1(v205)){
                bool v207;
                v207 = 0l <= v205;
                bool v209;
                if (v207){
                    bool v208;
                    v208 = v205 < 4l;
                    v209 = v208;
                } else {
                    v209 = false;
                }
                bool v210;
                v210 = v209 == false;
                if (v210){
                    assert("The indices should be inside the range of the dimension." && v209);
                } else {
                }
                bool v212;
                v212 = 0l <= v175;
                bool v214;
                if (v212){
                    bool v213;
                    v213 = v175 < 32l;
                    v214 = v213;
                } else {
                    v214 = false;
                }
                bool v215;
                v215 = v214 == false;
                if (v215){
                    assert("The indices should be inside the range of the dimension." && v214);
                } else {
                }
                int v217;
                v217 = v175 * 4l;
                int v218;
                v218 = v205 + v217;
                bool v219;
                v219 = 0l <= v203;
                bool v221;
                if (v219){
                    bool v220;
                    v220 = v203 < 1l;
                    v221 = v220;
                } else {
                    v221 = false;
                }
                bool v222;
                v222 = v221 == false;
                if (v222){
                    assert("The indices should be inside the range of the dimension." && v221);
                } else {
                }
                int v224;
                v224 = v203 * 128l;
                int v225;
                v225 = v218 + v224;
                assert("Tensor range check" && 0 <= v203 && v203 < 1l);
                assert("Tensor range check" && 0 <= v205 && v205 < 4l);
                int v226;
                v226 = 4l * v203;
                int v227;
                v227 = v226 + v205;
                v195[v227] = v225;
                v205 += 1l ;
            }
            v203 += 1l ;
        }
        bool v228;
        v228 = 0l <= v176;
        bool v229;
        v229 = v228 && v177;
        bool v230;
        v230 = v229 == false;
        if (v230){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v229);
        } else {
        }
        bool v232;
        v232 = v186 && v189;
        bool v233;
        v233 = v232 == false;
        if (v233){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v232);
        } else {
        }
        int v235;
        v235 = v184 + v176;
        assert("Tensor range check" && 0 <= v184 && v184 < 64l);
        v11[v235] = v235;
        v184 += 1l ;
    }
    v15.sync() ;
    int v236;
    v236 = threadIdx.x;
    bool v237;
    v237 = 0l <= v236;
    bool v238;
    v238 = v237 == false;
    if (v238){
        assert("The index needs to be zero or positive." && v237);
    } else {
    }
    int v240;
    v240 = v236 % 32l;
    int v241;
    v241 = v236 / 32l;
    bool v242;
    v242 = v241 < 1l;
    bool v243;
    v243 = v242 == false;
    if (v243){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v242);
    } else {
    }
    assert("Tensor range check" && 0 <= v241 && v241 < 1l);
    assert("Tensor range check" && 0 <= v240 && v240 < 32l);
    int v245;
    v245 = 4l * v240;
    int v246;
    v246 = 128l * v241;
    int v247;
    v247 = v246 + v245;
    assert("Tensor range check" && 0 <= v241 && v241 < 1l);
    assert("Tensor range check" && 0 <= v240 && v240 < 32l);
    int v248;
    v248 = blockIdx.x;
    int v249;
    v249 = v248;
    while (while_method_2(v249)){
        bool v251;
        v251 = 0l <= v249;
        bool v252;
        v252 = v251 == false;
        if (v252){
            assert("The index needs to be zero or positive." && v251);
        } else {
        }
        bool v254;
        v254 = v249 < 64l;
        bool v255;
        v255 = v254 == false;
        if (v255){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v254);
        } else {
        }
        assert("Tensor range check" && 0 <= v249 && v249 < 64l);
        int v257;
        v257 = 128l * v249;
        int v258;
        v258 = v257 + v247;
        float v259[4l];
        int v260[4l];
        int v261;
        v261 = 0l;
        while (while_method_3(v261)){
            assert("Tensor range check" && 0 <= v261 && v261 < 1l);
            int v263;
            v263 = 4l * v261;
            assert("Tensor range check" && 0 <= v261 && v261 < 1l);
            int v264;
            v264 = 128l * v261;
            int v265;
            v265 = v264 + v258;
            int4* v266;
            v266 = reinterpret_cast<int4*>(v1 + v265);
            int4* v267;
            v267 = reinterpret_cast<int4*>(v259 + v263);
            assert("Pointer alignment check" && (unsigned long long)(v266) % 4l == 0 && (unsigned long long)(v267) % 4l == 0);
            *v267 = *v266;
            v261 += 1l ;
        }
        int v268;
        v268 = 0l;
        while (while_method_3(v268)){
            int v270;
            v270 = 0l;
            while (while_method_1(v270)){
                bool v272;
                v272 = 0l <= v270;
                bool v274;
                if (v272){
                    bool v273;
                    v273 = v270 < 4l;
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
                bool v277;
                v277 = 0l <= v240;
                bool v279;
                if (v277){
                    bool v278;
                    v278 = v240 < 32l;
                    v279 = v278;
                } else {
                    v279 = false;
                }
                bool v280;
                v280 = v279 == false;
                if (v280){
                    assert("The indices should be inside the range of the dimension." && v279);
                } else {
                }
                int v282;
                v282 = v240 * 4l;
                int v283;
                v283 = v270 + v282;
                bool v284;
                v284 = 0l <= v268;
                bool v286;
                if (v284){
                    bool v285;
                    v285 = v268 < 1l;
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
                v289 = v268 * 128l;
                int v290;
                v290 = v283 + v289;
                assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                assert("Tensor range check" && 0 <= v270 && v270 < 4l);
                int v291;
                v291 = 4l * v268;
                int v292;
                v292 = v291 + v270;
                v260[v292] = v290;
                v270 += 1l ;
            }
            v268 += 1l ;
        }
        bool v293;
        v293 = 0l <= v241;
        bool v294;
        v294 = v293 && v242;
        bool v295;
        v295 = v294 == false;
        if (v295){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v294);
        } else {
        }
        bool v297;
        v297 = v251 && v254;
        bool v298;
        v298 = v297 == false;
        if (v298){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v297);
        } else {
        }
        int v300;
        v300 = v249 + v241;
        float v301;
        v301 = 0.0f;
        int v302;
        v302 = 0l;
        while (while_method_3(v302)){
            int v304;
            v304 = 0l;
            while (while_method_1(v304)){
                assert("Tensor range check" && 0 <= v302 && v302 < 1l);
                assert("Tensor range check" && 0 <= v304 && v304 < 4l);
                int v306;
                v306 = 4l * v302;
                int v307;
                v307 = v306 + v304;
                float v308;
                v308 = v259[v307];
                float v309;
                v309 = v301 + v308;
                v301 = v309;
                v304 += 1l ;
            }
            v302 += 1l ;
        }
        auto v310 = cooperative_groups::coalesced_threads();
        int v311;
        v311 = threadIdx.x;
        int v312;
        v312 = v311 / 32l;
        auto v313 = cooperative_groups::labeled_partition(v310,v312);
        Closure0 v314{};
        float v315;
        v315 = cooperative_groups::reduce(v313, v301, v314);
        float v316;
        v316 = v315 / 128.0f;
        float v317[4l];
        int v318;
        v318 = 0l;
        while (while_method_3(v318)){
            int v320;
            v320 = 0l;
            while (while_method_1(v320)){
                assert("Tensor range check" && 0 <= v318 && v318 < 1l);
                assert("Tensor range check" && 0 <= v320 && v320 < 4l);
                int v322;
                v322 = 4l * v318;
                int v323;
                v323 = v322 + v320;
                float v324;
                v324 = v259[v323];
                float v325;
                v325 = v324 - v316;
                float v326;
                v326 = exp(v325);
                assert("Tensor range check" && 0 <= v318 && v318 < 1l);
                assert("Tensor range check" && 0 <= v320 && v320 < 4l);
                v317[v323] = v326;
                v320 += 1l ;
            }
            v318 += 1l ;
        }
        float v327;
        v327 = 0.0f;
        int v328;
        v328 = 0l;
        while (while_method_3(v328)){
            int v330;
            v330 = 0l;
            while (while_method_1(v330)){
                assert("Tensor range check" && 0 <= v328 && v328 < 1l);
                assert("Tensor range check" && 0 <= v330 && v330 < 4l);
                int v332;
                v332 = 4l * v328;
                int v333;
                v333 = v332 + v330;
                float v334;
                v334 = v317[v333];
                float v335;
                v335 = v327 + v334;
                v327 = v335;
                v330 += 1l ;
            }
            v328 += 1l ;
        }
        auto v336 = cooperative_groups::coalesced_threads();
        int v337;
        v337 = threadIdx.x;
        int v338;
        v338 = v337 / 32l;
        auto v339 = cooperative_groups::labeled_partition(v336,v338);
        float v340;
        v340 = cooperative_groups::reduce(v339, v327, v314);
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
                v348 = v317[v347];
                float v349;
                v349 = v348 / v340;
                assert("Tensor range check" && 0 <= v342 && v342 < 1l);
                assert("Tensor range check" && 0 <= v344 && v344 < 4l);
                v341[v347] = v349;
                v344 += 1l ;
            }
            v342 += 1l ;
        }
        assert("Tensor range check" && 0 <= v249 && v249 < 64l);
        int v350;
        v350 = 0l;
        while (while_method_3(v350)){
            assert("Tensor range check" && 0 <= v350 && v350 < 1l);
            int v352;
            v352 = 128l * v350;
            int v353;
            v353 = v352 + v258;
            assert("Tensor range check" && 0 <= v350 && v350 < 1l);
            int v354;
            v354 = 4l * v350;
            int4* v355;
            v355 = reinterpret_cast<int4*>(v341 + v354);
            int4* v356;
            v356 = reinterpret_cast<int4*>(v3 + v353);
            assert("Pointer alignment check" && (unsigned long long)(v355) % 4l == 0 && (unsigned long long)(v356) % 4l == 0);
            *v356 = *v355;
            v350 += 1l ;
        }
        v249 += 1l ;
    }
    v15.sync() ;
    int v357;
    v357 = threadIdx.x;
    bool v358;
    v358 = 0l <= v357;
    bool v359;
    v359 = v358 == false;
    if (v359){
        assert("The index needs to be zero or positive." && v358);
    } else {
    }
    int v361;
    v361 = v357 % 32l;
    int v362;
    v362 = v357 / 32l;
    bool v363;
    v363 = v362 < 1l;
    bool v364;
    v364 = v363 == false;
    if (v364){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v363);
    } else {
    }
    assert("Tensor range check" && 0 <= v362 && v362 < 1l);
    assert("Tensor range check" && 0 <= v361 && v361 < 32l);
    int v366;
    v366 = 4l * v361;
    int v367;
    v367 = 128l * v362;
    int v368;
    v368 = v367 + v366;
    assert("Tensor range check" && 0 <= v362 && v362 < 1l);
    assert("Tensor range check" && 0 <= v361 && v361 < 32l);
    int v369;
    v369 = blockIdx.x;
    int v370;
    v370 = v369;
    while (while_method_2(v370)){
        bool v372;
        v372 = 0l <= v370;
        bool v373;
        v373 = v372 == false;
        if (v373){
            assert("The index needs to be zero or positive." && v372);
        } else {
        }
        bool v375;
        v375 = v370 < 64l;
        bool v376;
        v376 = v375 == false;
        if (v376){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v375);
        } else {
        }
        assert("Tensor range check" && 0 <= v370 && v370 < 64l);
        int v378;
        v378 = 128l * v370;
        int v379;
        v379 = v378 + v368;
        float v380[4l];
        int v381[4l];
        int v382;
        v382 = 0l;
        while (while_method_3(v382)){
            assert("Tensor range check" && 0 <= v382 && v382 < 1l);
            int v384;
            v384 = 4l * v382;
            assert("Tensor range check" && 0 <= v382 && v382 < 1l);
            int v385;
            v385 = 128l * v382;
            int v386;
            v386 = v385 + v379;
            int4* v387;
            v387 = reinterpret_cast<int4*>(v1 + v386);
            int4* v388;
            v388 = reinterpret_cast<int4*>(v380 + v384);
            assert("Pointer alignment check" && (unsigned long long)(v387) % 4l == 0 && (unsigned long long)(v388) % 4l == 0);
            *v388 = *v387;
            v382 += 1l ;
        }
        int v389;
        v389 = 0l;
        while (while_method_3(v389)){
            int v391;
            v391 = 0l;
            while (while_method_1(v391)){
                bool v393;
                v393 = 0l <= v391;
                bool v395;
                if (v393){
                    bool v394;
                    v394 = v391 < 4l;
                    v395 = v394;
                } else {
                    v395 = false;
                }
                bool v396;
                v396 = v395 == false;
                if (v396){
                    assert("The indices should be inside the range of the dimension." && v395);
                } else {
                }
                bool v398;
                v398 = 0l <= v361;
                bool v400;
                if (v398){
                    bool v399;
                    v399 = v361 < 32l;
                    v400 = v399;
                } else {
                    v400 = false;
                }
                bool v401;
                v401 = v400 == false;
                if (v401){
                    assert("The indices should be inside the range of the dimension." && v400);
                } else {
                }
                int v403;
                v403 = v361 * 4l;
                int v404;
                v404 = v391 + v403;
                bool v405;
                v405 = 0l <= v389;
                bool v407;
                if (v405){
                    bool v406;
                    v406 = v389 < 1l;
                    v407 = v406;
                } else {
                    v407 = false;
                }
                bool v408;
                v408 = v407 == false;
                if (v408){
                    assert("The indices should be inside the range of the dimension." && v407);
                } else {
                }
                int v410;
                v410 = v389 * 128l;
                int v411;
                v411 = v404 + v410;
                assert("Tensor range check" && 0 <= v389 && v389 < 1l);
                assert("Tensor range check" && 0 <= v391 && v391 < 4l);
                int v412;
                v412 = 4l * v389;
                int v413;
                v413 = v412 + v391;
                v381[v413] = v411;
                v391 += 1l ;
            }
            v389 += 1l ;
        }
        bool v414;
        v414 = 0l <= v362;
        bool v415;
        v415 = v414 && v363;
        bool v416;
        v416 = v415 == false;
        if (v416){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v415);
        } else {
        }
        bool v418;
        v418 = v372 && v375;
        bool v419;
        v419 = v418 == false;
        if (v419){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v418);
        } else {
        }
        int v421;
        v421 = v370 + v362;
        float v422[4l];
        int v423;
        v423 = 0l;
        while (while_method_3(v423)){
            int v425;
            v425 = 0l;
            while (while_method_1(v425)){
                assert("Tensor range check" && 0 <= v423 && v423 < 1l);
                assert("Tensor range check" && 0 <= v425 && v425 < 4l);
                int v427;
                v427 = 4l * v423;
                int v428;
                v428 = v427 + v425;
                float v429;
                v429 = v380[v428];
                float v430;
                v430 = v429 * v429;
                assert("Tensor range check" && 0 <= v423 && v423 < 1l);
                assert("Tensor range check" && 0 <= v425 && v425 < 4l);
                v422[v428] = v430;
                v425 += 1l ;
            }
            v423 += 1l ;
        }
        float v431;
        v431 = 0.0f;
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
                v438 = v422[v437];
                float v439;
                v439 = v431 + v438;
                v431 = v439;
                v434 += 1l ;
            }
            v432 += 1l ;
        }
        auto v440 = cooperative_groups::coalesced_threads();
        int v441;
        v441 = threadIdx.x;
        int v442;
        v442 = v441 / 32l;
        auto v443 = cooperative_groups::labeled_partition(v440,v442);
        Closure0 v444{};
        float v445;
        v445 = cooperative_groups::reduce(v443, v431, v444);
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
                v453 = v380[v452];
                bool v454;
                v454 = v445 == 0.0f;
                bool v455;
                v455 = v454 != true;
                float v457;
                if (v455){
                    float v456;
                    v456 = v453 / v445;
                    v457 = v456;
                } else {
                    v457 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v447 && v447 < 1l);
                assert("Tensor range check" && 0 <= v449 && v449 < 4l);
                v446[v452] = v457;
                v449 += 1l ;
            }
            v447 += 1l ;
        }
        assert("Tensor range check" && 0 <= v370 && v370 < 64l);
        int v458;
        v458 = 0l;
        while (while_method_3(v458)){
            assert("Tensor range check" && 0 <= v458 && v458 < 1l);
            int v460;
            v460 = 128l * v458;
            int v461;
            v461 = v460 + v379;
            assert("Tensor range check" && 0 <= v458 && v458 < 1l);
            int v462;
            v462 = 4l * v458;
            int4* v463;
            v463 = reinterpret_cast<int4*>(v446 + v462);
            int4* v464;
            v464 = reinterpret_cast<int4*>(v7 + v461);
            assert("Pointer alignment check" && (unsigned long long)(v463) % 4l == 0 && (unsigned long long)(v464) % 4l == 0);
            *v464 = *v463;
            v458 += 1l ;
        }
        v370 += 1l ;
    }
    v15.sync() ;
    int v465;
    v465 = threadIdx.x;
    bool v466;
    v466 = 0l <= v465;
    bool v467;
    v467 = v466 == false;
    if (v467){
        assert("The index needs to be zero or positive." && v466);
    } else {
    }
    int v469;
    v469 = v465 % 32l;
    int v470;
    v470 = v465 / 32l;
    bool v471;
    v471 = v470 < 1l;
    bool v472;
    v472 = v471 == false;
    if (v472){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v471);
    } else {
    }
    assert("Tensor range check" && 0 <= v470 && v470 < 1l);
    assert("Tensor range check" && 0 <= v469 && v469 < 32l);
    int v474;
    v474 = 4l * v469;
    int v475;
    v475 = 128l * v470;
    int v476;
    v476 = v475 + v474;
    assert("Tensor range check" && 0 <= v470 && v470 < 1l);
    int v477;
    v477 = blockIdx.x;
    int v478;
    v478 = v477;
    while (while_method_2(v478)){
        bool v480;
        v480 = 0l <= v478;
        bool v481;
        v481 = v480 == false;
        if (v481){
            assert("The index needs to be zero or positive." && v480);
        } else {
        }
        bool v483;
        v483 = v478 < 64l;
        bool v484;
        v484 = v483 == false;
        if (v484){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v483);
        } else {
        }
        assert("Tensor range check" && 0 <= v478 && v478 < 64l);
        int v486;
        v486 = 128l * v478;
        int v487;
        v487 = v486 + v476;
        float v488[4l];
        int v489[4l];
        int v490;
        v490 = 0l;
        while (while_method_3(v490)){
            assert("Tensor range check" && 0 <= v490 && v490 < 1l);
            int v492;
            v492 = 4l * v490;
            assert("Tensor range check" && 0 <= v490 && v490 < 1l);
            int v493;
            v493 = 128l * v490;
            int v494;
            v494 = v493 + v487;
            int4* v495;
            v495 = reinterpret_cast<int4*>(v1 + v494);
            int4* v496;
            v496 = reinterpret_cast<int4*>(v488 + v492);
            assert("Pointer alignment check" && (unsigned long long)(v495) % 4l == 0 && (unsigned long long)(v496) % 4l == 0);
            *v496 = *v495;
            v490 += 1l ;
        }
        int v497;
        v497 = 0l;
        while (while_method_3(v497)){
            int v499;
            v499 = 0l;
            while (while_method_1(v499)){
                bool v501;
                v501 = 0l <= v499;
                bool v503;
                if (v501){
                    bool v502;
                    v502 = v499 < 4l;
                    v503 = v502;
                } else {
                    v503 = false;
                }
                bool v504;
                v504 = v503 == false;
                if (v504){
                    assert("The indices should be inside the range of the dimension." && v503);
                } else {
                }
                bool v506;
                v506 = 0l <= v469;
                bool v508;
                if (v506){
                    bool v507;
                    v507 = v469 < 32l;
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
                int v511;
                v511 = v469 * 4l;
                int v512;
                v512 = v499 + v511;
                bool v513;
                v513 = 0l <= v497;
                bool v515;
                if (v513){
                    bool v514;
                    v514 = v497 < 1l;
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
                v518 = v497 * 128l;
                int v519;
                v519 = v512 + v518;
                assert("Tensor range check" && 0 <= v497 && v497 < 1l);
                assert("Tensor range check" && 0 <= v499 && v499 < 4l);
                int v520;
                v520 = 4l * v497;
                int v521;
                v521 = v520 + v499;
                v489[v521] = v519;
                v499 += 1l ;
            }
            v497 += 1l ;
        }
        bool v522;
        v522 = 0l <= v470;
        bool v523;
        v523 = v522 && v471;
        bool v524;
        v524 = v523 == false;
        if (v524){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v523);
        } else {
        }
        bool v526;
        v526 = v480 && v483;
        bool v527;
        v527 = v526 == false;
        if (v527){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v526);
        } else {
        }
        int v529;
        v529 = v478 + v470;
        float v530; int v531;
        Tuple1 tmp84 = Tuple1{-1.0f / 0.0f, 0l};
        v530 = tmp84.v0; v531 = tmp84.v1;
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
                v538 = v488[v537];
                int v539;
                v539 = v489[v537];
                bool v540;
                v540 = v530 > v538;
                float v541; int v542;
                if (v540){
                    v541 = v530; v542 = v531;
                } else {
                    v541 = v538; v542 = v539;
                }
                v530 = v541;
                v531 = v542;
                v534 += 1l ;
            }
            v532 += 1l ;
        }
        auto v543 = cooperative_groups::coalesced_threads();
        int v544;
        v544 = threadIdx.x;
        int v545;
        v545 = v544 / 32l;
        auto v546 = cooperative_groups::labeled_partition(v543,v545);
        Closure1 v547{};
        float v548; int v549;
        Tuple1 tmp85 = cooperative_groups::reduce(v546, Tuple1{v530, v531}, v547);
        v548 = tmp85.v0; v549 = tmp85.v1;
        assert("Tensor range check" && 0 <= v478 && v478 < 64l);
        v8[v529] = v549;
        v478 += 1l ;
    }
    v15.sync() ;
    int v550;
    v550 = threadIdx.x;
    bool v551;
    v551 = 0l <= v550;
    bool v552;
    v552 = v551 == false;
    if (v552){
        assert("The index needs to be zero or positive." && v551);
    } else {
    }
    int v554;
    v554 = v550 % 32l;
    int v555;
    v555 = v550 / 32l;
    bool v556;
    v556 = v555 < 1l;
    bool v557;
    v557 = v556 == false;
    if (v557){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v556);
    } else {
    }
    assert("Tensor range check" && 0 <= v555 && v555 < 1l);
    assert("Tensor range check" && 0 <= v554 && v554 < 32l);
    int v559;
    v559 = 4l * v554;
    int v560;
    v560 = 128l * v555;
    int v561;
    v561 = v560 + v559;
    assert("Tensor range check" && 0 <= v555 && v555 < 1l);
    assert("Tensor range check" && 0 <= v554 && v554 < 32l);
    int v562;
    v562 = blockIdx.x;
    int v563;
    v563 = v562;
    while (while_method_2(v563)){
        bool v565;
        v565 = 0l <= v563;
        bool v566;
        v566 = v565 == false;
        if (v566){
            assert("The index needs to be zero or positive." && v565);
        } else {
        }
        bool v568;
        v568 = v563 < 64l;
        bool v569;
        v569 = v568 == false;
        if (v569){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v568);
        } else {
        }
        assert("Tensor range check" && 0 <= v563 && v563 < 64l);
        int v571;
        v571 = 128l * v563;
        int v572;
        v572 = v571 + v561;
        float v573[4l];
        int v574[4l];
        int v575;
        v575 = 0l;
        while (while_method_3(v575)){
            assert("Tensor range check" && 0 <= v575 && v575 < 1l);
            int v577;
            v577 = 4l * v575;
            assert("Tensor range check" && 0 <= v575 && v575 < 1l);
            int v578;
            v578 = 128l * v575;
            int v579;
            v579 = v578 + v572;
            int4* v580;
            v580 = reinterpret_cast<int4*>(v1 + v579);
            int4* v581;
            v581 = reinterpret_cast<int4*>(v573 + v577);
            assert("Pointer alignment check" && (unsigned long long)(v580) % 4l == 0 && (unsigned long long)(v581) % 4l == 0);
            *v581 = *v580;
            v575 += 1l ;
        }
        int v582;
        v582 = 0l;
        while (while_method_3(v582)){
            int v584;
            v584 = 0l;
            while (while_method_1(v584)){
                bool v586;
                v586 = 0l <= v584;
                bool v588;
                if (v586){
                    bool v587;
                    v587 = v584 < 4l;
                    v588 = v587;
                } else {
                    v588 = false;
                }
                bool v589;
                v589 = v588 == false;
                if (v589){
                    assert("The indices should be inside the range of the dimension." && v588);
                } else {
                }
                bool v591;
                v591 = 0l <= v554;
                bool v593;
                if (v591){
                    bool v592;
                    v592 = v554 < 32l;
                    v593 = v592;
                } else {
                    v593 = false;
                }
                bool v594;
                v594 = v593 == false;
                if (v594){
                    assert("The indices should be inside the range of the dimension." && v593);
                } else {
                }
                int v596;
                v596 = v554 * 4l;
                int v597;
                v597 = v584 + v596;
                bool v598;
                v598 = 0l <= v582;
                bool v600;
                if (v598){
                    bool v599;
                    v599 = v582 < 1l;
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
                int v603;
                v603 = v582 * 128l;
                int v604;
                v604 = v597 + v603;
                assert("Tensor range check" && 0 <= v582 && v582 < 1l);
                assert("Tensor range check" && 0 <= v584 && v584 < 4l);
                int v605;
                v605 = 4l * v582;
                int v606;
                v606 = v605 + v584;
                v574[v606] = v604;
                v584 += 1l ;
            }
            v582 += 1l ;
        }
        bool v607;
        v607 = 0l <= v555;
        bool v608;
        v608 = v607 && v556;
        bool v609;
        v609 = v608 == false;
        if (v609){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v608);
        } else {
        }
        bool v611;
        v611 = v565 && v568;
        bool v612;
        v612 = v611 == false;
        if (v612){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v611);
        } else {
        }
        int v614;
        v614 = v563 + v555;
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
                v622 = v573[v621];
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
        v626 = v625 / 32l;
        auto v627 = cooperative_groups::labeled_partition(v624,v626);
        Closure0 v628{};
        float v629;
        v629 = cooperative_groups::reduce(v627, v615, v628);
        float v630;
        v630 = v629 / 128.0f;
        float v631[4l];
        int v632;
        v632 = 0l;
        while (while_method_3(v632)){
            int v634;
            v634 = 0l;
            while (while_method_1(v634)){
                assert("Tensor range check" && 0 <= v632 && v632 < 1l);
                assert("Tensor range check" && 0 <= v634 && v634 < 4l);
                int v636;
                v636 = 4l * v632;
                int v637;
                v637 = v636 + v634;
                float v638;
                v638 = v573[v637];
                float v639;
                v639 = v638 - v630;
                float v640;
                v640 = exp(v639);
                assert("Tensor range check" && 0 <= v632 && v632 < 1l);
                assert("Tensor range check" && 0 <= v634 && v634 < 4l);
                v631[v637] = v640;
                v634 += 1l ;
            }
            v632 += 1l ;
        }
        float v641;
        v641 = 0.0f;
        int v642;
        v642 = 0l;
        while (while_method_3(v642)){
            int v644;
            v644 = 0l;
            while (while_method_1(v644)){
                assert("Tensor range check" && 0 <= v642 && v642 < 1l);
                assert("Tensor range check" && 0 <= v644 && v644 < 4l);
                int v646;
                v646 = 4l * v642;
                int v647;
                v647 = v646 + v644;
                float v648;
                v648 = v631[v647];
                float v649;
                v649 = v641 + v648;
                v641 = v649;
                v644 += 1l ;
            }
            v642 += 1l ;
        }
        auto v650 = cooperative_groups::coalesced_threads();
        int v651;
        v651 = threadIdx.x;
        int v652;
        v652 = v651 / 32l;
        auto v653 = cooperative_groups::labeled_partition(v650,v652);
        float v654;
        v654 = cooperative_groups::reduce(v653, v641, v628);
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
                v662 = v631[v661];
                float v663;
                v663 = v662 / v654;
                assert("Tensor range check" && 0 <= v656 && v656 < 1l);
                assert("Tensor range check" && 0 <= v658 && v658 < 4l);
                v655[v661] = v663;
                v658 += 1l ;
            }
            v656 += 1l ;
        }
        float v664[4l];
        float v665;
        v665 = 0.0f;
        int v666;
        v666 = 0l;
        while (while_method_3(v666)){
            assert("Tensor range check" && 0 <= v666 && v666 < 1l);
            int v668;
            v668 = 4l * v666;
            assert("Tensor range check" && 0 <= v666 && v666 < 1l);
            int v669; float v670;
            Tuple0 tmp86 = Tuple0{0l, 0.0f};
            v669 = tmp86.v0; v670 = tmp86.v1;
            while (while_method_1(v669)){
                assert("Tensor range check" && 0 <= v669 && v669 < 4l);
                int v672;
                v672 = v669 + v668;
                float v673;
                v673 = v655[v672];
                float v674;
                v674 = v670 + v673;
                v670 = v674;
                v669 += 1l ;
            }
            auto v675 = cooperative_groups::coalesced_threads();
            int v676;
            v676 = threadIdx.x;
            int v677;
            v677 = v676 / 32l;
            auto v678 = cooperative_groups::labeled_partition(v675,v677);
            Closure2 v679{};
            float v680;
            v680 = cooperative_groups::inclusive_scan(v678, v670, v679);
            float v681;
            v681 = v678.shfl_up(v680,1);
            bool v682;
            v682 = v678.thread_rank() == 0;
            float v683;
            if (v682){
                v683 = 0.0f;
            } else {
                v683 = v681;
            }
            float v684;
            v684 = v678.shfl(v680,v678.num_threads()-1);
            float v685;
            v685 = v665 + v683;
            int v686; float v687;
            Tuple0 tmp87 = Tuple0{0l, v685};
            v686 = tmp87.v0; v687 = tmp87.v1;
            while (while_method_1(v686)){
                assert("Tensor range check" && 0 <= v686 && v686 < 4l);
                int v689;
                v689 = v686 + v668;
                float v690;
                v690 = v655[v689];
                float v691;
                v691 = v687 + v690;
                assert("Tensor range check" && 0 <= v686 && v686 < 4l);
                v664[v689] = v691;
                v687 = v691;
                v686 += 1l ;
            }
            float v692;
            v692 = v665 + v684;
            v665 = v692;
            v666 += 1l ;
        }
        assert("Tensor range check" && 0 <= v563 && v563 < 64l);
        int v693;
        v693 = 0l;
        while (while_method_3(v693)){
            assert("Tensor range check" && 0 <= v693 && v693 < 1l);
            int v695;
            v695 = 128l * v693;
            int v696;
            v696 = v695 + v572;
            assert("Tensor range check" && 0 <= v693 && v693 < 1l);
            int v697;
            v697 = 4l * v693;
            int4* v698;
            v698 = reinterpret_cast<int4*>(v655 + v697);
            int4* v699;
            v699 = reinterpret_cast<int4*>(v5 + v696);
            assert("Pointer alignment check" && (unsigned long long)(v698) % 4l == 0 && (unsigned long long)(v699) % 4l == 0);
            *v699 = *v698;
            int4* v700;
            v700 = reinterpret_cast<int4*>(v664 + v697);
            int4* v701;
            v701 = reinterpret_cast<int4*>(v6 + v696);
            assert("Pointer alignment check" && (unsigned long long)(v700) % 4l == 0 && (unsigned long long)(v701) % 4l == 0);
            *v701 = *v700;
            v693 += 1l ;
        }
        v563 += 1l ;
    }
    v15.sync() ;
    int v702;
    v702 = threadIdx.x;
    bool v703;
    v703 = 0l <= v702;
    bool v704;
    v704 = v703 == false;
    if (v704){
        assert("The index needs to be zero or positive." && v703);
    } else {
    }
    int v706;
    v706 = v702 % 32l;
    int v707;
    v707 = v702 / 32l;
    bool v708;
    v708 = v707 < 1l;
    bool v709;
    v709 = v708 == false;
    if (v709){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v708);
    } else {
    }
    assert("Tensor range check" && 0 <= v707 && v707 < 1l);
    assert("Tensor range check" && 0 <= v706 && v706 < 32l);
    int v711;
    v711 = 4l * v706;
    int v712;
    v712 = 128l * v707;
    int v713;
    v713 = v712 + v711;
    assert("Tensor range check" && 0 <= v707 && v707 < 1l);
    assert("Tensor range check" && 0 <= v706 && v706 < 32l);
    int v714;
    v714 = blockIdx.x;
    int v715;
    v715 = v714;
    while (while_method_2(v715)){
        bool v717;
        v717 = 0l <= v715;
        bool v718;
        v718 = v717 == false;
        if (v718){
            assert("The index needs to be zero or positive." && v717);
        } else {
        }
        bool v720;
        v720 = v715 < 64l;
        bool v721;
        v721 = v720 == false;
        if (v721){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v720);
        } else {
        }
        assert("Tensor range check" && 0 <= v715 && v715 < 64l);
        int v723;
        v723 = 128l * v715;
        int v724;
        v724 = v723 + v713;
        int v725[4l];
        int v726[4l];
        int v727;
        v727 = 0l;
        while (while_method_3(v727)){
            assert("Tensor range check" && 0 <= v727 && v727 < 1l);
            int v729;
            v729 = 4l * v727;
            assert("Tensor range check" && 0 <= v727 && v727 < 1l);
            int v730;
            v730 = 128l * v727;
            int v731;
            v731 = v730 + v724;
            int4* v732;
            v732 = reinterpret_cast<int4*>(v0 + v731);
            int4* v733;
            v733 = reinterpret_cast<int4*>(v725 + v729);
            assert("Pointer alignment check" && (unsigned long long)(v732) % 4l == 0 && (unsigned long long)(v733) % 4l == 0);
            *v733 = *v732;
            v727 += 1l ;
        }
        int v734;
        v734 = 0l;
        while (while_method_3(v734)){
            int v736;
            v736 = 0l;
            while (while_method_1(v736)){
                bool v738;
                v738 = 0l <= v736;
                bool v740;
                if (v738){
                    bool v739;
                    v739 = v736 < 4l;
                    v740 = v739;
                } else {
                    v740 = false;
                }
                bool v741;
                v741 = v740 == false;
                if (v741){
                    assert("The indices should be inside the range of the dimension." && v740);
                } else {
                }
                bool v743;
                v743 = 0l <= v706;
                bool v745;
                if (v743){
                    bool v744;
                    v744 = v706 < 32l;
                    v745 = v744;
                } else {
                    v745 = false;
                }
                bool v746;
                v746 = v745 == false;
                if (v746){
                    assert("The indices should be inside the range of the dimension." && v745);
                } else {
                }
                int v748;
                v748 = v706 * 4l;
                int v749;
                v749 = v736 + v748;
                bool v750;
                v750 = 0l <= v734;
                bool v752;
                if (v750){
                    bool v751;
                    v751 = v734 < 1l;
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
                v755 = v734 * 128l;
                int v756;
                v756 = v749 + v755;
                assert("Tensor range check" && 0 <= v734 && v734 < 1l);
                assert("Tensor range check" && 0 <= v736 && v736 < 4l);
                int v757;
                v757 = 4l * v734;
                int v758;
                v758 = v757 + v736;
                v726[v758] = v756;
                v736 += 1l ;
            }
            v734 += 1l ;
        }
        bool v759;
        v759 = 0l <= v707;
        bool v760;
        v760 = v759 && v708;
        bool v761;
        v761 = v760 == false;
        if (v761){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v760);
        } else {
        }
        bool v763;
        v763 = v717 && v720;
        bool v764;
        v764 = v763 == false;
        if (v764){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v763);
        } else {
        }
        int v766;
        v766 = v715 + v707;
        int v767[4l];
        int v768;
        v768 = 0l;
        int v769;
        v769 = 0l;
        while (while_method_3(v769)){
            assert("Tensor range check" && 0 <= v769 && v769 < 1l);
            int v771;
            v771 = 4l * v769;
            assert("Tensor range check" && 0 <= v769 && v769 < 1l);
            int v772; int v773;
            Tuple2 tmp88 = Tuple2{0l, 0l};
            v772 = tmp88.v0; v773 = tmp88.v1;
            while (while_method_1(v772)){
                assert("Tensor range check" && 0 <= v772 && v772 < 4l);
                int v775;
                v775 = v772 + v771;
                int v776;
                v776 = v725[v775];
                int v777;
                v777 = v773 + v776;
                v773 = v777;
                v772 += 1l ;
            }
            auto v778 = cooperative_groups::coalesced_threads();
            int v779;
            v779 = threadIdx.x;
            int v780;
            v780 = v779 / 32l;
            auto v781 = cooperative_groups::labeled_partition(v778,v780);
            Closure3 v782{};
            int v783;
            v783 = cooperative_groups::inclusive_scan(v781, v773, v782);
            int v784;
            v784 = v781.shfl_up(v783,1);
            bool v785;
            v785 = v781.thread_rank() == 0;
            int v786;
            if (v785){
                v786 = 0l;
            } else {
                v786 = v784;
            }
            int v787;
            v787 = v781.shfl(v783,v781.num_threads()-1);
            int v788;
            v788 = v768 + v786;
            int v789; int v790;
            Tuple2 tmp89 = Tuple2{0l, v788};
            v789 = tmp89.v0; v790 = tmp89.v1;
            while (while_method_1(v789)){
                assert("Tensor range check" && 0 <= v789 && v789 < 4l);
                int v792;
                v792 = v789 + v771;
                int v793;
                v793 = v725[v792];
                assert("Tensor range check" && 0 <= v789 && v789 < 4l);
                v767[v792] = v790;
                int v794;
                v794 = v790 + v793;
                v790 = v794;
                v789 += 1l ;
            }
            int v795;
            v795 = v768 + v787;
            v768 = v795;
            v769 += 1l ;
        }
        assert("Tensor range check" && 0 <= v715 && v715 < 64l);
        int v796;
        v796 = 0l;
        while (while_method_3(v796)){
            assert("Tensor range check" && 0 <= v796 && v796 < 1l);
            int v798;
            v798 = 128l * v796;
            int v799;
            v799 = v798 + v724;
            assert("Tensor range check" && 0 <= v796 && v796 < 1l);
            int v800;
            v800 = 4l * v796;
            int4* v801;
            v801 = reinterpret_cast<int4*>(v767 + v800);
            int4* v802;
            v802 = reinterpret_cast<int4*>(v12 + v799);
            assert("Pointer alignment check" && (unsigned long long)(v801) % 4l == 0 && (unsigned long long)(v802) % 4l == 0);
            *v802 = *v801;
            v796 += 1l ;
        }
        v715 += 1l ;
    }
    v15.sync() ;
    int v803;
    v803 = threadIdx.x;
    bool v804;
    v804 = 0l <= v803;
    bool v805;
    v805 = v804 == false;
    if (v805){
        assert("The index needs to be zero or positive." && v804);
    } else {
    }
    int v807;
    v807 = v803 % 32l;
    int v808;
    v808 = v803 / 32l;
    bool v809;
    v809 = v808 < 1l;
    bool v810;
    v810 = v809 == false;
    if (v810){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v809);
    } else {
    }
    assert("Tensor range check" && 0 <= v808 && v808 < 1l);
    assert("Tensor range check" && 0 <= v807 && v807 < 32l);
    int v812;
    v812 = 4l * v807;
    int v813;
    v813 = 128l * v808;
    int v814;
    v814 = v813 + v812;
    assert("Tensor range check" && 0 <= v808 && v808 < 1l);
    assert("Tensor range check" && 0 <= v807 && v807 < 32l);
    int v815;
    v815 = blockIdx.x;
    int v816;
    v816 = v815;
    while (while_method_2(v816)){
        bool v818;
        v818 = 0l <= v816;
        bool v819;
        v819 = v818 == false;
        if (v819){
            assert("The index needs to be zero or positive." && v818);
        } else {
        }
        bool v821;
        v821 = v816 < 64l;
        bool v822;
        v822 = v821 == false;
        if (v822){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v821);
        } else {
        }
        assert("Tensor range check" && 0 <= v816 && v816 < 64l);
        int v824;
        v824 = 128l * v816;
        int v825;
        v825 = v824 + v814;
        float v826[4l];
        int v827[4l];
        int v828;
        v828 = 0l;
        while (while_method_3(v828)){
            assert("Tensor range check" && 0 <= v828 && v828 < 1l);
            int v830;
            v830 = 4l * v828;
            assert("Tensor range check" && 0 <= v828 && v828 < 1l);
            int v831;
            v831 = 128l * v828;
            int v832;
            v832 = v831 + v825;
            int4* v833;
            v833 = reinterpret_cast<int4*>(v1 + v832);
            int4* v834;
            v834 = reinterpret_cast<int4*>(v826 + v830);
            assert("Pointer alignment check" && (unsigned long long)(v833) % 4l == 0 && (unsigned long long)(v834) % 4l == 0);
            *v834 = *v833;
            v828 += 1l ;
        }
        int v835;
        v835 = 0l;
        while (while_method_3(v835)){
            int v837;
            v837 = 0l;
            while (while_method_1(v837)){
                bool v839;
                v839 = 0l <= v837;
                bool v841;
                if (v839){
                    bool v840;
                    v840 = v837 < 4l;
                    v841 = v840;
                } else {
                    v841 = false;
                }
                bool v842;
                v842 = v841 == false;
                if (v842){
                    assert("The indices should be inside the range of the dimension." && v841);
                } else {
                }
                bool v844;
                v844 = 0l <= v807;
                bool v846;
                if (v844){
                    bool v845;
                    v845 = v807 < 32l;
                    v846 = v845;
                } else {
                    v846 = false;
                }
                bool v847;
                v847 = v846 == false;
                if (v847){
                    assert("The indices should be inside the range of the dimension." && v846);
                } else {
                }
                int v849;
                v849 = v807 * 4l;
                int v850;
                v850 = v837 + v849;
                bool v851;
                v851 = 0l <= v835;
                bool v853;
                if (v851){
                    bool v852;
                    v852 = v835 < 1l;
                    v853 = v852;
                } else {
                    v853 = false;
                }
                bool v854;
                v854 = v853 == false;
                if (v854){
                    assert("The indices should be inside the range of the dimension." && v853);
                } else {
                }
                int v856;
                v856 = v835 * 128l;
                int v857;
                v857 = v850 + v856;
                assert("Tensor range check" && 0 <= v835 && v835 < 1l);
                assert("Tensor range check" && 0 <= v837 && v837 < 4l);
                int v858;
                v858 = 4l * v835;
                int v859;
                v859 = v858 + v837;
                v827[v859] = v857;
                v837 += 1l ;
            }
            v835 += 1l ;
        }
        bool v860;
        v860 = 0l <= v808;
        bool v861;
        v861 = v860 && v809;
        bool v862;
        v862 = v861 == false;
        if (v862){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v861);
        } else {
        }
        bool v864;
        v864 = v818 && v821;
        bool v865;
        v865 = v864 == false;
        if (v865){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v864);
        } else {
        }
        int v867;
        v867 = v816 + v808;
        bool v868[4l];
        int v869;
        v869 = 0l;
        while (while_method_3(v869)){
            int v871;
            v871 = 0l;
            while (while_method_1(v871)){
                assert("Tensor range check" && 0 <= v869 && v869 < 1l);
                assert("Tensor range check" && 0 <= v871 && v871 < 4l);
                int v873;
                v873 = 4l * v869;
                int v874;
                v874 = v873 + v871;
                float v875;
                v875 = v826[v874];
                int v876;
                v876 = v827[v874];
                bool v877;
                v877 = v876 < 4l;
                assert("Tensor range check" && 0 <= v869 && v869 < 1l);
                assert("Tensor range check" && 0 <= v871 && v871 < 4l);
                v868[v874] = v877;
                v871 += 1l ;
            }
            v869 += 1l ;
        }
        int v878[4l];
        int v879;
        v879 = 0l;
        while (while_method_3(v879)){
            int v881;
            v881 = 0l;
            while (while_method_1(v881)){
                assert("Tensor range check" && 0 <= v879 && v879 < 1l);
                assert("Tensor range check" && 0 <= v881 && v881 < 4l);
                int v883;
                v883 = 4l * v879;
                int v884;
                v884 = v883 + v881;
                bool v885;
                v885 = v868[v884];
                int v886;
                if (v885){
                    v886 = 1l;
                } else {
                    v886 = 0l;
                }
                assert("Tensor range check" && 0 <= v879 && v879 < 1l);
                assert("Tensor range check" && 0 <= v881 && v881 < 4l);
                v878[v884] = v886;
                v881 += 1l ;
            }
            v879 += 1l ;
        }
        int v887;
        v887 = 0l;
        int v888;
        v888 = 0l;
        while (while_method_3(v888)){
            int v890;
            v890 = 0l;
            while (while_method_1(v890)){
                assert("Tensor range check" && 0 <= v888 && v888 < 1l);
                assert("Tensor range check" && 0 <= v890 && v890 < 4l);
                int v892;
                v892 = 4l * v888;
                int v893;
                v893 = v892 + v890;
                int v894;
                v894 = v878[v893];
                int v895;
                v895 = v887 + v894;
                v887 = v895;
                v890 += 1l ;
            }
            v888 += 1l ;
        }
        auto v896 = cooperative_groups::coalesced_threads();
        int v897;
        v897 = threadIdx.x;
        int v898;
        v898 = v897 / 32l;
        auto v899 = cooperative_groups::labeled_partition(v896,v898);
        Closure4 v900{};
        int v901;
        v901 = cooperative_groups::reduce(v899, v887, v900);
        float v902[4l];
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
                float v909;
                v909 = v826[v908];
                bool v910;
                v910 = v868[v908];
                float v911;
                if (v910){
                    v911 = v909;
                } else {
                    v911 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v903 && v903 < 1l);
                assert("Tensor range check" && 0 <= v905 && v905 < 4l);
                v902[v908] = v911;
                v905 += 1l ;
            }
            v903 += 1l ;
        }
        float v912;
        v912 = 0.0f;
        int v913;
        v913 = 0l;
        while (while_method_3(v913)){
            int v915;
            v915 = 0l;
            while (while_method_1(v915)){
                assert("Tensor range check" && 0 <= v913 && v913 < 1l);
                assert("Tensor range check" && 0 <= v915 && v915 < 4l);
                int v917;
                v917 = 4l * v913;
                int v918;
                v918 = v917 + v915;
                float v919;
                v919 = v902[v918];
                float v920;
                v920 = v912 + v919;
                v912 = v920;
                v915 += 1l ;
            }
            v913 += 1l ;
        }
        auto v921 = cooperative_groups::coalesced_threads();
        int v922;
        v922 = threadIdx.x;
        int v923;
        v923 = v922 / 32l;
        auto v924 = cooperative_groups::labeled_partition(v921,v923);
        Closure0 v925{};
        float v926;
        v926 = cooperative_groups::reduce(v924, v912, v925);
        float v927;
        v927 = (float)v901;
        float v928;
        v928 = v926 / v927;
        float v929[4l];
        int v930;
        v930 = 0l;
        while (while_method_3(v930)){
            int v932;
            v932 = 0l;
            while (while_method_1(v932)){
                assert("Tensor range check" && 0 <= v930 && v930 < 1l);
                assert("Tensor range check" && 0 <= v932 && v932 < 4l);
                int v934;
                v934 = 4l * v930;
                int v935;
                v935 = v934 + v932;
                float v936;
                v936 = v826[v935];
                bool v937;
                v937 = v868[v935];
                float v938;
                if (v937){
                    v938 = v936;
                } else {
                    v938 = -1.0f / 0.0f;
                }
                float v939;
                v939 = v938 - v928;
                float v940;
                v940 = exp(v939);
                assert("Tensor range check" && 0 <= v930 && v930 < 1l);
                assert("Tensor range check" && 0 <= v932 && v932 < 4l);
                v929[v935] = v940;
                v932 += 1l ;
            }
            v930 += 1l ;
        }
        float v941;
        v941 = 0.0f;
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
                v948 = v929[v947];
                float v949;
                v949 = v941 + v948;
                v941 = v949;
                v944 += 1l ;
            }
            v942 += 1l ;
        }
        auto v950 = cooperative_groups::coalesced_threads();
        int v951;
        v951 = threadIdx.x;
        int v952;
        v952 = v951 / 32l;
        auto v953 = cooperative_groups::labeled_partition(v950,v952);
        float v954;
        v954 = cooperative_groups::reduce(v953, v941, v925);
        float v955[4l];
        int v956;
        v956 = 0l;
        while (while_method_3(v956)){
            int v958;
            v958 = 0l;
            while (while_method_1(v958)){
                assert("Tensor range check" && 0 <= v956 && v956 < 1l);
                assert("Tensor range check" && 0 <= v958 && v958 < 4l);
                int v960;
                v960 = 4l * v956;
                int v961;
                v961 = v960 + v958;
                float v962;
                v962 = v929[v961];
                float v963;
                v963 = v962 / v954;
                assert("Tensor range check" && 0 <= v956 && v956 < 1l);
                assert("Tensor range check" && 0 <= v958 && v958 < 4l);
                v955[v961] = v963;
                v958 += 1l ;
            }
            v956 += 1l ;
        }
        assert("Tensor range check" && 0 <= v816 && v816 < 64l);
        int v964;
        v964 = 0l;
        while (while_method_3(v964)){
            assert("Tensor range check" && 0 <= v964 && v964 < 1l);
            int v966;
            v966 = 128l * v964;
            int v967;
            v967 = v966 + v825;
            assert("Tensor range check" && 0 <= v964 && v964 < 1l);
            int v968;
            v968 = 4l * v964;
            int4* v969;
            v969 = reinterpret_cast<int4*>(v955 + v968);
            int4* v970;
            v970 = reinterpret_cast<int4*>(v4 + v967);
            assert("Pointer alignment check" && (unsigned long long)(v969) % 4l == 0 && (unsigned long long)(v970) % 4l == 0);
            *v970 = *v969;
            v964 += 1l ;
        }
        v816 += 1l ;
    }
    v15.sync() ;
    int v971;
    v971 = threadIdx.x;
    unsigned long long v972;
    v972 = (unsigned long long)v971;
    curandStatePhilox4_32_10_t v973;
    curand_init(12344321ull,v972,0ull,&v973);
    int v974;
    v974 = threadIdx.x;
    bool v975;
    v975 = 0l <= v974;
    bool v976;
    v976 = v975 == false;
    if (v976){
        assert("The index needs to be zero or positive." && v975);
    } else {
    }
    int v978;
    v978 = v974 % 32l;
    int v979;
    v979 = v974 / 32l;
    bool v980;
    v980 = v979 < 1l;
    bool v981;
    v981 = v980 == false;
    if (v981){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v980);
    } else {
    }
    assert("Tensor range check" && 0 <= v979 && v979 < 1l);
    assert("Tensor range check" && 0 <= v978 && v978 < 32l);
    int v983;
    v983 = 4l * v978;
    int v984;
    v984 = 128l * v979;
    int v985;
    v985 = v984 + v983;
    assert("Tensor range check" && 0 <= v979 && v979 < 1l);
    assert("Tensor range check" && 0 <= v978 && v978 < 32l);
    assert("Tensor range check" && 0 <= v979 && v979 < 1l);
    int v986;
    v986 = blockIdx.x;
    int v987;
    v987 = v986;
    while (while_method_2(v987)){
        bool v989;
        v989 = 0l <= v987;
        bool v990;
        v990 = v989 == false;
        if (v990){
            assert("The index needs to be zero or positive." && v989);
        } else {
        }
        bool v992;
        v992 = v987 < 64l;
        bool v993;
        v993 = v992 == false;
        if (v993){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v992);
        } else {
        }
        assert("Tensor range check" && 0 <= v987 && v987 < 64l);
        int v995;
        v995 = 128l * v987;
        int v996;
        v996 = v995 + v985;
        float v997[4l];
        int v998[4l];
        int v999;
        v999 = 0l;
        while (while_method_3(v999)){
            assert("Tensor range check" && 0 <= v999 && v999 < 1l);
            int v1001;
            v1001 = 4l * v999;
            assert("Tensor range check" && 0 <= v999 && v999 < 1l);
            int v1002;
            v1002 = 128l * v999;
            int v1003;
            v1003 = v1002 + v996;
            int4* v1004;
            v1004 = reinterpret_cast<int4*>(v1 + v1003);
            int4* v1005;
            v1005 = reinterpret_cast<int4*>(v997 + v1001);
            assert("Pointer alignment check" && (unsigned long long)(v1004) % 4l == 0 && (unsigned long long)(v1005) % 4l == 0);
            *v1005 = *v1004;
            v999 += 1l ;
        }
        int v1006;
        v1006 = 0l;
        while (while_method_3(v1006)){
            int v1008;
            v1008 = 0l;
            while (while_method_1(v1008)){
                bool v1010;
                v1010 = 0l <= v1008;
                bool v1012;
                if (v1010){
                    bool v1011;
                    v1011 = v1008 < 4l;
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
                bool v1015;
                v1015 = 0l <= v978;
                bool v1017;
                if (v1015){
                    bool v1016;
                    v1016 = v978 < 32l;
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
                int v1020;
                v1020 = v978 * 4l;
                int v1021;
                v1021 = v1008 + v1020;
                bool v1022;
                v1022 = 0l <= v1006;
                bool v1024;
                if (v1022){
                    bool v1023;
                    v1023 = v1006 < 1l;
                    v1024 = v1023;
                } else {
                    v1024 = false;
                }
                bool v1025;
                v1025 = v1024 == false;
                if (v1025){
                    assert("The indices should be inside the range of the dimension." && v1024);
                } else {
                }
                int v1027;
                v1027 = v1006 * 128l;
                int v1028;
                v1028 = v1021 + v1027;
                assert("Tensor range check" && 0 <= v1006 && v1006 < 1l);
                assert("Tensor range check" && 0 <= v1008 && v1008 < 4l);
                int v1029;
                v1029 = 4l * v1006;
                int v1030;
                v1030 = v1029 + v1008;
                v998[v1030] = v1028;
                v1008 += 1l ;
            }
            v1006 += 1l ;
        }
        bool v1031;
        v1031 = 0l <= v979;
        bool v1032;
        v1032 = v1031 && v980;
        bool v1033;
        v1033 = v1032 == false;
        if (v1033){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1032);
        } else {
        }
        bool v1035;
        v1035 = v989 && v992;
        bool v1036;
        v1036 = v1035 == false;
        if (v1036){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1035);
        } else {
        }
        int v1038;
        v1038 = v987 + v979;
        float v1039;
        v1039 = 0.0f;
        int v1040;
        v1040 = 0l;
        while (while_method_3(v1040)){
            int v1042;
            v1042 = 0l;
            while (while_method_1(v1042)){
                assert("Tensor range check" && 0 <= v1040 && v1040 < 1l);
                assert("Tensor range check" && 0 <= v1042 && v1042 < 4l);
                int v1044;
                v1044 = 4l * v1040;
                int v1045;
                v1045 = v1044 + v1042;
                float v1046;
                v1046 = v997[v1045];
                float v1047;
                v1047 = v1039 + v1046;
                v1039 = v1047;
                v1042 += 1l ;
            }
            v1040 += 1l ;
        }
        auto v1048 = cooperative_groups::coalesced_threads();
        int v1049;
        v1049 = threadIdx.x;
        int v1050;
        v1050 = v1049 / 32l;
        auto v1051 = cooperative_groups::labeled_partition(v1048,v1050);
        Closure0 v1052{};
        float v1053;
        v1053 = cooperative_groups::reduce(v1051, v1039, v1052);
        float v1054;
        v1054 = v1053 / 128.0f;
        float v1055[4l];
        int v1056;
        v1056 = 0l;
        while (while_method_3(v1056)){
            int v1058;
            v1058 = 0l;
            while (while_method_1(v1058)){
                assert("Tensor range check" && 0 <= v1056 && v1056 < 1l);
                assert("Tensor range check" && 0 <= v1058 && v1058 < 4l);
                int v1060;
                v1060 = 4l * v1056;
                int v1061;
                v1061 = v1060 + v1058;
                float v1062;
                v1062 = v997[v1061];
                float v1063;
                v1063 = v1062 - v1054;
                float v1064;
                v1064 = exp(v1063);
                assert("Tensor range check" && 0 <= v1056 && v1056 < 1l);
                assert("Tensor range check" && 0 <= v1058 && v1058 < 4l);
                v1055[v1061] = v1064;
                v1058 += 1l ;
            }
            v1056 += 1l ;
        }
        float v1065;
        v1065 = 0.0f;
        int v1066;
        v1066 = 0l;
        while (while_method_3(v1066)){
            int v1068;
            v1068 = 0l;
            while (while_method_1(v1068)){
                assert("Tensor range check" && 0 <= v1066 && v1066 < 1l);
                assert("Tensor range check" && 0 <= v1068 && v1068 < 4l);
                int v1070;
                v1070 = 4l * v1066;
                int v1071;
                v1071 = v1070 + v1068;
                float v1072;
                v1072 = v1055[v1071];
                float v1073;
                v1073 = v1065 + v1072;
                v1065 = v1073;
                v1068 += 1l ;
            }
            v1066 += 1l ;
        }
        auto v1074 = cooperative_groups::coalesced_threads();
        int v1075;
        v1075 = threadIdx.x;
        int v1076;
        v1076 = v1075 / 32l;
        auto v1077 = cooperative_groups::labeled_partition(v1074,v1076);
        float v1078;
        v1078 = cooperative_groups::reduce(v1077, v1065, v1052);
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
                v1086 = v1055[v1085];
                float v1087;
                v1087 = v1086 / v1078;
                assert("Tensor range check" && 0 <= v1080 && v1080 < 1l);
                assert("Tensor range check" && 0 <= v1082 && v1082 < 4l);
                v1079[v1085] = v1087;
                v1082 += 1l ;
            }
            v1080 += 1l ;
        }
        float v1088[4l];
        float v1089;
        v1089 = 0.0f;
        int v1090;
        v1090 = 0l;
        while (while_method_3(v1090)){
            assert("Tensor range check" && 0 <= v1090 && v1090 < 1l);
            int v1092;
            v1092 = 4l * v1090;
            assert("Tensor range check" && 0 <= v1090 && v1090 < 1l);
            int v1093; float v1094;
            Tuple0 tmp90 = Tuple0{0l, 0.0f};
            v1093 = tmp90.v0; v1094 = tmp90.v1;
            while (while_method_1(v1093)){
                assert("Tensor range check" && 0 <= v1093 && v1093 < 4l);
                int v1096;
                v1096 = v1093 + v1092;
                float v1097;
                v1097 = v1079[v1096];
                float v1098;
                v1098 = v1094 + v1097;
                v1094 = v1098;
                v1093 += 1l ;
            }
            auto v1099 = cooperative_groups::coalesced_threads();
            int v1100;
            v1100 = threadIdx.x;
            int v1101;
            v1101 = v1100 / 32l;
            auto v1102 = cooperative_groups::labeled_partition(v1099,v1101);
            Closure2 v1103{};
            float v1104;
            v1104 = cooperative_groups::inclusive_scan(v1102, v1094, v1103);
            float v1105;
            v1105 = v1102.shfl_up(v1104,1);
            bool v1106;
            v1106 = v1102.thread_rank() == 0;
            float v1107;
            if (v1106){
                v1107 = 0.0f;
            } else {
                v1107 = v1105;
            }
            float v1108;
            v1108 = v1102.shfl(v1104,v1102.num_threads()-1);
            float v1109;
            v1109 = v1089 + v1107;
            int v1110; float v1111;
            Tuple0 tmp91 = Tuple0{0l, v1109};
            v1110 = tmp91.v0; v1111 = tmp91.v1;
            while (while_method_1(v1110)){
                assert("Tensor range check" && 0 <= v1110 && v1110 < 4l);
                int v1113;
                v1113 = v1110 + v1092;
                float v1114;
                v1114 = v1079[v1113];
                float v1115;
                v1115 = v1111 + v1114;
                assert("Tensor range check" && 0 <= v1110 && v1110 < 4l);
                v1088[v1113] = v1115;
                v1111 = v1115;
                v1110 += 1l ;
            }
            float v1116;
            v1116 = v1089 + v1108;
            v1089 = v1116;
            v1090 += 1l ;
        }
        float v1117[4l];
        bool v1118[4l];
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
                v1125 = v1088[v1124];
                float v1126;
                v1126 = v1079[v1124];
                bool v1127;
                v1127 = v1126 > 0.0f;
                assert("Tensor range check" && 0 <= v1119 && v1119 < 1l);
                assert("Tensor range check" && 0 <= v1121 && v1121 < 4l);
                v1117[v1124] = v1125;
                v1118[v1124] = v1127;
                v1121 += 1l ;
            }
            v1119 += 1l ;
        }
        float v1128; bool v1129;
        Tuple3 tmp92 = Tuple3{-1.0f / 0.0f, false};
        v1128 = tmp92.v0; v1129 = tmp92.v1;
        int v1130;
        v1130 = 0l;
        while (while_method_3(v1130)){
            int v1132;
            v1132 = 0l;
            while (while_method_1(v1132)){
                assert("Tensor range check" && 0 <= v1130 && v1130 < 1l);
                assert("Tensor range check" && 0 <= v1132 && v1132 < 4l);
                int v1134;
                v1134 = 4l * v1130;
                int v1135;
                v1135 = v1134 + v1132;
                float v1136;
                v1136 = v1117[v1135];
                bool v1137;
                v1137 = v1118[v1135];
                float v1144; bool v1145;
                if (v1129){
                    if (v1137){
                        bool v1138;
                        v1138 = v1128 >= v1136;
                        float v1139;
                        if (v1138){
                            v1139 = v1128;
                        } else {
                            v1139 = v1136;
                        }
                        v1144 = v1139; v1145 = true;
                    } else {
                        v1144 = v1128; v1145 = v1129;
                    }
                } else {
                    if (v1137){
                        v1144 = v1136; v1145 = v1137;
                    } else {
                        v1144 = v1128; v1145 = v1129;
                    }
                }
                v1128 = v1144;
                v1129 = v1145;
                v1132 += 1l ;
            }
            v1130 += 1l ;
        }
        auto v1146 = cooperative_groups::coalesced_threads();
        int v1147;
        v1147 = threadIdx.x;
        int v1148;
        v1148 = v1147 / 32l;
        auto v1149 = cooperative_groups::labeled_partition(v1146,v1148);
        Closure5 v1150{};
        float v1151; bool v1152;
        Tuple3 tmp93 = cooperative_groups::reduce(v1149, Tuple3{v1128, v1129}, v1150);
        v1151 = tmp93.v0; v1152 = tmp93.v1;
        bool v1153;
        v1153 = v1152 == false;
        if (v1153){
            assert("The local reduce must be true." && v1152);
        } else {
        }
        float v1155[4l];
        int v1156[4l];
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
                int v1163;
                v1163 = v998[v1162];
                float v1164;
                v1164 = curand_uniform(&v973);
                assert("Tensor range check" && 0 <= v1157 && v1157 < 1l);
                assert("Tensor range check" && 0 <= v1159 && v1159 < 4l);
                v1155[v1162] = v1164;
                v1156[v1162] = v1163;
                v1159 += 1l ;
            }
            v1157 += 1l ;
        }
        float v1165; int v1166;
        Tuple1 tmp94 = Tuple1{0.0f, 2147483647l};
        v1165 = tmp94.v0; v1166 = tmp94.v1;
        int v1167;
        v1167 = 0l;
        while (while_method_3(v1167)){
            int v1169;
            v1169 = 0l;
            while (while_method_1(v1169)){
                assert("Tensor range check" && 0 <= v1167 && v1167 < 1l);
                assert("Tensor range check" && 0 <= v1169 && v1169 < 4l);
                int v1171;
                v1171 = 4l * v1167;
                int v1172;
                v1172 = v1171 + v1169;
                float v1173;
                v1173 = v1155[v1172];
                int v1174;
                v1174 = v1156[v1172];
                bool v1175;
                v1175 = v1166 < v1174;
                float v1176; int v1177;
                if (v1175){
                    v1176 = v1165; v1177 = v1166;
                } else {
                    v1176 = v1173; v1177 = v1174;
                }
                v1165 = v1176;
                v1166 = v1177;
                v1169 += 1l ;
            }
            v1167 += 1l ;
        }
        auto v1178 = cooperative_groups::coalesced_threads();
        int v1179;
        v1179 = threadIdx.x;
        int v1180;
        v1180 = v1179 / 32l;
        auto v1181 = cooperative_groups::labeled_partition(v1178,v1180);
        Closure6 v1182{};
        float v1183; int v1184;
        Tuple1 tmp95 = cooperative_groups::reduce(v1181, Tuple1{v1165, v1166}, v1182);
        v1183 = tmp95.v0; v1184 = tmp95.v1;
        float v1185;
        v1185 = v1151 * v1183;
        int v1186[4l];
        bool v1187[4l];
        int v1188;
        v1188 = 0l;
        while (while_method_3(v1188)){
            int v1190;
            v1190 = 0l;
            while (while_method_1(v1190)){
                assert("Tensor range check" && 0 <= v1188 && v1188 < 1l);
                assert("Tensor range check" && 0 <= v1190 && v1190 < 4l);
                int v1192;
                v1192 = 4l * v1188;
                int v1193;
                v1193 = v1192 + v1190;
                float v1194;
                v1194 = v1117[v1193];
                bool v1195;
                v1195 = v1118[v1193];
                int v1196;
                v1196 = v998[v1193];
                int v1199; bool v1200;
                if (v1195){
                    float v1197;
                    v1197 = v1194 - v1185;
                    bool v1198;
                    v1198 = v1197 >= 0.0f;
                    v1199 = v1196; v1200 = v1198;
                } else {
                    v1199 = 2147483647l; v1200 = false;
                }
                assert("Tensor range check" && 0 <= v1188 && v1188 < 1l);
                assert("Tensor range check" && 0 <= v1190 && v1190 < 4l);
                v1186[v1193] = v1199;
                v1187[v1193] = v1200;
                v1190 += 1l ;
            }
            v1188 += 1l ;
        }
        int v1201; bool v1202;
        Tuple4 tmp96 = Tuple4{2147483647l, false};
        v1201 = tmp96.v0; v1202 = tmp96.v1;
        int v1203;
        v1203 = 0l;
        while (while_method_3(v1203)){
            int v1205;
            v1205 = 0l;
            while (while_method_1(v1205)){
                assert("Tensor range check" && 0 <= v1203 && v1203 < 1l);
                assert("Tensor range check" && 0 <= v1205 && v1205 < 4l);
                int v1207;
                v1207 = 4l * v1203;
                int v1208;
                v1208 = v1207 + v1205;
                int v1209;
                v1209 = v1186[v1208];
                bool v1210;
                v1210 = v1187[v1208];
                int v1217; bool v1218;
                if (v1202){
                    if (v1210){
                        bool v1211;
                        v1211 = v1201 < v1209;
                        int v1212;
                        if (v1211){
                            v1212 = v1201;
                        } else {
                            v1212 = v1209;
                        }
                        v1217 = v1212; v1218 = true;
                    } else {
                        v1217 = v1201; v1218 = v1202;
                    }
                } else {
                    if (v1210){
                        v1217 = v1209; v1218 = v1210;
                    } else {
                        v1217 = v1201; v1218 = v1202;
                    }
                }
                v1201 = v1217;
                v1202 = v1218;
                v1205 += 1l ;
            }
            v1203 += 1l ;
        }
        auto v1219 = cooperative_groups::coalesced_threads();
        int v1220;
        v1220 = threadIdx.x;
        int v1221;
        v1221 = v1220 / 32l;
        auto v1222 = cooperative_groups::labeled_partition(v1219,v1221);
        Closure7 v1223{};
        int v1224; bool v1225;
        Tuple4 tmp97 = cooperative_groups::reduce(v1222, Tuple4{v1201, v1202}, v1223);
        v1224 = tmp97.v0; v1225 = tmp97.v1;
        bool v1226;
        v1226 = v1225 == false;
        if (v1226){
            assert("The local reduce must be true." && v1225);
        } else {
        }
        assert("Tensor range check" && 0 <= v987 && v987 < 64l);
        int v1228;
        v1228 = 0l;
        while (while_method_3(v1228)){
            assert("Tensor range check" && 0 <= v1228 && v1228 < 1l);
            int v1230;
            v1230 = 128l * v1228;
            int v1231;
            v1231 = v1230 + v996;
            assert("Tensor range check" && 0 <= v1228 && v1228 < 1l);
            int v1232;
            v1232 = 4l * v1228;
            int4* v1233;
            v1233 = reinterpret_cast<int4*>(v1079 + v1232);
            int4* v1234;
            v1234 = reinterpret_cast<int4*>(v13 + v1231);
            assert("Pointer alignment check" && (unsigned long long)(v1233) % 4l == 0 && (unsigned long long)(v1234) % 4l == 0);
            *v1234 = *v1233;
            v1228 += 1l ;
        }
        assert("Tensor range check" && 0 <= v987 && v987 < 64l);
        v14[v1038] = v1224;
        v987 += 1l ;
    }
    v15.sync() ;
    int v1235;
    v1235 = threadIdx.x;
    unsigned long long v1236;
    v1236 = (unsigned long long)v1235;
    curandStatePhilox4_32_10_t v1237;
    curand_init(12344321ull,v1236,0ull,&v1237);
    int v1238;
    v1238 = threadIdx.x;
    bool v1239;
    v1239 = 0l <= v1238;
    bool v1240;
    v1240 = v1239 == false;
    if (v1240){
        assert("The index needs to be zero or positive." && v1239);
    } else {
    }
    int v1242;
    v1242 = v1238 % 32l;
    int v1243;
    v1243 = v1238 / 32l;
    bool v1244;
    v1244 = v1243 < 1l;
    bool v1245;
    v1245 = v1244 == false;
    if (v1245){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1244);
    } else {
    }
    assert("Tensor range check" && 0 <= v1243 && v1243 < 1l);
    assert("Tensor range check" && 0 <= v1242 && v1242 < 32l);
    int v1247;
    v1247 = 4l * v1242;
    int v1248;
    v1248 = 128l * v1243;
    int v1249;
    v1249 = v1248 + v1247;
    assert("Tensor range check" && 0 <= v1243 && v1243 < 1l);
    assert("Tensor range check" && 0 <= v1242 && v1242 < 32l);
    assert("Tensor range check" && 0 <= v1243 && v1243 < 1l);
    int v1250;
    v1250 = blockIdx.x;
    int v1251;
    v1251 = v1250;
    while (while_method_2(v1251)){
        bool v1253;
        v1253 = 0l <= v1251;
        bool v1254;
        v1254 = v1253 == false;
        if (v1254){
            assert("The index needs to be zero or positive." && v1253);
        } else {
        }
        bool v1256;
        v1256 = v1251 < 64l;
        bool v1257;
        v1257 = v1256 == false;
        if (v1257){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1256);
        } else {
        }
        assert("Tensor range check" && 0 <= v1251 && v1251 < 64l);
        int v1259;
        v1259 = 128l * v1251;
        int v1260;
        v1260 = v1259 + v1249;
        float v1261[4l];
        int v1262[4l];
        int v1263;
        v1263 = 0l;
        while (while_method_3(v1263)){
            assert("Tensor range check" && 0 <= v1263 && v1263 < 1l);
            int v1265;
            v1265 = 4l * v1263;
            assert("Tensor range check" && 0 <= v1263 && v1263 < 1l);
            int v1266;
            v1266 = 128l * v1263;
            int v1267;
            v1267 = v1266 + v1260;
            int4* v1268;
            v1268 = reinterpret_cast<int4*>(v1 + v1267);
            int4* v1269;
            v1269 = reinterpret_cast<int4*>(v1261 + v1265);
            assert("Pointer alignment check" && (unsigned long long)(v1268) % 4l == 0 && (unsigned long long)(v1269) % 4l == 0);
            *v1269 = *v1268;
            v1263 += 1l ;
        }
        int v1270;
        v1270 = 0l;
        while (while_method_3(v1270)){
            int v1272;
            v1272 = 0l;
            while (while_method_1(v1272)){
                bool v1274;
                v1274 = 0l <= v1272;
                bool v1276;
                if (v1274){
                    bool v1275;
                    v1275 = v1272 < 4l;
                    v1276 = v1275;
                } else {
                    v1276 = false;
                }
                bool v1277;
                v1277 = v1276 == false;
                if (v1277){
                    assert("The indices should be inside the range of the dimension." && v1276);
                } else {
                }
                bool v1279;
                v1279 = 0l <= v1242;
                bool v1281;
                if (v1279){
                    bool v1280;
                    v1280 = v1242 < 32l;
                    v1281 = v1280;
                } else {
                    v1281 = false;
                }
                bool v1282;
                v1282 = v1281 == false;
                if (v1282){
                    assert("The indices should be inside the range of the dimension." && v1281);
                } else {
                }
                int v1284;
                v1284 = v1242 * 4l;
                int v1285;
                v1285 = v1272 + v1284;
                bool v1286;
                v1286 = 0l <= v1270;
                bool v1288;
                if (v1286){
                    bool v1287;
                    v1287 = v1270 < 1l;
                    v1288 = v1287;
                } else {
                    v1288 = false;
                }
                bool v1289;
                v1289 = v1288 == false;
                if (v1289){
                    assert("The indices should be inside the range of the dimension." && v1288);
                } else {
                }
                int v1291;
                v1291 = v1270 * 128l;
                int v1292;
                v1292 = v1285 + v1291;
                assert("Tensor range check" && 0 <= v1270 && v1270 < 1l);
                assert("Tensor range check" && 0 <= v1272 && v1272 < 4l);
                int v1293;
                v1293 = 4l * v1270;
                int v1294;
                v1294 = v1293 + v1272;
                v1262[v1294] = v1292;
                v1272 += 1l ;
            }
            v1270 += 1l ;
        }
        bool v1295;
        v1295 = 0l <= v1243;
        bool v1296;
        v1296 = v1295 && v1244;
        bool v1297;
        v1297 = v1296 == false;
        if (v1297){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1296);
        } else {
        }
        bool v1299;
        v1299 = v1253 && v1256;
        bool v1300;
        v1300 = v1299 == false;
        if (v1300){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1299);
        } else {
        }
        int v1302;
        v1302 = v1251 + v1243;
        bool v1303[4l];
        int v1304;
        v1304 = 0l;
        while (while_method_3(v1304)){
            int v1306;
            v1306 = 0l;
            while (while_method_1(v1306)){
                assert("Tensor range check" && 0 <= v1304 && v1304 < 1l);
                assert("Tensor range check" && 0 <= v1306 && v1306 < 4l);
                int v1308;
                v1308 = 4l * v1304;
                int v1309;
                v1309 = v1308 + v1306;
                float v1310;
                v1310 = v1261[v1309];
                int v1311;
                v1311 = v1262[v1309];
                bool v1312;
                v1312 = v1311 < 11l;
                assert("Tensor range check" && 0 <= v1304 && v1304 < 1l);
                assert("Tensor range check" && 0 <= v1306 && v1306 < 4l);
                v1303[v1309] = v1312;
                v1306 += 1l ;
            }
            v1304 += 1l ;
        }
        int v1313[4l];
        int v1314;
        v1314 = 0l;
        while (while_method_3(v1314)){
            int v1316;
            v1316 = 0l;
            while (while_method_1(v1316)){
                assert("Tensor range check" && 0 <= v1314 && v1314 < 1l);
                assert("Tensor range check" && 0 <= v1316 && v1316 < 4l);
                int v1318;
                v1318 = 4l * v1314;
                int v1319;
                v1319 = v1318 + v1316;
                bool v1320;
                v1320 = v1303[v1319];
                int v1321;
                if (v1320){
                    v1321 = 1l;
                } else {
                    v1321 = 0l;
                }
                assert("Tensor range check" && 0 <= v1314 && v1314 < 1l);
                assert("Tensor range check" && 0 <= v1316 && v1316 < 4l);
                v1313[v1319] = v1321;
                v1316 += 1l ;
            }
            v1314 += 1l ;
        }
        int v1322;
        v1322 = 0l;
        int v1323;
        v1323 = 0l;
        while (while_method_3(v1323)){
            int v1325;
            v1325 = 0l;
            while (while_method_1(v1325)){
                assert("Tensor range check" && 0 <= v1323 && v1323 < 1l);
                assert("Tensor range check" && 0 <= v1325 && v1325 < 4l);
                int v1327;
                v1327 = 4l * v1323;
                int v1328;
                v1328 = v1327 + v1325;
                int v1329;
                v1329 = v1313[v1328];
                int v1330;
                v1330 = v1322 + v1329;
                v1322 = v1330;
                v1325 += 1l ;
            }
            v1323 += 1l ;
        }
        auto v1331 = cooperative_groups::coalesced_threads();
        int v1332;
        v1332 = threadIdx.x;
        int v1333;
        v1333 = v1332 / 32l;
        auto v1334 = cooperative_groups::labeled_partition(v1331,v1333);
        Closure4 v1335{};
        int v1336;
        v1336 = cooperative_groups::reduce(v1334, v1322, v1335);
        float v1337[4l];
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
                float v1344;
                v1344 = v1261[v1343];
                bool v1345;
                v1345 = v1303[v1343];
                float v1346;
                if (v1345){
                    v1346 = v1344;
                } else {
                    v1346 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1338 && v1338 < 1l);
                assert("Tensor range check" && 0 <= v1340 && v1340 < 4l);
                v1337[v1343] = v1346;
                v1340 += 1l ;
            }
            v1338 += 1l ;
        }
        float v1347;
        v1347 = 0.0f;
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
                float v1354;
                v1354 = v1337[v1353];
                float v1355;
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
        v1358 = v1357 / 32l;
        auto v1359 = cooperative_groups::labeled_partition(v1356,v1358);
        Closure0 v1360{};
        float v1361;
        v1361 = cooperative_groups::reduce(v1359, v1347, v1360);
        float v1362;
        v1362 = (float)v1336;
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
                v1371 = v1261[v1370];
                bool v1372;
                v1372 = v1303[v1370];
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
        v1389 = cooperative_groups::reduce(v1388, v1376, v1360);
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
            Tuple0 tmp98 = Tuple0{0l, 0.0f};
            v1404 = tmp98.v0; v1405 = tmp98.v1;
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
            Tuple0 tmp99 = Tuple0{0l, v1420};
            v1421 = tmp99.v0; v1422 = tmp99.v1;
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
        Tuple3 tmp100 = Tuple3{-1.0f / 0.0f, false};
        v1439 = tmp100.v0; v1440 = tmp100.v1;
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
        Tuple3 tmp101 = cooperative_groups::reduce(v1460, Tuple3{v1439, v1440}, v1461);
        v1462 = tmp101.v0; v1463 = tmp101.v1;
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
                v1474 = v1262[v1473];
                float v1475;
                v1475 = curand_uniform(&v1237);
                assert("Tensor range check" && 0 <= v1468 && v1468 < 1l);
                assert("Tensor range check" && 0 <= v1470 && v1470 < 4l);
                v1466[v1473] = v1475;
                v1467[v1473] = v1474;
                v1470 += 1l ;
            }
            v1468 += 1l ;
        }
        float v1476; int v1477;
        Tuple1 tmp102 = Tuple1{0.0f, 2147483647l};
        v1476 = tmp102.v0; v1477 = tmp102.v1;
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
        Tuple1 tmp103 = cooperative_groups::reduce(v1492, Tuple1{v1476, v1477}, v1493);
        v1494 = tmp103.v0; v1495 = tmp103.v1;
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
                v1507 = v1262[v1504];
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
        Tuple4 tmp104 = Tuple4{2147483647l, false};
        v1512 = tmp104.v0; v1513 = tmp104.v1;
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
        Tuple4 tmp105 = cooperative_groups::reduce(v1533, Tuple4{v1512, v1513}, v1534);
        v1535 = tmp105.v0; v1536 = tmp105.v1;
        bool v1537;
        v1537 = v1536 == false;
        if (v1537){
            assert("The local reduce must be true." && v1536);
        } else {
        }
        assert("Tensor range check" && 0 <= v1251 && v1251 < 64l);
        int v1539;
        v1539 = 0l;
        while (while_method_3(v1539)){
            assert("Tensor range check" && 0 <= v1539 && v1539 < 1l);
            int v1541;
            v1541 = 128l * v1539;
            int v1542;
            v1542 = v1541 + v1260;
            assert("Tensor range check" && 0 <= v1539 && v1539 < 1l);
            int v1543;
            v1543 = 4l * v1539;
            int4* v1544;
            v1544 = reinterpret_cast<int4*>(v1390 + v1543);
            int4* v1545;
            v1545 = reinterpret_cast<int4*>(v13 + v1542);
            assert("Pointer alignment check" && (unsigned long long)(v1544) % 4l == 0 && (unsigned long long)(v1545) % 4l == 0);
            *v1545 = *v1544;
            v1539 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1251 && v1251 < 64l);
        v14[v1302] = v1535;
        v1251 += 1l ;
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
    v25.max_dynamic_shared_size_bytes = 81920 
    v25((1,),(32,),(v0, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19),shared_mem=81920)
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
    v51.max_dynamic_shared_size_bytes = 81920 
    v51((1,),(32,),(v26, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45),shared_mem=81920)
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
    v69.max_dynamic_shared_size_bytes = 81920 
    v69((1,),(32,),(v52, v57, v58, v59, v60, v61, v62, v63),shared_mem=81920)
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
    v87.max_dynamic_shared_size_bytes = 81920 
    v87((1,),(32,),(v70, v75, v76, v77, v78, v79, v80, v81),shared_mem=81920)
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
    v112.max_dynamic_shared_size_bytes = 81920 
    v112((1,),(32,),(v88, v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106),shared_mem=81920)
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
    v137.max_dynamic_shared_size_bytes = 81920 
    v137((1,),(32,),(v113, v118, v119, v120, v121, v122, v123, v124, v125, v126, v127, v128, v129, v130, v131),shared_mem=81920)
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
