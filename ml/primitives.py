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
struct Closure0 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple0 {
    float v0;
    int v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure1 {
    __device__ Tuple0 operator()(Tuple0 tup0, Tuple0 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v0 > v2;
        if (v4){
            return Tuple0{v0, v1};
        } else {
            return Tuple0{v2, v3};
        }
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
    __device__ Tuple0 operator()(Tuple0 tup0, Tuple0 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple0{v0, v1};
        } else {
            return Tuple0{v2, v3};
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
    v1 = v0 < 64l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 2048l;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
extern "C" __global__ void entry0(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14) {
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
    v26 = 4096l * v21;
    int v27;
    v27 = v26 + v25;
    assert("Tensor range check" && 0 <= v21 && v21 < 1l);
    assert("Tensor range check" && 0 <= v20 && v20 < 32l);
    int v28;
    v28 = blockIdx.x;
    int v29;
    v29 = v28;
    while (while_method_0(v29)){
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
        v37 = 4096l * v29;
        int v38;
        v38 = v37 + v27;
        int v39[128l];
        int v40[128l];
        int v41;
        v41 = 0l;
        while (while_method_1(v41)){
            assert("Tensor range check" && 0 <= v41 && v41 < 32l);
            int v43;
            v43 = 4l * v41;
            assert("Tensor range check" && 0 <= v41 && v41 < 32l);
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
        while (while_method_1(v48)){
            int v50;
            v50 = 0l;
            while (while_method_2(v50)){
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
                    v65 = v48 < 32l;
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
                assert("Tensor range check" && 0 <= v48 && v48 < 32l);
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
        while (while_method_1(v81)){
            assert("Tensor range check" && 0 <= v81 && v81 < 32l);
            int v83;
            v83 = 128l * v81;
            int v84;
            v84 = v83 + v38;
            assert("Tensor range check" && 0 <= v81 && v81 < 32l);
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
        v29 += 24l ;
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
    v98 = 4096l * v93;
    int v99;
    v99 = v98 + v97;
    assert("Tensor range check" && 0 <= v93 && v93 < 1l);
    assert("Tensor range check" && 0 <= v92 && v92 < 32l);
    int v100;
    v100 = blockIdx.x;
    int v101;
    v101 = v100;
    while (while_method_0(v101)){
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
        v109 = 4096l * v101;
        int v110;
        v110 = v109 + v99;
        float v111[128l];
        int v112[128l];
        int v113;
        v113 = 0l;
        while (while_method_1(v113)){
            assert("Tensor range check" && 0 <= v113 && v113 < 32l);
            int v115;
            v115 = 4l * v113;
            assert("Tensor range check" && 0 <= v113 && v113 < 32l);
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
        while (while_method_1(v120)){
            int v122;
            v122 = 0l;
            while (while_method_2(v122)){
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
                    v137 = v120 < 32l;
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
                assert("Tensor range check" && 0 <= v120 && v120 < 32l);
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
        int v153[128l];
        int v154[128l];
        int v155;
        v155 = 0l;
        while (while_method_1(v155)){
            int v157;
            v157 = 0l;
            while (while_method_2(v157)){
                assert("Tensor range check" && 0 <= v155 && v155 < 32l);
                assert("Tensor range check" && 0 <= v157 && v157 < 4l);
                int v159;
                v159 = 4l * v155;
                int v160;
                v160 = v159 + v157;
                int v161;
                v161 = v112[v160];
                assert("Tensor range check" && 0 <= v155 && v155 < 32l);
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
        while (while_method_1(v162)){
            assert("Tensor range check" && 0 <= v162 && v162 < 32l);
            int v164;
            v164 = 128l * v162;
            int v165;
            v165 = v164 + v110;
            assert("Tensor range check" && 0 <= v162 && v162 < 32l);
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
        v101 += 24l ;
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
    v181 = 4096l * v176;
    int v182;
    v182 = v181 + v180;
    assert("Tensor range check" && 0 <= v176 && v176 < 1l);
    int v183;
    v183 = blockIdx.x;
    int v184;
    v184 = v183;
    while (while_method_0(v184)){
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
        v192 = 4096l * v184;
        int v193;
        v193 = v192 + v182;
        float v194[128l];
        int v195[128l];
        int v196;
        v196 = 0l;
        while (while_method_1(v196)){
            assert("Tensor range check" && 0 <= v196 && v196 < 32l);
            int v198;
            v198 = 4l * v196;
            assert("Tensor range check" && 0 <= v196 && v196 < 32l);
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
        while (while_method_1(v203)){
            int v205;
            v205 = 0l;
            while (while_method_2(v205)){
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
                    v220 = v203 < 32l;
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
                assert("Tensor range check" && 0 <= v203 && v203 < 32l);
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
        v184 += 24l ;
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
    v246 = 4096l * v241;
    int v247;
    v247 = v246 + v245;
    assert("Tensor range check" && 0 <= v241 && v241 < 1l);
    assert("Tensor range check" && 0 <= v240 && v240 < 32l);
    int v248;
    v248 = blockIdx.x;
    int v249;
    v249 = v248;
    while (while_method_0(v249)){
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
        v257 = 4096l * v249;
        int v258;
        v258 = v257 + v247;
        float v259[128l];
        int v260[128l];
        int v261;
        v261 = 0l;
        while (while_method_1(v261)){
            assert("Tensor range check" && 0 <= v261 && v261 < 32l);
            int v263;
            v263 = 4l * v261;
            assert("Tensor range check" && 0 <= v261 && v261 < 32l);
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
        while (while_method_1(v268)){
            int v270;
            v270 = 0l;
            while (while_method_2(v270)){
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
                    v285 = v268 < 32l;
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
                assert("Tensor range check" && 0 <= v268 && v268 < 32l);
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
        while (while_method_1(v302)){
            int v304;
            v304 = 0l;
            while (while_method_2(v304)){
                assert("Tensor range check" && 0 <= v302 && v302 < 32l);
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
        v316 = v315 / 4096.0f;
        float v317[128l];
        int v318;
        v318 = 0l;
        while (while_method_1(v318)){
            int v320;
            v320 = 0l;
            while (while_method_2(v320)){
                assert("Tensor range check" && 0 <= v318 && v318 < 32l);
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
                assert("Tensor range check" && 0 <= v318 && v318 < 32l);
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
        while (while_method_1(v328)){
            int v330;
            v330 = 0l;
            while (while_method_2(v330)){
                assert("Tensor range check" && 0 <= v328 && v328 < 32l);
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
        float v341[128l];
        int v342;
        v342 = 0l;
        while (while_method_1(v342)){
            int v344;
            v344 = 0l;
            while (while_method_2(v344)){
                assert("Tensor range check" && 0 <= v342 && v342 < 32l);
                assert("Tensor range check" && 0 <= v344 && v344 < 4l);
                int v346;
                v346 = 4l * v342;
                int v347;
                v347 = v346 + v344;
                float v348;
                v348 = v317[v347];
                float v349;
                v349 = v348 / v340;
                assert("Tensor range check" && 0 <= v342 && v342 < 32l);
                assert("Tensor range check" && 0 <= v344 && v344 < 4l);
                v341[v347] = v349;
                v344 += 1l ;
            }
            v342 += 1l ;
        }
        assert("Tensor range check" && 0 <= v249 && v249 < 64l);
        int v350;
        v350 = 0l;
        while (while_method_1(v350)){
            assert("Tensor range check" && 0 <= v350 && v350 < 32l);
            int v352;
            v352 = 128l * v350;
            int v353;
            v353 = v352 + v258;
            assert("Tensor range check" && 0 <= v350 && v350 < 32l);
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
        v249 += 24l ;
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
    v367 = 4096l * v362;
    int v368;
    v368 = v367 + v366;
    assert("Tensor range check" && 0 <= v362 && v362 < 1l);
    assert("Tensor range check" && 0 <= v361 && v361 < 32l);
    int v369;
    v369 = blockIdx.x;
    int v370;
    v370 = v369;
    while (while_method_0(v370)){
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
        v378 = 4096l * v370;
        int v379;
        v379 = v378 + v368;
        float v380[128l];
        int v381[128l];
        int v382;
        v382 = 0l;
        while (while_method_1(v382)){
            assert("Tensor range check" && 0 <= v382 && v382 < 32l);
            int v384;
            v384 = 4l * v382;
            assert("Tensor range check" && 0 <= v382 && v382 < 32l);
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
        while (while_method_1(v389)){
            int v391;
            v391 = 0l;
            while (while_method_2(v391)){
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
                    v406 = v389 < 32l;
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
                assert("Tensor range check" && 0 <= v389 && v389 < 32l);
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
        float v422[128l];
        int v423;
        v423 = 0l;
        while (while_method_1(v423)){
            int v425;
            v425 = 0l;
            while (while_method_2(v425)){
                assert("Tensor range check" && 0 <= v423 && v423 < 32l);
                assert("Tensor range check" && 0 <= v425 && v425 < 4l);
                int v427;
                v427 = 4l * v423;
                int v428;
                v428 = v427 + v425;
                float v429;
                v429 = v380[v428];
                float v430;
                v430 = v429 * v429;
                assert("Tensor range check" && 0 <= v423 && v423 < 32l);
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
        while (while_method_1(v432)){
            int v434;
            v434 = 0l;
            while (while_method_2(v434)){
                assert("Tensor range check" && 0 <= v432 && v432 < 32l);
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
        float v446[128l];
        int v447;
        v447 = 0l;
        while (while_method_1(v447)){
            int v449;
            v449 = 0l;
            while (while_method_2(v449)){
                assert("Tensor range check" && 0 <= v447 && v447 < 32l);
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
                assert("Tensor range check" && 0 <= v447 && v447 < 32l);
                assert("Tensor range check" && 0 <= v449 && v449 < 4l);
                v446[v452] = v457;
                v449 += 1l ;
            }
            v447 += 1l ;
        }
        assert("Tensor range check" && 0 <= v370 && v370 < 64l);
        int v458;
        v458 = 0l;
        while (while_method_1(v458)){
            assert("Tensor range check" && 0 <= v458 && v458 < 32l);
            int v460;
            v460 = 128l * v458;
            int v461;
            v461 = v460 + v379;
            assert("Tensor range check" && 0 <= v458 && v458 < 32l);
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
        v370 += 24l ;
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
    v475 = 4096l * v470;
    int v476;
    v476 = v475 + v474;
    assert("Tensor range check" && 0 <= v470 && v470 < 1l);
    int v477;
    v477 = blockIdx.x;
    int v478;
    v478 = v477;
    while (while_method_0(v478)){
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
        v486 = 4096l * v478;
        int v487;
        v487 = v486 + v476;
        float v488[128l];
        int v489[128l];
        int v490;
        v490 = 0l;
        while (while_method_1(v490)){
            assert("Tensor range check" && 0 <= v490 && v490 < 32l);
            int v492;
            v492 = 4l * v490;
            assert("Tensor range check" && 0 <= v490 && v490 < 32l);
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
        while (while_method_1(v497)){
            int v499;
            v499 = 0l;
            while (while_method_2(v499)){
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
                    v514 = v497 < 32l;
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
                assert("Tensor range check" && 0 <= v497 && v497 < 32l);
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
        Tuple0 tmp0 = Tuple0{-1.0f / 0.0f, 0l};
        v530 = tmp0.v0; v531 = tmp0.v1;
        int v532;
        v532 = 0l;
        while (while_method_1(v532)){
            int v534;
            v534 = 0l;
            while (while_method_2(v534)){
                assert("Tensor range check" && 0 <= v532 && v532 < 32l);
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
        Tuple0 tmp1 = cooperative_groups::reduce(v546, Tuple0{v530, v531}, v547);
        v548 = tmp1.v0; v549 = tmp1.v1;
        assert("Tensor range check" && 0 <= v478 && v478 < 64l);
        v8[v529] = v549;
        v478 += 24l ;
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
    v560 = 4096l * v555;
    int v561;
    v561 = v560 + v559;
    assert("Tensor range check" && 0 <= v555 && v555 < 1l);
    assert("Tensor range check" && 0 <= v554 && v554 < 32l);
    int v562;
    v562 = blockIdx.x;
    int v563;
    v563 = v562;
    while (while_method_0(v563)){
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
        v571 = 4096l * v563;
        int v572;
        v572 = v571 + v561;
        float v573[128l];
        int v574[128l];
        int v575;
        v575 = 0l;
        while (while_method_1(v575)){
            assert("Tensor range check" && 0 <= v575 && v575 < 32l);
            int v577;
            v577 = 4l * v575;
            assert("Tensor range check" && 0 <= v575 && v575 < 32l);
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
        while (while_method_1(v582)){
            int v584;
            v584 = 0l;
            while (while_method_2(v584)){
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
                    v599 = v582 < 32l;
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
                assert("Tensor range check" && 0 <= v582 && v582 < 32l);
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
        while (while_method_1(v616)){
            int v618;
            v618 = 0l;
            while (while_method_2(v618)){
                assert("Tensor range check" && 0 <= v616 && v616 < 32l);
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
        v630 = v629 / 4096.0f;
        float v631[128l];
        int v632;
        v632 = 0l;
        while (while_method_1(v632)){
            int v634;
            v634 = 0l;
            while (while_method_2(v634)){
                assert("Tensor range check" && 0 <= v632 && v632 < 32l);
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
                assert("Tensor range check" && 0 <= v632 && v632 < 32l);
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
        while (while_method_1(v642)){
            int v644;
            v644 = 0l;
            while (while_method_2(v644)){
                assert("Tensor range check" && 0 <= v642 && v642 < 32l);
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
        float v655[128l];
        int v656;
        v656 = 0l;
        while (while_method_1(v656)){
            int v658;
            v658 = 0l;
            while (while_method_2(v658)){
                assert("Tensor range check" && 0 <= v656 && v656 < 32l);
                assert("Tensor range check" && 0 <= v658 && v658 < 4l);
                int v660;
                v660 = 4l * v656;
                int v661;
                v661 = v660 + v658;
                float v662;
                v662 = v631[v661];
                float v663;
                v663 = v662 / v654;
                assert("Tensor range check" && 0 <= v656 && v656 < 32l);
                assert("Tensor range check" && 0 <= v658 && v658 < 4l);
                v655[v661] = v663;
                v658 += 1l ;
            }
            v656 += 1l ;
        }
        float v664[128l];
        float v665;
        v665 = 0.0f;
        int v666;
        v666 = 0l;
        while (while_method_1(v666)){
            assert("Tensor range check" && 0 <= v666 && v666 < 32l);
            int v668;
            v668 = 4l * v666;
            assert("Tensor range check" && 0 <= v666 && v666 < 32l);
            int v669; float v670;
            Tuple1 tmp2 = Tuple1{0l, 0.0f};
            v669 = tmp2.v0; v670 = tmp2.v1;
            while (while_method_2(v669)){
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
            Tuple1 tmp3 = Tuple1{0l, v685};
            v686 = tmp3.v0; v687 = tmp3.v1;
            while (while_method_2(v686)){
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
        while (while_method_1(v693)){
            assert("Tensor range check" && 0 <= v693 && v693 < 32l);
            int v695;
            v695 = 128l * v693;
            int v696;
            v696 = v695 + v572;
            assert("Tensor range check" && 0 <= v693 && v693 < 32l);
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
        v563 += 24l ;
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
    v712 = 4096l * v707;
    int v713;
    v713 = v712 + v711;
    assert("Tensor range check" && 0 <= v707 && v707 < 1l);
    assert("Tensor range check" && 0 <= v706 && v706 < 32l);
    int v714;
    v714 = blockIdx.x;
    int v715;
    v715 = v714;
    while (while_method_0(v715)){
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
        v723 = 4096l * v715;
        int v724;
        v724 = v723 + v713;
        int v725[128l];
        int v726[128l];
        int v727;
        v727 = 0l;
        while (while_method_1(v727)){
            assert("Tensor range check" && 0 <= v727 && v727 < 32l);
            int v729;
            v729 = 4l * v727;
            assert("Tensor range check" && 0 <= v727 && v727 < 32l);
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
        while (while_method_1(v734)){
            int v736;
            v736 = 0l;
            while (while_method_2(v736)){
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
                    v751 = v734 < 32l;
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
                assert("Tensor range check" && 0 <= v734 && v734 < 32l);
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
        int v767[128l];
        int v768;
        v768 = 0l;
        int v769;
        v769 = 0l;
        while (while_method_1(v769)){
            assert("Tensor range check" && 0 <= v769 && v769 < 32l);
            int v771;
            v771 = 4l * v769;
            assert("Tensor range check" && 0 <= v769 && v769 < 32l);
            int v772; int v773;
            Tuple2 tmp4 = Tuple2{0l, 0l};
            v772 = tmp4.v0; v773 = tmp4.v1;
            while (while_method_2(v772)){
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
            Tuple2 tmp5 = Tuple2{0l, v788};
            v789 = tmp5.v0; v790 = tmp5.v1;
            while (while_method_2(v789)){
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
        while (while_method_1(v796)){
            assert("Tensor range check" && 0 <= v796 && v796 < 32l);
            int v798;
            v798 = 128l * v796;
            int v799;
            v799 = v798 + v724;
            assert("Tensor range check" && 0 <= v796 && v796 < 32l);
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
        v715 += 24l ;
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
    v813 = 4096l * v808;
    int v814;
    v814 = v813 + v812;
    assert("Tensor range check" && 0 <= v808 && v808 < 1l);
    assert("Tensor range check" && 0 <= v807 && v807 < 32l);
    int v815;
    v815 = blockIdx.x;
    int v816;
    v816 = v815;
    while (while_method_0(v816)){
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
        v824 = 4096l * v816;
        int v825;
        v825 = v824 + v814;
        float v826[128l];
        int v827[128l];
        int v828;
        v828 = 0l;
        while (while_method_1(v828)){
            assert("Tensor range check" && 0 <= v828 && v828 < 32l);
            int v830;
            v830 = 4l * v828;
            assert("Tensor range check" && 0 <= v828 && v828 < 32l);
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
        while (while_method_1(v835)){
            int v837;
            v837 = 0l;
            while (while_method_2(v837)){
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
                    v852 = v835 < 32l;
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
                assert("Tensor range check" && 0 <= v835 && v835 < 32l);
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
        bool v868[128l];
        int v869;
        v869 = 0l;
        while (while_method_1(v869)){
            int v871;
            v871 = 0l;
            while (while_method_2(v871)){
                assert("Tensor range check" && 0 <= v869 && v869 < 32l);
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
                assert("Tensor range check" && 0 <= v869 && v869 < 32l);
                assert("Tensor range check" && 0 <= v871 && v871 < 4l);
                v868[v874] = v877;
                v871 += 1l ;
            }
            v869 += 1l ;
        }
        int v878[128l];
        int v879;
        v879 = 0l;
        while (while_method_1(v879)){
            int v881;
            v881 = 0l;
            while (while_method_2(v881)){
                assert("Tensor range check" && 0 <= v879 && v879 < 32l);
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
                assert("Tensor range check" && 0 <= v879 && v879 < 32l);
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
        while (while_method_1(v888)){
            int v890;
            v890 = 0l;
            while (while_method_2(v890)){
                assert("Tensor range check" && 0 <= v888 && v888 < 32l);
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
        float v902[128l];
        int v903;
        v903 = 0l;
        while (while_method_1(v903)){
            int v905;
            v905 = 0l;
            while (while_method_2(v905)){
                assert("Tensor range check" && 0 <= v903 && v903 < 32l);
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
                assert("Tensor range check" && 0 <= v903 && v903 < 32l);
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
        while (while_method_1(v913)){
            int v915;
            v915 = 0l;
            while (while_method_2(v915)){
                assert("Tensor range check" && 0 <= v913 && v913 < 32l);
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
        float v929[128l];
        int v930;
        v930 = 0l;
        while (while_method_1(v930)){
            int v932;
            v932 = 0l;
            while (while_method_2(v932)){
                assert("Tensor range check" && 0 <= v930 && v930 < 32l);
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
                assert("Tensor range check" && 0 <= v930 && v930 < 32l);
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
        while (while_method_1(v942)){
            int v944;
            v944 = 0l;
            while (while_method_2(v944)){
                assert("Tensor range check" && 0 <= v942 && v942 < 32l);
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
        float v955[128l];
        int v956;
        v956 = 0l;
        while (while_method_1(v956)){
            int v958;
            v958 = 0l;
            while (while_method_2(v958)){
                assert("Tensor range check" && 0 <= v956 && v956 < 32l);
                assert("Tensor range check" && 0 <= v958 && v958 < 4l);
                int v960;
                v960 = 4l * v956;
                int v961;
                v961 = v960 + v958;
                float v962;
                v962 = v929[v961];
                float v963;
                v963 = v962 / v954;
                assert("Tensor range check" && 0 <= v956 && v956 < 32l);
                assert("Tensor range check" && 0 <= v958 && v958 < 4l);
                v955[v961] = v963;
                v958 += 1l ;
            }
            v956 += 1l ;
        }
        assert("Tensor range check" && 0 <= v816 && v816 < 64l);
        int v964;
        v964 = 0l;
        while (while_method_1(v964)){
            assert("Tensor range check" && 0 <= v964 && v964 < 32l);
            int v966;
            v966 = 128l * v964;
            int v967;
            v967 = v966 + v825;
            assert("Tensor range check" && 0 <= v964 && v964 < 32l);
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
        v816 += 24l ;
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
    v984 = 4096l * v979;
    int v985;
    v985 = v984 + v983;
    assert("Tensor range check" && 0 <= v979 && v979 < 1l);
    assert("Tensor range check" && 0 <= v978 && v978 < 32l);
    assert("Tensor range check" && 0 <= v979 && v979 < 1l);
    int v986;
    v986 = blockIdx.x;
    int v987;
    v987 = v986;
    while (while_method_0(v987)){
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
        v995 = 4096l * v987;
        int v996;
        v996 = v995 + v985;
        float v997[128l];
        int v998[128l];
        int v999;
        v999 = 0l;
        while (while_method_1(v999)){
            assert("Tensor range check" && 0 <= v999 && v999 < 32l);
            int v1001;
            v1001 = 4l * v999;
            assert("Tensor range check" && 0 <= v999 && v999 < 32l);
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
        while (while_method_1(v1006)){
            int v1008;
            v1008 = 0l;
            while (while_method_2(v1008)){
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
                    v1023 = v1006 < 32l;
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
                assert("Tensor range check" && 0 <= v1006 && v1006 < 32l);
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
        while (while_method_1(v1040)){
            int v1042;
            v1042 = 0l;
            while (while_method_2(v1042)){
                assert("Tensor range check" && 0 <= v1040 && v1040 < 32l);
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
        v1054 = v1053 / 4096.0f;
        float v1055[128l];
        int v1056;
        v1056 = 0l;
        while (while_method_1(v1056)){
            int v1058;
            v1058 = 0l;
            while (while_method_2(v1058)){
                assert("Tensor range check" && 0 <= v1056 && v1056 < 32l);
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
                assert("Tensor range check" && 0 <= v1056 && v1056 < 32l);
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
        while (while_method_1(v1066)){
            int v1068;
            v1068 = 0l;
            while (while_method_2(v1068)){
                assert("Tensor range check" && 0 <= v1066 && v1066 < 32l);
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
        float v1079[128l];
        int v1080;
        v1080 = 0l;
        while (while_method_1(v1080)){
            int v1082;
            v1082 = 0l;
            while (while_method_2(v1082)){
                assert("Tensor range check" && 0 <= v1080 && v1080 < 32l);
                assert("Tensor range check" && 0 <= v1082 && v1082 < 4l);
                int v1084;
                v1084 = 4l * v1080;
                int v1085;
                v1085 = v1084 + v1082;
                float v1086;
                v1086 = v1055[v1085];
                float v1087;
                v1087 = v1086 / v1078;
                assert("Tensor range check" && 0 <= v1080 && v1080 < 32l);
                assert("Tensor range check" && 0 <= v1082 && v1082 < 4l);
                v1079[v1085] = v1087;
                v1082 += 1l ;
            }
            v1080 += 1l ;
        }
        float v1088[128l];
        float v1089;
        v1089 = 0.0f;
        int v1090;
        v1090 = 0l;
        while (while_method_1(v1090)){
            assert("Tensor range check" && 0 <= v1090 && v1090 < 32l);
            int v1092;
            v1092 = 4l * v1090;
            assert("Tensor range check" && 0 <= v1090 && v1090 < 32l);
            int v1093; float v1094;
            Tuple1 tmp6 = Tuple1{0l, 0.0f};
            v1093 = tmp6.v0; v1094 = tmp6.v1;
            while (while_method_2(v1093)){
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
            Tuple1 tmp7 = Tuple1{0l, v1109};
            v1110 = tmp7.v0; v1111 = tmp7.v1;
            while (while_method_2(v1110)){
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
        float v1117[128l];
        bool v1118[128l];
        int v1119;
        v1119 = 0l;
        while (while_method_1(v1119)){
            int v1121;
            v1121 = 0l;
            while (while_method_2(v1121)){
                assert("Tensor range check" && 0 <= v1119 && v1119 < 32l);
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
                assert("Tensor range check" && 0 <= v1119 && v1119 < 32l);
                assert("Tensor range check" && 0 <= v1121 && v1121 < 4l);
                v1117[v1124] = v1125;
                v1118[v1124] = v1127;
                v1121 += 1l ;
            }
            v1119 += 1l ;
        }
        float v1128; bool v1129;
        Tuple3 tmp8 = Tuple3{-1.0f / 0.0f, false};
        v1128 = tmp8.v0; v1129 = tmp8.v1;
        int v1130;
        v1130 = 0l;
        while (while_method_1(v1130)){
            int v1132;
            v1132 = 0l;
            while (while_method_2(v1132)){
                assert("Tensor range check" && 0 <= v1130 && v1130 < 32l);
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
        Tuple3 tmp9 = cooperative_groups::reduce(v1149, Tuple3{v1128, v1129}, v1150);
        v1151 = tmp9.v0; v1152 = tmp9.v1;
        bool v1153;
        v1153 = v1152 == false;
        if (v1153){
            assert("The local reduce must be true." && v1152);
        } else {
        }
        float v1155[128l];
        int v1156[128l];
        int v1157;
        v1157 = 0l;
        while (while_method_1(v1157)){
            int v1159;
            v1159 = 0l;
            while (while_method_2(v1159)){
                assert("Tensor range check" && 0 <= v1157 && v1157 < 32l);
                assert("Tensor range check" && 0 <= v1159 && v1159 < 4l);
                int v1161;
                v1161 = 4l * v1157;
                int v1162;
                v1162 = v1161 + v1159;
                int v1163;
                v1163 = v998[v1162];
                float v1164;
                v1164 = curand_uniform(&v973);
                assert("Tensor range check" && 0 <= v1157 && v1157 < 32l);
                assert("Tensor range check" && 0 <= v1159 && v1159 < 4l);
                v1155[v1162] = v1164;
                v1156[v1162] = v1163;
                v1159 += 1l ;
            }
            v1157 += 1l ;
        }
        float v1165; int v1166;
        Tuple0 tmp10 = Tuple0{0.0f, 2147483647l};
        v1165 = tmp10.v0; v1166 = tmp10.v1;
        int v1167;
        v1167 = 0l;
        while (while_method_1(v1167)){
            int v1169;
            v1169 = 0l;
            while (while_method_2(v1169)){
                assert("Tensor range check" && 0 <= v1167 && v1167 < 32l);
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
        Tuple0 tmp11 = cooperative_groups::reduce(v1181, Tuple0{v1165, v1166}, v1182);
        v1183 = tmp11.v0; v1184 = tmp11.v1;
        float v1185;
        v1185 = v1151 * v1183;
        int v1186[128l];
        bool v1187[128l];
        int v1188;
        v1188 = 0l;
        while (while_method_1(v1188)){
            int v1190;
            v1190 = 0l;
            while (while_method_2(v1190)){
                assert("Tensor range check" && 0 <= v1188 && v1188 < 32l);
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
                assert("Tensor range check" && 0 <= v1188 && v1188 < 32l);
                assert("Tensor range check" && 0 <= v1190 && v1190 < 4l);
                v1186[v1193] = v1199;
                v1187[v1193] = v1200;
                v1190 += 1l ;
            }
            v1188 += 1l ;
        }
        int v1201; bool v1202;
        Tuple4 tmp12 = Tuple4{2147483647l, false};
        v1201 = tmp12.v0; v1202 = tmp12.v1;
        int v1203;
        v1203 = 0l;
        while (while_method_1(v1203)){
            int v1205;
            v1205 = 0l;
            while (while_method_2(v1205)){
                assert("Tensor range check" && 0 <= v1203 && v1203 < 32l);
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
        Tuple4 tmp13 = cooperative_groups::reduce(v1222, Tuple4{v1201, v1202}, v1223);
        v1224 = tmp13.v0; v1225 = tmp13.v1;
        bool v1226;
        v1226 = v1225 == false;
        if (v1226){
            assert("The local reduce must be true." && v1225);
        } else {
        }
        assert("Tensor range check" && 0 <= v987 && v987 < 64l);
        int v1228;
        v1228 = 0l;
        while (while_method_1(v1228)){
            assert("Tensor range check" && 0 <= v1228 && v1228 < 32l);
            int v1230;
            v1230 = 128l * v1228;
            int v1231;
            v1231 = v1230 + v996;
            assert("Tensor range check" && 0 <= v1228 && v1228 < 32l);
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
        v987 += 24l ;
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
    v1248 = 4096l * v1243;
    int v1249;
    v1249 = v1248 + v1247;
    assert("Tensor range check" && 0 <= v1243 && v1243 < 1l);
    assert("Tensor range check" && 0 <= v1242 && v1242 < 32l);
    assert("Tensor range check" && 0 <= v1243 && v1243 < 1l);
    int v1250;
    v1250 = blockIdx.x;
    int v1251;
    v1251 = v1250;
    while (while_method_0(v1251)){
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
        v1259 = 4096l * v1251;
        int v1260;
        v1260 = v1259 + v1249;
        float v1261[128l];
        int v1262[128l];
        int v1263;
        v1263 = 0l;
        while (while_method_1(v1263)){
            assert("Tensor range check" && 0 <= v1263 && v1263 < 32l);
            int v1265;
            v1265 = 4l * v1263;
            assert("Tensor range check" && 0 <= v1263 && v1263 < 32l);
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
        while (while_method_1(v1270)){
            int v1272;
            v1272 = 0l;
            while (while_method_2(v1272)){
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
                    v1287 = v1270 < 32l;
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
                assert("Tensor range check" && 0 <= v1270 && v1270 < 32l);
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
        bool v1303[128l];
        int v1304;
        v1304 = 0l;
        while (while_method_1(v1304)){
            int v1306;
            v1306 = 0l;
            while (while_method_2(v1306)){
                assert("Tensor range check" && 0 <= v1304 && v1304 < 32l);
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
                assert("Tensor range check" && 0 <= v1304 && v1304 < 32l);
                assert("Tensor range check" && 0 <= v1306 && v1306 < 4l);
                v1303[v1309] = v1312;
                v1306 += 1l ;
            }
            v1304 += 1l ;
        }
        int v1313[128l];
        int v1314;
        v1314 = 0l;
        while (while_method_1(v1314)){
            int v1316;
            v1316 = 0l;
            while (while_method_2(v1316)){
                assert("Tensor range check" && 0 <= v1314 && v1314 < 32l);
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
                assert("Tensor range check" && 0 <= v1314 && v1314 < 32l);
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
        while (while_method_1(v1323)){
            int v1325;
            v1325 = 0l;
            while (while_method_2(v1325)){
                assert("Tensor range check" && 0 <= v1323 && v1323 < 32l);
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
        float v1337[128l];
        int v1338;
        v1338 = 0l;
        while (while_method_1(v1338)){
            int v1340;
            v1340 = 0l;
            while (while_method_2(v1340)){
                assert("Tensor range check" && 0 <= v1338 && v1338 < 32l);
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
                assert("Tensor range check" && 0 <= v1338 && v1338 < 32l);
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
        while (while_method_1(v1348)){
            int v1350;
            v1350 = 0l;
            while (while_method_2(v1350)){
                assert("Tensor range check" && 0 <= v1348 && v1348 < 32l);
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
        float v1364[128l];
        int v1365;
        v1365 = 0l;
        while (while_method_1(v1365)){
            int v1367;
            v1367 = 0l;
            while (while_method_2(v1367)){
                assert("Tensor range check" && 0 <= v1365 && v1365 < 32l);
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
                assert("Tensor range check" && 0 <= v1365 && v1365 < 32l);
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
        while (while_method_1(v1377)){
            int v1379;
            v1379 = 0l;
            while (while_method_2(v1379)){
                assert("Tensor range check" && 0 <= v1377 && v1377 < 32l);
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
        float v1390[128l];
        int v1391;
        v1391 = 0l;
        while (while_method_1(v1391)){
            int v1393;
            v1393 = 0l;
            while (while_method_2(v1393)){
                assert("Tensor range check" && 0 <= v1391 && v1391 < 32l);
                assert("Tensor range check" && 0 <= v1393 && v1393 < 4l);
                int v1395;
                v1395 = 4l * v1391;
                int v1396;
                v1396 = v1395 + v1393;
                float v1397;
                v1397 = v1364[v1396];
                float v1398;
                v1398 = v1397 / v1389;
                assert("Tensor range check" && 0 <= v1391 && v1391 < 32l);
                assert("Tensor range check" && 0 <= v1393 && v1393 < 4l);
                v1390[v1396] = v1398;
                v1393 += 1l ;
            }
            v1391 += 1l ;
        }
        float v1399[128l];
        float v1400;
        v1400 = 0.0f;
        int v1401;
        v1401 = 0l;
        while (while_method_1(v1401)){
            assert("Tensor range check" && 0 <= v1401 && v1401 < 32l);
            int v1403;
            v1403 = 4l * v1401;
            assert("Tensor range check" && 0 <= v1401 && v1401 < 32l);
            int v1404; float v1405;
            Tuple1 tmp14 = Tuple1{0l, 0.0f};
            v1404 = tmp14.v0; v1405 = tmp14.v1;
            while (while_method_2(v1404)){
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
            Tuple1 tmp15 = Tuple1{0l, v1420};
            v1421 = tmp15.v0; v1422 = tmp15.v1;
            while (while_method_2(v1421)){
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
        float v1428[128l];
        bool v1429[128l];
        int v1430;
        v1430 = 0l;
        while (while_method_1(v1430)){
            int v1432;
            v1432 = 0l;
            while (while_method_2(v1432)){
                assert("Tensor range check" && 0 <= v1430 && v1430 < 32l);
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
                assert("Tensor range check" && 0 <= v1430 && v1430 < 32l);
                assert("Tensor range check" && 0 <= v1432 && v1432 < 4l);
                v1428[v1435] = v1436;
                v1429[v1435] = v1438;
                v1432 += 1l ;
            }
            v1430 += 1l ;
        }
        float v1439; bool v1440;
        Tuple3 tmp16 = Tuple3{-1.0f / 0.0f, false};
        v1439 = tmp16.v0; v1440 = tmp16.v1;
        int v1441;
        v1441 = 0l;
        while (while_method_1(v1441)){
            int v1443;
            v1443 = 0l;
            while (while_method_2(v1443)){
                assert("Tensor range check" && 0 <= v1441 && v1441 < 32l);
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
        Tuple3 tmp17 = cooperative_groups::reduce(v1460, Tuple3{v1439, v1440}, v1461);
        v1462 = tmp17.v0; v1463 = tmp17.v1;
        bool v1464;
        v1464 = v1463 == false;
        if (v1464){
            assert("The local reduce must be true." && v1463);
        } else {
        }
        float v1466[128l];
        int v1467[128l];
        int v1468;
        v1468 = 0l;
        while (while_method_1(v1468)){
            int v1470;
            v1470 = 0l;
            while (while_method_2(v1470)){
                assert("Tensor range check" && 0 <= v1468 && v1468 < 32l);
                assert("Tensor range check" && 0 <= v1470 && v1470 < 4l);
                int v1472;
                v1472 = 4l * v1468;
                int v1473;
                v1473 = v1472 + v1470;
                int v1474;
                v1474 = v1262[v1473];
                float v1475;
                v1475 = curand_uniform(&v1237);
                assert("Tensor range check" && 0 <= v1468 && v1468 < 32l);
                assert("Tensor range check" && 0 <= v1470 && v1470 < 4l);
                v1466[v1473] = v1475;
                v1467[v1473] = v1474;
                v1470 += 1l ;
            }
            v1468 += 1l ;
        }
        float v1476; int v1477;
        Tuple0 tmp18 = Tuple0{0.0f, 2147483647l};
        v1476 = tmp18.v0; v1477 = tmp18.v1;
        int v1478;
        v1478 = 0l;
        while (while_method_1(v1478)){
            int v1480;
            v1480 = 0l;
            while (while_method_2(v1480)){
                assert("Tensor range check" && 0 <= v1478 && v1478 < 32l);
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
        Tuple0 tmp19 = cooperative_groups::reduce(v1492, Tuple0{v1476, v1477}, v1493);
        v1494 = tmp19.v0; v1495 = tmp19.v1;
        float v1496;
        v1496 = v1462 * v1494;
        int v1497[128l];
        bool v1498[128l];
        int v1499;
        v1499 = 0l;
        while (while_method_1(v1499)){
            int v1501;
            v1501 = 0l;
            while (while_method_2(v1501)){
                assert("Tensor range check" && 0 <= v1499 && v1499 < 32l);
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
                assert("Tensor range check" && 0 <= v1499 && v1499 < 32l);
                assert("Tensor range check" && 0 <= v1501 && v1501 < 4l);
                v1497[v1504] = v1510;
                v1498[v1504] = v1511;
                v1501 += 1l ;
            }
            v1499 += 1l ;
        }
        int v1512; bool v1513;
        Tuple4 tmp20 = Tuple4{2147483647l, false};
        v1512 = tmp20.v0; v1513 = tmp20.v1;
        int v1514;
        v1514 = 0l;
        while (while_method_1(v1514)){
            int v1516;
            v1516 = 0l;
            while (while_method_2(v1516)){
                assert("Tensor range check" && 0 <= v1514 && v1514 < 32l);
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
        Tuple4 tmp21 = cooperative_groups::reduce(v1533, Tuple4{v1512, v1513}, v1534);
        v1535 = tmp21.v0; v1536 = tmp21.v1;
        bool v1537;
        v1537 = v1536 == false;
        if (v1537){
            assert("The local reduce must be true." && v1536);
        } else {
        }
        assert("Tensor range check" && 0 <= v1251 && v1251 < 64l);
        int v1539;
        v1539 = 0l;
        while (while_method_1(v1539)){
            assert("Tensor range check" && 0 <= v1539 && v1539 < 32l);
            int v1541;
            v1541 = 128l * v1539;
            int v1542;
            v1542 = v1541 + v1260;
            assert("Tensor range check" && 0 <= v1539 && v1539 < 32l);
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
        v1251 += 24l ;
    }
    v15.sync() ;
    return ;
}
extern "C" __global__ void entry1(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14) {
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
    while (while_method_3(v29)){
        bool v31;
        v31 = 0l <= v29;
        bool v32;
        v32 = v31 == false;
        if (v32){
            assert("The index needs to be zero or positive." && v31);
        } else {
        }
        bool v34;
        v34 = v29 < 2048l;
        bool v35;
        v35 = v34 == false;
        if (v35){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v34);
        } else {
        }
        assert("Tensor range check" && 0 <= v29 && v29 < 2048l);
        int v37;
        v37 = 128l * v29;
        int v38;
        v38 = v37 + v27;
        int v39[4l];
        int v40[4l];
        int v41;
        v41 = 0l;
        while (while_method_4(v41)){
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
        while (while_method_4(v48)){
            int v50;
            v50 = 0l;
            while (while_method_2(v50)){
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
        assert("Tensor range check" && 0 <= v29 && v29 < 2048l);
        int v82;
        v82 = 0l;
        while (while_method_4(v82)){
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
        v29 += 24l ;
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
    while (while_method_3(v102)){
        bool v104;
        v104 = 0l <= v102;
        bool v105;
        v105 = v104 == false;
        if (v105){
            assert("The index needs to be zero or positive." && v104);
        } else {
        }
        bool v107;
        v107 = v102 < 2048l;
        bool v108;
        v108 = v107 == false;
        if (v108){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v107);
        } else {
        }
        assert("Tensor range check" && 0 <= v102 && v102 < 2048l);
        int v110;
        v110 = 128l * v102;
        int v111;
        v111 = v110 + v100;
        float v112[4l];
        int v113[4l];
        int v114;
        v114 = 0l;
        while (while_method_4(v114)){
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
        while (while_method_4(v121)){
            int v123;
            v123 = 0l;
            while (while_method_2(v123)){
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
        while (while_method_4(v157)){
            int v159;
            v159 = 0l;
            while (while_method_2(v159)){
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
        assert("Tensor range check" && 0 <= v102 && v102 < 2048l);
        int v164;
        v164 = 0l;
        while (while_method_4(v164)){
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
        v102 += 24l ;
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
    while (while_method_3(v186)){
        bool v188;
        v188 = 0l <= v186;
        bool v189;
        v189 = v188 == false;
        if (v189){
            assert("The index needs to be zero or positive." && v188);
        } else {
        }
        bool v191;
        v191 = v186 < 2048l;
        bool v192;
        v192 = v191 == false;
        if (v192){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v191);
        } else {
        }
        assert("Tensor range check" && 0 <= v186 && v186 < 2048l);
        int v194;
        v194 = 128l * v186;
        int v195;
        v195 = v194 + v184;
        float v196[4l];
        int v197[4l];
        int v198;
        v198 = 0l;
        while (while_method_4(v198)){
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
        while (while_method_4(v205)){
            int v207;
            v207 = 0l;
            while (while_method_2(v207)){
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
        assert("Tensor range check" && 0 <= v186 && v186 < 2048l);
        int v239;
        v239 = 2l * v186;
        int v240;
        v240 = v239 + v178;
        v11[v240] = v238;
        v186 += 24l ;
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
    while (while_method_3(v254)){
        bool v256;
        v256 = 0l <= v254;
        bool v257;
        v257 = v256 == false;
        if (v257){
            assert("The index needs to be zero or positive." && v256);
        } else {
        }
        bool v259;
        v259 = v254 < 2048l;
        bool v260;
        v260 = v259 == false;
        if (v260){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v259);
        } else {
        }
        assert("Tensor range check" && 0 <= v254 && v254 < 2048l);
        int v262;
        v262 = 128l * v254;
        int v263;
        v263 = v262 + v252;
        float v264[4l];
        int v265[4l];
        int v266;
        v266 = 0l;
        while (while_method_4(v266)){
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
        while (while_method_4(v273)){
            int v275;
            v275 = 0l;
            while (while_method_2(v275)){
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
        while (while_method_4(v308)){
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
        while (while_method_4(v324)){
            int v326;
            v326 = 0l;
            while (while_method_2(v326)){
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
        while (while_method_4(v334)){
            int v336;
            v336 = 0l;
            while (while_method_2(v336)){
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
        while (while_method_4(v348)){
            int v350;
            v350 = 0l;
            while (while_method_2(v350)){
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
        assert("Tensor range check" && 0 <= v254 && v254 < 2048l);
        int v356;
        v356 = 0l;
        while (while_method_4(v356)){
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
        v254 += 24l ;
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
    while (while_method_3(v376)){
        bool v378;
        v378 = 0l <= v376;
        bool v379;
        v379 = v378 == false;
        if (v379){
            assert("The index needs to be zero or positive." && v378);
        } else {
        }
        bool v381;
        v381 = v376 < 2048l;
        bool v382;
        v382 = v381 == false;
        if (v382){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v381);
        } else {
        }
        assert("Tensor range check" && 0 <= v376 && v376 < 2048l);
        int v384;
        v384 = 128l * v376;
        int v385;
        v385 = v384 + v374;
        float v386[4l];
        int v387[4l];
        int v388;
        v388 = 0l;
        while (while_method_4(v388)){
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
        while (while_method_4(v395)){
            int v397;
            v397 = 0l;
            while (while_method_2(v397)){
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
        while (while_method_4(v430)){
            int v432;
            v432 = 0l;
            while (while_method_2(v432)){
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
        while (while_method_4(v439)){
            int v441;
            v441 = 0l;
            while (while_method_2(v441)){
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
        while (while_method_4(v454)){
            int v456;
            v456 = 0l;
            while (while_method_2(v456)){
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
        assert("Tensor range check" && 0 <= v376 && v376 < 2048l);
        int v465;
        v465 = 0l;
        while (while_method_4(v465)){
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
        v376 += 24l ;
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
    while (while_method_3(v485)){
        bool v487;
        v487 = 0l <= v485;
        bool v488;
        v488 = v487 == false;
        if (v488){
            assert("The index needs to be zero or positive." && v487);
        } else {
        }
        bool v490;
        v490 = v485 < 2048l;
        bool v491;
        v491 = v490 == false;
        if (v491){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v490);
        } else {
        }
        assert("Tensor range check" && 0 <= v485 && v485 < 2048l);
        int v493;
        v493 = 128l * v485;
        int v494;
        v494 = v493 + v483;
        float v495[4l];
        int v496[4l];
        int v497;
        v497 = 0l;
        while (while_method_4(v497)){
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
        while (while_method_4(v504)){
            int v506;
            v506 = 0l;
            while (while_method_2(v506)){
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
        Tuple0 tmp22 = Tuple0{-1.0f / 0.0f, 0l};
        v538 = tmp22.v0; v539 = tmp22.v1;
        int v540;
        v540 = 0l;
        while (while_method_4(v540)){
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
        Tuple0 tmp23 = cooperative_groups::reduce(v554, Tuple0{v538, v539}, v555);
        v556 = tmp23.v0; v557 = tmp23.v1;
        assert("Tensor range check" && 0 <= v485 && v485 < 2048l);
        int v558;
        v558 = 2l * v485;
        int v559;
        v559 = v558 + v477;
        v8[v559] = v557;
        v485 += 24l ;
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
    while (while_method_3(v573)){
        bool v575;
        v575 = 0l <= v573;
        bool v576;
        v576 = v575 == false;
        if (v576){
            assert("The index needs to be zero or positive." && v575);
        } else {
        }
        bool v578;
        v578 = v573 < 2048l;
        bool v579;
        v579 = v578 == false;
        if (v579){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v578);
        } else {
        }
        assert("Tensor range check" && 0 <= v573 && v573 < 2048l);
        int v581;
        v581 = 128l * v573;
        int v582;
        v582 = v581 + v571;
        float v583[4l];
        int v584[4l];
        int v585;
        v585 = 0l;
        while (while_method_4(v585)){
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
        while (while_method_4(v592)){
            int v594;
            v594 = 0l;
            while (while_method_2(v594)){
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
        while (while_method_4(v627)){
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
        while (while_method_4(v643)){
            int v645;
            v645 = 0l;
            while (while_method_2(v645)){
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
        while (while_method_4(v653)){
            int v655;
            v655 = 0l;
            while (while_method_2(v655)){
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
        while (while_method_4(v667)){
            int v669;
            v669 = 0l;
            while (while_method_2(v669)){
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
        while (while_method_4(v677)){
            assert("Tensor range check" && 0 <= v677 && v677 < 1l);
            int v679;
            v679 = 4l * v677;
            assert("Tensor range check" && 0 <= v677 && v677 < 1l);
            int v680; float v681;
            Tuple1 tmp24 = Tuple1{0l, 0.0f};
            v680 = tmp24.v0; v681 = tmp24.v1;
            while (while_method_2(v680)){
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
            Tuple1 tmp25 = Tuple1{0l, v696};
            v697 = tmp25.v0; v698 = tmp25.v1;
            while (while_method_2(v697)){
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
        assert("Tensor range check" && 0 <= v573 && v573 < 2048l);
        int v704;
        v704 = 0l;
        while (while_method_4(v704)){
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
        v573 += 24l ;
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
    while (while_method_3(v726)){
        bool v728;
        v728 = 0l <= v726;
        bool v729;
        v729 = v728 == false;
        if (v729){
            assert("The index needs to be zero or positive." && v728);
        } else {
        }
        bool v731;
        v731 = v726 < 2048l;
        bool v732;
        v732 = v731 == false;
        if (v732){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v731);
        } else {
        }
        assert("Tensor range check" && 0 <= v726 && v726 < 2048l);
        int v734;
        v734 = 128l * v726;
        int v735;
        v735 = v734 + v724;
        int v736[4l];
        int v737[4l];
        int v738;
        v738 = 0l;
        while (while_method_4(v738)){
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
        while (while_method_4(v745)){
            int v747;
            v747 = 0l;
            while (while_method_2(v747)){
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
        while (while_method_4(v781)){
            assert("Tensor range check" && 0 <= v781 && v781 < 1l);
            int v783;
            v783 = 4l * v781;
            assert("Tensor range check" && 0 <= v781 && v781 < 1l);
            int v784; int v785;
            Tuple2 tmp26 = Tuple2{0l, 0l};
            v784 = tmp26.v0; v785 = tmp26.v1;
            while (while_method_2(v784)){
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
            Tuple2 tmp27 = Tuple2{0l, v800};
            v801 = tmp27.v0; v802 = tmp27.v1;
            while (while_method_2(v801)){
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
        assert("Tensor range check" && 0 <= v726 && v726 < 2048l);
        int v808;
        v808 = 0l;
        while (while_method_4(v808)){
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
        v726 += 24l ;
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
    while (while_method_3(v828)){
        bool v830;
        v830 = 0l <= v828;
        bool v831;
        v831 = v830 == false;
        if (v831){
            assert("The index needs to be zero or positive." && v830);
        } else {
        }
        bool v833;
        v833 = v828 < 2048l;
        bool v834;
        v834 = v833 == false;
        if (v834){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v833);
        } else {
        }
        assert("Tensor range check" && 0 <= v828 && v828 < 2048l);
        int v836;
        v836 = 128l * v828;
        int v837;
        v837 = v836 + v826;
        float v838[4l];
        int v839[4l];
        int v840;
        v840 = 0l;
        while (while_method_4(v840)){
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
        while (while_method_4(v847)){
            int v849;
            v849 = 0l;
            while (while_method_2(v849)){
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
        while (while_method_4(v882)){
            int v884;
            v884 = 0l;
            while (while_method_2(v884)){
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
        while (while_method_4(v892)){
            int v894;
            v894 = 0l;
            while (while_method_2(v894)){
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
        while (while_method_4(v901)){
            int v903;
            v903 = 0l;
            while (while_method_2(v903)){
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
        while (while_method_4(v916)){
            int v918;
            v918 = 0l;
            while (while_method_2(v918)){
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
        while (while_method_4(v926)){
            int v928;
            v928 = 0l;
            while (while_method_2(v928)){
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
        while (while_method_4(v943)){
            int v945;
            v945 = 0l;
            while (while_method_2(v945)){
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
        while (while_method_4(v955)){
            int v957;
            v957 = 0l;
            while (while_method_2(v957)){
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
        while (while_method_4(v969)){
            int v971;
            v971 = 0l;
            while (while_method_2(v971)){
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
        assert("Tensor range check" && 0 <= v828 && v828 < 2048l);
        int v977;
        v977 = 0l;
        while (while_method_4(v977)){
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
        v828 += 24l ;
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
    while (while_method_3(v1000)){
        bool v1002;
        v1002 = 0l <= v1000;
        bool v1003;
        v1003 = v1002 == false;
        if (v1003){
            assert("The index needs to be zero or positive." && v1002);
        } else {
        }
        bool v1005;
        v1005 = v1000 < 2048l;
        bool v1006;
        v1006 = v1005 == false;
        if (v1006){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1005);
        } else {
        }
        assert("Tensor range check" && 0 <= v1000 && v1000 < 2048l);
        int v1008;
        v1008 = 128l * v1000;
        int v1009;
        v1009 = v1008 + v998;
        float v1010[4l];
        int v1011[4l];
        int v1012;
        v1012 = 0l;
        while (while_method_4(v1012)){
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
        while (while_method_4(v1019)){
            int v1021;
            v1021 = 0l;
            while (while_method_2(v1021)){
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
        while (while_method_4(v1054)){
            int v1056;
            v1056 = 0l;
            while (while_method_2(v1056)){
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
        while (while_method_4(v1070)){
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
        while (while_method_4(v1080)){
            int v1082;
            v1082 = 0l;
            while (while_method_2(v1082)){
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
        while (while_method_4(v1094)){
            int v1096;
            v1096 = 0l;
            while (while_method_2(v1096)){
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
        while (while_method_4(v1104)){
            assert("Tensor range check" && 0 <= v1104 && v1104 < 1l);
            int v1106;
            v1106 = 4l * v1104;
            assert("Tensor range check" && 0 <= v1104 && v1104 < 1l);
            int v1107; float v1108;
            Tuple1 tmp28 = Tuple1{0l, 0.0f};
            v1107 = tmp28.v0; v1108 = tmp28.v1;
            while (while_method_2(v1107)){
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
            Tuple1 tmp29 = Tuple1{0l, v1123};
            v1124 = tmp29.v0; v1125 = tmp29.v1;
            while (while_method_2(v1124)){
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
        while (while_method_4(v1133)){
            int v1135;
            v1135 = 0l;
            while (while_method_2(v1135)){
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
        Tuple3 tmp30 = Tuple3{-1.0f / 0.0f, false};
        v1142 = tmp30.v0; v1143 = tmp30.v1;
        int v1144;
        v1144 = 0l;
        while (while_method_4(v1144)){
            int v1146;
            v1146 = 0l;
            while (while_method_2(v1146)){
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
        Tuple3 tmp31 = cooperative_groups::reduce(v1163, Tuple3{v1142, v1143}, v1164);
        v1165 = tmp31.v0; v1166 = tmp31.v1;
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
        while (while_method_4(v1171)){
            int v1173;
            v1173 = 0l;
            while (while_method_2(v1173)){
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
        Tuple0 tmp32 = Tuple0{0.0f, 2147483647l};
        v1179 = tmp32.v0; v1180 = tmp32.v1;
        int v1181;
        v1181 = 0l;
        while (while_method_4(v1181)){
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
        Tuple0 tmp33 = cooperative_groups::reduce(v1195, Tuple0{v1179, v1180}, v1196);
        v1197 = tmp33.v0; v1198 = tmp33.v1;
        float v1199;
        v1199 = v1165 * v1197;
        int v1200[4l];
        bool v1201[4l];
        int v1202;
        v1202 = 0l;
        while (while_method_4(v1202)){
            int v1204;
            v1204 = 0l;
            while (while_method_2(v1204)){
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
        Tuple4 tmp34 = Tuple4{2147483647l, false};
        v1215 = tmp34.v0; v1216 = tmp34.v1;
        int v1217;
        v1217 = 0l;
        while (while_method_4(v1217)){
            int v1219;
            v1219 = 0l;
            while (while_method_2(v1219)){
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
        Tuple4 tmp35 = cooperative_groups::reduce(v1236, Tuple4{v1215, v1216}, v1237);
        v1238 = tmp35.v0; v1239 = tmp35.v1;
        bool v1240;
        v1240 = v1239 == false;
        if (v1240){
            assert("The local reduce must be true." && v1239);
        } else {
        }
        assert("Tensor range check" && 0 <= v1000 && v1000 < 2048l);
        int v1242;
        v1242 = 0l;
        while (while_method_4(v1242)){
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
        assert("Tensor range check" && 0 <= v1000 && v1000 < 2048l);
        int v1249;
        v1249 = 2l * v1000;
        int v1250;
        v1250 = v1249 + v992;
        v14[v1250] = v1238;
        v1000 += 24l ;
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
    while (while_method_3(v1267)){
        bool v1269;
        v1269 = 0l <= v1267;
        bool v1270;
        v1270 = v1269 == false;
        if (v1270){
            assert("The index needs to be zero or positive." && v1269);
        } else {
        }
        bool v1272;
        v1272 = v1267 < 2048l;
        bool v1273;
        v1273 = v1272 == false;
        if (v1273){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1272);
        } else {
        }
        assert("Tensor range check" && 0 <= v1267 && v1267 < 2048l);
        int v1275;
        v1275 = 128l * v1267;
        int v1276;
        v1276 = v1275 + v1265;
        float v1277[4l];
        int v1278[4l];
        int v1279;
        v1279 = 0l;
        while (while_method_4(v1279)){
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
        while (while_method_4(v1286)){
            int v1288;
            v1288 = 0l;
            while (while_method_2(v1288)){
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
        while (while_method_4(v1321)){
            int v1323;
            v1323 = 0l;
            while (while_method_2(v1323)){
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
        while (while_method_4(v1331)){
            int v1333;
            v1333 = 0l;
            while (while_method_2(v1333)){
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
        while (while_method_4(v1340)){
            int v1342;
            v1342 = 0l;
            while (while_method_2(v1342)){
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
        while (while_method_4(v1355)){
            int v1357;
            v1357 = 0l;
            while (while_method_2(v1357)){
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
        while (while_method_4(v1365)){
            int v1367;
            v1367 = 0l;
            while (while_method_2(v1367)){
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
        while (while_method_4(v1382)){
            int v1384;
            v1384 = 0l;
            while (while_method_2(v1384)){
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
        while (while_method_4(v1394)){
            int v1396;
            v1396 = 0l;
            while (while_method_2(v1396)){
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
        while (while_method_4(v1408)){
            int v1410;
            v1410 = 0l;
            while (while_method_2(v1410)){
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
        while (while_method_4(v1418)){
            assert("Tensor range check" && 0 <= v1418 && v1418 < 1l);
            int v1420;
            v1420 = 4l * v1418;
            assert("Tensor range check" && 0 <= v1418 && v1418 < 1l);
            int v1421; float v1422;
            Tuple1 tmp36 = Tuple1{0l, 0.0f};
            v1421 = tmp36.v0; v1422 = tmp36.v1;
            while (while_method_2(v1421)){
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
            Tuple1 tmp37 = Tuple1{0l, v1437};
            v1438 = tmp37.v0; v1439 = tmp37.v1;
            while (while_method_2(v1438)){
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
        while (while_method_4(v1447)){
            int v1449;
            v1449 = 0l;
            while (while_method_2(v1449)){
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
        Tuple3 tmp38 = Tuple3{-1.0f / 0.0f, false};
        v1456 = tmp38.v0; v1457 = tmp38.v1;
        int v1458;
        v1458 = 0l;
        while (while_method_4(v1458)){
            int v1460;
            v1460 = 0l;
            while (while_method_2(v1460)){
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
        Tuple3 tmp39 = cooperative_groups::reduce(v1477, Tuple3{v1456, v1457}, v1478);
        v1479 = tmp39.v0; v1480 = tmp39.v1;
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
        while (while_method_4(v1485)){
            int v1487;
            v1487 = 0l;
            while (while_method_2(v1487)){
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
        Tuple0 tmp40 = Tuple0{0.0f, 2147483647l};
        v1493 = tmp40.v0; v1494 = tmp40.v1;
        int v1495;
        v1495 = 0l;
        while (while_method_4(v1495)){
            int v1497;
            v1497 = 0l;
            while (while_method_2(v1497)){
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
        Tuple0 tmp41 = cooperative_groups::reduce(v1509, Tuple0{v1493, v1494}, v1510);
        v1511 = tmp41.v0; v1512 = tmp41.v1;
        float v1513;
        v1513 = v1479 * v1511;
        int v1514[4l];
        bool v1515[4l];
        int v1516;
        v1516 = 0l;
        while (while_method_4(v1516)){
            int v1518;
            v1518 = 0l;
            while (while_method_2(v1518)){
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
        Tuple4 tmp42 = Tuple4{2147483647l, false};
        v1529 = tmp42.v0; v1530 = tmp42.v1;
        int v1531;
        v1531 = 0l;
        while (while_method_4(v1531)){
            int v1533;
            v1533 = 0l;
            while (while_method_2(v1533)){
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
        Tuple4 tmp43 = cooperative_groups::reduce(v1550, Tuple4{v1529, v1530}, v1551);
        v1552 = tmp43.v0; v1553 = tmp43.v1;
        bool v1554;
        v1554 = v1553 == false;
        if (v1554){
            assert("The local reduce must be true." && v1553);
        } else {
        }
        assert("Tensor range check" && 0 <= v1267 && v1267 < 2048l);
        int v1556;
        v1556 = 0l;
        while (while_method_4(v1556)){
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
        assert("Tensor range check" && 0 <= v1267 && v1267 < 2048l);
        int v1563;
        v1563 = 2l * v1267;
        int v1564;
        v1564 = v1563 + v1259;
        v14[v1564] = v1552;
        v1267 += 24l ;
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
    v1 = v0 < 4096
    del v0
    return v1
def method0(v0 : cp.ndarray) -> None:
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
            v50 = v33 * 4096
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
            v49 = v32 * 4096
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
def method4(v0 : cp.ndarray) -> None:
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
        while method2(v41):
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
            v50 = v33 * 4096
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
def method5(v0 : cp.ndarray) -> None:
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
            v50 = v33 * 4096
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
def method6(v0 : cp.ndarray) -> None:
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
            v50 = v33 * 4096
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
def method8(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
        v38 = v37 >= 262144
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
            v46 = v45 >= 262144
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
            v52 = v35 * 4096
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
def method9(v0 : cp.ndarray) -> None:
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
        v35 = v34 >= 262144
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
            v43 = v42 >= 262144
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
            v49 = v32 * 4096
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
def method10(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
        v38 = v37 >= 262144
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
            v46 = v45 >= 262144
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
            v52 = v35 * 4096
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
def method11(v0 : cp.ndarray) -> None:
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
        v25 = v24 >= 262144
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
def method12(v0 : cp.ndarray) -> None:
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
        v35 = v34 >= 262144
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
            v43 = v42 >= 262144
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
            v49 = v32 * 4096
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
def method13(v0 : cp.ndarray) -> None:
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
            v50 = v33 * 4096
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
def method14(v0 : cp.ndarray) -> None:
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
def method15(v0 : cp.ndarray) -> None:
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
def method16(v0 : cp.ndarray) -> None:
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
def method17(v0 : cp.ndarray) -> None:
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
        while method1(v41):
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
def method19(v0 : cp.ndarray) -> None:
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
def method20(v0 : cp.ndarray) -> None:
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
def method21(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
        v38 = v37 >= 262144
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
            v46 = v45 >= 262144
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
def method22(v0 : cp.ndarray) -> None:
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
        v35 = v34 >= 262144
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
            v43 = v42 >= 262144
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
def method23(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
        v38 = v37 >= 262144
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
            v46 = v45 >= 262144
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
def method24(v0 : cp.ndarray) -> None:
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
        v25 = v24 >= 262144
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
def method25(v0 : cp.ndarray) -> None:
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
        v35 = v34 >= 262144
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
            v43 = v42 >= 262144
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
def method26(v0 : cp.ndarray) -> None:
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
def method27(v0 : cp.ndarray) -> None:
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
def main_body():
    cp.random.seed(12344321)
    v0 = cp.arange(0,262144,1,dtype=cp.int32) # type: ignore
    v1 = v0.size
    v2 = 262144 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,262144,dtype=cp.float32) # type: ignore
    v6 = cp.empty(262144,dtype=cp.int32)
    v7 = cp.empty(262144,dtype=cp.float32)
    v8 = cp.empty(262144,dtype=cp.float32)
    v9 = cp.empty(262144,dtype=cp.float32)
    v10 = cp.empty(262144,dtype=cp.float32)
    v11 = cp.empty(262144,dtype=cp.float32)
    v12 = cp.empty(64,dtype=cp.int32)
    v13 = cp.empty(262144,dtype=cp.int32)
    v14 = cp.empty(262144,dtype=cp.int32)
    v15 = cp.empty(64,dtype=cp.int32)
    v16 = cp.empty(262144,dtype=cp.int32)
    v17 = cp.empty(262144,dtype=cp.float32)
    v18 = cp.empty(64,dtype=cp.int32)
    v19 = cp.cuda.Device().attributes['MultiProcessorCount']
    v20 = v19 == 24
    del v19
    v21 = v20 == False
    if v21:
        v22 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v20, v22
        del v22
    else:
        pass
    del v20, v21
    v23 = 0
    v24 = raw_module.get_function(f"entry{v23}")
    del v23
    v24.max_dynamic_shared_size_bytes = 0 
    v24((24,),(32,),(v0, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18),shared_mem=0)
    del v24
    method0(v5)
    del v5
    method3(v0)
    del v0
    method4(v7)
    del v7
    method5(v8)
    del v8
    method6(v11)
    del v11
    method7(v12)
    del v12
    method8(v9, v10)
    del v9, v10
    method9(v6)
    del v6
    method10(v13, v14)
    del v13, v14
    method11(v15)
    del v15
    method12(v16)
    del v16
    method13(v17)
    del v17
    method14(v18)
    del v18
    cp.random.seed(12344321)
    v25 = cp.arange(0,262144,1,dtype=cp.int32) # type: ignore
    v26 = v25.size
    v27 = 262144 == v26
    del v26
    v28 = v27 == False
    if v28:
        v29 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v27, v29
        del v29
    else:
        pass
    del v27, v28
    v30 = cp.random.normal(0.0,1.0,262144,dtype=cp.float32) # type: ignore
    v31 = cp.empty(262144,dtype=cp.int32)
    v32 = cp.empty(262144,dtype=cp.float32)
    v33 = cp.empty(262144,dtype=cp.float32)
    v34 = cp.empty(262144,dtype=cp.float32)
    v35 = cp.empty(262144,dtype=cp.float32)
    v36 = cp.empty(262144,dtype=cp.float32)
    v37 = cp.empty(4096,dtype=cp.int32)
    v38 = cp.empty(262144,dtype=cp.int32)
    v39 = cp.empty(262144,dtype=cp.int32)
    v40 = cp.empty(4096,dtype=cp.int32)
    v41 = cp.empty(262144,dtype=cp.int32)
    v42 = cp.empty(262144,dtype=cp.float32)
    v43 = cp.empty(4096,dtype=cp.int32)
    v44 = cp.cuda.Device().attributes['MultiProcessorCount']
    v45 = v44 == 24
    del v44
    v46 = v45 == False
    if v46:
        v47 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v45, v47
        del v47
    else:
        pass
    del v45, v46
    v48 = 1
    v49 = raw_module.get_function(f"entry{v48}")
    del v48
    v49.max_dynamic_shared_size_bytes = 0 
    v49((24,),(32,),(v25, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43),shared_mem=0)
    del v49
    method15(v30)
    del v30
    method16(v25)
    del v25
    method17(v32)
    del v32
    method18(v33)
    del v33
    method19(v36)
    del v36
    method20(v37)
    del v37
    method21(v34, v35)
    del v34, v35
    method22(v31)
    del v31
    method23(v38, v39)
    del v38, v39
    method24(v40)
    del v40
    method25(v41)
    del v41
    method26(v42)
    del v42
    return method27(v43)

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
