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
    __shared__ float v44[1l];
    assert("Tensor range check" && 0 <= v43 && v43 < 1l);
    v44[v43] = v41;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v45;
    v45 = threadIdx.x;
    int v46;
    v46 = v45 % 32l;
    bool v47;
    v47 = v43 == 0l;
    bool v49;
    if (v47){
        bool v48;
        v48 = v46 < 1l;
        v49 = v48;
    } else {
        v49 = false;
    }
    if (v49){
        auto v50 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v46 && v46 < 1l);
        float v51;
        v51 = v44[v46];
        float v52;
        v52 = cooperative_groups::reduce(v50, v51, v40);
        v2[0l] = v52;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v53;
    v53 = threadIdx.x;
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
    assert("Tensor range check" && 0 <= v57 && v57 < 32l);
    int v62;
    v62 = 4l * v57;
    int v63;
    v63 = 128l * v58;
    int v64;
    v64 = v63 + v62;
    assert("Tensor range check" && 0 <= v58 && v58 < 1l);
    assert("Tensor range check" && 0 <= v57 && v57 < 32l);
    int v65;
    v65 = 0l;
    while (while_method_2(v65)){
        assert("Tensor range check" && 0 <= v65 && v65 < 64l);
        int v67;
        v67 = 128l * v65;
        int v68;
        v68 = v67 + v64;
        int v69[4l];
        int v70[4l];
        int v71;
        v71 = 0l;
        while (while_method_3(v71)){
            assert("Tensor range check" && 0 <= v71 && v71 < 1l);
            int v73;
            v73 = 4l * v71;
            assert("Tensor range check" && 0 <= v71 && v71 < 1l);
            int v74;
            v74 = 128l * v71;
            int v75;
            v75 = v74 + v68;
            int4* v76;
            v76 = reinterpret_cast<int4*>(v0 + v75);
            int4* v77;
            v77 = reinterpret_cast<int4*>(v69 + v73);
            assert("Pointer alignment check" && (unsigned long long)(v76) % 4l == 0 && (unsigned long long)(v77) % 4l == 0);
            *v77 = *v76;
            v71 += 1l ;
        }
        int v78;
        v78 = 0l;
        while (while_method_3(v78)){
            int v80;
            v80 = 0l;
            while (while_method_1(v80)){
                bool v82;
                v82 = 0l <= v80;
                bool v84;
                if (v82){
                    bool v83;
                    v83 = v80 < 4l;
                    v84 = v83;
                } else {
                    v84 = false;
                }
                bool v85;
                v85 = v84 == false;
                if (v85){
                    assert("The indices should be inside the range of the dimension." && v84);
                } else {
                }
                bool v87;
                v87 = 0l <= v57;
                bool v89;
                if (v87){
                    bool v88;
                    v88 = v57 < 32l;
                    v89 = v88;
                } else {
                    v89 = false;
                }
                bool v90;
                v90 = v89 == false;
                if (v90){
                    assert("The indices should be inside the range of the dimension." && v89);
                } else {
                }
                int v92;
                v92 = v57 * 4l;
                int v93;
                v93 = v80 + v92;
                bool v94;
                v94 = 0l <= v78;
                bool v96;
                if (v94){
                    bool v95;
                    v95 = v78 < 1l;
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
                int v99;
                v99 = v78 * 128l;
                int v100;
                v100 = v93 + v99;
                assert("Tensor range check" && 0 <= v78 && v78 < 1l);
                assert("Tensor range check" && 0 <= v80 && v80 < 4l);
                int v101;
                v101 = 4l * v78;
                int v102;
                v102 = v101 + v80;
                v70[v102] = v100;
                v80 += 1l ;
            }
            v78 += 1l ;
        }
        bool v103;
        v103 = 0l <= v58;
        bool v104;
        v104 = v103 && v59;
        bool v105;
        v105 = v104 == false;
        if (v105){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v104);
        } else {
        }
        bool v107;
        v107 = 0l <= v65;
        bool v109;
        if (v107){
            bool v108;
            v108 = v65 < 64l;
            v109 = v108;
        } else {
            v109 = false;
        }
        bool v110;
        v110 = v109 == false;
        if (v110){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v109);
        } else {
        }
        int v112;
        v112 = v65 + v58;
        assert("Tensor range check" && 0 <= v65 && v65 < 64l);
        int v113;
        v113 = 0l;
        while (while_method_3(v113)){
            assert("Tensor range check" && 0 <= v113 && v113 < 1l);
            int v115;
            v115 = 128l * v113;
            int v116;
            v116 = v115 + v68;
            assert("Tensor range check" && 0 <= v113 && v113 < 1l);
            int v117;
            v117 = 4l * v113;
            int4* v118;
            v118 = reinterpret_cast<int4*>(v69 + v117);
            int4* v119;
            v119 = reinterpret_cast<int4*>(v3 + v116);
            assert("Pointer alignment check" && (unsigned long long)(v118) % 4l == 0 && (unsigned long long)(v119) % 4l == 0);
            *v119 = *v118;
            v113 += 1l ;
        }
        v65 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v120;
    v120 = threadIdx.x;
    bool v121;
    v121 = 0l <= v120;
    bool v122;
    v122 = v121 == false;
    if (v122){
        assert("The index needs to be zero or positive." && v121);
    } else {
    }
    int v124;
    v124 = v120 % 32l;
    int v125;
    v125 = v120 / 32l;
    bool v126;
    v126 = v125 < 1l;
    bool v127;
    v127 = v126 == false;
    if (v127){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v126);
    } else {
    }
    assert("Tensor range check" && 0 <= v125 && v125 < 1l);
    assert("Tensor range check" && 0 <= v124 && v124 < 32l);
    int v129;
    v129 = 4l * v124;
    int v130;
    v130 = 128l * v125;
    int v131;
    v131 = v130 + v129;
    assert("Tensor range check" && 0 <= v125 && v125 < 1l);
    assert("Tensor range check" && 0 <= v124 && v124 < 32l);
    int v132;
    v132 = 0l;
    while (while_method_2(v132)){
        assert("Tensor range check" && 0 <= v132 && v132 < 64l);
        int v134;
        v134 = 128l * v132;
        int v135;
        v135 = v134 + v131;
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
            v141 = 128l * v138;
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
                v154 = 0l <= v124;
                bool v156;
                if (v154){
                    bool v155;
                    v155 = v124 < 32l;
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
                v159 = v124 * 4l;
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
                v166 = v145 * 128l;
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
        v170 = 0l <= v125;
        bool v171;
        v171 = v170 && v126;
        bool v172;
        v172 = v171 == false;
        if (v172){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v171);
        } else {
        }
        bool v174;
        v174 = 0l <= v132;
        bool v176;
        if (v174){
            bool v175;
            v175 = v132 < 64l;
            v176 = v175;
        } else {
            v176 = false;
        }
        bool v177;
        v177 = v176 == false;
        if (v177){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v176);
        } else {
        }
        int v179;
        v179 = v132 + v125;
        int v180[4l];
        int v181[4l];
        int v182;
        v182 = 0l;
        while (while_method_3(v182)){
            int v184;
            v184 = 0l;
            while (while_method_1(v184)){
                assert("Tensor range check" && 0 <= v182 && v182 < 1l);
                assert("Tensor range check" && 0 <= v184 && v184 < 4l);
                int v186;
                v186 = 4l * v182;
                int v187;
                v187 = v186 + v184;
                int v188;
                v188 = v137[v187];
                assert("Tensor range check" && 0 <= v182 && v182 < 1l);
                assert("Tensor range check" && 0 <= v184 && v184 < 4l);
                v180[v187] = v179;
                v181[v187] = v188;
                v184 += 1l ;
            }
            v182 += 1l ;
        }
        assert("Tensor range check" && 0 <= v132 && v132 < 64l);
        int v189;
        v189 = 0l;
        while (while_method_3(v189)){
            assert("Tensor range check" && 0 <= v189 && v189 < 1l);
            int v191;
            v191 = 128l * v189;
            int v192;
            v192 = v191 + v135;
            assert("Tensor range check" && 0 <= v189 && v189 < 1l);
            int v193;
            v193 = 4l * v189;
            int4* v194;
            v194 = reinterpret_cast<int4*>(v180 + v193);
            int4* v195;
            v195 = reinterpret_cast<int4*>(v10 + v192);
            assert("Pointer alignment check" && (unsigned long long)(v194) % 4l == 0 && (unsigned long long)(v195) % 4l == 0);
            *v195 = *v194;
            int4* v196;
            v196 = reinterpret_cast<int4*>(v181 + v193);
            int4* v197;
            v197 = reinterpret_cast<int4*>(v11 + v192);
            assert("Pointer alignment check" && (unsigned long long)(v196) % 4l == 0 && (unsigned long long)(v197) % 4l == 0);
            *v197 = *v196;
            v189 += 1l ;
        }
        v132 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v198;
    v198 = threadIdx.x;
    bool v199;
    v199 = 0l <= v198;
    bool v200;
    v200 = v199 == false;
    if (v200){
        assert("The index needs to be zero or positive." && v199);
    } else {
    }
    int v202;
    v202 = v198 % 32l;
    int v203;
    v203 = v198 / 32l;
    bool v204;
    v204 = v203 < 1l;
    bool v205;
    v205 = v204 == false;
    if (v205){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v204);
    } else {
    }
    assert("Tensor range check" && 0 <= v203 && v203 < 1l);
    assert("Tensor range check" && 0 <= v202 && v202 < 32l);
    int v207;
    v207 = 4l * v202;
    int v208;
    v208 = 128l * v203;
    int v209;
    v209 = v208 + v207;
    assert("Tensor range check" && 0 <= v203 && v203 < 1l);
    int v210;
    v210 = 0l;
    while (while_method_2(v210)){
        assert("Tensor range check" && 0 <= v210 && v210 < 64l);
        int v212;
        v212 = 128l * v210;
        int v213;
        v213 = v212 + v209;
        float v214[4l];
        int v215[4l];
        int v216;
        v216 = 0l;
        while (while_method_3(v216)){
            assert("Tensor range check" && 0 <= v216 && v216 < 1l);
            int v218;
            v218 = 4l * v216;
            assert("Tensor range check" && 0 <= v216 && v216 < 1l);
            int v219;
            v219 = 128l * v216;
            int v220;
            v220 = v219 + v213;
            int4* v221;
            v221 = reinterpret_cast<int4*>(v1 + v220);
            int4* v222;
            v222 = reinterpret_cast<int4*>(v214 + v218);
            assert("Pointer alignment check" && (unsigned long long)(v221) % 4l == 0 && (unsigned long long)(v222) % 4l == 0);
            *v222 = *v221;
            v216 += 1l ;
        }
        int v223;
        v223 = 0l;
        while (while_method_3(v223)){
            int v225;
            v225 = 0l;
            while (while_method_1(v225)){
                bool v227;
                v227 = 0l <= v225;
                bool v229;
                if (v227){
                    bool v228;
                    v228 = v225 < 4l;
                    v229 = v228;
                } else {
                    v229 = false;
                }
                bool v230;
                v230 = v229 == false;
                if (v230){
                    assert("The indices should be inside the range of the dimension." && v229);
                } else {
                }
                bool v232;
                v232 = 0l <= v202;
                bool v234;
                if (v232){
                    bool v233;
                    v233 = v202 < 32l;
                    v234 = v233;
                } else {
                    v234 = false;
                }
                bool v235;
                v235 = v234 == false;
                if (v235){
                    assert("The indices should be inside the range of the dimension." && v234);
                } else {
                }
                int v237;
                v237 = v202 * 4l;
                int v238;
                v238 = v225 + v237;
                bool v239;
                v239 = 0l <= v223;
                bool v241;
                if (v239){
                    bool v240;
                    v240 = v223 < 1l;
                    v241 = v240;
                } else {
                    v241 = false;
                }
                bool v242;
                v242 = v241 == false;
                if (v242){
                    assert("The indices should be inside the range of the dimension." && v241);
                } else {
                }
                int v244;
                v244 = v223 * 128l;
                int v245;
                v245 = v238 + v244;
                assert("Tensor range check" && 0 <= v223 && v223 < 1l);
                assert("Tensor range check" && 0 <= v225 && v225 < 4l);
                int v246;
                v246 = 4l * v223;
                int v247;
                v247 = v246 + v225;
                v215[v247] = v245;
                v225 += 1l ;
            }
            v223 += 1l ;
        }
        bool v248;
        v248 = 0l <= v203;
        bool v249;
        v249 = v248 && v204;
        bool v250;
        v250 = v249 == false;
        if (v250){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v249);
        } else {
        }
        bool v252;
        v252 = 0l <= v210;
        bool v254;
        if (v252){
            bool v253;
            v253 = v210 < 64l;
            v254 = v253;
        } else {
            v254 = false;
        }
        bool v255;
        v255 = v254 == false;
        if (v255){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v254);
        } else {
        }
        int v257;
        v257 = v210 + v203;
        assert("Tensor range check" && 0 <= v210 && v210 < 64l);
        v12[v257] = v257;
        v210 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v258;
    v258 = threadIdx.x;
    bool v259;
    v259 = 0l <= v258;
    bool v260;
    v260 = v259 == false;
    if (v260){
        assert("The index needs to be zero or positive." && v259);
    } else {
    }
    int v262;
    v262 = v258 % 32l;
    int v263;
    v263 = v258 / 32l;
    bool v264;
    v264 = v263 < 1l;
    bool v265;
    v265 = v264 == false;
    if (v265){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v264);
    } else {
    }
    assert("Tensor range check" && 0 <= v263 && v263 < 1l);
    assert("Tensor range check" && 0 <= v262 && v262 < 32l);
    int v267;
    v267 = 4l * v262;
    int v268;
    v268 = 128l * v263;
    int v269;
    v269 = v268 + v267;
    assert("Tensor range check" && 0 <= v263 && v263 < 1l);
    assert("Tensor range check" && 0 <= v262 && v262 < 32l);
    int v270;
    v270 = 0l;
    while (while_method_2(v270)){
        assert("Tensor range check" && 0 <= v270 && v270 < 64l);
        int v272;
        v272 = 128l * v270;
        int v273;
        v273 = v272 + v269;
        float v274[4l];
        int v275[4l];
        int v276;
        v276 = 0l;
        while (while_method_3(v276)){
            assert("Tensor range check" && 0 <= v276 && v276 < 1l);
            int v278;
            v278 = 4l * v276;
            assert("Tensor range check" && 0 <= v276 && v276 < 1l);
            int v279;
            v279 = 128l * v276;
            int v280;
            v280 = v279 + v273;
            int4* v281;
            v281 = reinterpret_cast<int4*>(v1 + v280);
            int4* v282;
            v282 = reinterpret_cast<int4*>(v274 + v278);
            assert("Pointer alignment check" && (unsigned long long)(v281) % 4l == 0 && (unsigned long long)(v282) % 4l == 0);
            *v282 = *v281;
            v276 += 1l ;
        }
        int v283;
        v283 = 0l;
        while (while_method_3(v283)){
            int v285;
            v285 = 0l;
            while (while_method_1(v285)){
                bool v287;
                v287 = 0l <= v285;
                bool v289;
                if (v287){
                    bool v288;
                    v288 = v285 < 4l;
                    v289 = v288;
                } else {
                    v289 = false;
                }
                bool v290;
                v290 = v289 == false;
                if (v290){
                    assert("The indices should be inside the range of the dimension." && v289);
                } else {
                }
                bool v292;
                v292 = 0l <= v262;
                bool v294;
                if (v292){
                    bool v293;
                    v293 = v262 < 32l;
                    v294 = v293;
                } else {
                    v294 = false;
                }
                bool v295;
                v295 = v294 == false;
                if (v295){
                    assert("The indices should be inside the range of the dimension." && v294);
                } else {
                }
                int v297;
                v297 = v262 * 4l;
                int v298;
                v298 = v285 + v297;
                bool v299;
                v299 = 0l <= v283;
                bool v301;
                if (v299){
                    bool v300;
                    v300 = v283 < 1l;
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
                v304 = v283 * 128l;
                int v305;
                v305 = v298 + v304;
                assert("Tensor range check" && 0 <= v283 && v283 < 1l);
                assert("Tensor range check" && 0 <= v285 && v285 < 4l);
                int v306;
                v306 = 4l * v283;
                int v307;
                v307 = v306 + v285;
                v275[v307] = v305;
                v285 += 1l ;
            }
            v283 += 1l ;
        }
        bool v308;
        v308 = 0l <= v263;
        bool v309;
        v309 = v308 && v264;
        bool v310;
        v310 = v309 == false;
        if (v310){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v309);
        } else {
        }
        bool v312;
        v312 = 0l <= v270;
        bool v314;
        if (v312){
            bool v313;
            v313 = v270 < 64l;
            v314 = v313;
        } else {
            v314 = false;
        }
        bool v315;
        v315 = v314 == false;
        if (v315){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v314);
        } else {
        }
        int v317;
        v317 = v270 + v263;
        float v318;
        v318 = 0.0f;
        int v319;
        v319 = 0l;
        while (while_method_3(v319)){
            int v321;
            v321 = 0l;
            while (while_method_1(v321)){
                assert("Tensor range check" && 0 <= v319 && v319 < 1l);
                assert("Tensor range check" && 0 <= v321 && v321 < 4l);
                int v323;
                v323 = 4l * v319;
                int v324;
                v324 = v323 + v321;
                float v325;
                v325 = v274[v324];
                float v326;
                v326 = v318 + v325;
                v318 = v326;
                v321 += 1l ;
            }
            v319 += 1l ;
        }
        auto v327 = cooperative_groups::coalesced_threads();
        int v328;
        v328 = threadIdx.x;
        int v329;
        v329 = v328 / 32l;
        auto v330 = cooperative_groups::labeled_partition(v327,v329);
        float v331;
        v331 = cooperative_groups::reduce(v330, v318, v40);
        float v332;
        v332 = v331 / 128.0f;
        float v333[4l];
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
                v340 = v274[v339];
                float v341;
                v341 = v340 - v332;
                float v342;
                v342 = exp(v341);
                assert("Tensor range check" && 0 <= v334 && v334 < 1l);
                assert("Tensor range check" && 0 <= v336 && v336 < 4l);
                v333[v339] = v342;
                v336 += 1l ;
            }
            v334 += 1l ;
        }
        float v343;
        v343 = 0.0f;
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
                v350 = v333[v349];
                float v351;
                v351 = v343 + v350;
                v343 = v351;
                v346 += 1l ;
            }
            v344 += 1l ;
        }
        auto v352 = cooperative_groups::coalesced_threads();
        int v353;
        v353 = threadIdx.x;
        int v354;
        v354 = v353 / 32l;
        auto v355 = cooperative_groups::labeled_partition(v352,v354);
        float v356;
        v356 = cooperative_groups::reduce(v355, v343, v40);
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
                v364 = v333[v363];
                float v365;
                v365 = v364 / v356;
                assert("Tensor range check" && 0 <= v358 && v358 < 1l);
                assert("Tensor range check" && 0 <= v360 && v360 < 4l);
                v357[v363] = v365;
                v360 += 1l ;
            }
            v358 += 1l ;
        }
        assert("Tensor range check" && 0 <= v270 && v270 < 64l);
        int v366;
        v366 = 0l;
        while (while_method_3(v366)){
            assert("Tensor range check" && 0 <= v366 && v366 < 1l);
            int v368;
            v368 = 128l * v366;
            int v369;
            v369 = v368 + v273;
            assert("Tensor range check" && 0 <= v366 && v366 < 1l);
            int v370;
            v370 = 4l * v366;
            int4* v371;
            v371 = reinterpret_cast<int4*>(v357 + v370);
            int4* v372;
            v372 = reinterpret_cast<int4*>(v4 + v369);
            assert("Pointer alignment check" && (unsigned long long)(v371) % 4l == 0 && (unsigned long long)(v372) % 4l == 0);
            *v372 = *v371;
            v366 += 1l ;
        }
        v270 += 1l ;
    }
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
    v377 = v373 % 32l;
    int v378;
    v378 = v373 / 32l;
    bool v379;
    v379 = v378 < 1l;
    bool v380;
    v380 = v379 == false;
    if (v380){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v379);
    } else {
    }
    assert("Tensor range check" && 0 <= v378 && v378 < 1l);
    assert("Tensor range check" && 0 <= v377 && v377 < 32l);
    int v382;
    v382 = 4l * v377;
    int v383;
    v383 = 128l * v378;
    int v384;
    v384 = v383 + v382;
    assert("Tensor range check" && 0 <= v378 && v378 < 1l);
    assert("Tensor range check" && 0 <= v377 && v377 < 32l);
    int v385;
    v385 = 0l;
    while (while_method_2(v385)){
        assert("Tensor range check" && 0 <= v385 && v385 < 64l);
        int v387;
        v387 = 128l * v385;
        int v388;
        v388 = v387 + v384;
        float v389[4l];
        int v390[4l];
        int v391;
        v391 = 0l;
        while (while_method_3(v391)){
            assert("Tensor range check" && 0 <= v391 && v391 < 1l);
            int v393;
            v393 = 4l * v391;
            assert("Tensor range check" && 0 <= v391 && v391 < 1l);
            int v394;
            v394 = 128l * v391;
            int v395;
            v395 = v394 + v388;
            int4* v396;
            v396 = reinterpret_cast<int4*>(v1 + v395);
            int4* v397;
            v397 = reinterpret_cast<int4*>(v389 + v393);
            assert("Pointer alignment check" && (unsigned long long)(v396) % 4l == 0 && (unsigned long long)(v397) % 4l == 0);
            *v397 = *v396;
            v391 += 1l ;
        }
        int v398;
        v398 = 0l;
        while (while_method_3(v398)){
            int v400;
            v400 = 0l;
            while (while_method_1(v400)){
                bool v402;
                v402 = 0l <= v400;
                bool v404;
                if (v402){
                    bool v403;
                    v403 = v400 < 4l;
                    v404 = v403;
                } else {
                    v404 = false;
                }
                bool v405;
                v405 = v404 == false;
                if (v405){
                    assert("The indices should be inside the range of the dimension." && v404);
                } else {
                }
                bool v407;
                v407 = 0l <= v377;
                bool v409;
                if (v407){
                    bool v408;
                    v408 = v377 < 32l;
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
                v412 = v377 * 4l;
                int v413;
                v413 = v400 + v412;
                bool v414;
                v414 = 0l <= v398;
                bool v416;
                if (v414){
                    bool v415;
                    v415 = v398 < 1l;
                    v416 = v415;
                } else {
                    v416 = false;
                }
                bool v417;
                v417 = v416 == false;
                if (v417){
                    assert("The indices should be inside the range of the dimension." && v416);
                } else {
                }
                int v419;
                v419 = v398 * 128l;
                int v420;
                v420 = v413 + v419;
                assert("Tensor range check" && 0 <= v398 && v398 < 1l);
                assert("Tensor range check" && 0 <= v400 && v400 < 4l);
                int v421;
                v421 = 4l * v398;
                int v422;
                v422 = v421 + v400;
                v390[v422] = v420;
                v400 += 1l ;
            }
            v398 += 1l ;
        }
        bool v423;
        v423 = 0l <= v378;
        bool v424;
        v424 = v423 && v379;
        bool v425;
        v425 = v424 == false;
        if (v425){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v424);
        } else {
        }
        bool v427;
        v427 = 0l <= v385;
        bool v429;
        if (v427){
            bool v428;
            v428 = v385 < 64l;
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
        v432 = v385 + v378;
        float v433[4l];
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
                v440 = v389[v439];
                float v441;
                v441 = v440 * v440;
                assert("Tensor range check" && 0 <= v434 && v434 < 1l);
                assert("Tensor range check" && 0 <= v436 && v436 < 4l);
                v433[v439] = v441;
                v436 += 1l ;
            }
            v434 += 1l ;
        }
        float v442;
        v442 = 0.0f;
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
                v449 = v433[v448];
                float v450;
                v450 = v442 + v449;
                v442 = v450;
                v445 += 1l ;
            }
            v443 += 1l ;
        }
        auto v451 = cooperative_groups::coalesced_threads();
        int v452;
        v452 = threadIdx.x;
        int v453;
        v453 = v452 / 32l;
        auto v454 = cooperative_groups::labeled_partition(v451,v453);
        float v455;
        v455 = cooperative_groups::reduce(v454, v442, v40);
        float v456[4l];
        int v457;
        v457 = 0l;
        while (while_method_3(v457)){
            int v459;
            v459 = 0l;
            while (while_method_1(v459)){
                assert("Tensor range check" && 0 <= v457 && v457 < 1l);
                assert("Tensor range check" && 0 <= v459 && v459 < 4l);
                int v461;
                v461 = 4l * v457;
                int v462;
                v462 = v461 + v459;
                float v463;
                v463 = v389[v462];
                bool v464;
                v464 = v455 == 0.0f;
                bool v465;
                v465 = v464 != true;
                float v467;
                if (v465){
                    float v466;
                    v466 = v463 / v455;
                    v467 = v466;
                } else {
                    v467 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v457 && v457 < 1l);
                assert("Tensor range check" && 0 <= v459 && v459 < 4l);
                v456[v462] = v467;
                v459 += 1l ;
            }
            v457 += 1l ;
        }
        assert("Tensor range check" && 0 <= v385 && v385 < 64l);
        int v468;
        v468 = 0l;
        while (while_method_3(v468)){
            assert("Tensor range check" && 0 <= v468 && v468 < 1l);
            int v470;
            v470 = 128l * v468;
            int v471;
            v471 = v470 + v388;
            assert("Tensor range check" && 0 <= v468 && v468 < 1l);
            int v472;
            v472 = 4l * v468;
            int4* v473;
            v473 = reinterpret_cast<int4*>(v456 + v472);
            int4* v474;
            v474 = reinterpret_cast<int4*>(v8 + v471);
            assert("Pointer alignment check" && (unsigned long long)(v473) % 4l == 0 && (unsigned long long)(v474) % 4l == 0);
            *v474 = *v473;
            v468 += 1l ;
        }
        v385 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v475;
    v475 = threadIdx.x;
    bool v476;
    v476 = 0l <= v475;
    bool v477;
    v477 = v476 == false;
    if (v477){
        assert("The index needs to be zero or positive." && v476);
    } else {
    }
    int v479;
    v479 = v475 % 32l;
    int v480;
    v480 = v475 / 32l;
    bool v481;
    v481 = v480 < 1l;
    bool v482;
    v482 = v481 == false;
    if (v482){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v481);
    } else {
    }
    assert("Tensor range check" && 0 <= v480 && v480 < 1l);
    assert("Tensor range check" && 0 <= v479 && v479 < 32l);
    int v484;
    v484 = 4l * v479;
    int v485;
    v485 = 128l * v480;
    int v486;
    v486 = v485 + v484;
    assert("Tensor range check" && 0 <= v480 && v480 < 1l);
    int v487;
    v487 = 0l;
    while (while_method_2(v487)){
        assert("Tensor range check" && 0 <= v487 && v487 < 64l);
        int v489;
        v489 = 128l * v487;
        int v490;
        v490 = v489 + v486;
        float v491[4l];
        int v492[4l];
        int v493;
        v493 = 0l;
        while (while_method_3(v493)){
            assert("Tensor range check" && 0 <= v493 && v493 < 1l);
            int v495;
            v495 = 4l * v493;
            assert("Tensor range check" && 0 <= v493 && v493 < 1l);
            int v496;
            v496 = 128l * v493;
            int v497;
            v497 = v496 + v490;
            int4* v498;
            v498 = reinterpret_cast<int4*>(v1 + v497);
            int4* v499;
            v499 = reinterpret_cast<int4*>(v491 + v495);
            assert("Pointer alignment check" && (unsigned long long)(v498) % 4l == 0 && (unsigned long long)(v499) % 4l == 0);
            *v499 = *v498;
            v493 += 1l ;
        }
        int v500;
        v500 = 0l;
        while (while_method_3(v500)){
            int v502;
            v502 = 0l;
            while (while_method_1(v502)){
                bool v504;
                v504 = 0l <= v502;
                bool v506;
                if (v504){
                    bool v505;
                    v505 = v502 < 4l;
                    v506 = v505;
                } else {
                    v506 = false;
                }
                bool v507;
                v507 = v506 == false;
                if (v507){
                    assert("The indices should be inside the range of the dimension." && v506);
                } else {
                }
                bool v509;
                v509 = 0l <= v479;
                bool v511;
                if (v509){
                    bool v510;
                    v510 = v479 < 32l;
                    v511 = v510;
                } else {
                    v511 = false;
                }
                bool v512;
                v512 = v511 == false;
                if (v512){
                    assert("The indices should be inside the range of the dimension." && v511);
                } else {
                }
                int v514;
                v514 = v479 * 4l;
                int v515;
                v515 = v502 + v514;
                bool v516;
                v516 = 0l <= v500;
                bool v518;
                if (v516){
                    bool v517;
                    v517 = v500 < 1l;
                    v518 = v517;
                } else {
                    v518 = false;
                }
                bool v519;
                v519 = v518 == false;
                if (v519){
                    assert("The indices should be inside the range of the dimension." && v518);
                } else {
                }
                int v521;
                v521 = v500 * 128l;
                int v522;
                v522 = v515 + v521;
                assert("Tensor range check" && 0 <= v500 && v500 < 1l);
                assert("Tensor range check" && 0 <= v502 && v502 < 4l);
                int v523;
                v523 = 4l * v500;
                int v524;
                v524 = v523 + v502;
                v492[v524] = v522;
                v502 += 1l ;
            }
            v500 += 1l ;
        }
        bool v525;
        v525 = 0l <= v480;
        bool v526;
        v526 = v525 && v481;
        bool v527;
        v527 = v526 == false;
        if (v527){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v526);
        } else {
        }
        bool v529;
        v529 = 0l <= v487;
        bool v531;
        if (v529){
            bool v530;
            v530 = v487 < 64l;
            v531 = v530;
        } else {
            v531 = false;
        }
        bool v532;
        v532 = v531 == false;
        if (v532){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v531);
        } else {
        }
        int v534;
        v534 = v487 + v480;
        float v535; int v536;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0l};
        v535 = tmp1.v0; v536 = tmp1.v1;
        int v537;
        v537 = 0l;
        while (while_method_3(v537)){
            int v539;
            v539 = 0l;
            while (while_method_1(v539)){
                assert("Tensor range check" && 0 <= v537 && v537 < 1l);
                assert("Tensor range check" && 0 <= v539 && v539 < 4l);
                int v541;
                v541 = 4l * v537;
                int v542;
                v542 = v541 + v539;
                float v543;
                v543 = v491[v542];
                int v544;
                v544 = v492[v542];
                bool v545;
                v545 = v535 > v543;
                float v546; int v547;
                if (v545){
                    v546 = v535; v547 = v536;
                } else {
                    v546 = v543; v547 = v544;
                }
                v535 = v546;
                v536 = v547;
                v539 += 1l ;
            }
            v537 += 1l ;
        }
        auto v548 = cooperative_groups::coalesced_threads();
        int v549;
        v549 = threadIdx.x;
        int v550;
        v550 = v549 / 32l;
        auto v551 = cooperative_groups::labeled_partition(v548,v550);
        Closure1 v552{};
        float v553; int v554;
        Tuple1 tmp2 = cooperative_groups::reduce(v551, Tuple1{v535, v536}, v552);
        v553 = tmp2.v0; v554 = tmp2.v1;
        assert("Tensor range check" && 0 <= v487 && v487 < 64l);
        v9[v534] = v554;
        v487 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v555;
    v555 = threadIdx.x;
    bool v556;
    v556 = 0l <= v555;
    bool v557;
    v557 = v556 == false;
    if (v557){
        assert("The index needs to be zero or positive." && v556);
    } else {
    }
    int v559;
    v559 = v555 % 32l;
    int v560;
    v560 = v555 / 32l;
    bool v561;
    v561 = v560 < 1l;
    bool v562;
    v562 = v561 == false;
    if (v562){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v561);
    } else {
    }
    assert("Tensor range check" && 0 <= v560 && v560 < 1l);
    assert("Tensor range check" && 0 <= v559 && v559 < 32l);
    int v564;
    v564 = 4l * v559;
    int v565;
    v565 = 128l * v560;
    int v566;
    v566 = v565 + v564;
    assert("Tensor range check" && 0 <= v560 && v560 < 1l);
    assert("Tensor range check" && 0 <= v559 && v559 < 32l);
    int v567;
    v567 = 0l;
    while (while_method_2(v567)){
        assert("Tensor range check" && 0 <= v567 && v567 < 64l);
        int v569;
        v569 = 128l * v567;
        int v570;
        v570 = v569 + v566;
        float v571[4l];
        int v572[4l];
        int v573;
        v573 = 0l;
        while (while_method_3(v573)){
            assert("Tensor range check" && 0 <= v573 && v573 < 1l);
            int v575;
            v575 = 4l * v573;
            assert("Tensor range check" && 0 <= v573 && v573 < 1l);
            int v576;
            v576 = 128l * v573;
            int v577;
            v577 = v576 + v570;
            int4* v578;
            v578 = reinterpret_cast<int4*>(v1 + v577);
            int4* v579;
            v579 = reinterpret_cast<int4*>(v571 + v575);
            assert("Pointer alignment check" && (unsigned long long)(v578) % 4l == 0 && (unsigned long long)(v579) % 4l == 0);
            *v579 = *v578;
            v573 += 1l ;
        }
        int v580;
        v580 = 0l;
        while (while_method_3(v580)){
            int v582;
            v582 = 0l;
            while (while_method_1(v582)){
                bool v584;
                v584 = 0l <= v582;
                bool v586;
                if (v584){
                    bool v585;
                    v585 = v582 < 4l;
                    v586 = v585;
                } else {
                    v586 = false;
                }
                bool v587;
                v587 = v586 == false;
                if (v587){
                    assert("The indices should be inside the range of the dimension." && v586);
                } else {
                }
                bool v589;
                v589 = 0l <= v559;
                bool v591;
                if (v589){
                    bool v590;
                    v590 = v559 < 32l;
                    v591 = v590;
                } else {
                    v591 = false;
                }
                bool v592;
                v592 = v591 == false;
                if (v592){
                    assert("The indices should be inside the range of the dimension." && v591);
                } else {
                }
                int v594;
                v594 = v559 * 4l;
                int v595;
                v595 = v582 + v594;
                bool v596;
                v596 = 0l <= v580;
                bool v598;
                if (v596){
                    bool v597;
                    v597 = v580 < 1l;
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
                int v601;
                v601 = v580 * 128l;
                int v602;
                v602 = v595 + v601;
                assert("Tensor range check" && 0 <= v580 && v580 < 1l);
                assert("Tensor range check" && 0 <= v582 && v582 < 4l);
                int v603;
                v603 = 4l * v580;
                int v604;
                v604 = v603 + v582;
                v572[v604] = v602;
                v582 += 1l ;
            }
            v580 += 1l ;
        }
        bool v605;
        v605 = 0l <= v560;
        bool v606;
        v606 = v605 && v561;
        bool v607;
        v607 = v606 == false;
        if (v607){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v606);
        } else {
        }
        bool v609;
        v609 = 0l <= v567;
        bool v611;
        if (v609){
            bool v610;
            v610 = v567 < 64l;
            v611 = v610;
        } else {
            v611 = false;
        }
        bool v612;
        v612 = v611 == false;
        if (v612){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v611);
        } else {
        }
        int v614;
        v614 = v567 + v560;
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
                v622 = v571[v621];
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
        float v628;
        v628 = cooperative_groups::reduce(v627, v615, v40);
        float v629;
        v629 = v628 / 128.0f;
        float v630[4l];
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
                v637 = v571[v636];
                float v638;
                v638 = v637 - v629;
                float v639;
                v639 = exp(v638);
                assert("Tensor range check" && 0 <= v631 && v631 < 1l);
                assert("Tensor range check" && 0 <= v633 && v633 < 4l);
                v630[v636] = v639;
                v633 += 1l ;
            }
            v631 += 1l ;
        }
        float v640;
        v640 = 0.0f;
        int v641;
        v641 = 0l;
        while (while_method_3(v641)){
            int v643;
            v643 = 0l;
            while (while_method_1(v643)){
                assert("Tensor range check" && 0 <= v641 && v641 < 1l);
                assert("Tensor range check" && 0 <= v643 && v643 < 4l);
                int v645;
                v645 = 4l * v641;
                int v646;
                v646 = v645 + v643;
                float v647;
                v647 = v630[v646];
                float v648;
                v648 = v640 + v647;
                v640 = v648;
                v643 += 1l ;
            }
            v641 += 1l ;
        }
        auto v649 = cooperative_groups::coalesced_threads();
        int v650;
        v650 = threadIdx.x;
        int v651;
        v651 = v650 / 32l;
        auto v652 = cooperative_groups::labeled_partition(v649,v651);
        float v653;
        v653 = cooperative_groups::reduce(v652, v640, v40);
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
                v661 = v630[v660];
                float v662;
                v662 = v661 / v653;
                assert("Tensor range check" && 0 <= v655 && v655 < 1l);
                assert("Tensor range check" && 0 <= v657 && v657 < 4l);
                v654[v660] = v662;
                v657 += 1l ;
            }
            v655 += 1l ;
        }
        float v663[4l];
        float v664;
        v664 = 0.0f;
        int v665;
        v665 = 0l;
        while (while_method_3(v665)){
            assert("Tensor range check" && 0 <= v665 && v665 < 1l);
            int v667;
            v667 = 4l * v665;
            assert("Tensor range check" && 0 <= v665 && v665 < 1l);
            int v668; float v669;
            Tuple0 tmp3 = Tuple0{0l, 0.0f};
            v668 = tmp3.v0; v669 = tmp3.v1;
            while (while_method_1(v668)){
                assert("Tensor range check" && 0 <= v668 && v668 < 4l);
                int v671;
                v671 = v668 + v667;
                float v672;
                v672 = v654[v671];
                float v673;
                v673 = v669 + v672;
                v669 = v673;
                v668 += 1l ;
            }
            auto v674 = cooperative_groups::coalesced_threads();
            int v675;
            v675 = threadIdx.x;
            int v676;
            v676 = v675 / 32l;
            auto v677 = cooperative_groups::labeled_partition(v674,v676);
            Closure2 v678{};
            float v679;
            v679 = cooperative_groups::inclusive_scan(v677, v669, v678);
            float v680;
            v680 = v677.shfl_up(v679,1);
            bool v681;
            v681 = v677.thread_rank() == 0;
            float v682;
            if (v681){
                v682 = 0.0f;
            } else {
                v682 = v680;
            }
            float v683;
            v683 = v677.shfl(v679,v677.num_threads()-1);
            float v684;
            v684 = v664 + v682;
            int v685; float v686;
            Tuple0 tmp4 = Tuple0{0l, v684};
            v685 = tmp4.v0; v686 = tmp4.v1;
            while (while_method_1(v685)){
                assert("Tensor range check" && 0 <= v685 && v685 < 4l);
                int v688;
                v688 = v685 + v667;
                float v689;
                v689 = v654[v688];
                float v690;
                v690 = v686 + v689;
                assert("Tensor range check" && 0 <= v685 && v685 < 4l);
                v663[v688] = v690;
                v686 = v690;
                v685 += 1l ;
            }
            float v691;
            v691 = v664 + v683;
            v664 = v691;
            v665 += 1l ;
        }
        assert("Tensor range check" && 0 <= v567 && v567 < 64l);
        int v692;
        v692 = 0l;
        while (while_method_3(v692)){
            assert("Tensor range check" && 0 <= v692 && v692 < 1l);
            int v694;
            v694 = 128l * v692;
            int v695;
            v695 = v694 + v570;
            assert("Tensor range check" && 0 <= v692 && v692 < 1l);
            int v696;
            v696 = 4l * v692;
            int4* v697;
            v697 = reinterpret_cast<int4*>(v654 + v696);
            int4* v698;
            v698 = reinterpret_cast<int4*>(v6 + v695);
            assert("Pointer alignment check" && (unsigned long long)(v697) % 4l == 0 && (unsigned long long)(v698) % 4l == 0);
            *v698 = *v697;
            int4* v699;
            v699 = reinterpret_cast<int4*>(v663 + v696);
            int4* v700;
            v700 = reinterpret_cast<int4*>(v7 + v695);
            assert("Pointer alignment check" && (unsigned long long)(v699) % 4l == 0 && (unsigned long long)(v700) % 4l == 0);
            *v700 = *v699;
            v692 += 1l ;
        }
        v567 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v701;
    v701 = threadIdx.x;
    bool v702;
    v702 = 0l <= v701;
    bool v703;
    v703 = v702 == false;
    if (v703){
        assert("The index needs to be zero or positive." && v702);
    } else {
    }
    int v705;
    v705 = v701 % 32l;
    int v706;
    v706 = v701 / 32l;
    bool v707;
    v707 = v706 < 1l;
    bool v708;
    v708 = v707 == false;
    if (v708){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v707);
    } else {
    }
    assert("Tensor range check" && 0 <= v706 && v706 < 1l);
    assert("Tensor range check" && 0 <= v705 && v705 < 32l);
    int v710;
    v710 = 4l * v705;
    int v711;
    v711 = 128l * v706;
    int v712;
    v712 = v711 + v710;
    assert("Tensor range check" && 0 <= v706 && v706 < 1l);
    assert("Tensor range check" && 0 <= v705 && v705 < 32l);
    int v713;
    v713 = 0l;
    while (while_method_2(v713)){
        assert("Tensor range check" && 0 <= v713 && v713 < 64l);
        int v715;
        v715 = 128l * v713;
        int v716;
        v716 = v715 + v712;
        int v717[4l];
        int v718[4l];
        int v719;
        v719 = 0l;
        while (while_method_3(v719)){
            assert("Tensor range check" && 0 <= v719 && v719 < 1l);
            int v721;
            v721 = 4l * v719;
            assert("Tensor range check" && 0 <= v719 && v719 < 1l);
            int v722;
            v722 = 128l * v719;
            int v723;
            v723 = v722 + v716;
            int4* v724;
            v724 = reinterpret_cast<int4*>(v0 + v723);
            int4* v725;
            v725 = reinterpret_cast<int4*>(v717 + v721);
            assert("Pointer alignment check" && (unsigned long long)(v724) % 4l == 0 && (unsigned long long)(v725) % 4l == 0);
            *v725 = *v724;
            v719 += 1l ;
        }
        int v726;
        v726 = 0l;
        while (while_method_3(v726)){
            int v728;
            v728 = 0l;
            while (while_method_1(v728)){
                bool v730;
                v730 = 0l <= v728;
                bool v732;
                if (v730){
                    bool v731;
                    v731 = v728 < 4l;
                    v732 = v731;
                } else {
                    v732 = false;
                }
                bool v733;
                v733 = v732 == false;
                if (v733){
                    assert("The indices should be inside the range of the dimension." && v732);
                } else {
                }
                bool v735;
                v735 = 0l <= v705;
                bool v737;
                if (v735){
                    bool v736;
                    v736 = v705 < 32l;
                    v737 = v736;
                } else {
                    v737 = false;
                }
                bool v738;
                v738 = v737 == false;
                if (v738){
                    assert("The indices should be inside the range of the dimension." && v737);
                } else {
                }
                int v740;
                v740 = v705 * 4l;
                int v741;
                v741 = v728 + v740;
                bool v742;
                v742 = 0l <= v726;
                bool v744;
                if (v742){
                    bool v743;
                    v743 = v726 < 1l;
                    v744 = v743;
                } else {
                    v744 = false;
                }
                bool v745;
                v745 = v744 == false;
                if (v745){
                    assert("The indices should be inside the range of the dimension." && v744);
                } else {
                }
                int v747;
                v747 = v726 * 128l;
                int v748;
                v748 = v741 + v747;
                assert("Tensor range check" && 0 <= v726 && v726 < 1l);
                assert("Tensor range check" && 0 <= v728 && v728 < 4l);
                int v749;
                v749 = 4l * v726;
                int v750;
                v750 = v749 + v728;
                v718[v750] = v748;
                v728 += 1l ;
            }
            v726 += 1l ;
        }
        bool v751;
        v751 = 0l <= v706;
        bool v752;
        v752 = v751 && v707;
        bool v753;
        v753 = v752 == false;
        if (v753){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v752);
        } else {
        }
        bool v755;
        v755 = 0l <= v713;
        bool v757;
        if (v755){
            bool v756;
            v756 = v713 < 64l;
            v757 = v756;
        } else {
            v757 = false;
        }
        bool v758;
        v758 = v757 == false;
        if (v758){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v757);
        } else {
        }
        int v760;
        v760 = v713 + v706;
        int v761[4l];
        int v762;
        v762 = 0l;
        int v763;
        v763 = 0l;
        while (while_method_3(v763)){
            assert("Tensor range check" && 0 <= v763 && v763 < 1l);
            int v765;
            v765 = 4l * v763;
            assert("Tensor range check" && 0 <= v763 && v763 < 1l);
            int v766; int v767;
            Tuple2 tmp5 = Tuple2{0l, 0l};
            v766 = tmp5.v0; v767 = tmp5.v1;
            while (while_method_1(v766)){
                assert("Tensor range check" && 0 <= v766 && v766 < 4l);
                int v769;
                v769 = v766 + v765;
                int v770;
                v770 = v717[v769];
                int v771;
                v771 = v767 + v770;
                v767 = v771;
                v766 += 1l ;
            }
            auto v772 = cooperative_groups::coalesced_threads();
            int v773;
            v773 = threadIdx.x;
            int v774;
            v774 = v773 / 32l;
            auto v775 = cooperative_groups::labeled_partition(v772,v774);
            Closure3 v776{};
            int v777;
            v777 = cooperative_groups::inclusive_scan(v775, v767, v776);
            int v778;
            v778 = v775.shfl_up(v777,1);
            bool v779;
            v779 = v775.thread_rank() == 0;
            int v780;
            if (v779){
                v780 = 0l;
            } else {
                v780 = v778;
            }
            int v781;
            v781 = v775.shfl(v777,v775.num_threads()-1);
            int v782;
            v782 = v762 + v780;
            int v783; int v784;
            Tuple2 tmp6 = Tuple2{0l, v782};
            v783 = tmp6.v0; v784 = tmp6.v1;
            while (while_method_1(v783)){
                assert("Tensor range check" && 0 <= v783 && v783 < 4l);
                int v786;
                v786 = v783 + v765;
                int v787;
                v787 = v717[v786];
                assert("Tensor range check" && 0 <= v783 && v783 < 4l);
                v761[v786] = v784;
                int v788;
                v788 = v784 + v787;
                v784 = v788;
                v783 += 1l ;
            }
            int v789;
            v789 = v762 + v781;
            v762 = v789;
            v763 += 1l ;
        }
        assert("Tensor range check" && 0 <= v713 && v713 < 64l);
        int v790;
        v790 = 0l;
        while (while_method_3(v790)){
            assert("Tensor range check" && 0 <= v790 && v790 < 1l);
            int v792;
            v792 = 128l * v790;
            int v793;
            v793 = v792 + v716;
            assert("Tensor range check" && 0 <= v790 && v790 < 1l);
            int v794;
            v794 = 4l * v790;
            int4* v795;
            v795 = reinterpret_cast<int4*>(v761 + v794);
            int4* v796;
            v796 = reinterpret_cast<int4*>(v13 + v793);
            assert("Pointer alignment check" && (unsigned long long)(v795) % 4l == 0 && (unsigned long long)(v796) % 4l == 0);
            *v796 = *v795;
            v790 += 1l ;
        }
        v713 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v797;
    v797 = threadIdx.x;
    bool v798;
    v798 = 0l <= v797;
    bool v799;
    v799 = v798 == false;
    if (v799){
        assert("The index needs to be zero or positive." && v798);
    } else {
    }
    int v801;
    v801 = v797 % 32l;
    int v802;
    v802 = v797 / 32l;
    bool v803;
    v803 = v802 < 1l;
    bool v804;
    v804 = v803 == false;
    if (v804){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v803);
    } else {
    }
    assert("Tensor range check" && 0 <= v802 && v802 < 1l);
    assert("Tensor range check" && 0 <= v801 && v801 < 32l);
    int v806;
    v806 = 4l * v801;
    int v807;
    v807 = 128l * v802;
    int v808;
    v808 = v807 + v806;
    assert("Tensor range check" && 0 <= v802 && v802 < 1l);
    assert("Tensor range check" && 0 <= v801 && v801 < 32l);
    int v809;
    v809 = 0l;
    while (while_method_2(v809)){
        assert("Tensor range check" && 0 <= v809 && v809 < 64l);
        int v811;
        v811 = 128l * v809;
        int v812;
        v812 = v811 + v808;
        float v813[4l];
        int v814[4l];
        int v815;
        v815 = 0l;
        while (while_method_3(v815)){
            assert("Tensor range check" && 0 <= v815 && v815 < 1l);
            int v817;
            v817 = 4l * v815;
            assert("Tensor range check" && 0 <= v815 && v815 < 1l);
            int v818;
            v818 = 128l * v815;
            int v819;
            v819 = v818 + v812;
            int4* v820;
            v820 = reinterpret_cast<int4*>(v1 + v819);
            int4* v821;
            v821 = reinterpret_cast<int4*>(v813 + v817);
            assert("Pointer alignment check" && (unsigned long long)(v820) % 4l == 0 && (unsigned long long)(v821) % 4l == 0);
            *v821 = *v820;
            v815 += 1l ;
        }
        int v822;
        v822 = 0l;
        while (while_method_3(v822)){
            int v824;
            v824 = 0l;
            while (while_method_1(v824)){
                bool v826;
                v826 = 0l <= v824;
                bool v828;
                if (v826){
                    bool v827;
                    v827 = v824 < 4l;
                    v828 = v827;
                } else {
                    v828 = false;
                }
                bool v829;
                v829 = v828 == false;
                if (v829){
                    assert("The indices should be inside the range of the dimension." && v828);
                } else {
                }
                bool v831;
                v831 = 0l <= v801;
                bool v833;
                if (v831){
                    bool v832;
                    v832 = v801 < 32l;
                    v833 = v832;
                } else {
                    v833 = false;
                }
                bool v834;
                v834 = v833 == false;
                if (v834){
                    assert("The indices should be inside the range of the dimension." && v833);
                } else {
                }
                int v836;
                v836 = v801 * 4l;
                int v837;
                v837 = v824 + v836;
                bool v838;
                v838 = 0l <= v822;
                bool v840;
                if (v838){
                    bool v839;
                    v839 = v822 < 1l;
                    v840 = v839;
                } else {
                    v840 = false;
                }
                bool v841;
                v841 = v840 == false;
                if (v841){
                    assert("The indices should be inside the range of the dimension." && v840);
                } else {
                }
                int v843;
                v843 = v822 * 128l;
                int v844;
                v844 = v837 + v843;
                assert("Tensor range check" && 0 <= v822 && v822 < 1l);
                assert("Tensor range check" && 0 <= v824 && v824 < 4l);
                int v845;
                v845 = 4l * v822;
                int v846;
                v846 = v845 + v824;
                v814[v846] = v844;
                v824 += 1l ;
            }
            v822 += 1l ;
        }
        bool v847;
        v847 = 0l <= v802;
        bool v848;
        v848 = v847 && v803;
        bool v849;
        v849 = v848 == false;
        if (v849){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v848);
        } else {
        }
        bool v851;
        v851 = 0l <= v809;
        bool v853;
        if (v851){
            bool v852;
            v852 = v809 < 64l;
            v853 = v852;
        } else {
            v853 = false;
        }
        bool v854;
        v854 = v853 == false;
        if (v854){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v853);
        } else {
        }
        int v856;
        v856 = v809 + v802;
        bool v857[4l];
        int v858;
        v858 = 0l;
        while (while_method_3(v858)){
            int v860;
            v860 = 0l;
            while (while_method_1(v860)){
                assert("Tensor range check" && 0 <= v858 && v858 < 1l);
                assert("Tensor range check" && 0 <= v860 && v860 < 4l);
                int v862;
                v862 = 4l * v858;
                int v863;
                v863 = v862 + v860;
                float v864;
                v864 = v813[v863];
                int v865;
                v865 = v814[v863];
                bool v866;
                v866 = v865 < 4l;
                assert("Tensor range check" && 0 <= v858 && v858 < 1l);
                assert("Tensor range check" && 0 <= v860 && v860 < 4l);
                v857[v863] = v866;
                v860 += 1l ;
            }
            v858 += 1l ;
        }
        int v867[4l];
        int v868;
        v868 = 0l;
        while (while_method_3(v868)){
            int v870;
            v870 = 0l;
            while (while_method_1(v870)){
                assert("Tensor range check" && 0 <= v868 && v868 < 1l);
                assert("Tensor range check" && 0 <= v870 && v870 < 4l);
                int v872;
                v872 = 4l * v868;
                int v873;
                v873 = v872 + v870;
                bool v874;
                v874 = v857[v873];
                int v875;
                if (v874){
                    v875 = 1l;
                } else {
                    v875 = 0l;
                }
                assert("Tensor range check" && 0 <= v868 && v868 < 1l);
                assert("Tensor range check" && 0 <= v870 && v870 < 4l);
                v867[v873] = v875;
                v870 += 1l ;
            }
            v868 += 1l ;
        }
        int v876;
        v876 = 0l;
        int v877;
        v877 = 0l;
        while (while_method_3(v877)){
            int v879;
            v879 = 0l;
            while (while_method_1(v879)){
                assert("Tensor range check" && 0 <= v877 && v877 < 1l);
                assert("Tensor range check" && 0 <= v879 && v879 < 4l);
                int v881;
                v881 = 4l * v877;
                int v882;
                v882 = v881 + v879;
                int v883;
                v883 = v867[v882];
                int v884;
                v884 = v876 + v883;
                v876 = v884;
                v879 += 1l ;
            }
            v877 += 1l ;
        }
        auto v885 = cooperative_groups::coalesced_threads();
        int v886;
        v886 = threadIdx.x;
        int v887;
        v887 = v886 / 32l;
        auto v888 = cooperative_groups::labeled_partition(v885,v887);
        Closure4 v889{};
        int v890;
        v890 = cooperative_groups::reduce(v888, v876, v889);
        float v891[4l];
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
                float v898;
                v898 = v813[v897];
                bool v899;
                v899 = v857[v897];
                float v900;
                if (v899){
                    v900 = v898;
                } else {
                    v900 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v892 && v892 < 1l);
                assert("Tensor range check" && 0 <= v894 && v894 < 4l);
                v891[v897] = v900;
                v894 += 1l ;
            }
            v892 += 1l ;
        }
        float v901;
        v901 = 0.0f;
        int v902;
        v902 = 0l;
        while (while_method_3(v902)){
            int v904;
            v904 = 0l;
            while (while_method_1(v904)){
                assert("Tensor range check" && 0 <= v902 && v902 < 1l);
                assert("Tensor range check" && 0 <= v904 && v904 < 4l);
                int v906;
                v906 = 4l * v902;
                int v907;
                v907 = v906 + v904;
                float v908;
                v908 = v891[v907];
                float v909;
                v909 = v901 + v908;
                v901 = v909;
                v904 += 1l ;
            }
            v902 += 1l ;
        }
        auto v910 = cooperative_groups::coalesced_threads();
        int v911;
        v911 = threadIdx.x;
        int v912;
        v912 = v911 / 32l;
        auto v913 = cooperative_groups::labeled_partition(v910,v912);
        float v914;
        v914 = cooperative_groups::reduce(v913, v901, v40);
        float v915;
        v915 = (float)v890;
        float v916;
        v916 = v914 / v915;
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
                v924 = v813[v923];
                bool v925;
                v925 = v857[v923];
                float v926;
                if (v925){
                    v926 = v924;
                } else {
                    v926 = -1.0f / 0.0f;
                }
                float v927;
                v927 = v926 - v916;
                float v928;
                v928 = exp(v927);
                assert("Tensor range check" && 0 <= v918 && v918 < 1l);
                assert("Tensor range check" && 0 <= v920 && v920 < 4l);
                v917[v923] = v928;
                v920 += 1l ;
            }
            v918 += 1l ;
        }
        float v929;
        v929 = 0.0f;
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
                v936 = v917[v935];
                float v937;
                v937 = v929 + v936;
                v929 = v937;
                v932 += 1l ;
            }
            v930 += 1l ;
        }
        auto v938 = cooperative_groups::coalesced_threads();
        int v939;
        v939 = threadIdx.x;
        int v940;
        v940 = v939 / 32l;
        auto v941 = cooperative_groups::labeled_partition(v938,v940);
        float v942;
        v942 = cooperative_groups::reduce(v941, v929, v40);
        float v943[4l];
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
                v950 = v917[v949];
                float v951;
                v951 = v950 / v942;
                assert("Tensor range check" && 0 <= v944 && v944 < 1l);
                assert("Tensor range check" && 0 <= v946 && v946 < 4l);
                v943[v949] = v951;
                v946 += 1l ;
            }
            v944 += 1l ;
        }
        assert("Tensor range check" && 0 <= v809 && v809 < 64l);
        int v952;
        v952 = 0l;
        while (while_method_3(v952)){
            assert("Tensor range check" && 0 <= v952 && v952 < 1l);
            int v954;
            v954 = 128l * v952;
            int v955;
            v955 = v954 + v812;
            assert("Tensor range check" && 0 <= v952 && v952 < 1l);
            int v956;
            v956 = 4l * v952;
            int4* v957;
            v957 = reinterpret_cast<int4*>(v943 + v956);
            int4* v958;
            v958 = reinterpret_cast<int4*>(v5 + v955);
            assert("Pointer alignment check" && (unsigned long long)(v957) % 4l == 0 && (unsigned long long)(v958) % 4l == 0);
            *v958 = *v957;
            v952 += 1l ;
        }
        v809 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v959;
    v959 = threadIdx.x;
    unsigned long long v960;
    v960 = (unsigned long long)v959;
    curandStatePhilox4_32_10_t v961;
    curand_init(12344321ull,v960,0ull,&v961);
    int v962;
    v962 = threadIdx.x;
    bool v963;
    v963 = 0l <= v962;
    bool v964;
    v964 = v963 == false;
    if (v964){
        assert("The index needs to be zero or positive." && v963);
    } else {
    }
    int v966;
    v966 = v962 % 32l;
    int v967;
    v967 = v962 / 32l;
    bool v968;
    v968 = v967 < 1l;
    bool v969;
    v969 = v968 == false;
    if (v969){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v968);
    } else {
    }
    assert("Tensor range check" && 0 <= v967 && v967 < 1l);
    assert("Tensor range check" && 0 <= v966 && v966 < 32l);
    int v971;
    v971 = 4l * v966;
    int v972;
    v972 = 128l * v967;
    int v973;
    v973 = v972 + v971;
    assert("Tensor range check" && 0 <= v967 && v967 < 1l);
    assert("Tensor range check" && 0 <= v966 && v966 < 32l);
    assert("Tensor range check" && 0 <= v967 && v967 < 1l);
    int v974;
    v974 = 0l;
    while (while_method_2(v974)){
        assert("Tensor range check" && 0 <= v974 && v974 < 64l);
        int v976;
        v976 = 128l * v974;
        int v977;
        v977 = v976 + v973;
        float v978[4l];
        int v979[4l];
        int v980;
        v980 = 0l;
        while (while_method_3(v980)){
            assert("Tensor range check" && 0 <= v980 && v980 < 1l);
            int v982;
            v982 = 4l * v980;
            assert("Tensor range check" && 0 <= v980 && v980 < 1l);
            int v983;
            v983 = 128l * v980;
            int v984;
            v984 = v983 + v977;
            int4* v985;
            v985 = reinterpret_cast<int4*>(v1 + v984);
            int4* v986;
            v986 = reinterpret_cast<int4*>(v978 + v982);
            assert("Pointer alignment check" && (unsigned long long)(v985) % 4l == 0 && (unsigned long long)(v986) % 4l == 0);
            *v986 = *v985;
            v980 += 1l ;
        }
        int v987;
        v987 = 0l;
        while (while_method_3(v987)){
            int v989;
            v989 = 0l;
            while (while_method_1(v989)){
                bool v991;
                v991 = 0l <= v989;
                bool v993;
                if (v991){
                    bool v992;
                    v992 = v989 < 4l;
                    v993 = v992;
                } else {
                    v993 = false;
                }
                bool v994;
                v994 = v993 == false;
                if (v994){
                    assert("The indices should be inside the range of the dimension." && v993);
                } else {
                }
                bool v996;
                v996 = 0l <= v966;
                bool v998;
                if (v996){
                    bool v997;
                    v997 = v966 < 32l;
                    v998 = v997;
                } else {
                    v998 = false;
                }
                bool v999;
                v999 = v998 == false;
                if (v999){
                    assert("The indices should be inside the range of the dimension." && v998);
                } else {
                }
                int v1001;
                v1001 = v966 * 4l;
                int v1002;
                v1002 = v989 + v1001;
                bool v1003;
                v1003 = 0l <= v987;
                bool v1005;
                if (v1003){
                    bool v1004;
                    v1004 = v987 < 1l;
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
                v1008 = v987 * 128l;
                int v1009;
                v1009 = v1002 + v1008;
                assert("Tensor range check" && 0 <= v987 && v987 < 1l);
                assert("Tensor range check" && 0 <= v989 && v989 < 4l);
                int v1010;
                v1010 = 4l * v987;
                int v1011;
                v1011 = v1010 + v989;
                v979[v1011] = v1009;
                v989 += 1l ;
            }
            v987 += 1l ;
        }
        bool v1012;
        v1012 = 0l <= v967;
        bool v1013;
        v1013 = v1012 && v968;
        bool v1014;
        v1014 = v1013 == false;
        if (v1014){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1013);
        } else {
        }
        bool v1016;
        v1016 = 0l <= v974;
        bool v1018;
        if (v1016){
            bool v1017;
            v1017 = v974 < 64l;
            v1018 = v1017;
        } else {
            v1018 = false;
        }
        bool v1019;
        v1019 = v1018 == false;
        if (v1019){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1018);
        } else {
        }
        int v1021;
        v1021 = v974 + v967;
        float v1022;
        v1022 = 0.0f;
        int v1023;
        v1023 = 0l;
        while (while_method_3(v1023)){
            int v1025;
            v1025 = 0l;
            while (while_method_1(v1025)){
                assert("Tensor range check" && 0 <= v1023 && v1023 < 1l);
                assert("Tensor range check" && 0 <= v1025 && v1025 < 4l);
                int v1027;
                v1027 = 4l * v1023;
                int v1028;
                v1028 = v1027 + v1025;
                float v1029;
                v1029 = v978[v1028];
                float v1030;
                v1030 = v1022 + v1029;
                v1022 = v1030;
                v1025 += 1l ;
            }
            v1023 += 1l ;
        }
        auto v1031 = cooperative_groups::coalesced_threads();
        int v1032;
        v1032 = threadIdx.x;
        int v1033;
        v1033 = v1032 / 32l;
        auto v1034 = cooperative_groups::labeled_partition(v1031,v1033);
        float v1035;
        v1035 = cooperative_groups::reduce(v1034, v1022, v40);
        float v1036;
        v1036 = v1035 / 128.0f;
        float v1037[4l];
        int v1038;
        v1038 = 0l;
        while (while_method_3(v1038)){
            int v1040;
            v1040 = 0l;
            while (while_method_1(v1040)){
                assert("Tensor range check" && 0 <= v1038 && v1038 < 1l);
                assert("Tensor range check" && 0 <= v1040 && v1040 < 4l);
                int v1042;
                v1042 = 4l * v1038;
                int v1043;
                v1043 = v1042 + v1040;
                float v1044;
                v1044 = v978[v1043];
                float v1045;
                v1045 = v1044 - v1036;
                float v1046;
                v1046 = exp(v1045);
                assert("Tensor range check" && 0 <= v1038 && v1038 < 1l);
                assert("Tensor range check" && 0 <= v1040 && v1040 < 4l);
                v1037[v1043] = v1046;
                v1040 += 1l ;
            }
            v1038 += 1l ;
        }
        float v1047;
        v1047 = 0.0f;
        int v1048;
        v1048 = 0l;
        while (while_method_3(v1048)){
            int v1050;
            v1050 = 0l;
            while (while_method_1(v1050)){
                assert("Tensor range check" && 0 <= v1048 && v1048 < 1l);
                assert("Tensor range check" && 0 <= v1050 && v1050 < 4l);
                int v1052;
                v1052 = 4l * v1048;
                int v1053;
                v1053 = v1052 + v1050;
                float v1054;
                v1054 = v1037[v1053];
                float v1055;
                v1055 = v1047 + v1054;
                v1047 = v1055;
                v1050 += 1l ;
            }
            v1048 += 1l ;
        }
        auto v1056 = cooperative_groups::coalesced_threads();
        int v1057;
        v1057 = threadIdx.x;
        int v1058;
        v1058 = v1057 / 32l;
        auto v1059 = cooperative_groups::labeled_partition(v1056,v1058);
        float v1060;
        v1060 = cooperative_groups::reduce(v1059, v1047, v40);
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
                v1068 = v1037[v1067];
                float v1069;
                v1069 = v1068 / v1060;
                assert("Tensor range check" && 0 <= v1062 && v1062 < 1l);
                assert("Tensor range check" && 0 <= v1064 && v1064 < 4l);
                v1061[v1067] = v1069;
                v1064 += 1l ;
            }
            v1062 += 1l ;
        }
        float v1070[4l];
        float v1071;
        v1071 = 0.0f;
        int v1072;
        v1072 = 0l;
        while (while_method_3(v1072)){
            assert("Tensor range check" && 0 <= v1072 && v1072 < 1l);
            int v1074;
            v1074 = 4l * v1072;
            assert("Tensor range check" && 0 <= v1072 && v1072 < 1l);
            int v1075; float v1076;
            Tuple0 tmp7 = Tuple0{0l, 0.0f};
            v1075 = tmp7.v0; v1076 = tmp7.v1;
            while (while_method_1(v1075)){
                assert("Tensor range check" && 0 <= v1075 && v1075 < 4l);
                int v1078;
                v1078 = v1075 + v1074;
                float v1079;
                v1079 = v1061[v1078];
                float v1080;
                v1080 = v1076 + v1079;
                v1076 = v1080;
                v1075 += 1l ;
            }
            auto v1081 = cooperative_groups::coalesced_threads();
            int v1082;
            v1082 = threadIdx.x;
            int v1083;
            v1083 = v1082 / 32l;
            auto v1084 = cooperative_groups::labeled_partition(v1081,v1083);
            Closure2 v1085{};
            float v1086;
            v1086 = cooperative_groups::inclusive_scan(v1084, v1076, v1085);
            float v1087;
            v1087 = v1084.shfl_up(v1086,1);
            bool v1088;
            v1088 = v1084.thread_rank() == 0;
            float v1089;
            if (v1088){
                v1089 = 0.0f;
            } else {
                v1089 = v1087;
            }
            float v1090;
            v1090 = v1084.shfl(v1086,v1084.num_threads()-1);
            float v1091;
            v1091 = v1071 + v1089;
            int v1092; float v1093;
            Tuple0 tmp8 = Tuple0{0l, v1091};
            v1092 = tmp8.v0; v1093 = tmp8.v1;
            while (while_method_1(v1092)){
                assert("Tensor range check" && 0 <= v1092 && v1092 < 4l);
                int v1095;
                v1095 = v1092 + v1074;
                float v1096;
                v1096 = v1061[v1095];
                float v1097;
                v1097 = v1093 + v1096;
                assert("Tensor range check" && 0 <= v1092 && v1092 < 4l);
                v1070[v1095] = v1097;
                v1093 = v1097;
                v1092 += 1l ;
            }
            float v1098;
            v1098 = v1071 + v1090;
            v1071 = v1098;
            v1072 += 1l ;
        }
        float v1099[4l];
        bool v1100[4l];
        int v1101;
        v1101 = 0l;
        while (while_method_3(v1101)){
            int v1103;
            v1103 = 0l;
            while (while_method_1(v1103)){
                assert("Tensor range check" && 0 <= v1101 && v1101 < 1l);
                assert("Tensor range check" && 0 <= v1103 && v1103 < 4l);
                int v1105;
                v1105 = 4l * v1101;
                int v1106;
                v1106 = v1105 + v1103;
                float v1107;
                v1107 = v1070[v1106];
                float v1108;
                v1108 = v1061[v1106];
                bool v1109;
                v1109 = v1108 > 0.0f;
                assert("Tensor range check" && 0 <= v1101 && v1101 < 1l);
                assert("Tensor range check" && 0 <= v1103 && v1103 < 4l);
                v1099[v1106] = v1107;
                v1100[v1106] = v1109;
                v1103 += 1l ;
            }
            v1101 += 1l ;
        }
        float v1110; bool v1111;
        Tuple3 tmp9 = Tuple3{-1.0f / 0.0f, false};
        v1110 = tmp9.v0; v1111 = tmp9.v1;
        int v1112;
        v1112 = 0l;
        while (while_method_3(v1112)){
            int v1114;
            v1114 = 0l;
            while (while_method_1(v1114)){
                assert("Tensor range check" && 0 <= v1112 && v1112 < 1l);
                assert("Tensor range check" && 0 <= v1114 && v1114 < 4l);
                int v1116;
                v1116 = 4l * v1112;
                int v1117;
                v1117 = v1116 + v1114;
                float v1118;
                v1118 = v1099[v1117];
                bool v1119;
                v1119 = v1100[v1117];
                float v1126; bool v1127;
                if (v1111){
                    if (v1119){
                        bool v1120;
                        v1120 = v1110 >= v1118;
                        float v1121;
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
        int v1130;
        v1130 = v1129 / 32l;
        auto v1131 = cooperative_groups::labeled_partition(v1128,v1130);
        Closure5 v1132{};
        float v1133; bool v1134;
        Tuple3 tmp10 = cooperative_groups::reduce(v1131, Tuple3{v1110, v1111}, v1132);
        v1133 = tmp10.v0; v1134 = tmp10.v1;
        bool v1135;
        v1135 = v1134 == false;
        if (v1135){
            assert("The local reduce must be true." && v1134);
        } else {
        }
        float v1137[4l];
        int v1138[4l];
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
                int v1145;
                v1145 = v979[v1144];
                float v1146;
                v1146 = curand_uniform(&v961);
                assert("Tensor range check" && 0 <= v1139 && v1139 < 1l);
                assert("Tensor range check" && 0 <= v1141 && v1141 < 4l);
                v1137[v1144] = v1146;
                v1138[v1144] = v1145;
                v1141 += 1l ;
            }
            v1139 += 1l ;
        }
        float v1147; int v1148;
        Tuple1 tmp11 = Tuple1{0.0f, 2147483647l};
        v1147 = tmp11.v0; v1148 = tmp11.v1;
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
                v1155 = v1137[v1154];
                int v1156;
                v1156 = v1138[v1154];
                bool v1157;
                v1157 = v1148 < v1156;
                float v1158; int v1159;
                if (v1157){
                    v1158 = v1147; v1159 = v1148;
                } else {
                    v1158 = v1155; v1159 = v1156;
                }
                v1147 = v1158;
                v1148 = v1159;
                v1151 += 1l ;
            }
            v1149 += 1l ;
        }
        auto v1160 = cooperative_groups::coalesced_threads();
        int v1161;
        v1161 = threadIdx.x;
        int v1162;
        v1162 = v1161 / 32l;
        auto v1163 = cooperative_groups::labeled_partition(v1160,v1162);
        Closure6 v1164{};
        float v1165; int v1166;
        Tuple1 tmp12 = cooperative_groups::reduce(v1163, Tuple1{v1147, v1148}, v1164);
        v1165 = tmp12.v0; v1166 = tmp12.v1;
        float v1167;
        v1167 = v1133 * v1165;
        int v1168[4l];
        bool v1169[4l];
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
                v1176 = v1099[v1175];
                bool v1177;
                v1177 = v1100[v1175];
                int v1178;
                v1178 = v979[v1175];
                int v1181; bool v1182;
                if (v1177){
                    float v1179;
                    v1179 = v1176 - v1167;
                    bool v1180;
                    v1180 = v1179 >= 0.0f;
                    v1181 = v1178; v1182 = v1180;
                } else {
                    v1181 = 2147483647l; v1182 = false;
                }
                assert("Tensor range check" && 0 <= v1170 && v1170 < 1l);
                assert("Tensor range check" && 0 <= v1172 && v1172 < 4l);
                v1168[v1175] = v1181;
                v1169[v1175] = v1182;
                v1172 += 1l ;
            }
            v1170 += 1l ;
        }
        int v1183; bool v1184;
        Tuple4 tmp13 = Tuple4{2147483647l, false};
        v1183 = tmp13.v0; v1184 = tmp13.v1;
        int v1185;
        v1185 = 0l;
        while (while_method_3(v1185)){
            int v1187;
            v1187 = 0l;
            while (while_method_1(v1187)){
                assert("Tensor range check" && 0 <= v1185 && v1185 < 1l);
                assert("Tensor range check" && 0 <= v1187 && v1187 < 4l);
                int v1189;
                v1189 = 4l * v1185;
                int v1190;
                v1190 = v1189 + v1187;
                int v1191;
                v1191 = v1168[v1190];
                bool v1192;
                v1192 = v1169[v1190];
                int v1199; bool v1200;
                if (v1184){
                    if (v1192){
                        bool v1193;
                        v1193 = v1183 < v1191;
                        int v1194;
                        if (v1193){
                            v1194 = v1183;
                        } else {
                            v1194 = v1191;
                        }
                        v1199 = v1194; v1200 = true;
                    } else {
                        v1199 = v1183; v1200 = v1184;
                    }
                } else {
                    if (v1192){
                        v1199 = v1191; v1200 = v1192;
                    } else {
                        v1199 = v1183; v1200 = v1184;
                    }
                }
                v1183 = v1199;
                v1184 = v1200;
                v1187 += 1l ;
            }
            v1185 += 1l ;
        }
        auto v1201 = cooperative_groups::coalesced_threads();
        int v1202;
        v1202 = threadIdx.x;
        int v1203;
        v1203 = v1202 / 32l;
        auto v1204 = cooperative_groups::labeled_partition(v1201,v1203);
        Closure7 v1205{};
        int v1206; bool v1207;
        Tuple4 tmp14 = cooperative_groups::reduce(v1204, Tuple4{v1183, v1184}, v1205);
        v1206 = tmp14.v0; v1207 = tmp14.v1;
        bool v1208;
        v1208 = v1207 == false;
        if (v1208){
            assert("The local reduce must be true." && v1207);
        } else {
        }
        assert("Tensor range check" && 0 <= v974 && v974 < 64l);
        int v1210;
        v1210 = 0l;
        while (while_method_3(v1210)){
            assert("Tensor range check" && 0 <= v1210 && v1210 < 1l);
            int v1212;
            v1212 = 128l * v1210;
            int v1213;
            v1213 = v1212 + v977;
            assert("Tensor range check" && 0 <= v1210 && v1210 < 1l);
            int v1214;
            v1214 = 4l * v1210;
            int4* v1215;
            v1215 = reinterpret_cast<int4*>(v1061 + v1214);
            int4* v1216;
            v1216 = reinterpret_cast<int4*>(v14 + v1213);
            assert("Pointer alignment check" && (unsigned long long)(v1215) % 4l == 0 && (unsigned long long)(v1216) % 4l == 0);
            *v1216 = *v1215;
            v1210 += 1l ;
        }
        assert("Tensor range check" && 0 <= v974 && v974 < 64l);
        v15[v1021] = v1206;
        v974 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1217;
    v1217 = threadIdx.x;
    unsigned long long v1218;
    v1218 = (unsigned long long)v1217;
    curandStatePhilox4_32_10_t v1219;
    curand_init(12344321ull,v1218,0ull,&v1219);
    int v1220;
    v1220 = threadIdx.x;
    bool v1221;
    v1221 = 0l <= v1220;
    bool v1222;
    v1222 = v1221 == false;
    if (v1222){
        assert("The index needs to be zero or positive." && v1221);
    } else {
    }
    int v1224;
    v1224 = v1220 % 32l;
    int v1225;
    v1225 = v1220 / 32l;
    bool v1226;
    v1226 = v1225 < 1l;
    bool v1227;
    v1227 = v1226 == false;
    if (v1227){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1226);
    } else {
    }
    assert("Tensor range check" && 0 <= v1225 && v1225 < 1l);
    assert("Tensor range check" && 0 <= v1224 && v1224 < 32l);
    int v1229;
    v1229 = 4l * v1224;
    int v1230;
    v1230 = 128l * v1225;
    int v1231;
    v1231 = v1230 + v1229;
    assert("Tensor range check" && 0 <= v1225 && v1225 < 1l);
    assert("Tensor range check" && 0 <= v1224 && v1224 < 32l);
    assert("Tensor range check" && 0 <= v1225 && v1225 < 1l);
    int v1232;
    v1232 = 0l;
    while (while_method_2(v1232)){
        assert("Tensor range check" && 0 <= v1232 && v1232 < 64l);
        int v1234;
        v1234 = 128l * v1232;
        int v1235;
        v1235 = v1234 + v1231;
        float v1236[4l];
        int v1237[4l];
        int v1238;
        v1238 = 0l;
        while (while_method_3(v1238)){
            assert("Tensor range check" && 0 <= v1238 && v1238 < 1l);
            int v1240;
            v1240 = 4l * v1238;
            assert("Tensor range check" && 0 <= v1238 && v1238 < 1l);
            int v1241;
            v1241 = 128l * v1238;
            int v1242;
            v1242 = v1241 + v1235;
            int4* v1243;
            v1243 = reinterpret_cast<int4*>(v1 + v1242);
            int4* v1244;
            v1244 = reinterpret_cast<int4*>(v1236 + v1240);
            assert("Pointer alignment check" && (unsigned long long)(v1243) % 4l == 0 && (unsigned long long)(v1244) % 4l == 0);
            *v1244 = *v1243;
            v1238 += 1l ;
        }
        int v1245;
        v1245 = 0l;
        while (while_method_3(v1245)){
            int v1247;
            v1247 = 0l;
            while (while_method_1(v1247)){
                bool v1249;
                v1249 = 0l <= v1247;
                bool v1251;
                if (v1249){
                    bool v1250;
                    v1250 = v1247 < 4l;
                    v1251 = v1250;
                } else {
                    v1251 = false;
                }
                bool v1252;
                v1252 = v1251 == false;
                if (v1252){
                    assert("The indices should be inside the range of the dimension." && v1251);
                } else {
                }
                bool v1254;
                v1254 = 0l <= v1224;
                bool v1256;
                if (v1254){
                    bool v1255;
                    v1255 = v1224 < 32l;
                    v1256 = v1255;
                } else {
                    v1256 = false;
                }
                bool v1257;
                v1257 = v1256 == false;
                if (v1257){
                    assert("The indices should be inside the range of the dimension." && v1256);
                } else {
                }
                int v1259;
                v1259 = v1224 * 4l;
                int v1260;
                v1260 = v1247 + v1259;
                bool v1261;
                v1261 = 0l <= v1245;
                bool v1263;
                if (v1261){
                    bool v1262;
                    v1262 = v1245 < 1l;
                    v1263 = v1262;
                } else {
                    v1263 = false;
                }
                bool v1264;
                v1264 = v1263 == false;
                if (v1264){
                    assert("The indices should be inside the range of the dimension." && v1263);
                } else {
                }
                int v1266;
                v1266 = v1245 * 128l;
                int v1267;
                v1267 = v1260 + v1266;
                assert("Tensor range check" && 0 <= v1245 && v1245 < 1l);
                assert("Tensor range check" && 0 <= v1247 && v1247 < 4l);
                int v1268;
                v1268 = 4l * v1245;
                int v1269;
                v1269 = v1268 + v1247;
                v1237[v1269] = v1267;
                v1247 += 1l ;
            }
            v1245 += 1l ;
        }
        bool v1270;
        v1270 = 0l <= v1225;
        bool v1271;
        v1271 = v1270 && v1226;
        bool v1272;
        v1272 = v1271 == false;
        if (v1272){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1271);
        } else {
        }
        bool v1274;
        v1274 = 0l <= v1232;
        bool v1276;
        if (v1274){
            bool v1275;
            v1275 = v1232 < 64l;
            v1276 = v1275;
        } else {
            v1276 = false;
        }
        bool v1277;
        v1277 = v1276 == false;
        if (v1277){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1276);
        } else {
        }
        int v1279;
        v1279 = v1232 + v1225;
        bool v1280[4l];
        int v1281;
        v1281 = 0l;
        while (while_method_3(v1281)){
            int v1283;
            v1283 = 0l;
            while (while_method_1(v1283)){
                assert("Tensor range check" && 0 <= v1281 && v1281 < 1l);
                assert("Tensor range check" && 0 <= v1283 && v1283 < 4l);
                int v1285;
                v1285 = 4l * v1281;
                int v1286;
                v1286 = v1285 + v1283;
                float v1287;
                v1287 = v1236[v1286];
                int v1288;
                v1288 = v1237[v1286];
                bool v1289;
                v1289 = v1288 < 11l;
                assert("Tensor range check" && 0 <= v1281 && v1281 < 1l);
                assert("Tensor range check" && 0 <= v1283 && v1283 < 4l);
                v1280[v1286] = v1289;
                v1283 += 1l ;
            }
            v1281 += 1l ;
        }
        int v1290[4l];
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
                bool v1297;
                v1297 = v1280[v1296];
                int v1298;
                if (v1297){
                    v1298 = 1l;
                } else {
                    v1298 = 0l;
                }
                assert("Tensor range check" && 0 <= v1291 && v1291 < 1l);
                assert("Tensor range check" && 0 <= v1293 && v1293 < 4l);
                v1290[v1296] = v1298;
                v1293 += 1l ;
            }
            v1291 += 1l ;
        }
        int v1299;
        v1299 = 0l;
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
                int v1306;
                v1306 = v1290[v1305];
                int v1307;
                v1307 = v1299 + v1306;
                v1299 = v1307;
                v1302 += 1l ;
            }
            v1300 += 1l ;
        }
        auto v1308 = cooperative_groups::coalesced_threads();
        int v1309;
        v1309 = threadIdx.x;
        int v1310;
        v1310 = v1309 / 32l;
        auto v1311 = cooperative_groups::labeled_partition(v1308,v1310);
        Closure4 v1312{};
        int v1313;
        v1313 = cooperative_groups::reduce(v1311, v1299, v1312);
        float v1314[4l];
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
                float v1321;
                v1321 = v1236[v1320];
                bool v1322;
                v1322 = v1280[v1320];
                float v1323;
                if (v1322){
                    v1323 = v1321;
                } else {
                    v1323 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1315 && v1315 < 1l);
                assert("Tensor range check" && 0 <= v1317 && v1317 < 4l);
                v1314[v1320] = v1323;
                v1317 += 1l ;
            }
            v1315 += 1l ;
        }
        float v1324;
        v1324 = 0.0f;
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
                v1331 = v1314[v1330];
                float v1332;
                v1332 = v1324 + v1331;
                v1324 = v1332;
                v1327 += 1l ;
            }
            v1325 += 1l ;
        }
        auto v1333 = cooperative_groups::coalesced_threads();
        int v1334;
        v1334 = threadIdx.x;
        int v1335;
        v1335 = v1334 / 32l;
        auto v1336 = cooperative_groups::labeled_partition(v1333,v1335);
        float v1337;
        v1337 = cooperative_groups::reduce(v1336, v1324, v40);
        float v1338;
        v1338 = (float)v1313;
        float v1339;
        v1339 = v1337 / v1338;
        float v1340[4l];
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
                float v1347;
                v1347 = v1236[v1346];
                bool v1348;
                v1348 = v1280[v1346];
                float v1349;
                if (v1348){
                    v1349 = v1347;
                } else {
                    v1349 = -1.0f / 0.0f;
                }
                float v1350;
                v1350 = v1349 - v1339;
                float v1351;
                v1351 = exp(v1350);
                assert("Tensor range check" && 0 <= v1341 && v1341 < 1l);
                assert("Tensor range check" && 0 <= v1343 && v1343 < 4l);
                v1340[v1346] = v1351;
                v1343 += 1l ;
            }
            v1341 += 1l ;
        }
        float v1352;
        v1352 = 0.0f;
        int v1353;
        v1353 = 0l;
        while (while_method_3(v1353)){
            int v1355;
            v1355 = 0l;
            while (while_method_1(v1355)){
                assert("Tensor range check" && 0 <= v1353 && v1353 < 1l);
                assert("Tensor range check" && 0 <= v1355 && v1355 < 4l);
                int v1357;
                v1357 = 4l * v1353;
                int v1358;
                v1358 = v1357 + v1355;
                float v1359;
                v1359 = v1340[v1358];
                float v1360;
                v1360 = v1352 + v1359;
                v1352 = v1360;
                v1355 += 1l ;
            }
            v1353 += 1l ;
        }
        auto v1361 = cooperative_groups::coalesced_threads();
        int v1362;
        v1362 = threadIdx.x;
        int v1363;
        v1363 = v1362 / 32l;
        auto v1364 = cooperative_groups::labeled_partition(v1361,v1363);
        float v1365;
        v1365 = cooperative_groups::reduce(v1364, v1352, v40);
        float v1366[4l];
        int v1367;
        v1367 = 0l;
        while (while_method_3(v1367)){
            int v1369;
            v1369 = 0l;
            while (while_method_1(v1369)){
                assert("Tensor range check" && 0 <= v1367 && v1367 < 1l);
                assert("Tensor range check" && 0 <= v1369 && v1369 < 4l);
                int v1371;
                v1371 = 4l * v1367;
                int v1372;
                v1372 = v1371 + v1369;
                float v1373;
                v1373 = v1340[v1372];
                float v1374;
                v1374 = v1373 / v1365;
                assert("Tensor range check" && 0 <= v1367 && v1367 < 1l);
                assert("Tensor range check" && 0 <= v1369 && v1369 < 4l);
                v1366[v1372] = v1374;
                v1369 += 1l ;
            }
            v1367 += 1l ;
        }
        float v1375[4l];
        float v1376;
        v1376 = 0.0f;
        int v1377;
        v1377 = 0l;
        while (while_method_3(v1377)){
            assert("Tensor range check" && 0 <= v1377 && v1377 < 1l);
            int v1379;
            v1379 = 4l * v1377;
            assert("Tensor range check" && 0 <= v1377 && v1377 < 1l);
            int v1380; float v1381;
            Tuple0 tmp15 = Tuple0{0l, 0.0f};
            v1380 = tmp15.v0; v1381 = tmp15.v1;
            while (while_method_1(v1380)){
                assert("Tensor range check" && 0 <= v1380 && v1380 < 4l);
                int v1383;
                v1383 = v1380 + v1379;
                float v1384;
                v1384 = v1366[v1383];
                float v1385;
                v1385 = v1381 + v1384;
                v1381 = v1385;
                v1380 += 1l ;
            }
            auto v1386 = cooperative_groups::coalesced_threads();
            int v1387;
            v1387 = threadIdx.x;
            int v1388;
            v1388 = v1387 / 32l;
            auto v1389 = cooperative_groups::labeled_partition(v1386,v1388);
            Closure2 v1390{};
            float v1391;
            v1391 = cooperative_groups::inclusive_scan(v1389, v1381, v1390);
            float v1392;
            v1392 = v1389.shfl_up(v1391,1);
            bool v1393;
            v1393 = v1389.thread_rank() == 0;
            float v1394;
            if (v1393){
                v1394 = 0.0f;
            } else {
                v1394 = v1392;
            }
            float v1395;
            v1395 = v1389.shfl(v1391,v1389.num_threads()-1);
            float v1396;
            v1396 = v1376 + v1394;
            int v1397; float v1398;
            Tuple0 tmp16 = Tuple0{0l, v1396};
            v1397 = tmp16.v0; v1398 = tmp16.v1;
            while (while_method_1(v1397)){
                assert("Tensor range check" && 0 <= v1397 && v1397 < 4l);
                int v1400;
                v1400 = v1397 + v1379;
                float v1401;
                v1401 = v1366[v1400];
                float v1402;
                v1402 = v1398 + v1401;
                assert("Tensor range check" && 0 <= v1397 && v1397 < 4l);
                v1375[v1400] = v1402;
                v1398 = v1402;
                v1397 += 1l ;
            }
            float v1403;
            v1403 = v1376 + v1395;
            v1376 = v1403;
            v1377 += 1l ;
        }
        float v1404[4l];
        bool v1405[4l];
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
                v1412 = v1375[v1411];
                float v1413;
                v1413 = v1366[v1411];
                bool v1414;
                v1414 = v1413 > 0.0f;
                assert("Tensor range check" && 0 <= v1406 && v1406 < 1l);
                assert("Tensor range check" && 0 <= v1408 && v1408 < 4l);
                v1404[v1411] = v1412;
                v1405[v1411] = v1414;
                v1408 += 1l ;
            }
            v1406 += 1l ;
        }
        float v1415; bool v1416;
        Tuple3 tmp17 = Tuple3{-1.0f / 0.0f, false};
        v1415 = tmp17.v0; v1416 = tmp17.v1;
        int v1417;
        v1417 = 0l;
        while (while_method_3(v1417)){
            int v1419;
            v1419 = 0l;
            while (while_method_1(v1419)){
                assert("Tensor range check" && 0 <= v1417 && v1417 < 1l);
                assert("Tensor range check" && 0 <= v1419 && v1419 < 4l);
                int v1421;
                v1421 = 4l * v1417;
                int v1422;
                v1422 = v1421 + v1419;
                float v1423;
                v1423 = v1404[v1422];
                bool v1424;
                v1424 = v1405[v1422];
                float v1431; bool v1432;
                if (v1416){
                    if (v1424){
                        bool v1425;
                        v1425 = v1415 >= v1423;
                        float v1426;
                        if (v1425){
                            v1426 = v1415;
                        } else {
                            v1426 = v1423;
                        }
                        v1431 = v1426; v1432 = true;
                    } else {
                        v1431 = v1415; v1432 = v1416;
                    }
                } else {
                    if (v1424){
                        v1431 = v1423; v1432 = v1424;
                    } else {
                        v1431 = v1415; v1432 = v1416;
                    }
                }
                v1415 = v1431;
                v1416 = v1432;
                v1419 += 1l ;
            }
            v1417 += 1l ;
        }
        auto v1433 = cooperative_groups::coalesced_threads();
        int v1434;
        v1434 = threadIdx.x;
        int v1435;
        v1435 = v1434 / 32l;
        auto v1436 = cooperative_groups::labeled_partition(v1433,v1435);
        Closure5 v1437{};
        float v1438; bool v1439;
        Tuple3 tmp18 = cooperative_groups::reduce(v1436, Tuple3{v1415, v1416}, v1437);
        v1438 = tmp18.v0; v1439 = tmp18.v1;
        bool v1440;
        v1440 = v1439 == false;
        if (v1440){
            assert("The local reduce must be true." && v1439);
        } else {
        }
        float v1442[4l];
        int v1443[4l];
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
                int v1450;
                v1450 = v1237[v1449];
                float v1451;
                v1451 = curand_uniform(&v1219);
                assert("Tensor range check" && 0 <= v1444 && v1444 < 1l);
                assert("Tensor range check" && 0 <= v1446 && v1446 < 4l);
                v1442[v1449] = v1451;
                v1443[v1449] = v1450;
                v1446 += 1l ;
            }
            v1444 += 1l ;
        }
        float v1452; int v1453;
        Tuple1 tmp19 = Tuple1{0.0f, 2147483647l};
        v1452 = tmp19.v0; v1453 = tmp19.v1;
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
                v1460 = v1442[v1459];
                int v1461;
                v1461 = v1443[v1459];
                bool v1462;
                v1462 = v1453 < v1461;
                float v1463; int v1464;
                if (v1462){
                    v1463 = v1452; v1464 = v1453;
                } else {
                    v1463 = v1460; v1464 = v1461;
                }
                v1452 = v1463;
                v1453 = v1464;
                v1456 += 1l ;
            }
            v1454 += 1l ;
        }
        auto v1465 = cooperative_groups::coalesced_threads();
        int v1466;
        v1466 = threadIdx.x;
        int v1467;
        v1467 = v1466 / 32l;
        auto v1468 = cooperative_groups::labeled_partition(v1465,v1467);
        Closure6 v1469{};
        float v1470; int v1471;
        Tuple1 tmp20 = cooperative_groups::reduce(v1468, Tuple1{v1452, v1453}, v1469);
        v1470 = tmp20.v0; v1471 = tmp20.v1;
        float v1472;
        v1472 = v1438 * v1470;
        int v1473[4l];
        bool v1474[4l];
        int v1475;
        v1475 = 0l;
        while (while_method_3(v1475)){
            int v1477;
            v1477 = 0l;
            while (while_method_1(v1477)){
                assert("Tensor range check" && 0 <= v1475 && v1475 < 1l);
                assert("Tensor range check" && 0 <= v1477 && v1477 < 4l);
                int v1479;
                v1479 = 4l * v1475;
                int v1480;
                v1480 = v1479 + v1477;
                float v1481;
                v1481 = v1404[v1480];
                bool v1482;
                v1482 = v1405[v1480];
                int v1483;
                v1483 = v1237[v1480];
                int v1486; bool v1487;
                if (v1482){
                    float v1484;
                    v1484 = v1481 - v1472;
                    bool v1485;
                    v1485 = v1484 >= 0.0f;
                    v1486 = v1483; v1487 = v1485;
                } else {
                    v1486 = 2147483647l; v1487 = false;
                }
                assert("Tensor range check" && 0 <= v1475 && v1475 < 1l);
                assert("Tensor range check" && 0 <= v1477 && v1477 < 4l);
                v1473[v1480] = v1486;
                v1474[v1480] = v1487;
                v1477 += 1l ;
            }
            v1475 += 1l ;
        }
        int v1488; bool v1489;
        Tuple4 tmp21 = Tuple4{2147483647l, false};
        v1488 = tmp21.v0; v1489 = tmp21.v1;
        int v1490;
        v1490 = 0l;
        while (while_method_3(v1490)){
            int v1492;
            v1492 = 0l;
            while (while_method_1(v1492)){
                assert("Tensor range check" && 0 <= v1490 && v1490 < 1l);
                assert("Tensor range check" && 0 <= v1492 && v1492 < 4l);
                int v1494;
                v1494 = 4l * v1490;
                int v1495;
                v1495 = v1494 + v1492;
                int v1496;
                v1496 = v1473[v1495];
                bool v1497;
                v1497 = v1474[v1495];
                int v1504; bool v1505;
                if (v1489){
                    if (v1497){
                        bool v1498;
                        v1498 = v1488 < v1496;
                        int v1499;
                        if (v1498){
                            v1499 = v1488;
                        } else {
                            v1499 = v1496;
                        }
                        v1504 = v1499; v1505 = true;
                    } else {
                        v1504 = v1488; v1505 = v1489;
                    }
                } else {
                    if (v1497){
                        v1504 = v1496; v1505 = v1497;
                    } else {
                        v1504 = v1488; v1505 = v1489;
                    }
                }
                v1488 = v1504;
                v1489 = v1505;
                v1492 += 1l ;
            }
            v1490 += 1l ;
        }
        auto v1506 = cooperative_groups::coalesced_threads();
        int v1507;
        v1507 = threadIdx.x;
        int v1508;
        v1508 = v1507 / 32l;
        auto v1509 = cooperative_groups::labeled_partition(v1506,v1508);
        Closure7 v1510{};
        int v1511; bool v1512;
        Tuple4 tmp22 = cooperative_groups::reduce(v1509, Tuple4{v1488, v1489}, v1510);
        v1511 = tmp22.v0; v1512 = tmp22.v1;
        bool v1513;
        v1513 = v1512 == false;
        if (v1513){
            assert("The local reduce must be true." && v1512);
        } else {
        }
        assert("Tensor range check" && 0 <= v1232 && v1232 < 64l);
        int v1515;
        v1515 = 0l;
        while (while_method_3(v1515)){
            assert("Tensor range check" && 0 <= v1515 && v1515 < 1l);
            int v1517;
            v1517 = 128l * v1515;
            int v1518;
            v1518 = v1517 + v1235;
            assert("Tensor range check" && 0 <= v1515 && v1515 < 1l);
            int v1519;
            v1519 = 4l * v1515;
            int4* v1520;
            v1520 = reinterpret_cast<int4*>(v1366 + v1519);
            int4* v1521;
            v1521 = reinterpret_cast<int4*>(v14 + v1518);
            assert("Pointer alignment check" && (unsigned long long)(v1520) % 4l == 0 && (unsigned long long)(v1521) % 4l == 0);
            *v1521 = *v1520;
            v1515 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1232 && v1232 < 64l);
        v15[v1279] = v1511;
        v1232 += 1l ;
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
    __shared__ float v44[1l];
    assert("Tensor range check" && 0 <= v43 && v43 < 1l);
    v44[v43] = v41;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v45;
    v45 = threadIdx.x;
    int v46;
    v46 = v45 % 32l;
    bool v47;
    v47 = v43 == 0l;
    bool v49;
    if (v47){
        bool v48;
        v48 = v46 < 1l;
        v49 = v48;
    } else {
        v49 = false;
    }
    if (v49){
        auto v50 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v46 && v46 < 1l);
        float v51;
        v51 = v44[v46];
        float v52;
        v52 = cooperative_groups::reduce(v50, v51, v40);
        v2[0l] = v52;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v53;
    v53 = threadIdx.x;
    bool v54;
    v54 = 0l <= v53;
    bool v55;
    v55 = v54 == false;
    if (v55){
        assert("The index needs to be zero or positive." && v54);
    } else {
    }
    int v57;
    v57 = v53 % 16l;
    int v58;
    v58 = v53 / 16l;
    bool v59;
    v59 = v58 < 2l;
    bool v60;
    v60 = v59 == false;
    if (v60){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v59);
    } else {
    }
    assert("Tensor range check" && 0 <= v58 && v58 < 2l);
    assert("Tensor range check" && 0 <= v57 && v57 < 16l);
    int v62;
    v62 = 4l * v57;
    int v63;
    v63 = 64l * v58;
    int v64;
    v64 = v63 + v62;
    assert("Tensor range check" && 0 <= v58 && v58 < 2l);
    assert("Tensor range check" && 0 <= v57 && v57 < 16l);
    int v65;
    v65 = 0l;
    while (while_method_2(v65)){
        assert("Tensor range check" && 0 <= v65 && v65 < 64l);
        int v67;
        v67 = 128l * v65;
        int v68;
        v68 = v67 + v64;
        int v69[4l];
        int v70[4l];
        int v71;
        v71 = 0l;
        while (while_method_3(v71)){
            assert("Tensor range check" && 0 <= v71 && v71 < 1l);
            int v73;
            v73 = 4l * v71;
            assert("Tensor range check" && 0 <= v71 && v71 < 1l);
            int v74;
            v74 = 64l * v71;
            int v75;
            v75 = v74 + v68;
            int4* v76;
            v76 = reinterpret_cast<int4*>(v0 + v75);
            int4* v77;
            v77 = reinterpret_cast<int4*>(v69 + v73);
            assert("Pointer alignment check" && (unsigned long long)(v76) % 4l == 0 && (unsigned long long)(v77) % 4l == 0);
            *v77 = *v76;
            v71 += 1l ;
        }
        int v78;
        v78 = 0l;
        while (while_method_3(v78)){
            int v80;
            v80 = 0l;
            while (while_method_1(v80)){
                bool v82;
                v82 = 0l <= v80;
                bool v84;
                if (v82){
                    bool v83;
                    v83 = v80 < 4l;
                    v84 = v83;
                } else {
                    v84 = false;
                }
                bool v85;
                v85 = v84 == false;
                if (v85){
                    assert("The indices should be inside the range of the dimension." && v84);
                } else {
                }
                bool v87;
                v87 = 0l <= v57;
                bool v89;
                if (v87){
                    bool v88;
                    v88 = v57 < 16l;
                    v89 = v88;
                } else {
                    v89 = false;
                }
                bool v90;
                v90 = v89 == false;
                if (v90){
                    assert("The indices should be inside the range of the dimension." && v89);
                } else {
                }
                int v92;
                v92 = v57 * 4l;
                int v93;
                v93 = v80 + v92;
                bool v94;
                v94 = 0l <= v78;
                bool v96;
                if (v94){
                    bool v95;
                    v95 = v78 < 1l;
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
                int v99;
                v99 = v78 * 64l;
                int v100;
                v100 = v93 + v99;
                assert("Tensor range check" && 0 <= v78 && v78 < 1l);
                assert("Tensor range check" && 0 <= v80 && v80 < 4l);
                int v101;
                v101 = 4l * v78;
                int v102;
                v102 = v101 + v80;
                v70[v102] = v100;
                v80 += 1l ;
            }
            v78 += 1l ;
        }
        bool v103;
        v103 = 0l <= v58;
        bool v104;
        v104 = v103 && v59;
        bool v105;
        v105 = v104 == false;
        if (v105){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v104);
        } else {
        }
        bool v107;
        v107 = 0l <= v65;
        bool v109;
        if (v107){
            bool v108;
            v108 = v65 < 64l;
            v109 = v108;
        } else {
            v109 = false;
        }
        bool v110;
        v110 = v109 == false;
        if (v110){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v109);
        } else {
        }
        int v112;
        v112 = v65 * 2l;
        int v113;
        v113 = v112 + v58;
        assert("Tensor range check" && 0 <= v65 && v65 < 64l);
        int v114;
        v114 = 0l;
        while (while_method_3(v114)){
            assert("Tensor range check" && 0 <= v114 && v114 < 1l);
            int v116;
            v116 = 64l * v114;
            int v117;
            v117 = v116 + v68;
            assert("Tensor range check" && 0 <= v114 && v114 < 1l);
            int v118;
            v118 = 4l * v114;
            int4* v119;
            v119 = reinterpret_cast<int4*>(v69 + v118);
            int4* v120;
            v120 = reinterpret_cast<int4*>(v3 + v117);
            assert("Pointer alignment check" && (unsigned long long)(v119) % 4l == 0 && (unsigned long long)(v120) % 4l == 0);
            *v120 = *v119;
            v114 += 1l ;
        }
        v65 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v121;
    v121 = threadIdx.x;
    bool v122;
    v122 = 0l <= v121;
    bool v123;
    v123 = v122 == false;
    if (v123){
        assert("The index needs to be zero or positive." && v122);
    } else {
    }
    int v125;
    v125 = v121 % 16l;
    int v126;
    v126 = v121 / 16l;
    bool v127;
    v127 = v126 < 2l;
    bool v128;
    v128 = v127 == false;
    if (v128){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v127);
    } else {
    }
    assert("Tensor range check" && 0 <= v126 && v126 < 2l);
    assert("Tensor range check" && 0 <= v125 && v125 < 16l);
    int v130;
    v130 = 4l * v125;
    int v131;
    v131 = 64l * v126;
    int v132;
    v132 = v131 + v130;
    assert("Tensor range check" && 0 <= v126 && v126 < 2l);
    assert("Tensor range check" && 0 <= v125 && v125 < 16l);
    int v133;
    v133 = 0l;
    while (while_method_2(v133)){
        assert("Tensor range check" && 0 <= v133 && v133 < 64l);
        int v135;
        v135 = 128l * v133;
        int v136;
        v136 = v135 + v132;
        float v137[4l];
        int v138[4l];
        int v139;
        v139 = 0l;
        while (while_method_3(v139)){
            assert("Tensor range check" && 0 <= v139 && v139 < 1l);
            int v141;
            v141 = 4l * v139;
            assert("Tensor range check" && 0 <= v139 && v139 < 1l);
            int v142;
            v142 = 64l * v139;
            int v143;
            v143 = v142 + v136;
            int4* v144;
            v144 = reinterpret_cast<int4*>(v1 + v143);
            int4* v145;
            v145 = reinterpret_cast<int4*>(v137 + v141);
            assert("Pointer alignment check" && (unsigned long long)(v144) % 4l == 0 && (unsigned long long)(v145) % 4l == 0);
            *v145 = *v144;
            v139 += 1l ;
        }
        int v146;
        v146 = 0l;
        while (while_method_3(v146)){
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
                v155 = 0l <= v125;
                bool v157;
                if (v155){
                    bool v156;
                    v156 = v125 < 16l;
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
                v160 = v125 * 4l;
                int v161;
                v161 = v148 + v160;
                bool v162;
                v162 = 0l <= v146;
                bool v164;
                if (v162){
                    bool v163;
                    v163 = v146 < 1l;
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
                v167 = v146 * 64l;
                int v168;
                v168 = v161 + v167;
                assert("Tensor range check" && 0 <= v146 && v146 < 1l);
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
        bool v171;
        v171 = 0l <= v126;
        bool v172;
        v172 = v171 && v127;
        bool v173;
        v173 = v172 == false;
        if (v173){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v172);
        } else {
        }
        bool v175;
        v175 = 0l <= v133;
        bool v177;
        if (v175){
            bool v176;
            v176 = v133 < 64l;
            v177 = v176;
        } else {
            v177 = false;
        }
        bool v178;
        v178 = v177 == false;
        if (v178){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v177);
        } else {
        }
        int v180;
        v180 = v133 * 2l;
        int v181;
        v181 = v180 + v126;
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
                v190 = v138[v189];
                assert("Tensor range check" && 0 <= v184 && v184 < 1l);
                assert("Tensor range check" && 0 <= v186 && v186 < 4l);
                v182[v189] = v181;
                v183[v189] = v190;
                v186 += 1l ;
            }
            v184 += 1l ;
        }
        assert("Tensor range check" && 0 <= v133 && v133 < 64l);
        int v191;
        v191 = 0l;
        while (while_method_3(v191)){
            assert("Tensor range check" && 0 <= v191 && v191 < 1l);
            int v193;
            v193 = 64l * v191;
            int v194;
            v194 = v193 + v136;
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
        v133 += 1l ;
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
    v204 = v200 % 16l;
    int v205;
    v205 = v200 / 16l;
    bool v206;
    v206 = v205 < 2l;
    bool v207;
    v207 = v206 == false;
    if (v207){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v206);
    } else {
    }
    assert("Tensor range check" && 0 <= v205 && v205 < 2l);
    assert("Tensor range check" && 0 <= v204 && v204 < 16l);
    int v209;
    v209 = 4l * v204;
    int v210;
    v210 = 64l * v205;
    int v211;
    v211 = v210 + v209;
    assert("Tensor range check" && 0 <= v205 && v205 < 2l);
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
            v221 = 64l * v218;
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
                    v235 = v204 < 16l;
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
                v246 = v225 * 64l;
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
        v259 = v212 * 2l;
        int v260;
        v260 = v259 + v205;
        assert("Tensor range check" && 0 <= v212 && v212 < 64l);
        int v261;
        v261 = 2l * v212;
        int v262;
        v262 = v261 + v205;
        v12[v262] = v260;
        v212 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v263;
    v263 = threadIdx.x;
    bool v264;
    v264 = 0l <= v263;
    bool v265;
    v265 = v264 == false;
    if (v265){
        assert("The index needs to be zero or positive." && v264);
    } else {
    }
    int v267;
    v267 = v263 % 16l;
    int v268;
    v268 = v263 / 16l;
    bool v269;
    v269 = v268 < 2l;
    bool v270;
    v270 = v269 == false;
    if (v270){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v269);
    } else {
    }
    assert("Tensor range check" && 0 <= v268 && v268 < 2l);
    assert("Tensor range check" && 0 <= v267 && v267 < 16l);
    int v272;
    v272 = 4l * v267;
    int v273;
    v273 = 64l * v268;
    int v274;
    v274 = v273 + v272;
    assert("Tensor range check" && 0 <= v268 && v268 < 2l);
    assert("Tensor range check" && 0 <= v267 && v267 < 16l);
    int v275;
    v275 = 0l;
    while (while_method_2(v275)){
        assert("Tensor range check" && 0 <= v275 && v275 < 64l);
        int v277;
        v277 = 128l * v275;
        int v278;
        v278 = v277 + v274;
        float v279[4l];
        int v280[4l];
        int v281;
        v281 = 0l;
        while (while_method_3(v281)){
            assert("Tensor range check" && 0 <= v281 && v281 < 1l);
            int v283;
            v283 = 4l * v281;
            assert("Tensor range check" && 0 <= v281 && v281 < 1l);
            int v284;
            v284 = 64l * v281;
            int v285;
            v285 = v284 + v278;
            int4* v286;
            v286 = reinterpret_cast<int4*>(v1 + v285);
            int4* v287;
            v287 = reinterpret_cast<int4*>(v279 + v283);
            assert("Pointer alignment check" && (unsigned long long)(v286) % 4l == 0 && (unsigned long long)(v287) % 4l == 0);
            *v287 = *v286;
            v281 += 1l ;
        }
        int v288;
        v288 = 0l;
        while (while_method_3(v288)){
            int v290;
            v290 = 0l;
            while (while_method_1(v290)){
                bool v292;
                v292 = 0l <= v290;
                bool v294;
                if (v292){
                    bool v293;
                    v293 = v290 < 4l;
                    v294 = v293;
                } else {
                    v294 = false;
                }
                bool v295;
                v295 = v294 == false;
                if (v295){
                    assert("The indices should be inside the range of the dimension." && v294);
                } else {
                }
                bool v297;
                v297 = 0l <= v267;
                bool v299;
                if (v297){
                    bool v298;
                    v298 = v267 < 16l;
                    v299 = v298;
                } else {
                    v299 = false;
                }
                bool v300;
                v300 = v299 == false;
                if (v300){
                    assert("The indices should be inside the range of the dimension." && v299);
                } else {
                }
                int v302;
                v302 = v267 * 4l;
                int v303;
                v303 = v290 + v302;
                bool v304;
                v304 = 0l <= v288;
                bool v306;
                if (v304){
                    bool v305;
                    v305 = v288 < 1l;
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
                v309 = v288 * 64l;
                int v310;
                v310 = v303 + v309;
                assert("Tensor range check" && 0 <= v288 && v288 < 1l);
                assert("Tensor range check" && 0 <= v290 && v290 < 4l);
                int v311;
                v311 = 4l * v288;
                int v312;
                v312 = v311 + v290;
                v280[v312] = v310;
                v290 += 1l ;
            }
            v288 += 1l ;
        }
        bool v313;
        v313 = 0l <= v268;
        bool v314;
        v314 = v313 && v269;
        bool v315;
        v315 = v314 == false;
        if (v315){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v314);
        } else {
        }
        bool v317;
        v317 = 0l <= v275;
        bool v319;
        if (v317){
            bool v318;
            v318 = v275 < 64l;
            v319 = v318;
        } else {
            v319 = false;
        }
        bool v320;
        v320 = v319 == false;
        if (v320){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v319);
        } else {
        }
        int v322;
        v322 = v275 * 2l;
        int v323;
        v323 = v322 + v268;
        float v324;
        v324 = 0.0f;
        int v325;
        v325 = 0l;
        while (while_method_3(v325)){
            int v327;
            v327 = 0l;
            while (while_method_1(v327)){
                assert("Tensor range check" && 0 <= v325 && v325 < 1l);
                assert("Tensor range check" && 0 <= v327 && v327 < 4l);
                int v329;
                v329 = 4l * v325;
                int v330;
                v330 = v329 + v327;
                float v331;
                v331 = v279[v330];
                float v332;
                v332 = v324 + v331;
                v324 = v332;
                v327 += 1l ;
            }
            v325 += 1l ;
        }
        auto v333 = cooperative_groups::coalesced_threads();
        int v334;
        v334 = threadIdx.x;
        int v335;
        v335 = v334 / 16l;
        auto v336 = cooperative_groups::labeled_partition(v333,v335);
        float v337;
        v337 = cooperative_groups::reduce(v336, v324, v40);
        float v338;
        v338 = v337 / 64.0f;
        float v339[4l];
        int v340;
        v340 = 0l;
        while (while_method_3(v340)){
            int v342;
            v342 = 0l;
            while (while_method_1(v342)){
                assert("Tensor range check" && 0 <= v340 && v340 < 1l);
                assert("Tensor range check" && 0 <= v342 && v342 < 4l);
                int v344;
                v344 = 4l * v340;
                int v345;
                v345 = v344 + v342;
                float v346;
                v346 = v279[v345];
                float v347;
                v347 = v346 - v338;
                float v348;
                v348 = exp(v347);
                assert("Tensor range check" && 0 <= v340 && v340 < 1l);
                assert("Tensor range check" && 0 <= v342 && v342 < 4l);
                v339[v345] = v348;
                v342 += 1l ;
            }
            v340 += 1l ;
        }
        float v349;
        v349 = 0.0f;
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
                v356 = v339[v355];
                float v357;
                v357 = v349 + v356;
                v349 = v357;
                v352 += 1l ;
            }
            v350 += 1l ;
        }
        auto v358 = cooperative_groups::coalesced_threads();
        int v359;
        v359 = threadIdx.x;
        int v360;
        v360 = v359 / 16l;
        auto v361 = cooperative_groups::labeled_partition(v358,v360);
        float v362;
        v362 = cooperative_groups::reduce(v361, v349, v40);
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
                v370 = v339[v369];
                float v371;
                v371 = v370 / v362;
                assert("Tensor range check" && 0 <= v364 && v364 < 1l);
                assert("Tensor range check" && 0 <= v366 && v366 < 4l);
                v363[v369] = v371;
                v366 += 1l ;
            }
            v364 += 1l ;
        }
        assert("Tensor range check" && 0 <= v275 && v275 < 64l);
        int v372;
        v372 = 0l;
        while (while_method_3(v372)){
            assert("Tensor range check" && 0 <= v372 && v372 < 1l);
            int v374;
            v374 = 64l * v372;
            int v375;
            v375 = v374 + v278;
            assert("Tensor range check" && 0 <= v372 && v372 < 1l);
            int v376;
            v376 = 4l * v372;
            int4* v377;
            v377 = reinterpret_cast<int4*>(v363 + v376);
            int4* v378;
            v378 = reinterpret_cast<int4*>(v4 + v375);
            assert("Pointer alignment check" && (unsigned long long)(v377) % 4l == 0 && (unsigned long long)(v378) % 4l == 0);
            *v378 = *v377;
            v372 += 1l ;
        }
        v275 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v379;
    v379 = threadIdx.x;
    bool v380;
    v380 = 0l <= v379;
    bool v381;
    v381 = v380 == false;
    if (v381){
        assert("The index needs to be zero or positive." && v380);
    } else {
    }
    int v383;
    v383 = v379 % 16l;
    int v384;
    v384 = v379 / 16l;
    bool v385;
    v385 = v384 < 2l;
    bool v386;
    v386 = v385 == false;
    if (v386){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v385);
    } else {
    }
    assert("Tensor range check" && 0 <= v384 && v384 < 2l);
    assert("Tensor range check" && 0 <= v383 && v383 < 16l);
    int v388;
    v388 = 4l * v383;
    int v389;
    v389 = 64l * v384;
    int v390;
    v390 = v389 + v388;
    assert("Tensor range check" && 0 <= v384 && v384 < 2l);
    assert("Tensor range check" && 0 <= v383 && v383 < 16l);
    int v391;
    v391 = 0l;
    while (while_method_2(v391)){
        assert("Tensor range check" && 0 <= v391 && v391 < 64l);
        int v393;
        v393 = 128l * v391;
        int v394;
        v394 = v393 + v390;
        float v395[4l];
        int v396[4l];
        int v397;
        v397 = 0l;
        while (while_method_3(v397)){
            assert("Tensor range check" && 0 <= v397 && v397 < 1l);
            int v399;
            v399 = 4l * v397;
            assert("Tensor range check" && 0 <= v397 && v397 < 1l);
            int v400;
            v400 = 64l * v397;
            int v401;
            v401 = v400 + v394;
            int4* v402;
            v402 = reinterpret_cast<int4*>(v1 + v401);
            int4* v403;
            v403 = reinterpret_cast<int4*>(v395 + v399);
            assert("Pointer alignment check" && (unsigned long long)(v402) % 4l == 0 && (unsigned long long)(v403) % 4l == 0);
            *v403 = *v402;
            v397 += 1l ;
        }
        int v404;
        v404 = 0l;
        while (while_method_3(v404)){
            int v406;
            v406 = 0l;
            while (while_method_1(v406)){
                bool v408;
                v408 = 0l <= v406;
                bool v410;
                if (v408){
                    bool v409;
                    v409 = v406 < 4l;
                    v410 = v409;
                } else {
                    v410 = false;
                }
                bool v411;
                v411 = v410 == false;
                if (v411){
                    assert("The indices should be inside the range of the dimension." && v410);
                } else {
                }
                bool v413;
                v413 = 0l <= v383;
                bool v415;
                if (v413){
                    bool v414;
                    v414 = v383 < 16l;
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
                v418 = v383 * 4l;
                int v419;
                v419 = v406 + v418;
                bool v420;
                v420 = 0l <= v404;
                bool v422;
                if (v420){
                    bool v421;
                    v421 = v404 < 1l;
                    v422 = v421;
                } else {
                    v422 = false;
                }
                bool v423;
                v423 = v422 == false;
                if (v423){
                    assert("The indices should be inside the range of the dimension." && v422);
                } else {
                }
                int v425;
                v425 = v404 * 64l;
                int v426;
                v426 = v419 + v425;
                assert("Tensor range check" && 0 <= v404 && v404 < 1l);
                assert("Tensor range check" && 0 <= v406 && v406 < 4l);
                int v427;
                v427 = 4l * v404;
                int v428;
                v428 = v427 + v406;
                v396[v428] = v426;
                v406 += 1l ;
            }
            v404 += 1l ;
        }
        bool v429;
        v429 = 0l <= v384;
        bool v430;
        v430 = v429 && v385;
        bool v431;
        v431 = v430 == false;
        if (v431){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v430);
        } else {
        }
        bool v433;
        v433 = 0l <= v391;
        bool v435;
        if (v433){
            bool v434;
            v434 = v391 < 64l;
            v435 = v434;
        } else {
            v435 = false;
        }
        bool v436;
        v436 = v435 == false;
        if (v436){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v435);
        } else {
        }
        int v438;
        v438 = v391 * 2l;
        int v439;
        v439 = v438 + v384;
        float v440[4l];
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
                v447 = v395[v446];
                float v448;
                v448 = v447 * v447;
                assert("Tensor range check" && 0 <= v441 && v441 < 1l);
                assert("Tensor range check" && 0 <= v443 && v443 < 4l);
                v440[v446] = v448;
                v443 += 1l ;
            }
            v441 += 1l ;
        }
        float v449;
        v449 = 0.0f;
        int v450;
        v450 = 0l;
        while (while_method_3(v450)){
            int v452;
            v452 = 0l;
            while (while_method_1(v452)){
                assert("Tensor range check" && 0 <= v450 && v450 < 1l);
                assert("Tensor range check" && 0 <= v452 && v452 < 4l);
                int v454;
                v454 = 4l * v450;
                int v455;
                v455 = v454 + v452;
                float v456;
                v456 = v440[v455];
                float v457;
                v457 = v449 + v456;
                v449 = v457;
                v452 += 1l ;
            }
            v450 += 1l ;
        }
        auto v458 = cooperative_groups::coalesced_threads();
        int v459;
        v459 = threadIdx.x;
        int v460;
        v460 = v459 / 16l;
        auto v461 = cooperative_groups::labeled_partition(v458,v460);
        float v462;
        v462 = cooperative_groups::reduce(v461, v449, v40);
        float v463[4l];
        int v464;
        v464 = 0l;
        while (while_method_3(v464)){
            int v466;
            v466 = 0l;
            while (while_method_1(v466)){
                assert("Tensor range check" && 0 <= v464 && v464 < 1l);
                assert("Tensor range check" && 0 <= v466 && v466 < 4l);
                int v468;
                v468 = 4l * v464;
                int v469;
                v469 = v468 + v466;
                float v470;
                v470 = v395[v469];
                bool v471;
                v471 = v462 == 0.0f;
                bool v472;
                v472 = v471 != true;
                float v474;
                if (v472){
                    float v473;
                    v473 = v470 / v462;
                    v474 = v473;
                } else {
                    v474 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v464 && v464 < 1l);
                assert("Tensor range check" && 0 <= v466 && v466 < 4l);
                v463[v469] = v474;
                v466 += 1l ;
            }
            v464 += 1l ;
        }
        assert("Tensor range check" && 0 <= v391 && v391 < 64l);
        int v475;
        v475 = 0l;
        while (while_method_3(v475)){
            assert("Tensor range check" && 0 <= v475 && v475 < 1l);
            int v477;
            v477 = 64l * v475;
            int v478;
            v478 = v477 + v394;
            assert("Tensor range check" && 0 <= v475 && v475 < 1l);
            int v479;
            v479 = 4l * v475;
            int4* v480;
            v480 = reinterpret_cast<int4*>(v463 + v479);
            int4* v481;
            v481 = reinterpret_cast<int4*>(v8 + v478);
            assert("Pointer alignment check" && (unsigned long long)(v480) % 4l == 0 && (unsigned long long)(v481) % 4l == 0);
            *v481 = *v480;
            v475 += 1l ;
        }
        v391 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v482;
    v482 = threadIdx.x;
    bool v483;
    v483 = 0l <= v482;
    bool v484;
    v484 = v483 == false;
    if (v484){
        assert("The index needs to be zero or positive." && v483);
    } else {
    }
    int v486;
    v486 = v482 % 16l;
    int v487;
    v487 = v482 / 16l;
    bool v488;
    v488 = v487 < 2l;
    bool v489;
    v489 = v488 == false;
    if (v489){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v488);
    } else {
    }
    assert("Tensor range check" && 0 <= v487 && v487 < 2l);
    assert("Tensor range check" && 0 <= v486 && v486 < 16l);
    int v491;
    v491 = 4l * v486;
    int v492;
    v492 = 64l * v487;
    int v493;
    v493 = v492 + v491;
    assert("Tensor range check" && 0 <= v487 && v487 < 2l);
    int v494;
    v494 = 0l;
    while (while_method_2(v494)){
        assert("Tensor range check" && 0 <= v494 && v494 < 64l);
        int v496;
        v496 = 128l * v494;
        int v497;
        v497 = v496 + v493;
        float v498[4l];
        int v499[4l];
        int v500;
        v500 = 0l;
        while (while_method_3(v500)){
            assert("Tensor range check" && 0 <= v500 && v500 < 1l);
            int v502;
            v502 = 4l * v500;
            assert("Tensor range check" && 0 <= v500 && v500 < 1l);
            int v503;
            v503 = 64l * v500;
            int v504;
            v504 = v503 + v497;
            int4* v505;
            v505 = reinterpret_cast<int4*>(v1 + v504);
            int4* v506;
            v506 = reinterpret_cast<int4*>(v498 + v502);
            assert("Pointer alignment check" && (unsigned long long)(v505) % 4l == 0 && (unsigned long long)(v506) % 4l == 0);
            *v506 = *v505;
            v500 += 1l ;
        }
        int v507;
        v507 = 0l;
        while (while_method_3(v507)){
            int v509;
            v509 = 0l;
            while (while_method_1(v509)){
                bool v511;
                v511 = 0l <= v509;
                bool v513;
                if (v511){
                    bool v512;
                    v512 = v509 < 4l;
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
                bool v516;
                v516 = 0l <= v486;
                bool v518;
                if (v516){
                    bool v517;
                    v517 = v486 < 16l;
                    v518 = v517;
                } else {
                    v518 = false;
                }
                bool v519;
                v519 = v518 == false;
                if (v519){
                    assert("The indices should be inside the range of the dimension." && v518);
                } else {
                }
                int v521;
                v521 = v486 * 4l;
                int v522;
                v522 = v509 + v521;
                bool v523;
                v523 = 0l <= v507;
                bool v525;
                if (v523){
                    bool v524;
                    v524 = v507 < 1l;
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
                v528 = v507 * 64l;
                int v529;
                v529 = v522 + v528;
                assert("Tensor range check" && 0 <= v507 && v507 < 1l);
                assert("Tensor range check" && 0 <= v509 && v509 < 4l);
                int v530;
                v530 = 4l * v507;
                int v531;
                v531 = v530 + v509;
                v499[v531] = v529;
                v509 += 1l ;
            }
            v507 += 1l ;
        }
        bool v532;
        v532 = 0l <= v487;
        bool v533;
        v533 = v532 && v488;
        bool v534;
        v534 = v533 == false;
        if (v534){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v533);
        } else {
        }
        bool v536;
        v536 = 0l <= v494;
        bool v538;
        if (v536){
            bool v537;
            v537 = v494 < 64l;
            v538 = v537;
        } else {
            v538 = false;
        }
        bool v539;
        v539 = v538 == false;
        if (v539){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v538);
        } else {
        }
        int v541;
        v541 = v494 * 2l;
        int v542;
        v542 = v541 + v487;
        float v543; int v544;
        Tuple1 tmp24 = Tuple1{-1.0f / 0.0f, 0l};
        v543 = tmp24.v0; v544 = tmp24.v1;
        int v545;
        v545 = 0l;
        while (while_method_3(v545)){
            int v547;
            v547 = 0l;
            while (while_method_1(v547)){
                assert("Tensor range check" && 0 <= v545 && v545 < 1l);
                assert("Tensor range check" && 0 <= v547 && v547 < 4l);
                int v549;
                v549 = 4l * v545;
                int v550;
                v550 = v549 + v547;
                float v551;
                v551 = v498[v550];
                int v552;
                v552 = v499[v550];
                bool v553;
                v553 = v543 > v551;
                float v554; int v555;
                if (v553){
                    v554 = v543; v555 = v544;
                } else {
                    v554 = v551; v555 = v552;
                }
                v543 = v554;
                v544 = v555;
                v547 += 1l ;
            }
            v545 += 1l ;
        }
        auto v556 = cooperative_groups::coalesced_threads();
        int v557;
        v557 = threadIdx.x;
        int v558;
        v558 = v557 / 16l;
        auto v559 = cooperative_groups::labeled_partition(v556,v558);
        Closure1 v560{};
        float v561; int v562;
        Tuple1 tmp25 = cooperative_groups::reduce(v559, Tuple1{v543, v544}, v560);
        v561 = tmp25.v0; v562 = tmp25.v1;
        assert("Tensor range check" && 0 <= v494 && v494 < 64l);
        int v563;
        v563 = 2l * v494;
        int v564;
        v564 = v563 + v487;
        v9[v564] = v562;
        v494 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v565;
    v565 = threadIdx.x;
    bool v566;
    v566 = 0l <= v565;
    bool v567;
    v567 = v566 == false;
    if (v567){
        assert("The index needs to be zero or positive." && v566);
    } else {
    }
    int v569;
    v569 = v565 % 16l;
    int v570;
    v570 = v565 / 16l;
    bool v571;
    v571 = v570 < 2l;
    bool v572;
    v572 = v571 == false;
    if (v572){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v571);
    } else {
    }
    assert("Tensor range check" && 0 <= v570 && v570 < 2l);
    assert("Tensor range check" && 0 <= v569 && v569 < 16l);
    int v574;
    v574 = 4l * v569;
    int v575;
    v575 = 64l * v570;
    int v576;
    v576 = v575 + v574;
    assert("Tensor range check" && 0 <= v570 && v570 < 2l);
    assert("Tensor range check" && 0 <= v569 && v569 < 16l);
    int v577;
    v577 = 0l;
    while (while_method_2(v577)){
        assert("Tensor range check" && 0 <= v577 && v577 < 64l);
        int v579;
        v579 = 128l * v577;
        int v580;
        v580 = v579 + v576;
        float v581[4l];
        int v582[4l];
        int v583;
        v583 = 0l;
        while (while_method_3(v583)){
            assert("Tensor range check" && 0 <= v583 && v583 < 1l);
            int v585;
            v585 = 4l * v583;
            assert("Tensor range check" && 0 <= v583 && v583 < 1l);
            int v586;
            v586 = 64l * v583;
            int v587;
            v587 = v586 + v580;
            int4* v588;
            v588 = reinterpret_cast<int4*>(v1 + v587);
            int4* v589;
            v589 = reinterpret_cast<int4*>(v581 + v585);
            assert("Pointer alignment check" && (unsigned long long)(v588) % 4l == 0 && (unsigned long long)(v589) % 4l == 0);
            *v589 = *v588;
            v583 += 1l ;
        }
        int v590;
        v590 = 0l;
        while (while_method_3(v590)){
            int v592;
            v592 = 0l;
            while (while_method_1(v592)){
                bool v594;
                v594 = 0l <= v592;
                bool v596;
                if (v594){
                    bool v595;
                    v595 = v592 < 4l;
                    v596 = v595;
                } else {
                    v596 = false;
                }
                bool v597;
                v597 = v596 == false;
                if (v597){
                    assert("The indices should be inside the range of the dimension." && v596);
                } else {
                }
                bool v599;
                v599 = 0l <= v569;
                bool v601;
                if (v599){
                    bool v600;
                    v600 = v569 < 16l;
                    v601 = v600;
                } else {
                    v601 = false;
                }
                bool v602;
                v602 = v601 == false;
                if (v602){
                    assert("The indices should be inside the range of the dimension." && v601);
                } else {
                }
                int v604;
                v604 = v569 * 4l;
                int v605;
                v605 = v592 + v604;
                bool v606;
                v606 = 0l <= v590;
                bool v608;
                if (v606){
                    bool v607;
                    v607 = v590 < 1l;
                    v608 = v607;
                } else {
                    v608 = false;
                }
                bool v609;
                v609 = v608 == false;
                if (v609){
                    assert("The indices should be inside the range of the dimension." && v608);
                } else {
                }
                int v611;
                v611 = v590 * 64l;
                int v612;
                v612 = v605 + v611;
                assert("Tensor range check" && 0 <= v590 && v590 < 1l);
                assert("Tensor range check" && 0 <= v592 && v592 < 4l);
                int v613;
                v613 = 4l * v590;
                int v614;
                v614 = v613 + v592;
                v582[v614] = v612;
                v592 += 1l ;
            }
            v590 += 1l ;
        }
        bool v615;
        v615 = 0l <= v570;
        bool v616;
        v616 = v615 && v571;
        bool v617;
        v617 = v616 == false;
        if (v617){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v616);
        } else {
        }
        bool v619;
        v619 = 0l <= v577;
        bool v621;
        if (v619){
            bool v620;
            v620 = v577 < 64l;
            v621 = v620;
        } else {
            v621 = false;
        }
        bool v622;
        v622 = v621 == false;
        if (v622){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v621);
        } else {
        }
        int v624;
        v624 = v577 * 2l;
        int v625;
        v625 = v624 + v570;
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
                v633 = v581[v632];
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
        float v639;
        v639 = cooperative_groups::reduce(v638, v626, v40);
        float v640;
        v640 = v639 / 64.0f;
        float v641[4l];
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
                v648 = v581[v647];
                float v649;
                v649 = v648 - v640;
                float v650;
                v650 = exp(v649);
                assert("Tensor range check" && 0 <= v642 && v642 < 1l);
                assert("Tensor range check" && 0 <= v644 && v644 < 4l);
                v641[v647] = v650;
                v644 += 1l ;
            }
            v642 += 1l ;
        }
        float v651;
        v651 = 0.0f;
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
                float v658;
                v658 = v641[v657];
                float v659;
                v659 = v651 + v658;
                v651 = v659;
                v654 += 1l ;
            }
            v652 += 1l ;
        }
        auto v660 = cooperative_groups::coalesced_threads();
        int v661;
        v661 = threadIdx.x;
        int v662;
        v662 = v661 / 16l;
        auto v663 = cooperative_groups::labeled_partition(v660,v662);
        float v664;
        v664 = cooperative_groups::reduce(v663, v651, v40);
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
                v672 = v641[v671];
                float v673;
                v673 = v672 / v664;
                assert("Tensor range check" && 0 <= v666 && v666 < 1l);
                assert("Tensor range check" && 0 <= v668 && v668 < 4l);
                v665[v671] = v673;
                v668 += 1l ;
            }
            v666 += 1l ;
        }
        float v674[4l];
        float v675;
        v675 = 0.0f;
        int v676;
        v676 = 0l;
        while (while_method_3(v676)){
            assert("Tensor range check" && 0 <= v676 && v676 < 1l);
            int v678;
            v678 = 4l * v676;
            assert("Tensor range check" && 0 <= v676 && v676 < 1l);
            int v679; float v680;
            Tuple0 tmp26 = Tuple0{0l, 0.0f};
            v679 = tmp26.v0; v680 = tmp26.v1;
            while (while_method_1(v679)){
                assert("Tensor range check" && 0 <= v679 && v679 < 4l);
                int v682;
                v682 = v679 + v678;
                float v683;
                v683 = v665[v682];
                float v684;
                v684 = v680 + v683;
                v680 = v684;
                v679 += 1l ;
            }
            auto v685 = cooperative_groups::coalesced_threads();
            int v686;
            v686 = threadIdx.x;
            int v687;
            v687 = v686 / 16l;
            auto v688 = cooperative_groups::labeled_partition(v685,v687);
            Closure2 v689{};
            float v690;
            v690 = cooperative_groups::inclusive_scan(v688, v680, v689);
            float v691;
            v691 = v688.shfl_up(v690,1);
            bool v692;
            v692 = v688.thread_rank() == 0;
            float v693;
            if (v692){
                v693 = 0.0f;
            } else {
                v693 = v691;
            }
            float v694;
            v694 = v688.shfl(v690,v688.num_threads()-1);
            float v695;
            v695 = v675 + v693;
            int v696; float v697;
            Tuple0 tmp27 = Tuple0{0l, v695};
            v696 = tmp27.v0; v697 = tmp27.v1;
            while (while_method_1(v696)){
                assert("Tensor range check" && 0 <= v696 && v696 < 4l);
                int v699;
                v699 = v696 + v678;
                float v700;
                v700 = v665[v699];
                float v701;
                v701 = v697 + v700;
                assert("Tensor range check" && 0 <= v696 && v696 < 4l);
                v674[v699] = v701;
                v697 = v701;
                v696 += 1l ;
            }
            float v702;
            v702 = v675 + v694;
            v675 = v702;
            v676 += 1l ;
        }
        assert("Tensor range check" && 0 <= v577 && v577 < 64l);
        int v703;
        v703 = 0l;
        while (while_method_3(v703)){
            assert("Tensor range check" && 0 <= v703 && v703 < 1l);
            int v705;
            v705 = 64l * v703;
            int v706;
            v706 = v705 + v580;
            assert("Tensor range check" && 0 <= v703 && v703 < 1l);
            int v707;
            v707 = 4l * v703;
            int4* v708;
            v708 = reinterpret_cast<int4*>(v665 + v707);
            int4* v709;
            v709 = reinterpret_cast<int4*>(v6 + v706);
            assert("Pointer alignment check" && (unsigned long long)(v708) % 4l == 0 && (unsigned long long)(v709) % 4l == 0);
            *v709 = *v708;
            int4* v710;
            v710 = reinterpret_cast<int4*>(v674 + v707);
            int4* v711;
            v711 = reinterpret_cast<int4*>(v7 + v706);
            assert("Pointer alignment check" && (unsigned long long)(v710) % 4l == 0 && (unsigned long long)(v711) % 4l == 0);
            *v711 = *v710;
            v703 += 1l ;
        }
        v577 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v712;
    v712 = threadIdx.x;
    bool v713;
    v713 = 0l <= v712;
    bool v714;
    v714 = v713 == false;
    if (v714){
        assert("The index needs to be zero or positive." && v713);
    } else {
    }
    int v716;
    v716 = v712 % 16l;
    int v717;
    v717 = v712 / 16l;
    bool v718;
    v718 = v717 < 2l;
    bool v719;
    v719 = v718 == false;
    if (v719){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v718);
    } else {
    }
    assert("Tensor range check" && 0 <= v717 && v717 < 2l);
    assert("Tensor range check" && 0 <= v716 && v716 < 16l);
    int v721;
    v721 = 4l * v716;
    int v722;
    v722 = 64l * v717;
    int v723;
    v723 = v722 + v721;
    assert("Tensor range check" && 0 <= v717 && v717 < 2l);
    assert("Tensor range check" && 0 <= v716 && v716 < 16l);
    int v724;
    v724 = 0l;
    while (while_method_2(v724)){
        assert("Tensor range check" && 0 <= v724 && v724 < 64l);
        int v726;
        v726 = 128l * v724;
        int v727;
        v727 = v726 + v723;
        int v728[4l];
        int v729[4l];
        int v730;
        v730 = 0l;
        while (while_method_3(v730)){
            assert("Tensor range check" && 0 <= v730 && v730 < 1l);
            int v732;
            v732 = 4l * v730;
            assert("Tensor range check" && 0 <= v730 && v730 < 1l);
            int v733;
            v733 = 64l * v730;
            int v734;
            v734 = v733 + v727;
            int4* v735;
            v735 = reinterpret_cast<int4*>(v0 + v734);
            int4* v736;
            v736 = reinterpret_cast<int4*>(v728 + v732);
            assert("Pointer alignment check" && (unsigned long long)(v735) % 4l == 0 && (unsigned long long)(v736) % 4l == 0);
            *v736 = *v735;
            v730 += 1l ;
        }
        int v737;
        v737 = 0l;
        while (while_method_3(v737)){
            int v739;
            v739 = 0l;
            while (while_method_1(v739)){
                bool v741;
                v741 = 0l <= v739;
                bool v743;
                if (v741){
                    bool v742;
                    v742 = v739 < 4l;
                    v743 = v742;
                } else {
                    v743 = false;
                }
                bool v744;
                v744 = v743 == false;
                if (v744){
                    assert("The indices should be inside the range of the dimension." && v743);
                } else {
                }
                bool v746;
                v746 = 0l <= v716;
                bool v748;
                if (v746){
                    bool v747;
                    v747 = v716 < 16l;
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
                v751 = v716 * 4l;
                int v752;
                v752 = v739 + v751;
                bool v753;
                v753 = 0l <= v737;
                bool v755;
                if (v753){
                    bool v754;
                    v754 = v737 < 1l;
                    v755 = v754;
                } else {
                    v755 = false;
                }
                bool v756;
                v756 = v755 == false;
                if (v756){
                    assert("The indices should be inside the range of the dimension." && v755);
                } else {
                }
                int v758;
                v758 = v737 * 64l;
                int v759;
                v759 = v752 + v758;
                assert("Tensor range check" && 0 <= v737 && v737 < 1l);
                assert("Tensor range check" && 0 <= v739 && v739 < 4l);
                int v760;
                v760 = 4l * v737;
                int v761;
                v761 = v760 + v739;
                v729[v761] = v759;
                v739 += 1l ;
            }
            v737 += 1l ;
        }
        bool v762;
        v762 = 0l <= v717;
        bool v763;
        v763 = v762 && v718;
        bool v764;
        v764 = v763 == false;
        if (v764){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v763);
        } else {
        }
        bool v766;
        v766 = 0l <= v724;
        bool v768;
        if (v766){
            bool v767;
            v767 = v724 < 64l;
            v768 = v767;
        } else {
            v768 = false;
        }
        bool v769;
        v769 = v768 == false;
        if (v769){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v768);
        } else {
        }
        int v771;
        v771 = v724 * 2l;
        int v772;
        v772 = v771 + v717;
        int v773[4l];
        int v774;
        v774 = 0l;
        int v775;
        v775 = 0l;
        while (while_method_3(v775)){
            assert("Tensor range check" && 0 <= v775 && v775 < 1l);
            int v777;
            v777 = 4l * v775;
            assert("Tensor range check" && 0 <= v775 && v775 < 1l);
            int v778; int v779;
            Tuple2 tmp28 = Tuple2{0l, 0l};
            v778 = tmp28.v0; v779 = tmp28.v1;
            while (while_method_1(v778)){
                assert("Tensor range check" && 0 <= v778 && v778 < 4l);
                int v781;
                v781 = v778 + v777;
                int v782;
                v782 = v728[v781];
                int v783;
                v783 = v779 + v782;
                v779 = v783;
                v778 += 1l ;
            }
            auto v784 = cooperative_groups::coalesced_threads();
            int v785;
            v785 = threadIdx.x;
            int v786;
            v786 = v785 / 16l;
            auto v787 = cooperative_groups::labeled_partition(v784,v786);
            Closure3 v788{};
            int v789;
            v789 = cooperative_groups::inclusive_scan(v787, v779, v788);
            int v790;
            v790 = v787.shfl_up(v789,1);
            bool v791;
            v791 = v787.thread_rank() == 0;
            int v792;
            if (v791){
                v792 = 0l;
            } else {
                v792 = v790;
            }
            int v793;
            v793 = v787.shfl(v789,v787.num_threads()-1);
            int v794;
            v794 = v774 + v792;
            int v795; int v796;
            Tuple2 tmp29 = Tuple2{0l, v794};
            v795 = tmp29.v0; v796 = tmp29.v1;
            while (while_method_1(v795)){
                assert("Tensor range check" && 0 <= v795 && v795 < 4l);
                int v798;
                v798 = v795 + v777;
                int v799;
                v799 = v728[v798];
                assert("Tensor range check" && 0 <= v795 && v795 < 4l);
                v773[v798] = v796;
                int v800;
                v800 = v796 + v799;
                v796 = v800;
                v795 += 1l ;
            }
            int v801;
            v801 = v774 + v793;
            v774 = v801;
            v775 += 1l ;
        }
        assert("Tensor range check" && 0 <= v724 && v724 < 64l);
        int v802;
        v802 = 0l;
        while (while_method_3(v802)){
            assert("Tensor range check" && 0 <= v802 && v802 < 1l);
            int v804;
            v804 = 64l * v802;
            int v805;
            v805 = v804 + v727;
            assert("Tensor range check" && 0 <= v802 && v802 < 1l);
            int v806;
            v806 = 4l * v802;
            int4* v807;
            v807 = reinterpret_cast<int4*>(v773 + v806);
            int4* v808;
            v808 = reinterpret_cast<int4*>(v13 + v805);
            assert("Pointer alignment check" && (unsigned long long)(v807) % 4l == 0 && (unsigned long long)(v808) % 4l == 0);
            *v808 = *v807;
            v802 += 1l ;
        }
        v724 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v809;
    v809 = threadIdx.x;
    bool v810;
    v810 = 0l <= v809;
    bool v811;
    v811 = v810 == false;
    if (v811){
        assert("The index needs to be zero or positive." && v810);
    } else {
    }
    int v813;
    v813 = v809 % 16l;
    int v814;
    v814 = v809 / 16l;
    bool v815;
    v815 = v814 < 2l;
    bool v816;
    v816 = v815 == false;
    if (v816){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v815);
    } else {
    }
    assert("Tensor range check" && 0 <= v814 && v814 < 2l);
    assert("Tensor range check" && 0 <= v813 && v813 < 16l);
    int v818;
    v818 = 4l * v813;
    int v819;
    v819 = 64l * v814;
    int v820;
    v820 = v819 + v818;
    assert("Tensor range check" && 0 <= v814 && v814 < 2l);
    assert("Tensor range check" && 0 <= v813 && v813 < 16l);
    int v821;
    v821 = 0l;
    while (while_method_2(v821)){
        assert("Tensor range check" && 0 <= v821 && v821 < 64l);
        int v823;
        v823 = 128l * v821;
        int v824;
        v824 = v823 + v820;
        float v825[4l];
        int v826[4l];
        int v827;
        v827 = 0l;
        while (while_method_3(v827)){
            assert("Tensor range check" && 0 <= v827 && v827 < 1l);
            int v829;
            v829 = 4l * v827;
            assert("Tensor range check" && 0 <= v827 && v827 < 1l);
            int v830;
            v830 = 64l * v827;
            int v831;
            v831 = v830 + v824;
            int4* v832;
            v832 = reinterpret_cast<int4*>(v1 + v831);
            int4* v833;
            v833 = reinterpret_cast<int4*>(v825 + v829);
            assert("Pointer alignment check" && (unsigned long long)(v832) % 4l == 0 && (unsigned long long)(v833) % 4l == 0);
            *v833 = *v832;
            v827 += 1l ;
        }
        int v834;
        v834 = 0l;
        while (while_method_3(v834)){
            int v836;
            v836 = 0l;
            while (while_method_1(v836)){
                bool v838;
                v838 = 0l <= v836;
                bool v840;
                if (v838){
                    bool v839;
                    v839 = v836 < 4l;
                    v840 = v839;
                } else {
                    v840 = false;
                }
                bool v841;
                v841 = v840 == false;
                if (v841){
                    assert("The indices should be inside the range of the dimension." && v840);
                } else {
                }
                bool v843;
                v843 = 0l <= v813;
                bool v845;
                if (v843){
                    bool v844;
                    v844 = v813 < 16l;
                    v845 = v844;
                } else {
                    v845 = false;
                }
                bool v846;
                v846 = v845 == false;
                if (v846){
                    assert("The indices should be inside the range of the dimension." && v845);
                } else {
                }
                int v848;
                v848 = v813 * 4l;
                int v849;
                v849 = v836 + v848;
                bool v850;
                v850 = 0l <= v834;
                bool v852;
                if (v850){
                    bool v851;
                    v851 = v834 < 1l;
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
                int v855;
                v855 = v834 * 64l;
                int v856;
                v856 = v849 + v855;
                assert("Tensor range check" && 0 <= v834 && v834 < 1l);
                assert("Tensor range check" && 0 <= v836 && v836 < 4l);
                int v857;
                v857 = 4l * v834;
                int v858;
                v858 = v857 + v836;
                v826[v858] = v856;
                v836 += 1l ;
            }
            v834 += 1l ;
        }
        bool v859;
        v859 = 0l <= v814;
        bool v860;
        v860 = v859 && v815;
        bool v861;
        v861 = v860 == false;
        if (v861){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v860);
        } else {
        }
        bool v863;
        v863 = 0l <= v821;
        bool v865;
        if (v863){
            bool v864;
            v864 = v821 < 64l;
            v865 = v864;
        } else {
            v865 = false;
        }
        bool v866;
        v866 = v865 == false;
        if (v866){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v865);
        } else {
        }
        int v868;
        v868 = v821 * 2l;
        int v869;
        v869 = v868 + v814;
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
                v877 = v825[v876];
                int v878;
                v878 = v826[v876];
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
        v900 = v899 / 16l;
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
                v911 = v825[v910];
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
        v925 = v924 / 16l;
        auto v926 = cooperative_groups::labeled_partition(v923,v925);
        float v927;
        v927 = cooperative_groups::reduce(v926, v914, v40);
        float v928;
        v928 = (float)v903;
        float v929;
        v929 = v927 / v928;
        float v930[4l];
        int v931;
        v931 = 0l;
        while (while_method_3(v931)){
            int v933;
            v933 = 0l;
            while (while_method_1(v933)){
                assert("Tensor range check" && 0 <= v931 && v931 < 1l);
                assert("Tensor range check" && 0 <= v933 && v933 < 4l);
                int v935;
                v935 = 4l * v931;
                int v936;
                v936 = v935 + v933;
                float v937;
                v937 = v825[v936];
                bool v938;
                v938 = v870[v936];
                float v939;
                if (v938){
                    v939 = v937;
                } else {
                    v939 = -1.0f / 0.0f;
                }
                float v940;
                v940 = v939 - v929;
                float v941;
                v941 = exp(v940);
                assert("Tensor range check" && 0 <= v931 && v931 < 1l);
                assert("Tensor range check" && 0 <= v933 && v933 < 4l);
                v930[v936] = v941;
                v933 += 1l ;
            }
            v931 += 1l ;
        }
        float v942;
        v942 = 0.0f;
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
                v949 = v930[v948];
                float v950;
                v950 = v942 + v949;
                v942 = v950;
                v945 += 1l ;
            }
            v943 += 1l ;
        }
        auto v951 = cooperative_groups::coalesced_threads();
        int v952;
        v952 = threadIdx.x;
        int v953;
        v953 = v952 / 16l;
        auto v954 = cooperative_groups::labeled_partition(v951,v953);
        float v955;
        v955 = cooperative_groups::reduce(v954, v942, v40);
        float v956[4l];
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
                v963 = v930[v962];
                float v964;
                v964 = v963 / v955;
                assert("Tensor range check" && 0 <= v957 && v957 < 1l);
                assert("Tensor range check" && 0 <= v959 && v959 < 4l);
                v956[v962] = v964;
                v959 += 1l ;
            }
            v957 += 1l ;
        }
        assert("Tensor range check" && 0 <= v821 && v821 < 64l);
        int v965;
        v965 = 0l;
        while (while_method_3(v965)){
            assert("Tensor range check" && 0 <= v965 && v965 < 1l);
            int v967;
            v967 = 64l * v965;
            int v968;
            v968 = v967 + v824;
            assert("Tensor range check" && 0 <= v965 && v965 < 1l);
            int v969;
            v969 = 4l * v965;
            int4* v970;
            v970 = reinterpret_cast<int4*>(v956 + v969);
            int4* v971;
            v971 = reinterpret_cast<int4*>(v5 + v968);
            assert("Pointer alignment check" && (unsigned long long)(v970) % 4l == 0 && (unsigned long long)(v971) % 4l == 0);
            *v971 = *v970;
            v965 += 1l ;
        }
        v821 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v972;
    v972 = threadIdx.x;
    unsigned long long v973;
    v973 = (unsigned long long)v972;
    curandStatePhilox4_32_10_t v974;
    curand_init(12344321ull,v973,0ull,&v974);
    int v975;
    v975 = threadIdx.x;
    bool v976;
    v976 = 0l <= v975;
    bool v977;
    v977 = v976 == false;
    if (v977){
        assert("The index needs to be zero or positive." && v976);
    } else {
    }
    int v979;
    v979 = v975 % 16l;
    int v980;
    v980 = v975 / 16l;
    bool v981;
    v981 = v980 < 2l;
    bool v982;
    v982 = v981 == false;
    if (v982){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v981);
    } else {
    }
    assert("Tensor range check" && 0 <= v980 && v980 < 2l);
    assert("Tensor range check" && 0 <= v979 && v979 < 16l);
    int v984;
    v984 = 4l * v979;
    int v985;
    v985 = 64l * v980;
    int v986;
    v986 = v985 + v984;
    assert("Tensor range check" && 0 <= v980 && v980 < 2l);
    assert("Tensor range check" && 0 <= v979 && v979 < 16l);
    assert("Tensor range check" && 0 <= v980 && v980 < 2l);
    int v987;
    v987 = 0l;
    while (while_method_2(v987)){
        assert("Tensor range check" && 0 <= v987 && v987 < 64l);
        int v989;
        v989 = 128l * v987;
        int v990;
        v990 = v989 + v986;
        float v991[4l];
        int v992[4l];
        int v993;
        v993 = 0l;
        while (while_method_3(v993)){
            assert("Tensor range check" && 0 <= v993 && v993 < 1l);
            int v995;
            v995 = 4l * v993;
            assert("Tensor range check" && 0 <= v993 && v993 < 1l);
            int v996;
            v996 = 64l * v993;
            int v997;
            v997 = v996 + v990;
            int4* v998;
            v998 = reinterpret_cast<int4*>(v1 + v997);
            int4* v999;
            v999 = reinterpret_cast<int4*>(v991 + v995);
            assert("Pointer alignment check" && (unsigned long long)(v998) % 4l == 0 && (unsigned long long)(v999) % 4l == 0);
            *v999 = *v998;
            v993 += 1l ;
        }
        int v1000;
        v1000 = 0l;
        while (while_method_3(v1000)){
            int v1002;
            v1002 = 0l;
            while (while_method_1(v1002)){
                bool v1004;
                v1004 = 0l <= v1002;
                bool v1006;
                if (v1004){
                    bool v1005;
                    v1005 = v1002 < 4l;
                    v1006 = v1005;
                } else {
                    v1006 = false;
                }
                bool v1007;
                v1007 = v1006 == false;
                if (v1007){
                    assert("The indices should be inside the range of the dimension." && v1006);
                } else {
                }
                bool v1009;
                v1009 = 0l <= v979;
                bool v1011;
                if (v1009){
                    bool v1010;
                    v1010 = v979 < 16l;
                    v1011 = v1010;
                } else {
                    v1011 = false;
                }
                bool v1012;
                v1012 = v1011 == false;
                if (v1012){
                    assert("The indices should be inside the range of the dimension." && v1011);
                } else {
                }
                int v1014;
                v1014 = v979 * 4l;
                int v1015;
                v1015 = v1002 + v1014;
                bool v1016;
                v1016 = 0l <= v1000;
                bool v1018;
                if (v1016){
                    bool v1017;
                    v1017 = v1000 < 1l;
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
                v1021 = v1000 * 64l;
                int v1022;
                v1022 = v1015 + v1021;
                assert("Tensor range check" && 0 <= v1000 && v1000 < 1l);
                assert("Tensor range check" && 0 <= v1002 && v1002 < 4l);
                int v1023;
                v1023 = 4l * v1000;
                int v1024;
                v1024 = v1023 + v1002;
                v992[v1024] = v1022;
                v1002 += 1l ;
            }
            v1000 += 1l ;
        }
        bool v1025;
        v1025 = 0l <= v980;
        bool v1026;
        v1026 = v1025 && v981;
        bool v1027;
        v1027 = v1026 == false;
        if (v1027){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1026);
        } else {
        }
        bool v1029;
        v1029 = 0l <= v987;
        bool v1031;
        if (v1029){
            bool v1030;
            v1030 = v987 < 64l;
            v1031 = v1030;
        } else {
            v1031 = false;
        }
        bool v1032;
        v1032 = v1031 == false;
        if (v1032){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1031);
        } else {
        }
        int v1034;
        v1034 = v987 * 2l;
        int v1035;
        v1035 = v1034 + v980;
        float v1036;
        v1036 = 0.0f;
        int v1037;
        v1037 = 0l;
        while (while_method_3(v1037)){
            int v1039;
            v1039 = 0l;
            while (while_method_1(v1039)){
                assert("Tensor range check" && 0 <= v1037 && v1037 < 1l);
                assert("Tensor range check" && 0 <= v1039 && v1039 < 4l);
                int v1041;
                v1041 = 4l * v1037;
                int v1042;
                v1042 = v1041 + v1039;
                float v1043;
                v1043 = v991[v1042];
                float v1044;
                v1044 = v1036 + v1043;
                v1036 = v1044;
                v1039 += 1l ;
            }
            v1037 += 1l ;
        }
        auto v1045 = cooperative_groups::coalesced_threads();
        int v1046;
        v1046 = threadIdx.x;
        int v1047;
        v1047 = v1046 / 16l;
        auto v1048 = cooperative_groups::labeled_partition(v1045,v1047);
        float v1049;
        v1049 = cooperative_groups::reduce(v1048, v1036, v40);
        float v1050;
        v1050 = v1049 / 64.0f;
        float v1051[4l];
        int v1052;
        v1052 = 0l;
        while (while_method_3(v1052)){
            int v1054;
            v1054 = 0l;
            while (while_method_1(v1054)){
                assert("Tensor range check" && 0 <= v1052 && v1052 < 1l);
                assert("Tensor range check" && 0 <= v1054 && v1054 < 4l);
                int v1056;
                v1056 = 4l * v1052;
                int v1057;
                v1057 = v1056 + v1054;
                float v1058;
                v1058 = v991[v1057];
                float v1059;
                v1059 = v1058 - v1050;
                float v1060;
                v1060 = exp(v1059);
                assert("Tensor range check" && 0 <= v1052 && v1052 < 1l);
                assert("Tensor range check" && 0 <= v1054 && v1054 < 4l);
                v1051[v1057] = v1060;
                v1054 += 1l ;
            }
            v1052 += 1l ;
        }
        float v1061;
        v1061 = 0.0f;
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
                v1068 = v1051[v1067];
                float v1069;
                v1069 = v1061 + v1068;
                v1061 = v1069;
                v1064 += 1l ;
            }
            v1062 += 1l ;
        }
        auto v1070 = cooperative_groups::coalesced_threads();
        int v1071;
        v1071 = threadIdx.x;
        int v1072;
        v1072 = v1071 / 16l;
        auto v1073 = cooperative_groups::labeled_partition(v1070,v1072);
        float v1074;
        v1074 = cooperative_groups::reduce(v1073, v1061, v40);
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
                v1082 = v1051[v1081];
                float v1083;
                v1083 = v1082 / v1074;
                assert("Tensor range check" && 0 <= v1076 && v1076 < 1l);
                assert("Tensor range check" && 0 <= v1078 && v1078 < 4l);
                v1075[v1081] = v1083;
                v1078 += 1l ;
            }
            v1076 += 1l ;
        }
        float v1084[4l];
        float v1085;
        v1085 = 0.0f;
        int v1086;
        v1086 = 0l;
        while (while_method_3(v1086)){
            assert("Tensor range check" && 0 <= v1086 && v1086 < 1l);
            int v1088;
            v1088 = 4l * v1086;
            assert("Tensor range check" && 0 <= v1086 && v1086 < 1l);
            int v1089; float v1090;
            Tuple0 tmp30 = Tuple0{0l, 0.0f};
            v1089 = tmp30.v0; v1090 = tmp30.v1;
            while (while_method_1(v1089)){
                assert("Tensor range check" && 0 <= v1089 && v1089 < 4l);
                int v1092;
                v1092 = v1089 + v1088;
                float v1093;
                v1093 = v1075[v1092];
                float v1094;
                v1094 = v1090 + v1093;
                v1090 = v1094;
                v1089 += 1l ;
            }
            auto v1095 = cooperative_groups::coalesced_threads();
            int v1096;
            v1096 = threadIdx.x;
            int v1097;
            v1097 = v1096 / 16l;
            auto v1098 = cooperative_groups::labeled_partition(v1095,v1097);
            Closure2 v1099{};
            float v1100;
            v1100 = cooperative_groups::inclusive_scan(v1098, v1090, v1099);
            float v1101;
            v1101 = v1098.shfl_up(v1100,1);
            bool v1102;
            v1102 = v1098.thread_rank() == 0;
            float v1103;
            if (v1102){
                v1103 = 0.0f;
            } else {
                v1103 = v1101;
            }
            float v1104;
            v1104 = v1098.shfl(v1100,v1098.num_threads()-1);
            float v1105;
            v1105 = v1085 + v1103;
            int v1106; float v1107;
            Tuple0 tmp31 = Tuple0{0l, v1105};
            v1106 = tmp31.v0; v1107 = tmp31.v1;
            while (while_method_1(v1106)){
                assert("Tensor range check" && 0 <= v1106 && v1106 < 4l);
                int v1109;
                v1109 = v1106 + v1088;
                float v1110;
                v1110 = v1075[v1109];
                float v1111;
                v1111 = v1107 + v1110;
                assert("Tensor range check" && 0 <= v1106 && v1106 < 4l);
                v1084[v1109] = v1111;
                v1107 = v1111;
                v1106 += 1l ;
            }
            float v1112;
            v1112 = v1085 + v1104;
            v1085 = v1112;
            v1086 += 1l ;
        }
        float v1113[4l];
        bool v1114[4l];
        int v1115;
        v1115 = 0l;
        while (while_method_3(v1115)){
            int v1117;
            v1117 = 0l;
            while (while_method_1(v1117)){
                assert("Tensor range check" && 0 <= v1115 && v1115 < 1l);
                assert("Tensor range check" && 0 <= v1117 && v1117 < 4l);
                int v1119;
                v1119 = 4l * v1115;
                int v1120;
                v1120 = v1119 + v1117;
                float v1121;
                v1121 = v1084[v1120];
                float v1122;
                v1122 = v1075[v1120];
                bool v1123;
                v1123 = v1122 > 0.0f;
                assert("Tensor range check" && 0 <= v1115 && v1115 < 1l);
                assert("Tensor range check" && 0 <= v1117 && v1117 < 4l);
                v1113[v1120] = v1121;
                v1114[v1120] = v1123;
                v1117 += 1l ;
            }
            v1115 += 1l ;
        }
        float v1124; bool v1125;
        Tuple3 tmp32 = Tuple3{-1.0f / 0.0f, false};
        v1124 = tmp32.v0; v1125 = tmp32.v1;
        int v1126;
        v1126 = 0l;
        while (while_method_3(v1126)){
            int v1128;
            v1128 = 0l;
            while (while_method_1(v1128)){
                assert("Tensor range check" && 0 <= v1126 && v1126 < 1l);
                assert("Tensor range check" && 0 <= v1128 && v1128 < 4l);
                int v1130;
                v1130 = 4l * v1126;
                int v1131;
                v1131 = v1130 + v1128;
                float v1132;
                v1132 = v1113[v1131];
                bool v1133;
                v1133 = v1114[v1131];
                float v1140; bool v1141;
                if (v1125){
                    if (v1133){
                        bool v1134;
                        v1134 = v1124 >= v1132;
                        float v1135;
                        if (v1134){
                            v1135 = v1124;
                        } else {
                            v1135 = v1132;
                        }
                        v1140 = v1135; v1141 = true;
                    } else {
                        v1140 = v1124; v1141 = v1125;
                    }
                } else {
                    if (v1133){
                        v1140 = v1132; v1141 = v1133;
                    } else {
                        v1140 = v1124; v1141 = v1125;
                    }
                }
                v1124 = v1140;
                v1125 = v1141;
                v1128 += 1l ;
            }
            v1126 += 1l ;
        }
        auto v1142 = cooperative_groups::coalesced_threads();
        int v1143;
        v1143 = threadIdx.x;
        int v1144;
        v1144 = v1143 / 16l;
        auto v1145 = cooperative_groups::labeled_partition(v1142,v1144);
        Closure5 v1146{};
        float v1147; bool v1148;
        Tuple3 tmp33 = cooperative_groups::reduce(v1145, Tuple3{v1124, v1125}, v1146);
        v1147 = tmp33.v0; v1148 = tmp33.v1;
        bool v1149;
        v1149 = v1148 == false;
        if (v1149){
            assert("The local reduce must be true." && v1148);
        } else {
        }
        float v1151[4l];
        int v1152[4l];
        int v1153;
        v1153 = 0l;
        while (while_method_3(v1153)){
            int v1155;
            v1155 = 0l;
            while (while_method_1(v1155)){
                assert("Tensor range check" && 0 <= v1153 && v1153 < 1l);
                assert("Tensor range check" && 0 <= v1155 && v1155 < 4l);
                int v1157;
                v1157 = 4l * v1153;
                int v1158;
                v1158 = v1157 + v1155;
                int v1159;
                v1159 = v992[v1158];
                float v1160;
                v1160 = curand_uniform(&v974);
                assert("Tensor range check" && 0 <= v1153 && v1153 < 1l);
                assert("Tensor range check" && 0 <= v1155 && v1155 < 4l);
                v1151[v1158] = v1160;
                v1152[v1158] = v1159;
                v1155 += 1l ;
            }
            v1153 += 1l ;
        }
        float v1161; int v1162;
        Tuple1 tmp34 = Tuple1{0.0f, 2147483647l};
        v1161 = tmp34.v0; v1162 = tmp34.v1;
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
                float v1169;
                v1169 = v1151[v1168];
                int v1170;
                v1170 = v1152[v1168];
                bool v1171;
                v1171 = v1162 < v1170;
                float v1172; int v1173;
                if (v1171){
                    v1172 = v1161; v1173 = v1162;
                } else {
                    v1172 = v1169; v1173 = v1170;
                }
                v1161 = v1172;
                v1162 = v1173;
                v1165 += 1l ;
            }
            v1163 += 1l ;
        }
        auto v1174 = cooperative_groups::coalesced_threads();
        int v1175;
        v1175 = threadIdx.x;
        int v1176;
        v1176 = v1175 / 16l;
        auto v1177 = cooperative_groups::labeled_partition(v1174,v1176);
        Closure6 v1178{};
        float v1179; int v1180;
        Tuple1 tmp35 = cooperative_groups::reduce(v1177, Tuple1{v1161, v1162}, v1178);
        v1179 = tmp35.v0; v1180 = tmp35.v1;
        float v1181;
        v1181 = v1147 * v1179;
        int v1182[4l];
        bool v1183[4l];
        int v1184;
        v1184 = 0l;
        while (while_method_3(v1184)){
            int v1186;
            v1186 = 0l;
            while (while_method_1(v1186)){
                assert("Tensor range check" && 0 <= v1184 && v1184 < 1l);
                assert("Tensor range check" && 0 <= v1186 && v1186 < 4l);
                int v1188;
                v1188 = 4l * v1184;
                int v1189;
                v1189 = v1188 + v1186;
                float v1190;
                v1190 = v1113[v1189];
                bool v1191;
                v1191 = v1114[v1189];
                int v1192;
                v1192 = v992[v1189];
                int v1195; bool v1196;
                if (v1191){
                    float v1193;
                    v1193 = v1190 - v1181;
                    bool v1194;
                    v1194 = v1193 >= 0.0f;
                    v1195 = v1192; v1196 = v1194;
                } else {
                    v1195 = 2147483647l; v1196 = false;
                }
                assert("Tensor range check" && 0 <= v1184 && v1184 < 1l);
                assert("Tensor range check" && 0 <= v1186 && v1186 < 4l);
                v1182[v1189] = v1195;
                v1183[v1189] = v1196;
                v1186 += 1l ;
            }
            v1184 += 1l ;
        }
        int v1197; bool v1198;
        Tuple4 tmp36 = Tuple4{2147483647l, false};
        v1197 = tmp36.v0; v1198 = tmp36.v1;
        int v1199;
        v1199 = 0l;
        while (while_method_3(v1199)){
            int v1201;
            v1201 = 0l;
            while (while_method_1(v1201)){
                assert("Tensor range check" && 0 <= v1199 && v1199 < 1l);
                assert("Tensor range check" && 0 <= v1201 && v1201 < 4l);
                int v1203;
                v1203 = 4l * v1199;
                int v1204;
                v1204 = v1203 + v1201;
                int v1205;
                v1205 = v1182[v1204];
                bool v1206;
                v1206 = v1183[v1204];
                int v1213; bool v1214;
                if (v1198){
                    if (v1206){
                        bool v1207;
                        v1207 = v1197 < v1205;
                        int v1208;
                        if (v1207){
                            v1208 = v1197;
                        } else {
                            v1208 = v1205;
                        }
                        v1213 = v1208; v1214 = true;
                    } else {
                        v1213 = v1197; v1214 = v1198;
                    }
                } else {
                    if (v1206){
                        v1213 = v1205; v1214 = v1206;
                    } else {
                        v1213 = v1197; v1214 = v1198;
                    }
                }
                v1197 = v1213;
                v1198 = v1214;
                v1201 += 1l ;
            }
            v1199 += 1l ;
        }
        auto v1215 = cooperative_groups::coalesced_threads();
        int v1216;
        v1216 = threadIdx.x;
        int v1217;
        v1217 = v1216 / 16l;
        auto v1218 = cooperative_groups::labeled_partition(v1215,v1217);
        Closure7 v1219{};
        int v1220; bool v1221;
        Tuple4 tmp37 = cooperative_groups::reduce(v1218, Tuple4{v1197, v1198}, v1219);
        v1220 = tmp37.v0; v1221 = tmp37.v1;
        bool v1222;
        v1222 = v1221 == false;
        if (v1222){
            assert("The local reduce must be true." && v1221);
        } else {
        }
        assert("Tensor range check" && 0 <= v987 && v987 < 64l);
        int v1224;
        v1224 = 0l;
        while (while_method_3(v1224)){
            assert("Tensor range check" && 0 <= v1224 && v1224 < 1l);
            int v1226;
            v1226 = 64l * v1224;
            int v1227;
            v1227 = v1226 + v990;
            assert("Tensor range check" && 0 <= v1224 && v1224 < 1l);
            int v1228;
            v1228 = 4l * v1224;
            int4* v1229;
            v1229 = reinterpret_cast<int4*>(v1075 + v1228);
            int4* v1230;
            v1230 = reinterpret_cast<int4*>(v14 + v1227);
            assert("Pointer alignment check" && (unsigned long long)(v1229) % 4l == 0 && (unsigned long long)(v1230) % 4l == 0);
            *v1230 = *v1229;
            v1224 += 1l ;
        }
        assert("Tensor range check" && 0 <= v987 && v987 < 64l);
        int v1231;
        v1231 = 2l * v987;
        int v1232;
        v1232 = v1231 + v980;
        v15[v1232] = v1220;
        v987 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v1233;
    v1233 = threadIdx.x;
    unsigned long long v1234;
    v1234 = (unsigned long long)v1233;
    curandStatePhilox4_32_10_t v1235;
    curand_init(12344321ull,v1234,0ull,&v1235);
    int v1236;
    v1236 = threadIdx.x;
    bool v1237;
    v1237 = 0l <= v1236;
    bool v1238;
    v1238 = v1237 == false;
    if (v1238){
        assert("The index needs to be zero or positive." && v1237);
    } else {
    }
    int v1240;
    v1240 = v1236 % 16l;
    int v1241;
    v1241 = v1236 / 16l;
    bool v1242;
    v1242 = v1241 < 2l;
    bool v1243;
    v1243 = v1242 == false;
    if (v1243){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1242);
    } else {
    }
    assert("Tensor range check" && 0 <= v1241 && v1241 < 2l);
    assert("Tensor range check" && 0 <= v1240 && v1240 < 16l);
    int v1245;
    v1245 = 4l * v1240;
    int v1246;
    v1246 = 64l * v1241;
    int v1247;
    v1247 = v1246 + v1245;
    assert("Tensor range check" && 0 <= v1241 && v1241 < 2l);
    assert("Tensor range check" && 0 <= v1240 && v1240 < 16l);
    assert("Tensor range check" && 0 <= v1241 && v1241 < 2l);
    int v1248;
    v1248 = 0l;
    while (while_method_2(v1248)){
        assert("Tensor range check" && 0 <= v1248 && v1248 < 64l);
        int v1250;
        v1250 = 128l * v1248;
        int v1251;
        v1251 = v1250 + v1247;
        float v1252[4l];
        int v1253[4l];
        int v1254;
        v1254 = 0l;
        while (while_method_3(v1254)){
            assert("Tensor range check" && 0 <= v1254 && v1254 < 1l);
            int v1256;
            v1256 = 4l * v1254;
            assert("Tensor range check" && 0 <= v1254 && v1254 < 1l);
            int v1257;
            v1257 = 64l * v1254;
            int v1258;
            v1258 = v1257 + v1251;
            int4* v1259;
            v1259 = reinterpret_cast<int4*>(v1 + v1258);
            int4* v1260;
            v1260 = reinterpret_cast<int4*>(v1252 + v1256);
            assert("Pointer alignment check" && (unsigned long long)(v1259) % 4l == 0 && (unsigned long long)(v1260) % 4l == 0);
            *v1260 = *v1259;
            v1254 += 1l ;
        }
        int v1261;
        v1261 = 0l;
        while (while_method_3(v1261)){
            int v1263;
            v1263 = 0l;
            while (while_method_1(v1263)){
                bool v1265;
                v1265 = 0l <= v1263;
                bool v1267;
                if (v1265){
                    bool v1266;
                    v1266 = v1263 < 4l;
                    v1267 = v1266;
                } else {
                    v1267 = false;
                }
                bool v1268;
                v1268 = v1267 == false;
                if (v1268){
                    assert("The indices should be inside the range of the dimension." && v1267);
                } else {
                }
                bool v1270;
                v1270 = 0l <= v1240;
                bool v1272;
                if (v1270){
                    bool v1271;
                    v1271 = v1240 < 16l;
                    v1272 = v1271;
                } else {
                    v1272 = false;
                }
                bool v1273;
                v1273 = v1272 == false;
                if (v1273){
                    assert("The indices should be inside the range of the dimension." && v1272);
                } else {
                }
                int v1275;
                v1275 = v1240 * 4l;
                int v1276;
                v1276 = v1263 + v1275;
                bool v1277;
                v1277 = 0l <= v1261;
                bool v1279;
                if (v1277){
                    bool v1278;
                    v1278 = v1261 < 1l;
                    v1279 = v1278;
                } else {
                    v1279 = false;
                }
                bool v1280;
                v1280 = v1279 == false;
                if (v1280){
                    assert("The indices should be inside the range of the dimension." && v1279);
                } else {
                }
                int v1282;
                v1282 = v1261 * 64l;
                int v1283;
                v1283 = v1276 + v1282;
                assert("Tensor range check" && 0 <= v1261 && v1261 < 1l);
                assert("Tensor range check" && 0 <= v1263 && v1263 < 4l);
                int v1284;
                v1284 = 4l * v1261;
                int v1285;
                v1285 = v1284 + v1263;
                v1253[v1285] = v1283;
                v1263 += 1l ;
            }
            v1261 += 1l ;
        }
        bool v1286;
        v1286 = 0l <= v1241;
        bool v1287;
        v1287 = v1286 && v1242;
        bool v1288;
        v1288 = v1287 == false;
        if (v1288){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1287);
        } else {
        }
        bool v1290;
        v1290 = 0l <= v1248;
        bool v1292;
        if (v1290){
            bool v1291;
            v1291 = v1248 < 64l;
            v1292 = v1291;
        } else {
            v1292 = false;
        }
        bool v1293;
        v1293 = v1292 == false;
        if (v1293){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1292);
        } else {
        }
        int v1295;
        v1295 = v1248 * 2l;
        int v1296;
        v1296 = v1295 + v1241;
        bool v1297[4l];
        int v1298;
        v1298 = 0l;
        while (while_method_3(v1298)){
            int v1300;
            v1300 = 0l;
            while (while_method_1(v1300)){
                assert("Tensor range check" && 0 <= v1298 && v1298 < 1l);
                assert("Tensor range check" && 0 <= v1300 && v1300 < 4l);
                int v1302;
                v1302 = 4l * v1298;
                int v1303;
                v1303 = v1302 + v1300;
                float v1304;
                v1304 = v1252[v1303];
                int v1305;
                v1305 = v1253[v1303];
                bool v1306;
                v1306 = v1305 < 11l;
                assert("Tensor range check" && 0 <= v1298 && v1298 < 1l);
                assert("Tensor range check" && 0 <= v1300 && v1300 < 4l);
                v1297[v1303] = v1306;
                v1300 += 1l ;
            }
            v1298 += 1l ;
        }
        int v1307[4l];
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
                bool v1314;
                v1314 = v1297[v1313];
                int v1315;
                if (v1314){
                    v1315 = 1l;
                } else {
                    v1315 = 0l;
                }
                assert("Tensor range check" && 0 <= v1308 && v1308 < 1l);
                assert("Tensor range check" && 0 <= v1310 && v1310 < 4l);
                v1307[v1313] = v1315;
                v1310 += 1l ;
            }
            v1308 += 1l ;
        }
        int v1316;
        v1316 = 0l;
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
                int v1323;
                v1323 = v1307[v1322];
                int v1324;
                v1324 = v1316 + v1323;
                v1316 = v1324;
                v1319 += 1l ;
            }
            v1317 += 1l ;
        }
        auto v1325 = cooperative_groups::coalesced_threads();
        int v1326;
        v1326 = threadIdx.x;
        int v1327;
        v1327 = v1326 / 16l;
        auto v1328 = cooperative_groups::labeled_partition(v1325,v1327);
        Closure4 v1329{};
        int v1330;
        v1330 = cooperative_groups::reduce(v1328, v1316, v1329);
        float v1331[4l];
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
                float v1338;
                v1338 = v1252[v1337];
                bool v1339;
                v1339 = v1297[v1337];
                float v1340;
                if (v1339){
                    v1340 = v1338;
                } else {
                    v1340 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1332 && v1332 < 1l);
                assert("Tensor range check" && 0 <= v1334 && v1334 < 4l);
                v1331[v1337] = v1340;
                v1334 += 1l ;
            }
            v1332 += 1l ;
        }
        float v1341;
        v1341 = 0.0f;
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
                v1348 = v1331[v1347];
                float v1349;
                v1349 = v1341 + v1348;
                v1341 = v1349;
                v1344 += 1l ;
            }
            v1342 += 1l ;
        }
        auto v1350 = cooperative_groups::coalesced_threads();
        int v1351;
        v1351 = threadIdx.x;
        int v1352;
        v1352 = v1351 / 16l;
        auto v1353 = cooperative_groups::labeled_partition(v1350,v1352);
        float v1354;
        v1354 = cooperative_groups::reduce(v1353, v1341, v40);
        float v1355;
        v1355 = (float)v1330;
        float v1356;
        v1356 = v1354 / v1355;
        float v1357[4l];
        int v1358;
        v1358 = 0l;
        while (while_method_3(v1358)){
            int v1360;
            v1360 = 0l;
            while (while_method_1(v1360)){
                assert("Tensor range check" && 0 <= v1358 && v1358 < 1l);
                assert("Tensor range check" && 0 <= v1360 && v1360 < 4l);
                int v1362;
                v1362 = 4l * v1358;
                int v1363;
                v1363 = v1362 + v1360;
                float v1364;
                v1364 = v1252[v1363];
                bool v1365;
                v1365 = v1297[v1363];
                float v1366;
                if (v1365){
                    v1366 = v1364;
                } else {
                    v1366 = -1.0f / 0.0f;
                }
                float v1367;
                v1367 = v1366 - v1356;
                float v1368;
                v1368 = exp(v1367);
                assert("Tensor range check" && 0 <= v1358 && v1358 < 1l);
                assert("Tensor range check" && 0 <= v1360 && v1360 < 4l);
                v1357[v1363] = v1368;
                v1360 += 1l ;
            }
            v1358 += 1l ;
        }
        float v1369;
        v1369 = 0.0f;
        int v1370;
        v1370 = 0l;
        while (while_method_3(v1370)){
            int v1372;
            v1372 = 0l;
            while (while_method_1(v1372)){
                assert("Tensor range check" && 0 <= v1370 && v1370 < 1l);
                assert("Tensor range check" && 0 <= v1372 && v1372 < 4l);
                int v1374;
                v1374 = 4l * v1370;
                int v1375;
                v1375 = v1374 + v1372;
                float v1376;
                v1376 = v1357[v1375];
                float v1377;
                v1377 = v1369 + v1376;
                v1369 = v1377;
                v1372 += 1l ;
            }
            v1370 += 1l ;
        }
        auto v1378 = cooperative_groups::coalesced_threads();
        int v1379;
        v1379 = threadIdx.x;
        int v1380;
        v1380 = v1379 / 16l;
        auto v1381 = cooperative_groups::labeled_partition(v1378,v1380);
        float v1382;
        v1382 = cooperative_groups::reduce(v1381, v1369, v40);
        float v1383[4l];
        int v1384;
        v1384 = 0l;
        while (while_method_3(v1384)){
            int v1386;
            v1386 = 0l;
            while (while_method_1(v1386)){
                assert("Tensor range check" && 0 <= v1384 && v1384 < 1l);
                assert("Tensor range check" && 0 <= v1386 && v1386 < 4l);
                int v1388;
                v1388 = 4l * v1384;
                int v1389;
                v1389 = v1388 + v1386;
                float v1390;
                v1390 = v1357[v1389];
                float v1391;
                v1391 = v1390 / v1382;
                assert("Tensor range check" && 0 <= v1384 && v1384 < 1l);
                assert("Tensor range check" && 0 <= v1386 && v1386 < 4l);
                v1383[v1389] = v1391;
                v1386 += 1l ;
            }
            v1384 += 1l ;
        }
        float v1392[4l];
        float v1393;
        v1393 = 0.0f;
        int v1394;
        v1394 = 0l;
        while (while_method_3(v1394)){
            assert("Tensor range check" && 0 <= v1394 && v1394 < 1l);
            int v1396;
            v1396 = 4l * v1394;
            assert("Tensor range check" && 0 <= v1394 && v1394 < 1l);
            int v1397; float v1398;
            Tuple0 tmp38 = Tuple0{0l, 0.0f};
            v1397 = tmp38.v0; v1398 = tmp38.v1;
            while (while_method_1(v1397)){
                assert("Tensor range check" && 0 <= v1397 && v1397 < 4l);
                int v1400;
                v1400 = v1397 + v1396;
                float v1401;
                v1401 = v1383[v1400];
                float v1402;
                v1402 = v1398 + v1401;
                v1398 = v1402;
                v1397 += 1l ;
            }
            auto v1403 = cooperative_groups::coalesced_threads();
            int v1404;
            v1404 = threadIdx.x;
            int v1405;
            v1405 = v1404 / 16l;
            auto v1406 = cooperative_groups::labeled_partition(v1403,v1405);
            Closure2 v1407{};
            float v1408;
            v1408 = cooperative_groups::inclusive_scan(v1406, v1398, v1407);
            float v1409;
            v1409 = v1406.shfl_up(v1408,1);
            bool v1410;
            v1410 = v1406.thread_rank() == 0;
            float v1411;
            if (v1410){
                v1411 = 0.0f;
            } else {
                v1411 = v1409;
            }
            float v1412;
            v1412 = v1406.shfl(v1408,v1406.num_threads()-1);
            float v1413;
            v1413 = v1393 + v1411;
            int v1414; float v1415;
            Tuple0 tmp39 = Tuple0{0l, v1413};
            v1414 = tmp39.v0; v1415 = tmp39.v1;
            while (while_method_1(v1414)){
                assert("Tensor range check" && 0 <= v1414 && v1414 < 4l);
                int v1417;
                v1417 = v1414 + v1396;
                float v1418;
                v1418 = v1383[v1417];
                float v1419;
                v1419 = v1415 + v1418;
                assert("Tensor range check" && 0 <= v1414 && v1414 < 4l);
                v1392[v1417] = v1419;
                v1415 = v1419;
                v1414 += 1l ;
            }
            float v1420;
            v1420 = v1393 + v1412;
            v1393 = v1420;
            v1394 += 1l ;
        }
        float v1421[4l];
        bool v1422[4l];
        int v1423;
        v1423 = 0l;
        while (while_method_3(v1423)){
            int v1425;
            v1425 = 0l;
            while (while_method_1(v1425)){
                assert("Tensor range check" && 0 <= v1423 && v1423 < 1l);
                assert("Tensor range check" && 0 <= v1425 && v1425 < 4l);
                int v1427;
                v1427 = 4l * v1423;
                int v1428;
                v1428 = v1427 + v1425;
                float v1429;
                v1429 = v1392[v1428];
                float v1430;
                v1430 = v1383[v1428];
                bool v1431;
                v1431 = v1430 > 0.0f;
                assert("Tensor range check" && 0 <= v1423 && v1423 < 1l);
                assert("Tensor range check" && 0 <= v1425 && v1425 < 4l);
                v1421[v1428] = v1429;
                v1422[v1428] = v1431;
                v1425 += 1l ;
            }
            v1423 += 1l ;
        }
        float v1432; bool v1433;
        Tuple3 tmp40 = Tuple3{-1.0f / 0.0f, false};
        v1432 = tmp40.v0; v1433 = tmp40.v1;
        int v1434;
        v1434 = 0l;
        while (while_method_3(v1434)){
            int v1436;
            v1436 = 0l;
            while (while_method_1(v1436)){
                assert("Tensor range check" && 0 <= v1434 && v1434 < 1l);
                assert("Tensor range check" && 0 <= v1436 && v1436 < 4l);
                int v1438;
                v1438 = 4l * v1434;
                int v1439;
                v1439 = v1438 + v1436;
                float v1440;
                v1440 = v1421[v1439];
                bool v1441;
                v1441 = v1422[v1439];
                float v1448; bool v1449;
                if (v1433){
                    if (v1441){
                        bool v1442;
                        v1442 = v1432 >= v1440;
                        float v1443;
                        if (v1442){
                            v1443 = v1432;
                        } else {
                            v1443 = v1440;
                        }
                        v1448 = v1443; v1449 = true;
                    } else {
                        v1448 = v1432; v1449 = v1433;
                    }
                } else {
                    if (v1441){
                        v1448 = v1440; v1449 = v1441;
                    } else {
                        v1448 = v1432; v1449 = v1433;
                    }
                }
                v1432 = v1448;
                v1433 = v1449;
                v1436 += 1l ;
            }
            v1434 += 1l ;
        }
        auto v1450 = cooperative_groups::coalesced_threads();
        int v1451;
        v1451 = threadIdx.x;
        int v1452;
        v1452 = v1451 / 16l;
        auto v1453 = cooperative_groups::labeled_partition(v1450,v1452);
        Closure5 v1454{};
        float v1455; bool v1456;
        Tuple3 tmp41 = cooperative_groups::reduce(v1453, Tuple3{v1432, v1433}, v1454);
        v1455 = tmp41.v0; v1456 = tmp41.v1;
        bool v1457;
        v1457 = v1456 == false;
        if (v1457){
            assert("The local reduce must be true." && v1456);
        } else {
        }
        float v1459[4l];
        int v1460[4l];
        int v1461;
        v1461 = 0l;
        while (while_method_3(v1461)){
            int v1463;
            v1463 = 0l;
            while (while_method_1(v1463)){
                assert("Tensor range check" && 0 <= v1461 && v1461 < 1l);
                assert("Tensor range check" && 0 <= v1463 && v1463 < 4l);
                int v1465;
                v1465 = 4l * v1461;
                int v1466;
                v1466 = v1465 + v1463;
                int v1467;
                v1467 = v1253[v1466];
                float v1468;
                v1468 = curand_uniform(&v1235);
                assert("Tensor range check" && 0 <= v1461 && v1461 < 1l);
                assert("Tensor range check" && 0 <= v1463 && v1463 < 4l);
                v1459[v1466] = v1468;
                v1460[v1466] = v1467;
                v1463 += 1l ;
            }
            v1461 += 1l ;
        }
        float v1469; int v1470;
        Tuple1 tmp42 = Tuple1{0.0f, 2147483647l};
        v1469 = tmp42.v0; v1470 = tmp42.v1;
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
                v1477 = v1459[v1476];
                int v1478;
                v1478 = v1460[v1476];
                bool v1479;
                v1479 = v1470 < v1478;
                float v1480; int v1481;
                if (v1479){
                    v1480 = v1469; v1481 = v1470;
                } else {
                    v1480 = v1477; v1481 = v1478;
                }
                v1469 = v1480;
                v1470 = v1481;
                v1473 += 1l ;
            }
            v1471 += 1l ;
        }
        auto v1482 = cooperative_groups::coalesced_threads();
        int v1483;
        v1483 = threadIdx.x;
        int v1484;
        v1484 = v1483 / 16l;
        auto v1485 = cooperative_groups::labeled_partition(v1482,v1484);
        Closure6 v1486{};
        float v1487; int v1488;
        Tuple1 tmp43 = cooperative_groups::reduce(v1485, Tuple1{v1469, v1470}, v1486);
        v1487 = tmp43.v0; v1488 = tmp43.v1;
        float v1489;
        v1489 = v1455 * v1487;
        int v1490[4l];
        bool v1491[4l];
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
                float v1498;
                v1498 = v1421[v1497];
                bool v1499;
                v1499 = v1422[v1497];
                int v1500;
                v1500 = v1253[v1497];
                int v1503; bool v1504;
                if (v1499){
                    float v1501;
                    v1501 = v1498 - v1489;
                    bool v1502;
                    v1502 = v1501 >= 0.0f;
                    v1503 = v1500; v1504 = v1502;
                } else {
                    v1503 = 2147483647l; v1504 = false;
                }
                assert("Tensor range check" && 0 <= v1492 && v1492 < 1l);
                assert("Tensor range check" && 0 <= v1494 && v1494 < 4l);
                v1490[v1497] = v1503;
                v1491[v1497] = v1504;
                v1494 += 1l ;
            }
            v1492 += 1l ;
        }
        int v1505; bool v1506;
        Tuple4 tmp44 = Tuple4{2147483647l, false};
        v1505 = tmp44.v0; v1506 = tmp44.v1;
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
                int v1513;
                v1513 = v1490[v1512];
                bool v1514;
                v1514 = v1491[v1512];
                int v1521; bool v1522;
                if (v1506){
                    if (v1514){
                        bool v1515;
                        v1515 = v1505 < v1513;
                        int v1516;
                        if (v1515){
                            v1516 = v1505;
                        } else {
                            v1516 = v1513;
                        }
                        v1521 = v1516; v1522 = true;
                    } else {
                        v1521 = v1505; v1522 = v1506;
                    }
                } else {
                    if (v1514){
                        v1521 = v1513; v1522 = v1514;
                    } else {
                        v1521 = v1505; v1522 = v1506;
                    }
                }
                v1505 = v1521;
                v1506 = v1522;
                v1509 += 1l ;
            }
            v1507 += 1l ;
        }
        auto v1523 = cooperative_groups::coalesced_threads();
        int v1524;
        v1524 = threadIdx.x;
        int v1525;
        v1525 = v1524 / 16l;
        auto v1526 = cooperative_groups::labeled_partition(v1523,v1525);
        Closure7 v1527{};
        int v1528; bool v1529;
        Tuple4 tmp45 = cooperative_groups::reduce(v1526, Tuple4{v1505, v1506}, v1527);
        v1528 = tmp45.v0; v1529 = tmp45.v1;
        bool v1530;
        v1530 = v1529 == false;
        if (v1530){
            assert("The local reduce must be true." && v1529);
        } else {
        }
        assert("Tensor range check" && 0 <= v1248 && v1248 < 64l);
        int v1532;
        v1532 = 0l;
        while (while_method_3(v1532)){
            assert("Tensor range check" && 0 <= v1532 && v1532 < 1l);
            int v1534;
            v1534 = 64l * v1532;
            int v1535;
            v1535 = v1534 + v1251;
            assert("Tensor range check" && 0 <= v1532 && v1532 < 1l);
            int v1536;
            v1536 = 4l * v1532;
            int4* v1537;
            v1537 = reinterpret_cast<int4*>(v1383 + v1536);
            int4* v1538;
            v1538 = reinterpret_cast<int4*>(v14 + v1535);
            assert("Pointer alignment check" && (unsigned long long)(v1537) % 4l == 0 && (unsigned long long)(v1538) % 4l == 0);
            *v1538 = *v1537;
            v1532 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1248 && v1248 < 64l);
        int v1539;
        v1539 = 2l * v1248;
        int v1540;
        v1540 = v1539 + v1241;
        v15[v1540] = v1528;
        v1248 += 1l ;
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
    __shared__ float * v27[32l];
    __shared__ int * v28[32l];
    __shared__ int * v29[32l];
    /* void shared array create v30 */;
    /* void shared array create v31 */;
    int v32;
    v32 = threadIdx.x;
    assert("Tensor range check" && 0 <= v32 && v32 < 32l);
    v27[v32] = v21;
    v28[v32] = v23;
    v29[v32] = v25;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v33;
    v33 = 0l <= v32;
    bool v34;
    v34 = v33 == false;
    if (v34){
        assert("The index needs to be zero or positive." && v33);
    } else {
    }
    int v36;
    v36 = v32 % 4l;
    int v37;
    v37 = v32 / 4l;
    bool v38;
    v38 = v37 < 8l;
    bool v39;
    v39 = v38 == false;
    if (v39){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v38);
    } else {
    }
    assert("Tensor range check" && 0 <= v37 && v37 < 8l);
    int v41;
    v41 = 0l;
    while (while_method_1(v41)){
        bool v43;
        v43 = 0l <= v37;
        bool v44;
        v44 = v43 && v38;
        bool v45;
        v45 = v44 == false;
        if (v45){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v44);
        } else {
        }
        bool v47;
        v47 = 0l <= v41;
        bool v49;
        if (v47){
            bool v48;
            v48 = v41 < 4l;
            v49 = v48;
        } else {
            v49 = false;
        }
        bool v50;
        v50 = v49 == false;
        if (v50){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v49);
        } else {
        }
        int v52;
        v52 = v41 * 8l;
        int v53;
        v53 = v52 + v37;
        assert("Tensor range check" && 0 <= v41 && v41 < 4l);
        int v54;
        v54 = 8l * v41;
        int v55;
        v55 = v54 + v37;
        float * v56;
        v56 = v27[v55];
        int * v57;
        v57 = v28[v55];
        int * v58;
        v58 = v29[v55];
        /* void array index */;
        int v59;
        v59 = blockIdx.x;
        int v60;
        v60 = v59 * 32l;
        int v61;
        v61 = v60 + v53;
        assert("Tensor range check" && 0 <= v36 && v36 < 4l);
        int v62;
        v62 = 4l * v36;
        float v63[4l];
        int v64[4l];
        int v65;
        v65 = 0l;
        while (while_method_3(v65)){
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            int v67;
            v67 = 4l * v65;
            assert("Tensor range check" && 0 <= v65 && v65 < 1l);
            int v68;
            v68 = 16l * v65;
            int v69;
            v69 = v68 + v62;
            int4* v70;
            v70 = reinterpret_cast<int4*>(v56 + v69);
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
                v81 = 0l <= v36;
                bool v83;
                if (v81){
                    bool v82;
                    v82 = v36 < 4l;
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
                v86 = v36 * 4l;
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
                v93 = v72 * 16l;
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
        int v97[4l];
        int v98[4l];
        int v99;
        v99 = 0l;
        while (while_method_3(v99)){
            int v101;
            v101 = 0l;
            while (while_method_1(v101)){
                assert("Tensor range check" && 0 <= v99 && v99 < 1l);
                assert("Tensor range check" && 0 <= v101 && v101 < 4l);
                int v103;
                v103 = 4l * v99;
                int v104;
                v104 = v103 + v101;
                int v105;
                v105 = v64[v104];
                assert("Tensor range check" && 0 <= v99 && v99 < 1l);
                assert("Tensor range check" && 0 <= v101 && v101 < 4l);
                v97[v104] = v61;
                v98[v104] = v105;
                v101 += 1l ;
            }
            v99 += 1l ;
        }
        int v106;
        v106 = 0l;
        while (while_method_3(v106)){
            assert("Tensor range check" && 0 <= v106 && v106 < 1l);
            int v108;
            v108 = 16l * v106;
            int v109;
            v109 = v108 + v62;
            assert("Tensor range check" && 0 <= v106 && v106 < 1l);
            int v110;
            v110 = 4l * v106;
            int4* v111;
            v111 = reinterpret_cast<int4*>(v97 + v110);
            int4* v112;
            v112 = reinterpret_cast<int4*>(v57 + v109);
            assert("Pointer alignment check" && (unsigned long long)(v111) % 4l == 0 && (unsigned long long)(v112) % 4l == 0);
            *v112 = *v111;
            int4* v113;
            v113 = reinterpret_cast<int4*>(v98 + v110);
            int4* v114;
            v114 = reinterpret_cast<int4*>(v58 + v109);
            assert("Pointer alignment check" && (unsigned long long)(v113) % 4l == 0 && (unsigned long long)(v114) % 4l == 0);
            *v114 = *v113;
            v106 += 1l ;
        }
        assert("Tensor range check" && 0 <= v53 && v53 < 32l);
        /* void array set */;
        v41 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v32 && v32 < 32l);
    /* void array index */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v115;
    v115 = v1+v12;
    __shared__ float * v117[32l];
    /* void shared array create v118 */;
    __shared__ int v119[32l];
    int v120;
    v120 = threadIdx.x;
    assert("Tensor range check" && 0 <= v120 && v120 < 32l);
    v117[v120] = v115;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v121;
    v121 = 0l <= v120;
    bool v122;
    v122 = v121 == false;
    if (v122){
        assert("The index needs to be zero or positive." && v121);
    } else {
    }
    int v124;
    v124 = v120 % 4l;
    int v125;
    v125 = v120 / 4l;
    bool v126;
    v126 = v125 < 8l;
    bool v127;
    v127 = v126 == false;
    if (v127){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v126);
    } else {
    }
    assert("Tensor range check" && 0 <= v125 && v125 < 8l);
    int v129;
    v129 = 0l;
    while (while_method_1(v129)){
        bool v131;
        v131 = 0l <= v125;
        bool v132;
        v132 = v131 && v126;
        bool v133;
        v133 = v132 == false;
        if (v133){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v132);
        } else {
        }
        bool v135;
        v135 = 0l <= v129;
        bool v137;
        if (v135){
            bool v136;
            v136 = v129 < 4l;
            v137 = v136;
        } else {
            v137 = false;
        }
        bool v138;
        v138 = v137 == false;
        if (v138){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v137);
        } else {
        }
        int v140;
        v140 = v129 * 8l;
        int v141;
        v141 = v140 + v125;
        assert("Tensor range check" && 0 <= v129 && v129 < 4l);
        int v142;
        v142 = 8l * v129;
        int v143;
        v143 = v142 + v125;
        float * v144;
        v144 = v117[v143];
        /* void array index */;
        int v145;
        v145 = blockIdx.x;
        int v146;
        v146 = v145 * 32l;
        int v147;
        v147 = v146 + v141;
        assert("Tensor range check" && 0 <= v124 && v124 < 4l);
        int v148;
        v148 = 4l * v124;
        float v149[4l];
        int v150[4l];
        int v151;
        v151 = 0l;
        while (while_method_3(v151)){
            assert("Tensor range check" && 0 <= v151 && v151 < 1l);
            int v153;
            v153 = 4l * v151;
            assert("Tensor range check" && 0 <= v151 && v151 < 1l);
            int v154;
            v154 = 16l * v151;
            int v155;
            v155 = v154 + v148;
            int4* v156;
            v156 = reinterpret_cast<int4*>(v144 + v155);
            int4* v157;
            v157 = reinterpret_cast<int4*>(v149 + v153);
            assert("Pointer alignment check" && (unsigned long long)(v156) % 4l == 0 && (unsigned long long)(v157) % 4l == 0);
            *v157 = *v156;
            v151 += 1l ;
        }
        int v158;
        v158 = 0l;
        while (while_method_3(v158)){
            int v160;
            v160 = 0l;
            while (while_method_1(v160)){
                bool v162;
                v162 = 0l <= v160;
                bool v164;
                if (v162){
                    bool v163;
                    v163 = v160 < 4l;
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
                bool v167;
                v167 = 0l <= v124;
                bool v169;
                if (v167){
                    bool v168;
                    v168 = v124 < 4l;
                    v169 = v168;
                } else {
                    v169 = false;
                }
                bool v170;
                v170 = v169 == false;
                if (v170){
                    assert("The indices should be inside the range of the dimension." && v169);
                } else {
                }
                int v172;
                v172 = v124 * 4l;
                int v173;
                v173 = v160 + v172;
                bool v174;
                v174 = 0l <= v158;
                bool v176;
                if (v174){
                    bool v175;
                    v175 = v158 < 1l;
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
                v179 = v158 * 16l;
                int v180;
                v180 = v173 + v179;
                assert("Tensor range check" && 0 <= v158 && v158 < 1l);
                assert("Tensor range check" && 0 <= v160 && v160 < 4l);
                int v181;
                v181 = 4l * v158;
                int v182;
                v182 = v181 + v160;
                v150[v182] = v180;
                v160 += 1l ;
            }
            v158 += 1l ;
        }
        int v183;
        v183 = 0l;
        while (while_method_3(v183)){
            assert("Tensor range check" && 0 <= v183 && v183 < 1l);
            assert("Tensor range check" && 0 <= v183 && v183 < 1l);
            v183 += 1l ;
        }
        assert("Tensor range check" && 0 <= v141 && v141 < 32l);
        v119[v141] = v147;
        v129 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v120 && v120 < 32l);
    int v185;
    v185 = v119[v120];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v186;
    v186 = threadIdx.x;
    assert("Tensor range check" && 0 <= v186 && v186 < 32l);
    v4[v186] = v185;
    float * v187;
    v187 = v1+v12;
    float * v189;
    v189 = v6+v20;
    __shared__ float * v191[32l];
    __shared__ float * v192[32l];
    /* void shared array create v193 */;
    /* void shared array create v194 */;
    int v195;
    v195 = threadIdx.x;
    assert("Tensor range check" && 0 <= v195 && v195 < 32l);
    v191[v195] = v187;
    v192[v195] = v189;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v196;
    v196 = 0l <= v195;
    bool v197;
    v197 = v196 == false;
    if (v197){
        assert("The index needs to be zero or positive." && v196);
    } else {
    }
    int v199;
    v199 = v195 % 4l;
    int v200;
    v200 = v195 / 4l;
    bool v201;
    v201 = v200 < 8l;
    bool v202;
    v202 = v201 == false;
    if (v202){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v201);
    } else {
    }
    assert("Tensor range check" && 0 <= v200 && v200 < 8l);
    int v204;
    v204 = 0l;
    while (while_method_1(v204)){
        bool v206;
        v206 = 0l <= v200;
        bool v207;
        v207 = v206 && v201;
        bool v208;
        v208 = v207 == false;
        if (v208){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v207);
        } else {
        }
        bool v210;
        v210 = 0l <= v204;
        bool v212;
        if (v210){
            bool v211;
            v211 = v204 < 4l;
            v212 = v211;
        } else {
            v212 = false;
        }
        bool v213;
        v213 = v212 == false;
        if (v213){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v212);
        } else {
        }
        int v215;
        v215 = v204 * 8l;
        int v216;
        v216 = v215 + v200;
        assert("Tensor range check" && 0 <= v204 && v204 < 4l);
        int v217;
        v217 = 8l * v204;
        int v218;
        v218 = v217 + v200;
        float * v219;
        v219 = v191[v218];
        float * v220;
        v220 = v192[v218];
        /* void array index */;
        int v221;
        v221 = blockIdx.x;
        int v222;
        v222 = v221 * 32l;
        int v223;
        v223 = v222 + v216;
        assert("Tensor range check" && 0 <= v199 && v199 < 4l);
        int v224;
        v224 = 4l * v199;
        float v225[4l];
        int v226[4l];
        int v227;
        v227 = 0l;
        while (while_method_3(v227)){
            assert("Tensor range check" && 0 <= v227 && v227 < 1l);
            int v229;
            v229 = 4l * v227;
            assert("Tensor range check" && 0 <= v227 && v227 < 1l);
            int v230;
            v230 = 16l * v227;
            int v231;
            v231 = v230 + v224;
            int4* v232;
            v232 = reinterpret_cast<int4*>(v219 + v231);
            int4* v233;
            v233 = reinterpret_cast<int4*>(v225 + v229);
            assert("Pointer alignment check" && (unsigned long long)(v232) % 4l == 0 && (unsigned long long)(v233) % 4l == 0);
            *v233 = *v232;
            v227 += 1l ;
        }
        int v234;
        v234 = 0l;
        while (while_method_3(v234)){
            int v236;
            v236 = 0l;
            while (while_method_1(v236)){
                bool v238;
                v238 = 0l <= v236;
                bool v240;
                if (v238){
                    bool v239;
                    v239 = v236 < 4l;
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
                bool v243;
                v243 = 0l <= v199;
                bool v245;
                if (v243){
                    bool v244;
                    v244 = v199 < 4l;
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
                v248 = v199 * 4l;
                int v249;
                v249 = v236 + v248;
                bool v250;
                v250 = 0l <= v234;
                bool v252;
                if (v250){
                    bool v251;
                    v251 = v234 < 1l;
                    v252 = v251;
                } else {
                    v252 = false;
                }
                bool v253;
                v253 = v252 == false;
                if (v253){
                    assert("The indices should be inside the range of the dimension." && v252);
                } else {
                }
                int v255;
                v255 = v234 * 16l;
                int v256;
                v256 = v249 + v255;
                assert("Tensor range check" && 0 <= v234 && v234 < 1l);
                assert("Tensor range check" && 0 <= v236 && v236 < 4l);
                int v257;
                v257 = 4l * v234;
                int v258;
                v258 = v257 + v236;
                v226[v258] = v256;
                v236 += 1l ;
            }
            v234 += 1l ;
        }
        int v259;
        v259 = 0l;
        while (while_method_3(v259)){
            assert("Tensor range check" && 0 <= v259 && v259 < 1l);
            int v261;
            v261 = 16l * v259;
            int v262;
            v262 = v261 + v224;
            assert("Tensor range check" && 0 <= v259 && v259 < 1l);
            int v263;
            v263 = 4l * v259;
            int4* v264;
            v264 = reinterpret_cast<int4*>(v225 + v263);
            int4* v265;
            v265 = reinterpret_cast<int4*>(v220 + v262);
            assert("Pointer alignment check" && (unsigned long long)(v264) % 4l == 0 && (unsigned long long)(v265) % 4l == 0);
            *v265 = *v264;
            v259 += 1l ;
        }
        assert("Tensor range check" && 0 <= v216 && v216 < 32l);
        /* void array set */;
        v204 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v195 && v195 < 32l);
    /* void array index */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v266;
    v266 = v1+v12;
    float * v268;
    v268 = v7+v16;
    __shared__ float * v270[32l];
    __shared__ float * v271[32l];
    /* void shared array create v272 */;
    /* void shared array create v273 */;
    int v274;
    v274 = threadIdx.x;
    assert("Tensor range check" && 0 <= v274 && v274 < 32l);
    v270[v274] = v266;
    v271[v274] = v268;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v275;
    v275 = 0l <= v274;
    bool v276;
    v276 = v275 == false;
    if (v276){
        assert("The index needs to be zero or positive." && v275);
    } else {
    }
    int v278;
    v278 = v274 % 4l;
    int v279;
    v279 = v274 / 4l;
    bool v280;
    v280 = v279 < 8l;
    bool v281;
    v281 = v280 == false;
    if (v281){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v280);
    } else {
    }
    assert("Tensor range check" && 0 <= v279 && v279 < 8l);
    int v283;
    v283 = 0l;
    while (while_method_1(v283)){
        bool v285;
        v285 = 0l <= v279;
        bool v286;
        v286 = v285 && v280;
        bool v287;
        v287 = v286 == false;
        if (v287){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v286);
        } else {
        }
        bool v289;
        v289 = 0l <= v283;
        bool v291;
        if (v289){
            bool v290;
            v290 = v283 < 4l;
            v291 = v290;
        } else {
            v291 = false;
        }
        bool v292;
        v292 = v291 == false;
        if (v292){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v291);
        } else {
        }
        int v294;
        v294 = v283 * 8l;
        int v295;
        v295 = v294 + v279;
        assert("Tensor range check" && 0 <= v283 && v283 < 4l);
        int v296;
        v296 = 8l * v283;
        int v297;
        v297 = v296 + v279;
        float * v298;
        v298 = v270[v297];
        float * v299;
        v299 = v271[v297];
        /* void array index */;
        int v300;
        v300 = blockIdx.x;
        int v301;
        v301 = v300 * 32l;
        int v302;
        v302 = v301 + v295;
        assert("Tensor range check" && 0 <= v278 && v278 < 4l);
        int v303;
        v303 = 4l * v278;
        float v304[4l];
        int v305[4l];
        int v306;
        v306 = 0l;
        while (while_method_3(v306)){
            assert("Tensor range check" && 0 <= v306 && v306 < 1l);
            int v308;
            v308 = 4l * v306;
            assert("Tensor range check" && 0 <= v306 && v306 < 1l);
            int v309;
            v309 = 16l * v306;
            int v310;
            v310 = v309 + v303;
            int4* v311;
            v311 = reinterpret_cast<int4*>(v298 + v310);
            int4* v312;
            v312 = reinterpret_cast<int4*>(v304 + v308);
            assert("Pointer alignment check" && (unsigned long long)(v311) % 4l == 0 && (unsigned long long)(v312) % 4l == 0);
            *v312 = *v311;
            v306 += 1l ;
        }
        int v313;
        v313 = 0l;
        while (while_method_3(v313)){
            int v315;
            v315 = 0l;
            while (while_method_1(v315)){
                bool v317;
                v317 = 0l <= v315;
                bool v319;
                if (v317){
                    bool v318;
                    v318 = v315 < 4l;
                    v319 = v318;
                } else {
                    v319 = false;
                }
                bool v320;
                v320 = v319 == false;
                if (v320){
                    assert("The indices should be inside the range of the dimension." && v319);
                } else {
                }
                bool v322;
                v322 = 0l <= v278;
                bool v324;
                if (v322){
                    bool v323;
                    v323 = v278 < 4l;
                    v324 = v323;
                } else {
                    v324 = false;
                }
                bool v325;
                v325 = v324 == false;
                if (v325){
                    assert("The indices should be inside the range of the dimension." && v324);
                } else {
                }
                int v327;
                v327 = v278 * 4l;
                int v328;
                v328 = v315 + v327;
                bool v329;
                v329 = 0l <= v313;
                bool v331;
                if (v329){
                    bool v330;
                    v330 = v313 < 1l;
                    v331 = v330;
                } else {
                    v331 = false;
                }
                bool v332;
                v332 = v331 == false;
                if (v332){
                    assert("The indices should be inside the range of the dimension." && v331);
                } else {
                }
                int v334;
                v334 = v313 * 16l;
                int v335;
                v335 = v328 + v334;
                assert("Tensor range check" && 0 <= v313 && v313 < 1l);
                assert("Tensor range check" && 0 <= v315 && v315 < 4l);
                int v336;
                v336 = 4l * v313;
                int v337;
                v337 = v336 + v315;
                v305[v337] = v335;
                v315 += 1l ;
            }
            v313 += 1l ;
        }
        bool v338[4l];
        int v339;
        v339 = 0l;
        while (while_method_3(v339)){
            int v341;
            v341 = 0l;
            while (while_method_1(v341)){
                assert("Tensor range check" && 0 <= v339 && v339 < 1l);
                assert("Tensor range check" && 0 <= v341 && v341 < 4l);
                int v343;
                v343 = 4l * v339;
                int v344;
                v344 = v343 + v341;
                float v345;
                v345 = v304[v344];
                int v346;
                v346 = v305[v344];
                bool v347;
                v347 = v346 < 3l;
                assert("Tensor range check" && 0 <= v339 && v339 < 1l);
                assert("Tensor range check" && 0 <= v341 && v341 < 4l);
                v338[v344] = v347;
                v341 += 1l ;
            }
            v339 += 1l ;
        }
        float v348[4l];
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
                v355 = v304[v354];
                bool v356;
                v356 = v338[v354];
                float v359;
                if (v356){
                    bool v357;
                    v357 = 0.0f >= v355;
                    if (v357){
                        v359 = 0.0f;
                    } else {
                        v359 = v355;
                    }
                } else {
                    v359 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v349 && v349 < 1l);
                assert("Tensor range check" && 0 <= v351 && v351 < 4l);
                v348[v354] = v359;
                v351 += 1l ;
            }
            v349 += 1l ;
        }
        float v360;
        v360 = 0.0f;
        int v361;
        v361 = 0l;
        while (while_method_3(v361)){
            int v363;
            v363 = 0l;
            while (while_method_1(v363)){
                assert("Tensor range check" && 0 <= v361 && v361 < 1l);
                assert("Tensor range check" && 0 <= v363 && v363 < 4l);
                int v365;
                v365 = 4l * v361;
                int v366;
                v366 = v365 + v363;
                float v367;
                v367 = v348[v366];
                float v368;
                v368 = v360 + v367;
                v360 = v368;
                v363 += 1l ;
            }
            v361 += 1l ;
        }
        auto v369 = cooperative_groups::coalesced_threads();
        int v370;
        v370 = threadIdx.x;
        int v371;
        v371 = v370 / 4l;
        auto v372 = cooperative_groups::labeled_partition(v369,v371);
        Closure0 v373{};
        float v374;
        v374 = cooperative_groups::reduce(v372, v360, v373);
        int v375[4l];
        int v376;
        v376 = 0l;
        while (while_method_3(v376)){
            int v378;
            v378 = 0l;
            while (while_method_1(v378)){
                assert("Tensor range check" && 0 <= v376 && v376 < 1l);
                assert("Tensor range check" && 0 <= v378 && v378 < 4l);
                int v380;
                v380 = 4l * v376;
                int v381;
                v381 = v380 + v378;
                bool v382;
                v382 = v338[v381];
                int v383;
                if (v382){
                    v383 = 1l;
                } else {
                    v383 = 0l;
                }
                assert("Tensor range check" && 0 <= v376 && v376 < 1l);
                assert("Tensor range check" && 0 <= v378 && v378 < 4l);
                v375[v381] = v383;
                v378 += 1l ;
            }
            v376 += 1l ;
        }
        int v384;
        v384 = 0l;
        int v385;
        v385 = 0l;
        while (while_method_3(v385)){
            int v387;
            v387 = 0l;
            while (while_method_1(v387)){
                assert("Tensor range check" && 0 <= v385 && v385 < 1l);
                assert("Tensor range check" && 0 <= v387 && v387 < 4l);
                int v389;
                v389 = 4l * v385;
                int v390;
                v390 = v389 + v387;
                int v391;
                v391 = v375[v390];
                int v392;
                v392 = v384 + v391;
                v384 = v392;
                v387 += 1l ;
            }
            v385 += 1l ;
        }
        auto v393 = cooperative_groups::coalesced_threads();
        int v394;
        v394 = threadIdx.x;
        int v395;
        v395 = v394 / 4l;
        auto v396 = cooperative_groups::labeled_partition(v393,v395);
        Closure4 v397{};
        int v398;
        v398 = cooperative_groups::reduce(v396, v384, v397);
        float v399;
        v399 = (float)v398;
        float v400;
        v400 = 1.0f / v399;
        float v401[4l];
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
                v408 = v348[v407];
                bool v409;
                v409 = v338[v407];
                bool v410;
                v410 = v409 == false;
                float v415;
                if (v410){
                    v415 = 0.0f;
                } else {
                    bool v411;
                    v411 = v374 == 0.0f;
                    bool v412;
                    v412 = v411 != true;
                    if (v412){
                        float v413;
                        v413 = v408 / v374;
                        v415 = v413;
                    } else {
                        v415 = v400;
                    }
                }
                assert("Tensor range check" && 0 <= v402 && v402 < 1l);
                assert("Tensor range check" && 0 <= v404 && v404 < 4l);
                v401[v407] = v415;
                v404 += 1l ;
            }
            v402 += 1l ;
        }
        int v416;
        v416 = 0l;
        while (while_method_3(v416)){
            assert("Tensor range check" && 0 <= v416 && v416 < 1l);
            int v418;
            v418 = 16l * v416;
            int v419;
            v419 = v418 + v303;
            assert("Tensor range check" && 0 <= v416 && v416 < 1l);
            int v420;
            v420 = 4l * v416;
            int4* v421;
            v421 = reinterpret_cast<int4*>(v401 + v420);
            int4* v422;
            v422 = reinterpret_cast<int4*>(v299 + v419);
            assert("Pointer alignment check" && (unsigned long long)(v421) % 4l == 0 && (unsigned long long)(v422) % 4l == 0);
            *v422 = *v421;
            v416 += 1l ;
        }
        assert("Tensor range check" && 0 <= v295 && v295 < 32l);
        /* void array set */;
        v283 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v274 && v274 < 32l);
    /* void array index */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v423;
    v423 = v1+v12;
    __shared__ float * v425[32l];
    /* void shared array create v426 */;
    __shared__ int v427[32l];
    int v428;
    v428 = threadIdx.x;
    assert("Tensor range check" && 0 <= v428 && v428 < 32l);
    v425[v428] = v423;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v429;
    v429 = 0l <= v428;
    bool v430;
    v430 = v429 == false;
    if (v430){
        assert("The index needs to be zero or positive." && v429);
    } else {
    }
    int v432;
    v432 = v428 % 4l;
    int v433;
    v433 = v428 / 4l;
    bool v434;
    v434 = v433 < 8l;
    bool v435;
    v435 = v434 == false;
    if (v435){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v434);
    } else {
    }
    assert("Tensor range check" && 0 <= v433 && v433 < 8l);
    int v437;
    v437 = 0l;
    while (while_method_1(v437)){
        bool v439;
        v439 = 0l <= v433;
        bool v440;
        v440 = v439 && v434;
        bool v441;
        v441 = v440 == false;
        if (v441){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v440);
        } else {
        }
        bool v443;
        v443 = 0l <= v437;
        bool v445;
        if (v443){
            bool v444;
            v444 = v437 < 4l;
            v445 = v444;
        } else {
            v445 = false;
        }
        bool v446;
        v446 = v445 == false;
        if (v446){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v445);
        } else {
        }
        int v448;
        v448 = v437 * 8l;
        int v449;
        v449 = v448 + v433;
        assert("Tensor range check" && 0 <= v437 && v437 < 4l);
        int v450;
        v450 = 8l * v437;
        int v451;
        v451 = v450 + v433;
        float * v452;
        v452 = v425[v451];
        /* void array index */;
        int v453;
        v453 = blockIdx.x;
        int v454;
        v454 = v453 * 32l;
        int v455;
        v455 = v454 + v449;
        assert("Tensor range check" && 0 <= v432 && v432 < 4l);
        int v456;
        v456 = 4l * v432;
        float v457[4l];
        int v458[4l];
        int v459;
        v459 = 0l;
        while (while_method_3(v459)){
            assert("Tensor range check" && 0 <= v459 && v459 < 1l);
            int v461;
            v461 = 4l * v459;
            assert("Tensor range check" && 0 <= v459 && v459 < 1l);
            int v462;
            v462 = 16l * v459;
            int v463;
            v463 = v462 + v456;
            int4* v464;
            v464 = reinterpret_cast<int4*>(v452 + v463);
            int4* v465;
            v465 = reinterpret_cast<int4*>(v457 + v461);
            assert("Pointer alignment check" && (unsigned long long)(v464) % 4l == 0 && (unsigned long long)(v465) % 4l == 0);
            *v465 = *v464;
            v459 += 1l ;
        }
        int v466;
        v466 = 0l;
        while (while_method_3(v466)){
            int v468;
            v468 = 0l;
            while (while_method_1(v468)){
                bool v470;
                v470 = 0l <= v468;
                bool v472;
                if (v470){
                    bool v471;
                    v471 = v468 < 4l;
                    v472 = v471;
                } else {
                    v472 = false;
                }
                bool v473;
                v473 = v472 == false;
                if (v473){
                    assert("The indices should be inside the range of the dimension." && v472);
                } else {
                }
                bool v475;
                v475 = 0l <= v432;
                bool v477;
                if (v475){
                    bool v476;
                    v476 = v432 < 4l;
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
                v480 = v432 * 4l;
                int v481;
                v481 = v468 + v480;
                bool v482;
                v482 = 0l <= v466;
                bool v484;
                if (v482){
                    bool v483;
                    v483 = v466 < 1l;
                    v484 = v483;
                } else {
                    v484 = false;
                }
                bool v485;
                v485 = v484 == false;
                if (v485){
                    assert("The indices should be inside the range of the dimension." && v484);
                } else {
                }
                int v487;
                v487 = v466 * 16l;
                int v488;
                v488 = v481 + v487;
                assert("Tensor range check" && 0 <= v466 && v466 < 1l);
                assert("Tensor range check" && 0 <= v468 && v468 < 4l);
                int v489;
                v489 = 4l * v466;
                int v490;
                v490 = v489 + v468;
                v458[v490] = v488;
                v468 += 1l ;
            }
            v466 += 1l ;
        }
        bool v491[4l];
        int v492;
        v492 = 0l;
        while (while_method_3(v492)){
            int v494;
            v494 = 0l;
            while (while_method_1(v494)){
                assert("Tensor range check" && 0 <= v492 && v492 < 1l);
                assert("Tensor range check" && 0 <= v494 && v494 < 4l);
                int v496;
                v496 = 4l * v492;
                int v497;
                v497 = v496 + v494;
                float v498;
                v498 = v457[v497];
                int v499;
                v499 = v458[v497];
                bool v500;
                v500 = v499 < 3l;
                assert("Tensor range check" && 0 <= v492 && v492 < 1l);
                assert("Tensor range check" && 0 <= v494 && v494 < 4l);
                v491[v497] = v500;
                v494 += 1l ;
            }
            v492 += 1l ;
        }
        int v501[4l];
        int v502;
        v502 = 0l;
        while (while_method_3(v502)){
            int v504;
            v504 = 0l;
            while (while_method_1(v504)){
                assert("Tensor range check" && 0 <= v502 && v502 < 1l);
                assert("Tensor range check" && 0 <= v504 && v504 < 4l);
                int v506;
                v506 = 4l * v502;
                int v507;
                v507 = v506 + v504;
                bool v508;
                v508 = v491[v507];
                int v509;
                if (v508){
                    v509 = 1l;
                } else {
                    v509 = 0l;
                }
                assert("Tensor range check" && 0 <= v502 && v502 < 1l);
                assert("Tensor range check" && 0 <= v504 && v504 < 4l);
                v501[v507] = v509;
                v504 += 1l ;
            }
            v502 += 1l ;
        }
        int v510;
        v510 = 0l;
        int v511;
        v511 = 0l;
        while (while_method_3(v511)){
            int v513;
            v513 = 0l;
            while (while_method_1(v513)){
                assert("Tensor range check" && 0 <= v511 && v511 < 1l);
                assert("Tensor range check" && 0 <= v513 && v513 < 4l);
                int v515;
                v515 = 4l * v511;
                int v516;
                v516 = v515 + v513;
                int v517;
                v517 = v501[v516];
                int v518;
                v518 = v510 + v517;
                v510 = v518;
                v513 += 1l ;
            }
            v511 += 1l ;
        }
        auto v519 = cooperative_groups::coalesced_threads();
        int v520;
        v520 = threadIdx.x;
        int v521;
        v521 = v520 / 4l;
        auto v522 = cooperative_groups::labeled_partition(v519,v521);
        Closure4 v523{};
        int v524;
        v524 = cooperative_groups::reduce(v522, v510, v523);
        float v525[4l];
        int v526;
        v526 = 0l;
        while (while_method_3(v526)){
            int v528;
            v528 = 0l;
            while (while_method_1(v528)){
                assert("Tensor range check" && 0 <= v526 && v526 < 1l);
                assert("Tensor range check" && 0 <= v528 && v528 < 4l);
                int v530;
                v530 = 4l * v526;
                int v531;
                v531 = v530 + v528;
                float v532;
                v532 = v457[v531];
                bool v533;
                v533 = v491[v531];
                float v534;
                if (v533){
                    v534 = v532;
                } else {
                    v534 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v526 && v526 < 1l);
                assert("Tensor range check" && 0 <= v528 && v528 < 4l);
                v525[v531] = v534;
                v528 += 1l ;
            }
            v526 += 1l ;
        }
        float v535;
        v535 = 0.0f;
        int v536;
        v536 = 0l;
        while (while_method_3(v536)){
            int v538;
            v538 = 0l;
            while (while_method_1(v538)){
                assert("Tensor range check" && 0 <= v536 && v536 < 1l);
                assert("Tensor range check" && 0 <= v538 && v538 < 4l);
                int v540;
                v540 = 4l * v536;
                int v541;
                v541 = v540 + v538;
                float v542;
                v542 = v525[v541];
                float v543;
                v543 = v535 + v542;
                v535 = v543;
                v538 += 1l ;
            }
            v536 += 1l ;
        }
        auto v544 = cooperative_groups::coalesced_threads();
        int v545;
        v545 = threadIdx.x;
        int v546;
        v546 = v545 / 4l;
        auto v547 = cooperative_groups::labeled_partition(v544,v546);
        Closure0 v548{};
        float v549;
        v549 = cooperative_groups::reduce(v547, v535, v548);
        float v550;
        v550 = (float)v524;
        float v551;
        v551 = v549 / v550;
        float v552[4l];
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
                float v559;
                v559 = v457[v558];
                bool v560;
                v560 = v491[v558];
                float v561;
                if (v560){
                    v561 = v559;
                } else {
                    v561 = -1.0f / 0.0f;
                }
                float v562;
                v562 = v561 - v551;
                float v563;
                v563 = exp(v562);
                assert("Tensor range check" && 0 <= v553 && v553 < 1l);
                assert("Tensor range check" && 0 <= v555 && v555 < 4l);
                v552[v558] = v563;
                v555 += 1l ;
            }
            v553 += 1l ;
        }
        float v564;
        v564 = 0.0f;
        int v565;
        v565 = 0l;
        while (while_method_3(v565)){
            int v567;
            v567 = 0l;
            while (while_method_1(v567)){
                assert("Tensor range check" && 0 <= v565 && v565 < 1l);
                assert("Tensor range check" && 0 <= v567 && v567 < 4l);
                int v569;
                v569 = 4l * v565;
                int v570;
                v570 = v569 + v567;
                float v571;
                v571 = v552[v570];
                float v572;
                v572 = v564 + v571;
                v564 = v572;
                v567 += 1l ;
            }
            v565 += 1l ;
        }
        auto v573 = cooperative_groups::coalesced_threads();
        int v574;
        v574 = threadIdx.x;
        int v575;
        v575 = v574 / 4l;
        auto v576 = cooperative_groups::labeled_partition(v573,v575);
        float v577;
        v577 = cooperative_groups::reduce(v576, v564, v548);
        float v578[4l];
        int v579;
        v579 = 0l;
        while (while_method_3(v579)){
            int v581;
            v581 = 0l;
            while (while_method_1(v581)){
                assert("Tensor range check" && 0 <= v579 && v579 < 1l);
                assert("Tensor range check" && 0 <= v581 && v581 < 4l);
                int v583;
                v583 = 4l * v579;
                int v584;
                v584 = v583 + v581;
                float v585;
                v585 = v552[v584];
                float v586;
                v586 = v585 / v577;
                assert("Tensor range check" && 0 <= v579 && v579 < 1l);
                assert("Tensor range check" && 0 <= v581 && v581 < 4l);
                v578[v584] = v586;
                v581 += 1l ;
            }
            v579 += 1l ;
        }
        float v587[4l];
        float v588;
        v588 = 0.0f;
        int v589;
        v589 = 0l;
        while (while_method_3(v589)){
            assert("Tensor range check" && 0 <= v589 && v589 < 1l);
            int v591;
            v591 = 4l * v589;
            assert("Tensor range check" && 0 <= v589 && v589 < 1l);
            int v592; float v593;
            Tuple0 tmp46 = Tuple0{0l, 0.0f};
            v592 = tmp46.v0; v593 = tmp46.v1;
            while (while_method_1(v592)){
                assert("Tensor range check" && 0 <= v592 && v592 < 4l);
                int v595;
                v595 = v592 + v591;
                float v596;
                v596 = v578[v595];
                float v597;
                v597 = v593 + v596;
                v593 = v597;
                v592 += 1l ;
            }
            auto v598 = cooperative_groups::coalesced_threads();
            int v599;
            v599 = threadIdx.x;
            int v600;
            v600 = v599 / 4l;
            auto v601 = cooperative_groups::labeled_partition(v598,v600);
            Closure2 v602{};
            float v603;
            v603 = cooperative_groups::inclusive_scan(v601, v593, v602);
            float v604;
            v604 = v601.shfl_up(v603,1);
            bool v605;
            v605 = v601.thread_rank() == 0;
            float v606;
            if (v605){
                v606 = 0.0f;
            } else {
                v606 = v604;
            }
            float v607;
            v607 = v601.shfl(v603,v601.num_threads()-1);
            float v608;
            v608 = v588 + v606;
            int v609; float v610;
            Tuple0 tmp47 = Tuple0{0l, v608};
            v609 = tmp47.v0; v610 = tmp47.v1;
            while (while_method_1(v609)){
                assert("Tensor range check" && 0 <= v609 && v609 < 4l);
                int v612;
                v612 = v609 + v591;
                float v613;
                v613 = v578[v612];
                float v614;
                v614 = v610 + v613;
                assert("Tensor range check" && 0 <= v609 && v609 < 4l);
                v587[v612] = v614;
                v610 = v614;
                v609 += 1l ;
            }
            float v615;
            v615 = v588 + v607;
            v588 = v615;
            v589 += 1l ;
        }
        float v616[4l];
        bool v617[4l];
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
                v624 = v587[v623];
                float v625;
                v625 = v578[v623];
                bool v626;
                v626 = v625 > 0.0f;
                assert("Tensor range check" && 0 <= v618 && v618 < 1l);
                assert("Tensor range check" && 0 <= v620 && v620 < 4l);
                v616[v623] = v624;
                v617[v623] = v626;
                v620 += 1l ;
            }
            v618 += 1l ;
        }
        float v627; bool v628;
        Tuple3 tmp48 = Tuple3{-1.0f / 0.0f, false};
        v627 = tmp48.v0; v628 = tmp48.v1;
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
                v635 = v616[v634];
                bool v636;
                v636 = v617[v634];
                float v643; bool v644;
                if (v628){
                    if (v636){
                        bool v637;
                        v637 = v627 >= v635;
                        float v638;
                        if (v637){
                            v638 = v627;
                        } else {
                            v638 = v635;
                        }
                        v643 = v638; v644 = true;
                    } else {
                        v643 = v627; v644 = v628;
                    }
                } else {
                    if (v636){
                        v643 = v635; v644 = v636;
                    } else {
                        v643 = v627; v644 = v628;
                    }
                }
                v627 = v643;
                v628 = v644;
                v631 += 1l ;
            }
            v629 += 1l ;
        }
        auto v645 = cooperative_groups::coalesced_threads();
        int v646;
        v646 = threadIdx.x;
        int v647;
        v647 = v646 / 4l;
        auto v648 = cooperative_groups::labeled_partition(v645,v647);
        Closure5 v649{};
        float v650; bool v651;
        Tuple3 tmp49 = cooperative_groups::reduce(v648, Tuple3{v627, v628}, v649);
        v650 = tmp49.v0; v651 = tmp49.v1;
        bool v652;
        v652 = v651 == false;
        if (v652){
            assert("The local reduce must be true." && v651);
        } else {
        }
        float v654[4l];
        int v655[4l];
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
                int v662;
                v662 = v458[v661];
                float v663;
                v663 = curand_uniform(&v10);
                assert("Tensor range check" && 0 <= v656 && v656 < 1l);
                assert("Tensor range check" && 0 <= v658 && v658 < 4l);
                v654[v661] = v663;
                v655[v661] = v662;
                v658 += 1l ;
            }
            v656 += 1l ;
        }
        float v664; int v665;
        Tuple1 tmp50 = Tuple1{0.0f, 2147483647l};
        v664 = tmp50.v0; v665 = tmp50.v1;
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
                v672 = v654[v671];
                int v673;
                v673 = v655[v671];
                bool v674;
                v674 = v665 < v673;
                float v675; int v676;
                if (v674){
                    v675 = v664; v676 = v665;
                } else {
                    v675 = v672; v676 = v673;
                }
                v664 = v675;
                v665 = v676;
                v668 += 1l ;
            }
            v666 += 1l ;
        }
        auto v677 = cooperative_groups::coalesced_threads();
        int v678;
        v678 = threadIdx.x;
        int v679;
        v679 = v678 / 4l;
        auto v680 = cooperative_groups::labeled_partition(v677,v679);
        Closure6 v681{};
        float v682; int v683;
        Tuple1 tmp51 = cooperative_groups::reduce(v680, Tuple1{v664, v665}, v681);
        v682 = tmp51.v0; v683 = tmp51.v1;
        float v684;
        v684 = v650 * v682;
        int v685[4l];
        bool v686[4l];
        int v687;
        v687 = 0l;
        while (while_method_3(v687)){
            int v689;
            v689 = 0l;
            while (while_method_1(v689)){
                assert("Tensor range check" && 0 <= v687 && v687 < 1l);
                assert("Tensor range check" && 0 <= v689 && v689 < 4l);
                int v691;
                v691 = 4l * v687;
                int v692;
                v692 = v691 + v689;
                float v693;
                v693 = v616[v692];
                bool v694;
                v694 = v617[v692];
                int v695;
                v695 = v458[v692];
                int v698; bool v699;
                if (v694){
                    float v696;
                    v696 = v693 - v684;
                    bool v697;
                    v697 = v696 >= 0.0f;
                    v698 = v695; v699 = v697;
                } else {
                    v698 = 2147483647l; v699 = false;
                }
                assert("Tensor range check" && 0 <= v687 && v687 < 1l);
                assert("Tensor range check" && 0 <= v689 && v689 < 4l);
                v685[v692] = v698;
                v686[v692] = v699;
                v689 += 1l ;
            }
            v687 += 1l ;
        }
        int v700; bool v701;
        Tuple4 tmp52 = Tuple4{2147483647l, false};
        v700 = tmp52.v0; v701 = tmp52.v1;
        int v702;
        v702 = 0l;
        while (while_method_3(v702)){
            int v704;
            v704 = 0l;
            while (while_method_1(v704)){
                assert("Tensor range check" && 0 <= v702 && v702 < 1l);
                assert("Tensor range check" && 0 <= v704 && v704 < 4l);
                int v706;
                v706 = 4l * v702;
                int v707;
                v707 = v706 + v704;
                int v708;
                v708 = v685[v707];
                bool v709;
                v709 = v686[v707];
                int v716; bool v717;
                if (v701){
                    if (v709){
                        bool v710;
                        v710 = v700 < v708;
                        int v711;
                        if (v710){
                            v711 = v700;
                        } else {
                            v711 = v708;
                        }
                        v716 = v711; v717 = true;
                    } else {
                        v716 = v700; v717 = v701;
                    }
                } else {
                    if (v709){
                        v716 = v708; v717 = v709;
                    } else {
                        v716 = v700; v717 = v701;
                    }
                }
                v700 = v716;
                v701 = v717;
                v704 += 1l ;
            }
            v702 += 1l ;
        }
        auto v718 = cooperative_groups::coalesced_threads();
        int v719;
        v719 = threadIdx.x;
        int v720;
        v720 = v719 / 4l;
        auto v721 = cooperative_groups::labeled_partition(v718,v720);
        Closure7 v722{};
        int v723; bool v724;
        Tuple4 tmp53 = cooperative_groups::reduce(v721, Tuple4{v700, v701}, v722);
        v723 = tmp53.v0; v724 = tmp53.v1;
        bool v725;
        v725 = v724 == false;
        if (v725){
            assert("The local reduce must be true." && v724);
        } else {
        }
        int v727;
        v727 = 0l;
        while (while_method_3(v727)){
            assert("Tensor range check" && 0 <= v727 && v727 < 1l);
            assert("Tensor range check" && 0 <= v727 && v727 < 1l);
            v727 += 1l ;
        }
        assert("Tensor range check" && 0 <= v449 && v449 < 32l);
        v427[v449] = v723;
        v437 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v428 && v428 < 32l);
    int v729;
    v729 = v427[v428];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v730;
    v730 = threadIdx.x;
    assert("Tensor range check" && 0 <= v730 && v730 < 32l);
    v5[v730] = v729;
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
    __shared__ float * v27[32l];
    __shared__ int * v28[32l];
    __shared__ int * v29[32l];
    /* void shared array create v30 */;
    /* void shared array create v31 */;
    int v32;
    v32 = threadIdx.x;
    assert("Tensor range check" && 0 <= v32 && v32 < 32l);
    v27[v32] = v21;
    v28[v32] = v23;
    v29[v32] = v25;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v33;
    v33 = 0l <= v32;
    bool v34;
    v34 = v33 == false;
    if (v34){
        assert("The index needs to be zero or positive." && v33);
    } else {
    }
    int v36;
    v36 = v32 % 32l;
    int v37;
    v37 = v32 / 32l;
    bool v38;
    v38 = v37 < 1l;
    bool v39;
    v39 = v38 == false;
    if (v39){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v38);
    } else {
    }
    assert("Tensor range check" && 0 <= v37 && v37 < 1l);
    int v41;
    v41 = 0l;
    while (while_method_4(v41)){
        bool v43;
        v43 = 0l <= v37;
        bool v44;
        v44 = v43 && v38;
        bool v45;
        v45 = v44 == false;
        if (v45){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v44);
        } else {
        }
        bool v47;
        v47 = 0l <= v41;
        bool v49;
        if (v47){
            bool v48;
            v48 = v41 < 32l;
            v49 = v48;
        } else {
            v49 = false;
        }
        bool v50;
        v50 = v49 == false;
        if (v50){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v49);
        } else {
        }
        int v52;
        v52 = v41 + v37;
        assert("Tensor range check" && 0 <= v41 && v41 < 32l);
        float * v53;
        v53 = v27[v52];
        int * v54;
        v54 = v28[v52];
        int * v55;
        v55 = v29[v52];
        /* void array index */;
        int v56;
        v56 = blockIdx.x;
        int v57;
        v57 = v56 * 32l;
        int v58;
        v58 = v57 + v52;
        assert("Tensor range check" && 0 <= v36 && v36 < 32l);
        int v59;
        v59 = 4l * v36;
        float v60[8l];
        int v61[8l];
        int v62;
        v62 = 0l;
        while (while_method_5(v62)){
            assert("Tensor range check" && 0 <= v62 && v62 < 2l);
            int v64;
            v64 = 4l * v62;
            assert("Tensor range check" && 0 <= v62 && v62 < 2l);
            int v65;
            v65 = 128l * v62;
            int v66;
            v66 = v65 + v59;
            int4* v67;
            v67 = reinterpret_cast<int4*>(v53 + v66);
            int4* v68;
            v68 = reinterpret_cast<int4*>(v60 + v64);
            assert("Pointer alignment check" && (unsigned long long)(v67) % 4l == 0 && (unsigned long long)(v68) % 4l == 0);
            *v68 = *v67;
            v62 += 1l ;
        }
        int v69;
        v69 = 0l;
        while (while_method_5(v69)){
            int v71;
            v71 = 0l;
            while (while_method_1(v71)){
                bool v73;
                v73 = 0l <= v71;
                bool v75;
                if (v73){
                    bool v74;
                    v74 = v71 < 4l;
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
                bool v78;
                v78 = 0l <= v36;
                bool v80;
                if (v78){
                    bool v79;
                    v79 = v36 < 32l;
                    v80 = v79;
                } else {
                    v80 = false;
                }
                bool v81;
                v81 = v80 == false;
                if (v81){
                    assert("The indices should be inside the range of the dimension." && v80);
                } else {
                }
                int v83;
                v83 = v36 * 4l;
                int v84;
                v84 = v71 + v83;
                bool v85;
                v85 = 0l <= v69;
                bool v87;
                if (v85){
                    bool v86;
                    v86 = v69 < 2l;
                    v87 = v86;
                } else {
                    v87 = false;
                }
                bool v88;
                v88 = v87 == false;
                if (v88){
                    assert("The indices should be inside the range of the dimension." && v87);
                } else {
                }
                int v90;
                v90 = v69 * 128l;
                int v91;
                v91 = v84 + v90;
                assert("Tensor range check" && 0 <= v69 && v69 < 2l);
                assert("Tensor range check" && 0 <= v71 && v71 < 4l);
                int v92;
                v92 = 4l * v69;
                int v93;
                v93 = v92 + v71;
                v61[v93] = v91;
                v71 += 1l ;
            }
            v69 += 1l ;
        }
        int v94[8l];
        int v95[8l];
        int v96;
        v96 = 0l;
        while (while_method_5(v96)){
            int v98;
            v98 = 0l;
            while (while_method_1(v98)){
                assert("Tensor range check" && 0 <= v96 && v96 < 2l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                int v100;
                v100 = 4l * v96;
                int v101;
                v101 = v100 + v98;
                int v102;
                v102 = v61[v101];
                assert("Tensor range check" && 0 <= v96 && v96 < 2l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                v94[v101] = v58;
                v95[v101] = v102;
                v98 += 1l ;
            }
            v96 += 1l ;
        }
        int v103;
        v103 = 0l;
        while (while_method_5(v103)){
            assert("Tensor range check" && 0 <= v103 && v103 < 2l);
            int v105;
            v105 = 128l * v103;
            int v106;
            v106 = v105 + v59;
            assert("Tensor range check" && 0 <= v103 && v103 < 2l);
            int v107;
            v107 = 4l * v103;
            int4* v108;
            v108 = reinterpret_cast<int4*>(v94 + v107);
            int4* v109;
            v109 = reinterpret_cast<int4*>(v54 + v106);
            assert("Pointer alignment check" && (unsigned long long)(v108) % 4l == 0 && (unsigned long long)(v109) % 4l == 0);
            *v109 = *v108;
            int4* v110;
            v110 = reinterpret_cast<int4*>(v95 + v107);
            int4* v111;
            v111 = reinterpret_cast<int4*>(v55 + v106);
            assert("Pointer alignment check" && (unsigned long long)(v110) % 4l == 0 && (unsigned long long)(v111) % 4l == 0);
            *v111 = *v110;
            v103 += 1l ;
        }
        assert("Tensor range check" && 0 <= v52 && v52 < 32l);
        /* void array set */;
        v41 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v32 && v32 < 32l);
    /* void array index */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v112;
    v112 = v1+v12;
    __shared__ float * v114[32l];
    /* void shared array create v115 */;
    __shared__ int v116[32l];
    int v117;
    v117 = threadIdx.x;
    assert("Tensor range check" && 0 <= v117 && v117 < 32l);
    v114[v117] = v112;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v118;
    v118 = 0l <= v117;
    bool v119;
    v119 = v118 == false;
    if (v119){
        assert("The index needs to be zero or positive." && v118);
    } else {
    }
    int v121;
    v121 = v117 % 32l;
    int v122;
    v122 = v117 / 32l;
    bool v123;
    v123 = v122 < 1l;
    bool v124;
    v124 = v123 == false;
    if (v124){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v123);
    } else {
    }
    assert("Tensor range check" && 0 <= v122 && v122 < 1l);
    int v126;
    v126 = 0l;
    while (while_method_4(v126)){
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
            v133 = v126 < 32l;
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
        v137 = v126 + v122;
        assert("Tensor range check" && 0 <= v126 && v126 < 32l);
        float * v138;
        v138 = v114[v137];
        /* void array index */;
        int v139;
        v139 = blockIdx.x;
        int v140;
        v140 = v139 * 32l;
        int v141;
        v141 = v140 + v137;
        assert("Tensor range check" && 0 <= v121 && v121 < 32l);
        int v142;
        v142 = 4l * v121;
        float v143[8l];
        int v144[8l];
        int v145;
        v145 = 0l;
        while (while_method_5(v145)){
            assert("Tensor range check" && 0 <= v145 && v145 < 2l);
            int v147;
            v147 = 4l * v145;
            assert("Tensor range check" && 0 <= v145 && v145 < 2l);
            int v148;
            v148 = 128l * v145;
            int v149;
            v149 = v148 + v142;
            int4* v150;
            v150 = reinterpret_cast<int4*>(v138 + v149);
            int4* v151;
            v151 = reinterpret_cast<int4*>(v143 + v147);
            assert("Pointer alignment check" && (unsigned long long)(v150) % 4l == 0 && (unsigned long long)(v151) % 4l == 0);
            *v151 = *v150;
            v145 += 1l ;
        }
        int v152;
        v152 = 0l;
        while (while_method_5(v152)){
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
                    v162 = v121 < 32l;
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
                    v169 = v152 < 2l;
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
                v173 = v152 * 128l;
                int v174;
                v174 = v167 + v173;
                assert("Tensor range check" && 0 <= v152 && v152 < 2l);
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
        while (while_method_5(v177)){
            assert("Tensor range check" && 0 <= v177 && v177 < 2l);
            assert("Tensor range check" && 0 <= v177 && v177 < 2l);
            v177 += 1l ;
        }
        assert("Tensor range check" && 0 <= v137 && v137 < 32l);
        v116[v137] = v141;
        v126 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v117 && v117 < 32l);
    int v179;
    v179 = v116[v117];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v180;
    v180 = threadIdx.x;
    assert("Tensor range check" && 0 <= v180 && v180 < 32l);
    v4[v180] = v179;
    float * v181;
    v181 = v1+v12;
    float * v183;
    v183 = v6+v20;
    __shared__ float * v185[32l];
    __shared__ float * v186[32l];
    /* void shared array create v187 */;
    /* void shared array create v188 */;
    int v189;
    v189 = threadIdx.x;
    assert("Tensor range check" && 0 <= v189 && v189 < 32l);
    v185[v189] = v181;
    v186[v189] = v183;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v190;
    v190 = 0l <= v189;
    bool v191;
    v191 = v190 == false;
    if (v191){
        assert("The index needs to be zero or positive." && v190);
    } else {
    }
    int v193;
    v193 = v189 % 32l;
    int v194;
    v194 = v189 / 32l;
    bool v195;
    v195 = v194 < 1l;
    bool v196;
    v196 = v195 == false;
    if (v196){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v195);
    } else {
    }
    assert("Tensor range check" && 0 <= v194 && v194 < 1l);
    int v198;
    v198 = 0l;
    while (while_method_4(v198)){
        bool v200;
        v200 = 0l <= v194;
        bool v201;
        v201 = v200 && v195;
        bool v202;
        v202 = v201 == false;
        if (v202){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v201);
        } else {
        }
        bool v204;
        v204 = 0l <= v198;
        bool v206;
        if (v204){
            bool v205;
            v205 = v198 < 32l;
            v206 = v205;
        } else {
            v206 = false;
        }
        bool v207;
        v207 = v206 == false;
        if (v207){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v206);
        } else {
        }
        int v209;
        v209 = v198 + v194;
        assert("Tensor range check" && 0 <= v198 && v198 < 32l);
        float * v210;
        v210 = v185[v209];
        float * v211;
        v211 = v186[v209];
        /* void array index */;
        int v212;
        v212 = blockIdx.x;
        int v213;
        v213 = v212 * 32l;
        int v214;
        v214 = v213 + v209;
        assert("Tensor range check" && 0 <= v193 && v193 < 32l);
        int v215;
        v215 = 4l * v193;
        float v216[8l];
        int v217[8l];
        int v218;
        v218 = 0l;
        while (while_method_5(v218)){
            assert("Tensor range check" && 0 <= v218 && v218 < 2l);
            int v220;
            v220 = 4l * v218;
            assert("Tensor range check" && 0 <= v218 && v218 < 2l);
            int v221;
            v221 = 128l * v218;
            int v222;
            v222 = v221 + v215;
            int4* v223;
            v223 = reinterpret_cast<int4*>(v210 + v222);
            int4* v224;
            v224 = reinterpret_cast<int4*>(v216 + v220);
            assert("Pointer alignment check" && (unsigned long long)(v223) % 4l == 0 && (unsigned long long)(v224) % 4l == 0);
            *v224 = *v223;
            v218 += 1l ;
        }
        int v225;
        v225 = 0l;
        while (while_method_5(v225)){
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
                v234 = 0l <= v193;
                bool v236;
                if (v234){
                    bool v235;
                    v235 = v193 < 32l;
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
                v239 = v193 * 4l;
                int v240;
                v240 = v227 + v239;
                bool v241;
                v241 = 0l <= v225;
                bool v243;
                if (v241){
                    bool v242;
                    v242 = v225 < 2l;
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
                assert("Tensor range check" && 0 <= v225 && v225 < 2l);
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
        int v250;
        v250 = 0l;
        while (while_method_5(v250)){
            assert("Tensor range check" && 0 <= v250 && v250 < 2l);
            int v252;
            v252 = 128l * v250;
            int v253;
            v253 = v252 + v215;
            assert("Tensor range check" && 0 <= v250 && v250 < 2l);
            int v254;
            v254 = 4l * v250;
            int4* v255;
            v255 = reinterpret_cast<int4*>(v216 + v254);
            int4* v256;
            v256 = reinterpret_cast<int4*>(v211 + v253);
            assert("Pointer alignment check" && (unsigned long long)(v255) % 4l == 0 && (unsigned long long)(v256) % 4l == 0);
            *v256 = *v255;
            v250 += 1l ;
        }
        assert("Tensor range check" && 0 <= v209 && v209 < 32l);
        /* void array set */;
        v198 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v189 && v189 < 32l);
    /* void array index */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v257;
    v257 = v1+v12;
    float * v259;
    v259 = v7+v16;
    __shared__ float * v261[32l];
    __shared__ float * v262[32l];
    /* void shared array create v263 */;
    /* void shared array create v264 */;
    int v265;
    v265 = threadIdx.x;
    assert("Tensor range check" && 0 <= v265 && v265 < 32l);
    v261[v265] = v257;
    v262[v265] = v259;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v266;
    v266 = 0l <= v265;
    bool v267;
    v267 = v266 == false;
    if (v267){
        assert("The index needs to be zero or positive." && v266);
    } else {
    }
    int v269;
    v269 = v265 % 32l;
    int v270;
    v270 = v265 / 32l;
    bool v271;
    v271 = v270 < 1l;
    bool v272;
    v272 = v271 == false;
    if (v272){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v271);
    } else {
    }
    assert("Tensor range check" && 0 <= v270 && v270 < 1l);
    int v274;
    v274 = 0l;
    while (while_method_4(v274)){
        bool v276;
        v276 = 0l <= v270;
        bool v277;
        v277 = v276 && v271;
        bool v278;
        v278 = v277 == false;
        if (v278){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v277);
        } else {
        }
        bool v280;
        v280 = 0l <= v274;
        bool v282;
        if (v280){
            bool v281;
            v281 = v274 < 32l;
            v282 = v281;
        } else {
            v282 = false;
        }
        bool v283;
        v283 = v282 == false;
        if (v283){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v282);
        } else {
        }
        int v285;
        v285 = v274 + v270;
        assert("Tensor range check" && 0 <= v274 && v274 < 32l);
        float * v286;
        v286 = v261[v285];
        float * v287;
        v287 = v262[v285];
        /* void array index */;
        int v288;
        v288 = blockIdx.x;
        int v289;
        v289 = v288 * 32l;
        int v290;
        v290 = v289 + v285;
        assert("Tensor range check" && 0 <= v269 && v269 < 32l);
        int v291;
        v291 = 4l * v269;
        float v292[8l];
        int v293[8l];
        int v294;
        v294 = 0l;
        while (while_method_5(v294)){
            assert("Tensor range check" && 0 <= v294 && v294 < 2l);
            int v296;
            v296 = 4l * v294;
            assert("Tensor range check" && 0 <= v294 && v294 < 2l);
            int v297;
            v297 = 128l * v294;
            int v298;
            v298 = v297 + v291;
            int4* v299;
            v299 = reinterpret_cast<int4*>(v286 + v298);
            int4* v300;
            v300 = reinterpret_cast<int4*>(v292 + v296);
            assert("Pointer alignment check" && (unsigned long long)(v299) % 4l == 0 && (unsigned long long)(v300) % 4l == 0);
            *v300 = *v299;
            v294 += 1l ;
        }
        int v301;
        v301 = 0l;
        while (while_method_5(v301)){
            int v303;
            v303 = 0l;
            while (while_method_1(v303)){
                bool v305;
                v305 = 0l <= v303;
                bool v307;
                if (v305){
                    bool v306;
                    v306 = v303 < 4l;
                    v307 = v306;
                } else {
                    v307 = false;
                }
                bool v308;
                v308 = v307 == false;
                if (v308){
                    assert("The indices should be inside the range of the dimension." && v307);
                } else {
                }
                bool v310;
                v310 = 0l <= v269;
                bool v312;
                if (v310){
                    bool v311;
                    v311 = v269 < 32l;
                    v312 = v311;
                } else {
                    v312 = false;
                }
                bool v313;
                v313 = v312 == false;
                if (v313){
                    assert("The indices should be inside the range of the dimension." && v312);
                } else {
                }
                int v315;
                v315 = v269 * 4l;
                int v316;
                v316 = v303 + v315;
                bool v317;
                v317 = 0l <= v301;
                bool v319;
                if (v317){
                    bool v318;
                    v318 = v301 < 2l;
                    v319 = v318;
                } else {
                    v319 = false;
                }
                bool v320;
                v320 = v319 == false;
                if (v320){
                    assert("The indices should be inside the range of the dimension." && v319);
                } else {
                }
                int v322;
                v322 = v301 * 128l;
                int v323;
                v323 = v316 + v322;
                assert("Tensor range check" && 0 <= v301 && v301 < 2l);
                assert("Tensor range check" && 0 <= v303 && v303 < 4l);
                int v324;
                v324 = 4l * v301;
                int v325;
                v325 = v324 + v303;
                v293[v325] = v323;
                v303 += 1l ;
            }
            v301 += 1l ;
        }
        bool v326[8l];
        int v327;
        v327 = 0l;
        while (while_method_5(v327)){
            int v329;
            v329 = 0l;
            while (while_method_1(v329)){
                assert("Tensor range check" && 0 <= v327 && v327 < 2l);
                assert("Tensor range check" && 0 <= v329 && v329 < 4l);
                int v331;
                v331 = 4l * v327;
                int v332;
                v332 = v331 + v329;
                float v333;
                v333 = v292[v332];
                int v334;
                v334 = v293[v332];
                bool v335;
                v335 = v334 < 3l;
                assert("Tensor range check" && 0 <= v327 && v327 < 2l);
                assert("Tensor range check" && 0 <= v329 && v329 < 4l);
                v326[v332] = v335;
                v329 += 1l ;
            }
            v327 += 1l ;
        }
        float v336[8l];
        int v337;
        v337 = 0l;
        while (while_method_5(v337)){
            int v339;
            v339 = 0l;
            while (while_method_1(v339)){
                assert("Tensor range check" && 0 <= v337 && v337 < 2l);
                assert("Tensor range check" && 0 <= v339 && v339 < 4l);
                int v341;
                v341 = 4l * v337;
                int v342;
                v342 = v341 + v339;
                float v343;
                v343 = v292[v342];
                bool v344;
                v344 = v326[v342];
                float v347;
                if (v344){
                    bool v345;
                    v345 = 0.0f >= v343;
                    if (v345){
                        v347 = 0.0f;
                    } else {
                        v347 = v343;
                    }
                } else {
                    v347 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v337 && v337 < 2l);
                assert("Tensor range check" && 0 <= v339 && v339 < 4l);
                v336[v342] = v347;
                v339 += 1l ;
            }
            v337 += 1l ;
        }
        float v348;
        v348 = 0.0f;
        int v349;
        v349 = 0l;
        while (while_method_5(v349)){
            int v351;
            v351 = 0l;
            while (while_method_1(v351)){
                assert("Tensor range check" && 0 <= v349 && v349 < 2l);
                assert("Tensor range check" && 0 <= v351 && v351 < 4l);
                int v353;
                v353 = 4l * v349;
                int v354;
                v354 = v353 + v351;
                float v355;
                v355 = v336[v354];
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
        v359 = v358 / 32l;
        auto v360 = cooperative_groups::labeled_partition(v357,v359);
        Closure0 v361{};
        float v362;
        v362 = cooperative_groups::reduce(v360, v348, v361);
        int v363[8l];
        int v364;
        v364 = 0l;
        while (while_method_5(v364)){
            int v366;
            v366 = 0l;
            while (while_method_1(v366)){
                assert("Tensor range check" && 0 <= v364 && v364 < 2l);
                assert("Tensor range check" && 0 <= v366 && v366 < 4l);
                int v368;
                v368 = 4l * v364;
                int v369;
                v369 = v368 + v366;
                bool v370;
                v370 = v326[v369];
                int v371;
                if (v370){
                    v371 = 1l;
                } else {
                    v371 = 0l;
                }
                assert("Tensor range check" && 0 <= v364 && v364 < 2l);
                assert("Tensor range check" && 0 <= v366 && v366 < 4l);
                v363[v369] = v371;
                v366 += 1l ;
            }
            v364 += 1l ;
        }
        int v372;
        v372 = 0l;
        int v373;
        v373 = 0l;
        while (while_method_5(v373)){
            int v375;
            v375 = 0l;
            while (while_method_1(v375)){
                assert("Tensor range check" && 0 <= v373 && v373 < 2l);
                assert("Tensor range check" && 0 <= v375 && v375 < 4l);
                int v377;
                v377 = 4l * v373;
                int v378;
                v378 = v377 + v375;
                int v379;
                v379 = v363[v378];
                int v380;
                v380 = v372 + v379;
                v372 = v380;
                v375 += 1l ;
            }
            v373 += 1l ;
        }
        auto v381 = cooperative_groups::coalesced_threads();
        int v382;
        v382 = threadIdx.x;
        int v383;
        v383 = v382 / 32l;
        auto v384 = cooperative_groups::labeled_partition(v381,v383);
        Closure4 v385{};
        int v386;
        v386 = cooperative_groups::reduce(v384, v372, v385);
        float v387;
        v387 = (float)v386;
        float v388;
        v388 = 1.0f / v387;
        float v389[8l];
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
                v396 = v336[v395];
                bool v397;
                v397 = v326[v395];
                bool v398;
                v398 = v397 == false;
                float v403;
                if (v398){
                    v403 = 0.0f;
                } else {
                    bool v399;
                    v399 = v362 == 0.0f;
                    bool v400;
                    v400 = v399 != true;
                    if (v400){
                        float v401;
                        v401 = v396 / v362;
                        v403 = v401;
                    } else {
                        v403 = v388;
                    }
                }
                assert("Tensor range check" && 0 <= v390 && v390 < 2l);
                assert("Tensor range check" && 0 <= v392 && v392 < 4l);
                v389[v395] = v403;
                v392 += 1l ;
            }
            v390 += 1l ;
        }
        int v404;
        v404 = 0l;
        while (while_method_5(v404)){
            assert("Tensor range check" && 0 <= v404 && v404 < 2l);
            int v406;
            v406 = 128l * v404;
            int v407;
            v407 = v406 + v291;
            assert("Tensor range check" && 0 <= v404 && v404 < 2l);
            int v408;
            v408 = 4l * v404;
            int4* v409;
            v409 = reinterpret_cast<int4*>(v389 + v408);
            int4* v410;
            v410 = reinterpret_cast<int4*>(v287 + v407);
            assert("Pointer alignment check" && (unsigned long long)(v409) % 4l == 0 && (unsigned long long)(v410) % 4l == 0);
            *v410 = *v409;
            v404 += 1l ;
        }
        assert("Tensor range check" && 0 <= v285 && v285 < 32l);
        /* void array set */;
        v274 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v265 && v265 < 32l);
    /* void array index */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v411;
    v411 = v1+v12;
    __shared__ float * v413[32l];
    /* void shared array create v414 */;
    __shared__ int v415[32l];
    int v416;
    v416 = threadIdx.x;
    assert("Tensor range check" && 0 <= v416 && v416 < 32l);
    v413[v416] = v411;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v417;
    v417 = 0l <= v416;
    bool v418;
    v418 = v417 == false;
    if (v418){
        assert("The index needs to be zero or positive." && v417);
    } else {
    }
    int v420;
    v420 = v416 % 32l;
    int v421;
    v421 = v416 / 32l;
    bool v422;
    v422 = v421 < 1l;
    bool v423;
    v423 = v422 == false;
    if (v423){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v422);
    } else {
    }
    assert("Tensor range check" && 0 <= v421 && v421 < 1l);
    int v425;
    v425 = 0l;
    while (while_method_4(v425)){
        bool v427;
        v427 = 0l <= v421;
        bool v428;
        v428 = v427 && v422;
        bool v429;
        v429 = v428 == false;
        if (v429){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v428);
        } else {
        }
        bool v431;
        v431 = 0l <= v425;
        bool v433;
        if (v431){
            bool v432;
            v432 = v425 < 32l;
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
        v436 = v425 + v421;
        assert("Tensor range check" && 0 <= v425 && v425 < 32l);
        float * v437;
        v437 = v413[v436];
        /* void array index */;
        int v438;
        v438 = blockIdx.x;
        int v439;
        v439 = v438 * 32l;
        int v440;
        v440 = v439 + v436;
        assert("Tensor range check" && 0 <= v420 && v420 < 32l);
        int v441;
        v441 = 4l * v420;
        float v442[8l];
        int v443[8l];
        int v444;
        v444 = 0l;
        while (while_method_5(v444)){
            assert("Tensor range check" && 0 <= v444 && v444 < 2l);
            int v446;
            v446 = 4l * v444;
            assert("Tensor range check" && 0 <= v444 && v444 < 2l);
            int v447;
            v447 = 128l * v444;
            int v448;
            v448 = v447 + v441;
            int4* v449;
            v449 = reinterpret_cast<int4*>(v437 + v448);
            int4* v450;
            v450 = reinterpret_cast<int4*>(v442 + v446);
            assert("Pointer alignment check" && (unsigned long long)(v449) % 4l == 0 && (unsigned long long)(v450) % 4l == 0);
            *v450 = *v449;
            v444 += 1l ;
        }
        int v451;
        v451 = 0l;
        while (while_method_5(v451)){
            int v453;
            v453 = 0l;
            while (while_method_1(v453)){
                bool v455;
                v455 = 0l <= v453;
                bool v457;
                if (v455){
                    bool v456;
                    v456 = v453 < 4l;
                    v457 = v456;
                } else {
                    v457 = false;
                }
                bool v458;
                v458 = v457 == false;
                if (v458){
                    assert("The indices should be inside the range of the dimension." && v457);
                } else {
                }
                bool v460;
                v460 = 0l <= v420;
                bool v462;
                if (v460){
                    bool v461;
                    v461 = v420 < 32l;
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
                v465 = v420 * 4l;
                int v466;
                v466 = v453 + v465;
                bool v467;
                v467 = 0l <= v451;
                bool v469;
                if (v467){
                    bool v468;
                    v468 = v451 < 2l;
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
                v472 = v451 * 128l;
                int v473;
                v473 = v466 + v472;
                assert("Tensor range check" && 0 <= v451 && v451 < 2l);
                assert("Tensor range check" && 0 <= v453 && v453 < 4l);
                int v474;
                v474 = 4l * v451;
                int v475;
                v475 = v474 + v453;
                v443[v475] = v473;
                v453 += 1l ;
            }
            v451 += 1l ;
        }
        bool v476[8l];
        int v477;
        v477 = 0l;
        while (while_method_5(v477)){
            int v479;
            v479 = 0l;
            while (while_method_1(v479)){
                assert("Tensor range check" && 0 <= v477 && v477 < 2l);
                assert("Tensor range check" && 0 <= v479 && v479 < 4l);
                int v481;
                v481 = 4l * v477;
                int v482;
                v482 = v481 + v479;
                float v483;
                v483 = v442[v482];
                int v484;
                v484 = v443[v482];
                bool v485;
                v485 = v484 < 3l;
                assert("Tensor range check" && 0 <= v477 && v477 < 2l);
                assert("Tensor range check" && 0 <= v479 && v479 < 4l);
                v476[v482] = v485;
                v479 += 1l ;
            }
            v477 += 1l ;
        }
        int v486[8l];
        int v487;
        v487 = 0l;
        while (while_method_5(v487)){
            int v489;
            v489 = 0l;
            while (while_method_1(v489)){
                assert("Tensor range check" && 0 <= v487 && v487 < 2l);
                assert("Tensor range check" && 0 <= v489 && v489 < 4l);
                int v491;
                v491 = 4l * v487;
                int v492;
                v492 = v491 + v489;
                bool v493;
                v493 = v476[v492];
                int v494;
                if (v493){
                    v494 = 1l;
                } else {
                    v494 = 0l;
                }
                assert("Tensor range check" && 0 <= v487 && v487 < 2l);
                assert("Tensor range check" && 0 <= v489 && v489 < 4l);
                v486[v492] = v494;
                v489 += 1l ;
            }
            v487 += 1l ;
        }
        int v495;
        v495 = 0l;
        int v496;
        v496 = 0l;
        while (while_method_5(v496)){
            int v498;
            v498 = 0l;
            while (while_method_1(v498)){
                assert("Tensor range check" && 0 <= v496 && v496 < 2l);
                assert("Tensor range check" && 0 <= v498 && v498 < 4l);
                int v500;
                v500 = 4l * v496;
                int v501;
                v501 = v500 + v498;
                int v502;
                v502 = v486[v501];
                int v503;
                v503 = v495 + v502;
                v495 = v503;
                v498 += 1l ;
            }
            v496 += 1l ;
        }
        auto v504 = cooperative_groups::coalesced_threads();
        int v505;
        v505 = threadIdx.x;
        int v506;
        v506 = v505 / 32l;
        auto v507 = cooperative_groups::labeled_partition(v504,v506);
        Closure4 v508{};
        int v509;
        v509 = cooperative_groups::reduce(v507, v495, v508);
        float v510[8l];
        int v511;
        v511 = 0l;
        while (while_method_5(v511)){
            int v513;
            v513 = 0l;
            while (while_method_1(v513)){
                assert("Tensor range check" && 0 <= v511 && v511 < 2l);
                assert("Tensor range check" && 0 <= v513 && v513 < 4l);
                int v515;
                v515 = 4l * v511;
                int v516;
                v516 = v515 + v513;
                float v517;
                v517 = v442[v516];
                bool v518;
                v518 = v476[v516];
                float v519;
                if (v518){
                    v519 = v517;
                } else {
                    v519 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v511 && v511 < 2l);
                assert("Tensor range check" && 0 <= v513 && v513 < 4l);
                v510[v516] = v519;
                v513 += 1l ;
            }
            v511 += 1l ;
        }
        float v520;
        v520 = 0.0f;
        int v521;
        v521 = 0l;
        while (while_method_5(v521)){
            int v523;
            v523 = 0l;
            while (while_method_1(v523)){
                assert("Tensor range check" && 0 <= v521 && v521 < 2l);
                assert("Tensor range check" && 0 <= v523 && v523 < 4l);
                int v525;
                v525 = 4l * v521;
                int v526;
                v526 = v525 + v523;
                float v527;
                v527 = v510[v526];
                float v528;
                v528 = v520 + v527;
                v520 = v528;
                v523 += 1l ;
            }
            v521 += 1l ;
        }
        auto v529 = cooperative_groups::coalesced_threads();
        int v530;
        v530 = threadIdx.x;
        int v531;
        v531 = v530 / 32l;
        auto v532 = cooperative_groups::labeled_partition(v529,v531);
        Closure0 v533{};
        float v534;
        v534 = cooperative_groups::reduce(v532, v520, v533);
        float v535;
        v535 = (float)v509;
        float v536;
        v536 = v534 / v535;
        float v537[8l];
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
                float v544;
                v544 = v442[v543];
                bool v545;
                v545 = v476[v543];
                float v546;
                if (v545){
                    v546 = v544;
                } else {
                    v546 = -1.0f / 0.0f;
                }
                float v547;
                v547 = v546 - v536;
                float v548;
                v548 = exp(v547);
                assert("Tensor range check" && 0 <= v538 && v538 < 2l);
                assert("Tensor range check" && 0 <= v540 && v540 < 4l);
                v537[v543] = v548;
                v540 += 1l ;
            }
            v538 += 1l ;
        }
        float v549;
        v549 = 0.0f;
        int v550;
        v550 = 0l;
        while (while_method_5(v550)){
            int v552;
            v552 = 0l;
            while (while_method_1(v552)){
                assert("Tensor range check" && 0 <= v550 && v550 < 2l);
                assert("Tensor range check" && 0 <= v552 && v552 < 4l);
                int v554;
                v554 = 4l * v550;
                int v555;
                v555 = v554 + v552;
                float v556;
                v556 = v537[v555];
                float v557;
                v557 = v549 + v556;
                v549 = v557;
                v552 += 1l ;
            }
            v550 += 1l ;
        }
        auto v558 = cooperative_groups::coalesced_threads();
        int v559;
        v559 = threadIdx.x;
        int v560;
        v560 = v559 / 32l;
        auto v561 = cooperative_groups::labeled_partition(v558,v560);
        float v562;
        v562 = cooperative_groups::reduce(v561, v549, v533);
        float v563[8l];
        int v564;
        v564 = 0l;
        while (while_method_5(v564)){
            int v566;
            v566 = 0l;
            while (while_method_1(v566)){
                assert("Tensor range check" && 0 <= v564 && v564 < 2l);
                assert("Tensor range check" && 0 <= v566 && v566 < 4l);
                int v568;
                v568 = 4l * v564;
                int v569;
                v569 = v568 + v566;
                float v570;
                v570 = v537[v569];
                float v571;
                v571 = v570 / v562;
                assert("Tensor range check" && 0 <= v564 && v564 < 2l);
                assert("Tensor range check" && 0 <= v566 && v566 < 4l);
                v563[v569] = v571;
                v566 += 1l ;
            }
            v564 += 1l ;
        }
        float v572[8l];
        float v573;
        v573 = 0.0f;
        int v574;
        v574 = 0l;
        while (while_method_5(v574)){
            assert("Tensor range check" && 0 <= v574 && v574 < 2l);
            int v576;
            v576 = 4l * v574;
            assert("Tensor range check" && 0 <= v574 && v574 < 2l);
            int v577; float v578;
            Tuple0 tmp54 = Tuple0{0l, 0.0f};
            v577 = tmp54.v0; v578 = tmp54.v1;
            while (while_method_1(v577)){
                assert("Tensor range check" && 0 <= v577 && v577 < 4l);
                int v580;
                v580 = v577 + v576;
                float v581;
                v581 = v563[v580];
                float v582;
                v582 = v578 + v581;
                v578 = v582;
                v577 += 1l ;
            }
            auto v583 = cooperative_groups::coalesced_threads();
            int v584;
            v584 = threadIdx.x;
            int v585;
            v585 = v584 / 32l;
            auto v586 = cooperative_groups::labeled_partition(v583,v585);
            Closure2 v587{};
            float v588;
            v588 = cooperative_groups::inclusive_scan(v586, v578, v587);
            float v589;
            v589 = v586.shfl_up(v588,1);
            bool v590;
            v590 = v586.thread_rank() == 0;
            float v591;
            if (v590){
                v591 = 0.0f;
            } else {
                v591 = v589;
            }
            float v592;
            v592 = v586.shfl(v588,v586.num_threads()-1);
            float v593;
            v593 = v573 + v591;
            int v594; float v595;
            Tuple0 tmp55 = Tuple0{0l, v593};
            v594 = tmp55.v0; v595 = tmp55.v1;
            while (while_method_1(v594)){
                assert("Tensor range check" && 0 <= v594 && v594 < 4l);
                int v597;
                v597 = v594 + v576;
                float v598;
                v598 = v563[v597];
                float v599;
                v599 = v595 + v598;
                assert("Tensor range check" && 0 <= v594 && v594 < 4l);
                v572[v597] = v599;
                v595 = v599;
                v594 += 1l ;
            }
            float v600;
            v600 = v573 + v592;
            v573 = v600;
            v574 += 1l ;
        }
        float v601[8l];
        bool v602[8l];
        int v603;
        v603 = 0l;
        while (while_method_5(v603)){
            int v605;
            v605 = 0l;
            while (while_method_1(v605)){
                assert("Tensor range check" && 0 <= v603 && v603 < 2l);
                assert("Tensor range check" && 0 <= v605 && v605 < 4l);
                int v607;
                v607 = 4l * v603;
                int v608;
                v608 = v607 + v605;
                float v609;
                v609 = v572[v608];
                float v610;
                v610 = v563[v608];
                bool v611;
                v611 = v610 > 0.0f;
                assert("Tensor range check" && 0 <= v603 && v603 < 2l);
                assert("Tensor range check" && 0 <= v605 && v605 < 4l);
                v601[v608] = v609;
                v602[v608] = v611;
                v605 += 1l ;
            }
            v603 += 1l ;
        }
        float v612; bool v613;
        Tuple3 tmp56 = Tuple3{-1.0f / 0.0f, false};
        v612 = tmp56.v0; v613 = tmp56.v1;
        int v614;
        v614 = 0l;
        while (while_method_5(v614)){
            int v616;
            v616 = 0l;
            while (while_method_1(v616)){
                assert("Tensor range check" && 0 <= v614 && v614 < 2l);
                assert("Tensor range check" && 0 <= v616 && v616 < 4l);
                int v618;
                v618 = 4l * v614;
                int v619;
                v619 = v618 + v616;
                float v620;
                v620 = v601[v619];
                bool v621;
                v621 = v602[v619];
                float v628; bool v629;
                if (v613){
                    if (v621){
                        bool v622;
                        v622 = v612 >= v620;
                        float v623;
                        if (v622){
                            v623 = v612;
                        } else {
                            v623 = v620;
                        }
                        v628 = v623; v629 = true;
                    } else {
                        v628 = v612; v629 = v613;
                    }
                } else {
                    if (v621){
                        v628 = v620; v629 = v621;
                    } else {
                        v628 = v612; v629 = v613;
                    }
                }
                v612 = v628;
                v613 = v629;
                v616 += 1l ;
            }
            v614 += 1l ;
        }
        auto v630 = cooperative_groups::coalesced_threads();
        int v631;
        v631 = threadIdx.x;
        int v632;
        v632 = v631 / 32l;
        auto v633 = cooperative_groups::labeled_partition(v630,v632);
        Closure5 v634{};
        float v635; bool v636;
        Tuple3 tmp57 = cooperative_groups::reduce(v633, Tuple3{v612, v613}, v634);
        v635 = tmp57.v0; v636 = tmp57.v1;
        bool v637;
        v637 = v636 == false;
        if (v637){
            assert("The local reduce must be true." && v636);
        } else {
        }
        float v639[8l];
        int v640[8l];
        int v641;
        v641 = 0l;
        while (while_method_5(v641)){
            int v643;
            v643 = 0l;
            while (while_method_1(v643)){
                assert("Tensor range check" && 0 <= v641 && v641 < 2l);
                assert("Tensor range check" && 0 <= v643 && v643 < 4l);
                int v645;
                v645 = 4l * v641;
                int v646;
                v646 = v645 + v643;
                int v647;
                v647 = v443[v646];
                float v648;
                v648 = curand_uniform(&v10);
                assert("Tensor range check" && 0 <= v641 && v641 < 2l);
                assert("Tensor range check" && 0 <= v643 && v643 < 4l);
                v639[v646] = v648;
                v640[v646] = v647;
                v643 += 1l ;
            }
            v641 += 1l ;
        }
        float v649; int v650;
        Tuple1 tmp58 = Tuple1{0.0f, 2147483647l};
        v649 = tmp58.v0; v650 = tmp58.v1;
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
                v657 = v639[v656];
                int v658;
                v658 = v640[v656];
                bool v659;
                v659 = v650 < v658;
                float v660; int v661;
                if (v659){
                    v660 = v649; v661 = v650;
                } else {
                    v660 = v657; v661 = v658;
                }
                v649 = v660;
                v650 = v661;
                v653 += 1l ;
            }
            v651 += 1l ;
        }
        auto v662 = cooperative_groups::coalesced_threads();
        int v663;
        v663 = threadIdx.x;
        int v664;
        v664 = v663 / 32l;
        auto v665 = cooperative_groups::labeled_partition(v662,v664);
        Closure6 v666{};
        float v667; int v668;
        Tuple1 tmp59 = cooperative_groups::reduce(v665, Tuple1{v649, v650}, v666);
        v667 = tmp59.v0; v668 = tmp59.v1;
        float v669;
        v669 = v635 * v667;
        int v670[8l];
        bool v671[8l];
        int v672;
        v672 = 0l;
        while (while_method_5(v672)){
            int v674;
            v674 = 0l;
            while (while_method_1(v674)){
                assert("Tensor range check" && 0 <= v672 && v672 < 2l);
                assert("Tensor range check" && 0 <= v674 && v674 < 4l);
                int v676;
                v676 = 4l * v672;
                int v677;
                v677 = v676 + v674;
                float v678;
                v678 = v601[v677];
                bool v679;
                v679 = v602[v677];
                int v680;
                v680 = v443[v677];
                int v683; bool v684;
                if (v679){
                    float v681;
                    v681 = v678 - v669;
                    bool v682;
                    v682 = v681 >= 0.0f;
                    v683 = v680; v684 = v682;
                } else {
                    v683 = 2147483647l; v684 = false;
                }
                assert("Tensor range check" && 0 <= v672 && v672 < 2l);
                assert("Tensor range check" && 0 <= v674 && v674 < 4l);
                v670[v677] = v683;
                v671[v677] = v684;
                v674 += 1l ;
            }
            v672 += 1l ;
        }
        int v685; bool v686;
        Tuple4 tmp60 = Tuple4{2147483647l, false};
        v685 = tmp60.v0; v686 = tmp60.v1;
        int v687;
        v687 = 0l;
        while (while_method_5(v687)){
            int v689;
            v689 = 0l;
            while (while_method_1(v689)){
                assert("Tensor range check" && 0 <= v687 && v687 < 2l);
                assert("Tensor range check" && 0 <= v689 && v689 < 4l);
                int v691;
                v691 = 4l * v687;
                int v692;
                v692 = v691 + v689;
                int v693;
                v693 = v670[v692];
                bool v694;
                v694 = v671[v692];
                int v701; bool v702;
                if (v686){
                    if (v694){
                        bool v695;
                        v695 = v685 < v693;
                        int v696;
                        if (v695){
                            v696 = v685;
                        } else {
                            v696 = v693;
                        }
                        v701 = v696; v702 = true;
                    } else {
                        v701 = v685; v702 = v686;
                    }
                } else {
                    if (v694){
                        v701 = v693; v702 = v694;
                    } else {
                        v701 = v685; v702 = v686;
                    }
                }
                v685 = v701;
                v686 = v702;
                v689 += 1l ;
            }
            v687 += 1l ;
        }
        auto v703 = cooperative_groups::coalesced_threads();
        int v704;
        v704 = threadIdx.x;
        int v705;
        v705 = v704 / 32l;
        auto v706 = cooperative_groups::labeled_partition(v703,v705);
        Closure7 v707{};
        int v708; bool v709;
        Tuple4 tmp61 = cooperative_groups::reduce(v706, Tuple4{v685, v686}, v707);
        v708 = tmp61.v0; v709 = tmp61.v1;
        bool v710;
        v710 = v709 == false;
        if (v710){
            assert("The local reduce must be true." && v709);
        } else {
        }
        int v712;
        v712 = 0l;
        while (while_method_5(v712)){
            assert("Tensor range check" && 0 <= v712 && v712 < 2l);
            assert("Tensor range check" && 0 <= v712 && v712 < 2l);
            v712 += 1l ;
        }
        assert("Tensor range check" && 0 <= v436 && v436 < 32l);
        v415[v436] = v708;
        v425 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v416 && v416 < 32l);
    int v714;
    v714 = v415[v416];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v715;
    v715 = threadIdx.x;
    assert("Tensor range check" && 0 <= v715 && v715 < 32l);
    v5[v715] = v714;
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
        v1267 += 24l ;
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
        v1251 += 24l ;
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
