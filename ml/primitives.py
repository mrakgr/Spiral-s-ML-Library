kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups/reduce.h>
#include <curand_kernel.h>
#include <cooperative_groups/scan.h>
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
struct Tuple0 {
    int v0;
    float v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(int t0, float t1) : v0(t0), v1(t1) {}
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
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
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
    while (while_method_0(v39)){
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
        while (while_method_1(v57)){
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
        while (while_method_1(v64)){
            int v66;
            v66 = 0l;
            while (while_method_2(v66)){
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
        while (while_method_1(v91)){
            int v93;
            v93 = 0l;
            while (while_method_2(v93)){
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
        while (while_method_1(v98)){
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
    while (while_method_0(v123)){
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
        while (while_method_1(v139)){
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
        while (while_method_1(v146)){
            int v148;
            v148 = 0l;
            while (while_method_2(v148)){
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
        while (while_method_1(v171)){
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
    while (while_method_0(v196)){
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
        while (while_method_1(v213)){
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
        while (while_method_1(v220)){
            int v222;
            v222 = 0l;
            while (while_method_2(v222)){
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
        while (while_method_1(v245)){
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
    while (while_method_0(v271)){
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
        while (while_method_1(v288)){
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
        while (while_method_1(v295)){
            int v297;
            v297 = 0l;
            while (while_method_2(v297)){
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
        while (while_method_1(v321)){
            int v323;
            v323 = 0l;
            while (while_method_2(v323)){
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
        while (while_method_1(v331)){
            int v333;
            v333 = 0l;
            while (while_method_2(v333)){
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
        while (while_method_1(v343)){
            int v345;
            v345 = 0l;
            while (while_method_2(v345)){
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
        while (while_method_1(v358)){
            int v360;
            v360 = 0l;
            while (while_method_2(v360)){
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
        while (while_method_1(v367)){
            int v369;
            v369 = 0l;
            while (while_method_2(v369)){
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
        Closure1 v379{};
        int v380;
        v380 = cooperative_groups::reduce(v378, v366, v379);
        float v381;
        v381 = (float)v380;
        float v382;
        v382 = 1.0f / v381;
        float v383[8l];
        int v384;
        v384 = 0l;
        while (while_method_1(v384)){
            int v386;
            v386 = 0l;
            while (while_method_2(v386)){
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
        while (while_method_1(v398)){
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
    while (while_method_0(v421)){
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
        while (while_method_1(v437)){
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
        while (while_method_1(v444)){
            int v446;
            v446 = 0l;
            while (while_method_2(v446)){
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
        while (while_method_1(v473)){
            int v475;
            v475 = 0l;
            while (while_method_2(v475)){
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
        while (while_method_1(v483)){
            int v485;
            v485 = 0l;
            while (while_method_2(v485)){
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
        while (while_method_1(v492)){
            int v494;
            v494 = 0l;
            while (while_method_2(v494)){
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
        Closure1 v504{};
        int v505;
        v505 = cooperative_groups::reduce(v503, v491, v504);
        float v506[8l];
        int v507;
        v507 = 0l;
        while (while_method_1(v507)){
            int v509;
            v509 = 0l;
            while (while_method_2(v509)){
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
        while (while_method_1(v517)){
            int v519;
            v519 = 0l;
            while (while_method_2(v519)){
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
        while (while_method_1(v534)){
            int v536;
            v536 = 0l;
            while (while_method_2(v536)){
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
        while (while_method_1(v546)){
            int v548;
            v548 = 0l;
            while (while_method_2(v548)){
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
        while (while_method_1(v560)){
            int v562;
            v562 = 0l;
            while (while_method_2(v562)){
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
        while (while_method_1(v570)){
            assert("Tensor range check" && 0 <= v570 && v570 < 2l);
            int v572;
            v572 = 4l * v570;
            assert("Tensor range check" && 0 <= v570 && v570 < 2l);
            int v573; float v574;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v573 = tmp0.v0; v574 = tmp0.v1;
            while (while_method_2(v573)){
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
            Tuple0 tmp1 = Tuple0{0l, v589};
            v590 = tmp1.v0; v591 = tmp1.v1;
            while (while_method_2(v590)){
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
        while (while_method_1(v599)){
            int v601;
            v601 = 0l;
            while (while_method_2(v601)){
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
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, false};
        v608 = tmp2.v0; v609 = tmp2.v1;
        int v610;
        v610 = 0l;
        while (while_method_1(v610)){
            int v612;
            v612 = 0l;
            while (while_method_2(v612)){
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
        Closure3 v630{};
        float v631; bool v632;
        Tuple1 tmp3 = cooperative_groups::reduce(v629, Tuple1{v608, v609}, v630);
        v631 = tmp3.v0; v632 = tmp3.v1;
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
        while (while_method_1(v637)){
            int v639;
            v639 = 0l;
            while (while_method_2(v639)){
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
        Tuple2 tmp4 = Tuple2{0.0f, 2147483647l};
        v645 = tmp4.v0; v646 = tmp4.v1;
        int v647;
        v647 = 0l;
        while (while_method_1(v647)){
            int v649;
            v649 = 0l;
            while (while_method_2(v649)){
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
        Closure4 v662{};
        float v663; int v664;
        Tuple2 tmp5 = cooperative_groups::reduce(v661, Tuple2{v645, v646}, v662);
        v663 = tmp5.v0; v664 = tmp5.v1;
        float v665;
        v665 = v631 * v663;
        int v666[8l];
        bool v667[8l];
        int v668;
        v668 = 0l;
        while (while_method_1(v668)){
            int v670;
            v670 = 0l;
            while (while_method_2(v670)){
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
        Tuple3 tmp6 = Tuple3{2147483647l, false};
        v681 = tmp6.v0; v682 = tmp6.v1;
        int v683;
        v683 = 0l;
        while (while_method_1(v683)){
            int v685;
            v685 = 0l;
            while (while_method_2(v685)){
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
        Closure5 v703{};
        int v704; bool v705;
        Tuple3 tmp7 = cooperative_groups::reduce(v702, Tuple3{v681, v682}, v703);
        v704 = tmp7.v0; v705 = tmp7.v1;
        bool v706;
        v706 = v705 == false;
        if (v706){
            assert("The local reduce must be true." && v705);
        } else {
        }
        int v708;
        v708 = 0l;
        while (while_method_1(v708)){
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
    v1 = v0 < 32
    del v0
    return v1
def method2(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method0(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/test3/input_identity.txt"
    pathlib.Path(v1).parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(v1,'w')
    del v1
    v28 = 0
    v29 = "{}"
    print(v29.format('['),end="")
    v30 = 0
    while method1(v30):
        v32 = v28
        v33 = v32 >= 2147483647
        del v32
        if v33:
            v34 = " ..."
            print(v29.format(v34),end="")
            del v34
            break
        else:
            pass
        del v33
        v35 = v30 == 0
        v36 = v35 != True
        del v35
        if v36:
            v37 = "; "
            print(v29.format(v37),end="")
            del v37
        else:
            pass
        del v36
        print(v29.format('['),end="")
        v38 = 0
        while method2(v38):
            v40 = v28
            v41 = v40 >= 2147483647
            del v40
            if v41:
                v42 = " ..."
                print(v29.format(v42),end="")
                del v42
                break
            else:
                pass
            del v41
            v43 = v38 == 0
            v44 = v43 != True
            del v43
            if v44:
                v45 = "; "
                print(v29.format(v45),end="")
                del v45
            else:
                pass
            del v44
            v46 = v28 + 1
            v28 = v46
            del v46
            v47 = v30 * 256
            v48 = v47 + v38
            del v47
            v49 = v0[v48].item()
            del v48
            v50 = "{:.6f}"
            print(v50.format(v49),end="")
            del v49, v50
            v38 += 1 
        del v38
        print(v29.format(']'),end="")
        v30 += 1 
    del v0, v28, v30
    print(v29.format(']'),end="")
    del v29
    v51 = "\n"
    print(v51,end="")
    del v51
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method3(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/test3/output_sample_reduce.txt"
    pathlib.Path(v1).parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(v1,'w')
    del v1
    v17 = 0
    v18 = "{}"
    print(v18.format('['),end="")
    v19 = 0
    while method1(v19):
        v21 = v17
        v22 = v21 >= 2147483647
        del v21
        if v22:
            v23 = " ..."
            print(v18.format(v23),end="")
            del v23
            break
        else:
            pass
        del v22
        v24 = v19 == 0
        v25 = v24 != True
        del v24
        if v25:
            v26 = "; "
            print(v18.format(v26),end="")
            del v26
        else:
            pass
        del v25
        v27 = v17 + 1
        v17 = v27
        del v27
        v28 = v0[v19].item()
        print(v18.format(v28),end="")
        del v28
        v19 += 1 
    del v0, v17, v19
    print(v18.format(']'),end="")
    del v18
    v29 = "\n"
    print(v29,end="")
    del v29
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method4(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/test3/output_indices_map.txt"
    pathlib.Path(v2).parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(v2,'w')
    del v2
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method1(v32):
        v34 = v30
        v35 = v34 >= 2147483647
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
            v43 = v42 >= 2147483647
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
            v49 = v32 * 256
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            v52 = v1[v50].item()
            del v50
            v53 = "{}, {}"
            print(v53.format(v51, v52),end="")
            del v51, v52, v53
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v1, v30, v32
    print(v31.format(']'),end="")
    del v31
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method5(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/test3/output_indices_map.txt"
    pathlib.Path(v1).parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(v1,'w')
    del v1
    v17 = 0
    v18 = "{}"
    print(v18.format('['),end="")
    v19 = 0
    while method1(v19):
        v21 = v17
        v22 = v21 >= 2147483647
        del v21
        if v22:
            v23 = " ..."
            print(v18.format(v23),end="")
            del v23
            break
        else:
            pass
        del v22
        v24 = v19 == 0
        v25 = v24 != True
        del v24
        if v25:
            v26 = "; "
            print(v18.format(v26),end="")
            del v26
        else:
            pass
        del v25
        v27 = v17 + 1
        v17 = v27
        del v27
        v28 = v0[v19].item()
        print(v18.format(v28),end="")
        del v28
        v19 += 1 
    del v0, v17, v19
    print(v18.format(']'),end="")
    del v18
    v29 = "\n"
    print(v29,end="")
    del v29
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method6(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/test3/output_op_map.txt"
    pathlib.Path(v1).parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(v1,'w')
    del v1
    v28 = 0
    v29 = "{}"
    print(v29.format('['),end="")
    v30 = 0
    while method1(v30):
        v32 = v28
        v33 = v32 >= 2147483647
        del v32
        if v33:
            v34 = " ..."
            print(v29.format(v34),end="")
            del v34
            break
        else:
            pass
        del v33
        v35 = v30 == 0
        v36 = v35 != True
        del v35
        if v36:
            v37 = "; "
            print(v29.format(v37),end="")
            del v37
        else:
            pass
        del v36
        print(v29.format('['),end="")
        v38 = 0
        while method2(v38):
            v40 = v28
            v41 = v40 >= 2147483647
            del v40
            if v41:
                v42 = " ..."
                print(v29.format(v42),end="")
                del v42
                break
            else:
                pass
            del v41
            v43 = v38 == 0
            v44 = v43 != True
            del v43
            if v44:
                v45 = "; "
                print(v29.format(v45),end="")
                del v45
            else:
                pass
            del v44
            v46 = v28 + 1
            v28 = v46
            del v46
            v47 = v30 * 256
            v48 = v47 + v38
            del v47
            v49 = v0[v48].item()
            del v48
            v50 = "{:.6f}"
            print(v50.format(v49),end="")
            del v49, v50
            v38 += 1 
        del v38
        print(v29.format(']'),end="")
        v30 += 1 
    del v0, v28, v30
    print(v29.format(']'),end="")
    del v29
    v51 = "\n"
    print(v51,end="")
    del v51
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method7(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/test3/zip_input_output_identity_map.txt"
    pathlib.Path(v2).parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(v2,'w')
    del v2
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method1(v32):
        v34 = v30
        v35 = v34 >= 2147483647
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
            v43 = v42 >= 2147483647
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
            v49 = v32 * 256
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            v52 = v1[v50].item()
            del v50
            v53 = "{:.6f}, {:.6f}"
            print(v53.format(v51, v52),end="")
            del v51, v52, v53
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v1, v30, v32
    print(v31.format(']'),end="")
    del v31
    v54 = "\n"
    print(v54,end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def main_body():
    cp.random.seed(12344321)
    v0 = cp.arange(0,8192,1,dtype=cp.float32) # type: ignore
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
    v6 = cp.empty(8192,dtype=cp.int32)
    v7 = cp.empty(8192,dtype=cp.int32)
    v8 = cp.empty(32,dtype=cp.int32)
    v9 = cp.empty(32,dtype=cp.int32)
    v10 = cp.empty(8192,dtype=cp.float32)
    v11 = cp.empty(8192,dtype=cp.float32)
    v12 = 0
    v13 = raw_module.get_function(f"entry{v12}")
    del v12
    v13.max_dynamic_shared_size_bytes = 0 
    v13((1,),(32,),(v0, v5, v6, v7, v8, v9, v10, v11),shared_mem=0)
    del v13
    method0(v0)
    del v0
    method3(v9)
    del v9
    method4(v6, v7)
    del v6, v7
    method5(v8)
    del v8
    method6(v11)
    del v11
    return method7(v5, v10)

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
