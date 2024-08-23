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
        unsigned long long v469;
        v469 = clock64();
        int v470;
        v470 = threadIdx.x;
        unsigned long long v471;
        v471 = (unsigned long long)v470;
        curandStatePhilox4_32_10_t v472;
        curand_init(v469,v471,0ull,&v472);
        bool v473[8l];
        int v474;
        v474 = 0l;
        while (while_method_1(v474)){
            int v476;
            v476 = 0l;
            while (while_method_2(v476)){
                assert("Tensor range check" && 0 <= v474 && v474 < 2l);
                assert("Tensor range check" && 0 <= v476 && v476 < 4l);
                int v478;
                v478 = 4l * v474;
                int v479;
                v479 = v478 + v476;
                float v480;
                v480 = v435[v479];
                int v481;
                v481 = v436[v479];
                bool v482;
                v482 = v481 < 3l;
                assert("Tensor range check" && 0 <= v474 && v474 < 2l);
                assert("Tensor range check" && 0 <= v476 && v476 < 4l);
                v473[v479] = v482;
                v476 += 1l ;
            }
            v474 += 1l ;
        }
        int v483[8l];
        int v484;
        v484 = 0l;
        while (while_method_1(v484)){
            int v486;
            v486 = 0l;
            while (while_method_2(v486)){
                assert("Tensor range check" && 0 <= v484 && v484 < 2l);
                assert("Tensor range check" && 0 <= v486 && v486 < 4l);
                int v488;
                v488 = 4l * v484;
                int v489;
                v489 = v488 + v486;
                bool v490;
                v490 = v473[v489];
                int v491;
                if (v490){
                    v491 = 1l;
                } else {
                    v491 = 0l;
                }
                assert("Tensor range check" && 0 <= v484 && v484 < 2l);
                assert("Tensor range check" && 0 <= v486 && v486 < 4l);
                v483[v489] = v491;
                v486 += 1l ;
            }
            v484 += 1l ;
        }
        int v492;
        v492 = 0l;
        int v493;
        v493 = 0l;
        while (while_method_1(v493)){
            int v495;
            v495 = 0l;
            while (while_method_2(v495)){
                assert("Tensor range check" && 0 <= v493 && v493 < 2l);
                assert("Tensor range check" && 0 <= v495 && v495 < 4l);
                int v497;
                v497 = 4l * v493;
                int v498;
                v498 = v497 + v495;
                int v499;
                v499 = v483[v498];
                int v500;
                v500 = v492 + v499;
                v492 = v500;
                v495 += 1l ;
            }
            v493 += 1l ;
        }
        auto v501 = cooperative_groups::coalesced_threads();
        int v502;
        v502 = threadIdx.x;
        int v503;
        v503 = v502 / 32l;
        auto v504 = cooperative_groups::labeled_partition(v501,v503);
        Closure1 v505{};
        int v506;
        v506 = cooperative_groups::reduce(v504, v492, v505);
        float v507[8l];
        int v508;
        v508 = 0l;
        while (while_method_1(v508)){
            int v510;
            v510 = 0l;
            while (while_method_2(v510)){
                assert("Tensor range check" && 0 <= v508 && v508 < 2l);
                assert("Tensor range check" && 0 <= v510 && v510 < 4l);
                int v512;
                v512 = 4l * v508;
                int v513;
                v513 = v512 + v510;
                float v514;
                v514 = v435[v513];
                bool v515;
                v515 = v473[v513];
                float v516;
                if (v515){
                    v516 = v514;
                } else {
                    v516 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v508 && v508 < 2l);
                assert("Tensor range check" && 0 <= v510 && v510 < 4l);
                v507[v513] = v516;
                v510 += 1l ;
            }
            v508 += 1l ;
        }
        float v517;
        v517 = 0.0f;
        int v518;
        v518 = 0l;
        while (while_method_1(v518)){
            int v520;
            v520 = 0l;
            while (while_method_2(v520)){
                assert("Tensor range check" && 0 <= v518 && v518 < 2l);
                assert("Tensor range check" && 0 <= v520 && v520 < 4l);
                int v522;
                v522 = 4l * v518;
                int v523;
                v523 = v522 + v520;
                float v524;
                v524 = v507[v523];
                float v525;
                v525 = v517 + v524;
                v517 = v525;
                v520 += 1l ;
            }
            v518 += 1l ;
        }
        auto v526 = cooperative_groups::coalesced_threads();
        int v527;
        v527 = threadIdx.x;
        int v528;
        v528 = v527 / 32l;
        auto v529 = cooperative_groups::labeled_partition(v526,v528);
        Closure0 v530{};
        float v531;
        v531 = cooperative_groups::reduce(v529, v517, v530);
        float v532;
        v532 = (float)v506;
        float v533;
        v533 = v531 / v532;
        float v534[8l];
        int v535;
        v535 = 0l;
        while (while_method_1(v535)){
            int v537;
            v537 = 0l;
            while (while_method_2(v537)){
                assert("Tensor range check" && 0 <= v535 && v535 < 2l);
                assert("Tensor range check" && 0 <= v537 && v537 < 4l);
                int v539;
                v539 = 4l * v535;
                int v540;
                v540 = v539 + v537;
                float v541;
                v541 = v435[v540];
                bool v542;
                v542 = v473[v540];
                float v543;
                if (v542){
                    v543 = v541;
                } else {
                    v543 = -1.0f / 0.0f;
                }
                float v544;
                v544 = v543 - v533;
                float v545;
                v545 = exp(v544);
                assert("Tensor range check" && 0 <= v535 && v535 < 2l);
                assert("Tensor range check" && 0 <= v537 && v537 < 4l);
                v534[v540] = v545;
                v537 += 1l ;
            }
            v535 += 1l ;
        }
        float v546;
        v546 = 0.0f;
        int v547;
        v547 = 0l;
        while (while_method_1(v547)){
            int v549;
            v549 = 0l;
            while (while_method_2(v549)){
                assert("Tensor range check" && 0 <= v547 && v547 < 2l);
                assert("Tensor range check" && 0 <= v549 && v549 < 4l);
                int v551;
                v551 = 4l * v547;
                int v552;
                v552 = v551 + v549;
                float v553;
                v553 = v534[v552];
                float v554;
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
        float v559;
        v559 = cooperative_groups::reduce(v558, v546, v530);
        float v560[8l];
        int v561;
        v561 = 0l;
        while (while_method_1(v561)){
            int v563;
            v563 = 0l;
            while (while_method_2(v563)){
                assert("Tensor range check" && 0 <= v561 && v561 < 2l);
                assert("Tensor range check" && 0 <= v563 && v563 < 4l);
                int v565;
                v565 = 4l * v561;
                int v566;
                v566 = v565 + v563;
                float v567;
                v567 = v534[v566];
                float v568;
                v568 = v567 / v559;
                assert("Tensor range check" && 0 <= v561 && v561 < 2l);
                assert("Tensor range check" && 0 <= v563 && v563 < 4l);
                v560[v566] = v568;
                v563 += 1l ;
            }
            v561 += 1l ;
        }
        float v569[8l];
        float v570;
        v570 = 0.0f;
        int v571;
        v571 = 0l;
        while (while_method_1(v571)){
            assert("Tensor range check" && 0 <= v571 && v571 < 2l);
            int v573;
            v573 = 4l * v571;
            assert("Tensor range check" && 0 <= v571 && v571 < 2l);
            int v574; float v575;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v574 = tmp0.v0; v575 = tmp0.v1;
            while (while_method_2(v574)){
                assert("Tensor range check" && 0 <= v574 && v574 < 4l);
                int v577;
                v577 = v574 + v573;
                float v578;
                v578 = v560[v577];
                float v579;
                v579 = v575 + v578;
                v575 = v579;
                v574 += 1l ;
            }
            auto v580 = cooperative_groups::coalesced_threads();
            int v581;
            v581 = threadIdx.x;
            int v582;
            v582 = v581 / 32l;
            auto v583 = cooperative_groups::labeled_partition(v580,v582);
            Closure2 v584{};
            float v585;
            v585 = cooperative_groups::inclusive_scan(v583, v575, v584);
            float v586;
            v586 = v583.shfl_up(v585,1);
            bool v587;
            v587 = v583.thread_rank() == 0;
            float v588;
            if (v587){
                v588 = 0.0f;
            } else {
                v588 = v586;
            }
            float v589;
            v589 = v583.shfl(v585,v583.num_threads()-1);
            float v590;
            v590 = v570 + v588;
            int v591; float v592;
            Tuple0 tmp1 = Tuple0{0l, v590};
            v591 = tmp1.v0; v592 = tmp1.v1;
            while (while_method_2(v591)){
                assert("Tensor range check" && 0 <= v591 && v591 < 4l);
                int v594;
                v594 = v591 + v573;
                float v595;
                v595 = v560[v594];
                float v596;
                v596 = v592 + v595;
                assert("Tensor range check" && 0 <= v591 && v591 < 4l);
                v569[v594] = v596;
                v592 = v596;
                v591 += 1l ;
            }
            float v597;
            v597 = v570 + v589;
            v570 = v597;
            v571 += 1l ;
        }
        float v598[8l];
        bool v599[8l];
        int v600;
        v600 = 0l;
        while (while_method_1(v600)){
            int v602;
            v602 = 0l;
            while (while_method_2(v602)){
                assert("Tensor range check" && 0 <= v600 && v600 < 2l);
                assert("Tensor range check" && 0 <= v602 && v602 < 4l);
                int v604;
                v604 = 4l * v600;
                int v605;
                v605 = v604 + v602;
                float v606;
                v606 = v569[v605];
                float v607;
                v607 = v560[v605];
                bool v608;
                v608 = v607 > 0.0f;
                assert("Tensor range check" && 0 <= v600 && v600 < 2l);
                assert("Tensor range check" && 0 <= v602 && v602 < 4l);
                v598[v605] = v606;
                v599[v605] = v608;
                v602 += 1l ;
            }
            v600 += 1l ;
        }
        float v609; bool v610;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, false};
        v609 = tmp2.v0; v610 = tmp2.v1;
        int v611;
        v611 = 0l;
        while (while_method_1(v611)){
            int v613;
            v613 = 0l;
            while (while_method_2(v613)){
                assert("Tensor range check" && 0 <= v611 && v611 < 2l);
                assert("Tensor range check" && 0 <= v613 && v613 < 4l);
                int v615;
                v615 = 4l * v611;
                int v616;
                v616 = v615 + v613;
                float v617;
                v617 = v598[v616];
                bool v618;
                v618 = v599[v616];
                float v625; bool v626;
                if (v610){
                    if (v618){
                        bool v619;
                        v619 = v609 >= v617;
                        float v620;
                        if (v619){
                            v620 = v609;
                        } else {
                            v620 = v617;
                        }
                        v625 = v620; v626 = true;
                    } else {
                        v625 = v609; v626 = v610;
                    }
                } else {
                    if (v618){
                        v625 = v617; v626 = v618;
                    } else {
                        v625 = v609; v626 = v610;
                    }
                }
                v609 = v625;
                v610 = v626;
                v613 += 1l ;
            }
            v611 += 1l ;
        }
        auto v627 = cooperative_groups::coalesced_threads();
        int v628;
        v628 = threadIdx.x;
        int v629;
        v629 = v628 / 32l;
        auto v630 = cooperative_groups::labeled_partition(v627,v629);
        Closure3 v631{};
        float v632; bool v633;
        Tuple1 tmp3 = cooperative_groups::reduce(v630, Tuple1{v609, v610}, v631);
        v632 = tmp3.v0; v633 = tmp3.v1;
        bool v634;
        v634 = v633 == false;
        if (v634){
            assert("The local reduce must be true." && v633);
        } else {
        }
        float v636[8l];
        int v637[8l];
        int v638;
        v638 = 0l;
        while (while_method_1(v638)){
            int v640;
            v640 = 0l;
            while (while_method_2(v640)){
                assert("Tensor range check" && 0 <= v638 && v638 < 2l);
                assert("Tensor range check" && 0 <= v640 && v640 < 4l);
                int v642;
                v642 = 4l * v638;
                int v643;
                v643 = v642 + v640;
                int v644;
                v644 = v436[v643];
                float v645;
                v645 = curand_uniform(&v472);
                assert("Tensor range check" && 0 <= v638 && v638 < 2l);
                assert("Tensor range check" && 0 <= v640 && v640 < 4l);
                v636[v643] = v645;
                v637[v643] = v644;
                v640 += 1l ;
            }
            v638 += 1l ;
        }
        float v646; int v647;
        Tuple2 tmp4 = Tuple2{0.0f, 2147483647l};
        v646 = tmp4.v0; v647 = tmp4.v1;
        int v648;
        v648 = 0l;
        while (while_method_1(v648)){
            int v650;
            v650 = 0l;
            while (while_method_2(v650)){
                assert("Tensor range check" && 0 <= v648 && v648 < 2l);
                assert("Tensor range check" && 0 <= v650 && v650 < 4l);
                int v652;
                v652 = 4l * v648;
                int v653;
                v653 = v652 + v650;
                float v654;
                v654 = v636[v653];
                int v655;
                v655 = v637[v653];
                bool v656;
                v656 = v647 < v655;
                float v657; int v658;
                if (v656){
                    v657 = v646; v658 = v647;
                } else {
                    v657 = v654; v658 = v655;
                }
                v646 = v657;
                v647 = v658;
                v650 += 1l ;
            }
            v648 += 1l ;
        }
        auto v659 = cooperative_groups::coalesced_threads();
        int v660;
        v660 = threadIdx.x;
        int v661;
        v661 = v660 / 32l;
        auto v662 = cooperative_groups::labeled_partition(v659,v661);
        Closure4 v663{};
        float v664; int v665;
        Tuple2 tmp5 = cooperative_groups::reduce(v662, Tuple2{v646, v647}, v663);
        v664 = tmp5.v0; v665 = tmp5.v1;
        float v666;
        v666 = v632 * v664;
        int v667[8l];
        bool v668[8l];
        int v669;
        v669 = 0l;
        while (while_method_1(v669)){
            int v671;
            v671 = 0l;
            while (while_method_2(v671)){
                assert("Tensor range check" && 0 <= v669 && v669 < 2l);
                assert("Tensor range check" && 0 <= v671 && v671 < 4l);
                int v673;
                v673 = 4l * v669;
                int v674;
                v674 = v673 + v671;
                float v675;
                v675 = v598[v674];
                bool v676;
                v676 = v599[v674];
                int v677;
                v677 = v436[v674];
                int v680; bool v681;
                if (v676){
                    float v678;
                    v678 = v675 - v666;
                    bool v679;
                    v679 = v678 >= 0.0f;
                    v680 = v677; v681 = v679;
                } else {
                    v680 = 2147483647l; v681 = false;
                }
                assert("Tensor range check" && 0 <= v669 && v669 < 2l);
                assert("Tensor range check" && 0 <= v671 && v671 < 4l);
                v667[v674] = v680;
                v668[v674] = v681;
                v671 += 1l ;
            }
            v669 += 1l ;
        }
        int v682; bool v683;
        Tuple3 tmp6 = Tuple3{2147483647l, false};
        v682 = tmp6.v0; v683 = tmp6.v1;
        int v684;
        v684 = 0l;
        while (while_method_1(v684)){
            int v686;
            v686 = 0l;
            while (while_method_2(v686)){
                assert("Tensor range check" && 0 <= v684 && v684 < 2l);
                assert("Tensor range check" && 0 <= v686 && v686 < 4l);
                int v688;
                v688 = 4l * v684;
                int v689;
                v689 = v688 + v686;
                int v690;
                v690 = v667[v689];
                bool v691;
                v691 = v668[v689];
                int v698; bool v699;
                if (v683){
                    if (v691){
                        bool v692;
                        v692 = v682 < v690;
                        int v693;
                        if (v692){
                            v693 = v682;
                        } else {
                            v693 = v690;
                        }
                        v698 = v693; v699 = true;
                    } else {
                        v698 = v682; v699 = v683;
                    }
                } else {
                    if (v691){
                        v698 = v690; v699 = v691;
                    } else {
                        v698 = v682; v699 = v683;
                    }
                }
                v682 = v698;
                v683 = v699;
                v686 += 1l ;
            }
            v684 += 1l ;
        }
        auto v700 = cooperative_groups::coalesced_threads();
        int v701;
        v701 = threadIdx.x;
        int v702;
        v702 = v701 / 32l;
        auto v703 = cooperative_groups::labeled_partition(v700,v702);
        Closure5 v704{};
        int v705; bool v706;
        Tuple3 tmp7 = cooperative_groups::reduce(v703, Tuple3{v682, v683}, v704);
        v705 = tmp7.v0; v706 = tmp7.v1;
        bool v707;
        v707 = v706 == false;
        if (v707){
            assert("The local reduce must be true." && v706);
        } else {
        }
        int v709;
        v709 = 0l;
        while (while_method_1(v709)){
            assert("Tensor range check" && 0 <= v709 && v709 < 2l);
            assert("Tensor range check" && 0 <= v709 && v709 < 2l);
            v709 += 1l ;
        }
        assert("Tensor range check" && 0 <= v432 && v432 < 32l);
        v410[v432] = v705;
        v421 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v714;
    if (v412){
        assert("Tensor range check" && 0 <= v411 && v411 < 32l);
        int v711;
        v711 = v410[v411];
        v714 = v711;
    } else {
        int v712[1l];
        int v713;
        v713 = v712[0l];
        v714 = v713;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v715;
    v715 = threadIdx.x;
    assert("Tensor range check" && 0 <= v715 && v715 < 32l);
    v5[v715] = v714;
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
def method0(v0 : i32) -> bool:
    v1 = v0 < 32
    del v0
    return v1
def method1(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def main_body():
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
    del v0, v6, v7, v8, v9, v11, v13
    v41 = 0
    v42 = "{}"
    print(v42.format('['),end="")
    v43 = 0
    while method0(v43):
        v45 = v41
        v46 = v45 >= 2147483647
        del v45
        if v46:
            v47 = " ..."
            print(v42.format(v47),end="")
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
            print(v42.format(v50),end="")
            del v50
        else:
            pass
        del v49
        print(v42.format('['),end="")
        v51 = 0
        while method1(v51):
            v53 = v41
            v54 = v53 >= 2147483647
            del v53
            if v54:
                v55 = " ..."
                print(v42.format(v55),end="")
                del v55
                break
            else:
                pass
            del v54
            v56 = v51 == 0
            v57 = v56 != True
            del v56
            if v57:
                v58 = "; "
                print(v42.format(v58),end="")
                del v58
            else:
                pass
            del v57
            v59 = v41 + 1
            v41 = v59
            del v59
            v60 = v43 * 256
            v61 = v60 + v51
            del v60
            v62 = v5[v61].item()
            v63 = v10[v61].item()
            del v61
            v64 = "{:.6f}, {:.6f}"
            print(v64.format(v62, v63),end="")
            del v62, v63, v64
            v51 += 1 
        del v51
        print(v42.format(']'),end="")
        v43 += 1 
    del v5, v10, v41, v43
    print(v42.format(']'),end="")
    del v42
    v65 = "\n"
    print(v65,end="")
    del v65
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
