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
    assert("Tensor range check" && 0 <= v29 && v29 < 32l);
    v24[v29] = v18;
    v25[v29] = v20;
    v26[v29] = v22;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v30;
    v30 = threadIdx.x;
    bool v31;
    v31 = 0l <= v30;
    bool v32;
    v32 = v31 == false;
    if (v32){
        assert("The index needs to be zero or positive." && v31);
    } else {
    }
    int v34;
    v34 = v30 % 32l;
    int v35;
    v35 = v30 / 32l;
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
    int v107;
    v107 = threadIdx.x;
    assert("Tensor range check" && 0 <= v107 && v107 < 32l);
    /* void array index */;
    float * v108;
    v108 = v1+v9;
    __shared__ float * v110[32l];
    /* void shared array create v111 */;
    __shared__ int v112[32l];
    int v113;
    v113 = threadIdx.x;
    assert("Tensor range check" && 0 <= v113 && v113 < 32l);
    v110[v113] = v108;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v114;
    v114 = threadIdx.x;
    bool v115;
    v115 = 0l <= v114;
    bool v116;
    v116 = v115 == false;
    if (v116){
        assert("The index needs to be zero or positive." && v115);
    } else {
    }
    int v118;
    v118 = v114 % 32l;
    int v119;
    v119 = v114 / 32l;
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
    int v173;
    v173 = threadIdx.x;
    assert("Tensor range check" && 0 <= v173 && v173 < 32l);
    int v174;
    v174 = v112[v173];
    int v175;
    v175 = threadIdx.x;
    assert("Tensor range check" && 0 <= v175 && v175 < 32l);
    v4[v175] = v174;
    float * v176;
    v176 = v1+v9;
    float * v178;
    v178 = v6+v17;
    __shared__ float * v180[32l];
    __shared__ float * v181[32l];
    /* void shared array create v182 */;
    /* void shared array create v183 */;
    int v184;
    v184 = threadIdx.x;
    assert("Tensor range check" && 0 <= v184 && v184 < 32l);
    v180[v184] = v176;
    v181[v184] = v178;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v185;
    v185 = threadIdx.x;
    bool v186;
    v186 = 0l <= v185;
    bool v187;
    v187 = v186 == false;
    if (v187){
        assert("The index needs to be zero or positive." && v186);
    } else {
    }
    int v189;
    v189 = v185 % 32l;
    int v190;
    v190 = v185 / 32l;
    bool v191;
    v191 = v190 < 1l;
    bool v192;
    v192 = v191 == false;
    if (v192){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v191);
    } else {
    }
    assert("Tensor range check" && 0 <= v190 && v190 < 1l);
    int v194;
    v194 = 0l;
    while (while_method_0(v194)){
        bool v196;
        v196 = 0l <= v190;
        bool v197;
        v197 = v196 && v191;
        bool v198;
        v198 = v197 == false;
        if (v198){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v197);
        } else {
        }
        bool v200;
        v200 = 0l <= v194;
        bool v202;
        if (v200){
            bool v201;
            v201 = v194 < 32l;
            v202 = v201;
        } else {
            v202 = false;
        }
        bool v203;
        v203 = v202 == false;
        if (v203){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v202);
        } else {
        }
        int v205;
        v205 = v194 + v190;
        assert("Tensor range check" && 0 <= v194 && v194 < 32l);
        float * v206;
        v206 = v180[v205];
        float * v207;
        v207 = v181[v205];
        /* void array index */;
        assert("Tensor range check" && 0 <= v189 && v189 < 32l);
        int v208;
        v208 = 4l * v189;
        float v209[8l];
        int v210[8l];
        int v211;
        v211 = 0l;
        while (while_method_1(v211)){
            assert("Tensor range check" && 0 <= v211 && v211 < 2l);
            int v213;
            v213 = 4l * v211;
            assert("Tensor range check" && 0 <= v211 && v211 < 2l);
            int v214;
            v214 = 128l * v211;
            int v215;
            v215 = v214 + v208;
            int4* v216;
            v216 = reinterpret_cast<int4*>(v206 + v215);
            int4* v217;
            v217 = reinterpret_cast<int4*>(v209 + v213);
            assert("Pointer alignment check" && (unsigned long long)(v216) % 4l == 0 && (unsigned long long)(v217) % 4l == 0);
            *v217 = *v216;
            v211 += 1l ;
        }
        int v218;
        v218 = 0l;
        while (while_method_1(v218)){
            int v220;
            v220 = 0l;
            while (while_method_2(v220)){
                bool v222;
                v222 = 0l <= v220;
                bool v224;
                if (v222){
                    bool v223;
                    v223 = v220 < 4l;
                    v224 = v223;
                } else {
                    v224 = false;
                }
                bool v225;
                v225 = v224 == false;
                if (v225){
                    assert("The indices should be inside the range of the dimension." && v224);
                } else {
                }
                bool v227;
                v227 = 0l <= v189;
                bool v229;
                if (v227){
                    bool v228;
                    v228 = v189 < 32l;
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
                int v232;
                v232 = v189 * 4l;
                int v233;
                v233 = v220 + v232;
                bool v234;
                v234 = 0l <= v218;
                bool v236;
                if (v234){
                    bool v235;
                    v235 = v218 < 2l;
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
                v239 = v218 * 128l;
                int v240;
                v240 = v233 + v239;
                assert("Tensor range check" && 0 <= v218 && v218 < 2l);
                assert("Tensor range check" && 0 <= v220 && v220 < 4l);
                int v241;
                v241 = 4l * v218;
                int v242;
                v242 = v241 + v220;
                v210[v242] = v240;
                v220 += 1l ;
            }
            v218 += 1l ;
        }
        int v243;
        v243 = 0l;
        while (while_method_1(v243)){
            assert("Tensor range check" && 0 <= v243 && v243 < 2l);
            int v245;
            v245 = 128l * v243;
            int v246;
            v246 = v245 + v208;
            assert("Tensor range check" && 0 <= v243 && v243 < 2l);
            int v247;
            v247 = 4l * v243;
            int4* v248;
            v248 = reinterpret_cast<int4*>(v209 + v247);
            int4* v249;
            v249 = reinterpret_cast<int4*>(v207 + v246);
            assert("Pointer alignment check" && (unsigned long long)(v248) % 4l == 0 && (unsigned long long)(v249) % 4l == 0);
            *v249 = *v248;
            v243 += 1l ;
        }
        assert("Tensor range check" && 0 <= v205 && v205 < 32l);
        /* void array set */;
        v194 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v250;
    v250 = threadIdx.x;
    assert("Tensor range check" && 0 <= v250 && v250 < 32l);
    /* void array index */;
    float * v251;
    v251 = v1+v9;
    float * v253;
    v253 = v7+v13;
    __shared__ float * v255[32l];
    __shared__ float * v256[32l];
    /* void shared array create v257 */;
    /* void shared array create v258 */;
    int v259;
    v259 = threadIdx.x;
    assert("Tensor range check" && 0 <= v259 && v259 < 32l);
    v255[v259] = v251;
    v256[v259] = v253;
    /* void array set */;
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
    int v269;
    v269 = 0l;
    while (while_method_0(v269)){
        bool v271;
        v271 = 0l <= v265;
        bool v272;
        v272 = v271 && v266;
        bool v273;
        v273 = v272 == false;
        if (v273){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v272);
        } else {
        }
        bool v275;
        v275 = 0l <= v269;
        bool v277;
        if (v275){
            bool v276;
            v276 = v269 < 32l;
            v277 = v276;
        } else {
            v277 = false;
        }
        bool v278;
        v278 = v277 == false;
        if (v278){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v277);
        } else {
        }
        int v280;
        v280 = v269 + v265;
        assert("Tensor range check" && 0 <= v269 && v269 < 32l);
        float * v281;
        v281 = v255[v280];
        float * v282;
        v282 = v256[v280];
        /* void array index */;
        assert("Tensor range check" && 0 <= v264 && v264 < 32l);
        int v283;
        v283 = 4l * v264;
        float v284[8l];
        int v285[8l];
        int v286;
        v286 = 0l;
        while (while_method_1(v286)){
            assert("Tensor range check" && 0 <= v286 && v286 < 2l);
            int v288;
            v288 = 4l * v286;
            assert("Tensor range check" && 0 <= v286 && v286 < 2l);
            int v289;
            v289 = 128l * v286;
            int v290;
            v290 = v289 + v283;
            int4* v291;
            v291 = reinterpret_cast<int4*>(v281 + v290);
            int4* v292;
            v292 = reinterpret_cast<int4*>(v284 + v288);
            assert("Pointer alignment check" && (unsigned long long)(v291) % 4l == 0 && (unsigned long long)(v292) % 4l == 0);
            *v292 = *v291;
            v286 += 1l ;
        }
        int v293;
        v293 = 0l;
        while (while_method_1(v293)){
            int v295;
            v295 = 0l;
            while (while_method_2(v295)){
                bool v297;
                v297 = 0l <= v295;
                bool v299;
                if (v297){
                    bool v298;
                    v298 = v295 < 4l;
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
                bool v302;
                v302 = 0l <= v264;
                bool v304;
                if (v302){
                    bool v303;
                    v303 = v264 < 32l;
                    v304 = v303;
                } else {
                    v304 = false;
                }
                bool v305;
                v305 = v304 == false;
                if (v305){
                    assert("The indices should be inside the range of the dimension." && v304);
                } else {
                }
                int v307;
                v307 = v264 * 4l;
                int v308;
                v308 = v295 + v307;
                bool v309;
                v309 = 0l <= v293;
                bool v311;
                if (v309){
                    bool v310;
                    v310 = v293 < 2l;
                    v311 = v310;
                } else {
                    v311 = false;
                }
                bool v312;
                v312 = v311 == false;
                if (v312){
                    assert("The indices should be inside the range of the dimension." && v311);
                } else {
                }
                int v314;
                v314 = v293 * 128l;
                int v315;
                v315 = v308 + v314;
                assert("Tensor range check" && 0 <= v293 && v293 < 2l);
                assert("Tensor range check" && 0 <= v295 && v295 < 4l);
                int v316;
                v316 = 4l * v293;
                int v317;
                v317 = v316 + v295;
                v285[v317] = v315;
                v295 += 1l ;
            }
            v293 += 1l ;
        }
        bool v318[8l];
        int v319;
        v319 = 0l;
        while (while_method_1(v319)){
            int v321;
            v321 = 0l;
            while (while_method_2(v321)){
                assert("Tensor range check" && 0 <= v319 && v319 < 2l);
                assert("Tensor range check" && 0 <= v321 && v321 < 4l);
                int v323;
                v323 = 4l * v319;
                int v324;
                v324 = v323 + v321;
                float v325;
                v325 = v284[v324];
                int v326;
                v326 = v285[v324];
                bool v327;
                v327 = v326 < 3l;
                assert("Tensor range check" && 0 <= v319 && v319 < 2l);
                assert("Tensor range check" && 0 <= v321 && v321 < 4l);
                v318[v324] = v327;
                v321 += 1l ;
            }
            v319 += 1l ;
        }
        float v328[8l];
        int v329;
        v329 = 0l;
        while (while_method_1(v329)){
            int v331;
            v331 = 0l;
            while (while_method_2(v331)){
                assert("Tensor range check" && 0 <= v329 && v329 < 2l);
                assert("Tensor range check" && 0 <= v331 && v331 < 4l);
                int v333;
                v333 = 4l * v329;
                int v334;
                v334 = v333 + v331;
                float v335;
                v335 = v284[v334];
                bool v336;
                v336 = v318[v334];
                float v339;
                if (v336){
                    bool v337;
                    v337 = 0.0f >= v335;
                    if (v337){
                        v339 = 0.0f;
                    } else {
                        v339 = v335;
                    }
                } else {
                    v339 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v329 && v329 < 2l);
                assert("Tensor range check" && 0 <= v331 && v331 < 4l);
                v328[v334] = v339;
                v331 += 1l ;
            }
            v329 += 1l ;
        }
        float v340;
        v340 = 0.0f;
        int v341;
        v341 = 0l;
        while (while_method_1(v341)){
            int v343;
            v343 = 0l;
            while (while_method_2(v343)){
                assert("Tensor range check" && 0 <= v341 && v341 < 2l);
                assert("Tensor range check" && 0 <= v343 && v343 < 4l);
                int v345;
                v345 = 4l * v341;
                int v346;
                v346 = v345 + v343;
                float v347;
                v347 = v328[v346];
                float v348;
                v348 = v340 + v347;
                v340 = v348;
                v343 += 1l ;
            }
            v341 += 1l ;
        }
        auto v349 = cooperative_groups::coalesced_threads();
        int v350;
        v350 = threadIdx.x;
        int v351;
        v351 = v350 / 32l;
        auto v352 = cooperative_groups::labeled_partition(v349,v351);
        Closure0 v353{};
        float v354;
        v354 = cooperative_groups::reduce(v352, v340, v353);
        int v355[8l];
        int v356;
        v356 = 0l;
        while (while_method_1(v356)){
            int v358;
            v358 = 0l;
            while (while_method_2(v358)){
                assert("Tensor range check" && 0 <= v356 && v356 < 2l);
                assert("Tensor range check" && 0 <= v358 && v358 < 4l);
                int v360;
                v360 = 4l * v356;
                int v361;
                v361 = v360 + v358;
                bool v362;
                v362 = v318[v361];
                int v363;
                if (v362){
                    v363 = 1l;
                } else {
                    v363 = 0l;
                }
                assert("Tensor range check" && 0 <= v356 && v356 < 2l);
                assert("Tensor range check" && 0 <= v358 && v358 < 4l);
                v355[v361] = v363;
                v358 += 1l ;
            }
            v356 += 1l ;
        }
        int v364;
        v364 = 0l;
        int v365;
        v365 = 0l;
        while (while_method_1(v365)){
            int v367;
            v367 = 0l;
            while (while_method_2(v367)){
                assert("Tensor range check" && 0 <= v365 && v365 < 2l);
                assert("Tensor range check" && 0 <= v367 && v367 < 4l);
                int v369;
                v369 = 4l * v365;
                int v370;
                v370 = v369 + v367;
                int v371;
                v371 = v355[v370];
                int v372;
                v372 = v364 + v371;
                v364 = v372;
                v367 += 1l ;
            }
            v365 += 1l ;
        }
        auto v373 = cooperative_groups::coalesced_threads();
        int v374;
        v374 = threadIdx.x;
        int v375;
        v375 = v374 / 32l;
        auto v376 = cooperative_groups::labeled_partition(v373,v375);
        Closure1 v377{};
        int v378;
        v378 = cooperative_groups::reduce(v376, v364, v377);
        float v379;
        v379 = (float)v378;
        float v380;
        v380 = 1.0f / v379;
        float v381[8l];
        int v382;
        v382 = 0l;
        while (while_method_1(v382)){
            int v384;
            v384 = 0l;
            while (while_method_2(v384)){
                assert("Tensor range check" && 0 <= v382 && v382 < 2l);
                assert("Tensor range check" && 0 <= v384 && v384 < 4l);
                int v386;
                v386 = 4l * v382;
                int v387;
                v387 = v386 + v384;
                float v388;
                v388 = v328[v387];
                bool v389;
                v389 = v318[v387];
                bool v390;
                v390 = v389 == false;
                float v395;
                if (v390){
                    v395 = 0.0f;
                } else {
                    bool v391;
                    v391 = v354 == 0.0f;
                    bool v392;
                    v392 = v391 != true;
                    if (v392){
                        float v393;
                        v393 = v388 / v354;
                        v395 = v393;
                    } else {
                        v395 = v380;
                    }
                }
                assert("Tensor range check" && 0 <= v382 && v382 < 2l);
                assert("Tensor range check" && 0 <= v384 && v384 < 4l);
                v381[v387] = v395;
                v384 += 1l ;
            }
            v382 += 1l ;
        }
        int v396;
        v396 = 0l;
        while (while_method_1(v396)){
            assert("Tensor range check" && 0 <= v396 && v396 < 2l);
            int v398;
            v398 = 128l * v396;
            int v399;
            v399 = v398 + v283;
            assert("Tensor range check" && 0 <= v396 && v396 < 2l);
            int v400;
            v400 = 4l * v396;
            int4* v401;
            v401 = reinterpret_cast<int4*>(v381 + v400);
            int4* v402;
            v402 = reinterpret_cast<int4*>(v282 + v399);
            assert("Pointer alignment check" && (unsigned long long)(v401) % 4l == 0 && (unsigned long long)(v402) % 4l == 0);
            *v402 = *v401;
            v396 += 1l ;
        }
        assert("Tensor range check" && 0 <= v280 && v280 < 32l);
        /* void array set */;
        v269 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v403;
    v403 = threadIdx.x;
    assert("Tensor range check" && 0 <= v403 && v403 < 32l);
    /* void array index */;
    float * v404;
    v404 = v1+v9;
    __shared__ float * v406[32l];
    /* void shared array create v407 */;
    __shared__ int v408[32l];
    int v409;
    v409 = threadIdx.x;
    assert("Tensor range check" && 0 <= v409 && v409 < 32l);
    v406[v409] = v404;
    /* void array set */;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v410;
    v410 = threadIdx.x;
    bool v411;
    v411 = 0l <= v410;
    bool v412;
    v412 = v411 == false;
    if (v412){
        assert("The index needs to be zero or positive." && v411);
    } else {
    }
    int v414;
    v414 = v410 % 32l;
    int v415;
    v415 = v410 / 32l;
    bool v416;
    v416 = v415 < 1l;
    bool v417;
    v417 = v416 == false;
    if (v417){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v416);
    } else {
    }
    assert("Tensor range check" && 0 <= v415 && v415 < 1l);
    int v419;
    v419 = 0l;
    while (while_method_0(v419)){
        bool v421;
        v421 = 0l <= v415;
        bool v422;
        v422 = v421 && v416;
        bool v423;
        v423 = v422 == false;
        if (v423){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v422);
        } else {
        }
        bool v425;
        v425 = 0l <= v419;
        bool v427;
        if (v425){
            bool v426;
            v426 = v419 < 32l;
            v427 = v426;
        } else {
            v427 = false;
        }
        bool v428;
        v428 = v427 == false;
        if (v428){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v427);
        } else {
        }
        int v430;
        v430 = v419 + v415;
        assert("Tensor range check" && 0 <= v419 && v419 < 32l);
        float * v431;
        v431 = v406[v430];
        /* void array index */;
        assert("Tensor range check" && 0 <= v414 && v414 < 32l);
        int v432;
        v432 = 4l * v414;
        float v433[8l];
        int v434[8l];
        int v435;
        v435 = 0l;
        while (while_method_1(v435)){
            assert("Tensor range check" && 0 <= v435 && v435 < 2l);
            int v437;
            v437 = 4l * v435;
            assert("Tensor range check" && 0 <= v435 && v435 < 2l);
            int v438;
            v438 = 128l * v435;
            int v439;
            v439 = v438 + v432;
            int4* v440;
            v440 = reinterpret_cast<int4*>(v431 + v439);
            int4* v441;
            v441 = reinterpret_cast<int4*>(v433 + v437);
            assert("Pointer alignment check" && (unsigned long long)(v440) % 4l == 0 && (unsigned long long)(v441) % 4l == 0);
            *v441 = *v440;
            v435 += 1l ;
        }
        int v442;
        v442 = 0l;
        while (while_method_1(v442)){
            int v444;
            v444 = 0l;
            while (while_method_2(v444)){
                bool v446;
                v446 = 0l <= v444;
                bool v448;
                if (v446){
                    bool v447;
                    v447 = v444 < 4l;
                    v448 = v447;
                } else {
                    v448 = false;
                }
                bool v449;
                v449 = v448 == false;
                if (v449){
                    assert("The indices should be inside the range of the dimension." && v448);
                } else {
                }
                bool v451;
                v451 = 0l <= v414;
                bool v453;
                if (v451){
                    bool v452;
                    v452 = v414 < 32l;
                    v453 = v452;
                } else {
                    v453 = false;
                }
                bool v454;
                v454 = v453 == false;
                if (v454){
                    assert("The indices should be inside the range of the dimension." && v453);
                } else {
                }
                int v456;
                v456 = v414 * 4l;
                int v457;
                v457 = v444 + v456;
                bool v458;
                v458 = 0l <= v442;
                bool v460;
                if (v458){
                    bool v459;
                    v459 = v442 < 2l;
                    v460 = v459;
                } else {
                    v460 = false;
                }
                bool v461;
                v461 = v460 == false;
                if (v461){
                    assert("The indices should be inside the range of the dimension." && v460);
                } else {
                }
                int v463;
                v463 = v442 * 128l;
                int v464;
                v464 = v457 + v463;
                assert("Tensor range check" && 0 <= v442 && v442 < 2l);
                assert("Tensor range check" && 0 <= v444 && v444 < 4l);
                int v465;
                v465 = 4l * v442;
                int v466;
                v466 = v465 + v444;
                v434[v466] = v464;
                v444 += 1l ;
            }
            v442 += 1l ;
        }
        unsigned long long v467;
        v467 = clock64();
        int v468;
        v468 = threadIdx.x;
        unsigned long long v469;
        v469 = (unsigned long long)v468;
        curandStatePhilox4_32_10_t v470;
        curand_init(v467,v469,0ull,&v470);
        bool v471[8l];
        int v472;
        v472 = 0l;
        while (while_method_1(v472)){
            int v474;
            v474 = 0l;
            while (while_method_2(v474)){
                assert("Tensor range check" && 0 <= v472 && v472 < 2l);
                assert("Tensor range check" && 0 <= v474 && v474 < 4l);
                int v476;
                v476 = 4l * v472;
                int v477;
                v477 = v476 + v474;
                float v478;
                v478 = v433[v477];
                int v479;
                v479 = v434[v477];
                bool v480;
                v480 = v479 < 3l;
                assert("Tensor range check" && 0 <= v472 && v472 < 2l);
                assert("Tensor range check" && 0 <= v474 && v474 < 4l);
                v471[v477] = v480;
                v474 += 1l ;
            }
            v472 += 1l ;
        }
        int v481[8l];
        int v482;
        v482 = 0l;
        while (while_method_1(v482)){
            int v484;
            v484 = 0l;
            while (while_method_2(v484)){
                assert("Tensor range check" && 0 <= v482 && v482 < 2l);
                assert("Tensor range check" && 0 <= v484 && v484 < 4l);
                int v486;
                v486 = 4l * v482;
                int v487;
                v487 = v486 + v484;
                bool v488;
                v488 = v471[v487];
                int v489;
                if (v488){
                    v489 = 1l;
                } else {
                    v489 = 0l;
                }
                assert("Tensor range check" && 0 <= v482 && v482 < 2l);
                assert("Tensor range check" && 0 <= v484 && v484 < 4l);
                v481[v487] = v489;
                v484 += 1l ;
            }
            v482 += 1l ;
        }
        int v490;
        v490 = 0l;
        int v491;
        v491 = 0l;
        while (while_method_1(v491)){
            int v493;
            v493 = 0l;
            while (while_method_2(v493)){
                assert("Tensor range check" && 0 <= v491 && v491 < 2l);
                assert("Tensor range check" && 0 <= v493 && v493 < 4l);
                int v495;
                v495 = 4l * v491;
                int v496;
                v496 = v495 + v493;
                int v497;
                v497 = v481[v496];
                int v498;
                v498 = v490 + v497;
                v490 = v498;
                v493 += 1l ;
            }
            v491 += 1l ;
        }
        auto v499 = cooperative_groups::coalesced_threads();
        int v500;
        v500 = threadIdx.x;
        int v501;
        v501 = v500 / 32l;
        auto v502 = cooperative_groups::labeled_partition(v499,v501);
        Closure1 v503{};
        int v504;
        v504 = cooperative_groups::reduce(v502, v490, v503);
        float v505[8l];
        int v506;
        v506 = 0l;
        while (while_method_1(v506)){
            int v508;
            v508 = 0l;
            while (while_method_2(v508)){
                assert("Tensor range check" && 0 <= v506 && v506 < 2l);
                assert("Tensor range check" && 0 <= v508 && v508 < 4l);
                int v510;
                v510 = 4l * v506;
                int v511;
                v511 = v510 + v508;
                float v512;
                v512 = v433[v511];
                bool v513;
                v513 = v471[v511];
                float v514;
                if (v513){
                    v514 = v512;
                } else {
                    v514 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v506 && v506 < 2l);
                assert("Tensor range check" && 0 <= v508 && v508 < 4l);
                v505[v511] = v514;
                v508 += 1l ;
            }
            v506 += 1l ;
        }
        float v515;
        v515 = 0.0f;
        int v516;
        v516 = 0l;
        while (while_method_1(v516)){
            int v518;
            v518 = 0l;
            while (while_method_2(v518)){
                assert("Tensor range check" && 0 <= v516 && v516 < 2l);
                assert("Tensor range check" && 0 <= v518 && v518 < 4l);
                int v520;
                v520 = 4l * v516;
                int v521;
                v521 = v520 + v518;
                float v522;
                v522 = v505[v521];
                float v523;
                v523 = v515 + v522;
                v515 = v523;
                v518 += 1l ;
            }
            v516 += 1l ;
        }
        auto v524 = cooperative_groups::coalesced_threads();
        int v525;
        v525 = threadIdx.x;
        int v526;
        v526 = v525 / 32l;
        auto v527 = cooperative_groups::labeled_partition(v524,v526);
        Closure0 v528{};
        float v529;
        v529 = cooperative_groups::reduce(v527, v515, v528);
        float v530;
        v530 = (float)v504;
        float v531;
        v531 = v529 / v530;
        float v532[8l];
        int v533;
        v533 = 0l;
        while (while_method_1(v533)){
            int v535;
            v535 = 0l;
            while (while_method_2(v535)){
                assert("Tensor range check" && 0 <= v533 && v533 < 2l);
                assert("Tensor range check" && 0 <= v535 && v535 < 4l);
                int v537;
                v537 = 4l * v533;
                int v538;
                v538 = v537 + v535;
                float v539;
                v539 = v433[v538];
                bool v540;
                v540 = v471[v538];
                float v541;
                if (v540){
                    v541 = v539;
                } else {
                    v541 = -1.0f / 0.0f;
                }
                float v542;
                v542 = v541 - v531;
                float v543;
                v543 = exp(v542);
                assert("Tensor range check" && 0 <= v533 && v533 < 2l);
                assert("Tensor range check" && 0 <= v535 && v535 < 4l);
                v532[v538] = v543;
                v535 += 1l ;
            }
            v533 += 1l ;
        }
        float v544;
        v544 = 0.0f;
        int v545;
        v545 = 0l;
        while (while_method_1(v545)){
            int v547;
            v547 = 0l;
            while (while_method_2(v547)){
                assert("Tensor range check" && 0 <= v545 && v545 < 2l);
                assert("Tensor range check" && 0 <= v547 && v547 < 4l);
                int v549;
                v549 = 4l * v545;
                int v550;
                v550 = v549 + v547;
                float v551;
                v551 = v532[v550];
                float v552;
                v552 = v544 + v551;
                v544 = v552;
                v547 += 1l ;
            }
            v545 += 1l ;
        }
        auto v553 = cooperative_groups::coalesced_threads();
        int v554;
        v554 = threadIdx.x;
        int v555;
        v555 = v554 / 32l;
        auto v556 = cooperative_groups::labeled_partition(v553,v555);
        float v557;
        v557 = cooperative_groups::reduce(v556, v544, v528);
        float v558[8l];
        int v559;
        v559 = 0l;
        while (while_method_1(v559)){
            int v561;
            v561 = 0l;
            while (while_method_2(v561)){
                assert("Tensor range check" && 0 <= v559 && v559 < 2l);
                assert("Tensor range check" && 0 <= v561 && v561 < 4l);
                int v563;
                v563 = 4l * v559;
                int v564;
                v564 = v563 + v561;
                float v565;
                v565 = v532[v564];
                float v566;
                v566 = v565 / v557;
                assert("Tensor range check" && 0 <= v559 && v559 < 2l);
                assert("Tensor range check" && 0 <= v561 && v561 < 4l);
                v558[v564] = v566;
                v561 += 1l ;
            }
            v559 += 1l ;
        }
        float v567[8l];
        float v568;
        v568 = 0.0f;
        int v569;
        v569 = 0l;
        while (while_method_1(v569)){
            assert("Tensor range check" && 0 <= v569 && v569 < 2l);
            int v571;
            v571 = 4l * v569;
            assert("Tensor range check" && 0 <= v569 && v569 < 2l);
            int v572; float v573;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v572 = tmp0.v0; v573 = tmp0.v1;
            while (while_method_2(v572)){
                assert("Tensor range check" && 0 <= v572 && v572 < 4l);
                int v575;
                v575 = v572 + v571;
                float v576;
                v576 = v558[v575];
                float v577;
                v577 = v573 + v576;
                v573 = v577;
                v572 += 1l ;
            }
            auto v578 = cooperative_groups::coalesced_threads();
            int v579;
            v579 = threadIdx.x;
            int v580;
            v580 = v579 / 32l;
            auto v581 = cooperative_groups::labeled_partition(v578,v580);
            Closure2 v582{};
            float v583;
            v583 = cooperative_groups::inclusive_scan(v581, v573, v582);
            float v584;
            v584 = v581.shfl_up(v583,1);
            bool v585;
            v585 = v581.thread_rank() == 0;
            float v586;
            if (v585){
                v586 = 0.0f;
            } else {
                v586 = v584;
            }
            float v587;
            v587 = v581.shfl(v583,v581.num_threads()-1);
            float v588;
            v588 = v568 + v586;
            int v589; float v590;
            Tuple0 tmp1 = Tuple0{0l, v588};
            v589 = tmp1.v0; v590 = tmp1.v1;
            while (while_method_2(v589)){
                assert("Tensor range check" && 0 <= v589 && v589 < 4l);
                int v592;
                v592 = v589 + v571;
                float v593;
                v593 = v558[v592];
                float v594;
                v594 = v590 + v593;
                assert("Tensor range check" && 0 <= v589 && v589 < 4l);
                v567[v592] = v594;
                v590 = v594;
                v589 += 1l ;
            }
            float v595;
            v595 = v568 + v587;
            v568 = v595;
            v569 += 1l ;
        }
        float v596[8l];
        bool v597[8l];
        int v598;
        v598 = 0l;
        while (while_method_1(v598)){
            int v600;
            v600 = 0l;
            while (while_method_2(v600)){
                assert("Tensor range check" && 0 <= v598 && v598 < 2l);
                assert("Tensor range check" && 0 <= v600 && v600 < 4l);
                int v602;
                v602 = 4l * v598;
                int v603;
                v603 = v602 + v600;
                float v604;
                v604 = v567[v603];
                float v605;
                v605 = v558[v603];
                bool v606;
                v606 = v605 > 0.0f;
                assert("Tensor range check" && 0 <= v598 && v598 < 2l);
                assert("Tensor range check" && 0 <= v600 && v600 < 4l);
                v596[v603] = v604;
                v597[v603] = v606;
                v600 += 1l ;
            }
            v598 += 1l ;
        }
        float v607; bool v608;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, false};
        v607 = tmp2.v0; v608 = tmp2.v1;
        int v609;
        v609 = 0l;
        while (while_method_1(v609)){
            int v611;
            v611 = 0l;
            while (while_method_2(v611)){
                assert("Tensor range check" && 0 <= v609 && v609 < 2l);
                assert("Tensor range check" && 0 <= v611 && v611 < 4l);
                int v613;
                v613 = 4l * v609;
                int v614;
                v614 = v613 + v611;
                float v615;
                v615 = v596[v614];
                bool v616;
                v616 = v597[v614];
                float v623; bool v624;
                if (v608){
                    if (v616){
                        bool v617;
                        v617 = v607 >= v615;
                        float v618;
                        if (v617){
                            v618 = v607;
                        } else {
                            v618 = v615;
                        }
                        v623 = v618; v624 = true;
                    } else {
                        v623 = v607; v624 = v608;
                    }
                } else {
                    if (v616){
                        v623 = v615; v624 = v616;
                    } else {
                        v623 = v607; v624 = v608;
                    }
                }
                v607 = v623;
                v608 = v624;
                v611 += 1l ;
            }
            v609 += 1l ;
        }
        auto v625 = cooperative_groups::coalesced_threads();
        int v626;
        v626 = threadIdx.x;
        int v627;
        v627 = v626 / 32l;
        auto v628 = cooperative_groups::labeled_partition(v625,v627);
        Closure3 v629{};
        float v630; bool v631;
        Tuple1 tmp3 = cooperative_groups::reduce(v628, Tuple1{v607, v608}, v629);
        v630 = tmp3.v0; v631 = tmp3.v1;
        bool v632;
        v632 = v631 == false;
        if (v632){
            assert("The local reduce must be true." && v631);
        } else {
        }
        float v634[8l];
        int v635[8l];
        int v636;
        v636 = 0l;
        while (while_method_1(v636)){
            int v638;
            v638 = 0l;
            while (while_method_2(v638)){
                assert("Tensor range check" && 0 <= v636 && v636 < 2l);
                assert("Tensor range check" && 0 <= v638 && v638 < 4l);
                int v640;
                v640 = 4l * v636;
                int v641;
                v641 = v640 + v638;
                int v642;
                v642 = v434[v641];
                float v643;
                v643 = curand_uniform(&v470);
                assert("Tensor range check" && 0 <= v636 && v636 < 2l);
                assert("Tensor range check" && 0 <= v638 && v638 < 4l);
                v634[v641] = v643;
                v635[v641] = v642;
                v638 += 1l ;
            }
            v636 += 1l ;
        }
        float v644; int v645;
        Tuple2 tmp4 = Tuple2{0.0f, 2147483647l};
        v644 = tmp4.v0; v645 = tmp4.v1;
        int v646;
        v646 = 0l;
        while (while_method_1(v646)){
            int v648;
            v648 = 0l;
            while (while_method_2(v648)){
                assert("Tensor range check" && 0 <= v646 && v646 < 2l);
                assert("Tensor range check" && 0 <= v648 && v648 < 4l);
                int v650;
                v650 = 4l * v646;
                int v651;
                v651 = v650 + v648;
                float v652;
                v652 = v634[v651];
                int v653;
                v653 = v635[v651];
                bool v654;
                v654 = v645 < v653;
                float v655; int v656;
                if (v654){
                    v655 = v644; v656 = v645;
                } else {
                    v655 = v652; v656 = v653;
                }
                v644 = v655;
                v645 = v656;
                v648 += 1l ;
            }
            v646 += 1l ;
        }
        auto v657 = cooperative_groups::coalesced_threads();
        int v658;
        v658 = threadIdx.x;
        int v659;
        v659 = v658 / 32l;
        auto v660 = cooperative_groups::labeled_partition(v657,v659);
        Closure4 v661{};
        float v662; int v663;
        Tuple2 tmp5 = cooperative_groups::reduce(v660, Tuple2{v644, v645}, v661);
        v662 = tmp5.v0; v663 = tmp5.v1;
        float v664;
        v664 = v630 * v662;
        int v665[8l];
        bool v666[8l];
        int v667;
        v667 = 0l;
        while (while_method_1(v667)){
            int v669;
            v669 = 0l;
            while (while_method_2(v669)){
                assert("Tensor range check" && 0 <= v667 && v667 < 2l);
                assert("Tensor range check" && 0 <= v669 && v669 < 4l);
                int v671;
                v671 = 4l * v667;
                int v672;
                v672 = v671 + v669;
                float v673;
                v673 = v596[v672];
                bool v674;
                v674 = v597[v672];
                int v675;
                v675 = v434[v672];
                int v678; bool v679;
                if (v674){
                    float v676;
                    v676 = v673 - v664;
                    bool v677;
                    v677 = v676 >= 0.0f;
                    v678 = v675; v679 = v677;
                } else {
                    v678 = 2147483647l; v679 = false;
                }
                assert("Tensor range check" && 0 <= v667 && v667 < 2l);
                assert("Tensor range check" && 0 <= v669 && v669 < 4l);
                v665[v672] = v678;
                v666[v672] = v679;
                v669 += 1l ;
            }
            v667 += 1l ;
        }
        int v680; bool v681;
        Tuple3 tmp6 = Tuple3{2147483647l, false};
        v680 = tmp6.v0; v681 = tmp6.v1;
        int v682;
        v682 = 0l;
        while (while_method_1(v682)){
            int v684;
            v684 = 0l;
            while (while_method_2(v684)){
                assert("Tensor range check" && 0 <= v682 && v682 < 2l);
                assert("Tensor range check" && 0 <= v684 && v684 < 4l);
                int v686;
                v686 = 4l * v682;
                int v687;
                v687 = v686 + v684;
                int v688;
                v688 = v665[v687];
                bool v689;
                v689 = v666[v687];
                int v696; bool v697;
                if (v681){
                    if (v689){
                        bool v690;
                        v690 = v680 < v688;
                        int v691;
                        if (v690){
                            v691 = v680;
                        } else {
                            v691 = v688;
                        }
                        v696 = v691; v697 = true;
                    } else {
                        v696 = v680; v697 = v681;
                    }
                } else {
                    if (v689){
                        v696 = v688; v697 = v689;
                    } else {
                        v696 = v680; v697 = v681;
                    }
                }
                v680 = v696;
                v681 = v697;
                v684 += 1l ;
            }
            v682 += 1l ;
        }
        auto v698 = cooperative_groups::coalesced_threads();
        int v699;
        v699 = threadIdx.x;
        int v700;
        v700 = v699 / 32l;
        auto v701 = cooperative_groups::labeled_partition(v698,v700);
        Closure5 v702{};
        int v703; bool v704;
        Tuple3 tmp7 = cooperative_groups::reduce(v701, Tuple3{v680, v681}, v702);
        v703 = tmp7.v0; v704 = tmp7.v1;
        bool v705;
        v705 = v704 == false;
        if (v705){
            assert("The local reduce must be true." && v704);
        } else {
        }
        int v707;
        v707 = 0l;
        while (while_method_1(v707)){
            assert("Tensor range check" && 0 <= v707 && v707 < 2l);
            assert("Tensor range check" && 0 <= v707 && v707 < 2l);
            v707 += 1l ;
        }
        assert("Tensor range check" && 0 <= v430 && v430 < 32l);
        v408[v430] = v703;
        v419 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v709;
    v709 = threadIdx.x;
    assert("Tensor range check" && 0 <= v709 && v709 < 32l);
    int v710;
    v710 = v408[v709];
    int v711;
    v711 = threadIdx.x;
    assert("Tensor range check" && 0 <= v711 && v711 < 32l);
    v5[v711] = v710;
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
