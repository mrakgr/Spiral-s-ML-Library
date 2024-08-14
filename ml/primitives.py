kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <cooperative_groups/reduce.h>
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
struct Tuple4;
struct Tuple0 {
    int * v0;
    int * v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(int * t0, int * t1) : v0(t0), v1(t1) {}
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
    int v0;
    float v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure1 {
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
struct Closure2 {
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
struct Closure3 {
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
struct Closure4 {
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
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    unsigned long long v8;
    v8 = clock64();
    int v9;
    v9 = threadIdx.x;
    unsigned long long v10;
    v10 = (unsigned long long)v9;
    curandStatePhilox4_32_10_t v11;
    curand_init(v8,v10,0ull,&v11);
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
    int v18;
    v18 = threadIdx.x;
    assert("Tensor range check" && 0 <= v18 && v18 < 32l);
    int v19;
    v19 = 16l * v18;
    int v20;
    v20 = threadIdx.x;
    assert("Tensor range check" && 0 <= v20 && v20 < 32l);
    int v21;
    v21 = 16l * v20;
    __shared__ float * v22[32l];
    __shared__ Tuple0 v23[32l];
    int v24;
    v24 = threadIdx.x;
    float * v25;
    v25 = v1+v13;
    int * v27;
    v27 = v2+v17;
    int * v29;
    v29 = v3+v17;
    assert("Tensor range check" && 0 <= v24 && v24 < 32l);
    v22[v24] = v25;
    v23[v24] = Tuple0{v27, v29};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v31;
    v31 = threadIdx.x;
    bool v32;
    v32 = 0l <= v31;
    bool v33;
    v33 = v32 == false;
    if (v33){
        assert("The index needs to be zero or positive." && v32);
    } else {
    }
    int v35;
    v35 = v31 % 4l;
    int v36;
    v36 = v31 / 4l;
    bool v37;
    v37 = v36 < 8l;
    bool v38;
    v38 = v37 == false;
    if (v38){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v37);
    } else {
    }
    assert("Tensor range check" && 0 <= v36 && v36 < 8l);
    int v40;
    v40 = 4l * v36;
    int v41;
    v41 = 0l;
    while (while_method_0(v41)){
        assert("Tensor range check" && 0 <= v41 && v41 < 4l);
        int v43;
        v43 = v41 + v40;
        float * v44;
        v44 = v22[v43];
        int * v45; int * v46;
        Tuple0 tmp0 = v23[v43];
        v45 = tmp0.v0; v46 = tmp0.v1;
        assert("Tensor range check" && 0 <= v35 && v35 < 4l);
        int v47;
        v47 = 4l * v35;
        assert("Tensor range check" && 0 <= v35 && v35 < 4l);
        float v48[4l];
        int v49[4l];
        int v50;
        v50 = 0l;
        while (while_method_1(v50)){
            assert("Tensor range check" && 0 <= v50 && v50 < 1l);
            int v52;
            v52 = 4l * v50;
            assert("Tensor range check" && 0 <= v50 && v50 < 1l);
            int v53;
            v53 = v52 + v47;
            int4* v54;
            v54 = reinterpret_cast<int4*>(v44 + v53);
            int4* v55;
            v55 = reinterpret_cast<int4*>(v48 + v52);
            assert("Pointer alignment check" && (unsigned long long)(v54) % 4l == 0 && (unsigned long long)(v55) % 4l == 0);
            *v55 = *v54;
            v50 += 1l ;
        }
        int v56;
        v56 = 0l;
        while (while_method_1(v56)){
            int v58;
            v58 = 0l;
            while (while_method_0(v58)){
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
                v65 = 0l <= v56;
                bool v67;
                if (v65){
                    bool v66;
                    v66 = v56 < 1l;
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
                v70 = v56 * 4l;
                int v71;
                v71 = v58 + v70;
                bool v72;
                v72 = 0l <= v35;
                bool v74;
                if (v72){
                    bool v73;
                    v73 = v35 < 4l;
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
                v77 = v35 * 4l;
                int v78;
                v78 = v71 + v77;
                assert("Tensor range check" && 0 <= v56 && v56 < 1l);
                assert("Tensor range check" && 0 <= v58 && v58 < 4l);
                int v79;
                v79 = 4l * v56;
                int v80;
                v80 = v79 + v58;
                v49[v80] = v78;
                v58 += 1l ;
            }
            v56 += 1l ;
        }
        bool v81;
        v81 = 0l <= v41;
        bool v83;
        if (v81){
            bool v82;
            v82 = v41 < 4l;
            v83 = v82;
        } else {
            v83 = false;
        }
        bool v84;
        v84 = v83 == false;
        if (v84){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v83);
        } else {
        }
        bool v86;
        v86 = 0l <= v36;
        bool v87;
        v87 = v86 && v37;
        bool v88;
        v88 = v87 == false;
        if (v88){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v87);
        } else {
        }
        int v90;
        v90 = v36 * 4l;
        int v91;
        v91 = v90 + v41;
        int v92[4l];
        int v93[4l];
        int v94;
        v94 = 0l;
        while (while_method_1(v94)){
            int v96;
            v96 = 0l;
            while (while_method_0(v96)){
                assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                int v98;
                v98 = 4l * v94;
                int v99;
                v99 = v98 + v96;
                int v100;
                v100 = v49[v99];
                assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                v92[v99] = v91;
                v93[v99] = v100;
                v96 += 1l ;
            }
            v94 += 1l ;
        }
        int v101;
        v101 = 0l;
        while (while_method_1(v101)){
            assert("Tensor range check" && 0 <= v101 && v101 < 1l);
            int v103;
            v103 = 4l * v101;
            int v104;
            v104 = v103 + v47;
            assert("Tensor range check" && 0 <= v101 && v101 < 1l);
            int4* v105;
            v105 = reinterpret_cast<int4*>(v92 + v103);
            int4* v106;
            v106 = reinterpret_cast<int4*>(v45 + v104);
            assert("Pointer alignment check" && (unsigned long long)(v105) % 4l == 0 && (unsigned long long)(v106) % 4l == 0);
            *v106 = *v105;
            int4* v107;
            v107 = reinterpret_cast<int4*>(v93 + v103);
            int4* v108;
            v108 = reinterpret_cast<int4*>(v46 + v104);
            assert("Pointer alignment check" && (unsigned long long)(v107) % 4l == 0 && (unsigned long long)(v108) % 4l == 0);
            *v108 = *v107;
            v101 += 1l ;
        }
        v41 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    __shared__ float * v109[32l];
    __shared__ int v110[32l];
    int v111;
    v111 = threadIdx.x;
    float * v112;
    v112 = v1+v13;
    assert("Tensor range check" && 0 <= v111 && v111 < 32l);
    v109[v111] = v112;
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
    v118 = v114 % 4l;
    int v119;
    v119 = v114 / 4l;
    bool v120;
    v120 = v119 < 8l;
    bool v121;
    v121 = v120 == false;
    if (v121){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v120);
    } else {
    }
    assert("Tensor range check" && 0 <= v119 && v119 < 8l);
    int v123;
    v123 = 4l * v119;
    int v124;
    v124 = 0l;
    while (while_method_0(v124)){
        assert("Tensor range check" && 0 <= v124 && v124 < 4l);
        int v126;
        v126 = v124 + v123;
        float * v127;
        v127 = v109[v126];
        assert("Tensor range check" && 0 <= v118 && v118 < 4l);
        int v128;
        v128 = 4l * v118;
        float v129[4l];
        int v130[4l];
        int v131;
        v131 = 0l;
        while (while_method_1(v131)){
            assert("Tensor range check" && 0 <= v131 && v131 < 1l);
            int v133;
            v133 = 4l * v131;
            assert("Tensor range check" && 0 <= v131 && v131 < 1l);
            int v134;
            v134 = v133 + v128;
            int4* v135;
            v135 = reinterpret_cast<int4*>(v127 + v134);
            int4* v136;
            v136 = reinterpret_cast<int4*>(v129 + v133);
            assert("Pointer alignment check" && (unsigned long long)(v135) % 4l == 0 && (unsigned long long)(v136) % 4l == 0);
            *v136 = *v135;
            v131 += 1l ;
        }
        int v137;
        v137 = 0l;
        while (while_method_1(v137)){
            int v139;
            v139 = 0l;
            while (while_method_0(v139)){
                bool v141;
                v141 = 0l <= v139;
                bool v143;
                if (v141){
                    bool v142;
                    v142 = v139 < 4l;
                    v143 = v142;
                } else {
                    v143 = false;
                }
                bool v144;
                v144 = v143 == false;
                if (v144){
                    assert("The indices should be inside the range of the dimension." && v143);
                } else {
                }
                bool v146;
                v146 = 0l <= v137;
                bool v148;
                if (v146){
                    bool v147;
                    v147 = v137 < 1l;
                    v148 = v147;
                } else {
                    v148 = false;
                }
                bool v149;
                v149 = v148 == false;
                if (v149){
                    assert("The indices should be inside the range of the dimension." && v148);
                } else {
                }
                int v151;
                v151 = v137 * 4l;
                int v152;
                v152 = v139 + v151;
                bool v153;
                v153 = 0l <= v118;
                bool v155;
                if (v153){
                    bool v154;
                    v154 = v118 < 4l;
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
                int v158;
                v158 = v118 * 4l;
                int v159;
                v159 = v152 + v158;
                assert("Tensor range check" && 0 <= v137 && v137 < 1l);
                assert("Tensor range check" && 0 <= v139 && v139 < 4l);
                int v160;
                v160 = 4l * v137;
                int v161;
                v161 = v160 + v139;
                v130[v161] = v159;
                v139 += 1l ;
            }
            v137 += 1l ;
        }
        bool v162;
        v162 = 0l <= v124;
        bool v164;
        if (v162){
            bool v163;
            v163 = v124 < 4l;
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
        bool v167;
        v167 = 0l <= v119;
        bool v168;
        v168 = v167 && v120;
        bool v169;
        v169 = v168 == false;
        if (v169){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v168);
        } else {
        }
        int v171;
        v171 = v119 * 4l;
        int v172;
        v172 = v171 + v124;
        assert("Tensor range check" && 0 <= v124 && v124 < 4l);
        v110[v126] = v172;
        v124 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v173;
    v173 = threadIdx.x;
    assert("Tensor range check" && 0 <= v173 && v173 < 32l);
    int v174;
    v174 = v110[v173];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v175;
    v175 = threadIdx.x;
    assert("Tensor range check" && 0 <= v175 && v175 < 32l);
    v4[v175] = v174;
    __shared__ float * v176[32l];
    __shared__ float * v177[32l];
    int v178;
    v178 = threadIdx.x;
    float * v179;
    v179 = v1+v13;
    float * v181;
    v181 = v6+v19;
    assert("Tensor range check" && 0 <= v178 && v178 < 32l);
    v176[v178] = v179;
    v177[v178] = v181;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v183;
    v183 = threadIdx.x;
    bool v184;
    v184 = 0l <= v183;
    bool v185;
    v185 = v184 == false;
    if (v185){
        assert("The index needs to be zero or positive." && v184);
    } else {
    }
    int v187;
    v187 = v183 % 4l;
    int v188;
    v188 = v183 / 4l;
    bool v189;
    v189 = v188 < 8l;
    bool v190;
    v190 = v189 == false;
    if (v190){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v189);
    } else {
    }
    assert("Tensor range check" && 0 <= v188 && v188 < 8l);
    int v192;
    v192 = 4l * v188;
    int v193;
    v193 = 0l;
    while (while_method_0(v193)){
        assert("Tensor range check" && 0 <= v193 && v193 < 4l);
        int v195;
        v195 = v193 + v192;
        float * v196;
        v196 = v176[v195];
        float * v197;
        v197 = v177[v195];
        assert("Tensor range check" && 0 <= v187 && v187 < 4l);
        int v198;
        v198 = 4l * v187;
        assert("Tensor range check" && 0 <= v187 && v187 < 4l);
        float v199[4l];
        int v200[4l];
        int v201;
        v201 = 0l;
        while (while_method_1(v201)){
            assert("Tensor range check" && 0 <= v201 && v201 < 1l);
            int v203;
            v203 = 4l * v201;
            assert("Tensor range check" && 0 <= v201 && v201 < 1l);
            int v204;
            v204 = v203 + v198;
            int4* v205;
            v205 = reinterpret_cast<int4*>(v196 + v204);
            int4* v206;
            v206 = reinterpret_cast<int4*>(v199 + v203);
            assert("Pointer alignment check" && (unsigned long long)(v205) % 4l == 0 && (unsigned long long)(v206) % 4l == 0);
            *v206 = *v205;
            v201 += 1l ;
        }
        int v207;
        v207 = 0l;
        while (while_method_1(v207)){
            int v209;
            v209 = 0l;
            while (while_method_0(v209)){
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
                v216 = 0l <= v207;
                bool v218;
                if (v216){
                    bool v217;
                    v217 = v207 < 1l;
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
                v221 = v207 * 4l;
                int v222;
                v222 = v209 + v221;
                bool v223;
                v223 = 0l <= v187;
                bool v225;
                if (v223){
                    bool v224;
                    v224 = v187 < 4l;
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
                v228 = v187 * 4l;
                int v229;
                v229 = v222 + v228;
                assert("Tensor range check" && 0 <= v207 && v207 < 1l);
                assert("Tensor range check" && 0 <= v209 && v209 < 4l);
                int v230;
                v230 = 4l * v207;
                int v231;
                v231 = v230 + v209;
                v200[v231] = v229;
                v209 += 1l ;
            }
            v207 += 1l ;
        }
        bool v232;
        v232 = 0l <= v193;
        bool v234;
        if (v232){
            bool v233;
            v233 = v193 < 4l;
            v234 = v233;
        } else {
            v234 = false;
        }
        bool v235;
        v235 = v234 == false;
        if (v235){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v234);
        } else {
        }
        bool v237;
        v237 = 0l <= v188;
        bool v238;
        v238 = v237 && v189;
        bool v239;
        v239 = v238 == false;
        if (v239){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v238);
        } else {
        }
        int v241;
        v241 = v188 * 4l;
        int v242;
        v242 = v241 + v193;
        int v243;
        v243 = 0l;
        while (while_method_1(v243)){
            assert("Tensor range check" && 0 <= v243 && v243 < 1l);
            int v245;
            v245 = 4l * v243;
            int v246;
            v246 = v245 + v198;
            assert("Tensor range check" && 0 <= v243 && v243 < 1l);
            int4* v247;
            v247 = reinterpret_cast<int4*>(v199 + v245);
            int4* v248;
            v248 = reinterpret_cast<int4*>(v197 + v246);
            assert("Pointer alignment check" && (unsigned long long)(v247) % 4l == 0 && (unsigned long long)(v248) % 4l == 0);
            *v248 = *v247;
            v243 += 1l ;
        }
        v193 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    __shared__ float * v249[32l];
    __shared__ float * v250[32l];
    int v251;
    v251 = threadIdx.x;
    float * v252;
    v252 = v1+v13;
    float * v254;
    v254 = v7+v21;
    assert("Tensor range check" && 0 <= v251 && v251 < 32l);
    v249[v251] = v252;
    v250[v251] = v254;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v256;
    v256 = threadIdx.x;
    bool v257;
    v257 = 0l <= v256;
    bool v258;
    v258 = v257 == false;
    if (v258){
        assert("The index needs to be zero or positive." && v257);
    } else {
    }
    int v260;
    v260 = v256 % 4l;
    int v261;
    v261 = v256 / 4l;
    bool v262;
    v262 = v261 < 8l;
    bool v263;
    v263 = v262 == false;
    if (v263){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v262);
    } else {
    }
    assert("Tensor range check" && 0 <= v261 && v261 < 8l);
    int v265;
    v265 = 4l * v261;
    int v266;
    v266 = 0l;
    while (while_method_0(v266)){
        assert("Tensor range check" && 0 <= v266 && v266 < 4l);
        int v268;
        v268 = v266 + v265;
        float * v269;
        v269 = v249[v268];
        float * v270;
        v270 = v250[v268];
        assert("Tensor range check" && 0 <= v260 && v260 < 4l);
        int v271;
        v271 = 4l * v260;
        assert("Tensor range check" && 0 <= v260 && v260 < 4l);
        float v272[4l];
        int v273[4l];
        int v274;
        v274 = 0l;
        while (while_method_1(v274)){
            assert("Tensor range check" && 0 <= v274 && v274 < 1l);
            int v276;
            v276 = 4l * v274;
            assert("Tensor range check" && 0 <= v274 && v274 < 1l);
            int v277;
            v277 = v276 + v271;
            int4* v278;
            v278 = reinterpret_cast<int4*>(v269 + v277);
            int4* v279;
            v279 = reinterpret_cast<int4*>(v272 + v276);
            assert("Pointer alignment check" && (unsigned long long)(v278) % 4l == 0 && (unsigned long long)(v279) % 4l == 0);
            *v279 = *v278;
            v274 += 1l ;
        }
        int v280;
        v280 = 0l;
        while (while_method_1(v280)){
            int v282;
            v282 = 0l;
            while (while_method_0(v282)){
                bool v284;
                v284 = 0l <= v282;
                bool v286;
                if (v284){
                    bool v285;
                    v285 = v282 < 4l;
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
                bool v289;
                v289 = 0l <= v280;
                bool v291;
                if (v289){
                    bool v290;
                    v290 = v280 < 1l;
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
                v294 = v280 * 4l;
                int v295;
                v295 = v282 + v294;
                bool v296;
                v296 = 0l <= v260;
                bool v298;
                if (v296){
                    bool v297;
                    v297 = v260 < 4l;
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
                v301 = v260 * 4l;
                int v302;
                v302 = v295 + v301;
                assert("Tensor range check" && 0 <= v280 && v280 < 1l);
                assert("Tensor range check" && 0 <= v282 && v282 < 4l);
                int v303;
                v303 = 4l * v280;
                int v304;
                v304 = v303 + v282;
                v273[v304] = v302;
                v282 += 1l ;
            }
            v280 += 1l ;
        }
        bool v305;
        v305 = 0l <= v266;
        bool v307;
        if (v305){
            bool v306;
            v306 = v266 < 4l;
            v307 = v306;
        } else {
            v307 = false;
        }
        bool v308;
        v308 = v307 == false;
        if (v308){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v307);
        } else {
        }
        bool v310;
        v310 = 0l <= v261;
        bool v311;
        v311 = v310 && v262;
        bool v312;
        v312 = v311 == false;
        if (v312){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v311);
        } else {
        }
        int v314;
        v314 = v261 * 4l;
        int v315;
        v315 = v314 + v266;
        float v316;
        v316 = 0.0f;
        int v317;
        v317 = 0l;
        while (while_method_1(v317)){
            int v319;
            v319 = 0l;
            while (while_method_0(v319)){
                assert("Tensor range check" && 0 <= v317 && v317 < 1l);
                assert("Tensor range check" && 0 <= v319 && v319 < 4l);
                int v321;
                v321 = 4l * v317;
                int v322;
                v322 = v321 + v319;
                float v323;
                v323 = v272[v322];
                float v324;
                v324 = v316 + v323;
                v316 = v324;
                v319 += 1l ;
            }
            v317 += 1l ;
        }
        auto v325 = cooperative_groups::coalesced_threads();
        int v326;
        v326 = threadIdx.x;
        int v327;
        v327 = v326 / 4l;
        auto v328 = cooperative_groups::labeled_partition(v325,v327);
        Closure0 v329{};
        float v330;
        v330 = cooperative_groups::reduce(v328, v316, v329);
        float v331;
        v331 = v330 / 16.0f;
        float v332[4l];
        int v333;
        v333 = 0l;
        while (while_method_1(v333)){
            int v335;
            v335 = 0l;
            while (while_method_0(v335)){
                assert("Tensor range check" && 0 <= v333 && v333 < 1l);
                assert("Tensor range check" && 0 <= v335 && v335 < 4l);
                int v337;
                v337 = 4l * v333;
                int v338;
                v338 = v337 + v335;
                float v339;
                v339 = v272[v338];
                float v340;
                v340 = v339 - v331;
                float v341;
                v341 = exp(v340);
                assert("Tensor range check" && 0 <= v333 && v333 < 1l);
                assert("Tensor range check" && 0 <= v335 && v335 < 4l);
                v332[v338] = v341;
                v335 += 1l ;
            }
            v333 += 1l ;
        }
        float v342;
        v342 = 0.0f;
        int v343;
        v343 = 0l;
        while (while_method_1(v343)){
            int v345;
            v345 = 0l;
            while (while_method_0(v345)){
                assert("Tensor range check" && 0 <= v343 && v343 < 1l);
                assert("Tensor range check" && 0 <= v345 && v345 < 4l);
                int v347;
                v347 = 4l * v343;
                int v348;
                v348 = v347 + v345;
                float v349;
                v349 = v332[v348];
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
        v353 = v352 / 4l;
        auto v354 = cooperative_groups::labeled_partition(v351,v353);
        float v355;
        v355 = cooperative_groups::reduce(v354, v342, v329);
        float v356[4l];
        int v357;
        v357 = 0l;
        while (while_method_1(v357)){
            int v359;
            v359 = 0l;
            while (while_method_0(v359)){
                assert("Tensor range check" && 0 <= v357 && v357 < 1l);
                assert("Tensor range check" && 0 <= v359 && v359 < 4l);
                int v361;
                v361 = 4l * v357;
                int v362;
                v362 = v361 + v359;
                float v363;
                v363 = v332[v362];
                bool v364;
                v364 = v355 == 0.0f;
                bool v365;
                v365 = v364 != true;
                float v367;
                if (v365){
                    float v366;
                    v366 = v363 / v355;
                    v367 = v366;
                } else {
                    v367 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v357 && v357 < 1l);
                assert("Tensor range check" && 0 <= v359 && v359 < 4l);
                v356[v362] = v367;
                v359 += 1l ;
            }
            v357 += 1l ;
        }
        int v368;
        v368 = 0l;
        while (while_method_1(v368)){
            assert("Tensor range check" && 0 <= v368 && v368 < 1l);
            int v370;
            v370 = 4l * v368;
            int v371;
            v371 = v370 + v271;
            assert("Tensor range check" && 0 <= v368 && v368 < 1l);
            int4* v372;
            v372 = reinterpret_cast<int4*>(v356 + v370);
            int4* v373;
            v373 = reinterpret_cast<int4*>(v270 + v371);
            assert("Pointer alignment check" && (unsigned long long)(v372) % 4l == 0 && (unsigned long long)(v373) % 4l == 0);
            *v373 = *v372;
            v368 += 1l ;
        }
        v266 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    __shared__ float * v374[32l];
    __shared__ int v375[32l];
    int v376;
    v376 = threadIdx.x;
    float * v377;
    v377 = v0+v15;
    assert("Tensor range check" && 0 <= v376 && v376 < 32l);
    v374[v376] = v377;
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
    v383 = v379 % 4l;
    int v384;
    v384 = v379 / 4l;
    bool v385;
    v385 = v384 < 8l;
    bool v386;
    v386 = v385 == false;
    if (v386){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v385);
    } else {
    }
    assert("Tensor range check" && 0 <= v384 && v384 < 8l);
    int v388;
    v388 = 4l * v384;
    int v389;
    v389 = 0l;
    while (while_method_0(v389)){
        assert("Tensor range check" && 0 <= v389 && v389 < 4l);
        int v391;
        v391 = v389 + v388;
        float * v392;
        v392 = v374[v391];
        assert("Tensor range check" && 0 <= v383 && v383 < 4l);
        int v393;
        v393 = 4l * v383;
        float v394[4l];
        int v395[4l];
        int v396;
        v396 = 0l;
        while (while_method_1(v396)){
            assert("Tensor range check" && 0 <= v396 && v396 < 1l);
            int v398;
            v398 = 4l * v396;
            assert("Tensor range check" && 0 <= v396 && v396 < 1l);
            int v399;
            v399 = v398 + v393;
            int4* v400;
            v400 = reinterpret_cast<int4*>(v392 + v399);
            int4* v401;
            v401 = reinterpret_cast<int4*>(v394 + v398);
            assert("Pointer alignment check" && (unsigned long long)(v400) % 4l == 0 && (unsigned long long)(v401) % 4l == 0);
            *v401 = *v400;
            v396 += 1l ;
        }
        int v402;
        v402 = 0l;
        while (while_method_1(v402)){
            int v404;
            v404 = 0l;
            while (while_method_0(v404)){
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
                v411 = 0l <= v402;
                bool v413;
                if (v411){
                    bool v412;
                    v412 = v402 < 1l;
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
                v416 = v402 * 4l;
                int v417;
                v417 = v404 + v416;
                bool v418;
                v418 = 0l <= v383;
                bool v420;
                if (v418){
                    bool v419;
                    v419 = v383 < 4l;
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
                v423 = v383 * 4l;
                int v424;
                v424 = v417 + v423;
                assert("Tensor range check" && 0 <= v402 && v402 < 1l);
                assert("Tensor range check" && 0 <= v404 && v404 < 4l);
                int v425;
                v425 = 4l * v402;
                int v426;
                v426 = v425 + v404;
                v395[v426] = v424;
                v404 += 1l ;
            }
            v402 += 1l ;
        }
        bool v427;
        v427 = 0l <= v389;
        bool v429;
        if (v427){
            bool v428;
            v428 = v389 < 4l;
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
        bool v432;
        v432 = 0l <= v384;
        bool v433;
        v433 = v432 && v385;
        bool v434;
        v434 = v433 == false;
        if (v434){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v433);
        } else {
        }
        int v436;
        v436 = v384 * 4l;
        int v437;
        v437 = v436 + v389;
        float v438;
        v438 = 0.0f;
        int v439;
        v439 = 0l;
        while (while_method_1(v439)){
            int v441;
            v441 = 0l;
            while (while_method_0(v441)){
                assert("Tensor range check" && 0 <= v439 && v439 < 1l);
                assert("Tensor range check" && 0 <= v441 && v441 < 4l);
                int v443;
                v443 = 4l * v439;
                int v444;
                v444 = v443 + v441;
                float v445;
                v445 = v394[v444];
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
        v449 = v448 / 4l;
        auto v450 = cooperative_groups::labeled_partition(v447,v449);
        Closure0 v451{};
        float v452;
        v452 = cooperative_groups::reduce(v450, v438, v451);
        float v453;
        v453 = v452 / 16.0f;
        float v454[4l];
        int v455;
        v455 = 0l;
        while (while_method_1(v455)){
            int v457;
            v457 = 0l;
            while (while_method_0(v457)){
                assert("Tensor range check" && 0 <= v455 && v455 < 1l);
                assert("Tensor range check" && 0 <= v457 && v457 < 4l);
                int v459;
                v459 = 4l * v455;
                int v460;
                v460 = v459 + v457;
                float v461;
                v461 = v394[v460];
                float v462;
                v462 = v461 - v453;
                float v463;
                v463 = exp(v462);
                assert("Tensor range check" && 0 <= v455 && v455 < 1l);
                assert("Tensor range check" && 0 <= v457 && v457 < 4l);
                v454[v460] = v463;
                v457 += 1l ;
            }
            v455 += 1l ;
        }
        float v464;
        v464 = 0.0f;
        int v465;
        v465 = 0l;
        while (while_method_1(v465)){
            int v467;
            v467 = 0l;
            while (while_method_0(v467)){
                assert("Tensor range check" && 0 <= v465 && v465 < 1l);
                assert("Tensor range check" && 0 <= v467 && v467 < 4l);
                int v469;
                v469 = 4l * v465;
                int v470;
                v470 = v469 + v467;
                float v471;
                v471 = v454[v470];
                float v472;
                v472 = v464 + v471;
                v464 = v472;
                v467 += 1l ;
            }
            v465 += 1l ;
        }
        auto v473 = cooperative_groups::coalesced_threads();
        int v474;
        v474 = threadIdx.x;
        int v475;
        v475 = v474 / 4l;
        auto v476 = cooperative_groups::labeled_partition(v473,v475);
        float v477;
        v477 = cooperative_groups::reduce(v476, v464, v451);
        float v478[4l];
        int v479;
        v479 = 0l;
        while (while_method_1(v479)){
            int v481;
            v481 = 0l;
            while (while_method_0(v481)){
                assert("Tensor range check" && 0 <= v479 && v479 < 1l);
                assert("Tensor range check" && 0 <= v481 && v481 < 4l);
                int v483;
                v483 = 4l * v479;
                int v484;
                v484 = v483 + v481;
                float v485;
                v485 = v454[v484];
                bool v486;
                v486 = v477 == 0.0f;
                bool v487;
                v487 = v486 != true;
                float v489;
                if (v487){
                    float v488;
                    v488 = v485 / v477;
                    v489 = v488;
                } else {
                    v489 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v479 && v479 < 1l);
                assert("Tensor range check" && 0 <= v481 && v481 < 4l);
                v478[v484] = v489;
                v481 += 1l ;
            }
            v479 += 1l ;
        }
        float v490[4l];
        float v491;
        v491 = 0.0f;
        int v492;
        v492 = 0l;
        while (while_method_1(v492)){
            assert("Tensor range check" && 0 <= v492 && v492 < 1l);
            int v494;
            v494 = 4l * v492;
            assert("Tensor range check" && 0 <= v492 && v492 < 1l);
            int v495; float v496;
            Tuple1 tmp1 = Tuple1{0l, 0.0f};
            v495 = tmp1.v0; v496 = tmp1.v1;
            while (while_method_0(v495)){
                assert("Tensor range check" && 0 <= v495 && v495 < 4l);
                int v498;
                v498 = v495 + v494;
                float v499;
                v499 = v478[v498];
                float v500;
                v500 = v496 + v499;
                v496 = v500;
                v495 += 1l ;
            }
            auto v501 = cooperative_groups::coalesced_threads();
            int v502;
            v502 = threadIdx.x;
            int v503;
            v503 = v502 / 4l;
            auto v504 = cooperative_groups::labeled_partition(v501,v503);
            Closure1 v505{};
            float v506;
            v506 = cooperative_groups::inclusive_scan(v504, v496, v505);
            float v507;
            v507 = v504.shfl_up(v506,1);
            bool v508;
            v508 = v504.thread_rank() == 0;
            float v509;
            if (v508){
                v509 = 0.0f;
            } else {
                v509 = v507;
            }
            float v510;
            v510 = v504.shfl(v506,v504.num_threads()-1);
            float v511;
            v511 = v491 + v509;
            int v512; float v513;
            Tuple1 tmp2 = Tuple1{0l, v511};
            v512 = tmp2.v0; v513 = tmp2.v1;
            while (while_method_0(v512)){
                assert("Tensor range check" && 0 <= v512 && v512 < 4l);
                int v515;
                v515 = v512 + v494;
                float v516;
                v516 = v478[v515];
                float v517;
                v517 = v513 + v516;
                assert("Tensor range check" && 0 <= v512 && v512 < 4l);
                v490[v515] = v517;
                v513 = v517;
                v512 += 1l ;
            }
            float v518;
            v518 = v491 + v510;
            v491 = v518;
            v492 += 1l ;
        }
        float v519[4l];
        bool v520[4l];
        int v521;
        v521 = 0l;
        while (while_method_1(v521)){
            int v523;
            v523 = 0l;
            while (while_method_0(v523)){
                assert("Tensor range check" && 0 <= v521 && v521 < 1l);
                assert("Tensor range check" && 0 <= v523 && v523 < 4l);
                int v525;
                v525 = 4l * v521;
                int v526;
                v526 = v525 + v523;
                float v527;
                v527 = v490[v526];
                float v528;
                v528 = v478[v526];
                bool v529;
                v529 = v528 > 0.0f;
                assert("Tensor range check" && 0 <= v521 && v521 < 1l);
                assert("Tensor range check" && 0 <= v523 && v523 < 4l);
                v519[v526] = v527;
                v520[v526] = v529;
                v523 += 1l ;
            }
            v521 += 1l ;
        }
        float v530; bool v531;
        Tuple2 tmp3 = Tuple2{-1.0f / 0.0f, false};
        v530 = tmp3.v0; v531 = tmp3.v1;
        int v532;
        v532 = 0l;
        while (while_method_1(v532)){
            int v534;
            v534 = 0l;
            while (while_method_0(v534)){
                assert("Tensor range check" && 0 <= v532 && v532 < 1l);
                assert("Tensor range check" && 0 <= v534 && v534 < 4l);
                int v536;
                v536 = 4l * v532;
                int v537;
                v537 = v536 + v534;
                float v538;
                v538 = v519[v537];
                bool v539;
                v539 = v520[v537];
                float v546; bool v547;
                if (v531){
                    if (v539){
                        bool v540;
                        v540 = v530 >= v538;
                        float v541;
                        if (v540){
                            v541 = v530;
                        } else {
                            v541 = v538;
                        }
                        v546 = v541; v547 = true;
                    } else {
                        v546 = v530; v547 = v531;
                    }
                } else {
                    if (v539){
                        v546 = v538; v547 = v539;
                    } else {
                        v546 = v530; v547 = v531;
                    }
                }
                v530 = v546;
                v531 = v547;
                v534 += 1l ;
            }
            v532 += 1l ;
        }
        auto v548 = cooperative_groups::coalesced_threads();
        int v549;
        v549 = threadIdx.x;
        int v550;
        v550 = v549 / 4l;
        auto v551 = cooperative_groups::labeled_partition(v548,v550);
        Closure2 v552{};
        float v553; bool v554;
        Tuple2 tmp4 = cooperative_groups::reduce(v551, Tuple2{v530, v531}, v552);
        v553 = tmp4.v0; v554 = tmp4.v1;
        bool v555;
        v555 = v554 == false;
        if (v555){
            assert("The local reduce must be true." && v554);
        } else {
        }
        float v557[4l];
        int v558[4l];
        int v559;
        v559 = 0l;
        while (while_method_1(v559)){
            int v561;
            v561 = 0l;
            while (while_method_0(v561)){
                assert("Tensor range check" && 0 <= v559 && v559 < 1l);
                assert("Tensor range check" && 0 <= v561 && v561 < 4l);
                int v563;
                v563 = 4l * v559;
                int v564;
                v564 = v563 + v561;
                int v565;
                v565 = v395[v564];
                float v566;
                v566 = curand_uniform(&v11);
                assert("Tensor range check" && 0 <= v559 && v559 < 1l);
                assert("Tensor range check" && 0 <= v561 && v561 < 4l);
                v557[v564] = v566;
                v558[v564] = v565;
                v561 += 1l ;
            }
            v559 += 1l ;
        }
        float v567; int v568;
        Tuple3 tmp5 = Tuple3{0.0f, 2147483647l};
        v567 = tmp5.v0; v568 = tmp5.v1;
        int v569;
        v569 = 0l;
        while (while_method_1(v569)){
            int v571;
            v571 = 0l;
            while (while_method_0(v571)){
                assert("Tensor range check" && 0 <= v569 && v569 < 1l);
                assert("Tensor range check" && 0 <= v571 && v571 < 4l);
                int v573;
                v573 = 4l * v569;
                int v574;
                v574 = v573 + v571;
                float v575;
                v575 = v557[v574];
                int v576;
                v576 = v558[v574];
                bool v577;
                v577 = v568 < v576;
                float v578; int v579;
                if (v577){
                    v578 = v567; v579 = v568;
                } else {
                    v578 = v575; v579 = v576;
                }
                v567 = v578;
                v568 = v579;
                v571 += 1l ;
            }
            v569 += 1l ;
        }
        auto v580 = cooperative_groups::coalesced_threads();
        int v581;
        v581 = threadIdx.x;
        int v582;
        v582 = v581 / 4l;
        auto v583 = cooperative_groups::labeled_partition(v580,v582);
        Closure3 v584{};
        float v585; int v586;
        Tuple3 tmp6 = cooperative_groups::reduce(v583, Tuple3{v567, v568}, v584);
        v585 = tmp6.v0; v586 = tmp6.v1;
        float v587;
        v587 = v553 * v585;
        int v588[4l];
        bool v589[4l];
        int v590;
        v590 = 0l;
        while (while_method_1(v590)){
            int v592;
            v592 = 0l;
            while (while_method_0(v592)){
                assert("Tensor range check" && 0 <= v590 && v590 < 1l);
                assert("Tensor range check" && 0 <= v592 && v592 < 4l);
                int v594;
                v594 = 4l * v590;
                int v595;
                v595 = v594 + v592;
                float v596;
                v596 = v519[v595];
                bool v597;
                v597 = v520[v595];
                int v598;
                v598 = v395[v595];
                int v601; bool v602;
                if (v597){
                    float v599;
                    v599 = v596 - v587;
                    bool v600;
                    v600 = v599 >= 0.0f;
                    v601 = v598; v602 = v600;
                } else {
                    v601 = 2147483647l; v602 = false;
                }
                assert("Tensor range check" && 0 <= v590 && v590 < 1l);
                assert("Tensor range check" && 0 <= v592 && v592 < 4l);
                v588[v595] = v601;
                v589[v595] = v602;
                v592 += 1l ;
            }
            v590 += 1l ;
        }
        int v603; bool v604;
        Tuple4 tmp7 = Tuple4{2147483647l, false};
        v603 = tmp7.v0; v604 = tmp7.v1;
        int v605;
        v605 = 0l;
        while (while_method_1(v605)){
            int v607;
            v607 = 0l;
            while (while_method_0(v607)){
                assert("Tensor range check" && 0 <= v605 && v605 < 1l);
                assert("Tensor range check" && 0 <= v607 && v607 < 4l);
                int v609;
                v609 = 4l * v605;
                int v610;
                v610 = v609 + v607;
                int v611;
                v611 = v588[v610];
                bool v612;
                v612 = v589[v610];
                int v619; bool v620;
                if (v604){
                    if (v612){
                        bool v613;
                        v613 = v603 < v611;
                        int v614;
                        if (v613){
                            v614 = v603;
                        } else {
                            v614 = v611;
                        }
                        v619 = v614; v620 = true;
                    } else {
                        v619 = v603; v620 = v604;
                    }
                } else {
                    if (v612){
                        v619 = v611; v620 = v612;
                    } else {
                        v619 = v603; v620 = v604;
                    }
                }
                v603 = v619;
                v604 = v620;
                v607 += 1l ;
            }
            v605 += 1l ;
        }
        auto v621 = cooperative_groups::coalesced_threads();
        int v622;
        v622 = threadIdx.x;
        int v623;
        v623 = v622 / 4l;
        auto v624 = cooperative_groups::labeled_partition(v621,v623);
        Closure4 v625{};
        int v626; bool v627;
        Tuple4 tmp8 = cooperative_groups::reduce(v624, Tuple4{v603, v604}, v625);
        v626 = tmp8.v0; v627 = tmp8.v1;
        bool v628;
        v628 = v627 == false;
        if (v628){
            assert("The local reduce must be true." && v627);
        } else {
        }
        assert("Tensor range check" && 0 <= v389 && v389 < 4l);
        v375[v391] = v626;
        v389 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v630;
    v630 = threadIdx.x;
    assert("Tensor range check" && 0 <= v630 && v630 < 32l);
    int v631;
    v631 = v375[v630];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v632;
    v632 = threadIdx.x;
    assert("Tensor range check" && 0 <= v632 && v632 < 32l);
    v5[v632] = v631;
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
options.append('--diag-suppress=550,20012,68')
options.append('--restrict')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0(v0 : i32) -> bool:
    v1 = v0 < 32
    del v0
    return v1
def method1(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def main():
    v0 = cp.arange(0,512,1,dtype=cp.float32) # type: ignore
    v1 = v0.size
    v2 = 512 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,512,dtype=cp.float32) # type: ignore
    v6 = cp.empty(512,dtype=cp.int32)
    v7 = cp.empty(512,dtype=cp.int32)
    v8 = cp.empty(32,dtype=cp.int32)
    v9 = cp.empty(32,dtype=cp.int32)
    v10 = cp.empty(512,dtype=cp.float32)
    v11 = cp.empty(512,dtype=cp.float32)
    v12 = 0
    v13 = raw_module.get_function(f"entry{v12}")
    del v12
    v13.max_dynamic_shared_size_bytes = 0 
    v13((1,),(32,),(v0, v5, v6, v7, v8, v9, v10, v11),shared_mem=0)
    del v5, v6, v7, v8, v10, v11, v13
    v40 = 0
    v41 = "{}"
    print(v41.format('['),end="")
    v42 = 0
    while method0(v42):
        v44 = v40
        v45 = v44 >= 2147483647
        del v44
        if v45:
            v46 = " ..."
            print(v41.format(v46),end="")
            del v46
            break
        else:
            pass
        del v45
        v47 = v42 == 0
        v48 = v47 != True
        del v47
        if v48:
            v49 = "; "
            print(v41.format(v49),end="")
            del v49
        else:
            pass
        del v48
        print(v41.format('['),end="")
        v50 = 0
        while method1(v50):
            v52 = v40
            v53 = v52 >= 2147483647
            del v52
            if v53:
                v54 = " ..."
                print(v41.format(v54),end="")
                del v54
                break
            else:
                pass
            del v53
            v55 = v50 == 0
            v56 = v55 != True
            del v55
            if v56:
                v57 = "; "
                print(v41.format(v57),end="")
                del v57
            else:
                pass
            del v56
            v58 = v40 + 1
            v40 = v58
            del v58
            v59 = v42 * 16
            v60 = v59 + v50
            del v59
            v61 = v0[v60].item()
            del v60
            v62 = "{:.6f}"
            print(v62.format(v61),end="")
            del v61, v62
            v50 += 1 
        del v50
        print(v41.format(']'),end="")
        v42 += 1 
    del v0, v40, v42
    print(v41.format(']'),end="")
    v63 = "\n"
    print(v63,end="")
    v77 = 0
    print(v41.format('['),end="")
    v78 = 0
    while method0(v78):
        v80 = v77
        v81 = v80 >= 2147483647
        del v80
        if v81:
            v82 = " ..."
            print(v41.format(v82),end="")
            del v82
            break
        else:
            pass
        del v81
        v83 = v78 == 0
        v84 = v83 != True
        del v83
        if v84:
            v85 = "; "
            print(v41.format(v85),end="")
            del v85
        else:
            pass
        del v84
        v86 = v77 + 1
        v77 = v86
        del v86
        v87 = v9[v78].item()
        print(v41.format(v87),end="")
        del v87
        v78 += 1 
    del v9, v77, v78
    print(v41.format(']'),end="")
    del v41
    print(v63,end="")
    del v63
    return 

if __name__ == '__main__': print(main())
