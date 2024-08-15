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
    assert("Tensor range check" && 0 <= v24 && v24 < 32l);
    v22[v24] = v25;
    int v27;
    v27 = threadIdx.x;
    int * v28;
    v28 = v2+v17;
    int * v30;
    v30 = v3+v17;
    assert("Tensor range check" && 0 <= v27 && v27 < 32l);
    v23[v27] = Tuple0{v28, v30};
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v32;
    v32 = threadIdx.x;
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
    v41 = 4l * v37;
    assert("Tensor range check" && 0 <= v37 && v37 < 8l);
    int v42;
    v42 = 0l;
    while (while_method_0(v42)){
        assert("Tensor range check" && 0 <= v42 && v42 < 4l);
        int v44;
        v44 = v42 + v41;
        float * v45;
        v45 = v22[v44];
        assert("Tensor range check" && 0 <= v42 && v42 < 4l);
        int * v46; int * v47;
        Tuple0 tmp0 = v23[v44];
        v46 = tmp0.v0; v47 = tmp0.v1;
        assert("Tensor range check" && 0 <= v36 && v36 < 4l);
        int v48;
        v48 = 4l * v36;
        float v49[4l];
        int v50[4l];
        int v51;
        v51 = 0l;
        while (while_method_1(v51)){
            assert("Tensor range check" && 0 <= v51 && v51 < 1l);
            int v53;
            v53 = 4l * v51;
            assert("Tensor range check" && 0 <= v51 && v51 < 1l);
            int v54;
            v54 = v53 + v48;
            int4* v55;
            v55 = reinterpret_cast<int4*>(v45 + v54);
            int4* v56;
            v56 = reinterpret_cast<int4*>(v49 + v53);
            assert("Pointer alignment check" && (unsigned long long)(v55) % 4l == 0 && (unsigned long long)(v56) % 4l == 0);
            *v56 = *v55;
            v51 += 1l ;
        }
        int v57;
        v57 = 0l;
        while (while_method_1(v57)){
            int v59;
            v59 = 0l;
            while (while_method_0(v59)){
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
                v66 = 0l <= v57;
                bool v68;
                if (v66){
                    bool v67;
                    v67 = v57 < 1l;
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
                v71 = v57 * 4l;
                int v72;
                v72 = v59 + v71;
                bool v73;
                v73 = 0l <= v36;
                bool v75;
                if (v73){
                    bool v74;
                    v74 = v36 < 4l;
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
                v78 = v36 * 4l;
                int v79;
                v79 = v72 + v78;
                assert("Tensor range check" && 0 <= v57 && v57 < 1l);
                assert("Tensor range check" && 0 <= v59 && v59 < 4l);
                int v80;
                v80 = 4l * v57;
                int v81;
                v81 = v80 + v59;
                v50[v81] = v79;
                v59 += 1l ;
            }
            v57 += 1l ;
        }
        bool v82;
        v82 = 0l <= v42;
        bool v84;
        if (v82){
            bool v83;
            v83 = v42 < 4l;
            v84 = v83;
        } else {
            v84 = false;
        }
        bool v85;
        v85 = v84 == false;
        if (v85){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v84);
        } else {
        }
        bool v87;
        v87 = 0l <= v37;
        bool v88;
        v88 = v87 && v38;
        bool v89;
        v89 = v88 == false;
        if (v89){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v88);
        } else {
        }
        int v91;
        v91 = v37 * 4l;
        int v92;
        v92 = v91 + v42;
        int v93[4l];
        int v94[4l];
        int v95;
        v95 = 0l;
        while (while_method_1(v95)){
            int v97;
            v97 = 0l;
            while (while_method_0(v97)){
                assert("Tensor range check" && 0 <= v95 && v95 < 1l);
                assert("Tensor range check" && 0 <= v97 && v97 < 4l);
                int v99;
                v99 = 4l * v95;
                int v100;
                v100 = v99 + v97;
                int v101;
                v101 = v50[v100];
                assert("Tensor range check" && 0 <= v95 && v95 < 1l);
                assert("Tensor range check" && 0 <= v97 && v97 < 4l);
                v93[v100] = v92;
                v94[v100] = v101;
                v97 += 1l ;
            }
            v95 += 1l ;
        }
        assert("Tensor range check" && 0 <= v36 && v36 < 4l);
        int v102;
        v102 = 0l;
        while (while_method_1(v102)){
            assert("Tensor range check" && 0 <= v102 && v102 < 1l);
            int v104;
            v104 = 4l * v102;
            int v105;
            v105 = v104 + v48;
            assert("Tensor range check" && 0 <= v102 && v102 < 1l);
            int4* v106;
            v106 = reinterpret_cast<int4*>(v93 + v104);
            int4* v107;
            v107 = reinterpret_cast<int4*>(v46 + v105);
            assert("Pointer alignment check" && (unsigned long long)(v106) % 4l == 0 && (unsigned long long)(v107) % 4l == 0);
            *v107 = *v106;
            int4* v108;
            v108 = reinterpret_cast<int4*>(v94 + v104);
            int4* v109;
            v109 = reinterpret_cast<int4*>(v47 + v105);
            assert("Pointer alignment check" && (unsigned long long)(v108) % 4l == 0 && (unsigned long long)(v109) % 4l == 0);
            *v109 = *v108;
            v102 += 1l ;
        }
        v42 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v110[1l];
    __shared__ float * v111[32l];
    int v112;
    v112 = threadIdx.x;
    float * v113;
    v113 = v1+v13;
    assert("Tensor range check" && 0 <= v112 && v112 < 32l);
    v111[v112] = v113;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v115;
    v115 = threadIdx.x;
    bool v116;
    v116 = 0l <= v115;
    bool v117;
    v117 = v116 == false;
    if (v117){
        assert("The index needs to be zero or positive." && v116);
    } else {
    }
    int v119;
    v119 = v115 % 4l;
    int v120;
    v120 = v115 / 4l;
    bool v121;
    v121 = v120 < 8l;
    bool v122;
    v122 = v121 == false;
    if (v122){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v121);
    } else {
    }
    assert("Tensor range check" && 0 <= v120 && v120 < 8l);
    int v124;
    v124 = 4l * v120;
    int v125;
    v125 = 0l;
    while (while_method_0(v125)){
        assert("Tensor range check" && 0 <= v125 && v125 < 4l);
        int v127;
        v127 = v125 + v124;
        float * v128;
        v128 = v111[v127];
        assert("Tensor range check" && 0 <= v119 && v119 < 4l);
        int v129;
        v129 = 4l * v119;
        float v130[4l];
        int v131[4l];
        int v132;
        v132 = 0l;
        while (while_method_1(v132)){
            assert("Tensor range check" && 0 <= v132 && v132 < 1l);
            int v134;
            v134 = 4l * v132;
            assert("Tensor range check" && 0 <= v132 && v132 < 1l);
            int v135;
            v135 = v134 + v129;
            int4* v136;
            v136 = reinterpret_cast<int4*>(v128 + v135);
            int4* v137;
            v137 = reinterpret_cast<int4*>(v130 + v134);
            assert("Pointer alignment check" && (unsigned long long)(v136) % 4l == 0 && (unsigned long long)(v137) % 4l == 0);
            *v137 = *v136;
            v132 += 1l ;
        }
        int v138;
        v138 = 0l;
        while (while_method_1(v138)){
            int v140;
            v140 = 0l;
            while (while_method_0(v140)){
                bool v142;
                v142 = 0l <= v140;
                bool v144;
                if (v142){
                    bool v143;
                    v143 = v140 < 4l;
                    v144 = v143;
                } else {
                    v144 = false;
                }
                bool v145;
                v145 = v144 == false;
                if (v145){
                    assert("The indices should be inside the range of the dimension." && v144);
                } else {
                }
                bool v147;
                v147 = 0l <= v138;
                bool v149;
                if (v147){
                    bool v148;
                    v148 = v138 < 1l;
                    v149 = v148;
                } else {
                    v149 = false;
                }
                bool v150;
                v150 = v149 == false;
                if (v150){
                    assert("The indices should be inside the range of the dimension." && v149);
                } else {
                }
                int v152;
                v152 = v138 * 4l;
                int v153;
                v153 = v140 + v152;
                bool v154;
                v154 = 0l <= v119;
                bool v156;
                if (v154){
                    bool v155;
                    v155 = v119 < 4l;
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
                v159 = v119 * 4l;
                int v160;
                v160 = v153 + v159;
                assert("Tensor range check" && 0 <= v138 && v138 < 1l);
                assert("Tensor range check" && 0 <= v140 && v140 < 4l);
                int v161;
                v161 = 4l * v138;
                int v162;
                v162 = v161 + v140;
                v131[v162] = v160;
                v140 += 1l ;
            }
            v138 += 1l ;
        }
        bool v163;
        v163 = 0l <= v125;
        bool v165;
        if (v163){
            bool v164;
            v164 = v125 < 4l;
            v165 = v164;
        } else {
            v165 = false;
        }
        bool v166;
        v166 = v165 == false;
        if (v166){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v165);
        } else {
        }
        bool v168;
        v168 = 0l <= v120;
        bool v169;
        v169 = v168 && v121;
        bool v170;
        v170 = v169 == false;
        if (v170){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v169);
        } else {
        }
        int v172;
        v172 = v120 * 4l;
        int v173;
        v173 = v172 + v125;
        int v174;
        v174 = threadIdx.x;
        bool v175;
        v175 = v174 == v173;
        if (v175){
            v110[0l] = v173;
        } else {
        }
        __syncwarp();
        v125 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v176;
    v176 = v110[0l];
    int v177;
    v177 = threadIdx.x;
    assert("Tensor range check" && 0 <= v177 && v177 < 32l);
    v4[v177] = v176;
    __shared__ float * v178[32l];
    __shared__ float * v179[32l];
    int v180;
    v180 = threadIdx.x;
    float * v181;
    v181 = v1+v13;
    assert("Tensor range check" && 0 <= v180 && v180 < 32l);
    v178[v180] = v181;
    int v183;
    v183 = threadIdx.x;
    float * v184;
    v184 = v6+v19;
    assert("Tensor range check" && 0 <= v183 && v183 < 32l);
    v179[v183] = v184;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v186;
    v186 = threadIdx.x;
    bool v187;
    v187 = 0l <= v186;
    bool v188;
    v188 = v187 == false;
    if (v188){
        assert("The index needs to be zero or positive." && v187);
    } else {
    }
    int v190;
    v190 = v186 % 4l;
    int v191;
    v191 = v186 / 4l;
    bool v192;
    v192 = v191 < 8l;
    bool v193;
    v193 = v192 == false;
    if (v193){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v192);
    } else {
    }
    assert("Tensor range check" && 0 <= v191 && v191 < 8l);
    int v195;
    v195 = 4l * v191;
    assert("Tensor range check" && 0 <= v191 && v191 < 8l);
    int v196;
    v196 = 0l;
    while (while_method_0(v196)){
        assert("Tensor range check" && 0 <= v196 && v196 < 4l);
        int v198;
        v198 = v196 + v195;
        float * v199;
        v199 = v178[v198];
        assert("Tensor range check" && 0 <= v196 && v196 < 4l);
        float * v200;
        v200 = v179[v198];
        assert("Tensor range check" && 0 <= v190 && v190 < 4l);
        int v201;
        v201 = 4l * v190;
        float v202[4l];
        int v203[4l];
        int v204;
        v204 = 0l;
        while (while_method_1(v204)){
            assert("Tensor range check" && 0 <= v204 && v204 < 1l);
            int v206;
            v206 = 4l * v204;
            assert("Tensor range check" && 0 <= v204 && v204 < 1l);
            int v207;
            v207 = v206 + v201;
            int4* v208;
            v208 = reinterpret_cast<int4*>(v199 + v207);
            int4* v209;
            v209 = reinterpret_cast<int4*>(v202 + v206);
            assert("Pointer alignment check" && (unsigned long long)(v208) % 4l == 0 && (unsigned long long)(v209) % 4l == 0);
            *v209 = *v208;
            v204 += 1l ;
        }
        int v210;
        v210 = 0l;
        while (while_method_1(v210)){
            int v212;
            v212 = 0l;
            while (while_method_0(v212)){
                bool v214;
                v214 = 0l <= v212;
                bool v216;
                if (v214){
                    bool v215;
                    v215 = v212 < 4l;
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
                bool v219;
                v219 = 0l <= v210;
                bool v221;
                if (v219){
                    bool v220;
                    v220 = v210 < 1l;
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
                v224 = v210 * 4l;
                int v225;
                v225 = v212 + v224;
                bool v226;
                v226 = 0l <= v190;
                bool v228;
                if (v226){
                    bool v227;
                    v227 = v190 < 4l;
                    v228 = v227;
                } else {
                    v228 = false;
                }
                bool v229;
                v229 = v228 == false;
                if (v229){
                    assert("The indices should be inside the range of the dimension." && v228);
                } else {
                }
                int v231;
                v231 = v190 * 4l;
                int v232;
                v232 = v225 + v231;
                assert("Tensor range check" && 0 <= v210 && v210 < 1l);
                assert("Tensor range check" && 0 <= v212 && v212 < 4l);
                int v233;
                v233 = 4l * v210;
                int v234;
                v234 = v233 + v212;
                v203[v234] = v232;
                v212 += 1l ;
            }
            v210 += 1l ;
        }
        bool v235;
        v235 = 0l <= v196;
        bool v237;
        if (v235){
            bool v236;
            v236 = v196 < 4l;
            v237 = v236;
        } else {
            v237 = false;
        }
        bool v238;
        v238 = v237 == false;
        if (v238){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v237);
        } else {
        }
        bool v240;
        v240 = 0l <= v191;
        bool v241;
        v241 = v240 && v192;
        bool v242;
        v242 = v241 == false;
        if (v242){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v241);
        } else {
        }
        int v244;
        v244 = v191 * 4l;
        int v245;
        v245 = v244 + v196;
        assert("Tensor range check" && 0 <= v190 && v190 < 4l);
        int v246;
        v246 = 0l;
        while (while_method_1(v246)){
            assert("Tensor range check" && 0 <= v246 && v246 < 1l);
            int v248;
            v248 = 4l * v246;
            int v249;
            v249 = v248 + v201;
            assert("Tensor range check" && 0 <= v246 && v246 < 1l);
            int4* v250;
            v250 = reinterpret_cast<int4*>(v202 + v248);
            int4* v251;
            v251 = reinterpret_cast<int4*>(v200 + v249);
            assert("Pointer alignment check" && (unsigned long long)(v250) % 4l == 0 && (unsigned long long)(v251) % 4l == 0);
            *v251 = *v250;
            v246 += 1l ;
        }
        v196 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    __shared__ float * v252[32l];
    __shared__ float * v253[32l];
    int v254;
    v254 = threadIdx.x;
    float * v255;
    v255 = v1+v13;
    assert("Tensor range check" && 0 <= v254 && v254 < 32l);
    v252[v254] = v255;
    int v257;
    v257 = threadIdx.x;
    float * v258;
    v258 = v7+v21;
    assert("Tensor range check" && 0 <= v257 && v257 < 32l);
    v253[v257] = v258;
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
    v264 = v260 % 4l;
    int v265;
    v265 = v260 / 4l;
    bool v266;
    v266 = v265 < 8l;
    bool v267;
    v267 = v266 == false;
    if (v267){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v266);
    } else {
    }
    assert("Tensor range check" && 0 <= v265 && v265 < 8l);
    int v269;
    v269 = 4l * v265;
    assert("Tensor range check" && 0 <= v265 && v265 < 8l);
    int v270;
    v270 = 0l;
    while (while_method_0(v270)){
        assert("Tensor range check" && 0 <= v270 && v270 < 4l);
        int v272;
        v272 = v270 + v269;
        float * v273;
        v273 = v252[v272];
        assert("Tensor range check" && 0 <= v270 && v270 < 4l);
        float * v274;
        v274 = v253[v272];
        assert("Tensor range check" && 0 <= v264 && v264 < 4l);
        int v275;
        v275 = 4l * v264;
        float v276[4l];
        int v277[4l];
        int v278;
        v278 = 0l;
        while (while_method_1(v278)){
            assert("Tensor range check" && 0 <= v278 && v278 < 1l);
            int v280;
            v280 = 4l * v278;
            assert("Tensor range check" && 0 <= v278 && v278 < 1l);
            int v281;
            v281 = v280 + v275;
            int4* v282;
            v282 = reinterpret_cast<int4*>(v273 + v281);
            int4* v283;
            v283 = reinterpret_cast<int4*>(v276 + v280);
            assert("Pointer alignment check" && (unsigned long long)(v282) % 4l == 0 && (unsigned long long)(v283) % 4l == 0);
            *v283 = *v282;
            v278 += 1l ;
        }
        int v284;
        v284 = 0l;
        while (while_method_1(v284)){
            int v286;
            v286 = 0l;
            while (while_method_0(v286)){
                bool v288;
                v288 = 0l <= v286;
                bool v290;
                if (v288){
                    bool v289;
                    v289 = v286 < 4l;
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
                bool v293;
                v293 = 0l <= v284;
                bool v295;
                if (v293){
                    bool v294;
                    v294 = v284 < 1l;
                    v295 = v294;
                } else {
                    v295 = false;
                }
                bool v296;
                v296 = v295 == false;
                if (v296){
                    assert("The indices should be inside the range of the dimension." && v295);
                } else {
                }
                int v298;
                v298 = v284 * 4l;
                int v299;
                v299 = v286 + v298;
                bool v300;
                v300 = 0l <= v264;
                bool v302;
                if (v300){
                    bool v301;
                    v301 = v264 < 4l;
                    v302 = v301;
                } else {
                    v302 = false;
                }
                bool v303;
                v303 = v302 == false;
                if (v303){
                    assert("The indices should be inside the range of the dimension." && v302);
                } else {
                }
                int v305;
                v305 = v264 * 4l;
                int v306;
                v306 = v299 + v305;
                assert("Tensor range check" && 0 <= v284 && v284 < 1l);
                assert("Tensor range check" && 0 <= v286 && v286 < 4l);
                int v307;
                v307 = 4l * v284;
                int v308;
                v308 = v307 + v286;
                v277[v308] = v306;
                v286 += 1l ;
            }
            v284 += 1l ;
        }
        bool v309;
        v309 = 0l <= v270;
        bool v311;
        if (v309){
            bool v310;
            v310 = v270 < 4l;
            v311 = v310;
        } else {
            v311 = false;
        }
        bool v312;
        v312 = v311 == false;
        if (v312){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v311);
        } else {
        }
        bool v314;
        v314 = 0l <= v265;
        bool v315;
        v315 = v314 && v266;
        bool v316;
        v316 = v315 == false;
        if (v316){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v315);
        } else {
        }
        int v318;
        v318 = v265 * 4l;
        int v319;
        v319 = v318 + v270;
        float v320;
        v320 = 0.0f;
        int v321;
        v321 = 0l;
        while (while_method_1(v321)){
            int v323;
            v323 = 0l;
            while (while_method_0(v323)){
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
        v331 = v330 / 4l;
        auto v332 = cooperative_groups::labeled_partition(v329,v331);
        Closure0 v333{};
        float v334;
        v334 = cooperative_groups::reduce(v332, v320, v333);
        float v335;
        v335 = v334 / 16.0f;
        float v336[4l];
        int v337;
        v337 = 0l;
        while (while_method_1(v337)){
            int v339;
            v339 = 0l;
            while (while_method_0(v339)){
                assert("Tensor range check" && 0 <= v337 && v337 < 1l);
                assert("Tensor range check" && 0 <= v339 && v339 < 4l);
                int v341;
                v341 = 4l * v337;
                int v342;
                v342 = v341 + v339;
                float v343;
                v343 = v276[v342];
                float v344;
                v344 = v343 - v335;
                float v345;
                v345 = exp(v344);
                assert("Tensor range check" && 0 <= v337 && v337 < 1l);
                assert("Tensor range check" && 0 <= v339 && v339 < 4l);
                v336[v342] = v345;
                v339 += 1l ;
            }
            v337 += 1l ;
        }
        float v346;
        v346 = 0.0f;
        int v347;
        v347 = 0l;
        while (while_method_1(v347)){
            int v349;
            v349 = 0l;
            while (while_method_0(v349)){
                assert("Tensor range check" && 0 <= v347 && v347 < 1l);
                assert("Tensor range check" && 0 <= v349 && v349 < 4l);
                int v351;
                v351 = 4l * v347;
                int v352;
                v352 = v351 + v349;
                float v353;
                v353 = v336[v352];
                float v354;
                v354 = v346 + v353;
                v346 = v354;
                v349 += 1l ;
            }
            v347 += 1l ;
        }
        auto v355 = cooperative_groups::coalesced_threads();
        int v356;
        v356 = threadIdx.x;
        int v357;
        v357 = v356 / 4l;
        auto v358 = cooperative_groups::labeled_partition(v355,v357);
        float v359;
        v359 = cooperative_groups::reduce(v358, v346, v333);
        float v360[4l];
        int v361;
        v361 = 0l;
        while (while_method_1(v361)){
            int v363;
            v363 = 0l;
            while (while_method_0(v363)){
                assert("Tensor range check" && 0 <= v361 && v361 < 1l);
                assert("Tensor range check" && 0 <= v363 && v363 < 4l);
                int v365;
                v365 = 4l * v361;
                int v366;
                v366 = v365 + v363;
                float v367;
                v367 = v336[v366];
                bool v368;
                v368 = v359 == 0.0f;
                bool v369;
                v369 = v368 != true;
                float v371;
                if (v369){
                    float v370;
                    v370 = v367 / v359;
                    v371 = v370;
                } else {
                    v371 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v361 && v361 < 1l);
                assert("Tensor range check" && 0 <= v363 && v363 < 4l);
                v360[v366] = v371;
                v363 += 1l ;
            }
            v361 += 1l ;
        }
        assert("Tensor range check" && 0 <= v264 && v264 < 4l);
        int v372;
        v372 = 0l;
        while (while_method_1(v372)){
            assert("Tensor range check" && 0 <= v372 && v372 < 1l);
            int v374;
            v374 = 4l * v372;
            int v375;
            v375 = v374 + v275;
            assert("Tensor range check" && 0 <= v372 && v372 < 1l);
            int4* v376;
            v376 = reinterpret_cast<int4*>(v360 + v374);
            int4* v377;
            v377 = reinterpret_cast<int4*>(v274 + v375);
            assert("Pointer alignment check" && (unsigned long long)(v376) % 4l == 0 && (unsigned long long)(v377) % 4l == 0);
            *v377 = *v376;
            v372 += 1l ;
        }
        v270 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v378[1l];
    __shared__ float * v379[32l];
    int v380;
    v380 = threadIdx.x;
    float * v381;
    v381 = v0+v15;
    assert("Tensor range check" && 0 <= v380 && v380 < 32l);
    v379[v380] = v381;
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
    v387 = v383 % 4l;
    int v388;
    v388 = v383 / 4l;
    bool v389;
    v389 = v388 < 8l;
    bool v390;
    v390 = v389 == false;
    if (v390){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v389);
    } else {
    }
    assert("Tensor range check" && 0 <= v388 && v388 < 8l);
    int v392;
    v392 = 4l * v388;
    int v393;
    v393 = 0l;
    while (while_method_0(v393)){
        assert("Tensor range check" && 0 <= v393 && v393 < 4l);
        int v395;
        v395 = v393 + v392;
        float * v396;
        v396 = v379[v395];
        assert("Tensor range check" && 0 <= v387 && v387 < 4l);
        int v397;
        v397 = 4l * v387;
        float v398[4l];
        int v399[4l];
        int v400;
        v400 = 0l;
        while (while_method_1(v400)){
            assert("Tensor range check" && 0 <= v400 && v400 < 1l);
            int v402;
            v402 = 4l * v400;
            assert("Tensor range check" && 0 <= v400 && v400 < 1l);
            int v403;
            v403 = v402 + v397;
            int4* v404;
            v404 = reinterpret_cast<int4*>(v396 + v403);
            int4* v405;
            v405 = reinterpret_cast<int4*>(v398 + v402);
            assert("Pointer alignment check" && (unsigned long long)(v404) % 4l == 0 && (unsigned long long)(v405) % 4l == 0);
            *v405 = *v404;
            v400 += 1l ;
        }
        int v406;
        v406 = 0l;
        while (while_method_1(v406)){
            int v408;
            v408 = 0l;
            while (while_method_0(v408)){
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
                v415 = 0l <= v406;
                bool v417;
                if (v415){
                    bool v416;
                    v416 = v406 < 1l;
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
                v420 = v406 * 4l;
                int v421;
                v421 = v408 + v420;
                bool v422;
                v422 = 0l <= v387;
                bool v424;
                if (v422){
                    bool v423;
                    v423 = v387 < 4l;
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
                v427 = v387 * 4l;
                int v428;
                v428 = v421 + v427;
                assert("Tensor range check" && 0 <= v406 && v406 < 1l);
                assert("Tensor range check" && 0 <= v408 && v408 < 4l);
                int v429;
                v429 = 4l * v406;
                int v430;
                v430 = v429 + v408;
                v399[v430] = v428;
                v408 += 1l ;
            }
            v406 += 1l ;
        }
        bool v431;
        v431 = 0l <= v393;
        bool v433;
        if (v431){
            bool v432;
            v432 = v393 < 4l;
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
        bool v436;
        v436 = 0l <= v388;
        bool v437;
        v437 = v436 && v389;
        bool v438;
        v438 = v437 == false;
        if (v438){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v437);
        } else {
        }
        int v440;
        v440 = v388 * 4l;
        int v441;
        v441 = v440 + v393;
        float v442;
        v442 = 0.0f;
        int v443;
        v443 = 0l;
        while (while_method_1(v443)){
            int v445;
            v445 = 0l;
            while (while_method_0(v445)){
                assert("Tensor range check" && 0 <= v443 && v443 < 1l);
                assert("Tensor range check" && 0 <= v445 && v445 < 4l);
                int v447;
                v447 = 4l * v443;
                int v448;
                v448 = v447 + v445;
                float v449;
                v449 = v398[v448];
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
        v453 = v452 / 4l;
        auto v454 = cooperative_groups::labeled_partition(v451,v453);
        Closure0 v455{};
        float v456;
        v456 = cooperative_groups::reduce(v454, v442, v455);
        float v457;
        v457 = v456 / 16.0f;
        float v458[4l];
        int v459;
        v459 = 0l;
        while (while_method_1(v459)){
            int v461;
            v461 = 0l;
            while (while_method_0(v461)){
                assert("Tensor range check" && 0 <= v459 && v459 < 1l);
                assert("Tensor range check" && 0 <= v461 && v461 < 4l);
                int v463;
                v463 = 4l * v459;
                int v464;
                v464 = v463 + v461;
                float v465;
                v465 = v398[v464];
                float v466;
                v466 = v465 - v457;
                float v467;
                v467 = exp(v466);
                assert("Tensor range check" && 0 <= v459 && v459 < 1l);
                assert("Tensor range check" && 0 <= v461 && v461 < 4l);
                v458[v464] = v467;
                v461 += 1l ;
            }
            v459 += 1l ;
        }
        float v468;
        v468 = 0.0f;
        int v469;
        v469 = 0l;
        while (while_method_1(v469)){
            int v471;
            v471 = 0l;
            while (while_method_0(v471)){
                assert("Tensor range check" && 0 <= v469 && v469 < 1l);
                assert("Tensor range check" && 0 <= v471 && v471 < 4l);
                int v473;
                v473 = 4l * v469;
                int v474;
                v474 = v473 + v471;
                float v475;
                v475 = v458[v474];
                float v476;
                v476 = v468 + v475;
                v468 = v476;
                v471 += 1l ;
            }
            v469 += 1l ;
        }
        auto v477 = cooperative_groups::coalesced_threads();
        int v478;
        v478 = threadIdx.x;
        int v479;
        v479 = v478 / 4l;
        auto v480 = cooperative_groups::labeled_partition(v477,v479);
        float v481;
        v481 = cooperative_groups::reduce(v480, v468, v455);
        float v482[4l];
        int v483;
        v483 = 0l;
        while (while_method_1(v483)){
            int v485;
            v485 = 0l;
            while (while_method_0(v485)){
                assert("Tensor range check" && 0 <= v483 && v483 < 1l);
                assert("Tensor range check" && 0 <= v485 && v485 < 4l);
                int v487;
                v487 = 4l * v483;
                int v488;
                v488 = v487 + v485;
                float v489;
                v489 = v458[v488];
                bool v490;
                v490 = v481 == 0.0f;
                bool v491;
                v491 = v490 != true;
                float v493;
                if (v491){
                    float v492;
                    v492 = v489 / v481;
                    v493 = v492;
                } else {
                    v493 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v483 && v483 < 1l);
                assert("Tensor range check" && 0 <= v485 && v485 < 4l);
                v482[v488] = v493;
                v485 += 1l ;
            }
            v483 += 1l ;
        }
        float v494[4l];
        float v495;
        v495 = 0.0f;
        int v496;
        v496 = 0l;
        while (while_method_1(v496)){
            assert("Tensor range check" && 0 <= v496 && v496 < 1l);
            int v498;
            v498 = 4l * v496;
            assert("Tensor range check" && 0 <= v496 && v496 < 1l);
            int v499; float v500;
            Tuple1 tmp1 = Tuple1{0l, 0.0f};
            v499 = tmp1.v0; v500 = tmp1.v1;
            while (while_method_0(v499)){
                assert("Tensor range check" && 0 <= v499 && v499 < 4l);
                int v502;
                v502 = v499 + v498;
                float v503;
                v503 = v482[v502];
                float v504;
                v504 = v500 + v503;
                v500 = v504;
                v499 += 1l ;
            }
            auto v505 = cooperative_groups::coalesced_threads();
            int v506;
            v506 = threadIdx.x;
            int v507;
            v507 = v506 / 4l;
            auto v508 = cooperative_groups::labeled_partition(v505,v507);
            Closure1 v509{};
            float v510;
            v510 = cooperative_groups::inclusive_scan(v508, v500, v509);
            float v511;
            v511 = v508.shfl_up(v510,1);
            bool v512;
            v512 = v508.thread_rank() == 0;
            float v513;
            if (v512){
                v513 = 0.0f;
            } else {
                v513 = v511;
            }
            float v514;
            v514 = v508.shfl(v510,v508.num_threads()-1);
            float v515;
            v515 = v495 + v513;
            int v516; float v517;
            Tuple1 tmp2 = Tuple1{0l, v515};
            v516 = tmp2.v0; v517 = tmp2.v1;
            while (while_method_0(v516)){
                assert("Tensor range check" && 0 <= v516 && v516 < 4l);
                int v519;
                v519 = v516 + v498;
                float v520;
                v520 = v482[v519];
                float v521;
                v521 = v517 + v520;
                assert("Tensor range check" && 0 <= v516 && v516 < 4l);
                v494[v519] = v521;
                v517 = v521;
                v516 += 1l ;
            }
            float v522;
            v522 = v495 + v514;
            v495 = v522;
            v496 += 1l ;
        }
        float v523[4l];
        bool v524[4l];
        int v525;
        v525 = 0l;
        while (while_method_1(v525)){
            int v527;
            v527 = 0l;
            while (while_method_0(v527)){
                assert("Tensor range check" && 0 <= v525 && v525 < 1l);
                assert("Tensor range check" && 0 <= v527 && v527 < 4l);
                int v529;
                v529 = 4l * v525;
                int v530;
                v530 = v529 + v527;
                float v531;
                v531 = v494[v530];
                float v532;
                v532 = v482[v530];
                bool v533;
                v533 = v532 > 0.0f;
                assert("Tensor range check" && 0 <= v525 && v525 < 1l);
                assert("Tensor range check" && 0 <= v527 && v527 < 4l);
                v523[v530] = v531;
                v524[v530] = v533;
                v527 += 1l ;
            }
            v525 += 1l ;
        }
        float v534; bool v535;
        Tuple2 tmp3 = Tuple2{-1.0f / 0.0f, false};
        v534 = tmp3.v0; v535 = tmp3.v1;
        int v536;
        v536 = 0l;
        while (while_method_1(v536)){
            int v538;
            v538 = 0l;
            while (while_method_0(v538)){
                assert("Tensor range check" && 0 <= v536 && v536 < 1l);
                assert("Tensor range check" && 0 <= v538 && v538 < 4l);
                int v540;
                v540 = 4l * v536;
                int v541;
                v541 = v540 + v538;
                float v542;
                v542 = v523[v541];
                bool v543;
                v543 = v524[v541];
                float v550; bool v551;
                if (v535){
                    if (v543){
                        bool v544;
                        v544 = v534 >= v542;
                        float v545;
                        if (v544){
                            v545 = v534;
                        } else {
                            v545 = v542;
                        }
                        v550 = v545; v551 = true;
                    } else {
                        v550 = v534; v551 = v535;
                    }
                } else {
                    if (v543){
                        v550 = v542; v551 = v543;
                    } else {
                        v550 = v534; v551 = v535;
                    }
                }
                v534 = v550;
                v535 = v551;
                v538 += 1l ;
            }
            v536 += 1l ;
        }
        auto v552 = cooperative_groups::coalesced_threads();
        int v553;
        v553 = threadIdx.x;
        int v554;
        v554 = v553 / 4l;
        auto v555 = cooperative_groups::labeled_partition(v552,v554);
        Closure2 v556{};
        float v557; bool v558;
        Tuple2 tmp4 = cooperative_groups::reduce(v555, Tuple2{v534, v535}, v556);
        v557 = tmp4.v0; v558 = tmp4.v1;
        bool v559;
        v559 = v558 == false;
        if (v559){
            assert("The local reduce must be true." && v558);
        } else {
        }
        float v561[4l];
        int v562[4l];
        int v563;
        v563 = 0l;
        while (while_method_1(v563)){
            int v565;
            v565 = 0l;
            while (while_method_0(v565)){
                assert("Tensor range check" && 0 <= v563 && v563 < 1l);
                assert("Tensor range check" && 0 <= v565 && v565 < 4l);
                int v567;
                v567 = 4l * v563;
                int v568;
                v568 = v567 + v565;
                int v569;
                v569 = v399[v568];
                float v570;
                v570 = curand_uniform(&v11);
                assert("Tensor range check" && 0 <= v563 && v563 < 1l);
                assert("Tensor range check" && 0 <= v565 && v565 < 4l);
                v561[v568] = v570;
                v562[v568] = v569;
                v565 += 1l ;
            }
            v563 += 1l ;
        }
        float v571; int v572;
        Tuple3 tmp5 = Tuple3{0.0f, 2147483647l};
        v571 = tmp5.v0; v572 = tmp5.v1;
        int v573;
        v573 = 0l;
        while (while_method_1(v573)){
            int v575;
            v575 = 0l;
            while (while_method_0(v575)){
                assert("Tensor range check" && 0 <= v573 && v573 < 1l);
                assert("Tensor range check" && 0 <= v575 && v575 < 4l);
                int v577;
                v577 = 4l * v573;
                int v578;
                v578 = v577 + v575;
                float v579;
                v579 = v561[v578];
                int v580;
                v580 = v562[v578];
                bool v581;
                v581 = v572 < v580;
                float v582; int v583;
                if (v581){
                    v582 = v571; v583 = v572;
                } else {
                    v582 = v579; v583 = v580;
                }
                v571 = v582;
                v572 = v583;
                v575 += 1l ;
            }
            v573 += 1l ;
        }
        auto v584 = cooperative_groups::coalesced_threads();
        int v585;
        v585 = threadIdx.x;
        int v586;
        v586 = v585 / 4l;
        auto v587 = cooperative_groups::labeled_partition(v584,v586);
        Closure3 v588{};
        float v589; int v590;
        Tuple3 tmp6 = cooperative_groups::reduce(v587, Tuple3{v571, v572}, v588);
        v589 = tmp6.v0; v590 = tmp6.v1;
        float v591;
        v591 = v557 * v589;
        int v592[4l];
        bool v593[4l];
        int v594;
        v594 = 0l;
        while (while_method_1(v594)){
            int v596;
            v596 = 0l;
            while (while_method_0(v596)){
                assert("Tensor range check" && 0 <= v594 && v594 < 1l);
                assert("Tensor range check" && 0 <= v596 && v596 < 4l);
                int v598;
                v598 = 4l * v594;
                int v599;
                v599 = v598 + v596;
                float v600;
                v600 = v523[v599];
                bool v601;
                v601 = v524[v599];
                int v602;
                v602 = v399[v599];
                int v605; bool v606;
                if (v601){
                    float v603;
                    v603 = v600 - v591;
                    bool v604;
                    v604 = v603 >= 0.0f;
                    v605 = v602; v606 = v604;
                } else {
                    v605 = 2147483647l; v606 = false;
                }
                assert("Tensor range check" && 0 <= v594 && v594 < 1l);
                assert("Tensor range check" && 0 <= v596 && v596 < 4l);
                v592[v599] = v605;
                v593[v599] = v606;
                v596 += 1l ;
            }
            v594 += 1l ;
        }
        int v607; bool v608;
        Tuple4 tmp7 = Tuple4{2147483647l, false};
        v607 = tmp7.v0; v608 = tmp7.v1;
        int v609;
        v609 = 0l;
        while (while_method_1(v609)){
            int v611;
            v611 = 0l;
            while (while_method_0(v611)){
                assert("Tensor range check" && 0 <= v609 && v609 < 1l);
                assert("Tensor range check" && 0 <= v611 && v611 < 4l);
                int v613;
                v613 = 4l * v609;
                int v614;
                v614 = v613 + v611;
                int v615;
                v615 = v592[v614];
                bool v616;
                v616 = v593[v614];
                int v623; bool v624;
                if (v608){
                    if (v616){
                        bool v617;
                        v617 = v607 < v615;
                        int v618;
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
        v627 = v626 / 4l;
        auto v628 = cooperative_groups::labeled_partition(v625,v627);
        Closure4 v629{};
        int v630; bool v631;
        Tuple4 tmp8 = cooperative_groups::reduce(v628, Tuple4{v607, v608}, v629);
        v630 = tmp8.v0; v631 = tmp8.v1;
        bool v632;
        v632 = v631 == false;
        if (v632){
            assert("The local reduce must be true." && v631);
        } else {
        }
        int v634;
        v634 = threadIdx.x;
        bool v635;
        v635 = v634 == v441;
        if (v635){
            v378[0l] = v630;
        } else {
        }
        __syncwarp();
        v393 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v636;
    v636 = v378[0l];
    int v637;
    v637 = threadIdx.x;
    assert("Tensor range check" && 0 <= v637 && v637 < 32l);
    v5[v637] = v636;
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
