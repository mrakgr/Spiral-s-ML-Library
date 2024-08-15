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
struct Closure1 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
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
struct Closure2 {
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
struct Closure3 {
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
struct Closure4 {
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
struct Closure5 {
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
    v1 = v0 < 32l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 4l;
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
    v13 = 1024l * v12;
    int v14;
    v14 = threadIdx.x;
    assert("Tensor range check" && 0 <= v14 && v14 < 32l);
    int v15;
    v15 = 1024l * v14;
    int v16;
    v16 = threadIdx.x;
    assert("Tensor range check" && 0 <= v16 && v16 < 32l);
    int v17;
    v17 = 1024l * v16;
    int v18;
    v18 = threadIdx.x;
    assert("Tensor range check" && 0 <= v18 && v18 < 32l);
    int v19;
    v19 = 1024l * v18;
    int v20;
    v20 = threadIdx.x;
    assert("Tensor range check" && 0 <= v20 && v20 < 32l);
    int v21;
    v21 = 1024l * v20;
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
    v28 = v2+v19;
    int * v30;
    v30 = v3+v19;
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
    v41 = 32l * v37;
    assert("Tensor range check" && 0 <= v37 && v37 < 1l);
    int v42;
    v42 = 0l;
    while (while_method_0(v42)){
        assert("Tensor range check" && 0 <= v42 && v42 < 32l);
        int v44;
        v44 = v42 + v41;
        float * v45;
        v45 = v22[v44];
        assert("Tensor range check" && 0 <= v42 && v42 < 32l);
        int * v46; int * v47;
        Tuple0 tmp0 = v23[v44];
        v46 = tmp0.v0; v47 = tmp0.v1;
        assert("Tensor range check" && 0 <= v36 && v36 < 32l);
        int v48;
        v48 = 4l * v36;
        float v49[32l];
        int v50[32l];
        int v51;
        v51 = 0l;
        while (while_method_1(v51)){
            assert("Tensor range check" && 0 <= v51 && v51 < 8l);
            int v53;
            v53 = 4l * v51;
            assert("Tensor range check" && 0 <= v51 && v51 < 8l);
            int v54;
            v54 = 128l * v51;
            int v55;
            v55 = v54 + v48;
            int4* v56;
            v56 = reinterpret_cast<int4*>(v45 + v55);
            int4* v57;
            v57 = reinterpret_cast<int4*>(v49 + v53);
            assert("Pointer alignment check" && (unsigned long long)(v56) % 4l == 0 && (unsigned long long)(v57) % 4l == 0);
            *v57 = *v56;
            v51 += 1l ;
        }
        int v58;
        v58 = 0l;
        while (while_method_1(v58)){
            int v60;
            v60 = 0l;
            while (while_method_2(v60)){
                bool v62;
                v62 = 0l <= v60;
                bool v64;
                if (v62){
                    bool v63;
                    v63 = v60 < 4l;
                    v64 = v63;
                } else {
                    v64 = false;
                }
                bool v65;
                v65 = v64 == false;
                if (v65){
                    assert("The indices should be inside the range of the dimension." && v64);
                } else {
                }
                bool v67;
                v67 = 0l <= v36;
                bool v69;
                if (v67){
                    bool v68;
                    v68 = v36 < 32l;
                    v69 = v68;
                } else {
                    v69 = false;
                }
                bool v70;
                v70 = v69 == false;
                if (v70){
                    assert("The indices should be inside the range of the dimension." && v69);
                } else {
                }
                int v72;
                v72 = v36 * 4l;
                int v73;
                v73 = v60 + v72;
                bool v74;
                v74 = 0l <= v58;
                bool v76;
                if (v74){
                    bool v75;
                    v75 = v58 < 8l;
                    v76 = v75;
                } else {
                    v76 = false;
                }
                bool v77;
                v77 = v76 == false;
                if (v77){
                    assert("The indices should be inside the range of the dimension." && v76);
                } else {
                }
                int v79;
                v79 = v58 * 128l;
                int v80;
                v80 = v73 + v79;
                assert("Tensor range check" && 0 <= v58 && v58 < 8l);
                assert("Tensor range check" && 0 <= v60 && v60 < 4l);
                int v81;
                v81 = 4l * v58;
                int v82;
                v82 = v81 + v60;
                v50[v82] = v80;
                v60 += 1l ;
            }
            v58 += 1l ;
        }
        bool v83;
        v83 = 0l <= v42;
        bool v85;
        if (v83){
            bool v84;
            v84 = v42 < 32l;
            v85 = v84;
        } else {
            v85 = false;
        }
        bool v86;
        v86 = v85 == false;
        if (v86){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v85);
        } else {
        }
        bool v88;
        v88 = 0l <= v37;
        bool v89;
        v89 = v88 && v38;
        bool v90;
        v90 = v89 == false;
        if (v90){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v89);
        } else {
        }
        int v92;
        v92 = v37 * 32l;
        int v93;
        v93 = v92 + v42;
        int v94[32l];
        int v95[32l];
        int v96;
        v96 = 0l;
        while (while_method_1(v96)){
            int v98;
            v98 = 0l;
            while (while_method_2(v98)){
                assert("Tensor range check" && 0 <= v96 && v96 < 8l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                int v100;
                v100 = 4l * v96;
                int v101;
                v101 = v100 + v98;
                int v102;
                v102 = v50[v101];
                assert("Tensor range check" && 0 <= v96 && v96 < 8l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                v94[v101] = v93;
                v95[v101] = v102;
                v98 += 1l ;
            }
            v96 += 1l ;
        }
        assert("Tensor range check" && 0 <= v36 && v36 < 32l);
        int v103;
        v103 = 0l;
        while (while_method_1(v103)){
            assert("Tensor range check" && 0 <= v103 && v103 < 8l);
            int v105;
            v105 = 128l * v103;
            int v106;
            v106 = v105 + v48;
            assert("Tensor range check" && 0 <= v103 && v103 < 8l);
            int v107;
            v107 = 4l * v103;
            int4* v108;
            v108 = reinterpret_cast<int4*>(v94 + v107);
            int4* v109;
            v109 = reinterpret_cast<int4*>(v46 + v106);
            assert("Pointer alignment check" && (unsigned long long)(v108) % 4l == 0 && (unsigned long long)(v109) % 4l == 0);
            *v109 = *v108;
            int4* v110;
            v110 = reinterpret_cast<int4*>(v95 + v107);
            int4* v111;
            v111 = reinterpret_cast<int4*>(v47 + v106);
            assert("Pointer alignment check" && (unsigned long long)(v110) % 4l == 0 && (unsigned long long)(v111) % 4l == 0);
            *v111 = *v110;
            v103 += 1l ;
        }
        v42 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v112[1l];
    __shared__ float * v113[32l];
    __shared__ int v114[32l];
    int v115;
    v115 = threadIdx.x;
    float * v116;
    v116 = v1+v13;
    assert("Tensor range check" && 0 <= v115 && v115 < 32l);
    v113[v115] = v116;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v118;
    v118 = threadIdx.x;
    bool v119;
    v119 = 0l <= v118;
    bool v120;
    v120 = v119 == false;
    if (v120){
        assert("The index needs to be zero or positive." && v119);
    } else {
    }
    int v122;
    v122 = v118 % 32l;
    int v123;
    v123 = v118 / 32l;
    bool v124;
    v124 = v123 < 1l;
    bool v125;
    v125 = v124 == false;
    if (v125){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v124);
    } else {
    }
    assert("Tensor range check" && 0 <= v123 && v123 < 1l);
    int v127;
    v127 = 32l * v123;
    assert("Tensor range check" && 0 <= v123 && v123 < 1l);
    int v128;
    v128 = 0l;
    while (while_method_0(v128)){
        assert("Tensor range check" && 0 <= v128 && v128 < 32l);
        int v130;
        v130 = v128 + v127;
        float * v131;
        v131 = v113[v130];
        assert("Tensor range check" && 0 <= v122 && v122 < 32l);
        int v132;
        v132 = 4l * v122;
        float v133[32l];
        int v134[32l];
        int v135;
        v135 = 0l;
        while (while_method_1(v135)){
            assert("Tensor range check" && 0 <= v135 && v135 < 8l);
            int v137;
            v137 = 4l * v135;
            assert("Tensor range check" && 0 <= v135 && v135 < 8l);
            int v138;
            v138 = 128l * v135;
            int v139;
            v139 = v138 + v132;
            int4* v140;
            v140 = reinterpret_cast<int4*>(v131 + v139);
            int4* v141;
            v141 = reinterpret_cast<int4*>(v133 + v137);
            assert("Pointer alignment check" && (unsigned long long)(v140) % 4l == 0 && (unsigned long long)(v141) % 4l == 0);
            *v141 = *v140;
            v135 += 1l ;
        }
        int v142;
        v142 = 0l;
        while (while_method_1(v142)){
            int v144;
            v144 = 0l;
            while (while_method_2(v144)){
                bool v146;
                v146 = 0l <= v144;
                bool v148;
                if (v146){
                    bool v147;
                    v147 = v144 < 4l;
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
                bool v151;
                v151 = 0l <= v122;
                bool v153;
                if (v151){
                    bool v152;
                    v152 = v122 < 32l;
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
                int v156;
                v156 = v122 * 4l;
                int v157;
                v157 = v144 + v156;
                bool v158;
                v158 = 0l <= v142;
                bool v160;
                if (v158){
                    bool v159;
                    v159 = v142 < 8l;
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
                v163 = v142 * 128l;
                int v164;
                v164 = v157 + v163;
                assert("Tensor range check" && 0 <= v142 && v142 < 8l);
                assert("Tensor range check" && 0 <= v144 && v144 < 4l);
                int v165;
                v165 = 4l * v142;
                int v166;
                v166 = v165 + v144;
                v134[v166] = v164;
                v144 += 1l ;
            }
            v142 += 1l ;
        }
        bool v167;
        v167 = 0l <= v128;
        bool v169;
        if (v167){
            bool v168;
            v168 = v128 < 32l;
            v169 = v168;
        } else {
            v169 = false;
        }
        bool v170;
        v170 = v169 == false;
        if (v170){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v169);
        } else {
        }
        bool v172;
        v172 = 0l <= v123;
        bool v173;
        v173 = v172 && v124;
        bool v174;
        v174 = v173 == false;
        if (v174){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v173);
        } else {
        }
        int v176;
        v176 = v123 * 32l;
        int v177;
        v177 = v176 + v128;
        assert("Tensor range check" && 0 <= v128 && v128 < 32l);
        v114[v130] = v177;
        v128 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v178;
    v178 = threadIdx.x;
    assert("Tensor range check" && 0 <= v178 && v178 < 32l);
    int v179;
    v179 = v114[v178];
    v112[0l] = v179;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v180;
    v180 = v112[0l];
    int v181;
    v181 = threadIdx.x;
    assert("Tensor range check" && 0 <= v181 && v181 < 32l);
    v4[v181] = v180;
    __shared__ float * v182[32l];
    __shared__ float * v183[32l];
    int v184;
    v184 = threadIdx.x;
    float * v185;
    v185 = v1+v13;
    assert("Tensor range check" && 0 <= v184 && v184 < 32l);
    v182[v184] = v185;
    int v187;
    v187 = threadIdx.x;
    float * v188;
    v188 = v6+v21;
    assert("Tensor range check" && 0 <= v187 && v187 < 32l);
    v183[v187] = v188;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v190;
    v190 = threadIdx.x;
    bool v191;
    v191 = 0l <= v190;
    bool v192;
    v192 = v191 == false;
    if (v192){
        assert("The index needs to be zero or positive." && v191);
    } else {
    }
    int v194;
    v194 = v190 % 32l;
    int v195;
    v195 = v190 / 32l;
    bool v196;
    v196 = v195 < 1l;
    bool v197;
    v197 = v196 == false;
    if (v197){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v196);
    } else {
    }
    assert("Tensor range check" && 0 <= v195 && v195 < 1l);
    int v199;
    v199 = 32l * v195;
    assert("Tensor range check" && 0 <= v195 && v195 < 1l);
    int v200;
    v200 = 0l;
    while (while_method_0(v200)){
        assert("Tensor range check" && 0 <= v200 && v200 < 32l);
        int v202;
        v202 = v200 + v199;
        float * v203;
        v203 = v182[v202];
        assert("Tensor range check" && 0 <= v200 && v200 < 32l);
        float * v204;
        v204 = v183[v202];
        assert("Tensor range check" && 0 <= v194 && v194 < 32l);
        int v205;
        v205 = 4l * v194;
        float v206[32l];
        int v207[32l];
        int v208;
        v208 = 0l;
        while (while_method_1(v208)){
            assert("Tensor range check" && 0 <= v208 && v208 < 8l);
            int v210;
            v210 = 4l * v208;
            assert("Tensor range check" && 0 <= v208 && v208 < 8l);
            int v211;
            v211 = 128l * v208;
            int v212;
            v212 = v211 + v205;
            int4* v213;
            v213 = reinterpret_cast<int4*>(v203 + v212);
            int4* v214;
            v214 = reinterpret_cast<int4*>(v206 + v210);
            assert("Pointer alignment check" && (unsigned long long)(v213) % 4l == 0 && (unsigned long long)(v214) % 4l == 0);
            *v214 = *v213;
            v208 += 1l ;
        }
        int v215;
        v215 = 0l;
        while (while_method_1(v215)){
            int v217;
            v217 = 0l;
            while (while_method_2(v217)){
                bool v219;
                v219 = 0l <= v217;
                bool v221;
                if (v219){
                    bool v220;
                    v220 = v217 < 4l;
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
                bool v224;
                v224 = 0l <= v194;
                bool v226;
                if (v224){
                    bool v225;
                    v225 = v194 < 32l;
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
                int v229;
                v229 = v194 * 4l;
                int v230;
                v230 = v217 + v229;
                bool v231;
                v231 = 0l <= v215;
                bool v233;
                if (v231){
                    bool v232;
                    v232 = v215 < 8l;
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
                int v236;
                v236 = v215 * 128l;
                int v237;
                v237 = v230 + v236;
                assert("Tensor range check" && 0 <= v215 && v215 < 8l);
                assert("Tensor range check" && 0 <= v217 && v217 < 4l);
                int v238;
                v238 = 4l * v215;
                int v239;
                v239 = v238 + v217;
                v207[v239] = v237;
                v217 += 1l ;
            }
            v215 += 1l ;
        }
        bool v240;
        v240 = 0l <= v200;
        bool v242;
        if (v240){
            bool v241;
            v241 = v200 < 32l;
            v242 = v241;
        } else {
            v242 = false;
        }
        bool v243;
        v243 = v242 == false;
        if (v243){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v242);
        } else {
        }
        bool v245;
        v245 = 0l <= v195;
        bool v246;
        v246 = v245 && v196;
        bool v247;
        v247 = v246 == false;
        if (v247){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v246);
        } else {
        }
        int v249;
        v249 = v195 * 32l;
        int v250;
        v250 = v249 + v200;
        assert("Tensor range check" && 0 <= v194 && v194 < 32l);
        int v251;
        v251 = 0l;
        while (while_method_1(v251)){
            assert("Tensor range check" && 0 <= v251 && v251 < 8l);
            int v253;
            v253 = 128l * v251;
            int v254;
            v254 = v253 + v205;
            assert("Tensor range check" && 0 <= v251 && v251 < 8l);
            int v255;
            v255 = 4l * v251;
            int4* v256;
            v256 = reinterpret_cast<int4*>(v206 + v255);
            int4* v257;
            v257 = reinterpret_cast<int4*>(v204 + v254);
            assert("Pointer alignment check" && (unsigned long long)(v256) % 4l == 0 && (unsigned long long)(v257) % 4l == 0);
            *v257 = *v256;
            v251 += 1l ;
        }
        v200 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    __shared__ float * v258[32l];
    __shared__ float * v259[32l];
    int v260;
    v260 = threadIdx.x;
    float * v261;
    v261 = v1+v13;
    assert("Tensor range check" && 0 <= v260 && v260 < 32l);
    v258[v260] = v261;
    int v263;
    v263 = threadIdx.x;
    float * v264;
    v264 = v7+v17;
    assert("Tensor range check" && 0 <= v263 && v263 < 32l);
    v259[v263] = v264;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v266;
    v266 = threadIdx.x;
    bool v267;
    v267 = 0l <= v266;
    bool v268;
    v268 = v267 == false;
    if (v268){
        assert("The index needs to be zero or positive." && v267);
    } else {
    }
    int v270;
    v270 = v266 % 32l;
    int v271;
    v271 = v266 / 32l;
    bool v272;
    v272 = v271 < 1l;
    bool v273;
    v273 = v272 == false;
    if (v273){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v272);
    } else {
    }
    assert("Tensor range check" && 0 <= v271 && v271 < 1l);
    int v275;
    v275 = 32l * v271;
    assert("Tensor range check" && 0 <= v271 && v271 < 1l);
    int v276;
    v276 = 0l;
    while (while_method_0(v276)){
        assert("Tensor range check" && 0 <= v276 && v276 < 32l);
        int v278;
        v278 = v276 + v275;
        float * v279;
        v279 = v258[v278];
        assert("Tensor range check" && 0 <= v276 && v276 < 32l);
        float * v280;
        v280 = v259[v278];
        assert("Tensor range check" && 0 <= v270 && v270 < 32l);
        int v281;
        v281 = 4l * v270;
        float v282[32l];
        int v283[32l];
        int v284;
        v284 = 0l;
        while (while_method_1(v284)){
            assert("Tensor range check" && 0 <= v284 && v284 < 8l);
            int v286;
            v286 = 4l * v284;
            assert("Tensor range check" && 0 <= v284 && v284 < 8l);
            int v287;
            v287 = 128l * v284;
            int v288;
            v288 = v287 + v281;
            int4* v289;
            v289 = reinterpret_cast<int4*>(v279 + v288);
            int4* v290;
            v290 = reinterpret_cast<int4*>(v282 + v286);
            assert("Pointer alignment check" && (unsigned long long)(v289) % 4l == 0 && (unsigned long long)(v290) % 4l == 0);
            *v290 = *v289;
            v284 += 1l ;
        }
        int v291;
        v291 = 0l;
        while (while_method_1(v291)){
            int v293;
            v293 = 0l;
            while (while_method_2(v293)){
                bool v295;
                v295 = 0l <= v293;
                bool v297;
                if (v295){
                    bool v296;
                    v296 = v293 < 4l;
                    v297 = v296;
                } else {
                    v297 = false;
                }
                bool v298;
                v298 = v297 == false;
                if (v298){
                    assert("The indices should be inside the range of the dimension." && v297);
                } else {
                }
                bool v300;
                v300 = 0l <= v270;
                bool v302;
                if (v300){
                    bool v301;
                    v301 = v270 < 32l;
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
                v305 = v270 * 4l;
                int v306;
                v306 = v293 + v305;
                bool v307;
                v307 = 0l <= v291;
                bool v309;
                if (v307){
                    bool v308;
                    v308 = v291 < 8l;
                    v309 = v308;
                } else {
                    v309 = false;
                }
                bool v310;
                v310 = v309 == false;
                if (v310){
                    assert("The indices should be inside the range of the dimension." && v309);
                } else {
                }
                int v312;
                v312 = v291 * 128l;
                int v313;
                v313 = v306 + v312;
                assert("Tensor range check" && 0 <= v291 && v291 < 8l);
                assert("Tensor range check" && 0 <= v293 && v293 < 4l);
                int v314;
                v314 = 4l * v291;
                int v315;
                v315 = v314 + v293;
                v283[v315] = v313;
                v293 += 1l ;
            }
            v291 += 1l ;
        }
        bool v316;
        v316 = 0l <= v276;
        bool v318;
        if (v316){
            bool v317;
            v317 = v276 < 32l;
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
        bool v321;
        v321 = 0l <= v271;
        bool v322;
        v322 = v321 && v272;
        bool v323;
        v323 = v322 == false;
        if (v323){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v322);
        } else {
        }
        int v325;
        v325 = v271 * 32l;
        int v326;
        v326 = v325 + v276;
        float v327;
        v327 = 0.0f;
        int v328;
        v328 = 0l;
        while (while_method_1(v328)){
            int v330;
            v330 = 0l;
            while (while_method_2(v330)){
                assert("Tensor range check" && 0 <= v328 && v328 < 8l);
                assert("Tensor range check" && 0 <= v330 && v330 < 4l);
                int v332;
                v332 = 4l * v328;
                int v333;
                v333 = v332 + v330;
                float v334;
                v334 = v282[v333];
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
        Closure0 v340{};
        float v341;
        v341 = cooperative_groups::reduce(v339, v327, v340);
        float v342;
        v342 = v341 / 1024.0f;
        float v343[32l];
        int v344;
        v344 = 0l;
        while (while_method_1(v344)){
            int v346;
            v346 = 0l;
            while (while_method_2(v346)){
                assert("Tensor range check" && 0 <= v344 && v344 < 8l);
                assert("Tensor range check" && 0 <= v346 && v346 < 4l);
                int v348;
                v348 = 4l * v344;
                int v349;
                v349 = v348 + v346;
                float v350;
                v350 = v282[v349];
                float v351;
                v351 = v350 - v342;
                float v352;
                v352 = exp(v351);
                assert("Tensor range check" && 0 <= v344 && v344 < 8l);
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
        while (while_method_1(v354)){
            int v356;
            v356 = 0l;
            while (while_method_2(v356)){
                assert("Tensor range check" && 0 <= v354 && v354 < 8l);
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
        v364 = v363 / 32l;
        auto v365 = cooperative_groups::labeled_partition(v362,v364);
        float v366;
        v366 = cooperative_groups::reduce(v365, v353, v340);
        float v367[32l];
        int v368;
        v368 = 0l;
        while (while_method_1(v368)){
            int v370;
            v370 = 0l;
            while (while_method_2(v370)){
                assert("Tensor range check" && 0 <= v368 && v368 < 8l);
                assert("Tensor range check" && 0 <= v370 && v370 < 4l);
                int v372;
                v372 = 4l * v368;
                int v373;
                v373 = v372 + v370;
                float v374;
                v374 = v343[v373];
                bool v375;
                v375 = v366 == 0.0f;
                bool v376;
                v376 = v375 != true;
                float v378;
                if (v376){
                    float v377;
                    v377 = v374 / v366;
                    v378 = v377;
                } else {
                    v378 = 0.0009765625f;
                }
                assert("Tensor range check" && 0 <= v368 && v368 < 8l);
                assert("Tensor range check" && 0 <= v370 && v370 < 4l);
                v367[v373] = v378;
                v370 += 1l ;
            }
            v368 += 1l ;
        }
        assert("Tensor range check" && 0 <= v270 && v270 < 32l);
        int v379;
        v379 = 0l;
        while (while_method_1(v379)){
            assert("Tensor range check" && 0 <= v379 && v379 < 8l);
            int v381;
            v381 = 128l * v379;
            int v382;
            v382 = v381 + v281;
            assert("Tensor range check" && 0 <= v379 && v379 < 8l);
            int v383;
            v383 = 4l * v379;
            int4* v384;
            v384 = reinterpret_cast<int4*>(v367 + v383);
            int4* v385;
            v385 = reinterpret_cast<int4*>(v280 + v382);
            assert("Pointer alignment check" && (unsigned long long)(v384) % 4l == 0 && (unsigned long long)(v385) % 4l == 0);
            *v385 = *v384;
            v379 += 1l ;
        }
        v276 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v386[1l];
    __shared__ float * v387[32l];
    __shared__ int v388[32l];
    int v389;
    v389 = threadIdx.x;
    float * v390;
    v390 = v1+v13;
    assert("Tensor range check" && 0 <= v389 && v389 < 32l);
    v387[v389] = v390;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v392;
    v392 = threadIdx.x;
    bool v393;
    v393 = 0l <= v392;
    bool v394;
    v394 = v393 == false;
    if (v394){
        assert("The index needs to be zero or positive." && v393);
    } else {
    }
    int v396;
    v396 = v392 % 32l;
    int v397;
    v397 = v392 / 32l;
    bool v398;
    v398 = v397 < 1l;
    bool v399;
    v399 = v398 == false;
    if (v399){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v398);
    } else {
    }
    assert("Tensor range check" && 0 <= v397 && v397 < 1l);
    int v401;
    v401 = 32l * v397;
    assert("Tensor range check" && 0 <= v397 && v397 < 1l);
    int v402;
    v402 = 0l;
    while (while_method_0(v402)){
        assert("Tensor range check" && 0 <= v402 && v402 < 32l);
        int v404;
        v404 = v402 + v401;
        float * v405;
        v405 = v387[v404];
        assert("Tensor range check" && 0 <= v396 && v396 < 32l);
        int v406;
        v406 = 4l * v396;
        float v407[32l];
        int v408[32l];
        int v409;
        v409 = 0l;
        while (while_method_1(v409)){
            assert("Tensor range check" && 0 <= v409 && v409 < 8l);
            int v411;
            v411 = 4l * v409;
            assert("Tensor range check" && 0 <= v409 && v409 < 8l);
            int v412;
            v412 = 128l * v409;
            int v413;
            v413 = v412 + v406;
            int4* v414;
            v414 = reinterpret_cast<int4*>(v405 + v413);
            int4* v415;
            v415 = reinterpret_cast<int4*>(v407 + v411);
            assert("Pointer alignment check" && (unsigned long long)(v414) % 4l == 0 && (unsigned long long)(v415) % 4l == 0);
            *v415 = *v414;
            v409 += 1l ;
        }
        int v416;
        v416 = 0l;
        while (while_method_1(v416)){
            int v418;
            v418 = 0l;
            while (while_method_2(v418)){
                bool v420;
                v420 = 0l <= v418;
                bool v422;
                if (v420){
                    bool v421;
                    v421 = v418 < 4l;
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
                bool v425;
                v425 = 0l <= v396;
                bool v427;
                if (v425){
                    bool v426;
                    v426 = v396 < 32l;
                    v427 = v426;
                } else {
                    v427 = false;
                }
                bool v428;
                v428 = v427 == false;
                if (v428){
                    assert("The indices should be inside the range of the dimension." && v427);
                } else {
                }
                int v430;
                v430 = v396 * 4l;
                int v431;
                v431 = v418 + v430;
                bool v432;
                v432 = 0l <= v416;
                bool v434;
                if (v432){
                    bool v433;
                    v433 = v416 < 8l;
                    v434 = v433;
                } else {
                    v434 = false;
                }
                bool v435;
                v435 = v434 == false;
                if (v435){
                    assert("The indices should be inside the range of the dimension." && v434);
                } else {
                }
                int v437;
                v437 = v416 * 128l;
                int v438;
                v438 = v431 + v437;
                assert("Tensor range check" && 0 <= v416 && v416 < 8l);
                assert("Tensor range check" && 0 <= v418 && v418 < 4l);
                int v439;
                v439 = 4l * v416;
                int v440;
                v440 = v439 + v418;
                v408[v440] = v438;
                v418 += 1l ;
            }
            v416 += 1l ;
        }
        bool v441;
        v441 = 0l <= v402;
        bool v443;
        if (v441){
            bool v442;
            v442 = v402 < 32l;
            v443 = v442;
        } else {
            v443 = false;
        }
        bool v444;
        v444 = v443 == false;
        if (v444){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v443);
        } else {
        }
        bool v446;
        v446 = 0l <= v397;
        bool v447;
        v447 = v446 && v398;
        bool v448;
        v448 = v447 == false;
        if (v448){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v447);
        } else {
        }
        int v450;
        v450 = v397 * 32l;
        int v451;
        v451 = v450 + v402;
        bool v452[32l];
        int v453;
        v453 = 0l;
        while (while_method_1(v453)){
            int v455;
            v455 = 0l;
            while (while_method_2(v455)){
                assert("Tensor range check" && 0 <= v453 && v453 < 8l);
                assert("Tensor range check" && 0 <= v455 && v455 < 4l);
                int v457;
                v457 = 4l * v453;
                int v458;
                v458 = v457 + v455;
                float v459;
                v459 = v407[v458];
                int v460;
                v460 = v408[v458];
                bool v461;
                v461 = v460 < 11l;
                assert("Tensor range check" && 0 <= v453 && v453 < 8l);
                assert("Tensor range check" && 0 <= v455 && v455 < 4l);
                v452[v458] = v461;
                v455 += 1l ;
            }
            v453 += 1l ;
        }
        int v462[32l];
        int v463;
        v463 = 0l;
        while (while_method_1(v463)){
            int v465;
            v465 = 0l;
            while (while_method_2(v465)){
                assert("Tensor range check" && 0 <= v463 && v463 < 8l);
                assert("Tensor range check" && 0 <= v465 && v465 < 4l);
                int v467;
                v467 = 4l * v463;
                int v468;
                v468 = v467 + v465;
                bool v469;
                v469 = v452[v468];
                int v470;
                if (v469){
                    v470 = 1l;
                } else {
                    v470 = 0l;
                }
                assert("Tensor range check" && 0 <= v463 && v463 < 8l);
                assert("Tensor range check" && 0 <= v465 && v465 < 4l);
                v462[v468] = v470;
                v465 += 1l ;
            }
            v463 += 1l ;
        }
        int v471;
        v471 = 0l;
        int v472;
        v472 = 0l;
        while (while_method_1(v472)){
            int v474;
            v474 = 0l;
            while (while_method_2(v474)){
                assert("Tensor range check" && 0 <= v472 && v472 < 8l);
                assert("Tensor range check" && 0 <= v474 && v474 < 4l);
                int v476;
                v476 = 4l * v472;
                int v477;
                v477 = v476 + v474;
                int v478;
                v478 = v462[v477];
                int v479;
                v479 = v471 + v478;
                v471 = v479;
                v474 += 1l ;
            }
            v472 += 1l ;
        }
        auto v480 = cooperative_groups::coalesced_threads();
        int v481;
        v481 = threadIdx.x;
        int v482;
        v482 = v481 / 32l;
        auto v483 = cooperative_groups::labeled_partition(v480,v482);
        Closure1 v484{};
        int v485;
        v485 = cooperative_groups::reduce(v483, v471, v484);
        float v486[32l];
        int v487;
        v487 = 0l;
        while (while_method_1(v487)){
            int v489;
            v489 = 0l;
            while (while_method_2(v489)){
                assert("Tensor range check" && 0 <= v487 && v487 < 8l);
                assert("Tensor range check" && 0 <= v489 && v489 < 4l);
                int v491;
                v491 = 4l * v487;
                int v492;
                v492 = v491 + v489;
                float v493;
                v493 = v407[v492];
                bool v494;
                v494 = v452[v492];
                float v495;
                if (v494){
                    v495 = v493;
                } else {
                    v495 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v487 && v487 < 8l);
                assert("Tensor range check" && 0 <= v489 && v489 < 4l);
                v486[v492] = v495;
                v489 += 1l ;
            }
            v487 += 1l ;
        }
        float v496;
        v496 = 0.0f;
        int v497;
        v497 = 0l;
        while (while_method_1(v497)){
            int v499;
            v499 = 0l;
            while (while_method_2(v499)){
                assert("Tensor range check" && 0 <= v497 && v497 < 8l);
                assert("Tensor range check" && 0 <= v499 && v499 < 4l);
                int v501;
                v501 = 4l * v497;
                int v502;
                v502 = v501 + v499;
                float v503;
                v503 = v486[v502];
                float v504;
                v504 = v496 + v503;
                v496 = v504;
                v499 += 1l ;
            }
            v497 += 1l ;
        }
        auto v505 = cooperative_groups::coalesced_threads();
        int v506;
        v506 = threadIdx.x;
        int v507;
        v507 = v506 / 32l;
        auto v508 = cooperative_groups::labeled_partition(v505,v507);
        Closure0 v509{};
        float v510;
        v510 = cooperative_groups::reduce(v508, v496, v509);
        float v511;
        v511 = (float)v485;
        float v512;
        v512 = v510 / v511;
        float v513[32l];
        int v514;
        v514 = 0l;
        while (while_method_1(v514)){
            int v516;
            v516 = 0l;
            while (while_method_2(v516)){
                assert("Tensor range check" && 0 <= v514 && v514 < 8l);
                assert("Tensor range check" && 0 <= v516 && v516 < 4l);
                int v518;
                v518 = 4l * v514;
                int v519;
                v519 = v518 + v516;
                float v520;
                v520 = v407[v519];
                bool v521;
                v521 = v452[v519];
                float v522;
                if (v521){
                    v522 = v520;
                } else {
                    v522 = -1.0f / 0.0f;
                }
                float v523;
                v523 = v522 - v512;
                float v524;
                v524 = exp(v523);
                assert("Tensor range check" && 0 <= v514 && v514 < 8l);
                assert("Tensor range check" && 0 <= v516 && v516 < 4l);
                v513[v519] = v524;
                v516 += 1l ;
            }
            v514 += 1l ;
        }
        float v525;
        v525 = 0.0f;
        int v526;
        v526 = 0l;
        while (while_method_1(v526)){
            int v528;
            v528 = 0l;
            while (while_method_2(v528)){
                assert("Tensor range check" && 0 <= v526 && v526 < 8l);
                assert("Tensor range check" && 0 <= v528 && v528 < 4l);
                int v530;
                v530 = 4l * v526;
                int v531;
                v531 = v530 + v528;
                float v532;
                v532 = v513[v531];
                float v533;
                v533 = v525 + v532;
                v525 = v533;
                v528 += 1l ;
            }
            v526 += 1l ;
        }
        auto v534 = cooperative_groups::coalesced_threads();
        int v535;
        v535 = threadIdx.x;
        int v536;
        v536 = v535 / 32l;
        auto v537 = cooperative_groups::labeled_partition(v534,v536);
        float v538;
        v538 = cooperative_groups::reduce(v537, v525, v509);
        float v539[32l];
        int v540;
        v540 = 0l;
        while (while_method_1(v540)){
            int v542;
            v542 = 0l;
            while (while_method_2(v542)){
                assert("Tensor range check" && 0 <= v540 && v540 < 8l);
                assert("Tensor range check" && 0 <= v542 && v542 < 4l);
                int v544;
                v544 = 4l * v540;
                int v545;
                v545 = v544 + v542;
                float v546;
                v546 = v513[v545];
                bool v547;
                v547 = v538 == 0.0f;
                bool v548;
                v548 = v547 != true;
                float v550;
                if (v548){
                    float v549;
                    v549 = v546 / v538;
                    v550 = v549;
                } else {
                    v550 = 0.0009765625f;
                }
                assert("Tensor range check" && 0 <= v540 && v540 < 8l);
                assert("Tensor range check" && 0 <= v542 && v542 < 4l);
                v539[v545] = v550;
                v542 += 1l ;
            }
            v540 += 1l ;
        }
        float v551[32l];
        float v552;
        v552 = 0.0f;
        int v553;
        v553 = 0l;
        while (while_method_1(v553)){
            assert("Tensor range check" && 0 <= v553 && v553 < 8l);
            int v555;
            v555 = 4l * v553;
            assert("Tensor range check" && 0 <= v553 && v553 < 8l);
            int v556; float v557;
            Tuple1 tmp1 = Tuple1{0l, 0.0f};
            v556 = tmp1.v0; v557 = tmp1.v1;
            while (while_method_2(v556)){
                assert("Tensor range check" && 0 <= v556 && v556 < 4l);
                int v559;
                v559 = v556 + v555;
                float v560;
                v560 = v539[v559];
                float v561;
                v561 = v557 + v560;
                v557 = v561;
                v556 += 1l ;
            }
            auto v562 = cooperative_groups::coalesced_threads();
            int v563;
            v563 = threadIdx.x;
            int v564;
            v564 = v563 / 32l;
            auto v565 = cooperative_groups::labeled_partition(v562,v564);
            Closure2 v566{};
            float v567;
            v567 = cooperative_groups::inclusive_scan(v565, v557, v566);
            float v568;
            v568 = v565.shfl_up(v567,1);
            bool v569;
            v569 = v565.thread_rank() == 0;
            float v570;
            if (v569){
                v570 = 0.0f;
            } else {
                v570 = v568;
            }
            float v571;
            v571 = v565.shfl(v567,v565.num_threads()-1);
            float v572;
            v572 = v552 + v570;
            int v573; float v574;
            Tuple1 tmp2 = Tuple1{0l, v572};
            v573 = tmp2.v0; v574 = tmp2.v1;
            while (while_method_2(v573)){
                assert("Tensor range check" && 0 <= v573 && v573 < 4l);
                int v576;
                v576 = v573 + v555;
                float v577;
                v577 = v539[v576];
                float v578;
                v578 = v574 + v577;
                assert("Tensor range check" && 0 <= v573 && v573 < 4l);
                v551[v576] = v578;
                v574 = v578;
                v573 += 1l ;
            }
            float v579;
            v579 = v552 + v571;
            v552 = v579;
            v553 += 1l ;
        }
        float v580[32l];
        bool v581[32l];
        int v582;
        v582 = 0l;
        while (while_method_1(v582)){
            int v584;
            v584 = 0l;
            while (while_method_2(v584)){
                assert("Tensor range check" && 0 <= v582 && v582 < 8l);
                assert("Tensor range check" && 0 <= v584 && v584 < 4l);
                int v586;
                v586 = 4l * v582;
                int v587;
                v587 = v586 + v584;
                float v588;
                v588 = v551[v587];
                float v589;
                v589 = v539[v587];
                bool v590;
                v590 = v589 > 0.0f;
                assert("Tensor range check" && 0 <= v582 && v582 < 8l);
                assert("Tensor range check" && 0 <= v584 && v584 < 4l);
                v580[v587] = v588;
                v581[v587] = v590;
                v584 += 1l ;
            }
            v582 += 1l ;
        }
        float v591; bool v592;
        Tuple2 tmp3 = Tuple2{-1.0f / 0.0f, false};
        v591 = tmp3.v0; v592 = tmp3.v1;
        int v593;
        v593 = 0l;
        while (while_method_1(v593)){
            int v595;
            v595 = 0l;
            while (while_method_2(v595)){
                assert("Tensor range check" && 0 <= v593 && v593 < 8l);
                assert("Tensor range check" && 0 <= v595 && v595 < 4l);
                int v597;
                v597 = 4l * v593;
                int v598;
                v598 = v597 + v595;
                float v599;
                v599 = v580[v598];
                bool v600;
                v600 = v581[v598];
                float v607; bool v608;
                if (v592){
                    if (v600){
                        bool v601;
                        v601 = v591 >= v599;
                        float v602;
                        if (v601){
                            v602 = v591;
                        } else {
                            v602 = v599;
                        }
                        v607 = v602; v608 = true;
                    } else {
                        v607 = v591; v608 = v592;
                    }
                } else {
                    if (v600){
                        v607 = v599; v608 = v600;
                    } else {
                        v607 = v591; v608 = v592;
                    }
                }
                v591 = v607;
                v592 = v608;
                v595 += 1l ;
            }
            v593 += 1l ;
        }
        auto v609 = cooperative_groups::coalesced_threads();
        int v610;
        v610 = threadIdx.x;
        int v611;
        v611 = v610 / 32l;
        auto v612 = cooperative_groups::labeled_partition(v609,v611);
        Closure3 v613{};
        float v614; bool v615;
        Tuple2 tmp4 = cooperative_groups::reduce(v612, Tuple2{v591, v592}, v613);
        v614 = tmp4.v0; v615 = tmp4.v1;
        bool v616;
        v616 = v615 == false;
        if (v616){
            assert("The local reduce must be true." && v615);
        } else {
        }
        float v618[32l];
        int v619[32l];
        int v620;
        v620 = 0l;
        while (while_method_1(v620)){
            int v622;
            v622 = 0l;
            while (while_method_2(v622)){
                assert("Tensor range check" && 0 <= v620 && v620 < 8l);
                assert("Tensor range check" && 0 <= v622 && v622 < 4l);
                int v624;
                v624 = 4l * v620;
                int v625;
                v625 = v624 + v622;
                int v626;
                v626 = v408[v625];
                float v627;
                v627 = curand_uniform(&v11);
                assert("Tensor range check" && 0 <= v620 && v620 < 8l);
                assert("Tensor range check" && 0 <= v622 && v622 < 4l);
                v618[v625] = v627;
                v619[v625] = v626;
                v622 += 1l ;
            }
            v620 += 1l ;
        }
        float v628; int v629;
        Tuple3 tmp5 = Tuple3{0.0f, 2147483647l};
        v628 = tmp5.v0; v629 = tmp5.v1;
        int v630;
        v630 = 0l;
        while (while_method_1(v630)){
            int v632;
            v632 = 0l;
            while (while_method_2(v632)){
                assert("Tensor range check" && 0 <= v630 && v630 < 8l);
                assert("Tensor range check" && 0 <= v632 && v632 < 4l);
                int v634;
                v634 = 4l * v630;
                int v635;
                v635 = v634 + v632;
                float v636;
                v636 = v618[v635];
                int v637;
                v637 = v619[v635];
                bool v638;
                v638 = v629 < v637;
                float v639; int v640;
                if (v638){
                    v639 = v628; v640 = v629;
                } else {
                    v639 = v636; v640 = v637;
                }
                v628 = v639;
                v629 = v640;
                v632 += 1l ;
            }
            v630 += 1l ;
        }
        auto v641 = cooperative_groups::coalesced_threads();
        int v642;
        v642 = threadIdx.x;
        int v643;
        v643 = v642 / 32l;
        auto v644 = cooperative_groups::labeled_partition(v641,v643);
        Closure4 v645{};
        float v646; int v647;
        Tuple3 tmp6 = cooperative_groups::reduce(v644, Tuple3{v628, v629}, v645);
        v646 = tmp6.v0; v647 = tmp6.v1;
        float v648;
        v648 = v614 * v646;
        int v649[32l];
        bool v650[32l];
        int v651;
        v651 = 0l;
        while (while_method_1(v651)){
            int v653;
            v653 = 0l;
            while (while_method_2(v653)){
                assert("Tensor range check" && 0 <= v651 && v651 < 8l);
                assert("Tensor range check" && 0 <= v653 && v653 < 4l);
                int v655;
                v655 = 4l * v651;
                int v656;
                v656 = v655 + v653;
                float v657;
                v657 = v580[v656];
                bool v658;
                v658 = v581[v656];
                int v659;
                v659 = v408[v656];
                int v662; bool v663;
                if (v658){
                    float v660;
                    v660 = v657 - v648;
                    bool v661;
                    v661 = v660 >= 0.0f;
                    v662 = v659; v663 = v661;
                } else {
                    v662 = 2147483647l; v663 = false;
                }
                assert("Tensor range check" && 0 <= v651 && v651 < 8l);
                assert("Tensor range check" && 0 <= v653 && v653 < 4l);
                v649[v656] = v662;
                v650[v656] = v663;
                v653 += 1l ;
            }
            v651 += 1l ;
        }
        int v664; bool v665;
        Tuple4 tmp7 = Tuple4{2147483647l, false};
        v664 = tmp7.v0; v665 = tmp7.v1;
        int v666;
        v666 = 0l;
        while (while_method_1(v666)){
            int v668;
            v668 = 0l;
            while (while_method_2(v668)){
                assert("Tensor range check" && 0 <= v666 && v666 < 8l);
                assert("Tensor range check" && 0 <= v668 && v668 < 4l);
                int v670;
                v670 = 4l * v666;
                int v671;
                v671 = v670 + v668;
                int v672;
                v672 = v649[v671];
                bool v673;
                v673 = v650[v671];
                int v680; bool v681;
                if (v665){
                    if (v673){
                        bool v674;
                        v674 = v664 < v672;
                        int v675;
                        if (v674){
                            v675 = v664;
                        } else {
                            v675 = v672;
                        }
                        v680 = v675; v681 = true;
                    } else {
                        v680 = v664; v681 = v665;
                    }
                } else {
                    if (v673){
                        v680 = v672; v681 = v673;
                    } else {
                        v680 = v664; v681 = v665;
                    }
                }
                v664 = v680;
                v665 = v681;
                v668 += 1l ;
            }
            v666 += 1l ;
        }
        auto v682 = cooperative_groups::coalesced_threads();
        int v683;
        v683 = threadIdx.x;
        int v684;
        v684 = v683 / 32l;
        auto v685 = cooperative_groups::labeled_partition(v682,v684);
        Closure5 v686{};
        int v687; bool v688;
        Tuple4 tmp8 = cooperative_groups::reduce(v685, Tuple4{v664, v665}, v686);
        v687 = tmp8.v0; v688 = tmp8.v1;
        bool v689;
        v689 = v688 == false;
        if (v689){
            assert("The local reduce must be true." && v688);
        } else {
        }
        assert("Tensor range check" && 0 <= v402 && v402 < 32l);
        v388[v404] = v687;
        v402 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v691;
    v691 = threadIdx.x;
    assert("Tensor range check" && 0 <= v691 && v691 < 32l);
    int v692;
    v692 = v388[v691];
    v386[0l] = v692;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v693;
    v693 = v386[0l];
    int v694;
    v694 = threadIdx.x;
    assert("Tensor range check" && 0 <= v694 && v694 < 32l);
    v5[v694] = v693;
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
def main():
    v0 = cp.arange(0,32768,1,dtype=cp.float32) # type: ignore
    v1 = v0.size
    v2 = 32768 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,32768,dtype=cp.float32) # type: ignore
    v6 = cp.empty(32768,dtype=cp.int32)
    v7 = cp.empty(32768,dtype=cp.int32)
    v8 = cp.empty(32,dtype=cp.int32)
    v9 = cp.empty(32,dtype=cp.int32)
    v10 = cp.empty(32768,dtype=cp.float32)
    v11 = cp.empty(32768,dtype=cp.float32)
    v12 = 0
    v13 = raw_module.get_function(f"entry{v12}")
    del v12
    v13.max_dynamic_shared_size_bytes = 0 
    v13((1,),(32,),(v0, v5, v6, v7, v8, v9, v10, v11),shared_mem=0)
    del v0, v5, v6, v7, v8, v10, v11, v13
    v29 = 0
    v30 = "{}"
    print(v30.format('['),end="")
    v31 = 0
    while method0(v31):
        v33 = v29
        v34 = v33 >= 2147483647
        del v33
        if v34:
            v35 = " ..."
            print(v30.format(v35),end="")
            del v35
            break
        else:
            pass
        del v34
        v36 = v31 == 0
        v37 = v36 != True
        del v36
        if v37:
            v38 = "; "
            print(v30.format(v38),end="")
            del v38
        else:
            pass
        del v37
        v39 = v29 + 1
        v29 = v39
        del v39
        v40 = v9[v31].item()
        print(v30.format(v40),end="")
        del v40
        v31 += 1 
    del v9, v29, v31
    print(v30.format(']'),end="")
    del v30
    v41 = "\n"
    print(v41,end="")
    del v41
    return 

if __name__ == '__main__': print(main())
