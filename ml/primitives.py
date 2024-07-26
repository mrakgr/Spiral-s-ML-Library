kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
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
struct Closure3 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v0 >= 0.0f;
        bool v6;
        if (v4){
            bool v5;
            v5 = v2 >= 0.0f;
            v6 = v5;
        } else {
            v6 = false;
        }
        if (v6){
            bool v7;
            v7 = v0 <= v2;
            if (v7){
                return Tuple1{v0, v1};
            } else {
                return Tuple1{v2, v3};
            }
        } else {
            if (v4){
                return Tuple1{v0, v1};
            } else {
                bool v10;
                v10 = v2 >= 0.0f;
                if (v10){
                    return Tuple1{v2, v3};
                } else {
                    return Tuple1{v0, v1};
                }
            }
        }
    }
};
struct Tuple2 {
    int v0;
    int v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure5 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 1024l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
extern "C" __global__ void entry0(int * v0, float * v1, float * v2, int * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int * v9, int * v10, int * v11, int * v12, int * v13, int * v14) {
    unsigned long long v15;
    v15 = clock64();
    curandStatePhilox4_32_10_t v16;
    curand_init(v15,0ull,0ull,&v16);
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
        v23 = v18 % 64l;
        int v24;
        v24 = v18 / 64l;
        bool v25;
        v25 = v24 < 16l;
        bool v26;
        v26 = v25 == false;
        if (v26){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v25);
        } else {
        }
        assert("Tensor range check" && 0 <= v24 && v24 < 16l);
        assert("Tensor range check" && 0 <= v23 && v23 < 64l);
        int v28;
        v28 = 4l * v23;
        int v29;
        v29 = 256l * v24;
        int v30;
        v30 = v29 + v28;
        assert("Tensor range check" && 0 <= v24 && v24 < 16l);
        assert("Tensor range check" && 0 <= v23 && v23 < 64l);
        float v31[4l];
        float v32[4l];
        int4* v33;
        v33 = reinterpret_cast<int4*>(v1 + v30);
        int4* v34;
        v34 = reinterpret_cast<int4*>(v31 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v33) % 4l == 0 && (unsigned long long)(v34) % 4l == 0);
        *v34 = *v33;
        // Pushing the loop unrolling to: 0
        int v35;
        v35 = 0l;
        #pragma unroll
        while (while_method_1(v35)){
            assert("Tensor range check" && 0 <= v35 && v35 < 4l);
            float v37;
            v37 = v31[v35];
            float v38;
            v38 = 1.0f + v37;
            assert("Tensor range check" && 0 <= v35 && v35 < 4l);
            v32[v35] = v38;
            v35 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v39;
        v39 = reinterpret_cast<int4*>(v32 + 0l);
        int4* v40;
        v40 = reinterpret_cast<int4*>(v1 + v30);
        assert("Pointer alignment check" && (unsigned long long)(v39) % 4l == 0 && (unsigned long long)(v40) % 4l == 0);
        *v40 = *v39;
        v18 += 32l ;
    }
    __syncthreads();
    float v41;
    v41 = 0.0f;
    int v42;
    v42 = threadIdx.x;
    int v43;
    v43 = v42;
    while (while_method_0(v43)){
        bool v45;
        v45 = 0l <= v43;
        bool v46;
        v46 = v45 == false;
        if (v46){
            assert("The index needs to be zero or positive." && v45);
        } else {
        }
        int v48;
        v48 = v43 % 64l;
        int v49;
        v49 = v43 / 64l;
        bool v50;
        v50 = v49 < 16l;
        bool v51;
        v51 = v50 == false;
        if (v51){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v50);
        } else {
        }
        assert("Tensor range check" && 0 <= v49 && v49 < 16l);
        assert("Tensor range check" && 0 <= v48 && v48 < 64l);
        int v53;
        v53 = 4l * v48;
        int v54;
        v54 = 256l * v49;
        int v55;
        v55 = v54 + v53;
        float v56[4l];
        int4* v57;
        v57 = reinterpret_cast<int4*>(v1 + v55);
        int4* v58;
        v58 = reinterpret_cast<int4*>(v56 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v57) % 4l == 0 && (unsigned long long)(v58) % 4l == 0);
        *v58 = *v57;
        int v59; float v60;
        Tuple0 tmp0 = Tuple0{0l, v41};
        v59 = tmp0.v0; v60 = tmp0.v1;
        while (while_method_1(v59)){
            assert("Tensor range check" && 0 <= v59 && v59 < 4l);
            float v62;
            v62 = v56[v59];
            float v63;
            v63 = v60 + v62;
            v60 = v63;
            v59 += 1l ;
        }
        v41 = v60;
        v43 += 32l ;
    }
    auto v64 = cooperative_groups::coalesced_threads();
    Closure0 v65{};
    float v66;
    v66 = cooperative_groups::reduce(v64, v41, v65);
    int v67;
    v67 = threadIdx.x;
    int v68;
    v68 = v67 / 32l;
    __shared__ float v69[1l];
    assert("Tensor range check" && 0 <= v68 && v68 < 1l);
    v69[v68] = v66;
    __syncthreads();
    int v70;
    v70 = threadIdx.x;
    int v71;
    v71 = v70 % 32l;
    bool v72;
    v72 = v68 == 0l;
    bool v74;
    if (v72){
        bool v73;
        v73 = v71 < 1l;
        v74 = v73;
    } else {
        v74 = false;
    }
    if (v74){
        auto v75 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v71 && v71 < 1l);
        float v76;
        v76 = v69[v71];
        float v77;
        v77 = cooperative_groups::reduce(v75, v76, v65);
        v2[0l] = v77;
    } else {
    }
    __syncthreads();
    int v78;
    v78 = threadIdx.x;
    bool v79;
    v79 = 0l <= v78;
    bool v80;
    v80 = v79 == false;
    if (v80){
        assert("The index needs to be zero or positive." && v79);
    } else {
    }
    int v82;
    v82 = v78 % 32l;
    int v83;
    v83 = v78 / 32l;
    bool v84;
    v84 = v83 < 1l;
    bool v85;
    v85 = v84 == false;
    if (v85){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v84);
    } else {
    }
    assert("Tensor range check" && 0 <= v83 && v83 < 1l);
    assert("Tensor range check" && 0 <= v82 && v82 < 32l);
    int v87;
    v87 = 4l * v82;
    int v88;
    v88 = 256l * v83;
    int v89;
    v89 = v88 + v87;
    assert("Tensor range check" && 0 <= v83 && v83 < 1l);
    assert("Tensor range check" && 0 <= v82 && v82 < 32l);
    int v90;
    v90 = 0l;
    while (while_method_2(v90)){
        assert("Tensor range check" && 0 <= v90 && v90 < 16l);
        int v92;
        v92 = 256l * v90;
        int v93;
        v93 = v92 + v89;
        int v94[8l];
        int v95[8l];
        int v96;
        v96 = 0l;
        while (while_method_3(v96)){
            assert("Tensor range check" && 0 <= v96 && v96 < 2l);
            int v98;
            v98 = 4l * v96;
            assert("Tensor range check" && 0 <= v96 && v96 < 2l);
            int v99;
            v99 = 128l * v96;
            int v100;
            v100 = v99 + v93;
            int4* v101;
            v101 = reinterpret_cast<int4*>(v0 + v100);
            int4* v102;
            v102 = reinterpret_cast<int4*>(v94 + v98);
            assert("Pointer alignment check" && (unsigned long long)(v101) % 4l == 0 && (unsigned long long)(v102) % 4l == 0);
            *v102 = *v101;
            v96 += 1l ;
        }
        int v103;
        v103 = 0l;
        while (while_method_3(v103)){
            int v105;
            v105 = 0l;
            while (while_method_1(v105)){
                bool v107;
                v107 = 0l <= v105;
                bool v109;
                if (v107){
                    bool v108;
                    v108 = v105 < 4l;
                    v109 = v108;
                } else {
                    v109 = false;
                }
                bool v110;
                v110 = v109 == false;
                if (v110){
                    assert("The indices should be inside the range of the dimension." && v109);
                } else {
                }
                bool v112;
                v112 = 0l <= v82;
                bool v114;
                if (v112){
                    bool v113;
                    v113 = v82 < 32l;
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
                v117 = v82 * 4l;
                int v118;
                v118 = v105 + v117;
                bool v119;
                v119 = 0l <= v103;
                bool v121;
                if (v119){
                    bool v120;
                    v120 = v103 < 2l;
                    v121 = v120;
                } else {
                    v121 = false;
                }
                bool v122;
                v122 = v121 == false;
                if (v122){
                    assert("The indices should be inside the range of the dimension." && v121);
                } else {
                }
                int v124;
                v124 = v103 * 128l;
                int v125;
                v125 = v118 + v124;
                assert("Tensor range check" && 0 <= v103 && v103 < 2l);
                assert("Tensor range check" && 0 <= v105 && v105 < 4l);
                int v126;
                v126 = 4l * v103;
                int v127;
                v127 = v126 + v105;
                v95[v127] = v125;
                v105 += 1l ;
            }
            v103 += 1l ;
        }
        bool v128;
        v128 = 0l <= v83;
        bool v129;
        v129 = v128 && v84;
        bool v130;
        v130 = v129 == false;
        if (v130){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v129);
        } else {
        }
        bool v132;
        v132 = 0l <= v90;
        bool v134;
        if (v132){
            bool v133;
            v133 = v90 < 16l;
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
        v137 = v90 + v83;
        assert("Tensor range check" && 0 <= v90 && v90 < 16l);
        int v138;
        v138 = 0l;
        while (while_method_3(v138)){
            assert("Tensor range check" && 0 <= v138 && v138 < 2l);
            int v140;
            v140 = 128l * v138;
            int v141;
            v141 = v140 + v93;
            assert("Tensor range check" && 0 <= v138 && v138 < 2l);
            int v142;
            v142 = 4l * v138;
            int4* v143;
            v143 = reinterpret_cast<int4*>(v94 + v142);
            int4* v144;
            v144 = reinterpret_cast<int4*>(v3 + v141);
            assert("Pointer alignment check" && (unsigned long long)(v143) % 4l == 0 && (unsigned long long)(v144) % 4l == 0);
            *v144 = *v143;
            v138 += 1l ;
        }
        v90 += 1l ;
    }
    __syncthreads();
    int v145;
    v145 = threadIdx.x;
    bool v146;
    v146 = 0l <= v145;
    bool v147;
    v147 = v146 == false;
    if (v147){
        assert("The index needs to be zero or positive." && v146);
    } else {
    }
    int v149;
    v149 = v145 % 32l;
    int v150;
    v150 = v145 / 32l;
    bool v151;
    v151 = v150 < 1l;
    bool v152;
    v152 = v151 == false;
    if (v152){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v151);
    } else {
    }
    assert("Tensor range check" && 0 <= v150 && v150 < 1l);
    assert("Tensor range check" && 0 <= v149 && v149 < 32l);
    int v154;
    v154 = 4l * v149;
    int v155;
    v155 = 256l * v150;
    int v156;
    v156 = v155 + v154;
    assert("Tensor range check" && 0 <= v150 && v150 < 1l);
    assert("Tensor range check" && 0 <= v149 && v149 < 32l);
    int v157;
    v157 = 0l;
    while (while_method_2(v157)){
        assert("Tensor range check" && 0 <= v157 && v157 < 16l);
        int v159;
        v159 = 256l * v157;
        int v160;
        v160 = v159 + v156;
        float v161[8l];
        int v162[8l];
        int v163;
        v163 = 0l;
        while (while_method_3(v163)){
            assert("Tensor range check" && 0 <= v163 && v163 < 2l);
            int v165;
            v165 = 4l * v163;
            assert("Tensor range check" && 0 <= v163 && v163 < 2l);
            int v166;
            v166 = 128l * v163;
            int v167;
            v167 = v166 + v160;
            int4* v168;
            v168 = reinterpret_cast<int4*>(v1 + v167);
            int4* v169;
            v169 = reinterpret_cast<int4*>(v161 + v165);
            assert("Pointer alignment check" && (unsigned long long)(v168) % 4l == 0 && (unsigned long long)(v169) % 4l == 0);
            *v169 = *v168;
            v163 += 1l ;
        }
        int v170;
        v170 = 0l;
        while (while_method_3(v170)){
            int v172;
            v172 = 0l;
            while (while_method_1(v172)){
                bool v174;
                v174 = 0l <= v172;
                bool v176;
                if (v174){
                    bool v175;
                    v175 = v172 < 4l;
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
                bool v179;
                v179 = 0l <= v149;
                bool v181;
                if (v179){
                    bool v180;
                    v180 = v149 < 32l;
                    v181 = v180;
                } else {
                    v181 = false;
                }
                bool v182;
                v182 = v181 == false;
                if (v182){
                    assert("The indices should be inside the range of the dimension." && v181);
                } else {
                }
                int v184;
                v184 = v149 * 4l;
                int v185;
                v185 = v172 + v184;
                bool v186;
                v186 = 0l <= v170;
                bool v188;
                if (v186){
                    bool v187;
                    v187 = v170 < 2l;
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
                int v191;
                v191 = v170 * 128l;
                int v192;
                v192 = v185 + v191;
                assert("Tensor range check" && 0 <= v170 && v170 < 2l);
                assert("Tensor range check" && 0 <= v172 && v172 < 4l);
                int v193;
                v193 = 4l * v170;
                int v194;
                v194 = v193 + v172;
                v162[v194] = v192;
                v172 += 1l ;
            }
            v170 += 1l ;
        }
        bool v195;
        v195 = 0l <= v150;
        bool v196;
        v196 = v195 && v151;
        bool v197;
        v197 = v196 == false;
        if (v197){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v196);
        } else {
        }
        bool v199;
        v199 = 0l <= v157;
        bool v201;
        if (v199){
            bool v200;
            v200 = v157 < 16l;
            v201 = v200;
        } else {
            v201 = false;
        }
        bool v202;
        v202 = v201 == false;
        if (v202){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v201);
        } else {
        }
        int v204;
        v204 = v157 + v150;
        int v205[8l];
        int v206[8l];
        int v207;
        v207 = 0l;
        while (while_method_3(v207)){
            int v209;
            v209 = 0l;
            while (while_method_1(v209)){
                assert("Tensor range check" && 0 <= v207 && v207 < 2l);
                assert("Tensor range check" && 0 <= v209 && v209 < 4l);
                int v211;
                v211 = 4l * v207;
                int v212;
                v212 = v211 + v209;
                int v213;
                v213 = v162[v212];
                assert("Tensor range check" && 0 <= v207 && v207 < 2l);
                assert("Tensor range check" && 0 <= v209 && v209 < 4l);
                v205[v212] = v204;
                v206[v212] = v213;
                v209 += 1l ;
            }
            v207 += 1l ;
        }
        assert("Tensor range check" && 0 <= v157 && v157 < 16l);
        int v214;
        v214 = 0l;
        while (while_method_3(v214)){
            assert("Tensor range check" && 0 <= v214 && v214 < 2l);
            int v216;
            v216 = 128l * v214;
            int v217;
            v217 = v216 + v160;
            assert("Tensor range check" && 0 <= v214 && v214 < 2l);
            int v218;
            v218 = 4l * v214;
            int4* v219;
            v219 = reinterpret_cast<int4*>(v205 + v218);
            int4* v220;
            v220 = reinterpret_cast<int4*>(v11 + v217);
            assert("Pointer alignment check" && (unsigned long long)(v219) % 4l == 0 && (unsigned long long)(v220) % 4l == 0);
            *v220 = *v219;
            int4* v221;
            v221 = reinterpret_cast<int4*>(v206 + v218);
            int4* v222;
            v222 = reinterpret_cast<int4*>(v12 + v217);
            assert("Pointer alignment check" && (unsigned long long)(v221) % 4l == 0 && (unsigned long long)(v222) % 4l == 0);
            *v222 = *v221;
            v214 += 1l ;
        }
        v157 += 1l ;
    }
    __syncthreads();
    int v223;
    v223 = threadIdx.x;
    bool v224;
    v224 = 0l <= v223;
    bool v225;
    v225 = v224 == false;
    if (v225){
        assert("The index needs to be zero or positive." && v224);
    } else {
    }
    int v227;
    v227 = v223 % 32l;
    int v228;
    v228 = v223 / 32l;
    bool v229;
    v229 = v228 < 1l;
    bool v230;
    v230 = v229 == false;
    if (v230){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v229);
    } else {
    }
    assert("Tensor range check" && 0 <= v228 && v228 < 1l);
    assert("Tensor range check" && 0 <= v227 && v227 < 32l);
    int v232;
    v232 = 4l * v227;
    int v233;
    v233 = 256l * v228;
    int v234;
    v234 = v233 + v232;
    assert("Tensor range check" && 0 <= v228 && v228 < 1l);
    int v235;
    v235 = 0l;
    while (while_method_2(v235)){
        assert("Tensor range check" && 0 <= v235 && v235 < 16l);
        int v237;
        v237 = 256l * v235;
        int v238;
        v238 = v237 + v234;
        float v239[8l];
        int v240[8l];
        int v241;
        v241 = 0l;
        while (while_method_3(v241)){
            assert("Tensor range check" && 0 <= v241 && v241 < 2l);
            int v243;
            v243 = 4l * v241;
            assert("Tensor range check" && 0 <= v241 && v241 < 2l);
            int v244;
            v244 = 128l * v241;
            int v245;
            v245 = v244 + v238;
            int4* v246;
            v246 = reinterpret_cast<int4*>(v1 + v245);
            int4* v247;
            v247 = reinterpret_cast<int4*>(v239 + v243);
            assert("Pointer alignment check" && (unsigned long long)(v246) % 4l == 0 && (unsigned long long)(v247) % 4l == 0);
            *v247 = *v246;
            v241 += 1l ;
        }
        int v248;
        v248 = 0l;
        while (while_method_3(v248)){
            int v250;
            v250 = 0l;
            while (while_method_1(v250)){
                bool v252;
                v252 = 0l <= v250;
                bool v254;
                if (v252){
                    bool v253;
                    v253 = v250 < 4l;
                    v254 = v253;
                } else {
                    v254 = false;
                }
                bool v255;
                v255 = v254 == false;
                if (v255){
                    assert("The indices should be inside the range of the dimension." && v254);
                } else {
                }
                bool v257;
                v257 = 0l <= v227;
                bool v259;
                if (v257){
                    bool v258;
                    v258 = v227 < 32l;
                    v259 = v258;
                } else {
                    v259 = false;
                }
                bool v260;
                v260 = v259 == false;
                if (v260){
                    assert("The indices should be inside the range of the dimension." && v259);
                } else {
                }
                int v262;
                v262 = v227 * 4l;
                int v263;
                v263 = v250 + v262;
                bool v264;
                v264 = 0l <= v248;
                bool v266;
                if (v264){
                    bool v265;
                    v265 = v248 < 2l;
                    v266 = v265;
                } else {
                    v266 = false;
                }
                bool v267;
                v267 = v266 == false;
                if (v267){
                    assert("The indices should be inside the range of the dimension." && v266);
                } else {
                }
                int v269;
                v269 = v248 * 128l;
                int v270;
                v270 = v263 + v269;
                assert("Tensor range check" && 0 <= v248 && v248 < 2l);
                assert("Tensor range check" && 0 <= v250 && v250 < 4l);
                int v271;
                v271 = 4l * v248;
                int v272;
                v272 = v271 + v250;
                v240[v272] = v270;
                v250 += 1l ;
            }
            v248 += 1l ;
        }
        bool v273;
        v273 = 0l <= v228;
        bool v274;
        v274 = v273 && v229;
        bool v275;
        v275 = v274 == false;
        if (v275){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v274);
        } else {
        }
        bool v277;
        v277 = 0l <= v235;
        bool v279;
        if (v277){
            bool v278;
            v278 = v235 < 16l;
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
        v282 = v235 + v228;
        assert("Tensor range check" && 0 <= v235 && v235 < 16l);
        v13[v282] = v282;
        v235 += 1l ;
    }
    __syncthreads();
    int v283;
    v283 = threadIdx.x;
    bool v284;
    v284 = 0l <= v283;
    bool v285;
    v285 = v284 == false;
    if (v285){
        assert("The index needs to be zero or positive." && v284);
    } else {
    }
    int v287;
    v287 = v283 % 32l;
    int v288;
    v288 = v283 / 32l;
    bool v289;
    v289 = v288 < 1l;
    bool v290;
    v290 = v289 == false;
    if (v290){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v289);
    } else {
    }
    assert("Tensor range check" && 0 <= v288 && v288 < 1l);
    assert("Tensor range check" && 0 <= v287 && v287 < 32l);
    int v292;
    v292 = 4l * v287;
    int v293;
    v293 = 256l * v288;
    int v294;
    v294 = v293 + v292;
    assert("Tensor range check" && 0 <= v288 && v288 < 1l);
    assert("Tensor range check" && 0 <= v287 && v287 < 32l);
    int v295;
    v295 = 0l;
    while (while_method_2(v295)){
        assert("Tensor range check" && 0 <= v295 && v295 < 16l);
        int v297;
        v297 = 256l * v295;
        int v298;
        v298 = v297 + v294;
        float v299[8l];
        int v300[8l];
        int v301;
        v301 = 0l;
        while (while_method_3(v301)){
            assert("Tensor range check" && 0 <= v301 && v301 < 2l);
            int v303;
            v303 = 4l * v301;
            assert("Tensor range check" && 0 <= v301 && v301 < 2l);
            int v304;
            v304 = 128l * v301;
            int v305;
            v305 = v304 + v298;
            int4* v306;
            v306 = reinterpret_cast<int4*>(v1 + v305);
            int4* v307;
            v307 = reinterpret_cast<int4*>(v299 + v303);
            assert("Pointer alignment check" && (unsigned long long)(v306) % 4l == 0 && (unsigned long long)(v307) % 4l == 0);
            *v307 = *v306;
            v301 += 1l ;
        }
        int v308;
        v308 = 0l;
        while (while_method_3(v308)){
            int v310;
            v310 = 0l;
            while (while_method_1(v310)){
                bool v312;
                v312 = 0l <= v310;
                bool v314;
                if (v312){
                    bool v313;
                    v313 = v310 < 4l;
                    v314 = v313;
                } else {
                    v314 = false;
                }
                bool v315;
                v315 = v314 == false;
                if (v315){
                    assert("The indices should be inside the range of the dimension." && v314);
                } else {
                }
                bool v317;
                v317 = 0l <= v287;
                bool v319;
                if (v317){
                    bool v318;
                    v318 = v287 < 32l;
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
                v322 = v287 * 4l;
                int v323;
                v323 = v310 + v322;
                bool v324;
                v324 = 0l <= v308;
                bool v326;
                if (v324){
                    bool v325;
                    v325 = v308 < 2l;
                    v326 = v325;
                } else {
                    v326 = false;
                }
                bool v327;
                v327 = v326 == false;
                if (v327){
                    assert("The indices should be inside the range of the dimension." && v326);
                } else {
                }
                int v329;
                v329 = v308 * 128l;
                int v330;
                v330 = v323 + v329;
                assert("Tensor range check" && 0 <= v308 && v308 < 2l);
                assert("Tensor range check" && 0 <= v310 && v310 < 4l);
                int v331;
                v331 = 4l * v308;
                int v332;
                v332 = v331 + v310;
                v300[v332] = v330;
                v310 += 1l ;
            }
            v308 += 1l ;
        }
        bool v333;
        v333 = 0l <= v288;
        bool v334;
        v334 = v333 && v289;
        bool v335;
        v335 = v334 == false;
        if (v335){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v334);
        } else {
        }
        bool v337;
        v337 = 0l <= v295;
        bool v339;
        if (v337){
            bool v338;
            v338 = v295 < 16l;
            v339 = v338;
        } else {
            v339 = false;
        }
        bool v340;
        v340 = v339 == false;
        if (v340){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v339);
        } else {
        }
        int v342;
        v342 = v295 + v288;
        float v343;
        v343 = 0.0f;
        int v344;
        v344 = 0l;
        while (while_method_3(v344)){
            int v346;
            v346 = 0l;
            while (while_method_1(v346)){
                assert("Tensor range check" && 0 <= v344 && v344 < 2l);
                assert("Tensor range check" && 0 <= v346 && v346 < 4l);
                int v348;
                v348 = 4l * v344;
                int v349;
                v349 = v348 + v346;
                float v350;
                v350 = v299[v349];
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
        v356 = cooperative_groups::reduce(v355, v343, v65);
        float v357;
        v357 = v356 / 256.0f;
        float v358[8l];
        int v359;
        v359 = 0l;
        while (while_method_3(v359)){
            int v361;
            v361 = 0l;
            while (while_method_1(v361)){
                assert("Tensor range check" && 0 <= v359 && v359 < 2l);
                assert("Tensor range check" && 0 <= v361 && v361 < 4l);
                int v363;
                v363 = 4l * v359;
                int v364;
                v364 = v363 + v361;
                float v365;
                v365 = v299[v364];
                float v366;
                v366 = v365 - v357;
                float v367;
                v367 = exp(v366);
                assert("Tensor range check" && 0 <= v359 && v359 < 2l);
                assert("Tensor range check" && 0 <= v361 && v361 < 4l);
                v358[v364] = v367;
                v361 += 1l ;
            }
            v359 += 1l ;
        }
        float v368;
        v368 = 0.0f;
        int v369;
        v369 = 0l;
        while (while_method_3(v369)){
            int v371;
            v371 = 0l;
            while (while_method_1(v371)){
                assert("Tensor range check" && 0 <= v369 && v369 < 2l);
                assert("Tensor range check" && 0 <= v371 && v371 < 4l);
                int v373;
                v373 = 4l * v369;
                int v374;
                v374 = v373 + v371;
                float v375;
                v375 = v358[v374];
                float v376;
                v376 = v368 + v375;
                v368 = v376;
                v371 += 1l ;
            }
            v369 += 1l ;
        }
        auto v377 = cooperative_groups::coalesced_threads();
        int v378;
        v378 = threadIdx.x;
        int v379;
        v379 = v378 / 32l;
        auto v380 = cooperative_groups::labeled_partition(v377,v379);
        float v381;
        v381 = cooperative_groups::reduce(v380, v368, v65);
        float v382[8l];
        int v383;
        v383 = 0l;
        while (while_method_3(v383)){
            int v385;
            v385 = 0l;
            while (while_method_1(v385)){
                assert("Tensor range check" && 0 <= v383 && v383 < 2l);
                assert("Tensor range check" && 0 <= v385 && v385 < 4l);
                int v387;
                v387 = 4l * v383;
                int v388;
                v388 = v387 + v385;
                float v389;
                v389 = v358[v388];
                bool v390;
                v390 = v381 == 0.0f;
                bool v391;
                v391 = v390 != true;
                float v393;
                if (v391){
                    float v392;
                    v392 = v389 / v381;
                    v393 = v392;
                } else {
                    v393 = 0.00390625f;
                }
                assert("Tensor range check" && 0 <= v383 && v383 < 2l);
                assert("Tensor range check" && 0 <= v385 && v385 < 4l);
                v382[v388] = v393;
                v385 += 1l ;
            }
            v383 += 1l ;
        }
        assert("Tensor range check" && 0 <= v295 && v295 < 16l);
        int v394;
        v394 = 0l;
        while (while_method_3(v394)){
            assert("Tensor range check" && 0 <= v394 && v394 < 2l);
            int v396;
            v396 = 128l * v394;
            int v397;
            v397 = v396 + v298;
            assert("Tensor range check" && 0 <= v394 && v394 < 2l);
            int v398;
            v398 = 4l * v394;
            int4* v399;
            v399 = reinterpret_cast<int4*>(v382 + v398);
            int4* v400;
            v400 = reinterpret_cast<int4*>(v4 + v397);
            assert("Pointer alignment check" && (unsigned long long)(v399) % 4l == 0 && (unsigned long long)(v400) % 4l == 0);
            *v400 = *v399;
            v394 += 1l ;
        }
        v295 += 1l ;
    }
    __syncthreads();
    int v401;
    v401 = threadIdx.x;
    bool v402;
    v402 = 0l <= v401;
    bool v403;
    v403 = v402 == false;
    if (v403){
        assert("The index needs to be zero or positive." && v402);
    } else {
    }
    int v405;
    v405 = v401 % 32l;
    int v406;
    v406 = v401 / 32l;
    bool v407;
    v407 = v406 < 1l;
    bool v408;
    v408 = v407 == false;
    if (v408){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v407);
    } else {
    }
    assert("Tensor range check" && 0 <= v406 && v406 < 1l);
    assert("Tensor range check" && 0 <= v405 && v405 < 32l);
    int v410;
    v410 = 4l * v405;
    int v411;
    v411 = 256l * v406;
    int v412;
    v412 = v411 + v410;
    assert("Tensor range check" && 0 <= v406 && v406 < 1l);
    assert("Tensor range check" && 0 <= v405 && v405 < 32l);
    int v413;
    v413 = 0l;
    while (while_method_2(v413)){
        assert("Tensor range check" && 0 <= v413 && v413 < 16l);
        int v415;
        v415 = 256l * v413;
        int v416;
        v416 = v415 + v412;
        float v417[8l];
        int v418[8l];
        int v419;
        v419 = 0l;
        while (while_method_3(v419)){
            assert("Tensor range check" && 0 <= v419 && v419 < 2l);
            int v421;
            v421 = 4l * v419;
            assert("Tensor range check" && 0 <= v419 && v419 < 2l);
            int v422;
            v422 = 128l * v419;
            int v423;
            v423 = v422 + v416;
            int4* v424;
            v424 = reinterpret_cast<int4*>(v1 + v423);
            int4* v425;
            v425 = reinterpret_cast<int4*>(v417 + v421);
            assert("Pointer alignment check" && (unsigned long long)(v424) % 4l == 0 && (unsigned long long)(v425) % 4l == 0);
            *v425 = *v424;
            v419 += 1l ;
        }
        int v426;
        v426 = 0l;
        while (while_method_3(v426)){
            int v428;
            v428 = 0l;
            while (while_method_1(v428)){
                bool v430;
                v430 = 0l <= v428;
                bool v432;
                if (v430){
                    bool v431;
                    v431 = v428 < 4l;
                    v432 = v431;
                } else {
                    v432 = false;
                }
                bool v433;
                v433 = v432 == false;
                if (v433){
                    assert("The indices should be inside the range of the dimension." && v432);
                } else {
                }
                bool v435;
                v435 = 0l <= v405;
                bool v437;
                if (v435){
                    bool v436;
                    v436 = v405 < 32l;
                    v437 = v436;
                } else {
                    v437 = false;
                }
                bool v438;
                v438 = v437 == false;
                if (v438){
                    assert("The indices should be inside the range of the dimension." && v437);
                } else {
                }
                int v440;
                v440 = v405 * 4l;
                int v441;
                v441 = v428 + v440;
                bool v442;
                v442 = 0l <= v426;
                bool v444;
                if (v442){
                    bool v443;
                    v443 = v426 < 2l;
                    v444 = v443;
                } else {
                    v444 = false;
                }
                bool v445;
                v445 = v444 == false;
                if (v445){
                    assert("The indices should be inside the range of the dimension." && v444);
                } else {
                }
                int v447;
                v447 = v426 * 128l;
                int v448;
                v448 = v441 + v447;
                assert("Tensor range check" && 0 <= v426 && v426 < 2l);
                assert("Tensor range check" && 0 <= v428 && v428 < 4l);
                int v449;
                v449 = 4l * v426;
                int v450;
                v450 = v449 + v428;
                v418[v450] = v448;
                v428 += 1l ;
            }
            v426 += 1l ;
        }
        bool v451;
        v451 = 0l <= v406;
        bool v452;
        v452 = v451 && v407;
        bool v453;
        v453 = v452 == false;
        if (v453){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v452);
        } else {
        }
        bool v455;
        v455 = 0l <= v413;
        bool v457;
        if (v455){
            bool v456;
            v456 = v413 < 16l;
            v457 = v456;
        } else {
            v457 = false;
        }
        bool v458;
        v458 = v457 == false;
        if (v458){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v457);
        } else {
        }
        int v460;
        v460 = v413 + v406;
        float v461[8l];
        int v462;
        v462 = 0l;
        while (while_method_3(v462)){
            int v464;
            v464 = 0l;
            while (while_method_1(v464)){
                assert("Tensor range check" && 0 <= v462 && v462 < 2l);
                assert("Tensor range check" && 0 <= v464 && v464 < 4l);
                int v466;
                v466 = 4l * v462;
                int v467;
                v467 = v466 + v464;
                float v468;
                v468 = v417[v467];
                float v469;
                v469 = v468 * v468;
                assert("Tensor range check" && 0 <= v462 && v462 < 2l);
                assert("Tensor range check" && 0 <= v464 && v464 < 4l);
                v461[v467] = v469;
                v464 += 1l ;
            }
            v462 += 1l ;
        }
        float v470;
        v470 = 0.0f;
        int v471;
        v471 = 0l;
        while (while_method_3(v471)){
            int v473;
            v473 = 0l;
            while (while_method_1(v473)){
                assert("Tensor range check" && 0 <= v471 && v471 < 2l);
                assert("Tensor range check" && 0 <= v473 && v473 < 4l);
                int v475;
                v475 = 4l * v471;
                int v476;
                v476 = v475 + v473;
                float v477;
                v477 = v461[v476];
                float v478;
                v478 = v470 + v477;
                v470 = v478;
                v473 += 1l ;
            }
            v471 += 1l ;
        }
        auto v479 = cooperative_groups::coalesced_threads();
        int v480;
        v480 = threadIdx.x;
        int v481;
        v481 = v480 / 32l;
        auto v482 = cooperative_groups::labeled_partition(v479,v481);
        float v483;
        v483 = cooperative_groups::reduce(v482, v470, v65);
        float v484[8l];
        int v485;
        v485 = 0l;
        while (while_method_3(v485)){
            int v487;
            v487 = 0l;
            while (while_method_1(v487)){
                assert("Tensor range check" && 0 <= v485 && v485 < 2l);
                assert("Tensor range check" && 0 <= v487 && v487 < 4l);
                int v489;
                v489 = 4l * v485;
                int v490;
                v490 = v489 + v487;
                float v491;
                v491 = v417[v490];
                bool v492;
                v492 = v483 == 0.0f;
                bool v493;
                v493 = v492 != true;
                float v495;
                if (v493){
                    float v494;
                    v494 = v491 / v483;
                    v495 = v494;
                } else {
                    v495 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v485 && v485 < 2l);
                assert("Tensor range check" && 0 <= v487 && v487 < 4l);
                v484[v490] = v495;
                v487 += 1l ;
            }
            v485 += 1l ;
        }
        assert("Tensor range check" && 0 <= v413 && v413 < 16l);
        int v496;
        v496 = 0l;
        while (while_method_3(v496)){
            assert("Tensor range check" && 0 <= v496 && v496 < 2l);
            int v498;
            v498 = 128l * v496;
            int v499;
            v499 = v498 + v416;
            assert("Tensor range check" && 0 <= v496 && v496 < 2l);
            int v500;
            v500 = 4l * v496;
            int4* v501;
            v501 = reinterpret_cast<int4*>(v484 + v500);
            int4* v502;
            v502 = reinterpret_cast<int4*>(v8 + v499);
            assert("Pointer alignment check" && (unsigned long long)(v501) % 4l == 0 && (unsigned long long)(v502) % 4l == 0);
            *v502 = *v501;
            v496 += 1l ;
        }
        v413 += 1l ;
    }
    __syncthreads();
    int v503;
    v503 = threadIdx.x;
    bool v504;
    v504 = 0l <= v503;
    bool v505;
    v505 = v504 == false;
    if (v505){
        assert("The index needs to be zero or positive." && v504);
    } else {
    }
    int v507;
    v507 = v503 % 32l;
    int v508;
    v508 = v503 / 32l;
    bool v509;
    v509 = v508 < 1l;
    bool v510;
    v510 = v509 == false;
    if (v510){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v509);
    } else {
    }
    assert("Tensor range check" && 0 <= v508 && v508 < 1l);
    assert("Tensor range check" && 0 <= v507 && v507 < 32l);
    int v512;
    v512 = 4l * v507;
    int v513;
    v513 = 256l * v508;
    int v514;
    v514 = v513 + v512;
    assert("Tensor range check" && 0 <= v508 && v508 < 1l);
    int v515;
    v515 = 0l;
    while (while_method_2(v515)){
        assert("Tensor range check" && 0 <= v515 && v515 < 16l);
        int v517;
        v517 = 256l * v515;
        int v518;
        v518 = v517 + v514;
        float v519[8l];
        int v520[8l];
        int v521;
        v521 = 0l;
        while (while_method_3(v521)){
            assert("Tensor range check" && 0 <= v521 && v521 < 2l);
            int v523;
            v523 = 4l * v521;
            assert("Tensor range check" && 0 <= v521 && v521 < 2l);
            int v524;
            v524 = 128l * v521;
            int v525;
            v525 = v524 + v518;
            int4* v526;
            v526 = reinterpret_cast<int4*>(v1 + v525);
            int4* v527;
            v527 = reinterpret_cast<int4*>(v519 + v523);
            assert("Pointer alignment check" && (unsigned long long)(v526) % 4l == 0 && (unsigned long long)(v527) % 4l == 0);
            *v527 = *v526;
            v521 += 1l ;
        }
        int v528;
        v528 = 0l;
        while (while_method_3(v528)){
            int v530;
            v530 = 0l;
            while (while_method_1(v530)){
                bool v532;
                v532 = 0l <= v530;
                bool v534;
                if (v532){
                    bool v533;
                    v533 = v530 < 4l;
                    v534 = v533;
                } else {
                    v534 = false;
                }
                bool v535;
                v535 = v534 == false;
                if (v535){
                    assert("The indices should be inside the range of the dimension." && v534);
                } else {
                }
                bool v537;
                v537 = 0l <= v507;
                bool v539;
                if (v537){
                    bool v538;
                    v538 = v507 < 32l;
                    v539 = v538;
                } else {
                    v539 = false;
                }
                bool v540;
                v540 = v539 == false;
                if (v540){
                    assert("The indices should be inside the range of the dimension." && v539);
                } else {
                }
                int v542;
                v542 = v507 * 4l;
                int v543;
                v543 = v530 + v542;
                bool v544;
                v544 = 0l <= v528;
                bool v546;
                if (v544){
                    bool v545;
                    v545 = v528 < 2l;
                    v546 = v545;
                } else {
                    v546 = false;
                }
                bool v547;
                v547 = v546 == false;
                if (v547){
                    assert("The indices should be inside the range of the dimension." && v546);
                } else {
                }
                int v549;
                v549 = v528 * 128l;
                int v550;
                v550 = v543 + v549;
                assert("Tensor range check" && 0 <= v528 && v528 < 2l);
                assert("Tensor range check" && 0 <= v530 && v530 < 4l);
                int v551;
                v551 = 4l * v528;
                int v552;
                v552 = v551 + v530;
                v520[v552] = v550;
                v530 += 1l ;
            }
            v528 += 1l ;
        }
        bool v553;
        v553 = 0l <= v508;
        bool v554;
        v554 = v553 && v509;
        bool v555;
        v555 = v554 == false;
        if (v555){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v554);
        } else {
        }
        bool v557;
        v557 = 0l <= v515;
        bool v559;
        if (v557){
            bool v558;
            v558 = v515 < 16l;
            v559 = v558;
        } else {
            v559 = false;
        }
        bool v560;
        v560 = v559 == false;
        if (v560){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v559);
        } else {
        }
        int v562;
        v562 = v515 + v508;
        float v563; int v564;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0l};
        v563 = tmp1.v0; v564 = tmp1.v1;
        int v565;
        v565 = 0l;
        while (while_method_3(v565)){
            int v567;
            v567 = 0l;
            while (while_method_1(v567)){
                assert("Tensor range check" && 0 <= v565 && v565 < 2l);
                assert("Tensor range check" && 0 <= v567 && v567 < 4l);
                int v569;
                v569 = 4l * v565;
                int v570;
                v570 = v569 + v567;
                float v571;
                v571 = v519[v570];
                int v572;
                v572 = v520[v570];
                bool v573;
                v573 = v563 > v571;
                float v574; int v575;
                if (v573){
                    v574 = v563; v575 = v564;
                } else {
                    v574 = v571; v575 = v572;
                }
                v563 = v574;
                v564 = v575;
                v567 += 1l ;
            }
            v565 += 1l ;
        }
        auto v576 = cooperative_groups::coalesced_threads();
        int v577;
        v577 = threadIdx.x;
        int v578;
        v578 = v577 / 32l;
        auto v579 = cooperative_groups::labeled_partition(v576,v578);
        Closure1 v580{};
        float v581; int v582;
        Tuple1 tmp2 = cooperative_groups::reduce(v579, Tuple1{v563, v564}, v580);
        v581 = tmp2.v0; v582 = tmp2.v1;
        assert("Tensor range check" && 0 <= v515 && v515 < 16l);
        v9[v562] = v582;
        v515 += 1l ;
    }
    __syncthreads();
    int v583;
    v583 = threadIdx.x;
    bool v584;
    v584 = 0l <= v583;
    bool v585;
    v585 = v584 == false;
    if (v585){
        assert("The index needs to be zero or positive." && v584);
    } else {
    }
    int v587;
    v587 = v583 % 32l;
    int v588;
    v588 = v583 / 32l;
    bool v589;
    v589 = v588 < 1l;
    bool v590;
    v590 = v589 == false;
    if (v590){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v589);
    } else {
    }
    assert("Tensor range check" && 0 <= v588 && v588 < 1l);
    assert("Tensor range check" && 0 <= v587 && v587 < 32l);
    int v592;
    v592 = 4l * v587;
    int v593;
    v593 = 256l * v588;
    int v594;
    v594 = v593 + v592;
    assert("Tensor range check" && 0 <= v588 && v588 < 1l);
    assert("Tensor range check" && 0 <= v587 && v587 < 32l);
    int v595;
    v595 = 0l;
    while (while_method_2(v595)){
        assert("Tensor range check" && 0 <= v595 && v595 < 16l);
        int v597;
        v597 = 256l * v595;
        int v598;
        v598 = v597 + v594;
        float v599[8l];
        int v600[8l];
        int v601;
        v601 = 0l;
        while (while_method_3(v601)){
            assert("Tensor range check" && 0 <= v601 && v601 < 2l);
            int v603;
            v603 = 4l * v601;
            assert("Tensor range check" && 0 <= v601 && v601 < 2l);
            int v604;
            v604 = 128l * v601;
            int v605;
            v605 = v604 + v598;
            int4* v606;
            v606 = reinterpret_cast<int4*>(v1 + v605);
            int4* v607;
            v607 = reinterpret_cast<int4*>(v599 + v603);
            assert("Pointer alignment check" && (unsigned long long)(v606) % 4l == 0 && (unsigned long long)(v607) % 4l == 0);
            *v607 = *v606;
            v601 += 1l ;
        }
        int v608;
        v608 = 0l;
        while (while_method_3(v608)){
            int v610;
            v610 = 0l;
            while (while_method_1(v610)){
                bool v612;
                v612 = 0l <= v610;
                bool v614;
                if (v612){
                    bool v613;
                    v613 = v610 < 4l;
                    v614 = v613;
                } else {
                    v614 = false;
                }
                bool v615;
                v615 = v614 == false;
                if (v615){
                    assert("The indices should be inside the range of the dimension." && v614);
                } else {
                }
                bool v617;
                v617 = 0l <= v587;
                bool v619;
                if (v617){
                    bool v618;
                    v618 = v587 < 32l;
                    v619 = v618;
                } else {
                    v619 = false;
                }
                bool v620;
                v620 = v619 == false;
                if (v620){
                    assert("The indices should be inside the range of the dimension." && v619);
                } else {
                }
                int v622;
                v622 = v587 * 4l;
                int v623;
                v623 = v610 + v622;
                bool v624;
                v624 = 0l <= v608;
                bool v626;
                if (v624){
                    bool v625;
                    v625 = v608 < 2l;
                    v626 = v625;
                } else {
                    v626 = false;
                }
                bool v627;
                v627 = v626 == false;
                if (v627){
                    assert("The indices should be inside the range of the dimension." && v626);
                } else {
                }
                int v629;
                v629 = v608 * 128l;
                int v630;
                v630 = v623 + v629;
                assert("Tensor range check" && 0 <= v608 && v608 < 2l);
                assert("Tensor range check" && 0 <= v610 && v610 < 4l);
                int v631;
                v631 = 4l * v608;
                int v632;
                v632 = v631 + v610;
                v600[v632] = v630;
                v610 += 1l ;
            }
            v608 += 1l ;
        }
        bool v633;
        v633 = 0l <= v588;
        bool v634;
        v634 = v633 && v589;
        bool v635;
        v635 = v634 == false;
        if (v635){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v634);
        } else {
        }
        bool v637;
        v637 = 0l <= v595;
        bool v639;
        if (v637){
            bool v638;
            v638 = v595 < 16l;
            v639 = v638;
        } else {
            v639 = false;
        }
        bool v640;
        v640 = v639 == false;
        if (v640){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v639);
        } else {
        }
        int v642;
        v642 = v595 + v588;
        float v643;
        v643 = 0.0f;
        int v644;
        v644 = 0l;
        while (while_method_3(v644)){
            int v646;
            v646 = 0l;
            while (while_method_1(v646)){
                assert("Tensor range check" && 0 <= v644 && v644 < 2l);
                assert("Tensor range check" && 0 <= v646 && v646 < 4l);
                int v648;
                v648 = 4l * v644;
                int v649;
                v649 = v648 + v646;
                float v650;
                v650 = v599[v649];
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
        v656 = cooperative_groups::reduce(v655, v643, v65);
        float v657;
        v657 = v656 / 256.0f;
        float v658[8l];
        int v659;
        v659 = 0l;
        while (while_method_3(v659)){
            int v661;
            v661 = 0l;
            while (while_method_1(v661)){
                assert("Tensor range check" && 0 <= v659 && v659 < 2l);
                assert("Tensor range check" && 0 <= v661 && v661 < 4l);
                int v663;
                v663 = 4l * v659;
                int v664;
                v664 = v663 + v661;
                float v665;
                v665 = v599[v664];
                float v666;
                v666 = v665 - v657;
                float v667;
                v667 = exp(v666);
                assert("Tensor range check" && 0 <= v659 && v659 < 2l);
                assert("Tensor range check" && 0 <= v661 && v661 < 4l);
                v658[v664] = v667;
                v661 += 1l ;
            }
            v659 += 1l ;
        }
        float v668;
        v668 = 0.0f;
        int v669;
        v669 = 0l;
        while (while_method_3(v669)){
            int v671;
            v671 = 0l;
            while (while_method_1(v671)){
                assert("Tensor range check" && 0 <= v669 && v669 < 2l);
                assert("Tensor range check" && 0 <= v671 && v671 < 4l);
                int v673;
                v673 = 4l * v669;
                int v674;
                v674 = v673 + v671;
                float v675;
                v675 = v658[v674];
                float v676;
                v676 = v668 + v675;
                v668 = v676;
                v671 += 1l ;
            }
            v669 += 1l ;
        }
        auto v677 = cooperative_groups::coalesced_threads();
        int v678;
        v678 = threadIdx.x;
        int v679;
        v679 = v678 / 32l;
        auto v680 = cooperative_groups::labeled_partition(v677,v679);
        float v681;
        v681 = cooperative_groups::reduce(v680, v668, v65);
        float v682[8l];
        int v683;
        v683 = 0l;
        while (while_method_3(v683)){
            int v685;
            v685 = 0l;
            while (while_method_1(v685)){
                assert("Tensor range check" && 0 <= v683 && v683 < 2l);
                assert("Tensor range check" && 0 <= v685 && v685 < 4l);
                int v687;
                v687 = 4l * v683;
                int v688;
                v688 = v687 + v685;
                float v689;
                v689 = v658[v688];
                bool v690;
                v690 = v681 == 0.0f;
                bool v691;
                v691 = v690 != true;
                float v693;
                if (v691){
                    float v692;
                    v692 = v689 / v681;
                    v693 = v692;
                } else {
                    v693 = 0.00390625f;
                }
                assert("Tensor range check" && 0 <= v683 && v683 < 2l);
                assert("Tensor range check" && 0 <= v685 && v685 < 4l);
                v682[v688] = v693;
                v685 += 1l ;
            }
            v683 += 1l ;
        }
        float v694[8l];
        float v695;
        v695 = 0.0f;
        int v696;
        v696 = 0l;
        while (while_method_3(v696)){
            assert("Tensor range check" && 0 <= v696 && v696 < 2l);
            int v698;
            v698 = 4l * v696;
            assert("Tensor range check" && 0 <= v696 && v696 < 2l);
            int v699; float v700;
            Tuple0 tmp3 = Tuple0{0l, 0.0f};
            v699 = tmp3.v0; v700 = tmp3.v1;
            while (while_method_1(v699)){
                assert("Tensor range check" && 0 <= v699 && v699 < 4l);
                int v702;
                v702 = v699 + v698;
                float v703;
                v703 = v682[v702];
                float v704;
                v704 = v700 + v703;
                v700 = v704;
                v699 += 1l ;
            }
            auto v705 = cooperative_groups::coalesced_threads();
            int v706;
            v706 = threadIdx.x;
            int v707;
            v707 = v706 / 32l;
            auto v708 = cooperative_groups::labeled_partition(v705,v707);
            Closure2 v709{};
            float v710;
            v710 = cooperative_groups::inclusive_scan(v708, v700, v709);
            float v711;
            v711 = v708.shfl_up(v710,1);
            bool v712;
            v712 = v708.thread_rank() == 0;
            float v713;
            if (v712){
                v713 = 0.0f;
            } else {
                v713 = v711;
            }
            float v714;
            v714 = v708.shfl(v710,v708.num_threads()-1);
            float v715;
            v715 = v695 + v713;
            int v716; float v717;
            Tuple0 tmp4 = Tuple0{0l, v715};
            v716 = tmp4.v0; v717 = tmp4.v1;
            while (while_method_1(v716)){
                assert("Tensor range check" && 0 <= v716 && v716 < 4l);
                int v719;
                v719 = v716 + v698;
                float v720;
                v720 = v682[v719];
                float v721;
                v721 = v717 + v720;
                assert("Tensor range check" && 0 <= v716 && v716 < 4l);
                v694[v719] = v721;
                v717 = v721;
                v716 += 1l ;
            }
            float v722;
            v722 = v695 + v714;
            v695 = v722;
            v696 += 1l ;
        }
        assert("Tensor range check" && 0 <= v595 && v595 < 16l);
        int v723;
        v723 = 0l;
        while (while_method_3(v723)){
            assert("Tensor range check" && 0 <= v723 && v723 < 2l);
            int v725;
            v725 = 128l * v723;
            int v726;
            v726 = v725 + v598;
            assert("Tensor range check" && 0 <= v723 && v723 < 2l);
            int v727;
            v727 = 4l * v723;
            int4* v728;
            v728 = reinterpret_cast<int4*>(v682 + v727);
            int4* v729;
            v729 = reinterpret_cast<int4*>(v6 + v726);
            assert("Pointer alignment check" && (unsigned long long)(v728) % 4l == 0 && (unsigned long long)(v729) % 4l == 0);
            *v729 = *v728;
            int4* v730;
            v730 = reinterpret_cast<int4*>(v694 + v727);
            int4* v731;
            v731 = reinterpret_cast<int4*>(v7 + v726);
            assert("Pointer alignment check" && (unsigned long long)(v730) % 4l == 0 && (unsigned long long)(v731) % 4l == 0);
            *v731 = *v730;
            v723 += 1l ;
        }
        v595 += 1l ;
    }
    __syncthreads();
    int v732;
    v732 = threadIdx.x;
    bool v733;
    v733 = 0l <= v732;
    bool v734;
    v734 = v733 == false;
    if (v734){
        assert("The index needs to be zero or positive." && v733);
    } else {
    }
    int v736;
    v736 = v732 % 32l;
    int v737;
    v737 = v732 / 32l;
    bool v738;
    v738 = v737 < 1l;
    bool v739;
    v739 = v738 == false;
    if (v739){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v738);
    } else {
    }
    assert("Tensor range check" && 0 <= v737 && v737 < 1l);
    assert("Tensor range check" && 0 <= v736 && v736 < 32l);
    int v741;
    v741 = 4l * v736;
    int v742;
    v742 = 256l * v737;
    int v743;
    v743 = v742 + v741;
    assert("Tensor range check" && 0 <= v737 && v737 < 1l);
    int v744;
    v744 = 0l;
    while (while_method_2(v744)){
        assert("Tensor range check" && 0 <= v744 && v744 < 16l);
        int v746;
        v746 = 256l * v744;
        int v747;
        v747 = v746 + v743;
        float v748[8l];
        int v749[8l];
        int v750;
        v750 = 0l;
        while (while_method_3(v750)){
            assert("Tensor range check" && 0 <= v750 && v750 < 2l);
            int v752;
            v752 = 4l * v750;
            assert("Tensor range check" && 0 <= v750 && v750 < 2l);
            int v753;
            v753 = 128l * v750;
            int v754;
            v754 = v753 + v747;
            int4* v755;
            v755 = reinterpret_cast<int4*>(v1 + v754);
            int4* v756;
            v756 = reinterpret_cast<int4*>(v748 + v752);
            assert("Pointer alignment check" && (unsigned long long)(v755) % 4l == 0 && (unsigned long long)(v756) % 4l == 0);
            *v756 = *v755;
            v750 += 1l ;
        }
        int v757;
        v757 = 0l;
        while (while_method_3(v757)){
            int v759;
            v759 = 0l;
            while (while_method_1(v759)){
                bool v761;
                v761 = 0l <= v759;
                bool v763;
                if (v761){
                    bool v762;
                    v762 = v759 < 4l;
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
                bool v766;
                v766 = 0l <= v736;
                bool v768;
                if (v766){
                    bool v767;
                    v767 = v736 < 32l;
                    v768 = v767;
                } else {
                    v768 = false;
                }
                bool v769;
                v769 = v768 == false;
                if (v769){
                    assert("The indices should be inside the range of the dimension." && v768);
                } else {
                }
                int v771;
                v771 = v736 * 4l;
                int v772;
                v772 = v759 + v771;
                bool v773;
                v773 = 0l <= v757;
                bool v775;
                if (v773){
                    bool v774;
                    v774 = v757 < 2l;
                    v775 = v774;
                } else {
                    v775 = false;
                }
                bool v776;
                v776 = v775 == false;
                if (v776){
                    assert("The indices should be inside the range of the dimension." && v775);
                } else {
                }
                int v778;
                v778 = v757 * 128l;
                int v779;
                v779 = v772 + v778;
                assert("Tensor range check" && 0 <= v757 && v757 < 2l);
                assert("Tensor range check" && 0 <= v759 && v759 < 4l);
                int v780;
                v780 = 4l * v757;
                int v781;
                v781 = v780 + v759;
                v749[v781] = v779;
                v759 += 1l ;
            }
            v757 += 1l ;
        }
        bool v782;
        v782 = 0l <= v737;
        bool v783;
        v783 = v782 && v738;
        bool v784;
        v784 = v783 == false;
        if (v784){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v783);
        } else {
        }
        bool v786;
        v786 = 0l <= v744;
        bool v788;
        if (v786){
            bool v787;
            v787 = v744 < 16l;
            v788 = v787;
        } else {
            v788 = false;
        }
        bool v789;
        v789 = v788 == false;
        if (v789){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v788);
        } else {
        }
        int v791;
        v791 = v744 + v737;
        float v792;
        v792 = 0.0f;
        int v793;
        v793 = 0l;
        while (while_method_3(v793)){
            int v795;
            v795 = 0l;
            while (while_method_1(v795)){
                assert("Tensor range check" && 0 <= v793 && v793 < 2l);
                assert("Tensor range check" && 0 <= v795 && v795 < 4l);
                int v797;
                v797 = 4l * v793;
                int v798;
                v798 = v797 + v795;
                float v799;
                v799 = v748[v798];
                float v800;
                v800 = v792 + v799;
                v792 = v800;
                v795 += 1l ;
            }
            v793 += 1l ;
        }
        auto v801 = cooperative_groups::coalesced_threads();
        int v802;
        v802 = threadIdx.x;
        int v803;
        v803 = v802 / 32l;
        auto v804 = cooperative_groups::labeled_partition(v801,v803);
        float v805;
        v805 = cooperative_groups::reduce(v804, v792, v65);
        float v806;
        v806 = v805 / 256.0f;
        float v807[8l];
        int v808;
        v808 = 0l;
        while (while_method_3(v808)){
            int v810;
            v810 = 0l;
            while (while_method_1(v810)){
                assert("Tensor range check" && 0 <= v808 && v808 < 2l);
                assert("Tensor range check" && 0 <= v810 && v810 < 4l);
                int v812;
                v812 = 4l * v808;
                int v813;
                v813 = v812 + v810;
                float v814;
                v814 = v748[v813];
                float v815;
                v815 = v814 - v806;
                float v816;
                v816 = exp(v815);
                assert("Tensor range check" && 0 <= v808 && v808 < 2l);
                assert("Tensor range check" && 0 <= v810 && v810 < 4l);
                v807[v813] = v816;
                v810 += 1l ;
            }
            v808 += 1l ;
        }
        float v817;
        v817 = 0.0f;
        int v818;
        v818 = 0l;
        while (while_method_3(v818)){
            int v820;
            v820 = 0l;
            while (while_method_1(v820)){
                assert("Tensor range check" && 0 <= v818 && v818 < 2l);
                assert("Tensor range check" && 0 <= v820 && v820 < 4l);
                int v822;
                v822 = 4l * v818;
                int v823;
                v823 = v822 + v820;
                float v824;
                v824 = v807[v823];
                float v825;
                v825 = v817 + v824;
                v817 = v825;
                v820 += 1l ;
            }
            v818 += 1l ;
        }
        auto v826 = cooperative_groups::coalesced_threads();
        int v827;
        v827 = threadIdx.x;
        int v828;
        v828 = v827 / 32l;
        auto v829 = cooperative_groups::labeled_partition(v826,v828);
        float v830;
        v830 = cooperative_groups::reduce(v829, v817, v65);
        float v831[8l];
        int v832;
        v832 = 0l;
        while (while_method_3(v832)){
            int v834;
            v834 = 0l;
            while (while_method_1(v834)){
                assert("Tensor range check" && 0 <= v832 && v832 < 2l);
                assert("Tensor range check" && 0 <= v834 && v834 < 4l);
                int v836;
                v836 = 4l * v832;
                int v837;
                v837 = v836 + v834;
                float v838;
                v838 = v807[v837];
                bool v839;
                v839 = v830 == 0.0f;
                bool v840;
                v840 = v839 != true;
                float v842;
                if (v840){
                    float v841;
                    v841 = v838 / v830;
                    v842 = v841;
                } else {
                    v842 = 0.00390625f;
                }
                assert("Tensor range check" && 0 <= v832 && v832 < 2l);
                assert("Tensor range check" && 0 <= v834 && v834 < 4l);
                v831[v837] = v842;
                v834 += 1l ;
            }
            v832 += 1l ;
        }
        float v843[8l];
        float v844;
        v844 = 0.0f;
        int v845;
        v845 = 0l;
        while (while_method_3(v845)){
            assert("Tensor range check" && 0 <= v845 && v845 < 2l);
            int v847;
            v847 = 4l * v845;
            assert("Tensor range check" && 0 <= v845 && v845 < 2l);
            int v848; float v849;
            Tuple0 tmp5 = Tuple0{0l, 0.0f};
            v848 = tmp5.v0; v849 = tmp5.v1;
            while (while_method_1(v848)){
                assert("Tensor range check" && 0 <= v848 && v848 < 4l);
                int v851;
                v851 = v848 + v847;
                float v852;
                v852 = v831[v851];
                float v853;
                v853 = v849 + v852;
                v849 = v853;
                v848 += 1l ;
            }
            auto v854 = cooperative_groups::coalesced_threads();
            int v855;
            v855 = threadIdx.x;
            int v856;
            v856 = v855 / 32l;
            auto v857 = cooperative_groups::labeled_partition(v854,v856);
            Closure2 v858{};
            float v859;
            v859 = cooperative_groups::inclusive_scan(v857, v849, v858);
            float v860;
            v860 = v857.shfl_up(v859,1);
            bool v861;
            v861 = v857.thread_rank() == 0;
            float v862;
            if (v861){
                v862 = 0.0f;
            } else {
                v862 = v860;
            }
            float v863;
            v863 = v857.shfl(v859,v857.num_threads()-1);
            float v864;
            v864 = v844 + v862;
            int v865; float v866;
            Tuple0 tmp6 = Tuple0{0l, v864};
            v865 = tmp6.v0; v866 = tmp6.v1;
            while (while_method_1(v865)){
                assert("Tensor range check" && 0 <= v865 && v865 < 4l);
                int v868;
                v868 = v865 + v847;
                float v869;
                v869 = v831[v868];
                float v870;
                v870 = v866 + v869;
                assert("Tensor range check" && 0 <= v865 && v865 < 4l);
                v843[v868] = v870;
                v866 = v870;
                v865 += 1l ;
            }
            float v871;
            v871 = v844 + v863;
            v844 = v871;
            v845 += 1l ;
        }
        float v872;
        v872 = curand_uniform(&v16);
        float v873[8l];
        int v874;
        v874 = 0l;
        while (while_method_3(v874)){
            int v876;
            v876 = 0l;
            while (while_method_1(v876)){
                assert("Tensor range check" && 0 <= v874 && v874 < 2l);
                assert("Tensor range check" && 0 <= v876 && v876 < 4l);
                int v878;
                v878 = 4l * v874;
                int v879;
                v879 = v878 + v876;
                float v880;
                v880 = v843[v879];
                float v881;
                v881 = v880 - v872;
                assert("Tensor range check" && 0 <= v874 && v874 < 2l);
                assert("Tensor range check" && 0 <= v876 && v876 < 4l);
                v873[v879] = v881;
                v876 += 1l ;
            }
            v874 += 1l ;
        }
        float v882; int v883;
        Tuple1 tmp7 = Tuple1{-1.0f / 0.0f, 0l};
        v882 = tmp7.v0; v883 = tmp7.v1;
        int v884;
        v884 = 0l;
        while (while_method_3(v884)){
            int v886;
            v886 = 0l;
            while (while_method_1(v886)){
                assert("Tensor range check" && 0 <= v884 && v884 < 2l);
                assert("Tensor range check" && 0 <= v886 && v886 < 4l);
                int v888;
                v888 = 4l * v884;
                int v889;
                v889 = v888 + v886;
                float v890;
                v890 = v873[v889];
                int v891;
                v891 = v749[v889];
                bool v892;
                v892 = v882 >= 0.0f;
                bool v894;
                if (v892){
                    bool v893;
                    v893 = v890 >= 0.0f;
                    v894 = v893;
                } else {
                    v894 = false;
                }
                float v903; int v904;
                if (v894){
                    bool v895;
                    v895 = v882 <= v890;
                    if (v895){
                        v903 = v882; v904 = v883;
                    } else {
                        v903 = v890; v904 = v891;
                    }
                } else {
                    if (v892){
                        v903 = v882; v904 = v883;
                    } else {
                        bool v898;
                        v898 = v890 >= 0.0f;
                        if (v898){
                            v903 = v890; v904 = v891;
                        } else {
                            v903 = v882; v904 = v883;
                        }
                    }
                }
                v882 = v903;
                v883 = v904;
                v886 += 1l ;
            }
            v884 += 1l ;
        }
        auto v905 = cooperative_groups::coalesced_threads();
        int v906;
        v906 = threadIdx.x;
        int v907;
        v907 = v906 / 32l;
        auto v908 = cooperative_groups::labeled_partition(v905,v907);
        Closure3 v909{};
        float v910; int v911;
        Tuple1 tmp8 = cooperative_groups::reduce(v908, Tuple1{v882, v883}, v909);
        v910 = tmp8.v0; v911 = tmp8.v1;
        assert("Tensor range check" && 0 <= v744 && v744 < 16l);
        v10[v791] = v911;
        v744 += 1l ;
    }
    __syncthreads();
    int v912;
    v912 = threadIdx.x;
    bool v913;
    v913 = 0l <= v912;
    bool v914;
    v914 = v913 == false;
    if (v914){
        assert("The index needs to be zero or positive." && v913);
    } else {
    }
    int v916;
    v916 = v912 % 32l;
    int v917;
    v917 = v912 / 32l;
    bool v918;
    v918 = v917 < 1l;
    bool v919;
    v919 = v918 == false;
    if (v919){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v918);
    } else {
    }
    assert("Tensor range check" && 0 <= v917 && v917 < 1l);
    assert("Tensor range check" && 0 <= v916 && v916 < 32l);
    int v921;
    v921 = 4l * v916;
    int v922;
    v922 = 256l * v917;
    int v923;
    v923 = v922 + v921;
    assert("Tensor range check" && 0 <= v917 && v917 < 1l);
    assert("Tensor range check" && 0 <= v916 && v916 < 32l);
    int v924;
    v924 = 0l;
    while (while_method_2(v924)){
        assert("Tensor range check" && 0 <= v924 && v924 < 16l);
        int v926;
        v926 = 256l * v924;
        int v927;
        v927 = v926 + v923;
        int v928[8l];
        int v929[8l];
        int v930;
        v930 = 0l;
        while (while_method_3(v930)){
            assert("Tensor range check" && 0 <= v930 && v930 < 2l);
            int v932;
            v932 = 4l * v930;
            assert("Tensor range check" && 0 <= v930 && v930 < 2l);
            int v933;
            v933 = 128l * v930;
            int v934;
            v934 = v933 + v927;
            int4* v935;
            v935 = reinterpret_cast<int4*>(v0 + v934);
            int4* v936;
            v936 = reinterpret_cast<int4*>(v928 + v932);
            assert("Pointer alignment check" && (unsigned long long)(v935) % 4l == 0 && (unsigned long long)(v936) % 4l == 0);
            *v936 = *v935;
            v930 += 1l ;
        }
        int v937;
        v937 = 0l;
        while (while_method_3(v937)){
            int v939;
            v939 = 0l;
            while (while_method_1(v939)){
                bool v941;
                v941 = 0l <= v939;
                bool v943;
                if (v941){
                    bool v942;
                    v942 = v939 < 4l;
                    v943 = v942;
                } else {
                    v943 = false;
                }
                bool v944;
                v944 = v943 == false;
                if (v944){
                    assert("The indices should be inside the range of the dimension." && v943);
                } else {
                }
                bool v946;
                v946 = 0l <= v916;
                bool v948;
                if (v946){
                    bool v947;
                    v947 = v916 < 32l;
                    v948 = v947;
                } else {
                    v948 = false;
                }
                bool v949;
                v949 = v948 == false;
                if (v949){
                    assert("The indices should be inside the range of the dimension." && v948);
                } else {
                }
                int v951;
                v951 = v916 * 4l;
                int v952;
                v952 = v939 + v951;
                bool v953;
                v953 = 0l <= v937;
                bool v955;
                if (v953){
                    bool v954;
                    v954 = v937 < 2l;
                    v955 = v954;
                } else {
                    v955 = false;
                }
                bool v956;
                v956 = v955 == false;
                if (v956){
                    assert("The indices should be inside the range of the dimension." && v955);
                } else {
                }
                int v958;
                v958 = v937 * 128l;
                int v959;
                v959 = v952 + v958;
                assert("Tensor range check" && 0 <= v937 && v937 < 2l);
                assert("Tensor range check" && 0 <= v939 && v939 < 4l);
                int v960;
                v960 = 4l * v937;
                int v961;
                v961 = v960 + v939;
                v929[v961] = v959;
                v939 += 1l ;
            }
            v937 += 1l ;
        }
        bool v962;
        v962 = 0l <= v917;
        bool v963;
        v963 = v962 && v918;
        bool v964;
        v964 = v963 == false;
        if (v964){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v963);
        } else {
        }
        bool v966;
        v966 = 0l <= v924;
        bool v968;
        if (v966){
            bool v967;
            v967 = v924 < 16l;
            v968 = v967;
        } else {
            v968 = false;
        }
        bool v969;
        v969 = v968 == false;
        if (v969){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v968);
        } else {
        }
        int v971;
        v971 = v924 + v917;
        int v972[8l];
        int v973;
        v973 = 0l;
        int v974;
        v974 = 0l;
        while (while_method_3(v974)){
            assert("Tensor range check" && 0 <= v974 && v974 < 2l);
            int v976;
            v976 = 4l * v974;
            assert("Tensor range check" && 0 <= v974 && v974 < 2l);
            int v977; int v978;
            Tuple2 tmp9 = Tuple2{0l, 0l};
            v977 = tmp9.v0; v978 = tmp9.v1;
            while (while_method_1(v977)){
                assert("Tensor range check" && 0 <= v977 && v977 < 4l);
                int v980;
                v980 = v977 + v976;
                int v981;
                v981 = v928[v980];
                int v982;
                v982 = v978 + v981;
                v978 = v982;
                v977 += 1l ;
            }
            auto v983 = cooperative_groups::coalesced_threads();
            int v984;
            v984 = threadIdx.x;
            int v985;
            v985 = v984 / 32l;
            auto v986 = cooperative_groups::labeled_partition(v983,v985);
            Closure4 v987{};
            int v988;
            v988 = cooperative_groups::inclusive_scan(v986, v978, v987);
            int v989;
            v989 = v986.shfl_up(v988,1);
            bool v990;
            v990 = v986.thread_rank() == 0;
            int v991;
            if (v990){
                v991 = 0l;
            } else {
                v991 = v989;
            }
            int v992;
            v992 = v986.shfl(v988,v986.num_threads()-1);
            int v993;
            v993 = v973 + v991;
            int v994; int v995;
            Tuple2 tmp10 = Tuple2{0l, v993};
            v994 = tmp10.v0; v995 = tmp10.v1;
            while (while_method_1(v994)){
                assert("Tensor range check" && 0 <= v994 && v994 < 4l);
                int v997;
                v997 = v994 + v976;
                int v998;
                v998 = v928[v997];
                assert("Tensor range check" && 0 <= v994 && v994 < 4l);
                v972[v997] = v995;
                int v999;
                v999 = v995 + v998;
                v995 = v999;
                v994 += 1l ;
            }
            int v1000;
            v1000 = v973 + v992;
            v973 = v1000;
            v974 += 1l ;
        }
        assert("Tensor range check" && 0 <= v924 && v924 < 16l);
        int v1001;
        v1001 = 0l;
        while (while_method_3(v1001)){
            assert("Tensor range check" && 0 <= v1001 && v1001 < 2l);
            int v1003;
            v1003 = 128l * v1001;
            int v1004;
            v1004 = v1003 + v927;
            assert("Tensor range check" && 0 <= v1001 && v1001 < 2l);
            int v1005;
            v1005 = 4l * v1001;
            int4* v1006;
            v1006 = reinterpret_cast<int4*>(v972 + v1005);
            int4* v1007;
            v1007 = reinterpret_cast<int4*>(v14 + v1004);
            assert("Pointer alignment check" && (unsigned long long)(v1006) % 4l == 0 && (unsigned long long)(v1007) % 4l == 0);
            *v1007 = *v1006;
            v1001 += 1l ;
        }
        v924 += 1l ;
    }
    __syncthreads();
    int v1008;
    v1008 = threadIdx.x;
    bool v1009;
    v1009 = 0l <= v1008;
    bool v1010;
    v1010 = v1009 == false;
    if (v1010){
        assert("The index needs to be zero or positive." && v1009);
    } else {
    }
    int v1012;
    v1012 = v1008 % 32l;
    int v1013;
    v1013 = v1008 / 32l;
    bool v1014;
    v1014 = v1013 < 1l;
    bool v1015;
    v1015 = v1014 == false;
    if (v1015){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1014);
    } else {
    }
    assert("Tensor range check" && 0 <= v1013 && v1013 < 1l);
    assert("Tensor range check" && 0 <= v1012 && v1012 < 32l);
    int v1017;
    v1017 = 4l * v1012;
    int v1018;
    v1018 = 256l * v1013;
    int v1019;
    v1019 = v1018 + v1017;
    assert("Tensor range check" && 0 <= v1013 && v1013 < 1l);
    assert("Tensor range check" && 0 <= v1012 && v1012 < 32l);
    int v1020;
    v1020 = 0l;
    while (while_method_2(v1020)){
        assert("Tensor range check" && 0 <= v1020 && v1020 < 16l);
        int v1022;
        v1022 = 256l * v1020;
        int v1023;
        v1023 = v1022 + v1019;
        float v1024[8l];
        int v1025[8l];
        int v1026;
        v1026 = 0l;
        while (while_method_3(v1026)){
            assert("Tensor range check" && 0 <= v1026 && v1026 < 2l);
            int v1028;
            v1028 = 4l * v1026;
            assert("Tensor range check" && 0 <= v1026 && v1026 < 2l);
            int v1029;
            v1029 = 128l * v1026;
            int v1030;
            v1030 = v1029 + v1023;
            int4* v1031;
            v1031 = reinterpret_cast<int4*>(v1 + v1030);
            int4* v1032;
            v1032 = reinterpret_cast<int4*>(v1024 + v1028);
            assert("Pointer alignment check" && (unsigned long long)(v1031) % 4l == 0 && (unsigned long long)(v1032) % 4l == 0);
            *v1032 = *v1031;
            v1026 += 1l ;
        }
        int v1033;
        v1033 = 0l;
        while (while_method_3(v1033)){
            int v1035;
            v1035 = 0l;
            while (while_method_1(v1035)){
                bool v1037;
                v1037 = 0l <= v1035;
                bool v1039;
                if (v1037){
                    bool v1038;
                    v1038 = v1035 < 4l;
                    v1039 = v1038;
                } else {
                    v1039 = false;
                }
                bool v1040;
                v1040 = v1039 == false;
                if (v1040){
                    assert("The indices should be inside the range of the dimension." && v1039);
                } else {
                }
                bool v1042;
                v1042 = 0l <= v1012;
                bool v1044;
                if (v1042){
                    bool v1043;
                    v1043 = v1012 < 32l;
                    v1044 = v1043;
                } else {
                    v1044 = false;
                }
                bool v1045;
                v1045 = v1044 == false;
                if (v1045){
                    assert("The indices should be inside the range of the dimension." && v1044);
                } else {
                }
                int v1047;
                v1047 = v1012 * 4l;
                int v1048;
                v1048 = v1035 + v1047;
                bool v1049;
                v1049 = 0l <= v1033;
                bool v1051;
                if (v1049){
                    bool v1050;
                    v1050 = v1033 < 2l;
                    v1051 = v1050;
                } else {
                    v1051 = false;
                }
                bool v1052;
                v1052 = v1051 == false;
                if (v1052){
                    assert("The indices should be inside the range of the dimension." && v1051);
                } else {
                }
                int v1054;
                v1054 = v1033 * 128l;
                int v1055;
                v1055 = v1048 + v1054;
                assert("Tensor range check" && 0 <= v1033 && v1033 < 2l);
                assert("Tensor range check" && 0 <= v1035 && v1035 < 4l);
                int v1056;
                v1056 = 4l * v1033;
                int v1057;
                v1057 = v1056 + v1035;
                v1025[v1057] = v1055;
                v1035 += 1l ;
            }
            v1033 += 1l ;
        }
        bool v1058;
        v1058 = 0l <= v1013;
        bool v1059;
        v1059 = v1058 && v1014;
        bool v1060;
        v1060 = v1059 == false;
        if (v1060){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1059);
        } else {
        }
        bool v1062;
        v1062 = 0l <= v1020;
        bool v1064;
        if (v1062){
            bool v1063;
            v1063 = v1020 < 16l;
            v1064 = v1063;
        } else {
            v1064 = false;
        }
        bool v1065;
        v1065 = v1064 == false;
        if (v1065){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1064);
        } else {
        }
        int v1067;
        v1067 = v1020 + v1013;
        bool v1068[8l];
        int v1069;
        v1069 = 0l;
        while (while_method_3(v1069)){
            int v1071;
            v1071 = 0l;
            while (while_method_1(v1071)){
                assert("Tensor range check" && 0 <= v1069 && v1069 < 2l);
                assert("Tensor range check" && 0 <= v1071 && v1071 < 4l);
                int v1073;
                v1073 = 4l * v1069;
                int v1074;
                v1074 = v1073 + v1071;
                float v1075;
                v1075 = v1024[v1074];
                int v1076;
                v1076 = v1025[v1074];
                bool v1077;
                v1077 = v1076 < 4l;
                assert("Tensor range check" && 0 <= v1069 && v1069 < 2l);
                assert("Tensor range check" && 0 <= v1071 && v1071 < 4l);
                v1068[v1074] = v1077;
                v1071 += 1l ;
            }
            v1069 += 1l ;
        }
        int v1078[8l];
        int v1079;
        v1079 = 0l;
        while (while_method_3(v1079)){
            int v1081;
            v1081 = 0l;
            while (while_method_1(v1081)){
                assert("Tensor range check" && 0 <= v1079 && v1079 < 2l);
                assert("Tensor range check" && 0 <= v1081 && v1081 < 4l);
                int v1083;
                v1083 = 4l * v1079;
                int v1084;
                v1084 = v1083 + v1081;
                bool v1085;
                v1085 = v1068[v1084];
                int v1086;
                if (v1085){
                    v1086 = 1l;
                } else {
                    v1086 = 0l;
                }
                assert("Tensor range check" && 0 <= v1079 && v1079 < 2l);
                assert("Tensor range check" && 0 <= v1081 && v1081 < 4l);
                v1078[v1084] = v1086;
                v1081 += 1l ;
            }
            v1079 += 1l ;
        }
        int v1087;
        v1087 = 0l;
        int v1088;
        v1088 = 0l;
        while (while_method_3(v1088)){
            int v1090;
            v1090 = 0l;
            while (while_method_1(v1090)){
                assert("Tensor range check" && 0 <= v1088 && v1088 < 2l);
                assert("Tensor range check" && 0 <= v1090 && v1090 < 4l);
                int v1092;
                v1092 = 4l * v1088;
                int v1093;
                v1093 = v1092 + v1090;
                int v1094;
                v1094 = v1078[v1093];
                int v1095;
                v1095 = v1087 + v1094;
                v1087 = v1095;
                v1090 += 1l ;
            }
            v1088 += 1l ;
        }
        auto v1096 = cooperative_groups::coalesced_threads();
        int v1097;
        v1097 = threadIdx.x;
        int v1098;
        v1098 = v1097 / 32l;
        auto v1099 = cooperative_groups::labeled_partition(v1096,v1098);
        Closure5 v1100{};
        int v1101;
        v1101 = cooperative_groups::reduce(v1099, v1087, v1100);
        float v1102[8l];
        int v1103;
        v1103 = 0l;
        while (while_method_3(v1103)){
            int v1105;
            v1105 = 0l;
            while (while_method_1(v1105)){
                assert("Tensor range check" && 0 <= v1103 && v1103 < 2l);
                assert("Tensor range check" && 0 <= v1105 && v1105 < 4l);
                int v1107;
                v1107 = 4l * v1103;
                int v1108;
                v1108 = v1107 + v1105;
                float v1109;
                v1109 = v1024[v1108];
                bool v1110;
                v1110 = v1068[v1108];
                float v1111;
                if (v1110){
                    v1111 = v1109;
                } else {
                    v1111 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1103 && v1103 < 2l);
                assert("Tensor range check" && 0 <= v1105 && v1105 < 4l);
                v1102[v1108] = v1111;
                v1105 += 1l ;
            }
            v1103 += 1l ;
        }
        float v1112;
        v1112 = 0.0f;
        int v1113;
        v1113 = 0l;
        while (while_method_3(v1113)){
            int v1115;
            v1115 = 0l;
            while (while_method_1(v1115)){
                assert("Tensor range check" && 0 <= v1113 && v1113 < 2l);
                assert("Tensor range check" && 0 <= v1115 && v1115 < 4l);
                int v1117;
                v1117 = 4l * v1113;
                int v1118;
                v1118 = v1117 + v1115;
                float v1119;
                v1119 = v1102[v1118];
                float v1120;
                v1120 = v1112 + v1119;
                v1112 = v1120;
                v1115 += 1l ;
            }
            v1113 += 1l ;
        }
        auto v1121 = cooperative_groups::coalesced_threads();
        int v1122;
        v1122 = threadIdx.x;
        int v1123;
        v1123 = v1122 / 32l;
        auto v1124 = cooperative_groups::labeled_partition(v1121,v1123);
        float v1125;
        v1125 = cooperative_groups::reduce(v1124, v1112, v65);
        float v1126;
        v1126 = (float)v1101;
        float v1127;
        v1127 = v1125 / v1126;
        float v1128[8l];
        int v1129;
        v1129 = 0l;
        while (while_method_3(v1129)){
            int v1131;
            v1131 = 0l;
            while (while_method_1(v1131)){
                assert("Tensor range check" && 0 <= v1129 && v1129 < 2l);
                assert("Tensor range check" && 0 <= v1131 && v1131 < 4l);
                int v1133;
                v1133 = 4l * v1129;
                int v1134;
                v1134 = v1133 + v1131;
                float v1135;
                v1135 = v1024[v1134];
                bool v1136;
                v1136 = v1068[v1134];
                float v1137;
                if (v1136){
                    v1137 = v1135;
                } else {
                    v1137 = -1.0f / 0.0f;
                }
                float v1138;
                v1138 = v1137 - v1127;
                float v1139;
                v1139 = exp(v1138);
                assert("Tensor range check" && 0 <= v1129 && v1129 < 2l);
                assert("Tensor range check" && 0 <= v1131 && v1131 < 4l);
                v1128[v1134] = v1139;
                v1131 += 1l ;
            }
            v1129 += 1l ;
        }
        float v1140;
        v1140 = 0.0f;
        int v1141;
        v1141 = 0l;
        while (while_method_3(v1141)){
            int v1143;
            v1143 = 0l;
            while (while_method_1(v1143)){
                assert("Tensor range check" && 0 <= v1141 && v1141 < 2l);
                assert("Tensor range check" && 0 <= v1143 && v1143 < 4l);
                int v1145;
                v1145 = 4l * v1141;
                int v1146;
                v1146 = v1145 + v1143;
                float v1147;
                v1147 = v1128[v1146];
                float v1148;
                v1148 = v1140 + v1147;
                v1140 = v1148;
                v1143 += 1l ;
            }
            v1141 += 1l ;
        }
        auto v1149 = cooperative_groups::coalesced_threads();
        int v1150;
        v1150 = threadIdx.x;
        int v1151;
        v1151 = v1150 / 32l;
        auto v1152 = cooperative_groups::labeled_partition(v1149,v1151);
        float v1153;
        v1153 = cooperative_groups::reduce(v1152, v1140, v65);
        float v1154[8l];
        int v1155;
        v1155 = 0l;
        while (while_method_3(v1155)){
            int v1157;
            v1157 = 0l;
            while (while_method_1(v1157)){
                assert("Tensor range check" && 0 <= v1155 && v1155 < 2l);
                assert("Tensor range check" && 0 <= v1157 && v1157 < 4l);
                int v1159;
                v1159 = 4l * v1155;
                int v1160;
                v1160 = v1159 + v1157;
                float v1161;
                v1161 = v1128[v1160];
                bool v1162;
                v1162 = v1153 == 0.0f;
                bool v1163;
                v1163 = v1162 != true;
                float v1165;
                if (v1163){
                    float v1164;
                    v1164 = v1161 / v1153;
                    v1165 = v1164;
                } else {
                    v1165 = 0.00390625f;
                }
                assert("Tensor range check" && 0 <= v1155 && v1155 < 2l);
                assert("Tensor range check" && 0 <= v1157 && v1157 < 4l);
                v1154[v1160] = v1165;
                v1157 += 1l ;
            }
            v1155 += 1l ;
        }
        assert("Tensor range check" && 0 <= v1020 && v1020 < 16l);
        int v1166;
        v1166 = 0l;
        while (while_method_3(v1166)){
            assert("Tensor range check" && 0 <= v1166 && v1166 < 2l);
            int v1168;
            v1168 = 128l * v1166;
            int v1169;
            v1169 = v1168 + v1023;
            assert("Tensor range check" && 0 <= v1166 && v1166 < 2l);
            int v1170;
            v1170 = 4l * v1166;
            int4* v1171;
            v1171 = reinterpret_cast<int4*>(v1154 + v1170);
            int4* v1172;
            v1172 = reinterpret_cast<int4*>(v5 + v1169);
            assert("Pointer alignment check" && (unsigned long long)(v1171) % 4l == 0 && (unsigned long long)(v1172) % 4l == 0);
            *v1172 = *v1171;
            v1166 += 1l ;
        }
        v1020 += 1l ;
    }
    __syncthreads();
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
options.append('--diag-suppress=550,20012,68')
options.append('--dopt=on')
options.append('--restrict')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0(v0 : char) -> None:
    print(v0, end="")
    del v0
    return 
def method1(v0 : i32) -> bool:
    v1 = v0 < 1
    del v0
    return v1
def method2(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method3(v0 : f32) -> None:
    print("{:.6f}".format(v0), end="")
    del v0
    return 
def method4() -> None:
    return 
def main():
    v0 = cp.arange(0,4096,1,dtype=cp.int32) # type: ignore
    v1 = v0.size
    v2 = 4096 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,4096,dtype=cp.float32) # type: ignore
    v6 = cp.random.uniform(size=16,dtype=cp.float32) # type: ignore
    del v6
    v7 = cp.empty(1,dtype=cp.float32)
    v8 = cp.empty(4096,dtype=cp.int32)
    v9 = cp.empty(4096,dtype=cp.float32)
    v10 = cp.empty(4096,dtype=cp.float32)
    v11 = cp.empty(4096,dtype=cp.float32)
    v12 = cp.empty(4096,dtype=cp.float32)
    v13 = cp.empty(4096,dtype=cp.float32)
    v14 = cp.empty(16,dtype=cp.int32)
    v15 = cp.empty(16,dtype=cp.int32)
    v16 = cp.empty(4096,dtype=cp.int32)
    v17 = cp.empty(4096,dtype=cp.int32)
    v18 = cp.empty(16,dtype=cp.int32)
    v19 = cp.empty(4096,dtype=cp.int32)
    v20 = 0
    v21 = raw_module.get_function(f"entry{v20}")
    del v20
    v21.max_dynamic_shared_size_bytes = 0 
    v21((1,),(32,),(v0, v5, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19),shared_mem=0)
    del v0, v5, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v21
    v22 = 0
    v23 = '['
    method0(v23)
    del v23
    v24 = 0
    while method1(v24):
        v26 = v22
        v27 = v26 >= 1024
        del v26
        if v27:
            v28 = " ..."
            method2(v28)
            del v28
            break
        else:
            pass
        del v27
        v29 = v24 == 0
        v30 = v29 != True
        del v29
        if v30:
            v31 = "; "
            method2(v31)
        else:
            pass
        del v30
        v32 = v22 + 1
        v22 = v32
        del v32
        v33 = v7[v24].item()
        method3(v33)
        del v33
        v24 += 1 
    del v7, v22, v24
    v34 = ']'
    method0(v34)
    del v34
    method4()
    print()
    return 

if __name__ == '__main__': print(main())
