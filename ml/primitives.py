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
    v1 = v0 < 2l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
extern "C" __global__ void entry0(int * v0, float * v1, float * v2, int * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, int * v13) {
    unsigned long long v14;
    v14 = clock64();
    curandStatePhilox4_32_10_t v15;
    curand_init(v14,0ull,0ull,&v15);
    int v16;
    v16 = threadIdx.x;
    int v17;
    v17 = v16;
    while (while_method_0(v17)){
        bool v19;
        v19 = 0l <= v17;
        bool v20;
        v20 = v19 == false;
        if (v20){
            assert("The index needs to be zero or positive." && v19);
        } else {
        }
        int v21;
        v21 = v17 % 1l;
        bool v22;
        v22 = v17 < 1024l;
        bool v23;
        v23 = v22 == false;
        if (v23){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v22);
        } else {
        }
        assert("Tensor range check" && 0 <= v17 && v17 < 1024l);
        assert("Tensor range check" && 0 <= v21 && v21 < 1l);
        int v24;
        v24 = 4l * v21;
        int v25;
        v25 = 4l * v17;
        int v26;
        v26 = v25 + v24;
        assert("Tensor range check" && 0 <= v17 && v17 < 1024l);
        assert("Tensor range check" && 0 <= v21 && v21 < 1l);
        float v27[4l];
        float v28[4l];
        int4* v29;
        v29 = reinterpret_cast<int4*>(v1 + v26);
        int4* v30;
        v30 = reinterpret_cast<int4*>(v27 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v29) % 4l == 0 && (unsigned long long)(v30) % 4l == 0);
        *v30 = *v29;
        // Pushing the loop unrolling to: 0
        int v31;
        v31 = 0l;
        #pragma unroll
        while (while_method_1(v31)){
            assert("Tensor range check" && 0 <= v31 && v31 < 4l);
            float v33;
            v33 = v27[v31];
            float v34;
            v34 = 1.0f + v33;
            assert("Tensor range check" && 0 <= v31 && v31 < 4l);
            v28[v31] = v34;
            v31 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v35;
        v35 = reinterpret_cast<int4*>(v28 + 0l);
        int4* v36;
        v36 = reinterpret_cast<int4*>(v1 + v26);
        assert("Pointer alignment check" && (unsigned long long)(v35) % 4l == 0 && (unsigned long long)(v36) % 4l == 0);
        *v36 = *v35;
        v17 += 512l ;
    }
    __syncthreads();
    float v37;
    v37 = 0.0f;
    int v38;
    v38 = threadIdx.x;
    int v39;
    v39 = v38;
    while (while_method_0(v39)){
        bool v41;
        v41 = 0l <= v39;
        bool v42;
        v42 = v41 == false;
        if (v42){
            assert("The index needs to be zero or positive." && v41);
        } else {
        }
        int v43;
        v43 = v39 % 1l;
        bool v44;
        v44 = v39 < 1024l;
        bool v45;
        v45 = v44 == false;
        if (v45){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v44);
        } else {
        }
        assert("Tensor range check" && 0 <= v39 && v39 < 1024l);
        assert("Tensor range check" && 0 <= v43 && v43 < 1l);
        int v46;
        v46 = 4l * v43;
        int v47;
        v47 = 4l * v39;
        int v48;
        v48 = v47 + v46;
        float v49[4l];
        int4* v50;
        v50 = reinterpret_cast<int4*>(v1 + v48);
        int4* v51;
        v51 = reinterpret_cast<int4*>(v49 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v50) % 4l == 0 && (unsigned long long)(v51) % 4l == 0);
        *v51 = *v50;
        int v52; float v53;
        Tuple0 tmp0 = Tuple0{0l, v37};
        v52 = tmp0.v0; v53 = tmp0.v1;
        while (while_method_1(v52)){
            assert("Tensor range check" && 0 <= v52 && v52 < 4l);
            float v55;
            v55 = v49[v52];
            float v56;
            v56 = v53 + v55;
            v53 = v56;
            v52 += 1l ;
        }
        v37 = v53;
        v39 += 512l ;
    }
    auto v57 = cooperative_groups::coalesced_threads();
    Closure0 v58{};
    float v59;
    v59 = cooperative_groups::reduce(v57, v37, v58);
    int v60;
    v60 = threadIdx.x;
    int v61;
    v61 = v60 / 32l;
    __shared__ float v62[16l];
    assert("Tensor range check" && 0 <= v61 && v61 < 16l);
    v62[v61] = v59;
    __syncthreads();
    int v63;
    v63 = threadIdx.x;
    int v64;
    v64 = v63 % 32l;
    bool v65;
    v65 = v61 == 0l;
    bool v67;
    if (v65){
        bool v66;
        v66 = v64 < 16l;
        v67 = v66;
    } else {
        v67 = false;
    }
    if (v67){
        auto v68 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v64 && v64 < 16l);
        float v69;
        v69 = v62[v64];
        float v70;
        v70 = cooperative_groups::reduce(v68, v69, v58);
        v2[0l] = v70;
    } else {
    }
    __syncthreads();
    int v71;
    v71 = threadIdx.x;
    bool v72;
    v72 = 0l <= v71;
    bool v73;
    v73 = v72 == false;
    if (v73){
        assert("The index needs to be zero or positive." && v72);
    } else {
    }
    int v74;
    v74 = v71 % 1l;
    bool v75;
    v75 = v71 < 512l;
    bool v76;
    v76 = v75 == false;
    if (v76){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v75);
    } else {
    }
    assert("Tensor range check" && 0 <= v71 && v71 < 512l);
    assert("Tensor range check" && 0 <= v74 && v74 < 1l);
    int v77;
    v77 = 4l * v74;
    int v78;
    v78 = 4l * v71;
    int v79;
    v79 = v78 + v77;
    assert("Tensor range check" && 0 <= v71 && v71 < 512l);
    assert("Tensor range check" && 0 <= v74 && v74 < 1l);
    int v80;
    v80 = 0l;
    while (while_method_2(v80)){
        assert("Tensor range check" && 0 <= v80 && v80 < 2l);
        int v82;
        v82 = 2048l * v80;
        int v83;
        v83 = v82 + v79;
        assert("Tensor range check" && 0 <= v80 && v80 < 2l);
        int v84[4l];
        int v85[4l];
        int v86;
        v86 = 0l;
        while (while_method_3(v86)){
            assert("Tensor range check" && 0 <= v86 && v86 < 1l);
            int v88;
            v88 = 4l * v86;
            assert("Tensor range check" && 0 <= v86 && v86 < 1l);
            int v89;
            v89 = v88 + v83;
            int4* v90;
            v90 = reinterpret_cast<int4*>(v0 + v89);
            int4* v91;
            v91 = reinterpret_cast<int4*>(v84 + v88);
            assert("Pointer alignment check" && (unsigned long long)(v90) % 4l == 0 && (unsigned long long)(v91) % 4l == 0);
            *v91 = *v90;
            v86 += 1l ;
        }
        int v92;
        v92 = 0l;
        while (while_method_3(v92)){
            int v94;
            v94 = 0l;
            while (while_method_1(v94)){
                bool v96;
                v96 = 0l <= v94;
                bool v98;
                if (v96){
                    bool v97;
                    v97 = v94 < 4l;
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
                bool v100;
                v100 = 0l <= v74;
                bool v102;
                if (v100){
                    bool v101;
                    v101 = v74 < 1l;
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
                int v104;
                v104 = v74 * 4l;
                int v105;
                v105 = v94 + v104;
                bool v106;
                v106 = 0l <= v92;
                bool v108;
                if (v106){
                    bool v107;
                    v107 = v92 < 1l;
                    v108 = v107;
                } else {
                    v108 = false;
                }
                bool v109;
                v109 = v108 == false;
                if (v109){
                    assert("The indices should be inside the range of the dimension." && v108);
                } else {
                }
                int v110;
                v110 = v92 * 4l;
                int v111;
                v111 = v105 + v110;
                assert("Tensor range check" && 0 <= v92 && v92 < 1l);
                assert("Tensor range check" && 0 <= v94 && v94 < 4l);
                int v112;
                v112 = 4l * v92;
                int v113;
                v113 = v112 + v94;
                v85[v113] = v111;
                v94 += 1l ;
            }
            v92 += 1l ;
        }
        bool v114;
        v114 = v72 && v75;
        bool v115;
        v115 = v114 == false;
        if (v115){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v114);
        } else {
        }
        bool v116;
        v116 = 0l <= v80;
        bool v118;
        if (v116){
            bool v117;
            v117 = v80 < 2l;
            v118 = v117;
        } else {
            v118 = false;
        }
        bool v119;
        v119 = v118 == false;
        if (v119){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v118);
        } else {
        }
        int v120;
        v120 = v80 * 512l;
        int v121;
        v121 = v120 + v71;
        int v122;
        v122 = 0l;
        while (while_method_3(v122)){
            assert("Tensor range check" && 0 <= v122 && v122 < 1l);
            int v124;
            v124 = 4l * v122;
            int v125;
            v125 = v124 + v83;
            assert("Tensor range check" && 0 <= v122 && v122 < 1l);
            int4* v126;
            v126 = reinterpret_cast<int4*>(v84 + v124);
            int4* v127;
            v127 = reinterpret_cast<int4*>(v3 + v125);
            assert("Pointer alignment check" && (unsigned long long)(v126) % 4l == 0 && (unsigned long long)(v127) % 4l == 0);
            *v127 = *v126;
            v122 += 1l ;
        }
        v80 += 1l ;
    }
    __syncthreads();
    int v128;
    v128 = threadIdx.x;
    bool v129;
    v129 = 0l <= v128;
    bool v130;
    v130 = v129 == false;
    if (v130){
        assert("The index needs to be zero or positive." && v129);
    } else {
    }
    int v131;
    v131 = v128 % 1l;
    bool v132;
    v132 = v128 < 512l;
    bool v133;
    v133 = v132 == false;
    if (v133){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v132);
    } else {
    }
    assert("Tensor range check" && 0 <= v128 && v128 < 512l);
    assert("Tensor range check" && 0 <= v131 && v131 < 1l);
    int v134;
    v134 = 4l * v131;
    int v135;
    v135 = 4l * v128;
    int v136;
    v136 = v135 + v134;
    assert("Tensor range check" && 0 <= v128 && v128 < 512l);
    assert("Tensor range check" && 0 <= v131 && v131 < 1l);
    int v137;
    v137 = 0l;
    while (while_method_2(v137)){
        assert("Tensor range check" && 0 <= v137 && v137 < 2l);
        int v139;
        v139 = 2048l * v137;
        int v140;
        v140 = v139 + v136;
        assert("Tensor range check" && 0 <= v137 && v137 < 2l);
        float v141[4l];
        int v142[4l];
        int v143;
        v143 = 0l;
        while (while_method_3(v143)){
            assert("Tensor range check" && 0 <= v143 && v143 < 1l);
            int v145;
            v145 = 4l * v143;
            assert("Tensor range check" && 0 <= v143 && v143 < 1l);
            int v146;
            v146 = v145 + v140;
            int4* v147;
            v147 = reinterpret_cast<int4*>(v1 + v146);
            int4* v148;
            v148 = reinterpret_cast<int4*>(v141 + v145);
            assert("Pointer alignment check" && (unsigned long long)(v147) % 4l == 0 && (unsigned long long)(v148) % 4l == 0);
            *v148 = *v147;
            v143 += 1l ;
        }
        int v149;
        v149 = 0l;
        while (while_method_3(v149)){
            int v151;
            v151 = 0l;
            while (while_method_1(v151)){
                bool v153;
                v153 = 0l <= v151;
                bool v155;
                if (v153){
                    bool v154;
                    v154 = v151 < 4l;
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
                bool v157;
                v157 = 0l <= v131;
                bool v159;
                if (v157){
                    bool v158;
                    v158 = v131 < 1l;
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
                int v161;
                v161 = v131 * 4l;
                int v162;
                v162 = v151 + v161;
                bool v163;
                v163 = 0l <= v149;
                bool v165;
                if (v163){
                    bool v164;
                    v164 = v149 < 1l;
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
                int v167;
                v167 = v149 * 4l;
                int v168;
                v168 = v162 + v167;
                assert("Tensor range check" && 0 <= v149 && v149 < 1l);
                assert("Tensor range check" && 0 <= v151 && v151 < 4l);
                int v169;
                v169 = 4l * v149;
                int v170;
                v170 = v169 + v151;
                v142[v170] = v168;
                v151 += 1l ;
            }
            v149 += 1l ;
        }
        bool v171;
        v171 = v129 && v132;
        bool v172;
        v172 = v171 == false;
        if (v172){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v171);
        } else {
        }
        bool v173;
        v173 = 0l <= v137;
        bool v175;
        if (v173){
            bool v174;
            v174 = v137 < 2l;
            v175 = v174;
        } else {
            v175 = false;
        }
        bool v176;
        v176 = v175 == false;
        if (v176){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v175);
        } else {
        }
        int v177;
        v177 = v137 * 512l;
        int v178;
        v178 = v177 + v128;
        int v179[4l];
        int v180[4l];
        int v181;
        v181 = 0l;
        while (while_method_3(v181)){
            int v183;
            v183 = 0l;
            while (while_method_1(v183)){
                assert("Tensor range check" && 0 <= v181 && v181 < 1l);
                assert("Tensor range check" && 0 <= v183 && v183 < 4l);
                int v185;
                v185 = 4l * v181;
                int v186;
                v186 = v185 + v183;
                int v187;
                v187 = v142[v186];
                assert("Tensor range check" && 0 <= v181 && v181 < 1l);
                assert("Tensor range check" && 0 <= v183 && v183 < 4l);
                v179[v186] = v178;
                v180[v186] = v187;
                v183 += 1l ;
            }
            v181 += 1l ;
        }
        int v188;
        v188 = 0l;
        while (while_method_3(v188)){
            assert("Tensor range check" && 0 <= v188 && v188 < 1l);
            int v190;
            v190 = 4l * v188;
            int v191;
            v191 = v190 + v140;
            assert("Tensor range check" && 0 <= v188 && v188 < 1l);
            int4* v192;
            v192 = reinterpret_cast<int4*>(v179 + v190);
            int4* v193;
            v193 = reinterpret_cast<int4*>(v10 + v191);
            assert("Pointer alignment check" && (unsigned long long)(v192) % 4l == 0 && (unsigned long long)(v193) % 4l == 0);
            *v193 = *v192;
            int4* v194;
            v194 = reinterpret_cast<int4*>(v180 + v190);
            int4* v195;
            v195 = reinterpret_cast<int4*>(v11 + v191);
            assert("Pointer alignment check" && (unsigned long long)(v194) % 4l == 0 && (unsigned long long)(v195) % 4l == 0);
            *v195 = *v194;
            v188 += 1l ;
        }
        v137 += 1l ;
    }
    __syncthreads();
    int v196;
    v196 = threadIdx.x;
    bool v197;
    v197 = 0l <= v196;
    bool v198;
    v198 = v197 == false;
    if (v198){
        assert("The index needs to be zero or positive." && v197);
    } else {
    }
    int v199;
    v199 = v196 % 1l;
    bool v200;
    v200 = v196 < 512l;
    bool v201;
    v201 = v200 == false;
    if (v201){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v200);
    } else {
    }
    assert("Tensor range check" && 0 <= v196 && v196 < 512l);
    assert("Tensor range check" && 0 <= v199 && v199 < 1l);
    int v202;
    v202 = 4l * v199;
    int v203;
    v203 = 4l * v196;
    int v204;
    v204 = v203 + v202;
    assert("Tensor range check" && 0 <= v196 && v196 < 512l);
    int v205;
    v205 = 0l;
    while (while_method_2(v205)){
        assert("Tensor range check" && 0 <= v205 && v205 < 2l);
        int v207;
        v207 = 2048l * v205;
        int v208;
        v208 = v207 + v204;
        float v209[4l];
        int v210[4l];
        int v211;
        v211 = 0l;
        while (while_method_3(v211)){
            assert("Tensor range check" && 0 <= v211 && v211 < 1l);
            int v213;
            v213 = 4l * v211;
            assert("Tensor range check" && 0 <= v211 && v211 < 1l);
            int v214;
            v214 = v213 + v208;
            int4* v215;
            v215 = reinterpret_cast<int4*>(v1 + v214);
            int4* v216;
            v216 = reinterpret_cast<int4*>(v209 + v213);
            assert("Pointer alignment check" && (unsigned long long)(v215) % 4l == 0 && (unsigned long long)(v216) % 4l == 0);
            *v216 = *v215;
            v211 += 1l ;
        }
        int v217;
        v217 = 0l;
        while (while_method_3(v217)){
            int v219;
            v219 = 0l;
            while (while_method_1(v219)){
                bool v221;
                v221 = 0l <= v219;
                bool v223;
                if (v221){
                    bool v222;
                    v222 = v219 < 4l;
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
                bool v225;
                v225 = 0l <= v199;
                bool v227;
                if (v225){
                    bool v226;
                    v226 = v199 < 1l;
                    v227 = v226;
                } else {
                    v227 = false;
                }
                bool v228;
                v228 = v227 == false;
                if (v228){
                    assert("The indices should be inside the range of the dimension." && v227);
                } else {
                }
                int v229;
                v229 = v199 * 4l;
                int v230;
                v230 = v219 + v229;
                bool v231;
                v231 = 0l <= v217;
                bool v233;
                if (v231){
                    bool v232;
                    v232 = v217 < 1l;
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
                int v235;
                v235 = v217 * 4l;
                int v236;
                v236 = v230 + v235;
                assert("Tensor range check" && 0 <= v217 && v217 < 1l);
                assert("Tensor range check" && 0 <= v219 && v219 < 4l);
                int v237;
                v237 = 4l * v217;
                int v238;
                v238 = v237 + v219;
                v210[v238] = v236;
                v219 += 1l ;
            }
            v217 += 1l ;
        }
        bool v239;
        v239 = v197 && v200;
        bool v240;
        v240 = v239 == false;
        if (v240){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v239);
        } else {
        }
        bool v241;
        v241 = 0l <= v205;
        bool v243;
        if (v241){
            bool v242;
            v242 = v205 < 2l;
            v243 = v242;
        } else {
            v243 = false;
        }
        bool v244;
        v244 = v243 == false;
        if (v244){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v243);
        } else {
        }
        int v245;
        v245 = v205 * 512l;
        int v246;
        v246 = v245 + v196;
        assert("Tensor range check" && 0 <= v205 && v205 < 2l);
        int v247;
        v247 = 512l * v205;
        int v248;
        v248 = v247 + v196;
        v12[v248] = v246;
        v205 += 1l ;
    }
    __syncthreads();
    int v249;
    v249 = threadIdx.x;
    bool v250;
    v250 = 0l <= v249;
    bool v251;
    v251 = v250 == false;
    if (v251){
        assert("The index needs to be zero or positive." && v250);
    } else {
    }
    int v252;
    v252 = v249 % 1l;
    bool v253;
    v253 = v249 < 512l;
    bool v254;
    v254 = v253 == false;
    if (v254){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v253);
    } else {
    }
    assert("Tensor range check" && 0 <= v249 && v249 < 512l);
    assert("Tensor range check" && 0 <= v252 && v252 < 1l);
    int v255;
    v255 = 4l * v252;
    int v256;
    v256 = 4l * v249;
    int v257;
    v257 = v256 + v255;
    assert("Tensor range check" && 0 <= v249 && v249 < 512l);
    assert("Tensor range check" && 0 <= v252 && v252 < 1l);
    int v258;
    v258 = 0l;
    while (while_method_2(v258)){
        assert("Tensor range check" && 0 <= v258 && v258 < 2l);
        int v260;
        v260 = 2048l * v258;
        int v261;
        v261 = v260 + v257;
        assert("Tensor range check" && 0 <= v258 && v258 < 2l);
        float v262[4l];
        int v263[4l];
        int v264;
        v264 = 0l;
        while (while_method_3(v264)){
            assert("Tensor range check" && 0 <= v264 && v264 < 1l);
            int v266;
            v266 = 4l * v264;
            assert("Tensor range check" && 0 <= v264 && v264 < 1l);
            int v267;
            v267 = v266 + v261;
            int4* v268;
            v268 = reinterpret_cast<int4*>(v1 + v267);
            int4* v269;
            v269 = reinterpret_cast<int4*>(v262 + v266);
            assert("Pointer alignment check" && (unsigned long long)(v268) % 4l == 0 && (unsigned long long)(v269) % 4l == 0);
            *v269 = *v268;
            v264 += 1l ;
        }
        int v270;
        v270 = 0l;
        while (while_method_3(v270)){
            int v272;
            v272 = 0l;
            while (while_method_1(v272)){
                bool v274;
                v274 = 0l <= v272;
                bool v276;
                if (v274){
                    bool v275;
                    v275 = v272 < 4l;
                    v276 = v275;
                } else {
                    v276 = false;
                }
                bool v277;
                v277 = v276 == false;
                if (v277){
                    assert("The indices should be inside the range of the dimension." && v276);
                } else {
                }
                bool v278;
                v278 = 0l <= v252;
                bool v280;
                if (v278){
                    bool v279;
                    v279 = v252 < 1l;
                    v280 = v279;
                } else {
                    v280 = false;
                }
                bool v281;
                v281 = v280 == false;
                if (v281){
                    assert("The indices should be inside the range of the dimension." && v280);
                } else {
                }
                int v282;
                v282 = v252 * 4l;
                int v283;
                v283 = v272 + v282;
                bool v284;
                v284 = 0l <= v270;
                bool v286;
                if (v284){
                    bool v285;
                    v285 = v270 < 1l;
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
                int v288;
                v288 = v270 * 4l;
                int v289;
                v289 = v283 + v288;
                assert("Tensor range check" && 0 <= v270 && v270 < 1l);
                assert("Tensor range check" && 0 <= v272 && v272 < 4l);
                int v290;
                v290 = 4l * v270;
                int v291;
                v291 = v290 + v272;
                v263[v291] = v289;
                v272 += 1l ;
            }
            v270 += 1l ;
        }
        bool v292;
        v292 = v250 && v253;
        bool v293;
        v293 = v292 == false;
        if (v293){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v292);
        } else {
        }
        bool v294;
        v294 = 0l <= v258;
        bool v296;
        if (v294){
            bool v295;
            v295 = v258 < 2l;
            v296 = v295;
        } else {
            v296 = false;
        }
        bool v297;
        v297 = v296 == false;
        if (v297){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v296);
        } else {
        }
        int v298;
        v298 = v258 * 512l;
        int v299;
        v299 = v298 + v249;
        float v300;
        v300 = 0.0f;
        int v301;
        v301 = 0l;
        while (while_method_3(v301)){
            int v303;
            v303 = 0l;
            while (while_method_1(v303)){
                assert("Tensor range check" && 0 <= v301 && v301 < 1l);
                assert("Tensor range check" && 0 <= v303 && v303 < 4l);
                int v305;
                v305 = 4l * v301;
                int v306;
                v306 = v305 + v303;
                float v307;
                v307 = v262[v306];
                float v308;
                v308 = v300 + v307;
                v300 = v308;
                v303 += 1l ;
            }
            v301 += 1l ;
        }
        auto v309 = cooperative_groups::coalesced_threads();
        int v310;
        v310 = threadIdx.x;
        auto v311 = cooperative_groups::labeled_partition(v309,v310);
        float v312;
        v312 = cooperative_groups::reduce(v311, v300, v58);
        float v313;
        v313 = v312 / 4.0f;
        float v314[4l];
        int v315;
        v315 = 0l;
        while (while_method_3(v315)){
            int v317;
            v317 = 0l;
            while (while_method_1(v317)){
                assert("Tensor range check" && 0 <= v315 && v315 < 1l);
                assert("Tensor range check" && 0 <= v317 && v317 < 4l);
                int v319;
                v319 = 4l * v315;
                int v320;
                v320 = v319 + v317;
                float v321;
                v321 = v262[v320];
                float v322;
                v322 = v321 - v313;
                float v323;
                v323 = exp(v322);
                assert("Tensor range check" && 0 <= v315 && v315 < 1l);
                assert("Tensor range check" && 0 <= v317 && v317 < 4l);
                v314[v320] = v323;
                v317 += 1l ;
            }
            v315 += 1l ;
        }
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
                v331 = v314[v330];
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
        auto v335 = cooperative_groups::labeled_partition(v333,v334);
        float v336;
        v336 = cooperative_groups::reduce(v335, v324, v58);
        float v337[4l];
        int v338;
        v338 = 0l;
        while (while_method_3(v338)){
            int v340;
            v340 = 0l;
            while (while_method_1(v340)){
                assert("Tensor range check" && 0 <= v338 && v338 < 1l);
                assert("Tensor range check" && 0 <= v340 && v340 < 4l);
                int v342;
                v342 = 4l * v338;
                int v343;
                v343 = v342 + v340;
                float v344;
                v344 = v314[v343];
                bool v345;
                v345 = v336 == 0.0f;
                bool v346;
                v346 = v345 != true;
                float v348;
                if (v346){
                    float v347;
                    v347 = v344 / v336;
                    v348 = v347;
                } else {
                    v348 = 0.25f;
                }
                assert("Tensor range check" && 0 <= v338 && v338 < 1l);
                assert("Tensor range check" && 0 <= v340 && v340 < 4l);
                v337[v343] = v348;
                v340 += 1l ;
            }
            v338 += 1l ;
        }
        int v349;
        v349 = 0l;
        while (while_method_3(v349)){
            assert("Tensor range check" && 0 <= v349 && v349 < 1l);
            int v351;
            v351 = 4l * v349;
            int v352;
            v352 = v351 + v261;
            assert("Tensor range check" && 0 <= v349 && v349 < 1l);
            int4* v353;
            v353 = reinterpret_cast<int4*>(v337 + v351);
            int4* v354;
            v354 = reinterpret_cast<int4*>(v4 + v352);
            assert("Pointer alignment check" && (unsigned long long)(v353) % 4l == 0 && (unsigned long long)(v354) % 4l == 0);
            *v354 = *v353;
            v349 += 1l ;
        }
        v258 += 1l ;
    }
    __syncthreads();
    int v355;
    v355 = threadIdx.x;
    bool v356;
    v356 = 0l <= v355;
    bool v357;
    v357 = v356 == false;
    if (v357){
        assert("The index needs to be zero or positive." && v356);
    } else {
    }
    int v358;
    v358 = v355 % 1l;
    bool v359;
    v359 = v355 < 512l;
    bool v360;
    v360 = v359 == false;
    if (v360){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v359);
    } else {
    }
    assert("Tensor range check" && 0 <= v355 && v355 < 512l);
    assert("Tensor range check" && 0 <= v358 && v358 < 1l);
    int v361;
    v361 = 4l * v358;
    int v362;
    v362 = 4l * v355;
    int v363;
    v363 = v362 + v361;
    assert("Tensor range check" && 0 <= v355 && v355 < 512l);
    assert("Tensor range check" && 0 <= v358 && v358 < 1l);
    int v364;
    v364 = 0l;
    while (while_method_2(v364)){
        assert("Tensor range check" && 0 <= v364 && v364 < 2l);
        int v366;
        v366 = 2048l * v364;
        int v367;
        v367 = v366 + v363;
        assert("Tensor range check" && 0 <= v364 && v364 < 2l);
        float v368[4l];
        int v369[4l];
        int v370;
        v370 = 0l;
        while (while_method_3(v370)){
            assert("Tensor range check" && 0 <= v370 && v370 < 1l);
            int v372;
            v372 = 4l * v370;
            assert("Tensor range check" && 0 <= v370 && v370 < 1l);
            int v373;
            v373 = v372 + v367;
            int4* v374;
            v374 = reinterpret_cast<int4*>(v1 + v373);
            int4* v375;
            v375 = reinterpret_cast<int4*>(v368 + v372);
            assert("Pointer alignment check" && (unsigned long long)(v374) % 4l == 0 && (unsigned long long)(v375) % 4l == 0);
            *v375 = *v374;
            v370 += 1l ;
        }
        int v376;
        v376 = 0l;
        while (while_method_3(v376)){
            int v378;
            v378 = 0l;
            while (while_method_1(v378)){
                bool v380;
                v380 = 0l <= v378;
                bool v382;
                if (v380){
                    bool v381;
                    v381 = v378 < 4l;
                    v382 = v381;
                } else {
                    v382 = false;
                }
                bool v383;
                v383 = v382 == false;
                if (v383){
                    assert("The indices should be inside the range of the dimension." && v382);
                } else {
                }
                bool v384;
                v384 = 0l <= v358;
                bool v386;
                if (v384){
                    bool v385;
                    v385 = v358 < 1l;
                    v386 = v385;
                } else {
                    v386 = false;
                }
                bool v387;
                v387 = v386 == false;
                if (v387){
                    assert("The indices should be inside the range of the dimension." && v386);
                } else {
                }
                int v388;
                v388 = v358 * 4l;
                int v389;
                v389 = v378 + v388;
                bool v390;
                v390 = 0l <= v376;
                bool v392;
                if (v390){
                    bool v391;
                    v391 = v376 < 1l;
                    v392 = v391;
                } else {
                    v392 = false;
                }
                bool v393;
                v393 = v392 == false;
                if (v393){
                    assert("The indices should be inside the range of the dimension." && v392);
                } else {
                }
                int v394;
                v394 = v376 * 4l;
                int v395;
                v395 = v389 + v394;
                assert("Tensor range check" && 0 <= v376 && v376 < 1l);
                assert("Tensor range check" && 0 <= v378 && v378 < 4l);
                int v396;
                v396 = 4l * v376;
                int v397;
                v397 = v396 + v378;
                v369[v397] = v395;
                v378 += 1l ;
            }
            v376 += 1l ;
        }
        bool v398;
        v398 = v356 && v359;
        bool v399;
        v399 = v398 == false;
        if (v399){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v398);
        } else {
        }
        bool v400;
        v400 = 0l <= v364;
        bool v402;
        if (v400){
            bool v401;
            v401 = v364 < 2l;
            v402 = v401;
        } else {
            v402 = false;
        }
        bool v403;
        v403 = v402 == false;
        if (v403){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v402);
        } else {
        }
        int v404;
        v404 = v364 * 512l;
        int v405;
        v405 = v404 + v355;
        float v406[4l];
        int v407;
        v407 = 0l;
        while (while_method_3(v407)){
            int v409;
            v409 = 0l;
            while (while_method_1(v409)){
                assert("Tensor range check" && 0 <= v407 && v407 < 1l);
                assert("Tensor range check" && 0 <= v409 && v409 < 4l);
                int v411;
                v411 = 4l * v407;
                int v412;
                v412 = v411 + v409;
                float v413;
                v413 = v368[v412];
                float v414;
                v414 = v413 * v413;
                assert("Tensor range check" && 0 <= v407 && v407 < 1l);
                assert("Tensor range check" && 0 <= v409 && v409 < 4l);
                v406[v412] = v414;
                v409 += 1l ;
            }
            v407 += 1l ;
        }
        float v415;
        v415 = 0.0f;
        int v416;
        v416 = 0l;
        while (while_method_3(v416)){
            int v418;
            v418 = 0l;
            while (while_method_1(v418)){
                assert("Tensor range check" && 0 <= v416 && v416 < 1l);
                assert("Tensor range check" && 0 <= v418 && v418 < 4l);
                int v420;
                v420 = 4l * v416;
                int v421;
                v421 = v420 + v418;
                float v422;
                v422 = v406[v421];
                float v423;
                v423 = v415 + v422;
                v415 = v423;
                v418 += 1l ;
            }
            v416 += 1l ;
        }
        auto v424 = cooperative_groups::coalesced_threads();
        int v425;
        v425 = threadIdx.x;
        auto v426 = cooperative_groups::labeled_partition(v424,v425);
        float v427;
        v427 = cooperative_groups::reduce(v426, v415, v58);
        float v428[4l];
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
                float v435;
                v435 = v368[v434];
                bool v436;
                v436 = v427 == 0.0f;
                bool v437;
                v437 = v436 != true;
                float v439;
                if (v437){
                    float v438;
                    v438 = v435 / v427;
                    v439 = v438;
                } else {
                    v439 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v429 && v429 < 1l);
                assert("Tensor range check" && 0 <= v431 && v431 < 4l);
                v428[v434] = v439;
                v431 += 1l ;
            }
            v429 += 1l ;
        }
        int v440;
        v440 = 0l;
        while (while_method_3(v440)){
            assert("Tensor range check" && 0 <= v440 && v440 < 1l);
            int v442;
            v442 = 4l * v440;
            int v443;
            v443 = v442 + v367;
            assert("Tensor range check" && 0 <= v440 && v440 < 1l);
            int4* v444;
            v444 = reinterpret_cast<int4*>(v428 + v442);
            int4* v445;
            v445 = reinterpret_cast<int4*>(v7 + v443);
            assert("Pointer alignment check" && (unsigned long long)(v444) % 4l == 0 && (unsigned long long)(v445) % 4l == 0);
            *v445 = *v444;
            v440 += 1l ;
        }
        v364 += 1l ;
    }
    __syncthreads();
    int v446;
    v446 = threadIdx.x;
    bool v447;
    v447 = 0l <= v446;
    bool v448;
    v448 = v447 == false;
    if (v448){
        assert("The index needs to be zero or positive." && v447);
    } else {
    }
    int v449;
    v449 = v446 % 1l;
    bool v450;
    v450 = v446 < 512l;
    bool v451;
    v451 = v450 == false;
    if (v451){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v450);
    } else {
    }
    assert("Tensor range check" && 0 <= v446 && v446 < 512l);
    assert("Tensor range check" && 0 <= v449 && v449 < 1l);
    int v452;
    v452 = 4l * v449;
    int v453;
    v453 = 4l * v446;
    int v454;
    v454 = v453 + v452;
    assert("Tensor range check" && 0 <= v446 && v446 < 512l);
    int v455;
    v455 = 0l;
    while (while_method_2(v455)){
        assert("Tensor range check" && 0 <= v455 && v455 < 2l);
        int v457;
        v457 = 2048l * v455;
        int v458;
        v458 = v457 + v454;
        float v459[4l];
        int v460[4l];
        int v461;
        v461 = 0l;
        while (while_method_3(v461)){
            assert("Tensor range check" && 0 <= v461 && v461 < 1l);
            int v463;
            v463 = 4l * v461;
            assert("Tensor range check" && 0 <= v461 && v461 < 1l);
            int v464;
            v464 = v463 + v458;
            int4* v465;
            v465 = reinterpret_cast<int4*>(v1 + v464);
            int4* v466;
            v466 = reinterpret_cast<int4*>(v459 + v463);
            assert("Pointer alignment check" && (unsigned long long)(v465) % 4l == 0 && (unsigned long long)(v466) % 4l == 0);
            *v466 = *v465;
            v461 += 1l ;
        }
        int v467;
        v467 = 0l;
        while (while_method_3(v467)){
            int v469;
            v469 = 0l;
            while (while_method_1(v469)){
                bool v471;
                v471 = 0l <= v469;
                bool v473;
                if (v471){
                    bool v472;
                    v472 = v469 < 4l;
                    v473 = v472;
                } else {
                    v473 = false;
                }
                bool v474;
                v474 = v473 == false;
                if (v474){
                    assert("The indices should be inside the range of the dimension." && v473);
                } else {
                }
                bool v475;
                v475 = 0l <= v449;
                bool v477;
                if (v475){
                    bool v476;
                    v476 = v449 < 1l;
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
                int v479;
                v479 = v449 * 4l;
                int v480;
                v480 = v469 + v479;
                bool v481;
                v481 = 0l <= v467;
                bool v483;
                if (v481){
                    bool v482;
                    v482 = v467 < 1l;
                    v483 = v482;
                } else {
                    v483 = false;
                }
                bool v484;
                v484 = v483 == false;
                if (v484){
                    assert("The indices should be inside the range of the dimension." && v483);
                } else {
                }
                int v485;
                v485 = v467 * 4l;
                int v486;
                v486 = v480 + v485;
                assert("Tensor range check" && 0 <= v467 && v467 < 1l);
                assert("Tensor range check" && 0 <= v469 && v469 < 4l);
                int v487;
                v487 = 4l * v467;
                int v488;
                v488 = v487 + v469;
                v460[v488] = v486;
                v469 += 1l ;
            }
            v467 += 1l ;
        }
        bool v489;
        v489 = v447 && v450;
        bool v490;
        v490 = v489 == false;
        if (v490){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v489);
        } else {
        }
        bool v491;
        v491 = 0l <= v455;
        bool v493;
        if (v491){
            bool v492;
            v492 = v455 < 2l;
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
        int v495;
        v495 = v455 * 512l;
        int v496;
        v496 = v495 + v446;
        float v497; int v498;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0l};
        v497 = tmp1.v0; v498 = tmp1.v1;
        int v499;
        v499 = 0l;
        while (while_method_3(v499)){
            int v501;
            v501 = 0l;
            while (while_method_1(v501)){
                assert("Tensor range check" && 0 <= v499 && v499 < 1l);
                assert("Tensor range check" && 0 <= v501 && v501 < 4l);
                int v503;
                v503 = 4l * v499;
                int v504;
                v504 = v503 + v501;
                float v505;
                v505 = v459[v504];
                int v506;
                v506 = v460[v504];
                bool v507;
                v507 = v497 > v505;
                float v508; int v509;
                if (v507){
                    v508 = v497; v509 = v498;
                } else {
                    v508 = v505; v509 = v506;
                }
                v497 = v508;
                v498 = v509;
                v501 += 1l ;
            }
            v499 += 1l ;
        }
        auto v510 = cooperative_groups::coalesced_threads();
        int v511;
        v511 = threadIdx.x;
        auto v512 = cooperative_groups::labeled_partition(v510,v511);
        Closure1 v513{};
        float v514; int v515;
        Tuple1 tmp2 = cooperative_groups::reduce(v512, Tuple1{v497, v498}, v513);
        v514 = tmp2.v0; v515 = tmp2.v1;
        assert("Tensor range check" && 0 <= v455 && v455 < 2l);
        int v516;
        v516 = 512l * v455;
        int v517;
        v517 = v516 + v446;
        v8[v517] = v515;
        v455 += 1l ;
    }
    __syncthreads();
    int v518;
    v518 = threadIdx.x;
    bool v519;
    v519 = 0l <= v518;
    bool v520;
    v520 = v519 == false;
    if (v520){
        assert("The index needs to be zero or positive." && v519);
    } else {
    }
    int v521;
    v521 = v518 % 1l;
    bool v522;
    v522 = v518 < 512l;
    bool v523;
    v523 = v522 == false;
    if (v523){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v522);
    } else {
    }
    assert("Tensor range check" && 0 <= v518 && v518 < 512l);
    assert("Tensor range check" && 0 <= v521 && v521 < 1l);
    int v524;
    v524 = 4l * v521;
    int v525;
    v525 = 4l * v518;
    int v526;
    v526 = v525 + v524;
    assert("Tensor range check" && 0 <= v518 && v518 < 512l);
    assert("Tensor range check" && 0 <= v521 && v521 < 1l);
    int v527;
    v527 = 0l;
    while (while_method_2(v527)){
        assert("Tensor range check" && 0 <= v527 && v527 < 2l);
        int v529;
        v529 = 2048l * v527;
        int v530;
        v530 = v529 + v526;
        assert("Tensor range check" && 0 <= v527 && v527 < 2l);
        float v531[4l];
        int v532[4l];
        int v533;
        v533 = 0l;
        while (while_method_3(v533)){
            assert("Tensor range check" && 0 <= v533 && v533 < 1l);
            int v535;
            v535 = 4l * v533;
            assert("Tensor range check" && 0 <= v533 && v533 < 1l);
            int v536;
            v536 = v535 + v530;
            int4* v537;
            v537 = reinterpret_cast<int4*>(v1 + v536);
            int4* v538;
            v538 = reinterpret_cast<int4*>(v531 + v535);
            assert("Pointer alignment check" && (unsigned long long)(v537) % 4l == 0 && (unsigned long long)(v538) % 4l == 0);
            *v538 = *v537;
            v533 += 1l ;
        }
        int v539;
        v539 = 0l;
        while (while_method_3(v539)){
            int v541;
            v541 = 0l;
            while (while_method_1(v541)){
                bool v543;
                v543 = 0l <= v541;
                bool v545;
                if (v543){
                    bool v544;
                    v544 = v541 < 4l;
                    v545 = v544;
                } else {
                    v545 = false;
                }
                bool v546;
                v546 = v545 == false;
                if (v546){
                    assert("The indices should be inside the range of the dimension." && v545);
                } else {
                }
                bool v547;
                v547 = 0l <= v521;
                bool v549;
                if (v547){
                    bool v548;
                    v548 = v521 < 1l;
                    v549 = v548;
                } else {
                    v549 = false;
                }
                bool v550;
                v550 = v549 == false;
                if (v550){
                    assert("The indices should be inside the range of the dimension." && v549);
                } else {
                }
                int v551;
                v551 = v521 * 4l;
                int v552;
                v552 = v541 + v551;
                bool v553;
                v553 = 0l <= v539;
                bool v555;
                if (v553){
                    bool v554;
                    v554 = v539 < 1l;
                    v555 = v554;
                } else {
                    v555 = false;
                }
                bool v556;
                v556 = v555 == false;
                if (v556){
                    assert("The indices should be inside the range of the dimension." && v555);
                } else {
                }
                int v557;
                v557 = v539 * 4l;
                int v558;
                v558 = v552 + v557;
                assert("Tensor range check" && 0 <= v539 && v539 < 1l);
                assert("Tensor range check" && 0 <= v541 && v541 < 4l);
                int v559;
                v559 = 4l * v539;
                int v560;
                v560 = v559 + v541;
                v532[v560] = v558;
                v541 += 1l ;
            }
            v539 += 1l ;
        }
        bool v561;
        v561 = v519 && v522;
        bool v562;
        v562 = v561 == false;
        if (v562){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v561);
        } else {
        }
        bool v563;
        v563 = 0l <= v527;
        bool v565;
        if (v563){
            bool v564;
            v564 = v527 < 2l;
            v565 = v564;
        } else {
            v565 = false;
        }
        bool v566;
        v566 = v565 == false;
        if (v566){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v565);
        } else {
        }
        int v567;
        v567 = v527 * 512l;
        int v568;
        v568 = v567 + v518;
        float v569;
        v569 = 0.0f;
        int v570;
        v570 = 0l;
        while (while_method_3(v570)){
            int v572;
            v572 = 0l;
            while (while_method_1(v572)){
                assert("Tensor range check" && 0 <= v570 && v570 < 1l);
                assert("Tensor range check" && 0 <= v572 && v572 < 4l);
                int v574;
                v574 = 4l * v570;
                int v575;
                v575 = v574 + v572;
                float v576;
                v576 = v531[v575];
                float v577;
                v577 = v569 + v576;
                v569 = v577;
                v572 += 1l ;
            }
            v570 += 1l ;
        }
        auto v578 = cooperative_groups::coalesced_threads();
        int v579;
        v579 = threadIdx.x;
        auto v580 = cooperative_groups::labeled_partition(v578,v579);
        float v581;
        v581 = cooperative_groups::reduce(v580, v569, v58);
        float v582;
        v582 = v581 / 4.0f;
        float v583[4l];
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
                v590 = v531[v589];
                float v591;
                v591 = v590 - v582;
                float v592;
                v592 = exp(v591);
                assert("Tensor range check" && 0 <= v584 && v584 < 1l);
                assert("Tensor range check" && 0 <= v586 && v586 < 4l);
                v583[v589] = v592;
                v586 += 1l ;
            }
            v584 += 1l ;
        }
        float v593;
        v593 = 0.0f;
        int v594;
        v594 = 0l;
        while (while_method_3(v594)){
            int v596;
            v596 = 0l;
            while (while_method_1(v596)){
                assert("Tensor range check" && 0 <= v594 && v594 < 1l);
                assert("Tensor range check" && 0 <= v596 && v596 < 4l);
                int v598;
                v598 = 4l * v594;
                int v599;
                v599 = v598 + v596;
                float v600;
                v600 = v583[v599];
                float v601;
                v601 = v593 + v600;
                v593 = v601;
                v596 += 1l ;
            }
            v594 += 1l ;
        }
        auto v602 = cooperative_groups::coalesced_threads();
        int v603;
        v603 = threadIdx.x;
        auto v604 = cooperative_groups::labeled_partition(v602,v603);
        float v605;
        v605 = cooperative_groups::reduce(v604, v593, v58);
        float v606[4l];
        int v607;
        v607 = 0l;
        while (while_method_3(v607)){
            int v609;
            v609 = 0l;
            while (while_method_1(v609)){
                assert("Tensor range check" && 0 <= v607 && v607 < 1l);
                assert("Tensor range check" && 0 <= v609 && v609 < 4l);
                int v611;
                v611 = 4l * v607;
                int v612;
                v612 = v611 + v609;
                float v613;
                v613 = v583[v612];
                bool v614;
                v614 = v605 == 0.0f;
                bool v615;
                v615 = v614 != true;
                float v617;
                if (v615){
                    float v616;
                    v616 = v613 / v605;
                    v617 = v616;
                } else {
                    v617 = 0.25f;
                }
                assert("Tensor range check" && 0 <= v607 && v607 < 1l);
                assert("Tensor range check" && 0 <= v609 && v609 < 4l);
                v606[v612] = v617;
                v609 += 1l ;
            }
            v607 += 1l ;
        }
        float v618[4l];
        float v619;
        v619 = 0.0f;
        int v620;
        v620 = 0l;
        while (while_method_3(v620)){
            assert("Tensor range check" && 0 <= v620 && v620 < 1l);
            int v622;
            v622 = 4l * v620;
            assert("Tensor range check" && 0 <= v620 && v620 < 1l);
            int v623; float v624;
            Tuple0 tmp3 = Tuple0{0l, 0.0f};
            v623 = tmp3.v0; v624 = tmp3.v1;
            while (while_method_1(v623)){
                assert("Tensor range check" && 0 <= v623 && v623 < 4l);
                int v626;
                v626 = v623 + v622;
                float v627;
                v627 = v606[v626];
                float v628;
                v628 = v624 + v627;
                v624 = v628;
                v623 += 1l ;
            }
            auto v629 = cooperative_groups::coalesced_threads();
            int v630;
            v630 = threadIdx.x;
            auto v631 = cooperative_groups::labeled_partition(v629,v630);
            Closure2 v632{};
            float v633;
            v633 = cooperative_groups::inclusive_scan(v631, v624, v632);
            float v634;
            v634 = v631.shfl_up(v633,1);
            bool v635;
            v635 = v631.thread_rank() == 0;
            float v636;
            if (v635){
                v636 = 0.0f;
            } else {
                v636 = v634;
            }
            float v637;
            v637 = v631.shfl(v633,v631.num_threads()-1);
            float v638;
            v638 = v619 + v636;
            int v639; float v640;
            Tuple0 tmp4 = Tuple0{0l, v638};
            v639 = tmp4.v0; v640 = tmp4.v1;
            while (while_method_1(v639)){
                assert("Tensor range check" && 0 <= v639 && v639 < 4l);
                int v642;
                v642 = v639 + v622;
                float v643;
                v643 = v606[v642];
                float v644;
                v644 = v640 + v643;
                assert("Tensor range check" && 0 <= v639 && v639 < 4l);
                v618[v642] = v644;
                v640 = v644;
                v639 += 1l ;
            }
            float v645;
            v645 = v619 + v637;
            v619 = v645;
            v620 += 1l ;
        }
        int v646;
        v646 = 0l;
        while (while_method_3(v646)){
            assert("Tensor range check" && 0 <= v646 && v646 < 1l);
            int v648;
            v648 = 4l * v646;
            int v649;
            v649 = v648 + v530;
            assert("Tensor range check" && 0 <= v646 && v646 < 1l);
            int4* v650;
            v650 = reinterpret_cast<int4*>(v606 + v648);
            int4* v651;
            v651 = reinterpret_cast<int4*>(v5 + v649);
            assert("Pointer alignment check" && (unsigned long long)(v650) % 4l == 0 && (unsigned long long)(v651) % 4l == 0);
            *v651 = *v650;
            int4* v652;
            v652 = reinterpret_cast<int4*>(v618 + v648);
            int4* v653;
            v653 = reinterpret_cast<int4*>(v6 + v649);
            assert("Pointer alignment check" && (unsigned long long)(v652) % 4l == 0 && (unsigned long long)(v653) % 4l == 0);
            *v653 = *v652;
            v646 += 1l ;
        }
        v527 += 1l ;
    }
    __syncthreads();
    int v654;
    v654 = threadIdx.x;
    bool v655;
    v655 = 0l <= v654;
    bool v656;
    v656 = v655 == false;
    if (v656){
        assert("The index needs to be zero or positive." && v655);
    } else {
    }
    int v657;
    v657 = v654 % 1l;
    bool v658;
    v658 = v654 < 512l;
    bool v659;
    v659 = v658 == false;
    if (v659){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v658);
    } else {
    }
    assert("Tensor range check" && 0 <= v654 && v654 < 512l);
    assert("Tensor range check" && 0 <= v657 && v657 < 1l);
    int v660;
    v660 = 4l * v657;
    int v661;
    v661 = 4l * v654;
    int v662;
    v662 = v661 + v660;
    assert("Tensor range check" && 0 <= v654 && v654 < 512l);
    int v663;
    v663 = 0l;
    while (while_method_2(v663)){
        assert("Tensor range check" && 0 <= v663 && v663 < 2l);
        int v665;
        v665 = 2048l * v663;
        int v666;
        v666 = v665 + v662;
        float v667[4l];
        int v668[4l];
        int v669;
        v669 = 0l;
        while (while_method_3(v669)){
            assert("Tensor range check" && 0 <= v669 && v669 < 1l);
            int v671;
            v671 = 4l * v669;
            assert("Tensor range check" && 0 <= v669 && v669 < 1l);
            int v672;
            v672 = v671 + v666;
            int4* v673;
            v673 = reinterpret_cast<int4*>(v1 + v672);
            int4* v674;
            v674 = reinterpret_cast<int4*>(v667 + v671);
            assert("Pointer alignment check" && (unsigned long long)(v673) % 4l == 0 && (unsigned long long)(v674) % 4l == 0);
            *v674 = *v673;
            v669 += 1l ;
        }
        int v675;
        v675 = 0l;
        while (while_method_3(v675)){
            int v677;
            v677 = 0l;
            while (while_method_1(v677)){
                bool v679;
                v679 = 0l <= v677;
                bool v681;
                if (v679){
                    bool v680;
                    v680 = v677 < 4l;
                    v681 = v680;
                } else {
                    v681 = false;
                }
                bool v682;
                v682 = v681 == false;
                if (v682){
                    assert("The indices should be inside the range of the dimension." && v681);
                } else {
                }
                bool v683;
                v683 = 0l <= v657;
                bool v685;
                if (v683){
                    bool v684;
                    v684 = v657 < 1l;
                    v685 = v684;
                } else {
                    v685 = false;
                }
                bool v686;
                v686 = v685 == false;
                if (v686){
                    assert("The indices should be inside the range of the dimension." && v685);
                } else {
                }
                int v687;
                v687 = v657 * 4l;
                int v688;
                v688 = v677 + v687;
                bool v689;
                v689 = 0l <= v675;
                bool v691;
                if (v689){
                    bool v690;
                    v690 = v675 < 1l;
                    v691 = v690;
                } else {
                    v691 = false;
                }
                bool v692;
                v692 = v691 == false;
                if (v692){
                    assert("The indices should be inside the range of the dimension." && v691);
                } else {
                }
                int v693;
                v693 = v675 * 4l;
                int v694;
                v694 = v688 + v693;
                assert("Tensor range check" && 0 <= v675 && v675 < 1l);
                assert("Tensor range check" && 0 <= v677 && v677 < 4l);
                int v695;
                v695 = 4l * v675;
                int v696;
                v696 = v695 + v677;
                v668[v696] = v694;
                v677 += 1l ;
            }
            v675 += 1l ;
        }
        bool v697;
        v697 = v655 && v658;
        bool v698;
        v698 = v697 == false;
        if (v698){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v697);
        } else {
        }
        bool v699;
        v699 = 0l <= v663;
        bool v701;
        if (v699){
            bool v700;
            v700 = v663 < 2l;
            v701 = v700;
        } else {
            v701 = false;
        }
        bool v702;
        v702 = v701 == false;
        if (v702){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v701);
        } else {
        }
        int v703;
        v703 = v663 * 512l;
        int v704;
        v704 = v703 + v654;
        float v705;
        v705 = 0.0f;
        int v706;
        v706 = 0l;
        while (while_method_3(v706)){
            int v708;
            v708 = 0l;
            while (while_method_1(v708)){
                assert("Tensor range check" && 0 <= v706 && v706 < 1l);
                assert("Tensor range check" && 0 <= v708 && v708 < 4l);
                int v710;
                v710 = 4l * v706;
                int v711;
                v711 = v710 + v708;
                float v712;
                v712 = v667[v711];
                float v713;
                v713 = v705 + v712;
                v705 = v713;
                v708 += 1l ;
            }
            v706 += 1l ;
        }
        auto v714 = cooperative_groups::coalesced_threads();
        int v715;
        v715 = threadIdx.x;
        auto v716 = cooperative_groups::labeled_partition(v714,v715);
        float v717;
        v717 = cooperative_groups::reduce(v716, v705, v58);
        float v718;
        v718 = v717 / 4.0f;
        float v719[4l];
        int v720;
        v720 = 0l;
        while (while_method_3(v720)){
            int v722;
            v722 = 0l;
            while (while_method_1(v722)){
                assert("Tensor range check" && 0 <= v720 && v720 < 1l);
                assert("Tensor range check" && 0 <= v722 && v722 < 4l);
                int v724;
                v724 = 4l * v720;
                int v725;
                v725 = v724 + v722;
                float v726;
                v726 = v667[v725];
                float v727;
                v727 = v726 - v718;
                float v728;
                v728 = exp(v727);
                assert("Tensor range check" && 0 <= v720 && v720 < 1l);
                assert("Tensor range check" && 0 <= v722 && v722 < 4l);
                v719[v725] = v728;
                v722 += 1l ;
            }
            v720 += 1l ;
        }
        float v729;
        v729 = 0.0f;
        int v730;
        v730 = 0l;
        while (while_method_3(v730)){
            int v732;
            v732 = 0l;
            while (while_method_1(v732)){
                assert("Tensor range check" && 0 <= v730 && v730 < 1l);
                assert("Tensor range check" && 0 <= v732 && v732 < 4l);
                int v734;
                v734 = 4l * v730;
                int v735;
                v735 = v734 + v732;
                float v736;
                v736 = v719[v735];
                float v737;
                v737 = v729 + v736;
                v729 = v737;
                v732 += 1l ;
            }
            v730 += 1l ;
        }
        auto v738 = cooperative_groups::coalesced_threads();
        int v739;
        v739 = threadIdx.x;
        auto v740 = cooperative_groups::labeled_partition(v738,v739);
        float v741;
        v741 = cooperative_groups::reduce(v740, v729, v58);
        float v742[4l];
        int v743;
        v743 = 0l;
        while (while_method_3(v743)){
            int v745;
            v745 = 0l;
            while (while_method_1(v745)){
                assert("Tensor range check" && 0 <= v743 && v743 < 1l);
                assert("Tensor range check" && 0 <= v745 && v745 < 4l);
                int v747;
                v747 = 4l * v743;
                int v748;
                v748 = v747 + v745;
                float v749;
                v749 = v719[v748];
                bool v750;
                v750 = v741 == 0.0f;
                bool v751;
                v751 = v750 != true;
                float v753;
                if (v751){
                    float v752;
                    v752 = v749 / v741;
                    v753 = v752;
                } else {
                    v753 = 0.25f;
                }
                assert("Tensor range check" && 0 <= v743 && v743 < 1l);
                assert("Tensor range check" && 0 <= v745 && v745 < 4l);
                v742[v748] = v753;
                v745 += 1l ;
            }
            v743 += 1l ;
        }
        float v754[4l];
        float v755;
        v755 = 0.0f;
        int v756;
        v756 = 0l;
        while (while_method_3(v756)){
            assert("Tensor range check" && 0 <= v756 && v756 < 1l);
            int v758;
            v758 = 4l * v756;
            assert("Tensor range check" && 0 <= v756 && v756 < 1l);
            int v759; float v760;
            Tuple0 tmp5 = Tuple0{0l, 0.0f};
            v759 = tmp5.v0; v760 = tmp5.v1;
            while (while_method_1(v759)){
                assert("Tensor range check" && 0 <= v759 && v759 < 4l);
                int v762;
                v762 = v759 + v758;
                float v763;
                v763 = v742[v762];
                float v764;
                v764 = v760 + v763;
                v760 = v764;
                v759 += 1l ;
            }
            auto v765 = cooperative_groups::coalesced_threads();
            int v766;
            v766 = threadIdx.x;
            auto v767 = cooperative_groups::labeled_partition(v765,v766);
            Closure2 v768{};
            float v769;
            v769 = cooperative_groups::inclusive_scan(v767, v760, v768);
            float v770;
            v770 = v767.shfl_up(v769,1);
            bool v771;
            v771 = v767.thread_rank() == 0;
            float v772;
            if (v771){
                v772 = 0.0f;
            } else {
                v772 = v770;
            }
            float v773;
            v773 = v767.shfl(v769,v767.num_threads()-1);
            float v774;
            v774 = v755 + v772;
            int v775; float v776;
            Tuple0 tmp6 = Tuple0{0l, v774};
            v775 = tmp6.v0; v776 = tmp6.v1;
            while (while_method_1(v775)){
                assert("Tensor range check" && 0 <= v775 && v775 < 4l);
                int v778;
                v778 = v775 + v758;
                float v779;
                v779 = v742[v778];
                float v780;
                v780 = v776 + v779;
                assert("Tensor range check" && 0 <= v775 && v775 < 4l);
                v754[v778] = v780;
                v776 = v780;
                v775 += 1l ;
            }
            float v781;
            v781 = v755 + v773;
            v755 = v781;
            v756 += 1l ;
        }
        float v782;
        v782 = curand_uniform(&v15);
        float v783[4l];
        int v784;
        v784 = 0l;
        while (while_method_3(v784)){
            int v786;
            v786 = 0l;
            while (while_method_1(v786)){
                assert("Tensor range check" && 0 <= v784 && v784 < 1l);
                assert("Tensor range check" && 0 <= v786 && v786 < 4l);
                int v788;
                v788 = 4l * v784;
                int v789;
                v789 = v788 + v786;
                float v790;
                v790 = v754[v789];
                float v791;
                v791 = v790 - v782;
                assert("Tensor range check" && 0 <= v784 && v784 < 1l);
                assert("Tensor range check" && 0 <= v786 && v786 < 4l);
                v783[v789] = v791;
                v786 += 1l ;
            }
            v784 += 1l ;
        }
        float v792; int v793;
        Tuple1 tmp7 = Tuple1{-1.0f / 0.0f, 0l};
        v792 = tmp7.v0; v793 = tmp7.v1;
        int v794;
        v794 = 0l;
        while (while_method_3(v794)){
            int v796;
            v796 = 0l;
            while (while_method_1(v796)){
                assert("Tensor range check" && 0 <= v794 && v794 < 1l);
                assert("Tensor range check" && 0 <= v796 && v796 < 4l);
                int v798;
                v798 = 4l * v794;
                int v799;
                v799 = v798 + v796;
                float v800;
                v800 = v783[v799];
                int v801;
                v801 = v668[v799];
                bool v802;
                v802 = v792 >= 0.0f;
                bool v804;
                if (v802){
                    bool v803;
                    v803 = v800 >= 0.0f;
                    v804 = v803;
                } else {
                    v804 = false;
                }
                float v813; int v814;
                if (v804){
                    bool v805;
                    v805 = v792 <= v800;
                    if (v805){
                        v813 = v792; v814 = v793;
                    } else {
                        v813 = v800; v814 = v801;
                    }
                } else {
                    if (v802){
                        v813 = v792; v814 = v793;
                    } else {
                        bool v808;
                        v808 = v800 >= 0.0f;
                        if (v808){
                            v813 = v800; v814 = v801;
                        } else {
                            v813 = v792; v814 = v793;
                        }
                    }
                }
                v792 = v813;
                v793 = v814;
                v796 += 1l ;
            }
            v794 += 1l ;
        }
        auto v815 = cooperative_groups::coalesced_threads();
        int v816;
        v816 = threadIdx.x;
        auto v817 = cooperative_groups::labeled_partition(v815,v816);
        Closure3 v818{};
        float v819; int v820;
        Tuple1 tmp8 = cooperative_groups::reduce(v817, Tuple1{v792, v793}, v818);
        v819 = tmp8.v0; v820 = tmp8.v1;
        assert("Tensor range check" && 0 <= v663 && v663 < 2l);
        int v821;
        v821 = 512l * v663;
        int v822;
        v822 = v821 + v654;
        v9[v822] = v820;
        v663 += 1l ;
    }
    __syncthreads();
    int v823;
    v823 = threadIdx.x;
    bool v824;
    v824 = 0l <= v823;
    bool v825;
    v825 = v824 == false;
    if (v825){
        assert("The index needs to be zero or positive." && v824);
    } else {
    }
    int v826;
    v826 = v823 % 1l;
    bool v827;
    v827 = v823 < 512l;
    bool v828;
    v828 = v827 == false;
    if (v828){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v827);
    } else {
    }
    assert("Tensor range check" && 0 <= v823 && v823 < 512l);
    assert("Tensor range check" && 0 <= v826 && v826 < 1l);
    int v829;
    v829 = 4l * v826;
    int v830;
    v830 = 4l * v823;
    int v831;
    v831 = v830 + v829;
    assert("Tensor range check" && 0 <= v823 && v823 < 512l);
    assert("Tensor range check" && 0 <= v826 && v826 < 1l);
    int v832;
    v832 = 0l;
    while (while_method_2(v832)){
        assert("Tensor range check" && 0 <= v832 && v832 < 2l);
        int v834;
        v834 = 2048l * v832;
        int v835;
        v835 = v834 + v831;
        assert("Tensor range check" && 0 <= v832 && v832 < 2l);
        int v836[4l];
        int v837[4l];
        int v838;
        v838 = 0l;
        while (while_method_3(v838)){
            assert("Tensor range check" && 0 <= v838 && v838 < 1l);
            int v840;
            v840 = 4l * v838;
            assert("Tensor range check" && 0 <= v838 && v838 < 1l);
            int v841;
            v841 = v840 + v835;
            int4* v842;
            v842 = reinterpret_cast<int4*>(v0 + v841);
            int4* v843;
            v843 = reinterpret_cast<int4*>(v836 + v840);
            assert("Pointer alignment check" && (unsigned long long)(v842) % 4l == 0 && (unsigned long long)(v843) % 4l == 0);
            *v843 = *v842;
            v838 += 1l ;
        }
        int v844;
        v844 = 0l;
        while (while_method_3(v844)){
            int v846;
            v846 = 0l;
            while (while_method_1(v846)){
                bool v848;
                v848 = 0l <= v846;
                bool v850;
                if (v848){
                    bool v849;
                    v849 = v846 < 4l;
                    v850 = v849;
                } else {
                    v850 = false;
                }
                bool v851;
                v851 = v850 == false;
                if (v851){
                    assert("The indices should be inside the range of the dimension." && v850);
                } else {
                }
                bool v852;
                v852 = 0l <= v826;
                bool v854;
                if (v852){
                    bool v853;
                    v853 = v826 < 1l;
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
                int v856;
                v856 = v826 * 4l;
                int v857;
                v857 = v846 + v856;
                bool v858;
                v858 = 0l <= v844;
                bool v860;
                if (v858){
                    bool v859;
                    v859 = v844 < 1l;
                    v860 = v859;
                } else {
                    v860 = false;
                }
                bool v861;
                v861 = v860 == false;
                if (v861){
                    assert("The indices should be inside the range of the dimension." && v860);
                } else {
                }
                int v862;
                v862 = v844 * 4l;
                int v863;
                v863 = v857 + v862;
                assert("Tensor range check" && 0 <= v844 && v844 < 1l);
                assert("Tensor range check" && 0 <= v846 && v846 < 4l);
                int v864;
                v864 = 4l * v844;
                int v865;
                v865 = v864 + v846;
                v837[v865] = v863;
                v846 += 1l ;
            }
            v844 += 1l ;
        }
        bool v866;
        v866 = v824 && v827;
        bool v867;
        v867 = v866 == false;
        if (v867){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v866);
        } else {
        }
        bool v868;
        v868 = 0l <= v832;
        bool v870;
        if (v868){
            bool v869;
            v869 = v832 < 2l;
            v870 = v869;
        } else {
            v870 = false;
        }
        bool v871;
        v871 = v870 == false;
        if (v871){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v870);
        } else {
        }
        int v872;
        v872 = v832 * 512l;
        int v873;
        v873 = v872 + v823;
        int v874[4l];
        int v875;
        v875 = 0l;
        int v876;
        v876 = 0l;
        while (while_method_3(v876)){
            assert("Tensor range check" && 0 <= v876 && v876 < 1l);
            int v878;
            v878 = 4l * v876;
            assert("Tensor range check" && 0 <= v876 && v876 < 1l);
            int v879; int v880;
            Tuple2 tmp9 = Tuple2{0l, 0l};
            v879 = tmp9.v0; v880 = tmp9.v1;
            while (while_method_1(v879)){
                assert("Tensor range check" && 0 <= v879 && v879 < 4l);
                int v882;
                v882 = v879 + v878;
                int v883;
                v883 = v836[v882];
                int v884;
                v884 = v880 + v883;
                v880 = v884;
                v879 += 1l ;
            }
            auto v885 = cooperative_groups::coalesced_threads();
            int v886;
            v886 = threadIdx.x;
            auto v887 = cooperative_groups::labeled_partition(v885,v886);
            Closure4 v888{};
            int v889;
            v889 = cooperative_groups::inclusive_scan(v887, v880, v888);
            int v890;
            v890 = v887.shfl_up(v889,1);
            bool v891;
            v891 = v887.thread_rank() == 0;
            int v892;
            if (v891){
                v892 = 0l;
            } else {
                v892 = v890;
            }
            int v893;
            v893 = v887.shfl(v889,v887.num_threads()-1);
            int v894;
            v894 = v875 + v892;
            int v895; int v896;
            Tuple2 tmp10 = Tuple2{0l, v894};
            v895 = tmp10.v0; v896 = tmp10.v1;
            while (while_method_1(v895)){
                assert("Tensor range check" && 0 <= v895 && v895 < 4l);
                int v898;
                v898 = v895 + v878;
                int v899;
                v899 = v836[v898];
                assert("Tensor range check" && 0 <= v895 && v895 < 4l);
                v874[v898] = v896;
                int v900;
                v900 = v896 + v899;
                v896 = v900;
                v895 += 1l ;
            }
            int v901;
            v901 = v875 + v893;
            v875 = v901;
            v876 += 1l ;
        }
        int v902;
        v902 = 0l;
        while (while_method_3(v902)){
            assert("Tensor range check" && 0 <= v902 && v902 < 1l);
            int v904;
            v904 = 4l * v902;
            int v905;
            v905 = v904 + v835;
            assert("Tensor range check" && 0 <= v902 && v902 < 1l);
            int4* v906;
            v906 = reinterpret_cast<int4*>(v874 + v904);
            int4* v907;
            v907 = reinterpret_cast<int4*>(v13 + v905);
            assert("Pointer alignment check" && (unsigned long long)(v906) % 4l == 0 && (unsigned long long)(v907) % 4l == 0);
            *v907 = *v906;
            v902 += 1l ;
        }
        v832 += 1l ;
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
options.append('--maxrregcount=128')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0(v0 : char) -> None:
    print(v0, end="")
    del v0
    return 
def method1(v0 : i32) -> bool:
    v1 = v0 < 1024
    del v0
    return v1
def method2(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method3(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method4(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def method5() -> None:
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
    v6 = cp.random.uniform(size=1024,dtype=cp.float32) # type: ignore
    del v6
    v7 = cp.empty(1,dtype=cp.float32)
    v8 = cp.empty(4096,dtype=cp.int32)
    v9 = cp.empty(4096,dtype=cp.float32)
    v10 = cp.empty(4096,dtype=cp.float32)
    v11 = cp.empty(4096,dtype=cp.float32)
    v12 = cp.empty(4096,dtype=cp.float32)
    v13 = cp.empty(1024,dtype=cp.int32)
    v14 = cp.empty(1024,dtype=cp.int32)
    v15 = cp.empty(4096,dtype=cp.int32)
    v16 = cp.empty(4096,dtype=cp.int32)
    v17 = cp.empty(1024,dtype=cp.int32)
    v18 = cp.empty(4096,dtype=cp.int32)
    v19 = 0
    v20 = raw_module.get_function(f"entry{v19}")
    del v19
    v20.max_dynamic_shared_size_bytes = 0 
    v20((1,),(512,),(v0, v5, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18),shared_mem=0)
    del v0, v5, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v20
    v21 = 0
    v22 = '['
    method0(v22)
    del v22
    v23 = 0
    while method1(v23):
        v25 = v21
        v26 = v25 >= 4096
        del v25
        if v26:
            v27 = " ..."
            method2(v27)
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
            method2(v30)
        else:
            pass
        del v29
        v31 = '['
        method0(v31)
        del v31
        v32 = 0
        while method3(v32):
            v34 = v21
            v35 = v34 >= 4096
            del v34
            if v35:
                v36 = " ..."
                method2(v36)
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
                method2(v39)
            else:
                pass
            del v38
            v40 = v21 + 1
            v21 = v40
            del v40
            v41 = v23 * 4
            v42 = v41 + v32
            del v41
            v43 = v18[v42].item()
            del v42
            method4(v43)
            del v43
            v32 += 1 
        del v32
        v44 = ']'
        method0(v44)
        del v44
        v23 += 1 
    del v18, v21, v23
    v45 = ']'
    method0(v45)
    del v45
    method5()
    print()
    return 

if __name__ == '__main__': print(main())
