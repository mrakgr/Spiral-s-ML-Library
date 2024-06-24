kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
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
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 65536l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 128l;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
extern "C" __global__ void entry0(int * v0, float * v1, float * v2, float * v3, int * v4, float * v5, float * v6, float * v7, float * v8, int * v9, int * v10, int * v11, int * v12, int * v13) {
    int v14;
    v14 = threadIdx.x;
    int v15;
    v15 = v14;
    while (while_method_0(v15)){
        bool v17;
        v17 = 0l <= v15;
        bool v18;
        v18 = v17 == false;
        if (v18){
            assert("The index needs to be zero or positive." && v17);
        } else {
        }
        int v19;
        v19 = v15 % 64l;
        int v20;
        v20 = v15 / 64l;
        bool v21;
        v21 = v20 < 1024l;
        bool v22;
        v22 = v21 == false;
        if (v22){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v21);
        } else {
        }
        assert("Tensor range check" && 0 <= v20 && v20 < 1024l);
        assert("Tensor range check" && 0 <= v19 && v19 < 64l);
        int v23;
        v23 = 4l * v19;
        int v24;
        v24 = 256l * v20;
        int v25;
        v25 = v24 + v23;
        assert("Tensor range check" && 0 <= v20 && v20 < 1024l);
        assert("Tensor range check" && 0 <= v19 && v19 < 64l);
        float v26[4l];
        float v27[4l];
        int4* v28;
        v28 = reinterpret_cast<int4*>(v1 + v25);
        int4* v29;
        v29 = reinterpret_cast<int4*>(v26 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v28) % 4l == 0 && (unsigned long long)(v29) % 4l == 0);
        *v29 = *v28;
        // Pushing the loop unrolling to: 0
        int v30;
        v30 = 0l;
        #pragma unroll
        while (while_method_1(v30)){
            assert("Tensor range check" && 0 <= v30 && v30 < 4l);
            float v32;
            v32 = v26[v30];
            float v33;
            v33 = 1.0f + v32;
            assert("Tensor range check" && 0 <= v30 && v30 < 4l);
            v27[v30] = v33;
            v30 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v34;
        v34 = reinterpret_cast<int4*>(v27 + 0l);
        int4* v35;
        v35 = reinterpret_cast<int4*>(v1 + v25);
        assert("Pointer alignment check" && (unsigned long long)(v34) % 4l == 0 && (unsigned long long)(v35) % 4l == 0);
        *v35 = *v34;
        v15 += 512l ;
    }
    __syncthreads();
    float v36;
    v36 = 0.0f;
    int v37;
    v37 = threadIdx.x;
    int v38;
    v38 = v37;
    while (while_method_0(v38)){
        bool v40;
        v40 = 0l <= v38;
        bool v41;
        v41 = v40 == false;
        if (v41){
            assert("The index needs to be zero or positive." && v40);
        } else {
        }
        int v42;
        v42 = v38 % 64l;
        int v43;
        v43 = v38 / 64l;
        bool v44;
        v44 = v43 < 1024l;
        bool v45;
        v45 = v44 == false;
        if (v45){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v44);
        } else {
        }
        assert("Tensor range check" && 0 <= v43 && v43 < 1024l);
        assert("Tensor range check" && 0 <= v42 && v42 < 64l);
        int v46;
        v46 = 4l * v42;
        int v47;
        v47 = 256l * v43;
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
        Tuple0 tmp0 = Tuple0{0l, v36};
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
        v36 = v53;
        v38 += 512l ;
    }
    auto v57 = cooperative_groups::coalesced_threads();
    Closure0 v58{};
    float v59;
    v59 = cooperative_groups::reduce(v57, v36, v58);
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
        v3[0l] = v70;
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
    v74 = v71 % 64l;
    int v75;
    v75 = v71 / 64l;
    bool v76;
    v76 = v75 < 8l;
    bool v77;
    v77 = v76 == false;
    if (v77){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v76);
    } else {
    }
    assert("Tensor range check" && 0 <= v75 && v75 < 8l);
    assert("Tensor range check" && 0 <= v74 && v74 < 64l);
    int v78;
    v78 = 4l * v74;
    int v79;
    v79 = 256l * v75;
    int v80;
    v80 = v79 + v78;
    assert("Tensor range check" && 0 <= v75 && v75 < 8l);
    assert("Tensor range check" && 0 <= v74 && v74 < 64l);
    int v81;
    v81 = 0l;
    while (while_method_2(v81)){
        assert("Tensor range check" && 0 <= v81 && v81 < 128l);
        int v83;
        v83 = 2048l * v81;
        int v84;
        v84 = v83 + v80;
        assert("Tensor range check" && 0 <= v81 && v81 < 128l);
        int v85[4l];
        int v86[4l];
        int v87;
        v87 = 0l;
        while (while_method_3(v87)){
            assert("Tensor range check" && 0 <= v87 && v87 < 1l);
            int v89;
            v89 = 4l * v87;
            assert("Tensor range check" && 0 <= v87 && v87 < 1l);
            int v90;
            v90 = 256l * v87;
            int v91;
            v91 = v90 + v84;
            int4* v92;
            v92 = reinterpret_cast<int4*>(v0 + v91);
            int4* v93;
            v93 = reinterpret_cast<int4*>(v85 + v89);
            assert("Pointer alignment check" && (unsigned long long)(v92) % 4l == 0 && (unsigned long long)(v93) % 4l == 0);
            *v93 = *v92;
            v87 += 1l ;
        }
        int v94;
        v94 = 0l;
        while (while_method_3(v94)){
            int v96;
            v96 = 0l;
            while (while_method_1(v96)){
                bool v98;
                v98 = 0l <= v96;
                bool v100;
                if (v98){
                    bool v99;
                    v99 = v96 < 4l;
                    v100 = v99;
                } else {
                    v100 = false;
                }
                bool v101;
                v101 = v100 == false;
                if (v101){
                    assert("The indices should be inside the range of the dimension." && v100);
                } else {
                }
                bool v102;
                v102 = 0l <= v74;
                bool v104;
                if (v102){
                    bool v103;
                    v103 = v74 < 64l;
                    v104 = v103;
                } else {
                    v104 = false;
                }
                bool v105;
                v105 = v104 == false;
                if (v105){
                    assert("The indices should be inside the range of the dimension." && v104);
                } else {
                }
                int v106;
                v106 = v74 * 4l;
                int v107;
                v107 = v96 + v106;
                bool v108;
                v108 = 0l <= v94;
                bool v110;
                if (v108){
                    bool v109;
                    v109 = v94 < 1l;
                    v110 = v109;
                } else {
                    v110 = false;
                }
                bool v111;
                v111 = v110 == false;
                if (v111){
                    assert("The indices should be inside the range of the dimension." && v110);
                } else {
                }
                int v112;
                v112 = v94 * 256l;
                int v113;
                v113 = v107 + v112;
                assert("Tensor range check" && 0 <= v94 && v94 < 1l);
                assert("Tensor range check" && 0 <= v96 && v96 < 4l);
                int v114;
                v114 = 4l * v94;
                int v115;
                v115 = v114 + v96;
                v86[v115] = v113;
                v96 += 1l ;
            }
            v94 += 1l ;
        }
        bool v116;
        v116 = 0l <= v75;
        bool v117;
        v117 = v116 && v76;
        bool v118;
        v118 = v117 == false;
        if (v118){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v117);
        } else {
        }
        bool v119;
        v119 = 0l <= v81;
        bool v121;
        if (v119){
            bool v120;
            v120 = v81 < 128l;
            v121 = v120;
        } else {
            v121 = false;
        }
        bool v122;
        v122 = v121 == false;
        if (v122){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v121);
        } else {
        }
        int v123;
        v123 = v81 * 8l;
        int v124;
        v124 = v123 + v75;
        int v125;
        v125 = 0l;
        while (while_method_3(v125)){
            assert("Tensor range check" && 0 <= v125 && v125 < 1l);
            int v127;
            v127 = 256l * v125;
            int v128;
            v128 = v127 + v84;
            assert("Tensor range check" && 0 <= v125 && v125 < 1l);
            int v129;
            v129 = 4l * v125;
            int4* v130;
            v130 = reinterpret_cast<int4*>(v85 + v129);
            int4* v131;
            v131 = reinterpret_cast<int4*>(v4 + v128);
            assert("Pointer alignment check" && (unsigned long long)(v130) % 4l == 0 && (unsigned long long)(v131) % 4l == 0);
            *v131 = *v130;
            v125 += 1l ;
        }
        v81 += 1l ;
    }
    __syncthreads();
    int v132;
    v132 = threadIdx.x;
    bool v133;
    v133 = 0l <= v132;
    bool v134;
    v134 = v133 == false;
    if (v134){
        assert("The index needs to be zero or positive." && v133);
    } else {
    }
    int v135;
    v135 = v132 % 64l;
    int v136;
    v136 = v132 / 64l;
    bool v137;
    v137 = v136 < 8l;
    bool v138;
    v138 = v137 == false;
    if (v138){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v137);
    } else {
    }
    assert("Tensor range check" && 0 <= v136 && v136 < 8l);
    assert("Tensor range check" && 0 <= v135 && v135 < 64l);
    int v139;
    v139 = 4l * v135;
    int v140;
    v140 = 256l * v136;
    int v141;
    v141 = v140 + v139;
    assert("Tensor range check" && 0 <= v136 && v136 < 8l);
    assert("Tensor range check" && 0 <= v135 && v135 < 64l);
    int v142;
    v142 = 0l;
    while (while_method_2(v142)){
        assert("Tensor range check" && 0 <= v142 && v142 < 128l);
        int v144;
        v144 = 2048l * v142;
        int v145;
        v145 = v144 + v141;
        assert("Tensor range check" && 0 <= v142 && v142 < 128l);
        float v146[4l];
        int v147[4l];
        int v148;
        v148 = 0l;
        while (while_method_3(v148)){
            assert("Tensor range check" && 0 <= v148 && v148 < 1l);
            int v150;
            v150 = 4l * v148;
            assert("Tensor range check" && 0 <= v148 && v148 < 1l);
            int v151;
            v151 = 256l * v148;
            int v152;
            v152 = v151 + v145;
            int4* v153;
            v153 = reinterpret_cast<int4*>(v1 + v152);
            int4* v154;
            v154 = reinterpret_cast<int4*>(v146 + v150);
            assert("Pointer alignment check" && (unsigned long long)(v153) % 4l == 0 && (unsigned long long)(v154) % 4l == 0);
            *v154 = *v153;
            v148 += 1l ;
        }
        int v155;
        v155 = 0l;
        while (while_method_3(v155)){
            int v157;
            v157 = 0l;
            while (while_method_1(v157)){
                bool v159;
                v159 = 0l <= v157;
                bool v161;
                if (v159){
                    bool v160;
                    v160 = v157 < 4l;
                    v161 = v160;
                } else {
                    v161 = false;
                }
                bool v162;
                v162 = v161 == false;
                if (v162){
                    assert("The indices should be inside the range of the dimension." && v161);
                } else {
                }
                bool v163;
                v163 = 0l <= v135;
                bool v165;
                if (v163){
                    bool v164;
                    v164 = v135 < 64l;
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
                v167 = v135 * 4l;
                int v168;
                v168 = v157 + v167;
                bool v169;
                v169 = 0l <= v155;
                bool v171;
                if (v169){
                    bool v170;
                    v170 = v155 < 1l;
                    v171 = v170;
                } else {
                    v171 = false;
                }
                bool v172;
                v172 = v171 == false;
                if (v172){
                    assert("The indices should be inside the range of the dimension." && v171);
                } else {
                }
                int v173;
                v173 = v155 * 256l;
                int v174;
                v174 = v168 + v173;
                assert("Tensor range check" && 0 <= v155 && v155 < 1l);
                assert("Tensor range check" && 0 <= v157 && v157 < 4l);
                int v175;
                v175 = 4l * v155;
                int v176;
                v176 = v175 + v157;
                v147[v176] = v174;
                v157 += 1l ;
            }
            v155 += 1l ;
        }
        bool v177;
        v177 = 0l <= v136;
        bool v178;
        v178 = v177 && v137;
        bool v179;
        v179 = v178 == false;
        if (v179){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v178);
        } else {
        }
        bool v180;
        v180 = 0l <= v142;
        bool v182;
        if (v180){
            bool v181;
            v181 = v142 < 128l;
            v182 = v181;
        } else {
            v182 = false;
        }
        bool v183;
        v183 = v182 == false;
        if (v183){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v182);
        } else {
        }
        int v184;
        v184 = v142 * 8l;
        int v185;
        v185 = v184 + v136;
        int v186[4l];
        int v187[4l];
        int v188;
        v188 = 0l;
        while (while_method_3(v188)){
            int v190;
            v190 = 0l;
            while (while_method_1(v190)){
                assert("Tensor range check" && 0 <= v188 && v188 < 1l);
                assert("Tensor range check" && 0 <= v190 && v190 < 4l);
                int v192;
                v192 = 4l * v188;
                int v193;
                v193 = v192 + v190;
                int v194;
                v194 = v147[v193];
                assert("Tensor range check" && 0 <= v188 && v188 < 1l);
                assert("Tensor range check" && 0 <= v190 && v190 < 4l);
                v186[v193] = v185;
                v187[v193] = v194;
                v190 += 1l ;
            }
            v188 += 1l ;
        }
        int v195;
        v195 = 0l;
        while (while_method_3(v195)){
            assert("Tensor range check" && 0 <= v195 && v195 < 1l);
            int v197;
            v197 = 256l * v195;
            int v198;
            v198 = v197 + v145;
            assert("Tensor range check" && 0 <= v195 && v195 < 1l);
            int v199;
            v199 = 4l * v195;
            int4* v200;
            v200 = reinterpret_cast<int4*>(v186 + v199);
            int4* v201;
            v201 = reinterpret_cast<int4*>(v11 + v198);
            assert("Pointer alignment check" && (unsigned long long)(v200) % 4l == 0 && (unsigned long long)(v201) % 4l == 0);
            *v201 = *v200;
            int4* v202;
            v202 = reinterpret_cast<int4*>(v187 + v199);
            int4* v203;
            v203 = reinterpret_cast<int4*>(v12 + v198);
            assert("Pointer alignment check" && (unsigned long long)(v202) % 4l == 0 && (unsigned long long)(v203) % 4l == 0);
            *v203 = *v202;
            v195 += 1l ;
        }
        v142 += 1l ;
    }
    __syncthreads();
    int v204;
    v204 = threadIdx.x;
    bool v205;
    v205 = 0l <= v204;
    bool v206;
    v206 = v205 == false;
    if (v206){
        assert("The index needs to be zero or positive." && v205);
    } else {
    }
    int v207;
    v207 = v204 % 64l;
    int v208;
    v208 = v204 / 64l;
    bool v209;
    v209 = v208 < 8l;
    bool v210;
    v210 = v209 == false;
    if (v210){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v209);
    } else {
    }
    assert("Tensor range check" && 0 <= v208 && v208 < 8l);
    assert("Tensor range check" && 0 <= v207 && v207 < 64l);
    int v211;
    v211 = 4l * v207;
    int v212;
    v212 = 256l * v208;
    int v213;
    v213 = v212 + v211;
    assert("Tensor range check" && 0 <= v208 && v208 < 8l);
    int v214;
    v214 = 0l;
    while (while_method_2(v214)){
        assert("Tensor range check" && 0 <= v214 && v214 < 128l);
        int v216;
        v216 = 2048l * v214;
        int v217;
        v217 = v216 + v213;
        float v218[4l];
        int v219[4l];
        int v220;
        v220 = 0l;
        while (while_method_3(v220)){
            assert("Tensor range check" && 0 <= v220 && v220 < 1l);
            int v222;
            v222 = 4l * v220;
            assert("Tensor range check" && 0 <= v220 && v220 < 1l);
            int v223;
            v223 = 256l * v220;
            int v224;
            v224 = v223 + v217;
            int4* v225;
            v225 = reinterpret_cast<int4*>(v1 + v224);
            int4* v226;
            v226 = reinterpret_cast<int4*>(v218 + v222);
            assert("Pointer alignment check" && (unsigned long long)(v225) % 4l == 0 && (unsigned long long)(v226) % 4l == 0);
            *v226 = *v225;
            v220 += 1l ;
        }
        int v227;
        v227 = 0l;
        while (while_method_3(v227)){
            int v229;
            v229 = 0l;
            while (while_method_1(v229)){
                bool v231;
                v231 = 0l <= v229;
                bool v233;
                if (v231){
                    bool v232;
                    v232 = v229 < 4l;
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
                bool v235;
                v235 = 0l <= v207;
                bool v237;
                if (v235){
                    bool v236;
                    v236 = v207 < 64l;
                    v237 = v236;
                } else {
                    v237 = false;
                }
                bool v238;
                v238 = v237 == false;
                if (v238){
                    assert("The indices should be inside the range of the dimension." && v237);
                } else {
                }
                int v239;
                v239 = v207 * 4l;
                int v240;
                v240 = v229 + v239;
                bool v241;
                v241 = 0l <= v227;
                bool v243;
                if (v241){
                    bool v242;
                    v242 = v227 < 1l;
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
                int v245;
                v245 = v227 * 256l;
                int v246;
                v246 = v240 + v245;
                assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                assert("Tensor range check" && 0 <= v229 && v229 < 4l);
                int v247;
                v247 = 4l * v227;
                int v248;
                v248 = v247 + v229;
                v219[v248] = v246;
                v229 += 1l ;
            }
            v227 += 1l ;
        }
        bool v249;
        v249 = 0l <= v208;
        bool v250;
        v250 = v249 && v209;
        bool v251;
        v251 = v250 == false;
        if (v251){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v250);
        } else {
        }
        bool v252;
        v252 = 0l <= v214;
        bool v254;
        if (v252){
            bool v253;
            v253 = v214 < 128l;
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
        int v256;
        v256 = v214 * 8l;
        int v257;
        v257 = v256 + v208;
        assert("Tensor range check" && 0 <= v214 && v214 < 128l);
        int v258;
        v258 = 8l * v214;
        int v259;
        v259 = v258 + v208;
        v13[v259] = v257;
        v214 += 1l ;
    }
    __syncthreads();
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
    int v263;
    v263 = v260 % 64l;
    int v264;
    v264 = v260 / 64l;
    bool v265;
    v265 = v264 < 8l;
    bool v266;
    v266 = v265 == false;
    if (v266){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v265);
    } else {
    }
    assert("Tensor range check" && 0 <= v264 && v264 < 8l);
    assert("Tensor range check" && 0 <= v263 && v263 < 64l);
    int v267;
    v267 = 4l * v263;
    int v268;
    v268 = 256l * v264;
    int v269;
    v269 = v268 + v267;
    assert("Tensor range check" && 0 <= v264 && v264 < 8l);
    assert("Tensor range check" && 0 <= v263 && v263 < 64l);
    int v270;
    v270 = 0l;
    while (while_method_2(v270)){
        assert("Tensor range check" && 0 <= v270 && v270 < 128l);
        int v272;
        v272 = 2048l * v270;
        int v273;
        v273 = v272 + v269;
        assert("Tensor range check" && 0 <= v270 && v270 < 128l);
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
            v279 = 256l * v276;
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
                bool v291;
                v291 = 0l <= v263;
                bool v293;
                if (v291){
                    bool v292;
                    v292 = v263 < 64l;
                    v293 = v292;
                } else {
                    v293 = false;
                }
                bool v294;
                v294 = v293 == false;
                if (v294){
                    assert("The indices should be inside the range of the dimension." && v293);
                } else {
                }
                int v295;
                v295 = v263 * 4l;
                int v296;
                v296 = v285 + v295;
                bool v297;
                v297 = 0l <= v283;
                bool v299;
                if (v297){
                    bool v298;
                    v298 = v283 < 1l;
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
                int v301;
                v301 = v283 * 256l;
                int v302;
                v302 = v296 + v301;
                assert("Tensor range check" && 0 <= v283 && v283 < 1l);
                assert("Tensor range check" && 0 <= v285 && v285 < 4l);
                int v303;
                v303 = 4l * v283;
                int v304;
                v304 = v303 + v285;
                v275[v304] = v302;
                v285 += 1l ;
            }
            v283 += 1l ;
        }
        bool v305;
        v305 = 0l <= v264;
        bool v306;
        v306 = v305 && v265;
        bool v307;
        v307 = v306 == false;
        if (v307){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v306);
        } else {
        }
        bool v308;
        v308 = 0l <= v270;
        bool v310;
        if (v308){
            bool v309;
            v309 = v270 < 128l;
            v310 = v309;
        } else {
            v310 = false;
        }
        bool v311;
        v311 = v310 == false;
        if (v311){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v310);
        } else {
        }
        int v312;
        v312 = v270 * 8l;
        int v313;
        v313 = v312 + v264;
        float v314;
        v314 = 0.0f;
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
                v321 = v274[v320];
                float v322;
                v322 = v314 + v321;
                v314 = v322;
                v317 += 1l ;
            }
            v315 += 1l ;
        }
        auto v323 = cooperative_groups::coalesced_threads();
        float v324;
        v324 = cooperative_groups::reduce(v323, v314, v58);
        int v325;
        v325 = threadIdx.x;
        int v326;
        v326 = v325 / 32l;
        __shared__ float v327[16l];
        bool v328;
        v328 = 0l <= v326;
        bool v329;
        v329 = v328 == false;
        if (v329){
            assert("The index needs to be zero or positive." && v328);
        } else {
        }
        int v330;
        v330 = v326 % 2l;
        int v331;
        v331 = v326 / 2l;
        bool v332;
        v332 = v331 < 8l;
        bool v333;
        v333 = v332 == false;
        if (v333){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v332);
        } else {
        }
        assert("Tensor range check" && 0 <= v331 && v331 < 8l);
        assert("Tensor range check" && 0 <= v330 && v330 < 2l);
        int v334;
        v334 = 2l * v331;
        int v335;
        v335 = v334 + v330;
        v327[v335] = v324;
        int v336;
        v336 = v331 + 1l;
        asm("bar.cta.sync %0, %1;" :: "r"(v336), "r"(64l));
        int v337;
        v337 = threadIdx.x;
        int v338;
        v338 = v337 % 32l;
        bool v339;
        v339 = v338 < 2l;
        float v342;
        if (v339){
            assert("Tensor range check" && 0 <= v331 && v331 < 8l);
            assert("Tensor range check" && 0 <= v338 && v338 < 2l);
            int v340;
            v340 = v334 + v338;
            float v341;
            v341 = v327[v340];
            v342 = v341;
        } else {
            v342 = 0.0f;
        }
        asm("bar.cta.sync %0, %1;" :: "r"(v336), "r"(64l));
        float v343;
        v343 = cooperative_groups::reduce(v323, v342, v58);
        float v344;
        v344 = v343 / 256.0f;
        float v345[4l];
        int v346;
        v346 = 0l;
        while (while_method_3(v346)){
            int v348;
            v348 = 0l;
            while (while_method_1(v348)){
                assert("Tensor range check" && 0 <= v346 && v346 < 1l);
                assert("Tensor range check" && 0 <= v348 && v348 < 4l);
                int v350;
                v350 = 4l * v346;
                int v351;
                v351 = v350 + v348;
                float v352;
                v352 = v274[v351];
                float v353;
                v353 = v352 - v344;
                float v354;
                v354 = exp(v353);
                assert("Tensor range check" && 0 <= v346 && v346 < 1l);
                assert("Tensor range check" && 0 <= v348 && v348 < 4l);
                v345[v351] = v354;
                v348 += 1l ;
            }
            v346 += 1l ;
        }
        float v355;
        v355 = 0.0f;
        int v356;
        v356 = 0l;
        while (while_method_3(v356)){
            int v358;
            v358 = 0l;
            while (while_method_1(v358)){
                assert("Tensor range check" && 0 <= v356 && v356 < 1l);
                assert("Tensor range check" && 0 <= v358 && v358 < 4l);
                int v360;
                v360 = 4l * v356;
                int v361;
                v361 = v360 + v358;
                float v362;
                v362 = v345[v361];
                float v363;
                v363 = v355 + v362;
                v355 = v363;
                v358 += 1l ;
            }
            v356 += 1l ;
        }
        auto v364 = cooperative_groups::coalesced_threads();
        float v365;
        v365 = cooperative_groups::reduce(v364, v355, v58);
        int v366;
        v366 = threadIdx.x;
        int v367;
        v367 = v366 / 32l;
        __shared__ float v368[16l];
        bool v369;
        v369 = 0l <= v367;
        bool v370;
        v370 = v369 == false;
        if (v370){
            assert("The index needs to be zero or positive." && v369);
        } else {
        }
        int v371;
        v371 = v367 % 2l;
        int v372;
        v372 = v367 / 2l;
        bool v373;
        v373 = v372 < 8l;
        bool v374;
        v374 = v373 == false;
        if (v374){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v373);
        } else {
        }
        assert("Tensor range check" && 0 <= v372 && v372 < 8l);
        assert("Tensor range check" && 0 <= v371 && v371 < 2l);
        int v375;
        v375 = 2l * v372;
        int v376;
        v376 = v375 + v371;
        v368[v376] = v365;
        int v377;
        v377 = v372 + 1l;
        asm("bar.cta.sync %0, %1;" :: "r"(v377), "r"(64l));
        int v378;
        v378 = threadIdx.x;
        int v379;
        v379 = v378 % 32l;
        bool v380;
        v380 = v379 < 2l;
        float v383;
        if (v380){
            assert("Tensor range check" && 0 <= v372 && v372 < 8l);
            assert("Tensor range check" && 0 <= v379 && v379 < 2l);
            int v381;
            v381 = v375 + v379;
            float v382;
            v382 = v368[v381];
            v383 = v382;
        } else {
            v383 = 0.0f;
        }
        asm("bar.cta.sync %0, %1;" :: "r"(v377), "r"(64l));
        float v384;
        v384 = cooperative_groups::reduce(v364, v383, v58);
        float v385[4l];
        int v386;
        v386 = 0l;
        while (while_method_3(v386)){
            int v388;
            v388 = 0l;
            while (while_method_1(v388)){
                assert("Tensor range check" && 0 <= v386 && v386 < 1l);
                assert("Tensor range check" && 0 <= v388 && v388 < 4l);
                int v390;
                v390 = 4l * v386;
                int v391;
                v391 = v390 + v388;
                float v392;
                v392 = v345[v391];
                float v393;
                v393 = v392 / v384;
                assert("Tensor range check" && 0 <= v386 && v386 < 1l);
                assert("Tensor range check" && 0 <= v388 && v388 < 4l);
                v385[v391] = v393;
                v388 += 1l ;
            }
            v386 += 1l ;
        }
        int v394;
        v394 = 0l;
        while (while_method_3(v394)){
            assert("Tensor range check" && 0 <= v394 && v394 < 1l);
            int v396;
            v396 = 256l * v394;
            int v397;
            v397 = v396 + v273;
            assert("Tensor range check" && 0 <= v394 && v394 < 1l);
            int v398;
            v398 = 4l * v394;
            int4* v399;
            v399 = reinterpret_cast<int4*>(v385 + v398);
            int4* v400;
            v400 = reinterpret_cast<int4*>(v5 + v397);
            assert("Pointer alignment check" && (unsigned long long)(v399) % 4l == 0 && (unsigned long long)(v400) % 4l == 0);
            *v400 = *v399;
            v394 += 1l ;
        }
        v270 += 1l ;
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
    int v404;
    v404 = v401 % 64l;
    int v405;
    v405 = v401 / 64l;
    bool v406;
    v406 = v405 < 8l;
    bool v407;
    v407 = v406 == false;
    if (v407){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v406);
    } else {
    }
    assert("Tensor range check" && 0 <= v405 && v405 < 8l);
    assert("Tensor range check" && 0 <= v404 && v404 < 64l);
    int v408;
    v408 = 4l * v404;
    int v409;
    v409 = 256l * v405;
    int v410;
    v410 = v409 + v408;
    assert("Tensor range check" && 0 <= v405 && v405 < 8l);
    assert("Tensor range check" && 0 <= v404 && v404 < 64l);
    int v411;
    v411 = 0l;
    while (while_method_2(v411)){
        assert("Tensor range check" && 0 <= v411 && v411 < 128l);
        int v413;
        v413 = 2048l * v411;
        int v414;
        v414 = v413 + v410;
        assert("Tensor range check" && 0 <= v411 && v411 < 128l);
        float v415[4l];
        int v416[4l];
        int v417;
        v417 = 0l;
        while (while_method_3(v417)){
            assert("Tensor range check" && 0 <= v417 && v417 < 1l);
            int v419;
            v419 = 4l * v417;
            assert("Tensor range check" && 0 <= v417 && v417 < 1l);
            int v420;
            v420 = 256l * v417;
            int v421;
            v421 = v420 + v414;
            int4* v422;
            v422 = reinterpret_cast<int4*>(v1 + v421);
            int4* v423;
            v423 = reinterpret_cast<int4*>(v415 + v419);
            assert("Pointer alignment check" && (unsigned long long)(v422) % 4l == 0 && (unsigned long long)(v423) % 4l == 0);
            *v423 = *v422;
            v417 += 1l ;
        }
        int v424;
        v424 = 0l;
        while (while_method_3(v424)){
            int v426;
            v426 = 0l;
            while (while_method_1(v426)){
                bool v428;
                v428 = 0l <= v426;
                bool v430;
                if (v428){
                    bool v429;
                    v429 = v426 < 4l;
                    v430 = v429;
                } else {
                    v430 = false;
                }
                bool v431;
                v431 = v430 == false;
                if (v431){
                    assert("The indices should be inside the range of the dimension." && v430);
                } else {
                }
                bool v432;
                v432 = 0l <= v404;
                bool v434;
                if (v432){
                    bool v433;
                    v433 = v404 < 64l;
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
                int v436;
                v436 = v404 * 4l;
                int v437;
                v437 = v426 + v436;
                bool v438;
                v438 = 0l <= v424;
                bool v440;
                if (v438){
                    bool v439;
                    v439 = v424 < 1l;
                    v440 = v439;
                } else {
                    v440 = false;
                }
                bool v441;
                v441 = v440 == false;
                if (v441){
                    assert("The indices should be inside the range of the dimension." && v440);
                } else {
                }
                int v442;
                v442 = v424 * 256l;
                int v443;
                v443 = v437 + v442;
                assert("Tensor range check" && 0 <= v424 && v424 < 1l);
                assert("Tensor range check" && 0 <= v426 && v426 < 4l);
                int v444;
                v444 = 4l * v424;
                int v445;
                v445 = v444 + v426;
                v416[v445] = v443;
                v426 += 1l ;
            }
            v424 += 1l ;
        }
        bool v446;
        v446 = 0l <= v405;
        bool v447;
        v447 = v446 && v406;
        bool v448;
        v448 = v447 == false;
        if (v448){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v447);
        } else {
        }
        bool v449;
        v449 = 0l <= v411;
        bool v451;
        if (v449){
            bool v450;
            v450 = v411 < 128l;
            v451 = v450;
        } else {
            v451 = false;
        }
        bool v452;
        v452 = v451 == false;
        if (v452){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v451);
        } else {
        }
        int v453;
        v453 = v411 * 8l;
        int v454;
        v454 = v453 + v405;
        float v455[4l];
        int v456;
        v456 = 0l;
        while (while_method_3(v456)){
            int v458;
            v458 = 0l;
            while (while_method_1(v458)){
                assert("Tensor range check" && 0 <= v456 && v456 < 1l);
                assert("Tensor range check" && 0 <= v458 && v458 < 4l);
                int v460;
                v460 = 4l * v456;
                int v461;
                v461 = v460 + v458;
                float v462;
                v462 = v415[v461];
                float v463;
                v463 = v462 * v462;
                assert("Tensor range check" && 0 <= v456 && v456 < 1l);
                assert("Tensor range check" && 0 <= v458 && v458 < 4l);
                v455[v461] = v463;
                v458 += 1l ;
            }
            v456 += 1l ;
        }
        float v464;
        v464 = 0.0f;
        int v465;
        v465 = 0l;
        while (while_method_3(v465)){
            int v467;
            v467 = 0l;
            while (while_method_1(v467)){
                assert("Tensor range check" && 0 <= v465 && v465 < 1l);
                assert("Tensor range check" && 0 <= v467 && v467 < 4l);
                int v469;
                v469 = 4l * v465;
                int v470;
                v470 = v469 + v467;
                float v471;
                v471 = v455[v470];
                float v472;
                v472 = v464 + v471;
                v464 = v472;
                v467 += 1l ;
            }
            v465 += 1l ;
        }
        auto v473 = cooperative_groups::coalesced_threads();
        float v474;
        v474 = cooperative_groups::reduce(v473, v464, v58);
        int v475;
        v475 = threadIdx.x;
        int v476;
        v476 = v475 / 32l;
        __shared__ float v477[16l];
        bool v478;
        v478 = 0l <= v476;
        bool v479;
        v479 = v478 == false;
        if (v479){
            assert("The index needs to be zero or positive." && v478);
        } else {
        }
        int v480;
        v480 = v476 % 2l;
        int v481;
        v481 = v476 / 2l;
        bool v482;
        v482 = v481 < 8l;
        bool v483;
        v483 = v482 == false;
        if (v483){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v482);
        } else {
        }
        assert("Tensor range check" && 0 <= v481 && v481 < 8l);
        assert("Tensor range check" && 0 <= v480 && v480 < 2l);
        int v484;
        v484 = 2l * v481;
        int v485;
        v485 = v484 + v480;
        v477[v485] = v474;
        int v486;
        v486 = v481 + 1l;
        asm("bar.cta.sync %0, %1;" :: "r"(v486), "r"(64l));
        int v487;
        v487 = threadIdx.x;
        int v488;
        v488 = v487 % 32l;
        bool v489;
        v489 = v488 < 2l;
        float v492;
        if (v489){
            assert("Tensor range check" && 0 <= v481 && v481 < 8l);
            assert("Tensor range check" && 0 <= v488 && v488 < 2l);
            int v490;
            v490 = v484 + v488;
            float v491;
            v491 = v477[v490];
            v492 = v491;
        } else {
            v492 = 0.0f;
        }
        asm("bar.cta.sync %0, %1;" :: "r"(v486), "r"(64l));
        float v493;
        v493 = cooperative_groups::reduce(v473, v492, v58);
        float v494[4l];
        int v495;
        v495 = 0l;
        while (while_method_3(v495)){
            int v497;
            v497 = 0l;
            while (while_method_1(v497)){
                assert("Tensor range check" && 0 <= v495 && v495 < 1l);
                assert("Tensor range check" && 0 <= v497 && v497 < 4l);
                int v499;
                v499 = 4l * v495;
                int v500;
                v500 = v499 + v497;
                float v501;
                v501 = v415[v500];
                float v502;
                v502 = v501 / v493;
                assert("Tensor range check" && 0 <= v495 && v495 < 1l);
                assert("Tensor range check" && 0 <= v497 && v497 < 4l);
                v494[v500] = v502;
                v497 += 1l ;
            }
            v495 += 1l ;
        }
        int v503;
        v503 = 0l;
        while (while_method_3(v503)){
            assert("Tensor range check" && 0 <= v503 && v503 < 1l);
            int v505;
            v505 = 256l * v503;
            int v506;
            v506 = v505 + v414;
            assert("Tensor range check" && 0 <= v503 && v503 < 1l);
            int v507;
            v507 = 4l * v503;
            int4* v508;
            v508 = reinterpret_cast<int4*>(v494 + v507);
            int4* v509;
            v509 = reinterpret_cast<int4*>(v8 + v506);
            assert("Pointer alignment check" && (unsigned long long)(v508) % 4l == 0 && (unsigned long long)(v509) % 4l == 0);
            *v509 = *v508;
            v503 += 1l ;
        }
        v411 += 1l ;
    }
    __syncthreads();
    int v510;
    v510 = threadIdx.x;
    bool v511;
    v511 = 0l <= v510;
    bool v512;
    v512 = v511 == false;
    if (v512){
        assert("The index needs to be zero or positive." && v511);
    } else {
    }
    int v513;
    v513 = v510 % 64l;
    int v514;
    v514 = v510 / 64l;
    bool v515;
    v515 = v514 < 8l;
    bool v516;
    v516 = v515 == false;
    if (v516){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v515);
    } else {
    }
    assert("Tensor range check" && 0 <= v514 && v514 < 8l);
    assert("Tensor range check" && 0 <= v513 && v513 < 64l);
    int v517;
    v517 = 4l * v513;
    int v518;
    v518 = 256l * v514;
    int v519;
    v519 = v518 + v517;
    assert("Tensor range check" && 0 <= v514 && v514 < 8l);
    int v520;
    v520 = 0l;
    while (while_method_2(v520)){
        assert("Tensor range check" && 0 <= v520 && v520 < 128l);
        int v522;
        v522 = 2048l * v520;
        int v523;
        v523 = v522 + v519;
        float v524[4l];
        int v525[4l];
        int v526;
        v526 = 0l;
        while (while_method_3(v526)){
            assert("Tensor range check" && 0 <= v526 && v526 < 1l);
            int v528;
            v528 = 4l * v526;
            assert("Tensor range check" && 0 <= v526 && v526 < 1l);
            int v529;
            v529 = 256l * v526;
            int v530;
            v530 = v529 + v523;
            int4* v531;
            v531 = reinterpret_cast<int4*>(v1 + v530);
            int4* v532;
            v532 = reinterpret_cast<int4*>(v524 + v528);
            assert("Pointer alignment check" && (unsigned long long)(v531) % 4l == 0 && (unsigned long long)(v532) % 4l == 0);
            *v532 = *v531;
            v526 += 1l ;
        }
        int v533;
        v533 = 0l;
        while (while_method_3(v533)){
            int v535;
            v535 = 0l;
            while (while_method_1(v535)){
                bool v537;
                v537 = 0l <= v535;
                bool v539;
                if (v537){
                    bool v538;
                    v538 = v535 < 4l;
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
                bool v541;
                v541 = 0l <= v513;
                bool v543;
                if (v541){
                    bool v542;
                    v542 = v513 < 64l;
                    v543 = v542;
                } else {
                    v543 = false;
                }
                bool v544;
                v544 = v543 == false;
                if (v544){
                    assert("The indices should be inside the range of the dimension." && v543);
                } else {
                }
                int v545;
                v545 = v513 * 4l;
                int v546;
                v546 = v535 + v545;
                bool v547;
                v547 = 0l <= v533;
                bool v549;
                if (v547){
                    bool v548;
                    v548 = v533 < 1l;
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
                v551 = v533 * 256l;
                int v552;
                v552 = v546 + v551;
                assert("Tensor range check" && 0 <= v533 && v533 < 1l);
                assert("Tensor range check" && 0 <= v535 && v535 < 4l);
                int v553;
                v553 = 4l * v533;
                int v554;
                v554 = v553 + v535;
                v525[v554] = v552;
                v535 += 1l ;
            }
            v533 += 1l ;
        }
        bool v555;
        v555 = 0l <= v514;
        bool v556;
        v556 = v555 && v515;
        bool v557;
        v557 = v556 == false;
        if (v557){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v556);
        } else {
        }
        bool v558;
        v558 = 0l <= v520;
        bool v560;
        if (v558){
            bool v559;
            v559 = v520 < 128l;
            v560 = v559;
        } else {
            v560 = false;
        }
        bool v561;
        v561 = v560 == false;
        if (v561){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v560);
        } else {
        }
        int v562;
        v562 = v520 * 8l;
        int v563;
        v563 = v562 + v514;
        float v564; int v565;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0l};
        v564 = tmp1.v0; v565 = tmp1.v1;
        int v566;
        v566 = 0l;
        while (while_method_3(v566)){
            int v568;
            v568 = 0l;
            while (while_method_1(v568)){
                assert("Tensor range check" && 0 <= v566 && v566 < 1l);
                assert("Tensor range check" && 0 <= v568 && v568 < 4l);
                int v570;
                v570 = 4l * v566;
                int v571;
                v571 = v570 + v568;
                float v572;
                v572 = v524[v571];
                int v573;
                v573 = v525[v571];
                bool v574;
                v574 = v564 > v572;
                float v575; int v576;
                if (v574){
                    v575 = v564; v576 = v565;
                } else {
                    v575 = v572; v576 = v573;
                }
                v564 = v575;
                v565 = v576;
                v568 += 1l ;
            }
            v566 += 1l ;
        }
        auto v577 = cooperative_groups::coalesced_threads();
        Closure1 v578{};
        float v579; int v580;
        Tuple1 tmp2 = cooperative_groups::reduce(v577, Tuple1{v564, v565}, v578);
        v579 = tmp2.v0; v580 = tmp2.v1;
        int v581;
        v581 = threadIdx.x;
        int v582;
        v582 = v581 / 32l;
        __shared__ float v583[16l];
        __shared__ int v584[16l];
        bool v585;
        v585 = 0l <= v582;
        bool v586;
        v586 = v585 == false;
        if (v586){
            assert("The index needs to be zero or positive." && v585);
        } else {
        }
        int v587;
        v587 = v582 % 2l;
        int v588;
        v588 = v582 / 2l;
        bool v589;
        v589 = v588 < 8l;
        bool v590;
        v590 = v589 == false;
        if (v590){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v589);
        } else {
        }
        assert("Tensor range check" && 0 <= v588 && v588 < 8l);
        assert("Tensor range check" && 0 <= v587 && v587 < 2l);
        int v591;
        v591 = 2l * v588;
        int v592;
        v592 = v591 + v587;
        v583[v592] = v579;
        v584[v592] = v580;
        int v593;
        v593 = v588 + 1l;
        asm("bar.cta.sync %0, %1;" :: "r"(v593), "r"(64l));
        int v594;
        v594 = threadIdx.x;
        int v595;
        v595 = v594 % 32l;
        bool v596;
        v596 = v595 < 2l;
        float v600; int v601;
        if (v596){
            assert("Tensor range check" && 0 <= v588 && v588 < 8l);
            assert("Tensor range check" && 0 <= v595 && v595 < 2l);
            int v597;
            v597 = v591 + v595;
            float v598;
            v598 = v583[v597];
            int v599;
            v599 = v584[v597];
            v600 = v598; v601 = v599;
        } else {
            v600 = -1.0f / 0.0f; v601 = 0l;
        }
        asm("bar.cta.sync %0, %1;" :: "r"(v593), "r"(64l));
        float v602; int v603;
        Tuple1 tmp3 = cooperative_groups::reduce(v577, Tuple1{v600, v601}, v578);
        v602 = tmp3.v0; v603 = tmp3.v1;
        assert("Tensor range check" && 0 <= v520 && v520 < 128l);
        int v604;
        v604 = 8l * v520;
        int v605;
        v605 = v604 + v514;
        v9[v605] = v603;
        v520 += 1l ;
    }
    __syncthreads();
    int v606;
    v606 = threadIdx.x;
    bool v607;
    v607 = 0l <= v606;
    bool v608;
    v608 = v607 == false;
    if (v608){
        assert("The index needs to be zero or positive." && v607);
    } else {
    }
    int v609;
    v609 = v606 % 64l;
    int v610;
    v610 = v606 / 64l;
    bool v611;
    v611 = v610 < 8l;
    bool v612;
    v612 = v611 == false;
    if (v612){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v611);
    } else {
    }
    assert("Tensor range check" && 0 <= v610 && v610 < 8l);
    assert("Tensor range check" && 0 <= v609 && v609 < 64l);
    int v613;
    v613 = 4l * v609;
    int v614;
    v614 = 256l * v610;
    int v615;
    v615 = v614 + v613;
    assert("Tensor range check" && 0 <= v610 && v610 < 8l);
    assert("Tensor range check" && 0 <= v609 && v609 < 64l);
    int v616;
    v616 = 0l;
    while (while_method_2(v616)){
        assert("Tensor range check" && 0 <= v616 && v616 < 128l);
        int v618;
        v618 = 2048l * v616;
        int v619;
        v619 = v618 + v615;
        assert("Tensor range check" && 0 <= v616 && v616 < 128l);
        float v620[4l];
        int v621[4l];
        int v622;
        v622 = 0l;
        while (while_method_3(v622)){
            assert("Tensor range check" && 0 <= v622 && v622 < 1l);
            int v624;
            v624 = 4l * v622;
            assert("Tensor range check" && 0 <= v622 && v622 < 1l);
            int v625;
            v625 = 256l * v622;
            int v626;
            v626 = v625 + v619;
            int4* v627;
            v627 = reinterpret_cast<int4*>(v1 + v626);
            int4* v628;
            v628 = reinterpret_cast<int4*>(v620 + v624);
            assert("Pointer alignment check" && (unsigned long long)(v627) % 4l == 0 && (unsigned long long)(v628) % 4l == 0);
            *v628 = *v627;
            v622 += 1l ;
        }
        int v629;
        v629 = 0l;
        while (while_method_3(v629)){
            int v631;
            v631 = 0l;
            while (while_method_1(v631)){
                bool v633;
                v633 = 0l <= v631;
                bool v635;
                if (v633){
                    bool v634;
                    v634 = v631 < 4l;
                    v635 = v634;
                } else {
                    v635 = false;
                }
                bool v636;
                v636 = v635 == false;
                if (v636){
                    assert("The indices should be inside the range of the dimension." && v635);
                } else {
                }
                bool v637;
                v637 = 0l <= v609;
                bool v639;
                if (v637){
                    bool v638;
                    v638 = v609 < 64l;
                    v639 = v638;
                } else {
                    v639 = false;
                }
                bool v640;
                v640 = v639 == false;
                if (v640){
                    assert("The indices should be inside the range of the dimension." && v639);
                } else {
                }
                int v641;
                v641 = v609 * 4l;
                int v642;
                v642 = v631 + v641;
                bool v643;
                v643 = 0l <= v629;
                bool v645;
                if (v643){
                    bool v644;
                    v644 = v629 < 1l;
                    v645 = v644;
                } else {
                    v645 = false;
                }
                bool v646;
                v646 = v645 == false;
                if (v646){
                    assert("The indices should be inside the range of the dimension." && v645);
                } else {
                }
                int v647;
                v647 = v629 * 256l;
                int v648;
                v648 = v642 + v647;
                assert("Tensor range check" && 0 <= v629 && v629 < 1l);
                assert("Tensor range check" && 0 <= v631 && v631 < 4l);
                int v649;
                v649 = 4l * v629;
                int v650;
                v650 = v649 + v631;
                v621[v650] = v648;
                v631 += 1l ;
            }
            v629 += 1l ;
        }
        bool v651;
        v651 = 0l <= v610;
        bool v652;
        v652 = v651 && v611;
        bool v653;
        v653 = v652 == false;
        if (v653){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v652);
        } else {
        }
        bool v654;
        v654 = 0l <= v616;
        bool v656;
        if (v654){
            bool v655;
            v655 = v616 < 128l;
            v656 = v655;
        } else {
            v656 = false;
        }
        bool v657;
        v657 = v656 == false;
        if (v657){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v656);
        } else {
        }
        int v658;
        v658 = v616 * 8l;
        int v659;
        v659 = v658 + v610;
        float v660;
        v660 = 0.0f;
        int v661;
        v661 = 0l;
        while (while_method_3(v661)){
            int v663;
            v663 = 0l;
            while (while_method_1(v663)){
                assert("Tensor range check" && 0 <= v661 && v661 < 1l);
                assert("Tensor range check" && 0 <= v663 && v663 < 4l);
                int v665;
                v665 = 4l * v661;
                int v666;
                v666 = v665 + v663;
                float v667;
                v667 = v620[v666];
                float v668;
                v668 = v660 + v667;
                v660 = v668;
                v663 += 1l ;
            }
            v661 += 1l ;
        }
        auto v669 = cooperative_groups::coalesced_threads();
        float v670;
        v670 = cooperative_groups::reduce(v669, v660, v58);
        int v671;
        v671 = threadIdx.x;
        int v672;
        v672 = v671 / 32l;
        __shared__ float v673[16l];
        bool v674;
        v674 = 0l <= v672;
        bool v675;
        v675 = v674 == false;
        if (v675){
            assert("The index needs to be zero or positive." && v674);
        } else {
        }
        int v676;
        v676 = v672 % 2l;
        int v677;
        v677 = v672 / 2l;
        bool v678;
        v678 = v677 < 8l;
        bool v679;
        v679 = v678 == false;
        if (v679){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v678);
        } else {
        }
        assert("Tensor range check" && 0 <= v677 && v677 < 8l);
        assert("Tensor range check" && 0 <= v676 && v676 < 2l);
        int v680;
        v680 = 2l * v677;
        int v681;
        v681 = v680 + v676;
        v673[v681] = v670;
        int v682;
        v682 = v677 + 1l;
        asm("bar.cta.sync %0, %1;" :: "r"(v682), "r"(64l));
        int v683;
        v683 = threadIdx.x;
        int v684;
        v684 = v683 % 32l;
        bool v685;
        v685 = v684 < 2l;
        float v688;
        if (v685){
            assert("Tensor range check" && 0 <= v677 && v677 < 8l);
            assert("Tensor range check" && 0 <= v684 && v684 < 2l);
            int v686;
            v686 = v680 + v684;
            float v687;
            v687 = v673[v686];
            v688 = v687;
        } else {
            v688 = 0.0f;
        }
        asm("bar.cta.sync %0, %1;" :: "r"(v682), "r"(64l));
        float v689;
        v689 = cooperative_groups::reduce(v669, v688, v58);
        float v690;
        v690 = v689 / 256.0f;
        float v691[4l];
        int v692;
        v692 = 0l;
        while (while_method_3(v692)){
            int v694;
            v694 = 0l;
            while (while_method_1(v694)){
                assert("Tensor range check" && 0 <= v692 && v692 < 1l);
                assert("Tensor range check" && 0 <= v694 && v694 < 4l);
                int v696;
                v696 = 4l * v692;
                int v697;
                v697 = v696 + v694;
                float v698;
                v698 = v620[v697];
                float v699;
                v699 = v698 - v690;
                float v700;
                v700 = exp(v699);
                assert("Tensor range check" && 0 <= v692 && v692 < 1l);
                assert("Tensor range check" && 0 <= v694 && v694 < 4l);
                v691[v697] = v700;
                v694 += 1l ;
            }
            v692 += 1l ;
        }
        float v701;
        v701 = 0.0f;
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
                float v708;
                v708 = v691[v707];
                float v709;
                v709 = v701 + v708;
                v701 = v709;
                v704 += 1l ;
            }
            v702 += 1l ;
        }
        auto v710 = cooperative_groups::coalesced_threads();
        float v711;
        v711 = cooperative_groups::reduce(v710, v701, v58);
        int v712;
        v712 = threadIdx.x;
        int v713;
        v713 = v712 / 32l;
        __shared__ float v714[16l];
        bool v715;
        v715 = 0l <= v713;
        bool v716;
        v716 = v715 == false;
        if (v716){
            assert("The index needs to be zero or positive." && v715);
        } else {
        }
        int v717;
        v717 = v713 % 2l;
        int v718;
        v718 = v713 / 2l;
        bool v719;
        v719 = v718 < 8l;
        bool v720;
        v720 = v719 == false;
        if (v720){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v719);
        } else {
        }
        assert("Tensor range check" && 0 <= v718 && v718 < 8l);
        assert("Tensor range check" && 0 <= v717 && v717 < 2l);
        int v721;
        v721 = 2l * v718;
        int v722;
        v722 = v721 + v717;
        v714[v722] = v711;
        int v723;
        v723 = v718 + 1l;
        asm("bar.cta.sync %0, %1;" :: "r"(v723), "r"(64l));
        int v724;
        v724 = threadIdx.x;
        int v725;
        v725 = v724 % 32l;
        bool v726;
        v726 = v725 < 2l;
        float v729;
        if (v726){
            assert("Tensor range check" && 0 <= v718 && v718 < 8l);
            assert("Tensor range check" && 0 <= v725 && v725 < 2l);
            int v727;
            v727 = v721 + v725;
            float v728;
            v728 = v714[v727];
            v729 = v728;
        } else {
            v729 = 0.0f;
        }
        asm("bar.cta.sync %0, %1;" :: "r"(v723), "r"(64l));
        float v730;
        v730 = cooperative_groups::reduce(v710, v729, v58);
        float v731[4l];
        int v732;
        v732 = 0l;
        while (while_method_3(v732)){
            int v734;
            v734 = 0l;
            while (while_method_1(v734)){
                assert("Tensor range check" && 0 <= v732 && v732 < 1l);
                assert("Tensor range check" && 0 <= v734 && v734 < 4l);
                int v736;
                v736 = 4l * v732;
                int v737;
                v737 = v736 + v734;
                float v738;
                v738 = v691[v737];
                float v739;
                v739 = v738 / v730;
                assert("Tensor range check" && 0 <= v732 && v732 < 1l);
                assert("Tensor range check" && 0 <= v734 && v734 < 4l);
                v731[v737] = v739;
                v734 += 1l ;
            }
            v732 += 1l ;
        }
        float v740[4l];
        float v741;
        v741 = 0.0f;
        int v742;
        v742 = 0l;
        while (while_method_3(v742)){
            assert("Tensor range check" && 0 <= v742 && v742 < 1l);
            int v744;
            v744 = 4l * v742;
            assert("Tensor range check" && 0 <= v742 && v742 < 1l);
            int v745; float v746;
            Tuple0 tmp4 = Tuple0{0l, 0.0f};
            v745 = tmp4.v0; v746 = tmp4.v1;
            while (while_method_1(v745)){
                assert("Tensor range check" && 0 <= v745 && v745 < 4l);
                int v748;
                v748 = v745 + v744;
                float v749;
                v749 = v731[v748];
                float v750;
                v750 = v746 + v749;
                v746 = v750;
                v745 += 1l ;
            }
            auto v751 = cooperative_groups::coalesced_threads();
            int v752;
            v752 = threadIdx.x;
            int v753;
            v753 = v752 / 32l;
            __shared__ float v754[16l];
            Closure2 v755{};
            float v756;
            v756 = cooperative_groups::inclusive_scan(v751, v746, v755);
            float v757;
            v757 = v751.shfl_up(v756,1);
            bool v758;
            v758 = v751.thread_rank() == 0;
            float v759;
            if (v758){
                v759 = 0.0f;
            } else {
                v759 = v757;
            }
            float v760;
            v760 = v751.shfl(v756,v751.num_threads()-1);
            bool v761;
            v761 = 0l <= v753;
            bool v762;
            v762 = v761 == false;
            if (v762){
                assert("The index needs to be zero or positive." && v761);
            } else {
            }
            int v763;
            v763 = v753 % 2l;
            int v764;
            v764 = v753 / 2l;
            bool v765;
            v765 = v764 < 8l;
            bool v766;
            v766 = v765 == false;
            if (v766){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v765);
            } else {
            }
            assert("Tensor range check" && 0 <= v764 && v764 < 8l);
            assert("Tensor range check" && 0 <= v763 && v763 < 2l);
            int v767;
            v767 = 2l * v764;
            int v768;
            v768 = v767 + v763;
            v754[v768] = v760;
            int v769;
            v769 = v764 + 1l;
            asm("bar.cta.sync %0, %1;" :: "r"(v769), "r"(64l));
            int v770;
            v770 = threadIdx.x;
            int v771;
            v771 = v770 % 32l;
            bool v772;
            v772 = v771 < 2l;
            float v775;
            if (v772){
                assert("Tensor range check" && 0 <= v764 && v764 < 8l);
                assert("Tensor range check" && 0 <= v771 && v771 < 2l);
                int v773;
                v773 = v767 + v771;
                float v774;
                v774 = v754[v773];
                v775 = v774;
            } else {
                v775 = 0.0f;
            }
            asm("bar.cta.sync %0, %1;" :: "r"(v769), "r"(64l));
            float v776;
            v776 = cooperative_groups::inclusive_scan(v751, v775, v755);
            float v777;
            v777 = v751.shfl_up(v776,1);
            bool v778;
            v778 = v751.thread_rank() == 0;
            float v779;
            if (v778){
                v779 = 0.0f;
            } else {
                v779 = v777;
            }
            float v780;
            v780 = v751.shfl(v776,v751.num_threads()-1);
            float v781;
            v781 = v751.shfl(v779,v763);
            float v782;
            v782 = v781 + v759;
            float v783;
            v783 = v741 + v782;
            int v784; float v785;
            Tuple0 tmp5 = Tuple0{0l, v783};
            v784 = tmp5.v0; v785 = tmp5.v1;
            while (while_method_1(v784)){
                assert("Tensor range check" && 0 <= v784 && v784 < 4l);
                int v787;
                v787 = v784 + v744;
                float v788;
                v788 = v731[v787];
                float v789;
                v789 = v785 + v788;
                assert("Tensor range check" && 0 <= v784 && v784 < 4l);
                v740[v787] = v789;
                v785 = v789;
                v784 += 1l ;
            }
            float v790;
            v790 = v741 + v780;
            v741 = v790;
            v742 += 1l ;
        }
        int v791;
        v791 = 0l;
        while (while_method_3(v791)){
            assert("Tensor range check" && 0 <= v791 && v791 < 1l);
            int v793;
            v793 = 256l * v791;
            int v794;
            v794 = v793 + v619;
            assert("Tensor range check" && 0 <= v791 && v791 < 1l);
            int v795;
            v795 = 4l * v791;
            int4* v796;
            v796 = reinterpret_cast<int4*>(v731 + v795);
            int4* v797;
            v797 = reinterpret_cast<int4*>(v6 + v794);
            assert("Pointer alignment check" && (unsigned long long)(v796) % 4l == 0 && (unsigned long long)(v797) % 4l == 0);
            *v797 = *v796;
            int4* v798;
            v798 = reinterpret_cast<int4*>(v740 + v795);
            int4* v799;
            v799 = reinterpret_cast<int4*>(v7 + v794);
            assert("Pointer alignment check" && (unsigned long long)(v798) % 4l == 0 && (unsigned long long)(v799) % 4l == 0);
            *v799 = *v798;
            v791 += 1l ;
        }
        v616 += 1l ;
    }
    __syncthreads();
    int v800;
    v800 = threadIdx.x;
    bool v801;
    v801 = 0l <= v800;
    bool v802;
    v802 = v801 == false;
    if (v802){
        assert("The index needs to be zero or positive." && v801);
    } else {
    }
    int v803;
    v803 = v800 % 64l;
    int v804;
    v804 = v800 / 64l;
    bool v805;
    v805 = v804 < 8l;
    bool v806;
    v806 = v805 == false;
    if (v806){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v805);
    } else {
    }
    assert("Tensor range check" && 0 <= v804 && v804 < 8l);
    assert("Tensor range check" && 0 <= v803 && v803 < 64l);
    int v807;
    v807 = 4l * v803;
    int v808;
    v808 = 256l * v804;
    int v809;
    v809 = v808 + v807;
    assert("Tensor range check" && 0 <= v804 && v804 < 8l);
    int v810;
    v810 = 0l;
    while (while_method_2(v810)){
        assert("Tensor range check" && 0 <= v810 && v810 < 128l);
        int v812;
        v812 = 2048l * v810;
        int v813;
        v813 = v812 + v809;
        float v814[4l];
        int v815[4l];
        int v816;
        v816 = 0l;
        while (while_method_3(v816)){
            assert("Tensor range check" && 0 <= v816 && v816 < 1l);
            int v818;
            v818 = 4l * v816;
            assert("Tensor range check" && 0 <= v816 && v816 < 1l);
            int v819;
            v819 = 256l * v816;
            int v820;
            v820 = v819 + v813;
            int4* v821;
            v821 = reinterpret_cast<int4*>(v1 + v820);
            int4* v822;
            v822 = reinterpret_cast<int4*>(v814 + v818);
            assert("Pointer alignment check" && (unsigned long long)(v821) % 4l == 0 && (unsigned long long)(v822) % 4l == 0);
            *v822 = *v821;
            v816 += 1l ;
        }
        int v823;
        v823 = 0l;
        while (while_method_3(v823)){
            int v825;
            v825 = 0l;
            while (while_method_1(v825)){
                bool v827;
                v827 = 0l <= v825;
                bool v829;
                if (v827){
                    bool v828;
                    v828 = v825 < 4l;
                    v829 = v828;
                } else {
                    v829 = false;
                }
                bool v830;
                v830 = v829 == false;
                if (v830){
                    assert("The indices should be inside the range of the dimension." && v829);
                } else {
                }
                bool v831;
                v831 = 0l <= v803;
                bool v833;
                if (v831){
                    bool v832;
                    v832 = v803 < 64l;
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
                int v835;
                v835 = v803 * 4l;
                int v836;
                v836 = v825 + v835;
                bool v837;
                v837 = 0l <= v823;
                bool v839;
                if (v837){
                    bool v838;
                    v838 = v823 < 1l;
                    v839 = v838;
                } else {
                    v839 = false;
                }
                bool v840;
                v840 = v839 == false;
                if (v840){
                    assert("The indices should be inside the range of the dimension." && v839);
                } else {
                }
                int v841;
                v841 = v823 * 256l;
                int v842;
                v842 = v836 + v841;
                assert("Tensor range check" && 0 <= v823 && v823 < 1l);
                assert("Tensor range check" && 0 <= v825 && v825 < 4l);
                int v843;
                v843 = 4l * v823;
                int v844;
                v844 = v843 + v825;
                v815[v844] = v842;
                v825 += 1l ;
            }
            v823 += 1l ;
        }
        bool v845;
        v845 = 0l <= v804;
        bool v846;
        v846 = v845 && v805;
        bool v847;
        v847 = v846 == false;
        if (v847){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v846);
        } else {
        }
        bool v848;
        v848 = 0l <= v810;
        bool v850;
        if (v848){
            bool v849;
            v849 = v810 < 128l;
            v850 = v849;
        } else {
            v850 = false;
        }
        bool v851;
        v851 = v850 == false;
        if (v851){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v850);
        } else {
        }
        int v852;
        v852 = v810 * 8l;
        int v853;
        v853 = v852 + v804;
        float v854;
        v854 = 0.0f;
        int v855;
        v855 = 0l;
        while (while_method_3(v855)){
            int v857;
            v857 = 0l;
            while (while_method_1(v857)){
                assert("Tensor range check" && 0 <= v855 && v855 < 1l);
                assert("Tensor range check" && 0 <= v857 && v857 < 4l);
                int v859;
                v859 = 4l * v855;
                int v860;
                v860 = v859 + v857;
                float v861;
                v861 = v814[v860];
                float v862;
                v862 = v854 + v861;
                v854 = v862;
                v857 += 1l ;
            }
            v855 += 1l ;
        }
        auto v863 = cooperative_groups::coalesced_threads();
        float v864;
        v864 = cooperative_groups::reduce(v863, v854, v58);
        int v865;
        v865 = threadIdx.x;
        int v866;
        v866 = v865 / 32l;
        __shared__ float v867[16l];
        bool v868;
        v868 = 0l <= v866;
        bool v869;
        v869 = v868 == false;
        if (v869){
            assert("The index needs to be zero or positive." && v868);
        } else {
        }
        int v870;
        v870 = v866 % 2l;
        int v871;
        v871 = v866 / 2l;
        bool v872;
        v872 = v871 < 8l;
        bool v873;
        v873 = v872 == false;
        if (v873){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v872);
        } else {
        }
        assert("Tensor range check" && 0 <= v871 && v871 < 8l);
        assert("Tensor range check" && 0 <= v870 && v870 < 2l);
        int v874;
        v874 = 2l * v871;
        int v875;
        v875 = v874 + v870;
        v867[v875] = v864;
        int v876;
        v876 = v871 + 1l;
        asm("bar.cta.sync %0, %1;" :: "r"(v876), "r"(64l));
        int v877;
        v877 = threadIdx.x;
        int v878;
        v878 = v877 % 32l;
        bool v879;
        v879 = v878 < 2l;
        float v882;
        if (v879){
            assert("Tensor range check" && 0 <= v871 && v871 < 8l);
            assert("Tensor range check" && 0 <= v878 && v878 < 2l);
            int v880;
            v880 = v874 + v878;
            float v881;
            v881 = v867[v880];
            v882 = v881;
        } else {
            v882 = 0.0f;
        }
        asm("bar.cta.sync %0, %1;" :: "r"(v876), "r"(64l));
        float v883;
        v883 = cooperative_groups::reduce(v863, v882, v58);
        float v884;
        v884 = v883 / 256.0f;
        float v885[4l];
        int v886;
        v886 = 0l;
        while (while_method_3(v886)){
            int v888;
            v888 = 0l;
            while (while_method_1(v888)){
                assert("Tensor range check" && 0 <= v886 && v886 < 1l);
                assert("Tensor range check" && 0 <= v888 && v888 < 4l);
                int v890;
                v890 = 4l * v886;
                int v891;
                v891 = v890 + v888;
                float v892;
                v892 = v814[v891];
                float v893;
                v893 = v892 - v884;
                float v894;
                v894 = exp(v893);
                assert("Tensor range check" && 0 <= v886 && v886 < 1l);
                assert("Tensor range check" && 0 <= v888 && v888 < 4l);
                v885[v891] = v894;
                v888 += 1l ;
            }
            v886 += 1l ;
        }
        float v895;
        v895 = 0.0f;
        int v896;
        v896 = 0l;
        while (while_method_3(v896)){
            int v898;
            v898 = 0l;
            while (while_method_1(v898)){
                assert("Tensor range check" && 0 <= v896 && v896 < 1l);
                assert("Tensor range check" && 0 <= v898 && v898 < 4l);
                int v900;
                v900 = 4l * v896;
                int v901;
                v901 = v900 + v898;
                float v902;
                v902 = v885[v901];
                float v903;
                v903 = v895 + v902;
                v895 = v903;
                v898 += 1l ;
            }
            v896 += 1l ;
        }
        auto v904 = cooperative_groups::coalesced_threads();
        float v905;
        v905 = cooperative_groups::reduce(v904, v895, v58);
        int v906;
        v906 = threadIdx.x;
        int v907;
        v907 = v906 / 32l;
        __shared__ float v908[16l];
        bool v909;
        v909 = 0l <= v907;
        bool v910;
        v910 = v909 == false;
        if (v910){
            assert("The index needs to be zero or positive." && v909);
        } else {
        }
        int v911;
        v911 = v907 % 2l;
        int v912;
        v912 = v907 / 2l;
        bool v913;
        v913 = v912 < 8l;
        bool v914;
        v914 = v913 == false;
        if (v914){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v913);
        } else {
        }
        assert("Tensor range check" && 0 <= v912 && v912 < 8l);
        assert("Tensor range check" && 0 <= v911 && v911 < 2l);
        int v915;
        v915 = 2l * v912;
        int v916;
        v916 = v915 + v911;
        v908[v916] = v905;
        int v917;
        v917 = v912 + 1l;
        asm("bar.cta.sync %0, %1;" :: "r"(v917), "r"(64l));
        int v918;
        v918 = threadIdx.x;
        int v919;
        v919 = v918 % 32l;
        bool v920;
        v920 = v919 < 2l;
        float v923;
        if (v920){
            assert("Tensor range check" && 0 <= v912 && v912 < 8l);
            assert("Tensor range check" && 0 <= v919 && v919 < 2l);
            int v921;
            v921 = v915 + v919;
            float v922;
            v922 = v908[v921];
            v923 = v922;
        } else {
            v923 = 0.0f;
        }
        asm("bar.cta.sync %0, %1;" :: "r"(v917), "r"(64l));
        float v924;
        v924 = cooperative_groups::reduce(v904, v923, v58);
        float v925[4l];
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
                v932 = v885[v931];
                float v933;
                v933 = v932 / v924;
                assert("Tensor range check" && 0 <= v926 && v926 < 1l);
                assert("Tensor range check" && 0 <= v928 && v928 < 4l);
                v925[v931] = v933;
                v928 += 1l ;
            }
            v926 += 1l ;
        }
        float v934[4l];
        float v935;
        v935 = 0.0f;
        int v936;
        v936 = 0l;
        while (while_method_3(v936)){
            assert("Tensor range check" && 0 <= v936 && v936 < 1l);
            int v938;
            v938 = 4l * v936;
            assert("Tensor range check" && 0 <= v936 && v936 < 1l);
            int v939; float v940;
            Tuple0 tmp6 = Tuple0{0l, 0.0f};
            v939 = tmp6.v0; v940 = tmp6.v1;
            while (while_method_1(v939)){
                assert("Tensor range check" && 0 <= v939 && v939 < 4l);
                int v942;
                v942 = v939 + v938;
                float v943;
                v943 = v925[v942];
                float v944;
                v944 = v940 + v943;
                v940 = v944;
                v939 += 1l ;
            }
            auto v945 = cooperative_groups::coalesced_threads();
            int v946;
            v946 = threadIdx.x;
            int v947;
            v947 = v946 / 32l;
            __shared__ float v948[16l];
            Closure2 v949{};
            float v950;
            v950 = cooperative_groups::inclusive_scan(v945, v940, v949);
            float v951;
            v951 = v945.shfl_up(v950,1);
            bool v952;
            v952 = v945.thread_rank() == 0;
            float v953;
            if (v952){
                v953 = 0.0f;
            } else {
                v953 = v951;
            }
            float v954;
            v954 = v945.shfl(v950,v945.num_threads()-1);
            bool v955;
            v955 = 0l <= v947;
            bool v956;
            v956 = v955 == false;
            if (v956){
                assert("The index needs to be zero or positive." && v955);
            } else {
            }
            int v957;
            v957 = v947 % 2l;
            int v958;
            v958 = v947 / 2l;
            bool v959;
            v959 = v958 < 8l;
            bool v960;
            v960 = v959 == false;
            if (v960){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v959);
            } else {
            }
            assert("Tensor range check" && 0 <= v958 && v958 < 8l);
            assert("Tensor range check" && 0 <= v957 && v957 < 2l);
            int v961;
            v961 = 2l * v958;
            int v962;
            v962 = v961 + v957;
            v948[v962] = v954;
            int v963;
            v963 = v958 + 1l;
            asm("bar.cta.sync %0, %1;" :: "r"(v963), "r"(64l));
            int v964;
            v964 = threadIdx.x;
            int v965;
            v965 = v964 % 32l;
            bool v966;
            v966 = v965 < 2l;
            float v969;
            if (v966){
                assert("Tensor range check" && 0 <= v958 && v958 < 8l);
                assert("Tensor range check" && 0 <= v965 && v965 < 2l);
                int v967;
                v967 = v961 + v965;
                float v968;
                v968 = v948[v967];
                v969 = v968;
            } else {
                v969 = 0.0f;
            }
            asm("bar.cta.sync %0, %1;" :: "r"(v963), "r"(64l));
            float v970;
            v970 = cooperative_groups::inclusive_scan(v945, v969, v949);
            float v971;
            v971 = v945.shfl_up(v970,1);
            bool v972;
            v972 = v945.thread_rank() == 0;
            float v973;
            if (v972){
                v973 = 0.0f;
            } else {
                v973 = v971;
            }
            float v974;
            v974 = v945.shfl(v970,v945.num_threads()-1);
            float v975;
            v975 = v945.shfl(v973,v957);
            float v976;
            v976 = v975 + v953;
            float v977;
            v977 = v935 + v976;
            int v978; float v979;
            Tuple0 tmp7 = Tuple0{0l, v977};
            v978 = tmp7.v0; v979 = tmp7.v1;
            while (while_method_1(v978)){
                assert("Tensor range check" && 0 <= v978 && v978 < 4l);
                int v981;
                v981 = v978 + v938;
                float v982;
                v982 = v925[v981];
                float v983;
                v983 = v979 + v982;
                assert("Tensor range check" && 0 <= v978 && v978 < 4l);
                v934[v981] = v983;
                v979 = v983;
                v978 += 1l ;
            }
            float v984;
            v984 = v935 + v974;
            v935 = v984;
            v936 += 1l ;
        }
        assert("Tensor range check" && 0 <= v853 && v853 < 1024l);
        float v985;
        v985 = v2[v853];
        float v986[4l];
        int v987;
        v987 = 0l;
        while (while_method_3(v987)){
            int v989;
            v989 = 0l;
            while (while_method_1(v989)){
                assert("Tensor range check" && 0 <= v987 && v987 < 1l);
                assert("Tensor range check" && 0 <= v989 && v989 < 4l);
                int v991;
                v991 = 4l * v987;
                int v992;
                v992 = v991 + v989;
                float v993;
                v993 = v934[v992];
                float v994;
                v994 = v993 - v985;
                assert("Tensor range check" && 0 <= v987 && v987 < 1l);
                assert("Tensor range check" && 0 <= v989 && v989 < 4l);
                v986[v992] = v994;
                v989 += 1l ;
            }
            v987 += 1l ;
        }
        float v995; int v996;
        Tuple1 tmp8 = Tuple1{-1.0f / 0.0f, 0l};
        v995 = tmp8.v0; v996 = tmp8.v1;
        int v997;
        v997 = 0l;
        while (while_method_3(v997)){
            int v999;
            v999 = 0l;
            while (while_method_1(v999)){
                assert("Tensor range check" && 0 <= v997 && v997 < 1l);
                assert("Tensor range check" && 0 <= v999 && v999 < 4l);
                int v1001;
                v1001 = 4l * v997;
                int v1002;
                v1002 = v1001 + v999;
                float v1003;
                v1003 = v986[v1002];
                int v1004;
                v1004 = v815[v1002];
                bool v1005;
                v1005 = v995 >= 0.0f;
                bool v1007;
                if (v1005){
                    bool v1006;
                    v1006 = v1003 >= 0.0f;
                    v1007 = v1006;
                } else {
                    v1007 = false;
                }
                float v1016; int v1017;
                if (v1007){
                    bool v1008;
                    v1008 = v995 <= v1003;
                    if (v1008){
                        v1016 = v995; v1017 = v996;
                    } else {
                        v1016 = v1003; v1017 = v1004;
                    }
                } else {
                    if (v1005){
                        v1016 = v995; v1017 = v996;
                    } else {
                        bool v1011;
                        v1011 = v1003 >= 0.0f;
                        if (v1011){
                            v1016 = v1003; v1017 = v1004;
                        } else {
                            v1016 = v995; v1017 = v996;
                        }
                    }
                }
                v995 = v1016;
                v996 = v1017;
                v999 += 1l ;
            }
            v997 += 1l ;
        }
        auto v1018 = cooperative_groups::coalesced_threads();
        Closure3 v1019{};
        float v1020; int v1021;
        Tuple1 tmp9 = cooperative_groups::reduce(v1018, Tuple1{v995, v996}, v1019);
        v1020 = tmp9.v0; v1021 = tmp9.v1;
        int v1022;
        v1022 = threadIdx.x;
        int v1023;
        v1023 = v1022 / 32l;
        __shared__ float v1024[16l];
        __shared__ int v1025[16l];
        bool v1026;
        v1026 = 0l <= v1023;
        bool v1027;
        v1027 = v1026 == false;
        if (v1027){
            assert("The index needs to be zero or positive." && v1026);
        } else {
        }
        int v1028;
        v1028 = v1023 % 2l;
        int v1029;
        v1029 = v1023 / 2l;
        bool v1030;
        v1030 = v1029 < 8l;
        bool v1031;
        v1031 = v1030 == false;
        if (v1031){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1030);
        } else {
        }
        assert("Tensor range check" && 0 <= v1029 && v1029 < 8l);
        assert("Tensor range check" && 0 <= v1028 && v1028 < 2l);
        int v1032;
        v1032 = 2l * v1029;
        int v1033;
        v1033 = v1032 + v1028;
        v1024[v1033] = v1020;
        v1025[v1033] = v1021;
        int v1034;
        v1034 = v1029 + 1l;
        asm("bar.cta.sync %0, %1;" :: "r"(v1034), "r"(64l));
        int v1035;
        v1035 = threadIdx.x;
        int v1036;
        v1036 = v1035 % 32l;
        bool v1037;
        v1037 = v1036 < 2l;
        float v1041; int v1042;
        if (v1037){
            assert("Tensor range check" && 0 <= v1029 && v1029 < 8l);
            assert("Tensor range check" && 0 <= v1036 && v1036 < 2l);
            int v1038;
            v1038 = v1032 + v1036;
            float v1039;
            v1039 = v1024[v1038];
            int v1040;
            v1040 = v1025[v1038];
            v1041 = v1039; v1042 = v1040;
        } else {
            v1041 = -1.0f / 0.0f; v1042 = 0l;
        }
        asm("bar.cta.sync %0, %1;" :: "r"(v1034), "r"(64l));
        float v1043; int v1044;
        Tuple1 tmp10 = cooperative_groups::reduce(v1018, Tuple1{v1041, v1042}, v1019);
        v1043 = tmp10.v0; v1044 = tmp10.v1;
        assert("Tensor range check" && 0 <= v810 && v810 < 128l);
        int v1045;
        v1045 = 8l * v810;
        int v1046;
        v1046 = v1045 + v804;
        v10[v1046] = v1044;
        v810 += 1l ;
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
def method3(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def method4() -> None:
    return 
def main():
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
    v6 = cp.random.uniform(size=1024,dtype=cp.float32) # type: ignore
    v7 = cp.empty(1,dtype=cp.float32)
    v8 = cp.empty(262144,dtype=cp.int32)
    v9 = cp.empty(262144,dtype=cp.float32)
    v10 = cp.empty(262144,dtype=cp.float32)
    v11 = cp.empty(262144,dtype=cp.float32)
    v12 = cp.empty(262144,dtype=cp.float32)
    v13 = cp.empty(1024,dtype=cp.int32)
    v14 = cp.empty(1024,dtype=cp.int32)
    v15 = cp.empty(262144,dtype=cp.int32)
    v16 = cp.empty(262144,dtype=cp.int32)
    v17 = cp.empty(1024,dtype=cp.int32)
    v18 = 0
    v19 = raw_module.get_function(f"entry{v18}")
    del v18
    v19.max_dynamic_shared_size_bytes = 0 
    v19((1,),(512,),(v0, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17),shared_mem=0)
    del v0, v5, v6, v7, v8, v9, v10, v11, v12, v13, v15, v16, v17, v19
    v20 = 0
    v21 = '['
    method0(v21)
    del v21
    v22 = 0
    while method1(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            method2(v26)
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
            method2(v29)
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v14[v22].item()
        method3(v31)
        del v31
        v22 += 1 
    del v14, v20, v22
    v32 = ']'
    method0(v32)
    del v32
    method4()
    print()
    return 

if __name__ == '__main__': print(main())
