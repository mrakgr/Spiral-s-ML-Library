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
__device__ Tuple1 method_0(float v0, int v1, float v2, int v3);
struct Closure0 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v1 + v0;
        return v2;
    }
};
struct Closure1 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v1 + v0;
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
        v2 = v1 + v0;
        return v2;
    }
};
struct Closure3 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        return v0;
    }
};
struct Tuple1 {
    float v0;
    int v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        printf("inside: %i, %i\n", v3, v1);
        return method_0(v2, v3, v0, v1);
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ Tuple1 method_0(float v0, int v1, float v2, int v3){
    // wtf;
    // printf("%i, %i\n", v1, v3);
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
extern "C" __global__ void entry0(float * v0, float * v1, int * v2) {
    unsigned long long v3;
    v3 = clock64();
    curandStatePhilox4_32_10_t v4;
    curand_init(v3,0ull,0ull,&v4);
    int v5;
    v5 = threadIdx.x;
    bool v6;
    v6 = 0l <= v5;
    bool v7;
    v7 = v6 == false;
    if (v7){
        assert("The index needs to be zero or positive." && v6);
    } else {
    }
    int v9;
    v9 = v5 % 32l;
    int v10;
    v10 = v5 / 32l;
    bool v11;
    v11 = v10 < 1l;
    bool v12;
    v12 = v11 == false;
    if (v12){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v11);
    } else {
    }
    assert("Tensor range check" && 0 <= v10 && v10 < 1l);
    assert("Tensor range check" && 0 <= v9 && v9 < 32l);
    int v14;
    v14 = 4l * v9;
    int v15;
    v15 = 128l * v10;
    int v16;
    v16 = v15 + v14;
    assert("Tensor range check" && 0 <= v10 && v10 < 1l);
    assert("Tensor range check" && 0 <= v9 && v9 < 32l);
    assert("Tensor range check" && 0 <= v10 && v10 < 1l);
    int v17;
    v17 = 0l;
    while (while_method_0(v17)){
        assert("Tensor range check" && 0 <= v17 && v17 < 16l);
        int v19;
        v19 = 128l * v17;
        int v20;
        v20 = v19 + v16;
        float v21[4l];
        int v22[4l];
        int v23;
        v23 = 0l;
        while (while_method_1(v23)){
            assert("Tensor range check" && 0 <= v23 && v23 < 1l);
            int v25;
            v25 = 4l * v23;
            assert("Tensor range check" && 0 <= v23 && v23 < 1l);
            int v26;
            v26 = 128l * v23;
            int v27;
            v27 = v26 + v20;
            int4* v28;
            v28 = reinterpret_cast<int4*>(v0 + v27);
            int4* v29;
            v29 = reinterpret_cast<int4*>(v21 + v25);
            assert("Pointer alignment check" && (unsigned long long)(v28) % 4l == 0 && (unsigned long long)(v29) % 4l == 0);
            *v29 = *v28;
            v23 += 1l ;
        }
        int v30;
        v30 = 0l;
        while (while_method_1(v30)){
            int v32;
            v32 = 0l;
            while (while_method_2(v32)){
                bool v34;
                v34 = 0l <= v32;
                bool v36;
                if (v34){
                    bool v35;
                    v35 = v32 < 4l;
                    v36 = v35;
                } else {
                    v36 = false;
                }
                bool v37;
                v37 = v36 == false;
                if (v37){
                    assert("The indices should be inside the range of the dimension." && v36);
                } else {
                }
                bool v39;
                v39 = 0l <= v9;
                bool v41;
                if (v39){
                    bool v40;
                    v40 = v9 < 32l;
                    v41 = v40;
                } else {
                    v41 = false;
                }
                bool v42;
                v42 = v41 == false;
                if (v42){
                    assert("The indices should be inside the range of the dimension." && v41);
                } else {
                }
                int v44;
                v44 = v9 * 4l;
                int v45;
                v45 = v32 + v44;
                bool v46;
                v46 = 0l <= v30;
                bool v48;
                if (v46){
                    bool v47;
                    v47 = v30 < 1l;
                    v48 = v47;
                } else {
                    v48 = false;
                }
                bool v49;
                v49 = v48 == false;
                if (v49){
                    assert("The indices should be inside the range of the dimension." && v48);
                } else {
                }
                int v51;
                v51 = v30 * 128l;
                int v52;
                v52 = v45 + v51;
                assert("Tensor range check" && 0 <= v30 && v30 < 1l);
                assert("Tensor range check" && 0 <= v32 && v32 < 4l);
                int v53;
                v53 = 4l * v30;
                int v54;
                v54 = v53 + v32;
                v22[v54] = v52;
                v32 += 1l ;
            }
            v30 += 1l ;
        }
        bool v55;
        v55 = 0l <= v10;
        bool v56;
        v56 = v55 && v11;
        bool v57;
        v57 = v56 == false;
        if (v57){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v56);
        } else {
        }
        bool v59;
        v59 = 0l <= v17;
        bool v61;
        if (v59){
            bool v60;
            v60 = v17 < 16l;
            v61 = v60;
        } else {
            v61 = false;
        }
        bool v62;
        v62 = v61 == false;
        if (v62){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v61);
        } else {
        }
        int v64;
        v64 = v17 + v10;
        bool v65[4l];
        int v66;
        v66 = 0l;
        while (while_method_1(v66)){
            int v68;
            v68 = 0l;
            while (while_method_2(v68)){
                assert("Tensor range check" && 0 <= v66 && v66 < 1l);
                assert("Tensor range check" && 0 <= v68 && v68 < 4l);
                int v70;
                v70 = 4l * v66;
                int v71;
                v71 = v70 + v68;
                float v72;
                v72 = v21[v71];
                int v73;
                v73 = v22[v71];
                bool v74;
                v74 = v73 < 3l;
                assert("Tensor range check" && 0 <= v66 && v66 < 1l);
                assert("Tensor range check" && 0 <= v68 && v68 < 4l);
                v65[v71] = v74;
                v68 += 1l ;
            }
            v66 += 1l ;
        }
        int v75[4l];
        int v76;
        v76 = 0l;
        while (while_method_1(v76)){
            int v78;
            v78 = 0l;
            while (while_method_2(v78)){
                assert("Tensor range check" && 0 <= v76 && v76 < 1l);
                assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                int v80;
                v80 = 4l * v76;
                int v81;
                v81 = v80 + v78;
                bool v82;
                v82 = v65[v81];
                int v83;
                if (v82){
                    v83 = 1l;
                } else {
                    v83 = 0l;
                }
                assert("Tensor range check" && 0 <= v76 && v76 < 1l);
                assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                v75[v81] = v83;
                v78 += 1l ;
            }
            v76 += 1l ;
        }
        int v84;
        v84 = 0l;
        int v85;
        v85 = 0l;
        while (while_method_1(v85)){
            int v87;
            v87 = 0l;
            while (while_method_2(v87)){
                assert("Tensor range check" && 0 <= v85 && v85 < 1l);
                assert("Tensor range check" && 0 <= v87 && v87 < 4l);
                int v89;
                v89 = 4l * v85;
                int v90;
                v90 = v89 + v87;
                int v91;
                v91 = v75[v90];
                int v92;
                v92 = v84 + v91;
                v84 = v92;
                v87 += 1l ;
            }
            v85 += 1l ;
        }
        auto v93 = cooperative_groups::coalesced_threads();
        int v94;
        v94 = threadIdx.x;
        int v95;
        v95 = v94 / 32l;
        auto v96 = cooperative_groups::labeled_partition(v93,v95);
        Closure0 v97{};
        int v98;
        v98 = cooperative_groups::reduce(v96, v84, v97);
        float v99[4l];
        int v100;
        v100 = 0l;
        while (while_method_1(v100)){
            int v102;
            v102 = 0l;
            while (while_method_2(v102)){
                assert("Tensor range check" && 0 <= v100 && v100 < 1l);
                assert("Tensor range check" && 0 <= v102 && v102 < 4l);
                int v104;
                v104 = 4l * v100;
                int v105;
                v105 = v104 + v102;
                float v106;
                v106 = v21[v105];
                bool v107;
                v107 = v65[v105];
                float v108;
                if (v107){
                    v108 = v106;
                } else {
                    v108 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v100 && v100 < 1l);
                assert("Tensor range check" && 0 <= v102 && v102 < 4l);
                v99[v105] = v108;
                v102 += 1l ;
            }
            v100 += 1l ;
        }
        float v109;
        v109 = 0.0f;
        int v110;
        v110 = 0l;
        while (while_method_1(v110)){
            int v112;
            v112 = 0l;
            while (while_method_2(v112)){
                assert("Tensor range check" && 0 <= v110 && v110 < 1l);
                assert("Tensor range check" && 0 <= v112 && v112 < 4l);
                int v114;
                v114 = 4l * v110;
                int v115;
                v115 = v114 + v112;
                float v116;
                v116 = v99[v115];
                float v117;
                v117 = v109 + v116;
                v109 = v117;
                v112 += 1l ;
            }
            v110 += 1l ;
        }
        auto v118 = cooperative_groups::coalesced_threads();
        int v119;
        v119 = threadIdx.x;
        int v120;
        v120 = v119 / 32l;
        auto v121 = cooperative_groups::labeled_partition(v118,v120);
        Closure1 v122{};
        float v123;
        v123 = cooperative_groups::reduce(v121, v109, v122);
        float v124;
        v124 = (float)v98;
        float v125;
        v125 = v123 / v124;
        float v126[4l];
        int v127;
        v127 = 0l;
        while (while_method_1(v127)){
            int v129;
            v129 = 0l;
            while (while_method_2(v129)){
                assert("Tensor range check" && 0 <= v127 && v127 < 1l);
                assert("Tensor range check" && 0 <= v129 && v129 < 4l);
                int v131;
                v131 = 4l * v127;
                int v132;
                v132 = v131 + v129;
                float v133;
                v133 = v21[v132];
                bool v134;
                v134 = v65[v132];
                float v135;
                if (v134){
                    v135 = v133;
                } else {
                    v135 = -1.0f / 0.0f;
                }
                float v136;
                v136 = v135 - v125;
                float v137;
                v137 = exp(v136);
                assert("Tensor range check" && 0 <= v127 && v127 < 1l);
                assert("Tensor range check" && 0 <= v129 && v129 < 4l);
                v126[v132] = v137;
                v129 += 1l ;
            }
            v127 += 1l ;
        }
        float v138;
        v138 = 0.0f;
        int v139;
        v139 = 0l;
        while (while_method_1(v139)){
            int v141;
            v141 = 0l;
            while (while_method_2(v141)){
                assert("Tensor range check" && 0 <= v139 && v139 < 1l);
                assert("Tensor range check" && 0 <= v141 && v141 < 4l);
                int v143;
                v143 = 4l * v139;
                int v144;
                v144 = v143 + v141;
                float v145;
                v145 = v126[v144];
                float v146;
                v146 = v138 + v145;
                v138 = v146;
                v141 += 1l ;
            }
            v139 += 1l ;
        }
        auto v147 = cooperative_groups::coalesced_threads();
        int v148;
        v148 = threadIdx.x;
        int v149;
        v149 = v148 / 32l;
        auto v150 = cooperative_groups::labeled_partition(v147,v149);
        float v151;
        v151 = cooperative_groups::reduce(v150, v138, v122);
        float v152[4l];
        int v153;
        v153 = 0l;
        while (while_method_1(v153)){
            int v155;
            v155 = 0l;
            while (while_method_2(v155)){
                assert("Tensor range check" && 0 <= v153 && v153 < 1l);
                assert("Tensor range check" && 0 <= v155 && v155 < 4l);
                int v157;
                v157 = 4l * v153;
                int v158;
                v158 = v157 + v155;
                float v159;
                v159 = v126[v158];
                bool v160;
                v160 = v151 == 0.0f;
                bool v161;
                v161 = v160 != true;
                float v163;
                if (v161){
                    float v162;
                    v162 = v159 / v151;
                    v163 = v162;
                } else {
                    v163 = 0.0078125f;
                }
                assert("Tensor range check" && 0 <= v153 && v153 < 1l);
                assert("Tensor range check" && 0 <= v155 && v155 < 4l);
                v152[v158] = v163;
                v155 += 1l ;
            }
            v153 += 1l ;
        }
        float v164[4l];
        float v165;
        v165 = 0.0f;
        int v166;
        v166 = 0l;
        while (while_method_1(v166)){
            assert("Tensor range check" && 0 <= v166 && v166 < 1l);
            int v168;
            v168 = 4l * v166;
            assert("Tensor range check" && 0 <= v166 && v166 < 1l);
            int v169; float v170;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v169 = tmp0.v0; v170 = tmp0.v1;
            while (while_method_2(v169)){
                assert("Tensor range check" && 0 <= v169 && v169 < 4l);
                int v172;
                v172 = v169 + v168;
                float v173;
                v173 = v152[v172];
                float v174;
                v174 = v170 + v173;
                v170 = v174;
                v169 += 1l ;
            }
            auto v175 = cooperative_groups::coalesced_threads();
            int v176;
            v176 = threadIdx.x;
            int v177;
            v177 = v176 / 32l;
            auto v178 = cooperative_groups::labeled_partition(v175,v177);
            Closure2 v179{};
            float v180;
            v180 = cooperative_groups::inclusive_scan(v178, v170, v179);
            float v181;
            v181 = v178.shfl_up(v180,1);
            bool v182;
            v182 = v178.thread_rank() == 0;
            float v183;
            if (v182){
                v183 = 0.0f;
            } else {
                v183 = v181;
            }
            float v184;
            v184 = v178.shfl(v180,v178.num_threads()-1);
            float v185;
            v185 = v165 + v183;
            int v186; float v187;
            Tuple0 tmp1 = Tuple0{0l, v185};
            v186 = tmp1.v0; v187 = tmp1.v1;
            while (while_method_2(v186)){
                assert("Tensor range check" && 0 <= v186 && v186 < 4l);
                int v189;
                v189 = v186 + v168;
                float v190;
                v190 = v152[v189];
                float v191;
                v191 = v187 + v190;
                assert("Tensor range check" && 0 <= v186 && v186 < 4l);
                v164[v189] = v191;
                v187 = v191;
                v186 += 1l ;
            }
            float v192;
            v192 = v165 + v184;
            v165 = v192;
            v166 += 1l ;
        }
        float v193;
        v193 = curand_uniform(&v4);
        float v194[4l];
        int v195;
        v195 = 0l;
        while (while_method_1(v195)){
            int v197;
            v197 = 0l;
            while (while_method_2(v197)){
                assert("Tensor range check" && 0 <= v195 && v195 < 1l);
                assert("Tensor range check" && 0 <= v197 && v197 < 4l);
                int v199;
                v199 = 4l * v195;
                int v200;
                v200 = v199 + v197;
                int v201;
                v201 = v22[v200];
                assert("Tensor range check" && 0 <= v195 && v195 < 1l);
                assert("Tensor range check" && 0 <= v197 && v197 < 4l);
                v194[v200] = v193;
                v197 += 1l ;
            }
            v195 += 1l ;
        }
        float v202;
        v202 = 0.0f;
        int v203;
        v203 = 0l;
        while (while_method_1(v203)){
            int v205;
            v205 = 0l;
            while (while_method_2(v205)){
                assert("Tensor range check" && 0 <= v203 && v203 < 1l);
                assert("Tensor range check" && 0 <= v205 && v205 < 4l);
                int v207;
                v207 = 4l * v203;
                int v208;
                v208 = v207 + v205;
                float v209;
                v209 = v194[v208];
                v202 = v209;
                v205 += 1l ;
            }
            v203 += 1l ;
        }
        auto v210 = cooperative_groups::coalesced_threads();
        int v211;
        v211 = threadIdx.x;
        int v212;
        v212 = v211 / 32l;
        auto v213 = cooperative_groups::labeled_partition(v210,v212);
        Closure3 v214{};
        float v215;
        v215 = cooperative_groups::reduce(v213, v202, v214);
        float v216[4l];
        int v217;
        v217 = 0l;
        while (while_method_1(v217)){
            int v219;
            v219 = 0l;
            while (while_method_2(v219)){
                assert("Tensor range check" && 0 <= v217 && v217 < 1l);
                assert("Tensor range check" && 0 <= v219 && v219 < 4l);
                int v221;
                v221 = 4l * v217;
                int v222;
                v222 = v221 + v219;
                float v223;
                v223 = v164[v222];
                float v224;
                v224 = v223 - v215;
                assert("Tensor range check" && 0 <= v217 && v217 < 1l);
                assert("Tensor range check" && 0 <= v219 && v219 < 4l);
                v216[v222] = v224;
                v219 += 1l ;
            }
            v217 += 1l ;
        }
        float v225; int v226;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, 0l};
        v225 = tmp2.v0; v226 = tmp2.v1;
        int v227;
        v227 = 0l;
        while (while_method_1(v227)){
            int v229;
            v229 = 0l;
            while (while_method_2(v229)){
                assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                assert("Tensor range check" && 0 <= v229 && v229 < 4l);
                int v231;
                v231 = 4l * v227;
                int v232;
                v232 = v231 + v229;
                float v233;
                v233 = v216[v232];
                int v234;
                v234 = v22[v232];
                float v235; int v236;
                printf("outside: %i, %i\n", v226, v234);
                Tuple1 tmp3 = method_0(v225, v226, v233, v234);
                v235 = tmp3.v0; v236 = tmp3.v1;
                v225 = v235;
                v226 = v236;
                v229 += 1l ;
            }
            v227 += 1l ;
        }
        auto v237 = cooperative_groups::coalesced_threads();
        int v238;
        v238 = threadIdx.x;
        int v239;
        v239 = v238 / 32l;
        auto v240 = cooperative_groups::labeled_partition(v237,v239);
        Closure4 v241{};
        float v242; int v243;
        Tuple1 tmp4 = cooperative_groups::reduce(v240, Tuple1{v225, v226}, v241);
        v242 = tmp4.v0; v243 = tmp4.v1;
        assert("Tensor range check" && 0 <= v17 && v17 < 16l);
        int v244;
        v244 = 0l;
        while (while_method_1(v244)){
            assert("Tensor range check" && 0 <= v244 && v244 < 1l);
            int v246;
            v246 = 128l * v244;
            int v247;
            v247 = v246 + v20;
            assert("Tensor range check" && 0 <= v244 && v244 < 1l);
            int v248;
            v248 = 4l * v244;
            int4* v249;
            v249 = reinterpret_cast<int4*>(v152 + v248);
            int4* v250;
            v250 = reinterpret_cast<int4*>(v1 + v247);
            assert("Pointer alignment check" && (unsigned long long)(v249) % 4l == 0 && (unsigned long long)(v250) % 4l == 0);
            *v250 = *v249;
            v244 += 1l ;
        }
        assert("Tensor range check" && 0 <= v17 && v17 < 16l);
        v2[v64] = v243;
        v17 += 1l ;
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
    v1 = v0 < 16
    del v0
    return v1
def method2(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method3(v0 : i32) -> bool:
    v1 = v0 < 128
    del v0
    return v1
def method4(v0 : f32) -> None:
    print("{:.6f}".format(v0), end="")
    del v0
    return 
def method5() -> None:
    return 
def method6(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def main():
    v0 = cp.arange(0,2048,1,dtype=cp.int32) # type: ignore
    v1 = v0.size
    del v0
    v2 = 2048 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,2048,dtype=cp.float32) # type: ignore
    v6 = cp.random.uniform(size=16,dtype=cp.float32) # type: ignore
    del v6
    v7 = cp.empty(2048,dtype=cp.float32)
    v8 = cp.empty(16,dtype=cp.int32)
    v9 = 0
    v10 = raw_module.get_function(f"entry{v9}")
    del v9
    v10.max_dynamic_shared_size_bytes = 0 
    v10((1,),(32,),(v5, v7, v8),shared_mem=0)
    del v5, v10
    v11 = 0
    v12 = '['
    method0(v12)
    del v12
    v13 = 0
    while method1(v13):
        v15 = v11
        v16 = v15 >= 1024
        del v15
        if v16:
            v17 = " ..."
            method2(v17)
            del v17
            break
        else:
            pass
        del v16
        v18 = v13 == 0
        v19 = v18 != True
        del v18
        if v19:
            v20 = "; "
            method2(v20)
        else:
            pass
        del v19
        v21 = '['
        method0(v21)
        del v21
        v22 = 0
        while method3(v22):
            v24 = v11
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
            v30 = v11 + 1
            v11 = v30
            del v30
            v31 = v13 * 128
            v32 = v31 + v22
            del v31
            v33 = v7[v32].item()
            del v32
            method4(v33)
            del v33
            v22 += 1 
        del v22
        v34 = ']'
        method0(v34)
        del v34
        v13 += 1 
    del v7, v11, v13
    v35 = ']'
    method0(v35)
    del v35
    method5()
    print()
    v36 = 0
    v37 = '['
    method0(v37)
    del v37
    v38 = 0
    while method1(v38):
        v40 = v36
        v41 = v40 >= 1024
        del v40
        if v41:
            v42 = " ..."
            method2(v42)
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
            method2(v45)
        else:
            pass
        del v44
        v46 = v36 + 1
        v36 = v46
        del v46
        v47 = v8[v38].item()
        method6(v47)
        del v47
        v38 += 1 
    del v8, v36, v38
    v48 = ']'
    method0(v48)
    del v48
    method5()
    print()
    return 

if __name__ == '__main__': print(main())
