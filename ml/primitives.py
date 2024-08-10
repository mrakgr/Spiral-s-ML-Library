kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
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
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure1 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
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
struct Closure3 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        bool v2;
        v2 = v0 >= v1;
        if (v2){
            return v0;
        } else {
            return v1;
        }
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
        return method_0(v0, v1, v2, v3);
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ Tuple1 method_0(float v0, int v1, float v2, int v3){
    bool v4;
    v4 = v1 < v3;
    float v5; int v6; float v7; int v8;
    if (v4){
        v5 = v0; v6 = v1; v7 = v2; v8 = v3;
    } else {
        v5 = v2; v6 = v3; v7 = v0; v8 = v1;
    }
    bool v9;
    v9 = v5 >= 0.0f;
    bool v11;
    if (v9){
        bool v10;
        v10 = v7 >= 0.0f;
        v11 = v10;
    } else {
        v11 = false;
    }
    if (v11){
        return Tuple1{v5, v6};
    } else {
        if (v9){
            return Tuple1{v5, v6};
        } else {
            bool v12;
            v12 = v7 >= 0.0f;
            if (v12){
                return Tuple1{v7, v8};
            } else {
                return Tuple1{v5, v6};
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
    v15 = 4096l * v10;
    int v16;
    v16 = v15 + v14;
    assert("Tensor range check" && 0 <= v10 && v10 < 1l);
    assert("Tensor range check" && 0 <= v9 && v9 < 32l);
    assert("Tensor range check" && 0 <= v10 && v10 < 1l);
    int v17;
    v17 = 0l;
    while (while_method_0(v17)){
        assert("Tensor range check" && 0 <= v17 && v17 < 64l);
        int v19;
        v19 = 4096l * v17;
        int v20;
        v20 = v19 + v16;
        float v21[128l];
        int v22[128l];
        int v23;
        v23 = 0l;
        while (while_method_1(v23)){
            assert("Tensor range check" && 0 <= v23 && v23 < 32l);
            int v25;
            v25 = 4l * v23;
            assert("Tensor range check" && 0 <= v23 && v23 < 32l);
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
                    v47 = v30 < 32l;
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
                assert("Tensor range check" && 0 <= v30 && v30 < 32l);
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
            v60 = v17 < 64l;
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
        bool v65[128l];
        int v66;
        v66 = 0l;
        while (while_method_1(v66)){
            int v68;
            v68 = 0l;
            while (while_method_2(v68)){
                assert("Tensor range check" && 0 <= v66 && v66 < 32l);
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
                v74 = v73 < 11l;
                assert("Tensor range check" && 0 <= v66 && v66 < 32l);
                assert("Tensor range check" && 0 <= v68 && v68 < 4l);
                v65[v71] = v74;
                v68 += 1l ;
            }
            v66 += 1l ;
        }
        int v75[128l];
        int v76;
        v76 = 0l;
        while (while_method_1(v76)){
            int v78;
            v78 = 0l;
            while (while_method_2(v78)){
                assert("Tensor range check" && 0 <= v76 && v76 < 32l);
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
                assert("Tensor range check" && 0 <= v76 && v76 < 32l);
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
                assert("Tensor range check" && 0 <= v85 && v85 < 32l);
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
        float v99[128l];
        int v100;
        v100 = 0l;
        while (while_method_1(v100)){
            int v102;
            v102 = 0l;
            while (while_method_2(v102)){
                assert("Tensor range check" && 0 <= v100 && v100 < 32l);
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
                assert("Tensor range check" && 0 <= v100 && v100 < 32l);
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
                assert("Tensor range check" && 0 <= v110 && v110 < 32l);
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
        float v126[128l];
        int v127;
        v127 = 0l;
        while (while_method_1(v127)){
            int v129;
            v129 = 0l;
            while (while_method_2(v129)){
                assert("Tensor range check" && 0 <= v127 && v127 < 32l);
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
                assert("Tensor range check" && 0 <= v127 && v127 < 32l);
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
                assert("Tensor range check" && 0 <= v139 && v139 < 32l);
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
        float v152[128l];
        int v153;
        v153 = 0l;
        while (while_method_1(v153)){
            int v155;
            v155 = 0l;
            while (while_method_2(v155)){
                assert("Tensor range check" && 0 <= v153 && v153 < 32l);
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
                    v163 = 0.00024414062f;
                }
                assert("Tensor range check" && 0 <= v153 && v153 < 32l);
                assert("Tensor range check" && 0 <= v155 && v155 < 4l);
                v152[v158] = v163;
                v155 += 1l ;
            }
            v153 += 1l ;
        }
        bool v164[128l];
        int v165;
        v165 = 0l;
        while (while_method_1(v165)){
            int v167;
            v167 = 0l;
            while (while_method_2(v167)){
                assert("Tensor range check" && 0 <= v165 && v165 < 32l);
                assert("Tensor range check" && 0 <= v167 && v167 < 4l);
                int v169;
                v169 = 4l * v165;
                int v170;
                v170 = v169 + v167;
                float v171;
                v171 = v152[v170];
                bool v172;
                v172 = 0.0f == v171;
                bool v173;
                v173 = v172 != true;
                assert("Tensor range check" && 0 <= v165 && v165 < 32l);
                assert("Tensor range check" && 0 <= v167 && v167 < 4l);
                v164[v170] = v173;
                v167 += 1l ;
            }
            v165 += 1l ;
        }
        float v174[128l];
        float v175;
        v175 = 0.0f;
        int v176;
        v176 = 0l;
        while (while_method_1(v176)){
            assert("Tensor range check" && 0 <= v176 && v176 < 32l);
            int v178;
            v178 = 4l * v176;
            assert("Tensor range check" && 0 <= v176 && v176 < 32l);
            int v179; float v180;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v179 = tmp0.v0; v180 = tmp0.v1;
            while (while_method_2(v179)){
                assert("Tensor range check" && 0 <= v179 && v179 < 4l);
                int v182;
                v182 = v179 + v178;
                float v183;
                v183 = v152[v182];
                float v184;
                v184 = v180 + v183;
                v180 = v184;
                v179 += 1l ;
            }
            auto v185 = cooperative_groups::coalesced_threads();
            int v186;
            v186 = threadIdx.x;
            int v187;
            v187 = v186 / 32l;
            auto v188 = cooperative_groups::labeled_partition(v185,v187);
            Closure2 v189{};
            float v190;
            v190 = cooperative_groups::inclusive_scan(v188, v180, v189);
            float v191;
            v191 = v188.shfl_up(v190,1);
            bool v192;
            v192 = v188.thread_rank() == 0;
            float v193;
            if (v192){
                v193 = 0.0f;
            } else {
                v193 = v191;
            }
            float v194;
            v194 = v188.shfl(v190,v188.num_threads()-1);
            float v195;
            v195 = v175 + v193;
            int v196; float v197;
            Tuple0 tmp1 = Tuple0{0l, v195};
            v196 = tmp1.v0; v197 = tmp1.v1;
            while (while_method_2(v196)){
                assert("Tensor range check" && 0 <= v196 && v196 < 4l);
                int v199;
                v199 = v196 + v178;
                float v200;
                v200 = v152[v199];
                float v201;
                v201 = v197 + v200;
                assert("Tensor range check" && 0 <= v196 && v196 < 4l);
                v174[v199] = v201;
                v197 = v201;
                v196 += 1l ;
            }
            float v202;
            v202 = v175 + v194;
            v175 = v202;
            v176 += 1l ;
        }
        float v203;
        v203 = -1.0f / 0.0f;
        int v204;
        v204 = 0l;
        while (while_method_1(v204)){
            int v206;
            v206 = 0l;
            while (while_method_2(v206)){
                assert("Tensor range check" && 0 <= v204 && v204 < 32l);
                assert("Tensor range check" && 0 <= v206 && v206 < 4l);
                int v208;
                v208 = 4l * v204;
                int v209;
                v209 = v208 + v206;
                float v210;
                v210 = v174[v209];
                bool v211;
                v211 = v203 >= v210;
                float v212;
                if (v211){
                    v212 = v203;
                } else {
                    v212 = v210;
                }
                v203 = v212;
                v206 += 1l ;
            }
            v204 += 1l ;
        }
        auto v213 = cooperative_groups::coalesced_threads();
        int v214;
        v214 = threadIdx.x;
        int v215;
        v215 = v214 / 32l;
        auto v216 = cooperative_groups::labeled_partition(v213,v215);
        Closure3 v217{};
        float v218;
        v218 = cooperative_groups::reduce(v216, v203, v217);
        float v219[128l];
        int v220;
        v220 = 0l;
        while (while_method_1(v220)){
            int v222;
            v222 = 0l;
            while (while_method_2(v222)){
                assert("Tensor range check" && 0 <= v220 && v220 < 32l);
                assert("Tensor range check" && 0 <= v222 && v222 < 4l);
                int v224;
                v224 = 4l * v220;
                int v225;
                v225 = v224 + v222;
                float v226;
                v226 = v174[v225];
                bool v227;
                v227 = v164[v225];
                float v229;
                if (v227){
                    float v228;
                    v228 = v226 - v218;
                    v229 = v228;
                } else {
                    v229 = -1.0f / 0.0f;
                }
                assert("Tensor range check" && 0 <= v220 && v220 < 32l);
                assert("Tensor range check" && 0 <= v222 && v222 < 4l);
                v219[v225] = v229;
                v222 += 1l ;
            }
            v220 += 1l ;
        }
        float v230; int v231;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, 2147483647l};
        v230 = tmp2.v0; v231 = tmp2.v1;
        int v232;
        v232 = 0l;
        while (while_method_1(v232)){
            int v234;
            v234 = 0l;
            while (while_method_2(v234)){
                assert("Tensor range check" && 0 <= v232 && v232 < 32l);
                assert("Tensor range check" && 0 <= v234 && v234 < 4l);
                int v236;
                v236 = 4l * v232;
                int v237;
                v237 = v236 + v234;
                float v238;
                v238 = v219[v237];
                int v239;
                v239 = v22[v237];
                float v240; int v241;
                Tuple1 tmp3 = method_0(v230, v231, v238, v239);
                v240 = tmp3.v0; v241 = tmp3.v1;
                v230 = v240;
                v231 = v241;
                v234 += 1l ;
            }
            v232 += 1l ;
        }
        auto v242 = cooperative_groups::coalesced_threads();
        int v243;
        v243 = threadIdx.x;
        int v244;
        v244 = v243 / 32l;
        auto v245 = cooperative_groups::labeled_partition(v242,v244);
        Closure4 v246{};
        float v247; int v248;
        Tuple1 tmp4 = cooperative_groups::reduce(v245, Tuple1{v230, v231}, v246);
        v247 = tmp4.v0; v248 = tmp4.v1;
        bool v249;
        v249 = v248 < 11l;
        bool v250;
        v250 = v249 == false;
        if (v250){
            int v251;
            v251 = threadIdx.x;
            cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v252 = console_lock;
            auto v253 = cooperative_groups::coalesced_threads();
            v252.acquire();
            int v254;
            v254 = 0l;
            printf("{%s = %d; %s = %d; %s = %c","output_id", v248, "tid", v251, "x", '[');
            int v255;
            v255 = 0l;
            while (while_method_1(v255)){
                int v257;
                v257 = v254;
                bool v258;
                v258 = v257 >= 100l;
                if (v258){
                    printf("%s"," ...");
                    break;
                } else {
                }
                bool v259;
                v259 = v255 == 0l;
                bool v260;
                v260 = v259 != true;
                if (v260){
                    printf("%s","; ");
                } else {
                }
                printf("%c",'[');
                int v261;
                v261 = 0l;
                while (while_method_2(v261)){
                    int v263;
                    v263 = v254;
                    bool v264;
                    v264 = v263 >= 100l;
                    if (v264){
                        printf("%s"," ...");
                        break;
                    } else {
                    }
                    bool v265;
                    v265 = v261 == 0l;
                    bool v266;
                    v266 = v265 != true;
                    if (v266){
                        printf("%s","; ");
                    } else {
                    }
                    int v267;
                    v267 = v254 + 1l;
                    v254 = v267;
                    int v268;
                    v268 = v255 * 4l;
                    int v269;
                    v269 = v268 + v261;
                    float v270;
                    v270 = v219[v269];
                    printf("%f",v270);
                    v261 += 1l ;
                }
                printf("%c",']');
                v255 += 1l ;
            }
            printf("%c",']');
            printf("}\n");
            v252.release();
            v253.sync() ;
        } else {
        }
        assert("Tensor range check" && 0 <= v17 && v17 < 64l);
        int v302;
        v302 = 0l;
        while (while_method_1(v302)){
            assert("Tensor range check" && 0 <= v302 && v302 < 32l);
            int v304;
            v304 = 128l * v302;
            int v305;
            v305 = v304 + v20;
            assert("Tensor range check" && 0 <= v302 && v302 < 32l);
            int v306;
            v306 = 4l * v302;
            int4* v307;
            v307 = reinterpret_cast<int4*>(v152 + v306);
            int4* v308;
            v308 = reinterpret_cast<int4*>(v1 + v305);
            assert("Pointer alignment check" && (unsigned long long)(v307) % 4l == 0 && (unsigned long long)(v308) % 4l == 0);
            *v308 = *v307;
            v302 += 1l ;
        }
        assert("Tensor range check" && 0 <= v17 && v17 < 64l);
        v2[v64] = v248;
        v17 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
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
    v1 = v0 < 64
    del v0
    return v1
def method1(v0 : i32) -> bool:
    v1 = v0 < 4096
    del v0
    return v1
def main():
    v0 = cp.arange(0,262144,1,dtype=cp.int32) # type: ignore
    v1 = v0.size
    del v0
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
    v6 = cp.random.uniform(size=64,dtype=cp.float32) # type: ignore
    del v6
    v7 = cp.empty(262144,dtype=cp.float32)
    v8 = cp.empty(64,dtype=cp.int32)
    v9 = 0
    v10 = raw_module.get_function(f"entry{v9}")
    del v9
    v10.max_dynamic_shared_size_bytes = 0 
    v10((1,),(32,),(v5, v7, v8),shared_mem=0)
    del v5, v10
    v37 = 0
    v38 = "{}"
    print(v38.format('['),end="")
    v39 = 0
    while method0(v39):
        v41 = v37
        v42 = v41 >= 1024
        del v41
        if v42:
            v43 = " ..."
            print(v38.format(v43),end="")
            del v43
            break
        else:
            pass
        del v42
        v44 = v39 == 0
        v45 = v44 != True
        del v44
        if v45:
            v46 = "; "
            print(v38.format(v46),end="")
            del v46
        else:
            pass
        del v45
        print(v38.format('['),end="")
        v47 = 0
        while method1(v47):
            v49 = v37
            v50 = v49 >= 1024
            del v49
            if v50:
                v51 = " ..."
                print(v38.format(v51),end="")
                del v51
                break
            else:
                pass
            del v50
            v52 = v47 == 0
            v53 = v52 != True
            del v52
            if v53:
                v54 = "; "
                print(v38.format(v54),end="")
                del v54
            else:
                pass
            del v53
            v55 = v37 + 1
            v37 = v55
            del v55
            v56 = v39 * 4096
            v57 = v56 + v47
            del v56
            v58 = v7[v57].item()
            del v57
            v59 = "{:.6f}"
            print(v59.format(v58),end="")
            del v58, v59
            v47 += 1 
        del v47
        print(v38.format(']'),end="")
        v39 += 1 
    del v7, v37, v39
    print(v38.format(']'),end="")
    v60 = "\n"
    print(v60,end="")
    v74 = 0
    print(v38.format('['),end="")
    v75 = 0
    while method0(v75):
        v77 = v74
        v78 = v77 >= 1024
        del v77
        if v78:
            v79 = " ..."
            print(v38.format(v79),end="")
            del v79
            break
        else:
            pass
        del v78
        v80 = v75 == 0
        v81 = v80 != True
        del v80
        if v81:
            v82 = "; "
            print(v38.format(v82),end="")
            del v82
        else:
            pass
        del v81
        v83 = v74 + 1
        v74 = v83
        del v83
        v84 = v8[v75].item()
        print(v38.format(v84),end="")
        del v84
        v75 += 1 
    del v8, v74, v75
    print(v38.format(']'),end="")
    del v38
    print(v60,end="")
    del v60
    return 

if __name__ == '__main__': print(main())
