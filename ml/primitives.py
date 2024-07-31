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
        return method_0(v2, v3, v0, v1);
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 32l;
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
    printf("%i, %i\n", v1, v3);
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
    v9 = v5 % 4l;
    int v10;
    v10 = v5 / 4l;
    bool v11;
    v11 = v10 < 8l;
    bool v12;
    v12 = v11 == false;
    if (v12){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v11);
    } else {
    }
    assert("Tensor range check" && 0 <= v10 && v10 < 8l);
    assert("Tensor range check" && 0 <= v9 && v9 < 4l);
    int v14;
    v14 = 4l * v9;
    int v15;
    v15 = 16l * v10;
    int v16;
    v16 = v15 + v14;
    assert("Tensor range check" && 0 <= v10 && v10 < 8l);
    assert("Tensor range check" && 0 <= v9 && v9 < 4l);
    assert("Tensor range check" && 0 <= v10 && v10 < 8l);
    int v17;
    v17 = 0l;
    while (while_method_0(v17)){
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
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
            v26 = 16l * v23;
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
                    v40 = v9 < 4l;
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
                v51 = v30 * 16l;
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
            v60 = v17 < 32l;
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
        v64 = v17 * 8l;
        int v65;
        v65 = v64 + v10;
        bool v66[4l];
        int v67;
        v67 = 0l;
        while (while_method_1(v67)){
            int v69;
            v69 = 0l;
            while (while_method_2(v69)){
                assert("Tensor range check" && 0 <= v67 && v67 < 1l);
                assert("Tensor range check" && 0 <= v69 && v69 < 4l);
                int v71;
                v71 = 4l * v67;
                int v72;
                v72 = v71 + v69;
                float v73;
                v73 = v21[v72];
                int v74;
                v74 = v22[v72];
                bool v75;
                v75 = v74 < 3l;
                assert("Tensor range check" && 0 <= v67 && v67 < 1l);
                assert("Tensor range check" && 0 <= v69 && v69 < 4l);
                v66[v72] = v75;
                v69 += 1l ;
            }
            v67 += 1l ;
        }
        int v76[4l];
        int v77;
        v77 = 0l;
        while (while_method_1(v77)){
            int v79;
            v79 = 0l;
            while (while_method_2(v79)){
                assert("Tensor range check" && 0 <= v77 && v77 < 1l);
                assert("Tensor range check" && 0 <= v79 && v79 < 4l);
                int v81;
                v81 = 4l * v77;
                int v82;
                v82 = v81 + v79;
                bool v83;
                v83 = v66[v82];
                int v84;
                if (v83){
                    v84 = 1l;
                } else {
                    v84 = 0l;
                }
                assert("Tensor range check" && 0 <= v77 && v77 < 1l);
                assert("Tensor range check" && 0 <= v79 && v79 < 4l);
                v76[v82] = v84;
                v79 += 1l ;
            }
            v77 += 1l ;
        }
        int v85;
        v85 = 0l;
        int v86;
        v86 = 0l;
        while (while_method_1(v86)){
            int v88;
            v88 = 0l;
            while (while_method_2(v88)){
                assert("Tensor range check" && 0 <= v86 && v86 < 1l);
                assert("Tensor range check" && 0 <= v88 && v88 < 4l);
                int v90;
                v90 = 4l * v86;
                int v91;
                v91 = v90 + v88;
                int v92;
                v92 = v76[v91];
                int v93;
                v93 = v85 + v92;
                v85 = v93;
                v88 += 1l ;
            }
            v86 += 1l ;
        }
        auto v94 = cooperative_groups::coalesced_threads();
        int v95;
        v95 = threadIdx.x;
        int v96;
        v96 = v95 / 4l;
        auto v97 = cooperative_groups::labeled_partition(v94,v96);
        Closure0 v98{};
        int v99;
        v99 = cooperative_groups::reduce(v97, v85, v98);
        float v100[4l];
        int v101;
        v101 = 0l;
        while (while_method_1(v101)){
            int v103;
            v103 = 0l;
            while (while_method_2(v103)){
                assert("Tensor range check" && 0 <= v101 && v101 < 1l);
                assert("Tensor range check" && 0 <= v103 && v103 < 4l);
                int v105;
                v105 = 4l * v101;
                int v106;
                v106 = v105 + v103;
                float v107;
                v107 = v21[v106];
                bool v108;
                v108 = v66[v106];
                float v109;
                if (v108){
                    v109 = v107;
                } else {
                    v109 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v101 && v101 < 1l);
                assert("Tensor range check" && 0 <= v103 && v103 < 4l);
                v100[v106] = v109;
                v103 += 1l ;
            }
            v101 += 1l ;
        }
        float v110;
        v110 = 0.0f;
        int v111;
        v111 = 0l;
        while (while_method_1(v111)){
            int v113;
            v113 = 0l;
            while (while_method_2(v113)){
                assert("Tensor range check" && 0 <= v111 && v111 < 1l);
                assert("Tensor range check" && 0 <= v113 && v113 < 4l);
                int v115;
                v115 = 4l * v111;
                int v116;
                v116 = v115 + v113;
                float v117;
                v117 = v100[v116];
                float v118;
                v118 = v110 + v117;
                v110 = v118;
                v113 += 1l ;
            }
            v111 += 1l ;
        }
        auto v119 = cooperative_groups::coalesced_threads();
        int v120;
        v120 = threadIdx.x;
        int v121;
        v121 = v120 / 4l;
        auto v122 = cooperative_groups::labeled_partition(v119,v121);
        Closure1 v123{};
        float v124;
        v124 = cooperative_groups::reduce(v122, v110, v123);
        float v125;
        v125 = (float)v99;
        float v126;
        v126 = v124 / v125;
        float v127[4l];
        int v128;
        v128 = 0l;
        while (while_method_1(v128)){
            int v130;
            v130 = 0l;
            while (while_method_2(v130)){
                assert("Tensor range check" && 0 <= v128 && v128 < 1l);
                assert("Tensor range check" && 0 <= v130 && v130 < 4l);
                int v132;
                v132 = 4l * v128;
                int v133;
                v133 = v132 + v130;
                float v134;
                v134 = v21[v133];
                bool v135;
                v135 = v66[v133];
                float v136;
                if (v135){
                    v136 = v134;
                } else {
                    v136 = -1.0f / 0.0f;
                }
                float v137;
                v137 = v136 - v126;
                float v138;
                v138 = exp(v137);
                assert("Tensor range check" && 0 <= v128 && v128 < 1l);
                assert("Tensor range check" && 0 <= v130 && v130 < 4l);
                v127[v133] = v138;
                v130 += 1l ;
            }
            v128 += 1l ;
        }
        float v139;
        v139 = 0.0f;
        int v140;
        v140 = 0l;
        while (while_method_1(v140)){
            int v142;
            v142 = 0l;
            while (while_method_2(v142)){
                assert("Tensor range check" && 0 <= v140 && v140 < 1l);
                assert("Tensor range check" && 0 <= v142 && v142 < 4l);
                int v144;
                v144 = 4l * v140;
                int v145;
                v145 = v144 + v142;
                float v146;
                v146 = v127[v145];
                float v147;
                v147 = v139 + v146;
                v139 = v147;
                v142 += 1l ;
            }
            v140 += 1l ;
        }
        auto v148 = cooperative_groups::coalesced_threads();
        int v149;
        v149 = threadIdx.x;
        int v150;
        v150 = v149 / 4l;
        auto v151 = cooperative_groups::labeled_partition(v148,v150);
        float v152;
        v152 = cooperative_groups::reduce(v151, v139, v123);
        float v153[4l];
        int v154;
        v154 = 0l;
        while (while_method_1(v154)){
            int v156;
            v156 = 0l;
            while (while_method_2(v156)){
                assert("Tensor range check" && 0 <= v154 && v154 < 1l);
                assert("Tensor range check" && 0 <= v156 && v156 < 4l);
                int v158;
                v158 = 4l * v154;
                int v159;
                v159 = v158 + v156;
                float v160;
                v160 = v127[v159];
                bool v161;
                v161 = v152 == 0.0f;
                bool v162;
                v162 = v161 != true;
                float v164;
                if (v162){
                    float v163;
                    v163 = v160 / v152;
                    v164 = v163;
                } else {
                    v164 = 0.0625f;
                }
                assert("Tensor range check" && 0 <= v154 && v154 < 1l);
                assert("Tensor range check" && 0 <= v156 && v156 < 4l);
                v153[v159] = v164;
                v156 += 1l ;
            }
            v154 += 1l ;
        }
        float v165[4l];
        float v166;
        v166 = 0.0f;
        int v167;
        v167 = 0l;
        while (while_method_1(v167)){
            assert("Tensor range check" && 0 <= v167 && v167 < 1l);
            int v169;
            v169 = 4l * v167;
            assert("Tensor range check" && 0 <= v167 && v167 < 1l);
            int v170; float v171;
            Tuple0 tmp0 = Tuple0{0l, 0.0f};
            v170 = tmp0.v0; v171 = tmp0.v1;
            while (while_method_2(v170)){
                assert("Tensor range check" && 0 <= v170 && v170 < 4l);
                int v173;
                v173 = v170 + v169;
                float v174;
                v174 = v153[v173];
                float v175;
                v175 = v171 + v174;
                v171 = v175;
                v170 += 1l ;
            }
            auto v176 = cooperative_groups::coalesced_threads();
            int v177;
            v177 = threadIdx.x;
            int v178;
            v178 = v177 / 4l;
            auto v179 = cooperative_groups::labeled_partition(v176,v178);
            Closure2 v180{};
            float v181;
            v181 = cooperative_groups::inclusive_scan(v179, v171, v180);
            float v182;
            v182 = v179.shfl_up(v181,1);
            bool v183;
            v183 = v179.thread_rank() == 0;
            float v184;
            if (v183){
                v184 = 0.0f;
            } else {
                v184 = v182;
            }
            float v185;
            v185 = v179.shfl(v181,v179.num_threads()-1);
            float v186;
            v186 = v166 + v184;
            int v187; float v188;
            Tuple0 tmp1 = Tuple0{0l, v186};
            v187 = tmp1.v0; v188 = tmp1.v1;
            while (while_method_2(v187)){
                assert("Tensor range check" && 0 <= v187 && v187 < 4l);
                int v190;
                v190 = v187 + v169;
                float v191;
                v191 = v153[v190];
                float v192;
                v192 = v188 + v191;
                assert("Tensor range check" && 0 <= v187 && v187 < 4l);
                v165[v190] = v192;
                v188 = v192;
                v187 += 1l ;
            }
            float v193;
            v193 = v166 + v185;
            v166 = v193;
            v167 += 1l ;
        }
        float v194;
        v194 = curand_uniform(&v4);
        float v195[4l];
        int v196;
        v196 = 0l;
        while (while_method_1(v196)){
            int v198;
            v198 = 0l;
            while (while_method_2(v198)){
                assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                assert("Tensor range check" && 0 <= v198 && v198 < 4l);
                int v200;
                v200 = 4l * v196;
                int v201;
                v201 = v200 + v198;
                int v202;
                v202 = v22[v201];
                assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                assert("Tensor range check" && 0 <= v198 && v198 < 4l);
                v195[v201] = v194;
                v198 += 1l ;
            }
            v196 += 1l ;
        }
        float v203;
        v203 = 0.0f;
        int v204;
        v204 = 0l;
        while (while_method_1(v204)){
            int v206;
            v206 = 0l;
            while (while_method_2(v206)){
                assert("Tensor range check" && 0 <= v204 && v204 < 1l);
                assert("Tensor range check" && 0 <= v206 && v206 < 4l);
                int v208;
                v208 = 4l * v204;
                int v209;
                v209 = v208 + v206;
                float v210;
                v210 = v195[v209];
                v203 = v210;
                v206 += 1l ;
            }
            v204 += 1l ;
        }
        auto v211 = cooperative_groups::coalesced_threads();
        int v212;
        v212 = threadIdx.x;
        int v213;
        v213 = v212 / 4l;
        auto v214 = cooperative_groups::labeled_partition(v211,v213);
        Closure3 v215{};
        float v216;
        v216 = cooperative_groups::reduce(v214, v203, v215);
        float v217[4l];
        int v218;
        v218 = 0l;
        while (while_method_1(v218)){
            int v220;
            v220 = 0l;
            while (while_method_2(v220)){
                assert("Tensor range check" && 0 <= v218 && v218 < 1l);
                assert("Tensor range check" && 0 <= v220 && v220 < 4l);
                int v222;
                v222 = 4l * v218;
                int v223;
                v223 = v222 + v220;
                float v224;
                v224 = v165[v223];
                float v225;
                v225 = v224 - v216;
                assert("Tensor range check" && 0 <= v218 && v218 < 1l);
                assert("Tensor range check" && 0 <= v220 && v220 < 4l);
                v217[v223] = v225;
                v220 += 1l ;
            }
            v218 += 1l ;
        }
        float v226; int v227;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, 0l};
        v226 = tmp2.v0; v227 = tmp2.v1;
        int v228;
        v228 = 0l;
        while (while_method_1(v228)){
            int v230;
            v230 = 0l;
            while (while_method_2(v230)){
                assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                assert("Tensor range check" && 0 <= v230 && v230 < 4l);
                int v232;
                v232 = 4l * v228;
                int v233;
                v233 = v232 + v230;
                float v234;
                v234 = v217[v233];
                int v235;
                v235 = v22[v233];
                float v236; int v237;
                Tuple1 tmp3 = method_0(v226, v227, v234, v235);
                v236 = tmp3.v0; v237 = tmp3.v1;
                v226 = v236;
                v227 = v237;
                v230 += 1l ;
            }
            v228 += 1l ;
        }
        auto v238 = cooperative_groups::coalesced_threads();
        int v239;
        v239 = threadIdx.x;
        int v240;
        v240 = v239 / 4l;
        auto v241 = cooperative_groups::labeled_partition(v238,v240);
        Closure4 v242{};
        float v243; int v244;
        Tuple1 tmp4 = cooperative_groups::reduce(v241, Tuple1{v226, v227}, v242);
        v243 = tmp4.v0; v244 = tmp4.v1;
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
        int v245;
        v245 = 0l;
        while (while_method_1(v245)){
            assert("Tensor range check" && 0 <= v245 && v245 < 1l);
            int v247;
            v247 = 16l * v245;
            int v248;
            v248 = v247 + v20;
            assert("Tensor range check" && 0 <= v245 && v245 < 1l);
            int v249;
            v249 = 4l * v245;
            int4* v250;
            v250 = reinterpret_cast<int4*>(v153 + v249);
            int4* v251;
            v251 = reinterpret_cast<int4*>(v1 + v248);
            assert("Pointer alignment check" && (unsigned long long)(v250) % 4l == 0 && (unsigned long long)(v251) % 4l == 0);
            *v251 = *v250;
            v245 += 1l ;
        }
        assert("Tensor range check" && 0 <= v17 && v17 < 32l);
        int v252;
        v252 = 8l * v17;
        int v253;
        v253 = v252 + v10;
        v2[v253] = v244;
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
    v1 = v0 < 256
    del v0
    return v1
def method2(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method3(v0 : i32) -> bool:
    v1 = v0 < 16
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
    v0 = cp.arange(0,4096,1,dtype=cp.int32) # type: ignore
    v1 = v0.size
    del v0
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
    v6 = cp.random.uniform(size=256,dtype=cp.float32) # type: ignore
    del v6
    v7 = cp.empty(4096,dtype=cp.float32)
    v8 = cp.empty(256,dtype=cp.int32)
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
            v31 = v13 * 16
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
