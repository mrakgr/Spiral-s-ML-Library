kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
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
__device__ Tuple0 method_1(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10);
__device__ float method_2(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10);
__device__ void push_0(curandStatePhilox4_32_10_t & v0, double * v1, double * v2, int * v3, int * v4, float * v5, int * v6, int * v7, double * v8, double * v9, int * v10, float * v11, float * v12, float * v13, float * v14, float * v15, float * v16, float * v17, int v18, int v19, int v20, int v21);
__device__ unsigned int loop_4(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ int int_range_3(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Tuple0 {
    float v0;
    int v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(float t0, int t1) : v0(t0), v1(t1) {}
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
struct Closure4 {
    __device__ Tuple0 operator()(Tuple0 tup0, Tuple0 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple0{v0, v1};
        } else {
            return Tuple0{v2, v3};
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
struct Closure6 {
    int v0;
    __device__ Tuple0 operator()(Tuple0 tup0, Tuple0 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple0{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple0{v3, v4};
            } else {
                return Tuple0{v1, v2};
            }
        }
    }
    __device__ Closure6(int _v0) : v0(_v0) { }
};
struct Closure7 {
    __device__ bool operator()(bool tup0, bool tup1){
        bool v0 = tup0; bool v1 = tup1;
        bool v2;
        v2 = v0 || v1;
        return v2;
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ Tuple0 method_1(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10){
    assert("Tensor range check" && 0 <= v10 && v10 < 4l);
    int v11;
    v11 = 16384l * v10;
    assert("Tensor range check" && 0 <= v9 && v9 < 4096l);
    int v12;
    v12 = 4l * v9;
    int v13;
    v13 = v12 + v11;
    float * v14;
    v14 = v2+v13;
    float * v16;
    v16 = v3+v13;
    int v18;
    v18 = sizeof(float *);
    unsigned long long v19;
    v19 = (unsigned long long)v18;
    unsigned long long v20;
    v20 = 256ull * v19;
    unsigned long long v21;
    v21 = v20 + 16ull;
    unsigned long long v22;
    v22 = v21 - 1ull;
    unsigned long long v23;
    v23 = v22 % 16ull;
    unsigned long long v24;
    v24 = v22 - v23;
    unsigned long long v25;
    v25 = v24 + v20;
    unsigned long long v26;
    v26 = v25 + 16ull;
    unsigned long long v27;
    v27 = v26 - 1ull;
    unsigned long long v28;
    v28 = v27 % 16ull;
    unsigned long long v29;
    v29 = v27 - v28;
    unsigned long long v30;
    v30 = v29 + 1024ull;
    unsigned long long v31;
    v31 = v30 + 16ull;
    unsigned long long v32;
    v32 = v31 - 1ull;
    unsigned long long v33;
    v33 = v32 % 16ull;
    unsigned long long v34;
    v34 = v32 - v33;
    unsigned long long v35;
    v35 = v34 + 1024ull;
    bool v36;
    v36 = v35 <= 81920ull;
    bool v37;
    v37 = v36 == false;
    if (v37){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v36);
    } else {
    }
    extern __shared__ unsigned char v39[];
    bool v40;
    v40 = v35 <= v35;
    bool v41;
    v41 = v40 == false;
    if (v41){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v40);
    } else {
    }
    float * * v43;
    v43 = reinterpret_cast<float * *>(&v39[0ull]);
    float * * v45;
    v45 = reinterpret_cast<float * *>(&v39[v24]);
    float * v47;
    v47 = reinterpret_cast<float *>(&v39[v29]);
    int * v49;
    v49 = reinterpret_cast<int *>(&v39[v34]);
    int v51;
    v51 = threadIdx.x;
    assert("Tensor range check" && 0 <= v51 && v51 < 256l);
    v43[v51] = v14;
    v45[v51] = v16;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v52;
    v52 = 0l <= v51;
    bool v53;
    v53 = v52 == false;
    if (v53){
        assert("The index needs to be zero or positive." && v52);
    } else {
    }
    int v55;
    v55 = v51 % 1l;
    bool v56;
    v56 = v51 < 256l;
    bool v57;
    v57 = v56 == false;
    if (v57){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v56);
    } else {
    }
    assert("Tensor range check" && 0 <= v51 && v51 < 256l);
    int v59;
    v59 = 0l;
    while (while_method_0(v59)){
        bool v61;
        v61 = v52 && v56;
        bool v62;
        v62 = v61 == false;
        if (v62){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v61);
        } else {
        }
        bool v64;
        v64 = 0l <= v59;
        bool v66;
        if (v64){
            bool v65;
            v65 = v59 < 1l;
            v66 = v65;
        } else {
            v66 = false;
        }
        bool v67;
        v67 = v66 == false;
        if (v67){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v66);
        } else {
        }
        int v69;
        v69 = v59 * 256l;
        int v70;
        v70 = v69 + v51;
        assert("Tensor range check" && 0 <= v59 && v59 < 1l);
        int v71;
        v71 = 256l * v59;
        int v72;
        v72 = v71 + v51;
        float * v73;
        v73 = v43[v72];
        float * v74;
        v74 = v45[v72];
        int v75;
        v75 = blockIdx.x;
        int v76;
        v76 = v75 * 256l;
        int v77;
        v77 = v76 + v70;
        assert("Tensor range check" && 0 <= v55 && v55 < 1l);
        int v78;
        v78 = 4l * v55;
        float v79[4l];
        float v80[4l];
        int v81[4l];
        int v82;
        v82 = 0l;
        while (while_method_0(v82)){
            assert("Tensor range check" && 0 <= v82 && v82 < 1l);
            int v84;
            v84 = 4l * v82;
            assert("Tensor range check" && 0 <= v82 && v82 < 1l);
            int v85;
            v85 = v84 + v78;
            int4* v86;
            v86 = reinterpret_cast<int4*>(v73 + v85);
            int4* v87;
            v87 = reinterpret_cast<int4*>(v79 + v84);
            assert("Pointer alignment check" && (unsigned long long)(v86) % 4l == 0 && (unsigned long long)(v87) % 4l == 0);
            *v87 = *v86;
            int4* v88;
            v88 = reinterpret_cast<int4*>(v74 + v85);
            int4* v89;
            v89 = reinterpret_cast<int4*>(v80 + v84);
            assert("Pointer alignment check" && (unsigned long long)(v88) % 4l == 0 && (unsigned long long)(v89) % 4l == 0);
            *v89 = *v88;
            v82 += 1l ;
        }
        int v90;
        v90 = 0l;
        while (while_method_0(v90)){
            int v92;
            v92 = 0l;
            while (while_method_1(v92)){
                bool v94;
                v94 = 0l <= v92;
                bool v96;
                if (v94){
                    bool v95;
                    v95 = v92 < 4l;
                    v96 = v95;
                } else {
                    v96 = false;
                }
                bool v97;
                v97 = v96 == false;
                if (v97){
                    assert("The indices should be inside the range of the dimension." && v96);
                } else {
                }
                bool v99;
                v99 = 0l <= v55;
                bool v101;
                if (v99){
                    bool v100;
                    v100 = v55 < 1l;
                    v101 = v100;
                } else {
                    v101 = false;
                }
                bool v102;
                v102 = v101 == false;
                if (v102){
                    assert("The indices should be inside the range of the dimension." && v101);
                } else {
                }
                int v104;
                v104 = v55 * 4l;
                int v105;
                v105 = v92 + v104;
                bool v106;
                v106 = 0l <= v90;
                bool v108;
                if (v106){
                    bool v107;
                    v107 = v90 < 1l;
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
                int v111;
                v111 = v90 * 4l;
                int v112;
                v112 = v105 + v111;
                assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                assert("Tensor range check" && 0 <= v92 && v92 < 4l);
                int v113;
                v113 = 4l * v90;
                int v114;
                v114 = v113 + v92;
                v81[v114] = v112;
                v92 += 1l ;
            }
            v90 += 1l ;
        }
        bool v115[4l];
        int v116;
        v116 = 0l;
        while (while_method_0(v116)){
            int v118;
            v118 = 0l;
            while (while_method_1(v118)){
                assert("Tensor range check" && 0 <= v116 && v116 < 1l);
                assert("Tensor range check" && 0 <= v118 && v118 < 4l);
                int v120;
                v120 = 4l * v116;
                int v121;
                v121 = v120 + v118;
                float v122;
                v122 = v79[v121];
                int v123;
                v123 = v81[v121];
                bool v124;
                v124 = v123 < 3l;
                assert("Tensor range check" && 0 <= v116 && v116 < 1l);
                assert("Tensor range check" && 0 <= v118 && v118 < 4l);
                v115[v121] = v124;
                v118 += 1l ;
            }
            v116 += 1l ;
        }
        float v125[4l];
        int v126;
        v126 = 0l;
        while (while_method_0(v126)){
            int v128;
            v128 = 0l;
            while (while_method_1(v128)){
                assert("Tensor range check" && 0 <= v126 && v126 < 1l);
                assert("Tensor range check" && 0 <= v128 && v128 < 4l);
                int v130;
                v130 = 4l * v126;
                int v131;
                v131 = v130 + v128;
                float v132;
                v132 = v79[v131];
                bool v133;
                v133 = v115[v131];
                float v136;
                if (v133){
                    bool v134;
                    v134 = 0.0f >= v132;
                    if (v134){
                        v136 = 0.0f;
                    } else {
                        v136 = v132;
                    }
                } else {
                    v136 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v126 && v126 < 1l);
                assert("Tensor range check" && 0 <= v128 && v128 < 4l);
                v125[v131] = v136;
                v128 += 1l ;
            }
            v126 += 1l ;
        }
        float v137;
        v137 = 0.0f;
        int v138;
        v138 = 0l;
        while (while_method_0(v138)){
            int v140;
            v140 = 0l;
            while (while_method_1(v140)){
                assert("Tensor range check" && 0 <= v138 && v138 < 1l);
                assert("Tensor range check" && 0 <= v140 && v140 < 4l);
                int v142;
                v142 = 4l * v138;
                int v143;
                v143 = v142 + v140;
                float v144;
                v144 = v125[v143];
                float v145;
                v145 = v137 + v144;
                v137 = v145;
                v140 += 1l ;
            }
            v138 += 1l ;
        }
        auto v146 = cooperative_groups::coalesced_threads();
        int v147;
        v147 = threadIdx.x;
        auto v148 = cooperative_groups::labeled_partition(v146,v147);
        Closure0 v149{};
        float v150;
        v150 = cooperative_groups::reduce(v148, v137, v149);
        int v151[4l];
        int v152;
        v152 = 0l;
        while (while_method_0(v152)){
            int v154;
            v154 = 0l;
            while (while_method_1(v154)){
                assert("Tensor range check" && 0 <= v152 && v152 < 1l);
                assert("Tensor range check" && 0 <= v154 && v154 < 4l);
                int v156;
                v156 = 4l * v152;
                int v157;
                v157 = v156 + v154;
                bool v158;
                v158 = v115[v157];
                int v159;
                if (v158){
                    v159 = 1l;
                } else {
                    v159 = 0l;
                }
                assert("Tensor range check" && 0 <= v152 && v152 < 1l);
                assert("Tensor range check" && 0 <= v154 && v154 < 4l);
                v151[v157] = v159;
                v154 += 1l ;
            }
            v152 += 1l ;
        }
        int v160;
        v160 = 0l;
        int v161;
        v161 = 0l;
        while (while_method_0(v161)){
            int v163;
            v163 = 0l;
            while (while_method_1(v163)){
                assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                assert("Tensor range check" && 0 <= v163 && v163 < 4l);
                int v165;
                v165 = 4l * v161;
                int v166;
                v166 = v165 + v163;
                int v167;
                v167 = v151[v166];
                int v168;
                v168 = v160 + v167;
                v160 = v168;
                v163 += 1l ;
            }
            v161 += 1l ;
        }
        auto v169 = cooperative_groups::coalesced_threads();
        int v170;
        v170 = threadIdx.x;
        auto v171 = cooperative_groups::labeled_partition(v169,v170);
        Closure1 v172{};
        int v173;
        v173 = cooperative_groups::reduce(v171, v160, v172);
        float v174;
        v174 = (float)v173;
        float v175;
        v175 = 1.0f / v174;
        float v176[4l];
        int v177;
        v177 = 0l;
        while (while_method_0(v177)){
            int v179;
            v179 = 0l;
            while (while_method_1(v179)){
                assert("Tensor range check" && 0 <= v177 && v177 < 1l);
                assert("Tensor range check" && 0 <= v179 && v179 < 4l);
                int v181;
                v181 = 4l * v177;
                int v182;
                v182 = v181 + v179;
                float v183;
                v183 = v125[v182];
                bool v184;
                v184 = v115[v182];
                bool v185;
                v185 = v184 == false;
                float v190;
                if (v185){
                    v190 = 0.0f;
                } else {
                    bool v186;
                    v186 = v150 == 0.0f;
                    bool v187;
                    v187 = v186 != true;
                    if (v187){
                        float v188;
                        v188 = v183 / v150;
                        v190 = v188;
                    } else {
                        v190 = v175;
                    }
                }
                assert("Tensor range check" && 0 <= v177 && v177 < 1l);
                assert("Tensor range check" && 0 <= v179 && v179 < 4l);
                v176[v182] = v190;
                v179 += 1l ;
            }
            v177 += 1l ;
        }
        float v191[4l];
        float v192;
        v192 = 0.0f;
        int v193;
        v193 = 0l;
        while (while_method_0(v193)){
            assert("Tensor range check" && 0 <= v193 && v193 < 1l);
            int v195;
            v195 = 4l * v193;
            assert("Tensor range check" && 0 <= v193 && v193 < 1l);
            int v196; float v197;
            Tuple1 tmp0 = Tuple1{0l, 0.0f};
            v196 = tmp0.v0; v197 = tmp0.v1;
            while (while_method_1(v196)){
                assert("Tensor range check" && 0 <= v196 && v196 < 4l);
                int v199;
                v199 = v196 + v195;
                float v200;
                v200 = v176[v199];
                float v201;
                v201 = v197 + v200;
                v197 = v201;
                v196 += 1l ;
            }
            auto v202 = cooperative_groups::coalesced_threads();
            int v203;
            v203 = threadIdx.x;
            auto v204 = cooperative_groups::labeled_partition(v202,v203);
            Closure2 v205{};
            float v206;
            v206 = cooperative_groups::inclusive_scan(v204, v197, v205);
            float v207;
            v207 = v204.shfl_up(v206,1);
            bool v208;
            v208 = v204.thread_rank() == 0;
            float v209;
            if (v208){
                v209 = 0.0f;
            } else {
                v209 = v207;
            }
            float v210;
            v210 = v204.shfl(v206,v204.num_threads()-1);
            float v211;
            v211 = v192 + v209;
            int v212; float v213;
            Tuple1 tmp1 = Tuple1{0l, v211};
            v212 = tmp1.v0; v213 = tmp1.v1;
            while (while_method_1(v212)){
                assert("Tensor range check" && 0 <= v212 && v212 < 4l);
                int v215;
                v215 = v212 + v195;
                float v216;
                v216 = v176[v215];
                float v217;
                v217 = v213 + v216;
                assert("Tensor range check" && 0 <= v212 && v212 < 4l);
                v191[v215] = v217;
                v213 = v217;
                v212 += 1l ;
            }
            float v218;
            v218 = v192 + v210;
            v192 = v218;
            v193 += 1l ;
        }
        float v219[4l];
        bool v220[4l];
        int v221;
        v221 = 0l;
        while (while_method_0(v221)){
            int v223;
            v223 = 0l;
            while (while_method_1(v223)){
                assert("Tensor range check" && 0 <= v221 && v221 < 1l);
                assert("Tensor range check" && 0 <= v223 && v223 < 4l);
                int v225;
                v225 = 4l * v221;
                int v226;
                v226 = v225 + v223;
                float v227;
                v227 = v191[v226];
                float v228;
                v228 = v176[v226];
                bool v229;
                v229 = v228 > 0.0f;
                assert("Tensor range check" && 0 <= v221 && v221 < 1l);
                assert("Tensor range check" && 0 <= v223 && v223 < 4l);
                v219[v226] = v227;
                v220[v226] = v229;
                v223 += 1l ;
            }
            v221 += 1l ;
        }
        float v230; bool v231;
        Tuple2 tmp2 = Tuple2{-1.0f / 0.0f, false};
        v230 = tmp2.v0; v231 = tmp2.v1;
        int v232;
        v232 = 0l;
        while (while_method_0(v232)){
            int v234;
            v234 = 0l;
            while (while_method_1(v234)){
                assert("Tensor range check" && 0 <= v232 && v232 < 1l);
                assert("Tensor range check" && 0 <= v234 && v234 < 4l);
                int v236;
                v236 = 4l * v232;
                int v237;
                v237 = v236 + v234;
                float v238;
                v238 = v219[v237];
                bool v239;
                v239 = v220[v237];
                float v246; bool v247;
                if (v231){
                    if (v239){
                        bool v240;
                        v240 = v230 >= v238;
                        float v241;
                        if (v240){
                            v241 = v230;
                        } else {
                            v241 = v238;
                        }
                        v246 = v241; v247 = true;
                    } else {
                        v246 = v230; v247 = v231;
                    }
                } else {
                    if (v239){
                        v246 = v238; v247 = v239;
                    } else {
                        v246 = v230; v247 = v231;
                    }
                }
                v230 = v246;
                v231 = v247;
                v234 += 1l ;
            }
            v232 += 1l ;
        }
        auto v248 = cooperative_groups::coalesced_threads();
        int v249;
        v249 = threadIdx.x;
        auto v250 = cooperative_groups::labeled_partition(v248,v249);
        Closure3 v251{};
        float v252; bool v253;
        Tuple2 tmp3 = cooperative_groups::reduce(v250, Tuple2{v230, v231}, v251);
        v252 = tmp3.v0; v253 = tmp3.v1;
        bool v254;
        v254 = v253 == false;
        if (v254){
            assert("The local reduce must be true." && v253);
        } else {
        }
        float v256[4l];
        int v257[4l];
        int v258;
        v258 = 0l;
        while (while_method_0(v258)){
            int v260;
            v260 = 0l;
            while (while_method_1(v260)){
                assert("Tensor range check" && 0 <= v258 && v258 < 1l);
                assert("Tensor range check" && 0 <= v260 && v260 < 4l);
                int v262;
                v262 = 4l * v258;
                int v263;
                v263 = v262 + v260;
                int v264;
                v264 = v81[v263];
                float v265;
                v265 = curand_uniform(&v0);
                assert("Tensor range check" && 0 <= v258 && v258 < 1l);
                assert("Tensor range check" && 0 <= v260 && v260 < 4l);
                v256[v263] = v265;
                v257[v263] = v264;
                v260 += 1l ;
            }
            v258 += 1l ;
        }
        float v266; int v267;
        Tuple0 tmp4 = Tuple0{0.0f, 2147483647l};
        v266 = tmp4.v0; v267 = tmp4.v1;
        int v268;
        v268 = 0l;
        while (while_method_0(v268)){
            int v270;
            v270 = 0l;
            while (while_method_1(v270)){
                assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                assert("Tensor range check" && 0 <= v270 && v270 < 4l);
                int v272;
                v272 = 4l * v268;
                int v273;
                v273 = v272 + v270;
                float v274;
                v274 = v256[v273];
                int v275;
                v275 = v257[v273];
                bool v276;
                v276 = v267 < v275;
                float v277; int v278;
                if (v276){
                    v277 = v266; v278 = v267;
                } else {
                    v277 = v274; v278 = v275;
                }
                v266 = v277;
                v267 = v278;
                v270 += 1l ;
            }
            v268 += 1l ;
        }
        auto v279 = cooperative_groups::coalesced_threads();
        int v280;
        v280 = threadIdx.x;
        auto v281 = cooperative_groups::labeled_partition(v279,v280);
        Closure4 v282{};
        float v283; int v284;
        Tuple0 tmp5 = cooperative_groups::reduce(v281, Tuple0{v266, v267}, v282);
        v283 = tmp5.v0; v284 = tmp5.v1;
        float v285;
        v285 = v252 * v283;
        int v286[4l];
        bool v287[4l];
        int v288;
        v288 = 0l;
        while (while_method_0(v288)){
            int v290;
            v290 = 0l;
            while (while_method_1(v290)){
                assert("Tensor range check" && 0 <= v288 && v288 < 1l);
                assert("Tensor range check" && 0 <= v290 && v290 < 4l);
                int v292;
                v292 = 4l * v288;
                int v293;
                v293 = v292 + v290;
                float v294;
                v294 = v219[v293];
                bool v295;
                v295 = v220[v293];
                int v296;
                v296 = v81[v293];
                int v299; bool v300;
                if (v295){
                    float v297;
                    v297 = v294 - v285;
                    bool v298;
                    v298 = v297 >= 0.0f;
                    v299 = v296; v300 = v298;
                } else {
                    v299 = 2147483647l; v300 = false;
                }
                assert("Tensor range check" && 0 <= v288 && v288 < 1l);
                assert("Tensor range check" && 0 <= v290 && v290 < 4l);
                v286[v293] = v299;
                v287[v293] = v300;
                v290 += 1l ;
            }
            v288 += 1l ;
        }
        int v301; bool v302;
        Tuple3 tmp6 = Tuple3{2147483647l, false};
        v301 = tmp6.v0; v302 = tmp6.v1;
        int v303;
        v303 = 0l;
        while (while_method_0(v303)){
            int v305;
            v305 = 0l;
            while (while_method_1(v305)){
                assert("Tensor range check" && 0 <= v303 && v303 < 1l);
                assert("Tensor range check" && 0 <= v305 && v305 < 4l);
                int v307;
                v307 = 4l * v303;
                int v308;
                v308 = v307 + v305;
                int v309;
                v309 = v286[v308];
                bool v310;
                v310 = v287[v308];
                int v317; bool v318;
                if (v302){
                    if (v310){
                        bool v311;
                        v311 = v301 < v309;
                        int v312;
                        if (v311){
                            v312 = v301;
                        } else {
                            v312 = v309;
                        }
                        v317 = v312; v318 = true;
                    } else {
                        v317 = v301; v318 = v302;
                    }
                } else {
                    if (v310){
                        v317 = v309; v318 = v310;
                    } else {
                        v317 = v301; v318 = v302;
                    }
                }
                v301 = v317;
                v302 = v318;
                v305 += 1l ;
            }
            v303 += 1l ;
        }
        auto v319 = cooperative_groups::coalesced_threads();
        int v320;
        v320 = threadIdx.x;
        auto v321 = cooperative_groups::labeled_partition(v319,v320);
        Closure5 v322{};
        int v323; bool v324;
        Tuple3 tmp7 = cooperative_groups::reduce(v321, Tuple3{v301, v302}, v322);
        v323 = tmp7.v0; v324 = tmp7.v1;
        bool v325;
        v325 = v324 == false;
        if (v325){
            assert("The local reduce must be true." && v324);
        } else {
        }
        bool v327[4l];
        int v328;
        v328 = 0l;
        while (while_method_0(v328)){
            int v330;
            v330 = 0l;
            while (while_method_1(v330)){
                assert("Tensor range check" && 0 <= v328 && v328 < 1l);
                assert("Tensor range check" && 0 <= v330 && v330 < 4l);
                int v332;
                v332 = 4l * v328;
                int v333;
                v333 = v332 + v330;
                float v334;
                v334 = v80[v333];
                int v335;
                v335 = v81[v333];
                bool v336;
                v336 = v335 < 3l;
                assert("Tensor range check" && 0 <= v328 && v328 < 1l);
                assert("Tensor range check" && 0 <= v330 && v330 < 4l);
                v327[v333] = v336;
                v330 += 1l ;
            }
            v328 += 1l ;
        }
        float v337[4l];
        int v338;
        v338 = 0l;
        while (while_method_0(v338)){
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
                v344 = v80[v343];
                bool v345;
                v345 = v327[v343];
                float v348;
                if (v345){
                    bool v346;
                    v346 = 0.0f >= v344;
                    if (v346){
                        v348 = 0.0f;
                    } else {
                        v348 = v344;
                    }
                } else {
                    v348 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v338 && v338 < 1l);
                assert("Tensor range check" && 0 <= v340 && v340 < 4l);
                v337[v343] = v348;
                v340 += 1l ;
            }
            v338 += 1l ;
        }
        float v349;
        v349 = 0.0f;
        int v350;
        v350 = 0l;
        while (while_method_0(v350)){
            int v352;
            v352 = 0l;
            while (while_method_1(v352)){
                assert("Tensor range check" && 0 <= v350 && v350 < 1l);
                assert("Tensor range check" && 0 <= v352 && v352 < 4l);
                int v354;
                v354 = 4l * v350;
                int v355;
                v355 = v354 + v352;
                float v356;
                v356 = v337[v355];
                float v357;
                v357 = v349 + v356;
                v349 = v357;
                v352 += 1l ;
            }
            v350 += 1l ;
        }
        auto v358 = cooperative_groups::coalesced_threads();
        int v359;
        v359 = threadIdx.x;
        auto v360 = cooperative_groups::labeled_partition(v358,v359);
        float v361;
        v361 = cooperative_groups::reduce(v360, v349, v149);
        int v362[4l];
        int v363;
        v363 = 0l;
        while (while_method_0(v363)){
            int v365;
            v365 = 0l;
            while (while_method_1(v365)){
                assert("Tensor range check" && 0 <= v363 && v363 < 1l);
                assert("Tensor range check" && 0 <= v365 && v365 < 4l);
                int v367;
                v367 = 4l * v363;
                int v368;
                v368 = v367 + v365;
                bool v369;
                v369 = v327[v368];
                int v370;
                if (v369){
                    v370 = 1l;
                } else {
                    v370 = 0l;
                }
                assert("Tensor range check" && 0 <= v363 && v363 < 1l);
                assert("Tensor range check" && 0 <= v365 && v365 < 4l);
                v362[v368] = v370;
                v365 += 1l ;
            }
            v363 += 1l ;
        }
        int v371;
        v371 = 0l;
        int v372;
        v372 = 0l;
        while (while_method_0(v372)){
            int v374;
            v374 = 0l;
            while (while_method_1(v374)){
                assert("Tensor range check" && 0 <= v372 && v372 < 1l);
                assert("Tensor range check" && 0 <= v374 && v374 < 4l);
                int v376;
                v376 = 4l * v372;
                int v377;
                v377 = v376 + v374;
                int v378;
                v378 = v362[v377];
                int v379;
                v379 = v371 + v378;
                v371 = v379;
                v374 += 1l ;
            }
            v372 += 1l ;
        }
        auto v380 = cooperative_groups::coalesced_threads();
        int v381;
        v381 = threadIdx.x;
        auto v382 = cooperative_groups::labeled_partition(v380,v381);
        int v383;
        v383 = cooperative_groups::reduce(v382, v371, v172);
        float v384;
        v384 = (float)v383;
        float v385;
        v385 = 1.0f / v384;
        float v386[4l];
        int v387;
        v387 = 0l;
        while (while_method_0(v387)){
            int v389;
            v389 = 0l;
            while (while_method_1(v389)){
                assert("Tensor range check" && 0 <= v387 && v387 < 1l);
                assert("Tensor range check" && 0 <= v389 && v389 < 4l);
                int v391;
                v391 = 4l * v387;
                int v392;
                v392 = v391 + v389;
                float v393;
                v393 = v337[v392];
                bool v394;
                v394 = v327[v392];
                bool v395;
                v395 = v394 == false;
                float v400;
                if (v395){
                    v400 = 0.0f;
                } else {
                    bool v396;
                    v396 = v361 == 0.0f;
                    bool v397;
                    v397 = v396 != true;
                    if (v397){
                        float v398;
                        v398 = v393 / v361;
                        v400 = v398;
                    } else {
                        v400 = v385;
                    }
                }
                assert("Tensor range check" && 0 <= v387 && v387 < 1l);
                assert("Tensor range check" && 0 <= v389 && v389 < 4l);
                v386[v392] = v400;
                v389 += 1l ;
            }
            v387 += 1l ;
        }
        float v401; int v402;
        Tuple0 tmp8 = Tuple0{0.0f, 2147483647l};
        v401 = tmp8.v0; v402 = tmp8.v1;
        int v403;
        v403 = 0l;
        while (while_method_0(v403)){
            int v405;
            v405 = 0l;
            while (while_method_1(v405)){
                assert("Tensor range check" && 0 <= v403 && v403 < 1l);
                assert("Tensor range check" && 0 <= v405 && v405 < 4l);
                int v407;
                v407 = 4l * v403;
                int v408;
                v408 = v407 + v405;
                float v409;
                v409 = v176[v408];
                int v410;
                v410 = v81[v408];
                bool v411;
                v411 = v402 == v323;
                float v415; int v416;
                if (v411){
                    v415 = v401; v416 = v402;
                } else {
                    bool v412;
                    v412 = v410 == v323;
                    if (v412){
                        v415 = v409; v416 = v410;
                    } else {
                        v415 = v401; v416 = v402;
                    }
                }
                v401 = v415;
                v402 = v416;
                v405 += 1l ;
            }
            v403 += 1l ;
        }
        auto v417 = cooperative_groups::coalesced_threads();
        int v418;
        v418 = threadIdx.x;
        auto v419 = cooperative_groups::labeled_partition(v417,v418);
        Closure6 v420{v323};
        float v421; int v422;
        Tuple0 tmp9 = cooperative_groups::reduce(v419, Tuple0{v401, v402}, v420);
        v421 = tmp9.v0; v422 = tmp9.v1;
        bool v423;
        v423 = v422 == 2147483647l;
        bool v424;
        v424 = v423 != true;
        bool v425;
        v425 = v424 == false;
        if (v425){
            assert("Expected a valid action id in get_action." && v424);
        } else {
        }
        int v427;
        v427 = 0l;
        while (while_method_0(v427)){
            assert("Tensor range check" && 0 <= v427 && v427 < 1l);
            assert("Tensor range check" && 0 <= v427 && v427 < 1l);
            v427 += 1l ;
        }
        assert("Tensor range check" && 0 <= v70 && v70 < 256l);
        v47[v70] = v421;
        v49[v70] = v323;
        v59 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v51 && v51 < 256l);
    float v429;
    v429 = v47[v51];
    int v430;
    v430 = v49[v51];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return Tuple0{v429, v430};
}
__device__ float method_2(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10){
    assert("Tensor range check" && 0 <= v9 && v9 < 4l);
    int v11;
    v11 = 16384l * v9;
    assert("Tensor range check" && 0 <= v8 && v8 < 4096l);
    int v12;
    v12 = 4l * v8;
    int v13;
    v13 = v12 + v11;
    float * v14;
    v14 = v2+v13;
    int v16;
    v16 = sizeof(float *);
    unsigned long long v17;
    v17 = (unsigned long long)v16;
    unsigned long long v18;
    v18 = 256ull * v17;
    unsigned long long v19;
    v19 = 1024ull + v18;
    unsigned long long v20;
    v20 = v19 + 16ull;
    unsigned long long v21;
    v21 = v20 - 1ull;
    unsigned long long v22;
    v22 = v21 % 16ull;
    unsigned long long v23;
    v23 = v21 - v22;
    unsigned long long v24;
    v24 = v23 + 1024ull;
    bool v25;
    v25 = v24 <= 81920ull;
    bool v26;
    v26 = v25 == false;
    if (v26){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v25);
    } else {
    }
    extern __shared__ unsigned char v28[];
    bool v29;
    v29 = v24 <= v24;
    bool v30;
    v30 = v29 == false;
    if (v30){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v29);
    } else {
    }
    int * v32;
    v32 = reinterpret_cast<int *>(&v28[0ull]);
    float * * v34;
    v34 = reinterpret_cast<float * *>(&v28[1024ull]);
    float * v36;
    v36 = reinterpret_cast<float *>(&v28[v23]);
    int v38;
    v38 = threadIdx.x;
    assert("Tensor range check" && 0 <= v38 && v38 < 256l);
    v32[v38] = v10;
    v34[v38] = v14;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v39;
    v39 = 0l <= v38;
    bool v40;
    v40 = v39 == false;
    if (v40){
        assert("The index needs to be zero or positive." && v39);
    } else {
    }
    int v42;
    v42 = v38 % 1l;
    bool v43;
    v43 = v38 < 256l;
    bool v44;
    v44 = v43 == false;
    if (v44){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v43);
    } else {
    }
    assert("Tensor range check" && 0 <= v38 && v38 < 256l);
    int v46;
    v46 = 0l;
    while (while_method_0(v46)){
        bool v48;
        v48 = v39 && v43;
        bool v49;
        v49 = v48 == false;
        if (v49){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v48);
        } else {
        }
        bool v51;
        v51 = 0l <= v46;
        bool v53;
        if (v51){
            bool v52;
            v52 = v46 < 1l;
            v53 = v52;
        } else {
            v53 = false;
        }
        bool v54;
        v54 = v53 == false;
        if (v54){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v53);
        } else {
        }
        int v56;
        v56 = v46 * 256l;
        int v57;
        v57 = v56 + v38;
        assert("Tensor range check" && 0 <= v46 && v46 < 1l);
        int v58;
        v58 = 256l * v46;
        int v59;
        v59 = v58 + v38;
        int v60;
        v60 = v32[v59];
        float * v61;
        v61 = v34[v59];
        int v62;
        v62 = blockIdx.x;
        int v63;
        v63 = v62 * 256l;
        int v64;
        v64 = v63 + v57;
        assert("Tensor range check" && 0 <= v42 && v42 < 1l);
        int v65;
        v65 = 4l * v42;
        float v66[4l];
        int v67[4l];
        int v68;
        v68 = 0l;
        while (while_method_0(v68)){
            assert("Tensor range check" && 0 <= v68 && v68 < 1l);
            int v70;
            v70 = 4l * v68;
            assert("Tensor range check" && 0 <= v68 && v68 < 1l);
            int v71;
            v71 = v70 + v65;
            int4* v72;
            v72 = reinterpret_cast<int4*>(v61 + v71);
            int4* v73;
            v73 = reinterpret_cast<int4*>(v66 + v70);
            assert("Pointer alignment check" && (unsigned long long)(v72) % 4l == 0 && (unsigned long long)(v73) % 4l == 0);
            *v73 = *v72;
            v68 += 1l ;
        }
        int v74;
        v74 = 0l;
        while (while_method_0(v74)){
            int v76;
            v76 = 0l;
            while (while_method_1(v76)){
                bool v78;
                v78 = 0l <= v76;
                bool v80;
                if (v78){
                    bool v79;
                    v79 = v76 < 4l;
                    v80 = v79;
                } else {
                    v80 = false;
                }
                bool v81;
                v81 = v80 == false;
                if (v81){
                    assert("The indices should be inside the range of the dimension." && v80);
                } else {
                }
                bool v83;
                v83 = 0l <= v42;
                bool v85;
                if (v83){
                    bool v84;
                    v84 = v42 < 1l;
                    v85 = v84;
                } else {
                    v85 = false;
                }
                bool v86;
                v86 = v85 == false;
                if (v86){
                    assert("The indices should be inside the range of the dimension." && v85);
                } else {
                }
                int v88;
                v88 = v42 * 4l;
                int v89;
                v89 = v76 + v88;
                bool v90;
                v90 = 0l <= v74;
                bool v92;
                if (v90){
                    bool v91;
                    v91 = v74 < 1l;
                    v92 = v91;
                } else {
                    v92 = false;
                }
                bool v93;
                v93 = v92 == false;
                if (v93){
                    assert("The indices should be inside the range of the dimension." && v92);
                } else {
                }
                int v95;
                v95 = v74 * 4l;
                int v96;
                v96 = v89 + v95;
                assert("Tensor range check" && 0 <= v74 && v74 < 1l);
                assert("Tensor range check" && 0 <= v76 && v76 < 4l);
                int v97;
                v97 = 4l * v74;
                int v98;
                v98 = v97 + v76;
                v67[v98] = v96;
                v76 += 1l ;
            }
            v74 += 1l ;
        }
        bool v99[4l];
        int v100;
        v100 = 0l;
        while (while_method_0(v100)){
            int v102;
            v102 = 0l;
            while (while_method_1(v102)){
                assert("Tensor range check" && 0 <= v100 && v100 < 1l);
                assert("Tensor range check" && 0 <= v102 && v102 < 4l);
                int v104;
                v104 = 4l * v100;
                int v105;
                v105 = v104 + v102;
                float v106;
                v106 = v66[v105];
                int v107;
                v107 = v67[v105];
                bool v108;
                v108 = v107 < 3l;
                assert("Tensor range check" && 0 <= v100 && v100 < 1l);
                assert("Tensor range check" && 0 <= v102 && v102 < 4l);
                v99[v105] = v108;
                v102 += 1l ;
            }
            v100 += 1l ;
        }
        float v109[4l];
        int v110;
        v110 = 0l;
        while (while_method_0(v110)){
            int v112;
            v112 = 0l;
            while (while_method_1(v112)){
                assert("Tensor range check" && 0 <= v110 && v110 < 1l);
                assert("Tensor range check" && 0 <= v112 && v112 < 4l);
                int v114;
                v114 = 4l * v110;
                int v115;
                v115 = v114 + v112;
                float v116;
                v116 = v66[v115];
                bool v117;
                v117 = v99[v115];
                float v120;
                if (v117){
                    bool v118;
                    v118 = 0.0f >= v116;
                    if (v118){
                        v120 = 0.0f;
                    } else {
                        v120 = v116;
                    }
                } else {
                    v120 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v110 && v110 < 1l);
                assert("Tensor range check" && 0 <= v112 && v112 < 4l);
                v109[v115] = v120;
                v112 += 1l ;
            }
            v110 += 1l ;
        }
        float v121;
        v121 = 0.0f;
        int v122;
        v122 = 0l;
        while (while_method_0(v122)){
            int v124;
            v124 = 0l;
            while (while_method_1(v124)){
                assert("Tensor range check" && 0 <= v122 && v122 < 1l);
                assert("Tensor range check" && 0 <= v124 && v124 < 4l);
                int v126;
                v126 = 4l * v122;
                int v127;
                v127 = v126 + v124;
                float v128;
                v128 = v109[v127];
                float v129;
                v129 = v121 + v128;
                v121 = v129;
                v124 += 1l ;
            }
            v122 += 1l ;
        }
        auto v130 = cooperative_groups::coalesced_threads();
        int v131;
        v131 = threadIdx.x;
        auto v132 = cooperative_groups::labeled_partition(v130,v131);
        Closure0 v133{};
        float v134;
        v134 = cooperative_groups::reduce(v132, v121, v133);
        int v135[4l];
        int v136;
        v136 = 0l;
        while (while_method_0(v136)){
            int v138;
            v138 = 0l;
            while (while_method_1(v138)){
                assert("Tensor range check" && 0 <= v136 && v136 < 1l);
                assert("Tensor range check" && 0 <= v138 && v138 < 4l);
                int v140;
                v140 = 4l * v136;
                int v141;
                v141 = v140 + v138;
                bool v142;
                v142 = v99[v141];
                int v143;
                if (v142){
                    v143 = 1l;
                } else {
                    v143 = 0l;
                }
                assert("Tensor range check" && 0 <= v136 && v136 < 1l);
                assert("Tensor range check" && 0 <= v138 && v138 < 4l);
                v135[v141] = v143;
                v138 += 1l ;
            }
            v136 += 1l ;
        }
        int v144;
        v144 = 0l;
        int v145;
        v145 = 0l;
        while (while_method_0(v145)){
            int v147;
            v147 = 0l;
            while (while_method_1(v147)){
                assert("Tensor range check" && 0 <= v145 && v145 < 1l);
                assert("Tensor range check" && 0 <= v147 && v147 < 4l);
                int v149;
                v149 = 4l * v145;
                int v150;
                v150 = v149 + v147;
                int v151;
                v151 = v135[v150];
                int v152;
                v152 = v144 + v151;
                v144 = v152;
                v147 += 1l ;
            }
            v145 += 1l ;
        }
        auto v153 = cooperative_groups::coalesced_threads();
        int v154;
        v154 = threadIdx.x;
        auto v155 = cooperative_groups::labeled_partition(v153,v154);
        Closure1 v156{};
        int v157;
        v157 = cooperative_groups::reduce(v155, v144, v156);
        float v158;
        v158 = (float)v157;
        float v159;
        v159 = 1.0f / v158;
        float v160[4l];
        int v161;
        v161 = 0l;
        while (while_method_0(v161)){
            int v163;
            v163 = 0l;
            while (while_method_1(v163)){
                assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                assert("Tensor range check" && 0 <= v163 && v163 < 4l);
                int v165;
                v165 = 4l * v161;
                int v166;
                v166 = v165 + v163;
                float v167;
                v167 = v109[v166];
                bool v168;
                v168 = v99[v166];
                bool v169;
                v169 = v168 == false;
                float v174;
                if (v169){
                    v174 = 0.0f;
                } else {
                    bool v170;
                    v170 = v134 == 0.0f;
                    bool v171;
                    v171 = v170 != true;
                    if (v171){
                        float v172;
                        v172 = v167 / v134;
                        v174 = v172;
                    } else {
                        v174 = v159;
                    }
                }
                assert("Tensor range check" && 0 <= v161 && v161 < 1l);
                assert("Tensor range check" && 0 <= v163 && v163 < 4l);
                v160[v166] = v174;
                v163 += 1l ;
            }
            v161 += 1l ;
        }
        float v175; int v176;
        Tuple0 tmp11 = Tuple0{0.0f, 2147483647l};
        v175 = tmp11.v0; v176 = tmp11.v1;
        int v177;
        v177 = 0l;
        while (while_method_0(v177)){
            int v179;
            v179 = 0l;
            while (while_method_1(v179)){
                assert("Tensor range check" && 0 <= v177 && v177 < 1l);
                assert("Tensor range check" && 0 <= v179 && v179 < 4l);
                int v181;
                v181 = 4l * v177;
                int v182;
                v182 = v181 + v179;
                float v183;
                v183 = v160[v182];
                int v184;
                v184 = v67[v182];
                bool v185;
                v185 = v176 == v60;
                float v189; int v190;
                if (v185){
                    v189 = v175; v190 = v176;
                } else {
                    bool v186;
                    v186 = v184 == v60;
                    if (v186){
                        v189 = v183; v190 = v184;
                    } else {
                        v189 = v175; v190 = v176;
                    }
                }
                v175 = v189;
                v176 = v190;
                v179 += 1l ;
            }
            v177 += 1l ;
        }
        auto v191 = cooperative_groups::coalesced_threads();
        int v192;
        v192 = threadIdx.x;
        auto v193 = cooperative_groups::labeled_partition(v191,v192);
        Closure6 v194{v60};
        float v195; int v196;
        Tuple0 tmp12 = cooperative_groups::reduce(v193, Tuple0{v175, v176}, v194);
        v195 = tmp12.v0; v196 = tmp12.v1;
        bool v197;
        v197 = v196 == 2147483647l;
        bool v198;
        v198 = v197 != true;
        bool v199;
        v199 = v198 == false;
        if (v199){
            assert("Expected a valid action id in get_action." && v198);
        } else {
        }
        int v201;
        v201 = 0l;
        while (while_method_0(v201)){
            assert("Tensor range check" && 0 <= v201 && v201 < 1l);
            assert("Tensor range check" && 0 <= v201 && v201 < 1l);
            v201 += 1l ;
        }
        assert("Tensor range check" && 0 <= v57 && v57 < 256l);
        v36[v57] = v195;
        v46 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v38 && v38 < 256l);
    float v203;
    v203 = v36[v38];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return v203;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ void push_0(curandStatePhilox4_32_10_t & v0, double * v1, double * v2, int * v3, int * v4, float * v5, int * v6, int * v7, double * v8, double * v9, int * v10, float * v11, float * v12, float * v13, float * v14, float * v15, float * v16, float * v17, int v18, int v19, int v20, int v21){
    float v22; int v23;
    Tuple0 tmp10 = method_1(v0, v10, v11, v12, v13, v14, v15, v16, v17, v20, v21);
    v22 = tmp10.v0; v23 = tmp10.v1;
    float v24;
    v24 = method_2(v10, v11, v12, v13, v14, v15, v16, v17, v20, v21, v23);
    assert("Tensor range check" && 0 <= v21 && v21 < 4l);
    assert("Tensor range check" && 0 <= v18 && v18 < 6144l);
    int v25;
    v25 = 6144l * v21;
    int v26;
    v26 = v25 + v18;
    int v27;
    v27 = v3[v26];
    int v28;
    v28 = v27 + 1l;
    assert("Tensor range check" && 0 <= v21 && v21 < 4l);
    assert("Tensor range check" && 0 <= v18 && v18 < 6144l);
    v3[v26] = v28;
    // qwe;
    assert("Tensor range check" && 0 <= v21 && v21 < 4l);
    assert("Tensor range check" && 0 <= v27 && v27 < 16l);
    assert("Tensor range check" && 0 <= v18 && v18 < 6144l);
    int v29;
    v29 = 6144l * v27;
    int v30;
    v30 = v29 + v18;
    int v31;
    v31 = 98304l * v21;
    int v32;
    v32 = v31 + v30;
    v4[v32] = v23;
    v5[v32] = v22;
    v6[v32] = v19;
    v7[v32] = v20;
    assert("Tensor range check" && 0 <= v21 && v21 < 4l);
    int v33;
    v33 = 12288l * v21;
    assert("Tensor range check" && 0 <= v18 && v18 < 6144l);
    int v34;
    v34 = 2l * v18;
    int v35;
    v35 = v34 + v33;
    assert("Tensor range check" && 0 <= v21 && v21 < 4l);
    int v36;
    v36 = 196608l * v21;
    assert("Tensor range check" && 0 <= v27 && v27 < 16l);
    int v37;
    v37 = 12288l * v27;
    int v38;
    v38 = v37 + v36;
    assert("Tensor range check" && 0 <= v18 && v18 < 6144l);
    int v39;
    v39 = v34 + v38;
    double * v40;
    v40 = v1+v35;
    double * v42;
    v42 = v2+v35;
    double * v44;
    v44 = v8+v39;
    double * v46;
    v46 = v9+v39;
    int v48;
    v48 = sizeof(double *);
    unsigned long long v49;
    v49 = (unsigned long long)v48;
    unsigned long long v50;
    v50 = 256ull * v49;
    unsigned long long v51;
    v51 = v50 + 16ull;
    unsigned long long v52;
    v52 = v51 - 1ull;
    unsigned long long v53;
    v53 = v52 % 16ull;
    unsigned long long v54;
    v54 = v52 - v53;
    unsigned long long v55;
    v55 = v54 + v50;
    unsigned long long v56;
    v56 = v55 + 16ull;
    unsigned long long v57;
    v57 = v56 - 1ull;
    unsigned long long v58;
    v58 = v57 % 16ull;
    unsigned long long v59;
    v59 = v57 - v58;
    unsigned long long v60;
    v60 = v59 + v50;
    unsigned long long v61;
    v61 = v60 + 16ull;
    unsigned long long v62;
    v62 = v61 - 1ull;
    unsigned long long v63;
    v63 = v62 % 16ull;
    unsigned long long v64;
    v64 = v62 - v63;
    unsigned long long v65;
    v65 = v64 + v50;
    bool v66;
    v66 = v65 <= 81920ull;
    bool v67;
    v67 = v66 == false;
    if (v67){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v66);
    } else {
    }
    extern __shared__ unsigned char v69[];
    bool v70;
    v70 = v65 <= v65;
    bool v71;
    v71 = v70 == false;
    if (v71){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v70);
    } else {
    }
    double * * v73;
    v73 = reinterpret_cast<double * *>(&v69[0ull]);
    double * * v75;
    v75 = reinterpret_cast<double * *>(&v69[v54]);
    double * * v77;
    v77 = reinterpret_cast<double * *>(&v69[v59]);
    double * * v79;
    v79 = reinterpret_cast<double * *>(&v69[v64]);
    int v81;
    v81 = threadIdx.x;
    assert("Tensor range check" && 0 <= v81 && v81 < 256l);
    v73[v81] = v40;
    v75[v81] = v42;
    v77[v81] = v44;
    v79[v81] = v46;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v82;
    v82 = 0l <= v81;
    bool v83;
    v83 = v82 == false;
    if (v83){
        assert("The index needs to be zero or positive." && v82);
    } else {
    }
    int v85;
    v85 = v81 % 1l;
    bool v86;
    v86 = v81 < 256l;
    bool v87;
    v87 = v86 == false;
    if (v87){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v86);
    } else {
    }
    assert("Tensor range check" && 0 <= v81 && v81 < 256l);
    int v89;
    v89 = 0l;
    while (while_method_0(v89)){
        bool v91;
        v91 = v82 && v86;
        bool v92;
        v92 = v91 == false;
        if (v92){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v91);
        } else {
        }
        bool v94;
        v94 = 0l <= v89;
        bool v96;
        if (v94){
            bool v95;
            v95 = v89 < 1l;
            v96 = v95;
        } else {
            v96 = false;
        }
        bool v97;
        v97 = v96 == false;
        if (v97){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v96);
        } else {
        }
        int v99;
        v99 = v89 * 256l;
        int v100;
        v100 = v99 + v81;
        assert("Tensor range check" && 0 <= v89 && v89 < 1l);
        int v101;
        v101 = 256l * v89;
        int v102;
        v102 = v101 + v81;
        double * v103;
        v103 = v73[v102];
        double * v104;
        v104 = v75[v102];
        double * v105;
        v105 = v77[v102];
        double * v106;
        v106 = v79[v102];
        int v107;
        v107 = blockIdx.x;
        int v108;
        v108 = v107 * 256l;
        int v109;
        v109 = v108 + v100;
        assert("Tensor range check" && 0 <= v85 && v85 < 1l);
        int v110;
        v110 = 2l * v85;
        double v111[2l];
        double v112[2l];
        int v113[2l];
        int v114;
        v114 = 0l;
        while (while_method_0(v114)){
            assert("Tensor range check" && 0 <= v114 && v114 < 1l);
            int v116;
            v116 = 2l * v114;
            assert("Tensor range check" && 0 <= v114 && v114 < 1l);
            int v117;
            v117 = v116 + v110;
            int4* v118;
            v118 = reinterpret_cast<int4*>(v103 + v117);
            int4* v119;
            v119 = reinterpret_cast<int4*>(v111 + v116);
            assert("Pointer alignment check" && (unsigned long long)(v118) % 2l == 0 && (unsigned long long)(v119) % 2l == 0);
            *v119 = *v118;
            int4* v120;
            v120 = reinterpret_cast<int4*>(v104 + v117);
            int4* v121;
            v121 = reinterpret_cast<int4*>(v112 + v116);
            assert("Pointer alignment check" && (unsigned long long)(v120) % 2l == 0 && (unsigned long long)(v121) % 2l == 0);
            *v121 = *v120;
            v114 += 1l ;
        }
        int v122;
        v122 = 0l;
        while (while_method_0(v122)){
            int v124;
            v124 = 0l;
            while (while_method_2(v124)){
                bool v126;
                v126 = 0l <= v124;
                bool v128;
                if (v126){
                    bool v127;
                    v127 = v124 < 2l;
                    v128 = v127;
                } else {
                    v128 = false;
                }
                bool v129;
                v129 = v128 == false;
                if (v129){
                    assert("The indices should be inside the range of the dimension." && v128);
                } else {
                }
                bool v131;
                v131 = 0l <= v85;
                bool v133;
                if (v131){
                    bool v132;
                    v132 = v85 < 1l;
                    v133 = v132;
                } else {
                    v133 = false;
                }
                bool v134;
                v134 = v133 == false;
                if (v134){
                    assert("The indices should be inside the range of the dimension." && v133);
                } else {
                }
                int v136;
                v136 = v85 * 2l;
                int v137;
                v137 = v124 + v136;
                bool v138;
                v138 = 0l <= v122;
                bool v140;
                if (v138){
                    bool v139;
                    v139 = v122 < 1l;
                    v140 = v139;
                } else {
                    v140 = false;
                }
                bool v141;
                v141 = v140 == false;
                if (v141){
                    assert("The indices should be inside the range of the dimension." && v140);
                } else {
                }
                int v143;
                v143 = v122 * 2l;
                int v144;
                v144 = v137 + v143;
                assert("Tensor range check" && 0 <= v122 && v122 < 1l);
                assert("Tensor range check" && 0 <= v124 && v124 < 2l);
                int v145;
                v145 = 2l * v122;
                int v146;
                v146 = v145 + v124;
                v113[v146] = v144;
                v124 += 1l ;
            }
            v122 += 1l ;
        }
        int v147;
        v147 = 0l;
        while (while_method_0(v147)){
            assert("Tensor range check" && 0 <= v147 && v147 < 1l);
            int v149;
            v149 = 2l * v147;
            int v150;
            v150 = v149 + v110;
            assert("Tensor range check" && 0 <= v147 && v147 < 1l);
            int4* v151;
            v151 = reinterpret_cast<int4*>(v111 + v149);
            int4* v152;
            v152 = reinterpret_cast<int4*>(v105 + v150);
            assert("Pointer alignment check" && (unsigned long long)(v151) % 2l == 0 && (unsigned long long)(v152) % 2l == 0);
            *v152 = *v151;
            int4* v153;
            v153 = reinterpret_cast<int4*>(v112 + v149);
            int4* v154;
            v154 = reinterpret_cast<int4*>(v106 + v150);
            assert("Pointer alignment check" && (unsigned long long)(v153) % 2l == 0 && (unsigned long long)(v154) % 2l == 0);
            *v154 = *v153;
            v147 += 1l ;
        }
        assert("Tensor range check" && 0 <= v100 && v100 < 256l);
        v89 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v81 && v81 < 256l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    double v155;
    v155 = (double)v22;
    double v156;
    v156 = log(v155);
    double v157;
    v157 = (double)v24;
    double v158;
    v158 = log(v157);
    assert("Tensor range check" && 0 <= v21 && v21 < 4l);
    assert("Tensor range check" && 0 <= v18 && v18 < 6144l);
    assert("Tensor range check" && 0 <= v19 && v19 < 2l);
    int v159;
    v159 = v34 + v19;
    int v160;
    v160 = v33 + v159;
    double v161;
    v161 = v1[v160];
    double v162;
    v162 = v2[v160];
    double v163;
    v163 = v158 + v161;
    double v164;
    v164 = v156 + v162;
    assert("Tensor range check" && 0 <= v21 && v21 < 4l);
    assert("Tensor range check" && 0 <= v18 && v18 < 6144l);
    assert("Tensor range check" && 0 <= v19 && v19 < 2l);
    v1[v160] = v163;
    v2[v160] = v164;
    return ;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 > 0l;
    return v1;
}
__device__ unsigned int loop_4(unsigned int v0, curandStatePhilox4_32_10_t & v1){
    unsigned int v2;
    v2 = curand(&v1);
    unsigned int v3;
    v3 = v2 % v0;
    unsigned int v4;
    v4 = v2 - v3;
    unsigned int v5;
    v5 = 0ul - v0;
    bool v6;
    v6 = v4 <= v5;
    if (v6){
        return v3;
    } else {
        return loop_4(v0, v1);
    }
}
__device__ int int_range_3(int v0, int v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = v0 - v1;
    unsigned int v4;
    v4 = (unsigned int)v3;
    unsigned int v5;
    v5 = loop_4(v4, v2);
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned int v7;
    v7 = v5 + v6;
    int v8;
    v8 = (int)v7;
    return v8;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 64l;
    return v1;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    auto v2 = cooperative_groups::this_grid();
    unsigned long long v3;
    v3 = clock64();
    int v4;
    v4 = threadIdx.x;
    int v5;
    v5 = blockIdx.x;
    int v6;
    v6 = v5 * 256l;
    int v7;
    v7 = v4 + v6;
    unsigned long long v8;
    v8 = (unsigned long long)v7;
    curandStatePhilox4_32_10_t v9;
    curand_init(v3,v8,0ull,&v9);
    int * v10;
    v10 = reinterpret_cast<int *>(&v1[0ull]);
    float * v12;
    v12 = reinterpret_cast<float *>(&v1[16ull]);
    float * v14;
    v14 = reinterpret_cast<float *>(&v1[262160ull]);
    float * v16;
    v16 = reinterpret_cast<float *>(&v1[524304ull]);
    float * v18;
    v18 = reinterpret_cast<float *>(&v1[786448ull]);
    float * v20;
    v20 = reinterpret_cast<float *>(&v1[1048592ull]);
    float * v22;
    v22 = reinterpret_cast<float *>(&v1[1310736ull]);
    float * v24;
    v24 = reinterpret_cast<float *>(&v1[1572880ull]);
    int * v26;
    v26 = reinterpret_cast<int *>(&v0[0ull]);
    float * v28;
    v28 = reinterpret_cast<float *>(&v0[1572864ull]);
    int * v30;
    v30 = reinterpret_cast<int *>(&v0[3145728ull]);
    int * v32;
    v32 = reinterpret_cast<int *>(&v0[4718592ull]);
    double * v34;
    v34 = reinterpret_cast<double *>(&v0[6291456ull]);
    double * v36;
    v36 = reinterpret_cast<double *>(&v0[12582912ull]);
    double * v38;
    v38 = reinterpret_cast<double *>(&v1[1835024ull]);
    double * v40;
    v40 = reinterpret_cast<double *>(&v1[2228240ull]);
    int * v42;
    v42 = reinterpret_cast<int *>(&v1[2621456ull]);
    int v44;
    v44 = threadIdx.x;
    int v45;
    v45 = blockIdx.x;
    int v46;
    v46 = v45 * 256l;
    int v47;
    v47 = v44 + v46;
    int v48;
    v48 = 0l;
    int v49;
    v49 = 235l;
    int v50;
    v50 = 0l;
    push_0(v9, v38, v40, v42, v26, v28, v30, v32, v34, v36, v10, v12, v14, v16, v18, v20, v22, v24, v47, v50, v49, v48);
    int v51;
    v51 = 0l;
    int v52;
    v52 = 212l;
    int v53;
    v53 = 1l;
    push_0(v9, v38, v40, v42, v26, v28, v30, v32, v34, v36, v10, v12, v14, v16, v18, v20, v22, v24, v47, v53, v52, v51);
    int v54;
    v54 = 0l;
    int v55;
    v55 = 790l;
    int v56;
    v56 = 0l;
    push_0(v9, v38, v40, v42, v26, v28, v30, v32, v34, v36, v10, v12, v14, v16, v18, v20, v22, v24, v47, v56, v55, v54);
    int v57;
    v57 = 0l;
    int v58;
    v58 = 343l;
    int v59;
    v59 = 1l;
    push_0(v9, v38, v40, v42, v26, v28, v30, v32, v34, v36, v10, v12, v14, v16, v18, v20, v22, v24, v47, v59, v58, v57);
    int v60;
    v60 = 0l;
    int v61;
    v61 = 457l;
    int v62;
    v62 = 0l;
    push_0(v9, v38, v40, v42, v26, v28, v30, v32, v34, v36, v10, v12, v14, v16, v18, v20, v22, v24, v47, v62, v61, v60);
    int v63;
    v63 = 0l;
    int v64;
    v64 = 3447l;
    int v65;
    v65 = 1l;
    push_0(v9, v38, v40, v42, v26, v28, v30, v32, v34, v36, v10, v12, v14, v16, v18, v20, v22, v24, v47, v65, v64, v63);
    static_array<float,2l> v66;
    v66[0l] = 13.0f;
    v66[1l] = -13.0f;
    int v68;
    v68 = threadIdx.x;
    int v69;
    v69 = blockIdx.x;
    int v70;
    v70 = v69 * 256l;
    int v71;
    v71 = v68 + v70;
    float v72[2l];
    int v73;
    v73 = 0l;
    while (while_method_2(v73)){
        float v75;
        v75 = v66[v73];
        v72[v73] = v75;
        v73 += 1l ;
    }
    assert("Tensor range check" && 0 <= v71 && v71 < 6144l);
    int v77;
    v77 = v42[v71];
    int v78;
    v78 = v77;
    while (while_method_3(v78)){
        v78 -= 1l ;
        assert("Tensor range check" && 0 <= v78 && v78 < 16l);
        assert("Tensor range check" && 0 <= v71 && v71 < 6144l);
        int v80;
        v80 = 6144l * v78;
        int v81;
        v81 = v80 + v71;
        int v82;
        v82 = v26[v81];
        float v83;
        v83 = v28[v81];
        int v84;
        v84 = v30[v81];
        int v85;
        v85 = v32[v81];
        assert("Tensor range check" && 0 <= v84 && v84 < 2l);
        float v86;
        v86 = v72[v84];
        assert("Tensor range check" && 0 <= v85 && v85 < 4096l);
        int v87;
        v87 = 4l * v85;
        float * v88;
        v88 = v12+v87;
        float * v90;
        v90 = v14+v87;
        float * v92;
        v92 = v16+v87;
        float * v94;
        v94 = v18+v87;
        float * v96;
        v96 = v20+v87;
        float * v98;
        v98 = v22+v87;
        float * v100;
        v100 = v24+v87;
        assert("Tensor range check" && 0 <= v78 && v78 < 16l);
        int v102;
        v102 = 12288l * v78;
        assert("Tensor range check" && 0 <= v71 && v71 < 6144l);
        int v103;
        v103 = 2l * v71;
        int v104;
        v104 = v103 + v102;
        double v105[2l];
        int v106;
        v106 = 0l;
        while (while_method_2(v106)){
            assert("Tensor range check" && 0 <= v106 && v106 < 2l);
            int v108;
            v108 = v106 + v104;
            double v109;
            v109 = v34[v108];
            bool v110;
            v110 = v84 == v106;
            double v111;
            if (v110){
                v111 = 0.0;
            } else {
                v111 = v109;
            }
            assert("Tensor range check" && 0 <= v106 && v106 < 2l);
            v105[v106] = v111;
            v106 += 1l ;
        }
        double v112;
        v112 = 0.0;
        int v113;
        v113 = 0l;
        while (while_method_2(v113)){
            assert("Tensor range check" && 0 <= v113 && v113 < 2l);
            double v115;
            v115 = v105[v113];
            double v116;
            v116 = v112 + v115;
            v112 = v116;
            v113 += 1l ;
        }
        double v117;
        v117 = 0.0;
        int v118;
        v118 = 0l;
        while (while_method_2(v118)){
            assert("Tensor range check" && 0 <= v118 && v118 < 2l);
            int v120;
            v120 = v118 + v104;
            double v121;
            v121 = v36[v120];
            double v122;
            v122 = v117 + v121;
            v117 = v122;
            v118 += 1l ;
        }
        double v123;
        v123 = v112 - v117;
        double v124;
        v124 = exp(v123);
        float v125;
        v125 = (float)v124;
        float v126;
        v126 = v86 * v125;
        assert("Tensor range check" && 0 <= v82 && v82 < 4l);
        float * v127;
        v127 = v98+v82;
        float * v129;
        v129 = v100+v82;
        float v131;
        v131 = atomicAdd(v127,v126);
        float v132;
        v132 = atomicAdd(v129,v125);
        float * v133;
        v133 = v90+0l;
        float * v135;
        v135 = v94+0l;
        float * v137;
        v137 = v96+0l;
        int v139;
        v139 = sizeof(float *);
        unsigned long long v140;
        v140 = (unsigned long long)v139;
        unsigned long long v141;
        v141 = 256ull * v140;
        unsigned long long v142;
        v142 = 4096ull + v141;
        unsigned long long v143;
        v143 = v142 + 16ull;
        unsigned long long v144;
        v144 = v143 - 1ull;
        unsigned long long v145;
        v145 = v144 % 16ull;
        unsigned long long v146;
        v146 = v144 - v145;
        unsigned long long v147;
        v147 = v146 + v141;
        unsigned long long v148;
        v148 = v147 + 16ull;
        unsigned long long v149;
        v149 = v148 - 1ull;
        unsigned long long v150;
        v150 = v149 % 16ull;
        unsigned long long v151;
        v151 = v149 - v150;
        unsigned long long v152;
        v152 = v151 + v141;
        unsigned long long v153;
        v153 = v152 + 16ull;
        unsigned long long v154;
        v154 = v153 - 1ull;
        unsigned long long v155;
        v155 = v154 % 16ull;
        unsigned long long v156;
        v156 = v154 - v155;
        unsigned long long v157;
        v157 = v156 + v141;
        unsigned long long v158;
        v158 = v157 + 16ull;
        unsigned long long v159;
        v159 = v158 - 1ull;
        unsigned long long v160;
        v160 = v159 % 16ull;
        unsigned long long v161;
        v161 = v159 - v160;
        unsigned long long v162;
        v162 = v161 + 1024ull;
        bool v163;
        v163 = v162 <= 81920ull;
        bool v164;
        v164 = v163 == false;
        if (v164){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v163);
        } else {
        }
        extern __shared__ unsigned char v166[];
        bool v167;
        v167 = v162 <= v162;
        bool v168;
        v168 = v167 == false;
        if (v168){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v167);
        } else {
        }
        float * v170;
        v170 = reinterpret_cast<float *>(&v166[0ull]);
        int * v172;
        v172 = reinterpret_cast<int *>(&v166[1024ull]);
        float * v174;
        v174 = reinterpret_cast<float *>(&v166[2048ull]);
        float * v176;
        v176 = reinterpret_cast<float *>(&v166[3072ull]);
        float * * v178;
        v178 = reinterpret_cast<float * *>(&v166[4096ull]);
        float * * v180;
        v180 = reinterpret_cast<float * *>(&v166[v146]);
        float * * v182;
        v182 = reinterpret_cast<float * *>(&v166[v151]);
        float * * v184;
        v184 = reinterpret_cast<float * *>(&v166[v156]);
        float * v186;
        v186 = reinterpret_cast<float *>(&v166[v161]);
        int v188;
        v188 = threadIdx.x;
        assert("Tensor range check" && 0 <= v188 && v188 < 256l);
        v170[v188] = v83;
        v172[v188] = v82;
        v174[v188] = v86;
        v176[v188] = v125;
        v178[v188] = v92;
        v180[v188] = v133;
        v182[v188] = v135;
        v184[v188] = v137;
        asm("barrier.cta.sync %0;" :: "r"(0l));
        bool v189;
        v189 = 0l <= v188;
        bool v190;
        v190 = v189 == false;
        if (v190){
            assert("The index needs to be zero or positive." && v189);
        } else {
        }
        int v192;
        v192 = v188 % 1l;
        bool v193;
        v193 = v188 < 256l;
        bool v194;
        v194 = v193 == false;
        if (v194){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v193);
        } else {
        }
        assert("Tensor range check" && 0 <= v188 && v188 < 256l);
        int v196;
        v196 = 0l;
        while (while_method_0(v196)){
            bool v198;
            v198 = v189 && v193;
            bool v199;
            v199 = v198 == false;
            if (v199){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v198);
            } else {
            }
            bool v201;
            v201 = 0l <= v196;
            bool v203;
            if (v201){
                bool v202;
                v202 = v196 < 1l;
                v203 = v202;
            } else {
                v203 = false;
            }
            bool v204;
            v204 = v203 == false;
            if (v204){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v203);
            } else {
            }
            int v206;
            v206 = v196 * 256l;
            int v207;
            v207 = v206 + v188;
            assert("Tensor range check" && 0 <= v196 && v196 < 1l);
            int v208;
            v208 = 256l * v196;
            int v209;
            v209 = v208 + v188;
            float v210;
            v210 = v170[v209];
            int v211;
            v211 = v172[v209];
            float v212;
            v212 = v174[v209];
            float v213;
            v213 = v176[v209];
            float * v214;
            v214 = v178[v209];
            float * v215;
            v215 = v180[v209];
            float * v216;
            v216 = v182[v209];
            float * v217;
            v217 = v184[v209];
            int v218;
            v218 = blockIdx.x;
            int v219;
            v219 = v218 * 256l;
            int v220;
            v220 = v219 + v207;
            assert("Tensor range check" && 0 <= v192 && v192 < 1l);
            int v221;
            v221 = 4l * v192;
            float v222[4l];
            float v223[4l];
            float v224[4l];
            int v225[4l];
            int v226;
            v226 = 0l;
            while (while_method_0(v226)){
                assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                int v228;
                v228 = 4l * v226;
                assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                int v229;
                v229 = v228 + v221;
                int4* v230;
                v230 = reinterpret_cast<int4*>(v215 + v229);
                int4* v231;
                v231 = reinterpret_cast<int4*>(v222 + v228);
                assert("Pointer alignment check" && (unsigned long long)(v230) % 4l == 0 && (unsigned long long)(v231) % 4l == 0);
                *v231 = *v230;
                int4* v232;
                v232 = reinterpret_cast<int4*>(v216 + v229);
                int4* v233;
                v233 = reinterpret_cast<int4*>(v223 + v228);
                assert("Pointer alignment check" && (unsigned long long)(v232) % 4l == 0 && (unsigned long long)(v233) % 4l == 0);
                *v233 = *v232;
                int4* v234;
                v234 = reinterpret_cast<int4*>(v217 + v229);
                int4* v235;
                v235 = reinterpret_cast<int4*>(v224 + v228);
                assert("Pointer alignment check" && (unsigned long long)(v234) % 4l == 0 && (unsigned long long)(v235) % 4l == 0);
                *v235 = *v234;
                v226 += 1l ;
            }
            int v236;
            v236 = 0l;
            while (while_method_0(v236)){
                int v238;
                v238 = 0l;
                while (while_method_1(v238)){
                    bool v240;
                    v240 = 0l <= v238;
                    bool v242;
                    if (v240){
                        bool v241;
                        v241 = v238 < 4l;
                        v242 = v241;
                    } else {
                        v242 = false;
                    }
                    bool v243;
                    v243 = v242 == false;
                    if (v243){
                        assert("The indices should be inside the range of the dimension." && v242);
                    } else {
                    }
                    bool v245;
                    v245 = 0l <= v192;
                    bool v247;
                    if (v245){
                        bool v246;
                        v246 = v192 < 1l;
                        v247 = v246;
                    } else {
                        v247 = false;
                    }
                    bool v248;
                    v248 = v247 == false;
                    if (v248){
                        assert("The indices should be inside the range of the dimension." && v247);
                    } else {
                    }
                    int v250;
                    v250 = v192 * 4l;
                    int v251;
                    v251 = v238 + v250;
                    bool v252;
                    v252 = 0l <= v236;
                    bool v254;
                    if (v252){
                        bool v253;
                        v253 = v236 < 1l;
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
                    int v257;
                    v257 = v236 * 4l;
                    int v258;
                    v258 = v251 + v257;
                    assert("Tensor range check" && 0 <= v236 && v236 < 1l);
                    assert("Tensor range check" && 0 <= v238 && v238 < 4l);
                    int v259;
                    v259 = 4l * v236;
                    int v260;
                    v260 = v259 + v238;
                    v225[v260] = v258;
                    v238 += 1l ;
                }
                v236 += 1l ;
            }
            float v261[4l];
            int v262;
            v262 = 0l;
            while (while_method_0(v262)){
                int v264;
                v264 = 0l;
                while (while_method_1(v264)){
                    assert("Tensor range check" && 0 <= v262 && v262 < 1l);
                    assert("Tensor range check" && 0 <= v264 && v264 < 4l);
                    int v266;
                    v266 = 4l * v262;
                    int v267;
                    v267 = v266 + v264;
                    float v268;
                    v268 = v223[v267];
                    float v269;
                    v269 = v224[v267];
                    bool v270;
                    v270 = v269 == 0.0f;
                    bool v271;
                    v271 = v270 != true;
                    float v273;
                    if (v271){
                        float v272;
                        v272 = v268 / v269;
                        v273 = v272;
                    } else {
                        v273 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v262 && v262 < 1l);
                    assert("Tensor range check" && 0 <= v264 && v264 < 4l);
                    v261[v267] = v273;
                    v264 += 1l ;
                }
                v262 += 1l ;
            }
            bool v274[4l];
            int v275;
            v275 = 0l;
            while (while_method_0(v275)){
                int v277;
                v277 = 0l;
                while (while_method_1(v277)){
                    assert("Tensor range check" && 0 <= v275 && v275 < 1l);
                    assert("Tensor range check" && 0 <= v277 && v277 < 4l);
                    int v279;
                    v279 = 4l * v275;
                    int v280;
                    v280 = v279 + v277;
                    float v281;
                    v281 = v222[v280];
                    int v282;
                    v282 = v225[v280];
                    bool v283;
                    v283 = v282 < 3l;
                    assert("Tensor range check" && 0 <= v275 && v275 < 1l);
                    assert("Tensor range check" && 0 <= v277 && v277 < 4l);
                    v274[v280] = v283;
                    v277 += 1l ;
                }
                v275 += 1l ;
            }
            float v284[4l];
            int v285;
            v285 = 0l;
            while (while_method_0(v285)){
                int v287;
                v287 = 0l;
                while (while_method_1(v287)){
                    assert("Tensor range check" && 0 <= v285 && v285 < 1l);
                    assert("Tensor range check" && 0 <= v287 && v287 < 4l);
                    int v289;
                    v289 = 4l * v285;
                    int v290;
                    v290 = v289 + v287;
                    float v291;
                    v291 = v222[v290];
                    bool v292;
                    v292 = v274[v290];
                    float v295;
                    if (v292){
                        bool v293;
                        v293 = 0.0f >= v291;
                        if (v293){
                            v295 = 0.0f;
                        } else {
                            v295 = v291;
                        }
                    } else {
                        v295 = 0.0f;
                    }
                    assert("Tensor range check" && 0 <= v285 && v285 < 1l);
                    assert("Tensor range check" && 0 <= v287 && v287 < 4l);
                    v284[v290] = v295;
                    v287 += 1l ;
                }
                v285 += 1l ;
            }
            float v296;
            v296 = 0.0f;
            int v297;
            v297 = 0l;
            while (while_method_0(v297)){
                int v299;
                v299 = 0l;
                while (while_method_1(v299)){
                    assert("Tensor range check" && 0 <= v297 && v297 < 1l);
                    assert("Tensor range check" && 0 <= v299 && v299 < 4l);
                    int v301;
                    v301 = 4l * v297;
                    int v302;
                    v302 = v301 + v299;
                    float v303;
                    v303 = v284[v302];
                    float v304;
                    v304 = v296 + v303;
                    v296 = v304;
                    v299 += 1l ;
                }
                v297 += 1l ;
            }
            auto v305 = cooperative_groups::coalesced_threads();
            int v306;
            v306 = threadIdx.x;
            auto v307 = cooperative_groups::labeled_partition(v305,v306);
            Closure0 v308{};
            float v309;
            v309 = cooperative_groups::reduce(v307, v296, v308);
            int v310[4l];
            int v311;
            v311 = 0l;
            while (while_method_0(v311)){
                int v313;
                v313 = 0l;
                while (while_method_1(v313)){
                    assert("Tensor range check" && 0 <= v311 && v311 < 1l);
                    assert("Tensor range check" && 0 <= v313 && v313 < 4l);
                    int v315;
                    v315 = 4l * v311;
                    int v316;
                    v316 = v315 + v313;
                    bool v317;
                    v317 = v274[v316];
                    int v318;
                    if (v317){
                        v318 = 1l;
                    } else {
                        v318 = 0l;
                    }
                    assert("Tensor range check" && 0 <= v311 && v311 < 1l);
                    assert("Tensor range check" && 0 <= v313 && v313 < 4l);
                    v310[v316] = v318;
                    v313 += 1l ;
                }
                v311 += 1l ;
            }
            int v319;
            v319 = 0l;
            int v320;
            v320 = 0l;
            while (while_method_0(v320)){
                int v322;
                v322 = 0l;
                while (while_method_1(v322)){
                    assert("Tensor range check" && 0 <= v320 && v320 < 1l);
                    assert("Tensor range check" && 0 <= v322 && v322 < 4l);
                    int v324;
                    v324 = 4l * v320;
                    int v325;
                    v325 = v324 + v322;
                    int v326;
                    v326 = v310[v325];
                    int v327;
                    v327 = v319 + v326;
                    v319 = v327;
                    v322 += 1l ;
                }
                v320 += 1l ;
            }
            auto v328 = cooperative_groups::coalesced_threads();
            int v329;
            v329 = threadIdx.x;
            auto v330 = cooperative_groups::labeled_partition(v328,v329);
            Closure1 v331{};
            int v332;
            v332 = cooperative_groups::reduce(v330, v319, v331);
            float v333;
            v333 = (float)v332;
            float v334;
            v334 = 1.0f / v333;
            float v335[4l];
            int v336;
            v336 = 0l;
            while (while_method_0(v336)){
                int v338;
                v338 = 0l;
                while (while_method_1(v338)){
                    assert("Tensor range check" && 0 <= v336 && v336 < 1l);
                    assert("Tensor range check" && 0 <= v338 && v338 < 4l);
                    int v340;
                    v340 = 4l * v336;
                    int v341;
                    v341 = v340 + v338;
                    float v342;
                    v342 = v284[v341];
                    bool v343;
                    v343 = v274[v341];
                    bool v344;
                    v344 = v343 == false;
                    float v349;
                    if (v344){
                        v349 = 0.0f;
                    } else {
                        bool v345;
                        v345 = v309 == 0.0f;
                        bool v346;
                        v346 = v345 != true;
                        if (v346){
                            float v347;
                            v347 = v342 / v309;
                            v349 = v347;
                        } else {
                            v349 = v334;
                        }
                    }
                    assert("Tensor range check" && 0 <= v336 && v336 < 1l);
                    assert("Tensor range check" && 0 <= v338 && v338 < 4l);
                    v335[v341] = v349;
                    v338 += 1l ;
                }
                v336 += 1l ;
            }
            float v350[4l];
            int v351;
            v351 = 0l;
            while (while_method_0(v351)){
                int v353;
                v353 = 0l;
                while (while_method_1(v353)){
                    assert("Tensor range check" && 0 <= v351 && v351 < 1l);
                    assert("Tensor range check" && 0 <= v353 && v353 < 4l);
                    int v355;
                    v355 = 4l * v351;
                    int v356;
                    v356 = v355 + v353;
                    float v357;
                    v357 = v261[v356];
                    int v358;
                    v358 = v225[v356];
                    bool v359;
                    v359 = v211 == v358;
                    float v362;
                    if (v359){
                        float v360;
                        v360 = v212 - v357;
                        float v361;
                        v361 = v360 / v210;
                        v362 = v361;
                    } else {
                        v362 = 0.0f;
                    }
                    float v363;
                    v363 = v362 + v357;
                    assert("Tensor range check" && 0 <= v351 && v351 < 1l);
                    assert("Tensor range check" && 0 <= v353 && v353 < 4l);
                    v350[v356] = v363;
                    v353 += 1l ;
                }
                v351 += 1l ;
            }
            float v364[4l];
            int v365;
            v365 = 0l;
            while (while_method_0(v365)){
                int v367;
                v367 = 0l;
                while (while_method_1(v367)){
                    assert("Tensor range check" && 0 <= v365 && v365 < 1l);
                    assert("Tensor range check" && 0 <= v367 && v367 < 4l);
                    int v369;
                    v369 = 4l * v365;
                    int v370;
                    v370 = v369 + v367;
                    float v371;
                    v371 = v335[v370];
                    float v372;
                    v372 = v350[v370];
                    float v373;
                    v373 = v371 * v372;
                    assert("Tensor range check" && 0 <= v365 && v365 < 1l);
                    assert("Tensor range check" && 0 <= v367 && v367 < 4l);
                    v364[v370] = v373;
                    v367 += 1l ;
                }
                v365 += 1l ;
            }
            float v374;
            v374 = 0.0f;
            int v375;
            v375 = 0l;
            while (while_method_0(v375)){
                int v377;
                v377 = 0l;
                while (while_method_1(v377)){
                    assert("Tensor range check" && 0 <= v375 && v375 < 1l);
                    assert("Tensor range check" && 0 <= v377 && v377 < 4l);
                    int v379;
                    v379 = 4l * v375;
                    int v380;
                    v380 = v379 + v377;
                    float v381;
                    v381 = v364[v380];
                    float v382;
                    v382 = v374 + v381;
                    v374 = v382;
                    v377 += 1l ;
                }
                v375 += 1l ;
            }
            auto v383 = cooperative_groups::coalesced_threads();
            int v384;
            v384 = threadIdx.x;
            auto v385 = cooperative_groups::labeled_partition(v383,v384);
            float v386;
            v386 = cooperative_groups::reduce(v385, v374, v308);
            int v387;
            v387 = 0l;
            while (while_method_0(v387)){
                int v389;
                v389 = 0l;
                while (while_method_1(v389)){
                    assert("Tensor range check" && 0 <= v387 && v387 < 1l);
                    assert("Tensor range check" && 0 <= v389 && v389 < 4l);
                    int v391;
                    v391 = 4l * v387;
                    int v392;
                    v392 = v391 + v389;
                    float v393;
                    v393 = v350[v392];
                    int v394;
                    v394 = v225[v392];
                    float v395;
                    v395 = v393 - v386;
                    float v396;
                    v396 = v213 * v395;
                    assert("Tensor range check" && 0 <= v394 && v394 < 4l);
                    float * v397;
                    v397 = v214+v394;
                    float v399;
                    v399 = atomicAdd(v397,v396);
                    v389 += 1l ;
                }
                v387 += 1l ;
            }
            int v400;
            v400 = 0l;
            while (while_method_0(v400)){
                assert("Tensor range check" && 0 <= v400 && v400 < 1l);
                assert("Tensor range check" && 0 <= v400 && v400 < 1l);
                v400 += 1l ;
            }
            assert("Tensor range check" && 0 <= v207 && v207 < 256l);
            v186[v207] = v386;
            v196 += 1l ;
        }
        asm("barrier.cta.sync %0;" :: "r"(0l));
        assert("Tensor range check" && 0 <= v188 && v188 < 256l);
        float v402;
        v402 = v186[v188];
        asm("barrier.cta.sync %0;" :: "r"(0l));
        assert("Tensor range check" && 0 <= v84 && v84 < 2l);
        v72[v84] = v402;
    }
    int v403;
    v403 = threadIdx.x;
    int v404;
    v404 = blockIdx.x;
    int v405;
    v405 = v404 * 256l;
    int v406;
    v406 = v403 + v405;
    assert("Tensor range check" && 0 <= v406 && v406 < 6144l);
    int v407;
    v407 = 2l * v406;
    double * v408;
    v408 = v38+v407;
    double * v410;
    v410 = v40+v407;
    double * v412;
    v412 = v408+0l;
    double * v414;
    v414 = v410+0l;
    double * v416;
    v416 = v408+0l;
    double * v418;
    v418 = v410+0l;
    int v420;
    v420 = sizeof(double *);
    unsigned long long v421;
    v421 = (unsigned long long)v420;
    unsigned long long v422;
    v422 = 256ull * v421;
    unsigned long long v423;
    v423 = v422 + 16ull;
    unsigned long long v424;
    v424 = v423 - 1ull;
    unsigned long long v425;
    v425 = v424 % 16ull;
    unsigned long long v426;
    v426 = v424 - v425;
    unsigned long long v427;
    v427 = v426 + v422;
    unsigned long long v428;
    v428 = v427 + 16ull;
    unsigned long long v429;
    v429 = v428 - 1ull;
    unsigned long long v430;
    v430 = v429 % 16ull;
    unsigned long long v431;
    v431 = v429 - v430;
    unsigned long long v432;
    v432 = v431 + v422;
    unsigned long long v433;
    v433 = v432 + 16ull;
    unsigned long long v434;
    v434 = v433 - 1ull;
    unsigned long long v435;
    v435 = v434 % 16ull;
    unsigned long long v436;
    v436 = v434 - v435;
    unsigned long long v437;
    v437 = v436 + v422;
    bool v438;
    v438 = v437 <= 81920ull;
    bool v439;
    v439 = v438 == false;
    if (v439){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v438);
    } else {
    }
    extern __shared__ unsigned char v441[];
    bool v442;
    v442 = v437 <= v437;
    bool v443;
    v443 = v442 == false;
    if (v443){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v442);
    } else {
    }
    double * * v445;
    v445 = reinterpret_cast<double * *>(&v441[0ull]);
    double * * v447;
    v447 = reinterpret_cast<double * *>(&v441[v426]);
    double * * v449;
    v449 = reinterpret_cast<double * *>(&v441[v431]);
    double * * v451;
    v451 = reinterpret_cast<double * *>(&v441[v436]);
    int v453;
    v453 = threadIdx.x;
    assert("Tensor range check" && 0 <= v453 && v453 < 256l);
    v445[v453] = v412;
    v447[v453] = v414;
    v449[v453] = v416;
    v451[v453] = v418;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v454;
    v454 = 0l <= v453;
    bool v455;
    v455 = v454 == false;
    if (v455){
        assert("The index needs to be zero or positive." && v454);
    } else {
    }
    int v457;
    v457 = v453 % 1l;
    bool v458;
    v458 = v453 < 256l;
    bool v459;
    v459 = v458 == false;
    if (v459){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v458);
    } else {
    }
    assert("Tensor range check" && 0 <= v453 && v453 < 256l);
    int v461;
    v461 = 0l;
    while (while_method_0(v461)){
        bool v463;
        v463 = v454 && v458;
        bool v464;
        v464 = v463 == false;
        if (v464){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v463);
        } else {
        }
        bool v466;
        v466 = 0l <= v461;
        bool v468;
        if (v466){
            bool v467;
            v467 = v461 < 1l;
            v468 = v467;
        } else {
            v468 = false;
        }
        bool v469;
        v469 = v468 == false;
        if (v469){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v468);
        } else {
        }
        int v471;
        v471 = v461 * 256l;
        int v472;
        v472 = v471 + v453;
        assert("Tensor range check" && 0 <= v461 && v461 < 1l);
        int v473;
        v473 = 256l * v461;
        int v474;
        v474 = v473 + v453;
        double * v475;
        v475 = v445[v474];
        double * v476;
        v476 = v447[v474];
        double * v477;
        v477 = v449[v474];
        double * v478;
        v478 = v451[v474];
        int v479;
        v479 = blockIdx.x;
        int v480;
        v480 = v479 * 256l;
        int v481;
        v481 = v480 + v472;
        assert("Tensor range check" && 0 <= v457 && v457 < 1l);
        int v482;
        v482 = 2l * v457;
        double v483[2l];
        double v484[2l];
        int v485[2l];
        int v486;
        v486 = 0l;
        while (while_method_0(v486)){
            assert("Tensor range check" && 0 <= v486 && v486 < 1l);
            int v488;
            v488 = 2l * v486;
            assert("Tensor range check" && 0 <= v486 && v486 < 1l);
            int v489;
            v489 = v488 + v482;
            int4* v490;
            v490 = reinterpret_cast<int4*>(v475 + v489);
            int4* v491;
            v491 = reinterpret_cast<int4*>(v483 + v488);
            assert("Pointer alignment check" && (unsigned long long)(v490) % 2l == 0 && (unsigned long long)(v491) % 2l == 0);
            *v491 = *v490;
            int4* v492;
            v492 = reinterpret_cast<int4*>(v476 + v489);
            int4* v493;
            v493 = reinterpret_cast<int4*>(v484 + v488);
            assert("Pointer alignment check" && (unsigned long long)(v492) % 2l == 0 && (unsigned long long)(v493) % 2l == 0);
            *v493 = *v492;
            v486 += 1l ;
        }
        int v494;
        v494 = 0l;
        while (while_method_0(v494)){
            int v496;
            v496 = 0l;
            while (while_method_2(v496)){
                bool v498;
                v498 = 0l <= v496;
                bool v500;
                if (v498){
                    bool v499;
                    v499 = v496 < 2l;
                    v500 = v499;
                } else {
                    v500 = false;
                }
                bool v501;
                v501 = v500 == false;
                if (v501){
                    assert("The indices should be inside the range of the dimension." && v500);
                } else {
                }
                bool v503;
                v503 = 0l <= v457;
                bool v505;
                if (v503){
                    bool v504;
                    v504 = v457 < 1l;
                    v505 = v504;
                } else {
                    v505 = false;
                }
                bool v506;
                v506 = v505 == false;
                if (v506){
                    assert("The indices should be inside the range of the dimension." && v505);
                } else {
                }
                int v508;
                v508 = v457 * 2l;
                int v509;
                v509 = v496 + v508;
                bool v510;
                v510 = 0l <= v494;
                bool v512;
                if (v510){
                    bool v511;
                    v511 = v494 < 1l;
                    v512 = v511;
                } else {
                    v512 = false;
                }
                bool v513;
                v513 = v512 == false;
                if (v513){
                    assert("The indices should be inside the range of the dimension." && v512);
                } else {
                }
                int v515;
                v515 = v494 * 2l;
                int v516;
                v516 = v509 + v515;
                assert("Tensor range check" && 0 <= v494 && v494 < 1l);
                assert("Tensor range check" && 0 <= v496 && v496 < 2l);
                int v517;
                v517 = 2l * v494;
                int v518;
                v518 = v517 + v496;
                v485[v518] = v516;
                v496 += 1l ;
            }
            v494 += 1l ;
        }
        double v519[2l];
        double v520[2l];
        int v521;
        v521 = 0l;
        while (while_method_0(v521)){
            int v523;
            v523 = 0l;
            while (while_method_2(v523)){
                assert("Tensor range check" && 0 <= v521 && v521 < 1l);
                assert("Tensor range check" && 0 <= v523 && v523 < 2l);
                int v525;
                v525 = 2l * v521;
                int v526;
                v526 = v525 + v523;
                double v527;
                v527 = v483[v526];
                double v528;
                v528 = v484[v526];
                assert("Tensor range check" && 0 <= v521 && v521 < 1l);
                assert("Tensor range check" && 0 <= v523 && v523 < 2l);
                v519[v526] = 0.0;
                v520[v526] = 0.0;
                v523 += 1l ;
            }
            v521 += 1l ;
        }
        int v529;
        v529 = 0l;
        while (while_method_0(v529)){
            assert("Tensor range check" && 0 <= v529 && v529 < 1l);
            int v531;
            v531 = 2l * v529;
            int v532;
            v532 = v531 + v482;
            assert("Tensor range check" && 0 <= v529 && v529 < 1l);
            int4* v533;
            v533 = reinterpret_cast<int4*>(v519 + v531);
            int4* v534;
            v534 = reinterpret_cast<int4*>(v477 + v532);
            assert("Pointer alignment check" && (unsigned long long)(v533) % 2l == 0 && (unsigned long long)(v534) % 2l == 0);
            *v534 = *v533;
            int4* v535;
            v535 = reinterpret_cast<int4*>(v520 + v531);
            int4* v536;
            v536 = reinterpret_cast<int4*>(v478 + v532);
            assert("Pointer alignment check" && (unsigned long long)(v535) % 2l == 0 && (unsigned long long)(v536) % 2l == 0);
            *v536 = *v535;
            v529 += 1l ;
        }
        assert("Tensor range check" && 0 <= v472 && v472 < 256l);
        v461 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v453 && v453 < 256l);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    assert("Tensor range check" && 0 <= v406 && v406 < 6144l);
    v42[v406] = 0l;
    v2.sync() ;
    int v537;
    v537 = threadIdx.x;
    int v538;
    v538 = blockIdx.x;
    int v539;
    v539 = v538 * 256l;
    int v540;
    v540 = v537 + v539;
    bool v541;
    v541 = v540 == 0l;
    if (v541){
        int v542;
        v542 = 0l;
        int v543;
        v543 = 4l;
        int v544;
        v544 = int_range_3(v543, v542, v9);
        v10[0l] = v544;
    } else {
    }
    __syncwarp();
    int v545;
    v545 = threadIdx.x;
    bool v546;
    v546 = 0l <= v545;
    bool v547;
    v547 = v546 == false;
    if (v547){
        assert("The index needs to be zero or positive." && v546);
    } else {
    }
    int v549;
    v549 = v545 % 1l;
    int v550;
    v550 = v545 % 256l;
    int v551;
    v551 = v545 / 256l;
    bool v552;
    v552 = v551 < 1l;
    bool v553;
    v553 = v552 == false;
    if (v553){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v552);
    } else {
    }
    assert("Tensor range check" && 0 <= v551 && v551 < 1l);
    assert("Tensor range check" && 0 <= v550 && v550 < 256l);
    assert("Tensor range check" && 0 <= v549 && v549 < 1l);
    int v555;
    v555 = 4l * v549;
    int v556;
    v556 = 4l * v550;
    int v557;
    v557 = v556 + v555;
    int v558;
    v558 = 16384l * v551;
    int v559;
    v559 = v558 + v557;
    assert("Tensor range check" && 0 <= v551 && v551 < 1l);
    assert("Tensor range check" && 0 <= v550 && v550 < 256l);
    assert("Tensor range check" && 0 <= v549 && v549 < 1l);
    int v560;
    v560 = blockIdx.x;
    int v561;
    v561 = v560;
    while (while_method_4(v561)){
        bool v563;
        v563 = 0l <= v561;
        bool v564;
        v564 = v563 == false;
        if (v564){
            assert("The index needs to be zero or positive." && v563);
        } else {
        }
        int v566;
        v566 = v561 % 16l;
        int v567;
        v567 = v561 / 16l;
        bool v568;
        v568 = v567 < 4l;
        bool v569;
        v569 = v568 == false;
        if (v569){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v568);
        } else {
        }
        assert("Tensor range check" && 0 <= v567 && v567 < 4l);
        assert("Tensor range check" && 0 <= v566 && v566 < 16l);
        int v571;
        v571 = 1024l * v566;
        int v572;
        v572 = v571 + v559;
        int v573;
        v573 = 16384l * v567;
        int v574;
        v574 = v573 + v572;
        float v575[4l];
        float v576[4l];
        float v577[4l];
        float v578[4l];
        float v579[4l];
        float v580[4l];
        float v581[4l];
        int v582[4l];
        int v583;
        v583 = 0l;
        while (while_method_0(v583)){
            assert("Tensor range check" && 0 <= v583 && v583 < 1l);
            int v585;
            v585 = 4l * v583;
            assert("Tensor range check" && 0 <= v583 && v583 < 1l);
            int v586;
            v586 = v585 + v574;
            int4* v587;
            v587 = reinterpret_cast<int4*>(v12 + v586);
            int4* v588;
            v588 = reinterpret_cast<int4*>(v575 + v585);
            assert("Pointer alignment check" && (unsigned long long)(v587) % 4l == 0 && (unsigned long long)(v588) % 4l == 0);
            *v588 = *v587;
            int4* v589;
            v589 = reinterpret_cast<int4*>(v14 + v586);
            int4* v590;
            v590 = reinterpret_cast<int4*>(v576 + v585);
            assert("Pointer alignment check" && (unsigned long long)(v589) % 4l == 0 && (unsigned long long)(v590) % 4l == 0);
            *v590 = *v589;
            int4* v591;
            v591 = reinterpret_cast<int4*>(v16 + v586);
            int4* v592;
            v592 = reinterpret_cast<int4*>(v577 + v585);
            assert("Pointer alignment check" && (unsigned long long)(v591) % 4l == 0 && (unsigned long long)(v592) % 4l == 0);
            *v592 = *v591;
            int4* v593;
            v593 = reinterpret_cast<int4*>(v18 + v586);
            int4* v594;
            v594 = reinterpret_cast<int4*>(v578 + v585);
            assert("Pointer alignment check" && (unsigned long long)(v593) % 4l == 0 && (unsigned long long)(v594) % 4l == 0);
            *v594 = *v593;
            int4* v595;
            v595 = reinterpret_cast<int4*>(v20 + v586);
            int4* v596;
            v596 = reinterpret_cast<int4*>(v579 + v585);
            assert("Pointer alignment check" && (unsigned long long)(v595) % 4l == 0 && (unsigned long long)(v596) % 4l == 0);
            *v596 = *v595;
            int4* v597;
            v597 = reinterpret_cast<int4*>(v22 + v586);
            int4* v598;
            v598 = reinterpret_cast<int4*>(v580 + v585);
            assert("Pointer alignment check" && (unsigned long long)(v597) % 4l == 0 && (unsigned long long)(v598) % 4l == 0);
            *v598 = *v597;
            int4* v599;
            v599 = reinterpret_cast<int4*>(v24 + v586);
            int4* v600;
            v600 = reinterpret_cast<int4*>(v581 + v585);
            assert("Pointer alignment check" && (unsigned long long)(v599) % 4l == 0 && (unsigned long long)(v600) % 4l == 0);
            *v600 = *v599;
            v583 += 1l ;
        }
        int v601;
        v601 = 0l;
        while (while_method_0(v601)){
            int v603;
            v603 = 0l;
            while (while_method_1(v603)){
                bool v605;
                v605 = 0l <= v603;
                bool v607;
                if (v605){
                    bool v606;
                    v606 = v603 < 4l;
                    v607 = v606;
                } else {
                    v607 = false;
                }
                bool v608;
                v608 = v607 == false;
                if (v608){
                    assert("The indices should be inside the range of the dimension." && v607);
                } else {
                }
                bool v610;
                v610 = 0l <= v549;
                bool v612;
                if (v610){
                    bool v611;
                    v611 = v549 < 1l;
                    v612 = v611;
                } else {
                    v612 = false;
                }
                bool v613;
                v613 = v612 == false;
                if (v613){
                    assert("The indices should be inside the range of the dimension." && v612);
                } else {
                }
                int v615;
                v615 = v549 * 4l;
                int v616;
                v616 = v603 + v615;
                bool v617;
                v617 = 0l <= v601;
                bool v619;
                if (v617){
                    bool v618;
                    v618 = v601 < 1l;
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
                v622 = v601 * 4l;
                int v623;
                v623 = v616 + v622;
                assert("Tensor range check" && 0 <= v601 && v601 < 1l);
                assert("Tensor range check" && 0 <= v603 && v603 < 4l);
                int v624;
                v624 = 4l * v601;
                int v625;
                v625 = v624 + v603;
                v582[v625] = v623;
                v603 += 1l ;
            }
            v601 += 1l ;
        }
        bool v626;
        v626 = 0l <= v551;
        bool v627;
        v627 = v626 && v552;
        bool v628;
        v628 = v627 == false;
        if (v628){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v627);
        } else {
        }
        bool v630;
        v630 = 0l <= v550;
        bool v632;
        if (v630){
            bool v631;
            v631 = v550 < 256l;
            v632 = v631;
        } else {
            v632 = false;
        }
        bool v633;
        v633 = v632 == false;
        if (v633){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v632);
        } else {
        }
        bool v635;
        v635 = 0l <= v567;
        bool v636;
        v636 = v635 && v568;
        bool v637;
        v637 = v636 == false;
        if (v637){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v636);
        } else {
        }
        bool v639;
        v639 = 0l <= v566;
        bool v641;
        if (v639){
            bool v640;
            v640 = v566 < 16l;
            v641 = v640;
        } else {
            v641 = false;
        }
        bool v642;
        v642 = v641 == false;
        if (v642){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v641);
        } else {
        }
        int v644;
        v644 = v566 * 256l;
        int v645;
        v645 = v567 + v551;
        int v646;
        v646 = v644 + v550;
        bool v647[4l];
        int v648;
        v648 = 0l;
        while (while_method_0(v648)){
            int v650;
            v650 = 0l;
            while (while_method_1(v650)){
                assert("Tensor range check" && 0 <= v648 && v648 < 1l);
                assert("Tensor range check" && 0 <= v650 && v650 < 4l);
                int v652;
                v652 = 4l * v648;
                int v653;
                v653 = v652 + v650;
                float v654;
                v654 = v577[v653];
                bool v655;
                v655 = v654 == 0.0f;
                bool v656;
                v656 = v655 != true;
                assert("Tensor range check" && 0 <= v648 && v648 < 1l);
                assert("Tensor range check" && 0 <= v650 && v650 < 4l);
                v647[v653] = v656;
                v650 += 1l ;
            }
            v648 += 1l ;
        }
        bool v657;
        v657 = false;
        int v658;
        v658 = 0l;
        while (while_method_0(v658)){
            int v660;
            v660 = 0l;
            while (while_method_1(v660)){
                assert("Tensor range check" && 0 <= v658 && v658 < 1l);
                assert("Tensor range check" && 0 <= v660 && v660 < 4l);
                int v662;
                v662 = 4l * v658;
                int v663;
                v663 = v662 + v660;
                bool v664;
                v664 = v647[v663];
                bool v665;
                v665 = v657 || v664;
                v657 = v665;
                v660 += 1l ;
            }
            v658 += 1l ;
        }
        auto v666 = cooperative_groups::coalesced_threads();
        int v667;
        v667 = threadIdx.x;
        auto v668 = cooperative_groups::labeled_partition(v666,v667);
        Closure7 v669{};
        bool v670;
        v670 = cooperative_groups::reduce(v668, v657, v669);
        if (v670){
            float v671[4l];
            int v672;
            v672 = 0l;
            while (while_method_0(v672)){
                int v674;
                v674 = 0l;
                while (while_method_1(v674)){
                    assert("Tensor range check" && 0 <= v672 && v672 < 1l);
                    assert("Tensor range check" && 0 <= v674 && v674 < 4l);
                    int v676;
                    v676 = 4l * v672;
                    int v677;
                    v677 = v676 + v674;
                    float v678;
                    v678 = v576[v677];
                    float v679;
                    v679 = v577[v677];
                    float v680;
                    v680 = v678 + v679;
                    bool v681;
                    v681 = 0.0f >= v680;
                    float v682;
                    if (v681){
                        v682 = 0.0f;
                    } else {
                        v682 = v680;
                    }
                    assert("Tensor range check" && 0 <= v672 && v672 < 1l);
                    assert("Tensor range check" && 0 <= v674 && v674 < 4l);
                    v671[v677] = v682;
                    v674 += 1l ;
                }
                v672 += 1l ;
            }
            float v683[4l];
            int v684;
            v684 = 0l;
            while (while_method_0(v684)){
                int v686;
                v686 = 0l;
                while (while_method_1(v686)){
                    assert("Tensor range check" && 0 <= v684 && v684 < 1l);
                    assert("Tensor range check" && 0 <= v686 && v686 < 4l);
                    int v688;
                    v688 = 4l * v684;
                    int v689;
                    v689 = v688 + v686;
                    float v690;
                    v690 = v671[v689];
                    bool v691;
                    v691 = 0.0f >= v690;
                    float v692;
                    if (v691){
                        v692 = 0.0f;
                    } else {
                        v692 = v690;
                    }
                    assert("Tensor range check" && 0 <= v684 && v684 < 1l);
                    assert("Tensor range check" && 0 <= v686 && v686 < 4l);
                    v683[v689] = v692;
                    v686 += 1l ;
                }
                v684 += 1l ;
            }
            float v693;
            v693 = 0.0f;
            int v694;
            v694 = 0l;
            while (while_method_0(v694)){
                int v696;
                v696 = 0l;
                while (while_method_1(v696)){
                    assert("Tensor range check" && 0 <= v694 && v694 < 1l);
                    assert("Tensor range check" && 0 <= v696 && v696 < 4l);
                    int v698;
                    v698 = 4l * v694;
                    int v699;
                    v699 = v698 + v696;
                    float v700;
                    v700 = v683[v699];
                    float v701;
                    v701 = v693 + v700;
                    v693 = v701;
                    v696 += 1l ;
                }
                v694 += 1l ;
            }
            auto v702 = cooperative_groups::coalesced_threads();
            int v703;
            v703 = threadIdx.x;
            auto v704 = cooperative_groups::labeled_partition(v702,v703);
            Closure0 v705{};
            float v706;
            v706 = cooperative_groups::reduce(v704, v693, v705);
            float v707[4l];
            int v708;
            v708 = 0l;
            while (while_method_0(v708)){
                int v710;
                v710 = 0l;
                while (while_method_1(v710)){
                    assert("Tensor range check" && 0 <= v708 && v708 < 1l);
                    assert("Tensor range check" && 0 <= v710 && v710 < 4l);
                    int v712;
                    v712 = 4l * v708;
                    int v713;
                    v713 = v712 + v710;
                    float v714;
                    v714 = v683[v713];
                    bool v715;
                    v715 = v706 == 0.0f;
                    bool v716;
                    v716 = v715 != true;
                    float v718;
                    if (v716){
                        float v717;
                        v717 = v714 / v706;
                        v718 = v717;
                    } else {
                        v718 = 0.25f;
                    }
                    assert("Tensor range check" && 0 <= v708 && v708 < 1l);
                    assert("Tensor range check" && 0 <= v710 && v710 < 4l);
                    v707[v713] = v718;
                    v710 += 1l ;
                }
                v708 += 1l ;
            }
            float v719[4l];
            int v720;
            v720 = 0l;
            while (while_method_0(v720)){
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
                    v726 = v575[v725];
                    float v727;
                    v727 = v707[v725];
                    float v728;
                    v728 = v726 + v727;
                    assert("Tensor range check" && 0 <= v720 && v720 < 1l);
                    assert("Tensor range check" && 0 <= v722 && v722 < 4l);
                    v719[v725] = v728;
                    v722 += 1l ;
                }
                v720 += 1l ;
            }
            float v729[4l];
            int v730;
            v730 = 0l;
            while (while_method_0(v730)){
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
                    v737 = -v736;
                    bool v738;
                    v738 = v736 >= v737;
                    float v739;
                    if (v738){
                        v739 = v736;
                    } else {
                        v739 = v737;
                    }
                    assert("Tensor range check" && 0 <= v730 && v730 < 1l);
                    assert("Tensor range check" && 0 <= v732 && v732 < 4l);
                    v729[v735] = v739;
                    v732 += 1l ;
                }
                v730 += 1l ;
            }
            float v740;
            v740 = 0.0f;
            int v741;
            v741 = 0l;
            while (while_method_0(v741)){
                int v743;
                v743 = 0l;
                while (while_method_1(v743)){
                    assert("Tensor range check" && 0 <= v741 && v741 < 1l);
                    assert("Tensor range check" && 0 <= v743 && v743 < 4l);
                    int v745;
                    v745 = 4l * v741;
                    int v746;
                    v746 = v745 + v743;
                    float v747;
                    v747 = v729[v746];
                    float v748;
                    v748 = v740 + v747;
                    v740 = v748;
                    v743 += 1l ;
                }
                v741 += 1l ;
            }
            auto v749 = cooperative_groups::coalesced_threads();
            int v750;
            v750 = threadIdx.x;
            auto v751 = cooperative_groups::labeled_partition(v749,v750);
            float v752;
            v752 = cooperative_groups::reduce(v751, v740, v705);
            bool v753;
            v753 = v752 > 100.0f;
            float v755;
            if (v753){
                float v754;
                v754 = 100.0f / v752;
                v755 = v754;
            } else {
                v755 = 1.0f;
            }
            float v756[4l];
            int v757;
            v757 = 0l;
            while (while_method_0(v757)){
                int v759;
                v759 = 0l;
                while (while_method_1(v759)){
                    assert("Tensor range check" && 0 <= v757 && v757 < 1l);
                    assert("Tensor range check" && 0 <= v759 && v759 < 4l);
                    int v761;
                    v761 = 4l * v757;
                    int v762;
                    v762 = v761 + v759;
                    float v763;
                    v763 = v729[v762];
                    float v764;
                    v764 = v755 * v763;
                    assert("Tensor range check" && 0 <= v757 && v757 < 1l);
                    assert("Tensor range check" && 0 <= v759 && v759 < 4l);
                    v756[v762] = v764;
                    v759 += 1l ;
                }
                v757 += 1l ;
            }
            float v765[4l];
            float v766[4l];
            int v767;
            v767 = 0l;
            while (while_method_0(v767)){
                int v769;
                v769 = 0l;
                while (while_method_1(v769)){
                    assert("Tensor range check" && 0 <= v767 && v767 < 1l);
                    assert("Tensor range check" && 0 <= v769 && v769 < 4l);
                    int v771;
                    v771 = 4l * v767;
                    int v772;
                    v772 = v771 + v769;
                    float v773;
                    v773 = v575[v772];
                    float v774;
                    v774 = v576[v772];
                    float v775;
                    v775 = v577[v772];
                    float v776;
                    v776 = v578[v772];
                    float v777;
                    v777 = v579[v772];
                    float v778;
                    v778 = v580[v772];
                    float v779;
                    v779 = v581[v772];
                    float v780;
                    v780 = v776 + v778;
                    float v781;
                    v781 = v777 + v779;
                    assert("Tensor range check" && 0 <= v767 && v767 < 1l);
                    assert("Tensor range check" && 0 <= v769 && v769 < 4l);
                    v765[v772] = v780;
                    v766[v772] = v781;
                    v769 += 1l ;
                }
                v767 += 1l ;
            }
            int v782;
            v782 = 0l;
            while (while_method_0(v782)){
                int v784;
                v784 = 0l;
                while (while_method_1(v784)){
                    assert("Tensor range check" && 0 <= v782 && v782 < 1l);
                    assert("Tensor range check" && 0 <= v784 && v784 < 4l);
                    int v786;
                    v786 = 4l * v782;
                    int v787;
                    v787 = v786 + v784;
                    float v788;
                    v788 = v756[v787];
                    float v789;
                    v789 = v671[v787];
                    float v790;
                    v790 = v765[v787];
                    float v791;
                    v791 = v766[v787];
                    assert("Tensor range check" && 0 <= v782 && v782 < 1l);
                    assert("Tensor range check" && 0 <= v784 && v784 < 4l);
                    v575[v787] = v788;
                    v576[v787] = v789;
                    v577[v787] = 0.0f;
                    v578[v787] = v790;
                    v579[v787] = v791;
                    v580[v787] = 0.0f;
                    v581[v787] = 0.0f;
                    v784 += 1l ;
                }
                v782 += 1l ;
            }
        } else {
        }
        assert("Tensor range check" && 0 <= v567 && v567 < 4l);
        assert("Tensor range check" && 0 <= v566 && v566 < 16l);
        int v792;
        v792 = 0l;
        while (while_method_0(v792)){
            assert("Tensor range check" && 0 <= v792 && v792 < 1l);
            int v794;
            v794 = 4l * v792;
            int v795;
            v795 = v794 + v574;
            assert("Tensor range check" && 0 <= v792 && v792 < 1l);
            int4* v796;
            v796 = reinterpret_cast<int4*>(v575 + v794);
            int4* v797;
            v797 = reinterpret_cast<int4*>(v12 + v795);
            assert("Pointer alignment check" && (unsigned long long)(v796) % 4l == 0 && (unsigned long long)(v797) % 4l == 0);
            *v797 = *v796;
            int4* v798;
            v798 = reinterpret_cast<int4*>(v576 + v794);
            int4* v799;
            v799 = reinterpret_cast<int4*>(v14 + v795);
            assert("Pointer alignment check" && (unsigned long long)(v798) % 4l == 0 && (unsigned long long)(v799) % 4l == 0);
            *v799 = *v798;
            int4* v800;
            v800 = reinterpret_cast<int4*>(v577 + v794);
            int4* v801;
            v801 = reinterpret_cast<int4*>(v16 + v795);
            assert("Pointer alignment check" && (unsigned long long)(v800) % 4l == 0 && (unsigned long long)(v801) % 4l == 0);
            *v801 = *v800;
            int4* v802;
            v802 = reinterpret_cast<int4*>(v578 + v794);
            int4* v803;
            v803 = reinterpret_cast<int4*>(v18 + v795);
            assert("Pointer alignment check" && (unsigned long long)(v802) % 4l == 0 && (unsigned long long)(v803) % 4l == 0);
            *v803 = *v802;
            int4* v804;
            v804 = reinterpret_cast<int4*>(v579 + v794);
            int4* v805;
            v805 = reinterpret_cast<int4*>(v20 + v795);
            assert("Pointer alignment check" && (unsigned long long)(v804) % 4l == 0 && (unsigned long long)(v805) % 4l == 0);
            *v805 = *v804;
            int4* v806;
            v806 = reinterpret_cast<int4*>(v580 + v794);
            int4* v807;
            v807 = reinterpret_cast<int4*>(v22 + v795);
            assert("Pointer alignment check" && (unsigned long long)(v806) % 4l == 0 && (unsigned long long)(v807) % 4l == 0);
            *v807 = *v806;
            int4* v808;
            v808 = reinterpret_cast<int4*>(v581 + v794);
            int4* v809;
            v809 = reinterpret_cast<int4*>(v24 + v795);
            assert("Pointer alignment check" && (unsigned long long)(v808) % 4l == 0 && (unsigned long long)(v809) % 4l == 0);
            *v809 = *v808;
            v792 += 1l ;
        }
        v561 += 24l ;
    }
    v2.sync() ;
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
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

options = []
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=256')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def main_body():
    v0 = threadIdx.x
    v1 = blockIdx.x
    v2 = v1 * 256
    del v1
    v3 = v0 + v2
    del v0, v2, v3
    v4 = cp.empty(2719760,dtype=cp.uint8)
    v5 = cp.empty(18874368,dtype=cp.uint8)
    v7 = v4[0:0+4*1].view(cp.int32)
    v9 = v4[16:16+4*65536].view(cp.float32)
    v11 = v4[262160:262160+4*65536].view(cp.float32)
    v13 = v4[524304:524304+4*65536].view(cp.float32)
    v15 = v4[786448:786448+4*65536].view(cp.float32)
    v17 = v4[1048592:1048592+4*65536].view(cp.float32)
    v19 = v4[1310736:1310736+4*65536].view(cp.float32)
    v21 = v4[1572880:1572880+4*65536].view(cp.float32)
    v7[:] = 0
    del v7
    v9[:] = 0
    del v9
    v11[:] = 0
    del v11
    v13[:] = 0
    del v13
    v15[:] = 0
    del v15
    v17[:] = 0
    del v17
    v19[:] = 0
    del v19
    v21[:] = 0
    del v21
    v23 = v4[1835024:1835024+8*49152].view(cp.float64)
    v25 = v4[2228240:2228240+8*49152].view(cp.float64)
    v27 = v4[2621456:2621456+4*24576].view(cp.int32)
    v23[:] = 0
    del v23
    v25[:] = 0
    del v25
    v27[:] = 0
    del v27
    v28 = cp.cuda.Device().attributes['MultiProcessorCount']
    v29 = v28 == 24
    del v28
    v30 = v29 == False
    if v30:
        v31 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v29, v31
        del v31
    else:
        pass
    del v29, v30
    v32 = 0
    v33 = raw_module.get_function(f"entry{v32}")
    del v32
    v33.max_dynamic_shared_size_bytes = 81920 
    v33((24,),(256,),(v5, v4),shared_mem=81920)
    del v4, v5, v33
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
