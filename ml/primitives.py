kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <curand_kernel.h>
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
struct Tuple2;
struct Tuple3;
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
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure4 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
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
struct Closure5 {
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
struct Closure6 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple1{v0, v1};
        } else {
            return Tuple1{v2, v3};
        }
    }
};
struct Tuple3 {
    int v0;
    bool v1;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure7 {
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
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 2048;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
extern "C" __global__ void entry0(int * v0, float * v1, float * v2, int * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int * v9, int * v10, int * v11, int * v12, int * v13, float * v14, int * v15, float * v16, int * v17) {
    float v18;
    v18 = 0.0f;
    int v19;
    v19 = threadIdx.x;
    int v20;
    v20 = v19;
    while (while_method_0(v20)){
        bool v22;
        v22 = 0 <= v20;
        bool v23;
        v23 = v22 == false;
        if (v23){
            assert("The index needs to be zero or positive." && v22);
        } else {
        }
        int v25;
        v25 = v20 % 32;
        int v26;
        v26 = v20 / 32;
        bool v27;
        v27 = v26 < 64;
        bool v28;
        v28 = v27 == false;
        if (v28){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v27);
        } else {
        }
        assert("Tensor range check" && 0 <= v26 && v26 < 64);
        assert("Tensor range check" && 0 <= v25 && v25 < 32);
        int v30;
        v30 = 4 * v25;
        int v31;
        v31 = 128 * v26;
        int v32;
        v32 = v31 + v30;
        float v33[4];
        int4* v34;
        v34 = reinterpret_cast<int4*>(v1 + v32);
        int4* v35;
        v35 = reinterpret_cast<int4*>(v33 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v34) % 16 == 0 && reinterpret_cast<unsigned long long>(v35) % 16 == 0);
        *v35 = *v34;
        int v36; float v37;
        Tuple0 tmp0 = Tuple0{0, v18};
        v36 = tmp0.v0; v37 = tmp0.v1;
        while (while_method_1(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 4);
            float v39;
            v39 = v33[v36];
            float v40;
            v40 = v37 + v39;
            v37 = v40;
            v36 += 1 ;
        }
        v18 = v37;
        v20 += 256 ;
    }
    __syncwarp();
    auto v41 = cooperative_groups::coalesced_threads();
    Closure0 v42{};
    float v43;
    v43 = cooperative_groups::reduce(v41, v18, v42);
    int v44;
    v44 = threadIdx.x;
    int v45;
    v45 = v44 / 32;
    extern __shared__ unsigned char v46[];
    float * v47;
    v47 = reinterpret_cast<float *>(&v46[0ull]);
    assert("Tensor range check" && 0 <= v45 && v45 < 8);
    v47[v45] = v43;
    __syncthreads();
    int v49;
    v49 = threadIdx.x;
    int v50;
    v50 = v49 % 32;
    bool v51;
    v51 = v50 < 8;
    float v53;
    if (v51){
        assert("Tensor range check" && 0 <= v50 && v50 < 8);
        float v52;
        v52 = v47[v50];
        v53 = v52;
    } else {
        v53 = 0.0f;
    }
    __syncthreads();
    auto v54 = cooperative_groups::coalesced_threads();
    float v55;
    v55 = cooperative_groups::reduce(v54, v53, v42);
    v2[0] = v55;
    int v56;
    v56 = threadIdx.x;
    bool v57;
    v57 = 0 <= v56;
    bool v58;
    v58 = v57 == false;
    if (v58){
        assert("The index needs to be zero or positive." && v57);
    } else {
    }
    int v60;
    v60 = v56 % 32;
    int v61;
    v61 = v56 / 32;
    bool v62;
    v62 = v61 < 8;
    bool v63;
    v63 = v62 == false;
    if (v63){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v62);
    } else {
    }
    assert("Tensor range check" && 0 <= v61 && v61 < 8);
    assert("Tensor range check" && 0 <= v60 && v60 < 32);
    int v65;
    v65 = 4 * v60;
    int v66;
    v66 = 128 * v61;
    int v67;
    v67 = v66 + v65;
    assert("Tensor range check" && 0 <= v61 && v61 < 8);
    assert("Tensor range check" && 0 <= v60 && v60 < 32);
    int v68;
    v68 = 0;
    while (while_method_2(v68)){
        assert("Tensor range check" && 0 <= v68 && v68 < 8);
        int v70;
        v70 = 1024 * v68;
        int v71;
        v71 = v70 + v67;
        int v72[4];
        int v73[4];
        int v74;
        v74 = 0;
        while (while_method_3(v74)){
            assert("Tensor range check" && 0 <= v74 && v74 < 1);
            int v76;
            v76 = 4 * v74;
            assert("Tensor range check" && 0 <= v74 && v74 < 1);
            int v77;
            v77 = 128 * v74;
            int v78;
            v78 = v77 + v71;
            int4* v79;
            v79 = reinterpret_cast<int4*>(v0 + v78);
            int4* v80;
            v80 = reinterpret_cast<int4*>(v72 + v76);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v79) % 16 == 0 && reinterpret_cast<unsigned long long>(v80) % 16 == 0);
            *v80 = *v79;
            v74 += 1 ;
        }
        int v81;
        v81 = 0;
        while (while_method_3(v81)){
            int v83;
            v83 = 0;
            while (while_method_1(v83)){
                bool v85;
                v85 = 0 <= v83;
                bool v87;
                if (v85){
                    bool v86;
                    v86 = v83 < 4;
                    v87 = v86;
                } else {
                    v87 = false;
                }
                bool v88;
                v88 = v87 == false;
                if (v88){
                    assert("The indices should be inside the range of the dimension." && v87);
                } else {
                }
                bool v90;
                v90 = 0 <= v60;
                bool v92;
                if (v90){
                    bool v91;
                    v91 = v60 < 32;
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
                v95 = v60 * 4;
                int v96;
                v96 = v83 + v95;
                bool v97;
                v97 = 0 <= v81;
                bool v99;
                if (v97){
                    bool v98;
                    v98 = v81 < 1;
                    v99 = v98;
                } else {
                    v99 = false;
                }
                bool v100;
                v100 = v99 == false;
                if (v100){
                    assert("The indices should be inside the range of the dimension." && v99);
                } else {
                }
                int v102;
                v102 = v81 * 128;
                int v103;
                v103 = v96 + v102;
                assert("Tensor range check" && 0 <= v81 && v81 < 1);
                assert("Tensor range check" && 0 <= v83 && v83 < 4);
                int v104;
                v104 = 4 * v81;
                int v105;
                v105 = v104 + v83;
                v73[v105] = v103;
                v83 += 1 ;
            }
            v81 += 1 ;
        }
        bool v106;
        v106 = 0 <= v61;
        bool v107;
        v107 = v106 && v62;
        bool v108;
        v108 = v107 == false;
        if (v108){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v107);
        } else {
        }
        bool v110;
        v110 = 0 <= v68;
        bool v112;
        if (v110){
            bool v111;
            v111 = v68 < 8;
            v112 = v111;
        } else {
            v112 = false;
        }
        bool v113;
        v113 = v112 == false;
        if (v113){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v112);
        } else {
        }
        int v115;
        v115 = v68 * 8;
        int v116;
        v116 = v115 + v61;
        assert("Tensor range check" && 0 <= v68 && v68 < 8);
        int v117;
        v117 = 0;
        while (while_method_3(v117)){
            assert("Tensor range check" && 0 <= v117 && v117 < 1);
            int v119;
            v119 = 128 * v117;
            int v120;
            v120 = v119 + v71;
            assert("Tensor range check" && 0 <= v117 && v117 < 1);
            int v121;
            v121 = 4 * v117;
            int4* v122;
            v122 = reinterpret_cast<int4*>(v72 + v121);
            int4* v123;
            v123 = reinterpret_cast<int4*>(v3 + v120);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v122) % 16 == 0 && reinterpret_cast<unsigned long long>(v123) % 16 == 0);
            *v123 = *v122;
            v117 += 1 ;
        }
        v68 += 1 ;
    }
    __syncthreads();
    int v124;
    v124 = threadIdx.x;
    bool v125;
    v125 = 0 <= v124;
    bool v126;
    v126 = v125 == false;
    if (v126){
        assert("The index needs to be zero or positive." && v125);
    } else {
    }
    int v128;
    v128 = v124 % 32;
    int v129;
    v129 = v124 / 32;
    bool v130;
    v130 = v129 < 8;
    bool v131;
    v131 = v130 == false;
    if (v131){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v130);
    } else {
    }
    assert("Tensor range check" && 0 <= v129 && v129 < 8);
    assert("Tensor range check" && 0 <= v128 && v128 < 32);
    int v133;
    v133 = 4 * v128;
    int v134;
    v134 = 128 * v129;
    int v135;
    v135 = v134 + v133;
    assert("Tensor range check" && 0 <= v129 && v129 < 8);
    assert("Tensor range check" && 0 <= v128 && v128 < 32);
    int v136;
    v136 = 0;
    while (while_method_2(v136)){
        assert("Tensor range check" && 0 <= v136 && v136 < 8);
        int v138;
        v138 = 1024 * v136;
        int v139;
        v139 = v138 + v135;
        float v140[4];
        int v141[4];
        int v142;
        v142 = 0;
        while (while_method_3(v142)){
            assert("Tensor range check" && 0 <= v142 && v142 < 1);
            int v144;
            v144 = 4 * v142;
            assert("Tensor range check" && 0 <= v142 && v142 < 1);
            int v145;
            v145 = 128 * v142;
            int v146;
            v146 = v145 + v139;
            int4* v147;
            v147 = reinterpret_cast<int4*>(v1 + v146);
            int4* v148;
            v148 = reinterpret_cast<int4*>(v140 + v144);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v147) % 16 == 0 && reinterpret_cast<unsigned long long>(v148) % 16 == 0);
            *v148 = *v147;
            v142 += 1 ;
        }
        int v149;
        v149 = 0;
        while (while_method_3(v149)){
            int v151;
            v151 = 0;
            while (while_method_1(v151)){
                bool v153;
                v153 = 0 <= v151;
                bool v155;
                if (v153){
                    bool v154;
                    v154 = v151 < 4;
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
                bool v158;
                v158 = 0 <= v128;
                bool v160;
                if (v158){
                    bool v159;
                    v159 = v128 < 32;
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
                v163 = v128 * 4;
                int v164;
                v164 = v151 + v163;
                bool v165;
                v165 = 0 <= v149;
                bool v167;
                if (v165){
                    bool v166;
                    v166 = v149 < 1;
                    v167 = v166;
                } else {
                    v167 = false;
                }
                bool v168;
                v168 = v167 == false;
                if (v168){
                    assert("The indices should be inside the range of the dimension." && v167);
                } else {
                }
                int v170;
                v170 = v149 * 128;
                int v171;
                v171 = v164 + v170;
                assert("Tensor range check" && 0 <= v149 && v149 < 1);
                assert("Tensor range check" && 0 <= v151 && v151 < 4);
                int v172;
                v172 = 4 * v149;
                int v173;
                v173 = v172 + v151;
                v141[v173] = v171;
                v151 += 1 ;
            }
            v149 += 1 ;
        }
        bool v174;
        v174 = 0 <= v129;
        bool v175;
        v175 = v174 && v130;
        bool v176;
        v176 = v175 == false;
        if (v176){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v175);
        } else {
        }
        bool v178;
        v178 = 0 <= v136;
        bool v180;
        if (v178){
            bool v179;
            v179 = v136 < 8;
            v180 = v179;
        } else {
            v180 = false;
        }
        bool v181;
        v181 = v180 == false;
        if (v181){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v180);
        } else {
        }
        int v183;
        v183 = v136 * 8;
        int v184;
        v184 = v183 + v129;
        int v185[4];
        int v186[4];
        int v187;
        v187 = 0;
        while (while_method_3(v187)){
            int v189;
            v189 = 0;
            while (while_method_1(v189)){
                assert("Tensor range check" && 0 <= v187 && v187 < 1);
                assert("Tensor range check" && 0 <= v189 && v189 < 4);
                int v191;
                v191 = 4 * v187;
                int v192;
                v192 = v191 + v189;
                int v193;
                v193 = v141[v192];
                assert("Tensor range check" && 0 <= v187 && v187 < 1);
                assert("Tensor range check" && 0 <= v189 && v189 < 4);
                v185[v192] = v184;
                v186[v192] = v193;
                v189 += 1 ;
            }
            v187 += 1 ;
        }
        assert("Tensor range check" && 0 <= v136 && v136 < 8);
        int v194;
        v194 = 0;
        while (while_method_3(v194)){
            assert("Tensor range check" && 0 <= v194 && v194 < 1);
            int v196;
            v196 = 128 * v194;
            int v197;
            v197 = v196 + v139;
            assert("Tensor range check" && 0 <= v194 && v194 < 1);
            int v198;
            v198 = 4 * v194;
            int4* v199;
            v199 = reinterpret_cast<int4*>(v185 + v198);
            int4* v200;
            v200 = reinterpret_cast<int4*>(v10 + v197);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v199) % 16 == 0 && reinterpret_cast<unsigned long long>(v200) % 16 == 0);
            *v200 = *v199;
            int4* v201;
            v201 = reinterpret_cast<int4*>(v186 + v198);
            int4* v202;
            v202 = reinterpret_cast<int4*>(v11 + v197);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v201) % 16 == 0 && reinterpret_cast<unsigned long long>(v202) % 16 == 0);
            *v202 = *v201;
            v194 += 1 ;
        }
        v136 += 1 ;
    }
    __syncthreads();
    int v203;
    v203 = threadIdx.x;
    bool v204;
    v204 = 0 <= v203;
    bool v205;
    v205 = v204 == false;
    if (v205){
        assert("The index needs to be zero or positive." && v204);
    } else {
    }
    int v207;
    v207 = v203 % 32;
    int v208;
    v208 = v203 / 32;
    bool v209;
    v209 = v208 < 8;
    bool v210;
    v210 = v209 == false;
    if (v210){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v209);
    } else {
    }
    assert("Tensor range check" && 0 <= v208 && v208 < 8);
    assert("Tensor range check" && 0 <= v207 && v207 < 32);
    int v212;
    v212 = 4 * v207;
    int v213;
    v213 = 128 * v208;
    int v214;
    v214 = v213 + v212;
    assert("Tensor range check" && 0 <= v208 && v208 < 8);
    int v215;
    v215 = 0;
    while (while_method_2(v215)){
        assert("Tensor range check" && 0 <= v215 && v215 < 8);
        int v217;
        v217 = 1024 * v215;
        int v218;
        v218 = v217 + v214;
        float v219[4];
        int v220[4];
        int v221;
        v221 = 0;
        while (while_method_3(v221)){
            assert("Tensor range check" && 0 <= v221 && v221 < 1);
            int v223;
            v223 = 4 * v221;
            assert("Tensor range check" && 0 <= v221 && v221 < 1);
            int v224;
            v224 = 128 * v221;
            int v225;
            v225 = v224 + v218;
            int4* v226;
            v226 = reinterpret_cast<int4*>(v1 + v225);
            int4* v227;
            v227 = reinterpret_cast<int4*>(v219 + v223);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v226) % 16 == 0 && reinterpret_cast<unsigned long long>(v227) % 16 == 0);
            *v227 = *v226;
            v221 += 1 ;
        }
        int v228;
        v228 = 0;
        while (while_method_3(v228)){
            int v230;
            v230 = 0;
            while (while_method_1(v230)){
                bool v232;
                v232 = 0 <= v230;
                bool v234;
                if (v232){
                    bool v233;
                    v233 = v230 < 4;
                    v234 = v233;
                } else {
                    v234 = false;
                }
                bool v235;
                v235 = v234 == false;
                if (v235){
                    assert("The indices should be inside the range of the dimension." && v234);
                } else {
                }
                bool v237;
                v237 = 0 <= v207;
                bool v239;
                if (v237){
                    bool v238;
                    v238 = v207 < 32;
                    v239 = v238;
                } else {
                    v239 = false;
                }
                bool v240;
                v240 = v239 == false;
                if (v240){
                    assert("The indices should be inside the range of the dimension." && v239);
                } else {
                }
                int v242;
                v242 = v207 * 4;
                int v243;
                v243 = v230 + v242;
                bool v244;
                v244 = 0 <= v228;
                bool v246;
                if (v244){
                    bool v245;
                    v245 = v228 < 1;
                    v246 = v245;
                } else {
                    v246 = false;
                }
                bool v247;
                v247 = v246 == false;
                if (v247){
                    assert("The indices should be inside the range of the dimension." && v246);
                } else {
                }
                int v249;
                v249 = v228 * 128;
                int v250;
                v250 = v243 + v249;
                assert("Tensor range check" && 0 <= v228 && v228 < 1);
                assert("Tensor range check" && 0 <= v230 && v230 < 4);
                int v251;
                v251 = 4 * v228;
                int v252;
                v252 = v251 + v230;
                v220[v252] = v250;
                v230 += 1 ;
            }
            v228 += 1 ;
        }
        bool v253;
        v253 = 0 <= v208;
        bool v254;
        v254 = v253 && v209;
        bool v255;
        v255 = v254 == false;
        if (v255){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v254);
        } else {
        }
        bool v257;
        v257 = 0 <= v215;
        bool v259;
        if (v257){
            bool v258;
            v258 = v215 < 8;
            v259 = v258;
        } else {
            v259 = false;
        }
        bool v260;
        v260 = v259 == false;
        if (v260){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v259);
        } else {
        }
        int v262;
        v262 = v215 * 8;
        int v263;
        v263 = v262 + v208;
        assert("Tensor range check" && 0 <= v215 && v215 < 8);
        int v264;
        v264 = 8 * v215;
        int v265;
        v265 = v264 + v208;
        v12[v265] = v263;
        v215 += 1 ;
    }
    __syncthreads();
    int v266;
    v266 = threadIdx.x;
    bool v267;
    v267 = 0 <= v266;
    bool v268;
    v268 = v267 == false;
    if (v268){
        assert("The index needs to be zero or positive." && v267);
    } else {
    }
    int v270;
    v270 = v266 % 32;
    int v271;
    v271 = v266 / 32;
    bool v272;
    v272 = v271 < 8;
    bool v273;
    v273 = v272 == false;
    if (v273){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v272);
    } else {
    }
    assert("Tensor range check" && 0 <= v271 && v271 < 8);
    assert("Tensor range check" && 0 <= v270 && v270 < 32);
    int v275;
    v275 = 4 * v270;
    int v276;
    v276 = 128 * v271;
    int v277;
    v277 = v276 + v275;
    assert("Tensor range check" && 0 <= v271 && v271 < 8);
    assert("Tensor range check" && 0 <= v270 && v270 < 32);
    int v278;
    v278 = 0;
    while (while_method_2(v278)){
        assert("Tensor range check" && 0 <= v278 && v278 < 8);
        int v280;
        v280 = 1024 * v278;
        int v281;
        v281 = v280 + v277;
        float v282[4];
        int v283[4];
        int v284;
        v284 = 0;
        while (while_method_3(v284)){
            assert("Tensor range check" && 0 <= v284 && v284 < 1);
            int v286;
            v286 = 4 * v284;
            assert("Tensor range check" && 0 <= v284 && v284 < 1);
            int v287;
            v287 = 128 * v284;
            int v288;
            v288 = v287 + v281;
            int4* v289;
            v289 = reinterpret_cast<int4*>(v1 + v288);
            int4* v290;
            v290 = reinterpret_cast<int4*>(v282 + v286);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v289) % 16 == 0 && reinterpret_cast<unsigned long long>(v290) % 16 == 0);
            *v290 = *v289;
            v284 += 1 ;
        }
        int v291;
        v291 = 0;
        while (while_method_3(v291)){
            int v293;
            v293 = 0;
            while (while_method_1(v293)){
                bool v295;
                v295 = 0 <= v293;
                bool v297;
                if (v295){
                    bool v296;
                    v296 = v293 < 4;
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
                v300 = 0 <= v270;
                bool v302;
                if (v300){
                    bool v301;
                    v301 = v270 < 32;
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
                v305 = v270 * 4;
                int v306;
                v306 = v293 + v305;
                bool v307;
                v307 = 0 <= v291;
                bool v309;
                if (v307){
                    bool v308;
                    v308 = v291 < 1;
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
                v312 = v291 * 128;
                int v313;
                v313 = v306 + v312;
                assert("Tensor range check" && 0 <= v291 && v291 < 1);
                assert("Tensor range check" && 0 <= v293 && v293 < 4);
                int v314;
                v314 = 4 * v291;
                int v315;
                v315 = v314 + v293;
                v283[v315] = v313;
                v293 += 1 ;
            }
            v291 += 1 ;
        }
        bool v316;
        v316 = 0 <= v271;
        bool v317;
        v317 = v316 && v272;
        bool v318;
        v318 = v317 == false;
        if (v318){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v317);
        } else {
        }
        bool v320;
        v320 = 0 <= v278;
        bool v322;
        if (v320){
            bool v321;
            v321 = v278 < 8;
            v322 = v321;
        } else {
            v322 = false;
        }
        bool v323;
        v323 = v322 == false;
        if (v323){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v322);
        } else {
        }
        int v325;
        v325 = v278 * 8;
        int v326;
        v326 = v325 + v271;
        float v327;
        v327 = 0.0f;
        int v328;
        v328 = 0;
        while (while_method_3(v328)){
            int v330;
            v330 = 0;
            while (while_method_1(v330)){
                assert("Tensor range check" && 0 <= v328 && v328 < 1);
                assert("Tensor range check" && 0 <= v330 && v330 < 4);
                int v332;
                v332 = 4 * v328;
                int v333;
                v333 = v332 + v330;
                float v334;
                v334 = v282[v333];
                float v335;
                v335 = v327 + v334;
                v327 = v335;
                v330 += 1 ;
            }
            v328 += 1 ;
        }
        auto v336 = cooperative_groups::coalesced_threads();
        int v337;
        v337 = threadIdx.x;
        int v338;
        v338 = v337 / 32;
        auto v339 = cooperative_groups::labeled_partition(v336,v338);
        float v340;
        v340 = cooperative_groups::reduce(v339, v327, v42);
        float v341;
        v341 = v340 / 128.0f;
        float v342[4];
        int v343;
        v343 = 0;
        while (while_method_3(v343)){
            int v345;
            v345 = 0;
            while (while_method_1(v345)){
                assert("Tensor range check" && 0 <= v343 && v343 < 1);
                assert("Tensor range check" && 0 <= v345 && v345 < 4);
                int v347;
                v347 = 4 * v343;
                int v348;
                v348 = v347 + v345;
                float v349;
                v349 = v282[v348];
                float v350;
                v350 = v349 - v341;
                float v351;
                v351 = exp(v350);
                assert("Tensor range check" && 0 <= v343 && v343 < 1);
                assert("Tensor range check" && 0 <= v345 && v345 < 4);
                v342[v348] = v351;
                v345 += 1 ;
            }
            v343 += 1 ;
        }
        float v352;
        v352 = 0.0f;
        int v353;
        v353 = 0;
        while (while_method_3(v353)){
            int v355;
            v355 = 0;
            while (while_method_1(v355)){
                assert("Tensor range check" && 0 <= v353 && v353 < 1);
                assert("Tensor range check" && 0 <= v355 && v355 < 4);
                int v357;
                v357 = 4 * v353;
                int v358;
                v358 = v357 + v355;
                float v359;
                v359 = v342[v358];
                float v360;
                v360 = v352 + v359;
                v352 = v360;
                v355 += 1 ;
            }
            v353 += 1 ;
        }
        auto v361 = cooperative_groups::coalesced_threads();
        int v362;
        v362 = threadIdx.x;
        int v363;
        v363 = v362 / 32;
        auto v364 = cooperative_groups::labeled_partition(v361,v363);
        float v365;
        v365 = cooperative_groups::reduce(v364, v352, v42);
        float v366[4];
        int v367;
        v367 = 0;
        while (while_method_3(v367)){
            int v369;
            v369 = 0;
            while (while_method_1(v369)){
                assert("Tensor range check" && 0 <= v367 && v367 < 1);
                assert("Tensor range check" && 0 <= v369 && v369 < 4);
                int v371;
                v371 = 4 * v367;
                int v372;
                v372 = v371 + v369;
                float v373;
                v373 = v342[v372];
                float v374;
                v374 = v373 / v365;
                assert("Tensor range check" && 0 <= v367 && v367 < 1);
                assert("Tensor range check" && 0 <= v369 && v369 < 4);
                v366[v372] = v374;
                v369 += 1 ;
            }
            v367 += 1 ;
        }
        assert("Tensor range check" && 0 <= v278 && v278 < 8);
        int v375;
        v375 = 0;
        while (while_method_3(v375)){
            assert("Tensor range check" && 0 <= v375 && v375 < 1);
            int v377;
            v377 = 128 * v375;
            int v378;
            v378 = v377 + v281;
            assert("Tensor range check" && 0 <= v375 && v375 < 1);
            int v379;
            v379 = 4 * v375;
            int4* v380;
            v380 = reinterpret_cast<int4*>(v366 + v379);
            int4* v381;
            v381 = reinterpret_cast<int4*>(v4 + v378);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v380) % 16 == 0 && reinterpret_cast<unsigned long long>(v381) % 16 == 0);
            *v381 = *v380;
            v375 += 1 ;
        }
        v278 += 1 ;
    }
    __syncthreads();
    int v382;
    v382 = threadIdx.x;
    bool v383;
    v383 = 0 <= v382;
    bool v384;
    v384 = v383 == false;
    if (v384){
        assert("The index needs to be zero or positive." && v383);
    } else {
    }
    int v386;
    v386 = v382 % 32;
    int v387;
    v387 = v382 / 32;
    bool v388;
    v388 = v387 < 8;
    bool v389;
    v389 = v388 == false;
    if (v389){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v388);
    } else {
    }
    assert("Tensor range check" && 0 <= v387 && v387 < 8);
    assert("Tensor range check" && 0 <= v386 && v386 < 32);
    int v391;
    v391 = 4 * v386;
    int v392;
    v392 = 128 * v387;
    int v393;
    v393 = v392 + v391;
    assert("Tensor range check" && 0 <= v387 && v387 < 8);
    assert("Tensor range check" && 0 <= v386 && v386 < 32);
    int v394;
    v394 = 0;
    while (while_method_2(v394)){
        assert("Tensor range check" && 0 <= v394 && v394 < 8);
        int v396;
        v396 = 1024 * v394;
        int v397;
        v397 = v396 + v393;
        float v398[4];
        int v399[4];
        int v400;
        v400 = 0;
        while (while_method_3(v400)){
            assert("Tensor range check" && 0 <= v400 && v400 < 1);
            int v402;
            v402 = 4 * v400;
            assert("Tensor range check" && 0 <= v400 && v400 < 1);
            int v403;
            v403 = 128 * v400;
            int v404;
            v404 = v403 + v397;
            int4* v405;
            v405 = reinterpret_cast<int4*>(v1 + v404);
            int4* v406;
            v406 = reinterpret_cast<int4*>(v398 + v402);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v405) % 16 == 0 && reinterpret_cast<unsigned long long>(v406) % 16 == 0);
            *v406 = *v405;
            v400 += 1 ;
        }
        int v407;
        v407 = 0;
        while (while_method_3(v407)){
            int v409;
            v409 = 0;
            while (while_method_1(v409)){
                bool v411;
                v411 = 0 <= v409;
                bool v413;
                if (v411){
                    bool v412;
                    v412 = v409 < 4;
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
                bool v416;
                v416 = 0 <= v386;
                bool v418;
                if (v416){
                    bool v417;
                    v417 = v386 < 32;
                    v418 = v417;
                } else {
                    v418 = false;
                }
                bool v419;
                v419 = v418 == false;
                if (v419){
                    assert("The indices should be inside the range of the dimension." && v418);
                } else {
                }
                int v421;
                v421 = v386 * 4;
                int v422;
                v422 = v409 + v421;
                bool v423;
                v423 = 0 <= v407;
                bool v425;
                if (v423){
                    bool v424;
                    v424 = v407 < 1;
                    v425 = v424;
                } else {
                    v425 = false;
                }
                bool v426;
                v426 = v425 == false;
                if (v426){
                    assert("The indices should be inside the range of the dimension." && v425);
                } else {
                }
                int v428;
                v428 = v407 * 128;
                int v429;
                v429 = v422 + v428;
                assert("Tensor range check" && 0 <= v407 && v407 < 1);
                assert("Tensor range check" && 0 <= v409 && v409 < 4);
                int v430;
                v430 = 4 * v407;
                int v431;
                v431 = v430 + v409;
                v399[v431] = v429;
                v409 += 1 ;
            }
            v407 += 1 ;
        }
        bool v432;
        v432 = 0 <= v387;
        bool v433;
        v433 = v432 && v388;
        bool v434;
        v434 = v433 == false;
        if (v434){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v433);
        } else {
        }
        bool v436;
        v436 = 0 <= v394;
        bool v438;
        if (v436){
            bool v437;
            v437 = v394 < 8;
            v438 = v437;
        } else {
            v438 = false;
        }
        bool v439;
        v439 = v438 == false;
        if (v439){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v438);
        } else {
        }
        int v441;
        v441 = v394 * 8;
        int v442;
        v442 = v441 + v387;
        float v443[4];
        int v444;
        v444 = 0;
        while (while_method_3(v444)){
            int v446;
            v446 = 0;
            while (while_method_1(v446)){
                assert("Tensor range check" && 0 <= v444 && v444 < 1);
                assert("Tensor range check" && 0 <= v446 && v446 < 4);
                int v448;
                v448 = 4 * v444;
                int v449;
                v449 = v448 + v446;
                float v450;
                v450 = v398[v449];
                float v451;
                v451 = v450 * v450;
                assert("Tensor range check" && 0 <= v444 && v444 < 1);
                assert("Tensor range check" && 0 <= v446 && v446 < 4);
                v443[v449] = v451;
                v446 += 1 ;
            }
            v444 += 1 ;
        }
        float v452;
        v452 = 0.0f;
        int v453;
        v453 = 0;
        while (while_method_3(v453)){
            int v455;
            v455 = 0;
            while (while_method_1(v455)){
                assert("Tensor range check" && 0 <= v453 && v453 < 1);
                assert("Tensor range check" && 0 <= v455 && v455 < 4);
                int v457;
                v457 = 4 * v453;
                int v458;
                v458 = v457 + v455;
                float v459;
                v459 = v443[v458];
                float v460;
                v460 = v452 + v459;
                v452 = v460;
                v455 += 1 ;
            }
            v453 += 1 ;
        }
        auto v461 = cooperative_groups::coalesced_threads();
        int v462;
        v462 = threadIdx.x;
        int v463;
        v463 = v462 / 32;
        auto v464 = cooperative_groups::labeled_partition(v461,v463);
        float v465;
        v465 = cooperative_groups::reduce(v464, v452, v42);
        float v466[4];
        int v467;
        v467 = 0;
        while (while_method_3(v467)){
            int v469;
            v469 = 0;
            while (while_method_1(v469)){
                assert("Tensor range check" && 0 <= v467 && v467 < 1);
                assert("Tensor range check" && 0 <= v469 && v469 < 4);
                int v471;
                v471 = 4 * v467;
                int v472;
                v472 = v471 + v469;
                float v473;
                v473 = v398[v472];
                bool v474;
                v474 = v465 == 0.0f;
                bool v475;
                v475 = v474 != true;
                float v477;
                if (v475){
                    float v476;
                    v476 = v473 / v465;
                    v477 = v476;
                } else {
                    v477 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v467 && v467 < 1);
                assert("Tensor range check" && 0 <= v469 && v469 < 4);
                v466[v472] = v477;
                v469 += 1 ;
            }
            v467 += 1 ;
        }
        assert("Tensor range check" && 0 <= v394 && v394 < 8);
        int v478;
        v478 = 0;
        while (while_method_3(v478)){
            assert("Tensor range check" && 0 <= v478 && v478 < 1);
            int v480;
            v480 = 128 * v478;
            int v481;
            v481 = v480 + v397;
            assert("Tensor range check" && 0 <= v478 && v478 < 1);
            int v482;
            v482 = 4 * v478;
            int4* v483;
            v483 = reinterpret_cast<int4*>(v466 + v482);
            int4* v484;
            v484 = reinterpret_cast<int4*>(v8 + v481);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v483) % 16 == 0 && reinterpret_cast<unsigned long long>(v484) % 16 == 0);
            *v484 = *v483;
            v478 += 1 ;
        }
        v394 += 1 ;
    }
    __syncthreads();
    int v485;
    v485 = threadIdx.x;
    bool v486;
    v486 = 0 <= v485;
    bool v487;
    v487 = v486 == false;
    if (v487){
        assert("The index needs to be zero or positive." && v486);
    } else {
    }
    int v489;
    v489 = v485 % 32;
    int v490;
    v490 = v485 / 32;
    bool v491;
    v491 = v490 < 8;
    bool v492;
    v492 = v491 == false;
    if (v492){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v491);
    } else {
    }
    assert("Tensor range check" && 0 <= v490 && v490 < 8);
    assert("Tensor range check" && 0 <= v489 && v489 < 32);
    int v494;
    v494 = 4 * v489;
    int v495;
    v495 = 128 * v490;
    int v496;
    v496 = v495 + v494;
    assert("Tensor range check" && 0 <= v490 && v490 < 8);
    int v497;
    v497 = 0;
    while (while_method_2(v497)){
        assert("Tensor range check" && 0 <= v497 && v497 < 8);
        int v499;
        v499 = 1024 * v497;
        int v500;
        v500 = v499 + v496;
        float v501[4];
        int v502[4];
        int v503;
        v503 = 0;
        while (while_method_3(v503)){
            assert("Tensor range check" && 0 <= v503 && v503 < 1);
            int v505;
            v505 = 4 * v503;
            assert("Tensor range check" && 0 <= v503 && v503 < 1);
            int v506;
            v506 = 128 * v503;
            int v507;
            v507 = v506 + v500;
            int4* v508;
            v508 = reinterpret_cast<int4*>(v1 + v507);
            int4* v509;
            v509 = reinterpret_cast<int4*>(v501 + v505);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v508) % 16 == 0 && reinterpret_cast<unsigned long long>(v509) % 16 == 0);
            *v509 = *v508;
            v503 += 1 ;
        }
        int v510;
        v510 = 0;
        while (while_method_3(v510)){
            int v512;
            v512 = 0;
            while (while_method_1(v512)){
                bool v514;
                v514 = 0 <= v512;
                bool v516;
                if (v514){
                    bool v515;
                    v515 = v512 < 4;
                    v516 = v515;
                } else {
                    v516 = false;
                }
                bool v517;
                v517 = v516 == false;
                if (v517){
                    assert("The indices should be inside the range of the dimension." && v516);
                } else {
                }
                bool v519;
                v519 = 0 <= v489;
                bool v521;
                if (v519){
                    bool v520;
                    v520 = v489 < 32;
                    v521 = v520;
                } else {
                    v521 = false;
                }
                bool v522;
                v522 = v521 == false;
                if (v522){
                    assert("The indices should be inside the range of the dimension." && v521);
                } else {
                }
                int v524;
                v524 = v489 * 4;
                int v525;
                v525 = v512 + v524;
                bool v526;
                v526 = 0 <= v510;
                bool v528;
                if (v526){
                    bool v527;
                    v527 = v510 < 1;
                    v528 = v527;
                } else {
                    v528 = false;
                }
                bool v529;
                v529 = v528 == false;
                if (v529){
                    assert("The indices should be inside the range of the dimension." && v528);
                } else {
                }
                int v531;
                v531 = v510 * 128;
                int v532;
                v532 = v525 + v531;
                assert("Tensor range check" && 0 <= v510 && v510 < 1);
                assert("Tensor range check" && 0 <= v512 && v512 < 4);
                int v533;
                v533 = 4 * v510;
                int v534;
                v534 = v533 + v512;
                v502[v534] = v532;
                v512 += 1 ;
            }
            v510 += 1 ;
        }
        bool v535;
        v535 = 0 <= v490;
        bool v536;
        v536 = v535 && v491;
        bool v537;
        v537 = v536 == false;
        if (v537){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v536);
        } else {
        }
        bool v539;
        v539 = 0 <= v497;
        bool v541;
        if (v539){
            bool v540;
            v540 = v497 < 8;
            v541 = v540;
        } else {
            v541 = false;
        }
        bool v542;
        v542 = v541 == false;
        if (v542){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v541);
        } else {
        }
        int v544;
        v544 = v497 * 8;
        int v545;
        v545 = v544 + v490;
        float v546; int v547;
        Tuple1 tmp1 = Tuple1{-1.0f / 0.0f, 0};
        v546 = tmp1.v0; v547 = tmp1.v1;
        int v548;
        v548 = 0;
        while (while_method_3(v548)){
            int v550;
            v550 = 0;
            while (while_method_1(v550)){
                assert("Tensor range check" && 0 <= v548 && v548 < 1);
                assert("Tensor range check" && 0 <= v550 && v550 < 4);
                int v552;
                v552 = 4 * v548;
                int v553;
                v553 = v552 + v550;
                float v554;
                v554 = v501[v553];
                int v555;
                v555 = v502[v553];
                bool v556;
                v556 = v546 > v554;
                float v557; int v558;
                if (v556){
                    v557 = v546; v558 = v547;
                } else {
                    v557 = v554; v558 = v555;
                }
                v546 = v557;
                v547 = v558;
                v550 += 1 ;
            }
            v548 += 1 ;
        }
        auto v559 = cooperative_groups::coalesced_threads();
        int v560;
        v560 = threadIdx.x;
        int v561;
        v561 = v560 / 32;
        auto v562 = cooperative_groups::labeled_partition(v559,v561);
        Closure1 v563{};
        float v564; int v565;
        Tuple1 tmp2 = cooperative_groups::reduce(v562, Tuple1{v546, v547}, v563);
        v564 = tmp2.v0; v565 = tmp2.v1;
        assert("Tensor range check" && 0 <= v497 && v497 < 8);
        int v566;
        v566 = 8 * v497;
        int v567;
        v567 = v566 + v490;
        v9[v567] = v565;
        v497 += 1 ;
    }
    __syncthreads();
    int v568;
    v568 = threadIdx.x;
    bool v569;
    v569 = 0 <= v568;
    bool v570;
    v570 = v569 == false;
    if (v570){
        assert("The index needs to be zero or positive." && v569);
    } else {
    }
    int v572;
    v572 = v568 % 32;
    int v573;
    v573 = v568 / 32;
    bool v574;
    v574 = v573 < 8;
    bool v575;
    v575 = v574 == false;
    if (v575){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v574);
    } else {
    }
    assert("Tensor range check" && 0 <= v573 && v573 < 8);
    assert("Tensor range check" && 0 <= v572 && v572 < 32);
    int v577;
    v577 = 4 * v572;
    int v578;
    v578 = 128 * v573;
    int v579;
    v579 = v578 + v577;
    assert("Tensor range check" && 0 <= v573 && v573 < 8);
    assert("Tensor range check" && 0 <= v572 && v572 < 32);
    int v580;
    v580 = 0;
    while (while_method_2(v580)){
        assert("Tensor range check" && 0 <= v580 && v580 < 8);
        int v582;
        v582 = 1024 * v580;
        int v583;
        v583 = v582 + v579;
        float v584[4];
        int v585[4];
        int v586;
        v586 = 0;
        while (while_method_3(v586)){
            assert("Tensor range check" && 0 <= v586 && v586 < 1);
            int v588;
            v588 = 4 * v586;
            assert("Tensor range check" && 0 <= v586 && v586 < 1);
            int v589;
            v589 = 128 * v586;
            int v590;
            v590 = v589 + v583;
            int4* v591;
            v591 = reinterpret_cast<int4*>(v1 + v590);
            int4* v592;
            v592 = reinterpret_cast<int4*>(v584 + v588);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v591) % 16 == 0 && reinterpret_cast<unsigned long long>(v592) % 16 == 0);
            *v592 = *v591;
            v586 += 1 ;
        }
        int v593;
        v593 = 0;
        while (while_method_3(v593)){
            int v595;
            v595 = 0;
            while (while_method_1(v595)){
                bool v597;
                v597 = 0 <= v595;
                bool v599;
                if (v597){
                    bool v598;
                    v598 = v595 < 4;
                    v599 = v598;
                } else {
                    v599 = false;
                }
                bool v600;
                v600 = v599 == false;
                if (v600){
                    assert("The indices should be inside the range of the dimension." && v599);
                } else {
                }
                bool v602;
                v602 = 0 <= v572;
                bool v604;
                if (v602){
                    bool v603;
                    v603 = v572 < 32;
                    v604 = v603;
                } else {
                    v604 = false;
                }
                bool v605;
                v605 = v604 == false;
                if (v605){
                    assert("The indices should be inside the range of the dimension." && v604);
                } else {
                }
                int v607;
                v607 = v572 * 4;
                int v608;
                v608 = v595 + v607;
                bool v609;
                v609 = 0 <= v593;
                bool v611;
                if (v609){
                    bool v610;
                    v610 = v593 < 1;
                    v611 = v610;
                } else {
                    v611 = false;
                }
                bool v612;
                v612 = v611 == false;
                if (v612){
                    assert("The indices should be inside the range of the dimension." && v611);
                } else {
                }
                int v614;
                v614 = v593 * 128;
                int v615;
                v615 = v608 + v614;
                assert("Tensor range check" && 0 <= v593 && v593 < 1);
                assert("Tensor range check" && 0 <= v595 && v595 < 4);
                int v616;
                v616 = 4 * v593;
                int v617;
                v617 = v616 + v595;
                v585[v617] = v615;
                v595 += 1 ;
            }
            v593 += 1 ;
        }
        bool v618;
        v618 = 0 <= v573;
        bool v619;
        v619 = v618 && v574;
        bool v620;
        v620 = v619 == false;
        if (v620){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v619);
        } else {
        }
        bool v622;
        v622 = 0 <= v580;
        bool v624;
        if (v622){
            bool v623;
            v623 = v580 < 8;
            v624 = v623;
        } else {
            v624 = false;
        }
        bool v625;
        v625 = v624 == false;
        if (v625){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v624);
        } else {
        }
        int v627;
        v627 = v580 * 8;
        int v628;
        v628 = v627 + v573;
        float v629;
        v629 = 0.0f;
        int v630;
        v630 = 0;
        while (while_method_3(v630)){
            int v632;
            v632 = 0;
            while (while_method_1(v632)){
                assert("Tensor range check" && 0 <= v630 && v630 < 1);
                assert("Tensor range check" && 0 <= v632 && v632 < 4);
                int v634;
                v634 = 4 * v630;
                int v635;
                v635 = v634 + v632;
                float v636;
                v636 = v584[v635];
                float v637;
                v637 = v629 + v636;
                v629 = v637;
                v632 += 1 ;
            }
            v630 += 1 ;
        }
        auto v638 = cooperative_groups::coalesced_threads();
        int v639;
        v639 = threadIdx.x;
        int v640;
        v640 = v639 / 32;
        auto v641 = cooperative_groups::labeled_partition(v638,v640);
        float v642;
        v642 = cooperative_groups::reduce(v641, v629, v42);
        float v643;
        v643 = v642 / 128.0f;
        float v644[4];
        int v645;
        v645 = 0;
        while (while_method_3(v645)){
            int v647;
            v647 = 0;
            while (while_method_1(v647)){
                assert("Tensor range check" && 0 <= v645 && v645 < 1);
                assert("Tensor range check" && 0 <= v647 && v647 < 4);
                int v649;
                v649 = 4 * v645;
                int v650;
                v650 = v649 + v647;
                float v651;
                v651 = v584[v650];
                float v652;
                v652 = v651 - v643;
                float v653;
                v653 = exp(v652);
                assert("Tensor range check" && 0 <= v645 && v645 < 1);
                assert("Tensor range check" && 0 <= v647 && v647 < 4);
                v644[v650] = v653;
                v647 += 1 ;
            }
            v645 += 1 ;
        }
        float v654;
        v654 = 0.0f;
        int v655;
        v655 = 0;
        while (while_method_3(v655)){
            int v657;
            v657 = 0;
            while (while_method_1(v657)){
                assert("Tensor range check" && 0 <= v655 && v655 < 1);
                assert("Tensor range check" && 0 <= v657 && v657 < 4);
                int v659;
                v659 = 4 * v655;
                int v660;
                v660 = v659 + v657;
                float v661;
                v661 = v644[v660];
                float v662;
                v662 = v654 + v661;
                v654 = v662;
                v657 += 1 ;
            }
            v655 += 1 ;
        }
        auto v663 = cooperative_groups::coalesced_threads();
        int v664;
        v664 = threadIdx.x;
        int v665;
        v665 = v664 / 32;
        auto v666 = cooperative_groups::labeled_partition(v663,v665);
        float v667;
        v667 = cooperative_groups::reduce(v666, v654, v42);
        float v668[4];
        int v669;
        v669 = 0;
        while (while_method_3(v669)){
            int v671;
            v671 = 0;
            while (while_method_1(v671)){
                assert("Tensor range check" && 0 <= v669 && v669 < 1);
                assert("Tensor range check" && 0 <= v671 && v671 < 4);
                int v673;
                v673 = 4 * v669;
                int v674;
                v674 = v673 + v671;
                float v675;
                v675 = v644[v674];
                float v676;
                v676 = v675 / v667;
                assert("Tensor range check" && 0 <= v669 && v669 < 1);
                assert("Tensor range check" && 0 <= v671 && v671 < 4);
                v668[v674] = v676;
                v671 += 1 ;
            }
            v669 += 1 ;
        }
        float v677[4];
        float v678;
        v678 = 0.0f;
        int v679;
        v679 = 0;
        while (while_method_3(v679)){
            assert("Tensor range check" && 0 <= v679 && v679 < 1);
            int v681;
            v681 = 4 * v679;
            assert("Tensor range check" && 0 <= v679 && v679 < 1);
            float v682;
            v682 = 0.0f;
            int v683;
            v683 = 0;
            while (while_method_1(v683)){
                assert("Tensor range check" && 0 <= v683 && v683 < 4);
                int v685;
                v685 = v683 + v681;
                float v686;
                v686 = v668[v685];
                float v687;
                v687 = v682 + v686;
                v682 = v687;
                v683 += 1 ;
            }
            auto v688 = cooperative_groups::coalesced_threads();
            int v689;
            v689 = threadIdx.x;
            int v690;
            v690 = v689 / 32;
            auto v691 = cooperative_groups::labeled_partition(v688,v690);
            Closure2 v692{};
            float v693;
            v693 = cooperative_groups::inclusive_scan(v691, v682, v692);
            float v694;
            v694 = v691.shfl_up(v693,1);
            bool v695;
            v695 = v691.thread_rank() == 0;
            float v696;
            if (v695){
                v696 = 0.0f;
            } else {
                v696 = v694;
            }
            float v697;
            v697 = v691.shfl(v693,v691.num_threads()-1);
            float v698;
            v698 = v678 + v696;
            float v699;
            v699 = v698;
            int v700;
            v700 = 0;
            while (while_method_1(v700)){
                assert("Tensor range check" && 0 <= v700 && v700 < 4);
                int v702;
                v702 = v700 + v681;
                float v703;
                v703 = v668[v702];
                float v704;
                v704 = v699 + v703;
                assert("Tensor range check" && 0 <= v700 && v700 < 4);
                v677[v702] = v704;
                v699 = v704;
                v700 += 1 ;
            }
            float v705;
            v705 = v678 + v697;
            v678 = v705;
            v679 += 1 ;
        }
        assert("Tensor range check" && 0 <= v580 && v580 < 8);
        int v706;
        v706 = 0;
        while (while_method_3(v706)){
            assert("Tensor range check" && 0 <= v706 && v706 < 1);
            int v708;
            v708 = 128 * v706;
            int v709;
            v709 = v708 + v583;
            assert("Tensor range check" && 0 <= v706 && v706 < 1);
            int v710;
            v710 = 4 * v706;
            int4* v711;
            v711 = reinterpret_cast<int4*>(v668 + v710);
            int4* v712;
            v712 = reinterpret_cast<int4*>(v6 + v709);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v711) % 16 == 0 && reinterpret_cast<unsigned long long>(v712) % 16 == 0);
            *v712 = *v711;
            int4* v713;
            v713 = reinterpret_cast<int4*>(v677 + v710);
            int4* v714;
            v714 = reinterpret_cast<int4*>(v7 + v709);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v713) % 16 == 0 && reinterpret_cast<unsigned long long>(v714) % 16 == 0);
            *v714 = *v713;
            v706 += 1 ;
        }
        v580 += 1 ;
    }
    __syncthreads();
    int v715;
    v715 = threadIdx.x;
    bool v716;
    v716 = 0 <= v715;
    bool v717;
    v717 = v716 == false;
    if (v717){
        assert("The index needs to be zero or positive." && v716);
    } else {
    }
    int v719;
    v719 = v715 % 32;
    int v720;
    v720 = v715 / 32;
    bool v721;
    v721 = v720 < 8;
    bool v722;
    v722 = v721 == false;
    if (v722){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v721);
    } else {
    }
    assert("Tensor range check" && 0 <= v720 && v720 < 8);
    assert("Tensor range check" && 0 <= v719 && v719 < 32);
    int v724;
    v724 = 4 * v719;
    int v725;
    v725 = 128 * v720;
    int v726;
    v726 = v725 + v724;
    assert("Tensor range check" && 0 <= v720 && v720 < 8);
    assert("Tensor range check" && 0 <= v719 && v719 < 32);
    int v727;
    v727 = 0;
    while (while_method_2(v727)){
        assert("Tensor range check" && 0 <= v727 && v727 < 8);
        int v729;
        v729 = 1024 * v727;
        int v730;
        v730 = v729 + v726;
        int v731[4];
        int v732[4];
        int v733;
        v733 = 0;
        while (while_method_3(v733)){
            assert("Tensor range check" && 0 <= v733 && v733 < 1);
            int v735;
            v735 = 4 * v733;
            assert("Tensor range check" && 0 <= v733 && v733 < 1);
            int v736;
            v736 = 128 * v733;
            int v737;
            v737 = v736 + v730;
            int4* v738;
            v738 = reinterpret_cast<int4*>(v0 + v737);
            int4* v739;
            v739 = reinterpret_cast<int4*>(v731 + v735);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v738) % 16 == 0 && reinterpret_cast<unsigned long long>(v739) % 16 == 0);
            *v739 = *v738;
            v733 += 1 ;
        }
        int v740;
        v740 = 0;
        while (while_method_3(v740)){
            int v742;
            v742 = 0;
            while (while_method_1(v742)){
                bool v744;
                v744 = 0 <= v742;
                bool v746;
                if (v744){
                    bool v745;
                    v745 = v742 < 4;
                    v746 = v745;
                } else {
                    v746 = false;
                }
                bool v747;
                v747 = v746 == false;
                if (v747){
                    assert("The indices should be inside the range of the dimension." && v746);
                } else {
                }
                bool v749;
                v749 = 0 <= v719;
                bool v751;
                if (v749){
                    bool v750;
                    v750 = v719 < 32;
                    v751 = v750;
                } else {
                    v751 = false;
                }
                bool v752;
                v752 = v751 == false;
                if (v752){
                    assert("The indices should be inside the range of the dimension." && v751);
                } else {
                }
                int v754;
                v754 = v719 * 4;
                int v755;
                v755 = v742 + v754;
                bool v756;
                v756 = 0 <= v740;
                bool v758;
                if (v756){
                    bool v757;
                    v757 = v740 < 1;
                    v758 = v757;
                } else {
                    v758 = false;
                }
                bool v759;
                v759 = v758 == false;
                if (v759){
                    assert("The indices should be inside the range of the dimension." && v758);
                } else {
                }
                int v761;
                v761 = v740 * 128;
                int v762;
                v762 = v755 + v761;
                assert("Tensor range check" && 0 <= v740 && v740 < 1);
                assert("Tensor range check" && 0 <= v742 && v742 < 4);
                int v763;
                v763 = 4 * v740;
                int v764;
                v764 = v763 + v742;
                v732[v764] = v762;
                v742 += 1 ;
            }
            v740 += 1 ;
        }
        bool v765;
        v765 = 0 <= v720;
        bool v766;
        v766 = v765 && v721;
        bool v767;
        v767 = v766 == false;
        if (v767){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v766);
        } else {
        }
        bool v769;
        v769 = 0 <= v727;
        bool v771;
        if (v769){
            bool v770;
            v770 = v727 < 8;
            v771 = v770;
        } else {
            v771 = false;
        }
        bool v772;
        v772 = v771 == false;
        if (v772){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v771);
        } else {
        }
        int v774;
        v774 = v727 * 8;
        int v775;
        v775 = v774 + v720;
        int v776[4];
        int v777;
        v777 = 0;
        int v778;
        v778 = 0;
        while (while_method_3(v778)){
            assert("Tensor range check" && 0 <= v778 && v778 < 1);
            int v780;
            v780 = 4 * v778;
            assert("Tensor range check" && 0 <= v778 && v778 < 1);
            int v781;
            v781 = 0;
            int v782;
            v782 = 0;
            while (while_method_1(v782)){
                assert("Tensor range check" && 0 <= v782 && v782 < 4);
                int v784;
                v784 = v782 + v780;
                int v785;
                v785 = v731[v784];
                int v786;
                v786 = v781 + v785;
                v781 = v786;
                v782 += 1 ;
            }
            auto v787 = cooperative_groups::coalesced_threads();
            int v788;
            v788 = threadIdx.x;
            int v789;
            v789 = v788 / 32;
            auto v790 = cooperative_groups::labeled_partition(v787,v789);
            Closure3 v791{};
            int v792;
            v792 = cooperative_groups::inclusive_scan(v790, v781, v791);
            int v793;
            v793 = v790.shfl_up(v792,1);
            bool v794;
            v794 = v790.thread_rank() == 0;
            int v795;
            if (v794){
                v795 = 0;
            } else {
                v795 = v793;
            }
            int v796;
            v796 = v790.shfl(v792,v790.num_threads()-1);
            int v797;
            v797 = v777 + v795;
            int v798;
            v798 = v797;
            int v799;
            v799 = 0;
            while (while_method_1(v799)){
                assert("Tensor range check" && 0 <= v799 && v799 < 4);
                int v801;
                v801 = v799 + v780;
                int v802;
                v802 = v731[v801];
                assert("Tensor range check" && 0 <= v799 && v799 < 4);
                v776[v801] = v798;
                int v803;
                v803 = v798 + v802;
                v798 = v803;
                v799 += 1 ;
            }
            int v804;
            v804 = v777 + v796;
            v777 = v804;
            v778 += 1 ;
        }
        assert("Tensor range check" && 0 <= v727 && v727 < 8);
        int v805;
        v805 = 0;
        while (while_method_3(v805)){
            assert("Tensor range check" && 0 <= v805 && v805 < 1);
            int v807;
            v807 = 128 * v805;
            int v808;
            v808 = v807 + v730;
            assert("Tensor range check" && 0 <= v805 && v805 < 1);
            int v809;
            v809 = 4 * v805;
            int4* v810;
            v810 = reinterpret_cast<int4*>(v776 + v809);
            int4* v811;
            v811 = reinterpret_cast<int4*>(v13 + v808);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v810) % 16 == 0 && reinterpret_cast<unsigned long long>(v811) % 16 == 0);
            *v811 = *v810;
            v805 += 1 ;
        }
        v727 += 1 ;
    }
    __syncthreads();
    int v812;
    v812 = threadIdx.x;
    bool v813;
    v813 = 0 <= v812;
    bool v814;
    v814 = v813 == false;
    if (v814){
        assert("The index needs to be zero or positive." && v813);
    } else {
    }
    int v816;
    v816 = v812 % 32;
    int v817;
    v817 = v812 / 32;
    bool v818;
    v818 = v817 < 8;
    bool v819;
    v819 = v818 == false;
    if (v819){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v818);
    } else {
    }
    assert("Tensor range check" && 0 <= v817 && v817 < 8);
    assert("Tensor range check" && 0 <= v816 && v816 < 32);
    int v821;
    v821 = 4 * v816;
    int v822;
    v822 = 128 * v817;
    int v823;
    v823 = v822 + v821;
    assert("Tensor range check" && 0 <= v817 && v817 < 8);
    assert("Tensor range check" && 0 <= v816 && v816 < 32);
    int v824;
    v824 = 0;
    while (while_method_2(v824)){
        assert("Tensor range check" && 0 <= v824 && v824 < 8);
        int v826;
        v826 = 1024 * v824;
        int v827;
        v827 = v826 + v823;
        float v828[4];
        int v829[4];
        int v830;
        v830 = 0;
        while (while_method_3(v830)){
            assert("Tensor range check" && 0 <= v830 && v830 < 1);
            int v832;
            v832 = 4 * v830;
            assert("Tensor range check" && 0 <= v830 && v830 < 1);
            int v833;
            v833 = 128 * v830;
            int v834;
            v834 = v833 + v827;
            int4* v835;
            v835 = reinterpret_cast<int4*>(v1 + v834);
            int4* v836;
            v836 = reinterpret_cast<int4*>(v828 + v832);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v835) % 16 == 0 && reinterpret_cast<unsigned long long>(v836) % 16 == 0);
            *v836 = *v835;
            v830 += 1 ;
        }
        int v837;
        v837 = 0;
        while (while_method_3(v837)){
            int v839;
            v839 = 0;
            while (while_method_1(v839)){
                bool v841;
                v841 = 0 <= v839;
                bool v843;
                if (v841){
                    bool v842;
                    v842 = v839 < 4;
                    v843 = v842;
                } else {
                    v843 = false;
                }
                bool v844;
                v844 = v843 == false;
                if (v844){
                    assert("The indices should be inside the range of the dimension." && v843);
                } else {
                }
                bool v846;
                v846 = 0 <= v816;
                bool v848;
                if (v846){
                    bool v847;
                    v847 = v816 < 32;
                    v848 = v847;
                } else {
                    v848 = false;
                }
                bool v849;
                v849 = v848 == false;
                if (v849){
                    assert("The indices should be inside the range of the dimension." && v848);
                } else {
                }
                int v851;
                v851 = v816 * 4;
                int v852;
                v852 = v839 + v851;
                bool v853;
                v853 = 0 <= v837;
                bool v855;
                if (v853){
                    bool v854;
                    v854 = v837 < 1;
                    v855 = v854;
                } else {
                    v855 = false;
                }
                bool v856;
                v856 = v855 == false;
                if (v856){
                    assert("The indices should be inside the range of the dimension." && v855);
                } else {
                }
                int v858;
                v858 = v837 * 128;
                int v859;
                v859 = v852 + v858;
                assert("Tensor range check" && 0 <= v837 && v837 < 1);
                assert("Tensor range check" && 0 <= v839 && v839 < 4);
                int v860;
                v860 = 4 * v837;
                int v861;
                v861 = v860 + v839;
                v829[v861] = v859;
                v839 += 1 ;
            }
            v837 += 1 ;
        }
        bool v862;
        v862 = 0 <= v817;
        bool v863;
        v863 = v862 && v818;
        bool v864;
        v864 = v863 == false;
        if (v864){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v863);
        } else {
        }
        bool v866;
        v866 = 0 <= v824;
        bool v868;
        if (v866){
            bool v867;
            v867 = v824 < 8;
            v868 = v867;
        } else {
            v868 = false;
        }
        bool v869;
        v869 = v868 == false;
        if (v869){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v868);
        } else {
        }
        int v871;
        v871 = v824 * 8;
        int v872;
        v872 = v871 + v817;
        bool v873[4];
        int v874;
        v874 = 0;
        while (while_method_3(v874)){
            int v876;
            v876 = 0;
            while (while_method_1(v876)){
                assert("Tensor range check" && 0 <= v874 && v874 < 1);
                assert("Tensor range check" && 0 <= v876 && v876 < 4);
                int v878;
                v878 = 4 * v874;
                int v879;
                v879 = v878 + v876;
                float v880;
                v880 = v828[v879];
                int v881;
                v881 = v829[v879];
                bool v882;
                v882 = v881 < 4;
                assert("Tensor range check" && 0 <= v874 && v874 < 1);
                assert("Tensor range check" && 0 <= v876 && v876 < 4);
                v873[v879] = v882;
                v876 += 1 ;
            }
            v874 += 1 ;
        }
        float v883[4];
        int v884;
        v884 = 0;
        while (while_method_3(v884)){
            int v886;
            v886 = 0;
            while (while_method_1(v886)){
                assert("Tensor range check" && 0 <= v884 && v884 < 1);
                assert("Tensor range check" && 0 <= v886 && v886 < 4);
                int v888;
                v888 = 4 * v884;
                int v889;
                v889 = v888 + v886;
                float v890;
                v890 = v828[v889];
                bool v891;
                v891 = v873[v889];
                float v892;
                if (v891){
                    v892 = v890;
                } else {
                    v892 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v884 && v884 < 1);
                assert("Tensor range check" && 0 <= v886 && v886 < 4);
                v883[v889] = v892;
                v886 += 1 ;
            }
            v884 += 1 ;
        }
        float v893;
        v893 = 0.0f;
        int v894;
        v894 = 0;
        while (while_method_3(v894)){
            int v896;
            v896 = 0;
            while (while_method_1(v896)){
                assert("Tensor range check" && 0 <= v894 && v894 < 1);
                assert("Tensor range check" && 0 <= v896 && v896 < 4);
                int v898;
                v898 = 4 * v894;
                int v899;
                v899 = v898 + v896;
                float v900;
                v900 = v883[v899];
                float v901;
                v901 = v893 + v900;
                v893 = v901;
                v896 += 1 ;
            }
            v894 += 1 ;
        }
        auto v902 = cooperative_groups::coalesced_threads();
        int v903;
        v903 = threadIdx.x;
        int v904;
        v904 = v903 / 32;
        auto v905 = cooperative_groups::labeled_partition(v902,v904);
        float v906;
        v906 = cooperative_groups::reduce(v905, v893, v42);
        int v907[4];
        int v908;
        v908 = 0;
        while (while_method_3(v908)){
            int v910;
            v910 = 0;
            while (while_method_1(v910)){
                assert("Tensor range check" && 0 <= v908 && v908 < 1);
                assert("Tensor range check" && 0 <= v910 && v910 < 4);
                int v912;
                v912 = 4 * v908;
                int v913;
                v913 = v912 + v910;
                bool v914;
                v914 = v873[v913];
                int v915;
                if (v914){
                    v915 = 1;
                } else {
                    v915 = 0;
                }
                assert("Tensor range check" && 0 <= v908 && v908 < 1);
                assert("Tensor range check" && 0 <= v910 && v910 < 4);
                v907[v913] = v915;
                v910 += 1 ;
            }
            v908 += 1 ;
        }
        int v916;
        v916 = 0;
        int v917;
        v917 = 0;
        while (while_method_3(v917)){
            int v919;
            v919 = 0;
            while (while_method_1(v919)){
                assert("Tensor range check" && 0 <= v917 && v917 < 1);
                assert("Tensor range check" && 0 <= v919 && v919 < 4);
                int v921;
                v921 = 4 * v917;
                int v922;
                v922 = v921 + v919;
                int v923;
                v923 = v907[v922];
                int v924;
                v924 = v916 + v923;
                v916 = v924;
                v919 += 1 ;
            }
            v917 += 1 ;
        }
        auto v925 = cooperative_groups::coalesced_threads();
        int v926;
        v926 = threadIdx.x;
        int v927;
        v927 = v926 / 32;
        auto v928 = cooperative_groups::labeled_partition(v925,v927);
        Closure4 v929{};
        int v930;
        v930 = cooperative_groups::reduce(v928, v916, v929);
        float v931;
        v931 = (float)v930;
        float v932;
        v932 = v906 / v931;
        float v933[4];
        int v934;
        v934 = 0;
        while (while_method_3(v934)){
            int v936;
            v936 = 0;
            while (while_method_1(v936)){
                assert("Tensor range check" && 0 <= v934 && v934 < 1);
                assert("Tensor range check" && 0 <= v936 && v936 < 4);
                int v938;
                v938 = 4 * v934;
                int v939;
                v939 = v938 + v936;
                float v940;
                v940 = v828[v939];
                bool v941;
                v941 = v873[v939];
                float v942;
                if (v941){
                    v942 = v940;
                } else {
                    v942 = -1.0f / 0.0f;
                }
                float v943;
                v943 = v942 - v932;
                float v944;
                v944 = exp(v943);
                bool v945;
                v945 = v944 < 1.0f / 0.0f;
                bool v946;
                v946 = v945 == false;
                if (v946){
                    assert("The softmax values must not grow too large." && v945);
                } else {
                }
                bool v948;
                v948 = isnan(v944);
                bool v949;
                v949 = v948 == false;
                bool v950;
                v950 = v949 == false;
                if (v950){
                    assert("The softmax values must not be nans." && v949);
                } else {
                }
                assert("Tensor range check" && 0 <= v934 && v934 < 1);
                assert("Tensor range check" && 0 <= v936 && v936 < 4);
                v933[v939] = v944;
                v936 += 1 ;
            }
            v934 += 1 ;
        }
        float v952;
        v952 = 0.0f;
        int v953;
        v953 = 0;
        while (while_method_3(v953)){
            int v955;
            v955 = 0;
            while (while_method_1(v955)){
                assert("Tensor range check" && 0 <= v953 && v953 < 1);
                assert("Tensor range check" && 0 <= v955 && v955 < 4);
                int v957;
                v957 = 4 * v953;
                int v958;
                v958 = v957 + v955;
                float v959;
                v959 = v933[v958];
                float v960;
                v960 = v952 + v959;
                v952 = v960;
                v955 += 1 ;
            }
            v953 += 1 ;
        }
        auto v961 = cooperative_groups::coalesced_threads();
        int v962;
        v962 = threadIdx.x;
        int v963;
        v963 = v962 / 32;
        auto v964 = cooperative_groups::labeled_partition(v961,v963);
        float v965;
        v965 = cooperative_groups::reduce(v964, v952, v42);
        float v966[4];
        int v967;
        v967 = 0;
        while (while_method_3(v967)){
            int v969;
            v969 = 0;
            while (while_method_1(v969)){
                assert("Tensor range check" && 0 <= v967 && v967 < 1);
                assert("Tensor range check" && 0 <= v969 && v969 < 4);
                int v971;
                v971 = 4 * v967;
                int v972;
                v972 = v971 + v969;
                float v973;
                v973 = v933[v972];
                float v974;
                v974 = v973 / v965;
                assert("Tensor range check" && 0 <= v967 && v967 < 1);
                assert("Tensor range check" && 0 <= v969 && v969 < 4);
                v966[v972] = v974;
                v969 += 1 ;
            }
            v967 += 1 ;
        }
        assert("Tensor range check" && 0 <= v824 && v824 < 8);
        int v975;
        v975 = 0;
        while (while_method_3(v975)){
            assert("Tensor range check" && 0 <= v975 && v975 < 1);
            int v977;
            v977 = 128 * v975;
            int v978;
            v978 = v977 + v827;
            assert("Tensor range check" && 0 <= v975 && v975 < 1);
            int v979;
            v979 = 4 * v975;
            int4* v980;
            v980 = reinterpret_cast<int4*>(v966 + v979);
            int4* v981;
            v981 = reinterpret_cast<int4*>(v5 + v978);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v980) % 16 == 0 && reinterpret_cast<unsigned long long>(v981) % 16 == 0);
            *v981 = *v980;
            v975 += 1 ;
        }
        v824 += 1 ;
    }
    __syncthreads();
    int v982;
    v982 = threadIdx.x;
    int v983;
    v983 = blockIdx.x;
    int v984;
    v984 = v983 * 256;
    int v985;
    v985 = v982 + v984;
    unsigned long long v986;
    v986 = (unsigned long long)v985;
    curandStatePhilox4_32_10_t v987;
    curand_init(12344321ull,v986,0ull,&v987);
    int v988;
    v988 = threadIdx.x;
    bool v989;
    v989 = 0 <= v988;
    bool v990;
    v990 = v989 == false;
    if (v990){
        assert("The index needs to be zero or positive." && v989);
    } else {
    }
    int v992;
    v992 = v988 % 32;
    int v993;
    v993 = v988 / 32;
    bool v994;
    v994 = v993 < 8;
    bool v995;
    v995 = v994 == false;
    if (v995){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v994);
    } else {
    }
    assert("Tensor range check" && 0 <= v993 && v993 < 8);
    assert("Tensor range check" && 0 <= v992 && v992 < 32);
    int v997;
    v997 = 4 * v992;
    int v998;
    v998 = 128 * v993;
    int v999;
    v999 = v998 + v997;
    assert("Tensor range check" && 0 <= v993 && v993 < 8);
    assert("Tensor range check" && 0 <= v992 && v992 < 32);
    assert("Tensor range check" && 0 <= v993 && v993 < 8);
    int v1000;
    v1000 = 0;
    while (while_method_2(v1000)){
        assert("Tensor range check" && 0 <= v1000 && v1000 < 8);
        int v1002;
        v1002 = 1024 * v1000;
        int v1003;
        v1003 = v1002 + v999;
        float v1004[4];
        int v1005[4];
        int v1006;
        v1006 = 0;
        while (while_method_3(v1006)){
            assert("Tensor range check" && 0 <= v1006 && v1006 < 1);
            int v1008;
            v1008 = 4 * v1006;
            assert("Tensor range check" && 0 <= v1006 && v1006 < 1);
            int v1009;
            v1009 = 128 * v1006;
            int v1010;
            v1010 = v1009 + v1003;
            int4* v1011;
            v1011 = reinterpret_cast<int4*>(v1 + v1010);
            int4* v1012;
            v1012 = reinterpret_cast<int4*>(v1004 + v1008);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1011) % 16 == 0 && reinterpret_cast<unsigned long long>(v1012) % 16 == 0);
            *v1012 = *v1011;
            v1006 += 1 ;
        }
        int v1013;
        v1013 = 0;
        while (while_method_3(v1013)){
            int v1015;
            v1015 = 0;
            while (while_method_1(v1015)){
                bool v1017;
                v1017 = 0 <= v1015;
                bool v1019;
                if (v1017){
                    bool v1018;
                    v1018 = v1015 < 4;
                    v1019 = v1018;
                } else {
                    v1019 = false;
                }
                bool v1020;
                v1020 = v1019 == false;
                if (v1020){
                    assert("The indices should be inside the range of the dimension." && v1019);
                } else {
                }
                bool v1022;
                v1022 = 0 <= v992;
                bool v1024;
                if (v1022){
                    bool v1023;
                    v1023 = v992 < 32;
                    v1024 = v1023;
                } else {
                    v1024 = false;
                }
                bool v1025;
                v1025 = v1024 == false;
                if (v1025){
                    assert("The indices should be inside the range of the dimension." && v1024);
                } else {
                }
                int v1027;
                v1027 = v992 * 4;
                int v1028;
                v1028 = v1015 + v1027;
                bool v1029;
                v1029 = 0 <= v1013;
                bool v1031;
                if (v1029){
                    bool v1030;
                    v1030 = v1013 < 1;
                    v1031 = v1030;
                } else {
                    v1031 = false;
                }
                bool v1032;
                v1032 = v1031 == false;
                if (v1032){
                    assert("The indices should be inside the range of the dimension." && v1031);
                } else {
                }
                int v1034;
                v1034 = v1013 * 128;
                int v1035;
                v1035 = v1028 + v1034;
                assert("Tensor range check" && 0 <= v1013 && v1013 < 1);
                assert("Tensor range check" && 0 <= v1015 && v1015 < 4);
                int v1036;
                v1036 = 4 * v1013;
                int v1037;
                v1037 = v1036 + v1015;
                v1005[v1037] = v1035;
                v1015 += 1 ;
            }
            v1013 += 1 ;
        }
        bool v1038;
        v1038 = 0 <= v993;
        bool v1039;
        v1039 = v1038 && v994;
        bool v1040;
        v1040 = v1039 == false;
        if (v1040){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1039);
        } else {
        }
        bool v1042;
        v1042 = 0 <= v1000;
        bool v1044;
        if (v1042){
            bool v1043;
            v1043 = v1000 < 8;
            v1044 = v1043;
        } else {
            v1044 = false;
        }
        bool v1045;
        v1045 = v1044 == false;
        if (v1045){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1044);
        } else {
        }
        int v1047;
        v1047 = v1000 * 8;
        int v1048;
        v1048 = v1047 + v993;
        float v1049;
        v1049 = 0.0f;
        int v1050;
        v1050 = 0;
        while (while_method_3(v1050)){
            int v1052;
            v1052 = 0;
            while (while_method_1(v1052)){
                assert("Tensor range check" && 0 <= v1050 && v1050 < 1);
                assert("Tensor range check" && 0 <= v1052 && v1052 < 4);
                int v1054;
                v1054 = 4 * v1050;
                int v1055;
                v1055 = v1054 + v1052;
                float v1056;
                v1056 = v1004[v1055];
                float v1057;
                v1057 = v1049 + v1056;
                v1049 = v1057;
                v1052 += 1 ;
            }
            v1050 += 1 ;
        }
        auto v1058 = cooperative_groups::coalesced_threads();
        int v1059;
        v1059 = threadIdx.x;
        int v1060;
        v1060 = v1059 / 32;
        auto v1061 = cooperative_groups::labeled_partition(v1058,v1060);
        float v1062;
        v1062 = cooperative_groups::reduce(v1061, v1049, v42);
        float v1063;
        v1063 = v1062 / 128.0f;
        float v1064[4];
        int v1065;
        v1065 = 0;
        while (while_method_3(v1065)){
            int v1067;
            v1067 = 0;
            while (while_method_1(v1067)){
                assert("Tensor range check" && 0 <= v1065 && v1065 < 1);
                assert("Tensor range check" && 0 <= v1067 && v1067 < 4);
                int v1069;
                v1069 = 4 * v1065;
                int v1070;
                v1070 = v1069 + v1067;
                float v1071;
                v1071 = v1004[v1070];
                float v1072;
                v1072 = v1071 - v1063;
                float v1073;
                v1073 = exp(v1072);
                assert("Tensor range check" && 0 <= v1065 && v1065 < 1);
                assert("Tensor range check" && 0 <= v1067 && v1067 < 4);
                v1064[v1070] = v1073;
                v1067 += 1 ;
            }
            v1065 += 1 ;
        }
        float v1074;
        v1074 = 0.0f;
        int v1075;
        v1075 = 0;
        while (while_method_3(v1075)){
            int v1077;
            v1077 = 0;
            while (while_method_1(v1077)){
                assert("Tensor range check" && 0 <= v1075 && v1075 < 1);
                assert("Tensor range check" && 0 <= v1077 && v1077 < 4);
                int v1079;
                v1079 = 4 * v1075;
                int v1080;
                v1080 = v1079 + v1077;
                float v1081;
                v1081 = v1064[v1080];
                float v1082;
                v1082 = v1074 + v1081;
                v1074 = v1082;
                v1077 += 1 ;
            }
            v1075 += 1 ;
        }
        auto v1083 = cooperative_groups::coalesced_threads();
        int v1084;
        v1084 = threadIdx.x;
        int v1085;
        v1085 = v1084 / 32;
        auto v1086 = cooperative_groups::labeled_partition(v1083,v1085);
        float v1087;
        v1087 = cooperative_groups::reduce(v1086, v1074, v42);
        float v1088[4];
        int v1089;
        v1089 = 0;
        while (while_method_3(v1089)){
            int v1091;
            v1091 = 0;
            while (while_method_1(v1091)){
                assert("Tensor range check" && 0 <= v1089 && v1089 < 1);
                assert("Tensor range check" && 0 <= v1091 && v1091 < 4);
                int v1093;
                v1093 = 4 * v1089;
                int v1094;
                v1094 = v1093 + v1091;
                float v1095;
                v1095 = v1064[v1094];
                float v1096;
                v1096 = v1095 / v1087;
                assert("Tensor range check" && 0 <= v1089 && v1089 < 1);
                assert("Tensor range check" && 0 <= v1091 && v1091 < 4);
                v1088[v1094] = v1096;
                v1091 += 1 ;
            }
            v1089 += 1 ;
        }
        float v1097[4];
        float v1098;
        v1098 = 0.0f;
        int v1099;
        v1099 = 0;
        while (while_method_3(v1099)){
            assert("Tensor range check" && 0 <= v1099 && v1099 < 1);
            int v1101;
            v1101 = 4 * v1099;
            assert("Tensor range check" && 0 <= v1099 && v1099 < 1);
            float v1102;
            v1102 = 0.0f;
            int v1103;
            v1103 = 0;
            while (while_method_1(v1103)){
                assert("Tensor range check" && 0 <= v1103 && v1103 < 4);
                int v1105;
                v1105 = v1103 + v1101;
                float v1106;
                v1106 = v1088[v1105];
                float v1107;
                v1107 = v1102 + v1106;
                v1102 = v1107;
                v1103 += 1 ;
            }
            auto v1108 = cooperative_groups::coalesced_threads();
            int v1109;
            v1109 = threadIdx.x;
            int v1110;
            v1110 = v1109 / 32;
            auto v1111 = cooperative_groups::labeled_partition(v1108,v1110);
            Closure2 v1112{};
            float v1113;
            v1113 = cooperative_groups::inclusive_scan(v1111, v1102, v1112);
            float v1114;
            v1114 = v1111.shfl_up(v1113,1);
            bool v1115;
            v1115 = v1111.thread_rank() == 0;
            float v1116;
            if (v1115){
                v1116 = 0.0f;
            } else {
                v1116 = v1114;
            }
            float v1117;
            v1117 = v1111.shfl(v1113,v1111.num_threads()-1);
            float v1118;
            v1118 = v1098 + v1116;
            float v1119;
            v1119 = v1118;
            int v1120;
            v1120 = 0;
            while (while_method_1(v1120)){
                assert("Tensor range check" && 0 <= v1120 && v1120 < 4);
                int v1122;
                v1122 = v1120 + v1101;
                float v1123;
                v1123 = v1088[v1122];
                float v1124;
                v1124 = v1119 + v1123;
                assert("Tensor range check" && 0 <= v1120 && v1120 < 4);
                v1097[v1122] = v1124;
                v1119 = v1124;
                v1120 += 1 ;
            }
            float v1125;
            v1125 = v1098 + v1117;
            v1098 = v1125;
            v1099 += 1 ;
        }
        float v1126[4];
        bool v1127[4];
        int v1128;
        v1128 = 0;
        while (while_method_3(v1128)){
            int v1130;
            v1130 = 0;
            while (while_method_1(v1130)){
                assert("Tensor range check" && 0 <= v1128 && v1128 < 1);
                assert("Tensor range check" && 0 <= v1130 && v1130 < 4);
                int v1132;
                v1132 = 4 * v1128;
                int v1133;
                v1133 = v1132 + v1130;
                float v1134;
                v1134 = v1097[v1133];
                float v1135;
                v1135 = v1088[v1133];
                bool v1136;
                v1136 = v1135 > 0.0f;
                assert("Tensor range check" && 0 <= v1128 && v1128 < 1);
                assert("Tensor range check" && 0 <= v1130 && v1130 < 4);
                v1126[v1133] = v1134;
                v1127[v1133] = v1136;
                v1130 += 1 ;
            }
            v1128 += 1 ;
        }
        float v1137; bool v1138;
        Tuple2 tmp3 = Tuple2{-1.0f / 0.0f, false};
        v1137 = tmp3.v0; v1138 = tmp3.v1;
        int v1139;
        v1139 = 0;
        while (while_method_3(v1139)){
            int v1141;
            v1141 = 0;
            while (while_method_1(v1141)){
                assert("Tensor range check" && 0 <= v1139 && v1139 < 1);
                assert("Tensor range check" && 0 <= v1141 && v1141 < 4);
                int v1143;
                v1143 = 4 * v1139;
                int v1144;
                v1144 = v1143 + v1141;
                float v1145;
                v1145 = v1126[v1144];
                bool v1146;
                v1146 = v1127[v1144];
                float v1153; bool v1154;
                if (v1138){
                    if (v1146){
                        bool v1147;
                        v1147 = v1137 >= v1145;
                        float v1148;
                        if (v1147){
                            v1148 = v1137;
                        } else {
                            v1148 = v1145;
                        }
                        v1153 = v1148; v1154 = true;
                    } else {
                        v1153 = v1137; v1154 = v1138;
                    }
                } else {
                    if (v1146){
                        v1153 = v1145; v1154 = v1146;
                    } else {
                        v1153 = v1137; v1154 = v1138;
                    }
                }
                v1137 = v1153;
                v1138 = v1154;
                v1141 += 1 ;
            }
            v1139 += 1 ;
        }
        auto v1155 = cooperative_groups::coalesced_threads();
        int v1156;
        v1156 = threadIdx.x;
        int v1157;
        v1157 = v1156 / 32;
        auto v1158 = cooperative_groups::labeled_partition(v1155,v1157);
        Closure5 v1159{};
        float v1160; bool v1161;
        Tuple2 tmp4 = cooperative_groups::reduce(v1158, Tuple2{v1137, v1138}, v1159);
        v1160 = tmp4.v0; v1161 = tmp4.v1;
        bool v1162;
        v1162 = v1161 == false;
        if (v1162){
            assert("The local reduce must be true." && v1161);
        } else {
        }
        float v1164[4];
        int v1165[4];
        int v1166;
        v1166 = 0;
        while (while_method_3(v1166)){
            int v1168;
            v1168 = 0;
            while (while_method_1(v1168)){
                assert("Tensor range check" && 0 <= v1166 && v1166 < 1);
                assert("Tensor range check" && 0 <= v1168 && v1168 < 4);
                int v1170;
                v1170 = 4 * v1166;
                int v1171;
                v1171 = v1170 + v1168;
                int v1172;
                v1172 = v1005[v1171];
                float v1173;
                v1173 = curand_uniform(&v987);
                assert("Tensor range check" && 0 <= v1166 && v1166 < 1);
                assert("Tensor range check" && 0 <= v1168 && v1168 < 4);
                v1164[v1171] = v1173;
                v1165[v1171] = v1172;
                v1168 += 1 ;
            }
            v1166 += 1 ;
        }
        int v1174;
        v1174 = blockIdx.x;
        bool v1175;
        v1175 = v1174 == 0;
        bool v1178;
        if (v1175){
            int v1176;
            v1176 = threadIdx.x;
            bool v1177;
            v1177 = v1176 < 8;
            v1178 = v1177;
        } else {
            v1178 = false;
        }
        if (v1178){
            int v1179;
            v1179 = threadIdx.x;
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v1180 = console_lock;
            auto v1181 = cooperative_groups::coalesced_threads();
            v1180.acquire();
            int v1182;
            v1182 = 0;
            printf("{%s = %d; %s = %c","tid", v1179, "x", '[');
            int v1183;
            v1183 = 0;
            while (while_method_3(v1183)){
                int v1185;
                v1185 = v1182;
                bool v1186;
                v1186 = v1185 >= 100;
                if (v1186){
                    printf("%s"," ...");
                    break;
                } else {
                }
                bool v1187;
                v1187 = v1183 == 0;
                bool v1188;
                v1188 = v1187 != true;
                if (v1188){
                    printf("%s","; ");
                } else {
                }
                printf("%c",'[');
                int v1189;
                v1189 = 0;
                while (while_method_1(v1189)){
                    int v1191;
                    v1191 = v1182;
                    bool v1192;
                    v1192 = v1191 >= 100;
                    if (v1192){
                        printf("%s"," ...");
                        break;
                    } else {
                    }
                    bool v1193;
                    v1193 = v1189 == 0;
                    bool v1194;
                    v1194 = v1193 != true;
                    if (v1194){
                        printf("%s","; ");
                    } else {
                    }
                    int v1195;
                    v1195 = v1182 + 1;
                    v1182 = v1195;
                    int v1196;
                    v1196 = v1183 * 4;
                    int v1197;
                    v1197 = v1196 + v1189;
                    float v1198;
                    v1198 = v1164[v1197];
                    int v1199;
                    v1199 = v1165[v1197];
                    printf("%f, %d",v1198, v1199);
                    v1189 += 1 ;
                }
                printf("%c",']');
                v1183 += 1 ;
            }
            printf("%c",']');
            printf("}\n");
            v1180.release();
            v1181.sync() ;
        } else {
        }
        __syncthreads();
        float v1231; int v1232;
        Tuple1 tmp5 = Tuple1{0.0f, 2147483647};
        v1231 = tmp5.v0; v1232 = tmp5.v1;
        int v1233;
        v1233 = 0;
        while (while_method_3(v1233)){
            int v1235;
            v1235 = 0;
            while (while_method_1(v1235)){
                assert("Tensor range check" && 0 <= v1233 && v1233 < 1);
                assert("Tensor range check" && 0 <= v1235 && v1235 < 4);
                int v1237;
                v1237 = 4 * v1233;
                int v1238;
                v1238 = v1237 + v1235;
                float v1239;
                v1239 = v1164[v1238];
                int v1240;
                v1240 = v1165[v1238];
                bool v1241;
                v1241 = v1232 < v1240;
                float v1242; int v1243;
                if (v1241){
                    v1242 = v1231; v1243 = v1232;
                } else {
                    v1242 = v1239; v1243 = v1240;
                }
                v1231 = v1242;
                v1232 = v1243;
                v1235 += 1 ;
            }
            v1233 += 1 ;
        }
        auto v1244 = cooperative_groups::coalesced_threads();
        int v1245;
        v1245 = threadIdx.x;
        int v1246;
        v1246 = v1245 / 32;
        auto v1247 = cooperative_groups::labeled_partition(v1244,v1246);
        Closure6 v1248{};
        float v1249; int v1250;
        Tuple1 tmp6 = cooperative_groups::reduce(v1247, Tuple1{v1231, v1232}, v1248);
        v1249 = tmp6.v0; v1250 = tmp6.v1;
        float v1251;
        v1251 = v1160 * v1249;
        int v1252[4];
        bool v1253[4];
        int v1254;
        v1254 = 0;
        while (while_method_3(v1254)){
            int v1256;
            v1256 = 0;
            while (while_method_1(v1256)){
                assert("Tensor range check" && 0 <= v1254 && v1254 < 1);
                assert("Tensor range check" && 0 <= v1256 && v1256 < 4);
                int v1258;
                v1258 = 4 * v1254;
                int v1259;
                v1259 = v1258 + v1256;
                float v1260;
                v1260 = v1126[v1259];
                bool v1261;
                v1261 = v1127[v1259];
                int v1262;
                v1262 = v1005[v1259];
                int v1265; bool v1266;
                if (v1261){
                    float v1263;
                    v1263 = v1260 - v1251;
                    bool v1264;
                    v1264 = v1263 >= 0.0f;
                    v1265 = v1262; v1266 = v1264;
                } else {
                    v1265 = 2147483647; v1266 = false;
                }
                assert("Tensor range check" && 0 <= v1254 && v1254 < 1);
                assert("Tensor range check" && 0 <= v1256 && v1256 < 4);
                v1252[v1259] = v1265;
                v1253[v1259] = v1266;
                v1256 += 1 ;
            }
            v1254 += 1 ;
        }
        int v1267; bool v1268;
        Tuple3 tmp7 = Tuple3{2147483647, false};
        v1267 = tmp7.v0; v1268 = tmp7.v1;
        int v1269;
        v1269 = 0;
        while (while_method_3(v1269)){
            int v1271;
            v1271 = 0;
            while (while_method_1(v1271)){
                assert("Tensor range check" && 0 <= v1269 && v1269 < 1);
                assert("Tensor range check" && 0 <= v1271 && v1271 < 4);
                int v1273;
                v1273 = 4 * v1269;
                int v1274;
                v1274 = v1273 + v1271;
                int v1275;
                v1275 = v1252[v1274];
                bool v1276;
                v1276 = v1253[v1274];
                int v1283; bool v1284;
                if (v1268){
                    if (v1276){
                        bool v1277;
                        v1277 = v1267 < v1275;
                        int v1278;
                        if (v1277){
                            v1278 = v1267;
                        } else {
                            v1278 = v1275;
                        }
                        v1283 = v1278; v1284 = true;
                    } else {
                        v1283 = v1267; v1284 = v1268;
                    }
                } else {
                    if (v1276){
                        v1283 = v1275; v1284 = v1276;
                    } else {
                        v1283 = v1267; v1284 = v1268;
                    }
                }
                v1267 = v1283;
                v1268 = v1284;
                v1271 += 1 ;
            }
            v1269 += 1 ;
        }
        auto v1285 = cooperative_groups::coalesced_threads();
        int v1286;
        v1286 = threadIdx.x;
        int v1287;
        v1287 = v1286 / 32;
        auto v1288 = cooperative_groups::labeled_partition(v1285,v1287);
        Closure7 v1289{};
        int v1290; bool v1291;
        Tuple3 tmp8 = cooperative_groups::reduce(v1288, Tuple3{v1267, v1268}, v1289);
        v1290 = tmp8.v0; v1291 = tmp8.v1;
        bool v1292;
        v1292 = v1291 == false;
        if (v1292){
            assert("The local reduce must be true." && v1291);
        } else {
        }
        assert("Tensor range check" && 0 <= v1000 && v1000 < 8);
        int v1294;
        v1294 = 0;
        while (while_method_3(v1294)){
            assert("Tensor range check" && 0 <= v1294 && v1294 < 1);
            int v1296;
            v1296 = 128 * v1294;
            int v1297;
            v1297 = v1296 + v1003;
            assert("Tensor range check" && 0 <= v1294 && v1294 < 1);
            int v1298;
            v1298 = 4 * v1294;
            int4* v1299;
            v1299 = reinterpret_cast<int4*>(v1088 + v1298);
            int4* v1300;
            v1300 = reinterpret_cast<int4*>(v14 + v1297);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1299) % 16 == 0 && reinterpret_cast<unsigned long long>(v1300) % 16 == 0);
            *v1300 = *v1299;
            v1294 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1000 && v1000 < 8);
        int v1301;
        v1301 = 8 * v1000;
        int v1302;
        v1302 = v1301 + v993;
        v15[v1302] = v1290;
        v1000 += 1 ;
    }
    __syncthreads();
    int v1303;
    v1303 = threadIdx.x;
    int v1304;
    v1304 = blockIdx.x;
    int v1305;
    v1305 = v1304 * 256;
    int v1306;
    v1306 = v1303 + v1305;
    unsigned long long v1307;
    v1307 = (unsigned long long)v1306;
    curandStatePhilox4_32_10_t v1308;
    curand_init(12344321ull,v1307,0ull,&v1308);
    int v1309;
    v1309 = threadIdx.x;
    bool v1310;
    v1310 = 0 <= v1309;
    bool v1311;
    v1311 = v1310 == false;
    if (v1311){
        assert("The index needs to be zero or positive." && v1310);
    } else {
    }
    int v1313;
    v1313 = v1309 % 32;
    int v1314;
    v1314 = v1309 / 32;
    bool v1315;
    v1315 = v1314 < 8;
    bool v1316;
    v1316 = v1315 == false;
    if (v1316){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1315);
    } else {
    }
    assert("Tensor range check" && 0 <= v1314 && v1314 < 8);
    assert("Tensor range check" && 0 <= v1313 && v1313 < 32);
    int v1318;
    v1318 = 4 * v1313;
    int v1319;
    v1319 = 128 * v1314;
    int v1320;
    v1320 = v1319 + v1318;
    assert("Tensor range check" && 0 <= v1314 && v1314 < 8);
    assert("Tensor range check" && 0 <= v1313 && v1313 < 32);
    assert("Tensor range check" && 0 <= v1314 && v1314 < 8);
    int v1321;
    v1321 = 0;
    while (while_method_2(v1321)){
        assert("Tensor range check" && 0 <= v1321 && v1321 < 8);
        int v1323;
        v1323 = 1024 * v1321;
        int v1324;
        v1324 = v1323 + v1320;
        float v1325[4];
        int v1326[4];
        int v1327;
        v1327 = 0;
        while (while_method_3(v1327)){
            assert("Tensor range check" && 0 <= v1327 && v1327 < 1);
            int v1329;
            v1329 = 4 * v1327;
            assert("Tensor range check" && 0 <= v1327 && v1327 < 1);
            int v1330;
            v1330 = 128 * v1327;
            int v1331;
            v1331 = v1330 + v1324;
            int4* v1332;
            v1332 = reinterpret_cast<int4*>(v1 + v1331);
            int4* v1333;
            v1333 = reinterpret_cast<int4*>(v1325 + v1329);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1332) % 16 == 0 && reinterpret_cast<unsigned long long>(v1333) % 16 == 0);
            *v1333 = *v1332;
            v1327 += 1 ;
        }
        int v1334;
        v1334 = 0;
        while (while_method_3(v1334)){
            int v1336;
            v1336 = 0;
            while (while_method_1(v1336)){
                bool v1338;
                v1338 = 0 <= v1336;
                bool v1340;
                if (v1338){
                    bool v1339;
                    v1339 = v1336 < 4;
                    v1340 = v1339;
                } else {
                    v1340 = false;
                }
                bool v1341;
                v1341 = v1340 == false;
                if (v1341){
                    assert("The indices should be inside the range of the dimension." && v1340);
                } else {
                }
                bool v1343;
                v1343 = 0 <= v1313;
                bool v1345;
                if (v1343){
                    bool v1344;
                    v1344 = v1313 < 32;
                    v1345 = v1344;
                } else {
                    v1345 = false;
                }
                bool v1346;
                v1346 = v1345 == false;
                if (v1346){
                    assert("The indices should be inside the range of the dimension." && v1345);
                } else {
                }
                int v1348;
                v1348 = v1313 * 4;
                int v1349;
                v1349 = v1336 + v1348;
                bool v1350;
                v1350 = 0 <= v1334;
                bool v1352;
                if (v1350){
                    bool v1351;
                    v1351 = v1334 < 1;
                    v1352 = v1351;
                } else {
                    v1352 = false;
                }
                bool v1353;
                v1353 = v1352 == false;
                if (v1353){
                    assert("The indices should be inside the range of the dimension." && v1352);
                } else {
                }
                int v1355;
                v1355 = v1334 * 128;
                int v1356;
                v1356 = v1349 + v1355;
                assert("Tensor range check" && 0 <= v1334 && v1334 < 1);
                assert("Tensor range check" && 0 <= v1336 && v1336 < 4);
                int v1357;
                v1357 = 4 * v1334;
                int v1358;
                v1358 = v1357 + v1336;
                v1326[v1358] = v1356;
                v1336 += 1 ;
            }
            v1334 += 1 ;
        }
        bool v1359;
        v1359 = 0 <= v1314;
        bool v1360;
        v1360 = v1359 && v1315;
        bool v1361;
        v1361 = v1360 == false;
        if (v1361){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1360);
        } else {
        }
        bool v1363;
        v1363 = 0 <= v1321;
        bool v1365;
        if (v1363){
            bool v1364;
            v1364 = v1321 < 8;
            v1365 = v1364;
        } else {
            v1365 = false;
        }
        bool v1366;
        v1366 = v1365 == false;
        if (v1366){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1365);
        } else {
        }
        int v1368;
        v1368 = v1321 * 8;
        int v1369;
        v1369 = v1368 + v1314;
        bool v1370[4];
        int v1371;
        v1371 = 0;
        while (while_method_3(v1371)){
            int v1373;
            v1373 = 0;
            while (while_method_1(v1373)){
                assert("Tensor range check" && 0 <= v1371 && v1371 < 1);
                assert("Tensor range check" && 0 <= v1373 && v1373 < 4);
                int v1375;
                v1375 = 4 * v1371;
                int v1376;
                v1376 = v1375 + v1373;
                float v1377;
                v1377 = v1325[v1376];
                int v1378;
                v1378 = v1326[v1376];
                bool v1379;
                v1379 = v1378 < 11;
                assert("Tensor range check" && 0 <= v1371 && v1371 < 1);
                assert("Tensor range check" && 0 <= v1373 && v1373 < 4);
                v1370[v1376] = v1379;
                v1373 += 1 ;
            }
            v1371 += 1 ;
        }
        float v1380[4];
        int v1381;
        v1381 = 0;
        while (while_method_3(v1381)){
            int v1383;
            v1383 = 0;
            while (while_method_1(v1383)){
                assert("Tensor range check" && 0 <= v1381 && v1381 < 1);
                assert("Tensor range check" && 0 <= v1383 && v1383 < 4);
                int v1385;
                v1385 = 4 * v1381;
                int v1386;
                v1386 = v1385 + v1383;
                float v1387;
                v1387 = v1325[v1386];
                bool v1388;
                v1388 = v1370[v1386];
                float v1389;
                if (v1388){
                    v1389 = v1387;
                } else {
                    v1389 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1381 && v1381 < 1);
                assert("Tensor range check" && 0 <= v1383 && v1383 < 4);
                v1380[v1386] = v1389;
                v1383 += 1 ;
            }
            v1381 += 1 ;
        }
        float v1390;
        v1390 = 0.0f;
        int v1391;
        v1391 = 0;
        while (while_method_3(v1391)){
            int v1393;
            v1393 = 0;
            while (while_method_1(v1393)){
                assert("Tensor range check" && 0 <= v1391 && v1391 < 1);
                assert("Tensor range check" && 0 <= v1393 && v1393 < 4);
                int v1395;
                v1395 = 4 * v1391;
                int v1396;
                v1396 = v1395 + v1393;
                float v1397;
                v1397 = v1380[v1396];
                float v1398;
                v1398 = v1390 + v1397;
                v1390 = v1398;
                v1393 += 1 ;
            }
            v1391 += 1 ;
        }
        auto v1399 = cooperative_groups::coalesced_threads();
        int v1400;
        v1400 = threadIdx.x;
        int v1401;
        v1401 = v1400 / 32;
        auto v1402 = cooperative_groups::labeled_partition(v1399,v1401);
        float v1403;
        v1403 = cooperative_groups::reduce(v1402, v1390, v42);
        int v1404[4];
        int v1405;
        v1405 = 0;
        while (while_method_3(v1405)){
            int v1407;
            v1407 = 0;
            while (while_method_1(v1407)){
                assert("Tensor range check" && 0 <= v1405 && v1405 < 1);
                assert("Tensor range check" && 0 <= v1407 && v1407 < 4);
                int v1409;
                v1409 = 4 * v1405;
                int v1410;
                v1410 = v1409 + v1407;
                bool v1411;
                v1411 = v1370[v1410];
                int v1412;
                if (v1411){
                    v1412 = 1;
                } else {
                    v1412 = 0;
                }
                assert("Tensor range check" && 0 <= v1405 && v1405 < 1);
                assert("Tensor range check" && 0 <= v1407 && v1407 < 4);
                v1404[v1410] = v1412;
                v1407 += 1 ;
            }
            v1405 += 1 ;
        }
        int v1413;
        v1413 = 0;
        int v1414;
        v1414 = 0;
        while (while_method_3(v1414)){
            int v1416;
            v1416 = 0;
            while (while_method_1(v1416)){
                assert("Tensor range check" && 0 <= v1414 && v1414 < 1);
                assert("Tensor range check" && 0 <= v1416 && v1416 < 4);
                int v1418;
                v1418 = 4 * v1414;
                int v1419;
                v1419 = v1418 + v1416;
                int v1420;
                v1420 = v1404[v1419];
                int v1421;
                v1421 = v1413 + v1420;
                v1413 = v1421;
                v1416 += 1 ;
            }
            v1414 += 1 ;
        }
        auto v1422 = cooperative_groups::coalesced_threads();
        int v1423;
        v1423 = threadIdx.x;
        int v1424;
        v1424 = v1423 / 32;
        auto v1425 = cooperative_groups::labeled_partition(v1422,v1424);
        Closure4 v1426{};
        int v1427;
        v1427 = cooperative_groups::reduce(v1425, v1413, v1426);
        float v1428;
        v1428 = (float)v1427;
        float v1429;
        v1429 = v1403 / v1428;
        float v1430[4];
        int v1431;
        v1431 = 0;
        while (while_method_3(v1431)){
            int v1433;
            v1433 = 0;
            while (while_method_1(v1433)){
                assert("Tensor range check" && 0 <= v1431 && v1431 < 1);
                assert("Tensor range check" && 0 <= v1433 && v1433 < 4);
                int v1435;
                v1435 = 4 * v1431;
                int v1436;
                v1436 = v1435 + v1433;
                float v1437;
                v1437 = v1325[v1436];
                bool v1438;
                v1438 = v1370[v1436];
                float v1439;
                if (v1438){
                    v1439 = v1437;
                } else {
                    v1439 = -1.0f / 0.0f;
                }
                float v1440;
                v1440 = v1439 - v1429;
                float v1441;
                v1441 = exp(v1440);
                bool v1442;
                v1442 = v1441 < 1.0f / 0.0f;
                bool v1443;
                v1443 = v1442 == false;
                if (v1443){
                    assert("The softmax values must not grow too large." && v1442);
                } else {
                }
                bool v1445;
                v1445 = isnan(v1441);
                bool v1446;
                v1446 = v1445 == false;
                bool v1447;
                v1447 = v1446 == false;
                if (v1447){
                    assert("The softmax values must not be nans." && v1446);
                } else {
                }
                assert("Tensor range check" && 0 <= v1431 && v1431 < 1);
                assert("Tensor range check" && 0 <= v1433 && v1433 < 4);
                v1430[v1436] = v1441;
                v1433 += 1 ;
            }
            v1431 += 1 ;
        }
        float v1449;
        v1449 = 0.0f;
        int v1450;
        v1450 = 0;
        while (while_method_3(v1450)){
            int v1452;
            v1452 = 0;
            while (while_method_1(v1452)){
                assert("Tensor range check" && 0 <= v1450 && v1450 < 1);
                assert("Tensor range check" && 0 <= v1452 && v1452 < 4);
                int v1454;
                v1454 = 4 * v1450;
                int v1455;
                v1455 = v1454 + v1452;
                float v1456;
                v1456 = v1430[v1455];
                float v1457;
                v1457 = v1449 + v1456;
                v1449 = v1457;
                v1452 += 1 ;
            }
            v1450 += 1 ;
        }
        auto v1458 = cooperative_groups::coalesced_threads();
        int v1459;
        v1459 = threadIdx.x;
        int v1460;
        v1460 = v1459 / 32;
        auto v1461 = cooperative_groups::labeled_partition(v1458,v1460);
        float v1462;
        v1462 = cooperative_groups::reduce(v1461, v1449, v42);
        float v1463[4];
        int v1464;
        v1464 = 0;
        while (while_method_3(v1464)){
            int v1466;
            v1466 = 0;
            while (while_method_1(v1466)){
                assert("Tensor range check" && 0 <= v1464 && v1464 < 1);
                assert("Tensor range check" && 0 <= v1466 && v1466 < 4);
                int v1468;
                v1468 = 4 * v1464;
                int v1469;
                v1469 = v1468 + v1466;
                float v1470;
                v1470 = v1430[v1469];
                float v1471;
                v1471 = v1470 / v1462;
                assert("Tensor range check" && 0 <= v1464 && v1464 < 1);
                assert("Tensor range check" && 0 <= v1466 && v1466 < 4);
                v1463[v1469] = v1471;
                v1466 += 1 ;
            }
            v1464 += 1 ;
        }
        float v1472[4];
        float v1473;
        v1473 = 0.0f;
        int v1474;
        v1474 = 0;
        while (while_method_3(v1474)){
            assert("Tensor range check" && 0 <= v1474 && v1474 < 1);
            int v1476;
            v1476 = 4 * v1474;
            assert("Tensor range check" && 0 <= v1474 && v1474 < 1);
            float v1477;
            v1477 = 0.0f;
            int v1478;
            v1478 = 0;
            while (while_method_1(v1478)){
                assert("Tensor range check" && 0 <= v1478 && v1478 < 4);
                int v1480;
                v1480 = v1478 + v1476;
                float v1481;
                v1481 = v1463[v1480];
                float v1482;
                v1482 = v1477 + v1481;
                v1477 = v1482;
                v1478 += 1 ;
            }
            auto v1483 = cooperative_groups::coalesced_threads();
            int v1484;
            v1484 = threadIdx.x;
            int v1485;
            v1485 = v1484 / 32;
            auto v1486 = cooperative_groups::labeled_partition(v1483,v1485);
            Closure2 v1487{};
            float v1488;
            v1488 = cooperative_groups::inclusive_scan(v1486, v1477, v1487);
            float v1489;
            v1489 = v1486.shfl_up(v1488,1);
            bool v1490;
            v1490 = v1486.thread_rank() == 0;
            float v1491;
            if (v1490){
                v1491 = 0.0f;
            } else {
                v1491 = v1489;
            }
            float v1492;
            v1492 = v1486.shfl(v1488,v1486.num_threads()-1);
            float v1493;
            v1493 = v1473 + v1491;
            float v1494;
            v1494 = v1493;
            int v1495;
            v1495 = 0;
            while (while_method_1(v1495)){
                assert("Tensor range check" && 0 <= v1495 && v1495 < 4);
                int v1497;
                v1497 = v1495 + v1476;
                float v1498;
                v1498 = v1463[v1497];
                float v1499;
                v1499 = v1494 + v1498;
                assert("Tensor range check" && 0 <= v1495 && v1495 < 4);
                v1472[v1497] = v1499;
                v1494 = v1499;
                v1495 += 1 ;
            }
            float v1500;
            v1500 = v1473 + v1492;
            v1473 = v1500;
            v1474 += 1 ;
        }
        float v1501[4];
        bool v1502[4];
        int v1503;
        v1503 = 0;
        while (while_method_3(v1503)){
            int v1505;
            v1505 = 0;
            while (while_method_1(v1505)){
                assert("Tensor range check" && 0 <= v1503 && v1503 < 1);
                assert("Tensor range check" && 0 <= v1505 && v1505 < 4);
                int v1507;
                v1507 = 4 * v1503;
                int v1508;
                v1508 = v1507 + v1505;
                float v1509;
                v1509 = v1472[v1508];
                float v1510;
                v1510 = v1463[v1508];
                bool v1511;
                v1511 = v1510 > 0.0f;
                assert("Tensor range check" && 0 <= v1503 && v1503 < 1);
                assert("Tensor range check" && 0 <= v1505 && v1505 < 4);
                v1501[v1508] = v1509;
                v1502[v1508] = v1511;
                v1505 += 1 ;
            }
            v1503 += 1 ;
        }
        float v1512; bool v1513;
        Tuple2 tmp9 = Tuple2{-1.0f / 0.0f, false};
        v1512 = tmp9.v0; v1513 = tmp9.v1;
        int v1514;
        v1514 = 0;
        while (while_method_3(v1514)){
            int v1516;
            v1516 = 0;
            while (while_method_1(v1516)){
                assert("Tensor range check" && 0 <= v1514 && v1514 < 1);
                assert("Tensor range check" && 0 <= v1516 && v1516 < 4);
                int v1518;
                v1518 = 4 * v1514;
                int v1519;
                v1519 = v1518 + v1516;
                float v1520;
                v1520 = v1501[v1519];
                bool v1521;
                v1521 = v1502[v1519];
                float v1528; bool v1529;
                if (v1513){
                    if (v1521){
                        bool v1522;
                        v1522 = v1512 >= v1520;
                        float v1523;
                        if (v1522){
                            v1523 = v1512;
                        } else {
                            v1523 = v1520;
                        }
                        v1528 = v1523; v1529 = true;
                    } else {
                        v1528 = v1512; v1529 = v1513;
                    }
                } else {
                    if (v1521){
                        v1528 = v1520; v1529 = v1521;
                    } else {
                        v1528 = v1512; v1529 = v1513;
                    }
                }
                v1512 = v1528;
                v1513 = v1529;
                v1516 += 1 ;
            }
            v1514 += 1 ;
        }
        auto v1530 = cooperative_groups::coalesced_threads();
        int v1531;
        v1531 = threadIdx.x;
        int v1532;
        v1532 = v1531 / 32;
        auto v1533 = cooperative_groups::labeled_partition(v1530,v1532);
        Closure5 v1534{};
        float v1535; bool v1536;
        Tuple2 tmp10 = cooperative_groups::reduce(v1533, Tuple2{v1512, v1513}, v1534);
        v1535 = tmp10.v0; v1536 = tmp10.v1;
        bool v1537;
        v1537 = v1536 == false;
        if (v1537){
            assert("The local reduce must be true." && v1536);
        } else {
        }
        float v1539[4];
        int v1540[4];
        int v1541;
        v1541 = 0;
        while (while_method_3(v1541)){
            int v1543;
            v1543 = 0;
            while (while_method_1(v1543)){
                assert("Tensor range check" && 0 <= v1541 && v1541 < 1);
                assert("Tensor range check" && 0 <= v1543 && v1543 < 4);
                int v1545;
                v1545 = 4 * v1541;
                int v1546;
                v1546 = v1545 + v1543;
                int v1547;
                v1547 = v1326[v1546];
                float v1548;
                v1548 = curand_uniform(&v1308);
                assert("Tensor range check" && 0 <= v1541 && v1541 < 1);
                assert("Tensor range check" && 0 <= v1543 && v1543 < 4);
                v1539[v1546] = v1548;
                v1540[v1546] = v1547;
                v1543 += 1 ;
            }
            v1541 += 1 ;
        }
        int v1549;
        v1549 = blockIdx.x;
        bool v1550;
        v1550 = v1549 == 0;
        bool v1553;
        if (v1550){
            int v1551;
            v1551 = threadIdx.x;
            bool v1552;
            v1552 = v1551 < 8;
            v1553 = v1552;
        } else {
            v1553 = false;
        }
        if (v1553){
            int v1554;
            v1554 = threadIdx.x;
            cuda::counting_semaphore<cuda::thread_scope_system, 1> & v1555 = console_lock;
            auto v1556 = cooperative_groups::coalesced_threads();
            v1555.acquire();
            int v1557;
            v1557 = 0;
            printf("{%s = %d; %s = %c","tid", v1554, "x", '[');
            int v1558;
            v1558 = 0;
            while (while_method_3(v1558)){
                int v1560;
                v1560 = v1557;
                bool v1561;
                v1561 = v1560 >= 100;
                if (v1561){
                    printf("%s"," ...");
                    break;
                } else {
                }
                bool v1562;
                v1562 = v1558 == 0;
                bool v1563;
                v1563 = v1562 != true;
                if (v1563){
                    printf("%s","; ");
                } else {
                }
                printf("%c",'[');
                int v1564;
                v1564 = 0;
                while (while_method_1(v1564)){
                    int v1566;
                    v1566 = v1557;
                    bool v1567;
                    v1567 = v1566 >= 100;
                    if (v1567){
                        printf("%s"," ...");
                        break;
                    } else {
                    }
                    bool v1568;
                    v1568 = v1564 == 0;
                    bool v1569;
                    v1569 = v1568 != true;
                    if (v1569){
                        printf("%s","; ");
                    } else {
                    }
                    int v1570;
                    v1570 = v1557 + 1;
                    v1557 = v1570;
                    int v1571;
                    v1571 = v1558 * 4;
                    int v1572;
                    v1572 = v1571 + v1564;
                    float v1573;
                    v1573 = v1539[v1572];
                    int v1574;
                    v1574 = v1540[v1572];
                    printf("%f, %d",v1573, v1574);
                    v1564 += 1 ;
                }
                printf("%c",']');
                v1558 += 1 ;
            }
            printf("%c",']');
            printf("}\n");
            v1555.release();
            v1556.sync() ;
        } else {
        }
        __syncthreads();
        float v1606; int v1607;
        Tuple1 tmp11 = Tuple1{0.0f, 2147483647};
        v1606 = tmp11.v0; v1607 = tmp11.v1;
        int v1608;
        v1608 = 0;
        while (while_method_3(v1608)){
            int v1610;
            v1610 = 0;
            while (while_method_1(v1610)){
                assert("Tensor range check" && 0 <= v1608 && v1608 < 1);
                assert("Tensor range check" && 0 <= v1610 && v1610 < 4);
                int v1612;
                v1612 = 4 * v1608;
                int v1613;
                v1613 = v1612 + v1610;
                float v1614;
                v1614 = v1539[v1613];
                int v1615;
                v1615 = v1540[v1613];
                bool v1616;
                v1616 = v1607 < v1615;
                float v1617; int v1618;
                if (v1616){
                    v1617 = v1606; v1618 = v1607;
                } else {
                    v1617 = v1614; v1618 = v1615;
                }
                v1606 = v1617;
                v1607 = v1618;
                v1610 += 1 ;
            }
            v1608 += 1 ;
        }
        auto v1619 = cooperative_groups::coalesced_threads();
        int v1620;
        v1620 = threadIdx.x;
        int v1621;
        v1621 = v1620 / 32;
        auto v1622 = cooperative_groups::labeled_partition(v1619,v1621);
        Closure6 v1623{};
        float v1624; int v1625;
        Tuple1 tmp12 = cooperative_groups::reduce(v1622, Tuple1{v1606, v1607}, v1623);
        v1624 = tmp12.v0; v1625 = tmp12.v1;
        float v1626;
        v1626 = v1535 * v1624;
        int v1627[4];
        bool v1628[4];
        int v1629;
        v1629 = 0;
        while (while_method_3(v1629)){
            int v1631;
            v1631 = 0;
            while (while_method_1(v1631)){
                assert("Tensor range check" && 0 <= v1629 && v1629 < 1);
                assert("Tensor range check" && 0 <= v1631 && v1631 < 4);
                int v1633;
                v1633 = 4 * v1629;
                int v1634;
                v1634 = v1633 + v1631;
                float v1635;
                v1635 = v1501[v1634];
                bool v1636;
                v1636 = v1502[v1634];
                int v1637;
                v1637 = v1326[v1634];
                int v1640; bool v1641;
                if (v1636){
                    float v1638;
                    v1638 = v1635 - v1626;
                    bool v1639;
                    v1639 = v1638 >= 0.0f;
                    v1640 = v1637; v1641 = v1639;
                } else {
                    v1640 = 2147483647; v1641 = false;
                }
                assert("Tensor range check" && 0 <= v1629 && v1629 < 1);
                assert("Tensor range check" && 0 <= v1631 && v1631 < 4);
                v1627[v1634] = v1640;
                v1628[v1634] = v1641;
                v1631 += 1 ;
            }
            v1629 += 1 ;
        }
        int v1642; bool v1643;
        Tuple3 tmp13 = Tuple3{2147483647, false};
        v1642 = tmp13.v0; v1643 = tmp13.v1;
        int v1644;
        v1644 = 0;
        while (while_method_3(v1644)){
            int v1646;
            v1646 = 0;
            while (while_method_1(v1646)){
                assert("Tensor range check" && 0 <= v1644 && v1644 < 1);
                assert("Tensor range check" && 0 <= v1646 && v1646 < 4);
                int v1648;
                v1648 = 4 * v1644;
                int v1649;
                v1649 = v1648 + v1646;
                int v1650;
                v1650 = v1627[v1649];
                bool v1651;
                v1651 = v1628[v1649];
                int v1658; bool v1659;
                if (v1643){
                    if (v1651){
                        bool v1652;
                        v1652 = v1642 < v1650;
                        int v1653;
                        if (v1652){
                            v1653 = v1642;
                        } else {
                            v1653 = v1650;
                        }
                        v1658 = v1653; v1659 = true;
                    } else {
                        v1658 = v1642; v1659 = v1643;
                    }
                } else {
                    if (v1651){
                        v1658 = v1650; v1659 = v1651;
                    } else {
                        v1658 = v1642; v1659 = v1643;
                    }
                }
                v1642 = v1658;
                v1643 = v1659;
                v1646 += 1 ;
            }
            v1644 += 1 ;
        }
        auto v1660 = cooperative_groups::coalesced_threads();
        int v1661;
        v1661 = threadIdx.x;
        int v1662;
        v1662 = v1661 / 32;
        auto v1663 = cooperative_groups::labeled_partition(v1660,v1662);
        Closure7 v1664{};
        int v1665; bool v1666;
        Tuple3 tmp14 = cooperative_groups::reduce(v1663, Tuple3{v1642, v1643}, v1664);
        v1665 = tmp14.v0; v1666 = tmp14.v1;
        bool v1667;
        v1667 = v1666 == false;
        if (v1667){
            assert("The local reduce must be true." && v1666);
        } else {
        }
        assert("Tensor range check" && 0 <= v1321 && v1321 < 8);
        int v1669;
        v1669 = 0;
        while (while_method_3(v1669)){
            assert("Tensor range check" && 0 <= v1669 && v1669 < 1);
            int v1671;
            v1671 = 128 * v1669;
            int v1672;
            v1672 = v1671 + v1324;
            assert("Tensor range check" && 0 <= v1669 && v1669 < 1);
            int v1673;
            v1673 = 4 * v1669;
            int4* v1674;
            v1674 = reinterpret_cast<int4*>(v1463 + v1673);
            int4* v1675;
            v1675 = reinterpret_cast<int4*>(v16 + v1672);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1674) % 16 == 0 && reinterpret_cast<unsigned long long>(v1675) % 16 == 0);
            *v1675 = *v1674;
            v1669 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1321 && v1321 < 8);
        int v1676;
        v1676 = 8 * v1321;
        int v1677;
        v1677 = v1676 + v1314;
        v17[v1677] = v1665;
        v1321 += 1 ;
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
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

import sys
import pathlib
options = []
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=255')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : cp.ndarray, v9 : cp.ndarray, v10 : cp.ndarray, v11 : cp.ndarray, v12 : cp.ndarray, v13 : cp.ndarray, v14 : cp.ndarray, v15 : cp.ndarray, v16 : cp.ndarray, v17 : cp.ndarray) -> None:
    v18 = "test_text_outputs/primitives/"
    v19 = "test2/a/"
    v20 = "kernel_params.txt"
    v21 = pathlib.Path(v18,v19,v20)
    del v18, v19, v20
    v21.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v21),'w')
    del v21
    v22 = cp.cuda.Device().attributes['MultiProcessorCount']
    v23 = v22 == 24
    del v22
    v24 = v23 == False
    if v24:
        v25 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v23, v25
        del v25
    else:
        pass
    del v23, v24
    v26 = 0
    v27 = raw_module.get_function(f"entry{v26}")
    del v26
    v27.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v27((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v27
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method2(v0 : i32) -> bool:
    v1 = v0 < 64
    del v0
    return v1
def method3(v0 : i32) -> bool:
    v1 = v0 < 128
    del v0
    return v1
def method1(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "input.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method3(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method4(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "input_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method2(v32):
        v34 = v30
        v35 = v34 >= 1024
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
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
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method3(v40):
            v42 = v30
            v43 = v42 >= 1024
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 128
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method5(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v12 = 0
    v13 = v12 + 1
    v12 = v13
    del v12, v13
    v14 = v0[0].item()
    del v0
    v15 = "{:.6f}"
    print(v15.format(v14),end="")
    del v14, v15
    v16 = "\n"
    print(v16.format(),end="")
    del v16
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method6(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_softmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method3(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method7(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_masked_softmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method3(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method8(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_ln.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method3(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method9(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_argmax.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
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
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method10(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test2/a/"
    v4 = "output_softmax_scan.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method2(v35):
        v37 = v33
        v38 = v37 >= 8192
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method3(v43):
            v45 = v33
            v46 = v45 >= 8192
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 128
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{:.6f}, {:.6f}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method11(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method2(v32):
        v34 = v30
        v35 = v34 >= 8192
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
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
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method3(v40):
            v42 = v30
            v43 = v42 >= 8192
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 128
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method12(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test2/a/"
    v4 = "output_indices_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method2(v35):
        v37 = v33
        v38 = v37 >= 8192
        del v37
        if v38:
            v39 = " ..."
            print(v34.format(v39),end="")
            del v39
            break
        else:
            pass
        del v38
        v40 = v35 == 0
        v41 = v40 != True
        del v40
        if v41:
            v42 = "; "
            print(v34.format(v42),end="")
            del v42
        else:
            pass
        del v41
        print(v34.format('['),end="")
        v43 = 0
        while method3(v43):
            v45 = v33
            v46 = v45 >= 8192
            del v45
            if v46:
                v47 = " ..."
                print(v34.format(v47),end="")
                del v47
                break
            else:
                pass
            del v46
            v48 = v43 == 0
            v49 = v48 != True
            del v48
            if v49:
                v50 = "; "
                print(v34.format(v50),end="")
                del v50
            else:
                pass
            del v49
            v51 = v33 + 1
            v33 = v51
            del v51
            v52 = v35 * 128
            v53 = v52 + v43
            del v52
            v54 = v0[v53].item()
            v55 = v1[v53].item()
            del v53
            v56 = "{}, {}"
            print(v56.format(v54, v55),end="")
            del v54, v55, v56
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v0, v1, v33, v35
    print(v34.format(']'),end="")
    del v34
    v57 = "\n"
    print(v57.format(),end="")
    del v57
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method13(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_indices_reduction.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 8192
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
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
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method14(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_sum_exclusive.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method2(v32):
        v34 = v30
        v35 = v34 >= 8192
        del v34
        if v35:
            v36 = " ..."
            print(v31.format(v36),end="")
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
            print(v31.format(v39),end="")
            del v39
        else:
            pass
        del v38
        print(v31.format('['),end="")
        v40 = 0
        while method3(v40):
            v42 = v30
            v43 = v42 >= 8192
            del v42
            if v43:
                v44 = " ..."
                print(v31.format(v44),end="")
                del v44
                break
            else:
                pass
            del v43
            v45 = v40 == 0
            v46 = v45 != True
            del v45
            if v46:
                v47 = "; "
                print(v31.format(v47),end="")
                del v47
            else:
                pass
            del v46
            v48 = v30 + 1
            v30 = v48
            del v48
            v49 = v32 * 128
            v50 = v49 + v40
            del v49
            v51 = v0[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v0, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52.format(),end="")
    del v52
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method15(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_softmax'.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method3(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method16(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_sampling.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
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
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method17(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_softmax'.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method2(v33):
        v35 = v31
        v36 = v35 >= 1024
        del v35
        if v36:
            v37 = " ..."
            print(v32.format(v37),end="")
            del v37
            break
        else:
            pass
        del v36
        v38 = v33 == 0
        v39 = v38 != True
        del v38
        if v39:
            v40 = "; "
            print(v32.format(v40),end="")
            del v40
        else:
            pass
        del v39
        print(v32.format('['),end="")
        v41 = 0
        while method3(v41):
            v43 = v31
            v44 = v43 >= 1024
            del v43
            if v44:
                v45 = " ..."
                print(v32.format(v45),end="")
                del v45
                break
            else:
                pass
            del v44
            v46 = v41 == 0
            v47 = v46 != True
            del v46
            if v47:
                v48 = "; "
                print(v32.format(v48),end="")
                del v48
            else:
                pass
            del v47
            v49 = v31 + 1
            v31 = v49
            del v49
            v50 = v33 * 128
            v51 = v50 + v41
            del v50
            v52 = v0[v51].item()
            del v51
            v53 = "{:.6f}"
            print(v53.format(v52),end="")
            del v52, v53
            v41 += 1 
        del v41
        print(v32.format(']'),end="")
        v33 += 1 
    del v0, v31, v33
    print(v32.format(']'),end="")
    del v32
    v54 = "\n"
    print(v54.format(),end="")
    del v54
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method18(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/a/"
    v3 = "output_sampling'.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method2(v22):
        v24 = v20
        v25 = v24 >= 1024
        del v24
        if v25:
            v26 = " ..."
            print(v21.format(v26),end="")
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
            print(v21.format(v29),end="")
            del v29
        else:
            pass
        del v28
        v30 = v20 + 1
        v20 = v30
        del v30
        v31 = v0[v22].item()
        print(v21.format(v31),end="")
        del v31
        v22 += 1 
    del v0, v20, v22
    print(v21.format(']'),end="")
    del v21
    v32 = "\n"
    print(v32.format(),end="")
    del v32
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def main_body():
    cp.random.seed(12344321)
    v0 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v1 = v0.size
    v2 = 8192 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v6 = cp.empty(1,dtype=cp.float32)
    v7 = cp.empty(8192,dtype=cp.int32)
    v8 = cp.empty(8192,dtype=cp.float32)
    v9 = cp.empty(8192,dtype=cp.float32)
    v10 = cp.empty(8192,dtype=cp.float32)
    v11 = cp.empty(8192,dtype=cp.float32)
    v12 = cp.empty(8192,dtype=cp.float32)
    v13 = cp.empty(64,dtype=cp.int32)
    v14 = cp.empty(8192,dtype=cp.int32)
    v15 = cp.empty(8192,dtype=cp.int32)
    v16 = cp.empty(64,dtype=cp.int32)
    v17 = cp.empty(8192,dtype=cp.int32)
    v18 = cp.empty(8192,dtype=cp.float32)
    v19 = cp.empty(64,dtype=cp.int32)
    v20 = cp.empty(8192,dtype=cp.float32)
    v21 = cp.empty(64,dtype=cp.int32)
    method0(v0, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21)
    method1(v5)
    del v5
    method4(v0)
    del v0
    method5(v6)
    del v6
    method6(v8)
    del v8
    method7(v9)
    del v9
    method8(v12)
    del v12
    method9(v13)
    del v13
    method10(v10, v11)
    del v10, v11
    method11(v7)
    del v7
    method12(v14, v15)
    del v14, v15
    method13(v16)
    del v16
    method14(v17)
    del v17
    method15(v18)
    del v18
    method16(v19)
    del v19
    method17(v20)
    del v20
    return method18(v21)

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
