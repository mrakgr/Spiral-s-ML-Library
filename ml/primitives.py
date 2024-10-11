kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <curand_kernel.h>
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
struct Tuple4 {
    unsigned long long v1;
    int v0;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Closure8 {
    __device__ unsigned long long operator()(unsigned long long tup0, unsigned long long tup1){
        unsigned long long v0 = tup0; unsigned long long v1 = tup1;
        unsigned long long v2;
        v2 = v0 + v1;
        return v2;
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
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 64;
    return v1;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 32;
    return v1;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 16384;
    return v1;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 8192;
    return v1;
}
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 24;
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
        float v1174; int v1175;
        Tuple1 tmp5 = Tuple1{0.0f, 2147483647};
        v1174 = tmp5.v0; v1175 = tmp5.v1;
        int v1176;
        v1176 = 0;
        while (while_method_3(v1176)){
            int v1178;
            v1178 = 0;
            while (while_method_1(v1178)){
                assert("Tensor range check" && 0 <= v1176 && v1176 < 1);
                assert("Tensor range check" && 0 <= v1178 && v1178 < 4);
                int v1180;
                v1180 = 4 * v1176;
                int v1181;
                v1181 = v1180 + v1178;
                float v1182;
                v1182 = v1164[v1181];
                int v1183;
                v1183 = v1165[v1181];
                bool v1184;
                v1184 = v1175 < v1183;
                float v1185; int v1186;
                if (v1184){
                    v1185 = v1174; v1186 = v1175;
                } else {
                    v1185 = v1182; v1186 = v1183;
                }
                v1174 = v1185;
                v1175 = v1186;
                v1178 += 1 ;
            }
            v1176 += 1 ;
        }
        auto v1187 = cooperative_groups::coalesced_threads();
        int v1188;
        v1188 = threadIdx.x;
        int v1189;
        v1189 = v1188 / 32;
        auto v1190 = cooperative_groups::labeled_partition(v1187,v1189);
        Closure6 v1191{};
        float v1192; int v1193;
        Tuple1 tmp6 = cooperative_groups::reduce(v1190, Tuple1{v1174, v1175}, v1191);
        v1192 = tmp6.v0; v1193 = tmp6.v1;
        float v1194;
        v1194 = v1160 * v1192;
        int v1195[4];
        bool v1196[4];
        int v1197;
        v1197 = 0;
        while (while_method_3(v1197)){
            int v1199;
            v1199 = 0;
            while (while_method_1(v1199)){
                assert("Tensor range check" && 0 <= v1197 && v1197 < 1);
                assert("Tensor range check" && 0 <= v1199 && v1199 < 4);
                int v1201;
                v1201 = 4 * v1197;
                int v1202;
                v1202 = v1201 + v1199;
                float v1203;
                v1203 = v1126[v1202];
                bool v1204;
                v1204 = v1127[v1202];
                int v1205;
                v1205 = v1005[v1202];
                int v1208; bool v1209;
                if (v1204){
                    float v1206;
                    v1206 = v1203 - v1194;
                    bool v1207;
                    v1207 = v1206 >= 0.0f;
                    v1208 = v1205; v1209 = v1207;
                } else {
                    v1208 = 2147483647; v1209 = false;
                }
                assert("Tensor range check" && 0 <= v1197 && v1197 < 1);
                assert("Tensor range check" && 0 <= v1199 && v1199 < 4);
                v1195[v1202] = v1208;
                v1196[v1202] = v1209;
                v1199 += 1 ;
            }
            v1197 += 1 ;
        }
        int v1210; bool v1211;
        Tuple3 tmp7 = Tuple3{2147483647, false};
        v1210 = tmp7.v0; v1211 = tmp7.v1;
        int v1212;
        v1212 = 0;
        while (while_method_3(v1212)){
            int v1214;
            v1214 = 0;
            while (while_method_1(v1214)){
                assert("Tensor range check" && 0 <= v1212 && v1212 < 1);
                assert("Tensor range check" && 0 <= v1214 && v1214 < 4);
                int v1216;
                v1216 = 4 * v1212;
                int v1217;
                v1217 = v1216 + v1214;
                int v1218;
                v1218 = v1195[v1217];
                bool v1219;
                v1219 = v1196[v1217];
                int v1226; bool v1227;
                if (v1211){
                    if (v1219){
                        bool v1220;
                        v1220 = v1210 < v1218;
                        int v1221;
                        if (v1220){
                            v1221 = v1210;
                        } else {
                            v1221 = v1218;
                        }
                        v1226 = v1221; v1227 = true;
                    } else {
                        v1226 = v1210; v1227 = v1211;
                    }
                } else {
                    if (v1219){
                        v1226 = v1218; v1227 = v1219;
                    } else {
                        v1226 = v1210; v1227 = v1211;
                    }
                }
                v1210 = v1226;
                v1211 = v1227;
                v1214 += 1 ;
            }
            v1212 += 1 ;
        }
        auto v1228 = cooperative_groups::coalesced_threads();
        int v1229;
        v1229 = threadIdx.x;
        int v1230;
        v1230 = v1229 / 32;
        auto v1231 = cooperative_groups::labeled_partition(v1228,v1230);
        Closure7 v1232{};
        int v1233; bool v1234;
        Tuple3 tmp8 = cooperative_groups::reduce(v1231, Tuple3{v1210, v1211}, v1232);
        v1233 = tmp8.v0; v1234 = tmp8.v1;
        bool v1235;
        v1235 = v1234 == false;
        if (v1235){
            assert("The local reduce must be true." && v1234);
        } else {
        }
        assert("Tensor range check" && 0 <= v1000 && v1000 < 8);
        int v1237;
        v1237 = 0;
        while (while_method_3(v1237)){
            assert("Tensor range check" && 0 <= v1237 && v1237 < 1);
            int v1239;
            v1239 = 128 * v1237;
            int v1240;
            v1240 = v1239 + v1003;
            assert("Tensor range check" && 0 <= v1237 && v1237 < 1);
            int v1241;
            v1241 = 4 * v1237;
            int4* v1242;
            v1242 = reinterpret_cast<int4*>(v1088 + v1241);
            int4* v1243;
            v1243 = reinterpret_cast<int4*>(v14 + v1240);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1242) % 16 == 0 && reinterpret_cast<unsigned long long>(v1243) % 16 == 0);
            *v1243 = *v1242;
            v1237 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1000 && v1000 < 8);
        int v1244;
        v1244 = 8 * v1000;
        int v1245;
        v1245 = v1244 + v993;
        v15[v1245] = v1233;
        v1000 += 1 ;
    }
    __syncthreads();
    int v1246;
    v1246 = threadIdx.x;
    int v1247;
    v1247 = blockIdx.x;
    int v1248;
    v1248 = v1247 * 256;
    int v1249;
    v1249 = v1246 + v1248;
    unsigned long long v1250;
    v1250 = (unsigned long long)v1249;
    curandStatePhilox4_32_10_t v1251;
    curand_init(12344321ull,v1250,0ull,&v1251);
    int v1252;
    v1252 = threadIdx.x;
    bool v1253;
    v1253 = 0 <= v1252;
    bool v1254;
    v1254 = v1253 == false;
    if (v1254){
        assert("The index needs to be zero or positive." && v1253);
    } else {
    }
    int v1256;
    v1256 = v1252 % 32;
    int v1257;
    v1257 = v1252 / 32;
    bool v1258;
    v1258 = v1257 < 8;
    bool v1259;
    v1259 = v1258 == false;
    if (v1259){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1258);
    } else {
    }
    assert("Tensor range check" && 0 <= v1257 && v1257 < 8);
    assert("Tensor range check" && 0 <= v1256 && v1256 < 32);
    int v1261;
    v1261 = 4 * v1256;
    int v1262;
    v1262 = 128 * v1257;
    int v1263;
    v1263 = v1262 + v1261;
    assert("Tensor range check" && 0 <= v1257 && v1257 < 8);
    assert("Tensor range check" && 0 <= v1256 && v1256 < 32);
    assert("Tensor range check" && 0 <= v1257 && v1257 < 8);
    int v1264;
    v1264 = 0;
    while (while_method_2(v1264)){
        assert("Tensor range check" && 0 <= v1264 && v1264 < 8);
        int v1266;
        v1266 = 1024 * v1264;
        int v1267;
        v1267 = v1266 + v1263;
        float v1268[4];
        int v1269[4];
        int v1270;
        v1270 = 0;
        while (while_method_3(v1270)){
            assert("Tensor range check" && 0 <= v1270 && v1270 < 1);
            int v1272;
            v1272 = 4 * v1270;
            assert("Tensor range check" && 0 <= v1270 && v1270 < 1);
            int v1273;
            v1273 = 128 * v1270;
            int v1274;
            v1274 = v1273 + v1267;
            int4* v1275;
            v1275 = reinterpret_cast<int4*>(v1 + v1274);
            int4* v1276;
            v1276 = reinterpret_cast<int4*>(v1268 + v1272);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1275) % 16 == 0 && reinterpret_cast<unsigned long long>(v1276) % 16 == 0);
            *v1276 = *v1275;
            v1270 += 1 ;
        }
        int v1277;
        v1277 = 0;
        while (while_method_3(v1277)){
            int v1279;
            v1279 = 0;
            while (while_method_1(v1279)){
                bool v1281;
                v1281 = 0 <= v1279;
                bool v1283;
                if (v1281){
                    bool v1282;
                    v1282 = v1279 < 4;
                    v1283 = v1282;
                } else {
                    v1283 = false;
                }
                bool v1284;
                v1284 = v1283 == false;
                if (v1284){
                    assert("The indices should be inside the range of the dimension." && v1283);
                } else {
                }
                bool v1286;
                v1286 = 0 <= v1256;
                bool v1288;
                if (v1286){
                    bool v1287;
                    v1287 = v1256 < 32;
                    v1288 = v1287;
                } else {
                    v1288 = false;
                }
                bool v1289;
                v1289 = v1288 == false;
                if (v1289){
                    assert("The indices should be inside the range of the dimension." && v1288);
                } else {
                }
                int v1291;
                v1291 = v1256 * 4;
                int v1292;
                v1292 = v1279 + v1291;
                bool v1293;
                v1293 = 0 <= v1277;
                bool v1295;
                if (v1293){
                    bool v1294;
                    v1294 = v1277 < 1;
                    v1295 = v1294;
                } else {
                    v1295 = false;
                }
                bool v1296;
                v1296 = v1295 == false;
                if (v1296){
                    assert("The indices should be inside the range of the dimension." && v1295);
                } else {
                }
                int v1298;
                v1298 = v1277 * 128;
                int v1299;
                v1299 = v1292 + v1298;
                assert("Tensor range check" && 0 <= v1277 && v1277 < 1);
                assert("Tensor range check" && 0 <= v1279 && v1279 < 4);
                int v1300;
                v1300 = 4 * v1277;
                int v1301;
                v1301 = v1300 + v1279;
                v1269[v1301] = v1299;
                v1279 += 1 ;
            }
            v1277 += 1 ;
        }
        bool v1302;
        v1302 = 0 <= v1257;
        bool v1303;
        v1303 = v1302 && v1258;
        bool v1304;
        v1304 = v1303 == false;
        if (v1304){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1303);
        } else {
        }
        bool v1306;
        v1306 = 0 <= v1264;
        bool v1308;
        if (v1306){
            bool v1307;
            v1307 = v1264 < 8;
            v1308 = v1307;
        } else {
            v1308 = false;
        }
        bool v1309;
        v1309 = v1308 == false;
        if (v1309){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1308);
        } else {
        }
        int v1311;
        v1311 = v1264 * 8;
        int v1312;
        v1312 = v1311 + v1257;
        bool v1313[4];
        int v1314;
        v1314 = 0;
        while (while_method_3(v1314)){
            int v1316;
            v1316 = 0;
            while (while_method_1(v1316)){
                assert("Tensor range check" && 0 <= v1314 && v1314 < 1);
                assert("Tensor range check" && 0 <= v1316 && v1316 < 4);
                int v1318;
                v1318 = 4 * v1314;
                int v1319;
                v1319 = v1318 + v1316;
                float v1320;
                v1320 = v1268[v1319];
                int v1321;
                v1321 = v1269[v1319];
                bool v1322;
                v1322 = v1321 < 11;
                assert("Tensor range check" && 0 <= v1314 && v1314 < 1);
                assert("Tensor range check" && 0 <= v1316 && v1316 < 4);
                v1313[v1319] = v1322;
                v1316 += 1 ;
            }
            v1314 += 1 ;
        }
        float v1323[4];
        int v1324;
        v1324 = 0;
        while (while_method_3(v1324)){
            int v1326;
            v1326 = 0;
            while (while_method_1(v1326)){
                assert("Tensor range check" && 0 <= v1324 && v1324 < 1);
                assert("Tensor range check" && 0 <= v1326 && v1326 < 4);
                int v1328;
                v1328 = 4 * v1324;
                int v1329;
                v1329 = v1328 + v1326;
                float v1330;
                v1330 = v1268[v1329];
                bool v1331;
                v1331 = v1313[v1329];
                float v1332;
                if (v1331){
                    v1332 = v1330;
                } else {
                    v1332 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1324 && v1324 < 1);
                assert("Tensor range check" && 0 <= v1326 && v1326 < 4);
                v1323[v1329] = v1332;
                v1326 += 1 ;
            }
            v1324 += 1 ;
        }
        float v1333;
        v1333 = 0.0f;
        int v1334;
        v1334 = 0;
        while (while_method_3(v1334)){
            int v1336;
            v1336 = 0;
            while (while_method_1(v1336)){
                assert("Tensor range check" && 0 <= v1334 && v1334 < 1);
                assert("Tensor range check" && 0 <= v1336 && v1336 < 4);
                int v1338;
                v1338 = 4 * v1334;
                int v1339;
                v1339 = v1338 + v1336;
                float v1340;
                v1340 = v1323[v1339];
                float v1341;
                v1341 = v1333 + v1340;
                v1333 = v1341;
                v1336 += 1 ;
            }
            v1334 += 1 ;
        }
        auto v1342 = cooperative_groups::coalesced_threads();
        int v1343;
        v1343 = threadIdx.x;
        int v1344;
        v1344 = v1343 / 32;
        auto v1345 = cooperative_groups::labeled_partition(v1342,v1344);
        float v1346;
        v1346 = cooperative_groups::reduce(v1345, v1333, v42);
        int v1347[4];
        int v1348;
        v1348 = 0;
        while (while_method_3(v1348)){
            int v1350;
            v1350 = 0;
            while (while_method_1(v1350)){
                assert("Tensor range check" && 0 <= v1348 && v1348 < 1);
                assert("Tensor range check" && 0 <= v1350 && v1350 < 4);
                int v1352;
                v1352 = 4 * v1348;
                int v1353;
                v1353 = v1352 + v1350;
                bool v1354;
                v1354 = v1313[v1353];
                int v1355;
                if (v1354){
                    v1355 = 1;
                } else {
                    v1355 = 0;
                }
                assert("Tensor range check" && 0 <= v1348 && v1348 < 1);
                assert("Tensor range check" && 0 <= v1350 && v1350 < 4);
                v1347[v1353] = v1355;
                v1350 += 1 ;
            }
            v1348 += 1 ;
        }
        int v1356;
        v1356 = 0;
        int v1357;
        v1357 = 0;
        while (while_method_3(v1357)){
            int v1359;
            v1359 = 0;
            while (while_method_1(v1359)){
                assert("Tensor range check" && 0 <= v1357 && v1357 < 1);
                assert("Tensor range check" && 0 <= v1359 && v1359 < 4);
                int v1361;
                v1361 = 4 * v1357;
                int v1362;
                v1362 = v1361 + v1359;
                int v1363;
                v1363 = v1347[v1362];
                int v1364;
                v1364 = v1356 + v1363;
                v1356 = v1364;
                v1359 += 1 ;
            }
            v1357 += 1 ;
        }
        auto v1365 = cooperative_groups::coalesced_threads();
        int v1366;
        v1366 = threadIdx.x;
        int v1367;
        v1367 = v1366 / 32;
        auto v1368 = cooperative_groups::labeled_partition(v1365,v1367);
        Closure4 v1369{};
        int v1370;
        v1370 = cooperative_groups::reduce(v1368, v1356, v1369);
        float v1371;
        v1371 = (float)v1370;
        float v1372;
        v1372 = v1346 / v1371;
        float v1373[4];
        int v1374;
        v1374 = 0;
        while (while_method_3(v1374)){
            int v1376;
            v1376 = 0;
            while (while_method_1(v1376)){
                assert("Tensor range check" && 0 <= v1374 && v1374 < 1);
                assert("Tensor range check" && 0 <= v1376 && v1376 < 4);
                int v1378;
                v1378 = 4 * v1374;
                int v1379;
                v1379 = v1378 + v1376;
                float v1380;
                v1380 = v1268[v1379];
                bool v1381;
                v1381 = v1313[v1379];
                float v1382;
                if (v1381){
                    v1382 = v1380;
                } else {
                    v1382 = -1.0f / 0.0f;
                }
                float v1383;
                v1383 = v1382 - v1372;
                float v1384;
                v1384 = exp(v1383);
                bool v1385;
                v1385 = v1384 < 1.0f / 0.0f;
                bool v1386;
                v1386 = v1385 == false;
                if (v1386){
                    assert("The softmax values must not grow too large." && v1385);
                } else {
                }
                bool v1388;
                v1388 = isnan(v1384);
                bool v1389;
                v1389 = v1388 == false;
                bool v1390;
                v1390 = v1389 == false;
                if (v1390){
                    assert("The softmax values must not be nans." && v1389);
                } else {
                }
                assert("Tensor range check" && 0 <= v1374 && v1374 < 1);
                assert("Tensor range check" && 0 <= v1376 && v1376 < 4);
                v1373[v1379] = v1384;
                v1376 += 1 ;
            }
            v1374 += 1 ;
        }
        float v1392;
        v1392 = 0.0f;
        int v1393;
        v1393 = 0;
        while (while_method_3(v1393)){
            int v1395;
            v1395 = 0;
            while (while_method_1(v1395)){
                assert("Tensor range check" && 0 <= v1393 && v1393 < 1);
                assert("Tensor range check" && 0 <= v1395 && v1395 < 4);
                int v1397;
                v1397 = 4 * v1393;
                int v1398;
                v1398 = v1397 + v1395;
                float v1399;
                v1399 = v1373[v1398];
                float v1400;
                v1400 = v1392 + v1399;
                v1392 = v1400;
                v1395 += 1 ;
            }
            v1393 += 1 ;
        }
        auto v1401 = cooperative_groups::coalesced_threads();
        int v1402;
        v1402 = threadIdx.x;
        int v1403;
        v1403 = v1402 / 32;
        auto v1404 = cooperative_groups::labeled_partition(v1401,v1403);
        float v1405;
        v1405 = cooperative_groups::reduce(v1404, v1392, v42);
        float v1406[4];
        int v1407;
        v1407 = 0;
        while (while_method_3(v1407)){
            int v1409;
            v1409 = 0;
            while (while_method_1(v1409)){
                assert("Tensor range check" && 0 <= v1407 && v1407 < 1);
                assert("Tensor range check" && 0 <= v1409 && v1409 < 4);
                int v1411;
                v1411 = 4 * v1407;
                int v1412;
                v1412 = v1411 + v1409;
                float v1413;
                v1413 = v1373[v1412];
                float v1414;
                v1414 = v1413 / v1405;
                assert("Tensor range check" && 0 <= v1407 && v1407 < 1);
                assert("Tensor range check" && 0 <= v1409 && v1409 < 4);
                v1406[v1412] = v1414;
                v1409 += 1 ;
            }
            v1407 += 1 ;
        }
        float v1415[4];
        float v1416;
        v1416 = 0.0f;
        int v1417;
        v1417 = 0;
        while (while_method_3(v1417)){
            assert("Tensor range check" && 0 <= v1417 && v1417 < 1);
            int v1419;
            v1419 = 4 * v1417;
            assert("Tensor range check" && 0 <= v1417 && v1417 < 1);
            float v1420;
            v1420 = 0.0f;
            int v1421;
            v1421 = 0;
            while (while_method_1(v1421)){
                assert("Tensor range check" && 0 <= v1421 && v1421 < 4);
                int v1423;
                v1423 = v1421 + v1419;
                float v1424;
                v1424 = v1406[v1423];
                float v1425;
                v1425 = v1420 + v1424;
                v1420 = v1425;
                v1421 += 1 ;
            }
            auto v1426 = cooperative_groups::coalesced_threads();
            int v1427;
            v1427 = threadIdx.x;
            int v1428;
            v1428 = v1427 / 32;
            auto v1429 = cooperative_groups::labeled_partition(v1426,v1428);
            Closure2 v1430{};
            float v1431;
            v1431 = cooperative_groups::inclusive_scan(v1429, v1420, v1430);
            float v1432;
            v1432 = v1429.shfl_up(v1431,1);
            bool v1433;
            v1433 = v1429.thread_rank() == 0;
            float v1434;
            if (v1433){
                v1434 = 0.0f;
            } else {
                v1434 = v1432;
            }
            float v1435;
            v1435 = v1429.shfl(v1431,v1429.num_threads()-1);
            float v1436;
            v1436 = v1416 + v1434;
            float v1437;
            v1437 = v1436;
            int v1438;
            v1438 = 0;
            while (while_method_1(v1438)){
                assert("Tensor range check" && 0 <= v1438 && v1438 < 4);
                int v1440;
                v1440 = v1438 + v1419;
                float v1441;
                v1441 = v1406[v1440];
                float v1442;
                v1442 = v1437 + v1441;
                assert("Tensor range check" && 0 <= v1438 && v1438 < 4);
                v1415[v1440] = v1442;
                v1437 = v1442;
                v1438 += 1 ;
            }
            float v1443;
            v1443 = v1416 + v1435;
            v1416 = v1443;
            v1417 += 1 ;
        }
        float v1444[4];
        bool v1445[4];
        int v1446;
        v1446 = 0;
        while (while_method_3(v1446)){
            int v1448;
            v1448 = 0;
            while (while_method_1(v1448)){
                assert("Tensor range check" && 0 <= v1446 && v1446 < 1);
                assert("Tensor range check" && 0 <= v1448 && v1448 < 4);
                int v1450;
                v1450 = 4 * v1446;
                int v1451;
                v1451 = v1450 + v1448;
                float v1452;
                v1452 = v1415[v1451];
                float v1453;
                v1453 = v1406[v1451];
                bool v1454;
                v1454 = v1453 > 0.0f;
                assert("Tensor range check" && 0 <= v1446 && v1446 < 1);
                assert("Tensor range check" && 0 <= v1448 && v1448 < 4);
                v1444[v1451] = v1452;
                v1445[v1451] = v1454;
                v1448 += 1 ;
            }
            v1446 += 1 ;
        }
        float v1455; bool v1456;
        Tuple2 tmp9 = Tuple2{-1.0f / 0.0f, false};
        v1455 = tmp9.v0; v1456 = tmp9.v1;
        int v1457;
        v1457 = 0;
        while (while_method_3(v1457)){
            int v1459;
            v1459 = 0;
            while (while_method_1(v1459)){
                assert("Tensor range check" && 0 <= v1457 && v1457 < 1);
                assert("Tensor range check" && 0 <= v1459 && v1459 < 4);
                int v1461;
                v1461 = 4 * v1457;
                int v1462;
                v1462 = v1461 + v1459;
                float v1463;
                v1463 = v1444[v1462];
                bool v1464;
                v1464 = v1445[v1462];
                float v1471; bool v1472;
                if (v1456){
                    if (v1464){
                        bool v1465;
                        v1465 = v1455 >= v1463;
                        float v1466;
                        if (v1465){
                            v1466 = v1455;
                        } else {
                            v1466 = v1463;
                        }
                        v1471 = v1466; v1472 = true;
                    } else {
                        v1471 = v1455; v1472 = v1456;
                    }
                } else {
                    if (v1464){
                        v1471 = v1463; v1472 = v1464;
                    } else {
                        v1471 = v1455; v1472 = v1456;
                    }
                }
                v1455 = v1471;
                v1456 = v1472;
                v1459 += 1 ;
            }
            v1457 += 1 ;
        }
        auto v1473 = cooperative_groups::coalesced_threads();
        int v1474;
        v1474 = threadIdx.x;
        int v1475;
        v1475 = v1474 / 32;
        auto v1476 = cooperative_groups::labeled_partition(v1473,v1475);
        Closure5 v1477{};
        float v1478; bool v1479;
        Tuple2 tmp10 = cooperative_groups::reduce(v1476, Tuple2{v1455, v1456}, v1477);
        v1478 = tmp10.v0; v1479 = tmp10.v1;
        bool v1480;
        v1480 = v1479 == false;
        if (v1480){
            assert("The local reduce must be true." && v1479);
        } else {
        }
        float v1482[4];
        int v1483[4];
        int v1484;
        v1484 = 0;
        while (while_method_3(v1484)){
            int v1486;
            v1486 = 0;
            while (while_method_1(v1486)){
                assert("Tensor range check" && 0 <= v1484 && v1484 < 1);
                assert("Tensor range check" && 0 <= v1486 && v1486 < 4);
                int v1488;
                v1488 = 4 * v1484;
                int v1489;
                v1489 = v1488 + v1486;
                int v1490;
                v1490 = v1269[v1489];
                float v1491;
                v1491 = curand_uniform(&v1251);
                assert("Tensor range check" && 0 <= v1484 && v1484 < 1);
                assert("Tensor range check" && 0 <= v1486 && v1486 < 4);
                v1482[v1489] = v1491;
                v1483[v1489] = v1490;
                v1486 += 1 ;
            }
            v1484 += 1 ;
        }
        float v1492; int v1493;
        Tuple1 tmp11 = Tuple1{0.0f, 2147483647};
        v1492 = tmp11.v0; v1493 = tmp11.v1;
        int v1494;
        v1494 = 0;
        while (while_method_3(v1494)){
            int v1496;
            v1496 = 0;
            while (while_method_1(v1496)){
                assert("Tensor range check" && 0 <= v1494 && v1494 < 1);
                assert("Tensor range check" && 0 <= v1496 && v1496 < 4);
                int v1498;
                v1498 = 4 * v1494;
                int v1499;
                v1499 = v1498 + v1496;
                float v1500;
                v1500 = v1482[v1499];
                int v1501;
                v1501 = v1483[v1499];
                bool v1502;
                v1502 = v1493 < v1501;
                float v1503; int v1504;
                if (v1502){
                    v1503 = v1492; v1504 = v1493;
                } else {
                    v1503 = v1500; v1504 = v1501;
                }
                v1492 = v1503;
                v1493 = v1504;
                v1496 += 1 ;
            }
            v1494 += 1 ;
        }
        auto v1505 = cooperative_groups::coalesced_threads();
        int v1506;
        v1506 = threadIdx.x;
        int v1507;
        v1507 = v1506 / 32;
        auto v1508 = cooperative_groups::labeled_partition(v1505,v1507);
        Closure6 v1509{};
        float v1510; int v1511;
        Tuple1 tmp12 = cooperative_groups::reduce(v1508, Tuple1{v1492, v1493}, v1509);
        v1510 = tmp12.v0; v1511 = tmp12.v1;
        float v1512;
        v1512 = v1478 * v1510;
        int v1513[4];
        bool v1514[4];
        int v1515;
        v1515 = 0;
        while (while_method_3(v1515)){
            int v1517;
            v1517 = 0;
            while (while_method_1(v1517)){
                assert("Tensor range check" && 0 <= v1515 && v1515 < 1);
                assert("Tensor range check" && 0 <= v1517 && v1517 < 4);
                int v1519;
                v1519 = 4 * v1515;
                int v1520;
                v1520 = v1519 + v1517;
                float v1521;
                v1521 = v1444[v1520];
                bool v1522;
                v1522 = v1445[v1520];
                int v1523;
                v1523 = v1269[v1520];
                int v1526; bool v1527;
                if (v1522){
                    float v1524;
                    v1524 = v1521 - v1512;
                    bool v1525;
                    v1525 = v1524 >= 0.0f;
                    v1526 = v1523; v1527 = v1525;
                } else {
                    v1526 = 2147483647; v1527 = false;
                }
                assert("Tensor range check" && 0 <= v1515 && v1515 < 1);
                assert("Tensor range check" && 0 <= v1517 && v1517 < 4);
                v1513[v1520] = v1526;
                v1514[v1520] = v1527;
                v1517 += 1 ;
            }
            v1515 += 1 ;
        }
        int v1528; bool v1529;
        Tuple3 tmp13 = Tuple3{2147483647, false};
        v1528 = tmp13.v0; v1529 = tmp13.v1;
        int v1530;
        v1530 = 0;
        while (while_method_3(v1530)){
            int v1532;
            v1532 = 0;
            while (while_method_1(v1532)){
                assert("Tensor range check" && 0 <= v1530 && v1530 < 1);
                assert("Tensor range check" && 0 <= v1532 && v1532 < 4);
                int v1534;
                v1534 = 4 * v1530;
                int v1535;
                v1535 = v1534 + v1532;
                int v1536;
                v1536 = v1513[v1535];
                bool v1537;
                v1537 = v1514[v1535];
                int v1544; bool v1545;
                if (v1529){
                    if (v1537){
                        bool v1538;
                        v1538 = v1528 < v1536;
                        int v1539;
                        if (v1538){
                            v1539 = v1528;
                        } else {
                            v1539 = v1536;
                        }
                        v1544 = v1539; v1545 = true;
                    } else {
                        v1544 = v1528; v1545 = v1529;
                    }
                } else {
                    if (v1537){
                        v1544 = v1536; v1545 = v1537;
                    } else {
                        v1544 = v1528; v1545 = v1529;
                    }
                }
                v1528 = v1544;
                v1529 = v1545;
                v1532 += 1 ;
            }
            v1530 += 1 ;
        }
        auto v1546 = cooperative_groups::coalesced_threads();
        int v1547;
        v1547 = threadIdx.x;
        int v1548;
        v1548 = v1547 / 32;
        auto v1549 = cooperative_groups::labeled_partition(v1546,v1548);
        Closure7 v1550{};
        int v1551; bool v1552;
        Tuple3 tmp14 = cooperative_groups::reduce(v1549, Tuple3{v1528, v1529}, v1550);
        v1551 = tmp14.v0; v1552 = tmp14.v1;
        bool v1553;
        v1553 = v1552 == false;
        if (v1553){
            assert("The local reduce must be true." && v1552);
        } else {
        }
        assert("Tensor range check" && 0 <= v1264 && v1264 < 8);
        int v1555;
        v1555 = 0;
        while (while_method_3(v1555)){
            assert("Tensor range check" && 0 <= v1555 && v1555 < 1);
            int v1557;
            v1557 = 128 * v1555;
            int v1558;
            v1558 = v1557 + v1267;
            assert("Tensor range check" && 0 <= v1555 && v1555 < 1);
            int v1559;
            v1559 = 4 * v1555;
            int4* v1560;
            v1560 = reinterpret_cast<int4*>(v1406 + v1559);
            int4* v1561;
            v1561 = reinterpret_cast<int4*>(v16 + v1558);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1560) % 16 == 0 && reinterpret_cast<unsigned long long>(v1561) % 16 == 0);
            *v1561 = *v1560;
            v1555 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1264 && v1264 < 8);
        int v1562;
        v1562 = 8 * v1264;
        int v1563;
        v1563 = v1562 + v1257;
        v17[v1563] = v1551;
        v1264 += 1 ;
    }
    __syncthreads();
    return ;
}
extern "C" __global__ void entry1(int * v0, float * v1, float * v2, int * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int * v9, int * v10, int * v11, int * v12, int * v13, float * v14, int * v15, float * v16, int * v17) {
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
        v25 = v20 % 16;
        int v26;
        v26 = v20 / 16;
        bool v27;
        v27 = v26 < 128;
        bool v28;
        v28 = v27 == false;
        if (v28){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v27);
        } else {
        }
        assert("Tensor range check" && 0 <= v26 && v26 < 128);
        assert("Tensor range check" && 0 <= v25 && v25 < 16);
        int v30;
        v30 = 4 * v25;
        int v31;
        v31 = 64 * v26;
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
        Tuple0 tmp15 = Tuple0{0, v18};
        v36 = tmp15.v0; v37 = tmp15.v1;
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
    v60 = v56 % 16;
    int v61;
    v61 = v56 / 16;
    bool v62;
    v62 = v61 < 16;
    bool v63;
    v63 = v62 == false;
    if (v63){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v62);
    } else {
    }
    assert("Tensor range check" && 0 <= v61 && v61 < 16);
    assert("Tensor range check" && 0 <= v60 && v60 < 16);
    int v65;
    v65 = 4 * v60;
    int v66;
    v66 = 64 * v61;
    int v67;
    v67 = v66 + v65;
    assert("Tensor range check" && 0 <= v61 && v61 < 16);
    assert("Tensor range check" && 0 <= v60 && v60 < 16);
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
            v77 = 64 * v74;
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
                    v91 = v60 < 16;
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
                v102 = v81 * 64;
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
        v115 = v68 * 16;
        int v116;
        v116 = v115 + v61;
        assert("Tensor range check" && 0 <= v68 && v68 < 8);
        int v117;
        v117 = 0;
        while (while_method_3(v117)){
            assert("Tensor range check" && 0 <= v117 && v117 < 1);
            int v119;
            v119 = 64 * v117;
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
    v128 = v124 % 16;
    int v129;
    v129 = v124 / 16;
    bool v130;
    v130 = v129 < 16;
    bool v131;
    v131 = v130 == false;
    if (v131){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v130);
    } else {
    }
    assert("Tensor range check" && 0 <= v129 && v129 < 16);
    assert("Tensor range check" && 0 <= v128 && v128 < 16);
    int v133;
    v133 = 4 * v128;
    int v134;
    v134 = 64 * v129;
    int v135;
    v135 = v134 + v133;
    assert("Tensor range check" && 0 <= v129 && v129 < 16);
    assert("Tensor range check" && 0 <= v128 && v128 < 16);
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
            v145 = 64 * v142;
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
                    v159 = v128 < 16;
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
                v170 = v149 * 64;
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
        v183 = v136 * 16;
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
            v196 = 64 * v194;
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
    v207 = v203 % 16;
    int v208;
    v208 = v203 / 16;
    bool v209;
    v209 = v208 < 16;
    bool v210;
    v210 = v209 == false;
    if (v210){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v209);
    } else {
    }
    assert("Tensor range check" && 0 <= v208 && v208 < 16);
    assert("Tensor range check" && 0 <= v207 && v207 < 16);
    int v212;
    v212 = 4 * v207;
    int v213;
    v213 = 64 * v208;
    int v214;
    v214 = v213 + v212;
    assert("Tensor range check" && 0 <= v208 && v208 < 16);
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
            v224 = 64 * v221;
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
                    v238 = v207 < 16;
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
                v249 = v228 * 64;
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
        v262 = v215 * 16;
        int v263;
        v263 = v262 + v208;
        assert("Tensor range check" && 0 <= v215 && v215 < 8);
        int v264;
        v264 = 16 * v215;
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
    v270 = v266 % 16;
    int v271;
    v271 = v266 / 16;
    bool v272;
    v272 = v271 < 16;
    bool v273;
    v273 = v272 == false;
    if (v273){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v272);
    } else {
    }
    assert("Tensor range check" && 0 <= v271 && v271 < 16);
    assert("Tensor range check" && 0 <= v270 && v270 < 16);
    int v275;
    v275 = 4 * v270;
    int v276;
    v276 = 64 * v271;
    int v277;
    v277 = v276 + v275;
    assert("Tensor range check" && 0 <= v271 && v271 < 16);
    assert("Tensor range check" && 0 <= v270 && v270 < 16);
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
            v287 = 64 * v284;
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
                    v301 = v270 < 16;
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
                v312 = v291 * 64;
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
        v325 = v278 * 16;
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
        v338 = v337 / 16;
        auto v339 = cooperative_groups::labeled_partition(v336,v338);
        float v340;
        v340 = cooperative_groups::reduce(v339, v327, v42);
        float v341;
        v341 = v340 / 64.0f;
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
        v363 = v362 / 16;
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
            v377 = 64 * v375;
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
    v386 = v382 % 16;
    int v387;
    v387 = v382 / 16;
    bool v388;
    v388 = v387 < 16;
    bool v389;
    v389 = v388 == false;
    if (v389){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v388);
    } else {
    }
    assert("Tensor range check" && 0 <= v387 && v387 < 16);
    assert("Tensor range check" && 0 <= v386 && v386 < 16);
    int v391;
    v391 = 4 * v386;
    int v392;
    v392 = 64 * v387;
    int v393;
    v393 = v392 + v391;
    assert("Tensor range check" && 0 <= v387 && v387 < 16);
    assert("Tensor range check" && 0 <= v386 && v386 < 16);
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
            v403 = 64 * v400;
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
                    v417 = v386 < 16;
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
                v428 = v407 * 64;
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
        v441 = v394 * 16;
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
        v463 = v462 / 16;
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
            v480 = 64 * v478;
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
    v489 = v485 % 16;
    int v490;
    v490 = v485 / 16;
    bool v491;
    v491 = v490 < 16;
    bool v492;
    v492 = v491 == false;
    if (v492){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v491);
    } else {
    }
    assert("Tensor range check" && 0 <= v490 && v490 < 16);
    assert("Tensor range check" && 0 <= v489 && v489 < 16);
    int v494;
    v494 = 4 * v489;
    int v495;
    v495 = 64 * v490;
    int v496;
    v496 = v495 + v494;
    assert("Tensor range check" && 0 <= v490 && v490 < 16);
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
            v506 = 64 * v503;
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
                    v520 = v489 < 16;
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
                v531 = v510 * 64;
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
        v544 = v497 * 16;
        int v545;
        v545 = v544 + v490;
        float v546; int v547;
        Tuple1 tmp16 = Tuple1{-1.0f / 0.0f, 0};
        v546 = tmp16.v0; v547 = tmp16.v1;
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
        v561 = v560 / 16;
        auto v562 = cooperative_groups::labeled_partition(v559,v561);
        Closure1 v563{};
        float v564; int v565;
        Tuple1 tmp17 = cooperative_groups::reduce(v562, Tuple1{v546, v547}, v563);
        v564 = tmp17.v0; v565 = tmp17.v1;
        assert("Tensor range check" && 0 <= v497 && v497 < 8);
        int v566;
        v566 = 16 * v497;
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
    v572 = v568 % 16;
    int v573;
    v573 = v568 / 16;
    bool v574;
    v574 = v573 < 16;
    bool v575;
    v575 = v574 == false;
    if (v575){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v574);
    } else {
    }
    assert("Tensor range check" && 0 <= v573 && v573 < 16);
    assert("Tensor range check" && 0 <= v572 && v572 < 16);
    int v577;
    v577 = 4 * v572;
    int v578;
    v578 = 64 * v573;
    int v579;
    v579 = v578 + v577;
    assert("Tensor range check" && 0 <= v573 && v573 < 16);
    assert("Tensor range check" && 0 <= v572 && v572 < 16);
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
            v589 = 64 * v586;
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
                    v603 = v572 < 16;
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
                v614 = v593 * 64;
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
        v627 = v580 * 16;
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
        v640 = v639 / 16;
        auto v641 = cooperative_groups::labeled_partition(v638,v640);
        float v642;
        v642 = cooperative_groups::reduce(v641, v629, v42);
        float v643;
        v643 = v642 / 64.0f;
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
        v665 = v664 / 16;
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
            v690 = v689 / 16;
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
            v708 = 64 * v706;
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
    v719 = v715 % 16;
    int v720;
    v720 = v715 / 16;
    bool v721;
    v721 = v720 < 16;
    bool v722;
    v722 = v721 == false;
    if (v722){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v721);
    } else {
    }
    assert("Tensor range check" && 0 <= v720 && v720 < 16);
    assert("Tensor range check" && 0 <= v719 && v719 < 16);
    int v724;
    v724 = 4 * v719;
    int v725;
    v725 = 64 * v720;
    int v726;
    v726 = v725 + v724;
    assert("Tensor range check" && 0 <= v720 && v720 < 16);
    assert("Tensor range check" && 0 <= v719 && v719 < 16);
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
            v736 = 64 * v733;
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
                    v750 = v719 < 16;
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
                v761 = v740 * 64;
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
        v774 = v727 * 16;
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
            v789 = v788 / 16;
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
            v807 = 64 * v805;
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
    v816 = v812 % 16;
    int v817;
    v817 = v812 / 16;
    bool v818;
    v818 = v817 < 16;
    bool v819;
    v819 = v818 == false;
    if (v819){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v818);
    } else {
    }
    assert("Tensor range check" && 0 <= v817 && v817 < 16);
    assert("Tensor range check" && 0 <= v816 && v816 < 16);
    int v821;
    v821 = 4 * v816;
    int v822;
    v822 = 64 * v817;
    int v823;
    v823 = v822 + v821;
    assert("Tensor range check" && 0 <= v817 && v817 < 16);
    assert("Tensor range check" && 0 <= v816 && v816 < 16);
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
            v833 = 64 * v830;
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
                    v847 = v816 < 16;
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
                v858 = v837 * 64;
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
        v871 = v824 * 16;
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
        v904 = v903 / 16;
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
        v927 = v926 / 16;
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
        v963 = v962 / 16;
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
            v977 = 64 * v975;
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
    v992 = v988 % 16;
    int v993;
    v993 = v988 / 16;
    bool v994;
    v994 = v993 < 16;
    bool v995;
    v995 = v994 == false;
    if (v995){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v994);
    } else {
    }
    assert("Tensor range check" && 0 <= v993 && v993 < 16);
    assert("Tensor range check" && 0 <= v992 && v992 < 16);
    int v997;
    v997 = 4 * v992;
    int v998;
    v998 = 64 * v993;
    int v999;
    v999 = v998 + v997;
    assert("Tensor range check" && 0 <= v993 && v993 < 16);
    assert("Tensor range check" && 0 <= v992 && v992 < 16);
    assert("Tensor range check" && 0 <= v993 && v993 < 16);
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
            v1009 = 64 * v1006;
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
                    v1023 = v992 < 16;
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
                v1034 = v1013 * 64;
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
        v1047 = v1000 * 16;
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
        v1060 = v1059 / 16;
        auto v1061 = cooperative_groups::labeled_partition(v1058,v1060);
        float v1062;
        v1062 = cooperative_groups::reduce(v1061, v1049, v42);
        float v1063;
        v1063 = v1062 / 64.0f;
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
        v1085 = v1084 / 16;
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
            v1110 = v1109 / 16;
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
        Tuple2 tmp18 = Tuple2{-1.0f / 0.0f, false};
        v1137 = tmp18.v0; v1138 = tmp18.v1;
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
        v1157 = v1156 / 16;
        auto v1158 = cooperative_groups::labeled_partition(v1155,v1157);
        Closure5 v1159{};
        float v1160; bool v1161;
        Tuple2 tmp19 = cooperative_groups::reduce(v1158, Tuple2{v1137, v1138}, v1159);
        v1160 = tmp19.v0; v1161 = tmp19.v1;
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
        float v1174; int v1175;
        Tuple1 tmp20 = Tuple1{0.0f, 2147483647};
        v1174 = tmp20.v0; v1175 = tmp20.v1;
        int v1176;
        v1176 = 0;
        while (while_method_3(v1176)){
            int v1178;
            v1178 = 0;
            while (while_method_1(v1178)){
                assert("Tensor range check" && 0 <= v1176 && v1176 < 1);
                assert("Tensor range check" && 0 <= v1178 && v1178 < 4);
                int v1180;
                v1180 = 4 * v1176;
                int v1181;
                v1181 = v1180 + v1178;
                float v1182;
                v1182 = v1164[v1181];
                int v1183;
                v1183 = v1165[v1181];
                bool v1184;
                v1184 = v1175 < v1183;
                float v1185; int v1186;
                if (v1184){
                    v1185 = v1174; v1186 = v1175;
                } else {
                    v1185 = v1182; v1186 = v1183;
                }
                v1174 = v1185;
                v1175 = v1186;
                v1178 += 1 ;
            }
            v1176 += 1 ;
        }
        auto v1187 = cooperative_groups::coalesced_threads();
        int v1188;
        v1188 = threadIdx.x;
        int v1189;
        v1189 = v1188 / 16;
        auto v1190 = cooperative_groups::labeled_partition(v1187,v1189);
        Closure6 v1191{};
        float v1192; int v1193;
        Tuple1 tmp21 = cooperative_groups::reduce(v1190, Tuple1{v1174, v1175}, v1191);
        v1192 = tmp21.v0; v1193 = tmp21.v1;
        float v1194;
        v1194 = v1160 * v1192;
        int v1195[4];
        bool v1196[4];
        int v1197;
        v1197 = 0;
        while (while_method_3(v1197)){
            int v1199;
            v1199 = 0;
            while (while_method_1(v1199)){
                assert("Tensor range check" && 0 <= v1197 && v1197 < 1);
                assert("Tensor range check" && 0 <= v1199 && v1199 < 4);
                int v1201;
                v1201 = 4 * v1197;
                int v1202;
                v1202 = v1201 + v1199;
                float v1203;
                v1203 = v1126[v1202];
                bool v1204;
                v1204 = v1127[v1202];
                int v1205;
                v1205 = v1005[v1202];
                int v1208; bool v1209;
                if (v1204){
                    float v1206;
                    v1206 = v1203 - v1194;
                    bool v1207;
                    v1207 = v1206 >= 0.0f;
                    v1208 = v1205; v1209 = v1207;
                } else {
                    v1208 = 2147483647; v1209 = false;
                }
                assert("Tensor range check" && 0 <= v1197 && v1197 < 1);
                assert("Tensor range check" && 0 <= v1199 && v1199 < 4);
                v1195[v1202] = v1208;
                v1196[v1202] = v1209;
                v1199 += 1 ;
            }
            v1197 += 1 ;
        }
        int v1210; bool v1211;
        Tuple3 tmp22 = Tuple3{2147483647, false};
        v1210 = tmp22.v0; v1211 = tmp22.v1;
        int v1212;
        v1212 = 0;
        while (while_method_3(v1212)){
            int v1214;
            v1214 = 0;
            while (while_method_1(v1214)){
                assert("Tensor range check" && 0 <= v1212 && v1212 < 1);
                assert("Tensor range check" && 0 <= v1214 && v1214 < 4);
                int v1216;
                v1216 = 4 * v1212;
                int v1217;
                v1217 = v1216 + v1214;
                int v1218;
                v1218 = v1195[v1217];
                bool v1219;
                v1219 = v1196[v1217];
                int v1226; bool v1227;
                if (v1211){
                    if (v1219){
                        bool v1220;
                        v1220 = v1210 < v1218;
                        int v1221;
                        if (v1220){
                            v1221 = v1210;
                        } else {
                            v1221 = v1218;
                        }
                        v1226 = v1221; v1227 = true;
                    } else {
                        v1226 = v1210; v1227 = v1211;
                    }
                } else {
                    if (v1219){
                        v1226 = v1218; v1227 = v1219;
                    } else {
                        v1226 = v1210; v1227 = v1211;
                    }
                }
                v1210 = v1226;
                v1211 = v1227;
                v1214 += 1 ;
            }
            v1212 += 1 ;
        }
        auto v1228 = cooperative_groups::coalesced_threads();
        int v1229;
        v1229 = threadIdx.x;
        int v1230;
        v1230 = v1229 / 16;
        auto v1231 = cooperative_groups::labeled_partition(v1228,v1230);
        Closure7 v1232{};
        int v1233; bool v1234;
        Tuple3 tmp23 = cooperative_groups::reduce(v1231, Tuple3{v1210, v1211}, v1232);
        v1233 = tmp23.v0; v1234 = tmp23.v1;
        bool v1235;
        v1235 = v1234 == false;
        if (v1235){
            assert("The local reduce must be true." && v1234);
        } else {
        }
        assert("Tensor range check" && 0 <= v1000 && v1000 < 8);
        int v1237;
        v1237 = 0;
        while (while_method_3(v1237)){
            assert("Tensor range check" && 0 <= v1237 && v1237 < 1);
            int v1239;
            v1239 = 64 * v1237;
            int v1240;
            v1240 = v1239 + v1003;
            assert("Tensor range check" && 0 <= v1237 && v1237 < 1);
            int v1241;
            v1241 = 4 * v1237;
            int4* v1242;
            v1242 = reinterpret_cast<int4*>(v1088 + v1241);
            int4* v1243;
            v1243 = reinterpret_cast<int4*>(v14 + v1240);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1242) % 16 == 0 && reinterpret_cast<unsigned long long>(v1243) % 16 == 0);
            *v1243 = *v1242;
            v1237 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1000 && v1000 < 8);
        int v1244;
        v1244 = 16 * v1000;
        int v1245;
        v1245 = v1244 + v993;
        v15[v1245] = v1233;
        v1000 += 1 ;
    }
    __syncthreads();
    int v1246;
    v1246 = threadIdx.x;
    int v1247;
    v1247 = blockIdx.x;
    int v1248;
    v1248 = v1247 * 256;
    int v1249;
    v1249 = v1246 + v1248;
    unsigned long long v1250;
    v1250 = (unsigned long long)v1249;
    curandStatePhilox4_32_10_t v1251;
    curand_init(12344321ull,v1250,0ull,&v1251);
    int v1252;
    v1252 = threadIdx.x;
    bool v1253;
    v1253 = 0 <= v1252;
    bool v1254;
    v1254 = v1253 == false;
    if (v1254){
        assert("The index needs to be zero or positive." && v1253);
    } else {
    }
    int v1256;
    v1256 = v1252 % 16;
    int v1257;
    v1257 = v1252 / 16;
    bool v1258;
    v1258 = v1257 < 16;
    bool v1259;
    v1259 = v1258 == false;
    if (v1259){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1258);
    } else {
    }
    assert("Tensor range check" && 0 <= v1257 && v1257 < 16);
    assert("Tensor range check" && 0 <= v1256 && v1256 < 16);
    int v1261;
    v1261 = 4 * v1256;
    int v1262;
    v1262 = 64 * v1257;
    int v1263;
    v1263 = v1262 + v1261;
    assert("Tensor range check" && 0 <= v1257 && v1257 < 16);
    assert("Tensor range check" && 0 <= v1256 && v1256 < 16);
    assert("Tensor range check" && 0 <= v1257 && v1257 < 16);
    int v1264;
    v1264 = 0;
    while (while_method_2(v1264)){
        assert("Tensor range check" && 0 <= v1264 && v1264 < 8);
        int v1266;
        v1266 = 1024 * v1264;
        int v1267;
        v1267 = v1266 + v1263;
        float v1268[4];
        int v1269[4];
        int v1270;
        v1270 = 0;
        while (while_method_3(v1270)){
            assert("Tensor range check" && 0 <= v1270 && v1270 < 1);
            int v1272;
            v1272 = 4 * v1270;
            assert("Tensor range check" && 0 <= v1270 && v1270 < 1);
            int v1273;
            v1273 = 64 * v1270;
            int v1274;
            v1274 = v1273 + v1267;
            int4* v1275;
            v1275 = reinterpret_cast<int4*>(v1 + v1274);
            int4* v1276;
            v1276 = reinterpret_cast<int4*>(v1268 + v1272);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1275) % 16 == 0 && reinterpret_cast<unsigned long long>(v1276) % 16 == 0);
            *v1276 = *v1275;
            v1270 += 1 ;
        }
        int v1277;
        v1277 = 0;
        while (while_method_3(v1277)){
            int v1279;
            v1279 = 0;
            while (while_method_1(v1279)){
                bool v1281;
                v1281 = 0 <= v1279;
                bool v1283;
                if (v1281){
                    bool v1282;
                    v1282 = v1279 < 4;
                    v1283 = v1282;
                } else {
                    v1283 = false;
                }
                bool v1284;
                v1284 = v1283 == false;
                if (v1284){
                    assert("The indices should be inside the range of the dimension." && v1283);
                } else {
                }
                bool v1286;
                v1286 = 0 <= v1256;
                bool v1288;
                if (v1286){
                    bool v1287;
                    v1287 = v1256 < 16;
                    v1288 = v1287;
                } else {
                    v1288 = false;
                }
                bool v1289;
                v1289 = v1288 == false;
                if (v1289){
                    assert("The indices should be inside the range of the dimension." && v1288);
                } else {
                }
                int v1291;
                v1291 = v1256 * 4;
                int v1292;
                v1292 = v1279 + v1291;
                bool v1293;
                v1293 = 0 <= v1277;
                bool v1295;
                if (v1293){
                    bool v1294;
                    v1294 = v1277 < 1;
                    v1295 = v1294;
                } else {
                    v1295 = false;
                }
                bool v1296;
                v1296 = v1295 == false;
                if (v1296){
                    assert("The indices should be inside the range of the dimension." && v1295);
                } else {
                }
                int v1298;
                v1298 = v1277 * 64;
                int v1299;
                v1299 = v1292 + v1298;
                assert("Tensor range check" && 0 <= v1277 && v1277 < 1);
                assert("Tensor range check" && 0 <= v1279 && v1279 < 4);
                int v1300;
                v1300 = 4 * v1277;
                int v1301;
                v1301 = v1300 + v1279;
                v1269[v1301] = v1299;
                v1279 += 1 ;
            }
            v1277 += 1 ;
        }
        bool v1302;
        v1302 = 0 <= v1257;
        bool v1303;
        v1303 = v1302 && v1258;
        bool v1304;
        v1304 = v1303 == false;
        if (v1304){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1303);
        } else {
        }
        bool v1306;
        v1306 = 0 <= v1264;
        bool v1308;
        if (v1306){
            bool v1307;
            v1307 = v1264 < 8;
            v1308 = v1307;
        } else {
            v1308 = false;
        }
        bool v1309;
        v1309 = v1308 == false;
        if (v1309){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1308);
        } else {
        }
        int v1311;
        v1311 = v1264 * 16;
        int v1312;
        v1312 = v1311 + v1257;
        bool v1313[4];
        int v1314;
        v1314 = 0;
        while (while_method_3(v1314)){
            int v1316;
            v1316 = 0;
            while (while_method_1(v1316)){
                assert("Tensor range check" && 0 <= v1314 && v1314 < 1);
                assert("Tensor range check" && 0 <= v1316 && v1316 < 4);
                int v1318;
                v1318 = 4 * v1314;
                int v1319;
                v1319 = v1318 + v1316;
                float v1320;
                v1320 = v1268[v1319];
                int v1321;
                v1321 = v1269[v1319];
                bool v1322;
                v1322 = v1321 < 11;
                assert("Tensor range check" && 0 <= v1314 && v1314 < 1);
                assert("Tensor range check" && 0 <= v1316 && v1316 < 4);
                v1313[v1319] = v1322;
                v1316 += 1 ;
            }
            v1314 += 1 ;
        }
        float v1323[4];
        int v1324;
        v1324 = 0;
        while (while_method_3(v1324)){
            int v1326;
            v1326 = 0;
            while (while_method_1(v1326)){
                assert("Tensor range check" && 0 <= v1324 && v1324 < 1);
                assert("Tensor range check" && 0 <= v1326 && v1326 < 4);
                int v1328;
                v1328 = 4 * v1324;
                int v1329;
                v1329 = v1328 + v1326;
                float v1330;
                v1330 = v1268[v1329];
                bool v1331;
                v1331 = v1313[v1329];
                float v1332;
                if (v1331){
                    v1332 = v1330;
                } else {
                    v1332 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1324 && v1324 < 1);
                assert("Tensor range check" && 0 <= v1326 && v1326 < 4);
                v1323[v1329] = v1332;
                v1326 += 1 ;
            }
            v1324 += 1 ;
        }
        float v1333;
        v1333 = 0.0f;
        int v1334;
        v1334 = 0;
        while (while_method_3(v1334)){
            int v1336;
            v1336 = 0;
            while (while_method_1(v1336)){
                assert("Tensor range check" && 0 <= v1334 && v1334 < 1);
                assert("Tensor range check" && 0 <= v1336 && v1336 < 4);
                int v1338;
                v1338 = 4 * v1334;
                int v1339;
                v1339 = v1338 + v1336;
                float v1340;
                v1340 = v1323[v1339];
                float v1341;
                v1341 = v1333 + v1340;
                v1333 = v1341;
                v1336 += 1 ;
            }
            v1334 += 1 ;
        }
        auto v1342 = cooperative_groups::coalesced_threads();
        int v1343;
        v1343 = threadIdx.x;
        int v1344;
        v1344 = v1343 / 16;
        auto v1345 = cooperative_groups::labeled_partition(v1342,v1344);
        float v1346;
        v1346 = cooperative_groups::reduce(v1345, v1333, v42);
        int v1347[4];
        int v1348;
        v1348 = 0;
        while (while_method_3(v1348)){
            int v1350;
            v1350 = 0;
            while (while_method_1(v1350)){
                assert("Tensor range check" && 0 <= v1348 && v1348 < 1);
                assert("Tensor range check" && 0 <= v1350 && v1350 < 4);
                int v1352;
                v1352 = 4 * v1348;
                int v1353;
                v1353 = v1352 + v1350;
                bool v1354;
                v1354 = v1313[v1353];
                int v1355;
                if (v1354){
                    v1355 = 1;
                } else {
                    v1355 = 0;
                }
                assert("Tensor range check" && 0 <= v1348 && v1348 < 1);
                assert("Tensor range check" && 0 <= v1350 && v1350 < 4);
                v1347[v1353] = v1355;
                v1350 += 1 ;
            }
            v1348 += 1 ;
        }
        int v1356;
        v1356 = 0;
        int v1357;
        v1357 = 0;
        while (while_method_3(v1357)){
            int v1359;
            v1359 = 0;
            while (while_method_1(v1359)){
                assert("Tensor range check" && 0 <= v1357 && v1357 < 1);
                assert("Tensor range check" && 0 <= v1359 && v1359 < 4);
                int v1361;
                v1361 = 4 * v1357;
                int v1362;
                v1362 = v1361 + v1359;
                int v1363;
                v1363 = v1347[v1362];
                int v1364;
                v1364 = v1356 + v1363;
                v1356 = v1364;
                v1359 += 1 ;
            }
            v1357 += 1 ;
        }
        auto v1365 = cooperative_groups::coalesced_threads();
        int v1366;
        v1366 = threadIdx.x;
        int v1367;
        v1367 = v1366 / 16;
        auto v1368 = cooperative_groups::labeled_partition(v1365,v1367);
        Closure4 v1369{};
        int v1370;
        v1370 = cooperative_groups::reduce(v1368, v1356, v1369);
        float v1371;
        v1371 = (float)v1370;
        float v1372;
        v1372 = v1346 / v1371;
        float v1373[4];
        int v1374;
        v1374 = 0;
        while (while_method_3(v1374)){
            int v1376;
            v1376 = 0;
            while (while_method_1(v1376)){
                assert("Tensor range check" && 0 <= v1374 && v1374 < 1);
                assert("Tensor range check" && 0 <= v1376 && v1376 < 4);
                int v1378;
                v1378 = 4 * v1374;
                int v1379;
                v1379 = v1378 + v1376;
                float v1380;
                v1380 = v1268[v1379];
                bool v1381;
                v1381 = v1313[v1379];
                float v1382;
                if (v1381){
                    v1382 = v1380;
                } else {
                    v1382 = -1.0f / 0.0f;
                }
                float v1383;
                v1383 = v1382 - v1372;
                float v1384;
                v1384 = exp(v1383);
                bool v1385;
                v1385 = v1384 < 1.0f / 0.0f;
                bool v1386;
                v1386 = v1385 == false;
                if (v1386){
                    assert("The softmax values must not grow too large." && v1385);
                } else {
                }
                bool v1388;
                v1388 = isnan(v1384);
                bool v1389;
                v1389 = v1388 == false;
                bool v1390;
                v1390 = v1389 == false;
                if (v1390){
                    assert("The softmax values must not be nans." && v1389);
                } else {
                }
                assert("Tensor range check" && 0 <= v1374 && v1374 < 1);
                assert("Tensor range check" && 0 <= v1376 && v1376 < 4);
                v1373[v1379] = v1384;
                v1376 += 1 ;
            }
            v1374 += 1 ;
        }
        float v1392;
        v1392 = 0.0f;
        int v1393;
        v1393 = 0;
        while (while_method_3(v1393)){
            int v1395;
            v1395 = 0;
            while (while_method_1(v1395)){
                assert("Tensor range check" && 0 <= v1393 && v1393 < 1);
                assert("Tensor range check" && 0 <= v1395 && v1395 < 4);
                int v1397;
                v1397 = 4 * v1393;
                int v1398;
                v1398 = v1397 + v1395;
                float v1399;
                v1399 = v1373[v1398];
                float v1400;
                v1400 = v1392 + v1399;
                v1392 = v1400;
                v1395 += 1 ;
            }
            v1393 += 1 ;
        }
        auto v1401 = cooperative_groups::coalesced_threads();
        int v1402;
        v1402 = threadIdx.x;
        int v1403;
        v1403 = v1402 / 16;
        auto v1404 = cooperative_groups::labeled_partition(v1401,v1403);
        float v1405;
        v1405 = cooperative_groups::reduce(v1404, v1392, v42);
        float v1406[4];
        int v1407;
        v1407 = 0;
        while (while_method_3(v1407)){
            int v1409;
            v1409 = 0;
            while (while_method_1(v1409)){
                assert("Tensor range check" && 0 <= v1407 && v1407 < 1);
                assert("Tensor range check" && 0 <= v1409 && v1409 < 4);
                int v1411;
                v1411 = 4 * v1407;
                int v1412;
                v1412 = v1411 + v1409;
                float v1413;
                v1413 = v1373[v1412];
                float v1414;
                v1414 = v1413 / v1405;
                assert("Tensor range check" && 0 <= v1407 && v1407 < 1);
                assert("Tensor range check" && 0 <= v1409 && v1409 < 4);
                v1406[v1412] = v1414;
                v1409 += 1 ;
            }
            v1407 += 1 ;
        }
        float v1415[4];
        float v1416;
        v1416 = 0.0f;
        int v1417;
        v1417 = 0;
        while (while_method_3(v1417)){
            assert("Tensor range check" && 0 <= v1417 && v1417 < 1);
            int v1419;
            v1419 = 4 * v1417;
            assert("Tensor range check" && 0 <= v1417 && v1417 < 1);
            float v1420;
            v1420 = 0.0f;
            int v1421;
            v1421 = 0;
            while (while_method_1(v1421)){
                assert("Tensor range check" && 0 <= v1421 && v1421 < 4);
                int v1423;
                v1423 = v1421 + v1419;
                float v1424;
                v1424 = v1406[v1423];
                float v1425;
                v1425 = v1420 + v1424;
                v1420 = v1425;
                v1421 += 1 ;
            }
            auto v1426 = cooperative_groups::coalesced_threads();
            int v1427;
            v1427 = threadIdx.x;
            int v1428;
            v1428 = v1427 / 16;
            auto v1429 = cooperative_groups::labeled_partition(v1426,v1428);
            Closure2 v1430{};
            float v1431;
            v1431 = cooperative_groups::inclusive_scan(v1429, v1420, v1430);
            float v1432;
            v1432 = v1429.shfl_up(v1431,1);
            bool v1433;
            v1433 = v1429.thread_rank() == 0;
            float v1434;
            if (v1433){
                v1434 = 0.0f;
            } else {
                v1434 = v1432;
            }
            float v1435;
            v1435 = v1429.shfl(v1431,v1429.num_threads()-1);
            float v1436;
            v1436 = v1416 + v1434;
            float v1437;
            v1437 = v1436;
            int v1438;
            v1438 = 0;
            while (while_method_1(v1438)){
                assert("Tensor range check" && 0 <= v1438 && v1438 < 4);
                int v1440;
                v1440 = v1438 + v1419;
                float v1441;
                v1441 = v1406[v1440];
                float v1442;
                v1442 = v1437 + v1441;
                assert("Tensor range check" && 0 <= v1438 && v1438 < 4);
                v1415[v1440] = v1442;
                v1437 = v1442;
                v1438 += 1 ;
            }
            float v1443;
            v1443 = v1416 + v1435;
            v1416 = v1443;
            v1417 += 1 ;
        }
        float v1444[4];
        bool v1445[4];
        int v1446;
        v1446 = 0;
        while (while_method_3(v1446)){
            int v1448;
            v1448 = 0;
            while (while_method_1(v1448)){
                assert("Tensor range check" && 0 <= v1446 && v1446 < 1);
                assert("Tensor range check" && 0 <= v1448 && v1448 < 4);
                int v1450;
                v1450 = 4 * v1446;
                int v1451;
                v1451 = v1450 + v1448;
                float v1452;
                v1452 = v1415[v1451];
                float v1453;
                v1453 = v1406[v1451];
                bool v1454;
                v1454 = v1453 > 0.0f;
                assert("Tensor range check" && 0 <= v1446 && v1446 < 1);
                assert("Tensor range check" && 0 <= v1448 && v1448 < 4);
                v1444[v1451] = v1452;
                v1445[v1451] = v1454;
                v1448 += 1 ;
            }
            v1446 += 1 ;
        }
        float v1455; bool v1456;
        Tuple2 tmp24 = Tuple2{-1.0f / 0.0f, false};
        v1455 = tmp24.v0; v1456 = tmp24.v1;
        int v1457;
        v1457 = 0;
        while (while_method_3(v1457)){
            int v1459;
            v1459 = 0;
            while (while_method_1(v1459)){
                assert("Tensor range check" && 0 <= v1457 && v1457 < 1);
                assert("Tensor range check" && 0 <= v1459 && v1459 < 4);
                int v1461;
                v1461 = 4 * v1457;
                int v1462;
                v1462 = v1461 + v1459;
                float v1463;
                v1463 = v1444[v1462];
                bool v1464;
                v1464 = v1445[v1462];
                float v1471; bool v1472;
                if (v1456){
                    if (v1464){
                        bool v1465;
                        v1465 = v1455 >= v1463;
                        float v1466;
                        if (v1465){
                            v1466 = v1455;
                        } else {
                            v1466 = v1463;
                        }
                        v1471 = v1466; v1472 = true;
                    } else {
                        v1471 = v1455; v1472 = v1456;
                    }
                } else {
                    if (v1464){
                        v1471 = v1463; v1472 = v1464;
                    } else {
                        v1471 = v1455; v1472 = v1456;
                    }
                }
                v1455 = v1471;
                v1456 = v1472;
                v1459 += 1 ;
            }
            v1457 += 1 ;
        }
        auto v1473 = cooperative_groups::coalesced_threads();
        int v1474;
        v1474 = threadIdx.x;
        int v1475;
        v1475 = v1474 / 16;
        auto v1476 = cooperative_groups::labeled_partition(v1473,v1475);
        Closure5 v1477{};
        float v1478; bool v1479;
        Tuple2 tmp25 = cooperative_groups::reduce(v1476, Tuple2{v1455, v1456}, v1477);
        v1478 = tmp25.v0; v1479 = tmp25.v1;
        bool v1480;
        v1480 = v1479 == false;
        if (v1480){
            assert("The local reduce must be true." && v1479);
        } else {
        }
        float v1482[4];
        int v1483[4];
        int v1484;
        v1484 = 0;
        while (while_method_3(v1484)){
            int v1486;
            v1486 = 0;
            while (while_method_1(v1486)){
                assert("Tensor range check" && 0 <= v1484 && v1484 < 1);
                assert("Tensor range check" && 0 <= v1486 && v1486 < 4);
                int v1488;
                v1488 = 4 * v1484;
                int v1489;
                v1489 = v1488 + v1486;
                int v1490;
                v1490 = v1269[v1489];
                float v1491;
                v1491 = curand_uniform(&v1251);
                assert("Tensor range check" && 0 <= v1484 && v1484 < 1);
                assert("Tensor range check" && 0 <= v1486 && v1486 < 4);
                v1482[v1489] = v1491;
                v1483[v1489] = v1490;
                v1486 += 1 ;
            }
            v1484 += 1 ;
        }
        float v1492; int v1493;
        Tuple1 tmp26 = Tuple1{0.0f, 2147483647};
        v1492 = tmp26.v0; v1493 = tmp26.v1;
        int v1494;
        v1494 = 0;
        while (while_method_3(v1494)){
            int v1496;
            v1496 = 0;
            while (while_method_1(v1496)){
                assert("Tensor range check" && 0 <= v1494 && v1494 < 1);
                assert("Tensor range check" && 0 <= v1496 && v1496 < 4);
                int v1498;
                v1498 = 4 * v1494;
                int v1499;
                v1499 = v1498 + v1496;
                float v1500;
                v1500 = v1482[v1499];
                int v1501;
                v1501 = v1483[v1499];
                bool v1502;
                v1502 = v1493 < v1501;
                float v1503; int v1504;
                if (v1502){
                    v1503 = v1492; v1504 = v1493;
                } else {
                    v1503 = v1500; v1504 = v1501;
                }
                v1492 = v1503;
                v1493 = v1504;
                v1496 += 1 ;
            }
            v1494 += 1 ;
        }
        auto v1505 = cooperative_groups::coalesced_threads();
        int v1506;
        v1506 = threadIdx.x;
        int v1507;
        v1507 = v1506 / 16;
        auto v1508 = cooperative_groups::labeled_partition(v1505,v1507);
        Closure6 v1509{};
        float v1510; int v1511;
        Tuple1 tmp27 = cooperative_groups::reduce(v1508, Tuple1{v1492, v1493}, v1509);
        v1510 = tmp27.v0; v1511 = tmp27.v1;
        float v1512;
        v1512 = v1478 * v1510;
        int v1513[4];
        bool v1514[4];
        int v1515;
        v1515 = 0;
        while (while_method_3(v1515)){
            int v1517;
            v1517 = 0;
            while (while_method_1(v1517)){
                assert("Tensor range check" && 0 <= v1515 && v1515 < 1);
                assert("Tensor range check" && 0 <= v1517 && v1517 < 4);
                int v1519;
                v1519 = 4 * v1515;
                int v1520;
                v1520 = v1519 + v1517;
                float v1521;
                v1521 = v1444[v1520];
                bool v1522;
                v1522 = v1445[v1520];
                int v1523;
                v1523 = v1269[v1520];
                int v1526; bool v1527;
                if (v1522){
                    float v1524;
                    v1524 = v1521 - v1512;
                    bool v1525;
                    v1525 = v1524 >= 0.0f;
                    v1526 = v1523; v1527 = v1525;
                } else {
                    v1526 = 2147483647; v1527 = false;
                }
                assert("Tensor range check" && 0 <= v1515 && v1515 < 1);
                assert("Tensor range check" && 0 <= v1517 && v1517 < 4);
                v1513[v1520] = v1526;
                v1514[v1520] = v1527;
                v1517 += 1 ;
            }
            v1515 += 1 ;
        }
        int v1528; bool v1529;
        Tuple3 tmp28 = Tuple3{2147483647, false};
        v1528 = tmp28.v0; v1529 = tmp28.v1;
        int v1530;
        v1530 = 0;
        while (while_method_3(v1530)){
            int v1532;
            v1532 = 0;
            while (while_method_1(v1532)){
                assert("Tensor range check" && 0 <= v1530 && v1530 < 1);
                assert("Tensor range check" && 0 <= v1532 && v1532 < 4);
                int v1534;
                v1534 = 4 * v1530;
                int v1535;
                v1535 = v1534 + v1532;
                int v1536;
                v1536 = v1513[v1535];
                bool v1537;
                v1537 = v1514[v1535];
                int v1544; bool v1545;
                if (v1529){
                    if (v1537){
                        bool v1538;
                        v1538 = v1528 < v1536;
                        int v1539;
                        if (v1538){
                            v1539 = v1528;
                        } else {
                            v1539 = v1536;
                        }
                        v1544 = v1539; v1545 = true;
                    } else {
                        v1544 = v1528; v1545 = v1529;
                    }
                } else {
                    if (v1537){
                        v1544 = v1536; v1545 = v1537;
                    } else {
                        v1544 = v1528; v1545 = v1529;
                    }
                }
                v1528 = v1544;
                v1529 = v1545;
                v1532 += 1 ;
            }
            v1530 += 1 ;
        }
        auto v1546 = cooperative_groups::coalesced_threads();
        int v1547;
        v1547 = threadIdx.x;
        int v1548;
        v1548 = v1547 / 16;
        auto v1549 = cooperative_groups::labeled_partition(v1546,v1548);
        Closure7 v1550{};
        int v1551; bool v1552;
        Tuple3 tmp29 = cooperative_groups::reduce(v1549, Tuple3{v1528, v1529}, v1550);
        v1551 = tmp29.v0; v1552 = tmp29.v1;
        bool v1553;
        v1553 = v1552 == false;
        if (v1553){
            assert("The local reduce must be true." && v1552);
        } else {
        }
        assert("Tensor range check" && 0 <= v1264 && v1264 < 8);
        int v1555;
        v1555 = 0;
        while (while_method_3(v1555)){
            assert("Tensor range check" && 0 <= v1555 && v1555 < 1);
            int v1557;
            v1557 = 64 * v1555;
            int v1558;
            v1558 = v1557 + v1267;
            assert("Tensor range check" && 0 <= v1555 && v1555 < 1);
            int v1559;
            v1559 = 4 * v1555;
            int4* v1560;
            v1560 = reinterpret_cast<int4*>(v1406 + v1559);
            int4* v1561;
            v1561 = reinterpret_cast<int4*>(v16 + v1558);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1560) % 16 == 0 && reinterpret_cast<unsigned long long>(v1561) % 16 == 0);
            *v1561 = *v1560;
            v1555 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1264 && v1264 < 8);
        int v1562;
        v1562 = 16 * v1264;
        int v1563;
        v1563 = v1562 + v1257;
        v17[v1563] = v1551;
        v1264 += 1 ;
    }
    __syncthreads();
    return ;
}
extern "C" __global__ void entry2(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 256);
    int v9;
    v9 = 16 * v8;
    int v10;
    v10 = threadIdx.x;
    assert("Tensor range check" && 0 <= v10 && v10 < 256);
    int v11;
    v11 = 16 * v10;
    int v12;
    v12 = threadIdx.x;
    assert("Tensor range check" && 0 <= v12 && v12 < 256);
    int v13;
    v13 = 16 * v12;
    int v14;
    v14 = threadIdx.x;
    assert("Tensor range check" && 0 <= v14 && v14 < 256);
    int v15;
    v15 = 16 * v14;
    int v16;
    v16 = threadIdx.x;
    assert("Tensor range check" && 0 <= v16 && v16 < 256);
    int v17;
    v17 = 16 * v16;
    float * v18;
    v18 = v1+v9;
    int * v20;
    v20 = v2+v15;
    int * v22;
    v22 = v3+v15;
    int v24;
    v24 = sizeof(float *);
    unsigned long long v25;
    v25 = (unsigned long long)v24;
    unsigned long long v26;
    v26 = 256ull * v25;
    unsigned long long v27;
    v27 = v26 + 16ull;
    unsigned long long v28;
    v28 = v27 - 1ull;
    unsigned long long v29;
    v29 = v28 % 16ull;
    unsigned long long v30;
    v30 = v28 - v29;
    int v31;
    v31 = sizeof(int *);
    unsigned long long v32;
    v32 = (unsigned long long)v31;
    unsigned long long v33;
    v33 = 256ull * v32;
    unsigned long long v34;
    v34 = v30 + v33;
    unsigned long long v35;
    v35 = v34 + 16ull;
    unsigned long long v36;
    v36 = v35 - 1ull;
    unsigned long long v37;
    v37 = v36 % 16ull;
    unsigned long long v38;
    v38 = v36 - v37;
    unsigned long long v39;
    v39 = v38 + v33;
    bool v40;
    v40 = v39 <= 98304ull;
    bool v41;
    v41 = v40 == false;
    if (v41){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v40);
    } else {
    }
    extern __shared__ unsigned char v43[];
    bool v44;
    v44 = v39 <= v39;
    bool v45;
    v45 = v44 == false;
    if (v45){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v44);
    } else {
    }
    float * * v47;
    v47 = reinterpret_cast<float * *>(&v43[0ull]);
    int * * v49;
    v49 = reinterpret_cast<int * *>(&v43[v30]);
    int * * v51;
    v51 = reinterpret_cast<int * *>(&v43[v38]);
    int v53;
    v53 = threadIdx.x;
    assert("Tensor range check" && 0 <= v53 && v53 < 256);
    v47[v53] = v18;
    v49[v53] = v20;
    v51[v53] = v22;
    __syncthreads();
    bool v54;
    v54 = 0 <= v53;
    bool v55;
    v55 = v54 == false;
    if (v55){
        assert("The index needs to be zero or positive." && v54);
    } else {
    }
    int v57;
    v57 = v53 % 4;
    int v58;
    v58 = v53 / 4;
    bool v59;
    v59 = v58 < 64;
    bool v60;
    v60 = v59 == false;
    if (v60){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v59);
    } else {
    }
    assert("Tensor range check" && 0 <= v58 && v58 < 64);
    int v62;
    v62 = 0;
    while (while_method_1(v62)){
        bool v64;
        v64 = 0 <= v58;
        bool v65;
        v65 = v64 && v59;
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v65);
        } else {
        }
        bool v68;
        v68 = 0 <= v62;
        bool v70;
        if (v68){
            bool v69;
            v69 = v62 < 4;
            v70 = v69;
        } else {
            v70 = false;
        }
        bool v71;
        v71 = v70 == false;
        if (v71){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v70);
        } else {
        }
        int v73;
        v73 = v62 * 64;
        int v74;
        v74 = v73 + v58;
        assert("Tensor range check" && 0 <= v62 && v62 < 4);
        int v75;
        v75 = 64 * v62;
        int v76;
        v76 = v75 + v58;
        float * v77;
        v77 = v47[v76];
        int * v78;
        v78 = v49[v76];
        int * v79;
        v79 = v51[v76];
        int v80;
        v80 = blockIdx.x;
        int v81;
        v81 = v80 * 256;
        int v82;
        v82 = v81 + v74;
        assert("Tensor range check" && 0 <= v57 && v57 < 4);
        int v83;
        v83 = 4 * v57;
        float v84[4];
        int v85[4];
        int v86;
        v86 = 0;
        while (while_method_3(v86)){
            assert("Tensor range check" && 0 <= v86 && v86 < 1);
            int v88;
            v88 = 4 * v86;
            assert("Tensor range check" && 0 <= v86 && v86 < 1);
            int v89;
            v89 = 16 * v86;
            int v90;
            v90 = v89 + v83;
            int4* v91;
            v91 = reinterpret_cast<int4*>(v77 + v90);
            int4* v92;
            v92 = reinterpret_cast<int4*>(v84 + v88);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v91) % 16 == 0 && reinterpret_cast<unsigned long long>(v92) % 16 == 0);
            *v92 = *v91;
            v86 += 1 ;
        }
        int v93;
        v93 = 0;
        while (while_method_3(v93)){
            int v95;
            v95 = 0;
            while (while_method_1(v95)){
                bool v97;
                v97 = 0 <= v95;
                bool v99;
                if (v97){
                    bool v98;
                    v98 = v95 < 4;
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
                bool v102;
                v102 = 0 <= v57;
                bool v104;
                if (v102){
                    bool v103;
                    v103 = v57 < 4;
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
                int v107;
                v107 = v57 * 4;
                int v108;
                v108 = v95 + v107;
                bool v109;
                v109 = 0 <= v93;
                bool v111;
                if (v109){
                    bool v110;
                    v110 = v93 < 1;
                    v111 = v110;
                } else {
                    v111 = false;
                }
                bool v112;
                v112 = v111 == false;
                if (v112){
                    assert("The indices should be inside the range of the dimension." && v111);
                } else {
                }
                int v114;
                v114 = v93 * 16;
                int v115;
                v115 = v108 + v114;
                assert("Tensor range check" && 0 <= v93 && v93 < 1);
                assert("Tensor range check" && 0 <= v95 && v95 < 4);
                int v116;
                v116 = 4 * v93;
                int v117;
                v117 = v116 + v95;
                v85[v117] = v115;
                v95 += 1 ;
            }
            v93 += 1 ;
        }
        int v118[4];
        int v119[4];
        int v120;
        v120 = 0;
        while (while_method_3(v120)){
            int v122;
            v122 = 0;
            while (while_method_1(v122)){
                assert("Tensor range check" && 0 <= v120 && v120 < 1);
                assert("Tensor range check" && 0 <= v122 && v122 < 4);
                int v124;
                v124 = 4 * v120;
                int v125;
                v125 = v124 + v122;
                int v126;
                v126 = v85[v125];
                assert("Tensor range check" && 0 <= v120 && v120 < 1);
                assert("Tensor range check" && 0 <= v122 && v122 < 4);
                v118[v125] = v82;
                v119[v125] = v126;
                v122 += 1 ;
            }
            v120 += 1 ;
        }
        int v127;
        v127 = 0;
        while (while_method_3(v127)){
            assert("Tensor range check" && 0 <= v127 && v127 < 1);
            int v129;
            v129 = 16 * v127;
            int v130;
            v130 = v129 + v83;
            assert("Tensor range check" && 0 <= v127 && v127 < 1);
            int v131;
            v131 = 4 * v127;
            int4* v132;
            v132 = reinterpret_cast<int4*>(v118 + v131);
            int4* v133;
            v133 = reinterpret_cast<int4*>(v78 + v130);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v132) % 16 == 0 && reinterpret_cast<unsigned long long>(v133) % 16 == 0);
            *v133 = *v132;
            int4* v134;
            v134 = reinterpret_cast<int4*>(v119 + v131);
            int4* v135;
            v135 = reinterpret_cast<int4*>(v79 + v130);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v134) % 16 == 0 && reinterpret_cast<unsigned long long>(v135) % 16 == 0);
            *v135 = *v134;
            v127 += 1 ;
        }
        assert("Tensor range check" && 0 <= v74 && v74 < 256);
        v62 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v53 && v53 < 256);
    __syncthreads();
    float * v136;
    v136 = v1+v9;
    unsigned long long v138;
    v138 = v30 + 1024ull;
    bool v139;
    v139 = v138 <= 98304ull;
    bool v140;
    v140 = v139 == false;
    if (v140){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v139);
    } else {
    }
    extern __shared__ unsigned char v142[];
    bool v143;
    v143 = v138 <= v138;
    bool v144;
    v144 = v143 == false;
    if (v144){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v143);
    } else {
    }
    float * * v146;
    v146 = reinterpret_cast<float * *>(&v142[0ull]);
    int * v148;
    v148 = reinterpret_cast<int *>(&v142[v30]);
    int v150;
    v150 = threadIdx.x;
    assert("Tensor range check" && 0 <= v150 && v150 < 256);
    v146[v150] = v136;
    __syncthreads();
    bool v151;
    v151 = 0 <= v150;
    bool v152;
    v152 = v151 == false;
    if (v152){
        assert("The index needs to be zero or positive." && v151);
    } else {
    }
    int v154;
    v154 = v150 % 4;
    int v155;
    v155 = v150 / 4;
    bool v156;
    v156 = v155 < 64;
    bool v157;
    v157 = v156 == false;
    if (v157){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v156);
    } else {
    }
    assert("Tensor range check" && 0 <= v155 && v155 < 64);
    int v159;
    v159 = 0;
    while (while_method_1(v159)){
        bool v161;
        v161 = 0 <= v155;
        bool v162;
        v162 = v161 && v156;
        bool v163;
        v163 = v162 == false;
        if (v163){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v162);
        } else {
        }
        bool v165;
        v165 = 0 <= v159;
        bool v167;
        if (v165){
            bool v166;
            v166 = v159 < 4;
            v167 = v166;
        } else {
            v167 = false;
        }
        bool v168;
        v168 = v167 == false;
        if (v168){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v167);
        } else {
        }
        int v170;
        v170 = v159 * 64;
        int v171;
        v171 = v170 + v155;
        assert("Tensor range check" && 0 <= v159 && v159 < 4);
        int v172;
        v172 = 64 * v159;
        int v173;
        v173 = v172 + v155;
        float * v174;
        v174 = v146[v173];
        int v175;
        v175 = blockIdx.x;
        int v176;
        v176 = v175 * 256;
        int v177;
        v177 = v176 + v171;
        assert("Tensor range check" && 0 <= v154 && v154 < 4);
        int v178;
        v178 = 4 * v154;
        float v179[4];
        int v180[4];
        int v181;
        v181 = 0;
        while (while_method_3(v181)){
            assert("Tensor range check" && 0 <= v181 && v181 < 1);
            int v183;
            v183 = 4 * v181;
            assert("Tensor range check" && 0 <= v181 && v181 < 1);
            int v184;
            v184 = 16 * v181;
            int v185;
            v185 = v184 + v178;
            int4* v186;
            v186 = reinterpret_cast<int4*>(v174 + v185);
            int4* v187;
            v187 = reinterpret_cast<int4*>(v179 + v183);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v186) % 16 == 0 && reinterpret_cast<unsigned long long>(v187) % 16 == 0);
            *v187 = *v186;
            v181 += 1 ;
        }
        int v188;
        v188 = 0;
        while (while_method_3(v188)){
            int v190;
            v190 = 0;
            while (while_method_1(v190)){
                bool v192;
                v192 = 0 <= v190;
                bool v194;
                if (v192){
                    bool v193;
                    v193 = v190 < 4;
                    v194 = v193;
                } else {
                    v194 = false;
                }
                bool v195;
                v195 = v194 == false;
                if (v195){
                    assert("The indices should be inside the range of the dimension." && v194);
                } else {
                }
                bool v197;
                v197 = 0 <= v154;
                bool v199;
                if (v197){
                    bool v198;
                    v198 = v154 < 4;
                    v199 = v198;
                } else {
                    v199 = false;
                }
                bool v200;
                v200 = v199 == false;
                if (v200){
                    assert("The indices should be inside the range of the dimension." && v199);
                } else {
                }
                int v202;
                v202 = v154 * 4;
                int v203;
                v203 = v190 + v202;
                bool v204;
                v204 = 0 <= v188;
                bool v206;
                if (v204){
                    bool v205;
                    v205 = v188 < 1;
                    v206 = v205;
                } else {
                    v206 = false;
                }
                bool v207;
                v207 = v206 == false;
                if (v207){
                    assert("The indices should be inside the range of the dimension." && v206);
                } else {
                }
                int v209;
                v209 = v188 * 16;
                int v210;
                v210 = v203 + v209;
                assert("Tensor range check" && 0 <= v188 && v188 < 1);
                assert("Tensor range check" && 0 <= v190 && v190 < 4);
                int v211;
                v211 = 4 * v188;
                int v212;
                v212 = v211 + v190;
                v180[v212] = v210;
                v190 += 1 ;
            }
            v188 += 1 ;
        }
        int v213;
        v213 = 0;
        while (while_method_3(v213)){
            assert("Tensor range check" && 0 <= v213 && v213 < 1);
            assert("Tensor range check" && 0 <= v213 && v213 < 1);
            v213 += 1 ;
        }
        assert("Tensor range check" && 0 <= v171 && v171 < 256);
        v148[v171] = v177;
        v159 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v150 && v150 < 256);
    int v215;
    v215 = v148[v150];
    __syncthreads();
    int v216;
    v216 = threadIdx.x;
    assert("Tensor range check" && 0 <= v216 && v216 < 256);
    v4[v216] = v215;
    float * v217;
    v217 = v1+v9;
    float * v219;
    v219 = v6+v17;
    unsigned long long v221;
    v221 = v30 + v26;
    bool v222;
    v222 = v221 <= 98304ull;
    bool v223;
    v223 = v222 == false;
    if (v223){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v222);
    } else {
    }
    extern __shared__ unsigned char v225[];
    bool v226;
    v226 = v221 <= v221;
    bool v227;
    v227 = v226 == false;
    if (v227){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v226);
    } else {
    }
    float * * v229;
    v229 = reinterpret_cast<float * *>(&v225[0ull]);
    float * * v231;
    v231 = reinterpret_cast<float * *>(&v225[v30]);
    int v233;
    v233 = threadIdx.x;
    assert("Tensor range check" && 0 <= v233 && v233 < 256);
    v229[v233] = v217;
    v231[v233] = v219;
    __syncthreads();
    bool v234;
    v234 = 0 <= v233;
    bool v235;
    v235 = v234 == false;
    if (v235){
        assert("The index needs to be zero or positive." && v234);
    } else {
    }
    int v237;
    v237 = v233 % 4;
    int v238;
    v238 = v233 / 4;
    bool v239;
    v239 = v238 < 64;
    bool v240;
    v240 = v239 == false;
    if (v240){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v239);
    } else {
    }
    assert("Tensor range check" && 0 <= v238 && v238 < 64);
    int v242;
    v242 = 0;
    while (while_method_1(v242)){
        bool v244;
        v244 = 0 <= v238;
        bool v245;
        v245 = v244 && v239;
        bool v246;
        v246 = v245 == false;
        if (v246){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v245);
        } else {
        }
        bool v248;
        v248 = 0 <= v242;
        bool v250;
        if (v248){
            bool v249;
            v249 = v242 < 4;
            v250 = v249;
        } else {
            v250 = false;
        }
        bool v251;
        v251 = v250 == false;
        if (v251){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v250);
        } else {
        }
        int v253;
        v253 = v242 * 64;
        int v254;
        v254 = v253 + v238;
        assert("Tensor range check" && 0 <= v242 && v242 < 4);
        int v255;
        v255 = 64 * v242;
        int v256;
        v256 = v255 + v238;
        float * v257;
        v257 = v229[v256];
        float * v258;
        v258 = v231[v256];
        int v259;
        v259 = blockIdx.x;
        int v260;
        v260 = v259 * 256;
        int v261;
        v261 = v260 + v254;
        assert("Tensor range check" && 0 <= v237 && v237 < 4);
        int v262;
        v262 = 4 * v237;
        float v263[4];
        int v264[4];
        int v265;
        v265 = 0;
        while (while_method_3(v265)){
            assert("Tensor range check" && 0 <= v265 && v265 < 1);
            int v267;
            v267 = 4 * v265;
            assert("Tensor range check" && 0 <= v265 && v265 < 1);
            int v268;
            v268 = 16 * v265;
            int v269;
            v269 = v268 + v262;
            int4* v270;
            v270 = reinterpret_cast<int4*>(v257 + v269);
            int4* v271;
            v271 = reinterpret_cast<int4*>(v263 + v267);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v270) % 16 == 0 && reinterpret_cast<unsigned long long>(v271) % 16 == 0);
            *v271 = *v270;
            v265 += 1 ;
        }
        int v272;
        v272 = 0;
        while (while_method_3(v272)){
            int v274;
            v274 = 0;
            while (while_method_1(v274)){
                bool v276;
                v276 = 0 <= v274;
                bool v278;
                if (v276){
                    bool v277;
                    v277 = v274 < 4;
                    v278 = v277;
                } else {
                    v278 = false;
                }
                bool v279;
                v279 = v278 == false;
                if (v279){
                    assert("The indices should be inside the range of the dimension." && v278);
                } else {
                }
                bool v281;
                v281 = 0 <= v237;
                bool v283;
                if (v281){
                    bool v282;
                    v282 = v237 < 4;
                    v283 = v282;
                } else {
                    v283 = false;
                }
                bool v284;
                v284 = v283 == false;
                if (v284){
                    assert("The indices should be inside the range of the dimension." && v283);
                } else {
                }
                int v286;
                v286 = v237 * 4;
                int v287;
                v287 = v274 + v286;
                bool v288;
                v288 = 0 <= v272;
                bool v290;
                if (v288){
                    bool v289;
                    v289 = v272 < 1;
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
                int v293;
                v293 = v272 * 16;
                int v294;
                v294 = v287 + v293;
                assert("Tensor range check" && 0 <= v272 && v272 < 1);
                assert("Tensor range check" && 0 <= v274 && v274 < 4);
                int v295;
                v295 = 4 * v272;
                int v296;
                v296 = v295 + v274;
                v264[v296] = v294;
                v274 += 1 ;
            }
            v272 += 1 ;
        }
        int v297;
        v297 = 0;
        while (while_method_3(v297)){
            assert("Tensor range check" && 0 <= v297 && v297 < 1);
            int v299;
            v299 = 16 * v297;
            int v300;
            v300 = v299 + v262;
            assert("Tensor range check" && 0 <= v297 && v297 < 1);
            int v301;
            v301 = 4 * v297;
            int4* v302;
            v302 = reinterpret_cast<int4*>(v263 + v301);
            int4* v303;
            v303 = reinterpret_cast<int4*>(v258 + v300);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v302) % 16 == 0 && reinterpret_cast<unsigned long long>(v303) % 16 == 0);
            *v303 = *v302;
            v297 += 1 ;
        }
        assert("Tensor range check" && 0 <= v254 && v254 < 256);
        v242 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v233 && v233 < 256);
    __syncthreads();
    float * v304;
    v304 = v1+v9;
    float * v306;
    v306 = v7+v13;
    if (v223){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v222);
    } else {
    }
    extern __shared__ unsigned char v309[];
    if (v227){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v226);
    } else {
    }
    float * * v311;
    v311 = reinterpret_cast<float * *>(&v309[0ull]);
    float * * v313;
    v313 = reinterpret_cast<float * *>(&v309[v30]);
    int v315;
    v315 = threadIdx.x;
    assert("Tensor range check" && 0 <= v315 && v315 < 256);
    v311[v315] = v304;
    v313[v315] = v306;
    __syncthreads();
    bool v316;
    v316 = 0 <= v315;
    bool v317;
    v317 = v316 == false;
    if (v317){
        assert("The index needs to be zero or positive." && v316);
    } else {
    }
    int v319;
    v319 = v315 % 4;
    int v320;
    v320 = v315 / 4;
    bool v321;
    v321 = v320 < 64;
    bool v322;
    v322 = v321 == false;
    if (v322){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v321);
    } else {
    }
    assert("Tensor range check" && 0 <= v320 && v320 < 64);
    int v324;
    v324 = 0;
    while (while_method_1(v324)){
        bool v326;
        v326 = 0 <= v320;
        bool v327;
        v327 = v326 && v321;
        bool v328;
        v328 = v327 == false;
        if (v328){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v327);
        } else {
        }
        bool v330;
        v330 = 0 <= v324;
        bool v332;
        if (v330){
            bool v331;
            v331 = v324 < 4;
            v332 = v331;
        } else {
            v332 = false;
        }
        bool v333;
        v333 = v332 == false;
        if (v333){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v332);
        } else {
        }
        int v335;
        v335 = v324 * 64;
        int v336;
        v336 = v335 + v320;
        assert("Tensor range check" && 0 <= v324 && v324 < 4);
        int v337;
        v337 = 64 * v324;
        int v338;
        v338 = v337 + v320;
        float * v339;
        v339 = v311[v338];
        float * v340;
        v340 = v313[v338];
        int v341;
        v341 = blockIdx.x;
        int v342;
        v342 = v341 * 256;
        int v343;
        v343 = v342 + v336;
        assert("Tensor range check" && 0 <= v319 && v319 < 4);
        int v344;
        v344 = 4 * v319;
        float v345[4];
        int v346[4];
        int v347;
        v347 = 0;
        while (while_method_3(v347)){
            assert("Tensor range check" && 0 <= v347 && v347 < 1);
            int v349;
            v349 = 4 * v347;
            assert("Tensor range check" && 0 <= v347 && v347 < 1);
            int v350;
            v350 = 16 * v347;
            int v351;
            v351 = v350 + v344;
            int4* v352;
            v352 = reinterpret_cast<int4*>(v339 + v351);
            int4* v353;
            v353 = reinterpret_cast<int4*>(v345 + v349);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v352) % 16 == 0 && reinterpret_cast<unsigned long long>(v353) % 16 == 0);
            *v353 = *v352;
            v347 += 1 ;
        }
        int v354;
        v354 = 0;
        while (while_method_3(v354)){
            int v356;
            v356 = 0;
            while (while_method_1(v356)){
                bool v358;
                v358 = 0 <= v356;
                bool v360;
                if (v358){
                    bool v359;
                    v359 = v356 < 4;
                    v360 = v359;
                } else {
                    v360 = false;
                }
                bool v361;
                v361 = v360 == false;
                if (v361){
                    assert("The indices should be inside the range of the dimension." && v360);
                } else {
                }
                bool v363;
                v363 = 0 <= v319;
                bool v365;
                if (v363){
                    bool v364;
                    v364 = v319 < 4;
                    v365 = v364;
                } else {
                    v365 = false;
                }
                bool v366;
                v366 = v365 == false;
                if (v366){
                    assert("The indices should be inside the range of the dimension." && v365);
                } else {
                }
                int v368;
                v368 = v319 * 4;
                int v369;
                v369 = v356 + v368;
                bool v370;
                v370 = 0 <= v354;
                bool v372;
                if (v370){
                    bool v371;
                    v371 = v354 < 1;
                    v372 = v371;
                } else {
                    v372 = false;
                }
                bool v373;
                v373 = v372 == false;
                if (v373){
                    assert("The indices should be inside the range of the dimension." && v372);
                } else {
                }
                int v375;
                v375 = v354 * 16;
                int v376;
                v376 = v369 + v375;
                assert("Tensor range check" && 0 <= v354 && v354 < 1);
                assert("Tensor range check" && 0 <= v356 && v356 < 4);
                int v377;
                v377 = 4 * v354;
                int v378;
                v378 = v377 + v356;
                v346[v378] = v376;
                v356 += 1 ;
            }
            v354 += 1 ;
        }
        bool v379[4];
        int v380;
        v380 = 0;
        while (while_method_3(v380)){
            int v382;
            v382 = 0;
            while (while_method_1(v382)){
                assert("Tensor range check" && 0 <= v380 && v380 < 1);
                assert("Tensor range check" && 0 <= v382 && v382 < 4);
                int v384;
                v384 = 4 * v380;
                int v385;
                v385 = v384 + v382;
                float v386;
                v386 = v345[v385];
                int v387;
                v387 = v346[v385];
                bool v388;
                v388 = v387 < 3;
                assert("Tensor range check" && 0 <= v380 && v380 < 1);
                assert("Tensor range check" && 0 <= v382 && v382 < 4);
                v379[v385] = v388;
                v382 += 1 ;
            }
            v380 += 1 ;
        }
        float v389[4];
        int v390;
        v390 = 0;
        while (while_method_3(v390)){
            int v392;
            v392 = 0;
            while (while_method_1(v392)){
                assert("Tensor range check" && 0 <= v390 && v390 < 1);
                assert("Tensor range check" && 0 <= v392 && v392 < 4);
                int v394;
                v394 = 4 * v390;
                int v395;
                v395 = v394 + v392;
                float v396;
                v396 = v345[v395];
                bool v397;
                v397 = v379[v395];
                float v400;
                if (v397){
                    bool v398;
                    v398 = 0.0f >= v396;
                    if (v398){
                        v400 = 0.0f;
                    } else {
                        v400 = v396;
                    }
                } else {
                    v400 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v390 && v390 < 1);
                assert("Tensor range check" && 0 <= v392 && v392 < 4);
                v389[v395] = v400;
                v392 += 1 ;
            }
            v390 += 1 ;
        }
        float v401;
        v401 = 0.0f;
        int v402;
        v402 = 0;
        while (while_method_3(v402)){
            int v404;
            v404 = 0;
            while (while_method_1(v404)){
                assert("Tensor range check" && 0 <= v402 && v402 < 1);
                assert("Tensor range check" && 0 <= v404 && v404 < 4);
                int v406;
                v406 = 4 * v402;
                int v407;
                v407 = v406 + v404;
                float v408;
                v408 = v389[v407];
                float v409;
                v409 = v401 + v408;
                v401 = v409;
                v404 += 1 ;
            }
            v402 += 1 ;
        }
        auto v410 = cooperative_groups::coalesced_threads();
        int v411;
        v411 = threadIdx.x;
        int v412;
        v412 = v411 / 4;
        auto v413 = cooperative_groups::labeled_partition(v410,v412);
        Closure0 v414{};
        float v415;
        v415 = cooperative_groups::reduce(v413, v401, v414);
        int v416[4];
        int v417;
        v417 = 0;
        while (while_method_3(v417)){
            int v419;
            v419 = 0;
            while (while_method_1(v419)){
                assert("Tensor range check" && 0 <= v417 && v417 < 1);
                assert("Tensor range check" && 0 <= v419 && v419 < 4);
                int v421;
                v421 = 4 * v417;
                int v422;
                v422 = v421 + v419;
                bool v423;
                v423 = v379[v422];
                int v424;
                if (v423){
                    v424 = 1;
                } else {
                    v424 = 0;
                }
                assert("Tensor range check" && 0 <= v417 && v417 < 1);
                assert("Tensor range check" && 0 <= v419 && v419 < 4);
                v416[v422] = v424;
                v419 += 1 ;
            }
            v417 += 1 ;
        }
        int v425;
        v425 = 0;
        int v426;
        v426 = 0;
        while (while_method_3(v426)){
            int v428;
            v428 = 0;
            while (while_method_1(v428)){
                assert("Tensor range check" && 0 <= v426 && v426 < 1);
                assert("Tensor range check" && 0 <= v428 && v428 < 4);
                int v430;
                v430 = 4 * v426;
                int v431;
                v431 = v430 + v428;
                int v432;
                v432 = v416[v431];
                int v433;
                v433 = v425 + v432;
                v425 = v433;
                v428 += 1 ;
            }
            v426 += 1 ;
        }
        auto v434 = cooperative_groups::coalesced_threads();
        int v435;
        v435 = threadIdx.x;
        int v436;
        v436 = v435 / 4;
        auto v437 = cooperative_groups::labeled_partition(v434,v436);
        Closure4 v438{};
        int v439;
        v439 = cooperative_groups::reduce(v437, v425, v438);
        float v440;
        v440 = (float)v439;
        float v441;
        v441 = 1.0f / v440;
        float v442[4];
        int v443;
        v443 = 0;
        while (while_method_3(v443)){
            int v445;
            v445 = 0;
            while (while_method_1(v445)){
                assert("Tensor range check" && 0 <= v443 && v443 < 1);
                assert("Tensor range check" && 0 <= v445 && v445 < 4);
                int v447;
                v447 = 4 * v443;
                int v448;
                v448 = v447 + v445;
                float v449;
                v449 = v389[v448];
                bool v450;
                v450 = v379[v448];
                bool v451;
                v451 = v450 == false;
                float v456;
                if (v451){
                    v456 = 0.0f;
                } else {
                    bool v452;
                    v452 = v415 == 0.0f;
                    bool v453;
                    v453 = v452 != true;
                    if (v453){
                        float v454;
                        v454 = v449 / v415;
                        v456 = v454;
                    } else {
                        v456 = v441;
                    }
                }
                assert("Tensor range check" && 0 <= v443 && v443 < 1);
                assert("Tensor range check" && 0 <= v445 && v445 < 4);
                v442[v448] = v456;
                v445 += 1 ;
            }
            v443 += 1 ;
        }
        int v457;
        v457 = 0;
        while (while_method_3(v457)){
            assert("Tensor range check" && 0 <= v457 && v457 < 1);
            int v459;
            v459 = 16 * v457;
            int v460;
            v460 = v459 + v344;
            assert("Tensor range check" && 0 <= v457 && v457 < 1);
            int v461;
            v461 = 4 * v457;
            int4* v462;
            v462 = reinterpret_cast<int4*>(v442 + v461);
            int4* v463;
            v463 = reinterpret_cast<int4*>(v340 + v460);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v462) % 16 == 0 && reinterpret_cast<unsigned long long>(v463) % 16 == 0);
            *v463 = *v462;
            v457 += 1 ;
        }
        assert("Tensor range check" && 0 <= v336 && v336 < 256);
        v324 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v315 && v315 < 256);
    __syncthreads();
    int v464;
    v464 = threadIdx.x;
    int v465;
    v465 = blockIdx.x;
    int v466;
    v466 = v465 * 256;
    int v467;
    v467 = v464 + v466;
    unsigned long long v468;
    v468 = (unsigned long long)v467;
    curandStatePhilox4_32_10_t v469;
    curand_init(12344321ull,v468,0ull,&v469);
    float * v470;
    v470 = v1+v9;
    if (v140){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v139);
    } else {
    }
    extern __shared__ unsigned char v473[];
    if (v144){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v143);
    } else {
    }
    float * * v475;
    v475 = reinterpret_cast<float * *>(&v473[0ull]);
    int * v477;
    v477 = reinterpret_cast<int *>(&v473[v30]);
    int v479;
    v479 = threadIdx.x;
    assert("Tensor range check" && 0 <= v479 && v479 < 256);
    v475[v479] = v470;
    __syncthreads();
    bool v480;
    v480 = 0 <= v479;
    bool v481;
    v481 = v480 == false;
    if (v481){
        assert("The index needs to be zero or positive." && v480);
    } else {
    }
    int v483;
    v483 = v479 % 4;
    int v484;
    v484 = v479 / 4;
    bool v485;
    v485 = v484 < 64;
    bool v486;
    v486 = v485 == false;
    if (v486){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v485);
    } else {
    }
    assert("Tensor range check" && 0 <= v484 && v484 < 64);
    int v488;
    v488 = 0;
    while (while_method_1(v488)){
        bool v490;
        v490 = 0 <= v484;
        bool v491;
        v491 = v490 && v485;
        bool v492;
        v492 = v491 == false;
        if (v492){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v491);
        } else {
        }
        bool v494;
        v494 = 0 <= v488;
        bool v496;
        if (v494){
            bool v495;
            v495 = v488 < 4;
            v496 = v495;
        } else {
            v496 = false;
        }
        bool v497;
        v497 = v496 == false;
        if (v497){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v496);
        } else {
        }
        int v499;
        v499 = v488 * 64;
        int v500;
        v500 = v499 + v484;
        assert("Tensor range check" && 0 <= v488 && v488 < 4);
        int v501;
        v501 = 64 * v488;
        int v502;
        v502 = v501 + v484;
        float * v503;
        v503 = v475[v502];
        int v504;
        v504 = blockIdx.x;
        int v505;
        v505 = v504 * 256;
        int v506;
        v506 = v505 + v500;
        assert("Tensor range check" && 0 <= v483 && v483 < 4);
        int v507;
        v507 = 4 * v483;
        float v508[4];
        int v509[4];
        int v510;
        v510 = 0;
        while (while_method_3(v510)){
            assert("Tensor range check" && 0 <= v510 && v510 < 1);
            int v512;
            v512 = 4 * v510;
            assert("Tensor range check" && 0 <= v510 && v510 < 1);
            int v513;
            v513 = 16 * v510;
            int v514;
            v514 = v513 + v507;
            int4* v515;
            v515 = reinterpret_cast<int4*>(v503 + v514);
            int4* v516;
            v516 = reinterpret_cast<int4*>(v508 + v512);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v515) % 16 == 0 && reinterpret_cast<unsigned long long>(v516) % 16 == 0);
            *v516 = *v515;
            v510 += 1 ;
        }
        int v517;
        v517 = 0;
        while (while_method_3(v517)){
            int v519;
            v519 = 0;
            while (while_method_1(v519)){
                bool v521;
                v521 = 0 <= v519;
                bool v523;
                if (v521){
                    bool v522;
                    v522 = v519 < 4;
                    v523 = v522;
                } else {
                    v523 = false;
                }
                bool v524;
                v524 = v523 == false;
                if (v524){
                    assert("The indices should be inside the range of the dimension." && v523);
                } else {
                }
                bool v526;
                v526 = 0 <= v483;
                bool v528;
                if (v526){
                    bool v527;
                    v527 = v483 < 4;
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
                v531 = v483 * 4;
                int v532;
                v532 = v519 + v531;
                bool v533;
                v533 = 0 <= v517;
                bool v535;
                if (v533){
                    bool v534;
                    v534 = v517 < 1;
                    v535 = v534;
                } else {
                    v535 = false;
                }
                bool v536;
                v536 = v535 == false;
                if (v536){
                    assert("The indices should be inside the range of the dimension." && v535);
                } else {
                }
                int v538;
                v538 = v517 * 16;
                int v539;
                v539 = v532 + v538;
                assert("Tensor range check" && 0 <= v517 && v517 < 1);
                assert("Tensor range check" && 0 <= v519 && v519 < 4);
                int v540;
                v540 = 4 * v517;
                int v541;
                v541 = v540 + v519;
                v509[v541] = v539;
                v519 += 1 ;
            }
            v517 += 1 ;
        }
        bool v542[4];
        int v543;
        v543 = 0;
        while (while_method_3(v543)){
            int v545;
            v545 = 0;
            while (while_method_1(v545)){
                assert("Tensor range check" && 0 <= v543 && v543 < 1);
                assert("Tensor range check" && 0 <= v545 && v545 < 4);
                int v547;
                v547 = 4 * v543;
                int v548;
                v548 = v547 + v545;
                float v549;
                v549 = v508[v548];
                int v550;
                v550 = v509[v548];
                bool v551;
                v551 = v550 < 3;
                assert("Tensor range check" && 0 <= v543 && v543 < 1);
                assert("Tensor range check" && 0 <= v545 && v545 < 4);
                v542[v548] = v551;
                v545 += 1 ;
            }
            v543 += 1 ;
        }
        float v552[4];
        int v553;
        v553 = 0;
        while (while_method_3(v553)){
            int v555;
            v555 = 0;
            while (while_method_1(v555)){
                assert("Tensor range check" && 0 <= v553 && v553 < 1);
                assert("Tensor range check" && 0 <= v555 && v555 < 4);
                int v557;
                v557 = 4 * v553;
                int v558;
                v558 = v557 + v555;
                float v559;
                v559 = v508[v558];
                bool v560;
                v560 = v542[v558];
                float v561;
                if (v560){
                    v561 = v559;
                } else {
                    v561 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v553 && v553 < 1);
                assert("Tensor range check" && 0 <= v555 && v555 < 4);
                v552[v558] = v561;
                v555 += 1 ;
            }
            v553 += 1 ;
        }
        float v562;
        v562 = 0.0f;
        int v563;
        v563 = 0;
        while (while_method_3(v563)){
            int v565;
            v565 = 0;
            while (while_method_1(v565)){
                assert("Tensor range check" && 0 <= v563 && v563 < 1);
                assert("Tensor range check" && 0 <= v565 && v565 < 4);
                int v567;
                v567 = 4 * v563;
                int v568;
                v568 = v567 + v565;
                float v569;
                v569 = v552[v568];
                float v570;
                v570 = v562 + v569;
                v562 = v570;
                v565 += 1 ;
            }
            v563 += 1 ;
        }
        auto v571 = cooperative_groups::coalesced_threads();
        int v572;
        v572 = threadIdx.x;
        int v573;
        v573 = v572 / 4;
        auto v574 = cooperative_groups::labeled_partition(v571,v573);
        Closure0 v575{};
        float v576;
        v576 = cooperative_groups::reduce(v574, v562, v575);
        int v577[4];
        int v578;
        v578 = 0;
        while (while_method_3(v578)){
            int v580;
            v580 = 0;
            while (while_method_1(v580)){
                assert("Tensor range check" && 0 <= v578 && v578 < 1);
                assert("Tensor range check" && 0 <= v580 && v580 < 4);
                int v582;
                v582 = 4 * v578;
                int v583;
                v583 = v582 + v580;
                bool v584;
                v584 = v542[v583];
                int v585;
                if (v584){
                    v585 = 1;
                } else {
                    v585 = 0;
                }
                assert("Tensor range check" && 0 <= v578 && v578 < 1);
                assert("Tensor range check" && 0 <= v580 && v580 < 4);
                v577[v583] = v585;
                v580 += 1 ;
            }
            v578 += 1 ;
        }
        int v586;
        v586 = 0;
        int v587;
        v587 = 0;
        while (while_method_3(v587)){
            int v589;
            v589 = 0;
            while (while_method_1(v589)){
                assert("Tensor range check" && 0 <= v587 && v587 < 1);
                assert("Tensor range check" && 0 <= v589 && v589 < 4);
                int v591;
                v591 = 4 * v587;
                int v592;
                v592 = v591 + v589;
                int v593;
                v593 = v577[v592];
                int v594;
                v594 = v586 + v593;
                v586 = v594;
                v589 += 1 ;
            }
            v587 += 1 ;
        }
        auto v595 = cooperative_groups::coalesced_threads();
        int v596;
        v596 = threadIdx.x;
        int v597;
        v597 = v596 / 4;
        auto v598 = cooperative_groups::labeled_partition(v595,v597);
        Closure4 v599{};
        int v600;
        v600 = cooperative_groups::reduce(v598, v586, v599);
        float v601;
        v601 = (float)v600;
        float v602;
        v602 = v576 / v601;
        float v603[4];
        int v604;
        v604 = 0;
        while (while_method_3(v604)){
            int v606;
            v606 = 0;
            while (while_method_1(v606)){
                assert("Tensor range check" && 0 <= v604 && v604 < 1);
                assert("Tensor range check" && 0 <= v606 && v606 < 4);
                int v608;
                v608 = 4 * v604;
                int v609;
                v609 = v608 + v606;
                float v610;
                v610 = v508[v609];
                bool v611;
                v611 = v542[v609];
                float v612;
                if (v611){
                    v612 = v610;
                } else {
                    v612 = -1.0f / 0.0f;
                }
                float v613;
                v613 = v612 - v602;
                float v614;
                v614 = exp(v613);
                bool v615;
                v615 = v614 < 1.0f / 0.0f;
                bool v616;
                v616 = v615 == false;
                if (v616){
                    assert("The softmax values must not grow too large." && v615);
                } else {
                }
                bool v618;
                v618 = isnan(v614);
                bool v619;
                v619 = v618 == false;
                bool v620;
                v620 = v619 == false;
                if (v620){
                    assert("The softmax values must not be nans." && v619);
                } else {
                }
                assert("Tensor range check" && 0 <= v604 && v604 < 1);
                assert("Tensor range check" && 0 <= v606 && v606 < 4);
                v603[v609] = v614;
                v606 += 1 ;
            }
            v604 += 1 ;
        }
        float v622;
        v622 = 0.0f;
        int v623;
        v623 = 0;
        while (while_method_3(v623)){
            int v625;
            v625 = 0;
            while (while_method_1(v625)){
                assert("Tensor range check" && 0 <= v623 && v623 < 1);
                assert("Tensor range check" && 0 <= v625 && v625 < 4);
                int v627;
                v627 = 4 * v623;
                int v628;
                v628 = v627 + v625;
                float v629;
                v629 = v603[v628];
                float v630;
                v630 = v622 + v629;
                v622 = v630;
                v625 += 1 ;
            }
            v623 += 1 ;
        }
        auto v631 = cooperative_groups::coalesced_threads();
        int v632;
        v632 = threadIdx.x;
        int v633;
        v633 = v632 / 4;
        auto v634 = cooperative_groups::labeled_partition(v631,v633);
        float v635;
        v635 = cooperative_groups::reduce(v634, v622, v575);
        float v636[4];
        int v637;
        v637 = 0;
        while (while_method_3(v637)){
            int v639;
            v639 = 0;
            while (while_method_1(v639)){
                assert("Tensor range check" && 0 <= v637 && v637 < 1);
                assert("Tensor range check" && 0 <= v639 && v639 < 4);
                int v641;
                v641 = 4 * v637;
                int v642;
                v642 = v641 + v639;
                float v643;
                v643 = v603[v642];
                float v644;
                v644 = v643 / v635;
                assert("Tensor range check" && 0 <= v637 && v637 < 1);
                assert("Tensor range check" && 0 <= v639 && v639 < 4);
                v636[v642] = v644;
                v639 += 1 ;
            }
            v637 += 1 ;
        }
        float v645[4];
        float v646;
        v646 = 0.0f;
        int v647;
        v647 = 0;
        while (while_method_3(v647)){
            assert("Tensor range check" && 0 <= v647 && v647 < 1);
            int v649;
            v649 = 4 * v647;
            assert("Tensor range check" && 0 <= v647 && v647 < 1);
            float v650;
            v650 = 0.0f;
            int v651;
            v651 = 0;
            while (while_method_1(v651)){
                assert("Tensor range check" && 0 <= v651 && v651 < 4);
                int v653;
                v653 = v651 + v649;
                float v654;
                v654 = v636[v653];
                float v655;
                v655 = v650 + v654;
                v650 = v655;
                v651 += 1 ;
            }
            auto v656 = cooperative_groups::coalesced_threads();
            int v657;
            v657 = threadIdx.x;
            int v658;
            v658 = v657 / 4;
            auto v659 = cooperative_groups::labeled_partition(v656,v658);
            Closure2 v660{};
            float v661;
            v661 = cooperative_groups::inclusive_scan(v659, v650, v660);
            float v662;
            v662 = v659.shfl_up(v661,1);
            bool v663;
            v663 = v659.thread_rank() == 0;
            float v664;
            if (v663){
                v664 = 0.0f;
            } else {
                v664 = v662;
            }
            float v665;
            v665 = v659.shfl(v661,v659.num_threads()-1);
            float v666;
            v666 = v646 + v664;
            float v667;
            v667 = v666;
            int v668;
            v668 = 0;
            while (while_method_1(v668)){
                assert("Tensor range check" && 0 <= v668 && v668 < 4);
                int v670;
                v670 = v668 + v649;
                float v671;
                v671 = v636[v670];
                float v672;
                v672 = v667 + v671;
                assert("Tensor range check" && 0 <= v668 && v668 < 4);
                v645[v670] = v672;
                v667 = v672;
                v668 += 1 ;
            }
            float v673;
            v673 = v646 + v665;
            v646 = v673;
            v647 += 1 ;
        }
        float v674[4];
        bool v675[4];
        int v676;
        v676 = 0;
        while (while_method_3(v676)){
            int v678;
            v678 = 0;
            while (while_method_1(v678)){
                assert("Tensor range check" && 0 <= v676 && v676 < 1);
                assert("Tensor range check" && 0 <= v678 && v678 < 4);
                int v680;
                v680 = 4 * v676;
                int v681;
                v681 = v680 + v678;
                float v682;
                v682 = v645[v681];
                float v683;
                v683 = v636[v681];
                bool v684;
                v684 = v683 > 0.0f;
                assert("Tensor range check" && 0 <= v676 && v676 < 1);
                assert("Tensor range check" && 0 <= v678 && v678 < 4);
                v674[v681] = v682;
                v675[v681] = v684;
                v678 += 1 ;
            }
            v676 += 1 ;
        }
        float v685; bool v686;
        Tuple2 tmp30 = Tuple2{-1.0f / 0.0f, false};
        v685 = tmp30.v0; v686 = tmp30.v1;
        int v687;
        v687 = 0;
        while (while_method_3(v687)){
            int v689;
            v689 = 0;
            while (while_method_1(v689)){
                assert("Tensor range check" && 0 <= v687 && v687 < 1);
                assert("Tensor range check" && 0 <= v689 && v689 < 4);
                int v691;
                v691 = 4 * v687;
                int v692;
                v692 = v691 + v689;
                float v693;
                v693 = v674[v692];
                bool v694;
                v694 = v675[v692];
                float v701; bool v702;
                if (v686){
                    if (v694){
                        bool v695;
                        v695 = v685 >= v693;
                        float v696;
                        if (v695){
                            v696 = v685;
                        } else {
                            v696 = v693;
                        }
                        v701 = v696; v702 = true;
                    } else {
                        v701 = v685; v702 = v686;
                    }
                } else {
                    if (v694){
                        v701 = v693; v702 = v694;
                    } else {
                        v701 = v685; v702 = v686;
                    }
                }
                v685 = v701;
                v686 = v702;
                v689 += 1 ;
            }
            v687 += 1 ;
        }
        auto v703 = cooperative_groups::coalesced_threads();
        int v704;
        v704 = threadIdx.x;
        int v705;
        v705 = v704 / 4;
        auto v706 = cooperative_groups::labeled_partition(v703,v705);
        Closure5 v707{};
        float v708; bool v709;
        Tuple2 tmp31 = cooperative_groups::reduce(v706, Tuple2{v685, v686}, v707);
        v708 = tmp31.v0; v709 = tmp31.v1;
        bool v710;
        v710 = v709 == false;
        if (v710){
            assert("The local reduce must be true." && v709);
        } else {
        }
        float v712[4];
        int v713[4];
        int v714;
        v714 = 0;
        while (while_method_3(v714)){
            int v716;
            v716 = 0;
            while (while_method_1(v716)){
                assert("Tensor range check" && 0 <= v714 && v714 < 1);
                assert("Tensor range check" && 0 <= v716 && v716 < 4);
                int v718;
                v718 = 4 * v714;
                int v719;
                v719 = v718 + v716;
                int v720;
                v720 = v509[v719];
                float v721;
                v721 = curand_uniform(&v469);
                assert("Tensor range check" && 0 <= v714 && v714 < 1);
                assert("Tensor range check" && 0 <= v716 && v716 < 4);
                v712[v719] = v721;
                v713[v719] = v720;
                v716 += 1 ;
            }
            v714 += 1 ;
        }
        float v722; int v723;
        Tuple1 tmp32 = Tuple1{0.0f, 2147483647};
        v722 = tmp32.v0; v723 = tmp32.v1;
        int v724;
        v724 = 0;
        while (while_method_3(v724)){
            int v726;
            v726 = 0;
            while (while_method_1(v726)){
                assert("Tensor range check" && 0 <= v724 && v724 < 1);
                assert("Tensor range check" && 0 <= v726 && v726 < 4);
                int v728;
                v728 = 4 * v724;
                int v729;
                v729 = v728 + v726;
                float v730;
                v730 = v712[v729];
                int v731;
                v731 = v713[v729];
                bool v732;
                v732 = v723 < v731;
                float v733; int v734;
                if (v732){
                    v733 = v722; v734 = v723;
                } else {
                    v733 = v730; v734 = v731;
                }
                v722 = v733;
                v723 = v734;
                v726 += 1 ;
            }
            v724 += 1 ;
        }
        auto v735 = cooperative_groups::coalesced_threads();
        int v736;
        v736 = threadIdx.x;
        int v737;
        v737 = v736 / 4;
        auto v738 = cooperative_groups::labeled_partition(v735,v737);
        Closure6 v739{};
        float v740; int v741;
        Tuple1 tmp33 = cooperative_groups::reduce(v738, Tuple1{v722, v723}, v739);
        v740 = tmp33.v0; v741 = tmp33.v1;
        float v742;
        v742 = v708 * v740;
        int v743[4];
        bool v744[4];
        int v745;
        v745 = 0;
        while (while_method_3(v745)){
            int v747;
            v747 = 0;
            while (while_method_1(v747)){
                assert("Tensor range check" && 0 <= v745 && v745 < 1);
                assert("Tensor range check" && 0 <= v747 && v747 < 4);
                int v749;
                v749 = 4 * v745;
                int v750;
                v750 = v749 + v747;
                float v751;
                v751 = v674[v750];
                bool v752;
                v752 = v675[v750];
                int v753;
                v753 = v509[v750];
                int v756; bool v757;
                if (v752){
                    float v754;
                    v754 = v751 - v742;
                    bool v755;
                    v755 = v754 >= 0.0f;
                    v756 = v753; v757 = v755;
                } else {
                    v756 = 2147483647; v757 = false;
                }
                assert("Tensor range check" && 0 <= v745 && v745 < 1);
                assert("Tensor range check" && 0 <= v747 && v747 < 4);
                v743[v750] = v756;
                v744[v750] = v757;
                v747 += 1 ;
            }
            v745 += 1 ;
        }
        int v758; bool v759;
        Tuple3 tmp34 = Tuple3{2147483647, false};
        v758 = tmp34.v0; v759 = tmp34.v1;
        int v760;
        v760 = 0;
        while (while_method_3(v760)){
            int v762;
            v762 = 0;
            while (while_method_1(v762)){
                assert("Tensor range check" && 0 <= v760 && v760 < 1);
                assert("Tensor range check" && 0 <= v762 && v762 < 4);
                int v764;
                v764 = 4 * v760;
                int v765;
                v765 = v764 + v762;
                int v766;
                v766 = v743[v765];
                bool v767;
                v767 = v744[v765];
                int v774; bool v775;
                if (v759){
                    if (v767){
                        bool v768;
                        v768 = v758 < v766;
                        int v769;
                        if (v768){
                            v769 = v758;
                        } else {
                            v769 = v766;
                        }
                        v774 = v769; v775 = true;
                    } else {
                        v774 = v758; v775 = v759;
                    }
                } else {
                    if (v767){
                        v774 = v766; v775 = v767;
                    } else {
                        v774 = v758; v775 = v759;
                    }
                }
                v758 = v774;
                v759 = v775;
                v762 += 1 ;
            }
            v760 += 1 ;
        }
        auto v776 = cooperative_groups::coalesced_threads();
        int v777;
        v777 = threadIdx.x;
        int v778;
        v778 = v777 / 4;
        auto v779 = cooperative_groups::labeled_partition(v776,v778);
        Closure7 v780{};
        int v781; bool v782;
        Tuple3 tmp35 = cooperative_groups::reduce(v779, Tuple3{v758, v759}, v780);
        v781 = tmp35.v0; v782 = tmp35.v1;
        bool v783;
        v783 = v782 == false;
        if (v783){
            assert("The local reduce must be true." && v782);
        } else {
        }
        int v785;
        v785 = 0;
        while (while_method_3(v785)){
            assert("Tensor range check" && 0 <= v785 && v785 < 1);
            assert("Tensor range check" && 0 <= v785 && v785 < 1);
            v785 += 1 ;
        }
        assert("Tensor range check" && 0 <= v500 && v500 < 256);
        v477[v500] = v781;
        v488 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v479 && v479 < 256);
    int v787;
    v787 = v477[v479];
    __syncthreads();
    int v788;
    v788 = threadIdx.x;
    assert("Tensor range check" && 0 <= v788 && v788 < 256);
    v5[v788] = v787;
    return ;
}
extern "C" __global__ void entry3(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 256);
    int v9;
    v9 = 256 * v8;
    int v10;
    v10 = threadIdx.x;
    assert("Tensor range check" && 0 <= v10 && v10 < 256);
    int v11;
    v11 = 256 * v10;
    int v12;
    v12 = threadIdx.x;
    assert("Tensor range check" && 0 <= v12 && v12 < 256);
    int v13;
    v13 = 256 * v12;
    int v14;
    v14 = threadIdx.x;
    assert("Tensor range check" && 0 <= v14 && v14 < 256);
    int v15;
    v15 = 256 * v14;
    int v16;
    v16 = threadIdx.x;
    assert("Tensor range check" && 0 <= v16 && v16 < 256);
    int v17;
    v17 = 256 * v16;
    float * v18;
    v18 = v1+v9;
    int * v20;
    v20 = v2+v15;
    int * v22;
    v22 = v3+v15;
    int v24;
    v24 = sizeof(float *);
    unsigned long long v25;
    v25 = (unsigned long long)v24;
    unsigned long long v26;
    v26 = 256ull * v25;
    unsigned long long v27;
    v27 = v26 + 16ull;
    unsigned long long v28;
    v28 = v27 - 1ull;
    unsigned long long v29;
    v29 = v28 % 16ull;
    unsigned long long v30;
    v30 = v28 - v29;
    int v31;
    v31 = sizeof(int *);
    unsigned long long v32;
    v32 = (unsigned long long)v31;
    unsigned long long v33;
    v33 = 256ull * v32;
    unsigned long long v34;
    v34 = v30 + v33;
    unsigned long long v35;
    v35 = v34 + 16ull;
    unsigned long long v36;
    v36 = v35 - 1ull;
    unsigned long long v37;
    v37 = v36 % 16ull;
    unsigned long long v38;
    v38 = v36 - v37;
    unsigned long long v39;
    v39 = v38 + v33;
    bool v40;
    v40 = v39 <= 98304ull;
    bool v41;
    v41 = v40 == false;
    if (v41){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v40);
    } else {
    }
    extern __shared__ unsigned char v43[];
    bool v44;
    v44 = v39 <= v39;
    bool v45;
    v45 = v44 == false;
    if (v45){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v44);
    } else {
    }
    float * * v47;
    v47 = reinterpret_cast<float * *>(&v43[0ull]);
    int * * v49;
    v49 = reinterpret_cast<int * *>(&v43[v30]);
    int * * v51;
    v51 = reinterpret_cast<int * *>(&v43[v38]);
    int v53;
    v53 = threadIdx.x;
    assert("Tensor range check" && 0 <= v53 && v53 < 256);
    v47[v53] = v18;
    v49[v53] = v20;
    v51[v53] = v22;
    __syncthreads();
    bool v54;
    v54 = 0 <= v53;
    bool v55;
    v55 = v54 == false;
    if (v55){
        assert("The index needs to be zero or positive." && v54);
    } else {
    }
    int v57;
    v57 = v53 % 64;
    int v58;
    v58 = v53 / 64;
    bool v59;
    v59 = v58 < 4;
    bool v60;
    v60 = v59 == false;
    if (v60){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v59);
    } else {
    }
    assert("Tensor range check" && 0 <= v58 && v58 < 4);
    int v62;
    v62 = 0;
    while (while_method_4(v62)){
        bool v64;
        v64 = 0 <= v58;
        bool v65;
        v65 = v64 && v59;
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v65);
        } else {
        }
        bool v68;
        v68 = 0 <= v62;
        bool v70;
        if (v68){
            bool v69;
            v69 = v62 < 64;
            v70 = v69;
        } else {
            v70 = false;
        }
        bool v71;
        v71 = v70 == false;
        if (v71){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v70);
        } else {
        }
        int v73;
        v73 = v62 * 4;
        int v74;
        v74 = v73 + v58;
        assert("Tensor range check" && 0 <= v62 && v62 < 64);
        int v75;
        v75 = 4 * v62;
        int v76;
        v76 = v75 + v58;
        float * v77;
        v77 = v47[v76];
        int * v78;
        v78 = v49[v76];
        int * v79;
        v79 = v51[v76];
        int v80;
        v80 = blockIdx.x;
        int v81;
        v81 = v80 * 256;
        int v82;
        v82 = v81 + v74;
        assert("Tensor range check" && 0 <= v57 && v57 < 64);
        int v83;
        v83 = 4 * v57;
        float v84[4];
        int v85[4];
        int v86;
        v86 = 0;
        while (while_method_3(v86)){
            assert("Tensor range check" && 0 <= v86 && v86 < 1);
            int v88;
            v88 = 4 * v86;
            assert("Tensor range check" && 0 <= v86 && v86 < 1);
            int v89;
            v89 = 256 * v86;
            int v90;
            v90 = v89 + v83;
            int4* v91;
            v91 = reinterpret_cast<int4*>(v77 + v90);
            int4* v92;
            v92 = reinterpret_cast<int4*>(v84 + v88);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v91) % 16 == 0 && reinterpret_cast<unsigned long long>(v92) % 16 == 0);
            *v92 = *v91;
            v86 += 1 ;
        }
        int v93;
        v93 = 0;
        while (while_method_3(v93)){
            int v95;
            v95 = 0;
            while (while_method_1(v95)){
                bool v97;
                v97 = 0 <= v95;
                bool v99;
                if (v97){
                    bool v98;
                    v98 = v95 < 4;
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
                bool v102;
                v102 = 0 <= v57;
                bool v104;
                if (v102){
                    bool v103;
                    v103 = v57 < 64;
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
                int v107;
                v107 = v57 * 4;
                int v108;
                v108 = v95 + v107;
                bool v109;
                v109 = 0 <= v93;
                bool v111;
                if (v109){
                    bool v110;
                    v110 = v93 < 1;
                    v111 = v110;
                } else {
                    v111 = false;
                }
                bool v112;
                v112 = v111 == false;
                if (v112){
                    assert("The indices should be inside the range of the dimension." && v111);
                } else {
                }
                int v114;
                v114 = v93 * 256;
                int v115;
                v115 = v108 + v114;
                assert("Tensor range check" && 0 <= v93 && v93 < 1);
                assert("Tensor range check" && 0 <= v95 && v95 < 4);
                int v116;
                v116 = 4 * v93;
                int v117;
                v117 = v116 + v95;
                v85[v117] = v115;
                v95 += 1 ;
            }
            v93 += 1 ;
        }
        int v118[4];
        int v119[4];
        int v120;
        v120 = 0;
        while (while_method_3(v120)){
            int v122;
            v122 = 0;
            while (while_method_1(v122)){
                assert("Tensor range check" && 0 <= v120 && v120 < 1);
                assert("Tensor range check" && 0 <= v122 && v122 < 4);
                int v124;
                v124 = 4 * v120;
                int v125;
                v125 = v124 + v122;
                int v126;
                v126 = v85[v125];
                assert("Tensor range check" && 0 <= v120 && v120 < 1);
                assert("Tensor range check" && 0 <= v122 && v122 < 4);
                v118[v125] = v82;
                v119[v125] = v126;
                v122 += 1 ;
            }
            v120 += 1 ;
        }
        int v127;
        v127 = 0;
        while (while_method_3(v127)){
            assert("Tensor range check" && 0 <= v127 && v127 < 1);
            int v129;
            v129 = 256 * v127;
            int v130;
            v130 = v129 + v83;
            assert("Tensor range check" && 0 <= v127 && v127 < 1);
            int v131;
            v131 = 4 * v127;
            int4* v132;
            v132 = reinterpret_cast<int4*>(v118 + v131);
            int4* v133;
            v133 = reinterpret_cast<int4*>(v78 + v130);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v132) % 16 == 0 && reinterpret_cast<unsigned long long>(v133) % 16 == 0);
            *v133 = *v132;
            int4* v134;
            v134 = reinterpret_cast<int4*>(v119 + v131);
            int4* v135;
            v135 = reinterpret_cast<int4*>(v79 + v130);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v134) % 16 == 0 && reinterpret_cast<unsigned long long>(v135) % 16 == 0);
            *v135 = *v134;
            v127 += 1 ;
        }
        assert("Tensor range check" && 0 <= v74 && v74 < 256);
        v62 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v53 && v53 < 256);
    __syncthreads();
    float * v136;
    v136 = v1+v9;
    unsigned long long v138;
    v138 = v30 + 1024ull;
    bool v139;
    v139 = v138 <= 98304ull;
    bool v140;
    v140 = v139 == false;
    if (v140){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v139);
    } else {
    }
    extern __shared__ unsigned char v142[];
    bool v143;
    v143 = v138 <= v138;
    bool v144;
    v144 = v143 == false;
    if (v144){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v143);
    } else {
    }
    float * * v146;
    v146 = reinterpret_cast<float * *>(&v142[0ull]);
    int * v148;
    v148 = reinterpret_cast<int *>(&v142[v30]);
    int v150;
    v150 = threadIdx.x;
    assert("Tensor range check" && 0 <= v150 && v150 < 256);
    v146[v150] = v136;
    __syncthreads();
    bool v151;
    v151 = 0 <= v150;
    bool v152;
    v152 = v151 == false;
    if (v152){
        assert("The index needs to be zero or positive." && v151);
    } else {
    }
    int v154;
    v154 = v150 % 64;
    int v155;
    v155 = v150 / 64;
    bool v156;
    v156 = v155 < 4;
    bool v157;
    v157 = v156 == false;
    if (v157){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v156);
    } else {
    }
    assert("Tensor range check" && 0 <= v155 && v155 < 4);
    int v159;
    v159 = 0;
    while (while_method_4(v159)){
        bool v161;
        v161 = 0 <= v155;
        bool v162;
        v162 = v161 && v156;
        bool v163;
        v163 = v162 == false;
        if (v163){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v162);
        } else {
        }
        bool v165;
        v165 = 0 <= v159;
        bool v167;
        if (v165){
            bool v166;
            v166 = v159 < 64;
            v167 = v166;
        } else {
            v167 = false;
        }
        bool v168;
        v168 = v167 == false;
        if (v168){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v167);
        } else {
        }
        int v170;
        v170 = v159 * 4;
        int v171;
        v171 = v170 + v155;
        assert("Tensor range check" && 0 <= v159 && v159 < 64);
        int v172;
        v172 = 4 * v159;
        int v173;
        v173 = v172 + v155;
        float * v174;
        v174 = v146[v173];
        int v175;
        v175 = blockIdx.x;
        int v176;
        v176 = v175 * 256;
        int v177;
        v177 = v176 + v171;
        assert("Tensor range check" && 0 <= v154 && v154 < 64);
        int v178;
        v178 = 4 * v154;
        float v179[4];
        int v180[4];
        int v181;
        v181 = 0;
        while (while_method_3(v181)){
            assert("Tensor range check" && 0 <= v181 && v181 < 1);
            int v183;
            v183 = 4 * v181;
            assert("Tensor range check" && 0 <= v181 && v181 < 1);
            int v184;
            v184 = 256 * v181;
            int v185;
            v185 = v184 + v178;
            int4* v186;
            v186 = reinterpret_cast<int4*>(v174 + v185);
            int4* v187;
            v187 = reinterpret_cast<int4*>(v179 + v183);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v186) % 16 == 0 && reinterpret_cast<unsigned long long>(v187) % 16 == 0);
            *v187 = *v186;
            v181 += 1 ;
        }
        int v188;
        v188 = 0;
        while (while_method_3(v188)){
            int v190;
            v190 = 0;
            while (while_method_1(v190)){
                bool v192;
                v192 = 0 <= v190;
                bool v194;
                if (v192){
                    bool v193;
                    v193 = v190 < 4;
                    v194 = v193;
                } else {
                    v194 = false;
                }
                bool v195;
                v195 = v194 == false;
                if (v195){
                    assert("The indices should be inside the range of the dimension." && v194);
                } else {
                }
                bool v197;
                v197 = 0 <= v154;
                bool v199;
                if (v197){
                    bool v198;
                    v198 = v154 < 64;
                    v199 = v198;
                } else {
                    v199 = false;
                }
                bool v200;
                v200 = v199 == false;
                if (v200){
                    assert("The indices should be inside the range of the dimension." && v199);
                } else {
                }
                int v202;
                v202 = v154 * 4;
                int v203;
                v203 = v190 + v202;
                bool v204;
                v204 = 0 <= v188;
                bool v206;
                if (v204){
                    bool v205;
                    v205 = v188 < 1;
                    v206 = v205;
                } else {
                    v206 = false;
                }
                bool v207;
                v207 = v206 == false;
                if (v207){
                    assert("The indices should be inside the range of the dimension." && v206);
                } else {
                }
                int v209;
                v209 = v188 * 256;
                int v210;
                v210 = v203 + v209;
                assert("Tensor range check" && 0 <= v188 && v188 < 1);
                assert("Tensor range check" && 0 <= v190 && v190 < 4);
                int v211;
                v211 = 4 * v188;
                int v212;
                v212 = v211 + v190;
                v180[v212] = v210;
                v190 += 1 ;
            }
            v188 += 1 ;
        }
        int v213;
        v213 = 0;
        while (while_method_3(v213)){
            assert("Tensor range check" && 0 <= v213 && v213 < 1);
            assert("Tensor range check" && 0 <= v213 && v213 < 1);
            v213 += 1 ;
        }
        assert("Tensor range check" && 0 <= v171 && v171 < 256);
        v148[v171] = v177;
        v159 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v150 && v150 < 256);
    int v215;
    v215 = v148[v150];
    __syncthreads();
    int v216;
    v216 = threadIdx.x;
    assert("Tensor range check" && 0 <= v216 && v216 < 256);
    v4[v216] = v215;
    float * v217;
    v217 = v1+v9;
    float * v219;
    v219 = v6+v17;
    unsigned long long v221;
    v221 = v30 + v26;
    bool v222;
    v222 = v221 <= 98304ull;
    bool v223;
    v223 = v222 == false;
    if (v223){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v222);
    } else {
    }
    extern __shared__ unsigned char v225[];
    bool v226;
    v226 = v221 <= v221;
    bool v227;
    v227 = v226 == false;
    if (v227){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v226);
    } else {
    }
    float * * v229;
    v229 = reinterpret_cast<float * *>(&v225[0ull]);
    float * * v231;
    v231 = reinterpret_cast<float * *>(&v225[v30]);
    int v233;
    v233 = threadIdx.x;
    assert("Tensor range check" && 0 <= v233 && v233 < 256);
    v229[v233] = v217;
    v231[v233] = v219;
    __syncthreads();
    bool v234;
    v234 = 0 <= v233;
    bool v235;
    v235 = v234 == false;
    if (v235){
        assert("The index needs to be zero or positive." && v234);
    } else {
    }
    int v237;
    v237 = v233 % 64;
    int v238;
    v238 = v233 / 64;
    bool v239;
    v239 = v238 < 4;
    bool v240;
    v240 = v239 == false;
    if (v240){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v239);
    } else {
    }
    assert("Tensor range check" && 0 <= v238 && v238 < 4);
    int v242;
    v242 = 0;
    while (while_method_4(v242)){
        bool v244;
        v244 = 0 <= v238;
        bool v245;
        v245 = v244 && v239;
        bool v246;
        v246 = v245 == false;
        if (v246){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v245);
        } else {
        }
        bool v248;
        v248 = 0 <= v242;
        bool v250;
        if (v248){
            bool v249;
            v249 = v242 < 64;
            v250 = v249;
        } else {
            v250 = false;
        }
        bool v251;
        v251 = v250 == false;
        if (v251){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v250);
        } else {
        }
        int v253;
        v253 = v242 * 4;
        int v254;
        v254 = v253 + v238;
        assert("Tensor range check" && 0 <= v242 && v242 < 64);
        int v255;
        v255 = 4 * v242;
        int v256;
        v256 = v255 + v238;
        float * v257;
        v257 = v229[v256];
        float * v258;
        v258 = v231[v256];
        int v259;
        v259 = blockIdx.x;
        int v260;
        v260 = v259 * 256;
        int v261;
        v261 = v260 + v254;
        assert("Tensor range check" && 0 <= v237 && v237 < 64);
        int v262;
        v262 = 4 * v237;
        float v263[4];
        int v264[4];
        int v265;
        v265 = 0;
        while (while_method_3(v265)){
            assert("Tensor range check" && 0 <= v265 && v265 < 1);
            int v267;
            v267 = 4 * v265;
            assert("Tensor range check" && 0 <= v265 && v265 < 1);
            int v268;
            v268 = 256 * v265;
            int v269;
            v269 = v268 + v262;
            int4* v270;
            v270 = reinterpret_cast<int4*>(v257 + v269);
            int4* v271;
            v271 = reinterpret_cast<int4*>(v263 + v267);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v270) % 16 == 0 && reinterpret_cast<unsigned long long>(v271) % 16 == 0);
            *v271 = *v270;
            v265 += 1 ;
        }
        int v272;
        v272 = 0;
        while (while_method_3(v272)){
            int v274;
            v274 = 0;
            while (while_method_1(v274)){
                bool v276;
                v276 = 0 <= v274;
                bool v278;
                if (v276){
                    bool v277;
                    v277 = v274 < 4;
                    v278 = v277;
                } else {
                    v278 = false;
                }
                bool v279;
                v279 = v278 == false;
                if (v279){
                    assert("The indices should be inside the range of the dimension." && v278);
                } else {
                }
                bool v281;
                v281 = 0 <= v237;
                bool v283;
                if (v281){
                    bool v282;
                    v282 = v237 < 64;
                    v283 = v282;
                } else {
                    v283 = false;
                }
                bool v284;
                v284 = v283 == false;
                if (v284){
                    assert("The indices should be inside the range of the dimension." && v283);
                } else {
                }
                int v286;
                v286 = v237 * 4;
                int v287;
                v287 = v274 + v286;
                bool v288;
                v288 = 0 <= v272;
                bool v290;
                if (v288){
                    bool v289;
                    v289 = v272 < 1;
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
                int v293;
                v293 = v272 * 256;
                int v294;
                v294 = v287 + v293;
                assert("Tensor range check" && 0 <= v272 && v272 < 1);
                assert("Tensor range check" && 0 <= v274 && v274 < 4);
                int v295;
                v295 = 4 * v272;
                int v296;
                v296 = v295 + v274;
                v264[v296] = v294;
                v274 += 1 ;
            }
            v272 += 1 ;
        }
        int v297;
        v297 = 0;
        while (while_method_3(v297)){
            assert("Tensor range check" && 0 <= v297 && v297 < 1);
            int v299;
            v299 = 256 * v297;
            int v300;
            v300 = v299 + v262;
            assert("Tensor range check" && 0 <= v297 && v297 < 1);
            int v301;
            v301 = 4 * v297;
            int4* v302;
            v302 = reinterpret_cast<int4*>(v263 + v301);
            int4* v303;
            v303 = reinterpret_cast<int4*>(v258 + v300);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v302) % 16 == 0 && reinterpret_cast<unsigned long long>(v303) % 16 == 0);
            *v303 = *v302;
            v297 += 1 ;
        }
        assert("Tensor range check" && 0 <= v254 && v254 < 256);
        v242 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v233 && v233 < 256);
    __syncthreads();
    float * v304;
    v304 = v1+v9;
    float * v306;
    v306 = v7+v13;
    if (v223){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v222);
    } else {
    }
    extern __shared__ unsigned char v309[];
    if (v227){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v226);
    } else {
    }
    float * * v311;
    v311 = reinterpret_cast<float * *>(&v309[0ull]);
    float * * v313;
    v313 = reinterpret_cast<float * *>(&v309[v30]);
    int v315;
    v315 = threadIdx.x;
    assert("Tensor range check" && 0 <= v315 && v315 < 256);
    v311[v315] = v304;
    v313[v315] = v306;
    __syncthreads();
    bool v316;
    v316 = 0 <= v315;
    bool v317;
    v317 = v316 == false;
    if (v317){
        assert("The index needs to be zero or positive." && v316);
    } else {
    }
    int v319;
    v319 = v315 % 64;
    int v320;
    v320 = v315 / 64;
    bool v321;
    v321 = v320 < 4;
    bool v322;
    v322 = v321 == false;
    if (v322){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v321);
    } else {
    }
    assert("Tensor range check" && 0 <= v320 && v320 < 4);
    int v324;
    v324 = 0;
    while (while_method_4(v324)){
        bool v326;
        v326 = 0 <= v320;
        bool v327;
        v327 = v326 && v321;
        bool v328;
        v328 = v327 == false;
        if (v328){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v327);
        } else {
        }
        bool v330;
        v330 = 0 <= v324;
        bool v332;
        if (v330){
            bool v331;
            v331 = v324 < 64;
            v332 = v331;
        } else {
            v332 = false;
        }
        bool v333;
        v333 = v332 == false;
        if (v333){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v332);
        } else {
        }
        int v335;
        v335 = v324 * 4;
        int v336;
        v336 = v335 + v320;
        assert("Tensor range check" && 0 <= v324 && v324 < 64);
        int v337;
        v337 = 4 * v324;
        int v338;
        v338 = v337 + v320;
        float * v339;
        v339 = v311[v338];
        float * v340;
        v340 = v313[v338];
        int v341;
        v341 = blockIdx.x;
        int v342;
        v342 = v341 * 256;
        int v343;
        v343 = v342 + v336;
        assert("Tensor range check" && 0 <= v319 && v319 < 64);
        int v344;
        v344 = 4 * v319;
        float v345[4];
        int v346[4];
        int v347;
        v347 = 0;
        while (while_method_3(v347)){
            assert("Tensor range check" && 0 <= v347 && v347 < 1);
            int v349;
            v349 = 4 * v347;
            assert("Tensor range check" && 0 <= v347 && v347 < 1);
            int v350;
            v350 = 256 * v347;
            int v351;
            v351 = v350 + v344;
            int4* v352;
            v352 = reinterpret_cast<int4*>(v339 + v351);
            int4* v353;
            v353 = reinterpret_cast<int4*>(v345 + v349);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v352) % 16 == 0 && reinterpret_cast<unsigned long long>(v353) % 16 == 0);
            *v353 = *v352;
            v347 += 1 ;
        }
        int v354;
        v354 = 0;
        while (while_method_3(v354)){
            int v356;
            v356 = 0;
            while (while_method_1(v356)){
                bool v358;
                v358 = 0 <= v356;
                bool v360;
                if (v358){
                    bool v359;
                    v359 = v356 < 4;
                    v360 = v359;
                } else {
                    v360 = false;
                }
                bool v361;
                v361 = v360 == false;
                if (v361){
                    assert("The indices should be inside the range of the dimension." && v360);
                } else {
                }
                bool v363;
                v363 = 0 <= v319;
                bool v365;
                if (v363){
                    bool v364;
                    v364 = v319 < 64;
                    v365 = v364;
                } else {
                    v365 = false;
                }
                bool v366;
                v366 = v365 == false;
                if (v366){
                    assert("The indices should be inside the range of the dimension." && v365);
                } else {
                }
                int v368;
                v368 = v319 * 4;
                int v369;
                v369 = v356 + v368;
                bool v370;
                v370 = 0 <= v354;
                bool v372;
                if (v370){
                    bool v371;
                    v371 = v354 < 1;
                    v372 = v371;
                } else {
                    v372 = false;
                }
                bool v373;
                v373 = v372 == false;
                if (v373){
                    assert("The indices should be inside the range of the dimension." && v372);
                } else {
                }
                int v375;
                v375 = v354 * 256;
                int v376;
                v376 = v369 + v375;
                assert("Tensor range check" && 0 <= v354 && v354 < 1);
                assert("Tensor range check" && 0 <= v356 && v356 < 4);
                int v377;
                v377 = 4 * v354;
                int v378;
                v378 = v377 + v356;
                v346[v378] = v376;
                v356 += 1 ;
            }
            v354 += 1 ;
        }
        bool v379[4];
        int v380;
        v380 = 0;
        while (while_method_3(v380)){
            int v382;
            v382 = 0;
            while (while_method_1(v382)){
                assert("Tensor range check" && 0 <= v380 && v380 < 1);
                assert("Tensor range check" && 0 <= v382 && v382 < 4);
                int v384;
                v384 = 4 * v380;
                int v385;
                v385 = v384 + v382;
                float v386;
                v386 = v345[v385];
                int v387;
                v387 = v346[v385];
                bool v388;
                v388 = v387 < 3;
                assert("Tensor range check" && 0 <= v380 && v380 < 1);
                assert("Tensor range check" && 0 <= v382 && v382 < 4);
                v379[v385] = v388;
                v382 += 1 ;
            }
            v380 += 1 ;
        }
        float v389[4];
        int v390;
        v390 = 0;
        while (while_method_3(v390)){
            int v392;
            v392 = 0;
            while (while_method_1(v392)){
                assert("Tensor range check" && 0 <= v390 && v390 < 1);
                assert("Tensor range check" && 0 <= v392 && v392 < 4);
                int v394;
                v394 = 4 * v390;
                int v395;
                v395 = v394 + v392;
                float v396;
                v396 = v345[v395];
                bool v397;
                v397 = v379[v395];
                float v400;
                if (v397){
                    bool v398;
                    v398 = 0.0f >= v396;
                    if (v398){
                        v400 = 0.0f;
                    } else {
                        v400 = v396;
                    }
                } else {
                    v400 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v390 && v390 < 1);
                assert("Tensor range check" && 0 <= v392 && v392 < 4);
                v389[v395] = v400;
                v392 += 1 ;
            }
            v390 += 1 ;
        }
        float v401;
        v401 = 0.0f;
        int v402;
        v402 = 0;
        while (while_method_3(v402)){
            int v404;
            v404 = 0;
            while (while_method_1(v404)){
                assert("Tensor range check" && 0 <= v402 && v402 < 1);
                assert("Tensor range check" && 0 <= v404 && v404 < 4);
                int v406;
                v406 = 4 * v402;
                int v407;
                v407 = v406 + v404;
                float v408;
                v408 = v389[v407];
                float v409;
                v409 = v401 + v408;
                v401 = v409;
                v404 += 1 ;
            }
            v402 += 1 ;
        }
        auto v410 = cooperative_groups::coalesced_threads();
        Closure0 v411{};
        float v412;
        v412 = cooperative_groups::reduce(v410, v401, v411);
        int v413;
        v413 = threadIdx.x;
        int v414;
        v414 = v413 / 32;
        unsigned long long v415;
        v415 = v221 + 16ull;
        unsigned long long v416;
        v416 = v415 - 1ull;
        unsigned long long v417;
        v417 = v416 % 16ull;
        unsigned long long v418;
        v418 = v416 - v417;
        unsigned long long v419;
        v419 = v418 + 32ull;
        bool v420;
        v420 = v419 <= 98304ull;
        bool v421;
        v421 = v420 == false;
        if (v421){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v420);
        } else {
        }
        extern __shared__ unsigned char v423[];
        bool v424;
        v424 = v419 <= v419;
        bool v425;
        v425 = v424 == false;
        if (v425){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v424);
        } else {
        }
        float * v427;
        v427 = reinterpret_cast<float *>(&v423[v418]);
        bool v429;
        v429 = 0 <= v414;
        bool v430;
        v430 = v429 == false;
        if (v430){
            assert("The index needs to be zero or positive." && v429);
        } else {
        }
        int v432;
        v432 = v414 % 2;
        int v433;
        v433 = v414 / 2;
        bool v434;
        v434 = v433 < 4;
        bool v435;
        v435 = v434 == false;
        if (v435){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v434);
        } else {
        }
        assert("Tensor range check" && 0 <= v433 && v433 < 4);
        assert("Tensor range check" && 0 <= v432 && v432 < 2);
        int v437;
        v437 = 2 * v433;
        int v438;
        v438 = v437 + v432;
        v427[v438] = v412;
        int v439;
        v439 = v433 + 1;
        bool v440;
        v440 = v439 < 16;
        bool v441;
        v441 = v440 == false;
        if (v441){
            assert("The barrier_id has to be less than 16." && v440);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v439), "r"(64));
        int v443;
        v443 = threadIdx.x;
        int v444;
        v444 = v443 % 32;
        bool v445;
        v445 = v444 < 2;
        float v448;
        if (v445){
            assert("Tensor range check" && 0 <= v433 && v433 < 4);
            assert("Tensor range check" && 0 <= v444 && v444 < 2);
            int v446;
            v446 = v437 + v444;
            float v447;
            v447 = v427[v446];
            v448 = v447;
        } else {
            v448 = 0.0f;
        }
        __syncthreads();
        float v449;
        v449 = cooperative_groups::reduce(v410, v448, v411);
        int v450[4];
        int v451;
        v451 = 0;
        while (while_method_3(v451)){
            int v453;
            v453 = 0;
            while (while_method_1(v453)){
                assert("Tensor range check" && 0 <= v451 && v451 < 1);
                assert("Tensor range check" && 0 <= v453 && v453 < 4);
                int v455;
                v455 = 4 * v451;
                int v456;
                v456 = v455 + v453;
                bool v457;
                v457 = v379[v456];
                int v458;
                if (v457){
                    v458 = 1;
                } else {
                    v458 = 0;
                }
                assert("Tensor range check" && 0 <= v451 && v451 < 1);
                assert("Tensor range check" && 0 <= v453 && v453 < 4);
                v450[v456] = v458;
                v453 += 1 ;
            }
            v451 += 1 ;
        }
        int v459;
        v459 = 0;
        int v460;
        v460 = 0;
        while (while_method_3(v460)){
            int v462;
            v462 = 0;
            while (while_method_1(v462)){
                assert("Tensor range check" && 0 <= v460 && v460 < 1);
                assert("Tensor range check" && 0 <= v462 && v462 < 4);
                int v464;
                v464 = 4 * v460;
                int v465;
                v465 = v464 + v462;
                int v466;
                v466 = v450[v465];
                int v467;
                v467 = v459 + v466;
                v459 = v467;
                v462 += 1 ;
            }
            v460 += 1 ;
        }
        auto v468 = cooperative_groups::coalesced_threads();
        Closure4 v469{};
        int v470;
        v470 = cooperative_groups::reduce(v468, v459, v469);
        int v471;
        v471 = threadIdx.x;
        int v472;
        v472 = v471 / 32;
        if (v421){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v420);
        } else {
        }
        extern __shared__ unsigned char v474[];
        if (v425){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v424);
        } else {
        }
        int * v476;
        v476 = reinterpret_cast<int *>(&v474[v418]);
        bool v478;
        v478 = 0 <= v472;
        bool v479;
        v479 = v478 == false;
        if (v479){
            assert("The index needs to be zero or positive." && v478);
        } else {
        }
        int v481;
        v481 = v472 % 2;
        int v482;
        v482 = v472 / 2;
        bool v483;
        v483 = v482 < 4;
        bool v484;
        v484 = v483 == false;
        if (v484){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v483);
        } else {
        }
        assert("Tensor range check" && 0 <= v482 && v482 < 4);
        assert("Tensor range check" && 0 <= v481 && v481 < 2);
        int v486;
        v486 = 2 * v482;
        int v487;
        v487 = v486 + v481;
        v476[v487] = v470;
        int v488;
        v488 = v482 + 1;
        bool v489;
        v489 = v488 < 16;
        bool v490;
        v490 = v489 == false;
        if (v490){
            assert("The barrier_id has to be less than 16." && v489);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v488), "r"(64));
        int v492;
        v492 = threadIdx.x;
        int v493;
        v493 = v492 % 32;
        bool v494;
        v494 = v493 < 2;
        int v497;
        if (v494){
            assert("Tensor range check" && 0 <= v482 && v482 < 4);
            assert("Tensor range check" && 0 <= v493 && v493 < 2);
            int v495;
            v495 = v486 + v493;
            int v496;
            v496 = v476[v495];
            v497 = v496;
        } else {
            v497 = 0;
        }
        __syncthreads();
        int v498;
        v498 = cooperative_groups::reduce(v468, v497, v469);
        float v499;
        v499 = (float)v498;
        float v500;
        v500 = 1.0f / v499;
        float v501[4];
        int v502;
        v502 = 0;
        while (while_method_3(v502)){
            int v504;
            v504 = 0;
            while (while_method_1(v504)){
                assert("Tensor range check" && 0 <= v502 && v502 < 1);
                assert("Tensor range check" && 0 <= v504 && v504 < 4);
                int v506;
                v506 = 4 * v502;
                int v507;
                v507 = v506 + v504;
                float v508;
                v508 = v389[v507];
                bool v509;
                v509 = v379[v507];
                bool v510;
                v510 = v509 == false;
                float v515;
                if (v510){
                    v515 = 0.0f;
                } else {
                    bool v511;
                    v511 = v449 == 0.0f;
                    bool v512;
                    v512 = v511 != true;
                    if (v512){
                        float v513;
                        v513 = v508 / v449;
                        v515 = v513;
                    } else {
                        v515 = v500;
                    }
                }
                assert("Tensor range check" && 0 <= v502 && v502 < 1);
                assert("Tensor range check" && 0 <= v504 && v504 < 4);
                v501[v507] = v515;
                v504 += 1 ;
            }
            v502 += 1 ;
        }
        int v516;
        v516 = 0;
        while (while_method_3(v516)){
            assert("Tensor range check" && 0 <= v516 && v516 < 1);
            int v518;
            v518 = 256 * v516;
            int v519;
            v519 = v518 + v344;
            assert("Tensor range check" && 0 <= v516 && v516 < 1);
            int v520;
            v520 = 4 * v516;
            int4* v521;
            v521 = reinterpret_cast<int4*>(v501 + v520);
            int4* v522;
            v522 = reinterpret_cast<int4*>(v340 + v519);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v521) % 16 == 0 && reinterpret_cast<unsigned long long>(v522) % 16 == 0);
            *v522 = *v521;
            v516 += 1 ;
        }
        assert("Tensor range check" && 0 <= v336 && v336 < 256);
        v324 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v315 && v315 < 256);
    __syncthreads();
    int v523;
    v523 = threadIdx.x;
    int v524;
    v524 = blockIdx.x;
    int v525;
    v525 = v524 * 256;
    int v526;
    v526 = v523 + v525;
    unsigned long long v527;
    v527 = (unsigned long long)v526;
    curandStatePhilox4_32_10_t v528;
    curand_init(12344321ull,v527,0ull,&v528);
    float * v529;
    v529 = v1+v9;
    if (v140){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v139);
    } else {
    }
    extern __shared__ unsigned char v532[];
    if (v144){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v143);
    } else {
    }
    float * * v534;
    v534 = reinterpret_cast<float * *>(&v532[0ull]);
    int * v536;
    v536 = reinterpret_cast<int *>(&v532[v30]);
    int v538;
    v538 = threadIdx.x;
    assert("Tensor range check" && 0 <= v538 && v538 < 256);
    v534[v538] = v529;
    __syncthreads();
    bool v539;
    v539 = 0 <= v538;
    bool v540;
    v540 = v539 == false;
    if (v540){
        assert("The index needs to be zero or positive." && v539);
    } else {
    }
    int v542;
    v542 = v538 % 64;
    int v543;
    v543 = v538 / 64;
    bool v544;
    v544 = v543 < 4;
    bool v545;
    v545 = v544 == false;
    if (v545){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v544);
    } else {
    }
    assert("Tensor range check" && 0 <= v543 && v543 < 4);
    int v547;
    v547 = 0;
    while (while_method_4(v547)){
        bool v549;
        v549 = 0 <= v543;
        bool v550;
        v550 = v549 && v544;
        bool v551;
        v551 = v550 == false;
        if (v551){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v550);
        } else {
        }
        bool v553;
        v553 = 0 <= v547;
        bool v555;
        if (v553){
            bool v554;
            v554 = v547 < 64;
            v555 = v554;
        } else {
            v555 = false;
        }
        bool v556;
        v556 = v555 == false;
        if (v556){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v555);
        } else {
        }
        int v558;
        v558 = v547 * 4;
        int v559;
        v559 = v558 + v543;
        assert("Tensor range check" && 0 <= v547 && v547 < 64);
        int v560;
        v560 = 4 * v547;
        int v561;
        v561 = v560 + v543;
        float * v562;
        v562 = v534[v561];
        int v563;
        v563 = blockIdx.x;
        int v564;
        v564 = v563 * 256;
        int v565;
        v565 = v564 + v559;
        assert("Tensor range check" && 0 <= v542 && v542 < 64);
        int v566;
        v566 = 4 * v542;
        float v567[4];
        int v568[4];
        int v569;
        v569 = 0;
        while (while_method_3(v569)){
            assert("Tensor range check" && 0 <= v569 && v569 < 1);
            int v571;
            v571 = 4 * v569;
            assert("Tensor range check" && 0 <= v569 && v569 < 1);
            int v572;
            v572 = 256 * v569;
            int v573;
            v573 = v572 + v566;
            int4* v574;
            v574 = reinterpret_cast<int4*>(v562 + v573);
            int4* v575;
            v575 = reinterpret_cast<int4*>(v567 + v571);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v574) % 16 == 0 && reinterpret_cast<unsigned long long>(v575) % 16 == 0);
            *v575 = *v574;
            v569 += 1 ;
        }
        int v576;
        v576 = 0;
        while (while_method_3(v576)){
            int v578;
            v578 = 0;
            while (while_method_1(v578)){
                bool v580;
                v580 = 0 <= v578;
                bool v582;
                if (v580){
                    bool v581;
                    v581 = v578 < 4;
                    v582 = v581;
                } else {
                    v582 = false;
                }
                bool v583;
                v583 = v582 == false;
                if (v583){
                    assert("The indices should be inside the range of the dimension." && v582);
                } else {
                }
                bool v585;
                v585 = 0 <= v542;
                bool v587;
                if (v585){
                    bool v586;
                    v586 = v542 < 64;
                    v587 = v586;
                } else {
                    v587 = false;
                }
                bool v588;
                v588 = v587 == false;
                if (v588){
                    assert("The indices should be inside the range of the dimension." && v587);
                } else {
                }
                int v590;
                v590 = v542 * 4;
                int v591;
                v591 = v578 + v590;
                bool v592;
                v592 = 0 <= v576;
                bool v594;
                if (v592){
                    bool v593;
                    v593 = v576 < 1;
                    v594 = v593;
                } else {
                    v594 = false;
                }
                bool v595;
                v595 = v594 == false;
                if (v595){
                    assert("The indices should be inside the range of the dimension." && v594);
                } else {
                }
                int v597;
                v597 = v576 * 256;
                int v598;
                v598 = v591 + v597;
                assert("Tensor range check" && 0 <= v576 && v576 < 1);
                assert("Tensor range check" && 0 <= v578 && v578 < 4);
                int v599;
                v599 = 4 * v576;
                int v600;
                v600 = v599 + v578;
                v568[v600] = v598;
                v578 += 1 ;
            }
            v576 += 1 ;
        }
        bool v601[4];
        int v602;
        v602 = 0;
        while (while_method_3(v602)){
            int v604;
            v604 = 0;
            while (while_method_1(v604)){
                assert("Tensor range check" && 0 <= v602 && v602 < 1);
                assert("Tensor range check" && 0 <= v604 && v604 < 4);
                int v606;
                v606 = 4 * v602;
                int v607;
                v607 = v606 + v604;
                float v608;
                v608 = v567[v607];
                int v609;
                v609 = v568[v607];
                bool v610;
                v610 = v609 < 3;
                assert("Tensor range check" && 0 <= v602 && v602 < 1);
                assert("Tensor range check" && 0 <= v604 && v604 < 4);
                v601[v607] = v610;
                v604 += 1 ;
            }
            v602 += 1 ;
        }
        float v611[4];
        int v612;
        v612 = 0;
        while (while_method_3(v612)){
            int v614;
            v614 = 0;
            while (while_method_1(v614)){
                assert("Tensor range check" && 0 <= v612 && v612 < 1);
                assert("Tensor range check" && 0 <= v614 && v614 < 4);
                int v616;
                v616 = 4 * v612;
                int v617;
                v617 = v616 + v614;
                float v618;
                v618 = v567[v617];
                bool v619;
                v619 = v601[v617];
                float v620;
                if (v619){
                    v620 = v618;
                } else {
                    v620 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v612 && v612 < 1);
                assert("Tensor range check" && 0 <= v614 && v614 < 4);
                v611[v617] = v620;
                v614 += 1 ;
            }
            v612 += 1 ;
        }
        float v621;
        v621 = 0.0f;
        int v622;
        v622 = 0;
        while (while_method_3(v622)){
            int v624;
            v624 = 0;
            while (while_method_1(v624)){
                assert("Tensor range check" && 0 <= v622 && v622 < 1);
                assert("Tensor range check" && 0 <= v624 && v624 < 4);
                int v626;
                v626 = 4 * v622;
                int v627;
                v627 = v626 + v624;
                float v628;
                v628 = v611[v627];
                float v629;
                v629 = v621 + v628;
                v621 = v629;
                v624 += 1 ;
            }
            v622 += 1 ;
        }
        auto v630 = cooperative_groups::coalesced_threads();
        Closure0 v631{};
        float v632;
        v632 = cooperative_groups::reduce(v630, v621, v631);
        int v633;
        v633 = threadIdx.x;
        int v634;
        v634 = v633 / 32;
        unsigned long long v635;
        v635 = v138 + 16ull;
        unsigned long long v636;
        v636 = v635 - 1ull;
        unsigned long long v637;
        v637 = v636 % 16ull;
        unsigned long long v638;
        v638 = v636 - v637;
        unsigned long long v639;
        v639 = v638 + 32ull;
        bool v640;
        v640 = v639 <= 98304ull;
        bool v641;
        v641 = v640 == false;
        if (v641){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v640);
        } else {
        }
        extern __shared__ unsigned char v643[];
        bool v644;
        v644 = v639 <= v639;
        bool v645;
        v645 = v644 == false;
        if (v645){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v644);
        } else {
        }
        float * v647;
        v647 = reinterpret_cast<float *>(&v643[v638]);
        bool v649;
        v649 = 0 <= v634;
        bool v650;
        v650 = v649 == false;
        if (v650){
            assert("The index needs to be zero or positive." && v649);
        } else {
        }
        int v652;
        v652 = v634 % 2;
        int v653;
        v653 = v634 / 2;
        bool v654;
        v654 = v653 < 4;
        bool v655;
        v655 = v654 == false;
        if (v655){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v654);
        } else {
        }
        assert("Tensor range check" && 0 <= v653 && v653 < 4);
        assert("Tensor range check" && 0 <= v652 && v652 < 2);
        int v657;
        v657 = 2 * v653;
        int v658;
        v658 = v657 + v652;
        v647[v658] = v632;
        int v659;
        v659 = v653 + 1;
        bool v660;
        v660 = v659 < 16;
        bool v661;
        v661 = v660 == false;
        if (v661){
            assert("The barrier_id has to be less than 16." && v660);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v659), "r"(64));
        int v663;
        v663 = threadIdx.x;
        int v664;
        v664 = v663 % 32;
        bool v665;
        v665 = v664 < 2;
        float v668;
        if (v665){
            assert("Tensor range check" && 0 <= v653 && v653 < 4);
            assert("Tensor range check" && 0 <= v664 && v664 < 2);
            int v666;
            v666 = v657 + v664;
            float v667;
            v667 = v647[v666];
            v668 = v667;
        } else {
            v668 = 0.0f;
        }
        __syncthreads();
        float v669;
        v669 = cooperative_groups::reduce(v630, v668, v631);
        int v670[4];
        int v671;
        v671 = 0;
        while (while_method_3(v671)){
            int v673;
            v673 = 0;
            while (while_method_1(v673)){
                assert("Tensor range check" && 0 <= v671 && v671 < 1);
                assert("Tensor range check" && 0 <= v673 && v673 < 4);
                int v675;
                v675 = 4 * v671;
                int v676;
                v676 = v675 + v673;
                bool v677;
                v677 = v601[v676];
                int v678;
                if (v677){
                    v678 = 1;
                } else {
                    v678 = 0;
                }
                assert("Tensor range check" && 0 <= v671 && v671 < 1);
                assert("Tensor range check" && 0 <= v673 && v673 < 4);
                v670[v676] = v678;
                v673 += 1 ;
            }
            v671 += 1 ;
        }
        int v679;
        v679 = 0;
        int v680;
        v680 = 0;
        while (while_method_3(v680)){
            int v682;
            v682 = 0;
            while (while_method_1(v682)){
                assert("Tensor range check" && 0 <= v680 && v680 < 1);
                assert("Tensor range check" && 0 <= v682 && v682 < 4);
                int v684;
                v684 = 4 * v680;
                int v685;
                v685 = v684 + v682;
                int v686;
                v686 = v670[v685];
                int v687;
                v687 = v679 + v686;
                v679 = v687;
                v682 += 1 ;
            }
            v680 += 1 ;
        }
        auto v688 = cooperative_groups::coalesced_threads();
        Closure4 v689{};
        int v690;
        v690 = cooperative_groups::reduce(v688, v679, v689);
        int v691;
        v691 = threadIdx.x;
        int v692;
        v692 = v691 / 32;
        if (v641){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v640);
        } else {
        }
        extern __shared__ unsigned char v694[];
        if (v645){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v644);
        } else {
        }
        int * v696;
        v696 = reinterpret_cast<int *>(&v694[v638]);
        bool v698;
        v698 = 0 <= v692;
        bool v699;
        v699 = v698 == false;
        if (v699){
            assert("The index needs to be zero or positive." && v698);
        } else {
        }
        int v701;
        v701 = v692 % 2;
        int v702;
        v702 = v692 / 2;
        bool v703;
        v703 = v702 < 4;
        bool v704;
        v704 = v703 == false;
        if (v704){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v703);
        } else {
        }
        assert("Tensor range check" && 0 <= v702 && v702 < 4);
        assert("Tensor range check" && 0 <= v701 && v701 < 2);
        int v706;
        v706 = 2 * v702;
        int v707;
        v707 = v706 + v701;
        v696[v707] = v690;
        int v708;
        v708 = v702 + 1;
        bool v709;
        v709 = v708 < 16;
        bool v710;
        v710 = v709 == false;
        if (v710){
            assert("The barrier_id has to be less than 16." && v709);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v708), "r"(64));
        int v712;
        v712 = threadIdx.x;
        int v713;
        v713 = v712 % 32;
        bool v714;
        v714 = v713 < 2;
        int v717;
        if (v714){
            assert("Tensor range check" && 0 <= v702 && v702 < 4);
            assert("Tensor range check" && 0 <= v713 && v713 < 2);
            int v715;
            v715 = v706 + v713;
            int v716;
            v716 = v696[v715];
            v717 = v716;
        } else {
            v717 = 0;
        }
        __syncthreads();
        int v718;
        v718 = cooperative_groups::reduce(v688, v717, v689);
        float v719;
        v719 = (float)v718;
        float v720;
        v720 = v669 / v719;
        float v721[4];
        int v722;
        v722 = 0;
        while (while_method_3(v722)){
            int v724;
            v724 = 0;
            while (while_method_1(v724)){
                assert("Tensor range check" && 0 <= v722 && v722 < 1);
                assert("Tensor range check" && 0 <= v724 && v724 < 4);
                int v726;
                v726 = 4 * v722;
                int v727;
                v727 = v726 + v724;
                float v728;
                v728 = v567[v727];
                bool v729;
                v729 = v601[v727];
                float v730;
                if (v729){
                    v730 = v728;
                } else {
                    v730 = -1.0f / 0.0f;
                }
                float v731;
                v731 = v730 - v720;
                float v732;
                v732 = exp(v731);
                bool v733;
                v733 = v732 < 1.0f / 0.0f;
                bool v734;
                v734 = v733 == false;
                if (v734){
                    assert("The softmax values must not grow too large." && v733);
                } else {
                }
                bool v736;
                v736 = isnan(v732);
                bool v737;
                v737 = v736 == false;
                bool v738;
                v738 = v737 == false;
                if (v738){
                    assert("The softmax values must not be nans." && v737);
                } else {
                }
                assert("Tensor range check" && 0 <= v722 && v722 < 1);
                assert("Tensor range check" && 0 <= v724 && v724 < 4);
                v721[v727] = v732;
                v724 += 1 ;
            }
            v722 += 1 ;
        }
        float v740;
        v740 = 0.0f;
        int v741;
        v741 = 0;
        while (while_method_3(v741)){
            int v743;
            v743 = 0;
            while (while_method_1(v743)){
                assert("Tensor range check" && 0 <= v741 && v741 < 1);
                assert("Tensor range check" && 0 <= v743 && v743 < 4);
                int v745;
                v745 = 4 * v741;
                int v746;
                v746 = v745 + v743;
                float v747;
                v747 = v721[v746];
                float v748;
                v748 = v740 + v747;
                v740 = v748;
                v743 += 1 ;
            }
            v741 += 1 ;
        }
        auto v749 = cooperative_groups::coalesced_threads();
        float v750;
        v750 = cooperative_groups::reduce(v749, v740, v631);
        int v751;
        v751 = threadIdx.x;
        int v752;
        v752 = v751 / 32;
        if (v641){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v640);
        } else {
        }
        extern __shared__ unsigned char v754[];
        if (v645){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v644);
        } else {
        }
        float * v756;
        v756 = reinterpret_cast<float *>(&v754[v638]);
        bool v758;
        v758 = 0 <= v752;
        bool v759;
        v759 = v758 == false;
        if (v759){
            assert("The index needs to be zero or positive." && v758);
        } else {
        }
        int v761;
        v761 = v752 % 2;
        int v762;
        v762 = v752 / 2;
        bool v763;
        v763 = v762 < 4;
        bool v764;
        v764 = v763 == false;
        if (v764){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v763);
        } else {
        }
        assert("Tensor range check" && 0 <= v762 && v762 < 4);
        assert("Tensor range check" && 0 <= v761 && v761 < 2);
        int v766;
        v766 = 2 * v762;
        int v767;
        v767 = v766 + v761;
        v756[v767] = v750;
        int v768;
        v768 = v762 + 1;
        bool v769;
        v769 = v768 < 16;
        bool v770;
        v770 = v769 == false;
        if (v770){
            assert("The barrier_id has to be less than 16." && v769);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v768), "r"(64));
        int v772;
        v772 = threadIdx.x;
        int v773;
        v773 = v772 % 32;
        bool v774;
        v774 = v773 < 2;
        float v777;
        if (v774){
            assert("Tensor range check" && 0 <= v762 && v762 < 4);
            assert("Tensor range check" && 0 <= v773 && v773 < 2);
            int v775;
            v775 = v766 + v773;
            float v776;
            v776 = v756[v775];
            v777 = v776;
        } else {
            v777 = 0.0f;
        }
        __syncthreads();
        float v778;
        v778 = cooperative_groups::reduce(v749, v777, v631);
        float v779[4];
        int v780;
        v780 = 0;
        while (while_method_3(v780)){
            int v782;
            v782 = 0;
            while (while_method_1(v782)){
                assert("Tensor range check" && 0 <= v780 && v780 < 1);
                assert("Tensor range check" && 0 <= v782 && v782 < 4);
                int v784;
                v784 = 4 * v780;
                int v785;
                v785 = v784 + v782;
                float v786;
                v786 = v721[v785];
                float v787;
                v787 = v786 / v778;
                assert("Tensor range check" && 0 <= v780 && v780 < 1);
                assert("Tensor range check" && 0 <= v782 && v782 < 4);
                v779[v785] = v787;
                v782 += 1 ;
            }
            v780 += 1 ;
        }
        float v788[4];
        float v789;
        v789 = 0.0f;
        int v790;
        v790 = 0;
        while (while_method_3(v790)){
            assert("Tensor range check" && 0 <= v790 && v790 < 1);
            int v792;
            v792 = 4 * v790;
            assert("Tensor range check" && 0 <= v790 && v790 < 1);
            float v793;
            v793 = 0.0f;
            int v794;
            v794 = 0;
            while (while_method_1(v794)){
                assert("Tensor range check" && 0 <= v794 && v794 < 4);
                int v796;
                v796 = v794 + v792;
                float v797;
                v797 = v779[v796];
                float v798;
                v798 = v793 + v797;
                v793 = v798;
                v794 += 1 ;
            }
            auto v799 = cooperative_groups::coalesced_threads();
            int v800;
            v800 = threadIdx.x;
            int v801;
            v801 = v800 / 32;
            if (v641){
                assert("The dynamic shared memory is insufficient to allocate the tensor." && v640);
            } else {
            }
            extern __shared__ unsigned char v803[];
            if (v645){
                assert("The length of the partition has to be less than or equal to the length of the base array." && v644);
            } else {
            }
            float * v805;
            v805 = reinterpret_cast<float *>(&v803[v638]);
            Closure2 v807{};
            float v808;
            v808 = cooperative_groups::inclusive_scan(v799, v793, v807);
            float v809;
            v809 = v799.shfl_up(v808,1);
            bool v810;
            v810 = v799.thread_rank() == 0;
            float v811;
            if (v810){
                v811 = 0.0f;
            } else {
                v811 = v809;
            }
            float v812;
            v812 = v799.shfl(v808,v799.num_threads()-1);
            bool v813;
            v813 = 0 <= v801;
            bool v814;
            v814 = v813 == false;
            if (v814){
                assert("The index needs to be zero or positive." && v813);
            } else {
            }
            int v816;
            v816 = v801 % 2;
            int v817;
            v817 = v801 / 2;
            bool v818;
            v818 = v817 < 4;
            bool v819;
            v819 = v818 == false;
            if (v819){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v818);
            } else {
            }
            assert("Tensor range check" && 0 <= v817 && v817 < 4);
            assert("Tensor range check" && 0 <= v816 && v816 < 2);
            int v821;
            v821 = 2 * v817;
            int v822;
            v822 = v821 + v816;
            v805[v822] = v812;
            int v823;
            v823 = v817 + 1;
            bool v824;
            v824 = v823 < 16;
            bool v825;
            v825 = v824 == false;
            if (v825){
                assert("The barrier_id has to be less than 16." && v824);
            } else {
            }
            asm("barrier.cta.sync %0, %1;" :: "r"(v823), "r"(64));
            int v827;
            v827 = threadIdx.x;
            int v828;
            v828 = v827 % 32;
            bool v829;
            v829 = v828 < 2;
            float v832;
            if (v829){
                assert("Tensor range check" && 0 <= v817 && v817 < 4);
                assert("Tensor range check" && 0 <= v828 && v828 < 2);
                int v830;
                v830 = v821 + v828;
                float v831;
                v831 = v805[v830];
                v832 = v831;
            } else {
                v832 = 0.0f;
            }
            __syncthreads();
            float v833;
            v833 = cooperative_groups::inclusive_scan(v799, v832, v807);
            float v834;
            v834 = v799.shfl_up(v833,1);
            bool v835;
            v835 = v799.thread_rank() == 0;
            float v836;
            if (v835){
                v836 = 0.0f;
            } else {
                v836 = v834;
            }
            float v837;
            v837 = v799.shfl(v833,v799.num_threads()-1);
            float v838;
            v838 = v799.shfl(v836,v816);
            float v839;
            v839 = v838 + v811;
            float v840;
            v840 = v789 + v839;
            float v841;
            v841 = v840;
            int v842;
            v842 = 0;
            while (while_method_1(v842)){
                assert("Tensor range check" && 0 <= v842 && v842 < 4);
                int v844;
                v844 = v842 + v792;
                float v845;
                v845 = v779[v844];
                float v846;
                v846 = v841 + v845;
                assert("Tensor range check" && 0 <= v842 && v842 < 4);
                v788[v844] = v846;
                v841 = v846;
                v842 += 1 ;
            }
            float v847;
            v847 = v789 + v837;
            v789 = v847;
            v790 += 1 ;
        }
        float v848[4];
        bool v849[4];
        int v850;
        v850 = 0;
        while (while_method_3(v850)){
            int v852;
            v852 = 0;
            while (while_method_1(v852)){
                assert("Tensor range check" && 0 <= v850 && v850 < 1);
                assert("Tensor range check" && 0 <= v852 && v852 < 4);
                int v854;
                v854 = 4 * v850;
                int v855;
                v855 = v854 + v852;
                float v856;
                v856 = v788[v855];
                float v857;
                v857 = v779[v855];
                bool v858;
                v858 = v857 > 0.0f;
                assert("Tensor range check" && 0 <= v850 && v850 < 1);
                assert("Tensor range check" && 0 <= v852 && v852 < 4);
                v848[v855] = v856;
                v849[v855] = v858;
                v852 += 1 ;
            }
            v850 += 1 ;
        }
        float v859; bool v860;
        Tuple2 tmp36 = Tuple2{-1.0f / 0.0f, false};
        v859 = tmp36.v0; v860 = tmp36.v1;
        int v861;
        v861 = 0;
        while (while_method_3(v861)){
            int v863;
            v863 = 0;
            while (while_method_1(v863)){
                assert("Tensor range check" && 0 <= v861 && v861 < 1);
                assert("Tensor range check" && 0 <= v863 && v863 < 4);
                int v865;
                v865 = 4 * v861;
                int v866;
                v866 = v865 + v863;
                float v867;
                v867 = v848[v866];
                bool v868;
                v868 = v849[v866];
                float v875; bool v876;
                if (v860){
                    if (v868){
                        bool v869;
                        v869 = v859 >= v867;
                        float v870;
                        if (v869){
                            v870 = v859;
                        } else {
                            v870 = v867;
                        }
                        v875 = v870; v876 = true;
                    } else {
                        v875 = v859; v876 = v860;
                    }
                } else {
                    if (v868){
                        v875 = v867; v876 = v868;
                    } else {
                        v875 = v859; v876 = v860;
                    }
                }
                v859 = v875;
                v860 = v876;
                v863 += 1 ;
            }
            v861 += 1 ;
        }
        auto v877 = cooperative_groups::coalesced_threads();
        Closure5 v878{};
        float v879; bool v880;
        Tuple2 tmp37 = cooperative_groups::reduce(v877, Tuple2{v859, v860}, v878);
        v879 = tmp37.v0; v880 = tmp37.v1;
        int v881;
        v881 = threadIdx.x;
        int v882;
        v882 = v881 / 32;
        unsigned long long v883;
        v883 = v639 + 16ull;
        unsigned long long v884;
        v884 = v883 - 1ull;
        unsigned long long v885;
        v885 = v884 % 16ull;
        unsigned long long v886;
        v886 = v884 - v885;
        unsigned long long v887;
        v887 = v886 + 8ull;
        bool v888;
        v888 = v887 <= 98304ull;
        bool v889;
        v889 = v888 == false;
        if (v889){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v888);
        } else {
        }
        extern __shared__ unsigned char v891[];
        bool v892;
        v892 = v887 <= v887;
        bool v893;
        v893 = v892 == false;
        if (v893){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v892);
        } else {
        }
        float * v895;
        v895 = reinterpret_cast<float *>(&v891[v638]);
        bool * v897;
        v897 = reinterpret_cast<bool *>(&v891[v886]);
        bool v899;
        v899 = 0 <= v882;
        bool v900;
        v900 = v899 == false;
        if (v900){
            assert("The index needs to be zero or positive." && v899);
        } else {
        }
        int v902;
        v902 = v882 % 2;
        int v903;
        v903 = v882 / 2;
        bool v904;
        v904 = v903 < 4;
        bool v905;
        v905 = v904 == false;
        if (v905){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v904);
        } else {
        }
        assert("Tensor range check" && 0 <= v903 && v903 < 4);
        assert("Tensor range check" && 0 <= v902 && v902 < 2);
        int v907;
        v907 = 2 * v903;
        int v908;
        v908 = v907 + v902;
        v895[v908] = v879;
        v897[v908] = v880;
        int v909;
        v909 = v903 + 1;
        bool v910;
        v910 = v909 < 16;
        bool v911;
        v911 = v910 == false;
        if (v911){
            assert("The barrier_id has to be less than 16." && v910);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v909), "r"(64));
        int v913;
        v913 = threadIdx.x;
        int v914;
        v914 = v913 % 32;
        bool v915;
        v915 = v914 < 2;
        float v919; bool v920;
        if (v915){
            assert("Tensor range check" && 0 <= v903 && v903 < 4);
            assert("Tensor range check" && 0 <= v914 && v914 < 2);
            int v916;
            v916 = v907 + v914;
            float v917;
            v917 = v895[v916];
            bool v918;
            v918 = v897[v916];
            v919 = v917; v920 = v918;
        } else {
            v919 = -1.0f / 0.0f; v920 = false;
        }
        __syncthreads();
        float v921; bool v922;
        Tuple2 tmp38 = cooperative_groups::reduce(v877, Tuple2{v919, v920}, v878);
        v921 = tmp38.v0; v922 = tmp38.v1;
        bool v923;
        v923 = v922 == false;
        if (v923){
            assert("The local reduce must be true." && v922);
        } else {
        }
        float v925[4];
        int v926[4];
        int v927;
        v927 = 0;
        while (while_method_3(v927)){
            int v929;
            v929 = 0;
            while (while_method_1(v929)){
                assert("Tensor range check" && 0 <= v927 && v927 < 1);
                assert("Tensor range check" && 0 <= v929 && v929 < 4);
                int v931;
                v931 = 4 * v927;
                int v932;
                v932 = v931 + v929;
                int v933;
                v933 = v568[v932];
                float v934;
                v934 = curand_uniform(&v528);
                assert("Tensor range check" && 0 <= v927 && v927 < 1);
                assert("Tensor range check" && 0 <= v929 && v929 < 4);
                v925[v932] = v934;
                v926[v932] = v933;
                v929 += 1 ;
            }
            v927 += 1 ;
        }
        float v935; int v936;
        Tuple1 tmp39 = Tuple1{0.0f, 2147483647};
        v935 = tmp39.v0; v936 = tmp39.v1;
        int v937;
        v937 = 0;
        while (while_method_3(v937)){
            int v939;
            v939 = 0;
            while (while_method_1(v939)){
                assert("Tensor range check" && 0 <= v937 && v937 < 1);
                assert("Tensor range check" && 0 <= v939 && v939 < 4);
                int v941;
                v941 = 4 * v937;
                int v942;
                v942 = v941 + v939;
                float v943;
                v943 = v925[v942];
                int v944;
                v944 = v926[v942];
                bool v945;
                v945 = v936 < v944;
                float v946; int v947;
                if (v945){
                    v946 = v935; v947 = v936;
                } else {
                    v946 = v943; v947 = v944;
                }
                v935 = v946;
                v936 = v947;
                v939 += 1 ;
            }
            v937 += 1 ;
        }
        auto v948 = cooperative_groups::coalesced_threads();
        Closure6 v949{};
        float v950; int v951;
        Tuple1 tmp40 = cooperative_groups::reduce(v948, Tuple1{v935, v936}, v949);
        v950 = tmp40.v0; v951 = tmp40.v1;
        int v952;
        v952 = threadIdx.x;
        int v953;
        v953 = v952 / 32;
        unsigned long long v954;
        v954 = v886 + 32ull;
        bool v955;
        v955 = v954 <= 98304ull;
        bool v956;
        v956 = v955 == false;
        if (v956){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v955);
        } else {
        }
        extern __shared__ unsigned char v958[];
        bool v959;
        v959 = v954 <= v954;
        bool v960;
        v960 = v959 == false;
        if (v960){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v959);
        } else {
        }
        float * v962;
        v962 = reinterpret_cast<float *>(&v958[v638]);
        int * v964;
        v964 = reinterpret_cast<int *>(&v958[v886]);
        bool v966;
        v966 = 0 <= v953;
        bool v967;
        v967 = v966 == false;
        if (v967){
            assert("The index needs to be zero or positive." && v966);
        } else {
        }
        int v969;
        v969 = v953 % 2;
        int v970;
        v970 = v953 / 2;
        bool v971;
        v971 = v970 < 4;
        bool v972;
        v972 = v971 == false;
        if (v972){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v971);
        } else {
        }
        assert("Tensor range check" && 0 <= v970 && v970 < 4);
        assert("Tensor range check" && 0 <= v969 && v969 < 2);
        int v974;
        v974 = 2 * v970;
        int v975;
        v975 = v974 + v969;
        v962[v975] = v950;
        v964[v975] = v951;
        int v976;
        v976 = v970 + 1;
        bool v977;
        v977 = v976 < 16;
        bool v978;
        v978 = v977 == false;
        if (v978){
            assert("The barrier_id has to be less than 16." && v977);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v976), "r"(64));
        int v980;
        v980 = threadIdx.x;
        int v981;
        v981 = v980 % 32;
        bool v982;
        v982 = v981 < 2;
        float v986; int v987;
        if (v982){
            assert("Tensor range check" && 0 <= v970 && v970 < 4);
            assert("Tensor range check" && 0 <= v981 && v981 < 2);
            int v983;
            v983 = v974 + v981;
            float v984;
            v984 = v962[v983];
            int v985;
            v985 = v964[v983];
            v986 = v984; v987 = v985;
        } else {
            v986 = 0.0f; v987 = 2147483647;
        }
        __syncthreads();
        float v988; int v989;
        Tuple1 tmp41 = cooperative_groups::reduce(v948, Tuple1{v986, v987}, v949);
        v988 = tmp41.v0; v989 = tmp41.v1;
        float v990;
        v990 = v921 * v988;
        int v991[4];
        bool v992[4];
        int v993;
        v993 = 0;
        while (while_method_3(v993)){
            int v995;
            v995 = 0;
            while (while_method_1(v995)){
                assert("Tensor range check" && 0 <= v993 && v993 < 1);
                assert("Tensor range check" && 0 <= v995 && v995 < 4);
                int v997;
                v997 = 4 * v993;
                int v998;
                v998 = v997 + v995;
                float v999;
                v999 = v848[v998];
                bool v1000;
                v1000 = v849[v998];
                int v1001;
                v1001 = v568[v998];
                int v1004; bool v1005;
                if (v1000){
                    float v1002;
                    v1002 = v999 - v990;
                    bool v1003;
                    v1003 = v1002 >= 0.0f;
                    v1004 = v1001; v1005 = v1003;
                } else {
                    v1004 = 2147483647; v1005 = false;
                }
                assert("Tensor range check" && 0 <= v993 && v993 < 1);
                assert("Tensor range check" && 0 <= v995 && v995 < 4);
                v991[v998] = v1004;
                v992[v998] = v1005;
                v995 += 1 ;
            }
            v993 += 1 ;
        }
        int v1006; bool v1007;
        Tuple3 tmp42 = Tuple3{2147483647, false};
        v1006 = tmp42.v0; v1007 = tmp42.v1;
        int v1008;
        v1008 = 0;
        while (while_method_3(v1008)){
            int v1010;
            v1010 = 0;
            while (while_method_1(v1010)){
                assert("Tensor range check" && 0 <= v1008 && v1008 < 1);
                assert("Tensor range check" && 0 <= v1010 && v1010 < 4);
                int v1012;
                v1012 = 4 * v1008;
                int v1013;
                v1013 = v1012 + v1010;
                int v1014;
                v1014 = v991[v1013];
                bool v1015;
                v1015 = v992[v1013];
                int v1022; bool v1023;
                if (v1007){
                    if (v1015){
                        bool v1016;
                        v1016 = v1006 < v1014;
                        int v1017;
                        if (v1016){
                            v1017 = v1006;
                        } else {
                            v1017 = v1014;
                        }
                        v1022 = v1017; v1023 = true;
                    } else {
                        v1022 = v1006; v1023 = v1007;
                    }
                } else {
                    if (v1015){
                        v1022 = v1014; v1023 = v1015;
                    } else {
                        v1022 = v1006; v1023 = v1007;
                    }
                }
                v1006 = v1022;
                v1007 = v1023;
                v1010 += 1 ;
            }
            v1008 += 1 ;
        }
        auto v1024 = cooperative_groups::coalesced_threads();
        Closure7 v1025{};
        int v1026; bool v1027;
        Tuple3 tmp43 = cooperative_groups::reduce(v1024, Tuple3{v1006, v1007}, v1025);
        v1026 = tmp43.v0; v1027 = tmp43.v1;
        int v1028;
        v1028 = threadIdx.x;
        int v1029;
        v1029 = v1028 / 32;
        if (v889){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v888);
        } else {
        }
        extern __shared__ unsigned char v1031[];
        if (v893){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v892);
        } else {
        }
        int * v1033;
        v1033 = reinterpret_cast<int *>(&v1031[v638]);
        bool * v1035;
        v1035 = reinterpret_cast<bool *>(&v1031[v886]);
        bool v1037;
        v1037 = 0 <= v1029;
        bool v1038;
        v1038 = v1037 == false;
        if (v1038){
            assert("The index needs to be zero or positive." && v1037);
        } else {
        }
        int v1040;
        v1040 = v1029 % 2;
        int v1041;
        v1041 = v1029 / 2;
        bool v1042;
        v1042 = v1041 < 4;
        bool v1043;
        v1043 = v1042 == false;
        if (v1043){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1042);
        } else {
        }
        assert("Tensor range check" && 0 <= v1041 && v1041 < 4);
        assert("Tensor range check" && 0 <= v1040 && v1040 < 2);
        int v1045;
        v1045 = 2 * v1041;
        int v1046;
        v1046 = v1045 + v1040;
        v1033[v1046] = v1026;
        v1035[v1046] = v1027;
        int v1047;
        v1047 = v1041 + 1;
        bool v1048;
        v1048 = v1047 < 16;
        bool v1049;
        v1049 = v1048 == false;
        if (v1049){
            assert("The barrier_id has to be less than 16." && v1048);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v1047), "r"(64));
        int v1051;
        v1051 = threadIdx.x;
        int v1052;
        v1052 = v1051 % 32;
        bool v1053;
        v1053 = v1052 < 2;
        int v1057; bool v1058;
        if (v1053){
            assert("Tensor range check" && 0 <= v1041 && v1041 < 4);
            assert("Tensor range check" && 0 <= v1052 && v1052 < 2);
            int v1054;
            v1054 = v1045 + v1052;
            int v1055;
            v1055 = v1033[v1054];
            bool v1056;
            v1056 = v1035[v1054];
            v1057 = v1055; v1058 = v1056;
        } else {
            v1057 = 2147483647; v1058 = false;
        }
        __syncthreads();
        int v1059; bool v1060;
        Tuple3 tmp44 = cooperative_groups::reduce(v1024, Tuple3{v1057, v1058}, v1025);
        v1059 = tmp44.v0; v1060 = tmp44.v1;
        bool v1061;
        v1061 = v1060 == false;
        if (v1061){
            assert("The local reduce must be true." && v1060);
        } else {
        }
        int v1063;
        v1063 = 0;
        while (while_method_3(v1063)){
            assert("Tensor range check" && 0 <= v1063 && v1063 < 1);
            assert("Tensor range check" && 0 <= v1063 && v1063 < 1);
            v1063 += 1 ;
        }
        assert("Tensor range check" && 0 <= v559 && v559 < 256);
        v536[v559] = v1059;
        v547 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v538 && v538 < 256);
    int v1065;
    v1065 = v536[v538];
    __syncthreads();
    int v1066;
    v1066 = threadIdx.x;
    assert("Tensor range check" && 0 <= v1066 && v1066 < 256);
    v5[v1066] = v1065;
    return ;
}
extern "C" __global__ void entry4(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14, float * v15, int * v16) {
    auto v17 = cooperative_groups::this_grid();
    int v18;
    v18 = threadIdx.x;
    bool v19;
    v19 = 0 <= v18;
    bool v20;
    v20 = v19 == false;
    if (v20){
        assert("The index needs to be zero or positive." && v19);
    } else {
    }
    int v22;
    v22 = v18 % 16;
    int v23;
    v23 = v18 / 16;
    bool v24;
    v24 = v23 < 16;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
    } else {
    }
    assert("Tensor range check" && 0 <= v23 && v23 < 16);
    assert("Tensor range check" && 0 <= v22 && v22 < 16);
    int v27;
    v27 = 4 * v22;
    int v28;
    v28 = 64 * v23;
    int v29;
    v29 = v28 + v27;
    assert("Tensor range check" && 0 <= v23 && v23 < 16);
    assert("Tensor range check" && 0 <= v22 && v22 < 16);
    int v30;
    v30 = blockIdx.x;
    int v31;
    v31 = v30;
    while (while_method_2(v31)){
        bool v33;
        v33 = 0 <= v31;
        bool v34;
        v34 = v33 == false;
        if (v34){
            assert("The index needs to be zero or positive." && v33);
        } else {
        }
        bool v36;
        v36 = v31 < 8;
        bool v37;
        v37 = v36 == false;
        if (v37){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v36);
        } else {
        }
        assert("Tensor range check" && 0 <= v31 && v31 < 8);
        int v39;
        v39 = 1024 * v31;
        int v40;
        v40 = v39 + v29;
        int v41[4];
        int v42[4];
        int v43;
        v43 = 0;
        while (while_method_3(v43)){
            assert("Tensor range check" && 0 <= v43 && v43 < 1);
            int v45;
            v45 = 4 * v43;
            assert("Tensor range check" && 0 <= v43 && v43 < 1);
            int v46;
            v46 = 64 * v43;
            int v47;
            v47 = v46 + v40;
            int4* v48;
            v48 = reinterpret_cast<int4*>(v0 + v47);
            int4* v49;
            v49 = reinterpret_cast<int4*>(v41 + v45);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v48) % 16 == 0 && reinterpret_cast<unsigned long long>(v49) % 16 == 0);
            *v49 = *v48;
            v43 += 1 ;
        }
        int v50;
        v50 = 0;
        while (while_method_3(v50)){
            int v52;
            v52 = 0;
            while (while_method_1(v52)){
                bool v54;
                v54 = 0 <= v52;
                bool v56;
                if (v54){
                    bool v55;
                    v55 = v52 < 4;
                    v56 = v55;
                } else {
                    v56 = false;
                }
                bool v57;
                v57 = v56 == false;
                if (v57){
                    assert("The indices should be inside the range of the dimension." && v56);
                } else {
                }
                bool v59;
                v59 = 0 <= v22;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v22 < 16;
                    v61 = v60;
                } else {
                    v61 = false;
                }
                bool v62;
                v62 = v61 == false;
                if (v62){
                    assert("The indices should be inside the range of the dimension." && v61);
                } else {
                }
                int v64;
                v64 = v22 * 4;
                int v65;
                v65 = v52 + v64;
                bool v66;
                v66 = 0 <= v50;
                bool v68;
                if (v66){
                    bool v67;
                    v67 = v50 < 1;
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
                v71 = v50 * 64;
                int v72;
                v72 = v65 + v71;
                assert("Tensor range check" && 0 <= v50 && v50 < 1);
                assert("Tensor range check" && 0 <= v52 && v52 < 4);
                int v73;
                v73 = 4 * v50;
                int v74;
                v74 = v73 + v52;
                v42[v74] = v72;
                v52 += 1 ;
            }
            v50 += 1 ;
        }
        bool v75;
        v75 = 0 <= v23;
        bool v76;
        v76 = v75 && v24;
        bool v77;
        v77 = v76 == false;
        if (v77){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v76);
        } else {
        }
        bool v79;
        v79 = v33 && v36;
        bool v80;
        v80 = v79 == false;
        if (v80){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v79);
        } else {
        }
        int v82;
        v82 = v31 * 16;
        int v83;
        v83 = v82 + v23;
        assert("Tensor range check" && 0 <= v31 && v31 < 8);
        int v84;
        v84 = 0;
        while (while_method_3(v84)){
            assert("Tensor range check" && 0 <= v84 && v84 < 1);
            int v86;
            v86 = 64 * v84;
            int v87;
            v87 = v86 + v40;
            assert("Tensor range check" && 0 <= v84 && v84 < 1);
            int v88;
            v88 = 4 * v84;
            int4* v89;
            v89 = reinterpret_cast<int4*>(v41 + v88);
            int4* v90;
            v90 = reinterpret_cast<int4*>(v2 + v87);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v89) % 16 == 0 && reinterpret_cast<unsigned long long>(v90) % 16 == 0);
            *v90 = *v89;
            v84 += 1 ;
        }
        v31 += 24 ;
    }
    v17.sync() ;
    int v91;
    v91 = threadIdx.x;
    bool v92;
    v92 = 0 <= v91;
    bool v93;
    v93 = v92 == false;
    if (v93){
        assert("The index needs to be zero or positive." && v92);
    } else {
    }
    int v95;
    v95 = v91 % 16;
    int v96;
    v96 = v91 / 16;
    bool v97;
    v97 = v96 < 16;
    bool v98;
    v98 = v97 == false;
    if (v98){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v97);
    } else {
    }
    assert("Tensor range check" && 0 <= v96 && v96 < 16);
    assert("Tensor range check" && 0 <= v95 && v95 < 16);
    int v100;
    v100 = 4 * v95;
    int v101;
    v101 = 64 * v96;
    int v102;
    v102 = v101 + v100;
    assert("Tensor range check" && 0 <= v96 && v96 < 16);
    assert("Tensor range check" && 0 <= v95 && v95 < 16);
    int v103;
    v103 = blockIdx.x;
    int v104;
    v104 = v103;
    while (while_method_2(v104)){
        bool v106;
        v106 = 0 <= v104;
        bool v107;
        v107 = v106 == false;
        if (v107){
            assert("The index needs to be zero or positive." && v106);
        } else {
        }
        bool v109;
        v109 = v104 < 8;
        bool v110;
        v110 = v109 == false;
        if (v110){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v109);
        } else {
        }
        assert("Tensor range check" && 0 <= v104 && v104 < 8);
        int v112;
        v112 = 1024 * v104;
        int v113;
        v113 = v112 + v102;
        float v114[4];
        int v115[4];
        int v116;
        v116 = 0;
        while (while_method_3(v116)){
            assert("Tensor range check" && 0 <= v116 && v116 < 1);
            int v118;
            v118 = 4 * v116;
            assert("Tensor range check" && 0 <= v116 && v116 < 1);
            int v119;
            v119 = 64 * v116;
            int v120;
            v120 = v119 + v113;
            int4* v121;
            v121 = reinterpret_cast<int4*>(v1 + v120);
            int4* v122;
            v122 = reinterpret_cast<int4*>(v114 + v118);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v121) % 16 == 0 && reinterpret_cast<unsigned long long>(v122) % 16 == 0);
            *v122 = *v121;
            v116 += 1 ;
        }
        int v123;
        v123 = 0;
        while (while_method_3(v123)){
            int v125;
            v125 = 0;
            while (while_method_1(v125)){
                bool v127;
                v127 = 0 <= v125;
                bool v129;
                if (v127){
                    bool v128;
                    v128 = v125 < 4;
                    v129 = v128;
                } else {
                    v129 = false;
                }
                bool v130;
                v130 = v129 == false;
                if (v130){
                    assert("The indices should be inside the range of the dimension." && v129);
                } else {
                }
                bool v132;
                v132 = 0 <= v95;
                bool v134;
                if (v132){
                    bool v133;
                    v133 = v95 < 16;
                    v134 = v133;
                } else {
                    v134 = false;
                }
                bool v135;
                v135 = v134 == false;
                if (v135){
                    assert("The indices should be inside the range of the dimension." && v134);
                } else {
                }
                int v137;
                v137 = v95 * 4;
                int v138;
                v138 = v125 + v137;
                bool v139;
                v139 = 0 <= v123;
                bool v141;
                if (v139){
                    bool v140;
                    v140 = v123 < 1;
                    v141 = v140;
                } else {
                    v141 = false;
                }
                bool v142;
                v142 = v141 == false;
                if (v142){
                    assert("The indices should be inside the range of the dimension." && v141);
                } else {
                }
                int v144;
                v144 = v123 * 64;
                int v145;
                v145 = v138 + v144;
                assert("Tensor range check" && 0 <= v123 && v123 < 1);
                assert("Tensor range check" && 0 <= v125 && v125 < 4);
                int v146;
                v146 = 4 * v123;
                int v147;
                v147 = v146 + v125;
                v115[v147] = v145;
                v125 += 1 ;
            }
            v123 += 1 ;
        }
        bool v148;
        v148 = 0 <= v96;
        bool v149;
        v149 = v148 && v97;
        bool v150;
        v150 = v149 == false;
        if (v150){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v149);
        } else {
        }
        bool v152;
        v152 = v106 && v109;
        bool v153;
        v153 = v152 == false;
        if (v153){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v152);
        } else {
        }
        int v155;
        v155 = v104 * 16;
        int v156;
        v156 = v155 + v96;
        int v157[4];
        int v158[4];
        int v159;
        v159 = 0;
        while (while_method_3(v159)){
            int v161;
            v161 = 0;
            while (while_method_1(v161)){
                assert("Tensor range check" && 0 <= v159 && v159 < 1);
                assert("Tensor range check" && 0 <= v161 && v161 < 4);
                int v163;
                v163 = 4 * v159;
                int v164;
                v164 = v163 + v161;
                int v165;
                v165 = v115[v164];
                assert("Tensor range check" && 0 <= v159 && v159 < 1);
                assert("Tensor range check" && 0 <= v161 && v161 < 4);
                v157[v164] = v156;
                v158[v164] = v165;
                v161 += 1 ;
            }
            v159 += 1 ;
        }
        assert("Tensor range check" && 0 <= v104 && v104 < 8);
        int v166;
        v166 = 0;
        while (while_method_3(v166)){
            assert("Tensor range check" && 0 <= v166 && v166 < 1);
            int v168;
            v168 = 64 * v166;
            int v169;
            v169 = v168 + v113;
            assert("Tensor range check" && 0 <= v166 && v166 < 1);
            int v170;
            v170 = 4 * v166;
            int4* v171;
            v171 = reinterpret_cast<int4*>(v157 + v170);
            int4* v172;
            v172 = reinterpret_cast<int4*>(v9 + v169);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v171) % 16 == 0 && reinterpret_cast<unsigned long long>(v172) % 16 == 0);
            *v172 = *v171;
            int4* v173;
            v173 = reinterpret_cast<int4*>(v158 + v170);
            int4* v174;
            v174 = reinterpret_cast<int4*>(v10 + v169);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v173) % 16 == 0 && reinterpret_cast<unsigned long long>(v174) % 16 == 0);
            *v174 = *v173;
            v166 += 1 ;
        }
        v104 += 24 ;
    }
    v17.sync() ;
    int v175;
    v175 = threadIdx.x;
    bool v176;
    v176 = 0 <= v175;
    bool v177;
    v177 = v176 == false;
    if (v177){
        assert("The index needs to be zero or positive." && v176);
    } else {
    }
    int v179;
    v179 = v175 % 16;
    int v180;
    v180 = v175 / 16;
    bool v181;
    v181 = v180 < 16;
    bool v182;
    v182 = v181 == false;
    if (v182){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v181);
    } else {
    }
    assert("Tensor range check" && 0 <= v180 && v180 < 16);
    assert("Tensor range check" && 0 <= v179 && v179 < 16);
    int v184;
    v184 = 4 * v179;
    int v185;
    v185 = 64 * v180;
    int v186;
    v186 = v185 + v184;
    assert("Tensor range check" && 0 <= v180 && v180 < 16);
    int v187;
    v187 = blockIdx.x;
    int v188;
    v188 = v187;
    while (while_method_2(v188)){
        bool v190;
        v190 = 0 <= v188;
        bool v191;
        v191 = v190 == false;
        if (v191){
            assert("The index needs to be zero or positive." && v190);
        } else {
        }
        bool v193;
        v193 = v188 < 8;
        bool v194;
        v194 = v193 == false;
        if (v194){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v193);
        } else {
        }
        assert("Tensor range check" && 0 <= v188 && v188 < 8);
        int v196;
        v196 = 1024 * v188;
        int v197;
        v197 = v196 + v186;
        float v198[4];
        int v199[4];
        int v200;
        v200 = 0;
        while (while_method_3(v200)){
            assert("Tensor range check" && 0 <= v200 && v200 < 1);
            int v202;
            v202 = 4 * v200;
            assert("Tensor range check" && 0 <= v200 && v200 < 1);
            int v203;
            v203 = 64 * v200;
            int v204;
            v204 = v203 + v197;
            int4* v205;
            v205 = reinterpret_cast<int4*>(v1 + v204);
            int4* v206;
            v206 = reinterpret_cast<int4*>(v198 + v202);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v205) % 16 == 0 && reinterpret_cast<unsigned long long>(v206) % 16 == 0);
            *v206 = *v205;
            v200 += 1 ;
        }
        int v207;
        v207 = 0;
        while (while_method_3(v207)){
            int v209;
            v209 = 0;
            while (while_method_1(v209)){
                bool v211;
                v211 = 0 <= v209;
                bool v213;
                if (v211){
                    bool v212;
                    v212 = v209 < 4;
                    v213 = v212;
                } else {
                    v213 = false;
                }
                bool v214;
                v214 = v213 == false;
                if (v214){
                    assert("The indices should be inside the range of the dimension." && v213);
                } else {
                }
                bool v216;
                v216 = 0 <= v179;
                bool v218;
                if (v216){
                    bool v217;
                    v217 = v179 < 16;
                    v218 = v217;
                } else {
                    v218 = false;
                }
                bool v219;
                v219 = v218 == false;
                if (v219){
                    assert("The indices should be inside the range of the dimension." && v218);
                } else {
                }
                int v221;
                v221 = v179 * 4;
                int v222;
                v222 = v209 + v221;
                bool v223;
                v223 = 0 <= v207;
                bool v225;
                if (v223){
                    bool v224;
                    v224 = v207 < 1;
                    v225 = v224;
                } else {
                    v225 = false;
                }
                bool v226;
                v226 = v225 == false;
                if (v226){
                    assert("The indices should be inside the range of the dimension." && v225);
                } else {
                }
                int v228;
                v228 = v207 * 64;
                int v229;
                v229 = v222 + v228;
                assert("Tensor range check" && 0 <= v207 && v207 < 1);
                assert("Tensor range check" && 0 <= v209 && v209 < 4);
                int v230;
                v230 = 4 * v207;
                int v231;
                v231 = v230 + v209;
                v199[v231] = v229;
                v209 += 1 ;
            }
            v207 += 1 ;
        }
        bool v232;
        v232 = 0 <= v180;
        bool v233;
        v233 = v232 && v181;
        bool v234;
        v234 = v233 == false;
        if (v234){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v233);
        } else {
        }
        bool v236;
        v236 = v190 && v193;
        bool v237;
        v237 = v236 == false;
        if (v237){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v236);
        } else {
        }
        int v239;
        v239 = v188 * 16;
        int v240;
        v240 = v239 + v180;
        assert("Tensor range check" && 0 <= v188 && v188 < 8);
        int v241;
        v241 = 16 * v188;
        int v242;
        v242 = v241 + v180;
        v11[v242] = v240;
        v188 += 24 ;
    }
    v17.sync() ;
    int v243;
    v243 = threadIdx.x;
    bool v244;
    v244 = 0 <= v243;
    bool v245;
    v245 = v244 == false;
    if (v245){
        assert("The index needs to be zero or positive." && v244);
    } else {
    }
    int v247;
    v247 = v243 % 16;
    int v248;
    v248 = v243 / 16;
    bool v249;
    v249 = v248 < 16;
    bool v250;
    v250 = v249 == false;
    if (v250){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v249);
    } else {
    }
    assert("Tensor range check" && 0 <= v248 && v248 < 16);
    assert("Tensor range check" && 0 <= v247 && v247 < 16);
    int v252;
    v252 = 4 * v247;
    int v253;
    v253 = 64 * v248;
    int v254;
    v254 = v253 + v252;
    assert("Tensor range check" && 0 <= v248 && v248 < 16);
    assert("Tensor range check" && 0 <= v247 && v247 < 16);
    int v255;
    v255 = blockIdx.x;
    int v256;
    v256 = v255;
    while (while_method_2(v256)){
        bool v258;
        v258 = 0 <= v256;
        bool v259;
        v259 = v258 == false;
        if (v259){
            assert("The index needs to be zero or positive." && v258);
        } else {
        }
        bool v261;
        v261 = v256 < 8;
        bool v262;
        v262 = v261 == false;
        if (v262){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v261);
        } else {
        }
        assert("Tensor range check" && 0 <= v256 && v256 < 8);
        int v264;
        v264 = 1024 * v256;
        int v265;
        v265 = v264 + v254;
        float v266[4];
        int v267[4];
        int v268;
        v268 = 0;
        while (while_method_3(v268)){
            assert("Tensor range check" && 0 <= v268 && v268 < 1);
            int v270;
            v270 = 4 * v268;
            assert("Tensor range check" && 0 <= v268 && v268 < 1);
            int v271;
            v271 = 64 * v268;
            int v272;
            v272 = v271 + v265;
            int4* v273;
            v273 = reinterpret_cast<int4*>(v1 + v272);
            int4* v274;
            v274 = reinterpret_cast<int4*>(v266 + v270);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v273) % 16 == 0 && reinterpret_cast<unsigned long long>(v274) % 16 == 0);
            *v274 = *v273;
            v268 += 1 ;
        }
        int v275;
        v275 = 0;
        while (while_method_3(v275)){
            int v277;
            v277 = 0;
            while (while_method_1(v277)){
                bool v279;
                v279 = 0 <= v277;
                bool v281;
                if (v279){
                    bool v280;
                    v280 = v277 < 4;
                    v281 = v280;
                } else {
                    v281 = false;
                }
                bool v282;
                v282 = v281 == false;
                if (v282){
                    assert("The indices should be inside the range of the dimension." && v281);
                } else {
                }
                bool v284;
                v284 = 0 <= v247;
                bool v286;
                if (v284){
                    bool v285;
                    v285 = v247 < 16;
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
                int v289;
                v289 = v247 * 4;
                int v290;
                v290 = v277 + v289;
                bool v291;
                v291 = 0 <= v275;
                bool v293;
                if (v291){
                    bool v292;
                    v292 = v275 < 1;
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
                int v296;
                v296 = v275 * 64;
                int v297;
                v297 = v290 + v296;
                assert("Tensor range check" && 0 <= v275 && v275 < 1);
                assert("Tensor range check" && 0 <= v277 && v277 < 4);
                int v298;
                v298 = 4 * v275;
                int v299;
                v299 = v298 + v277;
                v267[v299] = v297;
                v277 += 1 ;
            }
            v275 += 1 ;
        }
        bool v300;
        v300 = 0 <= v248;
        bool v301;
        v301 = v300 && v249;
        bool v302;
        v302 = v301 == false;
        if (v302){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v301);
        } else {
        }
        bool v304;
        v304 = v258 && v261;
        bool v305;
        v305 = v304 == false;
        if (v305){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v304);
        } else {
        }
        int v307;
        v307 = v256 * 16;
        int v308;
        v308 = v307 + v248;
        float v309;
        v309 = 0.0f;
        int v310;
        v310 = 0;
        while (while_method_3(v310)){
            int v312;
            v312 = 0;
            while (while_method_1(v312)){
                assert("Tensor range check" && 0 <= v310 && v310 < 1);
                assert("Tensor range check" && 0 <= v312 && v312 < 4);
                int v314;
                v314 = 4 * v310;
                int v315;
                v315 = v314 + v312;
                float v316;
                v316 = v266[v315];
                float v317;
                v317 = v309 + v316;
                v309 = v317;
                v312 += 1 ;
            }
            v310 += 1 ;
        }
        auto v318 = cooperative_groups::coalesced_threads();
        int v319;
        v319 = threadIdx.x;
        int v320;
        v320 = v319 / 16;
        auto v321 = cooperative_groups::labeled_partition(v318,v320);
        Closure0 v322{};
        float v323;
        v323 = cooperative_groups::reduce(v321, v309, v322);
        float v324;
        v324 = v323 / 64.0f;
        float v325[4];
        int v326;
        v326 = 0;
        while (while_method_3(v326)){
            int v328;
            v328 = 0;
            while (while_method_1(v328)){
                assert("Tensor range check" && 0 <= v326 && v326 < 1);
                assert("Tensor range check" && 0 <= v328 && v328 < 4);
                int v330;
                v330 = 4 * v326;
                int v331;
                v331 = v330 + v328;
                float v332;
                v332 = v266[v331];
                float v333;
                v333 = v332 - v324;
                float v334;
                v334 = exp(v333);
                assert("Tensor range check" && 0 <= v326 && v326 < 1);
                assert("Tensor range check" && 0 <= v328 && v328 < 4);
                v325[v331] = v334;
                v328 += 1 ;
            }
            v326 += 1 ;
        }
        float v335;
        v335 = 0.0f;
        int v336;
        v336 = 0;
        while (while_method_3(v336)){
            int v338;
            v338 = 0;
            while (while_method_1(v338)){
                assert("Tensor range check" && 0 <= v336 && v336 < 1);
                assert("Tensor range check" && 0 <= v338 && v338 < 4);
                int v340;
                v340 = 4 * v336;
                int v341;
                v341 = v340 + v338;
                float v342;
                v342 = v325[v341];
                float v343;
                v343 = v335 + v342;
                v335 = v343;
                v338 += 1 ;
            }
            v336 += 1 ;
        }
        auto v344 = cooperative_groups::coalesced_threads();
        int v345;
        v345 = threadIdx.x;
        int v346;
        v346 = v345 / 16;
        auto v347 = cooperative_groups::labeled_partition(v344,v346);
        float v348;
        v348 = cooperative_groups::reduce(v347, v335, v322);
        float v349[4];
        int v350;
        v350 = 0;
        while (while_method_3(v350)){
            int v352;
            v352 = 0;
            while (while_method_1(v352)){
                assert("Tensor range check" && 0 <= v350 && v350 < 1);
                assert("Tensor range check" && 0 <= v352 && v352 < 4);
                int v354;
                v354 = 4 * v350;
                int v355;
                v355 = v354 + v352;
                float v356;
                v356 = v325[v355];
                float v357;
                v357 = v356 / v348;
                assert("Tensor range check" && 0 <= v350 && v350 < 1);
                assert("Tensor range check" && 0 <= v352 && v352 < 4);
                v349[v355] = v357;
                v352 += 1 ;
            }
            v350 += 1 ;
        }
        assert("Tensor range check" && 0 <= v256 && v256 < 8);
        int v358;
        v358 = 0;
        while (while_method_3(v358)){
            assert("Tensor range check" && 0 <= v358 && v358 < 1);
            int v360;
            v360 = 64 * v358;
            int v361;
            v361 = v360 + v265;
            assert("Tensor range check" && 0 <= v358 && v358 < 1);
            int v362;
            v362 = 4 * v358;
            int4* v363;
            v363 = reinterpret_cast<int4*>(v349 + v362);
            int4* v364;
            v364 = reinterpret_cast<int4*>(v3 + v361);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v363) % 16 == 0 && reinterpret_cast<unsigned long long>(v364) % 16 == 0);
            *v364 = *v363;
            v358 += 1 ;
        }
        v256 += 24 ;
    }
    v17.sync() ;
    int v365;
    v365 = threadIdx.x;
    bool v366;
    v366 = 0 <= v365;
    bool v367;
    v367 = v366 == false;
    if (v367){
        assert("The index needs to be zero or positive." && v366);
    } else {
    }
    int v369;
    v369 = v365 % 16;
    int v370;
    v370 = v365 / 16;
    bool v371;
    v371 = v370 < 16;
    bool v372;
    v372 = v371 == false;
    if (v372){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v371);
    } else {
    }
    assert("Tensor range check" && 0 <= v370 && v370 < 16);
    assert("Tensor range check" && 0 <= v369 && v369 < 16);
    int v374;
    v374 = 4 * v369;
    int v375;
    v375 = 64 * v370;
    int v376;
    v376 = v375 + v374;
    assert("Tensor range check" && 0 <= v370 && v370 < 16);
    assert("Tensor range check" && 0 <= v369 && v369 < 16);
    int v377;
    v377 = blockIdx.x;
    int v378;
    v378 = v377;
    while (while_method_2(v378)){
        bool v380;
        v380 = 0 <= v378;
        bool v381;
        v381 = v380 == false;
        if (v381){
            assert("The index needs to be zero or positive." && v380);
        } else {
        }
        bool v383;
        v383 = v378 < 8;
        bool v384;
        v384 = v383 == false;
        if (v384){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v383);
        } else {
        }
        assert("Tensor range check" && 0 <= v378 && v378 < 8);
        int v386;
        v386 = 1024 * v378;
        int v387;
        v387 = v386 + v376;
        float v388[4];
        int v389[4];
        int v390;
        v390 = 0;
        while (while_method_3(v390)){
            assert("Tensor range check" && 0 <= v390 && v390 < 1);
            int v392;
            v392 = 4 * v390;
            assert("Tensor range check" && 0 <= v390 && v390 < 1);
            int v393;
            v393 = 64 * v390;
            int v394;
            v394 = v393 + v387;
            int4* v395;
            v395 = reinterpret_cast<int4*>(v1 + v394);
            int4* v396;
            v396 = reinterpret_cast<int4*>(v388 + v392);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v395) % 16 == 0 && reinterpret_cast<unsigned long long>(v396) % 16 == 0);
            *v396 = *v395;
            v390 += 1 ;
        }
        int v397;
        v397 = 0;
        while (while_method_3(v397)){
            int v399;
            v399 = 0;
            while (while_method_1(v399)){
                bool v401;
                v401 = 0 <= v399;
                bool v403;
                if (v401){
                    bool v402;
                    v402 = v399 < 4;
                    v403 = v402;
                } else {
                    v403 = false;
                }
                bool v404;
                v404 = v403 == false;
                if (v404){
                    assert("The indices should be inside the range of the dimension." && v403);
                } else {
                }
                bool v406;
                v406 = 0 <= v369;
                bool v408;
                if (v406){
                    bool v407;
                    v407 = v369 < 16;
                    v408 = v407;
                } else {
                    v408 = false;
                }
                bool v409;
                v409 = v408 == false;
                if (v409){
                    assert("The indices should be inside the range of the dimension." && v408);
                } else {
                }
                int v411;
                v411 = v369 * 4;
                int v412;
                v412 = v399 + v411;
                bool v413;
                v413 = 0 <= v397;
                bool v415;
                if (v413){
                    bool v414;
                    v414 = v397 < 1;
                    v415 = v414;
                } else {
                    v415 = false;
                }
                bool v416;
                v416 = v415 == false;
                if (v416){
                    assert("The indices should be inside the range of the dimension." && v415);
                } else {
                }
                int v418;
                v418 = v397 * 64;
                int v419;
                v419 = v412 + v418;
                assert("Tensor range check" && 0 <= v397 && v397 < 1);
                assert("Tensor range check" && 0 <= v399 && v399 < 4);
                int v420;
                v420 = 4 * v397;
                int v421;
                v421 = v420 + v399;
                v389[v421] = v419;
                v399 += 1 ;
            }
            v397 += 1 ;
        }
        bool v422;
        v422 = 0 <= v370;
        bool v423;
        v423 = v422 && v371;
        bool v424;
        v424 = v423 == false;
        if (v424){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v423);
        } else {
        }
        bool v426;
        v426 = v380 && v383;
        bool v427;
        v427 = v426 == false;
        if (v427){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v426);
        } else {
        }
        int v429;
        v429 = v378 * 16;
        int v430;
        v430 = v429 + v370;
        float v431[4];
        int v432;
        v432 = 0;
        while (while_method_3(v432)){
            int v434;
            v434 = 0;
            while (while_method_1(v434)){
                assert("Tensor range check" && 0 <= v432 && v432 < 1);
                assert("Tensor range check" && 0 <= v434 && v434 < 4);
                int v436;
                v436 = 4 * v432;
                int v437;
                v437 = v436 + v434;
                float v438;
                v438 = v388[v437];
                float v439;
                v439 = v438 * v438;
                assert("Tensor range check" && 0 <= v432 && v432 < 1);
                assert("Tensor range check" && 0 <= v434 && v434 < 4);
                v431[v437] = v439;
                v434 += 1 ;
            }
            v432 += 1 ;
        }
        float v440;
        v440 = 0.0f;
        int v441;
        v441 = 0;
        while (while_method_3(v441)){
            int v443;
            v443 = 0;
            while (while_method_1(v443)){
                assert("Tensor range check" && 0 <= v441 && v441 < 1);
                assert("Tensor range check" && 0 <= v443 && v443 < 4);
                int v445;
                v445 = 4 * v441;
                int v446;
                v446 = v445 + v443;
                float v447;
                v447 = v431[v446];
                float v448;
                v448 = v440 + v447;
                v440 = v448;
                v443 += 1 ;
            }
            v441 += 1 ;
        }
        auto v449 = cooperative_groups::coalesced_threads();
        int v450;
        v450 = threadIdx.x;
        int v451;
        v451 = v450 / 16;
        auto v452 = cooperative_groups::labeled_partition(v449,v451);
        Closure0 v453{};
        float v454;
        v454 = cooperative_groups::reduce(v452, v440, v453);
        float v455[4];
        int v456;
        v456 = 0;
        while (while_method_3(v456)){
            int v458;
            v458 = 0;
            while (while_method_1(v458)){
                assert("Tensor range check" && 0 <= v456 && v456 < 1);
                assert("Tensor range check" && 0 <= v458 && v458 < 4);
                int v460;
                v460 = 4 * v456;
                int v461;
                v461 = v460 + v458;
                float v462;
                v462 = v388[v461];
                bool v463;
                v463 = v454 == 0.0f;
                bool v464;
                v464 = v463 != true;
                float v466;
                if (v464){
                    float v465;
                    v465 = v462 / v454;
                    v466 = v465;
                } else {
                    v466 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v456 && v456 < 1);
                assert("Tensor range check" && 0 <= v458 && v458 < 4);
                v455[v461] = v466;
                v458 += 1 ;
            }
            v456 += 1 ;
        }
        assert("Tensor range check" && 0 <= v378 && v378 < 8);
        int v467;
        v467 = 0;
        while (while_method_3(v467)){
            assert("Tensor range check" && 0 <= v467 && v467 < 1);
            int v469;
            v469 = 64 * v467;
            int v470;
            v470 = v469 + v387;
            assert("Tensor range check" && 0 <= v467 && v467 < 1);
            int v471;
            v471 = 4 * v467;
            int4* v472;
            v472 = reinterpret_cast<int4*>(v455 + v471);
            int4* v473;
            v473 = reinterpret_cast<int4*>(v7 + v470);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v472) % 16 == 0 && reinterpret_cast<unsigned long long>(v473) % 16 == 0);
            *v473 = *v472;
            v467 += 1 ;
        }
        v378 += 24 ;
    }
    v17.sync() ;
    int v474;
    v474 = threadIdx.x;
    bool v475;
    v475 = 0 <= v474;
    bool v476;
    v476 = v475 == false;
    if (v476){
        assert("The index needs to be zero or positive." && v475);
    } else {
    }
    int v478;
    v478 = v474 % 16;
    int v479;
    v479 = v474 / 16;
    bool v480;
    v480 = v479 < 16;
    bool v481;
    v481 = v480 == false;
    if (v481){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v480);
    } else {
    }
    assert("Tensor range check" && 0 <= v479 && v479 < 16);
    assert("Tensor range check" && 0 <= v478 && v478 < 16);
    int v483;
    v483 = 4 * v478;
    int v484;
    v484 = 64 * v479;
    int v485;
    v485 = v484 + v483;
    assert("Tensor range check" && 0 <= v479 && v479 < 16);
    int v486;
    v486 = blockIdx.x;
    int v487;
    v487 = v486;
    while (while_method_2(v487)){
        bool v489;
        v489 = 0 <= v487;
        bool v490;
        v490 = v489 == false;
        if (v490){
            assert("The index needs to be zero or positive." && v489);
        } else {
        }
        bool v492;
        v492 = v487 < 8;
        bool v493;
        v493 = v492 == false;
        if (v493){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v492);
        } else {
        }
        assert("Tensor range check" && 0 <= v487 && v487 < 8);
        int v495;
        v495 = 1024 * v487;
        int v496;
        v496 = v495 + v485;
        float v497[4];
        int v498[4];
        int v499;
        v499 = 0;
        while (while_method_3(v499)){
            assert("Tensor range check" && 0 <= v499 && v499 < 1);
            int v501;
            v501 = 4 * v499;
            assert("Tensor range check" && 0 <= v499 && v499 < 1);
            int v502;
            v502 = 64 * v499;
            int v503;
            v503 = v502 + v496;
            int4* v504;
            v504 = reinterpret_cast<int4*>(v1 + v503);
            int4* v505;
            v505 = reinterpret_cast<int4*>(v497 + v501);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v504) % 16 == 0 && reinterpret_cast<unsigned long long>(v505) % 16 == 0);
            *v505 = *v504;
            v499 += 1 ;
        }
        int v506;
        v506 = 0;
        while (while_method_3(v506)){
            int v508;
            v508 = 0;
            while (while_method_1(v508)){
                bool v510;
                v510 = 0 <= v508;
                bool v512;
                if (v510){
                    bool v511;
                    v511 = v508 < 4;
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
                bool v515;
                v515 = 0 <= v478;
                bool v517;
                if (v515){
                    bool v516;
                    v516 = v478 < 16;
                    v517 = v516;
                } else {
                    v517 = false;
                }
                bool v518;
                v518 = v517 == false;
                if (v518){
                    assert("The indices should be inside the range of the dimension." && v517);
                } else {
                }
                int v520;
                v520 = v478 * 4;
                int v521;
                v521 = v508 + v520;
                bool v522;
                v522 = 0 <= v506;
                bool v524;
                if (v522){
                    bool v523;
                    v523 = v506 < 1;
                    v524 = v523;
                } else {
                    v524 = false;
                }
                bool v525;
                v525 = v524 == false;
                if (v525){
                    assert("The indices should be inside the range of the dimension." && v524);
                } else {
                }
                int v527;
                v527 = v506 * 64;
                int v528;
                v528 = v521 + v527;
                assert("Tensor range check" && 0 <= v506 && v506 < 1);
                assert("Tensor range check" && 0 <= v508 && v508 < 4);
                int v529;
                v529 = 4 * v506;
                int v530;
                v530 = v529 + v508;
                v498[v530] = v528;
                v508 += 1 ;
            }
            v506 += 1 ;
        }
        bool v531;
        v531 = 0 <= v479;
        bool v532;
        v532 = v531 && v480;
        bool v533;
        v533 = v532 == false;
        if (v533){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v532);
        } else {
        }
        bool v535;
        v535 = v489 && v492;
        bool v536;
        v536 = v535 == false;
        if (v536){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v535);
        } else {
        }
        int v538;
        v538 = v487 * 16;
        int v539;
        v539 = v538 + v479;
        float v540; int v541;
        Tuple1 tmp45 = Tuple1{-1.0f / 0.0f, 0};
        v540 = tmp45.v0; v541 = tmp45.v1;
        int v542;
        v542 = 0;
        while (while_method_3(v542)){
            int v544;
            v544 = 0;
            while (while_method_1(v544)){
                assert("Tensor range check" && 0 <= v542 && v542 < 1);
                assert("Tensor range check" && 0 <= v544 && v544 < 4);
                int v546;
                v546 = 4 * v542;
                int v547;
                v547 = v546 + v544;
                float v548;
                v548 = v497[v547];
                int v549;
                v549 = v498[v547];
                bool v550;
                v550 = v540 > v548;
                float v551; int v552;
                if (v550){
                    v551 = v540; v552 = v541;
                } else {
                    v551 = v548; v552 = v549;
                }
                v540 = v551;
                v541 = v552;
                v544 += 1 ;
            }
            v542 += 1 ;
        }
        auto v553 = cooperative_groups::coalesced_threads();
        int v554;
        v554 = threadIdx.x;
        int v555;
        v555 = v554 / 16;
        auto v556 = cooperative_groups::labeled_partition(v553,v555);
        Closure1 v557{};
        float v558; int v559;
        Tuple1 tmp46 = cooperative_groups::reduce(v556, Tuple1{v540, v541}, v557);
        v558 = tmp46.v0; v559 = tmp46.v1;
        assert("Tensor range check" && 0 <= v487 && v487 < 8);
        int v560;
        v560 = 16 * v487;
        int v561;
        v561 = v560 + v479;
        v8[v561] = v559;
        v487 += 24 ;
    }
    v17.sync() ;
    int v562;
    v562 = threadIdx.x;
    bool v563;
    v563 = 0 <= v562;
    bool v564;
    v564 = v563 == false;
    if (v564){
        assert("The index needs to be zero or positive." && v563);
    } else {
    }
    int v566;
    v566 = v562 % 16;
    int v567;
    v567 = v562 / 16;
    bool v568;
    v568 = v567 < 16;
    bool v569;
    v569 = v568 == false;
    if (v569){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v568);
    } else {
    }
    assert("Tensor range check" && 0 <= v567 && v567 < 16);
    assert("Tensor range check" && 0 <= v566 && v566 < 16);
    int v571;
    v571 = 4 * v566;
    int v572;
    v572 = 64 * v567;
    int v573;
    v573 = v572 + v571;
    assert("Tensor range check" && 0 <= v567 && v567 < 16);
    assert("Tensor range check" && 0 <= v566 && v566 < 16);
    int v574;
    v574 = blockIdx.x;
    int v575;
    v575 = v574;
    while (while_method_2(v575)){
        bool v577;
        v577 = 0 <= v575;
        bool v578;
        v578 = v577 == false;
        if (v578){
            assert("The index needs to be zero or positive." && v577);
        } else {
        }
        bool v580;
        v580 = v575 < 8;
        bool v581;
        v581 = v580 == false;
        if (v581){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v580);
        } else {
        }
        assert("Tensor range check" && 0 <= v575 && v575 < 8);
        int v583;
        v583 = 1024 * v575;
        int v584;
        v584 = v583 + v573;
        float v585[4];
        int v586[4];
        int v587;
        v587 = 0;
        while (while_method_3(v587)){
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v589;
            v589 = 4 * v587;
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v590;
            v590 = 64 * v587;
            int v591;
            v591 = v590 + v584;
            int4* v592;
            v592 = reinterpret_cast<int4*>(v1 + v591);
            int4* v593;
            v593 = reinterpret_cast<int4*>(v585 + v589);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v592) % 16 == 0 && reinterpret_cast<unsigned long long>(v593) % 16 == 0);
            *v593 = *v592;
            v587 += 1 ;
        }
        int v594;
        v594 = 0;
        while (while_method_3(v594)){
            int v596;
            v596 = 0;
            while (while_method_1(v596)){
                bool v598;
                v598 = 0 <= v596;
                bool v600;
                if (v598){
                    bool v599;
                    v599 = v596 < 4;
                    v600 = v599;
                } else {
                    v600 = false;
                }
                bool v601;
                v601 = v600 == false;
                if (v601){
                    assert("The indices should be inside the range of the dimension." && v600);
                } else {
                }
                bool v603;
                v603 = 0 <= v566;
                bool v605;
                if (v603){
                    bool v604;
                    v604 = v566 < 16;
                    v605 = v604;
                } else {
                    v605 = false;
                }
                bool v606;
                v606 = v605 == false;
                if (v606){
                    assert("The indices should be inside the range of the dimension." && v605);
                } else {
                }
                int v608;
                v608 = v566 * 4;
                int v609;
                v609 = v596 + v608;
                bool v610;
                v610 = 0 <= v594;
                bool v612;
                if (v610){
                    bool v611;
                    v611 = v594 < 1;
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
                v615 = v594 * 64;
                int v616;
                v616 = v609 + v615;
                assert("Tensor range check" && 0 <= v594 && v594 < 1);
                assert("Tensor range check" && 0 <= v596 && v596 < 4);
                int v617;
                v617 = 4 * v594;
                int v618;
                v618 = v617 + v596;
                v586[v618] = v616;
                v596 += 1 ;
            }
            v594 += 1 ;
        }
        bool v619;
        v619 = 0 <= v567;
        bool v620;
        v620 = v619 && v568;
        bool v621;
        v621 = v620 == false;
        if (v621){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v620);
        } else {
        }
        bool v623;
        v623 = v577 && v580;
        bool v624;
        v624 = v623 == false;
        if (v624){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v623);
        } else {
        }
        int v626;
        v626 = v575 * 16;
        int v627;
        v627 = v626 + v567;
        float v628;
        v628 = 0.0f;
        int v629;
        v629 = 0;
        while (while_method_3(v629)){
            int v631;
            v631 = 0;
            while (while_method_1(v631)){
                assert("Tensor range check" && 0 <= v629 && v629 < 1);
                assert("Tensor range check" && 0 <= v631 && v631 < 4);
                int v633;
                v633 = 4 * v629;
                int v634;
                v634 = v633 + v631;
                float v635;
                v635 = v585[v634];
                float v636;
                v636 = v628 + v635;
                v628 = v636;
                v631 += 1 ;
            }
            v629 += 1 ;
        }
        auto v637 = cooperative_groups::coalesced_threads();
        int v638;
        v638 = threadIdx.x;
        int v639;
        v639 = v638 / 16;
        auto v640 = cooperative_groups::labeled_partition(v637,v639);
        Closure0 v641{};
        float v642;
        v642 = cooperative_groups::reduce(v640, v628, v641);
        float v643;
        v643 = v642 / 64.0f;
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
                v651 = v585[v650];
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
        v665 = v664 / 16;
        auto v666 = cooperative_groups::labeled_partition(v663,v665);
        float v667;
        v667 = cooperative_groups::reduce(v666, v654, v641);
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
            v690 = v689 / 16;
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
        assert("Tensor range check" && 0 <= v575 && v575 < 8);
        int v706;
        v706 = 0;
        while (while_method_3(v706)){
            assert("Tensor range check" && 0 <= v706 && v706 < 1);
            int v708;
            v708 = 64 * v706;
            int v709;
            v709 = v708 + v584;
            assert("Tensor range check" && 0 <= v706 && v706 < 1);
            int v710;
            v710 = 4 * v706;
            int4* v711;
            v711 = reinterpret_cast<int4*>(v668 + v710);
            int4* v712;
            v712 = reinterpret_cast<int4*>(v5 + v709);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v711) % 16 == 0 && reinterpret_cast<unsigned long long>(v712) % 16 == 0);
            *v712 = *v711;
            int4* v713;
            v713 = reinterpret_cast<int4*>(v677 + v710);
            int4* v714;
            v714 = reinterpret_cast<int4*>(v6 + v709);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v713) % 16 == 0 && reinterpret_cast<unsigned long long>(v714) % 16 == 0);
            *v714 = *v713;
            v706 += 1 ;
        }
        v575 += 24 ;
    }
    v17.sync() ;
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
    v719 = v715 % 16;
    int v720;
    v720 = v715 / 16;
    bool v721;
    v721 = v720 < 16;
    bool v722;
    v722 = v721 == false;
    if (v722){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v721);
    } else {
    }
    assert("Tensor range check" && 0 <= v720 && v720 < 16);
    assert("Tensor range check" && 0 <= v719 && v719 < 16);
    int v724;
    v724 = 4 * v719;
    int v725;
    v725 = 64 * v720;
    int v726;
    v726 = v725 + v724;
    assert("Tensor range check" && 0 <= v720 && v720 < 16);
    assert("Tensor range check" && 0 <= v719 && v719 < 16);
    int v727;
    v727 = blockIdx.x;
    int v728;
    v728 = v727;
    while (while_method_2(v728)){
        bool v730;
        v730 = 0 <= v728;
        bool v731;
        v731 = v730 == false;
        if (v731){
            assert("The index needs to be zero or positive." && v730);
        } else {
        }
        bool v733;
        v733 = v728 < 8;
        bool v734;
        v734 = v733 == false;
        if (v734){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v733);
        } else {
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v736;
        v736 = 1024 * v728;
        int v737;
        v737 = v736 + v726;
        int v738[4];
        int v739[4];
        int v740;
        v740 = 0;
        while (while_method_3(v740)){
            assert("Tensor range check" && 0 <= v740 && v740 < 1);
            int v742;
            v742 = 4 * v740;
            assert("Tensor range check" && 0 <= v740 && v740 < 1);
            int v743;
            v743 = 64 * v740;
            int v744;
            v744 = v743 + v737;
            int4* v745;
            v745 = reinterpret_cast<int4*>(v0 + v744);
            int4* v746;
            v746 = reinterpret_cast<int4*>(v738 + v742);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v745) % 16 == 0 && reinterpret_cast<unsigned long long>(v746) % 16 == 0);
            *v746 = *v745;
            v740 += 1 ;
        }
        int v747;
        v747 = 0;
        while (while_method_3(v747)){
            int v749;
            v749 = 0;
            while (while_method_1(v749)){
                bool v751;
                v751 = 0 <= v749;
                bool v753;
                if (v751){
                    bool v752;
                    v752 = v749 < 4;
                    v753 = v752;
                } else {
                    v753 = false;
                }
                bool v754;
                v754 = v753 == false;
                if (v754){
                    assert("The indices should be inside the range of the dimension." && v753);
                } else {
                }
                bool v756;
                v756 = 0 <= v719;
                bool v758;
                if (v756){
                    bool v757;
                    v757 = v719 < 16;
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
                v761 = v719 * 4;
                int v762;
                v762 = v749 + v761;
                bool v763;
                v763 = 0 <= v747;
                bool v765;
                if (v763){
                    bool v764;
                    v764 = v747 < 1;
                    v765 = v764;
                } else {
                    v765 = false;
                }
                bool v766;
                v766 = v765 == false;
                if (v766){
                    assert("The indices should be inside the range of the dimension." && v765);
                } else {
                }
                int v768;
                v768 = v747 * 64;
                int v769;
                v769 = v762 + v768;
                assert("Tensor range check" && 0 <= v747 && v747 < 1);
                assert("Tensor range check" && 0 <= v749 && v749 < 4);
                int v770;
                v770 = 4 * v747;
                int v771;
                v771 = v770 + v749;
                v739[v771] = v769;
                v749 += 1 ;
            }
            v747 += 1 ;
        }
        bool v772;
        v772 = 0 <= v720;
        bool v773;
        v773 = v772 && v721;
        bool v774;
        v774 = v773 == false;
        if (v774){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v773);
        } else {
        }
        bool v776;
        v776 = v730 && v733;
        bool v777;
        v777 = v776 == false;
        if (v777){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v776);
        } else {
        }
        int v779;
        v779 = v728 * 16;
        int v780;
        v780 = v779 + v720;
        int v781[4];
        int v782;
        v782 = 0;
        int v783;
        v783 = 0;
        while (while_method_3(v783)){
            assert("Tensor range check" && 0 <= v783 && v783 < 1);
            int v785;
            v785 = 4 * v783;
            assert("Tensor range check" && 0 <= v783 && v783 < 1);
            int v786;
            v786 = 0;
            int v787;
            v787 = 0;
            while (while_method_1(v787)){
                assert("Tensor range check" && 0 <= v787 && v787 < 4);
                int v789;
                v789 = v787 + v785;
                int v790;
                v790 = v738[v789];
                int v791;
                v791 = v786 + v790;
                v786 = v791;
                v787 += 1 ;
            }
            auto v792 = cooperative_groups::coalesced_threads();
            int v793;
            v793 = threadIdx.x;
            int v794;
            v794 = v793 / 16;
            auto v795 = cooperative_groups::labeled_partition(v792,v794);
            Closure3 v796{};
            int v797;
            v797 = cooperative_groups::inclusive_scan(v795, v786, v796);
            int v798;
            v798 = v795.shfl_up(v797,1);
            bool v799;
            v799 = v795.thread_rank() == 0;
            int v800;
            if (v799){
                v800 = 0;
            } else {
                v800 = v798;
            }
            int v801;
            v801 = v795.shfl(v797,v795.num_threads()-1);
            int v802;
            v802 = v782 + v800;
            int v803;
            v803 = v802;
            int v804;
            v804 = 0;
            while (while_method_1(v804)){
                assert("Tensor range check" && 0 <= v804 && v804 < 4);
                int v806;
                v806 = v804 + v785;
                int v807;
                v807 = v738[v806];
                assert("Tensor range check" && 0 <= v804 && v804 < 4);
                v781[v806] = v803;
                int v808;
                v808 = v803 + v807;
                v803 = v808;
                v804 += 1 ;
            }
            int v809;
            v809 = v782 + v801;
            v782 = v809;
            v783 += 1 ;
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v810;
        v810 = 0;
        while (while_method_3(v810)){
            assert("Tensor range check" && 0 <= v810 && v810 < 1);
            int v812;
            v812 = 64 * v810;
            int v813;
            v813 = v812 + v737;
            assert("Tensor range check" && 0 <= v810 && v810 < 1);
            int v814;
            v814 = 4 * v810;
            int4* v815;
            v815 = reinterpret_cast<int4*>(v781 + v814);
            int4* v816;
            v816 = reinterpret_cast<int4*>(v12 + v813);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v815) % 16 == 0 && reinterpret_cast<unsigned long long>(v816) % 16 == 0);
            *v816 = *v815;
            v810 += 1 ;
        }
        v728 += 24 ;
    }
    v17.sync() ;
    int v817;
    v817 = threadIdx.x;
    bool v818;
    v818 = 0 <= v817;
    bool v819;
    v819 = v818 == false;
    if (v819){
        assert("The index needs to be zero or positive." && v818);
    } else {
    }
    int v821;
    v821 = v817 % 16;
    int v822;
    v822 = v817 / 16;
    bool v823;
    v823 = v822 < 16;
    bool v824;
    v824 = v823 == false;
    if (v824){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v823);
    } else {
    }
    assert("Tensor range check" && 0 <= v822 && v822 < 16);
    assert("Tensor range check" && 0 <= v821 && v821 < 16);
    int v826;
    v826 = 4 * v821;
    int v827;
    v827 = 64 * v822;
    int v828;
    v828 = v827 + v826;
    assert("Tensor range check" && 0 <= v822 && v822 < 16);
    assert("Tensor range check" && 0 <= v821 && v821 < 16);
    int v829;
    v829 = blockIdx.x;
    int v830;
    v830 = v829;
    while (while_method_2(v830)){
        bool v832;
        v832 = 0 <= v830;
        bool v833;
        v833 = v832 == false;
        if (v833){
            assert("The index needs to be zero or positive." && v832);
        } else {
        }
        bool v835;
        v835 = v830 < 8;
        bool v836;
        v836 = v835 == false;
        if (v836){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v835);
        } else {
        }
        assert("Tensor range check" && 0 <= v830 && v830 < 8);
        int v838;
        v838 = 1024 * v830;
        int v839;
        v839 = v838 + v828;
        float v840[4];
        int v841[4];
        int v842;
        v842 = 0;
        while (while_method_3(v842)){
            assert("Tensor range check" && 0 <= v842 && v842 < 1);
            int v844;
            v844 = 4 * v842;
            assert("Tensor range check" && 0 <= v842 && v842 < 1);
            int v845;
            v845 = 64 * v842;
            int v846;
            v846 = v845 + v839;
            int4* v847;
            v847 = reinterpret_cast<int4*>(v1 + v846);
            int4* v848;
            v848 = reinterpret_cast<int4*>(v840 + v844);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v847) % 16 == 0 && reinterpret_cast<unsigned long long>(v848) % 16 == 0);
            *v848 = *v847;
            v842 += 1 ;
        }
        int v849;
        v849 = 0;
        while (while_method_3(v849)){
            int v851;
            v851 = 0;
            while (while_method_1(v851)){
                bool v853;
                v853 = 0 <= v851;
                bool v855;
                if (v853){
                    bool v854;
                    v854 = v851 < 4;
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
                bool v858;
                v858 = 0 <= v821;
                bool v860;
                if (v858){
                    bool v859;
                    v859 = v821 < 16;
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
                int v863;
                v863 = v821 * 4;
                int v864;
                v864 = v851 + v863;
                bool v865;
                v865 = 0 <= v849;
                bool v867;
                if (v865){
                    bool v866;
                    v866 = v849 < 1;
                    v867 = v866;
                } else {
                    v867 = false;
                }
                bool v868;
                v868 = v867 == false;
                if (v868){
                    assert("The indices should be inside the range of the dimension." && v867);
                } else {
                }
                int v870;
                v870 = v849 * 64;
                int v871;
                v871 = v864 + v870;
                assert("Tensor range check" && 0 <= v849 && v849 < 1);
                assert("Tensor range check" && 0 <= v851 && v851 < 4);
                int v872;
                v872 = 4 * v849;
                int v873;
                v873 = v872 + v851;
                v841[v873] = v871;
                v851 += 1 ;
            }
            v849 += 1 ;
        }
        bool v874;
        v874 = 0 <= v822;
        bool v875;
        v875 = v874 && v823;
        bool v876;
        v876 = v875 == false;
        if (v876){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v875);
        } else {
        }
        bool v878;
        v878 = v832 && v835;
        bool v879;
        v879 = v878 == false;
        if (v879){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v878);
        } else {
        }
        int v881;
        v881 = v830 * 16;
        int v882;
        v882 = v881 + v822;
        bool v883[4];
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
                v890 = v840[v889];
                int v891;
                v891 = v841[v889];
                bool v892;
                v892 = v891 < 4;
                assert("Tensor range check" && 0 <= v884 && v884 < 1);
                assert("Tensor range check" && 0 <= v886 && v886 < 4);
                v883[v889] = v892;
                v886 += 1 ;
            }
            v884 += 1 ;
        }
        float v893[4];
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
                v900 = v840[v899];
                bool v901;
                v901 = v883[v899];
                float v902;
                if (v901){
                    v902 = v900;
                } else {
                    v902 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v894 && v894 < 1);
                assert("Tensor range check" && 0 <= v896 && v896 < 4);
                v893[v899] = v902;
                v896 += 1 ;
            }
            v894 += 1 ;
        }
        float v903;
        v903 = 0.0f;
        int v904;
        v904 = 0;
        while (while_method_3(v904)){
            int v906;
            v906 = 0;
            while (while_method_1(v906)){
                assert("Tensor range check" && 0 <= v904 && v904 < 1);
                assert("Tensor range check" && 0 <= v906 && v906 < 4);
                int v908;
                v908 = 4 * v904;
                int v909;
                v909 = v908 + v906;
                float v910;
                v910 = v893[v909];
                float v911;
                v911 = v903 + v910;
                v903 = v911;
                v906 += 1 ;
            }
            v904 += 1 ;
        }
        auto v912 = cooperative_groups::coalesced_threads();
        int v913;
        v913 = threadIdx.x;
        int v914;
        v914 = v913 / 16;
        auto v915 = cooperative_groups::labeled_partition(v912,v914);
        Closure0 v916{};
        float v917;
        v917 = cooperative_groups::reduce(v915, v903, v916);
        int v918[4];
        int v919;
        v919 = 0;
        while (while_method_3(v919)){
            int v921;
            v921 = 0;
            while (while_method_1(v921)){
                assert("Tensor range check" && 0 <= v919 && v919 < 1);
                assert("Tensor range check" && 0 <= v921 && v921 < 4);
                int v923;
                v923 = 4 * v919;
                int v924;
                v924 = v923 + v921;
                bool v925;
                v925 = v883[v924];
                int v926;
                if (v925){
                    v926 = 1;
                } else {
                    v926 = 0;
                }
                assert("Tensor range check" && 0 <= v919 && v919 < 1);
                assert("Tensor range check" && 0 <= v921 && v921 < 4);
                v918[v924] = v926;
                v921 += 1 ;
            }
            v919 += 1 ;
        }
        int v927;
        v927 = 0;
        int v928;
        v928 = 0;
        while (while_method_3(v928)){
            int v930;
            v930 = 0;
            while (while_method_1(v930)){
                assert("Tensor range check" && 0 <= v928 && v928 < 1);
                assert("Tensor range check" && 0 <= v930 && v930 < 4);
                int v932;
                v932 = 4 * v928;
                int v933;
                v933 = v932 + v930;
                int v934;
                v934 = v918[v933];
                int v935;
                v935 = v927 + v934;
                v927 = v935;
                v930 += 1 ;
            }
            v928 += 1 ;
        }
        auto v936 = cooperative_groups::coalesced_threads();
        int v937;
        v937 = threadIdx.x;
        int v938;
        v938 = v937 / 16;
        auto v939 = cooperative_groups::labeled_partition(v936,v938);
        Closure4 v940{};
        int v941;
        v941 = cooperative_groups::reduce(v939, v927, v940);
        float v942;
        v942 = (float)v941;
        float v943;
        v943 = v917 / v942;
        float v944[4];
        int v945;
        v945 = 0;
        while (while_method_3(v945)){
            int v947;
            v947 = 0;
            while (while_method_1(v947)){
                assert("Tensor range check" && 0 <= v945 && v945 < 1);
                assert("Tensor range check" && 0 <= v947 && v947 < 4);
                int v949;
                v949 = 4 * v945;
                int v950;
                v950 = v949 + v947;
                float v951;
                v951 = v840[v950];
                bool v952;
                v952 = v883[v950];
                float v953;
                if (v952){
                    v953 = v951;
                } else {
                    v953 = -1.0f / 0.0f;
                }
                float v954;
                v954 = v953 - v943;
                float v955;
                v955 = exp(v954);
                bool v956;
                v956 = v955 < 1.0f / 0.0f;
                bool v957;
                v957 = v956 == false;
                if (v957){
                    assert("The softmax values must not grow too large." && v956);
                } else {
                }
                bool v959;
                v959 = isnan(v955);
                bool v960;
                v960 = v959 == false;
                bool v961;
                v961 = v960 == false;
                if (v961){
                    assert("The softmax values must not be nans." && v960);
                } else {
                }
                assert("Tensor range check" && 0 <= v945 && v945 < 1);
                assert("Tensor range check" && 0 <= v947 && v947 < 4);
                v944[v950] = v955;
                v947 += 1 ;
            }
            v945 += 1 ;
        }
        float v963;
        v963 = 0.0f;
        int v964;
        v964 = 0;
        while (while_method_3(v964)){
            int v966;
            v966 = 0;
            while (while_method_1(v966)){
                assert("Tensor range check" && 0 <= v964 && v964 < 1);
                assert("Tensor range check" && 0 <= v966 && v966 < 4);
                int v968;
                v968 = 4 * v964;
                int v969;
                v969 = v968 + v966;
                float v970;
                v970 = v944[v969];
                float v971;
                v971 = v963 + v970;
                v963 = v971;
                v966 += 1 ;
            }
            v964 += 1 ;
        }
        auto v972 = cooperative_groups::coalesced_threads();
        int v973;
        v973 = threadIdx.x;
        int v974;
        v974 = v973 / 16;
        auto v975 = cooperative_groups::labeled_partition(v972,v974);
        float v976;
        v976 = cooperative_groups::reduce(v975, v963, v916);
        float v977[4];
        int v978;
        v978 = 0;
        while (while_method_3(v978)){
            int v980;
            v980 = 0;
            while (while_method_1(v980)){
                assert("Tensor range check" && 0 <= v978 && v978 < 1);
                assert("Tensor range check" && 0 <= v980 && v980 < 4);
                int v982;
                v982 = 4 * v978;
                int v983;
                v983 = v982 + v980;
                float v984;
                v984 = v944[v983];
                float v985;
                v985 = v984 / v976;
                assert("Tensor range check" && 0 <= v978 && v978 < 1);
                assert("Tensor range check" && 0 <= v980 && v980 < 4);
                v977[v983] = v985;
                v980 += 1 ;
            }
            v978 += 1 ;
        }
        assert("Tensor range check" && 0 <= v830 && v830 < 8);
        int v986;
        v986 = 0;
        while (while_method_3(v986)){
            assert("Tensor range check" && 0 <= v986 && v986 < 1);
            int v988;
            v988 = 64 * v986;
            int v989;
            v989 = v988 + v839;
            assert("Tensor range check" && 0 <= v986 && v986 < 1);
            int v990;
            v990 = 4 * v986;
            int4* v991;
            v991 = reinterpret_cast<int4*>(v977 + v990);
            int4* v992;
            v992 = reinterpret_cast<int4*>(v4 + v989);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v991) % 16 == 0 && reinterpret_cast<unsigned long long>(v992) % 16 == 0);
            *v992 = *v991;
            v986 += 1 ;
        }
        v830 += 24 ;
    }
    v17.sync() ;
    int v993;
    v993 = threadIdx.x;
    int v994;
    v994 = blockIdx.x;
    int v995;
    v995 = v994 * 256;
    int v996;
    v996 = v993 + v995;
    unsigned long long v997;
    v997 = (unsigned long long)v996;
    curandStatePhilox4_32_10_t v998;
    curand_init(12344321ull,v997,0ull,&v998);
    int v999;
    v999 = threadIdx.x;
    bool v1000;
    v1000 = 0 <= v999;
    bool v1001;
    v1001 = v1000 == false;
    if (v1001){
        assert("The index needs to be zero or positive." && v1000);
    } else {
    }
    int v1003;
    v1003 = v999 % 16;
    int v1004;
    v1004 = v999 / 16;
    bool v1005;
    v1005 = v1004 < 16;
    bool v1006;
    v1006 = v1005 == false;
    if (v1006){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1005);
    } else {
    }
    assert("Tensor range check" && 0 <= v1004 && v1004 < 16);
    assert("Tensor range check" && 0 <= v1003 && v1003 < 16);
    int v1008;
    v1008 = 4 * v1003;
    int v1009;
    v1009 = 64 * v1004;
    int v1010;
    v1010 = v1009 + v1008;
    assert("Tensor range check" && 0 <= v1004 && v1004 < 16);
    assert("Tensor range check" && 0 <= v1003 && v1003 < 16);
    assert("Tensor range check" && 0 <= v1004 && v1004 < 16);
    int v1011;
    v1011 = blockIdx.x;
    int v1012;
    v1012 = v1011;
    while (while_method_2(v1012)){
        bool v1014;
        v1014 = 0 <= v1012;
        bool v1015;
        v1015 = v1014 == false;
        if (v1015){
            assert("The index needs to be zero or positive." && v1014);
        } else {
        }
        bool v1017;
        v1017 = v1012 < 8;
        bool v1018;
        v1018 = v1017 == false;
        if (v1018){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1017);
        } else {
        }
        assert("Tensor range check" && 0 <= v1012 && v1012 < 8);
        int v1020;
        v1020 = 1024 * v1012;
        int v1021;
        v1021 = v1020 + v1010;
        float v1022[4];
        int v1023[4];
        int v1024;
        v1024 = 0;
        while (while_method_3(v1024)){
            assert("Tensor range check" && 0 <= v1024 && v1024 < 1);
            int v1026;
            v1026 = 4 * v1024;
            assert("Tensor range check" && 0 <= v1024 && v1024 < 1);
            int v1027;
            v1027 = 64 * v1024;
            int v1028;
            v1028 = v1027 + v1021;
            int4* v1029;
            v1029 = reinterpret_cast<int4*>(v1 + v1028);
            int4* v1030;
            v1030 = reinterpret_cast<int4*>(v1022 + v1026);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1029) % 16 == 0 && reinterpret_cast<unsigned long long>(v1030) % 16 == 0);
            *v1030 = *v1029;
            v1024 += 1 ;
        }
        int v1031;
        v1031 = 0;
        while (while_method_3(v1031)){
            int v1033;
            v1033 = 0;
            while (while_method_1(v1033)){
                bool v1035;
                v1035 = 0 <= v1033;
                bool v1037;
                if (v1035){
                    bool v1036;
                    v1036 = v1033 < 4;
                    v1037 = v1036;
                } else {
                    v1037 = false;
                }
                bool v1038;
                v1038 = v1037 == false;
                if (v1038){
                    assert("The indices should be inside the range of the dimension." && v1037);
                } else {
                }
                bool v1040;
                v1040 = 0 <= v1003;
                bool v1042;
                if (v1040){
                    bool v1041;
                    v1041 = v1003 < 16;
                    v1042 = v1041;
                } else {
                    v1042 = false;
                }
                bool v1043;
                v1043 = v1042 == false;
                if (v1043){
                    assert("The indices should be inside the range of the dimension." && v1042);
                } else {
                }
                int v1045;
                v1045 = v1003 * 4;
                int v1046;
                v1046 = v1033 + v1045;
                bool v1047;
                v1047 = 0 <= v1031;
                bool v1049;
                if (v1047){
                    bool v1048;
                    v1048 = v1031 < 1;
                    v1049 = v1048;
                } else {
                    v1049 = false;
                }
                bool v1050;
                v1050 = v1049 == false;
                if (v1050){
                    assert("The indices should be inside the range of the dimension." && v1049);
                } else {
                }
                int v1052;
                v1052 = v1031 * 64;
                int v1053;
                v1053 = v1046 + v1052;
                assert("Tensor range check" && 0 <= v1031 && v1031 < 1);
                assert("Tensor range check" && 0 <= v1033 && v1033 < 4);
                int v1054;
                v1054 = 4 * v1031;
                int v1055;
                v1055 = v1054 + v1033;
                v1023[v1055] = v1053;
                v1033 += 1 ;
            }
            v1031 += 1 ;
        }
        bool v1056;
        v1056 = 0 <= v1004;
        bool v1057;
        v1057 = v1056 && v1005;
        bool v1058;
        v1058 = v1057 == false;
        if (v1058){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1057);
        } else {
        }
        bool v1060;
        v1060 = v1014 && v1017;
        bool v1061;
        v1061 = v1060 == false;
        if (v1061){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1060);
        } else {
        }
        int v1063;
        v1063 = v1012 * 16;
        int v1064;
        v1064 = v1063 + v1004;
        float v1065;
        v1065 = 0.0f;
        int v1066;
        v1066 = 0;
        while (while_method_3(v1066)){
            int v1068;
            v1068 = 0;
            while (while_method_1(v1068)){
                assert("Tensor range check" && 0 <= v1066 && v1066 < 1);
                assert("Tensor range check" && 0 <= v1068 && v1068 < 4);
                int v1070;
                v1070 = 4 * v1066;
                int v1071;
                v1071 = v1070 + v1068;
                float v1072;
                v1072 = v1022[v1071];
                float v1073;
                v1073 = v1065 + v1072;
                v1065 = v1073;
                v1068 += 1 ;
            }
            v1066 += 1 ;
        }
        auto v1074 = cooperative_groups::coalesced_threads();
        int v1075;
        v1075 = threadIdx.x;
        int v1076;
        v1076 = v1075 / 16;
        auto v1077 = cooperative_groups::labeled_partition(v1074,v1076);
        Closure0 v1078{};
        float v1079;
        v1079 = cooperative_groups::reduce(v1077, v1065, v1078);
        float v1080;
        v1080 = v1079 / 64.0f;
        float v1081[4];
        int v1082;
        v1082 = 0;
        while (while_method_3(v1082)){
            int v1084;
            v1084 = 0;
            while (while_method_1(v1084)){
                assert("Tensor range check" && 0 <= v1082 && v1082 < 1);
                assert("Tensor range check" && 0 <= v1084 && v1084 < 4);
                int v1086;
                v1086 = 4 * v1082;
                int v1087;
                v1087 = v1086 + v1084;
                float v1088;
                v1088 = v1022[v1087];
                float v1089;
                v1089 = v1088 - v1080;
                float v1090;
                v1090 = exp(v1089);
                assert("Tensor range check" && 0 <= v1082 && v1082 < 1);
                assert("Tensor range check" && 0 <= v1084 && v1084 < 4);
                v1081[v1087] = v1090;
                v1084 += 1 ;
            }
            v1082 += 1 ;
        }
        float v1091;
        v1091 = 0.0f;
        int v1092;
        v1092 = 0;
        while (while_method_3(v1092)){
            int v1094;
            v1094 = 0;
            while (while_method_1(v1094)){
                assert("Tensor range check" && 0 <= v1092 && v1092 < 1);
                assert("Tensor range check" && 0 <= v1094 && v1094 < 4);
                int v1096;
                v1096 = 4 * v1092;
                int v1097;
                v1097 = v1096 + v1094;
                float v1098;
                v1098 = v1081[v1097];
                float v1099;
                v1099 = v1091 + v1098;
                v1091 = v1099;
                v1094 += 1 ;
            }
            v1092 += 1 ;
        }
        auto v1100 = cooperative_groups::coalesced_threads();
        int v1101;
        v1101 = threadIdx.x;
        int v1102;
        v1102 = v1101 / 16;
        auto v1103 = cooperative_groups::labeled_partition(v1100,v1102);
        float v1104;
        v1104 = cooperative_groups::reduce(v1103, v1091, v1078);
        float v1105[4];
        int v1106;
        v1106 = 0;
        while (while_method_3(v1106)){
            int v1108;
            v1108 = 0;
            while (while_method_1(v1108)){
                assert("Tensor range check" && 0 <= v1106 && v1106 < 1);
                assert("Tensor range check" && 0 <= v1108 && v1108 < 4);
                int v1110;
                v1110 = 4 * v1106;
                int v1111;
                v1111 = v1110 + v1108;
                float v1112;
                v1112 = v1081[v1111];
                float v1113;
                v1113 = v1112 / v1104;
                assert("Tensor range check" && 0 <= v1106 && v1106 < 1);
                assert("Tensor range check" && 0 <= v1108 && v1108 < 4);
                v1105[v1111] = v1113;
                v1108 += 1 ;
            }
            v1106 += 1 ;
        }
        float v1114[4];
        float v1115;
        v1115 = 0.0f;
        int v1116;
        v1116 = 0;
        while (while_method_3(v1116)){
            assert("Tensor range check" && 0 <= v1116 && v1116 < 1);
            int v1118;
            v1118 = 4 * v1116;
            assert("Tensor range check" && 0 <= v1116 && v1116 < 1);
            float v1119;
            v1119 = 0.0f;
            int v1120;
            v1120 = 0;
            while (while_method_1(v1120)){
                assert("Tensor range check" && 0 <= v1120 && v1120 < 4);
                int v1122;
                v1122 = v1120 + v1118;
                float v1123;
                v1123 = v1105[v1122];
                float v1124;
                v1124 = v1119 + v1123;
                v1119 = v1124;
                v1120 += 1 ;
            }
            auto v1125 = cooperative_groups::coalesced_threads();
            int v1126;
            v1126 = threadIdx.x;
            int v1127;
            v1127 = v1126 / 16;
            auto v1128 = cooperative_groups::labeled_partition(v1125,v1127);
            Closure2 v1129{};
            float v1130;
            v1130 = cooperative_groups::inclusive_scan(v1128, v1119, v1129);
            float v1131;
            v1131 = v1128.shfl_up(v1130,1);
            bool v1132;
            v1132 = v1128.thread_rank() == 0;
            float v1133;
            if (v1132){
                v1133 = 0.0f;
            } else {
                v1133 = v1131;
            }
            float v1134;
            v1134 = v1128.shfl(v1130,v1128.num_threads()-1);
            float v1135;
            v1135 = v1115 + v1133;
            float v1136;
            v1136 = v1135;
            int v1137;
            v1137 = 0;
            while (while_method_1(v1137)){
                assert("Tensor range check" && 0 <= v1137 && v1137 < 4);
                int v1139;
                v1139 = v1137 + v1118;
                float v1140;
                v1140 = v1105[v1139];
                float v1141;
                v1141 = v1136 + v1140;
                assert("Tensor range check" && 0 <= v1137 && v1137 < 4);
                v1114[v1139] = v1141;
                v1136 = v1141;
                v1137 += 1 ;
            }
            float v1142;
            v1142 = v1115 + v1134;
            v1115 = v1142;
            v1116 += 1 ;
        }
        float v1143[4];
        bool v1144[4];
        int v1145;
        v1145 = 0;
        while (while_method_3(v1145)){
            int v1147;
            v1147 = 0;
            while (while_method_1(v1147)){
                assert("Tensor range check" && 0 <= v1145 && v1145 < 1);
                assert("Tensor range check" && 0 <= v1147 && v1147 < 4);
                int v1149;
                v1149 = 4 * v1145;
                int v1150;
                v1150 = v1149 + v1147;
                float v1151;
                v1151 = v1114[v1150];
                float v1152;
                v1152 = v1105[v1150];
                bool v1153;
                v1153 = v1152 > 0.0f;
                assert("Tensor range check" && 0 <= v1145 && v1145 < 1);
                assert("Tensor range check" && 0 <= v1147 && v1147 < 4);
                v1143[v1150] = v1151;
                v1144[v1150] = v1153;
                v1147 += 1 ;
            }
            v1145 += 1 ;
        }
        float v1154; bool v1155;
        Tuple2 tmp47 = Tuple2{-1.0f / 0.0f, false};
        v1154 = tmp47.v0; v1155 = tmp47.v1;
        int v1156;
        v1156 = 0;
        while (while_method_3(v1156)){
            int v1158;
            v1158 = 0;
            while (while_method_1(v1158)){
                assert("Tensor range check" && 0 <= v1156 && v1156 < 1);
                assert("Tensor range check" && 0 <= v1158 && v1158 < 4);
                int v1160;
                v1160 = 4 * v1156;
                int v1161;
                v1161 = v1160 + v1158;
                float v1162;
                v1162 = v1143[v1161];
                bool v1163;
                v1163 = v1144[v1161];
                float v1170; bool v1171;
                if (v1155){
                    if (v1163){
                        bool v1164;
                        v1164 = v1154 >= v1162;
                        float v1165;
                        if (v1164){
                            v1165 = v1154;
                        } else {
                            v1165 = v1162;
                        }
                        v1170 = v1165; v1171 = true;
                    } else {
                        v1170 = v1154; v1171 = v1155;
                    }
                } else {
                    if (v1163){
                        v1170 = v1162; v1171 = v1163;
                    } else {
                        v1170 = v1154; v1171 = v1155;
                    }
                }
                v1154 = v1170;
                v1155 = v1171;
                v1158 += 1 ;
            }
            v1156 += 1 ;
        }
        auto v1172 = cooperative_groups::coalesced_threads();
        int v1173;
        v1173 = threadIdx.x;
        int v1174;
        v1174 = v1173 / 16;
        auto v1175 = cooperative_groups::labeled_partition(v1172,v1174);
        Closure5 v1176{};
        float v1177; bool v1178;
        Tuple2 tmp48 = cooperative_groups::reduce(v1175, Tuple2{v1154, v1155}, v1176);
        v1177 = tmp48.v0; v1178 = tmp48.v1;
        bool v1179;
        v1179 = v1178 == false;
        if (v1179){
            assert("The local reduce must be true." && v1178);
        } else {
        }
        float v1181[4];
        int v1182[4];
        int v1183;
        v1183 = 0;
        while (while_method_3(v1183)){
            int v1185;
            v1185 = 0;
            while (while_method_1(v1185)){
                assert("Tensor range check" && 0 <= v1183 && v1183 < 1);
                assert("Tensor range check" && 0 <= v1185 && v1185 < 4);
                int v1187;
                v1187 = 4 * v1183;
                int v1188;
                v1188 = v1187 + v1185;
                int v1189;
                v1189 = v1023[v1188];
                float v1190;
                v1190 = curand_uniform(&v998);
                assert("Tensor range check" && 0 <= v1183 && v1183 < 1);
                assert("Tensor range check" && 0 <= v1185 && v1185 < 4);
                v1181[v1188] = v1190;
                v1182[v1188] = v1189;
                v1185 += 1 ;
            }
            v1183 += 1 ;
        }
        float v1191; int v1192;
        Tuple1 tmp49 = Tuple1{0.0f, 2147483647};
        v1191 = tmp49.v0; v1192 = tmp49.v1;
        int v1193;
        v1193 = 0;
        while (while_method_3(v1193)){
            int v1195;
            v1195 = 0;
            while (while_method_1(v1195)){
                assert("Tensor range check" && 0 <= v1193 && v1193 < 1);
                assert("Tensor range check" && 0 <= v1195 && v1195 < 4);
                int v1197;
                v1197 = 4 * v1193;
                int v1198;
                v1198 = v1197 + v1195;
                float v1199;
                v1199 = v1181[v1198];
                int v1200;
                v1200 = v1182[v1198];
                bool v1201;
                v1201 = v1192 < v1200;
                float v1202; int v1203;
                if (v1201){
                    v1202 = v1191; v1203 = v1192;
                } else {
                    v1202 = v1199; v1203 = v1200;
                }
                v1191 = v1202;
                v1192 = v1203;
                v1195 += 1 ;
            }
            v1193 += 1 ;
        }
        auto v1204 = cooperative_groups::coalesced_threads();
        int v1205;
        v1205 = threadIdx.x;
        int v1206;
        v1206 = v1205 / 16;
        auto v1207 = cooperative_groups::labeled_partition(v1204,v1206);
        Closure6 v1208{};
        float v1209; int v1210;
        Tuple1 tmp50 = cooperative_groups::reduce(v1207, Tuple1{v1191, v1192}, v1208);
        v1209 = tmp50.v0; v1210 = tmp50.v1;
        float v1211;
        v1211 = v1177 * v1209;
        int v1212[4];
        bool v1213[4];
        int v1214;
        v1214 = 0;
        while (while_method_3(v1214)){
            int v1216;
            v1216 = 0;
            while (while_method_1(v1216)){
                assert("Tensor range check" && 0 <= v1214 && v1214 < 1);
                assert("Tensor range check" && 0 <= v1216 && v1216 < 4);
                int v1218;
                v1218 = 4 * v1214;
                int v1219;
                v1219 = v1218 + v1216;
                float v1220;
                v1220 = v1143[v1219];
                bool v1221;
                v1221 = v1144[v1219];
                int v1222;
                v1222 = v1023[v1219];
                int v1225; bool v1226;
                if (v1221){
                    float v1223;
                    v1223 = v1220 - v1211;
                    bool v1224;
                    v1224 = v1223 >= 0.0f;
                    v1225 = v1222; v1226 = v1224;
                } else {
                    v1225 = 2147483647; v1226 = false;
                }
                assert("Tensor range check" && 0 <= v1214 && v1214 < 1);
                assert("Tensor range check" && 0 <= v1216 && v1216 < 4);
                v1212[v1219] = v1225;
                v1213[v1219] = v1226;
                v1216 += 1 ;
            }
            v1214 += 1 ;
        }
        int v1227; bool v1228;
        Tuple3 tmp51 = Tuple3{2147483647, false};
        v1227 = tmp51.v0; v1228 = tmp51.v1;
        int v1229;
        v1229 = 0;
        while (while_method_3(v1229)){
            int v1231;
            v1231 = 0;
            while (while_method_1(v1231)){
                assert("Tensor range check" && 0 <= v1229 && v1229 < 1);
                assert("Tensor range check" && 0 <= v1231 && v1231 < 4);
                int v1233;
                v1233 = 4 * v1229;
                int v1234;
                v1234 = v1233 + v1231;
                int v1235;
                v1235 = v1212[v1234];
                bool v1236;
                v1236 = v1213[v1234];
                int v1243; bool v1244;
                if (v1228){
                    if (v1236){
                        bool v1237;
                        v1237 = v1227 < v1235;
                        int v1238;
                        if (v1237){
                            v1238 = v1227;
                        } else {
                            v1238 = v1235;
                        }
                        v1243 = v1238; v1244 = true;
                    } else {
                        v1243 = v1227; v1244 = v1228;
                    }
                } else {
                    if (v1236){
                        v1243 = v1235; v1244 = v1236;
                    } else {
                        v1243 = v1227; v1244 = v1228;
                    }
                }
                v1227 = v1243;
                v1228 = v1244;
                v1231 += 1 ;
            }
            v1229 += 1 ;
        }
        auto v1245 = cooperative_groups::coalesced_threads();
        int v1246;
        v1246 = threadIdx.x;
        int v1247;
        v1247 = v1246 / 16;
        auto v1248 = cooperative_groups::labeled_partition(v1245,v1247);
        Closure7 v1249{};
        int v1250; bool v1251;
        Tuple3 tmp52 = cooperative_groups::reduce(v1248, Tuple3{v1227, v1228}, v1249);
        v1250 = tmp52.v0; v1251 = tmp52.v1;
        bool v1252;
        v1252 = v1251 == false;
        if (v1252){
            assert("The local reduce must be true." && v1251);
        } else {
        }
        assert("Tensor range check" && 0 <= v1012 && v1012 < 8);
        int v1254;
        v1254 = 0;
        while (while_method_3(v1254)){
            assert("Tensor range check" && 0 <= v1254 && v1254 < 1);
            int v1256;
            v1256 = 64 * v1254;
            int v1257;
            v1257 = v1256 + v1021;
            assert("Tensor range check" && 0 <= v1254 && v1254 < 1);
            int v1258;
            v1258 = 4 * v1254;
            int4* v1259;
            v1259 = reinterpret_cast<int4*>(v1105 + v1258);
            int4* v1260;
            v1260 = reinterpret_cast<int4*>(v13 + v1257);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1259) % 16 == 0 && reinterpret_cast<unsigned long long>(v1260) % 16 == 0);
            *v1260 = *v1259;
            v1254 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1012 && v1012 < 8);
        int v1261;
        v1261 = 16 * v1012;
        int v1262;
        v1262 = v1261 + v1004;
        v14[v1262] = v1250;
        v1012 += 24 ;
    }
    v17.sync() ;
    int v1263;
    v1263 = threadIdx.x;
    int v1264;
    v1264 = blockIdx.x;
    int v1265;
    v1265 = v1264 * 256;
    int v1266;
    v1266 = v1263 + v1265;
    unsigned long long v1267;
    v1267 = (unsigned long long)v1266;
    curandStatePhilox4_32_10_t v1268;
    curand_init(12344321ull,v1267,0ull,&v1268);
    int v1269;
    v1269 = threadIdx.x;
    bool v1270;
    v1270 = 0 <= v1269;
    bool v1271;
    v1271 = v1270 == false;
    if (v1271){
        assert("The index needs to be zero or positive." && v1270);
    } else {
    }
    int v1273;
    v1273 = v1269 % 16;
    int v1274;
    v1274 = v1269 / 16;
    bool v1275;
    v1275 = v1274 < 16;
    bool v1276;
    v1276 = v1275 == false;
    if (v1276){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1275);
    } else {
    }
    assert("Tensor range check" && 0 <= v1274 && v1274 < 16);
    assert("Tensor range check" && 0 <= v1273 && v1273 < 16);
    int v1278;
    v1278 = 4 * v1273;
    int v1279;
    v1279 = 64 * v1274;
    int v1280;
    v1280 = v1279 + v1278;
    assert("Tensor range check" && 0 <= v1274 && v1274 < 16);
    assert("Tensor range check" && 0 <= v1273 && v1273 < 16);
    assert("Tensor range check" && 0 <= v1274 && v1274 < 16);
    int v1281;
    v1281 = blockIdx.x;
    int v1282;
    v1282 = v1281;
    while (while_method_2(v1282)){
        bool v1284;
        v1284 = 0 <= v1282;
        bool v1285;
        v1285 = v1284 == false;
        if (v1285){
            assert("The index needs to be zero or positive." && v1284);
        } else {
        }
        bool v1287;
        v1287 = v1282 < 8;
        bool v1288;
        v1288 = v1287 == false;
        if (v1288){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1287);
        } else {
        }
        assert("Tensor range check" && 0 <= v1282 && v1282 < 8);
        int v1290;
        v1290 = 1024 * v1282;
        int v1291;
        v1291 = v1290 + v1280;
        float v1292[4];
        int v1293[4];
        int v1294;
        v1294 = 0;
        while (while_method_3(v1294)){
            assert("Tensor range check" && 0 <= v1294 && v1294 < 1);
            int v1296;
            v1296 = 4 * v1294;
            assert("Tensor range check" && 0 <= v1294 && v1294 < 1);
            int v1297;
            v1297 = 64 * v1294;
            int v1298;
            v1298 = v1297 + v1291;
            int4* v1299;
            v1299 = reinterpret_cast<int4*>(v1 + v1298);
            int4* v1300;
            v1300 = reinterpret_cast<int4*>(v1292 + v1296);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1299) % 16 == 0 && reinterpret_cast<unsigned long long>(v1300) % 16 == 0);
            *v1300 = *v1299;
            v1294 += 1 ;
        }
        int v1301;
        v1301 = 0;
        while (while_method_3(v1301)){
            int v1303;
            v1303 = 0;
            while (while_method_1(v1303)){
                bool v1305;
                v1305 = 0 <= v1303;
                bool v1307;
                if (v1305){
                    bool v1306;
                    v1306 = v1303 < 4;
                    v1307 = v1306;
                } else {
                    v1307 = false;
                }
                bool v1308;
                v1308 = v1307 == false;
                if (v1308){
                    assert("The indices should be inside the range of the dimension." && v1307);
                } else {
                }
                bool v1310;
                v1310 = 0 <= v1273;
                bool v1312;
                if (v1310){
                    bool v1311;
                    v1311 = v1273 < 16;
                    v1312 = v1311;
                } else {
                    v1312 = false;
                }
                bool v1313;
                v1313 = v1312 == false;
                if (v1313){
                    assert("The indices should be inside the range of the dimension." && v1312);
                } else {
                }
                int v1315;
                v1315 = v1273 * 4;
                int v1316;
                v1316 = v1303 + v1315;
                bool v1317;
                v1317 = 0 <= v1301;
                bool v1319;
                if (v1317){
                    bool v1318;
                    v1318 = v1301 < 1;
                    v1319 = v1318;
                } else {
                    v1319 = false;
                }
                bool v1320;
                v1320 = v1319 == false;
                if (v1320){
                    assert("The indices should be inside the range of the dimension." && v1319);
                } else {
                }
                int v1322;
                v1322 = v1301 * 64;
                int v1323;
                v1323 = v1316 + v1322;
                assert("Tensor range check" && 0 <= v1301 && v1301 < 1);
                assert("Tensor range check" && 0 <= v1303 && v1303 < 4);
                int v1324;
                v1324 = 4 * v1301;
                int v1325;
                v1325 = v1324 + v1303;
                v1293[v1325] = v1323;
                v1303 += 1 ;
            }
            v1301 += 1 ;
        }
        bool v1326;
        v1326 = 0 <= v1274;
        bool v1327;
        v1327 = v1326 && v1275;
        bool v1328;
        v1328 = v1327 == false;
        if (v1328){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1327);
        } else {
        }
        bool v1330;
        v1330 = v1284 && v1287;
        bool v1331;
        v1331 = v1330 == false;
        if (v1331){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1330);
        } else {
        }
        int v1333;
        v1333 = v1282 * 16;
        int v1334;
        v1334 = v1333 + v1274;
        bool v1335[4];
        int v1336;
        v1336 = 0;
        while (while_method_3(v1336)){
            int v1338;
            v1338 = 0;
            while (while_method_1(v1338)){
                assert("Tensor range check" && 0 <= v1336 && v1336 < 1);
                assert("Tensor range check" && 0 <= v1338 && v1338 < 4);
                int v1340;
                v1340 = 4 * v1336;
                int v1341;
                v1341 = v1340 + v1338;
                float v1342;
                v1342 = v1292[v1341];
                int v1343;
                v1343 = v1293[v1341];
                bool v1344;
                v1344 = v1343 < 11;
                assert("Tensor range check" && 0 <= v1336 && v1336 < 1);
                assert("Tensor range check" && 0 <= v1338 && v1338 < 4);
                v1335[v1341] = v1344;
                v1338 += 1 ;
            }
            v1336 += 1 ;
        }
        float v1345[4];
        int v1346;
        v1346 = 0;
        while (while_method_3(v1346)){
            int v1348;
            v1348 = 0;
            while (while_method_1(v1348)){
                assert("Tensor range check" && 0 <= v1346 && v1346 < 1);
                assert("Tensor range check" && 0 <= v1348 && v1348 < 4);
                int v1350;
                v1350 = 4 * v1346;
                int v1351;
                v1351 = v1350 + v1348;
                float v1352;
                v1352 = v1292[v1351];
                bool v1353;
                v1353 = v1335[v1351];
                float v1354;
                if (v1353){
                    v1354 = v1352;
                } else {
                    v1354 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1346 && v1346 < 1);
                assert("Tensor range check" && 0 <= v1348 && v1348 < 4);
                v1345[v1351] = v1354;
                v1348 += 1 ;
            }
            v1346 += 1 ;
        }
        float v1355;
        v1355 = 0.0f;
        int v1356;
        v1356 = 0;
        while (while_method_3(v1356)){
            int v1358;
            v1358 = 0;
            while (while_method_1(v1358)){
                assert("Tensor range check" && 0 <= v1356 && v1356 < 1);
                assert("Tensor range check" && 0 <= v1358 && v1358 < 4);
                int v1360;
                v1360 = 4 * v1356;
                int v1361;
                v1361 = v1360 + v1358;
                float v1362;
                v1362 = v1345[v1361];
                float v1363;
                v1363 = v1355 + v1362;
                v1355 = v1363;
                v1358 += 1 ;
            }
            v1356 += 1 ;
        }
        auto v1364 = cooperative_groups::coalesced_threads();
        int v1365;
        v1365 = threadIdx.x;
        int v1366;
        v1366 = v1365 / 16;
        auto v1367 = cooperative_groups::labeled_partition(v1364,v1366);
        Closure0 v1368{};
        float v1369;
        v1369 = cooperative_groups::reduce(v1367, v1355, v1368);
        int v1370[4];
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
                bool v1377;
                v1377 = v1335[v1376];
                int v1378;
                if (v1377){
                    v1378 = 1;
                } else {
                    v1378 = 0;
                }
                assert("Tensor range check" && 0 <= v1371 && v1371 < 1);
                assert("Tensor range check" && 0 <= v1373 && v1373 < 4);
                v1370[v1376] = v1378;
                v1373 += 1 ;
            }
            v1371 += 1 ;
        }
        int v1379;
        v1379 = 0;
        int v1380;
        v1380 = 0;
        while (while_method_3(v1380)){
            int v1382;
            v1382 = 0;
            while (while_method_1(v1382)){
                assert("Tensor range check" && 0 <= v1380 && v1380 < 1);
                assert("Tensor range check" && 0 <= v1382 && v1382 < 4);
                int v1384;
                v1384 = 4 * v1380;
                int v1385;
                v1385 = v1384 + v1382;
                int v1386;
                v1386 = v1370[v1385];
                int v1387;
                v1387 = v1379 + v1386;
                v1379 = v1387;
                v1382 += 1 ;
            }
            v1380 += 1 ;
        }
        auto v1388 = cooperative_groups::coalesced_threads();
        int v1389;
        v1389 = threadIdx.x;
        int v1390;
        v1390 = v1389 / 16;
        auto v1391 = cooperative_groups::labeled_partition(v1388,v1390);
        Closure4 v1392{};
        int v1393;
        v1393 = cooperative_groups::reduce(v1391, v1379, v1392);
        float v1394;
        v1394 = (float)v1393;
        float v1395;
        v1395 = v1369 / v1394;
        float v1396[4];
        int v1397;
        v1397 = 0;
        while (while_method_3(v1397)){
            int v1399;
            v1399 = 0;
            while (while_method_1(v1399)){
                assert("Tensor range check" && 0 <= v1397 && v1397 < 1);
                assert("Tensor range check" && 0 <= v1399 && v1399 < 4);
                int v1401;
                v1401 = 4 * v1397;
                int v1402;
                v1402 = v1401 + v1399;
                float v1403;
                v1403 = v1292[v1402];
                bool v1404;
                v1404 = v1335[v1402];
                float v1405;
                if (v1404){
                    v1405 = v1403;
                } else {
                    v1405 = -1.0f / 0.0f;
                }
                float v1406;
                v1406 = v1405 - v1395;
                float v1407;
                v1407 = exp(v1406);
                bool v1408;
                v1408 = v1407 < 1.0f / 0.0f;
                bool v1409;
                v1409 = v1408 == false;
                if (v1409){
                    assert("The softmax values must not grow too large." && v1408);
                } else {
                }
                bool v1411;
                v1411 = isnan(v1407);
                bool v1412;
                v1412 = v1411 == false;
                bool v1413;
                v1413 = v1412 == false;
                if (v1413){
                    assert("The softmax values must not be nans." && v1412);
                } else {
                }
                assert("Tensor range check" && 0 <= v1397 && v1397 < 1);
                assert("Tensor range check" && 0 <= v1399 && v1399 < 4);
                v1396[v1402] = v1407;
                v1399 += 1 ;
            }
            v1397 += 1 ;
        }
        float v1415;
        v1415 = 0.0f;
        int v1416;
        v1416 = 0;
        while (while_method_3(v1416)){
            int v1418;
            v1418 = 0;
            while (while_method_1(v1418)){
                assert("Tensor range check" && 0 <= v1416 && v1416 < 1);
                assert("Tensor range check" && 0 <= v1418 && v1418 < 4);
                int v1420;
                v1420 = 4 * v1416;
                int v1421;
                v1421 = v1420 + v1418;
                float v1422;
                v1422 = v1396[v1421];
                float v1423;
                v1423 = v1415 + v1422;
                v1415 = v1423;
                v1418 += 1 ;
            }
            v1416 += 1 ;
        }
        auto v1424 = cooperative_groups::coalesced_threads();
        int v1425;
        v1425 = threadIdx.x;
        int v1426;
        v1426 = v1425 / 16;
        auto v1427 = cooperative_groups::labeled_partition(v1424,v1426);
        float v1428;
        v1428 = cooperative_groups::reduce(v1427, v1415, v1368);
        float v1429[4];
        int v1430;
        v1430 = 0;
        while (while_method_3(v1430)){
            int v1432;
            v1432 = 0;
            while (while_method_1(v1432)){
                assert("Tensor range check" && 0 <= v1430 && v1430 < 1);
                assert("Tensor range check" && 0 <= v1432 && v1432 < 4);
                int v1434;
                v1434 = 4 * v1430;
                int v1435;
                v1435 = v1434 + v1432;
                float v1436;
                v1436 = v1396[v1435];
                float v1437;
                v1437 = v1436 / v1428;
                assert("Tensor range check" && 0 <= v1430 && v1430 < 1);
                assert("Tensor range check" && 0 <= v1432 && v1432 < 4);
                v1429[v1435] = v1437;
                v1432 += 1 ;
            }
            v1430 += 1 ;
        }
        float v1438[4];
        float v1439;
        v1439 = 0.0f;
        int v1440;
        v1440 = 0;
        while (while_method_3(v1440)){
            assert("Tensor range check" && 0 <= v1440 && v1440 < 1);
            int v1442;
            v1442 = 4 * v1440;
            assert("Tensor range check" && 0 <= v1440 && v1440 < 1);
            float v1443;
            v1443 = 0.0f;
            int v1444;
            v1444 = 0;
            while (while_method_1(v1444)){
                assert("Tensor range check" && 0 <= v1444 && v1444 < 4);
                int v1446;
                v1446 = v1444 + v1442;
                float v1447;
                v1447 = v1429[v1446];
                float v1448;
                v1448 = v1443 + v1447;
                v1443 = v1448;
                v1444 += 1 ;
            }
            auto v1449 = cooperative_groups::coalesced_threads();
            int v1450;
            v1450 = threadIdx.x;
            int v1451;
            v1451 = v1450 / 16;
            auto v1452 = cooperative_groups::labeled_partition(v1449,v1451);
            Closure2 v1453{};
            float v1454;
            v1454 = cooperative_groups::inclusive_scan(v1452, v1443, v1453);
            float v1455;
            v1455 = v1452.shfl_up(v1454,1);
            bool v1456;
            v1456 = v1452.thread_rank() == 0;
            float v1457;
            if (v1456){
                v1457 = 0.0f;
            } else {
                v1457 = v1455;
            }
            float v1458;
            v1458 = v1452.shfl(v1454,v1452.num_threads()-1);
            float v1459;
            v1459 = v1439 + v1457;
            float v1460;
            v1460 = v1459;
            int v1461;
            v1461 = 0;
            while (while_method_1(v1461)){
                assert("Tensor range check" && 0 <= v1461 && v1461 < 4);
                int v1463;
                v1463 = v1461 + v1442;
                float v1464;
                v1464 = v1429[v1463];
                float v1465;
                v1465 = v1460 + v1464;
                assert("Tensor range check" && 0 <= v1461 && v1461 < 4);
                v1438[v1463] = v1465;
                v1460 = v1465;
                v1461 += 1 ;
            }
            float v1466;
            v1466 = v1439 + v1458;
            v1439 = v1466;
            v1440 += 1 ;
        }
        float v1467[4];
        bool v1468[4];
        int v1469;
        v1469 = 0;
        while (while_method_3(v1469)){
            int v1471;
            v1471 = 0;
            while (while_method_1(v1471)){
                assert("Tensor range check" && 0 <= v1469 && v1469 < 1);
                assert("Tensor range check" && 0 <= v1471 && v1471 < 4);
                int v1473;
                v1473 = 4 * v1469;
                int v1474;
                v1474 = v1473 + v1471;
                float v1475;
                v1475 = v1438[v1474];
                float v1476;
                v1476 = v1429[v1474];
                bool v1477;
                v1477 = v1476 > 0.0f;
                assert("Tensor range check" && 0 <= v1469 && v1469 < 1);
                assert("Tensor range check" && 0 <= v1471 && v1471 < 4);
                v1467[v1474] = v1475;
                v1468[v1474] = v1477;
                v1471 += 1 ;
            }
            v1469 += 1 ;
        }
        float v1478; bool v1479;
        Tuple2 tmp53 = Tuple2{-1.0f / 0.0f, false};
        v1478 = tmp53.v0; v1479 = tmp53.v1;
        int v1480;
        v1480 = 0;
        while (while_method_3(v1480)){
            int v1482;
            v1482 = 0;
            while (while_method_1(v1482)){
                assert("Tensor range check" && 0 <= v1480 && v1480 < 1);
                assert("Tensor range check" && 0 <= v1482 && v1482 < 4);
                int v1484;
                v1484 = 4 * v1480;
                int v1485;
                v1485 = v1484 + v1482;
                float v1486;
                v1486 = v1467[v1485];
                bool v1487;
                v1487 = v1468[v1485];
                float v1494; bool v1495;
                if (v1479){
                    if (v1487){
                        bool v1488;
                        v1488 = v1478 >= v1486;
                        float v1489;
                        if (v1488){
                            v1489 = v1478;
                        } else {
                            v1489 = v1486;
                        }
                        v1494 = v1489; v1495 = true;
                    } else {
                        v1494 = v1478; v1495 = v1479;
                    }
                } else {
                    if (v1487){
                        v1494 = v1486; v1495 = v1487;
                    } else {
                        v1494 = v1478; v1495 = v1479;
                    }
                }
                v1478 = v1494;
                v1479 = v1495;
                v1482 += 1 ;
            }
            v1480 += 1 ;
        }
        auto v1496 = cooperative_groups::coalesced_threads();
        int v1497;
        v1497 = threadIdx.x;
        int v1498;
        v1498 = v1497 / 16;
        auto v1499 = cooperative_groups::labeled_partition(v1496,v1498);
        Closure5 v1500{};
        float v1501; bool v1502;
        Tuple2 tmp54 = cooperative_groups::reduce(v1499, Tuple2{v1478, v1479}, v1500);
        v1501 = tmp54.v0; v1502 = tmp54.v1;
        bool v1503;
        v1503 = v1502 == false;
        if (v1503){
            assert("The local reduce must be true." && v1502);
        } else {
        }
        float v1505[4];
        int v1506[4];
        int v1507;
        v1507 = 0;
        while (while_method_3(v1507)){
            int v1509;
            v1509 = 0;
            while (while_method_1(v1509)){
                assert("Tensor range check" && 0 <= v1507 && v1507 < 1);
                assert("Tensor range check" && 0 <= v1509 && v1509 < 4);
                int v1511;
                v1511 = 4 * v1507;
                int v1512;
                v1512 = v1511 + v1509;
                int v1513;
                v1513 = v1293[v1512];
                float v1514;
                v1514 = curand_uniform(&v1268);
                assert("Tensor range check" && 0 <= v1507 && v1507 < 1);
                assert("Tensor range check" && 0 <= v1509 && v1509 < 4);
                v1505[v1512] = v1514;
                v1506[v1512] = v1513;
                v1509 += 1 ;
            }
            v1507 += 1 ;
        }
        float v1515; int v1516;
        Tuple1 tmp55 = Tuple1{0.0f, 2147483647};
        v1515 = tmp55.v0; v1516 = tmp55.v1;
        int v1517;
        v1517 = 0;
        while (while_method_3(v1517)){
            int v1519;
            v1519 = 0;
            while (while_method_1(v1519)){
                assert("Tensor range check" && 0 <= v1517 && v1517 < 1);
                assert("Tensor range check" && 0 <= v1519 && v1519 < 4);
                int v1521;
                v1521 = 4 * v1517;
                int v1522;
                v1522 = v1521 + v1519;
                float v1523;
                v1523 = v1505[v1522];
                int v1524;
                v1524 = v1506[v1522];
                bool v1525;
                v1525 = v1516 < v1524;
                float v1526; int v1527;
                if (v1525){
                    v1526 = v1515; v1527 = v1516;
                } else {
                    v1526 = v1523; v1527 = v1524;
                }
                v1515 = v1526;
                v1516 = v1527;
                v1519 += 1 ;
            }
            v1517 += 1 ;
        }
        auto v1528 = cooperative_groups::coalesced_threads();
        int v1529;
        v1529 = threadIdx.x;
        int v1530;
        v1530 = v1529 / 16;
        auto v1531 = cooperative_groups::labeled_partition(v1528,v1530);
        Closure6 v1532{};
        float v1533; int v1534;
        Tuple1 tmp56 = cooperative_groups::reduce(v1531, Tuple1{v1515, v1516}, v1532);
        v1533 = tmp56.v0; v1534 = tmp56.v1;
        float v1535;
        v1535 = v1501 * v1533;
        int v1536[4];
        bool v1537[4];
        int v1538;
        v1538 = 0;
        while (while_method_3(v1538)){
            int v1540;
            v1540 = 0;
            while (while_method_1(v1540)){
                assert("Tensor range check" && 0 <= v1538 && v1538 < 1);
                assert("Tensor range check" && 0 <= v1540 && v1540 < 4);
                int v1542;
                v1542 = 4 * v1538;
                int v1543;
                v1543 = v1542 + v1540;
                float v1544;
                v1544 = v1467[v1543];
                bool v1545;
                v1545 = v1468[v1543];
                int v1546;
                v1546 = v1293[v1543];
                int v1549; bool v1550;
                if (v1545){
                    float v1547;
                    v1547 = v1544 - v1535;
                    bool v1548;
                    v1548 = v1547 >= 0.0f;
                    v1549 = v1546; v1550 = v1548;
                } else {
                    v1549 = 2147483647; v1550 = false;
                }
                assert("Tensor range check" && 0 <= v1538 && v1538 < 1);
                assert("Tensor range check" && 0 <= v1540 && v1540 < 4);
                v1536[v1543] = v1549;
                v1537[v1543] = v1550;
                v1540 += 1 ;
            }
            v1538 += 1 ;
        }
        int v1551; bool v1552;
        Tuple3 tmp57 = Tuple3{2147483647, false};
        v1551 = tmp57.v0; v1552 = tmp57.v1;
        int v1553;
        v1553 = 0;
        while (while_method_3(v1553)){
            int v1555;
            v1555 = 0;
            while (while_method_1(v1555)){
                assert("Tensor range check" && 0 <= v1553 && v1553 < 1);
                assert("Tensor range check" && 0 <= v1555 && v1555 < 4);
                int v1557;
                v1557 = 4 * v1553;
                int v1558;
                v1558 = v1557 + v1555;
                int v1559;
                v1559 = v1536[v1558];
                bool v1560;
                v1560 = v1537[v1558];
                int v1567; bool v1568;
                if (v1552){
                    if (v1560){
                        bool v1561;
                        v1561 = v1551 < v1559;
                        int v1562;
                        if (v1561){
                            v1562 = v1551;
                        } else {
                            v1562 = v1559;
                        }
                        v1567 = v1562; v1568 = true;
                    } else {
                        v1567 = v1551; v1568 = v1552;
                    }
                } else {
                    if (v1560){
                        v1567 = v1559; v1568 = v1560;
                    } else {
                        v1567 = v1551; v1568 = v1552;
                    }
                }
                v1551 = v1567;
                v1552 = v1568;
                v1555 += 1 ;
            }
            v1553 += 1 ;
        }
        auto v1569 = cooperative_groups::coalesced_threads();
        int v1570;
        v1570 = threadIdx.x;
        int v1571;
        v1571 = v1570 / 16;
        auto v1572 = cooperative_groups::labeled_partition(v1569,v1571);
        Closure7 v1573{};
        int v1574; bool v1575;
        Tuple3 tmp58 = cooperative_groups::reduce(v1572, Tuple3{v1551, v1552}, v1573);
        v1574 = tmp58.v0; v1575 = tmp58.v1;
        bool v1576;
        v1576 = v1575 == false;
        if (v1576){
            assert("The local reduce must be true." && v1575);
        } else {
        }
        assert("Tensor range check" && 0 <= v1282 && v1282 < 8);
        int v1578;
        v1578 = 0;
        while (while_method_3(v1578)){
            assert("Tensor range check" && 0 <= v1578 && v1578 < 1);
            int v1580;
            v1580 = 64 * v1578;
            int v1581;
            v1581 = v1580 + v1291;
            assert("Tensor range check" && 0 <= v1578 && v1578 < 1);
            int v1582;
            v1582 = 4 * v1578;
            int4* v1583;
            v1583 = reinterpret_cast<int4*>(v1429 + v1582);
            int4* v1584;
            v1584 = reinterpret_cast<int4*>(v15 + v1581);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1583) % 16 == 0 && reinterpret_cast<unsigned long long>(v1584) % 16 == 0);
            *v1584 = *v1583;
            v1578 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1282 && v1282 < 8);
        int v1585;
        v1585 = 16 * v1282;
        int v1586;
        v1586 = v1585 + v1274;
        v16[v1586] = v1574;
        v1282 += 24 ;
    }
    v17.sync() ;
    return ;
}
extern "C" __global__ void entry5(int * v0, float * v1, int * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int * v8, int * v9, int * v10, int * v11, int * v12, float * v13, int * v14, float * v15, int * v16) {
    auto v17 = cooperative_groups::this_grid();
    int v18;
    v18 = threadIdx.x;
    bool v19;
    v19 = 0 <= v18;
    bool v20;
    v20 = v19 == false;
    if (v20){
        assert("The index needs to be zero or positive." && v19);
    } else {
    }
    int v22;
    v22 = v18 % 32;
    int v23;
    v23 = v18 / 32;
    bool v24;
    v24 = v23 < 8;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
    } else {
    }
    assert("Tensor range check" && 0 <= v23 && v23 < 8);
    assert("Tensor range check" && 0 <= v22 && v22 < 32);
    int v27;
    v27 = 4 * v22;
    int v28;
    v28 = 128 * v23;
    int v29;
    v29 = v28 + v27;
    assert("Tensor range check" && 0 <= v23 && v23 < 8);
    assert("Tensor range check" && 0 <= v22 && v22 < 32);
    int v30;
    v30 = blockIdx.x;
    int v31;
    v31 = v30;
    while (while_method_2(v31)){
        bool v33;
        v33 = 0 <= v31;
        bool v34;
        v34 = v33 == false;
        if (v34){
            assert("The index needs to be zero or positive." && v33);
        } else {
        }
        bool v36;
        v36 = v31 < 8;
        bool v37;
        v37 = v36 == false;
        if (v37){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v36);
        } else {
        }
        assert("Tensor range check" && 0 <= v31 && v31 < 8);
        int v39;
        v39 = 1024 * v31;
        int v40;
        v40 = v39 + v29;
        int v41[4];
        int v42[4];
        int v43;
        v43 = 0;
        while (while_method_3(v43)){
            assert("Tensor range check" && 0 <= v43 && v43 < 1);
            int v45;
            v45 = 4 * v43;
            assert("Tensor range check" && 0 <= v43 && v43 < 1);
            int v46;
            v46 = 128 * v43;
            int v47;
            v47 = v46 + v40;
            int4* v48;
            v48 = reinterpret_cast<int4*>(v0 + v47);
            int4* v49;
            v49 = reinterpret_cast<int4*>(v41 + v45);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v48) % 16 == 0 && reinterpret_cast<unsigned long long>(v49) % 16 == 0);
            *v49 = *v48;
            v43 += 1 ;
        }
        int v50;
        v50 = 0;
        while (while_method_3(v50)){
            int v52;
            v52 = 0;
            while (while_method_1(v52)){
                bool v54;
                v54 = 0 <= v52;
                bool v56;
                if (v54){
                    bool v55;
                    v55 = v52 < 4;
                    v56 = v55;
                } else {
                    v56 = false;
                }
                bool v57;
                v57 = v56 == false;
                if (v57){
                    assert("The indices should be inside the range of the dimension." && v56);
                } else {
                }
                bool v59;
                v59 = 0 <= v22;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v22 < 32;
                    v61 = v60;
                } else {
                    v61 = false;
                }
                bool v62;
                v62 = v61 == false;
                if (v62){
                    assert("The indices should be inside the range of the dimension." && v61);
                } else {
                }
                int v64;
                v64 = v22 * 4;
                int v65;
                v65 = v52 + v64;
                bool v66;
                v66 = 0 <= v50;
                bool v68;
                if (v66){
                    bool v67;
                    v67 = v50 < 1;
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
                v71 = v50 * 128;
                int v72;
                v72 = v65 + v71;
                assert("Tensor range check" && 0 <= v50 && v50 < 1);
                assert("Tensor range check" && 0 <= v52 && v52 < 4);
                int v73;
                v73 = 4 * v50;
                int v74;
                v74 = v73 + v52;
                v42[v74] = v72;
                v52 += 1 ;
            }
            v50 += 1 ;
        }
        bool v75;
        v75 = 0 <= v23;
        bool v76;
        v76 = v75 && v24;
        bool v77;
        v77 = v76 == false;
        if (v77){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v76);
        } else {
        }
        bool v79;
        v79 = v33 && v36;
        bool v80;
        v80 = v79 == false;
        if (v80){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v79);
        } else {
        }
        int v82;
        v82 = v31 * 8;
        int v83;
        v83 = v82 + v23;
        assert("Tensor range check" && 0 <= v31 && v31 < 8);
        int v84;
        v84 = 0;
        while (while_method_3(v84)){
            assert("Tensor range check" && 0 <= v84 && v84 < 1);
            int v86;
            v86 = 128 * v84;
            int v87;
            v87 = v86 + v40;
            assert("Tensor range check" && 0 <= v84 && v84 < 1);
            int v88;
            v88 = 4 * v84;
            int4* v89;
            v89 = reinterpret_cast<int4*>(v41 + v88);
            int4* v90;
            v90 = reinterpret_cast<int4*>(v2 + v87);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v89) % 16 == 0 && reinterpret_cast<unsigned long long>(v90) % 16 == 0);
            *v90 = *v89;
            v84 += 1 ;
        }
        v31 += 24 ;
    }
    v17.sync() ;
    int v91;
    v91 = threadIdx.x;
    bool v92;
    v92 = 0 <= v91;
    bool v93;
    v93 = v92 == false;
    if (v93){
        assert("The index needs to be zero or positive." && v92);
    } else {
    }
    int v95;
    v95 = v91 % 32;
    int v96;
    v96 = v91 / 32;
    bool v97;
    v97 = v96 < 8;
    bool v98;
    v98 = v97 == false;
    if (v98){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v97);
    } else {
    }
    assert("Tensor range check" && 0 <= v96 && v96 < 8);
    assert("Tensor range check" && 0 <= v95 && v95 < 32);
    int v100;
    v100 = 4 * v95;
    int v101;
    v101 = 128 * v96;
    int v102;
    v102 = v101 + v100;
    assert("Tensor range check" && 0 <= v96 && v96 < 8);
    assert("Tensor range check" && 0 <= v95 && v95 < 32);
    int v103;
    v103 = blockIdx.x;
    int v104;
    v104 = v103;
    while (while_method_2(v104)){
        bool v106;
        v106 = 0 <= v104;
        bool v107;
        v107 = v106 == false;
        if (v107){
            assert("The index needs to be zero or positive." && v106);
        } else {
        }
        bool v109;
        v109 = v104 < 8;
        bool v110;
        v110 = v109 == false;
        if (v110){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v109);
        } else {
        }
        assert("Tensor range check" && 0 <= v104 && v104 < 8);
        int v112;
        v112 = 1024 * v104;
        int v113;
        v113 = v112 + v102;
        float v114[4];
        int v115[4];
        int v116;
        v116 = 0;
        while (while_method_3(v116)){
            assert("Tensor range check" && 0 <= v116 && v116 < 1);
            int v118;
            v118 = 4 * v116;
            assert("Tensor range check" && 0 <= v116 && v116 < 1);
            int v119;
            v119 = 128 * v116;
            int v120;
            v120 = v119 + v113;
            int4* v121;
            v121 = reinterpret_cast<int4*>(v1 + v120);
            int4* v122;
            v122 = reinterpret_cast<int4*>(v114 + v118);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v121) % 16 == 0 && reinterpret_cast<unsigned long long>(v122) % 16 == 0);
            *v122 = *v121;
            v116 += 1 ;
        }
        int v123;
        v123 = 0;
        while (while_method_3(v123)){
            int v125;
            v125 = 0;
            while (while_method_1(v125)){
                bool v127;
                v127 = 0 <= v125;
                bool v129;
                if (v127){
                    bool v128;
                    v128 = v125 < 4;
                    v129 = v128;
                } else {
                    v129 = false;
                }
                bool v130;
                v130 = v129 == false;
                if (v130){
                    assert("The indices should be inside the range of the dimension." && v129);
                } else {
                }
                bool v132;
                v132 = 0 <= v95;
                bool v134;
                if (v132){
                    bool v133;
                    v133 = v95 < 32;
                    v134 = v133;
                } else {
                    v134 = false;
                }
                bool v135;
                v135 = v134 == false;
                if (v135){
                    assert("The indices should be inside the range of the dimension." && v134);
                } else {
                }
                int v137;
                v137 = v95 * 4;
                int v138;
                v138 = v125 + v137;
                bool v139;
                v139 = 0 <= v123;
                bool v141;
                if (v139){
                    bool v140;
                    v140 = v123 < 1;
                    v141 = v140;
                } else {
                    v141 = false;
                }
                bool v142;
                v142 = v141 == false;
                if (v142){
                    assert("The indices should be inside the range of the dimension." && v141);
                } else {
                }
                int v144;
                v144 = v123 * 128;
                int v145;
                v145 = v138 + v144;
                assert("Tensor range check" && 0 <= v123 && v123 < 1);
                assert("Tensor range check" && 0 <= v125 && v125 < 4);
                int v146;
                v146 = 4 * v123;
                int v147;
                v147 = v146 + v125;
                v115[v147] = v145;
                v125 += 1 ;
            }
            v123 += 1 ;
        }
        bool v148;
        v148 = 0 <= v96;
        bool v149;
        v149 = v148 && v97;
        bool v150;
        v150 = v149 == false;
        if (v150){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v149);
        } else {
        }
        bool v152;
        v152 = v106 && v109;
        bool v153;
        v153 = v152 == false;
        if (v153){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v152);
        } else {
        }
        int v155;
        v155 = v104 * 8;
        int v156;
        v156 = v155 + v96;
        int v157[4];
        int v158[4];
        int v159;
        v159 = 0;
        while (while_method_3(v159)){
            int v161;
            v161 = 0;
            while (while_method_1(v161)){
                assert("Tensor range check" && 0 <= v159 && v159 < 1);
                assert("Tensor range check" && 0 <= v161 && v161 < 4);
                int v163;
                v163 = 4 * v159;
                int v164;
                v164 = v163 + v161;
                int v165;
                v165 = v115[v164];
                assert("Tensor range check" && 0 <= v159 && v159 < 1);
                assert("Tensor range check" && 0 <= v161 && v161 < 4);
                v157[v164] = v156;
                v158[v164] = v165;
                v161 += 1 ;
            }
            v159 += 1 ;
        }
        assert("Tensor range check" && 0 <= v104 && v104 < 8);
        int v166;
        v166 = 0;
        while (while_method_3(v166)){
            assert("Tensor range check" && 0 <= v166 && v166 < 1);
            int v168;
            v168 = 128 * v166;
            int v169;
            v169 = v168 + v113;
            assert("Tensor range check" && 0 <= v166 && v166 < 1);
            int v170;
            v170 = 4 * v166;
            int4* v171;
            v171 = reinterpret_cast<int4*>(v157 + v170);
            int4* v172;
            v172 = reinterpret_cast<int4*>(v9 + v169);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v171) % 16 == 0 && reinterpret_cast<unsigned long long>(v172) % 16 == 0);
            *v172 = *v171;
            int4* v173;
            v173 = reinterpret_cast<int4*>(v158 + v170);
            int4* v174;
            v174 = reinterpret_cast<int4*>(v10 + v169);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v173) % 16 == 0 && reinterpret_cast<unsigned long long>(v174) % 16 == 0);
            *v174 = *v173;
            v166 += 1 ;
        }
        v104 += 24 ;
    }
    v17.sync() ;
    int v175;
    v175 = threadIdx.x;
    bool v176;
    v176 = 0 <= v175;
    bool v177;
    v177 = v176 == false;
    if (v177){
        assert("The index needs to be zero or positive." && v176);
    } else {
    }
    int v179;
    v179 = v175 % 32;
    int v180;
    v180 = v175 / 32;
    bool v181;
    v181 = v180 < 8;
    bool v182;
    v182 = v181 == false;
    if (v182){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v181);
    } else {
    }
    assert("Tensor range check" && 0 <= v180 && v180 < 8);
    assert("Tensor range check" && 0 <= v179 && v179 < 32);
    int v184;
    v184 = 4 * v179;
    int v185;
    v185 = 128 * v180;
    int v186;
    v186 = v185 + v184;
    assert("Tensor range check" && 0 <= v180 && v180 < 8);
    int v187;
    v187 = blockIdx.x;
    int v188;
    v188 = v187;
    while (while_method_2(v188)){
        bool v190;
        v190 = 0 <= v188;
        bool v191;
        v191 = v190 == false;
        if (v191){
            assert("The index needs to be zero or positive." && v190);
        } else {
        }
        bool v193;
        v193 = v188 < 8;
        bool v194;
        v194 = v193 == false;
        if (v194){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v193);
        } else {
        }
        assert("Tensor range check" && 0 <= v188 && v188 < 8);
        int v196;
        v196 = 1024 * v188;
        int v197;
        v197 = v196 + v186;
        float v198[4];
        int v199[4];
        int v200;
        v200 = 0;
        while (while_method_3(v200)){
            assert("Tensor range check" && 0 <= v200 && v200 < 1);
            int v202;
            v202 = 4 * v200;
            assert("Tensor range check" && 0 <= v200 && v200 < 1);
            int v203;
            v203 = 128 * v200;
            int v204;
            v204 = v203 + v197;
            int4* v205;
            v205 = reinterpret_cast<int4*>(v1 + v204);
            int4* v206;
            v206 = reinterpret_cast<int4*>(v198 + v202);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v205) % 16 == 0 && reinterpret_cast<unsigned long long>(v206) % 16 == 0);
            *v206 = *v205;
            v200 += 1 ;
        }
        int v207;
        v207 = 0;
        while (while_method_3(v207)){
            int v209;
            v209 = 0;
            while (while_method_1(v209)){
                bool v211;
                v211 = 0 <= v209;
                bool v213;
                if (v211){
                    bool v212;
                    v212 = v209 < 4;
                    v213 = v212;
                } else {
                    v213 = false;
                }
                bool v214;
                v214 = v213 == false;
                if (v214){
                    assert("The indices should be inside the range of the dimension." && v213);
                } else {
                }
                bool v216;
                v216 = 0 <= v179;
                bool v218;
                if (v216){
                    bool v217;
                    v217 = v179 < 32;
                    v218 = v217;
                } else {
                    v218 = false;
                }
                bool v219;
                v219 = v218 == false;
                if (v219){
                    assert("The indices should be inside the range of the dimension." && v218);
                } else {
                }
                int v221;
                v221 = v179 * 4;
                int v222;
                v222 = v209 + v221;
                bool v223;
                v223 = 0 <= v207;
                bool v225;
                if (v223){
                    bool v224;
                    v224 = v207 < 1;
                    v225 = v224;
                } else {
                    v225 = false;
                }
                bool v226;
                v226 = v225 == false;
                if (v226){
                    assert("The indices should be inside the range of the dimension." && v225);
                } else {
                }
                int v228;
                v228 = v207 * 128;
                int v229;
                v229 = v222 + v228;
                assert("Tensor range check" && 0 <= v207 && v207 < 1);
                assert("Tensor range check" && 0 <= v209 && v209 < 4);
                int v230;
                v230 = 4 * v207;
                int v231;
                v231 = v230 + v209;
                v199[v231] = v229;
                v209 += 1 ;
            }
            v207 += 1 ;
        }
        bool v232;
        v232 = 0 <= v180;
        bool v233;
        v233 = v232 && v181;
        bool v234;
        v234 = v233 == false;
        if (v234){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v233);
        } else {
        }
        bool v236;
        v236 = v190 && v193;
        bool v237;
        v237 = v236 == false;
        if (v237){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v236);
        } else {
        }
        int v239;
        v239 = v188 * 8;
        int v240;
        v240 = v239 + v180;
        assert("Tensor range check" && 0 <= v188 && v188 < 8);
        int v241;
        v241 = 8 * v188;
        int v242;
        v242 = v241 + v180;
        v11[v242] = v240;
        v188 += 24 ;
    }
    v17.sync() ;
    int v243;
    v243 = threadIdx.x;
    bool v244;
    v244 = 0 <= v243;
    bool v245;
    v245 = v244 == false;
    if (v245){
        assert("The index needs to be zero or positive." && v244);
    } else {
    }
    int v247;
    v247 = v243 % 32;
    int v248;
    v248 = v243 / 32;
    bool v249;
    v249 = v248 < 8;
    bool v250;
    v250 = v249 == false;
    if (v250){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v249);
    } else {
    }
    assert("Tensor range check" && 0 <= v248 && v248 < 8);
    assert("Tensor range check" && 0 <= v247 && v247 < 32);
    int v252;
    v252 = 4 * v247;
    int v253;
    v253 = 128 * v248;
    int v254;
    v254 = v253 + v252;
    assert("Tensor range check" && 0 <= v248 && v248 < 8);
    assert("Tensor range check" && 0 <= v247 && v247 < 32);
    int v255;
    v255 = blockIdx.x;
    int v256;
    v256 = v255;
    while (while_method_2(v256)){
        bool v258;
        v258 = 0 <= v256;
        bool v259;
        v259 = v258 == false;
        if (v259){
            assert("The index needs to be zero or positive." && v258);
        } else {
        }
        bool v261;
        v261 = v256 < 8;
        bool v262;
        v262 = v261 == false;
        if (v262){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v261);
        } else {
        }
        assert("Tensor range check" && 0 <= v256 && v256 < 8);
        int v264;
        v264 = 1024 * v256;
        int v265;
        v265 = v264 + v254;
        float v266[4];
        int v267[4];
        int v268;
        v268 = 0;
        while (while_method_3(v268)){
            assert("Tensor range check" && 0 <= v268 && v268 < 1);
            int v270;
            v270 = 4 * v268;
            assert("Tensor range check" && 0 <= v268 && v268 < 1);
            int v271;
            v271 = 128 * v268;
            int v272;
            v272 = v271 + v265;
            int4* v273;
            v273 = reinterpret_cast<int4*>(v1 + v272);
            int4* v274;
            v274 = reinterpret_cast<int4*>(v266 + v270);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v273) % 16 == 0 && reinterpret_cast<unsigned long long>(v274) % 16 == 0);
            *v274 = *v273;
            v268 += 1 ;
        }
        int v275;
        v275 = 0;
        while (while_method_3(v275)){
            int v277;
            v277 = 0;
            while (while_method_1(v277)){
                bool v279;
                v279 = 0 <= v277;
                bool v281;
                if (v279){
                    bool v280;
                    v280 = v277 < 4;
                    v281 = v280;
                } else {
                    v281 = false;
                }
                bool v282;
                v282 = v281 == false;
                if (v282){
                    assert("The indices should be inside the range of the dimension." && v281);
                } else {
                }
                bool v284;
                v284 = 0 <= v247;
                bool v286;
                if (v284){
                    bool v285;
                    v285 = v247 < 32;
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
                int v289;
                v289 = v247 * 4;
                int v290;
                v290 = v277 + v289;
                bool v291;
                v291 = 0 <= v275;
                bool v293;
                if (v291){
                    bool v292;
                    v292 = v275 < 1;
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
                int v296;
                v296 = v275 * 128;
                int v297;
                v297 = v290 + v296;
                assert("Tensor range check" && 0 <= v275 && v275 < 1);
                assert("Tensor range check" && 0 <= v277 && v277 < 4);
                int v298;
                v298 = 4 * v275;
                int v299;
                v299 = v298 + v277;
                v267[v299] = v297;
                v277 += 1 ;
            }
            v275 += 1 ;
        }
        bool v300;
        v300 = 0 <= v248;
        bool v301;
        v301 = v300 && v249;
        bool v302;
        v302 = v301 == false;
        if (v302){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v301);
        } else {
        }
        bool v304;
        v304 = v258 && v261;
        bool v305;
        v305 = v304 == false;
        if (v305){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v304);
        } else {
        }
        int v307;
        v307 = v256 * 8;
        int v308;
        v308 = v307 + v248;
        float v309;
        v309 = 0.0f;
        int v310;
        v310 = 0;
        while (while_method_3(v310)){
            int v312;
            v312 = 0;
            while (while_method_1(v312)){
                assert("Tensor range check" && 0 <= v310 && v310 < 1);
                assert("Tensor range check" && 0 <= v312 && v312 < 4);
                int v314;
                v314 = 4 * v310;
                int v315;
                v315 = v314 + v312;
                float v316;
                v316 = v266[v315];
                float v317;
                v317 = v309 + v316;
                v309 = v317;
                v312 += 1 ;
            }
            v310 += 1 ;
        }
        auto v318 = cooperative_groups::coalesced_threads();
        int v319;
        v319 = threadIdx.x;
        int v320;
        v320 = v319 / 32;
        auto v321 = cooperative_groups::labeled_partition(v318,v320);
        Closure0 v322{};
        float v323;
        v323 = cooperative_groups::reduce(v321, v309, v322);
        float v324;
        v324 = v323 / 128.0f;
        float v325[4];
        int v326;
        v326 = 0;
        while (while_method_3(v326)){
            int v328;
            v328 = 0;
            while (while_method_1(v328)){
                assert("Tensor range check" && 0 <= v326 && v326 < 1);
                assert("Tensor range check" && 0 <= v328 && v328 < 4);
                int v330;
                v330 = 4 * v326;
                int v331;
                v331 = v330 + v328;
                float v332;
                v332 = v266[v331];
                float v333;
                v333 = v332 - v324;
                float v334;
                v334 = exp(v333);
                assert("Tensor range check" && 0 <= v326 && v326 < 1);
                assert("Tensor range check" && 0 <= v328 && v328 < 4);
                v325[v331] = v334;
                v328 += 1 ;
            }
            v326 += 1 ;
        }
        float v335;
        v335 = 0.0f;
        int v336;
        v336 = 0;
        while (while_method_3(v336)){
            int v338;
            v338 = 0;
            while (while_method_1(v338)){
                assert("Tensor range check" && 0 <= v336 && v336 < 1);
                assert("Tensor range check" && 0 <= v338 && v338 < 4);
                int v340;
                v340 = 4 * v336;
                int v341;
                v341 = v340 + v338;
                float v342;
                v342 = v325[v341];
                float v343;
                v343 = v335 + v342;
                v335 = v343;
                v338 += 1 ;
            }
            v336 += 1 ;
        }
        auto v344 = cooperative_groups::coalesced_threads();
        int v345;
        v345 = threadIdx.x;
        int v346;
        v346 = v345 / 32;
        auto v347 = cooperative_groups::labeled_partition(v344,v346);
        float v348;
        v348 = cooperative_groups::reduce(v347, v335, v322);
        float v349[4];
        int v350;
        v350 = 0;
        while (while_method_3(v350)){
            int v352;
            v352 = 0;
            while (while_method_1(v352)){
                assert("Tensor range check" && 0 <= v350 && v350 < 1);
                assert("Tensor range check" && 0 <= v352 && v352 < 4);
                int v354;
                v354 = 4 * v350;
                int v355;
                v355 = v354 + v352;
                float v356;
                v356 = v325[v355];
                float v357;
                v357 = v356 / v348;
                assert("Tensor range check" && 0 <= v350 && v350 < 1);
                assert("Tensor range check" && 0 <= v352 && v352 < 4);
                v349[v355] = v357;
                v352 += 1 ;
            }
            v350 += 1 ;
        }
        assert("Tensor range check" && 0 <= v256 && v256 < 8);
        int v358;
        v358 = 0;
        while (while_method_3(v358)){
            assert("Tensor range check" && 0 <= v358 && v358 < 1);
            int v360;
            v360 = 128 * v358;
            int v361;
            v361 = v360 + v265;
            assert("Tensor range check" && 0 <= v358 && v358 < 1);
            int v362;
            v362 = 4 * v358;
            int4* v363;
            v363 = reinterpret_cast<int4*>(v349 + v362);
            int4* v364;
            v364 = reinterpret_cast<int4*>(v3 + v361);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v363) % 16 == 0 && reinterpret_cast<unsigned long long>(v364) % 16 == 0);
            *v364 = *v363;
            v358 += 1 ;
        }
        v256 += 24 ;
    }
    v17.sync() ;
    int v365;
    v365 = threadIdx.x;
    bool v366;
    v366 = 0 <= v365;
    bool v367;
    v367 = v366 == false;
    if (v367){
        assert("The index needs to be zero or positive." && v366);
    } else {
    }
    int v369;
    v369 = v365 % 32;
    int v370;
    v370 = v365 / 32;
    bool v371;
    v371 = v370 < 8;
    bool v372;
    v372 = v371 == false;
    if (v372){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v371);
    } else {
    }
    assert("Tensor range check" && 0 <= v370 && v370 < 8);
    assert("Tensor range check" && 0 <= v369 && v369 < 32);
    int v374;
    v374 = 4 * v369;
    int v375;
    v375 = 128 * v370;
    int v376;
    v376 = v375 + v374;
    assert("Tensor range check" && 0 <= v370 && v370 < 8);
    assert("Tensor range check" && 0 <= v369 && v369 < 32);
    int v377;
    v377 = blockIdx.x;
    int v378;
    v378 = v377;
    while (while_method_2(v378)){
        bool v380;
        v380 = 0 <= v378;
        bool v381;
        v381 = v380 == false;
        if (v381){
            assert("The index needs to be zero or positive." && v380);
        } else {
        }
        bool v383;
        v383 = v378 < 8;
        bool v384;
        v384 = v383 == false;
        if (v384){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v383);
        } else {
        }
        assert("Tensor range check" && 0 <= v378 && v378 < 8);
        int v386;
        v386 = 1024 * v378;
        int v387;
        v387 = v386 + v376;
        float v388[4];
        int v389[4];
        int v390;
        v390 = 0;
        while (while_method_3(v390)){
            assert("Tensor range check" && 0 <= v390 && v390 < 1);
            int v392;
            v392 = 4 * v390;
            assert("Tensor range check" && 0 <= v390 && v390 < 1);
            int v393;
            v393 = 128 * v390;
            int v394;
            v394 = v393 + v387;
            int4* v395;
            v395 = reinterpret_cast<int4*>(v1 + v394);
            int4* v396;
            v396 = reinterpret_cast<int4*>(v388 + v392);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v395) % 16 == 0 && reinterpret_cast<unsigned long long>(v396) % 16 == 0);
            *v396 = *v395;
            v390 += 1 ;
        }
        int v397;
        v397 = 0;
        while (while_method_3(v397)){
            int v399;
            v399 = 0;
            while (while_method_1(v399)){
                bool v401;
                v401 = 0 <= v399;
                bool v403;
                if (v401){
                    bool v402;
                    v402 = v399 < 4;
                    v403 = v402;
                } else {
                    v403 = false;
                }
                bool v404;
                v404 = v403 == false;
                if (v404){
                    assert("The indices should be inside the range of the dimension." && v403);
                } else {
                }
                bool v406;
                v406 = 0 <= v369;
                bool v408;
                if (v406){
                    bool v407;
                    v407 = v369 < 32;
                    v408 = v407;
                } else {
                    v408 = false;
                }
                bool v409;
                v409 = v408 == false;
                if (v409){
                    assert("The indices should be inside the range of the dimension." && v408);
                } else {
                }
                int v411;
                v411 = v369 * 4;
                int v412;
                v412 = v399 + v411;
                bool v413;
                v413 = 0 <= v397;
                bool v415;
                if (v413){
                    bool v414;
                    v414 = v397 < 1;
                    v415 = v414;
                } else {
                    v415 = false;
                }
                bool v416;
                v416 = v415 == false;
                if (v416){
                    assert("The indices should be inside the range of the dimension." && v415);
                } else {
                }
                int v418;
                v418 = v397 * 128;
                int v419;
                v419 = v412 + v418;
                assert("Tensor range check" && 0 <= v397 && v397 < 1);
                assert("Tensor range check" && 0 <= v399 && v399 < 4);
                int v420;
                v420 = 4 * v397;
                int v421;
                v421 = v420 + v399;
                v389[v421] = v419;
                v399 += 1 ;
            }
            v397 += 1 ;
        }
        bool v422;
        v422 = 0 <= v370;
        bool v423;
        v423 = v422 && v371;
        bool v424;
        v424 = v423 == false;
        if (v424){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v423);
        } else {
        }
        bool v426;
        v426 = v380 && v383;
        bool v427;
        v427 = v426 == false;
        if (v427){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v426);
        } else {
        }
        int v429;
        v429 = v378 * 8;
        int v430;
        v430 = v429 + v370;
        float v431[4];
        int v432;
        v432 = 0;
        while (while_method_3(v432)){
            int v434;
            v434 = 0;
            while (while_method_1(v434)){
                assert("Tensor range check" && 0 <= v432 && v432 < 1);
                assert("Tensor range check" && 0 <= v434 && v434 < 4);
                int v436;
                v436 = 4 * v432;
                int v437;
                v437 = v436 + v434;
                float v438;
                v438 = v388[v437];
                float v439;
                v439 = v438 * v438;
                assert("Tensor range check" && 0 <= v432 && v432 < 1);
                assert("Tensor range check" && 0 <= v434 && v434 < 4);
                v431[v437] = v439;
                v434 += 1 ;
            }
            v432 += 1 ;
        }
        float v440;
        v440 = 0.0f;
        int v441;
        v441 = 0;
        while (while_method_3(v441)){
            int v443;
            v443 = 0;
            while (while_method_1(v443)){
                assert("Tensor range check" && 0 <= v441 && v441 < 1);
                assert("Tensor range check" && 0 <= v443 && v443 < 4);
                int v445;
                v445 = 4 * v441;
                int v446;
                v446 = v445 + v443;
                float v447;
                v447 = v431[v446];
                float v448;
                v448 = v440 + v447;
                v440 = v448;
                v443 += 1 ;
            }
            v441 += 1 ;
        }
        auto v449 = cooperative_groups::coalesced_threads();
        int v450;
        v450 = threadIdx.x;
        int v451;
        v451 = v450 / 32;
        auto v452 = cooperative_groups::labeled_partition(v449,v451);
        Closure0 v453{};
        float v454;
        v454 = cooperative_groups::reduce(v452, v440, v453);
        float v455[4];
        int v456;
        v456 = 0;
        while (while_method_3(v456)){
            int v458;
            v458 = 0;
            while (while_method_1(v458)){
                assert("Tensor range check" && 0 <= v456 && v456 < 1);
                assert("Tensor range check" && 0 <= v458 && v458 < 4);
                int v460;
                v460 = 4 * v456;
                int v461;
                v461 = v460 + v458;
                float v462;
                v462 = v388[v461];
                bool v463;
                v463 = v454 == 0.0f;
                bool v464;
                v464 = v463 != true;
                float v466;
                if (v464){
                    float v465;
                    v465 = v462 / v454;
                    v466 = v465;
                } else {
                    v466 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v456 && v456 < 1);
                assert("Tensor range check" && 0 <= v458 && v458 < 4);
                v455[v461] = v466;
                v458 += 1 ;
            }
            v456 += 1 ;
        }
        assert("Tensor range check" && 0 <= v378 && v378 < 8);
        int v467;
        v467 = 0;
        while (while_method_3(v467)){
            assert("Tensor range check" && 0 <= v467 && v467 < 1);
            int v469;
            v469 = 128 * v467;
            int v470;
            v470 = v469 + v387;
            assert("Tensor range check" && 0 <= v467 && v467 < 1);
            int v471;
            v471 = 4 * v467;
            int4* v472;
            v472 = reinterpret_cast<int4*>(v455 + v471);
            int4* v473;
            v473 = reinterpret_cast<int4*>(v7 + v470);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v472) % 16 == 0 && reinterpret_cast<unsigned long long>(v473) % 16 == 0);
            *v473 = *v472;
            v467 += 1 ;
        }
        v378 += 24 ;
    }
    v17.sync() ;
    int v474;
    v474 = threadIdx.x;
    bool v475;
    v475 = 0 <= v474;
    bool v476;
    v476 = v475 == false;
    if (v476){
        assert("The index needs to be zero or positive." && v475);
    } else {
    }
    int v478;
    v478 = v474 % 32;
    int v479;
    v479 = v474 / 32;
    bool v480;
    v480 = v479 < 8;
    bool v481;
    v481 = v480 == false;
    if (v481){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v480);
    } else {
    }
    assert("Tensor range check" && 0 <= v479 && v479 < 8);
    assert("Tensor range check" && 0 <= v478 && v478 < 32);
    int v483;
    v483 = 4 * v478;
    int v484;
    v484 = 128 * v479;
    int v485;
    v485 = v484 + v483;
    assert("Tensor range check" && 0 <= v479 && v479 < 8);
    int v486;
    v486 = blockIdx.x;
    int v487;
    v487 = v486;
    while (while_method_2(v487)){
        bool v489;
        v489 = 0 <= v487;
        bool v490;
        v490 = v489 == false;
        if (v490){
            assert("The index needs to be zero or positive." && v489);
        } else {
        }
        bool v492;
        v492 = v487 < 8;
        bool v493;
        v493 = v492 == false;
        if (v493){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v492);
        } else {
        }
        assert("Tensor range check" && 0 <= v487 && v487 < 8);
        int v495;
        v495 = 1024 * v487;
        int v496;
        v496 = v495 + v485;
        float v497[4];
        int v498[4];
        int v499;
        v499 = 0;
        while (while_method_3(v499)){
            assert("Tensor range check" && 0 <= v499 && v499 < 1);
            int v501;
            v501 = 4 * v499;
            assert("Tensor range check" && 0 <= v499 && v499 < 1);
            int v502;
            v502 = 128 * v499;
            int v503;
            v503 = v502 + v496;
            int4* v504;
            v504 = reinterpret_cast<int4*>(v1 + v503);
            int4* v505;
            v505 = reinterpret_cast<int4*>(v497 + v501);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v504) % 16 == 0 && reinterpret_cast<unsigned long long>(v505) % 16 == 0);
            *v505 = *v504;
            v499 += 1 ;
        }
        int v506;
        v506 = 0;
        while (while_method_3(v506)){
            int v508;
            v508 = 0;
            while (while_method_1(v508)){
                bool v510;
                v510 = 0 <= v508;
                bool v512;
                if (v510){
                    bool v511;
                    v511 = v508 < 4;
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
                bool v515;
                v515 = 0 <= v478;
                bool v517;
                if (v515){
                    bool v516;
                    v516 = v478 < 32;
                    v517 = v516;
                } else {
                    v517 = false;
                }
                bool v518;
                v518 = v517 == false;
                if (v518){
                    assert("The indices should be inside the range of the dimension." && v517);
                } else {
                }
                int v520;
                v520 = v478 * 4;
                int v521;
                v521 = v508 + v520;
                bool v522;
                v522 = 0 <= v506;
                bool v524;
                if (v522){
                    bool v523;
                    v523 = v506 < 1;
                    v524 = v523;
                } else {
                    v524 = false;
                }
                bool v525;
                v525 = v524 == false;
                if (v525){
                    assert("The indices should be inside the range of the dimension." && v524);
                } else {
                }
                int v527;
                v527 = v506 * 128;
                int v528;
                v528 = v521 + v527;
                assert("Tensor range check" && 0 <= v506 && v506 < 1);
                assert("Tensor range check" && 0 <= v508 && v508 < 4);
                int v529;
                v529 = 4 * v506;
                int v530;
                v530 = v529 + v508;
                v498[v530] = v528;
                v508 += 1 ;
            }
            v506 += 1 ;
        }
        bool v531;
        v531 = 0 <= v479;
        bool v532;
        v532 = v531 && v480;
        bool v533;
        v533 = v532 == false;
        if (v533){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v532);
        } else {
        }
        bool v535;
        v535 = v489 && v492;
        bool v536;
        v536 = v535 == false;
        if (v536){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v535);
        } else {
        }
        int v538;
        v538 = v487 * 8;
        int v539;
        v539 = v538 + v479;
        float v540; int v541;
        Tuple1 tmp59 = Tuple1{-1.0f / 0.0f, 0};
        v540 = tmp59.v0; v541 = tmp59.v1;
        int v542;
        v542 = 0;
        while (while_method_3(v542)){
            int v544;
            v544 = 0;
            while (while_method_1(v544)){
                assert("Tensor range check" && 0 <= v542 && v542 < 1);
                assert("Tensor range check" && 0 <= v544 && v544 < 4);
                int v546;
                v546 = 4 * v542;
                int v547;
                v547 = v546 + v544;
                float v548;
                v548 = v497[v547];
                int v549;
                v549 = v498[v547];
                bool v550;
                v550 = v540 > v548;
                float v551; int v552;
                if (v550){
                    v551 = v540; v552 = v541;
                } else {
                    v551 = v548; v552 = v549;
                }
                v540 = v551;
                v541 = v552;
                v544 += 1 ;
            }
            v542 += 1 ;
        }
        auto v553 = cooperative_groups::coalesced_threads();
        int v554;
        v554 = threadIdx.x;
        int v555;
        v555 = v554 / 32;
        auto v556 = cooperative_groups::labeled_partition(v553,v555);
        Closure1 v557{};
        float v558; int v559;
        Tuple1 tmp60 = cooperative_groups::reduce(v556, Tuple1{v540, v541}, v557);
        v558 = tmp60.v0; v559 = tmp60.v1;
        assert("Tensor range check" && 0 <= v487 && v487 < 8);
        int v560;
        v560 = 8 * v487;
        int v561;
        v561 = v560 + v479;
        v8[v561] = v559;
        v487 += 24 ;
    }
    v17.sync() ;
    int v562;
    v562 = threadIdx.x;
    bool v563;
    v563 = 0 <= v562;
    bool v564;
    v564 = v563 == false;
    if (v564){
        assert("The index needs to be zero or positive." && v563);
    } else {
    }
    int v566;
    v566 = v562 % 32;
    int v567;
    v567 = v562 / 32;
    bool v568;
    v568 = v567 < 8;
    bool v569;
    v569 = v568 == false;
    if (v569){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v568);
    } else {
    }
    assert("Tensor range check" && 0 <= v567 && v567 < 8);
    assert("Tensor range check" && 0 <= v566 && v566 < 32);
    int v571;
    v571 = 4 * v566;
    int v572;
    v572 = 128 * v567;
    int v573;
    v573 = v572 + v571;
    assert("Tensor range check" && 0 <= v567 && v567 < 8);
    assert("Tensor range check" && 0 <= v566 && v566 < 32);
    int v574;
    v574 = blockIdx.x;
    int v575;
    v575 = v574;
    while (while_method_2(v575)){
        bool v577;
        v577 = 0 <= v575;
        bool v578;
        v578 = v577 == false;
        if (v578){
            assert("The index needs to be zero or positive." && v577);
        } else {
        }
        bool v580;
        v580 = v575 < 8;
        bool v581;
        v581 = v580 == false;
        if (v581){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v580);
        } else {
        }
        assert("Tensor range check" && 0 <= v575 && v575 < 8);
        int v583;
        v583 = 1024 * v575;
        int v584;
        v584 = v583 + v573;
        float v585[4];
        int v586[4];
        int v587;
        v587 = 0;
        while (while_method_3(v587)){
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v589;
            v589 = 4 * v587;
            assert("Tensor range check" && 0 <= v587 && v587 < 1);
            int v590;
            v590 = 128 * v587;
            int v591;
            v591 = v590 + v584;
            int4* v592;
            v592 = reinterpret_cast<int4*>(v1 + v591);
            int4* v593;
            v593 = reinterpret_cast<int4*>(v585 + v589);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v592) % 16 == 0 && reinterpret_cast<unsigned long long>(v593) % 16 == 0);
            *v593 = *v592;
            v587 += 1 ;
        }
        int v594;
        v594 = 0;
        while (while_method_3(v594)){
            int v596;
            v596 = 0;
            while (while_method_1(v596)){
                bool v598;
                v598 = 0 <= v596;
                bool v600;
                if (v598){
                    bool v599;
                    v599 = v596 < 4;
                    v600 = v599;
                } else {
                    v600 = false;
                }
                bool v601;
                v601 = v600 == false;
                if (v601){
                    assert("The indices should be inside the range of the dimension." && v600);
                } else {
                }
                bool v603;
                v603 = 0 <= v566;
                bool v605;
                if (v603){
                    bool v604;
                    v604 = v566 < 32;
                    v605 = v604;
                } else {
                    v605 = false;
                }
                bool v606;
                v606 = v605 == false;
                if (v606){
                    assert("The indices should be inside the range of the dimension." && v605);
                } else {
                }
                int v608;
                v608 = v566 * 4;
                int v609;
                v609 = v596 + v608;
                bool v610;
                v610 = 0 <= v594;
                bool v612;
                if (v610){
                    bool v611;
                    v611 = v594 < 1;
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
                v615 = v594 * 128;
                int v616;
                v616 = v609 + v615;
                assert("Tensor range check" && 0 <= v594 && v594 < 1);
                assert("Tensor range check" && 0 <= v596 && v596 < 4);
                int v617;
                v617 = 4 * v594;
                int v618;
                v618 = v617 + v596;
                v586[v618] = v616;
                v596 += 1 ;
            }
            v594 += 1 ;
        }
        bool v619;
        v619 = 0 <= v567;
        bool v620;
        v620 = v619 && v568;
        bool v621;
        v621 = v620 == false;
        if (v621){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v620);
        } else {
        }
        bool v623;
        v623 = v577 && v580;
        bool v624;
        v624 = v623 == false;
        if (v624){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v623);
        } else {
        }
        int v626;
        v626 = v575 * 8;
        int v627;
        v627 = v626 + v567;
        float v628;
        v628 = 0.0f;
        int v629;
        v629 = 0;
        while (while_method_3(v629)){
            int v631;
            v631 = 0;
            while (while_method_1(v631)){
                assert("Tensor range check" && 0 <= v629 && v629 < 1);
                assert("Tensor range check" && 0 <= v631 && v631 < 4);
                int v633;
                v633 = 4 * v629;
                int v634;
                v634 = v633 + v631;
                float v635;
                v635 = v585[v634];
                float v636;
                v636 = v628 + v635;
                v628 = v636;
                v631 += 1 ;
            }
            v629 += 1 ;
        }
        auto v637 = cooperative_groups::coalesced_threads();
        int v638;
        v638 = threadIdx.x;
        int v639;
        v639 = v638 / 32;
        auto v640 = cooperative_groups::labeled_partition(v637,v639);
        Closure0 v641{};
        float v642;
        v642 = cooperative_groups::reduce(v640, v628, v641);
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
                v651 = v585[v650];
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
        v667 = cooperative_groups::reduce(v666, v654, v641);
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
        assert("Tensor range check" && 0 <= v575 && v575 < 8);
        int v706;
        v706 = 0;
        while (while_method_3(v706)){
            assert("Tensor range check" && 0 <= v706 && v706 < 1);
            int v708;
            v708 = 128 * v706;
            int v709;
            v709 = v708 + v584;
            assert("Tensor range check" && 0 <= v706 && v706 < 1);
            int v710;
            v710 = 4 * v706;
            int4* v711;
            v711 = reinterpret_cast<int4*>(v668 + v710);
            int4* v712;
            v712 = reinterpret_cast<int4*>(v5 + v709);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v711) % 16 == 0 && reinterpret_cast<unsigned long long>(v712) % 16 == 0);
            *v712 = *v711;
            int4* v713;
            v713 = reinterpret_cast<int4*>(v677 + v710);
            int4* v714;
            v714 = reinterpret_cast<int4*>(v6 + v709);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v713) % 16 == 0 && reinterpret_cast<unsigned long long>(v714) % 16 == 0);
            *v714 = *v713;
            v706 += 1 ;
        }
        v575 += 24 ;
    }
    v17.sync() ;
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
    v727 = blockIdx.x;
    int v728;
    v728 = v727;
    while (while_method_2(v728)){
        bool v730;
        v730 = 0 <= v728;
        bool v731;
        v731 = v730 == false;
        if (v731){
            assert("The index needs to be zero or positive." && v730);
        } else {
        }
        bool v733;
        v733 = v728 < 8;
        bool v734;
        v734 = v733 == false;
        if (v734){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v733);
        } else {
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v736;
        v736 = 1024 * v728;
        int v737;
        v737 = v736 + v726;
        int v738[4];
        int v739[4];
        int v740;
        v740 = 0;
        while (while_method_3(v740)){
            assert("Tensor range check" && 0 <= v740 && v740 < 1);
            int v742;
            v742 = 4 * v740;
            assert("Tensor range check" && 0 <= v740 && v740 < 1);
            int v743;
            v743 = 128 * v740;
            int v744;
            v744 = v743 + v737;
            int4* v745;
            v745 = reinterpret_cast<int4*>(v0 + v744);
            int4* v746;
            v746 = reinterpret_cast<int4*>(v738 + v742);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v745) % 16 == 0 && reinterpret_cast<unsigned long long>(v746) % 16 == 0);
            *v746 = *v745;
            v740 += 1 ;
        }
        int v747;
        v747 = 0;
        while (while_method_3(v747)){
            int v749;
            v749 = 0;
            while (while_method_1(v749)){
                bool v751;
                v751 = 0 <= v749;
                bool v753;
                if (v751){
                    bool v752;
                    v752 = v749 < 4;
                    v753 = v752;
                } else {
                    v753 = false;
                }
                bool v754;
                v754 = v753 == false;
                if (v754){
                    assert("The indices should be inside the range of the dimension." && v753);
                } else {
                }
                bool v756;
                v756 = 0 <= v719;
                bool v758;
                if (v756){
                    bool v757;
                    v757 = v719 < 32;
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
                v761 = v719 * 4;
                int v762;
                v762 = v749 + v761;
                bool v763;
                v763 = 0 <= v747;
                bool v765;
                if (v763){
                    bool v764;
                    v764 = v747 < 1;
                    v765 = v764;
                } else {
                    v765 = false;
                }
                bool v766;
                v766 = v765 == false;
                if (v766){
                    assert("The indices should be inside the range of the dimension." && v765);
                } else {
                }
                int v768;
                v768 = v747 * 128;
                int v769;
                v769 = v762 + v768;
                assert("Tensor range check" && 0 <= v747 && v747 < 1);
                assert("Tensor range check" && 0 <= v749 && v749 < 4);
                int v770;
                v770 = 4 * v747;
                int v771;
                v771 = v770 + v749;
                v739[v771] = v769;
                v749 += 1 ;
            }
            v747 += 1 ;
        }
        bool v772;
        v772 = 0 <= v720;
        bool v773;
        v773 = v772 && v721;
        bool v774;
        v774 = v773 == false;
        if (v774){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v773);
        } else {
        }
        bool v776;
        v776 = v730 && v733;
        bool v777;
        v777 = v776 == false;
        if (v777){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v776);
        } else {
        }
        int v779;
        v779 = v728 * 8;
        int v780;
        v780 = v779 + v720;
        int v781[4];
        int v782;
        v782 = 0;
        int v783;
        v783 = 0;
        while (while_method_3(v783)){
            assert("Tensor range check" && 0 <= v783 && v783 < 1);
            int v785;
            v785 = 4 * v783;
            assert("Tensor range check" && 0 <= v783 && v783 < 1);
            int v786;
            v786 = 0;
            int v787;
            v787 = 0;
            while (while_method_1(v787)){
                assert("Tensor range check" && 0 <= v787 && v787 < 4);
                int v789;
                v789 = v787 + v785;
                int v790;
                v790 = v738[v789];
                int v791;
                v791 = v786 + v790;
                v786 = v791;
                v787 += 1 ;
            }
            auto v792 = cooperative_groups::coalesced_threads();
            int v793;
            v793 = threadIdx.x;
            int v794;
            v794 = v793 / 32;
            auto v795 = cooperative_groups::labeled_partition(v792,v794);
            Closure3 v796{};
            int v797;
            v797 = cooperative_groups::inclusive_scan(v795, v786, v796);
            int v798;
            v798 = v795.shfl_up(v797,1);
            bool v799;
            v799 = v795.thread_rank() == 0;
            int v800;
            if (v799){
                v800 = 0;
            } else {
                v800 = v798;
            }
            int v801;
            v801 = v795.shfl(v797,v795.num_threads()-1);
            int v802;
            v802 = v782 + v800;
            int v803;
            v803 = v802;
            int v804;
            v804 = 0;
            while (while_method_1(v804)){
                assert("Tensor range check" && 0 <= v804 && v804 < 4);
                int v806;
                v806 = v804 + v785;
                int v807;
                v807 = v738[v806];
                assert("Tensor range check" && 0 <= v804 && v804 < 4);
                v781[v806] = v803;
                int v808;
                v808 = v803 + v807;
                v803 = v808;
                v804 += 1 ;
            }
            int v809;
            v809 = v782 + v801;
            v782 = v809;
            v783 += 1 ;
        }
        assert("Tensor range check" && 0 <= v728 && v728 < 8);
        int v810;
        v810 = 0;
        while (while_method_3(v810)){
            assert("Tensor range check" && 0 <= v810 && v810 < 1);
            int v812;
            v812 = 128 * v810;
            int v813;
            v813 = v812 + v737;
            assert("Tensor range check" && 0 <= v810 && v810 < 1);
            int v814;
            v814 = 4 * v810;
            int4* v815;
            v815 = reinterpret_cast<int4*>(v781 + v814);
            int4* v816;
            v816 = reinterpret_cast<int4*>(v12 + v813);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v815) % 16 == 0 && reinterpret_cast<unsigned long long>(v816) % 16 == 0);
            *v816 = *v815;
            v810 += 1 ;
        }
        v728 += 24 ;
    }
    v17.sync() ;
    int v817;
    v817 = threadIdx.x;
    bool v818;
    v818 = 0 <= v817;
    bool v819;
    v819 = v818 == false;
    if (v819){
        assert("The index needs to be zero or positive." && v818);
    } else {
    }
    int v821;
    v821 = v817 % 32;
    int v822;
    v822 = v817 / 32;
    bool v823;
    v823 = v822 < 8;
    bool v824;
    v824 = v823 == false;
    if (v824){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v823);
    } else {
    }
    assert("Tensor range check" && 0 <= v822 && v822 < 8);
    assert("Tensor range check" && 0 <= v821 && v821 < 32);
    int v826;
    v826 = 4 * v821;
    int v827;
    v827 = 128 * v822;
    int v828;
    v828 = v827 + v826;
    assert("Tensor range check" && 0 <= v822 && v822 < 8);
    assert("Tensor range check" && 0 <= v821 && v821 < 32);
    int v829;
    v829 = blockIdx.x;
    int v830;
    v830 = v829;
    while (while_method_2(v830)){
        bool v832;
        v832 = 0 <= v830;
        bool v833;
        v833 = v832 == false;
        if (v833){
            assert("The index needs to be zero or positive." && v832);
        } else {
        }
        bool v835;
        v835 = v830 < 8;
        bool v836;
        v836 = v835 == false;
        if (v836){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v835);
        } else {
        }
        assert("Tensor range check" && 0 <= v830 && v830 < 8);
        int v838;
        v838 = 1024 * v830;
        int v839;
        v839 = v838 + v828;
        float v840[4];
        int v841[4];
        int v842;
        v842 = 0;
        while (while_method_3(v842)){
            assert("Tensor range check" && 0 <= v842 && v842 < 1);
            int v844;
            v844 = 4 * v842;
            assert("Tensor range check" && 0 <= v842 && v842 < 1);
            int v845;
            v845 = 128 * v842;
            int v846;
            v846 = v845 + v839;
            int4* v847;
            v847 = reinterpret_cast<int4*>(v1 + v846);
            int4* v848;
            v848 = reinterpret_cast<int4*>(v840 + v844);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v847) % 16 == 0 && reinterpret_cast<unsigned long long>(v848) % 16 == 0);
            *v848 = *v847;
            v842 += 1 ;
        }
        int v849;
        v849 = 0;
        while (while_method_3(v849)){
            int v851;
            v851 = 0;
            while (while_method_1(v851)){
                bool v853;
                v853 = 0 <= v851;
                bool v855;
                if (v853){
                    bool v854;
                    v854 = v851 < 4;
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
                bool v858;
                v858 = 0 <= v821;
                bool v860;
                if (v858){
                    bool v859;
                    v859 = v821 < 32;
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
                int v863;
                v863 = v821 * 4;
                int v864;
                v864 = v851 + v863;
                bool v865;
                v865 = 0 <= v849;
                bool v867;
                if (v865){
                    bool v866;
                    v866 = v849 < 1;
                    v867 = v866;
                } else {
                    v867 = false;
                }
                bool v868;
                v868 = v867 == false;
                if (v868){
                    assert("The indices should be inside the range of the dimension." && v867);
                } else {
                }
                int v870;
                v870 = v849 * 128;
                int v871;
                v871 = v864 + v870;
                assert("Tensor range check" && 0 <= v849 && v849 < 1);
                assert("Tensor range check" && 0 <= v851 && v851 < 4);
                int v872;
                v872 = 4 * v849;
                int v873;
                v873 = v872 + v851;
                v841[v873] = v871;
                v851 += 1 ;
            }
            v849 += 1 ;
        }
        bool v874;
        v874 = 0 <= v822;
        bool v875;
        v875 = v874 && v823;
        bool v876;
        v876 = v875 == false;
        if (v876){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v875);
        } else {
        }
        bool v878;
        v878 = v832 && v835;
        bool v879;
        v879 = v878 == false;
        if (v879){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v878);
        } else {
        }
        int v881;
        v881 = v830 * 8;
        int v882;
        v882 = v881 + v822;
        bool v883[4];
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
                v890 = v840[v889];
                int v891;
                v891 = v841[v889];
                bool v892;
                v892 = v891 < 4;
                assert("Tensor range check" && 0 <= v884 && v884 < 1);
                assert("Tensor range check" && 0 <= v886 && v886 < 4);
                v883[v889] = v892;
                v886 += 1 ;
            }
            v884 += 1 ;
        }
        float v893[4];
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
                v900 = v840[v899];
                bool v901;
                v901 = v883[v899];
                float v902;
                if (v901){
                    v902 = v900;
                } else {
                    v902 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v894 && v894 < 1);
                assert("Tensor range check" && 0 <= v896 && v896 < 4);
                v893[v899] = v902;
                v896 += 1 ;
            }
            v894 += 1 ;
        }
        float v903;
        v903 = 0.0f;
        int v904;
        v904 = 0;
        while (while_method_3(v904)){
            int v906;
            v906 = 0;
            while (while_method_1(v906)){
                assert("Tensor range check" && 0 <= v904 && v904 < 1);
                assert("Tensor range check" && 0 <= v906 && v906 < 4);
                int v908;
                v908 = 4 * v904;
                int v909;
                v909 = v908 + v906;
                float v910;
                v910 = v893[v909];
                float v911;
                v911 = v903 + v910;
                v903 = v911;
                v906 += 1 ;
            }
            v904 += 1 ;
        }
        auto v912 = cooperative_groups::coalesced_threads();
        int v913;
        v913 = threadIdx.x;
        int v914;
        v914 = v913 / 32;
        auto v915 = cooperative_groups::labeled_partition(v912,v914);
        Closure0 v916{};
        float v917;
        v917 = cooperative_groups::reduce(v915, v903, v916);
        int v918[4];
        int v919;
        v919 = 0;
        while (while_method_3(v919)){
            int v921;
            v921 = 0;
            while (while_method_1(v921)){
                assert("Tensor range check" && 0 <= v919 && v919 < 1);
                assert("Tensor range check" && 0 <= v921 && v921 < 4);
                int v923;
                v923 = 4 * v919;
                int v924;
                v924 = v923 + v921;
                bool v925;
                v925 = v883[v924];
                int v926;
                if (v925){
                    v926 = 1;
                } else {
                    v926 = 0;
                }
                assert("Tensor range check" && 0 <= v919 && v919 < 1);
                assert("Tensor range check" && 0 <= v921 && v921 < 4);
                v918[v924] = v926;
                v921 += 1 ;
            }
            v919 += 1 ;
        }
        int v927;
        v927 = 0;
        int v928;
        v928 = 0;
        while (while_method_3(v928)){
            int v930;
            v930 = 0;
            while (while_method_1(v930)){
                assert("Tensor range check" && 0 <= v928 && v928 < 1);
                assert("Tensor range check" && 0 <= v930 && v930 < 4);
                int v932;
                v932 = 4 * v928;
                int v933;
                v933 = v932 + v930;
                int v934;
                v934 = v918[v933];
                int v935;
                v935 = v927 + v934;
                v927 = v935;
                v930 += 1 ;
            }
            v928 += 1 ;
        }
        auto v936 = cooperative_groups::coalesced_threads();
        int v937;
        v937 = threadIdx.x;
        int v938;
        v938 = v937 / 32;
        auto v939 = cooperative_groups::labeled_partition(v936,v938);
        Closure4 v940{};
        int v941;
        v941 = cooperative_groups::reduce(v939, v927, v940);
        float v942;
        v942 = (float)v941;
        float v943;
        v943 = v917 / v942;
        float v944[4];
        int v945;
        v945 = 0;
        while (while_method_3(v945)){
            int v947;
            v947 = 0;
            while (while_method_1(v947)){
                assert("Tensor range check" && 0 <= v945 && v945 < 1);
                assert("Tensor range check" && 0 <= v947 && v947 < 4);
                int v949;
                v949 = 4 * v945;
                int v950;
                v950 = v949 + v947;
                float v951;
                v951 = v840[v950];
                bool v952;
                v952 = v883[v950];
                float v953;
                if (v952){
                    v953 = v951;
                } else {
                    v953 = -1.0f / 0.0f;
                }
                float v954;
                v954 = v953 - v943;
                float v955;
                v955 = exp(v954);
                bool v956;
                v956 = v955 < 1.0f / 0.0f;
                bool v957;
                v957 = v956 == false;
                if (v957){
                    assert("The softmax values must not grow too large." && v956);
                } else {
                }
                bool v959;
                v959 = isnan(v955);
                bool v960;
                v960 = v959 == false;
                bool v961;
                v961 = v960 == false;
                if (v961){
                    assert("The softmax values must not be nans." && v960);
                } else {
                }
                assert("Tensor range check" && 0 <= v945 && v945 < 1);
                assert("Tensor range check" && 0 <= v947 && v947 < 4);
                v944[v950] = v955;
                v947 += 1 ;
            }
            v945 += 1 ;
        }
        float v963;
        v963 = 0.0f;
        int v964;
        v964 = 0;
        while (while_method_3(v964)){
            int v966;
            v966 = 0;
            while (while_method_1(v966)){
                assert("Tensor range check" && 0 <= v964 && v964 < 1);
                assert("Tensor range check" && 0 <= v966 && v966 < 4);
                int v968;
                v968 = 4 * v964;
                int v969;
                v969 = v968 + v966;
                float v970;
                v970 = v944[v969];
                float v971;
                v971 = v963 + v970;
                v963 = v971;
                v966 += 1 ;
            }
            v964 += 1 ;
        }
        auto v972 = cooperative_groups::coalesced_threads();
        int v973;
        v973 = threadIdx.x;
        int v974;
        v974 = v973 / 32;
        auto v975 = cooperative_groups::labeled_partition(v972,v974);
        float v976;
        v976 = cooperative_groups::reduce(v975, v963, v916);
        float v977[4];
        int v978;
        v978 = 0;
        while (while_method_3(v978)){
            int v980;
            v980 = 0;
            while (while_method_1(v980)){
                assert("Tensor range check" && 0 <= v978 && v978 < 1);
                assert("Tensor range check" && 0 <= v980 && v980 < 4);
                int v982;
                v982 = 4 * v978;
                int v983;
                v983 = v982 + v980;
                float v984;
                v984 = v944[v983];
                float v985;
                v985 = v984 / v976;
                assert("Tensor range check" && 0 <= v978 && v978 < 1);
                assert("Tensor range check" && 0 <= v980 && v980 < 4);
                v977[v983] = v985;
                v980 += 1 ;
            }
            v978 += 1 ;
        }
        assert("Tensor range check" && 0 <= v830 && v830 < 8);
        int v986;
        v986 = 0;
        while (while_method_3(v986)){
            assert("Tensor range check" && 0 <= v986 && v986 < 1);
            int v988;
            v988 = 128 * v986;
            int v989;
            v989 = v988 + v839;
            assert("Tensor range check" && 0 <= v986 && v986 < 1);
            int v990;
            v990 = 4 * v986;
            int4* v991;
            v991 = reinterpret_cast<int4*>(v977 + v990);
            int4* v992;
            v992 = reinterpret_cast<int4*>(v4 + v989);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v991) % 16 == 0 && reinterpret_cast<unsigned long long>(v992) % 16 == 0);
            *v992 = *v991;
            v986 += 1 ;
        }
        v830 += 24 ;
    }
    v17.sync() ;
    int v993;
    v993 = threadIdx.x;
    int v994;
    v994 = blockIdx.x;
    int v995;
    v995 = v994 * 256;
    int v996;
    v996 = v993 + v995;
    unsigned long long v997;
    v997 = (unsigned long long)v996;
    curandStatePhilox4_32_10_t v998;
    curand_init(12344321ull,v997,0ull,&v998);
    int v999;
    v999 = threadIdx.x;
    bool v1000;
    v1000 = 0 <= v999;
    bool v1001;
    v1001 = v1000 == false;
    if (v1001){
        assert("The index needs to be zero or positive." && v1000);
    } else {
    }
    int v1003;
    v1003 = v999 % 32;
    int v1004;
    v1004 = v999 / 32;
    bool v1005;
    v1005 = v1004 < 8;
    bool v1006;
    v1006 = v1005 == false;
    if (v1006){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1005);
    } else {
    }
    assert("Tensor range check" && 0 <= v1004 && v1004 < 8);
    assert("Tensor range check" && 0 <= v1003 && v1003 < 32);
    int v1008;
    v1008 = 4 * v1003;
    int v1009;
    v1009 = 128 * v1004;
    int v1010;
    v1010 = v1009 + v1008;
    assert("Tensor range check" && 0 <= v1004 && v1004 < 8);
    assert("Tensor range check" && 0 <= v1003 && v1003 < 32);
    assert("Tensor range check" && 0 <= v1004 && v1004 < 8);
    int v1011;
    v1011 = blockIdx.x;
    int v1012;
    v1012 = v1011;
    while (while_method_2(v1012)){
        bool v1014;
        v1014 = 0 <= v1012;
        bool v1015;
        v1015 = v1014 == false;
        if (v1015){
            assert("The index needs to be zero or positive." && v1014);
        } else {
        }
        bool v1017;
        v1017 = v1012 < 8;
        bool v1018;
        v1018 = v1017 == false;
        if (v1018){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1017);
        } else {
        }
        assert("Tensor range check" && 0 <= v1012 && v1012 < 8);
        int v1020;
        v1020 = 1024 * v1012;
        int v1021;
        v1021 = v1020 + v1010;
        float v1022[4];
        int v1023[4];
        int v1024;
        v1024 = 0;
        while (while_method_3(v1024)){
            assert("Tensor range check" && 0 <= v1024 && v1024 < 1);
            int v1026;
            v1026 = 4 * v1024;
            assert("Tensor range check" && 0 <= v1024 && v1024 < 1);
            int v1027;
            v1027 = 128 * v1024;
            int v1028;
            v1028 = v1027 + v1021;
            int4* v1029;
            v1029 = reinterpret_cast<int4*>(v1 + v1028);
            int4* v1030;
            v1030 = reinterpret_cast<int4*>(v1022 + v1026);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1029) % 16 == 0 && reinterpret_cast<unsigned long long>(v1030) % 16 == 0);
            *v1030 = *v1029;
            v1024 += 1 ;
        }
        int v1031;
        v1031 = 0;
        while (while_method_3(v1031)){
            int v1033;
            v1033 = 0;
            while (while_method_1(v1033)){
                bool v1035;
                v1035 = 0 <= v1033;
                bool v1037;
                if (v1035){
                    bool v1036;
                    v1036 = v1033 < 4;
                    v1037 = v1036;
                } else {
                    v1037 = false;
                }
                bool v1038;
                v1038 = v1037 == false;
                if (v1038){
                    assert("The indices should be inside the range of the dimension." && v1037);
                } else {
                }
                bool v1040;
                v1040 = 0 <= v1003;
                bool v1042;
                if (v1040){
                    bool v1041;
                    v1041 = v1003 < 32;
                    v1042 = v1041;
                } else {
                    v1042 = false;
                }
                bool v1043;
                v1043 = v1042 == false;
                if (v1043){
                    assert("The indices should be inside the range of the dimension." && v1042);
                } else {
                }
                int v1045;
                v1045 = v1003 * 4;
                int v1046;
                v1046 = v1033 + v1045;
                bool v1047;
                v1047 = 0 <= v1031;
                bool v1049;
                if (v1047){
                    bool v1048;
                    v1048 = v1031 < 1;
                    v1049 = v1048;
                } else {
                    v1049 = false;
                }
                bool v1050;
                v1050 = v1049 == false;
                if (v1050){
                    assert("The indices should be inside the range of the dimension." && v1049);
                } else {
                }
                int v1052;
                v1052 = v1031 * 128;
                int v1053;
                v1053 = v1046 + v1052;
                assert("Tensor range check" && 0 <= v1031 && v1031 < 1);
                assert("Tensor range check" && 0 <= v1033 && v1033 < 4);
                int v1054;
                v1054 = 4 * v1031;
                int v1055;
                v1055 = v1054 + v1033;
                v1023[v1055] = v1053;
                v1033 += 1 ;
            }
            v1031 += 1 ;
        }
        bool v1056;
        v1056 = 0 <= v1004;
        bool v1057;
        v1057 = v1056 && v1005;
        bool v1058;
        v1058 = v1057 == false;
        if (v1058){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1057);
        } else {
        }
        bool v1060;
        v1060 = v1014 && v1017;
        bool v1061;
        v1061 = v1060 == false;
        if (v1061){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1060);
        } else {
        }
        int v1063;
        v1063 = v1012 * 8;
        int v1064;
        v1064 = v1063 + v1004;
        float v1065;
        v1065 = 0.0f;
        int v1066;
        v1066 = 0;
        while (while_method_3(v1066)){
            int v1068;
            v1068 = 0;
            while (while_method_1(v1068)){
                assert("Tensor range check" && 0 <= v1066 && v1066 < 1);
                assert("Tensor range check" && 0 <= v1068 && v1068 < 4);
                int v1070;
                v1070 = 4 * v1066;
                int v1071;
                v1071 = v1070 + v1068;
                float v1072;
                v1072 = v1022[v1071];
                float v1073;
                v1073 = v1065 + v1072;
                v1065 = v1073;
                v1068 += 1 ;
            }
            v1066 += 1 ;
        }
        auto v1074 = cooperative_groups::coalesced_threads();
        int v1075;
        v1075 = threadIdx.x;
        int v1076;
        v1076 = v1075 / 32;
        auto v1077 = cooperative_groups::labeled_partition(v1074,v1076);
        Closure0 v1078{};
        float v1079;
        v1079 = cooperative_groups::reduce(v1077, v1065, v1078);
        float v1080;
        v1080 = v1079 / 128.0f;
        float v1081[4];
        int v1082;
        v1082 = 0;
        while (while_method_3(v1082)){
            int v1084;
            v1084 = 0;
            while (while_method_1(v1084)){
                assert("Tensor range check" && 0 <= v1082 && v1082 < 1);
                assert("Tensor range check" && 0 <= v1084 && v1084 < 4);
                int v1086;
                v1086 = 4 * v1082;
                int v1087;
                v1087 = v1086 + v1084;
                float v1088;
                v1088 = v1022[v1087];
                float v1089;
                v1089 = v1088 - v1080;
                float v1090;
                v1090 = exp(v1089);
                assert("Tensor range check" && 0 <= v1082 && v1082 < 1);
                assert("Tensor range check" && 0 <= v1084 && v1084 < 4);
                v1081[v1087] = v1090;
                v1084 += 1 ;
            }
            v1082 += 1 ;
        }
        float v1091;
        v1091 = 0.0f;
        int v1092;
        v1092 = 0;
        while (while_method_3(v1092)){
            int v1094;
            v1094 = 0;
            while (while_method_1(v1094)){
                assert("Tensor range check" && 0 <= v1092 && v1092 < 1);
                assert("Tensor range check" && 0 <= v1094 && v1094 < 4);
                int v1096;
                v1096 = 4 * v1092;
                int v1097;
                v1097 = v1096 + v1094;
                float v1098;
                v1098 = v1081[v1097];
                float v1099;
                v1099 = v1091 + v1098;
                v1091 = v1099;
                v1094 += 1 ;
            }
            v1092 += 1 ;
        }
        auto v1100 = cooperative_groups::coalesced_threads();
        int v1101;
        v1101 = threadIdx.x;
        int v1102;
        v1102 = v1101 / 32;
        auto v1103 = cooperative_groups::labeled_partition(v1100,v1102);
        float v1104;
        v1104 = cooperative_groups::reduce(v1103, v1091, v1078);
        float v1105[4];
        int v1106;
        v1106 = 0;
        while (while_method_3(v1106)){
            int v1108;
            v1108 = 0;
            while (while_method_1(v1108)){
                assert("Tensor range check" && 0 <= v1106 && v1106 < 1);
                assert("Tensor range check" && 0 <= v1108 && v1108 < 4);
                int v1110;
                v1110 = 4 * v1106;
                int v1111;
                v1111 = v1110 + v1108;
                float v1112;
                v1112 = v1081[v1111];
                float v1113;
                v1113 = v1112 / v1104;
                assert("Tensor range check" && 0 <= v1106 && v1106 < 1);
                assert("Tensor range check" && 0 <= v1108 && v1108 < 4);
                v1105[v1111] = v1113;
                v1108 += 1 ;
            }
            v1106 += 1 ;
        }
        float v1114[4];
        float v1115;
        v1115 = 0.0f;
        int v1116;
        v1116 = 0;
        while (while_method_3(v1116)){
            assert("Tensor range check" && 0 <= v1116 && v1116 < 1);
            int v1118;
            v1118 = 4 * v1116;
            assert("Tensor range check" && 0 <= v1116 && v1116 < 1);
            float v1119;
            v1119 = 0.0f;
            int v1120;
            v1120 = 0;
            while (while_method_1(v1120)){
                assert("Tensor range check" && 0 <= v1120 && v1120 < 4);
                int v1122;
                v1122 = v1120 + v1118;
                float v1123;
                v1123 = v1105[v1122];
                float v1124;
                v1124 = v1119 + v1123;
                v1119 = v1124;
                v1120 += 1 ;
            }
            auto v1125 = cooperative_groups::coalesced_threads();
            int v1126;
            v1126 = threadIdx.x;
            int v1127;
            v1127 = v1126 / 32;
            auto v1128 = cooperative_groups::labeled_partition(v1125,v1127);
            Closure2 v1129{};
            float v1130;
            v1130 = cooperative_groups::inclusive_scan(v1128, v1119, v1129);
            float v1131;
            v1131 = v1128.shfl_up(v1130,1);
            bool v1132;
            v1132 = v1128.thread_rank() == 0;
            float v1133;
            if (v1132){
                v1133 = 0.0f;
            } else {
                v1133 = v1131;
            }
            float v1134;
            v1134 = v1128.shfl(v1130,v1128.num_threads()-1);
            float v1135;
            v1135 = v1115 + v1133;
            float v1136;
            v1136 = v1135;
            int v1137;
            v1137 = 0;
            while (while_method_1(v1137)){
                assert("Tensor range check" && 0 <= v1137 && v1137 < 4);
                int v1139;
                v1139 = v1137 + v1118;
                float v1140;
                v1140 = v1105[v1139];
                float v1141;
                v1141 = v1136 + v1140;
                assert("Tensor range check" && 0 <= v1137 && v1137 < 4);
                v1114[v1139] = v1141;
                v1136 = v1141;
                v1137 += 1 ;
            }
            float v1142;
            v1142 = v1115 + v1134;
            v1115 = v1142;
            v1116 += 1 ;
        }
        float v1143[4];
        bool v1144[4];
        int v1145;
        v1145 = 0;
        while (while_method_3(v1145)){
            int v1147;
            v1147 = 0;
            while (while_method_1(v1147)){
                assert("Tensor range check" && 0 <= v1145 && v1145 < 1);
                assert("Tensor range check" && 0 <= v1147 && v1147 < 4);
                int v1149;
                v1149 = 4 * v1145;
                int v1150;
                v1150 = v1149 + v1147;
                float v1151;
                v1151 = v1114[v1150];
                float v1152;
                v1152 = v1105[v1150];
                bool v1153;
                v1153 = v1152 > 0.0f;
                assert("Tensor range check" && 0 <= v1145 && v1145 < 1);
                assert("Tensor range check" && 0 <= v1147 && v1147 < 4);
                v1143[v1150] = v1151;
                v1144[v1150] = v1153;
                v1147 += 1 ;
            }
            v1145 += 1 ;
        }
        float v1154; bool v1155;
        Tuple2 tmp61 = Tuple2{-1.0f / 0.0f, false};
        v1154 = tmp61.v0; v1155 = tmp61.v1;
        int v1156;
        v1156 = 0;
        while (while_method_3(v1156)){
            int v1158;
            v1158 = 0;
            while (while_method_1(v1158)){
                assert("Tensor range check" && 0 <= v1156 && v1156 < 1);
                assert("Tensor range check" && 0 <= v1158 && v1158 < 4);
                int v1160;
                v1160 = 4 * v1156;
                int v1161;
                v1161 = v1160 + v1158;
                float v1162;
                v1162 = v1143[v1161];
                bool v1163;
                v1163 = v1144[v1161];
                float v1170; bool v1171;
                if (v1155){
                    if (v1163){
                        bool v1164;
                        v1164 = v1154 >= v1162;
                        float v1165;
                        if (v1164){
                            v1165 = v1154;
                        } else {
                            v1165 = v1162;
                        }
                        v1170 = v1165; v1171 = true;
                    } else {
                        v1170 = v1154; v1171 = v1155;
                    }
                } else {
                    if (v1163){
                        v1170 = v1162; v1171 = v1163;
                    } else {
                        v1170 = v1154; v1171 = v1155;
                    }
                }
                v1154 = v1170;
                v1155 = v1171;
                v1158 += 1 ;
            }
            v1156 += 1 ;
        }
        auto v1172 = cooperative_groups::coalesced_threads();
        int v1173;
        v1173 = threadIdx.x;
        int v1174;
        v1174 = v1173 / 32;
        auto v1175 = cooperative_groups::labeled_partition(v1172,v1174);
        Closure5 v1176{};
        float v1177; bool v1178;
        Tuple2 tmp62 = cooperative_groups::reduce(v1175, Tuple2{v1154, v1155}, v1176);
        v1177 = tmp62.v0; v1178 = tmp62.v1;
        bool v1179;
        v1179 = v1178 == false;
        if (v1179){
            assert("The local reduce must be true." && v1178);
        } else {
        }
        float v1181[4];
        int v1182[4];
        int v1183;
        v1183 = 0;
        while (while_method_3(v1183)){
            int v1185;
            v1185 = 0;
            while (while_method_1(v1185)){
                assert("Tensor range check" && 0 <= v1183 && v1183 < 1);
                assert("Tensor range check" && 0 <= v1185 && v1185 < 4);
                int v1187;
                v1187 = 4 * v1183;
                int v1188;
                v1188 = v1187 + v1185;
                int v1189;
                v1189 = v1023[v1188];
                float v1190;
                v1190 = curand_uniform(&v998);
                assert("Tensor range check" && 0 <= v1183 && v1183 < 1);
                assert("Tensor range check" && 0 <= v1185 && v1185 < 4);
                v1181[v1188] = v1190;
                v1182[v1188] = v1189;
                v1185 += 1 ;
            }
            v1183 += 1 ;
        }
        float v1191; int v1192;
        Tuple1 tmp63 = Tuple1{0.0f, 2147483647};
        v1191 = tmp63.v0; v1192 = tmp63.v1;
        int v1193;
        v1193 = 0;
        while (while_method_3(v1193)){
            int v1195;
            v1195 = 0;
            while (while_method_1(v1195)){
                assert("Tensor range check" && 0 <= v1193 && v1193 < 1);
                assert("Tensor range check" && 0 <= v1195 && v1195 < 4);
                int v1197;
                v1197 = 4 * v1193;
                int v1198;
                v1198 = v1197 + v1195;
                float v1199;
                v1199 = v1181[v1198];
                int v1200;
                v1200 = v1182[v1198];
                bool v1201;
                v1201 = v1192 < v1200;
                float v1202; int v1203;
                if (v1201){
                    v1202 = v1191; v1203 = v1192;
                } else {
                    v1202 = v1199; v1203 = v1200;
                }
                v1191 = v1202;
                v1192 = v1203;
                v1195 += 1 ;
            }
            v1193 += 1 ;
        }
        auto v1204 = cooperative_groups::coalesced_threads();
        int v1205;
        v1205 = threadIdx.x;
        int v1206;
        v1206 = v1205 / 32;
        auto v1207 = cooperative_groups::labeled_partition(v1204,v1206);
        Closure6 v1208{};
        float v1209; int v1210;
        Tuple1 tmp64 = cooperative_groups::reduce(v1207, Tuple1{v1191, v1192}, v1208);
        v1209 = tmp64.v0; v1210 = tmp64.v1;
        float v1211;
        v1211 = v1177 * v1209;
        int v1212[4];
        bool v1213[4];
        int v1214;
        v1214 = 0;
        while (while_method_3(v1214)){
            int v1216;
            v1216 = 0;
            while (while_method_1(v1216)){
                assert("Tensor range check" && 0 <= v1214 && v1214 < 1);
                assert("Tensor range check" && 0 <= v1216 && v1216 < 4);
                int v1218;
                v1218 = 4 * v1214;
                int v1219;
                v1219 = v1218 + v1216;
                float v1220;
                v1220 = v1143[v1219];
                bool v1221;
                v1221 = v1144[v1219];
                int v1222;
                v1222 = v1023[v1219];
                int v1225; bool v1226;
                if (v1221){
                    float v1223;
                    v1223 = v1220 - v1211;
                    bool v1224;
                    v1224 = v1223 >= 0.0f;
                    v1225 = v1222; v1226 = v1224;
                } else {
                    v1225 = 2147483647; v1226 = false;
                }
                assert("Tensor range check" && 0 <= v1214 && v1214 < 1);
                assert("Tensor range check" && 0 <= v1216 && v1216 < 4);
                v1212[v1219] = v1225;
                v1213[v1219] = v1226;
                v1216 += 1 ;
            }
            v1214 += 1 ;
        }
        int v1227; bool v1228;
        Tuple3 tmp65 = Tuple3{2147483647, false};
        v1227 = tmp65.v0; v1228 = tmp65.v1;
        int v1229;
        v1229 = 0;
        while (while_method_3(v1229)){
            int v1231;
            v1231 = 0;
            while (while_method_1(v1231)){
                assert("Tensor range check" && 0 <= v1229 && v1229 < 1);
                assert("Tensor range check" && 0 <= v1231 && v1231 < 4);
                int v1233;
                v1233 = 4 * v1229;
                int v1234;
                v1234 = v1233 + v1231;
                int v1235;
                v1235 = v1212[v1234];
                bool v1236;
                v1236 = v1213[v1234];
                int v1243; bool v1244;
                if (v1228){
                    if (v1236){
                        bool v1237;
                        v1237 = v1227 < v1235;
                        int v1238;
                        if (v1237){
                            v1238 = v1227;
                        } else {
                            v1238 = v1235;
                        }
                        v1243 = v1238; v1244 = true;
                    } else {
                        v1243 = v1227; v1244 = v1228;
                    }
                } else {
                    if (v1236){
                        v1243 = v1235; v1244 = v1236;
                    } else {
                        v1243 = v1227; v1244 = v1228;
                    }
                }
                v1227 = v1243;
                v1228 = v1244;
                v1231 += 1 ;
            }
            v1229 += 1 ;
        }
        auto v1245 = cooperative_groups::coalesced_threads();
        int v1246;
        v1246 = threadIdx.x;
        int v1247;
        v1247 = v1246 / 32;
        auto v1248 = cooperative_groups::labeled_partition(v1245,v1247);
        Closure7 v1249{};
        int v1250; bool v1251;
        Tuple3 tmp66 = cooperative_groups::reduce(v1248, Tuple3{v1227, v1228}, v1249);
        v1250 = tmp66.v0; v1251 = tmp66.v1;
        bool v1252;
        v1252 = v1251 == false;
        if (v1252){
            assert("The local reduce must be true." && v1251);
        } else {
        }
        assert("Tensor range check" && 0 <= v1012 && v1012 < 8);
        int v1254;
        v1254 = 0;
        while (while_method_3(v1254)){
            assert("Tensor range check" && 0 <= v1254 && v1254 < 1);
            int v1256;
            v1256 = 128 * v1254;
            int v1257;
            v1257 = v1256 + v1021;
            assert("Tensor range check" && 0 <= v1254 && v1254 < 1);
            int v1258;
            v1258 = 4 * v1254;
            int4* v1259;
            v1259 = reinterpret_cast<int4*>(v1105 + v1258);
            int4* v1260;
            v1260 = reinterpret_cast<int4*>(v13 + v1257);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1259) % 16 == 0 && reinterpret_cast<unsigned long long>(v1260) % 16 == 0);
            *v1260 = *v1259;
            v1254 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1012 && v1012 < 8);
        int v1261;
        v1261 = 8 * v1012;
        int v1262;
        v1262 = v1261 + v1004;
        v14[v1262] = v1250;
        v1012 += 24 ;
    }
    v17.sync() ;
    int v1263;
    v1263 = threadIdx.x;
    int v1264;
    v1264 = blockIdx.x;
    int v1265;
    v1265 = v1264 * 256;
    int v1266;
    v1266 = v1263 + v1265;
    unsigned long long v1267;
    v1267 = (unsigned long long)v1266;
    curandStatePhilox4_32_10_t v1268;
    curand_init(12344321ull,v1267,0ull,&v1268);
    int v1269;
    v1269 = threadIdx.x;
    bool v1270;
    v1270 = 0 <= v1269;
    bool v1271;
    v1271 = v1270 == false;
    if (v1271){
        assert("The index needs to be zero or positive." && v1270);
    } else {
    }
    int v1273;
    v1273 = v1269 % 32;
    int v1274;
    v1274 = v1269 / 32;
    bool v1275;
    v1275 = v1274 < 8;
    bool v1276;
    v1276 = v1275 == false;
    if (v1276){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1275);
    } else {
    }
    assert("Tensor range check" && 0 <= v1274 && v1274 < 8);
    assert("Tensor range check" && 0 <= v1273 && v1273 < 32);
    int v1278;
    v1278 = 4 * v1273;
    int v1279;
    v1279 = 128 * v1274;
    int v1280;
    v1280 = v1279 + v1278;
    assert("Tensor range check" && 0 <= v1274 && v1274 < 8);
    assert("Tensor range check" && 0 <= v1273 && v1273 < 32);
    assert("Tensor range check" && 0 <= v1274 && v1274 < 8);
    int v1281;
    v1281 = blockIdx.x;
    int v1282;
    v1282 = v1281;
    while (while_method_2(v1282)){
        bool v1284;
        v1284 = 0 <= v1282;
        bool v1285;
        v1285 = v1284 == false;
        if (v1285){
            assert("The index needs to be zero or positive." && v1284);
        } else {
        }
        bool v1287;
        v1287 = v1282 < 8;
        bool v1288;
        v1288 = v1287 == false;
        if (v1288){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1287);
        } else {
        }
        assert("Tensor range check" && 0 <= v1282 && v1282 < 8);
        int v1290;
        v1290 = 1024 * v1282;
        int v1291;
        v1291 = v1290 + v1280;
        float v1292[4];
        int v1293[4];
        int v1294;
        v1294 = 0;
        while (while_method_3(v1294)){
            assert("Tensor range check" && 0 <= v1294 && v1294 < 1);
            int v1296;
            v1296 = 4 * v1294;
            assert("Tensor range check" && 0 <= v1294 && v1294 < 1);
            int v1297;
            v1297 = 128 * v1294;
            int v1298;
            v1298 = v1297 + v1291;
            int4* v1299;
            v1299 = reinterpret_cast<int4*>(v1 + v1298);
            int4* v1300;
            v1300 = reinterpret_cast<int4*>(v1292 + v1296);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1299) % 16 == 0 && reinterpret_cast<unsigned long long>(v1300) % 16 == 0);
            *v1300 = *v1299;
            v1294 += 1 ;
        }
        int v1301;
        v1301 = 0;
        while (while_method_3(v1301)){
            int v1303;
            v1303 = 0;
            while (while_method_1(v1303)){
                bool v1305;
                v1305 = 0 <= v1303;
                bool v1307;
                if (v1305){
                    bool v1306;
                    v1306 = v1303 < 4;
                    v1307 = v1306;
                } else {
                    v1307 = false;
                }
                bool v1308;
                v1308 = v1307 == false;
                if (v1308){
                    assert("The indices should be inside the range of the dimension." && v1307);
                } else {
                }
                bool v1310;
                v1310 = 0 <= v1273;
                bool v1312;
                if (v1310){
                    bool v1311;
                    v1311 = v1273 < 32;
                    v1312 = v1311;
                } else {
                    v1312 = false;
                }
                bool v1313;
                v1313 = v1312 == false;
                if (v1313){
                    assert("The indices should be inside the range of the dimension." && v1312);
                } else {
                }
                int v1315;
                v1315 = v1273 * 4;
                int v1316;
                v1316 = v1303 + v1315;
                bool v1317;
                v1317 = 0 <= v1301;
                bool v1319;
                if (v1317){
                    bool v1318;
                    v1318 = v1301 < 1;
                    v1319 = v1318;
                } else {
                    v1319 = false;
                }
                bool v1320;
                v1320 = v1319 == false;
                if (v1320){
                    assert("The indices should be inside the range of the dimension." && v1319);
                } else {
                }
                int v1322;
                v1322 = v1301 * 128;
                int v1323;
                v1323 = v1316 + v1322;
                assert("Tensor range check" && 0 <= v1301 && v1301 < 1);
                assert("Tensor range check" && 0 <= v1303 && v1303 < 4);
                int v1324;
                v1324 = 4 * v1301;
                int v1325;
                v1325 = v1324 + v1303;
                v1293[v1325] = v1323;
                v1303 += 1 ;
            }
            v1301 += 1 ;
        }
        bool v1326;
        v1326 = 0 <= v1274;
        bool v1327;
        v1327 = v1326 && v1275;
        bool v1328;
        v1328 = v1327 == false;
        if (v1328){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1327);
        } else {
        }
        bool v1330;
        v1330 = v1284 && v1287;
        bool v1331;
        v1331 = v1330 == false;
        if (v1331){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1330);
        } else {
        }
        int v1333;
        v1333 = v1282 * 8;
        int v1334;
        v1334 = v1333 + v1274;
        bool v1335[4];
        int v1336;
        v1336 = 0;
        while (while_method_3(v1336)){
            int v1338;
            v1338 = 0;
            while (while_method_1(v1338)){
                assert("Tensor range check" && 0 <= v1336 && v1336 < 1);
                assert("Tensor range check" && 0 <= v1338 && v1338 < 4);
                int v1340;
                v1340 = 4 * v1336;
                int v1341;
                v1341 = v1340 + v1338;
                float v1342;
                v1342 = v1292[v1341];
                int v1343;
                v1343 = v1293[v1341];
                bool v1344;
                v1344 = v1343 < 11;
                assert("Tensor range check" && 0 <= v1336 && v1336 < 1);
                assert("Tensor range check" && 0 <= v1338 && v1338 < 4);
                v1335[v1341] = v1344;
                v1338 += 1 ;
            }
            v1336 += 1 ;
        }
        float v1345[4];
        int v1346;
        v1346 = 0;
        while (while_method_3(v1346)){
            int v1348;
            v1348 = 0;
            while (while_method_1(v1348)){
                assert("Tensor range check" && 0 <= v1346 && v1346 < 1);
                assert("Tensor range check" && 0 <= v1348 && v1348 < 4);
                int v1350;
                v1350 = 4 * v1346;
                int v1351;
                v1351 = v1350 + v1348;
                float v1352;
                v1352 = v1292[v1351];
                bool v1353;
                v1353 = v1335[v1351];
                float v1354;
                if (v1353){
                    v1354 = v1352;
                } else {
                    v1354 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1346 && v1346 < 1);
                assert("Tensor range check" && 0 <= v1348 && v1348 < 4);
                v1345[v1351] = v1354;
                v1348 += 1 ;
            }
            v1346 += 1 ;
        }
        float v1355;
        v1355 = 0.0f;
        int v1356;
        v1356 = 0;
        while (while_method_3(v1356)){
            int v1358;
            v1358 = 0;
            while (while_method_1(v1358)){
                assert("Tensor range check" && 0 <= v1356 && v1356 < 1);
                assert("Tensor range check" && 0 <= v1358 && v1358 < 4);
                int v1360;
                v1360 = 4 * v1356;
                int v1361;
                v1361 = v1360 + v1358;
                float v1362;
                v1362 = v1345[v1361];
                float v1363;
                v1363 = v1355 + v1362;
                v1355 = v1363;
                v1358 += 1 ;
            }
            v1356 += 1 ;
        }
        auto v1364 = cooperative_groups::coalesced_threads();
        int v1365;
        v1365 = threadIdx.x;
        int v1366;
        v1366 = v1365 / 32;
        auto v1367 = cooperative_groups::labeled_partition(v1364,v1366);
        Closure0 v1368{};
        float v1369;
        v1369 = cooperative_groups::reduce(v1367, v1355, v1368);
        int v1370[4];
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
                bool v1377;
                v1377 = v1335[v1376];
                int v1378;
                if (v1377){
                    v1378 = 1;
                } else {
                    v1378 = 0;
                }
                assert("Tensor range check" && 0 <= v1371 && v1371 < 1);
                assert("Tensor range check" && 0 <= v1373 && v1373 < 4);
                v1370[v1376] = v1378;
                v1373 += 1 ;
            }
            v1371 += 1 ;
        }
        int v1379;
        v1379 = 0;
        int v1380;
        v1380 = 0;
        while (while_method_3(v1380)){
            int v1382;
            v1382 = 0;
            while (while_method_1(v1382)){
                assert("Tensor range check" && 0 <= v1380 && v1380 < 1);
                assert("Tensor range check" && 0 <= v1382 && v1382 < 4);
                int v1384;
                v1384 = 4 * v1380;
                int v1385;
                v1385 = v1384 + v1382;
                int v1386;
                v1386 = v1370[v1385];
                int v1387;
                v1387 = v1379 + v1386;
                v1379 = v1387;
                v1382 += 1 ;
            }
            v1380 += 1 ;
        }
        auto v1388 = cooperative_groups::coalesced_threads();
        int v1389;
        v1389 = threadIdx.x;
        int v1390;
        v1390 = v1389 / 32;
        auto v1391 = cooperative_groups::labeled_partition(v1388,v1390);
        Closure4 v1392{};
        int v1393;
        v1393 = cooperative_groups::reduce(v1391, v1379, v1392);
        float v1394;
        v1394 = (float)v1393;
        float v1395;
        v1395 = v1369 / v1394;
        float v1396[4];
        int v1397;
        v1397 = 0;
        while (while_method_3(v1397)){
            int v1399;
            v1399 = 0;
            while (while_method_1(v1399)){
                assert("Tensor range check" && 0 <= v1397 && v1397 < 1);
                assert("Tensor range check" && 0 <= v1399 && v1399 < 4);
                int v1401;
                v1401 = 4 * v1397;
                int v1402;
                v1402 = v1401 + v1399;
                float v1403;
                v1403 = v1292[v1402];
                bool v1404;
                v1404 = v1335[v1402];
                float v1405;
                if (v1404){
                    v1405 = v1403;
                } else {
                    v1405 = -1.0f / 0.0f;
                }
                float v1406;
                v1406 = v1405 - v1395;
                float v1407;
                v1407 = exp(v1406);
                bool v1408;
                v1408 = v1407 < 1.0f / 0.0f;
                bool v1409;
                v1409 = v1408 == false;
                if (v1409){
                    assert("The softmax values must not grow too large." && v1408);
                } else {
                }
                bool v1411;
                v1411 = isnan(v1407);
                bool v1412;
                v1412 = v1411 == false;
                bool v1413;
                v1413 = v1412 == false;
                if (v1413){
                    assert("The softmax values must not be nans." && v1412);
                } else {
                }
                assert("Tensor range check" && 0 <= v1397 && v1397 < 1);
                assert("Tensor range check" && 0 <= v1399 && v1399 < 4);
                v1396[v1402] = v1407;
                v1399 += 1 ;
            }
            v1397 += 1 ;
        }
        float v1415;
        v1415 = 0.0f;
        int v1416;
        v1416 = 0;
        while (while_method_3(v1416)){
            int v1418;
            v1418 = 0;
            while (while_method_1(v1418)){
                assert("Tensor range check" && 0 <= v1416 && v1416 < 1);
                assert("Tensor range check" && 0 <= v1418 && v1418 < 4);
                int v1420;
                v1420 = 4 * v1416;
                int v1421;
                v1421 = v1420 + v1418;
                float v1422;
                v1422 = v1396[v1421];
                float v1423;
                v1423 = v1415 + v1422;
                v1415 = v1423;
                v1418 += 1 ;
            }
            v1416 += 1 ;
        }
        auto v1424 = cooperative_groups::coalesced_threads();
        int v1425;
        v1425 = threadIdx.x;
        int v1426;
        v1426 = v1425 / 32;
        auto v1427 = cooperative_groups::labeled_partition(v1424,v1426);
        float v1428;
        v1428 = cooperative_groups::reduce(v1427, v1415, v1368);
        float v1429[4];
        int v1430;
        v1430 = 0;
        while (while_method_3(v1430)){
            int v1432;
            v1432 = 0;
            while (while_method_1(v1432)){
                assert("Tensor range check" && 0 <= v1430 && v1430 < 1);
                assert("Tensor range check" && 0 <= v1432 && v1432 < 4);
                int v1434;
                v1434 = 4 * v1430;
                int v1435;
                v1435 = v1434 + v1432;
                float v1436;
                v1436 = v1396[v1435];
                float v1437;
                v1437 = v1436 / v1428;
                assert("Tensor range check" && 0 <= v1430 && v1430 < 1);
                assert("Tensor range check" && 0 <= v1432 && v1432 < 4);
                v1429[v1435] = v1437;
                v1432 += 1 ;
            }
            v1430 += 1 ;
        }
        float v1438[4];
        float v1439;
        v1439 = 0.0f;
        int v1440;
        v1440 = 0;
        while (while_method_3(v1440)){
            assert("Tensor range check" && 0 <= v1440 && v1440 < 1);
            int v1442;
            v1442 = 4 * v1440;
            assert("Tensor range check" && 0 <= v1440 && v1440 < 1);
            float v1443;
            v1443 = 0.0f;
            int v1444;
            v1444 = 0;
            while (while_method_1(v1444)){
                assert("Tensor range check" && 0 <= v1444 && v1444 < 4);
                int v1446;
                v1446 = v1444 + v1442;
                float v1447;
                v1447 = v1429[v1446];
                float v1448;
                v1448 = v1443 + v1447;
                v1443 = v1448;
                v1444 += 1 ;
            }
            auto v1449 = cooperative_groups::coalesced_threads();
            int v1450;
            v1450 = threadIdx.x;
            int v1451;
            v1451 = v1450 / 32;
            auto v1452 = cooperative_groups::labeled_partition(v1449,v1451);
            Closure2 v1453{};
            float v1454;
            v1454 = cooperative_groups::inclusive_scan(v1452, v1443, v1453);
            float v1455;
            v1455 = v1452.shfl_up(v1454,1);
            bool v1456;
            v1456 = v1452.thread_rank() == 0;
            float v1457;
            if (v1456){
                v1457 = 0.0f;
            } else {
                v1457 = v1455;
            }
            float v1458;
            v1458 = v1452.shfl(v1454,v1452.num_threads()-1);
            float v1459;
            v1459 = v1439 + v1457;
            float v1460;
            v1460 = v1459;
            int v1461;
            v1461 = 0;
            while (while_method_1(v1461)){
                assert("Tensor range check" && 0 <= v1461 && v1461 < 4);
                int v1463;
                v1463 = v1461 + v1442;
                float v1464;
                v1464 = v1429[v1463];
                float v1465;
                v1465 = v1460 + v1464;
                assert("Tensor range check" && 0 <= v1461 && v1461 < 4);
                v1438[v1463] = v1465;
                v1460 = v1465;
                v1461 += 1 ;
            }
            float v1466;
            v1466 = v1439 + v1458;
            v1439 = v1466;
            v1440 += 1 ;
        }
        float v1467[4];
        bool v1468[4];
        int v1469;
        v1469 = 0;
        while (while_method_3(v1469)){
            int v1471;
            v1471 = 0;
            while (while_method_1(v1471)){
                assert("Tensor range check" && 0 <= v1469 && v1469 < 1);
                assert("Tensor range check" && 0 <= v1471 && v1471 < 4);
                int v1473;
                v1473 = 4 * v1469;
                int v1474;
                v1474 = v1473 + v1471;
                float v1475;
                v1475 = v1438[v1474];
                float v1476;
                v1476 = v1429[v1474];
                bool v1477;
                v1477 = v1476 > 0.0f;
                assert("Tensor range check" && 0 <= v1469 && v1469 < 1);
                assert("Tensor range check" && 0 <= v1471 && v1471 < 4);
                v1467[v1474] = v1475;
                v1468[v1474] = v1477;
                v1471 += 1 ;
            }
            v1469 += 1 ;
        }
        float v1478; bool v1479;
        Tuple2 tmp67 = Tuple2{-1.0f / 0.0f, false};
        v1478 = tmp67.v0; v1479 = tmp67.v1;
        int v1480;
        v1480 = 0;
        while (while_method_3(v1480)){
            int v1482;
            v1482 = 0;
            while (while_method_1(v1482)){
                assert("Tensor range check" && 0 <= v1480 && v1480 < 1);
                assert("Tensor range check" && 0 <= v1482 && v1482 < 4);
                int v1484;
                v1484 = 4 * v1480;
                int v1485;
                v1485 = v1484 + v1482;
                float v1486;
                v1486 = v1467[v1485];
                bool v1487;
                v1487 = v1468[v1485];
                float v1494; bool v1495;
                if (v1479){
                    if (v1487){
                        bool v1488;
                        v1488 = v1478 >= v1486;
                        float v1489;
                        if (v1488){
                            v1489 = v1478;
                        } else {
                            v1489 = v1486;
                        }
                        v1494 = v1489; v1495 = true;
                    } else {
                        v1494 = v1478; v1495 = v1479;
                    }
                } else {
                    if (v1487){
                        v1494 = v1486; v1495 = v1487;
                    } else {
                        v1494 = v1478; v1495 = v1479;
                    }
                }
                v1478 = v1494;
                v1479 = v1495;
                v1482 += 1 ;
            }
            v1480 += 1 ;
        }
        auto v1496 = cooperative_groups::coalesced_threads();
        int v1497;
        v1497 = threadIdx.x;
        int v1498;
        v1498 = v1497 / 32;
        auto v1499 = cooperative_groups::labeled_partition(v1496,v1498);
        Closure5 v1500{};
        float v1501; bool v1502;
        Tuple2 tmp68 = cooperative_groups::reduce(v1499, Tuple2{v1478, v1479}, v1500);
        v1501 = tmp68.v0; v1502 = tmp68.v1;
        bool v1503;
        v1503 = v1502 == false;
        if (v1503){
            assert("The local reduce must be true." && v1502);
        } else {
        }
        float v1505[4];
        int v1506[4];
        int v1507;
        v1507 = 0;
        while (while_method_3(v1507)){
            int v1509;
            v1509 = 0;
            while (while_method_1(v1509)){
                assert("Tensor range check" && 0 <= v1507 && v1507 < 1);
                assert("Tensor range check" && 0 <= v1509 && v1509 < 4);
                int v1511;
                v1511 = 4 * v1507;
                int v1512;
                v1512 = v1511 + v1509;
                int v1513;
                v1513 = v1293[v1512];
                float v1514;
                v1514 = curand_uniform(&v1268);
                assert("Tensor range check" && 0 <= v1507 && v1507 < 1);
                assert("Tensor range check" && 0 <= v1509 && v1509 < 4);
                v1505[v1512] = v1514;
                v1506[v1512] = v1513;
                v1509 += 1 ;
            }
            v1507 += 1 ;
        }
        float v1515; int v1516;
        Tuple1 tmp69 = Tuple1{0.0f, 2147483647};
        v1515 = tmp69.v0; v1516 = tmp69.v1;
        int v1517;
        v1517 = 0;
        while (while_method_3(v1517)){
            int v1519;
            v1519 = 0;
            while (while_method_1(v1519)){
                assert("Tensor range check" && 0 <= v1517 && v1517 < 1);
                assert("Tensor range check" && 0 <= v1519 && v1519 < 4);
                int v1521;
                v1521 = 4 * v1517;
                int v1522;
                v1522 = v1521 + v1519;
                float v1523;
                v1523 = v1505[v1522];
                int v1524;
                v1524 = v1506[v1522];
                bool v1525;
                v1525 = v1516 < v1524;
                float v1526; int v1527;
                if (v1525){
                    v1526 = v1515; v1527 = v1516;
                } else {
                    v1526 = v1523; v1527 = v1524;
                }
                v1515 = v1526;
                v1516 = v1527;
                v1519 += 1 ;
            }
            v1517 += 1 ;
        }
        auto v1528 = cooperative_groups::coalesced_threads();
        int v1529;
        v1529 = threadIdx.x;
        int v1530;
        v1530 = v1529 / 32;
        auto v1531 = cooperative_groups::labeled_partition(v1528,v1530);
        Closure6 v1532{};
        float v1533; int v1534;
        Tuple1 tmp70 = cooperative_groups::reduce(v1531, Tuple1{v1515, v1516}, v1532);
        v1533 = tmp70.v0; v1534 = tmp70.v1;
        float v1535;
        v1535 = v1501 * v1533;
        int v1536[4];
        bool v1537[4];
        int v1538;
        v1538 = 0;
        while (while_method_3(v1538)){
            int v1540;
            v1540 = 0;
            while (while_method_1(v1540)){
                assert("Tensor range check" && 0 <= v1538 && v1538 < 1);
                assert("Tensor range check" && 0 <= v1540 && v1540 < 4);
                int v1542;
                v1542 = 4 * v1538;
                int v1543;
                v1543 = v1542 + v1540;
                float v1544;
                v1544 = v1467[v1543];
                bool v1545;
                v1545 = v1468[v1543];
                int v1546;
                v1546 = v1293[v1543];
                int v1549; bool v1550;
                if (v1545){
                    float v1547;
                    v1547 = v1544 - v1535;
                    bool v1548;
                    v1548 = v1547 >= 0.0f;
                    v1549 = v1546; v1550 = v1548;
                } else {
                    v1549 = 2147483647; v1550 = false;
                }
                assert("Tensor range check" && 0 <= v1538 && v1538 < 1);
                assert("Tensor range check" && 0 <= v1540 && v1540 < 4);
                v1536[v1543] = v1549;
                v1537[v1543] = v1550;
                v1540 += 1 ;
            }
            v1538 += 1 ;
        }
        int v1551; bool v1552;
        Tuple3 tmp71 = Tuple3{2147483647, false};
        v1551 = tmp71.v0; v1552 = tmp71.v1;
        int v1553;
        v1553 = 0;
        while (while_method_3(v1553)){
            int v1555;
            v1555 = 0;
            while (while_method_1(v1555)){
                assert("Tensor range check" && 0 <= v1553 && v1553 < 1);
                assert("Tensor range check" && 0 <= v1555 && v1555 < 4);
                int v1557;
                v1557 = 4 * v1553;
                int v1558;
                v1558 = v1557 + v1555;
                int v1559;
                v1559 = v1536[v1558];
                bool v1560;
                v1560 = v1537[v1558];
                int v1567; bool v1568;
                if (v1552){
                    if (v1560){
                        bool v1561;
                        v1561 = v1551 < v1559;
                        int v1562;
                        if (v1561){
                            v1562 = v1551;
                        } else {
                            v1562 = v1559;
                        }
                        v1567 = v1562; v1568 = true;
                    } else {
                        v1567 = v1551; v1568 = v1552;
                    }
                } else {
                    if (v1560){
                        v1567 = v1559; v1568 = v1560;
                    } else {
                        v1567 = v1551; v1568 = v1552;
                    }
                }
                v1551 = v1567;
                v1552 = v1568;
                v1555 += 1 ;
            }
            v1553 += 1 ;
        }
        auto v1569 = cooperative_groups::coalesced_threads();
        int v1570;
        v1570 = threadIdx.x;
        int v1571;
        v1571 = v1570 / 32;
        auto v1572 = cooperative_groups::labeled_partition(v1569,v1571);
        Closure7 v1573{};
        int v1574; bool v1575;
        Tuple3 tmp72 = cooperative_groups::reduce(v1572, Tuple3{v1551, v1552}, v1573);
        v1574 = tmp72.v0; v1575 = tmp72.v1;
        bool v1576;
        v1576 = v1575 == false;
        if (v1576){
            assert("The local reduce must be true." && v1575);
        } else {
        }
        assert("Tensor range check" && 0 <= v1282 && v1282 < 8);
        int v1578;
        v1578 = 0;
        while (while_method_3(v1578)){
            assert("Tensor range check" && 0 <= v1578 && v1578 < 1);
            int v1580;
            v1580 = 128 * v1578;
            int v1581;
            v1581 = v1580 + v1291;
            assert("Tensor range check" && 0 <= v1578 && v1578 < 1);
            int v1582;
            v1582 = 4 * v1578;
            int4* v1583;
            v1583 = reinterpret_cast<int4*>(v1429 + v1582);
            int4* v1584;
            v1584 = reinterpret_cast<int4*>(v15 + v1581);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1583) % 16 == 0 && reinterpret_cast<unsigned long long>(v1584) % 16 == 0);
            *v1584 = *v1583;
            v1578 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1282 && v1282 < 8);
        int v1585;
        v1585 = 8 * v1282;
        int v1586;
        v1586 = v1585 + v1274;
        v16[v1586] = v1574;
        v1282 += 24 ;
    }
    v17.sync() ;
    return ;
}
extern "C" __global__ void entry6(int * v0, int * v1) {
    extern __shared__ unsigned char v2[];
    int * v3;
    v3 = reinterpret_cast<int *>(&v2[0ull]);
    int v5;
    v5 = blockIdx.x;
    int v6;
    v6 = v5;
    while (while_method_5(v6)){
        bool v8;
        v8 = 0 <= v6;
        bool v9;
        v9 = v8 == false;
        if (v9){
            assert("The index needs to be zero or positive." && v8);
        } else {
        }
        int v11;
        v11 = v6 % 1;
        bool v12;
        v12 = v6 < 2;
        bool v13;
        v13 = v12 == false;
        if (v13){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v12);
        } else {
        }
        assert("Tensor range check" && 0 <= v6 && v6 < 2);
        assert("Tensor range check" && 0 <= v11 && v11 < 1);
        assert("Tensor range check" && 0 <= v11 && v11 < 1);
        int v15;
        v15 = 8 * v11;
        int v16;
        v16 = 32 * v11;
        int v17;
        v17 = v16 + v15;
        int v18;
        v18 = 32 * v6;
        int v19;
        v19 = v18 + v17;
        int v20;
        v20 = 4 * v11;
        int v21;
        v21 = v20 + v16;
        int v22;
        v22 = v18 + v21;
        int v23;
        v23 = threadIdx.x;
        int v24;
        v24 = v23;
        while (while_method_6(v24)){
            bool v26;
            v26 = 0 <= v24;
            bool v27;
            v27 = v26 == false;
            if (v27){
                assert("The index needs to be zero or positive." && v26);
            } else {
            }
            int v29;
            v29 = v24 % 8;
            int v30;
            v30 = v24 / 8;
            bool v31;
            v31 = v30 < 4;
            bool v32;
            v32 = v31 == false;
            if (v32){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v31);
            } else {
            }
            assert("Tensor range check" && 0 <= v30 && v30 < 4);
            assert("Tensor range check" && 0 <= v29 && v29 < 8);
            int v34;
            v34 = v29 + v19;
            int v35;
            v35 = 8 * v30;
            int v36;
            v36 = v35 + v34;
            int v37;
            v37 = v0[v36];
            assert("Tensor range check" && 0 <= v30 && v30 < 4);
            assert("Tensor range check" && 0 <= v29 && v29 < 8);
            int v38;
            v38 = 33 * v30;
            int v39;
            v39 = v38 + v29;
            v3[v39] = v37;
            v24 += 256 ;
        }
        __syncthreads();
        int v40;
        v40 = threadIdx.x;
        int v41;
        v41 = v40;
        while (while_method_6(v41)){
            bool v43;
            v43 = 0 <= v41;
            bool v44;
            v44 = v43 == false;
            if (v44){
                assert("The index needs to be zero or positive." && v43);
            } else {
            }
            int v46;
            v46 = v41 % 4;
            int v47;
            v47 = v41 / 4;
            bool v48;
            v48 = v47 < 8;
            bool v49;
            v49 = v48 == false;
            if (v49){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v48);
            } else {
            }
            assert("Tensor range check" && 0 <= v47 && v47 < 8);
            assert("Tensor range check" && 0 <= v46 && v46 < 4);
            int v51;
            v51 = 33 * v46;
            int v52;
            v52 = v47 + v51;
            int v53;
            v53 = v3[v52];
            assert("Tensor range check" && 0 <= v47 && v47 < 8);
            assert("Tensor range check" && 0 <= v46 && v46 < 4);
            int v54;
            v54 = v46 + v22;
            int v55;
            v55 = 4 * v47;
            int v56;
            v56 = v55 + v54;
            v1[v56] = v53;
            v41 += 256 ;
        }
        __syncthreads();
        v6 += 24 ;
    }
    return ;
}
extern "C" __global__ void entry7(int * v0, int * v1) {
    extern __shared__ unsigned char v2[];
    int * v3;
    v3 = reinterpret_cast<int *>(&v2[0ull]);
    int v5;
    v5 = blockIdx.x;
    int v6;
    v6 = v5;
    while (while_method_1(v6)){
        bool v8;
        v8 = 0 <= v6;
        bool v9;
        v9 = v8 == false;
        if (v9){
            assert("The index needs to be zero or positive." && v8);
        } else {
        }
        int v11;
        v11 = v6 % 2;
        int v12;
        v12 = v6 / 2;
        int v13;
        v13 = v12 % 1;
        bool v14;
        v14 = v12 < 2;
        bool v15;
        v15 = v14 == false;
        if (v15){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
        } else {
        }
        assert("Tensor range check" && 0 <= v12 && v12 < 2);
        assert("Tensor range check" && 0 <= v13 && v13 < 1);
        assert("Tensor range check" && 0 <= v11 && v11 < 2);
        int v17;
        v17 = 128 * v11;
        int v18;
        v18 = 32768 * v13;
        int v19;
        v19 = v18 + v17;
        int v20;
        v20 = 32768 * v12;
        int v21;
        v21 = v20 + v19;
        int v22;
        v22 = 16384 * v11;
        int v23;
        v23 = 128 * v13;
        int v24;
        v24 = v23 + v22;
        int v25;
        v25 = v20 + v24;
        int v26;
        v26 = threadIdx.x;
        int v27;
        v27 = v26;
        while (while_method_7(v27)){
            bool v29;
            v29 = 0 <= v27;
            bool v30;
            v30 = v29 == false;
            if (v30){
                assert("The index needs to be zero or positive." && v29);
            } else {
            }
            int v32;
            v32 = v27 % 128;
            int v33;
            v33 = v27 / 128;
            bool v34;
            v34 = v33 < 128;
            bool v35;
            v35 = v34 == false;
            if (v35){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v34);
            } else {
            }
            assert("Tensor range check" && 0 <= v33 && v33 < 128);
            assert("Tensor range check" && 0 <= v32 && v32 < 128);
            int v37;
            v37 = v32 + v21;
            int v38;
            v38 = 256 * v33;
            int v39;
            v39 = v38 + v37;
            int v40;
            v40 = v0[v39];
            assert("Tensor range check" && 0 <= v33 && v33 < 128);
            assert("Tensor range check" && 0 <= v32 && v32 < 128);
            int v41;
            v41 = 129 * v33;
            int v42;
            v42 = v41 + v32;
            v3[v42] = v40;
            v27 += 256 ;
        }
        __syncthreads();
        int v43;
        v43 = threadIdx.x;
        int v44;
        v44 = v43;
        while (while_method_7(v44)){
            bool v46;
            v46 = 0 <= v44;
            bool v47;
            v47 = v46 == false;
            if (v47){
                assert("The index needs to be zero or positive." && v46);
            } else {
            }
            int v49;
            v49 = v44 % 128;
            int v50;
            v50 = v44 / 128;
            bool v51;
            v51 = v50 < 128;
            bool v52;
            v52 = v51 == false;
            if (v52){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v51);
            } else {
            }
            assert("Tensor range check" && 0 <= v50 && v50 < 128);
            assert("Tensor range check" && 0 <= v49 && v49 < 128);
            int v54;
            v54 = 129 * v49;
            int v55;
            v55 = v50 + v54;
            int v56;
            v56 = v3[v55];
            assert("Tensor range check" && 0 <= v50 && v50 < 128);
            assert("Tensor range check" && 0 <= v49 && v49 < 128);
            int v57;
            v57 = v49 + v25;
            int v58;
            v58 = 128 * v50;
            int v59;
            v59 = v58 + v57;
            v1[v59] = v56;
            v44 += 256 ;
        }
        __syncthreads();
        v6 += 24 ;
    }
    return ;
}
extern "C" __global__ void entry8(unsigned long long * v0, unsigned long long * v1, unsigned long long * v2) {
    auto v3 = cooperative_groups::this_grid();
    unsigned long long v4;
    v4 = 0ull;
    int v5;
    v5 = threadIdx.x;
    int v6;
    v6 = v5;
    while (while_method_8(v6)){
        bool v8;
        v8 = 0 <= v6;
        bool v9;
        v9 = v8 == false;
        if (v9){
            assert("The index needs to be zero or positive." && v8);
        } else {
        }
        int v11;
        v11 = v6 % 64;
        int v12;
        v12 = v6 / 64;
        bool v13;
        v13 = v12 < 128;
        bool v14;
        v14 = v13 == false;
        if (v14){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v13);
        } else {
        }
        assert("Tensor range check" && 0 <= v12 && v12 < 128);
        assert("Tensor range check" && 0 <= v11 && v11 < 64);
        int v16;
        v16 = 2 * v11;
        int v17;
        v17 = 128 * v12;
        int v18;
        v18 = v17 + v16;
        unsigned long long v19[2];
        int4* v20;
        v20 = reinterpret_cast<int4*>(v0 + v18);
        int4* v21;
        v21 = reinterpret_cast<int4*>(v19 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v20) % 16 == 0 && reinterpret_cast<unsigned long long>(v21) % 16 == 0);
        *v21 = *v20;
        int v22; unsigned long long v23;
        Tuple4 tmp73 = Tuple4{0, v4};
        v22 = tmp73.v0; v23 = tmp73.v1;
        while (while_method_5(v22)){
            assert("Tensor range check" && 0 <= v22 && v22 < 2);
            unsigned long long v25;
            v25 = v19[v22];
            unsigned long long v26;
            v26 = v23 + v25;
            v23 = v26;
            v22 += 1 ;
        }
        v4 = v23;
        v6 += 256 ;
    }
    __syncwarp();
    auto v27 = cooperative_groups::coalesced_threads();
    Closure8 v28{};
    unsigned long long v29;
    v29 = cooperative_groups::reduce(v27, v4, v28);
    int v30;
    v30 = threadIdx.x;
    int v31;
    v31 = v30 / 32;
    extern __shared__ unsigned char v32[];
    unsigned long long * v33;
    v33 = reinterpret_cast<unsigned long long *>(&v32[0ull]);
    assert("Tensor range check" && 0 <= v31 && v31 < 8);
    v33[v31] = v29;
    __syncthreads();
    int v35;
    v35 = threadIdx.x;
    int v36;
    v36 = v35 % 32;
    bool v37;
    v37 = v36 < 8;
    unsigned long long v39;
    if (v37){
        assert("Tensor range check" && 0 <= v36 && v36 < 8);
        unsigned long long v38;
        v38 = v33[v36];
        v39 = v38;
    } else {
        v39 = 0ull;
    }
    __syncthreads();
    auto v40 = cooperative_groups::coalesced_threads();
    unsigned long long v41;
    v41 = cooperative_groups::reduce(v40, v39, v28);
    v1[0] = v41;
    unsigned long long v42;
    v42 = 0ull;
    int v43;
    v43 = threadIdx.x;
    int v44;
    v44 = blockIdx.x;
    int v45;
    v45 = v44 * 256;
    int v46;
    v46 = v43 + v45;
    int v47;
    v47 = v46;
    while (while_method_8(v47)){
        bool v49;
        v49 = 0 <= v47;
        bool v50;
        v50 = v49 == false;
        if (v50){
            assert("The index needs to be zero or positive." && v49);
        } else {
        }
        int v52;
        v52 = v47 % 64;
        int v53;
        v53 = v47 / 64;
        bool v54;
        v54 = v53 < 128;
        bool v55;
        v55 = v54 == false;
        if (v55){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v54);
        } else {
        }
        assert("Tensor range check" && 0 <= v53 && v53 < 128);
        assert("Tensor range check" && 0 <= v52 && v52 < 64);
        int v57;
        v57 = 2 * v52;
        int v58;
        v58 = 128 * v53;
        int v59;
        v59 = v58 + v57;
        unsigned long long v60[2];
        int4* v61;
        v61 = reinterpret_cast<int4*>(v0 + v59);
        int4* v62;
        v62 = reinterpret_cast<int4*>(v60 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v61) % 16 == 0 && reinterpret_cast<unsigned long long>(v62) % 16 == 0);
        *v62 = *v61;
        int v63; unsigned long long v64;
        Tuple4 tmp74 = Tuple4{0, v42};
        v63 = tmp74.v0; v64 = tmp74.v1;
        while (while_method_5(v63)){
            assert("Tensor range check" && 0 <= v63 && v63 < 2);
            unsigned long long v66;
            v66 = v60[v63];
            unsigned long long v67;
            v67 = v64 + v66;
            v64 = v67;
            v63 += 1 ;
        }
        v42 = v64;
        v47 += 6144 ;
    }
    __syncwarp();
    auto v68 = cooperative_groups::coalesced_threads();
    unsigned long long v69;
    v69 = cooperative_groups::reduce(v68, v42, v28);
    int v70;
    v70 = threadIdx.x;
    int v71;
    v71 = v70 / 32;
    extern __shared__ unsigned char v72[];
    unsigned long long * v73;
    v73 = reinterpret_cast<unsigned long long *>(&v72[0ull]);
    assert("Tensor range check" && 0 <= v71 && v71 < 8);
    v73[v71] = v69;
    __syncthreads();
    int v75;
    v75 = threadIdx.x;
    int v76;
    v76 = v75 % 32;
    bool v77;
    v77 = v76 < 8;
    unsigned long long v79;
    if (v77){
        assert("Tensor range check" && 0 <= v76 && v76 < 8);
        unsigned long long v78;
        v78 = v73[v76];
        v79 = v78;
    } else {
        v79 = 0ull;
    }
    __syncthreads();
    auto v80 = cooperative_groups::coalesced_threads();
    unsigned long long v81;
    v81 = cooperative_groups::reduce(v80, v79, v28);
    int v82;
    v82 = blockIdx.x;
    static unsigned long long v83[24];
    assert("Tensor range check" && 0 <= v82 && v82 < 24);
    v83[v82] = v81;
    v3.sync() ;
    unsigned long long v84;
    v84 = 0ull;
    int v85;
    v85 = threadIdx.x;
    int v86;
    v86 = v85 % 32;
    int v87;
    v87 = v86;
    while (while_method_9(v87)){
        bool v89;
        v89 = 0 <= v87;
        bool v90;
        v90 = v89 == false;
        if (v90){
            assert("The index needs to be zero or positive." && v89);
        } else {
        }
        bool v92;
        v92 = v87 < 24;
        bool v93;
        v93 = v92 == false;
        if (v93){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v92);
        } else {
        }
        assert("Tensor range check" && 0 <= v87 && v87 < 24);
        unsigned long long v95;
        v95 = v83[v87];
        unsigned long long v96;
        v96 = v84 + v95;
        v84 = v96;
        v87 += 32 ;
    }
    __syncwarp();
    auto v97 = cooperative_groups::coalesced_threads();
    unsigned long long v98;
    v98 = cooperative_groups::reduce(v97, v84, v28);
    v2[0] = v98;
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
def method19(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : cp.ndarray, v9 : cp.ndarray, v10 : cp.ndarray, v11 : cp.ndarray, v12 : cp.ndarray, v13 : cp.ndarray, v14 : cp.ndarray, v15 : cp.ndarray, v16 : cp.ndarray, v17 : cp.ndarray) -> None:
    v18 = "test_text_outputs/primitives/"
    v19 = "test2/b/"
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
    v26 = 1
    v27 = raw_module.get_function(f"entry{v26}")
    del v26
    v27.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v27((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v27
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method20(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method21(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v32):
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
        while method2(v40):
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
            v49 = v32 * 64
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
def method22(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
def method23(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method24(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method25(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method26(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v22):
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
def method27(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test2/b/"
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
    while method3(v35):
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
        while method2(v43):
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
            v52 = v35 * 64
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
def method28(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v32):
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
        while method2(v40):
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
            v49 = v32 * 64
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
def method29(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test2/b/"
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
    while method3(v35):
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
        while method2(v43):
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
            v52 = v35 * 64
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
def method30(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v22):
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
def method31(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v32):
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
        while method2(v40):
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
            v49 = v32 * 64
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
def method32(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method33(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v22):
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
def method34(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method35(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test2/b/"
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
    while method3(v22):
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
def method36(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> None:
    v8 = "test_text_outputs/primitives/"
    v9 = "test3/a"
    v10 = "kernel_params.txt"
    v11 = pathlib.Path(v8,v9,v10)
    del v8, v9, v10
    v11.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v11),'w')
    del v11
    v12 = cp.cuda.Device().attributes['MultiProcessorCount']
    v13 = v12 == 24
    del v12
    v14 = v13 == False
    if v14:
        v15 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v13, v15
        del v15
    else:
        pass
    del v13, v14
    v16 = 2
    v17 = raw_module.get_function(f"entry{v16}")
    del v16
    v17.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v17((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v17
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method38(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method39(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def method37(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "input_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method38(v33):
        v35 = v31
        v36 = v35 >= 2147483647
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
        while method39(v41):
            v43 = v31
            v44 = v43 >= 2147483647
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
            v50 = v33 * 16
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
def method40(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "output_sample_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method38(v22):
        v24 = v20
        v25 = v24 >= 2147483647
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
def method41(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/a"
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
    while method38(v35):
        v37 = v33
        v38 = v37 >= 2147483647
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
        while method39(v43):
            v45 = v33
            v46 = v45 >= 2147483647
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
            v52 = v35 * 16
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
def method42(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "output_indices_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method38(v22):
        v24 = v20
        v25 = v24 >= 2147483647
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
def method43(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/a"
    v3 = "output_op_map.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method38(v33):
        v35 = v31
        v36 = v35 >= 2147483647
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
        while method39(v41):
            v43 = v31
            v44 = v43 >= 2147483647
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
            v50 = v33 * 16
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
def method44(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/a"
    v4 = "zip_input_output_identity_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method38(v35):
        v37 = v33
        v38 = v37 >= 2147483647
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
        while method39(v43):
            v45 = v33
            v46 = v45 >= 2147483647
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
            v52 = v35 * 16
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
def method45(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray) -> None:
    v8 = "test_text_outputs/primitives/"
    v9 = "test3/b"
    v10 = "kernel_params.txt"
    v11 = pathlib.Path(v8,v9,v10)
    del v8, v9, v10
    v11.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v11),'w')
    del v11
    v12 = cp.cuda.Device().attributes['MultiProcessorCount']
    v13 = v12 == 24
    del v12
    v14 = v13 == False
    if v14:
        v15 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v13, v15
        del v15
    else:
        pass
    del v13, v14
    v16 = 3
    v17 = raw_module.get_function(f"entry{v16}")
    del v16
    v17.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v17((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v17
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method46(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "input_identity.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method38(v33):
        v35 = v31
        v36 = v35 >= 2147483647
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
        while method38(v41):
            v43 = v31
            v44 = v43 >= 2147483647
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
            v50 = v33 * 256
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
def method47(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "output_sample_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method38(v22):
        v24 = v20
        v25 = v24 >= 2147483647
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
def method48(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/b"
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
    while method38(v35):
        v37 = v33
        v38 = v37 >= 2147483647
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
        while method38(v43):
            v45 = v33
            v46 = v45 >= 2147483647
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
            v52 = v35 * 256
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
def method49(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "output_indices_reduce.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v20 = 0
    v21 = "{}"
    print(v21.format('['),end="")
    v22 = 0
    while method38(v22):
        v24 = v20
        v25 = v24 >= 2147483647
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
def method50(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test3/b"
    v3 = "output_op_map.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method38(v33):
        v35 = v31
        v36 = v35 >= 2147483647
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
        while method38(v41):
            v43 = v31
            v44 = v43 >= 2147483647
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
            v50 = v33 * 256
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
def method51(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test3/b"
    v4 = "zip_input_output_identity_map.txt"
    v5 = pathlib.Path(v2,v3,v4)
    del v2, v3, v4
    v5.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v5),'w')
    del v5
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method38(v35):
        v37 = v33
        v38 = v37 >= 2147483647
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
        while method38(v43):
            v45 = v33
            v46 = v45 >= 2147483647
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
            v52 = v35 * 256
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
def method52(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : cp.ndarray, v9 : cp.ndarray, v10 : cp.ndarray, v11 : cp.ndarray, v12 : cp.ndarray, v13 : cp.ndarray, v14 : cp.ndarray, v15 : cp.ndarray, v16 : cp.ndarray) -> None:
    v17 = "test_text_outputs/primitives/"
    v18 = "test4/b/"
    v19 = "kernel_params.txt"
    v20 = pathlib.Path(v17,v18,v19)
    del v17, v18, v19
    v20.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v20),'w')
    del v20
    v21 = cp.cuda.Device().attributes['MultiProcessorCount']
    v22 = v21 == 24
    del v21
    v23 = v22 == False
    if v23:
        v24 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v22, v24
        del v24
    else:
        pass
    del v22, v23
    v25 = 4
    v26 = raw_module.get_function(f"entry{v25}")
    del v25
    v26.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v26((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v26
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method53(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method54(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v32):
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
        while method2(v40):
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
            v49 = v32 * 64
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
def method55(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method56(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method57(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method58(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v22):
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
def method59(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test4/b/"
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
    while method3(v35):
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
        while method2(v43):
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
            v52 = v35 * 64
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
def method60(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v32):
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
        while method2(v40):
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
            v49 = v32 * 64
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
def method61(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test4/b/"
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
    while method3(v35):
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
        while method2(v43):
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
            v52 = v35 * 64
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
def method62(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v22):
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
def method63(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v32):
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
        while method2(v40):
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
            v49 = v32 * 64
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
def method64(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method65(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v22):
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
def method66(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
    v3 = "output_softmax''.txt"
    v4 = pathlib.Path(v1,v2,v3)
    del v1, v2, v3
    v4.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v4),'w')
    del v4
    v31 = 0
    v32 = "{}"
    print(v32.format('['),end="")
    v33 = 0
    while method3(v33):
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
        while method2(v41):
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
            v50 = v33 * 64
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
def method67(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/b/"
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
    while method3(v22):
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
def method68(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : cp.ndarray, v9 : cp.ndarray, v10 : cp.ndarray, v11 : cp.ndarray, v12 : cp.ndarray, v13 : cp.ndarray, v14 : cp.ndarray, v15 : cp.ndarray, v16 : cp.ndarray) -> None:
    v17 = "test_text_outputs/primitives/"
    v18 = "test4/a/"
    v19 = "kernel_params.txt"
    v20 = pathlib.Path(v17,v18,v19)
    del v17, v18, v19
    v20.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v20),'w')
    del v20
    v21 = cp.cuda.Device().attributes['MultiProcessorCount']
    v22 = v21 == 24
    del v21
    v23 = v22 == False
    if v23:
        v24 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v22, v24
        del v24
    else:
        pass
    del v22, v23
    v25 = 5
    v26 = raw_module.get_function(f"entry{v25}")
    del v25
    v26.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v26((24,),(256,),(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16),shared_mem=98304)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v26
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method69(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method70(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method71(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method72(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method73(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method74(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method75(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test4/a/"
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
def method76(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method77(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    v2 = "test_text_outputs/primitives/"
    v3 = "test4/a/"
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
def method78(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method79(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method80(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method81(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method82(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
    v3 = "output_softmax''.txt"
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
def method83(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test4/a/"
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
def method85(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method86(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method87(v0 : i32) -> bool:
    v1 = v0 < 8
    del v0
    return v1
def method84() -> None:
    v0 = "test_text_outputs/primitives/"
    v1 = "test5/a"
    v2 = "transpose.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.arange(0,64,1,dtype=cp.int32) # type: ignore
    v5 = v4.size
    v6 = 64 == v5
    del v5
    v7 = v6 == False
    if v7:
        v8 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v6, v8
        del v8
    else:
        pass
    del v6, v7
    v9 = cp.empty(64,dtype=cp.int32)
    v45 = 0
    v46 = "{}"
    print(v46.format('['),end="")
    v47 = 0
    while method85(v47):
        v49 = v45
        v50 = v49 >= 2147483647
        del v49
        if v50:
            v51 = " ..."
            print(v46.format(v51),end="")
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
            print(v46.format(v54),end="")
            del v54
        else:
            pass
        del v53
        print(v46.format('['),end="")
        v55 = 0
        while method86(v55):
            v57 = v45
            v58 = v57 >= 2147483647
            del v57
            if v58:
                v59 = " ..."
                print(v46.format(v59),end="")
                del v59
                break
            else:
                pass
            del v58
            v60 = v55 == 0
            v61 = v60 != True
            del v60
            if v61:
                v62 = "; "
                print(v46.format(v62),end="")
                del v62
            else:
                pass
            del v61
            print(v46.format('['),end="")
            v63 = 0
            while method87(v63):
                v65 = v45
                v66 = v65 >= 2147483647
                del v65
                if v66:
                    v67 = " ..."
                    print(v46.format(v67),end="")
                    del v67
                    break
                else:
                    pass
                del v66
                v68 = v63 == 0
                v69 = v68 != True
                del v68
                if v69:
                    v70 = "; "
                    print(v46.format(v70),end="")
                    del v70
                else:
                    pass
                del v69
                v71 = v45 + 1
                v45 = v71
                del v71
                v72 = v47 * 32
                v73 = v55 * 8
                v74 = v72 + v73
                del v72, v73
                v75 = v74 + v63
                del v74
                v76 = v4[v75].item()
                del v75
                print(v46.format(v76),end="")
                del v76
                v63 += 1 
            del v63
            print(v46.format(']'),end="")
            v55 += 1 
        del v55
        print(v46.format(']'),end="")
        v47 += 1 
    del v45, v47
    print(v46.format(']'),end="")
    v77 = "\n"
    print(v77.format(),end="")
    v78 = cp.cuda.Device().attributes['MultiProcessorCount']
    v79 = v78 == 24
    del v78
    v80 = v79 == False
    if v80:
        v81 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v79, v81
        del v81
    else:
        pass
    del v79, v80
    v82 = 6
    v83 = raw_module.get_function(f"entry{v82}")
    del v82
    v83.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v83((24,),(256,),(v4, v9),shared_mem=98304)
    del v4, v83
    v117 = 0
    print(v46.format('['),end="")
    v118 = 0
    while method85(v118):
        v120 = v117
        v121 = v120 >= 2147483647
        del v120
        if v121:
            v122 = " ..."
            print(v46.format(v122),end="")
            del v122
            break
        else:
            pass
        del v121
        v123 = v118 == 0
        v124 = v123 != True
        del v123
        if v124:
            v125 = "; "
            print(v46.format(v125),end="")
            del v125
        else:
            pass
        del v124
        print(v46.format('['),end="")
        v126 = 0
        while method87(v126):
            v128 = v117
            v129 = v128 >= 2147483647
            del v128
            if v129:
                v130 = " ..."
                print(v46.format(v130),end="")
                del v130
                break
            else:
                pass
            del v129
            v131 = v126 == 0
            v132 = v131 != True
            del v131
            if v132:
                v133 = "; "
                print(v46.format(v133),end="")
                del v133
            else:
                pass
            del v132
            print(v46.format('['),end="")
            v134 = 0
            while method86(v134):
                v136 = v117
                v137 = v136 >= 2147483647
                del v136
                if v137:
                    v138 = " ..."
                    print(v46.format(v138),end="")
                    del v138
                    break
                else:
                    pass
                del v137
                v139 = v134 == 0
                v140 = v139 != True
                del v139
                if v140:
                    v141 = "; "
                    print(v46.format(v141),end="")
                    del v141
                else:
                    pass
                del v140
                v142 = v117 + 1
                v117 = v142
                del v142
                v143 = v118 * 32
                v144 = v126 * 4
                v145 = v143 + v144
                del v143, v144
                v146 = v145 + v134
                del v145
                v147 = v9[v146].item()
                del v146
                print(v46.format(v147),end="")
                del v147
                v134 += 1 
            del v134
            print(v46.format(']'),end="")
            v126 += 1 
        del v126
        print(v46.format(']'),end="")
        v118 += 1 
    del v9, v117, v118
    print(v46.format(']'),end="")
    del v46
    print(v77.format(),end="")
    del v77
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method88() -> None:
    v0 = "test_text_outputs/primitives/"
    v1 = "test5/b"
    v2 = "transpose.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.arange(0,65536,1,dtype=cp.int32) # type: ignore
    v5 = v4.size
    v6 = 65536 == v5
    del v5
    v7 = v6 == False
    if v7:
        v8 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v6, v8
        del v8
    else:
        pass
    del v6, v7
    v9 = cp.empty(65536,dtype=cp.int32)
    v45 = 0
    v46 = "{}"
    print(v46.format('['),end="")
    v47 = 0
    while method85(v47):
        v49 = v45
        v50 = v49 >= 2147483647
        del v49
        if v50:
            v51 = " ..."
            print(v46.format(v51),end="")
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
            print(v46.format(v54),end="")
            del v54
        else:
            pass
        del v53
        print(v46.format('['),end="")
        v55 = 0
        while method3(v55):
            v57 = v45
            v58 = v57 >= 2147483647
            del v57
            if v58:
                v59 = " ..."
                print(v46.format(v59),end="")
                del v59
                break
            else:
                pass
            del v58
            v60 = v55 == 0
            v61 = v60 != True
            del v60
            if v61:
                v62 = "; "
                print(v46.format(v62),end="")
                del v62
            else:
                pass
            del v61
            print(v46.format('['),end="")
            v63 = 0
            while method38(v63):
                v65 = v45
                v66 = v65 >= 2147483647
                del v65
                if v66:
                    v67 = " ..."
                    print(v46.format(v67),end="")
                    del v67
                    break
                else:
                    pass
                del v66
                v68 = v63 == 0
                v69 = v68 != True
                del v68
                if v69:
                    v70 = "; "
                    print(v46.format(v70),end="")
                    del v70
                else:
                    pass
                del v69
                v71 = v45 + 1
                v45 = v71
                del v71
                v72 = v47 * 32768
                v73 = v55 * 256
                v74 = v72 + v73
                del v72, v73
                v75 = v74 + v63
                del v74
                v76 = v4[v75].item()
                del v75
                print(v46.format(v76),end="")
                del v76
                v63 += 1 
            del v63
            print(v46.format(']'),end="")
            v55 += 1 
        del v55
        print(v46.format(']'),end="")
        v47 += 1 
    del v45, v47
    print(v46.format(']'),end="")
    v77 = "\n"
    print(v77.format(),end="")
    v78 = cp.cuda.Device().attributes['MultiProcessorCount']
    v79 = v78 == 24
    del v78
    v80 = v79 == False
    if v80:
        v81 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v79, v81
        del v81
    else:
        pass
    del v79, v80
    v82 = 7
    v83 = raw_module.get_function(f"entry{v82}")
    del v82
    v83.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v83((24,),(256,),(v4, v9),shared_mem=98304)
    del v4, v83
    v117 = 0
    print(v46.format('['),end="")
    v118 = 0
    while method85(v118):
        v120 = v117
        v121 = v120 >= 2147483647
        del v120
        if v121:
            v122 = " ..."
            print(v46.format(v122),end="")
            del v122
            break
        else:
            pass
        del v121
        v123 = v118 == 0
        v124 = v123 != True
        del v123
        if v124:
            v125 = "; "
            print(v46.format(v125),end="")
            del v125
        else:
            pass
        del v124
        print(v46.format('['),end="")
        v126 = 0
        while method38(v126):
            v128 = v117
            v129 = v128 >= 2147483647
            del v128
            if v129:
                v130 = " ..."
                print(v46.format(v130),end="")
                del v130
                break
            else:
                pass
            del v129
            v131 = v126 == 0
            v132 = v131 != True
            del v131
            if v132:
                v133 = "; "
                print(v46.format(v133),end="")
                del v133
            else:
                pass
            del v132
            print(v46.format('['),end="")
            v134 = 0
            while method3(v134):
                v136 = v117
                v137 = v136 >= 2147483647
                del v136
                if v137:
                    v138 = " ..."
                    print(v46.format(v138),end="")
                    del v138
                    break
                else:
                    pass
                del v137
                v139 = v134 == 0
                v140 = v139 != True
                del v139
                if v140:
                    v141 = "; "
                    print(v46.format(v141),end="")
                    del v141
                else:
                    pass
                del v140
                v142 = v117 + 1
                v117 = v142
                del v142
                v143 = v118 * 32768
                v144 = v126 * 128
                v145 = v143 + v144
                del v143, v144
                v146 = v145 + v134
                del v145
                v147 = v9[v146].item()
                del v146
                print(v46.format(v147),end="")
                del v147
                v134 += 1 
            del v134
            print(v46.format(']'),end="")
            v126 += 1 
        del v126
        print(v46.format(']'),end="")
        v118 += 1 
    del v9, v117, v118
    print(v46.format(']'),end="")
    del v46
    print(v77.format(),end="")
    del v77
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method89(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> None:
    v3 = "test_text_outputs/primitives/"
    v4 = "test6/a"
    v5 = "kernel_params.txt"
    v6 = pathlib.Path(v3,v4,v5)
    del v3, v4, v5
    v6.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v6),'w')
    del v6
    v7 = cp.cuda.Device().attributes['MultiProcessorCount']
    v8 = v7 == 24
    del v7
    v9 = v8 == False
    if v9:
        v10 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v8, v10
        del v10
    else:
        pass
    del v8, v9
    v11 = 8
    v12 = raw_module.get_function(f"entry{v11}")
    del v11
    v12.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v12((24,),(256,),(v0, v1, v2),shared_mem=98304)
    del v0, v1, v2, v12
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method90(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test6/a"
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
    while method3(v32):
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
def method91(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test6/a"
    v3 = "output_reduce_block.txt"
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
    v15 = "{}"
    print(v15.format(v14),end="")
    del v14, v15
    v16 = "\n"
    print(v16.format(),end="")
    del v16
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method92(v0 : cp.ndarray) -> None:
    v1 = "test_text_outputs/primitives/"
    v2 = "test6/a"
    v3 = "output_reduce_grid.txt"
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
    v15 = "{}"
    print(v15.format(v14),end="")
    del v14, v15
    v16 = "\n"
    print(v16.format(),end="")
    del v16
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
    method18(v21)
    del v21
    cp.random.seed(12344321)
    v22 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v23 = v22.size
    v24 = 8192 == v23
    del v23
    v25 = v24 == False
    if v25:
        v26 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v24, v26
        del v26
    else:
        pass
    del v24, v25
    v27 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v28 = cp.empty(1,dtype=cp.float32)
    v29 = cp.empty(8192,dtype=cp.int32)
    v30 = cp.empty(8192,dtype=cp.float32)
    v31 = cp.empty(8192,dtype=cp.float32)
    v32 = cp.empty(8192,dtype=cp.float32)
    v33 = cp.empty(8192,dtype=cp.float32)
    v34 = cp.empty(8192,dtype=cp.float32)
    v35 = cp.empty(128,dtype=cp.int32)
    v36 = cp.empty(8192,dtype=cp.int32)
    v37 = cp.empty(8192,dtype=cp.int32)
    v38 = cp.empty(128,dtype=cp.int32)
    v39 = cp.empty(8192,dtype=cp.int32)
    v40 = cp.empty(8192,dtype=cp.float32)
    v41 = cp.empty(128,dtype=cp.int32)
    v42 = cp.empty(8192,dtype=cp.float32)
    v43 = cp.empty(128,dtype=cp.int32)
    method19(v22, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43)
    method20(v27)
    del v27
    method21(v22)
    del v22
    method22(v28)
    del v28
    method23(v30)
    del v30
    method24(v31)
    del v31
    method25(v34)
    del v34
    method26(v35)
    del v35
    method27(v32, v33)
    del v32, v33
    method28(v29)
    del v29
    method29(v36, v37)
    del v36, v37
    method30(v38)
    del v38
    method31(v39)
    del v39
    method32(v40)
    del v40
    method33(v41)
    del v41
    method34(v42)
    del v42
    method35(v43)
    del v43
    cp.random.seed(12344321)
    v44 = cp.arange(0,4096,1,dtype=cp.float32) # type: ignore
    v45 = v44.size
    v46 = 4096 == v45
    del v45
    v47 = v46 == False
    if v47:
        v48 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v46, v48
        del v48
    else:
        pass
    del v46, v47
    v49 = cp.random.normal(0.0,1.0,4096,dtype=cp.float32) # type: ignore
    v50 = cp.empty(4096,dtype=cp.int32)
    v51 = cp.empty(4096,dtype=cp.int32)
    v52 = cp.empty(256,dtype=cp.int32)
    v53 = cp.empty(256,dtype=cp.int32)
    v54 = cp.empty(4096,dtype=cp.float32)
    v55 = cp.empty(4096,dtype=cp.float32)
    method36(v44, v49, v50, v51, v52, v53, v54, v55)
    method37(v44)
    del v44
    method40(v53)
    del v53
    method41(v50, v51)
    del v50, v51
    method42(v52)
    del v52
    method43(v55)
    del v55
    method44(v49, v54)
    del v49, v54
    cp.random.seed(12344321)
    v56 = cp.arange(0,65536,1,dtype=cp.float32) # type: ignore
    v57 = v56.size
    v58 = 65536 == v57
    del v57
    v59 = v58 == False
    if v59:
        v60 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v58, v60
        del v60
    else:
        pass
    del v58, v59
    v61 = cp.random.normal(0.0,1.0,65536,dtype=cp.float32) # type: ignore
    v62 = cp.empty(65536,dtype=cp.int32)
    v63 = cp.empty(65536,dtype=cp.int32)
    v64 = cp.empty(256,dtype=cp.int32)
    v65 = cp.empty(256,dtype=cp.int32)
    v66 = cp.empty(65536,dtype=cp.float32)
    v67 = cp.empty(65536,dtype=cp.float32)
    method45(v56, v61, v62, v63, v64, v65, v66, v67)
    method46(v56)
    del v56
    method47(v65)
    del v65
    method48(v62, v63)
    del v62, v63
    method49(v64)
    del v64
    method50(v67)
    del v67
    method51(v61, v66)
    del v61, v66
    cp.random.seed(12344321)
    v68 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v69 = v68.size
    v70 = 8192 == v69
    del v69
    v71 = v70 == False
    if v71:
        v72 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v70, v72
        del v72
    else:
        pass
    del v70, v71
    v73 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v74 = cp.empty(8192,dtype=cp.int32)
    v75 = cp.empty(8192,dtype=cp.float32)
    v76 = cp.empty(8192,dtype=cp.float32)
    v77 = cp.empty(8192,dtype=cp.float32)
    v78 = cp.empty(8192,dtype=cp.float32)
    v79 = cp.empty(8192,dtype=cp.float32)
    v80 = cp.empty(128,dtype=cp.int32)
    v81 = cp.empty(8192,dtype=cp.int32)
    v82 = cp.empty(8192,dtype=cp.int32)
    v83 = cp.empty(128,dtype=cp.int32)
    v84 = cp.empty(8192,dtype=cp.int32)
    v85 = cp.empty(8192,dtype=cp.float32)
    v86 = cp.empty(128,dtype=cp.int32)
    v87 = cp.empty(8192,dtype=cp.float32)
    v88 = cp.empty(128,dtype=cp.int32)
    method52(v68, v73, v74, v75, v76, v77, v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88)
    method53(v73)
    del v73
    method54(v68)
    del v68
    method55(v75)
    del v75
    method56(v76)
    del v76
    method57(v79)
    del v79
    method58(v80)
    del v80
    method59(v77, v78)
    del v77, v78
    method60(v74)
    del v74
    method61(v81, v82)
    del v81, v82
    method62(v83)
    del v83
    method63(v84)
    del v84
    method64(v85)
    del v85
    method65(v86)
    del v86
    method66(v87)
    del v87
    method67(v88)
    del v88
    cp.random.seed(12344321)
    v89 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v90 = v89.size
    v91 = 8192 == v90
    del v90
    v92 = v91 == False
    if v92:
        v93 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v91, v93
        del v93
    else:
        pass
    del v91, v92
    v94 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v95 = cp.empty(8192,dtype=cp.int32)
    v96 = cp.empty(8192,dtype=cp.float32)
    v97 = cp.empty(8192,dtype=cp.float32)
    v98 = cp.empty(8192,dtype=cp.float32)
    v99 = cp.empty(8192,dtype=cp.float32)
    v100 = cp.empty(8192,dtype=cp.float32)
    v101 = cp.empty(64,dtype=cp.int32)
    v102 = cp.empty(8192,dtype=cp.int32)
    v103 = cp.empty(8192,dtype=cp.int32)
    v104 = cp.empty(64,dtype=cp.int32)
    v105 = cp.empty(8192,dtype=cp.int32)
    v106 = cp.empty(8192,dtype=cp.float32)
    v107 = cp.empty(64,dtype=cp.int32)
    v108 = cp.empty(8192,dtype=cp.float32)
    v109 = cp.empty(64,dtype=cp.int32)
    method68(v89, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107, v108, v109)
    method69(v94)
    del v94
    method70(v89)
    del v89
    method71(v96)
    del v96
    method72(v97)
    del v97
    method73(v100)
    del v100
    method74(v101)
    del v101
    method75(v98, v99)
    del v98, v99
    method76(v95)
    del v95
    method77(v102, v103)
    del v102, v103
    method78(v104)
    del v104
    method79(v105)
    del v105
    method80(v106)
    del v106
    method81(v107)
    del v107
    method82(v108)
    del v108
    method83(v109)
    del v109
    method84()
    method88()
    cp.random.seed(12344321)
    v110 = cp.arange(0,16384,1,dtype=cp.uint64) # type: ignore
    v111 = v110.size
    v112 = 16384 == v111
    del v111
    v113 = v112 == False
    if v113:
        v114 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v112, v114
        del v114
    else:
        pass
    del v112, v113
    v115 = cp.empty(1,dtype=cp.uint64)
    v116 = cp.empty(1,dtype=cp.uint64)
    method89(v110, v115, v116)
    method90(v110)
    del v110
    method91(v115)
    del v115
    return method92(v116)

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
