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
    unsigned long long v983;
    v983 = (unsigned long long)v982;
    curandStatePhilox4_32_10_t v984;
    curand_init(12344321ull,v983,0ull,&v984);
    int v985;
    v985 = threadIdx.x;
    bool v986;
    v986 = 0 <= v985;
    bool v987;
    v987 = v986 == false;
    if (v987){
        assert("The index needs to be zero or positive." && v986);
    } else {
    }
    int v989;
    v989 = v985 % 32;
    int v990;
    v990 = v985 / 32;
    bool v991;
    v991 = v990 < 8;
    bool v992;
    v992 = v991 == false;
    if (v992){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v991);
    } else {
    }
    assert("Tensor range check" && 0 <= v990 && v990 < 8);
    assert("Tensor range check" && 0 <= v989 && v989 < 32);
    int v994;
    v994 = 4 * v989;
    int v995;
    v995 = 128 * v990;
    int v996;
    v996 = v995 + v994;
    assert("Tensor range check" && 0 <= v990 && v990 < 8);
    assert("Tensor range check" && 0 <= v989 && v989 < 32);
    assert("Tensor range check" && 0 <= v990 && v990 < 8);
    int v997;
    v997 = 0;
    while (while_method_2(v997)){
        assert("Tensor range check" && 0 <= v997 && v997 < 8);
        int v999;
        v999 = 1024 * v997;
        int v1000;
        v1000 = v999 + v996;
        float v1001[4];
        int v1002[4];
        int v1003;
        v1003 = 0;
        while (while_method_3(v1003)){
            assert("Tensor range check" && 0 <= v1003 && v1003 < 1);
            int v1005;
            v1005 = 4 * v1003;
            assert("Tensor range check" && 0 <= v1003 && v1003 < 1);
            int v1006;
            v1006 = 128 * v1003;
            int v1007;
            v1007 = v1006 + v1000;
            int4* v1008;
            v1008 = reinterpret_cast<int4*>(v1 + v1007);
            int4* v1009;
            v1009 = reinterpret_cast<int4*>(v1001 + v1005);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1008) % 16 == 0 && reinterpret_cast<unsigned long long>(v1009) % 16 == 0);
            *v1009 = *v1008;
            v1003 += 1 ;
        }
        int v1010;
        v1010 = 0;
        while (while_method_3(v1010)){
            int v1012;
            v1012 = 0;
            while (while_method_1(v1012)){
                bool v1014;
                v1014 = 0 <= v1012;
                bool v1016;
                if (v1014){
                    bool v1015;
                    v1015 = v1012 < 4;
                    v1016 = v1015;
                } else {
                    v1016 = false;
                }
                bool v1017;
                v1017 = v1016 == false;
                if (v1017){
                    assert("The indices should be inside the range of the dimension." && v1016);
                } else {
                }
                bool v1019;
                v1019 = 0 <= v989;
                bool v1021;
                if (v1019){
                    bool v1020;
                    v1020 = v989 < 32;
                    v1021 = v1020;
                } else {
                    v1021 = false;
                }
                bool v1022;
                v1022 = v1021 == false;
                if (v1022){
                    assert("The indices should be inside the range of the dimension." && v1021);
                } else {
                }
                int v1024;
                v1024 = v989 * 4;
                int v1025;
                v1025 = v1012 + v1024;
                bool v1026;
                v1026 = 0 <= v1010;
                bool v1028;
                if (v1026){
                    bool v1027;
                    v1027 = v1010 < 1;
                    v1028 = v1027;
                } else {
                    v1028 = false;
                }
                bool v1029;
                v1029 = v1028 == false;
                if (v1029){
                    assert("The indices should be inside the range of the dimension." && v1028);
                } else {
                }
                int v1031;
                v1031 = v1010 * 128;
                int v1032;
                v1032 = v1025 + v1031;
                assert("Tensor range check" && 0 <= v1010 && v1010 < 1);
                assert("Tensor range check" && 0 <= v1012 && v1012 < 4);
                int v1033;
                v1033 = 4 * v1010;
                int v1034;
                v1034 = v1033 + v1012;
                v1002[v1034] = v1032;
                v1012 += 1 ;
            }
            v1010 += 1 ;
        }
        bool v1035;
        v1035 = 0 <= v990;
        bool v1036;
        v1036 = v1035 && v991;
        bool v1037;
        v1037 = v1036 == false;
        if (v1037){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1036);
        } else {
        }
        bool v1039;
        v1039 = 0 <= v997;
        bool v1041;
        if (v1039){
            bool v1040;
            v1040 = v997 < 8;
            v1041 = v1040;
        } else {
            v1041 = false;
        }
        bool v1042;
        v1042 = v1041 == false;
        if (v1042){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1041);
        } else {
        }
        int v1044;
        v1044 = v997 * 8;
        int v1045;
        v1045 = v1044 + v990;
        float v1046;
        v1046 = 0.0f;
        int v1047;
        v1047 = 0;
        while (while_method_3(v1047)){
            int v1049;
            v1049 = 0;
            while (while_method_1(v1049)){
                assert("Tensor range check" && 0 <= v1047 && v1047 < 1);
                assert("Tensor range check" && 0 <= v1049 && v1049 < 4);
                int v1051;
                v1051 = 4 * v1047;
                int v1052;
                v1052 = v1051 + v1049;
                float v1053;
                v1053 = v1001[v1052];
                float v1054;
                v1054 = v1046 + v1053;
                v1046 = v1054;
                v1049 += 1 ;
            }
            v1047 += 1 ;
        }
        auto v1055 = cooperative_groups::coalesced_threads();
        int v1056;
        v1056 = threadIdx.x;
        int v1057;
        v1057 = v1056 / 32;
        auto v1058 = cooperative_groups::labeled_partition(v1055,v1057);
        float v1059;
        v1059 = cooperative_groups::reduce(v1058, v1046, v42);
        float v1060;
        v1060 = v1059 / 128.0f;
        float v1061[4];
        int v1062;
        v1062 = 0;
        while (while_method_3(v1062)){
            int v1064;
            v1064 = 0;
            while (while_method_1(v1064)){
                assert("Tensor range check" && 0 <= v1062 && v1062 < 1);
                assert("Tensor range check" && 0 <= v1064 && v1064 < 4);
                int v1066;
                v1066 = 4 * v1062;
                int v1067;
                v1067 = v1066 + v1064;
                float v1068;
                v1068 = v1001[v1067];
                float v1069;
                v1069 = v1068 - v1060;
                float v1070;
                v1070 = exp(v1069);
                assert("Tensor range check" && 0 <= v1062 && v1062 < 1);
                assert("Tensor range check" && 0 <= v1064 && v1064 < 4);
                v1061[v1067] = v1070;
                v1064 += 1 ;
            }
            v1062 += 1 ;
        }
        float v1071;
        v1071 = 0.0f;
        int v1072;
        v1072 = 0;
        while (while_method_3(v1072)){
            int v1074;
            v1074 = 0;
            while (while_method_1(v1074)){
                assert("Tensor range check" && 0 <= v1072 && v1072 < 1);
                assert("Tensor range check" && 0 <= v1074 && v1074 < 4);
                int v1076;
                v1076 = 4 * v1072;
                int v1077;
                v1077 = v1076 + v1074;
                float v1078;
                v1078 = v1061[v1077];
                float v1079;
                v1079 = v1071 + v1078;
                v1071 = v1079;
                v1074 += 1 ;
            }
            v1072 += 1 ;
        }
        auto v1080 = cooperative_groups::coalesced_threads();
        int v1081;
        v1081 = threadIdx.x;
        int v1082;
        v1082 = v1081 / 32;
        auto v1083 = cooperative_groups::labeled_partition(v1080,v1082);
        float v1084;
        v1084 = cooperative_groups::reduce(v1083, v1071, v42);
        float v1085[4];
        int v1086;
        v1086 = 0;
        while (while_method_3(v1086)){
            int v1088;
            v1088 = 0;
            while (while_method_1(v1088)){
                assert("Tensor range check" && 0 <= v1086 && v1086 < 1);
                assert("Tensor range check" && 0 <= v1088 && v1088 < 4);
                int v1090;
                v1090 = 4 * v1086;
                int v1091;
                v1091 = v1090 + v1088;
                float v1092;
                v1092 = v1061[v1091];
                float v1093;
                v1093 = v1092 / v1084;
                assert("Tensor range check" && 0 <= v1086 && v1086 < 1);
                assert("Tensor range check" && 0 <= v1088 && v1088 < 4);
                v1085[v1091] = v1093;
                v1088 += 1 ;
            }
            v1086 += 1 ;
        }
        float v1094[4];
        float v1095;
        v1095 = 0.0f;
        int v1096;
        v1096 = 0;
        while (while_method_3(v1096)){
            assert("Tensor range check" && 0 <= v1096 && v1096 < 1);
            int v1098;
            v1098 = 4 * v1096;
            assert("Tensor range check" && 0 <= v1096 && v1096 < 1);
            float v1099;
            v1099 = 0.0f;
            int v1100;
            v1100 = 0;
            while (while_method_1(v1100)){
                assert("Tensor range check" && 0 <= v1100 && v1100 < 4);
                int v1102;
                v1102 = v1100 + v1098;
                float v1103;
                v1103 = v1085[v1102];
                float v1104;
                v1104 = v1099 + v1103;
                v1099 = v1104;
                v1100 += 1 ;
            }
            auto v1105 = cooperative_groups::coalesced_threads();
            int v1106;
            v1106 = threadIdx.x;
            int v1107;
            v1107 = v1106 / 32;
            auto v1108 = cooperative_groups::labeled_partition(v1105,v1107);
            Closure2 v1109{};
            float v1110;
            v1110 = cooperative_groups::inclusive_scan(v1108, v1099, v1109);
            float v1111;
            v1111 = v1108.shfl_up(v1110,1);
            bool v1112;
            v1112 = v1108.thread_rank() == 0;
            float v1113;
            if (v1112){
                v1113 = 0.0f;
            } else {
                v1113 = v1111;
            }
            float v1114;
            v1114 = v1108.shfl(v1110,v1108.num_threads()-1);
            float v1115;
            v1115 = v1095 + v1113;
            float v1116;
            v1116 = v1115;
            int v1117;
            v1117 = 0;
            while (while_method_1(v1117)){
                assert("Tensor range check" && 0 <= v1117 && v1117 < 4);
                int v1119;
                v1119 = v1117 + v1098;
                float v1120;
                v1120 = v1085[v1119];
                float v1121;
                v1121 = v1116 + v1120;
                assert("Tensor range check" && 0 <= v1117 && v1117 < 4);
                v1094[v1119] = v1121;
                v1116 = v1121;
                v1117 += 1 ;
            }
            float v1122;
            v1122 = v1095 + v1114;
            v1095 = v1122;
            v1096 += 1 ;
        }
        float v1123[4];
        bool v1124[4];
        int v1125;
        v1125 = 0;
        while (while_method_3(v1125)){
            int v1127;
            v1127 = 0;
            while (while_method_1(v1127)){
                assert("Tensor range check" && 0 <= v1125 && v1125 < 1);
                assert("Tensor range check" && 0 <= v1127 && v1127 < 4);
                int v1129;
                v1129 = 4 * v1125;
                int v1130;
                v1130 = v1129 + v1127;
                float v1131;
                v1131 = v1094[v1130];
                float v1132;
                v1132 = v1085[v1130];
                bool v1133;
                v1133 = v1132 > 0.0f;
                assert("Tensor range check" && 0 <= v1125 && v1125 < 1);
                assert("Tensor range check" && 0 <= v1127 && v1127 < 4);
                v1123[v1130] = v1131;
                v1124[v1130] = v1133;
                v1127 += 1 ;
            }
            v1125 += 1 ;
        }
        float v1134; bool v1135;
        Tuple2 tmp3 = Tuple2{-1.0f / 0.0f, false};
        v1134 = tmp3.v0; v1135 = tmp3.v1;
        int v1136;
        v1136 = 0;
        while (while_method_3(v1136)){
            int v1138;
            v1138 = 0;
            while (while_method_1(v1138)){
                assert("Tensor range check" && 0 <= v1136 && v1136 < 1);
                assert("Tensor range check" && 0 <= v1138 && v1138 < 4);
                int v1140;
                v1140 = 4 * v1136;
                int v1141;
                v1141 = v1140 + v1138;
                float v1142;
                v1142 = v1123[v1141];
                bool v1143;
                v1143 = v1124[v1141];
                float v1150; bool v1151;
                if (v1135){
                    if (v1143){
                        bool v1144;
                        v1144 = v1134 >= v1142;
                        float v1145;
                        if (v1144){
                            v1145 = v1134;
                        } else {
                            v1145 = v1142;
                        }
                        v1150 = v1145; v1151 = true;
                    } else {
                        v1150 = v1134; v1151 = v1135;
                    }
                } else {
                    if (v1143){
                        v1150 = v1142; v1151 = v1143;
                    } else {
                        v1150 = v1134; v1151 = v1135;
                    }
                }
                v1134 = v1150;
                v1135 = v1151;
                v1138 += 1 ;
            }
            v1136 += 1 ;
        }
        auto v1152 = cooperative_groups::coalesced_threads();
        int v1153;
        v1153 = threadIdx.x;
        int v1154;
        v1154 = v1153 / 32;
        auto v1155 = cooperative_groups::labeled_partition(v1152,v1154);
        Closure5 v1156{};
        float v1157; bool v1158;
        Tuple2 tmp4 = cooperative_groups::reduce(v1155, Tuple2{v1134, v1135}, v1156);
        v1157 = tmp4.v0; v1158 = tmp4.v1;
        bool v1159;
        v1159 = v1158 == false;
        if (v1159){
            assert("The local reduce must be true." && v1158);
        } else {
        }
        float v1161[4];
        int v1162[4];
        int v1163;
        v1163 = 0;
        while (while_method_3(v1163)){
            int v1165;
            v1165 = 0;
            while (while_method_1(v1165)){
                assert("Tensor range check" && 0 <= v1163 && v1163 < 1);
                assert("Tensor range check" && 0 <= v1165 && v1165 < 4);
                int v1167;
                v1167 = 4 * v1163;
                int v1168;
                v1168 = v1167 + v1165;
                int v1169;
                v1169 = v1002[v1168];
                float v1170;
                v1170 = curand_uniform(&v984);
                assert("Tensor range check" && 0 <= v1163 && v1163 < 1);
                assert("Tensor range check" && 0 <= v1165 && v1165 < 4);
                v1161[v1168] = v1170;
                v1162[v1168] = v1169;
                v1165 += 1 ;
            }
            v1163 += 1 ;
        }
        float v1171; int v1172;
        Tuple1 tmp5 = Tuple1{0.0f, 2147483647};
        v1171 = tmp5.v0; v1172 = tmp5.v1;
        int v1173;
        v1173 = 0;
        while (while_method_3(v1173)){
            int v1175;
            v1175 = 0;
            while (while_method_1(v1175)){
                assert("Tensor range check" && 0 <= v1173 && v1173 < 1);
                assert("Tensor range check" && 0 <= v1175 && v1175 < 4);
                int v1177;
                v1177 = 4 * v1173;
                int v1178;
                v1178 = v1177 + v1175;
                float v1179;
                v1179 = v1161[v1178];
                int v1180;
                v1180 = v1162[v1178];
                bool v1181;
                v1181 = v1172 < v1180;
                float v1182; int v1183;
                if (v1181){
                    v1182 = v1171; v1183 = v1172;
                } else {
                    v1182 = v1179; v1183 = v1180;
                }
                v1171 = v1182;
                v1172 = v1183;
                v1175 += 1 ;
            }
            v1173 += 1 ;
        }
        auto v1184 = cooperative_groups::coalesced_threads();
        int v1185;
        v1185 = threadIdx.x;
        int v1186;
        v1186 = v1185 / 32;
        auto v1187 = cooperative_groups::labeled_partition(v1184,v1186);
        Closure6 v1188{};
        float v1189; int v1190;
        Tuple1 tmp6 = cooperative_groups::reduce(v1187, Tuple1{v1171, v1172}, v1188);
        v1189 = tmp6.v0; v1190 = tmp6.v1;
        float v1191;
        v1191 = v1157 * v1189;
        int v1192[4];
        bool v1193[4];
        int v1194;
        v1194 = 0;
        while (while_method_3(v1194)){
            int v1196;
            v1196 = 0;
            while (while_method_1(v1196)){
                assert("Tensor range check" && 0 <= v1194 && v1194 < 1);
                assert("Tensor range check" && 0 <= v1196 && v1196 < 4);
                int v1198;
                v1198 = 4 * v1194;
                int v1199;
                v1199 = v1198 + v1196;
                float v1200;
                v1200 = v1123[v1199];
                bool v1201;
                v1201 = v1124[v1199];
                int v1202;
                v1202 = v1002[v1199];
                int v1205; bool v1206;
                if (v1201){
                    float v1203;
                    v1203 = v1200 - v1191;
                    bool v1204;
                    v1204 = v1203 >= 0.0f;
                    v1205 = v1202; v1206 = v1204;
                } else {
                    v1205 = 2147483647; v1206 = false;
                }
                assert("Tensor range check" && 0 <= v1194 && v1194 < 1);
                assert("Tensor range check" && 0 <= v1196 && v1196 < 4);
                v1192[v1199] = v1205;
                v1193[v1199] = v1206;
                v1196 += 1 ;
            }
            v1194 += 1 ;
        }
        int v1207; bool v1208;
        Tuple3 tmp7 = Tuple3{2147483647, false};
        v1207 = tmp7.v0; v1208 = tmp7.v1;
        int v1209;
        v1209 = 0;
        while (while_method_3(v1209)){
            int v1211;
            v1211 = 0;
            while (while_method_1(v1211)){
                assert("Tensor range check" && 0 <= v1209 && v1209 < 1);
                assert("Tensor range check" && 0 <= v1211 && v1211 < 4);
                int v1213;
                v1213 = 4 * v1209;
                int v1214;
                v1214 = v1213 + v1211;
                int v1215;
                v1215 = v1192[v1214];
                bool v1216;
                v1216 = v1193[v1214];
                int v1223; bool v1224;
                if (v1208){
                    if (v1216){
                        bool v1217;
                        v1217 = v1207 < v1215;
                        int v1218;
                        if (v1217){
                            v1218 = v1207;
                        } else {
                            v1218 = v1215;
                        }
                        v1223 = v1218; v1224 = true;
                    } else {
                        v1223 = v1207; v1224 = v1208;
                    }
                } else {
                    if (v1216){
                        v1223 = v1215; v1224 = v1216;
                    } else {
                        v1223 = v1207; v1224 = v1208;
                    }
                }
                v1207 = v1223;
                v1208 = v1224;
                v1211 += 1 ;
            }
            v1209 += 1 ;
        }
        auto v1225 = cooperative_groups::coalesced_threads();
        int v1226;
        v1226 = threadIdx.x;
        int v1227;
        v1227 = v1226 / 32;
        auto v1228 = cooperative_groups::labeled_partition(v1225,v1227);
        Closure7 v1229{};
        int v1230; bool v1231;
        Tuple3 tmp8 = cooperative_groups::reduce(v1228, Tuple3{v1207, v1208}, v1229);
        v1230 = tmp8.v0; v1231 = tmp8.v1;
        bool v1232;
        v1232 = v1231 == false;
        if (v1232){
            assert("The local reduce must be true." && v1231);
        } else {
        }
        assert("Tensor range check" && 0 <= v997 && v997 < 8);
        int v1234;
        v1234 = 0;
        while (while_method_3(v1234)){
            assert("Tensor range check" && 0 <= v1234 && v1234 < 1);
            int v1236;
            v1236 = 128 * v1234;
            int v1237;
            v1237 = v1236 + v1000;
            assert("Tensor range check" && 0 <= v1234 && v1234 < 1);
            int v1238;
            v1238 = 4 * v1234;
            int4* v1239;
            v1239 = reinterpret_cast<int4*>(v1085 + v1238);
            int4* v1240;
            v1240 = reinterpret_cast<int4*>(v14 + v1237);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1239) % 16 == 0 && reinterpret_cast<unsigned long long>(v1240) % 16 == 0);
            *v1240 = *v1239;
            v1234 += 1 ;
        }
        assert("Tensor range check" && 0 <= v997 && v997 < 8);
        int v1241;
        v1241 = 8 * v997;
        int v1242;
        v1242 = v1241 + v990;
        v15[v1242] = v1230;
        v997 += 1 ;
    }
    __syncthreads();
    int v1243;
    v1243 = threadIdx.x;
    unsigned long long v1244;
    v1244 = (unsigned long long)v1243;
    curandStatePhilox4_32_10_t v1245;
    curand_init(12344321ull,v1244,0ull,&v1245);
    int v1246;
    v1246 = threadIdx.x;
    bool v1247;
    v1247 = 0 <= v1246;
    bool v1248;
    v1248 = v1247 == false;
    if (v1248){
        assert("The index needs to be zero or positive." && v1247);
    } else {
    }
    int v1250;
    v1250 = v1246 % 32;
    int v1251;
    v1251 = v1246 / 32;
    bool v1252;
    v1252 = v1251 < 8;
    bool v1253;
    v1253 = v1252 == false;
    if (v1253){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1252);
    } else {
    }
    assert("Tensor range check" && 0 <= v1251 && v1251 < 8);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 32);
    int v1255;
    v1255 = 4 * v1250;
    int v1256;
    v1256 = 128 * v1251;
    int v1257;
    v1257 = v1256 + v1255;
    assert("Tensor range check" && 0 <= v1251 && v1251 < 8);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 32);
    assert("Tensor range check" && 0 <= v1251 && v1251 < 8);
    int v1258;
    v1258 = 0;
    while (while_method_2(v1258)){
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1260;
        v1260 = 1024 * v1258;
        int v1261;
        v1261 = v1260 + v1257;
        float v1262[4];
        int v1263[4];
        int v1264;
        v1264 = 0;
        while (while_method_3(v1264)){
            assert("Tensor range check" && 0 <= v1264 && v1264 < 1);
            int v1266;
            v1266 = 4 * v1264;
            assert("Tensor range check" && 0 <= v1264 && v1264 < 1);
            int v1267;
            v1267 = 128 * v1264;
            int v1268;
            v1268 = v1267 + v1261;
            int4* v1269;
            v1269 = reinterpret_cast<int4*>(v1 + v1268);
            int4* v1270;
            v1270 = reinterpret_cast<int4*>(v1262 + v1266);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1269) % 16 == 0 && reinterpret_cast<unsigned long long>(v1270) % 16 == 0);
            *v1270 = *v1269;
            v1264 += 1 ;
        }
        int v1271;
        v1271 = 0;
        while (while_method_3(v1271)){
            int v1273;
            v1273 = 0;
            while (while_method_1(v1273)){
                bool v1275;
                v1275 = 0 <= v1273;
                bool v1277;
                if (v1275){
                    bool v1276;
                    v1276 = v1273 < 4;
                    v1277 = v1276;
                } else {
                    v1277 = false;
                }
                bool v1278;
                v1278 = v1277 == false;
                if (v1278){
                    assert("The indices should be inside the range of the dimension." && v1277);
                } else {
                }
                bool v1280;
                v1280 = 0 <= v1250;
                bool v1282;
                if (v1280){
                    bool v1281;
                    v1281 = v1250 < 32;
                    v1282 = v1281;
                } else {
                    v1282 = false;
                }
                bool v1283;
                v1283 = v1282 == false;
                if (v1283){
                    assert("The indices should be inside the range of the dimension." && v1282);
                } else {
                }
                int v1285;
                v1285 = v1250 * 4;
                int v1286;
                v1286 = v1273 + v1285;
                bool v1287;
                v1287 = 0 <= v1271;
                bool v1289;
                if (v1287){
                    bool v1288;
                    v1288 = v1271 < 1;
                    v1289 = v1288;
                } else {
                    v1289 = false;
                }
                bool v1290;
                v1290 = v1289 == false;
                if (v1290){
                    assert("The indices should be inside the range of the dimension." && v1289);
                } else {
                }
                int v1292;
                v1292 = v1271 * 128;
                int v1293;
                v1293 = v1286 + v1292;
                assert("Tensor range check" && 0 <= v1271 && v1271 < 1);
                assert("Tensor range check" && 0 <= v1273 && v1273 < 4);
                int v1294;
                v1294 = 4 * v1271;
                int v1295;
                v1295 = v1294 + v1273;
                v1263[v1295] = v1293;
                v1273 += 1 ;
            }
            v1271 += 1 ;
        }
        bool v1296;
        v1296 = 0 <= v1251;
        bool v1297;
        v1297 = v1296 && v1252;
        bool v1298;
        v1298 = v1297 == false;
        if (v1298){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1297);
        } else {
        }
        bool v1300;
        v1300 = 0 <= v1258;
        bool v1302;
        if (v1300){
            bool v1301;
            v1301 = v1258 < 8;
            v1302 = v1301;
        } else {
            v1302 = false;
        }
        bool v1303;
        v1303 = v1302 == false;
        if (v1303){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1302);
        } else {
        }
        int v1305;
        v1305 = v1258 * 8;
        int v1306;
        v1306 = v1305 + v1251;
        bool v1307[4];
        int v1308;
        v1308 = 0;
        while (while_method_3(v1308)){
            int v1310;
            v1310 = 0;
            while (while_method_1(v1310)){
                assert("Tensor range check" && 0 <= v1308 && v1308 < 1);
                assert("Tensor range check" && 0 <= v1310 && v1310 < 4);
                int v1312;
                v1312 = 4 * v1308;
                int v1313;
                v1313 = v1312 + v1310;
                float v1314;
                v1314 = v1262[v1313];
                int v1315;
                v1315 = v1263[v1313];
                bool v1316;
                v1316 = v1315 < 11;
                assert("Tensor range check" && 0 <= v1308 && v1308 < 1);
                assert("Tensor range check" && 0 <= v1310 && v1310 < 4);
                v1307[v1313] = v1316;
                v1310 += 1 ;
            }
            v1308 += 1 ;
        }
        float v1317[4];
        int v1318;
        v1318 = 0;
        while (while_method_3(v1318)){
            int v1320;
            v1320 = 0;
            while (while_method_1(v1320)){
                assert("Tensor range check" && 0 <= v1318 && v1318 < 1);
                assert("Tensor range check" && 0 <= v1320 && v1320 < 4);
                int v1322;
                v1322 = 4 * v1318;
                int v1323;
                v1323 = v1322 + v1320;
                float v1324;
                v1324 = v1262[v1323];
                bool v1325;
                v1325 = v1307[v1323];
                float v1326;
                if (v1325){
                    v1326 = v1324;
                } else {
                    v1326 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1318 && v1318 < 1);
                assert("Tensor range check" && 0 <= v1320 && v1320 < 4);
                v1317[v1323] = v1326;
                v1320 += 1 ;
            }
            v1318 += 1 ;
        }
        float v1327;
        v1327 = 0.0f;
        int v1328;
        v1328 = 0;
        while (while_method_3(v1328)){
            int v1330;
            v1330 = 0;
            while (while_method_1(v1330)){
                assert("Tensor range check" && 0 <= v1328 && v1328 < 1);
                assert("Tensor range check" && 0 <= v1330 && v1330 < 4);
                int v1332;
                v1332 = 4 * v1328;
                int v1333;
                v1333 = v1332 + v1330;
                float v1334;
                v1334 = v1317[v1333];
                float v1335;
                v1335 = v1327 + v1334;
                v1327 = v1335;
                v1330 += 1 ;
            }
            v1328 += 1 ;
        }
        auto v1336 = cooperative_groups::coalesced_threads();
        int v1337;
        v1337 = threadIdx.x;
        int v1338;
        v1338 = v1337 / 32;
        auto v1339 = cooperative_groups::labeled_partition(v1336,v1338);
        float v1340;
        v1340 = cooperative_groups::reduce(v1339, v1327, v42);
        int v1341[4];
        int v1342;
        v1342 = 0;
        while (while_method_3(v1342)){
            int v1344;
            v1344 = 0;
            while (while_method_1(v1344)){
                assert("Tensor range check" && 0 <= v1342 && v1342 < 1);
                assert("Tensor range check" && 0 <= v1344 && v1344 < 4);
                int v1346;
                v1346 = 4 * v1342;
                int v1347;
                v1347 = v1346 + v1344;
                bool v1348;
                v1348 = v1307[v1347];
                int v1349;
                if (v1348){
                    v1349 = 1;
                } else {
                    v1349 = 0;
                }
                assert("Tensor range check" && 0 <= v1342 && v1342 < 1);
                assert("Tensor range check" && 0 <= v1344 && v1344 < 4);
                v1341[v1347] = v1349;
                v1344 += 1 ;
            }
            v1342 += 1 ;
        }
        int v1350;
        v1350 = 0;
        int v1351;
        v1351 = 0;
        while (while_method_3(v1351)){
            int v1353;
            v1353 = 0;
            while (while_method_1(v1353)){
                assert("Tensor range check" && 0 <= v1351 && v1351 < 1);
                assert("Tensor range check" && 0 <= v1353 && v1353 < 4);
                int v1355;
                v1355 = 4 * v1351;
                int v1356;
                v1356 = v1355 + v1353;
                int v1357;
                v1357 = v1341[v1356];
                int v1358;
                v1358 = v1350 + v1357;
                v1350 = v1358;
                v1353 += 1 ;
            }
            v1351 += 1 ;
        }
        auto v1359 = cooperative_groups::coalesced_threads();
        int v1360;
        v1360 = threadIdx.x;
        int v1361;
        v1361 = v1360 / 32;
        auto v1362 = cooperative_groups::labeled_partition(v1359,v1361);
        Closure4 v1363{};
        int v1364;
        v1364 = cooperative_groups::reduce(v1362, v1350, v1363);
        float v1365;
        v1365 = (float)v1364;
        float v1366;
        v1366 = v1340 / v1365;
        float v1367[4];
        int v1368;
        v1368 = 0;
        while (while_method_3(v1368)){
            int v1370;
            v1370 = 0;
            while (while_method_1(v1370)){
                assert("Tensor range check" && 0 <= v1368 && v1368 < 1);
                assert("Tensor range check" && 0 <= v1370 && v1370 < 4);
                int v1372;
                v1372 = 4 * v1368;
                int v1373;
                v1373 = v1372 + v1370;
                float v1374;
                v1374 = v1262[v1373];
                bool v1375;
                v1375 = v1307[v1373];
                float v1376;
                if (v1375){
                    v1376 = v1374;
                } else {
                    v1376 = -1.0f / 0.0f;
                }
                float v1377;
                v1377 = v1376 - v1366;
                float v1378;
                v1378 = exp(v1377);
                bool v1379;
                v1379 = v1378 < 1.0f / 0.0f;
                bool v1380;
                v1380 = v1379 == false;
                if (v1380){
                    assert("The softmax values must not grow too large." && v1379);
                } else {
                }
                bool v1382;
                v1382 = isnan(v1378);
                bool v1383;
                v1383 = v1382 == false;
                bool v1384;
                v1384 = v1383 == false;
                if (v1384){
                    assert("The softmax values must not be nans." && v1383);
                } else {
                }
                assert("Tensor range check" && 0 <= v1368 && v1368 < 1);
                assert("Tensor range check" && 0 <= v1370 && v1370 < 4);
                v1367[v1373] = v1378;
                v1370 += 1 ;
            }
            v1368 += 1 ;
        }
        float v1386;
        v1386 = 0.0f;
        int v1387;
        v1387 = 0;
        while (while_method_3(v1387)){
            int v1389;
            v1389 = 0;
            while (while_method_1(v1389)){
                assert("Tensor range check" && 0 <= v1387 && v1387 < 1);
                assert("Tensor range check" && 0 <= v1389 && v1389 < 4);
                int v1391;
                v1391 = 4 * v1387;
                int v1392;
                v1392 = v1391 + v1389;
                float v1393;
                v1393 = v1367[v1392];
                float v1394;
                v1394 = v1386 + v1393;
                v1386 = v1394;
                v1389 += 1 ;
            }
            v1387 += 1 ;
        }
        auto v1395 = cooperative_groups::coalesced_threads();
        int v1396;
        v1396 = threadIdx.x;
        int v1397;
        v1397 = v1396 / 32;
        auto v1398 = cooperative_groups::labeled_partition(v1395,v1397);
        float v1399;
        v1399 = cooperative_groups::reduce(v1398, v1386, v42);
        float v1400[4];
        int v1401;
        v1401 = 0;
        while (while_method_3(v1401)){
            int v1403;
            v1403 = 0;
            while (while_method_1(v1403)){
                assert("Tensor range check" && 0 <= v1401 && v1401 < 1);
                assert("Tensor range check" && 0 <= v1403 && v1403 < 4);
                int v1405;
                v1405 = 4 * v1401;
                int v1406;
                v1406 = v1405 + v1403;
                float v1407;
                v1407 = v1367[v1406];
                float v1408;
                v1408 = v1407 / v1399;
                assert("Tensor range check" && 0 <= v1401 && v1401 < 1);
                assert("Tensor range check" && 0 <= v1403 && v1403 < 4);
                v1400[v1406] = v1408;
                v1403 += 1 ;
            }
            v1401 += 1 ;
        }
        float v1409[4];
        float v1410;
        v1410 = 0.0f;
        int v1411;
        v1411 = 0;
        while (while_method_3(v1411)){
            assert("Tensor range check" && 0 <= v1411 && v1411 < 1);
            int v1413;
            v1413 = 4 * v1411;
            assert("Tensor range check" && 0 <= v1411 && v1411 < 1);
            float v1414;
            v1414 = 0.0f;
            int v1415;
            v1415 = 0;
            while (while_method_1(v1415)){
                assert("Tensor range check" && 0 <= v1415 && v1415 < 4);
                int v1417;
                v1417 = v1415 + v1413;
                float v1418;
                v1418 = v1400[v1417];
                float v1419;
                v1419 = v1414 + v1418;
                v1414 = v1419;
                v1415 += 1 ;
            }
            auto v1420 = cooperative_groups::coalesced_threads();
            int v1421;
            v1421 = threadIdx.x;
            int v1422;
            v1422 = v1421 / 32;
            auto v1423 = cooperative_groups::labeled_partition(v1420,v1422);
            Closure2 v1424{};
            float v1425;
            v1425 = cooperative_groups::inclusive_scan(v1423, v1414, v1424);
            float v1426;
            v1426 = v1423.shfl_up(v1425,1);
            bool v1427;
            v1427 = v1423.thread_rank() == 0;
            float v1428;
            if (v1427){
                v1428 = 0.0f;
            } else {
                v1428 = v1426;
            }
            float v1429;
            v1429 = v1423.shfl(v1425,v1423.num_threads()-1);
            float v1430;
            v1430 = v1410 + v1428;
            float v1431;
            v1431 = v1430;
            int v1432;
            v1432 = 0;
            while (while_method_1(v1432)){
                assert("Tensor range check" && 0 <= v1432 && v1432 < 4);
                int v1434;
                v1434 = v1432 + v1413;
                float v1435;
                v1435 = v1400[v1434];
                float v1436;
                v1436 = v1431 + v1435;
                assert("Tensor range check" && 0 <= v1432 && v1432 < 4);
                v1409[v1434] = v1436;
                v1431 = v1436;
                v1432 += 1 ;
            }
            float v1437;
            v1437 = v1410 + v1429;
            v1410 = v1437;
            v1411 += 1 ;
        }
        float v1438[4];
        bool v1439[4];
        int v1440;
        v1440 = 0;
        while (while_method_3(v1440)){
            int v1442;
            v1442 = 0;
            while (while_method_1(v1442)){
                assert("Tensor range check" && 0 <= v1440 && v1440 < 1);
                assert("Tensor range check" && 0 <= v1442 && v1442 < 4);
                int v1444;
                v1444 = 4 * v1440;
                int v1445;
                v1445 = v1444 + v1442;
                float v1446;
                v1446 = v1409[v1445];
                float v1447;
                v1447 = v1400[v1445];
                bool v1448;
                v1448 = v1447 > 0.0f;
                assert("Tensor range check" && 0 <= v1440 && v1440 < 1);
                assert("Tensor range check" && 0 <= v1442 && v1442 < 4);
                v1438[v1445] = v1446;
                v1439[v1445] = v1448;
                v1442 += 1 ;
            }
            v1440 += 1 ;
        }
        float v1449; bool v1450;
        Tuple2 tmp9 = Tuple2{-1.0f / 0.0f, false};
        v1449 = tmp9.v0; v1450 = tmp9.v1;
        int v1451;
        v1451 = 0;
        while (while_method_3(v1451)){
            int v1453;
            v1453 = 0;
            while (while_method_1(v1453)){
                assert("Tensor range check" && 0 <= v1451 && v1451 < 1);
                assert("Tensor range check" && 0 <= v1453 && v1453 < 4);
                int v1455;
                v1455 = 4 * v1451;
                int v1456;
                v1456 = v1455 + v1453;
                float v1457;
                v1457 = v1438[v1456];
                bool v1458;
                v1458 = v1439[v1456];
                float v1465; bool v1466;
                if (v1450){
                    if (v1458){
                        bool v1459;
                        v1459 = v1449 >= v1457;
                        float v1460;
                        if (v1459){
                            v1460 = v1449;
                        } else {
                            v1460 = v1457;
                        }
                        v1465 = v1460; v1466 = true;
                    } else {
                        v1465 = v1449; v1466 = v1450;
                    }
                } else {
                    if (v1458){
                        v1465 = v1457; v1466 = v1458;
                    } else {
                        v1465 = v1449; v1466 = v1450;
                    }
                }
                v1449 = v1465;
                v1450 = v1466;
                v1453 += 1 ;
            }
            v1451 += 1 ;
        }
        auto v1467 = cooperative_groups::coalesced_threads();
        int v1468;
        v1468 = threadIdx.x;
        int v1469;
        v1469 = v1468 / 32;
        auto v1470 = cooperative_groups::labeled_partition(v1467,v1469);
        Closure5 v1471{};
        float v1472; bool v1473;
        Tuple2 tmp10 = cooperative_groups::reduce(v1470, Tuple2{v1449, v1450}, v1471);
        v1472 = tmp10.v0; v1473 = tmp10.v1;
        bool v1474;
        v1474 = v1473 == false;
        if (v1474){
            assert("The local reduce must be true." && v1473);
        } else {
        }
        float v1476[4];
        int v1477[4];
        int v1478;
        v1478 = 0;
        while (while_method_3(v1478)){
            int v1480;
            v1480 = 0;
            while (while_method_1(v1480)){
                assert("Tensor range check" && 0 <= v1478 && v1478 < 1);
                assert("Tensor range check" && 0 <= v1480 && v1480 < 4);
                int v1482;
                v1482 = 4 * v1478;
                int v1483;
                v1483 = v1482 + v1480;
                int v1484;
                v1484 = v1263[v1483];
                float v1485;
                v1485 = curand_uniform(&v1245);
                assert("Tensor range check" && 0 <= v1478 && v1478 < 1);
                assert("Tensor range check" && 0 <= v1480 && v1480 < 4);
                v1476[v1483] = v1485;
                v1477[v1483] = v1484;
                v1480 += 1 ;
            }
            v1478 += 1 ;
        }
        float v1486; int v1487;
        Tuple1 tmp11 = Tuple1{0.0f, 2147483647};
        v1486 = tmp11.v0; v1487 = tmp11.v1;
        int v1488;
        v1488 = 0;
        while (while_method_3(v1488)){
            int v1490;
            v1490 = 0;
            while (while_method_1(v1490)){
                assert("Tensor range check" && 0 <= v1488 && v1488 < 1);
                assert("Tensor range check" && 0 <= v1490 && v1490 < 4);
                int v1492;
                v1492 = 4 * v1488;
                int v1493;
                v1493 = v1492 + v1490;
                float v1494;
                v1494 = v1476[v1493];
                int v1495;
                v1495 = v1477[v1493];
                bool v1496;
                v1496 = v1487 < v1495;
                float v1497; int v1498;
                if (v1496){
                    v1497 = v1486; v1498 = v1487;
                } else {
                    v1497 = v1494; v1498 = v1495;
                }
                v1486 = v1497;
                v1487 = v1498;
                v1490 += 1 ;
            }
            v1488 += 1 ;
        }
        auto v1499 = cooperative_groups::coalesced_threads();
        int v1500;
        v1500 = threadIdx.x;
        int v1501;
        v1501 = v1500 / 32;
        auto v1502 = cooperative_groups::labeled_partition(v1499,v1501);
        Closure6 v1503{};
        float v1504; int v1505;
        Tuple1 tmp12 = cooperative_groups::reduce(v1502, Tuple1{v1486, v1487}, v1503);
        v1504 = tmp12.v0; v1505 = tmp12.v1;
        float v1506;
        v1506 = v1472 * v1504;
        int v1507[4];
        bool v1508[4];
        int v1509;
        v1509 = 0;
        while (while_method_3(v1509)){
            int v1511;
            v1511 = 0;
            while (while_method_1(v1511)){
                assert("Tensor range check" && 0 <= v1509 && v1509 < 1);
                assert("Tensor range check" && 0 <= v1511 && v1511 < 4);
                int v1513;
                v1513 = 4 * v1509;
                int v1514;
                v1514 = v1513 + v1511;
                float v1515;
                v1515 = v1438[v1514];
                bool v1516;
                v1516 = v1439[v1514];
                int v1517;
                v1517 = v1263[v1514];
                int v1520; bool v1521;
                if (v1516){
                    float v1518;
                    v1518 = v1515 - v1506;
                    bool v1519;
                    v1519 = v1518 >= 0.0f;
                    v1520 = v1517; v1521 = v1519;
                } else {
                    v1520 = 2147483647; v1521 = false;
                }
                assert("Tensor range check" && 0 <= v1509 && v1509 < 1);
                assert("Tensor range check" && 0 <= v1511 && v1511 < 4);
                v1507[v1514] = v1520;
                v1508[v1514] = v1521;
                v1511 += 1 ;
            }
            v1509 += 1 ;
        }
        int v1522; bool v1523;
        Tuple3 tmp13 = Tuple3{2147483647, false};
        v1522 = tmp13.v0; v1523 = tmp13.v1;
        int v1524;
        v1524 = 0;
        while (while_method_3(v1524)){
            int v1526;
            v1526 = 0;
            while (while_method_1(v1526)){
                assert("Tensor range check" && 0 <= v1524 && v1524 < 1);
                assert("Tensor range check" && 0 <= v1526 && v1526 < 4);
                int v1528;
                v1528 = 4 * v1524;
                int v1529;
                v1529 = v1528 + v1526;
                int v1530;
                v1530 = v1507[v1529];
                bool v1531;
                v1531 = v1508[v1529];
                int v1538; bool v1539;
                if (v1523){
                    if (v1531){
                        bool v1532;
                        v1532 = v1522 < v1530;
                        int v1533;
                        if (v1532){
                            v1533 = v1522;
                        } else {
                            v1533 = v1530;
                        }
                        v1538 = v1533; v1539 = true;
                    } else {
                        v1538 = v1522; v1539 = v1523;
                    }
                } else {
                    if (v1531){
                        v1538 = v1530; v1539 = v1531;
                    } else {
                        v1538 = v1522; v1539 = v1523;
                    }
                }
                v1522 = v1538;
                v1523 = v1539;
                v1526 += 1 ;
            }
            v1524 += 1 ;
        }
        auto v1540 = cooperative_groups::coalesced_threads();
        int v1541;
        v1541 = threadIdx.x;
        int v1542;
        v1542 = v1541 / 32;
        auto v1543 = cooperative_groups::labeled_partition(v1540,v1542);
        Closure7 v1544{};
        int v1545; bool v1546;
        Tuple3 tmp14 = cooperative_groups::reduce(v1543, Tuple3{v1522, v1523}, v1544);
        v1545 = tmp14.v0; v1546 = tmp14.v1;
        bool v1547;
        v1547 = v1546 == false;
        if (v1547){
            assert("The local reduce must be true." && v1546);
        } else {
        }
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1549;
        v1549 = 0;
        while (while_method_3(v1549)){
            assert("Tensor range check" && 0 <= v1549 && v1549 < 1);
            int v1551;
            v1551 = 128 * v1549;
            int v1552;
            v1552 = v1551 + v1261;
            assert("Tensor range check" && 0 <= v1549 && v1549 < 1);
            int v1553;
            v1553 = 4 * v1549;
            int4* v1554;
            v1554 = reinterpret_cast<int4*>(v1400 + v1553);
            int4* v1555;
            v1555 = reinterpret_cast<int4*>(v16 + v1552);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1554) % 16 == 0 && reinterpret_cast<unsigned long long>(v1555) % 16 == 0);
            *v1555 = *v1554;
            v1549 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1556;
        v1556 = 8 * v1258;
        int v1557;
        v1557 = v1556 + v1251;
        v17[v1557] = v1545;
        v1258 += 1 ;
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
    unsigned long long v983;
    v983 = (unsigned long long)v982;
    curandStatePhilox4_32_10_t v984;
    curand_init(12344321ull,v983,0ull,&v984);
    int v985;
    v985 = threadIdx.x;
    bool v986;
    v986 = 0 <= v985;
    bool v987;
    v987 = v986 == false;
    if (v987){
        assert("The index needs to be zero or positive." && v986);
    } else {
    }
    int v989;
    v989 = v985 % 16;
    int v990;
    v990 = v985 / 16;
    bool v991;
    v991 = v990 < 16;
    bool v992;
    v992 = v991 == false;
    if (v992){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v991);
    } else {
    }
    assert("Tensor range check" && 0 <= v990 && v990 < 16);
    assert("Tensor range check" && 0 <= v989 && v989 < 16);
    int v994;
    v994 = 4 * v989;
    int v995;
    v995 = 64 * v990;
    int v996;
    v996 = v995 + v994;
    assert("Tensor range check" && 0 <= v990 && v990 < 16);
    assert("Tensor range check" && 0 <= v989 && v989 < 16);
    assert("Tensor range check" && 0 <= v990 && v990 < 16);
    int v997;
    v997 = 0;
    while (while_method_2(v997)){
        assert("Tensor range check" && 0 <= v997 && v997 < 8);
        int v999;
        v999 = 1024 * v997;
        int v1000;
        v1000 = v999 + v996;
        float v1001[4];
        int v1002[4];
        int v1003;
        v1003 = 0;
        while (while_method_3(v1003)){
            assert("Tensor range check" && 0 <= v1003 && v1003 < 1);
            int v1005;
            v1005 = 4 * v1003;
            assert("Tensor range check" && 0 <= v1003 && v1003 < 1);
            int v1006;
            v1006 = 64 * v1003;
            int v1007;
            v1007 = v1006 + v1000;
            int4* v1008;
            v1008 = reinterpret_cast<int4*>(v1 + v1007);
            int4* v1009;
            v1009 = reinterpret_cast<int4*>(v1001 + v1005);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1008) % 16 == 0 && reinterpret_cast<unsigned long long>(v1009) % 16 == 0);
            *v1009 = *v1008;
            v1003 += 1 ;
        }
        int v1010;
        v1010 = 0;
        while (while_method_3(v1010)){
            int v1012;
            v1012 = 0;
            while (while_method_1(v1012)){
                bool v1014;
                v1014 = 0 <= v1012;
                bool v1016;
                if (v1014){
                    bool v1015;
                    v1015 = v1012 < 4;
                    v1016 = v1015;
                } else {
                    v1016 = false;
                }
                bool v1017;
                v1017 = v1016 == false;
                if (v1017){
                    assert("The indices should be inside the range of the dimension." && v1016);
                } else {
                }
                bool v1019;
                v1019 = 0 <= v989;
                bool v1021;
                if (v1019){
                    bool v1020;
                    v1020 = v989 < 16;
                    v1021 = v1020;
                } else {
                    v1021 = false;
                }
                bool v1022;
                v1022 = v1021 == false;
                if (v1022){
                    assert("The indices should be inside the range of the dimension." && v1021);
                } else {
                }
                int v1024;
                v1024 = v989 * 4;
                int v1025;
                v1025 = v1012 + v1024;
                bool v1026;
                v1026 = 0 <= v1010;
                bool v1028;
                if (v1026){
                    bool v1027;
                    v1027 = v1010 < 1;
                    v1028 = v1027;
                } else {
                    v1028 = false;
                }
                bool v1029;
                v1029 = v1028 == false;
                if (v1029){
                    assert("The indices should be inside the range of the dimension." && v1028);
                } else {
                }
                int v1031;
                v1031 = v1010 * 64;
                int v1032;
                v1032 = v1025 + v1031;
                assert("Tensor range check" && 0 <= v1010 && v1010 < 1);
                assert("Tensor range check" && 0 <= v1012 && v1012 < 4);
                int v1033;
                v1033 = 4 * v1010;
                int v1034;
                v1034 = v1033 + v1012;
                v1002[v1034] = v1032;
                v1012 += 1 ;
            }
            v1010 += 1 ;
        }
        bool v1035;
        v1035 = 0 <= v990;
        bool v1036;
        v1036 = v1035 && v991;
        bool v1037;
        v1037 = v1036 == false;
        if (v1037){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1036);
        } else {
        }
        bool v1039;
        v1039 = 0 <= v997;
        bool v1041;
        if (v1039){
            bool v1040;
            v1040 = v997 < 8;
            v1041 = v1040;
        } else {
            v1041 = false;
        }
        bool v1042;
        v1042 = v1041 == false;
        if (v1042){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1041);
        } else {
        }
        int v1044;
        v1044 = v997 * 16;
        int v1045;
        v1045 = v1044 + v990;
        float v1046;
        v1046 = 0.0f;
        int v1047;
        v1047 = 0;
        while (while_method_3(v1047)){
            int v1049;
            v1049 = 0;
            while (while_method_1(v1049)){
                assert("Tensor range check" && 0 <= v1047 && v1047 < 1);
                assert("Tensor range check" && 0 <= v1049 && v1049 < 4);
                int v1051;
                v1051 = 4 * v1047;
                int v1052;
                v1052 = v1051 + v1049;
                float v1053;
                v1053 = v1001[v1052];
                float v1054;
                v1054 = v1046 + v1053;
                v1046 = v1054;
                v1049 += 1 ;
            }
            v1047 += 1 ;
        }
        auto v1055 = cooperative_groups::coalesced_threads();
        int v1056;
        v1056 = threadIdx.x;
        int v1057;
        v1057 = v1056 / 16;
        auto v1058 = cooperative_groups::labeled_partition(v1055,v1057);
        float v1059;
        v1059 = cooperative_groups::reduce(v1058, v1046, v42);
        float v1060;
        v1060 = v1059 / 64.0f;
        float v1061[4];
        int v1062;
        v1062 = 0;
        while (while_method_3(v1062)){
            int v1064;
            v1064 = 0;
            while (while_method_1(v1064)){
                assert("Tensor range check" && 0 <= v1062 && v1062 < 1);
                assert("Tensor range check" && 0 <= v1064 && v1064 < 4);
                int v1066;
                v1066 = 4 * v1062;
                int v1067;
                v1067 = v1066 + v1064;
                float v1068;
                v1068 = v1001[v1067];
                float v1069;
                v1069 = v1068 - v1060;
                float v1070;
                v1070 = exp(v1069);
                assert("Tensor range check" && 0 <= v1062 && v1062 < 1);
                assert("Tensor range check" && 0 <= v1064 && v1064 < 4);
                v1061[v1067] = v1070;
                v1064 += 1 ;
            }
            v1062 += 1 ;
        }
        float v1071;
        v1071 = 0.0f;
        int v1072;
        v1072 = 0;
        while (while_method_3(v1072)){
            int v1074;
            v1074 = 0;
            while (while_method_1(v1074)){
                assert("Tensor range check" && 0 <= v1072 && v1072 < 1);
                assert("Tensor range check" && 0 <= v1074 && v1074 < 4);
                int v1076;
                v1076 = 4 * v1072;
                int v1077;
                v1077 = v1076 + v1074;
                float v1078;
                v1078 = v1061[v1077];
                float v1079;
                v1079 = v1071 + v1078;
                v1071 = v1079;
                v1074 += 1 ;
            }
            v1072 += 1 ;
        }
        auto v1080 = cooperative_groups::coalesced_threads();
        int v1081;
        v1081 = threadIdx.x;
        int v1082;
        v1082 = v1081 / 16;
        auto v1083 = cooperative_groups::labeled_partition(v1080,v1082);
        float v1084;
        v1084 = cooperative_groups::reduce(v1083, v1071, v42);
        float v1085[4];
        int v1086;
        v1086 = 0;
        while (while_method_3(v1086)){
            int v1088;
            v1088 = 0;
            while (while_method_1(v1088)){
                assert("Tensor range check" && 0 <= v1086 && v1086 < 1);
                assert("Tensor range check" && 0 <= v1088 && v1088 < 4);
                int v1090;
                v1090 = 4 * v1086;
                int v1091;
                v1091 = v1090 + v1088;
                float v1092;
                v1092 = v1061[v1091];
                float v1093;
                v1093 = v1092 / v1084;
                assert("Tensor range check" && 0 <= v1086 && v1086 < 1);
                assert("Tensor range check" && 0 <= v1088 && v1088 < 4);
                v1085[v1091] = v1093;
                v1088 += 1 ;
            }
            v1086 += 1 ;
        }
        float v1094[4];
        float v1095;
        v1095 = 0.0f;
        int v1096;
        v1096 = 0;
        while (while_method_3(v1096)){
            assert("Tensor range check" && 0 <= v1096 && v1096 < 1);
            int v1098;
            v1098 = 4 * v1096;
            assert("Tensor range check" && 0 <= v1096 && v1096 < 1);
            float v1099;
            v1099 = 0.0f;
            int v1100;
            v1100 = 0;
            while (while_method_1(v1100)){
                assert("Tensor range check" && 0 <= v1100 && v1100 < 4);
                int v1102;
                v1102 = v1100 + v1098;
                float v1103;
                v1103 = v1085[v1102];
                float v1104;
                v1104 = v1099 + v1103;
                v1099 = v1104;
                v1100 += 1 ;
            }
            auto v1105 = cooperative_groups::coalesced_threads();
            int v1106;
            v1106 = threadIdx.x;
            int v1107;
            v1107 = v1106 / 16;
            auto v1108 = cooperative_groups::labeled_partition(v1105,v1107);
            Closure2 v1109{};
            float v1110;
            v1110 = cooperative_groups::inclusive_scan(v1108, v1099, v1109);
            float v1111;
            v1111 = v1108.shfl_up(v1110,1);
            bool v1112;
            v1112 = v1108.thread_rank() == 0;
            float v1113;
            if (v1112){
                v1113 = 0.0f;
            } else {
                v1113 = v1111;
            }
            float v1114;
            v1114 = v1108.shfl(v1110,v1108.num_threads()-1);
            float v1115;
            v1115 = v1095 + v1113;
            float v1116;
            v1116 = v1115;
            int v1117;
            v1117 = 0;
            while (while_method_1(v1117)){
                assert("Tensor range check" && 0 <= v1117 && v1117 < 4);
                int v1119;
                v1119 = v1117 + v1098;
                float v1120;
                v1120 = v1085[v1119];
                float v1121;
                v1121 = v1116 + v1120;
                assert("Tensor range check" && 0 <= v1117 && v1117 < 4);
                v1094[v1119] = v1121;
                v1116 = v1121;
                v1117 += 1 ;
            }
            float v1122;
            v1122 = v1095 + v1114;
            v1095 = v1122;
            v1096 += 1 ;
        }
        float v1123[4];
        bool v1124[4];
        int v1125;
        v1125 = 0;
        while (while_method_3(v1125)){
            int v1127;
            v1127 = 0;
            while (while_method_1(v1127)){
                assert("Tensor range check" && 0 <= v1125 && v1125 < 1);
                assert("Tensor range check" && 0 <= v1127 && v1127 < 4);
                int v1129;
                v1129 = 4 * v1125;
                int v1130;
                v1130 = v1129 + v1127;
                float v1131;
                v1131 = v1094[v1130];
                float v1132;
                v1132 = v1085[v1130];
                bool v1133;
                v1133 = v1132 > 0.0f;
                assert("Tensor range check" && 0 <= v1125 && v1125 < 1);
                assert("Tensor range check" && 0 <= v1127 && v1127 < 4);
                v1123[v1130] = v1131;
                v1124[v1130] = v1133;
                v1127 += 1 ;
            }
            v1125 += 1 ;
        }
        float v1134; bool v1135;
        Tuple2 tmp18 = Tuple2{-1.0f / 0.0f, false};
        v1134 = tmp18.v0; v1135 = tmp18.v1;
        int v1136;
        v1136 = 0;
        while (while_method_3(v1136)){
            int v1138;
            v1138 = 0;
            while (while_method_1(v1138)){
                assert("Tensor range check" && 0 <= v1136 && v1136 < 1);
                assert("Tensor range check" && 0 <= v1138 && v1138 < 4);
                int v1140;
                v1140 = 4 * v1136;
                int v1141;
                v1141 = v1140 + v1138;
                float v1142;
                v1142 = v1123[v1141];
                bool v1143;
                v1143 = v1124[v1141];
                float v1150; bool v1151;
                if (v1135){
                    if (v1143){
                        bool v1144;
                        v1144 = v1134 >= v1142;
                        float v1145;
                        if (v1144){
                            v1145 = v1134;
                        } else {
                            v1145 = v1142;
                        }
                        v1150 = v1145; v1151 = true;
                    } else {
                        v1150 = v1134; v1151 = v1135;
                    }
                } else {
                    if (v1143){
                        v1150 = v1142; v1151 = v1143;
                    } else {
                        v1150 = v1134; v1151 = v1135;
                    }
                }
                v1134 = v1150;
                v1135 = v1151;
                v1138 += 1 ;
            }
            v1136 += 1 ;
        }
        auto v1152 = cooperative_groups::coalesced_threads();
        int v1153;
        v1153 = threadIdx.x;
        int v1154;
        v1154 = v1153 / 16;
        auto v1155 = cooperative_groups::labeled_partition(v1152,v1154);
        Closure5 v1156{};
        float v1157; bool v1158;
        Tuple2 tmp19 = cooperative_groups::reduce(v1155, Tuple2{v1134, v1135}, v1156);
        v1157 = tmp19.v0; v1158 = tmp19.v1;
        bool v1159;
        v1159 = v1158 == false;
        if (v1159){
            assert("The local reduce must be true." && v1158);
        } else {
        }
        float v1161[4];
        int v1162[4];
        int v1163;
        v1163 = 0;
        while (while_method_3(v1163)){
            int v1165;
            v1165 = 0;
            while (while_method_1(v1165)){
                assert("Tensor range check" && 0 <= v1163 && v1163 < 1);
                assert("Tensor range check" && 0 <= v1165 && v1165 < 4);
                int v1167;
                v1167 = 4 * v1163;
                int v1168;
                v1168 = v1167 + v1165;
                int v1169;
                v1169 = v1002[v1168];
                float v1170;
                v1170 = curand_uniform(&v984);
                assert("Tensor range check" && 0 <= v1163 && v1163 < 1);
                assert("Tensor range check" && 0 <= v1165 && v1165 < 4);
                v1161[v1168] = v1170;
                v1162[v1168] = v1169;
                v1165 += 1 ;
            }
            v1163 += 1 ;
        }
        float v1171; int v1172;
        Tuple1 tmp20 = Tuple1{0.0f, 2147483647};
        v1171 = tmp20.v0; v1172 = tmp20.v1;
        int v1173;
        v1173 = 0;
        while (while_method_3(v1173)){
            int v1175;
            v1175 = 0;
            while (while_method_1(v1175)){
                assert("Tensor range check" && 0 <= v1173 && v1173 < 1);
                assert("Tensor range check" && 0 <= v1175 && v1175 < 4);
                int v1177;
                v1177 = 4 * v1173;
                int v1178;
                v1178 = v1177 + v1175;
                float v1179;
                v1179 = v1161[v1178];
                int v1180;
                v1180 = v1162[v1178];
                bool v1181;
                v1181 = v1172 < v1180;
                float v1182; int v1183;
                if (v1181){
                    v1182 = v1171; v1183 = v1172;
                } else {
                    v1182 = v1179; v1183 = v1180;
                }
                v1171 = v1182;
                v1172 = v1183;
                v1175 += 1 ;
            }
            v1173 += 1 ;
        }
        auto v1184 = cooperative_groups::coalesced_threads();
        int v1185;
        v1185 = threadIdx.x;
        int v1186;
        v1186 = v1185 / 16;
        auto v1187 = cooperative_groups::labeled_partition(v1184,v1186);
        Closure6 v1188{};
        float v1189; int v1190;
        Tuple1 tmp21 = cooperative_groups::reduce(v1187, Tuple1{v1171, v1172}, v1188);
        v1189 = tmp21.v0; v1190 = tmp21.v1;
        float v1191;
        v1191 = v1157 * v1189;
        int v1192[4];
        bool v1193[4];
        int v1194;
        v1194 = 0;
        while (while_method_3(v1194)){
            int v1196;
            v1196 = 0;
            while (while_method_1(v1196)){
                assert("Tensor range check" && 0 <= v1194 && v1194 < 1);
                assert("Tensor range check" && 0 <= v1196 && v1196 < 4);
                int v1198;
                v1198 = 4 * v1194;
                int v1199;
                v1199 = v1198 + v1196;
                float v1200;
                v1200 = v1123[v1199];
                bool v1201;
                v1201 = v1124[v1199];
                int v1202;
                v1202 = v1002[v1199];
                int v1205; bool v1206;
                if (v1201){
                    float v1203;
                    v1203 = v1200 - v1191;
                    bool v1204;
                    v1204 = v1203 >= 0.0f;
                    v1205 = v1202; v1206 = v1204;
                } else {
                    v1205 = 2147483647; v1206 = false;
                }
                assert("Tensor range check" && 0 <= v1194 && v1194 < 1);
                assert("Tensor range check" && 0 <= v1196 && v1196 < 4);
                v1192[v1199] = v1205;
                v1193[v1199] = v1206;
                v1196 += 1 ;
            }
            v1194 += 1 ;
        }
        int v1207; bool v1208;
        Tuple3 tmp22 = Tuple3{2147483647, false};
        v1207 = tmp22.v0; v1208 = tmp22.v1;
        int v1209;
        v1209 = 0;
        while (while_method_3(v1209)){
            int v1211;
            v1211 = 0;
            while (while_method_1(v1211)){
                assert("Tensor range check" && 0 <= v1209 && v1209 < 1);
                assert("Tensor range check" && 0 <= v1211 && v1211 < 4);
                int v1213;
                v1213 = 4 * v1209;
                int v1214;
                v1214 = v1213 + v1211;
                int v1215;
                v1215 = v1192[v1214];
                bool v1216;
                v1216 = v1193[v1214];
                int v1223; bool v1224;
                if (v1208){
                    if (v1216){
                        bool v1217;
                        v1217 = v1207 < v1215;
                        int v1218;
                        if (v1217){
                            v1218 = v1207;
                        } else {
                            v1218 = v1215;
                        }
                        v1223 = v1218; v1224 = true;
                    } else {
                        v1223 = v1207; v1224 = v1208;
                    }
                } else {
                    if (v1216){
                        v1223 = v1215; v1224 = v1216;
                    } else {
                        v1223 = v1207; v1224 = v1208;
                    }
                }
                v1207 = v1223;
                v1208 = v1224;
                v1211 += 1 ;
            }
            v1209 += 1 ;
        }
        auto v1225 = cooperative_groups::coalesced_threads();
        int v1226;
        v1226 = threadIdx.x;
        int v1227;
        v1227 = v1226 / 16;
        auto v1228 = cooperative_groups::labeled_partition(v1225,v1227);
        Closure7 v1229{};
        int v1230; bool v1231;
        Tuple3 tmp23 = cooperative_groups::reduce(v1228, Tuple3{v1207, v1208}, v1229);
        v1230 = tmp23.v0; v1231 = tmp23.v1;
        bool v1232;
        v1232 = v1231 == false;
        if (v1232){
            assert("The local reduce must be true." && v1231);
        } else {
        }
        assert("Tensor range check" && 0 <= v997 && v997 < 8);
        int v1234;
        v1234 = 0;
        while (while_method_3(v1234)){
            assert("Tensor range check" && 0 <= v1234 && v1234 < 1);
            int v1236;
            v1236 = 64 * v1234;
            int v1237;
            v1237 = v1236 + v1000;
            assert("Tensor range check" && 0 <= v1234 && v1234 < 1);
            int v1238;
            v1238 = 4 * v1234;
            int4* v1239;
            v1239 = reinterpret_cast<int4*>(v1085 + v1238);
            int4* v1240;
            v1240 = reinterpret_cast<int4*>(v14 + v1237);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1239) % 16 == 0 && reinterpret_cast<unsigned long long>(v1240) % 16 == 0);
            *v1240 = *v1239;
            v1234 += 1 ;
        }
        assert("Tensor range check" && 0 <= v997 && v997 < 8);
        int v1241;
        v1241 = 16 * v997;
        int v1242;
        v1242 = v1241 + v990;
        v15[v1242] = v1230;
        v997 += 1 ;
    }
    __syncthreads();
    int v1243;
    v1243 = threadIdx.x;
    unsigned long long v1244;
    v1244 = (unsigned long long)v1243;
    curandStatePhilox4_32_10_t v1245;
    curand_init(12344321ull,v1244,0ull,&v1245);
    int v1246;
    v1246 = threadIdx.x;
    bool v1247;
    v1247 = 0 <= v1246;
    bool v1248;
    v1248 = v1247 == false;
    if (v1248){
        assert("The index needs to be zero or positive." && v1247);
    } else {
    }
    int v1250;
    v1250 = v1246 % 16;
    int v1251;
    v1251 = v1246 / 16;
    bool v1252;
    v1252 = v1251 < 16;
    bool v1253;
    v1253 = v1252 == false;
    if (v1253){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1252);
    } else {
    }
    assert("Tensor range check" && 0 <= v1251 && v1251 < 16);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 16);
    int v1255;
    v1255 = 4 * v1250;
    int v1256;
    v1256 = 64 * v1251;
    int v1257;
    v1257 = v1256 + v1255;
    assert("Tensor range check" && 0 <= v1251 && v1251 < 16);
    assert("Tensor range check" && 0 <= v1250 && v1250 < 16);
    assert("Tensor range check" && 0 <= v1251 && v1251 < 16);
    int v1258;
    v1258 = 0;
    while (while_method_2(v1258)){
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1260;
        v1260 = 1024 * v1258;
        int v1261;
        v1261 = v1260 + v1257;
        float v1262[4];
        int v1263[4];
        int v1264;
        v1264 = 0;
        while (while_method_3(v1264)){
            assert("Tensor range check" && 0 <= v1264 && v1264 < 1);
            int v1266;
            v1266 = 4 * v1264;
            assert("Tensor range check" && 0 <= v1264 && v1264 < 1);
            int v1267;
            v1267 = 64 * v1264;
            int v1268;
            v1268 = v1267 + v1261;
            int4* v1269;
            v1269 = reinterpret_cast<int4*>(v1 + v1268);
            int4* v1270;
            v1270 = reinterpret_cast<int4*>(v1262 + v1266);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1269) % 16 == 0 && reinterpret_cast<unsigned long long>(v1270) % 16 == 0);
            *v1270 = *v1269;
            v1264 += 1 ;
        }
        int v1271;
        v1271 = 0;
        while (while_method_3(v1271)){
            int v1273;
            v1273 = 0;
            while (while_method_1(v1273)){
                bool v1275;
                v1275 = 0 <= v1273;
                bool v1277;
                if (v1275){
                    bool v1276;
                    v1276 = v1273 < 4;
                    v1277 = v1276;
                } else {
                    v1277 = false;
                }
                bool v1278;
                v1278 = v1277 == false;
                if (v1278){
                    assert("The indices should be inside the range of the dimension." && v1277);
                } else {
                }
                bool v1280;
                v1280 = 0 <= v1250;
                bool v1282;
                if (v1280){
                    bool v1281;
                    v1281 = v1250 < 16;
                    v1282 = v1281;
                } else {
                    v1282 = false;
                }
                bool v1283;
                v1283 = v1282 == false;
                if (v1283){
                    assert("The indices should be inside the range of the dimension." && v1282);
                } else {
                }
                int v1285;
                v1285 = v1250 * 4;
                int v1286;
                v1286 = v1273 + v1285;
                bool v1287;
                v1287 = 0 <= v1271;
                bool v1289;
                if (v1287){
                    bool v1288;
                    v1288 = v1271 < 1;
                    v1289 = v1288;
                } else {
                    v1289 = false;
                }
                bool v1290;
                v1290 = v1289 == false;
                if (v1290){
                    assert("The indices should be inside the range of the dimension." && v1289);
                } else {
                }
                int v1292;
                v1292 = v1271 * 64;
                int v1293;
                v1293 = v1286 + v1292;
                assert("Tensor range check" && 0 <= v1271 && v1271 < 1);
                assert("Tensor range check" && 0 <= v1273 && v1273 < 4);
                int v1294;
                v1294 = 4 * v1271;
                int v1295;
                v1295 = v1294 + v1273;
                v1263[v1295] = v1293;
                v1273 += 1 ;
            }
            v1271 += 1 ;
        }
        bool v1296;
        v1296 = 0 <= v1251;
        bool v1297;
        v1297 = v1296 && v1252;
        bool v1298;
        v1298 = v1297 == false;
        if (v1298){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1297);
        } else {
        }
        bool v1300;
        v1300 = 0 <= v1258;
        bool v1302;
        if (v1300){
            bool v1301;
            v1301 = v1258 < 8;
            v1302 = v1301;
        } else {
            v1302 = false;
        }
        bool v1303;
        v1303 = v1302 == false;
        if (v1303){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v1302);
        } else {
        }
        int v1305;
        v1305 = v1258 * 16;
        int v1306;
        v1306 = v1305 + v1251;
        bool v1307[4];
        int v1308;
        v1308 = 0;
        while (while_method_3(v1308)){
            int v1310;
            v1310 = 0;
            while (while_method_1(v1310)){
                assert("Tensor range check" && 0 <= v1308 && v1308 < 1);
                assert("Tensor range check" && 0 <= v1310 && v1310 < 4);
                int v1312;
                v1312 = 4 * v1308;
                int v1313;
                v1313 = v1312 + v1310;
                float v1314;
                v1314 = v1262[v1313];
                int v1315;
                v1315 = v1263[v1313];
                bool v1316;
                v1316 = v1315 < 11;
                assert("Tensor range check" && 0 <= v1308 && v1308 < 1);
                assert("Tensor range check" && 0 <= v1310 && v1310 < 4);
                v1307[v1313] = v1316;
                v1310 += 1 ;
            }
            v1308 += 1 ;
        }
        float v1317[4];
        int v1318;
        v1318 = 0;
        while (while_method_3(v1318)){
            int v1320;
            v1320 = 0;
            while (while_method_1(v1320)){
                assert("Tensor range check" && 0 <= v1318 && v1318 < 1);
                assert("Tensor range check" && 0 <= v1320 && v1320 < 4);
                int v1322;
                v1322 = 4 * v1318;
                int v1323;
                v1323 = v1322 + v1320;
                float v1324;
                v1324 = v1262[v1323];
                bool v1325;
                v1325 = v1307[v1323];
                float v1326;
                if (v1325){
                    v1326 = v1324;
                } else {
                    v1326 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v1318 && v1318 < 1);
                assert("Tensor range check" && 0 <= v1320 && v1320 < 4);
                v1317[v1323] = v1326;
                v1320 += 1 ;
            }
            v1318 += 1 ;
        }
        float v1327;
        v1327 = 0.0f;
        int v1328;
        v1328 = 0;
        while (while_method_3(v1328)){
            int v1330;
            v1330 = 0;
            while (while_method_1(v1330)){
                assert("Tensor range check" && 0 <= v1328 && v1328 < 1);
                assert("Tensor range check" && 0 <= v1330 && v1330 < 4);
                int v1332;
                v1332 = 4 * v1328;
                int v1333;
                v1333 = v1332 + v1330;
                float v1334;
                v1334 = v1317[v1333];
                float v1335;
                v1335 = v1327 + v1334;
                v1327 = v1335;
                v1330 += 1 ;
            }
            v1328 += 1 ;
        }
        auto v1336 = cooperative_groups::coalesced_threads();
        int v1337;
        v1337 = threadIdx.x;
        int v1338;
        v1338 = v1337 / 16;
        auto v1339 = cooperative_groups::labeled_partition(v1336,v1338);
        float v1340;
        v1340 = cooperative_groups::reduce(v1339, v1327, v42);
        int v1341[4];
        int v1342;
        v1342 = 0;
        while (while_method_3(v1342)){
            int v1344;
            v1344 = 0;
            while (while_method_1(v1344)){
                assert("Tensor range check" && 0 <= v1342 && v1342 < 1);
                assert("Tensor range check" && 0 <= v1344 && v1344 < 4);
                int v1346;
                v1346 = 4 * v1342;
                int v1347;
                v1347 = v1346 + v1344;
                bool v1348;
                v1348 = v1307[v1347];
                int v1349;
                if (v1348){
                    v1349 = 1;
                } else {
                    v1349 = 0;
                }
                assert("Tensor range check" && 0 <= v1342 && v1342 < 1);
                assert("Tensor range check" && 0 <= v1344 && v1344 < 4);
                v1341[v1347] = v1349;
                v1344 += 1 ;
            }
            v1342 += 1 ;
        }
        int v1350;
        v1350 = 0;
        int v1351;
        v1351 = 0;
        while (while_method_3(v1351)){
            int v1353;
            v1353 = 0;
            while (while_method_1(v1353)){
                assert("Tensor range check" && 0 <= v1351 && v1351 < 1);
                assert("Tensor range check" && 0 <= v1353 && v1353 < 4);
                int v1355;
                v1355 = 4 * v1351;
                int v1356;
                v1356 = v1355 + v1353;
                int v1357;
                v1357 = v1341[v1356];
                int v1358;
                v1358 = v1350 + v1357;
                v1350 = v1358;
                v1353 += 1 ;
            }
            v1351 += 1 ;
        }
        auto v1359 = cooperative_groups::coalesced_threads();
        int v1360;
        v1360 = threadIdx.x;
        int v1361;
        v1361 = v1360 / 16;
        auto v1362 = cooperative_groups::labeled_partition(v1359,v1361);
        Closure4 v1363{};
        int v1364;
        v1364 = cooperative_groups::reduce(v1362, v1350, v1363);
        float v1365;
        v1365 = (float)v1364;
        float v1366;
        v1366 = v1340 / v1365;
        float v1367[4];
        int v1368;
        v1368 = 0;
        while (while_method_3(v1368)){
            int v1370;
            v1370 = 0;
            while (while_method_1(v1370)){
                assert("Tensor range check" && 0 <= v1368 && v1368 < 1);
                assert("Tensor range check" && 0 <= v1370 && v1370 < 4);
                int v1372;
                v1372 = 4 * v1368;
                int v1373;
                v1373 = v1372 + v1370;
                float v1374;
                v1374 = v1262[v1373];
                bool v1375;
                v1375 = v1307[v1373];
                float v1376;
                if (v1375){
                    v1376 = v1374;
                } else {
                    v1376 = -1.0f / 0.0f;
                }
                float v1377;
                v1377 = v1376 - v1366;
                float v1378;
                v1378 = exp(v1377);
                bool v1379;
                v1379 = v1378 < 1.0f / 0.0f;
                bool v1380;
                v1380 = v1379 == false;
                if (v1380){
                    assert("The softmax values must not grow too large." && v1379);
                } else {
                }
                bool v1382;
                v1382 = isnan(v1378);
                bool v1383;
                v1383 = v1382 == false;
                bool v1384;
                v1384 = v1383 == false;
                if (v1384){
                    assert("The softmax values must not be nans." && v1383);
                } else {
                }
                assert("Tensor range check" && 0 <= v1368 && v1368 < 1);
                assert("Tensor range check" && 0 <= v1370 && v1370 < 4);
                v1367[v1373] = v1378;
                v1370 += 1 ;
            }
            v1368 += 1 ;
        }
        float v1386;
        v1386 = 0.0f;
        int v1387;
        v1387 = 0;
        while (while_method_3(v1387)){
            int v1389;
            v1389 = 0;
            while (while_method_1(v1389)){
                assert("Tensor range check" && 0 <= v1387 && v1387 < 1);
                assert("Tensor range check" && 0 <= v1389 && v1389 < 4);
                int v1391;
                v1391 = 4 * v1387;
                int v1392;
                v1392 = v1391 + v1389;
                float v1393;
                v1393 = v1367[v1392];
                float v1394;
                v1394 = v1386 + v1393;
                v1386 = v1394;
                v1389 += 1 ;
            }
            v1387 += 1 ;
        }
        auto v1395 = cooperative_groups::coalesced_threads();
        int v1396;
        v1396 = threadIdx.x;
        int v1397;
        v1397 = v1396 / 16;
        auto v1398 = cooperative_groups::labeled_partition(v1395,v1397);
        float v1399;
        v1399 = cooperative_groups::reduce(v1398, v1386, v42);
        float v1400[4];
        int v1401;
        v1401 = 0;
        while (while_method_3(v1401)){
            int v1403;
            v1403 = 0;
            while (while_method_1(v1403)){
                assert("Tensor range check" && 0 <= v1401 && v1401 < 1);
                assert("Tensor range check" && 0 <= v1403 && v1403 < 4);
                int v1405;
                v1405 = 4 * v1401;
                int v1406;
                v1406 = v1405 + v1403;
                float v1407;
                v1407 = v1367[v1406];
                float v1408;
                v1408 = v1407 / v1399;
                assert("Tensor range check" && 0 <= v1401 && v1401 < 1);
                assert("Tensor range check" && 0 <= v1403 && v1403 < 4);
                v1400[v1406] = v1408;
                v1403 += 1 ;
            }
            v1401 += 1 ;
        }
        float v1409[4];
        float v1410;
        v1410 = 0.0f;
        int v1411;
        v1411 = 0;
        while (while_method_3(v1411)){
            assert("Tensor range check" && 0 <= v1411 && v1411 < 1);
            int v1413;
            v1413 = 4 * v1411;
            assert("Tensor range check" && 0 <= v1411 && v1411 < 1);
            float v1414;
            v1414 = 0.0f;
            int v1415;
            v1415 = 0;
            while (while_method_1(v1415)){
                assert("Tensor range check" && 0 <= v1415 && v1415 < 4);
                int v1417;
                v1417 = v1415 + v1413;
                float v1418;
                v1418 = v1400[v1417];
                float v1419;
                v1419 = v1414 + v1418;
                v1414 = v1419;
                v1415 += 1 ;
            }
            auto v1420 = cooperative_groups::coalesced_threads();
            int v1421;
            v1421 = threadIdx.x;
            int v1422;
            v1422 = v1421 / 16;
            auto v1423 = cooperative_groups::labeled_partition(v1420,v1422);
            Closure2 v1424{};
            float v1425;
            v1425 = cooperative_groups::inclusive_scan(v1423, v1414, v1424);
            float v1426;
            v1426 = v1423.shfl_up(v1425,1);
            bool v1427;
            v1427 = v1423.thread_rank() == 0;
            float v1428;
            if (v1427){
                v1428 = 0.0f;
            } else {
                v1428 = v1426;
            }
            float v1429;
            v1429 = v1423.shfl(v1425,v1423.num_threads()-1);
            float v1430;
            v1430 = v1410 + v1428;
            float v1431;
            v1431 = v1430;
            int v1432;
            v1432 = 0;
            while (while_method_1(v1432)){
                assert("Tensor range check" && 0 <= v1432 && v1432 < 4);
                int v1434;
                v1434 = v1432 + v1413;
                float v1435;
                v1435 = v1400[v1434];
                float v1436;
                v1436 = v1431 + v1435;
                assert("Tensor range check" && 0 <= v1432 && v1432 < 4);
                v1409[v1434] = v1436;
                v1431 = v1436;
                v1432 += 1 ;
            }
            float v1437;
            v1437 = v1410 + v1429;
            v1410 = v1437;
            v1411 += 1 ;
        }
        float v1438[4];
        bool v1439[4];
        int v1440;
        v1440 = 0;
        while (while_method_3(v1440)){
            int v1442;
            v1442 = 0;
            while (while_method_1(v1442)){
                assert("Tensor range check" && 0 <= v1440 && v1440 < 1);
                assert("Tensor range check" && 0 <= v1442 && v1442 < 4);
                int v1444;
                v1444 = 4 * v1440;
                int v1445;
                v1445 = v1444 + v1442;
                float v1446;
                v1446 = v1409[v1445];
                float v1447;
                v1447 = v1400[v1445];
                bool v1448;
                v1448 = v1447 > 0.0f;
                assert("Tensor range check" && 0 <= v1440 && v1440 < 1);
                assert("Tensor range check" && 0 <= v1442 && v1442 < 4);
                v1438[v1445] = v1446;
                v1439[v1445] = v1448;
                v1442 += 1 ;
            }
            v1440 += 1 ;
        }
        float v1449; bool v1450;
        Tuple2 tmp24 = Tuple2{-1.0f / 0.0f, false};
        v1449 = tmp24.v0; v1450 = tmp24.v1;
        int v1451;
        v1451 = 0;
        while (while_method_3(v1451)){
            int v1453;
            v1453 = 0;
            while (while_method_1(v1453)){
                assert("Tensor range check" && 0 <= v1451 && v1451 < 1);
                assert("Tensor range check" && 0 <= v1453 && v1453 < 4);
                int v1455;
                v1455 = 4 * v1451;
                int v1456;
                v1456 = v1455 + v1453;
                float v1457;
                v1457 = v1438[v1456];
                bool v1458;
                v1458 = v1439[v1456];
                float v1465; bool v1466;
                if (v1450){
                    if (v1458){
                        bool v1459;
                        v1459 = v1449 >= v1457;
                        float v1460;
                        if (v1459){
                            v1460 = v1449;
                        } else {
                            v1460 = v1457;
                        }
                        v1465 = v1460; v1466 = true;
                    } else {
                        v1465 = v1449; v1466 = v1450;
                    }
                } else {
                    if (v1458){
                        v1465 = v1457; v1466 = v1458;
                    } else {
                        v1465 = v1449; v1466 = v1450;
                    }
                }
                v1449 = v1465;
                v1450 = v1466;
                v1453 += 1 ;
            }
            v1451 += 1 ;
        }
        auto v1467 = cooperative_groups::coalesced_threads();
        int v1468;
        v1468 = threadIdx.x;
        int v1469;
        v1469 = v1468 / 16;
        auto v1470 = cooperative_groups::labeled_partition(v1467,v1469);
        Closure5 v1471{};
        float v1472; bool v1473;
        Tuple2 tmp25 = cooperative_groups::reduce(v1470, Tuple2{v1449, v1450}, v1471);
        v1472 = tmp25.v0; v1473 = tmp25.v1;
        bool v1474;
        v1474 = v1473 == false;
        if (v1474){
            assert("The local reduce must be true." && v1473);
        } else {
        }
        float v1476[4];
        int v1477[4];
        int v1478;
        v1478 = 0;
        while (while_method_3(v1478)){
            int v1480;
            v1480 = 0;
            while (while_method_1(v1480)){
                assert("Tensor range check" && 0 <= v1478 && v1478 < 1);
                assert("Tensor range check" && 0 <= v1480 && v1480 < 4);
                int v1482;
                v1482 = 4 * v1478;
                int v1483;
                v1483 = v1482 + v1480;
                int v1484;
                v1484 = v1263[v1483];
                float v1485;
                v1485 = curand_uniform(&v1245);
                assert("Tensor range check" && 0 <= v1478 && v1478 < 1);
                assert("Tensor range check" && 0 <= v1480 && v1480 < 4);
                v1476[v1483] = v1485;
                v1477[v1483] = v1484;
                v1480 += 1 ;
            }
            v1478 += 1 ;
        }
        float v1486; int v1487;
        Tuple1 tmp26 = Tuple1{0.0f, 2147483647};
        v1486 = tmp26.v0; v1487 = tmp26.v1;
        int v1488;
        v1488 = 0;
        while (while_method_3(v1488)){
            int v1490;
            v1490 = 0;
            while (while_method_1(v1490)){
                assert("Tensor range check" && 0 <= v1488 && v1488 < 1);
                assert("Tensor range check" && 0 <= v1490 && v1490 < 4);
                int v1492;
                v1492 = 4 * v1488;
                int v1493;
                v1493 = v1492 + v1490;
                float v1494;
                v1494 = v1476[v1493];
                int v1495;
                v1495 = v1477[v1493];
                bool v1496;
                v1496 = v1487 < v1495;
                float v1497; int v1498;
                if (v1496){
                    v1497 = v1486; v1498 = v1487;
                } else {
                    v1497 = v1494; v1498 = v1495;
                }
                v1486 = v1497;
                v1487 = v1498;
                v1490 += 1 ;
            }
            v1488 += 1 ;
        }
        auto v1499 = cooperative_groups::coalesced_threads();
        int v1500;
        v1500 = threadIdx.x;
        int v1501;
        v1501 = v1500 / 16;
        auto v1502 = cooperative_groups::labeled_partition(v1499,v1501);
        Closure6 v1503{};
        float v1504; int v1505;
        Tuple1 tmp27 = cooperative_groups::reduce(v1502, Tuple1{v1486, v1487}, v1503);
        v1504 = tmp27.v0; v1505 = tmp27.v1;
        float v1506;
        v1506 = v1472 * v1504;
        int v1507[4];
        bool v1508[4];
        int v1509;
        v1509 = 0;
        while (while_method_3(v1509)){
            int v1511;
            v1511 = 0;
            while (while_method_1(v1511)){
                assert("Tensor range check" && 0 <= v1509 && v1509 < 1);
                assert("Tensor range check" && 0 <= v1511 && v1511 < 4);
                int v1513;
                v1513 = 4 * v1509;
                int v1514;
                v1514 = v1513 + v1511;
                float v1515;
                v1515 = v1438[v1514];
                bool v1516;
                v1516 = v1439[v1514];
                int v1517;
                v1517 = v1263[v1514];
                int v1520; bool v1521;
                if (v1516){
                    float v1518;
                    v1518 = v1515 - v1506;
                    bool v1519;
                    v1519 = v1518 >= 0.0f;
                    v1520 = v1517; v1521 = v1519;
                } else {
                    v1520 = 2147483647; v1521 = false;
                }
                assert("Tensor range check" && 0 <= v1509 && v1509 < 1);
                assert("Tensor range check" && 0 <= v1511 && v1511 < 4);
                v1507[v1514] = v1520;
                v1508[v1514] = v1521;
                v1511 += 1 ;
            }
            v1509 += 1 ;
        }
        int v1522; bool v1523;
        Tuple3 tmp28 = Tuple3{2147483647, false};
        v1522 = tmp28.v0; v1523 = tmp28.v1;
        int v1524;
        v1524 = 0;
        while (while_method_3(v1524)){
            int v1526;
            v1526 = 0;
            while (while_method_1(v1526)){
                assert("Tensor range check" && 0 <= v1524 && v1524 < 1);
                assert("Tensor range check" && 0 <= v1526 && v1526 < 4);
                int v1528;
                v1528 = 4 * v1524;
                int v1529;
                v1529 = v1528 + v1526;
                int v1530;
                v1530 = v1507[v1529];
                bool v1531;
                v1531 = v1508[v1529];
                int v1538; bool v1539;
                if (v1523){
                    if (v1531){
                        bool v1532;
                        v1532 = v1522 < v1530;
                        int v1533;
                        if (v1532){
                            v1533 = v1522;
                        } else {
                            v1533 = v1530;
                        }
                        v1538 = v1533; v1539 = true;
                    } else {
                        v1538 = v1522; v1539 = v1523;
                    }
                } else {
                    if (v1531){
                        v1538 = v1530; v1539 = v1531;
                    } else {
                        v1538 = v1522; v1539 = v1523;
                    }
                }
                v1522 = v1538;
                v1523 = v1539;
                v1526 += 1 ;
            }
            v1524 += 1 ;
        }
        auto v1540 = cooperative_groups::coalesced_threads();
        int v1541;
        v1541 = threadIdx.x;
        int v1542;
        v1542 = v1541 / 16;
        auto v1543 = cooperative_groups::labeled_partition(v1540,v1542);
        Closure7 v1544{};
        int v1545; bool v1546;
        Tuple3 tmp29 = cooperative_groups::reduce(v1543, Tuple3{v1522, v1523}, v1544);
        v1545 = tmp29.v0; v1546 = tmp29.v1;
        bool v1547;
        v1547 = v1546 == false;
        if (v1547){
            assert("The local reduce must be true." && v1546);
        } else {
        }
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1549;
        v1549 = 0;
        while (while_method_3(v1549)){
            assert("Tensor range check" && 0 <= v1549 && v1549 < 1);
            int v1551;
            v1551 = 64 * v1549;
            int v1552;
            v1552 = v1551 + v1261;
            assert("Tensor range check" && 0 <= v1549 && v1549 < 1);
            int v1553;
            v1553 = 4 * v1549;
            int4* v1554;
            v1554 = reinterpret_cast<int4*>(v1400 + v1553);
            int4* v1555;
            v1555 = reinterpret_cast<int4*>(v16 + v1552);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v1554) % 16 == 0 && reinterpret_cast<unsigned long long>(v1555) % 16 == 0);
            *v1555 = *v1554;
            v1549 += 1 ;
        }
        assert("Tensor range check" && 0 <= v1258 && v1258 < 8);
        int v1556;
        v1556 = 16 * v1258;
        int v1557;
        v1557 = v1556 + v1251;
        v17[v1557] = v1545;
        v1258 += 1 ;
    }
    __syncthreads();
    return ;
}
extern "C" __global__ void entry2(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    int v9;
    v9 = blockIdx.x;
    int v10;
    v10 = v9 * 256;
    int v11;
    v11 = v8 + v10;
    assert("Tensor range check" && 0 <= v11 && v11 < 6144);
    int v12;
    v12 = 16 * v11;
    int v13;
    v13 = threadIdx.x;
    int v14;
    v14 = blockIdx.x;
    int v15;
    v15 = v14 * 256;
    int v16;
    v16 = v13 + v15;
    assert("Tensor range check" && 0 <= v16 && v16 < 6144);
    int v17;
    v17 = 16 * v16;
    int v18;
    v18 = threadIdx.x;
    int v19;
    v19 = blockIdx.x;
    int v20;
    v20 = v19 * 256;
    int v21;
    v21 = v18 + v20;
    assert("Tensor range check" && 0 <= v21 && v21 < 6144);
    int v22;
    v22 = 16 * v21;
    int v23;
    v23 = threadIdx.x;
    int v24;
    v24 = blockIdx.x;
    int v25;
    v25 = v24 * 256;
    int v26;
    v26 = v23 + v25;
    assert("Tensor range check" && 0 <= v26 && v26 < 6144);
    int v27;
    v27 = 16 * v26;
    int v28;
    v28 = threadIdx.x;
    int v29;
    v29 = blockIdx.x;
    int v30;
    v30 = v29 * 256;
    int v31;
    v31 = v28 + v30;
    assert("Tensor range check" && 0 <= v31 && v31 < 6144);
    int v32;
    v32 = 16 * v31;
    float * v33;
    v33 = v1+v12;
    int * v35;
    v35 = v2+v27;
    int * v37;
    v37 = v3+v27;
    int v39;
    v39 = sizeof(float *);
    unsigned long long v40;
    v40 = (unsigned long long)v39;
    unsigned long long v41;
    v41 = 256ull * v40;
    unsigned long long v42;
    v42 = v41 + 16ull;
    unsigned long long v43;
    v43 = v42 - 1ull;
    unsigned long long v44;
    v44 = v43 % 16ull;
    unsigned long long v45;
    v45 = v43 - v44;
    int v46;
    v46 = sizeof(int *);
    unsigned long long v47;
    v47 = (unsigned long long)v46;
    unsigned long long v48;
    v48 = 256ull * v47;
    unsigned long long v49;
    v49 = v45 + v48;
    unsigned long long v50;
    v50 = v49 + 16ull;
    unsigned long long v51;
    v51 = v50 - 1ull;
    unsigned long long v52;
    v52 = v51 % 16ull;
    unsigned long long v53;
    v53 = v51 - v52;
    unsigned long long v54;
    v54 = v53 + v48;
    bool v55;
    v55 = v54 <= 98304ull;
    bool v56;
    v56 = v55 == false;
    if (v56){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v55);
    } else {
    }
    extern __shared__ unsigned char v58[];
    bool v59;
    v59 = v54 <= v54;
    bool v60;
    v60 = v59 == false;
    if (v60){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v59);
    } else {
    }
    float * * v62;
    v62 = reinterpret_cast<float * *>(&v58[0ull]);
    int * * v64;
    v64 = reinterpret_cast<int * *>(&v58[v45]);
    int * * v66;
    v66 = reinterpret_cast<int * *>(&v58[v53]);
    int v68;
    v68 = threadIdx.x;
    assert("Tensor range check" && 0 <= v68 && v68 < 256);
    v62[v68] = v33;
    v64[v68] = v35;
    v66[v68] = v37;
    __syncthreads();
    bool v69;
    v69 = 0 <= v68;
    bool v70;
    v70 = v69 == false;
    if (v70){
        assert("The index needs to be zero or positive." && v69);
    } else {
    }
    int v72;
    v72 = v68 % 4;
    int v73;
    v73 = v68 / 4;
    bool v74;
    v74 = v73 < 64;
    bool v75;
    v75 = v74 == false;
    if (v75){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v74);
    } else {
    }
    assert("Tensor range check" && 0 <= v73 && v73 < 64);
    int v77;
    v77 = 0;
    while (while_method_1(v77)){
        bool v79;
        v79 = 0 <= v73;
        bool v80;
        v80 = v79 && v74;
        bool v81;
        v81 = v80 == false;
        if (v81){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v80);
        } else {
        }
        bool v83;
        v83 = 0 <= v77;
        bool v85;
        if (v83){
            bool v84;
            v84 = v77 < 4;
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
        int v88;
        v88 = v77 * 64;
        int v89;
        v89 = v88 + v73;
        assert("Tensor range check" && 0 <= v77 && v77 < 4);
        int v90;
        v90 = 64 * v77;
        int v91;
        v91 = v90 + v73;
        float * v92;
        v92 = v62[v91];
        int * v93;
        v93 = v64[v91];
        int * v94;
        v94 = v66[v91];
        int v95;
        v95 = blockIdx.x;
        int v96;
        v96 = v95 * 256;
        int v97;
        v97 = v96 + v89;
        assert("Tensor range check" && 0 <= v72 && v72 < 4);
        int v98;
        v98 = 4 * v72;
        float v99[4];
        int v100[4];
        int v101;
        v101 = 0;
        while (while_method_3(v101)){
            assert("Tensor range check" && 0 <= v101 && v101 < 1);
            int v103;
            v103 = 4 * v101;
            assert("Tensor range check" && 0 <= v101 && v101 < 1);
            int v104;
            v104 = 16 * v101;
            int v105;
            v105 = v104 + v98;
            int4* v106;
            v106 = reinterpret_cast<int4*>(v92 + v105);
            int4* v107;
            v107 = reinterpret_cast<int4*>(v99 + v103);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v106) % 16 == 0 && reinterpret_cast<unsigned long long>(v107) % 16 == 0);
            *v107 = *v106;
            v101 += 1 ;
        }
        int v108;
        v108 = 0;
        while (while_method_3(v108)){
            int v110;
            v110 = 0;
            while (while_method_1(v110)){
                bool v112;
                v112 = 0 <= v110;
                bool v114;
                if (v112){
                    bool v113;
                    v113 = v110 < 4;
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
                bool v117;
                v117 = 0 <= v72;
                bool v119;
                if (v117){
                    bool v118;
                    v118 = v72 < 4;
                    v119 = v118;
                } else {
                    v119 = false;
                }
                bool v120;
                v120 = v119 == false;
                if (v120){
                    assert("The indices should be inside the range of the dimension." && v119);
                } else {
                }
                int v122;
                v122 = v72 * 4;
                int v123;
                v123 = v110 + v122;
                bool v124;
                v124 = 0 <= v108;
                bool v126;
                if (v124){
                    bool v125;
                    v125 = v108 < 1;
                    v126 = v125;
                } else {
                    v126 = false;
                }
                bool v127;
                v127 = v126 == false;
                if (v127){
                    assert("The indices should be inside the range of the dimension." && v126);
                } else {
                }
                int v129;
                v129 = v108 * 16;
                int v130;
                v130 = v123 + v129;
                assert("Tensor range check" && 0 <= v108 && v108 < 1);
                assert("Tensor range check" && 0 <= v110 && v110 < 4);
                int v131;
                v131 = 4 * v108;
                int v132;
                v132 = v131 + v110;
                v100[v132] = v130;
                v110 += 1 ;
            }
            v108 += 1 ;
        }
        int v133[4];
        int v134[4];
        int v135;
        v135 = 0;
        while (while_method_3(v135)){
            int v137;
            v137 = 0;
            while (while_method_1(v137)){
                assert("Tensor range check" && 0 <= v135 && v135 < 1);
                assert("Tensor range check" && 0 <= v137 && v137 < 4);
                int v139;
                v139 = 4 * v135;
                int v140;
                v140 = v139 + v137;
                int v141;
                v141 = v100[v140];
                assert("Tensor range check" && 0 <= v135 && v135 < 1);
                assert("Tensor range check" && 0 <= v137 && v137 < 4);
                v133[v140] = v97;
                v134[v140] = v141;
                v137 += 1 ;
            }
            v135 += 1 ;
        }
        int v142;
        v142 = 0;
        while (while_method_3(v142)){
            assert("Tensor range check" && 0 <= v142 && v142 < 1);
            int v144;
            v144 = 16 * v142;
            int v145;
            v145 = v144 + v98;
            assert("Tensor range check" && 0 <= v142 && v142 < 1);
            int v146;
            v146 = 4 * v142;
            int4* v147;
            v147 = reinterpret_cast<int4*>(v133 + v146);
            int4* v148;
            v148 = reinterpret_cast<int4*>(v93 + v145);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v147) % 16 == 0 && reinterpret_cast<unsigned long long>(v148) % 16 == 0);
            *v148 = *v147;
            int4* v149;
            v149 = reinterpret_cast<int4*>(v134 + v146);
            int4* v150;
            v150 = reinterpret_cast<int4*>(v94 + v145);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v149) % 16 == 0 && reinterpret_cast<unsigned long long>(v150) % 16 == 0);
            *v150 = *v149;
            v142 += 1 ;
        }
        assert("Tensor range check" && 0 <= v89 && v89 < 256);
        v77 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v68 && v68 < 256);
    __syncthreads();
    float * v151;
    v151 = v1+v12;
    unsigned long long v153;
    v153 = v45 + 1024ull;
    bool v154;
    v154 = v153 <= 98304ull;
    bool v155;
    v155 = v154 == false;
    if (v155){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v154);
    } else {
    }
    extern __shared__ unsigned char v157[];
    bool v158;
    v158 = v153 <= v153;
    bool v159;
    v159 = v158 == false;
    if (v159){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v158);
    } else {
    }
    float * * v161;
    v161 = reinterpret_cast<float * *>(&v157[0ull]);
    int * v163;
    v163 = reinterpret_cast<int *>(&v157[v45]);
    int v165;
    v165 = threadIdx.x;
    assert("Tensor range check" && 0 <= v165 && v165 < 256);
    v161[v165] = v151;
    __syncthreads();
    bool v166;
    v166 = 0 <= v165;
    bool v167;
    v167 = v166 == false;
    if (v167){
        assert("The index needs to be zero or positive." && v166);
    } else {
    }
    int v169;
    v169 = v165 % 4;
    int v170;
    v170 = v165 / 4;
    bool v171;
    v171 = v170 < 64;
    bool v172;
    v172 = v171 == false;
    if (v172){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v171);
    } else {
    }
    assert("Tensor range check" && 0 <= v170 && v170 < 64);
    int v174;
    v174 = 0;
    while (while_method_1(v174)){
        bool v176;
        v176 = 0 <= v170;
        bool v177;
        v177 = v176 && v171;
        bool v178;
        v178 = v177 == false;
        if (v178){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v177);
        } else {
        }
        bool v180;
        v180 = 0 <= v174;
        bool v182;
        if (v180){
            bool v181;
            v181 = v174 < 4;
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
        int v185;
        v185 = v174 * 64;
        int v186;
        v186 = v185 + v170;
        assert("Tensor range check" && 0 <= v174 && v174 < 4);
        int v187;
        v187 = 64 * v174;
        int v188;
        v188 = v187 + v170;
        float * v189;
        v189 = v161[v188];
        int v190;
        v190 = blockIdx.x;
        int v191;
        v191 = v190 * 256;
        int v192;
        v192 = v191 + v186;
        assert("Tensor range check" && 0 <= v169 && v169 < 4);
        int v193;
        v193 = 4 * v169;
        float v194[4];
        int v195[4];
        int v196;
        v196 = 0;
        while (while_method_3(v196)){
            assert("Tensor range check" && 0 <= v196 && v196 < 1);
            int v198;
            v198 = 4 * v196;
            assert("Tensor range check" && 0 <= v196 && v196 < 1);
            int v199;
            v199 = 16 * v196;
            int v200;
            v200 = v199 + v193;
            int4* v201;
            v201 = reinterpret_cast<int4*>(v189 + v200);
            int4* v202;
            v202 = reinterpret_cast<int4*>(v194 + v198);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v201) % 16 == 0 && reinterpret_cast<unsigned long long>(v202) % 16 == 0);
            *v202 = *v201;
            v196 += 1 ;
        }
        int v203;
        v203 = 0;
        while (while_method_3(v203)){
            int v205;
            v205 = 0;
            while (while_method_1(v205)){
                bool v207;
                v207 = 0 <= v205;
                bool v209;
                if (v207){
                    bool v208;
                    v208 = v205 < 4;
                    v209 = v208;
                } else {
                    v209 = false;
                }
                bool v210;
                v210 = v209 == false;
                if (v210){
                    assert("The indices should be inside the range of the dimension." && v209);
                } else {
                }
                bool v212;
                v212 = 0 <= v169;
                bool v214;
                if (v212){
                    bool v213;
                    v213 = v169 < 4;
                    v214 = v213;
                } else {
                    v214 = false;
                }
                bool v215;
                v215 = v214 == false;
                if (v215){
                    assert("The indices should be inside the range of the dimension." && v214);
                } else {
                }
                int v217;
                v217 = v169 * 4;
                int v218;
                v218 = v205 + v217;
                bool v219;
                v219 = 0 <= v203;
                bool v221;
                if (v219){
                    bool v220;
                    v220 = v203 < 1;
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
                int v224;
                v224 = v203 * 16;
                int v225;
                v225 = v218 + v224;
                assert("Tensor range check" && 0 <= v203 && v203 < 1);
                assert("Tensor range check" && 0 <= v205 && v205 < 4);
                int v226;
                v226 = 4 * v203;
                int v227;
                v227 = v226 + v205;
                v195[v227] = v225;
                v205 += 1 ;
            }
            v203 += 1 ;
        }
        int v228;
        v228 = 0;
        while (while_method_3(v228)){
            assert("Tensor range check" && 0 <= v228 && v228 < 1);
            assert("Tensor range check" && 0 <= v228 && v228 < 1);
            v228 += 1 ;
        }
        assert("Tensor range check" && 0 <= v186 && v186 < 256);
        v163[v186] = v192;
        v174 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v165 && v165 < 256);
    int v230;
    v230 = v163[v165];
    __syncthreads();
    int v231;
    v231 = threadIdx.x;
    int v232;
    v232 = blockIdx.x;
    int v233;
    v233 = v232 * 256;
    int v234;
    v234 = v231 + v233;
    assert("Tensor range check" && 0 <= v234 && v234 < 6144);
    v4[v234] = v230;
    float * v235;
    v235 = v1+v12;
    float * v237;
    v237 = v6+v32;
    unsigned long long v239;
    v239 = v45 + v41;
    bool v240;
    v240 = v239 <= 98304ull;
    bool v241;
    v241 = v240 == false;
    if (v241){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v240);
    } else {
    }
    extern __shared__ unsigned char v243[];
    bool v244;
    v244 = v239 <= v239;
    bool v245;
    v245 = v244 == false;
    if (v245){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v244);
    } else {
    }
    float * * v247;
    v247 = reinterpret_cast<float * *>(&v243[0ull]);
    float * * v249;
    v249 = reinterpret_cast<float * *>(&v243[v45]);
    int v251;
    v251 = threadIdx.x;
    assert("Tensor range check" && 0 <= v251 && v251 < 256);
    v247[v251] = v235;
    v249[v251] = v237;
    __syncthreads();
    bool v252;
    v252 = 0 <= v251;
    bool v253;
    v253 = v252 == false;
    if (v253){
        assert("The index needs to be zero or positive." && v252);
    } else {
    }
    int v255;
    v255 = v251 % 4;
    int v256;
    v256 = v251 / 4;
    bool v257;
    v257 = v256 < 64;
    bool v258;
    v258 = v257 == false;
    if (v258){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v257);
    } else {
    }
    assert("Tensor range check" && 0 <= v256 && v256 < 64);
    int v260;
    v260 = 0;
    while (while_method_1(v260)){
        bool v262;
        v262 = 0 <= v256;
        bool v263;
        v263 = v262 && v257;
        bool v264;
        v264 = v263 == false;
        if (v264){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v263);
        } else {
        }
        bool v266;
        v266 = 0 <= v260;
        bool v268;
        if (v266){
            bool v267;
            v267 = v260 < 4;
            v268 = v267;
        } else {
            v268 = false;
        }
        bool v269;
        v269 = v268 == false;
        if (v269){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v268);
        } else {
        }
        int v271;
        v271 = v260 * 64;
        int v272;
        v272 = v271 + v256;
        assert("Tensor range check" && 0 <= v260 && v260 < 4);
        int v273;
        v273 = 64 * v260;
        int v274;
        v274 = v273 + v256;
        float * v275;
        v275 = v247[v274];
        float * v276;
        v276 = v249[v274];
        int v277;
        v277 = blockIdx.x;
        int v278;
        v278 = v277 * 256;
        int v279;
        v279 = v278 + v272;
        assert("Tensor range check" && 0 <= v255 && v255 < 4);
        int v280;
        v280 = 4 * v255;
        float v281[4];
        int v282[4];
        int v283;
        v283 = 0;
        while (while_method_3(v283)){
            assert("Tensor range check" && 0 <= v283 && v283 < 1);
            int v285;
            v285 = 4 * v283;
            assert("Tensor range check" && 0 <= v283 && v283 < 1);
            int v286;
            v286 = 16 * v283;
            int v287;
            v287 = v286 + v280;
            int4* v288;
            v288 = reinterpret_cast<int4*>(v275 + v287);
            int4* v289;
            v289 = reinterpret_cast<int4*>(v281 + v285);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v288) % 16 == 0 && reinterpret_cast<unsigned long long>(v289) % 16 == 0);
            *v289 = *v288;
            v283 += 1 ;
        }
        int v290;
        v290 = 0;
        while (while_method_3(v290)){
            int v292;
            v292 = 0;
            while (while_method_1(v292)){
                bool v294;
                v294 = 0 <= v292;
                bool v296;
                if (v294){
                    bool v295;
                    v295 = v292 < 4;
                    v296 = v295;
                } else {
                    v296 = false;
                }
                bool v297;
                v297 = v296 == false;
                if (v297){
                    assert("The indices should be inside the range of the dimension." && v296);
                } else {
                }
                bool v299;
                v299 = 0 <= v255;
                bool v301;
                if (v299){
                    bool v300;
                    v300 = v255 < 4;
                    v301 = v300;
                } else {
                    v301 = false;
                }
                bool v302;
                v302 = v301 == false;
                if (v302){
                    assert("The indices should be inside the range of the dimension." && v301);
                } else {
                }
                int v304;
                v304 = v255 * 4;
                int v305;
                v305 = v292 + v304;
                bool v306;
                v306 = 0 <= v290;
                bool v308;
                if (v306){
                    bool v307;
                    v307 = v290 < 1;
                    v308 = v307;
                } else {
                    v308 = false;
                }
                bool v309;
                v309 = v308 == false;
                if (v309){
                    assert("The indices should be inside the range of the dimension." && v308);
                } else {
                }
                int v311;
                v311 = v290 * 16;
                int v312;
                v312 = v305 + v311;
                assert("Tensor range check" && 0 <= v290 && v290 < 1);
                assert("Tensor range check" && 0 <= v292 && v292 < 4);
                int v313;
                v313 = 4 * v290;
                int v314;
                v314 = v313 + v292;
                v282[v314] = v312;
                v292 += 1 ;
            }
            v290 += 1 ;
        }
        int v315;
        v315 = 0;
        while (while_method_3(v315)){
            assert("Tensor range check" && 0 <= v315 && v315 < 1);
            int v317;
            v317 = 16 * v315;
            int v318;
            v318 = v317 + v280;
            assert("Tensor range check" && 0 <= v315 && v315 < 1);
            int v319;
            v319 = 4 * v315;
            int4* v320;
            v320 = reinterpret_cast<int4*>(v281 + v319);
            int4* v321;
            v321 = reinterpret_cast<int4*>(v276 + v318);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v320) % 16 == 0 && reinterpret_cast<unsigned long long>(v321) % 16 == 0);
            *v321 = *v320;
            v315 += 1 ;
        }
        assert("Tensor range check" && 0 <= v272 && v272 < 256);
        v260 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v251 && v251 < 256);
    __syncthreads();
    float * v322;
    v322 = v1+v12;
    float * v324;
    v324 = v7+v22;
    if (v241){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v240);
    } else {
    }
    extern __shared__ unsigned char v327[];
    if (v245){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v244);
    } else {
    }
    float * * v329;
    v329 = reinterpret_cast<float * *>(&v327[0ull]);
    float * * v331;
    v331 = reinterpret_cast<float * *>(&v327[v45]);
    int v333;
    v333 = threadIdx.x;
    assert("Tensor range check" && 0 <= v333 && v333 < 256);
    v329[v333] = v322;
    v331[v333] = v324;
    __syncthreads();
    bool v334;
    v334 = 0 <= v333;
    bool v335;
    v335 = v334 == false;
    if (v335){
        assert("The index needs to be zero or positive." && v334);
    } else {
    }
    int v337;
    v337 = v333 % 4;
    int v338;
    v338 = v333 / 4;
    bool v339;
    v339 = v338 < 64;
    bool v340;
    v340 = v339 == false;
    if (v340){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v339);
    } else {
    }
    assert("Tensor range check" && 0 <= v338 && v338 < 64);
    int v342;
    v342 = 0;
    while (while_method_1(v342)){
        bool v344;
        v344 = 0 <= v338;
        bool v345;
        v345 = v344 && v339;
        bool v346;
        v346 = v345 == false;
        if (v346){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v345);
        } else {
        }
        bool v348;
        v348 = 0 <= v342;
        bool v350;
        if (v348){
            bool v349;
            v349 = v342 < 4;
            v350 = v349;
        } else {
            v350 = false;
        }
        bool v351;
        v351 = v350 == false;
        if (v351){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v350);
        } else {
        }
        int v353;
        v353 = v342 * 64;
        int v354;
        v354 = v353 + v338;
        assert("Tensor range check" && 0 <= v342 && v342 < 4);
        int v355;
        v355 = 64 * v342;
        int v356;
        v356 = v355 + v338;
        float * v357;
        v357 = v329[v356];
        float * v358;
        v358 = v331[v356];
        int v359;
        v359 = blockIdx.x;
        int v360;
        v360 = v359 * 256;
        int v361;
        v361 = v360 + v354;
        assert("Tensor range check" && 0 <= v337 && v337 < 4);
        int v362;
        v362 = 4 * v337;
        float v363[4];
        int v364[4];
        int v365;
        v365 = 0;
        while (while_method_3(v365)){
            assert("Tensor range check" && 0 <= v365 && v365 < 1);
            int v367;
            v367 = 4 * v365;
            assert("Tensor range check" && 0 <= v365 && v365 < 1);
            int v368;
            v368 = 16 * v365;
            int v369;
            v369 = v368 + v362;
            int4* v370;
            v370 = reinterpret_cast<int4*>(v357 + v369);
            int4* v371;
            v371 = reinterpret_cast<int4*>(v363 + v367);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v370) % 16 == 0 && reinterpret_cast<unsigned long long>(v371) % 16 == 0);
            *v371 = *v370;
            v365 += 1 ;
        }
        int v372;
        v372 = 0;
        while (while_method_3(v372)){
            int v374;
            v374 = 0;
            while (while_method_1(v374)){
                bool v376;
                v376 = 0 <= v374;
                bool v378;
                if (v376){
                    bool v377;
                    v377 = v374 < 4;
                    v378 = v377;
                } else {
                    v378 = false;
                }
                bool v379;
                v379 = v378 == false;
                if (v379){
                    assert("The indices should be inside the range of the dimension." && v378);
                } else {
                }
                bool v381;
                v381 = 0 <= v337;
                bool v383;
                if (v381){
                    bool v382;
                    v382 = v337 < 4;
                    v383 = v382;
                } else {
                    v383 = false;
                }
                bool v384;
                v384 = v383 == false;
                if (v384){
                    assert("The indices should be inside the range of the dimension." && v383);
                } else {
                }
                int v386;
                v386 = v337 * 4;
                int v387;
                v387 = v374 + v386;
                bool v388;
                v388 = 0 <= v372;
                bool v390;
                if (v388){
                    bool v389;
                    v389 = v372 < 1;
                    v390 = v389;
                } else {
                    v390 = false;
                }
                bool v391;
                v391 = v390 == false;
                if (v391){
                    assert("The indices should be inside the range of the dimension." && v390);
                } else {
                }
                int v393;
                v393 = v372 * 16;
                int v394;
                v394 = v387 + v393;
                assert("Tensor range check" && 0 <= v372 && v372 < 1);
                assert("Tensor range check" && 0 <= v374 && v374 < 4);
                int v395;
                v395 = 4 * v372;
                int v396;
                v396 = v395 + v374;
                v364[v396] = v394;
                v374 += 1 ;
            }
            v372 += 1 ;
        }
        bool v397[4];
        int v398;
        v398 = 0;
        while (while_method_3(v398)){
            int v400;
            v400 = 0;
            while (while_method_1(v400)){
                assert("Tensor range check" && 0 <= v398 && v398 < 1);
                assert("Tensor range check" && 0 <= v400 && v400 < 4);
                int v402;
                v402 = 4 * v398;
                int v403;
                v403 = v402 + v400;
                float v404;
                v404 = v363[v403];
                int v405;
                v405 = v364[v403];
                bool v406;
                v406 = v405 < 3;
                assert("Tensor range check" && 0 <= v398 && v398 < 1);
                assert("Tensor range check" && 0 <= v400 && v400 < 4);
                v397[v403] = v406;
                v400 += 1 ;
            }
            v398 += 1 ;
        }
        float v407[4];
        int v408;
        v408 = 0;
        while (while_method_3(v408)){
            int v410;
            v410 = 0;
            while (while_method_1(v410)){
                assert("Tensor range check" && 0 <= v408 && v408 < 1);
                assert("Tensor range check" && 0 <= v410 && v410 < 4);
                int v412;
                v412 = 4 * v408;
                int v413;
                v413 = v412 + v410;
                float v414;
                v414 = v363[v413];
                bool v415;
                v415 = v397[v413];
                float v418;
                if (v415){
                    bool v416;
                    v416 = 0.0f >= v414;
                    if (v416){
                        v418 = 0.0f;
                    } else {
                        v418 = v414;
                    }
                } else {
                    v418 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v408 && v408 < 1);
                assert("Tensor range check" && 0 <= v410 && v410 < 4);
                v407[v413] = v418;
                v410 += 1 ;
            }
            v408 += 1 ;
        }
        float v419;
        v419 = 0.0f;
        int v420;
        v420 = 0;
        while (while_method_3(v420)){
            int v422;
            v422 = 0;
            while (while_method_1(v422)){
                assert("Tensor range check" && 0 <= v420 && v420 < 1);
                assert("Tensor range check" && 0 <= v422 && v422 < 4);
                int v424;
                v424 = 4 * v420;
                int v425;
                v425 = v424 + v422;
                float v426;
                v426 = v407[v425];
                float v427;
                v427 = v419 + v426;
                v419 = v427;
                v422 += 1 ;
            }
            v420 += 1 ;
        }
        auto v428 = cooperative_groups::coalesced_threads();
        int v429;
        v429 = threadIdx.x;
        int v430;
        v430 = v429 / 4;
        auto v431 = cooperative_groups::labeled_partition(v428,v430);
        Closure0 v432{};
        float v433;
        v433 = cooperative_groups::reduce(v431, v419, v432);
        int v434[4];
        int v435;
        v435 = 0;
        while (while_method_3(v435)){
            int v437;
            v437 = 0;
            while (while_method_1(v437)){
                assert("Tensor range check" && 0 <= v435 && v435 < 1);
                assert("Tensor range check" && 0 <= v437 && v437 < 4);
                int v439;
                v439 = 4 * v435;
                int v440;
                v440 = v439 + v437;
                bool v441;
                v441 = v397[v440];
                int v442;
                if (v441){
                    v442 = 1;
                } else {
                    v442 = 0;
                }
                assert("Tensor range check" && 0 <= v435 && v435 < 1);
                assert("Tensor range check" && 0 <= v437 && v437 < 4);
                v434[v440] = v442;
                v437 += 1 ;
            }
            v435 += 1 ;
        }
        int v443;
        v443 = 0;
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
                int v450;
                v450 = v434[v449];
                int v451;
                v451 = v443 + v450;
                v443 = v451;
                v446 += 1 ;
            }
            v444 += 1 ;
        }
        auto v452 = cooperative_groups::coalesced_threads();
        int v453;
        v453 = threadIdx.x;
        int v454;
        v454 = v453 / 4;
        auto v455 = cooperative_groups::labeled_partition(v452,v454);
        Closure4 v456{};
        int v457;
        v457 = cooperative_groups::reduce(v455, v443, v456);
        float v458;
        v458 = (float)v457;
        float v459;
        v459 = 1.0f / v458;
        float v460[4];
        int v461;
        v461 = 0;
        while (while_method_3(v461)){
            int v463;
            v463 = 0;
            while (while_method_1(v463)){
                assert("Tensor range check" && 0 <= v461 && v461 < 1);
                assert("Tensor range check" && 0 <= v463 && v463 < 4);
                int v465;
                v465 = 4 * v461;
                int v466;
                v466 = v465 + v463;
                float v467;
                v467 = v407[v466];
                bool v468;
                v468 = v397[v466];
                bool v469;
                v469 = v468 == false;
                float v474;
                if (v469){
                    v474 = 0.0f;
                } else {
                    bool v470;
                    v470 = v433 == 0.0f;
                    bool v471;
                    v471 = v470 != true;
                    if (v471){
                        float v472;
                        v472 = v467 / v433;
                        v474 = v472;
                    } else {
                        v474 = v459;
                    }
                }
                assert("Tensor range check" && 0 <= v461 && v461 < 1);
                assert("Tensor range check" && 0 <= v463 && v463 < 4);
                v460[v466] = v474;
                v463 += 1 ;
            }
            v461 += 1 ;
        }
        int v475;
        v475 = 0;
        while (while_method_3(v475)){
            assert("Tensor range check" && 0 <= v475 && v475 < 1);
            int v477;
            v477 = 16 * v475;
            int v478;
            v478 = v477 + v362;
            assert("Tensor range check" && 0 <= v475 && v475 < 1);
            int v479;
            v479 = 4 * v475;
            int4* v480;
            v480 = reinterpret_cast<int4*>(v460 + v479);
            int4* v481;
            v481 = reinterpret_cast<int4*>(v358 + v478);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v480) % 16 == 0 && reinterpret_cast<unsigned long long>(v481) % 16 == 0);
            *v481 = *v480;
            v475 += 1 ;
        }
        assert("Tensor range check" && 0 <= v354 && v354 < 256);
        v342 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v333 && v333 < 256);
    __syncthreads();
    int v482;
    v482 = threadIdx.x;
    int v483;
    v483 = blockIdx.x;
    int v484;
    v484 = v483 * 256;
    int v485;
    v485 = v482 + v484;
    unsigned long long v486;
    v486 = (unsigned long long)v485;
    curandStatePhilox4_32_10_t v487;
    curand_init(12344321ull,v486,0ull,&v487);
    float * v488;
    v488 = v1+v12;
    if (v155){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v154);
    } else {
    }
    extern __shared__ unsigned char v491[];
    if (v159){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v158);
    } else {
    }
    float * * v493;
    v493 = reinterpret_cast<float * *>(&v491[0ull]);
    int * v495;
    v495 = reinterpret_cast<int *>(&v491[v45]);
    int v497;
    v497 = threadIdx.x;
    assert("Tensor range check" && 0 <= v497 && v497 < 256);
    v493[v497] = v488;
    __syncthreads();
    bool v498;
    v498 = 0 <= v497;
    bool v499;
    v499 = v498 == false;
    if (v499){
        assert("The index needs to be zero or positive." && v498);
    } else {
    }
    int v501;
    v501 = v497 % 4;
    int v502;
    v502 = v497 / 4;
    bool v503;
    v503 = v502 < 64;
    bool v504;
    v504 = v503 == false;
    if (v504){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v503);
    } else {
    }
    assert("Tensor range check" && 0 <= v502 && v502 < 64);
    int v506;
    v506 = 0;
    while (while_method_1(v506)){
        bool v508;
        v508 = 0 <= v502;
        bool v509;
        v509 = v508 && v503;
        bool v510;
        v510 = v509 == false;
        if (v510){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v509);
        } else {
        }
        bool v512;
        v512 = 0 <= v506;
        bool v514;
        if (v512){
            bool v513;
            v513 = v506 < 4;
            v514 = v513;
        } else {
            v514 = false;
        }
        bool v515;
        v515 = v514 == false;
        if (v515){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v514);
        } else {
        }
        int v517;
        v517 = v506 * 64;
        int v518;
        v518 = v517 + v502;
        assert("Tensor range check" && 0 <= v506 && v506 < 4);
        int v519;
        v519 = 64 * v506;
        int v520;
        v520 = v519 + v502;
        float * v521;
        v521 = v493[v520];
        int v522;
        v522 = blockIdx.x;
        int v523;
        v523 = v522 * 256;
        int v524;
        v524 = v523 + v518;
        assert("Tensor range check" && 0 <= v501 && v501 < 4);
        int v525;
        v525 = 4 * v501;
        float v526[4];
        int v527[4];
        int v528;
        v528 = 0;
        while (while_method_3(v528)){
            assert("Tensor range check" && 0 <= v528 && v528 < 1);
            int v530;
            v530 = 4 * v528;
            assert("Tensor range check" && 0 <= v528 && v528 < 1);
            int v531;
            v531 = 16 * v528;
            int v532;
            v532 = v531 + v525;
            int4* v533;
            v533 = reinterpret_cast<int4*>(v521 + v532);
            int4* v534;
            v534 = reinterpret_cast<int4*>(v526 + v530);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v533) % 16 == 0 && reinterpret_cast<unsigned long long>(v534) % 16 == 0);
            *v534 = *v533;
            v528 += 1 ;
        }
        int v535;
        v535 = 0;
        while (while_method_3(v535)){
            int v537;
            v537 = 0;
            while (while_method_1(v537)){
                bool v539;
                v539 = 0 <= v537;
                bool v541;
                if (v539){
                    bool v540;
                    v540 = v537 < 4;
                    v541 = v540;
                } else {
                    v541 = false;
                }
                bool v542;
                v542 = v541 == false;
                if (v542){
                    assert("The indices should be inside the range of the dimension." && v541);
                } else {
                }
                bool v544;
                v544 = 0 <= v501;
                bool v546;
                if (v544){
                    bool v545;
                    v545 = v501 < 4;
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
                v549 = v501 * 4;
                int v550;
                v550 = v537 + v549;
                bool v551;
                v551 = 0 <= v535;
                bool v553;
                if (v551){
                    bool v552;
                    v552 = v535 < 1;
                    v553 = v552;
                } else {
                    v553 = false;
                }
                bool v554;
                v554 = v553 == false;
                if (v554){
                    assert("The indices should be inside the range of the dimension." && v553);
                } else {
                }
                int v556;
                v556 = v535 * 16;
                int v557;
                v557 = v550 + v556;
                assert("Tensor range check" && 0 <= v535 && v535 < 1);
                assert("Tensor range check" && 0 <= v537 && v537 < 4);
                int v558;
                v558 = 4 * v535;
                int v559;
                v559 = v558 + v537;
                v527[v559] = v557;
                v537 += 1 ;
            }
            v535 += 1 ;
        }
        bool v560[4];
        int v561;
        v561 = 0;
        while (while_method_3(v561)){
            int v563;
            v563 = 0;
            while (while_method_1(v563)){
                assert("Tensor range check" && 0 <= v561 && v561 < 1);
                assert("Tensor range check" && 0 <= v563 && v563 < 4);
                int v565;
                v565 = 4 * v561;
                int v566;
                v566 = v565 + v563;
                float v567;
                v567 = v526[v566];
                int v568;
                v568 = v527[v566];
                bool v569;
                v569 = v568 < 3;
                assert("Tensor range check" && 0 <= v561 && v561 < 1);
                assert("Tensor range check" && 0 <= v563 && v563 < 4);
                v560[v566] = v569;
                v563 += 1 ;
            }
            v561 += 1 ;
        }
        float v570[4];
        int v571;
        v571 = 0;
        while (while_method_3(v571)){
            int v573;
            v573 = 0;
            while (while_method_1(v573)){
                assert("Tensor range check" && 0 <= v571 && v571 < 1);
                assert("Tensor range check" && 0 <= v573 && v573 < 4);
                int v575;
                v575 = 4 * v571;
                int v576;
                v576 = v575 + v573;
                float v577;
                v577 = v526[v576];
                bool v578;
                v578 = v560[v576];
                float v579;
                if (v578){
                    v579 = v577;
                } else {
                    v579 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v571 && v571 < 1);
                assert("Tensor range check" && 0 <= v573 && v573 < 4);
                v570[v576] = v579;
                v573 += 1 ;
            }
            v571 += 1 ;
        }
        float v580;
        v580 = 0.0f;
        int v581;
        v581 = 0;
        while (while_method_3(v581)){
            int v583;
            v583 = 0;
            while (while_method_1(v583)){
                assert("Tensor range check" && 0 <= v581 && v581 < 1);
                assert("Tensor range check" && 0 <= v583 && v583 < 4);
                int v585;
                v585 = 4 * v581;
                int v586;
                v586 = v585 + v583;
                float v587;
                v587 = v570[v586];
                float v588;
                v588 = v580 + v587;
                v580 = v588;
                v583 += 1 ;
            }
            v581 += 1 ;
        }
        auto v589 = cooperative_groups::coalesced_threads();
        int v590;
        v590 = threadIdx.x;
        int v591;
        v591 = v590 / 4;
        auto v592 = cooperative_groups::labeled_partition(v589,v591);
        Closure0 v593{};
        float v594;
        v594 = cooperative_groups::reduce(v592, v580, v593);
        int v595[4];
        int v596;
        v596 = 0;
        while (while_method_3(v596)){
            int v598;
            v598 = 0;
            while (while_method_1(v598)){
                assert("Tensor range check" && 0 <= v596 && v596 < 1);
                assert("Tensor range check" && 0 <= v598 && v598 < 4);
                int v600;
                v600 = 4 * v596;
                int v601;
                v601 = v600 + v598;
                bool v602;
                v602 = v560[v601];
                int v603;
                if (v602){
                    v603 = 1;
                } else {
                    v603 = 0;
                }
                assert("Tensor range check" && 0 <= v596 && v596 < 1);
                assert("Tensor range check" && 0 <= v598 && v598 < 4);
                v595[v601] = v603;
                v598 += 1 ;
            }
            v596 += 1 ;
        }
        int v604;
        v604 = 0;
        int v605;
        v605 = 0;
        while (while_method_3(v605)){
            int v607;
            v607 = 0;
            while (while_method_1(v607)){
                assert("Tensor range check" && 0 <= v605 && v605 < 1);
                assert("Tensor range check" && 0 <= v607 && v607 < 4);
                int v609;
                v609 = 4 * v605;
                int v610;
                v610 = v609 + v607;
                int v611;
                v611 = v595[v610];
                int v612;
                v612 = v604 + v611;
                v604 = v612;
                v607 += 1 ;
            }
            v605 += 1 ;
        }
        auto v613 = cooperative_groups::coalesced_threads();
        int v614;
        v614 = threadIdx.x;
        int v615;
        v615 = v614 / 4;
        auto v616 = cooperative_groups::labeled_partition(v613,v615);
        Closure4 v617{};
        int v618;
        v618 = cooperative_groups::reduce(v616, v604, v617);
        float v619;
        v619 = (float)v618;
        float v620;
        v620 = v594 / v619;
        float v621[4];
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
                v628 = v526[v627];
                bool v629;
                v629 = v560[v627];
                float v630;
                if (v629){
                    v630 = v628;
                } else {
                    v630 = -1.0f / 0.0f;
                }
                float v631;
                v631 = v630 - v620;
                float v632;
                v632 = exp(v631);
                bool v633;
                v633 = v632 < 1.0f / 0.0f;
                bool v634;
                v634 = v633 == false;
                if (v634){
                    assert("The softmax values must not grow too large." && v633);
                } else {
                }
                bool v636;
                v636 = isnan(v632);
                bool v637;
                v637 = v636 == false;
                bool v638;
                v638 = v637 == false;
                if (v638){
                    assert("The softmax values must not be nans." && v637);
                } else {
                }
                assert("Tensor range check" && 0 <= v622 && v622 < 1);
                assert("Tensor range check" && 0 <= v624 && v624 < 4);
                v621[v627] = v632;
                v624 += 1 ;
            }
            v622 += 1 ;
        }
        float v640;
        v640 = 0.0f;
        int v641;
        v641 = 0;
        while (while_method_3(v641)){
            int v643;
            v643 = 0;
            while (while_method_1(v643)){
                assert("Tensor range check" && 0 <= v641 && v641 < 1);
                assert("Tensor range check" && 0 <= v643 && v643 < 4);
                int v645;
                v645 = 4 * v641;
                int v646;
                v646 = v645 + v643;
                float v647;
                v647 = v621[v646];
                float v648;
                v648 = v640 + v647;
                v640 = v648;
                v643 += 1 ;
            }
            v641 += 1 ;
        }
        auto v649 = cooperative_groups::coalesced_threads();
        int v650;
        v650 = threadIdx.x;
        int v651;
        v651 = v650 / 4;
        auto v652 = cooperative_groups::labeled_partition(v649,v651);
        float v653;
        v653 = cooperative_groups::reduce(v652, v640, v593);
        float v654[4];
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
                v661 = v621[v660];
                float v662;
                v662 = v661 / v653;
                assert("Tensor range check" && 0 <= v655 && v655 < 1);
                assert("Tensor range check" && 0 <= v657 && v657 < 4);
                v654[v660] = v662;
                v657 += 1 ;
            }
            v655 += 1 ;
        }
        float v663[4];
        float v664;
        v664 = 0.0f;
        int v665;
        v665 = 0;
        while (while_method_3(v665)){
            assert("Tensor range check" && 0 <= v665 && v665 < 1);
            int v667;
            v667 = 4 * v665;
            assert("Tensor range check" && 0 <= v665 && v665 < 1);
            float v668;
            v668 = 0.0f;
            int v669;
            v669 = 0;
            while (while_method_1(v669)){
                assert("Tensor range check" && 0 <= v669 && v669 < 4);
                int v671;
                v671 = v669 + v667;
                float v672;
                v672 = v654[v671];
                float v673;
                v673 = v668 + v672;
                v668 = v673;
                v669 += 1 ;
            }
            auto v674 = cooperative_groups::coalesced_threads();
            int v675;
            v675 = threadIdx.x;
            int v676;
            v676 = v675 / 4;
            auto v677 = cooperative_groups::labeled_partition(v674,v676);
            Closure2 v678{};
            float v679;
            v679 = cooperative_groups::inclusive_scan(v677, v668, v678);
            float v680;
            v680 = v677.shfl_up(v679,1);
            bool v681;
            v681 = v677.thread_rank() == 0;
            float v682;
            if (v681){
                v682 = 0.0f;
            } else {
                v682 = v680;
            }
            float v683;
            v683 = v677.shfl(v679,v677.num_threads()-1);
            float v684;
            v684 = v664 + v682;
            float v685;
            v685 = v684;
            int v686;
            v686 = 0;
            while (while_method_1(v686)){
                assert("Tensor range check" && 0 <= v686 && v686 < 4);
                int v688;
                v688 = v686 + v667;
                float v689;
                v689 = v654[v688];
                float v690;
                v690 = v685 + v689;
                assert("Tensor range check" && 0 <= v686 && v686 < 4);
                v663[v688] = v690;
                v685 = v690;
                v686 += 1 ;
            }
            float v691;
            v691 = v664 + v683;
            v664 = v691;
            v665 += 1 ;
        }
        float v692[4];
        bool v693[4];
        int v694;
        v694 = 0;
        while (while_method_3(v694)){
            int v696;
            v696 = 0;
            while (while_method_1(v696)){
                assert("Tensor range check" && 0 <= v694 && v694 < 1);
                assert("Tensor range check" && 0 <= v696 && v696 < 4);
                int v698;
                v698 = 4 * v694;
                int v699;
                v699 = v698 + v696;
                float v700;
                v700 = v663[v699];
                float v701;
                v701 = v654[v699];
                bool v702;
                v702 = v701 > 0.0f;
                assert("Tensor range check" && 0 <= v694 && v694 < 1);
                assert("Tensor range check" && 0 <= v696 && v696 < 4);
                v692[v699] = v700;
                v693[v699] = v702;
                v696 += 1 ;
            }
            v694 += 1 ;
        }
        float v703; bool v704;
        Tuple2 tmp30 = Tuple2{-1.0f / 0.0f, false};
        v703 = tmp30.v0; v704 = tmp30.v1;
        int v705;
        v705 = 0;
        while (while_method_3(v705)){
            int v707;
            v707 = 0;
            while (while_method_1(v707)){
                assert("Tensor range check" && 0 <= v705 && v705 < 1);
                assert("Tensor range check" && 0 <= v707 && v707 < 4);
                int v709;
                v709 = 4 * v705;
                int v710;
                v710 = v709 + v707;
                float v711;
                v711 = v692[v710];
                bool v712;
                v712 = v693[v710];
                float v719; bool v720;
                if (v704){
                    if (v712){
                        bool v713;
                        v713 = v703 >= v711;
                        float v714;
                        if (v713){
                            v714 = v703;
                        } else {
                            v714 = v711;
                        }
                        v719 = v714; v720 = true;
                    } else {
                        v719 = v703; v720 = v704;
                    }
                } else {
                    if (v712){
                        v719 = v711; v720 = v712;
                    } else {
                        v719 = v703; v720 = v704;
                    }
                }
                v703 = v719;
                v704 = v720;
                v707 += 1 ;
            }
            v705 += 1 ;
        }
        auto v721 = cooperative_groups::coalesced_threads();
        int v722;
        v722 = threadIdx.x;
        int v723;
        v723 = v722 / 4;
        auto v724 = cooperative_groups::labeled_partition(v721,v723);
        Closure5 v725{};
        float v726; bool v727;
        Tuple2 tmp31 = cooperative_groups::reduce(v724, Tuple2{v703, v704}, v725);
        v726 = tmp31.v0; v727 = tmp31.v1;
        bool v728;
        v728 = v727 == false;
        if (v728){
            assert("The local reduce must be true." && v727);
        } else {
        }
        float v730[4];
        int v731[4];
        int v732;
        v732 = 0;
        while (while_method_3(v732)){
            int v734;
            v734 = 0;
            while (while_method_1(v734)){
                assert("Tensor range check" && 0 <= v732 && v732 < 1);
                assert("Tensor range check" && 0 <= v734 && v734 < 4);
                int v736;
                v736 = 4 * v732;
                int v737;
                v737 = v736 + v734;
                int v738;
                v738 = v527[v737];
                float v739;
                v739 = curand_uniform(&v487);
                assert("Tensor range check" && 0 <= v732 && v732 < 1);
                assert("Tensor range check" && 0 <= v734 && v734 < 4);
                v730[v737] = v739;
                v731[v737] = v738;
                v734 += 1 ;
            }
            v732 += 1 ;
        }
        float v740; int v741;
        Tuple1 tmp32 = Tuple1{0.0f, 2147483647};
        v740 = tmp32.v0; v741 = tmp32.v1;
        int v742;
        v742 = 0;
        while (while_method_3(v742)){
            int v744;
            v744 = 0;
            while (while_method_1(v744)){
                assert("Tensor range check" && 0 <= v742 && v742 < 1);
                assert("Tensor range check" && 0 <= v744 && v744 < 4);
                int v746;
                v746 = 4 * v742;
                int v747;
                v747 = v746 + v744;
                float v748;
                v748 = v730[v747];
                int v749;
                v749 = v731[v747];
                bool v750;
                v750 = v741 < v749;
                float v751; int v752;
                if (v750){
                    v751 = v740; v752 = v741;
                } else {
                    v751 = v748; v752 = v749;
                }
                v740 = v751;
                v741 = v752;
                v744 += 1 ;
            }
            v742 += 1 ;
        }
        auto v753 = cooperative_groups::coalesced_threads();
        int v754;
        v754 = threadIdx.x;
        int v755;
        v755 = v754 / 4;
        auto v756 = cooperative_groups::labeled_partition(v753,v755);
        Closure6 v757{};
        float v758; int v759;
        Tuple1 tmp33 = cooperative_groups::reduce(v756, Tuple1{v740, v741}, v757);
        v758 = tmp33.v0; v759 = tmp33.v1;
        float v760;
        v760 = v726 * v758;
        int v761[4];
        bool v762[4];
        int v763;
        v763 = 0;
        while (while_method_3(v763)){
            int v765;
            v765 = 0;
            while (while_method_1(v765)){
                assert("Tensor range check" && 0 <= v763 && v763 < 1);
                assert("Tensor range check" && 0 <= v765 && v765 < 4);
                int v767;
                v767 = 4 * v763;
                int v768;
                v768 = v767 + v765;
                float v769;
                v769 = v692[v768];
                bool v770;
                v770 = v693[v768];
                int v771;
                v771 = v527[v768];
                int v774; bool v775;
                if (v770){
                    float v772;
                    v772 = v769 - v760;
                    bool v773;
                    v773 = v772 >= 0.0f;
                    v774 = v771; v775 = v773;
                } else {
                    v774 = 2147483647; v775 = false;
                }
                assert("Tensor range check" && 0 <= v763 && v763 < 1);
                assert("Tensor range check" && 0 <= v765 && v765 < 4);
                v761[v768] = v774;
                v762[v768] = v775;
                v765 += 1 ;
            }
            v763 += 1 ;
        }
        int v776; bool v777;
        Tuple3 tmp34 = Tuple3{2147483647, false};
        v776 = tmp34.v0; v777 = tmp34.v1;
        int v778;
        v778 = 0;
        while (while_method_3(v778)){
            int v780;
            v780 = 0;
            while (while_method_1(v780)){
                assert("Tensor range check" && 0 <= v778 && v778 < 1);
                assert("Tensor range check" && 0 <= v780 && v780 < 4);
                int v782;
                v782 = 4 * v778;
                int v783;
                v783 = v782 + v780;
                int v784;
                v784 = v761[v783];
                bool v785;
                v785 = v762[v783];
                int v792; bool v793;
                if (v777){
                    if (v785){
                        bool v786;
                        v786 = v776 < v784;
                        int v787;
                        if (v786){
                            v787 = v776;
                        } else {
                            v787 = v784;
                        }
                        v792 = v787; v793 = true;
                    } else {
                        v792 = v776; v793 = v777;
                    }
                } else {
                    if (v785){
                        v792 = v784; v793 = v785;
                    } else {
                        v792 = v776; v793 = v777;
                    }
                }
                v776 = v792;
                v777 = v793;
                v780 += 1 ;
            }
            v778 += 1 ;
        }
        auto v794 = cooperative_groups::coalesced_threads();
        int v795;
        v795 = threadIdx.x;
        int v796;
        v796 = v795 / 4;
        auto v797 = cooperative_groups::labeled_partition(v794,v796);
        Closure7 v798{};
        int v799; bool v800;
        Tuple3 tmp35 = cooperative_groups::reduce(v797, Tuple3{v776, v777}, v798);
        v799 = tmp35.v0; v800 = tmp35.v1;
        bool v801;
        v801 = v800 == false;
        if (v801){
            assert("The local reduce must be true." && v800);
        } else {
        }
        int v803;
        v803 = 0;
        while (while_method_3(v803)){
            assert("Tensor range check" && 0 <= v803 && v803 < 1);
            assert("Tensor range check" && 0 <= v803 && v803 < 1);
            v803 += 1 ;
        }
        assert("Tensor range check" && 0 <= v518 && v518 < 256);
        v495[v518] = v799;
        v506 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v497 && v497 < 256);
    int v805;
    v805 = v495[v497];
    __syncthreads();
    int v806;
    v806 = threadIdx.x;
    int v807;
    v807 = blockIdx.x;
    int v808;
    v808 = v807 * 256;
    int v809;
    v809 = v806 + v808;
    assert("Tensor range check" && 0 <= v809 && v809 < 6144);
    v5[v809] = v805;
    return ;
}
extern "C" __global__ void entry3(float * v0, float * v1, int * v2, int * v3, int * v4, int * v5, float * v6, float * v7) {
    int v8;
    v8 = threadIdx.x;
    int v9;
    v9 = blockIdx.x;
    int v10;
    v10 = v9 * 256;
    int v11;
    v11 = v8 + v10;
    assert("Tensor range check" && 0 <= v11 && v11 < 6144);
    int v12;
    v12 = 256 * v11;
    int v13;
    v13 = threadIdx.x;
    int v14;
    v14 = blockIdx.x;
    int v15;
    v15 = v14 * 256;
    int v16;
    v16 = v13 + v15;
    assert("Tensor range check" && 0 <= v16 && v16 < 6144);
    int v17;
    v17 = 256 * v16;
    int v18;
    v18 = threadIdx.x;
    int v19;
    v19 = blockIdx.x;
    int v20;
    v20 = v19 * 256;
    int v21;
    v21 = v18 + v20;
    assert("Tensor range check" && 0 <= v21 && v21 < 6144);
    int v22;
    v22 = 256 * v21;
    int v23;
    v23 = threadIdx.x;
    int v24;
    v24 = blockIdx.x;
    int v25;
    v25 = v24 * 256;
    int v26;
    v26 = v23 + v25;
    assert("Tensor range check" && 0 <= v26 && v26 < 6144);
    int v27;
    v27 = 256 * v26;
    int v28;
    v28 = threadIdx.x;
    int v29;
    v29 = blockIdx.x;
    int v30;
    v30 = v29 * 256;
    int v31;
    v31 = v28 + v30;
    assert("Tensor range check" && 0 <= v31 && v31 < 6144);
    int v32;
    v32 = 256 * v31;
    float * v33;
    v33 = v1+v12;
    int * v35;
    v35 = v2+v27;
    int * v37;
    v37 = v3+v27;
    int v39;
    v39 = sizeof(float *);
    unsigned long long v40;
    v40 = (unsigned long long)v39;
    unsigned long long v41;
    v41 = 256ull * v40;
    unsigned long long v42;
    v42 = v41 + 16ull;
    unsigned long long v43;
    v43 = v42 - 1ull;
    unsigned long long v44;
    v44 = v43 % 16ull;
    unsigned long long v45;
    v45 = v43 - v44;
    int v46;
    v46 = sizeof(int *);
    unsigned long long v47;
    v47 = (unsigned long long)v46;
    unsigned long long v48;
    v48 = 256ull * v47;
    unsigned long long v49;
    v49 = v45 + v48;
    unsigned long long v50;
    v50 = v49 + 16ull;
    unsigned long long v51;
    v51 = v50 - 1ull;
    unsigned long long v52;
    v52 = v51 % 16ull;
    unsigned long long v53;
    v53 = v51 - v52;
    unsigned long long v54;
    v54 = v53 + v48;
    bool v55;
    v55 = v54 <= 98304ull;
    bool v56;
    v56 = v55 == false;
    if (v56){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v55);
    } else {
    }
    extern __shared__ unsigned char v58[];
    bool v59;
    v59 = v54 <= v54;
    bool v60;
    v60 = v59 == false;
    if (v60){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v59);
    } else {
    }
    float * * v62;
    v62 = reinterpret_cast<float * *>(&v58[0ull]);
    int * * v64;
    v64 = reinterpret_cast<int * *>(&v58[v45]);
    int * * v66;
    v66 = reinterpret_cast<int * *>(&v58[v53]);
    int v68;
    v68 = threadIdx.x;
    assert("Tensor range check" && 0 <= v68 && v68 < 256);
    v62[v68] = v33;
    v64[v68] = v35;
    v66[v68] = v37;
    __syncthreads();
    bool v69;
    v69 = 0 <= v68;
    bool v70;
    v70 = v69 == false;
    if (v70){
        assert("The index needs to be zero or positive." && v69);
    } else {
    }
    int v72;
    v72 = v68 % 64;
    int v73;
    v73 = v68 / 64;
    bool v74;
    v74 = v73 < 4;
    bool v75;
    v75 = v74 == false;
    if (v75){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v74);
    } else {
    }
    assert("Tensor range check" && 0 <= v73 && v73 < 4);
    int v77;
    v77 = 0;
    while (while_method_4(v77)){
        bool v79;
        v79 = 0 <= v73;
        bool v80;
        v80 = v79 && v74;
        bool v81;
        v81 = v80 == false;
        if (v81){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v80);
        } else {
        }
        bool v83;
        v83 = 0 <= v77;
        bool v85;
        if (v83){
            bool v84;
            v84 = v77 < 64;
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
        int v88;
        v88 = v77 * 4;
        int v89;
        v89 = v88 + v73;
        assert("Tensor range check" && 0 <= v77 && v77 < 64);
        int v90;
        v90 = 4 * v77;
        int v91;
        v91 = v90 + v73;
        float * v92;
        v92 = v62[v91];
        int * v93;
        v93 = v64[v91];
        int * v94;
        v94 = v66[v91];
        int v95;
        v95 = blockIdx.x;
        int v96;
        v96 = v95 * 256;
        int v97;
        v97 = v96 + v89;
        assert("Tensor range check" && 0 <= v72 && v72 < 64);
        int v98;
        v98 = 4 * v72;
        float v99[4];
        int v100[4];
        int v101;
        v101 = 0;
        while (while_method_3(v101)){
            assert("Tensor range check" && 0 <= v101 && v101 < 1);
            int v103;
            v103 = 4 * v101;
            assert("Tensor range check" && 0 <= v101 && v101 < 1);
            int v104;
            v104 = 256 * v101;
            int v105;
            v105 = v104 + v98;
            int4* v106;
            v106 = reinterpret_cast<int4*>(v92 + v105);
            int4* v107;
            v107 = reinterpret_cast<int4*>(v99 + v103);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v106) % 16 == 0 && reinterpret_cast<unsigned long long>(v107) % 16 == 0);
            *v107 = *v106;
            v101 += 1 ;
        }
        int v108;
        v108 = 0;
        while (while_method_3(v108)){
            int v110;
            v110 = 0;
            while (while_method_1(v110)){
                bool v112;
                v112 = 0 <= v110;
                bool v114;
                if (v112){
                    bool v113;
                    v113 = v110 < 4;
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
                bool v117;
                v117 = 0 <= v72;
                bool v119;
                if (v117){
                    bool v118;
                    v118 = v72 < 64;
                    v119 = v118;
                } else {
                    v119 = false;
                }
                bool v120;
                v120 = v119 == false;
                if (v120){
                    assert("The indices should be inside the range of the dimension." && v119);
                } else {
                }
                int v122;
                v122 = v72 * 4;
                int v123;
                v123 = v110 + v122;
                bool v124;
                v124 = 0 <= v108;
                bool v126;
                if (v124){
                    bool v125;
                    v125 = v108 < 1;
                    v126 = v125;
                } else {
                    v126 = false;
                }
                bool v127;
                v127 = v126 == false;
                if (v127){
                    assert("The indices should be inside the range of the dimension." && v126);
                } else {
                }
                int v129;
                v129 = v108 * 256;
                int v130;
                v130 = v123 + v129;
                assert("Tensor range check" && 0 <= v108 && v108 < 1);
                assert("Tensor range check" && 0 <= v110 && v110 < 4);
                int v131;
                v131 = 4 * v108;
                int v132;
                v132 = v131 + v110;
                v100[v132] = v130;
                v110 += 1 ;
            }
            v108 += 1 ;
        }
        int v133[4];
        int v134[4];
        int v135;
        v135 = 0;
        while (while_method_3(v135)){
            int v137;
            v137 = 0;
            while (while_method_1(v137)){
                assert("Tensor range check" && 0 <= v135 && v135 < 1);
                assert("Tensor range check" && 0 <= v137 && v137 < 4);
                int v139;
                v139 = 4 * v135;
                int v140;
                v140 = v139 + v137;
                int v141;
                v141 = v100[v140];
                assert("Tensor range check" && 0 <= v135 && v135 < 1);
                assert("Tensor range check" && 0 <= v137 && v137 < 4);
                v133[v140] = v97;
                v134[v140] = v141;
                v137 += 1 ;
            }
            v135 += 1 ;
        }
        int v142;
        v142 = 0;
        while (while_method_3(v142)){
            assert("Tensor range check" && 0 <= v142 && v142 < 1);
            int v144;
            v144 = 256 * v142;
            int v145;
            v145 = v144 + v98;
            assert("Tensor range check" && 0 <= v142 && v142 < 1);
            int v146;
            v146 = 4 * v142;
            int4* v147;
            v147 = reinterpret_cast<int4*>(v133 + v146);
            int4* v148;
            v148 = reinterpret_cast<int4*>(v93 + v145);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v147) % 16 == 0 && reinterpret_cast<unsigned long long>(v148) % 16 == 0);
            *v148 = *v147;
            int4* v149;
            v149 = reinterpret_cast<int4*>(v134 + v146);
            int4* v150;
            v150 = reinterpret_cast<int4*>(v94 + v145);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v149) % 16 == 0 && reinterpret_cast<unsigned long long>(v150) % 16 == 0);
            *v150 = *v149;
            v142 += 1 ;
        }
        assert("Tensor range check" && 0 <= v89 && v89 < 256);
        v77 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v68 && v68 < 256);
    __syncthreads();
    float * v151;
    v151 = v1+v12;
    unsigned long long v153;
    v153 = v45 + 1024ull;
    bool v154;
    v154 = v153 <= 98304ull;
    bool v155;
    v155 = v154 == false;
    if (v155){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v154);
    } else {
    }
    extern __shared__ unsigned char v157[];
    bool v158;
    v158 = v153 <= v153;
    bool v159;
    v159 = v158 == false;
    if (v159){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v158);
    } else {
    }
    float * * v161;
    v161 = reinterpret_cast<float * *>(&v157[0ull]);
    int * v163;
    v163 = reinterpret_cast<int *>(&v157[v45]);
    int v165;
    v165 = threadIdx.x;
    assert("Tensor range check" && 0 <= v165 && v165 < 256);
    v161[v165] = v151;
    __syncthreads();
    bool v166;
    v166 = 0 <= v165;
    bool v167;
    v167 = v166 == false;
    if (v167){
        assert("The index needs to be zero or positive." && v166);
    } else {
    }
    int v169;
    v169 = v165 % 64;
    int v170;
    v170 = v165 / 64;
    bool v171;
    v171 = v170 < 4;
    bool v172;
    v172 = v171 == false;
    if (v172){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v171);
    } else {
    }
    assert("Tensor range check" && 0 <= v170 && v170 < 4);
    int v174;
    v174 = 0;
    while (while_method_4(v174)){
        bool v176;
        v176 = 0 <= v170;
        bool v177;
        v177 = v176 && v171;
        bool v178;
        v178 = v177 == false;
        if (v178){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v177);
        } else {
        }
        bool v180;
        v180 = 0 <= v174;
        bool v182;
        if (v180){
            bool v181;
            v181 = v174 < 64;
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
        int v185;
        v185 = v174 * 4;
        int v186;
        v186 = v185 + v170;
        assert("Tensor range check" && 0 <= v174 && v174 < 64);
        int v187;
        v187 = 4 * v174;
        int v188;
        v188 = v187 + v170;
        float * v189;
        v189 = v161[v188];
        int v190;
        v190 = blockIdx.x;
        int v191;
        v191 = v190 * 256;
        int v192;
        v192 = v191 + v186;
        assert("Tensor range check" && 0 <= v169 && v169 < 64);
        int v193;
        v193 = 4 * v169;
        float v194[4];
        int v195[4];
        int v196;
        v196 = 0;
        while (while_method_3(v196)){
            assert("Tensor range check" && 0 <= v196 && v196 < 1);
            int v198;
            v198 = 4 * v196;
            assert("Tensor range check" && 0 <= v196 && v196 < 1);
            int v199;
            v199 = 256 * v196;
            int v200;
            v200 = v199 + v193;
            int4* v201;
            v201 = reinterpret_cast<int4*>(v189 + v200);
            int4* v202;
            v202 = reinterpret_cast<int4*>(v194 + v198);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v201) % 16 == 0 && reinterpret_cast<unsigned long long>(v202) % 16 == 0);
            *v202 = *v201;
            v196 += 1 ;
        }
        int v203;
        v203 = 0;
        while (while_method_3(v203)){
            int v205;
            v205 = 0;
            while (while_method_1(v205)){
                bool v207;
                v207 = 0 <= v205;
                bool v209;
                if (v207){
                    bool v208;
                    v208 = v205 < 4;
                    v209 = v208;
                } else {
                    v209 = false;
                }
                bool v210;
                v210 = v209 == false;
                if (v210){
                    assert("The indices should be inside the range of the dimension." && v209);
                } else {
                }
                bool v212;
                v212 = 0 <= v169;
                bool v214;
                if (v212){
                    bool v213;
                    v213 = v169 < 64;
                    v214 = v213;
                } else {
                    v214 = false;
                }
                bool v215;
                v215 = v214 == false;
                if (v215){
                    assert("The indices should be inside the range of the dimension." && v214);
                } else {
                }
                int v217;
                v217 = v169 * 4;
                int v218;
                v218 = v205 + v217;
                bool v219;
                v219 = 0 <= v203;
                bool v221;
                if (v219){
                    bool v220;
                    v220 = v203 < 1;
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
                int v224;
                v224 = v203 * 256;
                int v225;
                v225 = v218 + v224;
                assert("Tensor range check" && 0 <= v203 && v203 < 1);
                assert("Tensor range check" && 0 <= v205 && v205 < 4);
                int v226;
                v226 = 4 * v203;
                int v227;
                v227 = v226 + v205;
                v195[v227] = v225;
                v205 += 1 ;
            }
            v203 += 1 ;
        }
        int v228;
        v228 = 0;
        while (while_method_3(v228)){
            assert("Tensor range check" && 0 <= v228 && v228 < 1);
            assert("Tensor range check" && 0 <= v228 && v228 < 1);
            v228 += 1 ;
        }
        assert("Tensor range check" && 0 <= v186 && v186 < 256);
        v163[v186] = v192;
        v174 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v165 && v165 < 256);
    int v230;
    v230 = v163[v165];
    __syncthreads();
    int v231;
    v231 = threadIdx.x;
    int v232;
    v232 = blockIdx.x;
    int v233;
    v233 = v232 * 256;
    int v234;
    v234 = v231 + v233;
    assert("Tensor range check" && 0 <= v234 && v234 < 6144);
    v4[v234] = v230;
    float * v235;
    v235 = v1+v12;
    float * v237;
    v237 = v6+v32;
    unsigned long long v239;
    v239 = v45 + v41;
    bool v240;
    v240 = v239 <= 98304ull;
    bool v241;
    v241 = v240 == false;
    if (v241){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v240);
    } else {
    }
    extern __shared__ unsigned char v243[];
    bool v244;
    v244 = v239 <= v239;
    bool v245;
    v245 = v244 == false;
    if (v245){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v244);
    } else {
    }
    float * * v247;
    v247 = reinterpret_cast<float * *>(&v243[0ull]);
    float * * v249;
    v249 = reinterpret_cast<float * *>(&v243[v45]);
    int v251;
    v251 = threadIdx.x;
    assert("Tensor range check" && 0 <= v251 && v251 < 256);
    v247[v251] = v235;
    v249[v251] = v237;
    __syncthreads();
    bool v252;
    v252 = 0 <= v251;
    bool v253;
    v253 = v252 == false;
    if (v253){
        assert("The index needs to be zero or positive." && v252);
    } else {
    }
    int v255;
    v255 = v251 % 64;
    int v256;
    v256 = v251 / 64;
    bool v257;
    v257 = v256 < 4;
    bool v258;
    v258 = v257 == false;
    if (v258){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v257);
    } else {
    }
    assert("Tensor range check" && 0 <= v256 && v256 < 4);
    int v260;
    v260 = 0;
    while (while_method_4(v260)){
        bool v262;
        v262 = 0 <= v256;
        bool v263;
        v263 = v262 && v257;
        bool v264;
        v264 = v263 == false;
        if (v264){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v263);
        } else {
        }
        bool v266;
        v266 = 0 <= v260;
        bool v268;
        if (v266){
            bool v267;
            v267 = v260 < 64;
            v268 = v267;
        } else {
            v268 = false;
        }
        bool v269;
        v269 = v268 == false;
        if (v269){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v268);
        } else {
        }
        int v271;
        v271 = v260 * 4;
        int v272;
        v272 = v271 + v256;
        assert("Tensor range check" && 0 <= v260 && v260 < 64);
        int v273;
        v273 = 4 * v260;
        int v274;
        v274 = v273 + v256;
        float * v275;
        v275 = v247[v274];
        float * v276;
        v276 = v249[v274];
        int v277;
        v277 = blockIdx.x;
        int v278;
        v278 = v277 * 256;
        int v279;
        v279 = v278 + v272;
        assert("Tensor range check" && 0 <= v255 && v255 < 64);
        int v280;
        v280 = 4 * v255;
        float v281[4];
        int v282[4];
        int v283;
        v283 = 0;
        while (while_method_3(v283)){
            assert("Tensor range check" && 0 <= v283 && v283 < 1);
            int v285;
            v285 = 4 * v283;
            assert("Tensor range check" && 0 <= v283 && v283 < 1);
            int v286;
            v286 = 256 * v283;
            int v287;
            v287 = v286 + v280;
            int4* v288;
            v288 = reinterpret_cast<int4*>(v275 + v287);
            int4* v289;
            v289 = reinterpret_cast<int4*>(v281 + v285);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v288) % 16 == 0 && reinterpret_cast<unsigned long long>(v289) % 16 == 0);
            *v289 = *v288;
            v283 += 1 ;
        }
        int v290;
        v290 = 0;
        while (while_method_3(v290)){
            int v292;
            v292 = 0;
            while (while_method_1(v292)){
                bool v294;
                v294 = 0 <= v292;
                bool v296;
                if (v294){
                    bool v295;
                    v295 = v292 < 4;
                    v296 = v295;
                } else {
                    v296 = false;
                }
                bool v297;
                v297 = v296 == false;
                if (v297){
                    assert("The indices should be inside the range of the dimension." && v296);
                } else {
                }
                bool v299;
                v299 = 0 <= v255;
                bool v301;
                if (v299){
                    bool v300;
                    v300 = v255 < 64;
                    v301 = v300;
                } else {
                    v301 = false;
                }
                bool v302;
                v302 = v301 == false;
                if (v302){
                    assert("The indices should be inside the range of the dimension." && v301);
                } else {
                }
                int v304;
                v304 = v255 * 4;
                int v305;
                v305 = v292 + v304;
                bool v306;
                v306 = 0 <= v290;
                bool v308;
                if (v306){
                    bool v307;
                    v307 = v290 < 1;
                    v308 = v307;
                } else {
                    v308 = false;
                }
                bool v309;
                v309 = v308 == false;
                if (v309){
                    assert("The indices should be inside the range of the dimension." && v308);
                } else {
                }
                int v311;
                v311 = v290 * 256;
                int v312;
                v312 = v305 + v311;
                assert("Tensor range check" && 0 <= v290 && v290 < 1);
                assert("Tensor range check" && 0 <= v292 && v292 < 4);
                int v313;
                v313 = 4 * v290;
                int v314;
                v314 = v313 + v292;
                v282[v314] = v312;
                v292 += 1 ;
            }
            v290 += 1 ;
        }
        int v315;
        v315 = 0;
        while (while_method_3(v315)){
            assert("Tensor range check" && 0 <= v315 && v315 < 1);
            int v317;
            v317 = 256 * v315;
            int v318;
            v318 = v317 + v280;
            assert("Tensor range check" && 0 <= v315 && v315 < 1);
            int v319;
            v319 = 4 * v315;
            int4* v320;
            v320 = reinterpret_cast<int4*>(v281 + v319);
            int4* v321;
            v321 = reinterpret_cast<int4*>(v276 + v318);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v320) % 16 == 0 && reinterpret_cast<unsigned long long>(v321) % 16 == 0);
            *v321 = *v320;
            v315 += 1 ;
        }
        assert("Tensor range check" && 0 <= v272 && v272 < 256);
        v260 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v251 && v251 < 256);
    __syncthreads();
    float * v322;
    v322 = v1+v12;
    float * v324;
    v324 = v7+v22;
    if (v241){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v240);
    } else {
    }
    extern __shared__ unsigned char v327[];
    if (v245){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v244);
    } else {
    }
    float * * v329;
    v329 = reinterpret_cast<float * *>(&v327[0ull]);
    float * * v331;
    v331 = reinterpret_cast<float * *>(&v327[v45]);
    int v333;
    v333 = threadIdx.x;
    assert("Tensor range check" && 0 <= v333 && v333 < 256);
    v329[v333] = v322;
    v331[v333] = v324;
    __syncthreads();
    bool v334;
    v334 = 0 <= v333;
    bool v335;
    v335 = v334 == false;
    if (v335){
        assert("The index needs to be zero or positive." && v334);
    } else {
    }
    int v337;
    v337 = v333 % 64;
    int v338;
    v338 = v333 / 64;
    bool v339;
    v339 = v338 < 4;
    bool v340;
    v340 = v339 == false;
    if (v340){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v339);
    } else {
    }
    assert("Tensor range check" && 0 <= v338 && v338 < 4);
    int v342;
    v342 = 0;
    while (while_method_4(v342)){
        bool v344;
        v344 = 0 <= v338;
        bool v345;
        v345 = v344 && v339;
        bool v346;
        v346 = v345 == false;
        if (v346){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v345);
        } else {
        }
        bool v348;
        v348 = 0 <= v342;
        bool v350;
        if (v348){
            bool v349;
            v349 = v342 < 64;
            v350 = v349;
        } else {
            v350 = false;
        }
        bool v351;
        v351 = v350 == false;
        if (v351){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v350);
        } else {
        }
        int v353;
        v353 = v342 * 4;
        int v354;
        v354 = v353 + v338;
        assert("Tensor range check" && 0 <= v342 && v342 < 64);
        int v355;
        v355 = 4 * v342;
        int v356;
        v356 = v355 + v338;
        float * v357;
        v357 = v329[v356];
        float * v358;
        v358 = v331[v356];
        int v359;
        v359 = blockIdx.x;
        int v360;
        v360 = v359 * 256;
        int v361;
        v361 = v360 + v354;
        assert("Tensor range check" && 0 <= v337 && v337 < 64);
        int v362;
        v362 = 4 * v337;
        float v363[4];
        int v364[4];
        int v365;
        v365 = 0;
        while (while_method_3(v365)){
            assert("Tensor range check" && 0 <= v365 && v365 < 1);
            int v367;
            v367 = 4 * v365;
            assert("Tensor range check" && 0 <= v365 && v365 < 1);
            int v368;
            v368 = 256 * v365;
            int v369;
            v369 = v368 + v362;
            int4* v370;
            v370 = reinterpret_cast<int4*>(v357 + v369);
            int4* v371;
            v371 = reinterpret_cast<int4*>(v363 + v367);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v370) % 16 == 0 && reinterpret_cast<unsigned long long>(v371) % 16 == 0);
            *v371 = *v370;
            v365 += 1 ;
        }
        int v372;
        v372 = 0;
        while (while_method_3(v372)){
            int v374;
            v374 = 0;
            while (while_method_1(v374)){
                bool v376;
                v376 = 0 <= v374;
                bool v378;
                if (v376){
                    bool v377;
                    v377 = v374 < 4;
                    v378 = v377;
                } else {
                    v378 = false;
                }
                bool v379;
                v379 = v378 == false;
                if (v379){
                    assert("The indices should be inside the range of the dimension." && v378);
                } else {
                }
                bool v381;
                v381 = 0 <= v337;
                bool v383;
                if (v381){
                    bool v382;
                    v382 = v337 < 64;
                    v383 = v382;
                } else {
                    v383 = false;
                }
                bool v384;
                v384 = v383 == false;
                if (v384){
                    assert("The indices should be inside the range of the dimension." && v383);
                } else {
                }
                int v386;
                v386 = v337 * 4;
                int v387;
                v387 = v374 + v386;
                bool v388;
                v388 = 0 <= v372;
                bool v390;
                if (v388){
                    bool v389;
                    v389 = v372 < 1;
                    v390 = v389;
                } else {
                    v390 = false;
                }
                bool v391;
                v391 = v390 == false;
                if (v391){
                    assert("The indices should be inside the range of the dimension." && v390);
                } else {
                }
                int v393;
                v393 = v372 * 256;
                int v394;
                v394 = v387 + v393;
                assert("Tensor range check" && 0 <= v372 && v372 < 1);
                assert("Tensor range check" && 0 <= v374 && v374 < 4);
                int v395;
                v395 = 4 * v372;
                int v396;
                v396 = v395 + v374;
                v364[v396] = v394;
                v374 += 1 ;
            }
            v372 += 1 ;
        }
        bool v397[4];
        int v398;
        v398 = 0;
        while (while_method_3(v398)){
            int v400;
            v400 = 0;
            while (while_method_1(v400)){
                assert("Tensor range check" && 0 <= v398 && v398 < 1);
                assert("Tensor range check" && 0 <= v400 && v400 < 4);
                int v402;
                v402 = 4 * v398;
                int v403;
                v403 = v402 + v400;
                float v404;
                v404 = v363[v403];
                int v405;
                v405 = v364[v403];
                bool v406;
                v406 = v405 < 3;
                assert("Tensor range check" && 0 <= v398 && v398 < 1);
                assert("Tensor range check" && 0 <= v400 && v400 < 4);
                v397[v403] = v406;
                v400 += 1 ;
            }
            v398 += 1 ;
        }
        float v407[4];
        int v408;
        v408 = 0;
        while (while_method_3(v408)){
            int v410;
            v410 = 0;
            while (while_method_1(v410)){
                assert("Tensor range check" && 0 <= v408 && v408 < 1);
                assert("Tensor range check" && 0 <= v410 && v410 < 4);
                int v412;
                v412 = 4 * v408;
                int v413;
                v413 = v412 + v410;
                float v414;
                v414 = v363[v413];
                bool v415;
                v415 = v397[v413];
                float v418;
                if (v415){
                    bool v416;
                    v416 = 0.0f >= v414;
                    if (v416){
                        v418 = 0.0f;
                    } else {
                        v418 = v414;
                    }
                } else {
                    v418 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v408 && v408 < 1);
                assert("Tensor range check" && 0 <= v410 && v410 < 4);
                v407[v413] = v418;
                v410 += 1 ;
            }
            v408 += 1 ;
        }
        float v419;
        v419 = 0.0f;
        int v420;
        v420 = 0;
        while (while_method_3(v420)){
            int v422;
            v422 = 0;
            while (while_method_1(v422)){
                assert("Tensor range check" && 0 <= v420 && v420 < 1);
                assert("Tensor range check" && 0 <= v422 && v422 < 4);
                int v424;
                v424 = 4 * v420;
                int v425;
                v425 = v424 + v422;
                float v426;
                v426 = v407[v425];
                float v427;
                v427 = v419 + v426;
                v419 = v427;
                v422 += 1 ;
            }
            v420 += 1 ;
        }
        auto v428 = cooperative_groups::coalesced_threads();
        Closure0 v429{};
        float v430;
        v430 = cooperative_groups::reduce(v428, v419, v429);
        int v431;
        v431 = threadIdx.x;
        int v432;
        v432 = v431 / 32;
        unsigned long long v433;
        v433 = v239 + 16ull;
        unsigned long long v434;
        v434 = v433 - 1ull;
        unsigned long long v435;
        v435 = v434 % 16ull;
        unsigned long long v436;
        v436 = v434 - v435;
        unsigned long long v437;
        v437 = v436 + 32ull;
        bool v438;
        v438 = v437 <= 98304ull;
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
        float * v445;
        v445 = reinterpret_cast<float *>(&v441[v436]);
        bool v447;
        v447 = 0 <= v432;
        bool v448;
        v448 = v447 == false;
        if (v448){
            assert("The index needs to be zero or positive." && v447);
        } else {
        }
        int v450;
        v450 = v432 % 2;
        int v451;
        v451 = v432 / 2;
        bool v452;
        v452 = v451 < 4;
        bool v453;
        v453 = v452 == false;
        if (v453){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v452);
        } else {
        }
        assert("Tensor range check" && 0 <= v451 && v451 < 4);
        assert("Tensor range check" && 0 <= v450 && v450 < 2);
        int v455;
        v455 = 2 * v451;
        int v456;
        v456 = v455 + v450;
        v445[v456] = v430;
        int v457;
        v457 = v451 + 1;
        bool v458;
        v458 = v457 < 16;
        bool v459;
        v459 = v458 == false;
        if (v459){
            assert("The barrier_id has to be less than 16." && v458);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v457), "r"(64));
        int v461;
        v461 = threadIdx.x;
        int v462;
        v462 = v461 % 32;
        bool v463;
        v463 = v462 < 2;
        float v466;
        if (v463){
            assert("Tensor range check" && 0 <= v451 && v451 < 4);
            assert("Tensor range check" && 0 <= v462 && v462 < 2);
            int v464;
            v464 = v455 + v462;
            float v465;
            v465 = v445[v464];
            v466 = v465;
        } else {
            v466 = 0.0f;
        }
        __syncthreads();
        float v467;
        v467 = cooperative_groups::reduce(v428, v466, v429);
        int v468[4];
        int v469;
        v469 = 0;
        while (while_method_3(v469)){
            int v471;
            v471 = 0;
            while (while_method_1(v471)){
                assert("Tensor range check" && 0 <= v469 && v469 < 1);
                assert("Tensor range check" && 0 <= v471 && v471 < 4);
                int v473;
                v473 = 4 * v469;
                int v474;
                v474 = v473 + v471;
                bool v475;
                v475 = v397[v474];
                int v476;
                if (v475){
                    v476 = 1;
                } else {
                    v476 = 0;
                }
                assert("Tensor range check" && 0 <= v469 && v469 < 1);
                assert("Tensor range check" && 0 <= v471 && v471 < 4);
                v468[v474] = v476;
                v471 += 1 ;
            }
            v469 += 1 ;
        }
        int v477;
        v477 = 0;
        int v478;
        v478 = 0;
        while (while_method_3(v478)){
            int v480;
            v480 = 0;
            while (while_method_1(v480)){
                assert("Tensor range check" && 0 <= v478 && v478 < 1);
                assert("Tensor range check" && 0 <= v480 && v480 < 4);
                int v482;
                v482 = 4 * v478;
                int v483;
                v483 = v482 + v480;
                int v484;
                v484 = v468[v483];
                int v485;
                v485 = v477 + v484;
                v477 = v485;
                v480 += 1 ;
            }
            v478 += 1 ;
        }
        auto v486 = cooperative_groups::coalesced_threads();
        Closure4 v487{};
        int v488;
        v488 = cooperative_groups::reduce(v486, v477, v487);
        int v489;
        v489 = threadIdx.x;
        int v490;
        v490 = v489 / 32;
        if (v439){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v438);
        } else {
        }
        extern __shared__ unsigned char v492[];
        if (v443){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v442);
        } else {
        }
        int * v494;
        v494 = reinterpret_cast<int *>(&v492[v436]);
        bool v496;
        v496 = 0 <= v490;
        bool v497;
        v497 = v496 == false;
        if (v497){
            assert("The index needs to be zero or positive." && v496);
        } else {
        }
        int v499;
        v499 = v490 % 2;
        int v500;
        v500 = v490 / 2;
        bool v501;
        v501 = v500 < 4;
        bool v502;
        v502 = v501 == false;
        if (v502){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v501);
        } else {
        }
        assert("Tensor range check" && 0 <= v500 && v500 < 4);
        assert("Tensor range check" && 0 <= v499 && v499 < 2);
        int v504;
        v504 = 2 * v500;
        int v505;
        v505 = v504 + v499;
        v494[v505] = v488;
        int v506;
        v506 = v500 + 1;
        bool v507;
        v507 = v506 < 16;
        bool v508;
        v508 = v507 == false;
        if (v508){
            assert("The barrier_id has to be less than 16." && v507);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v506), "r"(64));
        int v510;
        v510 = threadIdx.x;
        int v511;
        v511 = v510 % 32;
        bool v512;
        v512 = v511 < 2;
        int v515;
        if (v512){
            assert("Tensor range check" && 0 <= v500 && v500 < 4);
            assert("Tensor range check" && 0 <= v511 && v511 < 2);
            int v513;
            v513 = v504 + v511;
            int v514;
            v514 = v494[v513];
            v515 = v514;
        } else {
            v515 = 0;
        }
        __syncthreads();
        int v516;
        v516 = cooperative_groups::reduce(v486, v515, v487);
        float v517;
        v517 = (float)v516;
        float v518;
        v518 = 1.0f / v517;
        float v519[4];
        int v520;
        v520 = 0;
        while (while_method_3(v520)){
            int v522;
            v522 = 0;
            while (while_method_1(v522)){
                assert("Tensor range check" && 0 <= v520 && v520 < 1);
                assert("Tensor range check" && 0 <= v522 && v522 < 4);
                int v524;
                v524 = 4 * v520;
                int v525;
                v525 = v524 + v522;
                float v526;
                v526 = v407[v525];
                bool v527;
                v527 = v397[v525];
                bool v528;
                v528 = v527 == false;
                float v533;
                if (v528){
                    v533 = 0.0f;
                } else {
                    bool v529;
                    v529 = v467 == 0.0f;
                    bool v530;
                    v530 = v529 != true;
                    if (v530){
                        float v531;
                        v531 = v526 / v467;
                        v533 = v531;
                    } else {
                        v533 = v518;
                    }
                }
                assert("Tensor range check" && 0 <= v520 && v520 < 1);
                assert("Tensor range check" && 0 <= v522 && v522 < 4);
                v519[v525] = v533;
                v522 += 1 ;
            }
            v520 += 1 ;
        }
        int v534;
        v534 = 0;
        while (while_method_3(v534)){
            assert("Tensor range check" && 0 <= v534 && v534 < 1);
            int v536;
            v536 = 256 * v534;
            int v537;
            v537 = v536 + v362;
            assert("Tensor range check" && 0 <= v534 && v534 < 1);
            int v538;
            v538 = 4 * v534;
            int4* v539;
            v539 = reinterpret_cast<int4*>(v519 + v538);
            int4* v540;
            v540 = reinterpret_cast<int4*>(v358 + v537);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v539) % 16 == 0 && reinterpret_cast<unsigned long long>(v540) % 16 == 0);
            *v540 = *v539;
            v534 += 1 ;
        }
        assert("Tensor range check" && 0 <= v354 && v354 < 256);
        v342 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v333 && v333 < 256);
    __syncthreads();
    int v541;
    v541 = threadIdx.x;
    int v542;
    v542 = blockIdx.x;
    int v543;
    v543 = v542 * 256;
    int v544;
    v544 = v541 + v543;
    unsigned long long v545;
    v545 = (unsigned long long)v544;
    curandStatePhilox4_32_10_t v546;
    curand_init(12344321ull,v545,0ull,&v546);
    float * v547;
    v547 = v1+v12;
    if (v155){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v154);
    } else {
    }
    extern __shared__ unsigned char v550[];
    if (v159){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v158);
    } else {
    }
    float * * v552;
    v552 = reinterpret_cast<float * *>(&v550[0ull]);
    int * v554;
    v554 = reinterpret_cast<int *>(&v550[v45]);
    int v556;
    v556 = threadIdx.x;
    assert("Tensor range check" && 0 <= v556 && v556 < 256);
    v552[v556] = v547;
    __syncthreads();
    bool v557;
    v557 = 0 <= v556;
    bool v558;
    v558 = v557 == false;
    if (v558){
        assert("The index needs to be zero or positive." && v557);
    } else {
    }
    int v560;
    v560 = v556 % 64;
    int v561;
    v561 = v556 / 64;
    bool v562;
    v562 = v561 < 4;
    bool v563;
    v563 = v562 == false;
    if (v563){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v562);
    } else {
    }
    assert("Tensor range check" && 0 <= v561 && v561 < 4);
    int v565;
    v565 = 0;
    while (while_method_4(v565)){
        bool v567;
        v567 = 0 <= v561;
        bool v568;
        v568 = v567 && v562;
        bool v569;
        v569 = v568 == false;
        if (v569){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v568);
        } else {
        }
        bool v571;
        v571 = 0 <= v565;
        bool v573;
        if (v571){
            bool v572;
            v572 = v565 < 64;
            v573 = v572;
        } else {
            v573 = false;
        }
        bool v574;
        v574 = v573 == false;
        if (v574){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v573);
        } else {
        }
        int v576;
        v576 = v565 * 4;
        int v577;
        v577 = v576 + v561;
        assert("Tensor range check" && 0 <= v565 && v565 < 64);
        int v578;
        v578 = 4 * v565;
        int v579;
        v579 = v578 + v561;
        float * v580;
        v580 = v552[v579];
        int v581;
        v581 = blockIdx.x;
        int v582;
        v582 = v581 * 256;
        int v583;
        v583 = v582 + v577;
        assert("Tensor range check" && 0 <= v560 && v560 < 64);
        int v584;
        v584 = 4 * v560;
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
            v590 = 256 * v587;
            int v591;
            v591 = v590 + v584;
            int4* v592;
            v592 = reinterpret_cast<int4*>(v580 + v591);
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
                v603 = 0 <= v560;
                bool v605;
                if (v603){
                    bool v604;
                    v604 = v560 < 64;
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
                v608 = v560 * 4;
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
                v615 = v594 * 256;
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
        bool v619[4];
        int v620;
        v620 = 0;
        while (while_method_3(v620)){
            int v622;
            v622 = 0;
            while (while_method_1(v622)){
                assert("Tensor range check" && 0 <= v620 && v620 < 1);
                assert("Tensor range check" && 0 <= v622 && v622 < 4);
                int v624;
                v624 = 4 * v620;
                int v625;
                v625 = v624 + v622;
                float v626;
                v626 = v585[v625];
                int v627;
                v627 = v586[v625];
                bool v628;
                v628 = v627 < 3;
                assert("Tensor range check" && 0 <= v620 && v620 < 1);
                assert("Tensor range check" && 0 <= v622 && v622 < 4);
                v619[v625] = v628;
                v622 += 1 ;
            }
            v620 += 1 ;
        }
        float v629[4];
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
                v636 = v585[v635];
                bool v637;
                v637 = v619[v635];
                float v638;
                if (v637){
                    v638 = v636;
                } else {
                    v638 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v630 && v630 < 1);
                assert("Tensor range check" && 0 <= v632 && v632 < 4);
                v629[v635] = v638;
                v632 += 1 ;
            }
            v630 += 1 ;
        }
        float v639;
        v639 = 0.0f;
        int v640;
        v640 = 0;
        while (while_method_3(v640)){
            int v642;
            v642 = 0;
            while (while_method_1(v642)){
                assert("Tensor range check" && 0 <= v640 && v640 < 1);
                assert("Tensor range check" && 0 <= v642 && v642 < 4);
                int v644;
                v644 = 4 * v640;
                int v645;
                v645 = v644 + v642;
                float v646;
                v646 = v629[v645];
                float v647;
                v647 = v639 + v646;
                v639 = v647;
                v642 += 1 ;
            }
            v640 += 1 ;
        }
        auto v648 = cooperative_groups::coalesced_threads();
        Closure0 v649{};
        float v650;
        v650 = cooperative_groups::reduce(v648, v639, v649);
        int v651;
        v651 = threadIdx.x;
        int v652;
        v652 = v651 / 32;
        unsigned long long v653;
        v653 = v153 + 16ull;
        unsigned long long v654;
        v654 = v653 - 1ull;
        unsigned long long v655;
        v655 = v654 % 16ull;
        unsigned long long v656;
        v656 = v654 - v655;
        unsigned long long v657;
        v657 = v656 + 32ull;
        bool v658;
        v658 = v657 <= 98304ull;
        bool v659;
        v659 = v658 == false;
        if (v659){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v658);
        } else {
        }
        extern __shared__ unsigned char v661[];
        bool v662;
        v662 = v657 <= v657;
        bool v663;
        v663 = v662 == false;
        if (v663){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v662);
        } else {
        }
        float * v665;
        v665 = reinterpret_cast<float *>(&v661[v656]);
        bool v667;
        v667 = 0 <= v652;
        bool v668;
        v668 = v667 == false;
        if (v668){
            assert("The index needs to be zero or positive." && v667);
        } else {
        }
        int v670;
        v670 = v652 % 2;
        int v671;
        v671 = v652 / 2;
        bool v672;
        v672 = v671 < 4;
        bool v673;
        v673 = v672 == false;
        if (v673){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v672);
        } else {
        }
        assert("Tensor range check" && 0 <= v671 && v671 < 4);
        assert("Tensor range check" && 0 <= v670 && v670 < 2);
        int v675;
        v675 = 2 * v671;
        int v676;
        v676 = v675 + v670;
        v665[v676] = v650;
        int v677;
        v677 = v671 + 1;
        bool v678;
        v678 = v677 < 16;
        bool v679;
        v679 = v678 == false;
        if (v679){
            assert("The barrier_id has to be less than 16." && v678);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v677), "r"(64));
        int v681;
        v681 = threadIdx.x;
        int v682;
        v682 = v681 % 32;
        bool v683;
        v683 = v682 < 2;
        float v686;
        if (v683){
            assert("Tensor range check" && 0 <= v671 && v671 < 4);
            assert("Tensor range check" && 0 <= v682 && v682 < 2);
            int v684;
            v684 = v675 + v682;
            float v685;
            v685 = v665[v684];
            v686 = v685;
        } else {
            v686 = 0.0f;
        }
        __syncthreads();
        float v687;
        v687 = cooperative_groups::reduce(v648, v686, v649);
        int v688[4];
        int v689;
        v689 = 0;
        while (while_method_3(v689)){
            int v691;
            v691 = 0;
            while (while_method_1(v691)){
                assert("Tensor range check" && 0 <= v689 && v689 < 1);
                assert("Tensor range check" && 0 <= v691 && v691 < 4);
                int v693;
                v693 = 4 * v689;
                int v694;
                v694 = v693 + v691;
                bool v695;
                v695 = v619[v694];
                int v696;
                if (v695){
                    v696 = 1;
                } else {
                    v696 = 0;
                }
                assert("Tensor range check" && 0 <= v689 && v689 < 1);
                assert("Tensor range check" && 0 <= v691 && v691 < 4);
                v688[v694] = v696;
                v691 += 1 ;
            }
            v689 += 1 ;
        }
        int v697;
        v697 = 0;
        int v698;
        v698 = 0;
        while (while_method_3(v698)){
            int v700;
            v700 = 0;
            while (while_method_1(v700)){
                assert("Tensor range check" && 0 <= v698 && v698 < 1);
                assert("Tensor range check" && 0 <= v700 && v700 < 4);
                int v702;
                v702 = 4 * v698;
                int v703;
                v703 = v702 + v700;
                int v704;
                v704 = v688[v703];
                int v705;
                v705 = v697 + v704;
                v697 = v705;
                v700 += 1 ;
            }
            v698 += 1 ;
        }
        auto v706 = cooperative_groups::coalesced_threads();
        Closure4 v707{};
        int v708;
        v708 = cooperative_groups::reduce(v706, v697, v707);
        int v709;
        v709 = threadIdx.x;
        int v710;
        v710 = v709 / 32;
        if (v659){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v658);
        } else {
        }
        extern __shared__ unsigned char v712[];
        if (v663){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v662);
        } else {
        }
        int * v714;
        v714 = reinterpret_cast<int *>(&v712[v656]);
        bool v716;
        v716 = 0 <= v710;
        bool v717;
        v717 = v716 == false;
        if (v717){
            assert("The index needs to be zero or positive." && v716);
        } else {
        }
        int v719;
        v719 = v710 % 2;
        int v720;
        v720 = v710 / 2;
        bool v721;
        v721 = v720 < 4;
        bool v722;
        v722 = v721 == false;
        if (v722){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v721);
        } else {
        }
        assert("Tensor range check" && 0 <= v720 && v720 < 4);
        assert("Tensor range check" && 0 <= v719 && v719 < 2);
        int v724;
        v724 = 2 * v720;
        int v725;
        v725 = v724 + v719;
        v714[v725] = v708;
        int v726;
        v726 = v720 + 1;
        bool v727;
        v727 = v726 < 16;
        bool v728;
        v728 = v727 == false;
        if (v728){
            assert("The barrier_id has to be less than 16." && v727);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v726), "r"(64));
        int v730;
        v730 = threadIdx.x;
        int v731;
        v731 = v730 % 32;
        bool v732;
        v732 = v731 < 2;
        int v735;
        if (v732){
            assert("Tensor range check" && 0 <= v720 && v720 < 4);
            assert("Tensor range check" && 0 <= v731 && v731 < 2);
            int v733;
            v733 = v724 + v731;
            int v734;
            v734 = v714[v733];
            v735 = v734;
        } else {
            v735 = 0;
        }
        __syncthreads();
        int v736;
        v736 = cooperative_groups::reduce(v706, v735, v707);
        float v737;
        v737 = (float)v736;
        float v738;
        v738 = v687 / v737;
        float v739[4];
        int v740;
        v740 = 0;
        while (while_method_3(v740)){
            int v742;
            v742 = 0;
            while (while_method_1(v742)){
                assert("Tensor range check" && 0 <= v740 && v740 < 1);
                assert("Tensor range check" && 0 <= v742 && v742 < 4);
                int v744;
                v744 = 4 * v740;
                int v745;
                v745 = v744 + v742;
                float v746;
                v746 = v585[v745];
                bool v747;
                v747 = v619[v745];
                float v748;
                if (v747){
                    v748 = v746;
                } else {
                    v748 = -1.0f / 0.0f;
                }
                float v749;
                v749 = v748 - v738;
                float v750;
                v750 = exp(v749);
                bool v751;
                v751 = v750 < 1.0f / 0.0f;
                bool v752;
                v752 = v751 == false;
                if (v752){
                    assert("The softmax values must not grow too large." && v751);
                } else {
                }
                bool v754;
                v754 = isnan(v750);
                bool v755;
                v755 = v754 == false;
                bool v756;
                v756 = v755 == false;
                if (v756){
                    assert("The softmax values must not be nans." && v755);
                } else {
                }
                assert("Tensor range check" && 0 <= v740 && v740 < 1);
                assert("Tensor range check" && 0 <= v742 && v742 < 4);
                v739[v745] = v750;
                v742 += 1 ;
            }
            v740 += 1 ;
        }
        float v758;
        v758 = 0.0f;
        int v759;
        v759 = 0;
        while (while_method_3(v759)){
            int v761;
            v761 = 0;
            while (while_method_1(v761)){
                assert("Tensor range check" && 0 <= v759 && v759 < 1);
                assert("Tensor range check" && 0 <= v761 && v761 < 4);
                int v763;
                v763 = 4 * v759;
                int v764;
                v764 = v763 + v761;
                float v765;
                v765 = v739[v764];
                float v766;
                v766 = v758 + v765;
                v758 = v766;
                v761 += 1 ;
            }
            v759 += 1 ;
        }
        auto v767 = cooperative_groups::coalesced_threads();
        float v768;
        v768 = cooperative_groups::reduce(v767, v758, v649);
        int v769;
        v769 = threadIdx.x;
        int v770;
        v770 = v769 / 32;
        if (v659){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v658);
        } else {
        }
        extern __shared__ unsigned char v772[];
        if (v663){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v662);
        } else {
        }
        float * v774;
        v774 = reinterpret_cast<float *>(&v772[v656]);
        bool v776;
        v776 = 0 <= v770;
        bool v777;
        v777 = v776 == false;
        if (v777){
            assert("The index needs to be zero or positive." && v776);
        } else {
        }
        int v779;
        v779 = v770 % 2;
        int v780;
        v780 = v770 / 2;
        bool v781;
        v781 = v780 < 4;
        bool v782;
        v782 = v781 == false;
        if (v782){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v781);
        } else {
        }
        assert("Tensor range check" && 0 <= v780 && v780 < 4);
        assert("Tensor range check" && 0 <= v779 && v779 < 2);
        int v784;
        v784 = 2 * v780;
        int v785;
        v785 = v784 + v779;
        v774[v785] = v768;
        int v786;
        v786 = v780 + 1;
        bool v787;
        v787 = v786 < 16;
        bool v788;
        v788 = v787 == false;
        if (v788){
            assert("The barrier_id has to be less than 16." && v787);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v786), "r"(64));
        int v790;
        v790 = threadIdx.x;
        int v791;
        v791 = v790 % 32;
        bool v792;
        v792 = v791 < 2;
        float v795;
        if (v792){
            assert("Tensor range check" && 0 <= v780 && v780 < 4);
            assert("Tensor range check" && 0 <= v791 && v791 < 2);
            int v793;
            v793 = v784 + v791;
            float v794;
            v794 = v774[v793];
            v795 = v794;
        } else {
            v795 = 0.0f;
        }
        __syncthreads();
        float v796;
        v796 = cooperative_groups::reduce(v767, v795, v649);
        float v797[4];
        int v798;
        v798 = 0;
        while (while_method_3(v798)){
            int v800;
            v800 = 0;
            while (while_method_1(v800)){
                assert("Tensor range check" && 0 <= v798 && v798 < 1);
                assert("Tensor range check" && 0 <= v800 && v800 < 4);
                int v802;
                v802 = 4 * v798;
                int v803;
                v803 = v802 + v800;
                float v804;
                v804 = v739[v803];
                float v805;
                v805 = v804 / v796;
                assert("Tensor range check" && 0 <= v798 && v798 < 1);
                assert("Tensor range check" && 0 <= v800 && v800 < 4);
                v797[v803] = v805;
                v800 += 1 ;
            }
            v798 += 1 ;
        }
        float v806[4];
        float v807;
        v807 = 0.0f;
        int v808;
        v808 = 0;
        while (while_method_3(v808)){
            assert("Tensor range check" && 0 <= v808 && v808 < 1);
            int v810;
            v810 = 4 * v808;
            assert("Tensor range check" && 0 <= v808 && v808 < 1);
            float v811;
            v811 = 0.0f;
            int v812;
            v812 = 0;
            while (while_method_1(v812)){
                assert("Tensor range check" && 0 <= v812 && v812 < 4);
                int v814;
                v814 = v812 + v810;
                float v815;
                v815 = v797[v814];
                float v816;
                v816 = v811 + v815;
                v811 = v816;
                v812 += 1 ;
            }
            auto v817 = cooperative_groups::coalesced_threads();
            int v818;
            v818 = threadIdx.x;
            int v819;
            v819 = v818 / 32;
            if (v659){
                assert("The dynamic shared memory is insufficient to allocate the tensor." && v658);
            } else {
            }
            extern __shared__ unsigned char v821[];
            if (v663){
                assert("The length of the partition has to be less than or equal to the length of the base array." && v662);
            } else {
            }
            float * v823;
            v823 = reinterpret_cast<float *>(&v821[v656]);
            Closure2 v825{};
            float v826;
            v826 = cooperative_groups::inclusive_scan(v817, v811, v825);
            float v827;
            v827 = v817.shfl_up(v826,1);
            bool v828;
            v828 = v817.thread_rank() == 0;
            float v829;
            if (v828){
                v829 = 0.0f;
            } else {
                v829 = v827;
            }
            float v830;
            v830 = v817.shfl(v826,v817.num_threads()-1);
            bool v831;
            v831 = 0 <= v819;
            bool v832;
            v832 = v831 == false;
            if (v832){
                assert("The index needs to be zero or positive." && v831);
            } else {
            }
            int v834;
            v834 = v819 % 2;
            int v835;
            v835 = v819 / 2;
            bool v836;
            v836 = v835 < 4;
            bool v837;
            v837 = v836 == false;
            if (v837){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v836);
            } else {
            }
            assert("Tensor range check" && 0 <= v835 && v835 < 4);
            assert("Tensor range check" && 0 <= v834 && v834 < 2);
            int v839;
            v839 = 2 * v835;
            int v840;
            v840 = v839 + v834;
            v823[v840] = v830;
            int v841;
            v841 = v835 + 1;
            bool v842;
            v842 = v841 < 16;
            bool v843;
            v843 = v842 == false;
            if (v843){
                assert("The barrier_id has to be less than 16." && v842);
            } else {
            }
            asm("barrier.cta.sync %0, %1;" :: "r"(v841), "r"(64));
            int v845;
            v845 = threadIdx.x;
            int v846;
            v846 = v845 % 32;
            bool v847;
            v847 = v846 < 2;
            float v850;
            if (v847){
                assert("Tensor range check" && 0 <= v835 && v835 < 4);
                assert("Tensor range check" && 0 <= v846 && v846 < 2);
                int v848;
                v848 = v839 + v846;
                float v849;
                v849 = v823[v848];
                v850 = v849;
            } else {
                v850 = 0.0f;
            }
            __syncthreads();
            float v851;
            v851 = cooperative_groups::inclusive_scan(v817, v850, v825);
            float v852;
            v852 = v817.shfl_up(v851,1);
            bool v853;
            v853 = v817.thread_rank() == 0;
            float v854;
            if (v853){
                v854 = 0.0f;
            } else {
                v854 = v852;
            }
            float v855;
            v855 = v817.shfl(v851,v817.num_threads()-1);
            float v856;
            v856 = v817.shfl(v854,v834);
            float v857;
            v857 = v856 + v829;
            float v858;
            v858 = v807 + v857;
            float v859;
            v859 = v858;
            int v860;
            v860 = 0;
            while (while_method_1(v860)){
                assert("Tensor range check" && 0 <= v860 && v860 < 4);
                int v862;
                v862 = v860 + v810;
                float v863;
                v863 = v797[v862];
                float v864;
                v864 = v859 + v863;
                assert("Tensor range check" && 0 <= v860 && v860 < 4);
                v806[v862] = v864;
                v859 = v864;
                v860 += 1 ;
            }
            float v865;
            v865 = v807 + v855;
            v807 = v865;
            v808 += 1 ;
        }
        float v866[4];
        bool v867[4];
        int v868;
        v868 = 0;
        while (while_method_3(v868)){
            int v870;
            v870 = 0;
            while (while_method_1(v870)){
                assert("Tensor range check" && 0 <= v868 && v868 < 1);
                assert("Tensor range check" && 0 <= v870 && v870 < 4);
                int v872;
                v872 = 4 * v868;
                int v873;
                v873 = v872 + v870;
                float v874;
                v874 = v806[v873];
                float v875;
                v875 = v797[v873];
                bool v876;
                v876 = v875 > 0.0f;
                assert("Tensor range check" && 0 <= v868 && v868 < 1);
                assert("Tensor range check" && 0 <= v870 && v870 < 4);
                v866[v873] = v874;
                v867[v873] = v876;
                v870 += 1 ;
            }
            v868 += 1 ;
        }
        float v877; bool v878;
        Tuple2 tmp36 = Tuple2{-1.0f / 0.0f, false};
        v877 = tmp36.v0; v878 = tmp36.v1;
        int v879;
        v879 = 0;
        while (while_method_3(v879)){
            int v881;
            v881 = 0;
            while (while_method_1(v881)){
                assert("Tensor range check" && 0 <= v879 && v879 < 1);
                assert("Tensor range check" && 0 <= v881 && v881 < 4);
                int v883;
                v883 = 4 * v879;
                int v884;
                v884 = v883 + v881;
                float v885;
                v885 = v866[v884];
                bool v886;
                v886 = v867[v884];
                float v893; bool v894;
                if (v878){
                    if (v886){
                        bool v887;
                        v887 = v877 >= v885;
                        float v888;
                        if (v887){
                            v888 = v877;
                        } else {
                            v888 = v885;
                        }
                        v893 = v888; v894 = true;
                    } else {
                        v893 = v877; v894 = v878;
                    }
                } else {
                    if (v886){
                        v893 = v885; v894 = v886;
                    } else {
                        v893 = v877; v894 = v878;
                    }
                }
                v877 = v893;
                v878 = v894;
                v881 += 1 ;
            }
            v879 += 1 ;
        }
        auto v895 = cooperative_groups::coalesced_threads();
        Closure5 v896{};
        float v897; bool v898;
        Tuple2 tmp37 = cooperative_groups::reduce(v895, Tuple2{v877, v878}, v896);
        v897 = tmp37.v0; v898 = tmp37.v1;
        int v899;
        v899 = threadIdx.x;
        int v900;
        v900 = v899 / 32;
        unsigned long long v901;
        v901 = v657 + 16ull;
        unsigned long long v902;
        v902 = v901 - 1ull;
        unsigned long long v903;
        v903 = v902 % 16ull;
        unsigned long long v904;
        v904 = v902 - v903;
        unsigned long long v905;
        v905 = v904 + 8ull;
        bool v906;
        v906 = v905 <= 98304ull;
        bool v907;
        v907 = v906 == false;
        if (v907){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v906);
        } else {
        }
        extern __shared__ unsigned char v909[];
        bool v910;
        v910 = v905 <= v905;
        bool v911;
        v911 = v910 == false;
        if (v911){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v910);
        } else {
        }
        float * v913;
        v913 = reinterpret_cast<float *>(&v909[v656]);
        bool * v915;
        v915 = reinterpret_cast<bool *>(&v909[v904]);
        bool v917;
        v917 = 0 <= v900;
        bool v918;
        v918 = v917 == false;
        if (v918){
            assert("The index needs to be zero or positive." && v917);
        } else {
        }
        int v920;
        v920 = v900 % 2;
        int v921;
        v921 = v900 / 2;
        bool v922;
        v922 = v921 < 4;
        bool v923;
        v923 = v922 == false;
        if (v923){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v922);
        } else {
        }
        assert("Tensor range check" && 0 <= v921 && v921 < 4);
        assert("Tensor range check" && 0 <= v920 && v920 < 2);
        int v925;
        v925 = 2 * v921;
        int v926;
        v926 = v925 + v920;
        v913[v926] = v897;
        v915[v926] = v898;
        int v927;
        v927 = v921 + 1;
        bool v928;
        v928 = v927 < 16;
        bool v929;
        v929 = v928 == false;
        if (v929){
            assert("The barrier_id has to be less than 16." && v928);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v927), "r"(64));
        int v931;
        v931 = threadIdx.x;
        int v932;
        v932 = v931 % 32;
        bool v933;
        v933 = v932 < 2;
        float v937; bool v938;
        if (v933){
            assert("Tensor range check" && 0 <= v921 && v921 < 4);
            assert("Tensor range check" && 0 <= v932 && v932 < 2);
            int v934;
            v934 = v925 + v932;
            float v935;
            v935 = v913[v934];
            bool v936;
            v936 = v915[v934];
            v937 = v935; v938 = v936;
        } else {
            v937 = -1.0f / 0.0f; v938 = false;
        }
        __syncthreads();
        float v939; bool v940;
        Tuple2 tmp38 = cooperative_groups::reduce(v895, Tuple2{v937, v938}, v896);
        v939 = tmp38.v0; v940 = tmp38.v1;
        bool v941;
        v941 = v940 == false;
        if (v941){
            assert("The local reduce must be true." && v940);
        } else {
        }
        float v943[4];
        int v944[4];
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
                int v951;
                v951 = v586[v950];
                float v952;
                v952 = curand_uniform(&v546);
                assert("Tensor range check" && 0 <= v945 && v945 < 1);
                assert("Tensor range check" && 0 <= v947 && v947 < 4);
                v943[v950] = v952;
                v944[v950] = v951;
                v947 += 1 ;
            }
            v945 += 1 ;
        }
        float v953; int v954;
        Tuple1 tmp39 = Tuple1{0.0f, 2147483647};
        v953 = tmp39.v0; v954 = tmp39.v1;
        int v955;
        v955 = 0;
        while (while_method_3(v955)){
            int v957;
            v957 = 0;
            while (while_method_1(v957)){
                assert("Tensor range check" && 0 <= v955 && v955 < 1);
                assert("Tensor range check" && 0 <= v957 && v957 < 4);
                int v959;
                v959 = 4 * v955;
                int v960;
                v960 = v959 + v957;
                float v961;
                v961 = v943[v960];
                int v962;
                v962 = v944[v960];
                bool v963;
                v963 = v954 < v962;
                float v964; int v965;
                if (v963){
                    v964 = v953; v965 = v954;
                } else {
                    v964 = v961; v965 = v962;
                }
                v953 = v964;
                v954 = v965;
                v957 += 1 ;
            }
            v955 += 1 ;
        }
        auto v966 = cooperative_groups::coalesced_threads();
        Closure6 v967{};
        float v968; int v969;
        Tuple1 tmp40 = cooperative_groups::reduce(v966, Tuple1{v953, v954}, v967);
        v968 = tmp40.v0; v969 = tmp40.v1;
        int v970;
        v970 = threadIdx.x;
        int v971;
        v971 = v970 / 32;
        unsigned long long v972;
        v972 = v904 + 32ull;
        bool v973;
        v973 = v972 <= 98304ull;
        bool v974;
        v974 = v973 == false;
        if (v974){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v973);
        } else {
        }
        extern __shared__ unsigned char v976[];
        bool v977;
        v977 = v972 <= v972;
        bool v978;
        v978 = v977 == false;
        if (v978){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v977);
        } else {
        }
        float * v980;
        v980 = reinterpret_cast<float *>(&v976[v656]);
        int * v982;
        v982 = reinterpret_cast<int *>(&v976[v904]);
        bool v984;
        v984 = 0 <= v971;
        bool v985;
        v985 = v984 == false;
        if (v985){
            assert("The index needs to be zero or positive." && v984);
        } else {
        }
        int v987;
        v987 = v971 % 2;
        int v988;
        v988 = v971 / 2;
        bool v989;
        v989 = v988 < 4;
        bool v990;
        v990 = v989 == false;
        if (v990){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v989);
        } else {
        }
        assert("Tensor range check" && 0 <= v988 && v988 < 4);
        assert("Tensor range check" && 0 <= v987 && v987 < 2);
        int v992;
        v992 = 2 * v988;
        int v993;
        v993 = v992 + v987;
        v980[v993] = v968;
        v982[v993] = v969;
        int v994;
        v994 = v988 + 1;
        bool v995;
        v995 = v994 < 16;
        bool v996;
        v996 = v995 == false;
        if (v996){
            assert("The barrier_id has to be less than 16." && v995);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v994), "r"(64));
        int v998;
        v998 = threadIdx.x;
        int v999;
        v999 = v998 % 32;
        bool v1000;
        v1000 = v999 < 2;
        float v1004; int v1005;
        if (v1000){
            assert("Tensor range check" && 0 <= v988 && v988 < 4);
            assert("Tensor range check" && 0 <= v999 && v999 < 2);
            int v1001;
            v1001 = v992 + v999;
            float v1002;
            v1002 = v980[v1001];
            int v1003;
            v1003 = v982[v1001];
            v1004 = v1002; v1005 = v1003;
        } else {
            v1004 = 0.0f; v1005 = 2147483647;
        }
        __syncthreads();
        float v1006; int v1007;
        Tuple1 tmp41 = cooperative_groups::reduce(v966, Tuple1{v1004, v1005}, v967);
        v1006 = tmp41.v0; v1007 = tmp41.v1;
        float v1008;
        v1008 = v939 * v1006;
        int v1009[4];
        bool v1010[4];
        int v1011;
        v1011 = 0;
        while (while_method_3(v1011)){
            int v1013;
            v1013 = 0;
            while (while_method_1(v1013)){
                assert("Tensor range check" && 0 <= v1011 && v1011 < 1);
                assert("Tensor range check" && 0 <= v1013 && v1013 < 4);
                int v1015;
                v1015 = 4 * v1011;
                int v1016;
                v1016 = v1015 + v1013;
                float v1017;
                v1017 = v866[v1016];
                bool v1018;
                v1018 = v867[v1016];
                int v1019;
                v1019 = v586[v1016];
                int v1022; bool v1023;
                if (v1018){
                    float v1020;
                    v1020 = v1017 - v1008;
                    bool v1021;
                    v1021 = v1020 >= 0.0f;
                    v1022 = v1019; v1023 = v1021;
                } else {
                    v1022 = 2147483647; v1023 = false;
                }
                assert("Tensor range check" && 0 <= v1011 && v1011 < 1);
                assert("Tensor range check" && 0 <= v1013 && v1013 < 4);
                v1009[v1016] = v1022;
                v1010[v1016] = v1023;
                v1013 += 1 ;
            }
            v1011 += 1 ;
        }
        int v1024; bool v1025;
        Tuple3 tmp42 = Tuple3{2147483647, false};
        v1024 = tmp42.v0; v1025 = tmp42.v1;
        int v1026;
        v1026 = 0;
        while (while_method_3(v1026)){
            int v1028;
            v1028 = 0;
            while (while_method_1(v1028)){
                assert("Tensor range check" && 0 <= v1026 && v1026 < 1);
                assert("Tensor range check" && 0 <= v1028 && v1028 < 4);
                int v1030;
                v1030 = 4 * v1026;
                int v1031;
                v1031 = v1030 + v1028;
                int v1032;
                v1032 = v1009[v1031];
                bool v1033;
                v1033 = v1010[v1031];
                int v1040; bool v1041;
                if (v1025){
                    if (v1033){
                        bool v1034;
                        v1034 = v1024 < v1032;
                        int v1035;
                        if (v1034){
                            v1035 = v1024;
                        } else {
                            v1035 = v1032;
                        }
                        v1040 = v1035; v1041 = true;
                    } else {
                        v1040 = v1024; v1041 = v1025;
                    }
                } else {
                    if (v1033){
                        v1040 = v1032; v1041 = v1033;
                    } else {
                        v1040 = v1024; v1041 = v1025;
                    }
                }
                v1024 = v1040;
                v1025 = v1041;
                v1028 += 1 ;
            }
            v1026 += 1 ;
        }
        auto v1042 = cooperative_groups::coalesced_threads();
        Closure7 v1043{};
        int v1044; bool v1045;
        Tuple3 tmp43 = cooperative_groups::reduce(v1042, Tuple3{v1024, v1025}, v1043);
        v1044 = tmp43.v0; v1045 = tmp43.v1;
        int v1046;
        v1046 = threadIdx.x;
        int v1047;
        v1047 = v1046 / 32;
        if (v907){
            assert("The dynamic shared memory is insufficient to allocate the tensor." && v906);
        } else {
        }
        extern __shared__ unsigned char v1049[];
        if (v911){
            assert("The length of the partition has to be less than or equal to the length of the base array." && v910);
        } else {
        }
        int * v1051;
        v1051 = reinterpret_cast<int *>(&v1049[v656]);
        bool * v1053;
        v1053 = reinterpret_cast<bool *>(&v1049[v904]);
        bool v1055;
        v1055 = 0 <= v1047;
        bool v1056;
        v1056 = v1055 == false;
        if (v1056){
            assert("The index needs to be zero or positive." && v1055);
        } else {
        }
        int v1058;
        v1058 = v1047 % 2;
        int v1059;
        v1059 = v1047 / 2;
        bool v1060;
        v1060 = v1059 < 4;
        bool v1061;
        v1061 = v1060 == false;
        if (v1061){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v1060);
        } else {
        }
        assert("Tensor range check" && 0 <= v1059 && v1059 < 4);
        assert("Tensor range check" && 0 <= v1058 && v1058 < 2);
        int v1063;
        v1063 = 2 * v1059;
        int v1064;
        v1064 = v1063 + v1058;
        v1051[v1064] = v1044;
        v1053[v1064] = v1045;
        int v1065;
        v1065 = v1059 + 1;
        bool v1066;
        v1066 = v1065 < 16;
        bool v1067;
        v1067 = v1066 == false;
        if (v1067){
            assert("The barrier_id has to be less than 16." && v1066);
        } else {
        }
        asm("barrier.cta.sync %0, %1;" :: "r"(v1065), "r"(64));
        int v1069;
        v1069 = threadIdx.x;
        int v1070;
        v1070 = v1069 % 32;
        bool v1071;
        v1071 = v1070 < 2;
        int v1075; bool v1076;
        if (v1071){
            assert("Tensor range check" && 0 <= v1059 && v1059 < 4);
            assert("Tensor range check" && 0 <= v1070 && v1070 < 2);
            int v1072;
            v1072 = v1063 + v1070;
            int v1073;
            v1073 = v1051[v1072];
            bool v1074;
            v1074 = v1053[v1072];
            v1075 = v1073; v1076 = v1074;
        } else {
            v1075 = 2147483647; v1076 = false;
        }
        __syncthreads();
        int v1077; bool v1078;
        Tuple3 tmp44 = cooperative_groups::reduce(v1042, Tuple3{v1075, v1076}, v1043);
        v1077 = tmp44.v0; v1078 = tmp44.v1;
        bool v1079;
        v1079 = v1078 == false;
        if (v1079){
            assert("The local reduce must be true." && v1078);
        } else {
        }
        int v1081;
        v1081 = 0;
        while (while_method_3(v1081)){
            assert("Tensor range check" && 0 <= v1081 && v1081 < 1);
            assert("Tensor range check" && 0 <= v1081 && v1081 < 1);
            v1081 += 1 ;
        }
        assert("Tensor range check" && 0 <= v577 && v577 < 256);
        v554[v577] = v1077;
        v565 += 1 ;
    }
    __syncthreads();
    assert("Tensor range check" && 0 <= v556 && v556 < 256);
    int v1083;
    v1083 = v554[v556];
    __syncthreads();
    int v1084;
    v1084 = threadIdx.x;
    int v1085;
    v1085 = blockIdx.x;
    int v1086;
    v1086 = v1085 * 256;
    int v1087;
    v1087 = v1084 + v1086;
    assert("Tensor range check" && 0 <= v1087 && v1087 < 6144);
    v5[v1087] = v1083;
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
    v1 = v0 < 6144
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
        while method39(v41):
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
        v38 = v37 >= 1024
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
            v46 = v45 >= 1024
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
        while method39(v41):
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
        v38 = v37 >= 1024
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
            v46 = v45 >= 1024
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
def method47(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
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
        while method47(v41):
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
def method48(v0 : cp.ndarray) -> None:
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
def method49(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
        v38 = v37 >= 1024
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
        while method47(v43):
            v45 = v33
            v46 = v45 >= 1024
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
def method50(v0 : cp.ndarray) -> None:
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
def method51(v0 : cp.ndarray) -> None:
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
        while method47(v41):
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
def method52(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
        v38 = v37 >= 1024
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
        while method47(v43):
            v45 = v33
            v46 = v45 >= 1024
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
def method53(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : cp.ndarray, v9 : cp.ndarray, v10 : cp.ndarray, v11 : cp.ndarray, v12 : cp.ndarray, v13 : cp.ndarray, v14 : cp.ndarray, v15 : cp.ndarray, v16 : cp.ndarray) -> None:
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
def method54(v0 : cp.ndarray) -> None:
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
def method55(v0 : cp.ndarray) -> None:
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
def method56(v0 : cp.ndarray) -> None:
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
def method57(v0 : cp.ndarray) -> None:
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
def method58(v0 : cp.ndarray) -> None:
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
def method59(v0 : cp.ndarray) -> None:
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
def method60(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
def method61(v0 : cp.ndarray) -> None:
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
def method62(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
def method63(v0 : cp.ndarray) -> None:
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
def method64(v0 : cp.ndarray) -> None:
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
def method65(v0 : cp.ndarray) -> None:
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
def method66(v0 : cp.ndarray) -> None:
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
def method67(v0 : cp.ndarray) -> None:
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
def method68(v0 : cp.ndarray) -> None:
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
def method69(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray, v3 : cp.ndarray, v4 : cp.ndarray, v5 : cp.ndarray, v6 : cp.ndarray, v7 : cp.ndarray, v8 : cp.ndarray, v9 : cp.ndarray, v10 : cp.ndarray, v11 : cp.ndarray, v12 : cp.ndarray, v13 : cp.ndarray, v14 : cp.ndarray, v15 : cp.ndarray, v16 : cp.ndarray) -> None:
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
def method70(v0 : cp.ndarray) -> None:
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
def method71(v0 : cp.ndarray) -> None:
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
def method72(v0 : cp.ndarray) -> None:
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
def method73(v0 : cp.ndarray) -> None:
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
def method74(v0 : cp.ndarray) -> None:
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
def method75(v0 : cp.ndarray) -> None:
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
def method76(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
def method77(v0 : cp.ndarray) -> None:
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
def method78(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
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
def method79(v0 : cp.ndarray) -> None:
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
def method80(v0 : cp.ndarray) -> None:
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
def method81(v0 : cp.ndarray) -> None:
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
def method82(v0 : cp.ndarray) -> None:
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
def method83(v0 : cp.ndarray) -> None:
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
def method84(v0 : cp.ndarray) -> None:
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
def method86(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method87(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method88(v0 : i32) -> bool:
    v1 = v0 < 8
    del v0
    return v1
def method85() -> None:
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
    while method86(v47):
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
        while method87(v55):
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
            while method88(v63):
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
    while method86(v118):
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
        while method88(v126):
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
            while method87(v134):
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
def method89() -> None:
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
    while method86(v47):
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
            while method47(v63):
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
    while method86(v118):
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
        while method47(v126):
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
def method90(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> None:
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
def method91(v0 : cp.ndarray) -> None:
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
def method92(v0 : cp.ndarray) -> None:
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
def method93(v0 : cp.ndarray) -> None:
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
    v2 = "{}\n"
    v3 = "2"
    print(v2.format(v3),end="")
    del v3
    cp.random.seed(12344321)
    v4 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v5 = v4.size
    v6 = 8192 == v5
    del v5
    v7 = v6 == False
    if v7:
        v8 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v6, v8
        del v8
    else:
        pass
    del v6, v7
    v9 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v10 = cp.empty(1,dtype=cp.float32)
    v11 = cp.empty(8192,dtype=cp.int32)
    v12 = cp.empty(8192,dtype=cp.float32)
    v13 = cp.empty(8192,dtype=cp.float32)
    v14 = cp.empty(8192,dtype=cp.float32)
    v15 = cp.empty(8192,dtype=cp.float32)
    v16 = cp.empty(8192,dtype=cp.float32)
    v17 = cp.empty(64,dtype=cp.int32)
    v18 = cp.empty(8192,dtype=cp.int32)
    v19 = cp.empty(8192,dtype=cp.int32)
    v20 = cp.empty(64,dtype=cp.int32)
    v21 = cp.empty(8192,dtype=cp.int32)
    v22 = cp.empty(8192,dtype=cp.float32)
    v23 = cp.empty(64,dtype=cp.int32)
    v24 = cp.empty(8192,dtype=cp.float32)
    v25 = cp.empty(64,dtype=cp.int32)
    method0(v4, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25)
    method1(v9)
    del v9
    method4(v4)
    del v4
    method5(v10)
    del v10
    method6(v12)
    del v12
    method7(v13)
    del v13
    method8(v16)
    del v16
    method9(v17)
    del v17
    method10(v14, v15)
    del v14, v15
    method11(v11)
    del v11
    method12(v18, v19)
    del v18, v19
    method13(v20)
    del v20
    method14(v21)
    del v21
    method15(v22)
    del v22
    method16(v23)
    del v23
    method17(v24)
    del v24
    method18(v25)
    del v25
    cp.random.seed(12344321)
    v26 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v27 = v26.size
    v28 = 8192 == v27
    del v27
    v29 = v28 == False
    if v29:
        v30 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v28, v30
        del v30
    else:
        pass
    del v28, v29
    v31 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v32 = cp.empty(1,dtype=cp.float32)
    v33 = cp.empty(8192,dtype=cp.int32)
    v34 = cp.empty(8192,dtype=cp.float32)
    v35 = cp.empty(8192,dtype=cp.float32)
    v36 = cp.empty(8192,dtype=cp.float32)
    v37 = cp.empty(8192,dtype=cp.float32)
    v38 = cp.empty(8192,dtype=cp.float32)
    v39 = cp.empty(128,dtype=cp.int32)
    v40 = cp.empty(8192,dtype=cp.int32)
    v41 = cp.empty(8192,dtype=cp.int32)
    v42 = cp.empty(128,dtype=cp.int32)
    v43 = cp.empty(8192,dtype=cp.int32)
    v44 = cp.empty(8192,dtype=cp.float32)
    v45 = cp.empty(128,dtype=cp.int32)
    v46 = cp.empty(8192,dtype=cp.float32)
    v47 = cp.empty(128,dtype=cp.int32)
    method19(v26, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47)
    method20(v31)
    del v31
    method21(v26)
    del v26
    method22(v32)
    del v32
    method23(v34)
    del v34
    method24(v35)
    del v35
    method25(v38)
    del v38
    method26(v39)
    del v39
    method27(v36, v37)
    del v36, v37
    method28(v33)
    del v33
    method29(v40, v41)
    del v40, v41
    method30(v42)
    del v42
    method31(v43)
    del v43
    method32(v44)
    del v44
    method33(v45)
    del v45
    method34(v46)
    del v46
    method35(v47)
    del v47
    cp.cuda.get_current_stream().synchronize()
    v50 = "3"
    print(v2.format(v50),end="")
    del v50
    cp.random.seed(12344321)
    v51 = cp.arange(0,98304,1,dtype=cp.float32) # type: ignore
    v52 = v51.size
    v53 = 98304 == v52
    del v52
    v54 = v53 == False
    if v54:
        v55 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v53, v55
        del v55
    else:
        pass
    del v53, v54
    v56 = cp.random.normal(0.0,1.0,98304,dtype=cp.float32) # type: ignore
    v57 = cp.empty(98304,dtype=cp.int32)
    v58 = cp.empty(98304,dtype=cp.int32)
    v59 = cp.empty(6144,dtype=cp.int32)
    v60 = cp.empty(6144,dtype=cp.int32)
    v61 = cp.empty(98304,dtype=cp.float32)
    v62 = cp.empty(98304,dtype=cp.float32)
    method36(v51, v56, v57, v58, v59, v60, v61, v62)
    method37(v51)
    del v51
    method40(v60)
    del v60
    method41(v57, v58)
    del v57, v58
    method42(v59)
    del v59
    method43(v62)
    del v62
    method44(v56, v61)
    del v56, v61
    cp.random.seed(12344321)
    v63 = cp.arange(0,1572864,1,dtype=cp.float32) # type: ignore
    v64 = v63.size
    v65 = 1572864 == v64
    del v64
    v66 = v65 == False
    if v66:
        v67 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v65, v67
        del v67
    else:
        pass
    del v65, v66
    v68 = cp.random.normal(0.0,1.0,1572864,dtype=cp.float32) # type: ignore
    v69 = cp.empty(1572864,dtype=cp.int32)
    v70 = cp.empty(1572864,dtype=cp.int32)
    v71 = cp.empty(6144,dtype=cp.int32)
    v72 = cp.empty(6144,dtype=cp.int32)
    v73 = cp.empty(1572864,dtype=cp.float32)
    v74 = cp.empty(1572864,dtype=cp.float32)
    method45(v63, v68, v69, v70, v71, v72, v73, v74)
    method46(v63)
    del v63
    method48(v72)
    del v72
    method49(v69, v70)
    del v69, v70
    method50(v71)
    del v71
    method51(v74)
    del v74
    method52(v68, v73)
    del v68, v73
    cp.cuda.get_current_stream().synchronize()
    v77 = "4"
    print(v2.format(v77),end="")
    del v77
    cp.random.seed(12344321)
    v78 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v79 = v78.size
    v80 = 8192 == v79
    del v79
    v81 = v80 == False
    if v81:
        v82 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v80, v82
        del v82
    else:
        pass
    del v80, v81
    v83 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v84 = cp.empty(8192,dtype=cp.int32)
    v85 = cp.empty(8192,dtype=cp.float32)
    v86 = cp.empty(8192,dtype=cp.float32)
    v87 = cp.empty(8192,dtype=cp.float32)
    v88 = cp.empty(8192,dtype=cp.float32)
    v89 = cp.empty(8192,dtype=cp.float32)
    v90 = cp.empty(128,dtype=cp.int32)
    v91 = cp.empty(8192,dtype=cp.int32)
    v92 = cp.empty(8192,dtype=cp.int32)
    v93 = cp.empty(128,dtype=cp.int32)
    v94 = cp.empty(8192,dtype=cp.int32)
    v95 = cp.empty(8192,dtype=cp.float32)
    v96 = cp.empty(128,dtype=cp.int32)
    v97 = cp.empty(8192,dtype=cp.float32)
    v98 = cp.empty(128,dtype=cp.int32)
    method53(v78, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92, v93, v94, v95, v96, v97, v98)
    method54(v83)
    del v83
    method55(v78)
    del v78
    method56(v85)
    del v85
    method57(v86)
    del v86
    method58(v89)
    del v89
    method59(v90)
    del v90
    method60(v87, v88)
    del v87, v88
    method61(v84)
    del v84
    method62(v91, v92)
    del v91, v92
    method63(v93)
    del v93
    method64(v94)
    del v94
    method65(v95)
    del v95
    method66(v96)
    del v96
    method67(v97)
    del v97
    method68(v98)
    del v98
    cp.random.seed(12344321)
    v99 = cp.arange(0,8192,1,dtype=cp.int32) # type: ignore
    v100 = v99.size
    v101 = 8192 == v100
    del v100
    v102 = v101 == False
    if v102:
        v103 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v101, v103
        del v103
    else:
        pass
    del v101, v102
    v104 = cp.random.normal(0.0,1.0,8192,dtype=cp.float32) # type: ignore
    v105 = cp.empty(8192,dtype=cp.int32)
    v106 = cp.empty(8192,dtype=cp.float32)
    v107 = cp.empty(8192,dtype=cp.float32)
    v108 = cp.empty(8192,dtype=cp.float32)
    v109 = cp.empty(8192,dtype=cp.float32)
    v110 = cp.empty(8192,dtype=cp.float32)
    v111 = cp.empty(64,dtype=cp.int32)
    v112 = cp.empty(8192,dtype=cp.int32)
    v113 = cp.empty(8192,dtype=cp.int32)
    v114 = cp.empty(64,dtype=cp.int32)
    v115 = cp.empty(8192,dtype=cp.int32)
    v116 = cp.empty(8192,dtype=cp.float32)
    v117 = cp.empty(64,dtype=cp.int32)
    v118 = cp.empty(8192,dtype=cp.float32)
    v119 = cp.empty(64,dtype=cp.int32)
    method69(v99, v104, v105, v106, v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117, v118, v119)
    method70(v104)
    del v104
    method71(v99)
    del v99
    method72(v106)
    del v106
    method73(v107)
    del v107
    method74(v110)
    del v110
    method75(v111)
    del v111
    method76(v108, v109)
    del v108, v109
    method77(v105)
    del v105
    method78(v112, v113)
    del v112, v113
    method79(v114)
    del v114
    method80(v115)
    del v115
    method81(v116)
    del v116
    method82(v117)
    del v117
    method83(v118)
    del v118
    method84(v119)
    del v119
    cp.cuda.get_current_stream().synchronize()
    v122 = "5"
    print(v2.format(v122),end="")
    del v122
    method85()
    method89()
    cp.cuda.get_current_stream().synchronize()
    v125 = "6"
    print(v2.format(v125),end="")
    del v125
    cp.random.seed(12344321)
    v126 = cp.arange(0,16384,1,dtype=cp.uint64) # type: ignore
    v127 = v126.size
    v128 = 16384 == v127
    del v127
    v129 = v128 == False
    if v129:
        v130 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v128, v130
        del v130
    else:
        pass
    del v128, v129
    v131 = cp.empty(1,dtype=cp.uint64)
    v132 = cp.empty(1,dtype=cp.uint64)
    method90(v126, v131, v132)
    method91(v126)
    del v126
    method92(v131)
    del v131
    method93(v132)
    del v132
    cp.cuda.get_current_stream().synchronize()
    v135 = "Done."
    print(v2.format(v135),end="")
    del v2, v135
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
