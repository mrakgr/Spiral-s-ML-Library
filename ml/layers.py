kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <mma.h>
using namespace nvcuda;
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>
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

struct Union0;
__device__ void block_matmul_0(float * v0, float * v1, float * v2);
__device__ void block_row_map_1(float * v0, float * v1);
__device__ void block_map_2(float * v0, float * v1);
struct Tuple0;
struct Tuple1;
struct Tuple2;
struct Tuple3;
__device__ void block_row_map_reduce_3(int * v0, float * v1, float * v2, curandStatePhilox4_32_10_t & v3);
__device__ void block_matmul_4(float * v0, float * v1, int v2, float * v3);
__device__ void block_row_map_reduce_5(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5);
__device__ void block_row_map_reduce_6(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5);
struct Union0_0 { // None
};
struct Union0_1 { // Some
    int v0;
    __device__ Union0_1(int t0) : v0(t0) {}
    __device__ Union0_1() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // None
        Union0_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // None
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Some
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // None
            case 1: new (&this->case1) Union0_1(x.case1); break; // Some
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{x};
        }
        return *this;
    }
    __device__ Union0 & operator=(Union0 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // None
            case 1: this->case1.~Union0_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Closure0 {
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
struct Closure1 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple1 {
    float v0;
    bool v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure2 {
    __device__ Tuple1 operator()(Tuple1 tup0, Tuple1 tup1){
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
                return Tuple1{v5, true};
            } else {
                return Tuple1{v0, v1};
            }
        } else {
            if (v3){
                return Tuple1{v2, v3};
            } else {
                return Tuple1{v0, v1};
            }
        }
    }
};
struct Tuple2 {
    float v0;
    int v1;
    __device__ Tuple2() = default;
    __device__ Tuple2(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple2 operator()(Tuple2 tup0, Tuple2 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple2{v0, v1};
        } else {
            return Tuple2{v2, v3};
        }
    }
};
struct Tuple3 {
    int v0;
    bool v1;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
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
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ void block_matmul_0(float * v0, float * v1, float * v2){
    int v3;
    v3 = blockIdx.x;
    assert("Tensor range check" && 0 <= v3 && v3 < 24);
    int v4;
    v4 = 4096 * v3;
    int v5;
    v5 = blockIdx.x;
    assert("Tensor range check" && 0 <= v5 && v5 < 24);
    int v6;
    v6 = 4096 * v5;
    cuda::pipeline<cuda::thread_scope_thread> v7 = cuda::make_pipeline();
    extern __shared__ unsigned char v8[];
    float * v9;
    v9 = reinterpret_cast<float *>(&v8[0ull]);
    float * v11;
    v11 = reinterpret_cast<float *>(&v8[17408ull]);
    float * v13;
    v13 = reinterpret_cast<float *>(&v8[0ull]);
    int v15;
    v15 = threadIdx.x;
    int v16;
    v16 = v15 / 32;
    bool v17;
    v17 = 0 <= v16;
    bool v18;
    v18 = v17 == false;
    if (v18){
        assert("The index needs to be zero or positive." && v17);
    } else {
    }
    int v20;
    v20 = v16 % 4;
    int v21;
    v21 = v16 / 4;
    bool v22;
    v22 = v21 < 2;
    bool v23;
    v23 = v22 == false;
    if (v23){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v22);
    } else {
    }
    assert("Tensor range check" && 0 <= v21 && v21 < 2);
    assert("Tensor range check" && 0 <= v20 && v20 < 4);
    int v25;
    v25 = 16 * v20;
    int v26;
    v26 = 2304 * v21;
    int v27;
    v27 = v26 + v25;
    float * v28;
    v28 = v13+v27;
    assert("Tensor range check" && 0 <= v21 && v21 < 2);
    int v30;
    v30 = 2176 * v21;
    int v31;
    v31 = threadIdx.x;
    int v32;
    v32 = v31 % 32;
    bool v33;
    v33 = 0 <= v32;
    bool v34;
    v34 = v33 == false;
    if (v34){
        assert("The index needs to be zero or positive." && v33);
    } else {
    }
    int v36;
    v36 = v32 % 4;
    int v37;
    v37 = v32 / 4;
    bool v38;
    v38 = v37 < 8;
    bool v39;
    v39 = v38 == false;
    if (v39){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v38);
    } else {
    }
    assert("Tensor range check" && 0 <= v37 && v37 < 8);
    assert("Tensor range check" && 0 <= v36 && v36 < 4);
    int v41;
    v41 = v36 + v30;
    int v42;
    v42 = 68 * v37;
    int v43;
    v43 = v42 + v41;
    float * v44;
    v44 = v9+v43;
    assert("Tensor range check" && 0 <= v20 && v20 < 4);
    int v46;
    v46 = 1088 * v20;
    int v47;
    v47 = threadIdx.x;
    int v48;
    v48 = v47 % 32;
    bool v49;
    v49 = 0 <= v48;
    bool v50;
    v50 = v49 == false;
    if (v50){
        assert("The index needs to be zero or positive." && v49);
    } else {
    }
    int v52;
    v52 = v48 % 4;
    int v53;
    v53 = v48 / 4;
    bool v54;
    v54 = v53 < 8;
    bool v55;
    v55 = v54 == false;
    if (v55){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v54);
    } else {
    }
    assert("Tensor range check" && 0 <= v53 && v53 < 8);
    assert("Tensor range check" && 0 <= v52 && v52 < 4);
    int v57;
    v57 = v52 + v46;
    int v58;
    v58 = 68 * v53;
    int v59;
    v59 = v58 + v57;
    float * v60;
    v60 = v11+v59;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v62[2];
    int v63;
    v63 = 0;
    while (while_method_0(v63)){
        int v65;
        v65 = 0;
        while (while_method_0(v65)){
            assert("Tensor range check" && 0 <= v63 && v63 < 1);
            assert("Tensor range check" && 0 <= v65 && v65 < 1);
            int v67;
            v67 = 64 * v65;
            int v68;
            v68 = v67 + v6;
            int v69;
            v69 = 4096 * v63;
            int v70;
            v70 = v69 + v68;
            float * v71;
            v71 = v0+v70;
            // Pushing the loop unrolling to: 0
            int v73;
            v73 = 0;
            #pragma unroll
            while (while_method_1(v73)){
                int v75;
                v75 = 0;
                #pragma unroll
                while (while_method_0(v75)){
                    assert("Tensor range check" && 0 <= v73 && v73 < 2);
                    assert("Tensor range check" && 0 <= v75 && v75 < 1);
                    int v77;
                    v77 = v73 + v75;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v78 = v62[v77];
                    wmma::fill_fragment(v78, 0.0f);
                    v75 += 1 ;
                }
                v73 += 1 ;
            }
            // Poping the loop unrolling to: 0
            int v79;
            v79 = 0;
            while (while_method_2(v79)){
                int v81;
                v81 = v79 + 1;
                bool v82;
                v82 = v79 == 0;
                int v83;
                v83 = v79 % 2;
                bool v84;
                v84 = 0 <= v79;
                bool v85;
                v85 = v84 == false;
                if (v85){
                    assert("The index needs to be zero or positive." && v84);
                } else {
                }
                bool v87;
                v87 = v79 < 1;
                bool v88;
                v88 = v87 == false;
                if (v88){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v87);
                } else {
                }
                bool v90;
                v90 = v81 < 1;
                Union0 v96;
                if (v90){
                    bool v91;
                    v91 = 0 <= v81;
                    bool v92;
                    v92 = v91 == false;
                    if (v92){
                        assert("The index needs to be zero or positive." && v91);
                    } else {
                    }
                    v96 = Union0{Union0_1{v81}};
                } else {
                    v96 = Union0{Union0_0{}};
                }
                assert("Tensor range check" && 0 <= v63 && v63 < 1);
                int v97;
                v97 = v69 + v4;
                assert("Tensor range check" && 0 <= v79 && v79 < 1);
                int v98;
                v98 = 64 * v79;
                int v99;
                v99 = v98 + v97;
                float * v100;
                v100 = v2+v99;
                assert("Tensor range check" && 0 <= v65 && v65 < 1);
                int v102;
                v102 = 4096 * v65;
                if (v82){
                    assert("Tensor range check" && 0 <= v79 && v79 < 1);
                    int v103;
                    v103 = v98 + v102;
                    float * v104;
                    v104 = v1+v103;
                    // Pushing the loop unrolling to: 0
                    v7.producer_acquire();
                    int v106;
                    v106 = threadIdx.x;
                    bool v107;
                    v107 = 0 <= v106;
                    bool v108;
                    v108 = v107 == false;
                    if (v108){
                        assert("The index needs to be zero or positive." && v107);
                    } else {
                    }
                    int v110;
                    v110 = v106 % 16;
                    int v111;
                    v111 = v106 / 16;
                    bool v112;
                    v112 = v111 < 16;
                    bool v113;
                    v113 = v112 == false;
                    if (v113){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v112);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v111 && v111 < 16);
                    assert("Tensor range check" && 0 <= v110 && v110 < 16);
                    int v115;
                    v115 = 4 * v110;
                    int v116;
                    v116 = 68 * v111;
                    int v117;
                    v117 = v116 + v115;
                    int v118;
                    v118 = 64 * v111;
                    int v119;
                    v119 = v118 + v115;
                    float * v120;
                    v120 = v11+v117;
                    float * v122;
                    v122 = v104+v119;
                    int v124;
                    v124 = 0;
                    #pragma unroll
                    while (while_method_3(v124)){
                        int v126;
                        v126 = 0;
                        #pragma unroll
                        while (while_method_0(v126)){
                            assert("Tensor range check" && 0 <= v124 && v124 < 4);
                            assert("Tensor range check" && 0 <= v126 && v126 < 1);
                            int v128;
                            v128 = 64 * v126;
                            int v129;
                            v129 = 1088 * v124;
                            int v130;
                            v130 = v129 + v128;
                            int v131;
                            v131 = 1024 * v124;
                            int v132;
                            v132 = v131 + v128;
                            constexpr int v133 = sizeof(float) * 4;
                            assert("Pointer alignment check" && (unsigned long long)(v122 + v132) % v133 == 0 && (unsigned long long)(v120 + v130) % v133 == 0);
                            cuda::memcpy_async(v120 + v130, v122 + v132, cuda::aligned_size_t<v133>(v133), v7);
                            v126 += 1 ;
                        }
                        v124 += 1 ;
                    }
                    v7.producer_commit();
                    // Poping the loop unrolling to: 0
                } else {
                }
                // Pushing the loop unrolling to: 0
                int v134;
                v134 = threadIdx.x;
                bool v135;
                v135 = 0 <= v134;
                bool v136;
                v136 = v135 == false;
                if (v136){
                    assert("The index needs to be zero or positive." && v135);
                } else {
                }
                int v138;
                v138 = v134 % 16;
                int v139;
                v139 = v134 / 16;
                bool v140;
                v140 = v139 < 16;
                bool v141;
                v141 = v140 == false;
                if (v141){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v140);
                } else {
                }
                assert("Tensor range check" && 0 <= v139 && v139 < 16);
                assert("Tensor range check" && 0 <= v138 && v138 < 16);
                int v143;
                v143 = 4 * v138;
                int v144;
                v144 = 68 * v139;
                int v145;
                v145 = v144 + v143;
                int v146;
                v146 = 64 * v139;
                int v147;
                v147 = v146 + v143;
                float * v148;
                v148 = v9+v145;
                float * v150;
                v150 = v100+v147;
                int v152;
                v152 = 0;
                #pragma unroll
                while (while_method_3(v152)){
                    int v154;
                    v154 = 0;
                    #pragma unroll
                    while (while_method_0(v154)){
                        assert("Tensor range check" && 0 <= v152 && v152 < 4);
                        assert("Tensor range check" && 0 <= v154 && v154 < 1);
                        int v156;
                        v156 = 64 * v154;
                        int v157;
                        v157 = 1088 * v152;
                        int v158;
                        v158 = v157 + v156;
                        int v159;
                        v159 = 1024 * v152;
                        int v160;
                        v160 = v159 + v156;
                        int4* v161;
                        v161 = reinterpret_cast<int4*>(v150 + v160);
                        int4* v162;
                        v162 = reinterpret_cast<int4*>(v148 + v158);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v161) % 16 == 0 && reinterpret_cast<unsigned long long>(v162) % 16 == 0);
                        *v162 = *v161;
                        v154 += 1 ;
                    }
                    v152 += 1 ;
                }
                // Poping the loop unrolling to: 0
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v163[1];
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v164[8];
                cuda::pipeline_consumer_wait_prior<0>(v7);;
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v165;
                v165 = 0;
                #pragma unroll
                while (while_method_0(v165)){
                    int v167;
                    v167 = 0;
                    #pragma unroll
                    while (while_method_4(v167)){
                        assert("Tensor range check" && 0 <= v165 && v165 < 1);
                        assert("Tensor range check" && 0 <= v167 && v167 < 8);
                        int v169;
                        v169 = 8 * v165;
                        int v170;
                        v170 = v169 + v167;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v171 = v164[v170];
                        assert("Tensor range check" && 0 <= v165 && v165 < 1);
                        int v172;
                        v172 = 1088 * v165;
                        assert("Tensor range check" && 0 <= v167 && v167 < 8);
                        int v173;
                        v173 = 8 * v167;
                        int v174;
                        v174 = v173 + v172;
                        int v175;
                        v175 = 0;
                        #pragma unroll
                        while (while_method_1(v175)){
                            int v177;
                            v177 = 0;
                            #pragma unroll
                            while (while_method_1(v177)){
                                assert("Tensor range check" && 0 <= v175 && v175 < 2);
                                assert("Tensor range check" && 0 <= v177 && v177 < 2);
                                int v179;
                                v179 = 4 * v177;
                                int v180;
                                v180 = v179 + v174;
                                int v181;
                                v181 = 544 * v175;
                                int v182;
                                v182 = v181 + v180;
                                float v183;
                                v183 = v60[v182];
                                bool v184;
                                v184 = 0 <= v177;
                                bool v186;
                                if (v184){
                                    bool v185;
                                    v185 = v177 < 2;
                                    v186 = v185;
                                } else {
                                    v186 = false;
                                }
                                bool v187;
                                v187 = v186 == false;
                                if (v187){
                                    assert("The indices should be inside the range of the dimension." && v186);
                                } else {
                                }
                                bool v189;
                                v189 = 0 <= v175;
                                bool v191;
                                if (v189){
                                    bool v190;
                                    v190 = v175 < 2;
                                    v191 = v190;
                                } else {
                                    v191 = false;
                                }
                                bool v192;
                                v192 = v191 == false;
                                if (v192){
                                    assert("The indices should be inside the range of the dimension." && v191);
                                } else {
                                }
                                int v194;
                                v194 = v175 * 2;
                                int v195;
                                v195 = v177 + v194;
                                v171.x[v195] = wmma::__float_to_tf32(v183);
                                v177 += 1 ;
                            }
                            v175 += 1 ;
                        }
                        v167 += 1 ;
                    }
                    v165 += 1 ;
                }
                // Poping the loop unrolling to: 0
                v7.consumer_release();
                switch (v96.tag) {
                    case 0: { // None
                        break;
                    }
                    case 1: { // Some
                        int v196 = v96.case1.v0;
                        assert("Tensor range check" && 0 <= v196 && v196 < 1);
                        int v197;
                        v197 = 64 * v196;
                        int v198;
                        v198 = v197 + v102;
                        float * v199;
                        v199 = v1+v198;
                        __syncthreads();
                        // Pushing the loop unrolling to: 0
                        v7.producer_acquire();
                        int v201;
                        v201 = threadIdx.x;
                        bool v202;
                        v202 = 0 <= v201;
                        bool v203;
                        v203 = v202 == false;
                        if (v203){
                            assert("The index needs to be zero or positive." && v202);
                        } else {
                        }
                        int v205;
                        v205 = v201 % 16;
                        int v206;
                        v206 = v201 / 16;
                        bool v207;
                        v207 = v206 < 16;
                        bool v208;
                        v208 = v207 == false;
                        if (v208){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v207);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v206 && v206 < 16);
                        assert("Tensor range check" && 0 <= v205 && v205 < 16);
                        int v210;
                        v210 = 4 * v205;
                        int v211;
                        v211 = 68 * v206;
                        int v212;
                        v212 = v211 + v210;
                        int v213;
                        v213 = 64 * v206;
                        int v214;
                        v214 = v213 + v210;
                        float * v215;
                        v215 = v11+v212;
                        float * v217;
                        v217 = v199+v214;
                        int v219;
                        v219 = 0;
                        #pragma unroll
                        while (while_method_3(v219)){
                            int v221;
                            v221 = 0;
                            #pragma unroll
                            while (while_method_0(v221)){
                                assert("Tensor range check" && 0 <= v219 && v219 < 4);
                                assert("Tensor range check" && 0 <= v221 && v221 < 1);
                                int v223;
                                v223 = 64 * v221;
                                int v224;
                                v224 = 1088 * v219;
                                int v225;
                                v225 = v224 + v223;
                                int v226;
                                v226 = 1024 * v219;
                                int v227;
                                v227 = v226 + v223;
                                constexpr int v228 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v217 + v227) % v228 == 0 && (unsigned long long)(v215 + v225) % v228 == 0);
                                cuda::memcpy_async(v215 + v225, v217 + v227, cuda::aligned_size_t<v228>(v228), v7);
                                v221 += 1 ;
                            }
                            v219 += 1 ;
                        }
                        v7.producer_commit();
                        // Poping the loop unrolling to: 0
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                // Pushing the loop unrolling to: 0
                int v229;
                v229 = 0;
                #pragma unroll
                while (while_method_1(v229)){
                    int v231;
                    v231 = 0;
                    #pragma unroll
                    while (while_method_4(v231)){
                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v233 = v163[0];
                        assert("Tensor range check" && 0 <= v229 && v229 < 2);
                        int v234;
                        v234 = 1088 * v229;
                        assert("Tensor range check" && 0 <= v231 && v231 < 8);
                        int v235;
                        v235 = 8 * v231;
                        int v236;
                        v236 = v235 + v234;
                        int v237;
                        v237 = 0;
                        #pragma unroll
                        while (while_method_1(v237)){
                            int v239;
                            v239 = 0;
                            #pragma unroll
                            while (while_method_1(v239)){
                                assert("Tensor range check" && 0 <= v237 && v237 < 2);
                                assert("Tensor range check" && 0 <= v239 && v239 < 2);
                                int v241;
                                v241 = 544 * v239;
                                int v242;
                                v242 = v241 + v236;
                                int v243;
                                v243 = 4 * v237;
                                int v244;
                                v244 = v243 + v242;
                                float v245;
                                v245 = v44[v244];
                                bool v246;
                                v246 = 0 <= v239;
                                bool v248;
                                if (v246){
                                    bool v247;
                                    v247 = v239 < 2;
                                    v248 = v247;
                                } else {
                                    v248 = false;
                                }
                                bool v249;
                                v249 = v248 == false;
                                if (v249){
                                    assert("The indices should be inside the range of the dimension." && v248);
                                } else {
                                }
                                bool v251;
                                v251 = 0 <= v237;
                                bool v253;
                                if (v251){
                                    bool v252;
                                    v252 = v237 < 2;
                                    v253 = v252;
                                } else {
                                    v253 = false;
                                }
                                bool v254;
                                v254 = v253 == false;
                                if (v254){
                                    assert("The indices should be inside the range of the dimension." && v253);
                                } else {
                                }
                                int v256;
                                v256 = v237 * 2;
                                int v257;
                                v257 = v239 + v256;
                                v233.x[v257] = wmma::__float_to_tf32(v245);
                                v239 += 1 ;
                            }
                            v237 += 1 ;
                        }
                        int v258;
                        v258 = 0;
                        #pragma unroll
                        while (while_method_0(v258)){
                            assert("Tensor range check" && 0 <= v229 && v229 < 2);
                            assert("Tensor range check" && 0 <= v258 && v258 < 1);
                            int v260;
                            v260 = v229 + v258;
                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v261 = v62[v260];
                            assert("Tensor range check" && 0 <= v258 && v258 < 1);
                            assert("Tensor range check" && 0 <= v231 && v231 < 8);
                            int v262;
                            v262 = 8 * v258;
                            int v263;
                            v263 = v262 + v231;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v264 = v164[v263];
                            wmma::mma_sync(v261, v233, v264, v261);
                            v258 += 1 ;
                        }
                        v231 += 1 ;
                    }
                    v229 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v79 = v81;
            }
            // Pushing the loop unrolling to: 0
            int v265;
            v265 = 0;
            #pragma unroll
            while (while_method_1(v265)){
                int v267;
                v267 = 0;
                #pragma unroll
                while (while_method_0(v267)){
                    assert("Tensor range check" && 0 <= v265 && v265 < 2);
                    assert("Tensor range check" && 0 <= v267 && v267 < 1);
                    int v269;
                    v269 = v265 + v267;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v270 = v62[v269];
                    assert("Tensor range check" && 0 <= v265 && v265 < 2);
                    assert("Tensor range check" && 0 <= v267 && v267 < 1);
                    int v271;
                    v271 = 16 * v267;
                    int v272;
                    v272 = 1152 * v265;
                    int v273;
                    v273 = v272 + v271;
                    float * v274;
                    v274 = v28+v273;
                    wmma::store_matrix_sync(v274, v270, 72, wmma::mem_row_major);
                    v267 += 1 ;
                }
                v265 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            // Pushing the loop unrolling to: 0
            int v276;
            v276 = threadIdx.x;
            bool v277;
            v277 = 0 <= v276;
            bool v278;
            v278 = v277 == false;
            if (v278){
                assert("The index needs to be zero or positive." && v277);
            } else {
            }
            int v280;
            v280 = v276 % 16;
            int v281;
            v281 = v276 / 16;
            bool v282;
            v282 = v281 < 16;
            bool v283;
            v283 = v282 == false;
            if (v283){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v282);
            } else {
            }
            assert("Tensor range check" && 0 <= v281 && v281 < 16);
            assert("Tensor range check" && 0 <= v280 && v280 < 16);
            int v285;
            v285 = 4 * v280;
            int v286;
            v286 = 64 * v281;
            int v287;
            v287 = v286 + v285;
            int v288;
            v288 = 72 * v281;
            int v289;
            v289 = v288 + v285;
            float * v290;
            v290 = v71+v287;
            float * v292;
            v292 = v13+v289;
            int v294;
            v294 = 0;
            #pragma unroll
            while (while_method_3(v294)){
                int v296;
                v296 = 0;
                #pragma unroll
                while (while_method_0(v296)){
                    assert("Tensor range check" && 0 <= v294 && v294 < 4);
                    assert("Tensor range check" && 0 <= v296 && v296 < 1);
                    int v298;
                    v298 = 64 * v296;
                    int v299;
                    v299 = 1024 * v294;
                    int v300;
                    v300 = v299 + v298;
                    int v301;
                    v301 = 1152 * v294;
                    int v302;
                    v302 = v301 + v298;
                    int4* v303;
                    v303 = reinterpret_cast<int4*>(v292 + v302);
                    int4* v304;
                    v304 = reinterpret_cast<int4*>(v290 + v300);
                    assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v303) % 16 == 0 && reinterpret_cast<unsigned long long>(v304) % 16 == 0);
                    *v304 = *v303;
                    v296 += 1 ;
                }
                v294 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            v65 += 1 ;
        }
        v63 += 1 ;
    }
    return ;
}
__device__ void block_row_map_1(float * v0, float * v1){
    int v2;
    v2 = blockIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 24);
    int v3;
    v3 = 4096 * v2;
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 24);
    int v5;
    v5 = 4096 * v4;
    int v6;
    v6 = threadIdx.x;
    bool v7;
    v7 = 0 <= v6;
    bool v8;
    v8 = v7 == false;
    if (v8){
        assert("The index needs to be zero or positive." && v7);
    } else {
    }
    int v10;
    v10 = v6 % 16;
    int v11;
    v11 = v6 / 16;
    bool v12;
    v12 = v11 < 16;
    bool v13;
    v13 = v12 == false;
    if (v13){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v12);
    } else {
    }
    assert("Tensor range check" && 0 <= v11 && v11 < 16);
    assert("Tensor range check" && 0 <= v10 && v10 < 16);
    int v15;
    v15 = 4 * v10;
    int v16;
    v16 = v15 + v3;
    int v17;
    v17 = 64 * v11;
    int v18;
    v18 = v17 + v16;
    assert("Tensor range check" && 0 <= v11 && v11 < 16);
    assert("Tensor range check" && 0 <= v10 && v10 < 16);
    int v19;
    v19 = v15 + v5;
    int v20;
    v20 = v17 + v19;
    int v21;
    v21 = 0;
    while (while_method_3(v21)){
        assert("Tensor range check" && 0 <= v21 && v21 < 4);
        int v23;
        v23 = 1024 * v21;
        int v24;
        v24 = v23 + v18;
        float v25[4];
        int v26[4];
        int v27;
        v27 = 0;
        while (while_method_0(v27)){
            assert("Tensor range check" && 0 <= v27 && v27 < 1);
            int v29;
            v29 = 4 * v27;
            assert("Tensor range check" && 0 <= v27 && v27 < 1);
            int v30;
            v30 = 64 * v27;
            int v31;
            v31 = v30 + v24;
            int4* v32;
            v32 = reinterpret_cast<int4*>(v1 + v31);
            int4* v33;
            v33 = reinterpret_cast<int4*>(v25 + v29);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v32) % 16 == 0 && reinterpret_cast<unsigned long long>(v33) % 16 == 0);
            *v33 = *v32;
            v27 += 1 ;
        }
        int v34;
        v34 = 0;
        while (while_method_0(v34)){
            int v36;
            v36 = 0;
            while (while_method_3(v36)){
                bool v38;
                v38 = 0 <= v36;
                bool v40;
                if (v38){
                    bool v39;
                    v39 = v36 < 4;
                    v40 = v39;
                } else {
                    v40 = false;
                }
                bool v41;
                v41 = v40 == false;
                if (v41){
                    assert("The indices should be inside the range of the dimension." && v40);
                } else {
                }
                bool v43;
                v43 = 0 <= v10;
                bool v45;
                if (v43){
                    bool v44;
                    v44 = v10 < 16;
                    v45 = v44;
                } else {
                    v45 = false;
                }
                bool v46;
                v46 = v45 == false;
                if (v46){
                    assert("The indices should be inside the range of the dimension." && v45);
                } else {
                }
                int v48;
                v48 = v10 * 4;
                int v49;
                v49 = v36 + v48;
                bool v50;
                v50 = 0 <= v34;
                bool v52;
                if (v50){
                    bool v51;
                    v51 = v34 < 1;
                    v52 = v51;
                } else {
                    v52 = false;
                }
                bool v53;
                v53 = v52 == false;
                if (v53){
                    assert("The indices should be inside the range of the dimension." && v52);
                } else {
                }
                int v55;
                v55 = v34 * 64;
                int v56;
                v56 = v49 + v55;
                assert("Tensor range check" && 0 <= v34 && v34 < 1);
                assert("Tensor range check" && 0 <= v36 && v36 < 4);
                int v57;
                v57 = 4 * v34;
                int v58;
                v58 = v57 + v36;
                v26[v58] = v56;
                v36 += 1 ;
            }
            v34 += 1 ;
        }
        bool v59;
        v59 = 0 <= v11;
        bool v60;
        v60 = v59 && v12;
        bool v61;
        v61 = v60 == false;
        if (v61){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v60);
        } else {
        }
        bool v63;
        v63 = 0 <= v21;
        bool v65;
        if (v63){
            bool v64;
            v64 = v21 < 4;
            v65 = v64;
        } else {
            v65 = false;
        }
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v65);
        } else {
        }
        int v68;
        v68 = v21 * 16;
        int v69;
        v69 = v68 + v11;
        float v70[4];
        int v71;
        v71 = 0;
        while (while_method_0(v71)){
            int v73;
            v73 = 0;
            while (while_method_3(v73)){
                assert("Tensor range check" && 0 <= v71 && v71 < 1);
                assert("Tensor range check" && 0 <= v73 && v73 < 4);
                int v75;
                v75 = 4 * v71;
                int v76;
                v76 = v75 + v73;
                float v77;
                v77 = v25[v76];
                float v78;
                v78 = v77 * v77;
                assert("Tensor range check" && 0 <= v71 && v71 < 1);
                assert("Tensor range check" && 0 <= v73 && v73 < 4);
                v70[v76] = v78;
                v73 += 1 ;
            }
            v71 += 1 ;
        }
        float v79;
        v79 = 0.0f;
        int v80;
        v80 = 0;
        while (while_method_0(v80)){
            int v82;
            v82 = 0;
            while (while_method_3(v82)){
                assert("Tensor range check" && 0 <= v80 && v80 < 1);
                assert("Tensor range check" && 0 <= v82 && v82 < 4);
                int v84;
                v84 = 4 * v80;
                int v85;
                v85 = v84 + v82;
                float v86;
                v86 = v70[v85];
                float v87;
                v87 = v79 + v86;
                v79 = v87;
                v82 += 1 ;
            }
            v80 += 1 ;
        }
        auto v88 = cooperative_groups::coalesced_threads();
        int v89;
        v89 = threadIdx.x;
        int v90;
        v90 = v89 / 16;
        auto v91 = cooperative_groups::labeled_partition(v88,v90);
        Closure0 v92{};
        float v93;
        v93 = cooperative_groups::reduce(v91, v79, v92);
        float v94[4];
        int v95;
        v95 = 0;
        while (while_method_0(v95)){
            int v97;
            v97 = 0;
            while (while_method_3(v97)){
                assert("Tensor range check" && 0 <= v95 && v95 < 1);
                assert("Tensor range check" && 0 <= v97 && v97 < 4);
                int v99;
                v99 = 4 * v95;
                int v100;
                v100 = v99 + v97;
                float v101;
                v101 = v25[v100];
                bool v102;
                v102 = v93 == 0.0f;
                bool v103;
                v103 = v102 != true;
                float v105;
                if (v103){
                    float v104;
                    v104 = v101 / v93;
                    v105 = v104;
                } else {
                    v105 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v95 && v95 < 1);
                assert("Tensor range check" && 0 <= v97 && v97 < 4);
                v94[v100] = v105;
                v97 += 1 ;
            }
            v95 += 1 ;
        }
        assert("Tensor range check" && 0 <= v21 && v21 < 4);
        int v106;
        v106 = v23 + v20;
        int v107;
        v107 = 0;
        while (while_method_0(v107)){
            assert("Tensor range check" && 0 <= v107 && v107 < 1);
            int v109;
            v109 = 64 * v107;
            int v110;
            v110 = v109 + v106;
            assert("Tensor range check" && 0 <= v107 && v107 < 1);
            int v111;
            v111 = 4 * v107;
            int4* v112;
            v112 = reinterpret_cast<int4*>(v94 + v111);
            int4* v113;
            v113 = reinterpret_cast<int4*>(v0 + v110);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v112) % 16 == 0 && reinterpret_cast<unsigned long long>(v113) % 16 == 0);
            *v113 = *v112;
            v107 += 1 ;
        }
        v21 += 1 ;
    }
    __syncthreads();
    return ;
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 1024;
    return v1;
}
__device__ void block_map_2(float * v0, float * v1){
    int v2;
    v2 = blockIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 24);
    int v3;
    v3 = 4096 * v2;
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 24);
    int v5;
    v5 = 4096 * v4;
    int v6;
    v6 = threadIdx.x;
    int v7;
    v7 = v6;
    while (while_method_5(v7)){
        bool v9;
        v9 = 0 <= v7;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The index needs to be zero or positive." && v9);
        } else {
        }
        int v12;
        v12 = v7 % 16;
        int v13;
        v13 = v7 / 16;
        bool v14;
        v14 = v13 < 64;
        bool v15;
        v15 = v14 == false;
        if (v15){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
        } else {
        }
        assert("Tensor range check" && 0 <= v13 && v13 < 64);
        assert("Tensor range check" && 0 <= v12 && v12 < 16);
        int v17;
        v17 = 4 * v12;
        int v18;
        v18 = v17 + v3;
        int v19;
        v19 = 64 * v13;
        int v20;
        v20 = v19 + v18;
        assert("Tensor range check" && 0 <= v13 && v13 < 64);
        assert("Tensor range check" && 0 <= v12 && v12 < 16);
        int v21;
        v21 = v17 + v5;
        int v22;
        v22 = v19 + v21;
        float v23[4];
        float v24[4];
        int4* v25;
        v25 = reinterpret_cast<int4*>(v1 + v20);
        int4* v26;
        v26 = reinterpret_cast<int4*>(v23 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v25) % 16 == 0 && reinterpret_cast<unsigned long long>(v26) % 16 == 0);
        *v26 = *v25;
        // Pushing the loop unrolling to: 0
        int v27;
        v27 = 0;
        #pragma unroll
        while (while_method_3(v27)){
            assert("Tensor range check" && 0 <= v27 && v27 < 4);
            float v29;
            v29 = v23[v27];
            bool v30;
            v30 = 0.0f >= v29;
            float v31;
            if (v30){
                v31 = 0.0f;
            } else {
                v31 = v29;
            }
            assert("Tensor range check" && 0 <= v27 && v27 < 4);
            v24[v27] = v31;
            v27 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v32;
        v32 = reinterpret_cast<int4*>(v24 + 0);
        int4* v33;
        v33 = reinterpret_cast<int4*>(v0 + v22);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v32) % 16 == 0 && reinterpret_cast<unsigned long long>(v33) % 16 == 0);
        *v33 = *v32;
        v7 += 256 ;
    }
    __syncthreads();
    return ;
}
__device__ void block_row_map_reduce_3(int * v0, float * v1, float * v2, curandStatePhilox4_32_10_t & v3){
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 24);
    int v5;
    v5 = 4096 * v4;
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 24);
    int v7;
    v7 = 4096 * v6;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 24);
    int v9;
    v9 = 64 * v8;
    int v10;
    v10 = threadIdx.x;
    bool v11;
    v11 = 0 <= v10;
    bool v12;
    v12 = v11 == false;
    if (v12){
        assert("The index needs to be zero or positive." && v11);
    } else {
    }
    int v14;
    v14 = v10 % 16;
    int v15;
    v15 = v10 / 16;
    bool v16;
    v16 = v15 < 16;
    bool v17;
    v17 = v16 == false;
    if (v17){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v16);
    } else {
    }
    assert("Tensor range check" && 0 <= v15 && v15 < 16);
    assert("Tensor range check" && 0 <= v14 && v14 < 16);
    int v19;
    v19 = 4 * v14;
    int v20;
    v20 = v19 + v5;
    int v21;
    v21 = 64 * v15;
    int v22;
    v22 = v21 + v20;
    assert("Tensor range check" && 0 <= v15 && v15 < 16);
    assert("Tensor range check" && 0 <= v14 && v14 < 16);
    int v23;
    v23 = v19 + v7;
    int v24;
    v24 = v21 + v23;
    assert("Tensor range check" && 0 <= v15 && v15 < 16);
    int v25;
    v25 = v15 + v9;
    int v26;
    v26 = 0;
    while (while_method_3(v26)){
        assert("Tensor range check" && 0 <= v26 && v26 < 4);
        int v28;
        v28 = 1024 * v26;
        int v29;
        v29 = v28 + v22;
        float v30[4];
        int v31[4];
        int v32;
        v32 = 0;
        while (while_method_0(v32)){
            assert("Tensor range check" && 0 <= v32 && v32 < 1);
            int v34;
            v34 = 4 * v32;
            assert("Tensor range check" && 0 <= v32 && v32 < 1);
            int v35;
            v35 = 64 * v32;
            int v36;
            v36 = v35 + v29;
            int4* v37;
            v37 = reinterpret_cast<int4*>(v2 + v36);
            int4* v38;
            v38 = reinterpret_cast<int4*>(v30 + v34);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v37) % 16 == 0 && reinterpret_cast<unsigned long long>(v38) % 16 == 0);
            *v38 = *v37;
            v32 += 1 ;
        }
        int v39;
        v39 = 0;
        while (while_method_0(v39)){
            int v41;
            v41 = 0;
            while (while_method_3(v41)){
                bool v43;
                v43 = 0 <= v41;
                bool v45;
                if (v43){
                    bool v44;
                    v44 = v41 < 4;
                    v45 = v44;
                } else {
                    v45 = false;
                }
                bool v46;
                v46 = v45 == false;
                if (v46){
                    assert("The indices should be inside the range of the dimension." && v45);
                } else {
                }
                bool v48;
                v48 = 0 <= v14;
                bool v50;
                if (v48){
                    bool v49;
                    v49 = v14 < 16;
                    v50 = v49;
                } else {
                    v50 = false;
                }
                bool v51;
                v51 = v50 == false;
                if (v51){
                    assert("The indices should be inside the range of the dimension." && v50);
                } else {
                }
                int v53;
                v53 = v14 * 4;
                int v54;
                v54 = v41 + v53;
                bool v55;
                v55 = 0 <= v39;
                bool v57;
                if (v55){
                    bool v56;
                    v56 = v39 < 1;
                    v57 = v56;
                } else {
                    v57 = false;
                }
                bool v58;
                v58 = v57 == false;
                if (v58){
                    assert("The indices should be inside the range of the dimension." && v57);
                } else {
                }
                int v60;
                v60 = v39 * 64;
                int v61;
                v61 = v54 + v60;
                assert("Tensor range check" && 0 <= v39 && v39 < 1);
                assert("Tensor range check" && 0 <= v41 && v41 < 4);
                int v62;
                v62 = 4 * v39;
                int v63;
                v63 = v62 + v41;
                v31[v63] = v61;
                v41 += 1 ;
            }
            v39 += 1 ;
        }
        bool v64;
        v64 = 0 <= v15;
        bool v65;
        v65 = v64 && v16;
        bool v66;
        v66 = v65 == false;
        if (v66){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v65);
        } else {
        }
        bool v68;
        v68 = 0 <= v26;
        bool v70;
        if (v68){
            bool v69;
            v69 = v26 < 4;
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
        v73 = v26 * 16;
        int v74;
        v74 = v73 + v15;
        float v75;
        v75 = 0.0f;
        int v76;
        v76 = 0;
        while (while_method_0(v76)){
            int v78;
            v78 = 0;
            while (while_method_3(v78)){
                assert("Tensor range check" && 0 <= v76 && v76 < 1);
                assert("Tensor range check" && 0 <= v78 && v78 < 4);
                int v80;
                v80 = 4 * v76;
                int v81;
                v81 = v80 + v78;
                float v82;
                v82 = v30[v81];
                float v83;
                v83 = v75 + v82;
                v75 = v83;
                v78 += 1 ;
            }
            v76 += 1 ;
        }
        auto v84 = cooperative_groups::coalesced_threads();
        int v85;
        v85 = threadIdx.x;
        int v86;
        v86 = v85 / 16;
        auto v87 = cooperative_groups::labeled_partition(v84,v86);
        Closure0 v88{};
        float v89;
        v89 = cooperative_groups::reduce(v87, v75, v88);
        float v90;
        v90 = v89 / 64.0f;
        float v91[4];
        int v92;
        v92 = 0;
        while (while_method_0(v92)){
            int v94;
            v94 = 0;
            while (while_method_3(v94)){
                assert("Tensor range check" && 0 <= v92 && v92 < 1);
                assert("Tensor range check" && 0 <= v94 && v94 < 4);
                int v96;
                v96 = 4 * v92;
                int v97;
                v97 = v96 + v94;
                float v98;
                v98 = v30[v97];
                float v99;
                v99 = v98 - v90;
                float v100;
                v100 = exp(v99);
                assert("Tensor range check" && 0 <= v92 && v92 < 1);
                assert("Tensor range check" && 0 <= v94 && v94 < 4);
                v91[v97] = v100;
                v94 += 1 ;
            }
            v92 += 1 ;
        }
        float v101;
        v101 = 0.0f;
        int v102;
        v102 = 0;
        while (while_method_0(v102)){
            int v104;
            v104 = 0;
            while (while_method_3(v104)){
                assert("Tensor range check" && 0 <= v102 && v102 < 1);
                assert("Tensor range check" && 0 <= v104 && v104 < 4);
                int v106;
                v106 = 4 * v102;
                int v107;
                v107 = v106 + v104;
                float v108;
                v108 = v91[v107];
                float v109;
                v109 = v101 + v108;
                v101 = v109;
                v104 += 1 ;
            }
            v102 += 1 ;
        }
        auto v110 = cooperative_groups::coalesced_threads();
        int v111;
        v111 = threadIdx.x;
        int v112;
        v112 = v111 / 16;
        auto v113 = cooperative_groups::labeled_partition(v110,v112);
        float v114;
        v114 = cooperative_groups::reduce(v113, v101, v88);
        float v115[4];
        int v116;
        v116 = 0;
        while (while_method_0(v116)){
            int v118;
            v118 = 0;
            while (while_method_3(v118)){
                assert("Tensor range check" && 0 <= v116 && v116 < 1);
                assert("Tensor range check" && 0 <= v118 && v118 < 4);
                int v120;
                v120 = 4 * v116;
                int v121;
                v121 = v120 + v118;
                float v122;
                v122 = v91[v121];
                float v123;
                v123 = v122 / v114;
                assert("Tensor range check" && 0 <= v116 && v116 < 1);
                assert("Tensor range check" && 0 <= v118 && v118 < 4);
                v115[v121] = v123;
                v118 += 1 ;
            }
            v116 += 1 ;
        }
        float v124[4];
        float v125;
        v125 = 0.0f;
        int v126;
        v126 = 0;
        while (while_method_0(v126)){
            assert("Tensor range check" && 0 <= v126 && v126 < 1);
            int v128;
            v128 = 4 * v126;
            assert("Tensor range check" && 0 <= v126 && v126 < 1);
            int v129; float v130;
            Tuple0 tmp0 = Tuple0{0, 0.0f};
            v129 = tmp0.v0; v130 = tmp0.v1;
            while (while_method_3(v129)){
                assert("Tensor range check" && 0 <= v129 && v129 < 4);
                int v132;
                v132 = v129 + v128;
                float v133;
                v133 = v115[v132];
                float v134;
                v134 = v130 + v133;
                v130 = v134;
                v129 += 1 ;
            }
            auto v135 = cooperative_groups::coalesced_threads();
            int v136;
            v136 = threadIdx.x;
            int v137;
            v137 = v136 / 16;
            auto v138 = cooperative_groups::labeled_partition(v135,v137);
            Closure1 v139{};
            float v140;
            v140 = cooperative_groups::inclusive_scan(v138, v130, v139);
            float v141;
            v141 = v138.shfl_up(v140,1);
            bool v142;
            v142 = v138.thread_rank() == 0;
            float v143;
            if (v142){
                v143 = 0.0f;
            } else {
                v143 = v141;
            }
            float v144;
            v144 = v138.shfl(v140,v138.num_threads()-1);
            float v145;
            v145 = v125 + v143;
            int v146; float v147;
            Tuple0 tmp1 = Tuple0{0, v145};
            v146 = tmp1.v0; v147 = tmp1.v1;
            while (while_method_3(v146)){
                assert("Tensor range check" && 0 <= v146 && v146 < 4);
                int v149;
                v149 = v146 + v128;
                float v150;
                v150 = v115[v149];
                float v151;
                v151 = v147 + v150;
                assert("Tensor range check" && 0 <= v146 && v146 < 4);
                v124[v149] = v151;
                v147 = v151;
                v146 += 1 ;
            }
            float v152;
            v152 = v125 + v144;
            v125 = v152;
            v126 += 1 ;
        }
        float v153[4];
        bool v154[4];
        int v155;
        v155 = 0;
        while (while_method_0(v155)){
            int v157;
            v157 = 0;
            while (while_method_3(v157)){
                assert("Tensor range check" && 0 <= v155 && v155 < 1);
                assert("Tensor range check" && 0 <= v157 && v157 < 4);
                int v159;
                v159 = 4 * v155;
                int v160;
                v160 = v159 + v157;
                float v161;
                v161 = v124[v160];
                float v162;
                v162 = v115[v160];
                bool v163;
                v163 = v162 > 0.0f;
                assert("Tensor range check" && 0 <= v155 && v155 < 1);
                assert("Tensor range check" && 0 <= v157 && v157 < 4);
                v153[v160] = v161;
                v154[v160] = v163;
                v157 += 1 ;
            }
            v155 += 1 ;
        }
        float v164; bool v165;
        Tuple1 tmp2 = Tuple1{-1.0f / 0.0f, false};
        v164 = tmp2.v0; v165 = tmp2.v1;
        int v166;
        v166 = 0;
        while (while_method_0(v166)){
            int v168;
            v168 = 0;
            while (while_method_3(v168)){
                assert("Tensor range check" && 0 <= v166 && v166 < 1);
                assert("Tensor range check" && 0 <= v168 && v168 < 4);
                int v170;
                v170 = 4 * v166;
                int v171;
                v171 = v170 + v168;
                float v172;
                v172 = v153[v171];
                bool v173;
                v173 = v154[v171];
                float v180; bool v181;
                if (v165){
                    if (v173){
                        bool v174;
                        v174 = v164 >= v172;
                        float v175;
                        if (v174){
                            v175 = v164;
                        } else {
                            v175 = v172;
                        }
                        v180 = v175; v181 = true;
                    } else {
                        v180 = v164; v181 = v165;
                    }
                } else {
                    if (v173){
                        v180 = v172; v181 = v173;
                    } else {
                        v180 = v164; v181 = v165;
                    }
                }
                v164 = v180;
                v165 = v181;
                v168 += 1 ;
            }
            v166 += 1 ;
        }
        auto v182 = cooperative_groups::coalesced_threads();
        int v183;
        v183 = threadIdx.x;
        int v184;
        v184 = v183 / 16;
        auto v185 = cooperative_groups::labeled_partition(v182,v184);
        Closure2 v186{};
        float v187; bool v188;
        Tuple1 tmp3 = cooperative_groups::reduce(v185, Tuple1{v164, v165}, v186);
        v187 = tmp3.v0; v188 = tmp3.v1;
        bool v189;
        v189 = v188 == false;
        if (v189){
            assert("The local reduce must be true." && v188);
        } else {
        }
        float v191[4];
        int v192[4];
        int v193;
        v193 = 0;
        while (while_method_0(v193)){
            int v195;
            v195 = 0;
            while (while_method_3(v195)){
                assert("Tensor range check" && 0 <= v193 && v193 < 1);
                assert("Tensor range check" && 0 <= v195 && v195 < 4);
                int v197;
                v197 = 4 * v193;
                int v198;
                v198 = v197 + v195;
                int v199;
                v199 = v31[v198];
                float v200;
                v200 = curand_uniform(&v3);
                assert("Tensor range check" && 0 <= v193 && v193 < 1);
                assert("Tensor range check" && 0 <= v195 && v195 < 4);
                v191[v198] = v200;
                v192[v198] = v199;
                v195 += 1 ;
            }
            v193 += 1 ;
        }
        float v201; int v202;
        Tuple2 tmp4 = Tuple2{0.0f, 2147483647};
        v201 = tmp4.v0; v202 = tmp4.v1;
        int v203;
        v203 = 0;
        while (while_method_0(v203)){
            int v205;
            v205 = 0;
            while (while_method_3(v205)){
                assert("Tensor range check" && 0 <= v203 && v203 < 1);
                assert("Tensor range check" && 0 <= v205 && v205 < 4);
                int v207;
                v207 = 4 * v203;
                int v208;
                v208 = v207 + v205;
                float v209;
                v209 = v191[v208];
                int v210;
                v210 = v192[v208];
                bool v211;
                v211 = v202 < v210;
                float v212; int v213;
                if (v211){
                    v212 = v201; v213 = v202;
                } else {
                    v212 = v209; v213 = v210;
                }
                v201 = v212;
                v202 = v213;
                v205 += 1 ;
            }
            v203 += 1 ;
        }
        auto v214 = cooperative_groups::coalesced_threads();
        int v215;
        v215 = threadIdx.x;
        int v216;
        v216 = v215 / 16;
        auto v217 = cooperative_groups::labeled_partition(v214,v216);
        Closure3 v218{};
        float v219; int v220;
        Tuple2 tmp5 = cooperative_groups::reduce(v217, Tuple2{v201, v202}, v218);
        v219 = tmp5.v0; v220 = tmp5.v1;
        float v221;
        v221 = v187 * v219;
        int v222[4];
        bool v223[4];
        int v224;
        v224 = 0;
        while (while_method_0(v224)){
            int v226;
            v226 = 0;
            while (while_method_3(v226)){
                assert("Tensor range check" && 0 <= v224 && v224 < 1);
                assert("Tensor range check" && 0 <= v226 && v226 < 4);
                int v228;
                v228 = 4 * v224;
                int v229;
                v229 = v228 + v226;
                float v230;
                v230 = v153[v229];
                bool v231;
                v231 = v154[v229];
                int v232;
                v232 = v31[v229];
                int v235; bool v236;
                if (v231){
                    float v233;
                    v233 = v230 - v221;
                    bool v234;
                    v234 = v233 >= 0.0f;
                    v235 = v232; v236 = v234;
                } else {
                    v235 = 2147483647; v236 = false;
                }
                assert("Tensor range check" && 0 <= v224 && v224 < 1);
                assert("Tensor range check" && 0 <= v226 && v226 < 4);
                v222[v229] = v235;
                v223[v229] = v236;
                v226 += 1 ;
            }
            v224 += 1 ;
        }
        int v237; bool v238;
        Tuple3 tmp6 = Tuple3{2147483647, false};
        v237 = tmp6.v0; v238 = tmp6.v1;
        int v239;
        v239 = 0;
        while (while_method_0(v239)){
            int v241;
            v241 = 0;
            while (while_method_3(v241)){
                assert("Tensor range check" && 0 <= v239 && v239 < 1);
                assert("Tensor range check" && 0 <= v241 && v241 < 4);
                int v243;
                v243 = 4 * v239;
                int v244;
                v244 = v243 + v241;
                int v245;
                v245 = v222[v244];
                bool v246;
                v246 = v223[v244];
                int v253; bool v254;
                if (v238){
                    if (v246){
                        bool v247;
                        v247 = v237 < v245;
                        int v248;
                        if (v247){
                            v248 = v237;
                        } else {
                            v248 = v245;
                        }
                        v253 = v248; v254 = true;
                    } else {
                        v253 = v237; v254 = v238;
                    }
                } else {
                    if (v246){
                        v253 = v245; v254 = v246;
                    } else {
                        v253 = v237; v254 = v238;
                    }
                }
                v237 = v253;
                v238 = v254;
                v241 += 1 ;
            }
            v239 += 1 ;
        }
        auto v255 = cooperative_groups::coalesced_threads();
        int v256;
        v256 = threadIdx.x;
        int v257;
        v257 = v256 / 16;
        auto v258 = cooperative_groups::labeled_partition(v255,v257);
        Closure4 v259{};
        int v260; bool v261;
        Tuple3 tmp7 = cooperative_groups::reduce(v258, Tuple3{v237, v238}, v259);
        v260 = tmp7.v0; v261 = tmp7.v1;
        bool v262;
        v262 = v261 == false;
        if (v262){
            assert("The local reduce must be true." && v261);
        } else {
        }
        assert("Tensor range check" && 0 <= v26 && v26 < 4);
        int v264;
        v264 = v28 + v24;
        int v265;
        v265 = 0;
        while (while_method_0(v265)){
            assert("Tensor range check" && 0 <= v265 && v265 < 1);
            int v267;
            v267 = 64 * v265;
            int v268;
            v268 = v267 + v264;
            assert("Tensor range check" && 0 <= v265 && v265 < 1);
            int v269;
            v269 = 4 * v265;
            int4* v270;
            v270 = reinterpret_cast<int4*>(v115 + v269);
            int4* v271;
            v271 = reinterpret_cast<int4*>(v1 + v268);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v270) % 16 == 0 && reinterpret_cast<unsigned long long>(v271) % 16 == 0);
            *v271 = *v270;
            v265 += 1 ;
        }
        assert("Tensor range check" && 0 <= v26 && v26 < 4);
        int v272;
        v272 = 16 * v26;
        int v273;
        v273 = v272 + v25;
        v0[v273] = v260;
        v26 += 1 ;
    }
    __syncthreads();
    return ;
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 16;
    return v1;
}
__device__ void block_matmul_4(float * v0, float * v1, int v2, float * v3){
    int v4;
    v4 = blockIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 24);
    int v5;
    v5 = 4096 * v4;
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 24);
    int v7;
    v7 = 4096 * v6;
    cuda::pipeline<cuda::thread_scope_thread> v8 = cuda::make_pipeline();
    extern __shared__ unsigned char v9[];
    float * v10;
    v10 = reinterpret_cast<float *>(&v9[0ull]);
    float * v12;
    v12 = reinterpret_cast<float *>(&v9[17408ull]);
    float * v14;
    v14 = reinterpret_cast<float *>(&v9[0ull]);
    int v16;
    v16 = threadIdx.x;
    int v17;
    v17 = v16 / 32;
    bool v18;
    v18 = 0 <= v17;
    bool v19;
    v19 = v18 == false;
    if (v19){
        assert("The index needs to be zero or positive." && v18);
    } else {
    }
    int v21;
    v21 = v17 % 4;
    int v22;
    v22 = v17 / 4;
    bool v23;
    v23 = v22 < 2;
    bool v24;
    v24 = v23 == false;
    if (v24){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v23);
    } else {
    }
    assert("Tensor range check" && 0 <= v22 && v22 < 2);
    assert("Tensor range check" && 0 <= v21 && v21 < 4);
    int v26;
    v26 = 16 * v21;
    int v27;
    v27 = 2304 * v22;
    int v28;
    v28 = v27 + v26;
    float * v29;
    v29 = v14+v28;
    assert("Tensor range check" && 0 <= v22 && v22 < 2);
    int v31;
    v31 = 2176 * v22;
    int v32;
    v32 = threadIdx.x;
    int v33;
    v33 = v32 % 32;
    bool v34;
    v34 = 0 <= v33;
    bool v35;
    v35 = v34 == false;
    if (v35){
        assert("The index needs to be zero or positive." && v34);
    } else {
    }
    int v37;
    v37 = v33 % 4;
    int v38;
    v38 = v33 / 4;
    bool v39;
    v39 = v38 < 8;
    bool v40;
    v40 = v39 == false;
    if (v40){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v39);
    } else {
    }
    assert("Tensor range check" && 0 <= v38 && v38 < 8);
    assert("Tensor range check" && 0 <= v37 && v37 < 4);
    int v42;
    v42 = v37 + v31;
    int v43;
    v43 = 68 * v38;
    int v44;
    v44 = v43 + v42;
    float * v45;
    v45 = v10+v44;
    assert("Tensor range check" && 0 <= v21 && v21 < 4);
    int v47;
    v47 = 1088 * v21;
    int v48;
    v48 = threadIdx.x;
    int v49;
    v49 = v48 % 32;
    bool v50;
    v50 = 0 <= v49;
    bool v51;
    v51 = v50 == false;
    if (v51){
        assert("The index needs to be zero or positive." && v50);
    } else {
    }
    int v53;
    v53 = v49 % 4;
    int v54;
    v54 = v49 / 4;
    bool v55;
    v55 = v54 < 8;
    bool v56;
    v56 = v55 == false;
    if (v56){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v55);
    } else {
    }
    assert("Tensor range check" && 0 <= v54 && v54 < 8);
    assert("Tensor range check" && 0 <= v53 && v53 < 4);
    int v58;
    v58 = v53 + v47;
    int v59;
    v59 = 68 * v54;
    int v60;
    v60 = v59 + v58;
    float * v61;
    v61 = v12+v60;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v63[2];
    int v64;
    v64 = 0;
    while (while_method_0(v64)){
        int v66;
        v66 = 0;
        while (while_method_0(v66)){
            assert("Tensor range check" && 0 <= v64 && v64 < 1);
            assert("Tensor range check" && 0 <= v66 && v66 < 1);
            int v68;
            v68 = 64 * v66;
            int v69;
            v69 = v68 + v7;
            int v70;
            v70 = 4096 * v64;
            int v71;
            v71 = v70 + v69;
            float * v72;
            v72 = v0+v71;
            // Pushing the loop unrolling to: 0
            int v74;
            v74 = 0;
            #pragma unroll
            while (while_method_1(v74)){
                int v76;
                v76 = 0;
                #pragma unroll
                while (while_method_0(v76)){
                    assert("Tensor range check" && 0 <= v74 && v74 < 2);
                    assert("Tensor range check" && 0 <= v76 && v76 < 1);
                    int v78;
                    v78 = v74 + v76;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v79 = v63[v78];
                    wmma::fill_fragment(v79, 0.0f);
                    v76 += 1 ;
                }
                v74 += 1 ;
            }
            // Poping the loop unrolling to: 0
            int v80;
            v80 = 0;
            while (while_method_2(v80)){
                int v82;
                v82 = v80 + 1;
                bool v83;
                v83 = v80 == 0;
                int v84;
                v84 = v80 % 2;
                bool v85;
                v85 = 0 <= v80;
                bool v86;
                v86 = v85 == false;
                if (v86){
                    assert("The index needs to be zero or positive." && v85);
                } else {
                }
                bool v88;
                v88 = v80 < 1;
                bool v89;
                v89 = v88 == false;
                if (v89){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v88);
                } else {
                }
                bool v91;
                v91 = v82 < 1;
                Union0 v97;
                if (v91){
                    bool v92;
                    v92 = 0 <= v82;
                    bool v93;
                    v93 = v92 == false;
                    if (v93){
                        assert("The index needs to be zero or positive." && v92);
                    } else {
                    }
                    v97 = Union0{Union0_1{v82}};
                } else {
                    v97 = Union0{Union0_0{}};
                }
                assert("Tensor range check" && 0 <= v64 && v64 < 1);
                int v98;
                v98 = v70 + v5;
                assert("Tensor range check" && 0 <= v80 && v80 < 1);
                int v99;
                v99 = 64 * v80;
                int v100;
                v100 = v99 + v98;
                float * v101;
                v101 = v3+v100;
                assert("Tensor range check" && 0 <= v66 && v66 < 1);
                int v103;
                v103 = 4096 * v66;
                int v104;
                v104 = v103 + v2;
                if (v83){
                    assert("Tensor range check" && 0 <= v80 && v80 < 1);
                    int v105;
                    v105 = v99 + v104;
                    float * v106;
                    v106 = v1+v105;
                    // Pushing the loop unrolling to: 0
                    v8.producer_acquire();
                    int v108;
                    v108 = threadIdx.x;
                    bool v109;
                    v109 = 0 <= v108;
                    bool v110;
                    v110 = v109 == false;
                    if (v110){
                        assert("The index needs to be zero or positive." && v109);
                    } else {
                    }
                    int v112;
                    v112 = v108 % 16;
                    int v113;
                    v113 = v108 / 16;
                    bool v114;
                    v114 = v113 < 16;
                    bool v115;
                    v115 = v114 == false;
                    if (v115){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v114);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v113 && v113 < 16);
                    assert("Tensor range check" && 0 <= v112 && v112 < 16);
                    int v117;
                    v117 = 4 * v112;
                    int v118;
                    v118 = 68 * v113;
                    int v119;
                    v119 = v118 + v117;
                    int v120;
                    v120 = 64 * v113;
                    int v121;
                    v121 = v120 + v117;
                    float * v122;
                    v122 = v12+v119;
                    float * v124;
                    v124 = v106+v121;
                    int v126;
                    v126 = 0;
                    #pragma unroll
                    while (while_method_3(v126)){
                        int v128;
                        v128 = 0;
                        #pragma unroll
                        while (while_method_0(v128)){
                            assert("Tensor range check" && 0 <= v126 && v126 < 4);
                            assert("Tensor range check" && 0 <= v128 && v128 < 1);
                            int v130;
                            v130 = 64 * v128;
                            int v131;
                            v131 = 1088 * v126;
                            int v132;
                            v132 = v131 + v130;
                            int v133;
                            v133 = 1024 * v126;
                            int v134;
                            v134 = v133 + v130;
                            constexpr int v135 = sizeof(float) * 4;
                            assert("Pointer alignment check" && (unsigned long long)(v124 + v134) % v135 == 0 && (unsigned long long)(v122 + v132) % v135 == 0);
                            cuda::memcpy_async(v122 + v132, v124 + v134, cuda::aligned_size_t<v135>(v135), v8);
                            v128 += 1 ;
                        }
                        v126 += 1 ;
                    }
                    v8.producer_commit();
                    // Poping the loop unrolling to: 0
                } else {
                }
                // Pushing the loop unrolling to: 0
                int v136;
                v136 = threadIdx.x;
                bool v137;
                v137 = 0 <= v136;
                bool v138;
                v138 = v137 == false;
                if (v138){
                    assert("The index needs to be zero or positive." && v137);
                } else {
                }
                int v140;
                v140 = v136 % 16;
                int v141;
                v141 = v136 / 16;
                bool v142;
                v142 = v141 < 16;
                bool v143;
                v143 = v142 == false;
                if (v143){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v142);
                } else {
                }
                assert("Tensor range check" && 0 <= v141 && v141 < 16);
                assert("Tensor range check" && 0 <= v140 && v140 < 16);
                int v145;
                v145 = 4 * v140;
                int v146;
                v146 = 68 * v141;
                int v147;
                v147 = v146 + v145;
                int v148;
                v148 = 64 * v141;
                int v149;
                v149 = v148 + v145;
                float * v150;
                v150 = v10+v147;
                float * v152;
                v152 = v101+v149;
                int v154;
                v154 = 0;
                #pragma unroll
                while (while_method_3(v154)){
                    int v156;
                    v156 = 0;
                    #pragma unroll
                    while (while_method_0(v156)){
                        assert("Tensor range check" && 0 <= v154 && v154 < 4);
                        assert("Tensor range check" && 0 <= v156 && v156 < 1);
                        int v158;
                        v158 = 64 * v156;
                        int v159;
                        v159 = 1088 * v154;
                        int v160;
                        v160 = v159 + v158;
                        int v161;
                        v161 = 1024 * v154;
                        int v162;
                        v162 = v161 + v158;
                        int4* v163;
                        v163 = reinterpret_cast<int4*>(v152 + v162);
                        int4* v164;
                        v164 = reinterpret_cast<int4*>(v150 + v160);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v163) % 16 == 0 && reinterpret_cast<unsigned long long>(v164) % 16 == 0);
                        *v164 = *v163;
                        v156 += 1 ;
                    }
                    v154 += 1 ;
                }
                // Poping the loop unrolling to: 0
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v165[1];
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v166[8];
                cuda::pipeline_consumer_wait_prior<0>(v8);;
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v167;
                v167 = 0;
                #pragma unroll
                while (while_method_0(v167)){
                    int v169;
                    v169 = 0;
                    #pragma unroll
                    while (while_method_4(v169)){
                        assert("Tensor range check" && 0 <= v167 && v167 < 1);
                        assert("Tensor range check" && 0 <= v169 && v169 < 8);
                        int v171;
                        v171 = 8 * v167;
                        int v172;
                        v172 = v171 + v169;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v173 = v166[v172];
                        assert("Tensor range check" && 0 <= v167 && v167 < 1);
                        int v174;
                        v174 = 1088 * v167;
                        assert("Tensor range check" && 0 <= v169 && v169 < 8);
                        int v175;
                        v175 = 8 * v169;
                        int v176;
                        v176 = v175 + v174;
                        int v177;
                        v177 = 0;
                        #pragma unroll
                        while (while_method_1(v177)){
                            int v179;
                            v179 = 0;
                            #pragma unroll
                            while (while_method_1(v179)){
                                assert("Tensor range check" && 0 <= v177 && v177 < 2);
                                assert("Tensor range check" && 0 <= v179 && v179 < 2);
                                int v181;
                                v181 = 4 * v179;
                                int v182;
                                v182 = v181 + v176;
                                int v183;
                                v183 = 544 * v177;
                                int v184;
                                v184 = v183 + v182;
                                float v185;
                                v185 = v61[v184];
                                bool v186;
                                v186 = 0 <= v179;
                                bool v188;
                                if (v186){
                                    bool v187;
                                    v187 = v179 < 2;
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
                                bool v191;
                                v191 = 0 <= v177;
                                bool v193;
                                if (v191){
                                    bool v192;
                                    v192 = v177 < 2;
                                    v193 = v192;
                                } else {
                                    v193 = false;
                                }
                                bool v194;
                                v194 = v193 == false;
                                if (v194){
                                    assert("The indices should be inside the range of the dimension." && v193);
                                } else {
                                }
                                int v196;
                                v196 = v177 * 2;
                                int v197;
                                v197 = v179 + v196;
                                v173.x[v197] = wmma::__float_to_tf32(v185);
                                v179 += 1 ;
                            }
                            v177 += 1 ;
                        }
                        v169 += 1 ;
                    }
                    v167 += 1 ;
                }
                // Poping the loop unrolling to: 0
                v8.consumer_release();
                switch (v97.tag) {
                    case 0: { // None
                        break;
                    }
                    case 1: { // Some
                        int v198 = v97.case1.v0;
                        assert("Tensor range check" && 0 <= v198 && v198 < 1);
                        int v199;
                        v199 = 64 * v198;
                        int v200;
                        v200 = v199 + v104;
                        float * v201;
                        v201 = v1+v200;
                        __syncthreads();
                        // Pushing the loop unrolling to: 0
                        v8.producer_acquire();
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
                        v213 = 68 * v208;
                        int v214;
                        v214 = v213 + v212;
                        int v215;
                        v215 = 64 * v208;
                        int v216;
                        v216 = v215 + v212;
                        float * v217;
                        v217 = v12+v214;
                        float * v219;
                        v219 = v201+v216;
                        int v221;
                        v221 = 0;
                        #pragma unroll
                        while (while_method_3(v221)){
                            int v223;
                            v223 = 0;
                            #pragma unroll
                            while (while_method_0(v223)){
                                assert("Tensor range check" && 0 <= v221 && v221 < 4);
                                assert("Tensor range check" && 0 <= v223 && v223 < 1);
                                int v225;
                                v225 = 64 * v223;
                                int v226;
                                v226 = 1088 * v221;
                                int v227;
                                v227 = v226 + v225;
                                int v228;
                                v228 = 1024 * v221;
                                int v229;
                                v229 = v228 + v225;
                                constexpr int v230 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v219 + v229) % v230 == 0 && (unsigned long long)(v217 + v227) % v230 == 0);
                                cuda::memcpy_async(v217 + v227, v219 + v229, cuda::aligned_size_t<v230>(v230), v8);
                                v223 += 1 ;
                            }
                            v221 += 1 ;
                        }
                        v8.producer_commit();
                        // Poping the loop unrolling to: 0
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                // Pushing the loop unrolling to: 0
                int v231;
                v231 = 0;
                #pragma unroll
                while (while_method_1(v231)){
                    int v233;
                    v233 = 0;
                    #pragma unroll
                    while (while_method_4(v233)){
                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v235 = v165[0];
                        assert("Tensor range check" && 0 <= v231 && v231 < 2);
                        int v236;
                        v236 = 1088 * v231;
                        assert("Tensor range check" && 0 <= v233 && v233 < 8);
                        int v237;
                        v237 = 8 * v233;
                        int v238;
                        v238 = v237 + v236;
                        int v239;
                        v239 = 0;
                        #pragma unroll
                        while (while_method_1(v239)){
                            int v241;
                            v241 = 0;
                            #pragma unroll
                            while (while_method_1(v241)){
                                assert("Tensor range check" && 0 <= v239 && v239 < 2);
                                assert("Tensor range check" && 0 <= v241 && v241 < 2);
                                int v243;
                                v243 = 544 * v241;
                                int v244;
                                v244 = v243 + v238;
                                int v245;
                                v245 = 4 * v239;
                                int v246;
                                v246 = v245 + v244;
                                float v247;
                                v247 = v45[v246];
                                bool v248;
                                v248 = 0 <= v241;
                                bool v250;
                                if (v248){
                                    bool v249;
                                    v249 = v241 < 2;
                                    v250 = v249;
                                } else {
                                    v250 = false;
                                }
                                bool v251;
                                v251 = v250 == false;
                                if (v251){
                                    assert("The indices should be inside the range of the dimension." && v250);
                                } else {
                                }
                                bool v253;
                                v253 = 0 <= v239;
                                bool v255;
                                if (v253){
                                    bool v254;
                                    v254 = v239 < 2;
                                    v255 = v254;
                                } else {
                                    v255 = false;
                                }
                                bool v256;
                                v256 = v255 == false;
                                if (v256){
                                    assert("The indices should be inside the range of the dimension." && v255);
                                } else {
                                }
                                int v258;
                                v258 = v239 * 2;
                                int v259;
                                v259 = v241 + v258;
                                v235.x[v259] = wmma::__float_to_tf32(v247);
                                v241 += 1 ;
                            }
                            v239 += 1 ;
                        }
                        int v260;
                        v260 = 0;
                        #pragma unroll
                        while (while_method_0(v260)){
                            assert("Tensor range check" && 0 <= v231 && v231 < 2);
                            assert("Tensor range check" && 0 <= v260 && v260 < 1);
                            int v262;
                            v262 = v231 + v260;
                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v263 = v63[v262];
                            assert("Tensor range check" && 0 <= v260 && v260 < 1);
                            assert("Tensor range check" && 0 <= v233 && v233 < 8);
                            int v264;
                            v264 = 8 * v260;
                            int v265;
                            v265 = v264 + v233;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v266 = v166[v265];
                            wmma::mma_sync(v263, v235, v266, v263);
                            v260 += 1 ;
                        }
                        v233 += 1 ;
                    }
                    v231 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v80 = v82;
            }
            // Pushing the loop unrolling to: 0
            int v267;
            v267 = 0;
            #pragma unroll
            while (while_method_1(v267)){
                int v269;
                v269 = 0;
                #pragma unroll
                while (while_method_0(v269)){
                    assert("Tensor range check" && 0 <= v267 && v267 < 2);
                    assert("Tensor range check" && 0 <= v269 && v269 < 1);
                    int v271;
                    v271 = v267 + v269;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v272 = v63[v271];
                    assert("Tensor range check" && 0 <= v267 && v267 < 2);
                    assert("Tensor range check" && 0 <= v269 && v269 < 1);
                    int v273;
                    v273 = 16 * v269;
                    int v274;
                    v274 = 1152 * v267;
                    int v275;
                    v275 = v274 + v273;
                    float * v276;
                    v276 = v29+v275;
                    wmma::store_matrix_sync(v276, v272, 72, wmma::mem_row_major);
                    v269 += 1 ;
                }
                v267 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            // Pushing the loop unrolling to: 0
            int v278;
            v278 = threadIdx.x;
            bool v279;
            v279 = 0 <= v278;
            bool v280;
            v280 = v279 == false;
            if (v280){
                assert("The index needs to be zero or positive." && v279);
            } else {
            }
            int v282;
            v282 = v278 % 16;
            int v283;
            v283 = v278 / 16;
            bool v284;
            v284 = v283 < 16;
            bool v285;
            v285 = v284 == false;
            if (v285){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v284);
            } else {
            }
            assert("Tensor range check" && 0 <= v283 && v283 < 16);
            assert("Tensor range check" && 0 <= v282 && v282 < 16);
            int v287;
            v287 = 4 * v282;
            int v288;
            v288 = 64 * v283;
            int v289;
            v289 = v288 + v287;
            int v290;
            v290 = 72 * v283;
            int v291;
            v291 = v290 + v287;
            float * v292;
            v292 = v72+v289;
            float * v294;
            v294 = v14+v291;
            int v296;
            v296 = 0;
            #pragma unroll
            while (while_method_3(v296)){
                int v298;
                v298 = 0;
                #pragma unroll
                while (while_method_0(v298)){
                    assert("Tensor range check" && 0 <= v296 && v296 < 4);
                    assert("Tensor range check" && 0 <= v298 && v298 < 1);
                    int v300;
                    v300 = 64 * v298;
                    int v301;
                    v301 = 1024 * v296;
                    int v302;
                    v302 = v301 + v300;
                    int v303;
                    v303 = 1152 * v296;
                    int v304;
                    v304 = v303 + v300;
                    int4* v305;
                    v305 = reinterpret_cast<int4*>(v294 + v304);
                    int4* v306;
                    v306 = reinterpret_cast<int4*>(v292 + v302);
                    assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v305) % 16 == 0 && reinterpret_cast<unsigned long long>(v306) % 16 == 0);
                    *v306 = *v305;
                    v298 += 1 ;
                }
                v296 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            v66 += 1 ;
        }
        v64 += 1 ;
    }
    return ;
}
__device__ void block_row_map_reduce_5(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5){
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 24);
    int v7;
    v7 = 4096 * v6;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 24);
    int v9;
    v9 = 4096 * v8;
    int v10;
    v10 = v9 + v3;
    int v11;
    v11 = blockIdx.x;
    assert("Tensor range check" && 0 <= v11 && v11 < 24);
    int v12;
    v12 = 64 * v11;
    int v13;
    v13 = v12 + v1;
    int v14;
    v14 = threadIdx.x;
    bool v15;
    v15 = 0 <= v14;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("The index needs to be zero or positive." && v15);
    } else {
    }
    int v18;
    v18 = v14 % 16;
    int v19;
    v19 = v14 / 16;
    bool v20;
    v20 = v19 < 16;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v20);
    } else {
    }
    assert("Tensor range check" && 0 <= v19 && v19 < 16);
    assert("Tensor range check" && 0 <= v18 && v18 < 16);
    int v23;
    v23 = 4 * v18;
    int v24;
    v24 = v23 + v7;
    int v25;
    v25 = 64 * v19;
    int v26;
    v26 = v25 + v24;
    assert("Tensor range check" && 0 <= v19 && v19 < 16);
    assert("Tensor range check" && 0 <= v18 && v18 < 16);
    int v27;
    v27 = v23 + v10;
    int v28;
    v28 = v25 + v27;
    assert("Tensor range check" && 0 <= v19 && v19 < 16);
    int v29;
    v29 = v19 + v13;
    int v30;
    v30 = 0;
    while (while_method_3(v30)){
        assert("Tensor range check" && 0 <= v30 && v30 < 4);
        int v32;
        v32 = 1024 * v30;
        int v33;
        v33 = v32 + v26;
        float v34[4];
        int v35[4];
        int v36;
        v36 = 0;
        while (while_method_0(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 1);
            int v38;
            v38 = 4 * v36;
            assert("Tensor range check" && 0 <= v36 && v36 < 1);
            int v39;
            v39 = 64 * v36;
            int v40;
            v40 = v39 + v33;
            int4* v41;
            v41 = reinterpret_cast<int4*>(v4 + v40);
            int4* v42;
            v42 = reinterpret_cast<int4*>(v34 + v38);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v41) % 16 == 0 && reinterpret_cast<unsigned long long>(v42) % 16 == 0);
            *v42 = *v41;
            v36 += 1 ;
        }
        int v43;
        v43 = 0;
        while (while_method_0(v43)){
            int v45;
            v45 = 0;
            while (while_method_3(v45)){
                bool v47;
                v47 = 0 <= v45;
                bool v49;
                if (v47){
                    bool v48;
                    v48 = v45 < 4;
                    v49 = v48;
                } else {
                    v49 = false;
                }
                bool v50;
                v50 = v49 == false;
                if (v50){
                    assert("The indices should be inside the range of the dimension." && v49);
                } else {
                }
                bool v52;
                v52 = 0 <= v18;
                bool v54;
                if (v52){
                    bool v53;
                    v53 = v18 < 16;
                    v54 = v53;
                } else {
                    v54 = false;
                }
                bool v55;
                v55 = v54 == false;
                if (v55){
                    assert("The indices should be inside the range of the dimension." && v54);
                } else {
                }
                int v57;
                v57 = v18 * 4;
                int v58;
                v58 = v45 + v57;
                bool v59;
                v59 = 0 <= v43;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v43 < 1;
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
                v64 = v43 * 64;
                int v65;
                v65 = v58 + v64;
                assert("Tensor range check" && 0 <= v43 && v43 < 1);
                assert("Tensor range check" && 0 <= v45 && v45 < 4);
                int v66;
                v66 = 4 * v43;
                int v67;
                v67 = v66 + v45;
                v35[v67] = v65;
                v45 += 1 ;
            }
            v43 += 1 ;
        }
        bool v68;
        v68 = 0 <= v19;
        bool v69;
        v69 = v68 && v20;
        bool v70;
        v70 = v69 == false;
        if (v70){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v69);
        } else {
        }
        bool v72;
        v72 = 0 <= v30;
        bool v74;
        if (v72){
            bool v73;
            v73 = v30 < 4;
            v74 = v73;
        } else {
            v74 = false;
        }
        bool v75;
        v75 = v74 == false;
        if (v75){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v74);
        } else {
        }
        int v77;
        v77 = v30 * 16;
        int v78;
        v78 = v77 + v19;
        float v79;
        v79 = 0.0f;
        int v80;
        v80 = 0;
        while (while_method_0(v80)){
            int v82;
            v82 = 0;
            while (while_method_3(v82)){
                assert("Tensor range check" && 0 <= v80 && v80 < 1);
                assert("Tensor range check" && 0 <= v82 && v82 < 4);
                int v84;
                v84 = 4 * v80;
                int v85;
                v85 = v84 + v82;
                float v86;
                v86 = v34[v85];
                float v87;
                v87 = v79 + v86;
                v79 = v87;
                v82 += 1 ;
            }
            v80 += 1 ;
        }
        auto v88 = cooperative_groups::coalesced_threads();
        int v89;
        v89 = threadIdx.x;
        int v90;
        v90 = v89 / 16;
        auto v91 = cooperative_groups::labeled_partition(v88,v90);
        Closure0 v92{};
        float v93;
        v93 = cooperative_groups::reduce(v91, v79, v92);
        float v94;
        v94 = v93 / 64.0f;
        float v95[4];
        int v96;
        v96 = 0;
        while (while_method_0(v96)){
            int v98;
            v98 = 0;
            while (while_method_3(v98)){
                assert("Tensor range check" && 0 <= v96 && v96 < 1);
                assert("Tensor range check" && 0 <= v98 && v98 < 4);
                int v100;
                v100 = 4 * v96;
                int v101;
                v101 = v100 + v98;
                float v102;
                v102 = v34[v101];
                float v103;
                v103 = v102 - v94;
                float v104;
                v104 = exp(v103);
                assert("Tensor range check" && 0 <= v96 && v96 < 1);
                assert("Tensor range check" && 0 <= v98 && v98 < 4);
                v95[v101] = v104;
                v98 += 1 ;
            }
            v96 += 1 ;
        }
        float v105;
        v105 = 0.0f;
        int v106;
        v106 = 0;
        while (while_method_0(v106)){
            int v108;
            v108 = 0;
            while (while_method_3(v108)){
                assert("Tensor range check" && 0 <= v106 && v106 < 1);
                assert("Tensor range check" && 0 <= v108 && v108 < 4);
                int v110;
                v110 = 4 * v106;
                int v111;
                v111 = v110 + v108;
                float v112;
                v112 = v95[v111];
                float v113;
                v113 = v105 + v112;
                v105 = v113;
                v108 += 1 ;
            }
            v106 += 1 ;
        }
        auto v114 = cooperative_groups::coalesced_threads();
        int v115;
        v115 = threadIdx.x;
        int v116;
        v116 = v115 / 16;
        auto v117 = cooperative_groups::labeled_partition(v114,v116);
        float v118;
        v118 = cooperative_groups::reduce(v117, v105, v92);
        float v119[4];
        int v120;
        v120 = 0;
        while (while_method_0(v120)){
            int v122;
            v122 = 0;
            while (while_method_3(v122)){
                assert("Tensor range check" && 0 <= v120 && v120 < 1);
                assert("Tensor range check" && 0 <= v122 && v122 < 4);
                int v124;
                v124 = 4 * v120;
                int v125;
                v125 = v124 + v122;
                float v126;
                v126 = v95[v125];
                float v127;
                v127 = v126 / v118;
                assert("Tensor range check" && 0 <= v120 && v120 < 1);
                assert("Tensor range check" && 0 <= v122 && v122 < 4);
                v119[v125] = v127;
                v122 += 1 ;
            }
            v120 += 1 ;
        }
        float v128[4];
        float v129;
        v129 = 0.0f;
        int v130;
        v130 = 0;
        while (while_method_0(v130)){
            assert("Tensor range check" && 0 <= v130 && v130 < 1);
            int v132;
            v132 = 4 * v130;
            assert("Tensor range check" && 0 <= v130 && v130 < 1);
            int v133; float v134;
            Tuple0 tmp8 = Tuple0{0, 0.0f};
            v133 = tmp8.v0; v134 = tmp8.v1;
            while (while_method_3(v133)){
                assert("Tensor range check" && 0 <= v133 && v133 < 4);
                int v136;
                v136 = v133 + v132;
                float v137;
                v137 = v119[v136];
                float v138;
                v138 = v134 + v137;
                v134 = v138;
                v133 += 1 ;
            }
            auto v139 = cooperative_groups::coalesced_threads();
            int v140;
            v140 = threadIdx.x;
            int v141;
            v141 = v140 / 16;
            auto v142 = cooperative_groups::labeled_partition(v139,v141);
            Closure1 v143{};
            float v144;
            v144 = cooperative_groups::inclusive_scan(v142, v134, v143);
            float v145;
            v145 = v142.shfl_up(v144,1);
            bool v146;
            v146 = v142.thread_rank() == 0;
            float v147;
            if (v146){
                v147 = 0.0f;
            } else {
                v147 = v145;
            }
            float v148;
            v148 = v142.shfl(v144,v142.num_threads()-1);
            float v149;
            v149 = v129 + v147;
            int v150; float v151;
            Tuple0 tmp9 = Tuple0{0, v149};
            v150 = tmp9.v0; v151 = tmp9.v1;
            while (while_method_3(v150)){
                assert("Tensor range check" && 0 <= v150 && v150 < 4);
                int v153;
                v153 = v150 + v132;
                float v154;
                v154 = v119[v153];
                float v155;
                v155 = v151 + v154;
                assert("Tensor range check" && 0 <= v150 && v150 < 4);
                v128[v153] = v155;
                v151 = v155;
                v150 += 1 ;
            }
            float v156;
            v156 = v129 + v148;
            v129 = v156;
            v130 += 1 ;
        }
        float v157[4];
        bool v158[4];
        int v159;
        v159 = 0;
        while (while_method_0(v159)){
            int v161;
            v161 = 0;
            while (while_method_3(v161)){
                assert("Tensor range check" && 0 <= v159 && v159 < 1);
                assert("Tensor range check" && 0 <= v161 && v161 < 4);
                int v163;
                v163 = 4 * v159;
                int v164;
                v164 = v163 + v161;
                float v165;
                v165 = v128[v164];
                float v166;
                v166 = v119[v164];
                bool v167;
                v167 = v166 > 0.0f;
                assert("Tensor range check" && 0 <= v159 && v159 < 1);
                assert("Tensor range check" && 0 <= v161 && v161 < 4);
                v157[v164] = v165;
                v158[v164] = v167;
                v161 += 1 ;
            }
            v159 += 1 ;
        }
        float v168; bool v169;
        Tuple1 tmp10 = Tuple1{-1.0f / 0.0f, false};
        v168 = tmp10.v0; v169 = tmp10.v1;
        int v170;
        v170 = 0;
        while (while_method_0(v170)){
            int v172;
            v172 = 0;
            while (while_method_3(v172)){
                assert("Tensor range check" && 0 <= v170 && v170 < 1);
                assert("Tensor range check" && 0 <= v172 && v172 < 4);
                int v174;
                v174 = 4 * v170;
                int v175;
                v175 = v174 + v172;
                float v176;
                v176 = v157[v175];
                bool v177;
                v177 = v158[v175];
                float v184; bool v185;
                if (v169){
                    if (v177){
                        bool v178;
                        v178 = v168 >= v176;
                        float v179;
                        if (v178){
                            v179 = v168;
                        } else {
                            v179 = v176;
                        }
                        v184 = v179; v185 = true;
                    } else {
                        v184 = v168; v185 = v169;
                    }
                } else {
                    if (v177){
                        v184 = v176; v185 = v177;
                    } else {
                        v184 = v168; v185 = v169;
                    }
                }
                v168 = v184;
                v169 = v185;
                v172 += 1 ;
            }
            v170 += 1 ;
        }
        auto v186 = cooperative_groups::coalesced_threads();
        int v187;
        v187 = threadIdx.x;
        int v188;
        v188 = v187 / 16;
        auto v189 = cooperative_groups::labeled_partition(v186,v188);
        Closure2 v190{};
        float v191; bool v192;
        Tuple1 tmp11 = cooperative_groups::reduce(v189, Tuple1{v168, v169}, v190);
        v191 = tmp11.v0; v192 = tmp11.v1;
        bool v193;
        v193 = v192 == false;
        if (v193){
            assert("The local reduce must be true." && v192);
        } else {
        }
        float v195[4];
        int v196[4];
        int v197;
        v197 = 0;
        while (while_method_0(v197)){
            int v199;
            v199 = 0;
            while (while_method_3(v199)){
                assert("Tensor range check" && 0 <= v197 && v197 < 1);
                assert("Tensor range check" && 0 <= v199 && v199 < 4);
                int v201;
                v201 = 4 * v197;
                int v202;
                v202 = v201 + v199;
                int v203;
                v203 = v35[v202];
                float v204;
                v204 = curand_uniform(&v5);
                assert("Tensor range check" && 0 <= v197 && v197 < 1);
                assert("Tensor range check" && 0 <= v199 && v199 < 4);
                v195[v202] = v204;
                v196[v202] = v203;
                v199 += 1 ;
            }
            v197 += 1 ;
        }
        float v205; int v206;
        Tuple2 tmp12 = Tuple2{0.0f, 2147483647};
        v205 = tmp12.v0; v206 = tmp12.v1;
        int v207;
        v207 = 0;
        while (while_method_0(v207)){
            int v209;
            v209 = 0;
            while (while_method_3(v209)){
                assert("Tensor range check" && 0 <= v207 && v207 < 1);
                assert("Tensor range check" && 0 <= v209 && v209 < 4);
                int v211;
                v211 = 4 * v207;
                int v212;
                v212 = v211 + v209;
                float v213;
                v213 = v195[v212];
                int v214;
                v214 = v196[v212];
                bool v215;
                v215 = v206 < v214;
                float v216; int v217;
                if (v215){
                    v216 = v205; v217 = v206;
                } else {
                    v216 = v213; v217 = v214;
                }
                v205 = v216;
                v206 = v217;
                v209 += 1 ;
            }
            v207 += 1 ;
        }
        auto v218 = cooperative_groups::coalesced_threads();
        int v219;
        v219 = threadIdx.x;
        int v220;
        v220 = v219 / 16;
        auto v221 = cooperative_groups::labeled_partition(v218,v220);
        Closure3 v222{};
        float v223; int v224;
        Tuple2 tmp13 = cooperative_groups::reduce(v221, Tuple2{v205, v206}, v222);
        v223 = tmp13.v0; v224 = tmp13.v1;
        float v225;
        v225 = v191 * v223;
        int v226[4];
        bool v227[4];
        int v228;
        v228 = 0;
        while (while_method_0(v228)){
            int v230;
            v230 = 0;
            while (while_method_3(v230)){
                assert("Tensor range check" && 0 <= v228 && v228 < 1);
                assert("Tensor range check" && 0 <= v230 && v230 < 4);
                int v232;
                v232 = 4 * v228;
                int v233;
                v233 = v232 + v230;
                float v234;
                v234 = v157[v233];
                bool v235;
                v235 = v158[v233];
                int v236;
                v236 = v35[v233];
                int v239; bool v240;
                if (v235){
                    float v237;
                    v237 = v234 - v225;
                    bool v238;
                    v238 = v237 >= 0.0f;
                    v239 = v236; v240 = v238;
                } else {
                    v239 = 2147483647; v240 = false;
                }
                assert("Tensor range check" && 0 <= v228 && v228 < 1);
                assert("Tensor range check" && 0 <= v230 && v230 < 4);
                v226[v233] = v239;
                v227[v233] = v240;
                v230 += 1 ;
            }
            v228 += 1 ;
        }
        int v241; bool v242;
        Tuple3 tmp14 = Tuple3{2147483647, false};
        v241 = tmp14.v0; v242 = tmp14.v1;
        int v243;
        v243 = 0;
        while (while_method_0(v243)){
            int v245;
            v245 = 0;
            while (while_method_3(v245)){
                assert("Tensor range check" && 0 <= v243 && v243 < 1);
                assert("Tensor range check" && 0 <= v245 && v245 < 4);
                int v247;
                v247 = 4 * v243;
                int v248;
                v248 = v247 + v245;
                int v249;
                v249 = v226[v248];
                bool v250;
                v250 = v227[v248];
                int v257; bool v258;
                if (v242){
                    if (v250){
                        bool v251;
                        v251 = v241 < v249;
                        int v252;
                        if (v251){
                            v252 = v241;
                        } else {
                            v252 = v249;
                        }
                        v257 = v252; v258 = true;
                    } else {
                        v257 = v241; v258 = v242;
                    }
                } else {
                    if (v250){
                        v257 = v249; v258 = v250;
                    } else {
                        v257 = v241; v258 = v242;
                    }
                }
                v241 = v257;
                v242 = v258;
                v245 += 1 ;
            }
            v243 += 1 ;
        }
        auto v259 = cooperative_groups::coalesced_threads();
        int v260;
        v260 = threadIdx.x;
        int v261;
        v261 = v260 / 16;
        auto v262 = cooperative_groups::labeled_partition(v259,v261);
        Closure4 v263{};
        int v264; bool v265;
        Tuple3 tmp15 = cooperative_groups::reduce(v262, Tuple3{v241, v242}, v263);
        v264 = tmp15.v0; v265 = tmp15.v1;
        bool v266;
        v266 = v265 == false;
        if (v266){
            assert("The local reduce must be true." && v265);
        } else {
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 4);
        int v268;
        v268 = v32 + v28;
        int v269;
        v269 = 0;
        while (while_method_0(v269)){
            assert("Tensor range check" && 0 <= v269 && v269 < 1);
            int v271;
            v271 = 64 * v269;
            int v272;
            v272 = v271 + v268;
            assert("Tensor range check" && 0 <= v269 && v269 < 1);
            int v273;
            v273 = 4 * v269;
            int4* v274;
            v274 = reinterpret_cast<int4*>(v119 + v273);
            int4* v275;
            v275 = reinterpret_cast<int4*>(v2 + v272);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v274) % 16 == 0 && reinterpret_cast<unsigned long long>(v275) % 16 == 0);
            *v275 = *v274;
            v269 += 1 ;
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 4);
        int v276;
        v276 = 16 * v30;
        int v277;
        v277 = v276 + v29;
        v0[v277] = v264;
        v30 += 1 ;
    }
    __syncthreads();
    return ;
}
__device__ void block_row_map_reduce_6(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5){
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 24);
    int v7;
    v7 = 4096 * v6;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 24);
    int v9;
    v9 = 4096 * v8;
    int v10;
    v10 = v9 + v3;
    int v11;
    v11 = blockIdx.x;
    assert("Tensor range check" && 0 <= v11 && v11 < 24);
    int v12;
    v12 = 64 * v11;
    int v13;
    v13 = v12 + v1;
    int v14;
    v14 = threadIdx.x;
    bool v15;
    v15 = 0 <= v14;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("The index needs to be zero or positive." && v15);
    } else {
    }
    int v18;
    v18 = v14 % 16;
    int v19;
    v19 = v14 / 16;
    bool v20;
    v20 = v19 < 16;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v20);
    } else {
    }
    assert("Tensor range check" && 0 <= v19 && v19 < 16);
    assert("Tensor range check" && 0 <= v18 && v18 < 16);
    int v23;
    v23 = 4 * v18;
    int v24;
    v24 = v23 + v7;
    int v25;
    v25 = 64 * v19;
    int v26;
    v26 = v25 + v24;
    assert("Tensor range check" && 0 <= v19 && v19 < 16);
    assert("Tensor range check" && 0 <= v18 && v18 < 16);
    int v27;
    v27 = v23 + v10;
    int v28;
    v28 = v25 + v27;
    assert("Tensor range check" && 0 <= v19 && v19 < 16);
    int v29;
    v29 = v19 + v13;
    int v30;
    v30 = 0;
    while (while_method_3(v30)){
        assert("Tensor range check" && 0 <= v30 && v30 < 4);
        int v32;
        v32 = 1024 * v30;
        int v33;
        v33 = v32 + v26;
        float v34[4];
        int v35[4];
        int v36;
        v36 = 0;
        while (while_method_0(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 1);
            int v38;
            v38 = 4 * v36;
            assert("Tensor range check" && 0 <= v36 && v36 < 1);
            int v39;
            v39 = 64 * v36;
            int v40;
            v40 = v39 + v33;
            int4* v41;
            v41 = reinterpret_cast<int4*>(v4 + v40);
            int4* v42;
            v42 = reinterpret_cast<int4*>(v34 + v38);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v41) % 16 == 0 && reinterpret_cast<unsigned long long>(v42) % 16 == 0);
            *v42 = *v41;
            v36 += 1 ;
        }
        int v43;
        v43 = 0;
        while (while_method_0(v43)){
            int v45;
            v45 = 0;
            while (while_method_3(v45)){
                bool v47;
                v47 = 0 <= v45;
                bool v49;
                if (v47){
                    bool v48;
                    v48 = v45 < 4;
                    v49 = v48;
                } else {
                    v49 = false;
                }
                bool v50;
                v50 = v49 == false;
                if (v50){
                    assert("The indices should be inside the range of the dimension." && v49);
                } else {
                }
                bool v52;
                v52 = 0 <= v18;
                bool v54;
                if (v52){
                    bool v53;
                    v53 = v18 < 16;
                    v54 = v53;
                } else {
                    v54 = false;
                }
                bool v55;
                v55 = v54 == false;
                if (v55){
                    assert("The indices should be inside the range of the dimension." && v54);
                } else {
                }
                int v57;
                v57 = v18 * 4;
                int v58;
                v58 = v45 + v57;
                bool v59;
                v59 = 0 <= v43;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v43 < 1;
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
                v64 = v43 * 64;
                int v65;
                v65 = v58 + v64;
                assert("Tensor range check" && 0 <= v43 && v43 < 1);
                assert("Tensor range check" && 0 <= v45 && v45 < 4);
                int v66;
                v66 = 4 * v43;
                int v67;
                v67 = v66 + v45;
                v35[v67] = v65;
                v45 += 1 ;
            }
            v43 += 1 ;
        }
        bool v68;
        v68 = 0 <= v19;
        bool v69;
        v69 = v68 && v20;
        bool v70;
        v70 = v69 == false;
        if (v70){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v69);
        } else {
        }
        bool v72;
        v72 = 0 <= v30;
        bool v74;
        if (v72){
            bool v73;
            v73 = v30 < 4;
            v74 = v73;
        } else {
            v74 = false;
        }
        bool v75;
        v75 = v74 == false;
        if (v75){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v74);
        } else {
        }
        int v77;
        v77 = v30 * 16;
        int v78;
        v78 = v77 + v19;
        bool v79[4];
        int v80;
        v80 = 0;
        while (while_method_0(v80)){
            int v82;
            v82 = 0;
            while (while_method_3(v82)){
                assert("Tensor range check" && 0 <= v80 && v80 < 1);
                assert("Tensor range check" && 0 <= v82 && v82 < 4);
                int v84;
                v84 = 4 * v80;
                int v85;
                v85 = v84 + v82;
                float v86;
                v86 = v34[v85];
                int v87;
                v87 = v35[v85];
                bool v88;
                v88 = v87 < 11;
                assert("Tensor range check" && 0 <= v80 && v80 < 1);
                assert("Tensor range check" && 0 <= v82 && v82 < 4);
                v79[v85] = v88;
                v82 += 1 ;
            }
            v80 += 1 ;
        }
        float v89[4];
        int v90;
        v90 = 0;
        while (while_method_0(v90)){
            int v92;
            v92 = 0;
            while (while_method_3(v92)){
                assert("Tensor range check" && 0 <= v90 && v90 < 1);
                assert("Tensor range check" && 0 <= v92 && v92 < 4);
                int v94;
                v94 = 4 * v90;
                int v95;
                v95 = v94 + v92;
                float v96;
                v96 = v34[v95];
                bool v97;
                v97 = v79[v95];
                float v98;
                if (v97){
                    v98 = v96;
                } else {
                    v98 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v90 && v90 < 1);
                assert("Tensor range check" && 0 <= v92 && v92 < 4);
                v89[v95] = v98;
                v92 += 1 ;
            }
            v90 += 1 ;
        }
        float v99;
        v99 = 0.0f;
        int v100;
        v100 = 0;
        while (while_method_0(v100)){
            int v102;
            v102 = 0;
            while (while_method_3(v102)){
                assert("Tensor range check" && 0 <= v100 && v100 < 1);
                assert("Tensor range check" && 0 <= v102 && v102 < 4);
                int v104;
                v104 = 4 * v100;
                int v105;
                v105 = v104 + v102;
                float v106;
                v106 = v89[v105];
                float v107;
                v107 = v99 + v106;
                v99 = v107;
                v102 += 1 ;
            }
            v100 += 1 ;
        }
        auto v108 = cooperative_groups::coalesced_threads();
        int v109;
        v109 = threadIdx.x;
        int v110;
        v110 = v109 / 16;
        auto v111 = cooperative_groups::labeled_partition(v108,v110);
        Closure0 v112{};
        float v113;
        v113 = cooperative_groups::reduce(v111, v99, v112);
        int v114[4];
        int v115;
        v115 = 0;
        while (while_method_0(v115)){
            int v117;
            v117 = 0;
            while (while_method_3(v117)){
                assert("Tensor range check" && 0 <= v115 && v115 < 1);
                assert("Tensor range check" && 0 <= v117 && v117 < 4);
                int v119;
                v119 = 4 * v115;
                int v120;
                v120 = v119 + v117;
                bool v121;
                v121 = v79[v120];
                int v122;
                if (v121){
                    v122 = 1;
                } else {
                    v122 = 0;
                }
                assert("Tensor range check" && 0 <= v115 && v115 < 1);
                assert("Tensor range check" && 0 <= v117 && v117 < 4);
                v114[v120] = v122;
                v117 += 1 ;
            }
            v115 += 1 ;
        }
        int v123;
        v123 = 0;
        int v124;
        v124 = 0;
        while (while_method_0(v124)){
            int v126;
            v126 = 0;
            while (while_method_3(v126)){
                assert("Tensor range check" && 0 <= v124 && v124 < 1);
                assert("Tensor range check" && 0 <= v126 && v126 < 4);
                int v128;
                v128 = 4 * v124;
                int v129;
                v129 = v128 + v126;
                int v130;
                v130 = v114[v129];
                int v131;
                v131 = v123 + v130;
                v123 = v131;
                v126 += 1 ;
            }
            v124 += 1 ;
        }
        auto v132 = cooperative_groups::coalesced_threads();
        int v133;
        v133 = threadIdx.x;
        int v134;
        v134 = v133 / 16;
        auto v135 = cooperative_groups::labeled_partition(v132,v134);
        Closure5 v136{};
        int v137;
        v137 = cooperative_groups::reduce(v135, v123, v136);
        float v138;
        v138 = (float)v137;
        float v139;
        v139 = v113 / v138;
        float v140[4];
        int v141;
        v141 = 0;
        while (while_method_0(v141)){
            int v143;
            v143 = 0;
            while (while_method_3(v143)){
                assert("Tensor range check" && 0 <= v141 && v141 < 1);
                assert("Tensor range check" && 0 <= v143 && v143 < 4);
                int v145;
                v145 = 4 * v141;
                int v146;
                v146 = v145 + v143;
                float v147;
                v147 = v34[v146];
                bool v148;
                v148 = v79[v146];
                float v149;
                if (v148){
                    v149 = v147;
                } else {
                    v149 = -1.0f / 0.0f;
                }
                float v150;
                v150 = v149 - v139;
                float v151;
                v151 = exp(v150);
                assert("Tensor range check" && 0 <= v141 && v141 < 1);
                assert("Tensor range check" && 0 <= v143 && v143 < 4);
                v140[v146] = v151;
                v143 += 1 ;
            }
            v141 += 1 ;
        }
        float v152;
        v152 = 0.0f;
        int v153;
        v153 = 0;
        while (while_method_0(v153)){
            int v155;
            v155 = 0;
            while (while_method_3(v155)){
                assert("Tensor range check" && 0 <= v153 && v153 < 1);
                assert("Tensor range check" && 0 <= v155 && v155 < 4);
                int v157;
                v157 = 4 * v153;
                int v158;
                v158 = v157 + v155;
                float v159;
                v159 = v140[v158];
                float v160;
                v160 = v152 + v159;
                v152 = v160;
                v155 += 1 ;
            }
            v153 += 1 ;
        }
        auto v161 = cooperative_groups::coalesced_threads();
        int v162;
        v162 = threadIdx.x;
        int v163;
        v163 = v162 / 16;
        auto v164 = cooperative_groups::labeled_partition(v161,v163);
        float v165;
        v165 = cooperative_groups::reduce(v164, v152, v112);
        float v166[4];
        int v167;
        v167 = 0;
        while (while_method_0(v167)){
            int v169;
            v169 = 0;
            while (while_method_3(v169)){
                assert("Tensor range check" && 0 <= v167 && v167 < 1);
                assert("Tensor range check" && 0 <= v169 && v169 < 4);
                int v171;
                v171 = 4 * v167;
                int v172;
                v172 = v171 + v169;
                float v173;
                v173 = v140[v172];
                float v174;
                v174 = v173 / v165;
                assert("Tensor range check" && 0 <= v167 && v167 < 1);
                assert("Tensor range check" && 0 <= v169 && v169 < 4);
                v166[v172] = v174;
                v169 += 1 ;
            }
            v167 += 1 ;
        }
        float v175[4];
        float v176;
        v176 = 0.0f;
        int v177;
        v177 = 0;
        while (while_method_0(v177)){
            assert("Tensor range check" && 0 <= v177 && v177 < 1);
            int v179;
            v179 = 4 * v177;
            assert("Tensor range check" && 0 <= v177 && v177 < 1);
            int v180; float v181;
            Tuple0 tmp16 = Tuple0{0, 0.0f};
            v180 = tmp16.v0; v181 = tmp16.v1;
            while (while_method_3(v180)){
                assert("Tensor range check" && 0 <= v180 && v180 < 4);
                int v183;
                v183 = v180 + v179;
                float v184;
                v184 = v166[v183];
                float v185;
                v185 = v181 + v184;
                v181 = v185;
                v180 += 1 ;
            }
            auto v186 = cooperative_groups::coalesced_threads();
            int v187;
            v187 = threadIdx.x;
            int v188;
            v188 = v187 / 16;
            auto v189 = cooperative_groups::labeled_partition(v186,v188);
            Closure1 v190{};
            float v191;
            v191 = cooperative_groups::inclusive_scan(v189, v181, v190);
            float v192;
            v192 = v189.shfl_up(v191,1);
            bool v193;
            v193 = v189.thread_rank() == 0;
            float v194;
            if (v193){
                v194 = 0.0f;
            } else {
                v194 = v192;
            }
            float v195;
            v195 = v189.shfl(v191,v189.num_threads()-1);
            float v196;
            v196 = v176 + v194;
            int v197; float v198;
            Tuple0 tmp17 = Tuple0{0, v196};
            v197 = tmp17.v0; v198 = tmp17.v1;
            while (while_method_3(v197)){
                assert("Tensor range check" && 0 <= v197 && v197 < 4);
                int v200;
                v200 = v197 + v179;
                float v201;
                v201 = v166[v200];
                float v202;
                v202 = v198 + v201;
                assert("Tensor range check" && 0 <= v197 && v197 < 4);
                v175[v200] = v202;
                v198 = v202;
                v197 += 1 ;
            }
            float v203;
            v203 = v176 + v195;
            v176 = v203;
            v177 += 1 ;
        }
        float v204[4];
        bool v205[4];
        int v206;
        v206 = 0;
        while (while_method_0(v206)){
            int v208;
            v208 = 0;
            while (while_method_3(v208)){
                assert("Tensor range check" && 0 <= v206 && v206 < 1);
                assert("Tensor range check" && 0 <= v208 && v208 < 4);
                int v210;
                v210 = 4 * v206;
                int v211;
                v211 = v210 + v208;
                float v212;
                v212 = v175[v211];
                float v213;
                v213 = v166[v211];
                bool v214;
                v214 = v213 > 0.0f;
                assert("Tensor range check" && 0 <= v206 && v206 < 1);
                assert("Tensor range check" && 0 <= v208 && v208 < 4);
                v204[v211] = v212;
                v205[v211] = v214;
                v208 += 1 ;
            }
            v206 += 1 ;
        }
        float v215; bool v216;
        Tuple1 tmp18 = Tuple1{-1.0f / 0.0f, false};
        v215 = tmp18.v0; v216 = tmp18.v1;
        int v217;
        v217 = 0;
        while (while_method_0(v217)){
            int v219;
            v219 = 0;
            while (while_method_3(v219)){
                assert("Tensor range check" && 0 <= v217 && v217 < 1);
                assert("Tensor range check" && 0 <= v219 && v219 < 4);
                int v221;
                v221 = 4 * v217;
                int v222;
                v222 = v221 + v219;
                float v223;
                v223 = v204[v222];
                bool v224;
                v224 = v205[v222];
                float v231; bool v232;
                if (v216){
                    if (v224){
                        bool v225;
                        v225 = v215 >= v223;
                        float v226;
                        if (v225){
                            v226 = v215;
                        } else {
                            v226 = v223;
                        }
                        v231 = v226; v232 = true;
                    } else {
                        v231 = v215; v232 = v216;
                    }
                } else {
                    if (v224){
                        v231 = v223; v232 = v224;
                    } else {
                        v231 = v215; v232 = v216;
                    }
                }
                v215 = v231;
                v216 = v232;
                v219 += 1 ;
            }
            v217 += 1 ;
        }
        auto v233 = cooperative_groups::coalesced_threads();
        int v234;
        v234 = threadIdx.x;
        int v235;
        v235 = v234 / 16;
        auto v236 = cooperative_groups::labeled_partition(v233,v235);
        Closure2 v237{};
        float v238; bool v239;
        Tuple1 tmp19 = cooperative_groups::reduce(v236, Tuple1{v215, v216}, v237);
        v238 = tmp19.v0; v239 = tmp19.v1;
        bool v240;
        v240 = v239 == false;
        if (v240){
            assert("The local reduce must be true." && v239);
        } else {
        }
        float v242[4];
        int v243[4];
        int v244;
        v244 = 0;
        while (while_method_0(v244)){
            int v246;
            v246 = 0;
            while (while_method_3(v246)){
                assert("Tensor range check" && 0 <= v244 && v244 < 1);
                assert("Tensor range check" && 0 <= v246 && v246 < 4);
                int v248;
                v248 = 4 * v244;
                int v249;
                v249 = v248 + v246;
                int v250;
                v250 = v35[v249];
                float v251;
                v251 = curand_uniform(&v5);
                assert("Tensor range check" && 0 <= v244 && v244 < 1);
                assert("Tensor range check" && 0 <= v246 && v246 < 4);
                v242[v249] = v251;
                v243[v249] = v250;
                v246 += 1 ;
            }
            v244 += 1 ;
        }
        float v252; int v253;
        Tuple2 tmp20 = Tuple2{0.0f, 2147483647};
        v252 = tmp20.v0; v253 = tmp20.v1;
        int v254;
        v254 = 0;
        while (while_method_0(v254)){
            int v256;
            v256 = 0;
            while (while_method_3(v256)){
                assert("Tensor range check" && 0 <= v254 && v254 < 1);
                assert("Tensor range check" && 0 <= v256 && v256 < 4);
                int v258;
                v258 = 4 * v254;
                int v259;
                v259 = v258 + v256;
                float v260;
                v260 = v242[v259];
                int v261;
                v261 = v243[v259];
                bool v262;
                v262 = v253 < v261;
                float v263; int v264;
                if (v262){
                    v263 = v252; v264 = v253;
                } else {
                    v263 = v260; v264 = v261;
                }
                v252 = v263;
                v253 = v264;
                v256 += 1 ;
            }
            v254 += 1 ;
        }
        auto v265 = cooperative_groups::coalesced_threads();
        int v266;
        v266 = threadIdx.x;
        int v267;
        v267 = v266 / 16;
        auto v268 = cooperative_groups::labeled_partition(v265,v267);
        Closure3 v269{};
        float v270; int v271;
        Tuple2 tmp21 = cooperative_groups::reduce(v268, Tuple2{v252, v253}, v269);
        v270 = tmp21.v0; v271 = tmp21.v1;
        float v272;
        v272 = v238 * v270;
        int v273[4];
        bool v274[4];
        int v275;
        v275 = 0;
        while (while_method_0(v275)){
            int v277;
            v277 = 0;
            while (while_method_3(v277)){
                assert("Tensor range check" && 0 <= v275 && v275 < 1);
                assert("Tensor range check" && 0 <= v277 && v277 < 4);
                int v279;
                v279 = 4 * v275;
                int v280;
                v280 = v279 + v277;
                float v281;
                v281 = v204[v280];
                bool v282;
                v282 = v205[v280];
                int v283;
                v283 = v35[v280];
                int v286; bool v287;
                if (v282){
                    float v284;
                    v284 = v281 - v272;
                    bool v285;
                    v285 = v284 >= 0.0f;
                    v286 = v283; v287 = v285;
                } else {
                    v286 = 2147483647; v287 = false;
                }
                assert("Tensor range check" && 0 <= v275 && v275 < 1);
                assert("Tensor range check" && 0 <= v277 && v277 < 4);
                v273[v280] = v286;
                v274[v280] = v287;
                v277 += 1 ;
            }
            v275 += 1 ;
        }
        int v288; bool v289;
        Tuple3 tmp22 = Tuple3{2147483647, false};
        v288 = tmp22.v0; v289 = tmp22.v1;
        int v290;
        v290 = 0;
        while (while_method_0(v290)){
            int v292;
            v292 = 0;
            while (while_method_3(v292)){
                assert("Tensor range check" && 0 <= v290 && v290 < 1);
                assert("Tensor range check" && 0 <= v292 && v292 < 4);
                int v294;
                v294 = 4 * v290;
                int v295;
                v295 = v294 + v292;
                int v296;
                v296 = v273[v295];
                bool v297;
                v297 = v274[v295];
                int v304; bool v305;
                if (v289){
                    if (v297){
                        bool v298;
                        v298 = v288 < v296;
                        int v299;
                        if (v298){
                            v299 = v288;
                        } else {
                            v299 = v296;
                        }
                        v304 = v299; v305 = true;
                    } else {
                        v304 = v288; v305 = v289;
                    }
                } else {
                    if (v297){
                        v304 = v296; v305 = v297;
                    } else {
                        v304 = v288; v305 = v289;
                    }
                }
                v288 = v304;
                v289 = v305;
                v292 += 1 ;
            }
            v290 += 1 ;
        }
        auto v306 = cooperative_groups::coalesced_threads();
        int v307;
        v307 = threadIdx.x;
        int v308;
        v308 = v307 / 16;
        auto v309 = cooperative_groups::labeled_partition(v306,v308);
        Closure4 v310{};
        int v311; bool v312;
        Tuple3 tmp23 = cooperative_groups::reduce(v309, Tuple3{v288, v289}, v310);
        v311 = tmp23.v0; v312 = tmp23.v1;
        bool v313;
        v313 = v312 == false;
        if (v313){
            assert("The local reduce must be true." && v312);
        } else {
        }
        bool v315;
        v315 = v311 < 11;
        bool v316;
        v316 = v315 == false;
        if (v316){
            assert("The masking requirement is violated in masked_softmax_and_discrete_sample_." && v315);
        } else {
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 4);
        int v318;
        v318 = v32 + v28;
        int v319;
        v319 = 0;
        while (while_method_0(v319)){
            assert("Tensor range check" && 0 <= v319 && v319 < 1);
            int v321;
            v321 = 64 * v319;
            int v322;
            v322 = v321 + v318;
            assert("Tensor range check" && 0 <= v319 && v319 < 1);
            int v323;
            v323 = 4 * v319;
            int4* v324;
            v324 = reinterpret_cast<int4*>(v166 + v323);
            int4* v325;
            v325 = reinterpret_cast<int4*>(v2 + v322);
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v324) % 16 == 0 && reinterpret_cast<unsigned long long>(v325) % 16 == 0);
            *v325 = *v324;
            v319 += 1 ;
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 4);
        int v326;
        v326 = 16 * v30;
        int v327;
        v327 = v326 + v29;
        v0[v327] = v311;
        v30 += 1 ;
    }
    __syncthreads();
    return ;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned char * v2) {
    int v3;
    v3 = threadIdx.x;
    int v4;
    v4 = blockIdx.x;
    int v5;
    v5 = v4 * 256;
    int v6;
    v6 = v3 + v5;
    unsigned long long v7;
    v7 = (unsigned long long)v6;
    curandStatePhilox4_32_10_t v8;
    curand_init(12344321ull,v7,0ull,&v8);
    float * v9;
    v9 = reinterpret_cast<float *>(&v1[3145728ull]);
    int * v11;
    v11 = reinterpret_cast<int *>(&v1[3538944ull]);
    float * v13;
    v13 = reinterpret_cast<float *>(&v1[2359296ull]);
    float * v15;
    v15 = reinterpret_cast<float *>(&v1[1966080ull]);
    float * v17;
    v17 = reinterpret_cast<float *>(&v1[1179648ull]);
    float * v19;
    v19 = reinterpret_cast<float *>(&v1[786432ull]);
    float * v21;
    v21 = reinterpret_cast<float *>(&v1[0ull]);
    float * v23;
    v23 = reinterpret_cast<float *>(&v0[0ull]);
    float * v25;
    v25 = reinterpret_cast<float *>(&v1[393216ull]);
    block_matmul_0(v25, v23, v21);
    block_row_map_1(v19, v25);
    block_map_2(v17, v19);
    float * v27;
    v27 = reinterpret_cast<float *>(&v0[16384ull]);
    float * v29;
    v29 = reinterpret_cast<float *>(&v1[1572864ull]);
    block_matmul_0(v29, v27, v17);
    block_row_map_1(v15, v29);
    block_map_2(v13, v15);
    float * v31;
    v31 = reinterpret_cast<float *>(&v0[32768ull]);
    float * v33;
    v33 = reinterpret_cast<float *>(&v1[2752512ull]);
    block_matmul_0(v33, v31, v13);
    return block_row_map_reduce_3(v11, v9, v33, v8);
}
extern "C" __global__ void entry1(unsigned char * v0, unsigned char * v1, unsigned char * v2) {
    int v3;
    v3 = threadIdx.x;
    int v4;
    v4 = blockIdx.x;
    int v5;
    v5 = v4 * 256;
    int v6;
    v6 = v3 + v5;
    unsigned long long v7;
    v7 = (unsigned long long)v6;
    curandStatePhilox4_32_10_t v8;
    curand_init(12344321ull,v7,0ull,&v8);
    int v9;
    v9 = 0;
    while (while_method_6(v9)){
        float * v11;
        v11 = reinterpret_cast<float *>(&v1[3145728ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16);
        int v13;
        v13 = 98304 * v9;
        int * v14;
        v14 = reinterpret_cast<int *>(&v1[9437184ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16);
        int v16;
        v16 = 1536 * v9;
        float * v17;
        v17 = reinterpret_cast<float *>(&v1[2359296ull]);
        float * v19;
        v19 = reinterpret_cast<float *>(&v1[1966080ull]);
        float * v21;
        v21 = reinterpret_cast<float *>(&v1[1179648ull]);
        float * v23;
        v23 = reinterpret_cast<float *>(&v1[786432ull]);
        float * v25;
        v25 = reinterpret_cast<float *>(&v1[0ull]);
        float * v27;
        v27 = reinterpret_cast<float *>(&v0[0ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16);
        int v29;
        v29 = 4096 * v9;
        float * v30;
        v30 = reinterpret_cast<float *>(&v1[393216ull]);
        block_matmul_4(v30, v27, v29, v25);
        block_row_map_1(v23, v30);
        block_map_2(v21, v23);
        float * v32;
        v32 = reinterpret_cast<float *>(&v0[262144ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16);
        float * v34;
        v34 = reinterpret_cast<float *>(&v1[1572864ull]);
        block_matmul_4(v34, v32, v29, v21);
        block_row_map_1(v19, v34);
        block_map_2(v17, v19);
        float * v36;
        v36 = reinterpret_cast<float *>(&v0[524288ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16);
        float * v38;
        v38 = reinterpret_cast<float *>(&v1[2752512ull]);
        block_matmul_4(v38, v36, v29, v17);
        block_row_map_reduce_5(v14, v16, v11, v13, v38, v8);
        v9 += 1 ;
    }
    return ;
}
extern "C" __global__ void entry2(unsigned char * v0, unsigned char * v1, unsigned char * v2) {
    int v3;
    v3 = threadIdx.x;
    int v4;
    v4 = blockIdx.x;
    int v5;
    v5 = v4 * 256;
    int v6;
    v6 = v3 + v5;
    unsigned long long v7;
    v7 = (unsigned long long)v6;
    curandStatePhilox4_32_10_t v8;
    curand_init(12344321ull,v7,0ull,&v8);
    int v9;
    v9 = 0;
    while (while_method_3(v9)){
        float * v11;
        v11 = reinterpret_cast<float *>(&v1[786432ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 4);
        int v13;
        v13 = 98304 * v9;
        int * v14;
        v14 = reinterpret_cast<int *>(&v1[2359296ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 4);
        int v16;
        v16 = 1536 * v9;
        float * v17;
        v17 = reinterpret_cast<float *>(&v1[0ull]);
        float * v19;
        v19 = reinterpret_cast<float *>(&v0[0ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 4);
        int v21;
        v21 = 4096 * v9;
        float * v22;
        v22 = reinterpret_cast<float *>(&v1[393216ull]);
        block_matmul_4(v22, v19, v21, v17);
        block_row_map_reduce_6(v14, v16, v11, v13, v22, v8);
        v9 += 1 ;
    }
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
def method1(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method2(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method0() -> None:
    v0 = "test_text_outputs/layers/"
    v1 = "test1"
    v2 = "layers.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.empty(0,dtype=cp.uint8)
    del v4
    v5 = cp.empty(2304,dtype=cp.uint8)
    v6 = cp.empty(160,dtype=cp.uint8)
    v9 = "{}\n"
    v10 = "---"
    print(v9.format(v10),end="")
    del v10
    v12 = v6[0:0+4*16].view(cp.float32)
    v14 = v6[64:64+4*16].view(cp.float32)
    v16 = v6[128:128+4*8].view(cp.float32)
    v36 = 0
    v37 = "{}"
    print(v37.format('['),end="")
    v38 = 0
    while method1(v38):
        v40 = v36
        v41 = v40 >= 100
        del v40
        if v41:
            v42 = " ..."
            print(v37.format(v42),end="")
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
            print(v37.format(v45),end="")
            del v45
        else:
            pass
        del v44
        print(v37.format('['),end="")
        v46 = 0
        while method1(v46):
            v48 = v36
            v49 = v48 >= 100
            del v48
            if v49:
                v50 = " ..."
                print(v37.format(v50),end="")
                del v50
                break
            else:
                pass
            del v49
            v51 = v46 == 0
            v52 = v51 != True
            del v51
            if v52:
                v53 = "; "
                print(v37.format(v53),end="")
                del v53
            else:
                pass
            del v52
            v54 = v36 + 1
            v36 = v54
            del v54
            v55 = v38 * 4
            v56 = v55 + v46
            del v55
            v57 = v12[v56].item()
            del v56
            v58 = "{:.6f}"
            print(v58.format(v57),end="")
            del v57, v58
            v46 += 1 
        del v46
        print(v37.format(']'),end="")
        v38 += 1 
    del v12, v36, v38
    print(v37.format(']'),end="")
    v59 = "\n"
    print(v59.format(),end="")
    v79 = 0
    print(v37.format('['),end="")
    v80 = 0
    while method1(v80):
        v82 = v79
        v83 = v82 >= 100
        del v82
        if v83:
            v84 = " ..."
            print(v37.format(v84),end="")
            del v84
            break
        else:
            pass
        del v83
        v85 = v80 == 0
        v86 = v85 != True
        del v85
        if v86:
            v87 = "; "
            print(v37.format(v87),end="")
            del v87
        else:
            pass
        del v86
        print(v37.format('['),end="")
        v88 = 0
        while method1(v88):
            v90 = v79
            v91 = v90 >= 100
            del v90
            if v91:
                v92 = " ..."
                print(v37.format(v92),end="")
                del v92
                break
            else:
                pass
            del v91
            v93 = v88 == 0
            v94 = v93 != True
            del v93
            if v94:
                v95 = "; "
                print(v37.format(v95),end="")
                del v95
            else:
                pass
            del v94
            v96 = v79 + 1
            v79 = v96
            del v96
            v97 = v80 * 4
            v98 = v97 + v88
            del v97
            v99 = v14[v98].item()
            del v98
            v100 = "{:.6f}"
            print(v100.format(v99),end="")
            del v99, v100
            v88 += 1 
        del v88
        print(v37.format(']'),end="")
        v80 += 1 
    del v14, v79, v80
    print(v37.format(']'),end="")
    print(v59.format(),end="")
    v120 = 0
    print(v37.format('['),end="")
    v121 = 0
    while method2(v121):
        v123 = v120
        v124 = v123 >= 100
        del v123
        if v124:
            v125 = " ..."
            print(v37.format(v125),end="")
            del v125
            break
        else:
            pass
        del v124
        v126 = v121 == 0
        v127 = v126 != True
        del v126
        if v127:
            v128 = "; "
            print(v37.format(v128),end="")
            del v128
        else:
            pass
        del v127
        print(v37.format('['),end="")
        v129 = 0
        while method1(v129):
            v131 = v120
            v132 = v131 >= 100
            del v131
            if v132:
                v133 = " ..."
                print(v37.format(v133),end="")
                del v133
                break
            else:
                pass
            del v132
            v134 = v129 == 0
            v135 = v134 != True
            del v134
            if v135:
                v136 = "; "
                print(v37.format(v136),end="")
                del v136
            else:
                pass
            del v135
            v137 = v120 + 1
            v120 = v137
            del v137
            v138 = v121 * 4
            v139 = v138 + v129
            del v138
            v140 = v16[v139].item()
            del v139
            v141 = "{:.6f}"
            print(v141.format(v140),end="")
            del v140, v141
            v129 += 1 
        del v129
        print(v37.format(']'),end="")
        v121 += 1 
    del v16, v120, v121
    print(v37.format(']'),end="")
    print(v59.format(),end="")
    v143 = v5[0:0+4*96].view(cp.float32)
    del v5, v143
    v145 = v6[0:0+4*16].view(cp.float32)
    v147 = v6[64:64+4*16].view(cp.float32)
    v149 = v6[128:128+4*8].view(cp.float32)
    v150 = cp.random.normal(0.0,0.25,16,dtype=cp.float32) # type: ignore
    cp.copyto(v145[0:0+16],v150[0:0+16])
    del v145, v150
    v151 = cp.random.normal(0.0,0.25,16,dtype=cp.float32) # type: ignore
    cp.copyto(v147[0:0+16],v151[0:0+16])
    del v147, v151
    v152 = cp.random.normal(0.0,0.35355338,8,dtype=cp.float32) # type: ignore
    cp.copyto(v149[0:0+8],v152[0:0+8])
    del v149, v152
    v155 = "Done initing."
    print(v9.format(v155),end="")
    del v9, v155
    v157 = v6[0:0+4*16].view(cp.float32)
    v159 = v6[64:64+4*16].view(cp.float32)
    v161 = v6[128:128+4*8].view(cp.float32)
    del v6
    v181 = 0
    print(v37.format('['),end="")
    v182 = 0
    while method1(v182):
        v184 = v181
        v185 = v184 >= 100
        del v184
        if v185:
            v186 = " ..."
            print(v37.format(v186),end="")
            del v186
            break
        else:
            pass
        del v185
        v187 = v182 == 0
        v188 = v187 != True
        del v187
        if v188:
            v189 = "; "
            print(v37.format(v189),end="")
            del v189
        else:
            pass
        del v188
        print(v37.format('['),end="")
        v190 = 0
        while method1(v190):
            v192 = v181
            v193 = v192 >= 100
            del v192
            if v193:
                v194 = " ..."
                print(v37.format(v194),end="")
                del v194
                break
            else:
                pass
            del v193
            v195 = v190 == 0
            v196 = v195 != True
            del v195
            if v196:
                v197 = "; "
                print(v37.format(v197),end="")
                del v197
            else:
                pass
            del v196
            v198 = v181 + 1
            v181 = v198
            del v198
            v199 = v182 * 4
            v200 = v199 + v190
            del v199
            v201 = v157[v200].item()
            del v200
            v202 = "{:.6f}"
            print(v202.format(v201),end="")
            del v201, v202
            v190 += 1 
        del v190
        print(v37.format(']'),end="")
        v182 += 1 
    del v157, v181, v182
    print(v37.format(']'),end="")
    print(v59.format(),end="")
    v222 = 0
    print(v37.format('['),end="")
    v223 = 0
    while method1(v223):
        v225 = v222
        v226 = v225 >= 100
        del v225
        if v226:
            v227 = " ..."
            print(v37.format(v227),end="")
            del v227
            break
        else:
            pass
        del v226
        v228 = v223 == 0
        v229 = v228 != True
        del v228
        if v229:
            v230 = "; "
            print(v37.format(v230),end="")
            del v230
        else:
            pass
        del v229
        print(v37.format('['),end="")
        v231 = 0
        while method1(v231):
            v233 = v222
            v234 = v233 >= 100
            del v233
            if v234:
                v235 = " ..."
                print(v37.format(v235),end="")
                del v235
                break
            else:
                pass
            del v234
            v236 = v231 == 0
            v237 = v236 != True
            del v236
            if v237:
                v238 = "; "
                print(v37.format(v238),end="")
                del v238
            else:
                pass
            del v237
            v239 = v222 + 1
            v222 = v239
            del v239
            v240 = v223 * 4
            v241 = v240 + v231
            del v240
            v242 = v159[v241].item()
            del v241
            v243 = "{:.6f}"
            print(v243.format(v242),end="")
            del v242, v243
            v231 += 1 
        del v231
        print(v37.format(']'),end="")
        v223 += 1 
    del v159, v222, v223
    print(v37.format(']'),end="")
    print(v59.format(),end="")
    v263 = 0
    print(v37.format('['),end="")
    v264 = 0
    while method2(v264):
        v266 = v263
        v267 = v266 >= 100
        del v266
        if v267:
            v268 = " ..."
            print(v37.format(v268),end="")
            del v268
            break
        else:
            pass
        del v267
        v269 = v264 == 0
        v270 = v269 != True
        del v269
        if v270:
            v271 = "; "
            print(v37.format(v271),end="")
            del v271
        else:
            pass
        del v270
        print(v37.format('['),end="")
        v272 = 0
        while method1(v272):
            v274 = v263
            v275 = v274 >= 100
            del v274
            if v275:
                v276 = " ..."
                print(v37.format(v276),end="")
                del v276
                break
            else:
                pass
            del v275
            v277 = v272 == 0
            v278 = v277 != True
            del v277
            if v278:
                v279 = "; "
                print(v37.format(v279),end="")
                del v279
            else:
                pass
            del v278
            v280 = v263 + 1
            v263 = v280
            del v280
            v281 = v264 * 4
            v282 = v281 + v272
            del v281
            v283 = v161[v282].item()
            del v282
            v284 = "{:.6f}"
            print(v284.format(v283),end="")
            del v283, v284
            v272 += 1 
        del v272
        print(v37.format(']'),end="")
        v264 += 1 
    del v161, v263, v264
    print(v37.format(']'),end="")
    del v37
    print(v59.format(),end="")
    del v59
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method4(v0 : i32) -> bool:
    v1 = v0 < 24
    del v0
    return v1
def method5(v0 : i32) -> bool:
    v1 = v0 < 1
    del v0
    return v1
def method3() -> None:
    v0 = "test_text_outputs/layers/"
    v1 = "test2"
    v2 = "layers.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.empty(0,dtype=cp.uint8)
    del v4
    v5 = cp.empty(2112,dtype=cp.uint8)
    v6 = cp.empty(128,dtype=cp.uint8)
    v9 = "{}\n"
    v10 = "---"
    print(v9.format(v10),end="")
    del v10
    v12 = v5[0:0+4*48].view(cp.float32)
    del v12
    v14 = v6[0:0+4*8].view(cp.float32)
    v16 = v6[32:32+4*16].view(cp.float32)
    v18 = v6[96:96+4*8].view(cp.float32)
    v19 = cp.random.normal(0.0,0.35355338,8,dtype=cp.float32) # type: ignore
    cp.copyto(v14[0:0+8],v19[0:0+8])
    del v14, v19
    v20 = cp.random.normal(0.0,0.25,16,dtype=cp.float32) # type: ignore
    cp.copyto(v16[0:0+16],v20[0:0+16])
    del v16, v20
    v21 = cp.random.normal(0.0,0.35355338,8,dtype=cp.float32) # type: ignore
    cp.copyto(v18[0:0+8],v21[0:0+8])
    del v18, v21
    v24 = "Here are the weight matrices."
    print(v9.format(v24),end="")
    del v24
    v26 = v6[0:0+4*8].view(cp.float32)
    v28 = v6[32:32+4*16].view(cp.float32)
    v30 = v6[96:96+4*8].view(cp.float32)
    del v6
    v50 = 0
    v51 = "{}"
    print(v51.format('['),end="")
    v52 = 0
    while method1(v52):
        v54 = v50
        v55 = v54 >= 100
        del v54
        if v55:
            v56 = " ..."
            print(v51.format(v56),end="")
            del v56
            break
        else:
            pass
        del v55
        v57 = v52 == 0
        v58 = v57 != True
        del v57
        if v58:
            v59 = "; "
            print(v51.format(v59),end="")
            del v59
        else:
            pass
        del v58
        print(v51.format('['),end="")
        v60 = 0
        while method2(v60):
            v62 = v50
            v63 = v62 >= 100
            del v62
            if v63:
                v64 = " ..."
                print(v51.format(v64),end="")
                del v64
                break
            else:
                pass
            del v63
            v65 = v60 == 0
            v66 = v65 != True
            del v65
            if v66:
                v67 = "; "
                print(v51.format(v67),end="")
                del v67
            else:
                pass
            del v66
            v68 = v50 + 1
            v50 = v68
            del v68
            v69 = v52 * 2
            v70 = v69 + v60
            del v69
            v71 = v26[v70].item()
            del v70
            v72 = "{:.6f}"
            print(v72.format(v71),end="")
            del v71, v72
            v60 += 1 
        del v60
        print(v51.format(']'),end="")
        v52 += 1 
    del v26, v50, v52
    print(v51.format(']'),end="")
    v73 = "\n"
    print(v73.format(),end="")
    v93 = 0
    print(v51.format('['),end="")
    v94 = 0
    while method1(v94):
        v96 = v93
        v97 = v96 >= 100
        del v96
        if v97:
            v98 = " ..."
            print(v51.format(v98),end="")
            del v98
            break
        else:
            pass
        del v97
        v99 = v94 == 0
        v100 = v99 != True
        del v99
        if v100:
            v101 = "; "
            print(v51.format(v101),end="")
            del v101
        else:
            pass
        del v100
        print(v51.format('['),end="")
        v102 = 0
        while method1(v102):
            v104 = v93
            v105 = v104 >= 100
            del v104
            if v105:
                v106 = " ..."
                print(v51.format(v106),end="")
                del v106
                break
            else:
                pass
            del v105
            v107 = v102 == 0
            v108 = v107 != True
            del v107
            if v108:
                v109 = "; "
                print(v51.format(v109),end="")
                del v109
            else:
                pass
            del v108
            v110 = v93 + 1
            v93 = v110
            del v110
            v111 = v94 * 4
            v112 = v111 + v102
            del v111
            v113 = v28[v112].item()
            del v112
            v114 = "{:.6f}"
            print(v114.format(v113),end="")
            del v113, v114
            v102 += 1 
        del v102
        print(v51.format(']'),end="")
        v94 += 1 
    del v28, v93, v94
    print(v51.format(']'),end="")
    print(v73.format(),end="")
    v134 = 0
    print(v51.format('['),end="")
    v135 = 0
    while method2(v135):
        v137 = v134
        v138 = v137 >= 100
        del v137
        if v138:
            v139 = " ..."
            print(v51.format(v139),end="")
            del v139
            break
        else:
            pass
        del v138
        v140 = v135 == 0
        v141 = v140 != True
        del v140
        if v141:
            v142 = "; "
            print(v51.format(v142),end="")
            del v142
        else:
            pass
        del v141
        print(v51.format('['),end="")
        v143 = 0
        while method1(v143):
            v145 = v134
            v146 = v145 >= 100
            del v145
            if v146:
                v147 = " ..."
                print(v51.format(v147),end="")
                del v147
                break
            else:
                pass
            del v146
            v148 = v143 == 0
            v149 = v148 != True
            del v148
            if v149:
                v150 = "; "
                print(v51.format(v150),end="")
                del v150
            else:
                pass
            del v149
            v151 = v134 + 1
            v134 = v151
            del v151
            v152 = v135 * 4
            v153 = v152 + v143
            del v152
            v154 = v30[v153].item()
            del v153
            v155 = "{:.6f}"
            print(v155.format(v154),end="")
            del v154, v155
            v143 += 1 
        del v143
        print(v51.format(']'),end="")
        v135 += 1 
    del v30, v134, v135
    print(v51.format(']'),end="")
    print(v73.format(),end="")
    v158 = "Here is the input tensor."
    print(v9.format(v158),end="")
    del v9, v158
    v160 = v5[0:0+4*48].view(cp.float32)
    del v5
    v161 = cp.random.normal(0.0,1.0,48,dtype=cp.float32) # type: ignore
    cp.copyto(v160[0:0+48],v161[0:0+48])
    del v161
    v189 = 0
    print(v51.format('['),end="")
    v190 = 0
    while method4(v190):
        v192 = v189
        v193 = v192 >= 100
        del v192
        if v193:
            v194 = " ..."
            print(v51.format(v194),end="")
            del v194
            break
        else:
            pass
        del v193
        v195 = v190 == 0
        v196 = v195 != True
        del v195
        if v196:
            v197 = "; "
            print(v51.format(v197),end="")
            del v197
        else:
            pass
        del v196
        print(v51.format('['),end="")
        v198 = 0
        while method5(v198):
            v200 = v189
            v201 = v200 >= 100
            del v200
            if v201:
                v202 = " ..."
                print(v51.format(v202),end="")
                del v202
                break
            else:
                pass
            del v201
            v203 = v198 == 0
            v204 = v203 != True
            del v203
            if v204:
                v205 = "; "
                print(v51.format(v205),end="")
                del v205
            else:
                pass
            del v204
            print(v51.format('['),end="")
            v206 = 0
            while method2(v206):
                v208 = v189
                v209 = v208 >= 100
                del v208
                if v209:
                    v210 = " ..."
                    print(v51.format(v210),end="")
                    del v210
                    break
                else:
                    pass
                del v209
                v211 = v206 == 0
                v212 = v211 != True
                del v211
                if v212:
                    v213 = "; "
                    print(v51.format(v213),end="")
                    del v213
                else:
                    pass
                del v212
                v214 = v189 + 1
                v189 = v214
                del v214
                v215 = v190 * 2
                v216 = v198 * 2
                v217 = v215 + v216
                del v215, v216
                v218 = v217 + v206
                del v217
                v219 = v160[v218].item()
                del v218
                v220 = "{:.6f}"
                print(v220.format(v219),end="")
                del v219, v220
                v206 += 1 
            del v206
            print(v51.format(']'),end="")
            v198 += 1 
        del v198
        print(v51.format(']'),end="")
        v190 += 1 
    del v160, v189, v190
    print(v51.format(']'),end="")
    del v51
    print(v73.format(),end="")
    del v73
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method7(v0 : i32) -> bool:
    v1 = v0 < 64
    del v0
    return v1
def method6() -> None:
    v0 = "test_text_outputs/layers/"
    v1 = "test3"
    v2 = "layers.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.empty(0,dtype=cp.uint8)
    v5 = cp.empty(3545088,dtype=cp.uint8)
    v6 = cp.empty(49152,dtype=cp.uint8)
    v8 = v5[0:0+4*98304].view(cp.float32)
    del v8
    v10 = v6[0:0+4*4096].view(cp.float32)
    v12 = v6[16384:16384+4*4096].view(cp.float32)
    v14 = v6[32768:32768+4*4096].view(cp.float32)
    v15 = cp.random.normal(0.0,0.015625,4096,dtype=cp.float32) # type: ignore
    cp.copyto(v10[0:0+4096],v15[0:0+4096])
    del v10, v15
    v16 = cp.random.normal(0.0,0.015625,4096,dtype=cp.float32) # type: ignore
    cp.copyto(v12[0:0+4096],v16[0:0+4096])
    del v12, v16
    v17 = cp.random.normal(0.0,0.015625,4096,dtype=cp.float32) # type: ignore
    cp.copyto(v14[0:0+4096],v17[0:0+4096])
    del v14, v17
    v20 = "{}\n"
    v21 = "Here are the weight matrices."
    print(v20.format(v21),end="")
    del v21
    v23 = v6[0:0+4*4096].view(cp.float32)
    v25 = v6[16384:16384+4*4096].view(cp.float32)
    v27 = v6[32768:32768+4*4096].view(cp.float32)
    v47 = 0
    v48 = "{}"
    print(v48.format('['),end="")
    v49 = 0
    while method7(v49):
        v51 = v47
        v52 = v51 >= 100
        del v51
        if v52:
            v53 = " ..."
            print(v48.format(v53),end="")
            del v53
            break
        else:
            pass
        del v52
        v54 = v49 == 0
        v55 = v54 != True
        del v54
        if v55:
            v56 = "; "
            print(v48.format(v56),end="")
            del v56
        else:
            pass
        del v55
        print(v48.format('['),end="")
        v57 = 0
        while method7(v57):
            v59 = v47
            v60 = v59 >= 100
            del v59
            if v60:
                v61 = " ..."
                print(v48.format(v61),end="")
                del v61
                break
            else:
                pass
            del v60
            v62 = v57 == 0
            v63 = v62 != True
            del v62
            if v63:
                v64 = "; "
                print(v48.format(v64),end="")
                del v64
            else:
                pass
            del v63
            v65 = v47 + 1
            v47 = v65
            del v65
            v66 = v49 * 64
            v67 = v66 + v57
            del v66
            v68 = v23[v67].item()
            del v67
            v69 = "{:.6f}"
            print(v69.format(v68),end="")
            del v68, v69
            v57 += 1 
        del v57
        print(v48.format(']'),end="")
        v49 += 1 
    del v23, v47, v49
    print(v48.format(']'),end="")
    v70 = "\n"
    print(v70.format(),end="")
    v90 = 0
    print(v48.format('['),end="")
    v91 = 0
    while method7(v91):
        v93 = v90
        v94 = v93 >= 100
        del v93
        if v94:
            v95 = " ..."
            print(v48.format(v95),end="")
            del v95
            break
        else:
            pass
        del v94
        v96 = v91 == 0
        v97 = v96 != True
        del v96
        if v97:
            v98 = "; "
            print(v48.format(v98),end="")
            del v98
        else:
            pass
        del v97
        print(v48.format('['),end="")
        v99 = 0
        while method7(v99):
            v101 = v90
            v102 = v101 >= 100
            del v101
            if v102:
                v103 = " ..."
                print(v48.format(v103),end="")
                del v103
                break
            else:
                pass
            del v102
            v104 = v99 == 0
            v105 = v104 != True
            del v104
            if v105:
                v106 = "; "
                print(v48.format(v106),end="")
                del v106
            else:
                pass
            del v105
            v107 = v90 + 1
            v90 = v107
            del v107
            v108 = v91 * 64
            v109 = v108 + v99
            del v108
            v110 = v25[v109].item()
            del v109
            v111 = "{:.6f}"
            print(v111.format(v110),end="")
            del v110, v111
            v99 += 1 
        del v99
        print(v48.format(']'),end="")
        v91 += 1 
    del v25, v90, v91
    print(v48.format(']'),end="")
    print(v70.format(),end="")
    v131 = 0
    print(v48.format('['),end="")
    v132 = 0
    while method7(v132):
        v134 = v131
        v135 = v134 >= 100
        del v134
        if v135:
            v136 = " ..."
            print(v48.format(v136),end="")
            del v136
            break
        else:
            pass
        del v135
        v137 = v132 == 0
        v138 = v137 != True
        del v137
        if v138:
            v139 = "; "
            print(v48.format(v139),end="")
            del v139
        else:
            pass
        del v138
        print(v48.format('['),end="")
        v140 = 0
        while method7(v140):
            v142 = v131
            v143 = v142 >= 100
            del v142
            if v143:
                v144 = " ..."
                print(v48.format(v144),end="")
                del v144
                break
            else:
                pass
            del v143
            v145 = v140 == 0
            v146 = v145 != True
            del v145
            if v146:
                v147 = "; "
                print(v48.format(v147),end="")
                del v147
            else:
                pass
            del v146
            v148 = v131 + 1
            v131 = v148
            del v148
            v149 = v132 * 64
            v150 = v149 + v140
            del v149
            v151 = v27[v150].item()
            del v150
            v152 = "{:.6f}"
            print(v152.format(v151),end="")
            del v151, v152
            v140 += 1 
        del v140
        print(v48.format(']'),end="")
        v132 += 1 
    del v27, v131, v132
    print(v48.format(']'),end="")
    print(v70.format(),end="")
    v154 = v5[0:0+4*98304].view(cp.float32)
    v155 = cp.random.normal(0.0,1.0,98304,dtype=cp.float32) # type: ignore
    cp.copyto(v154[0:0+98304],v155[0:0+98304])
    del v155
    v183 = 0
    print(v48.format('['),end="")
    v184 = 0
    while method4(v184):
        v186 = v183
        v187 = v186 >= 100
        del v186
        if v187:
            v188 = " ..."
            print(v48.format(v188),end="")
            del v188
            break
        else:
            pass
        del v187
        v189 = v184 == 0
        v190 = v189 != True
        del v189
        if v190:
            v191 = "; "
            print(v48.format(v191),end="")
            del v191
        else:
            pass
        del v190
        print(v48.format('['),end="")
        v192 = 0
        while method7(v192):
            v194 = v183
            v195 = v194 >= 100
            del v194
            if v195:
                v196 = " ..."
                print(v48.format(v196),end="")
                del v196
                break
            else:
                pass
            del v195
            v197 = v192 == 0
            v198 = v197 != True
            del v197
            if v198:
                v199 = "; "
                print(v48.format(v199),end="")
                del v199
            else:
                pass
            del v198
            print(v48.format('['),end="")
            v200 = 0
            while method7(v200):
                v202 = v183
                v203 = v202 >= 100
                del v202
                if v203:
                    v204 = " ..."
                    print(v48.format(v204),end="")
                    del v204
                    break
                else:
                    pass
                del v203
                v205 = v200 == 0
                v206 = v205 != True
                del v205
                if v206:
                    v207 = "; "
                    print(v48.format(v207),end="")
                    del v207
                else:
                    pass
                del v206
                v208 = v183 + 1
                v183 = v208
                del v208
                v209 = v184 * 4096
                v210 = v192 * 64
                v211 = v209 + v210
                del v209, v210
                v212 = v211 + v200
                del v211
                v213 = v154[v212].item()
                del v212
                v214 = "{:.6f}"
                print(v214.format(v213),end="")
                del v213, v214
                v200 += 1 
            del v200
            print(v48.format(']'),end="")
            v192 += 1 
        del v192
        print(v48.format(']'),end="")
        v184 += 1 
    del v154, v183, v184
    print(v48.format(']'),end="")
    print(v70.format(),end="")
    v217 = "Here is the output tensor."
    print(v20.format(v217),end="")
    del v217
    v218 = cp.cuda.Device().attributes['MultiProcessorCount']
    v219 = v218 == 24
    del v218
    v220 = v219 == False
    if v220:
        v221 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v219, v221
        del v221
    else:
        pass
    del v219, v220
    v222 = 0
    v223 = raw_module.get_function(f"entry{v222}")
    del v222
    v223.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v223((24,),(256,),(v6, v5, v4),shared_mem=98304)
    del v4, v6, v223
    v225 = v5[3145728:3145728+4*98304].view(cp.float32)
    v227 = v5[3538944:3538944+4*1536].view(cp.int32)
    del v5
    v272 = 0
    print(v48.format('['),end="")
    v273 = 0
    while method4(v273):
        v275 = v272
        v276 = v275 >= 100
        del v275
        if v276:
            v277 = " ..."
            print(v48.format(v277),end="")
            del v277
            break
        else:
            pass
        del v276
        v278 = v273 == 0
        v279 = v278 != True
        del v278
        if v279:
            v280 = "; "
            print(v48.format(v280),end="")
            del v280
        else:
            pass
        del v279
        print(v48.format('['),end="")
        v281 = 0
        while method7(v281):
            v283 = v272
            v284 = v283 >= 100
            del v283
            if v284:
                v285 = " ..."
                print(v48.format(v285),end="")
                del v285
                break
            else:
                pass
            del v284
            v286 = v281 == 0
            v287 = v286 != True
            del v286
            if v287:
                v288 = "; "
                print(v48.format(v288),end="")
                del v288
            else:
                pass
            del v287
            print(v48.format('['),end="")
            v289 = 0
            while method7(v289):
                v291 = v272
                v292 = v291 >= 100
                del v291
                if v292:
                    v293 = " ..."
                    print(v48.format(v293),end="")
                    del v293
                    break
                else:
                    pass
                del v292
                v294 = v289 == 0
                v295 = v294 != True
                del v294
                if v295:
                    v296 = "; "
                    print(v48.format(v296),end="")
                    del v296
                else:
                    pass
                del v295
                v297 = v272 + 1
                v272 = v297
                del v297
                v298 = v273 * 4096
                v299 = v281 * 64
                v300 = v298 + v299
                del v298, v299
                v301 = v300 + v289
                del v300
                v302 = v225[v301].item()
                del v301
                v303 = "{:.6f}"
                print(v303.format(v302),end="")
                del v302, v303
                v289 += 1 
            del v289
            print(v48.format(']'),end="")
            v281 += 1 
        del v281
        print(v48.format(']'),end="")
        v273 += 1 
    del v225, v272, v273
    print(v48.format(']'),end="")
    v304 = 0
    v305 = ", {}"
    print(v305.format('['),end="")
    del v305
    v306 = 0
    while method4(v306):
        v308 = v304
        v309 = v308 >= 100
        del v308
        if v309:
            v310 = " ..."
            print(v48.format(v310),end="")
            del v310
            break
        else:
            pass
        del v309
        v311 = v306 == 0
        v312 = v311 != True
        del v311
        if v312:
            v313 = "; "
            print(v48.format(v313),end="")
            del v313
        else:
            pass
        del v312
        print(v48.format('['),end="")
        v314 = 0
        while method7(v314):
            v316 = v304
            v317 = v316 >= 100
            del v316
            if v317:
                v318 = " ..."
                print(v48.format(v318),end="")
                del v318
                break
            else:
                pass
            del v317
            v319 = v314 == 0
            v320 = v319 != True
            del v319
            if v320:
                v321 = "; "
                print(v48.format(v321),end="")
                del v321
            else:
                pass
            del v320
            v322 = v304 + 1
            v304 = v322
            del v322
            v323 = v306 * 64
            v324 = v323 + v314
            del v323
            v325 = v227[v324].item()
            del v324
            print(v48.format(v325),end="")
            del v325
            v314 += 1 
        del v314
        print(v48.format(']'),end="")
        v306 += 1 
    del v227, v304, v306
    print(v48.format(']'),end="")
    del v48
    print(v70.format(),end="")
    del v70
    v328 = "===="
    print(v20.format(v328),end="")
    del v20, v328
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method9(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def method8() -> None:
    v0 = "test_text_outputs/layers/"
    v1 = "test4"
    v2 = "layers.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.empty(0,dtype=cp.uint8)
    v5 = cp.empty(9535488,dtype=cp.uint8)
    v6 = cp.empty(786432,dtype=cp.uint8)
    v8 = v5[0:0+4*98304].view(cp.float32)
    del v8
    v10 = v6[0:0+4*65536].view(cp.float32)
    v12 = v6[262144:262144+4*65536].view(cp.float32)
    v14 = v6[524288:524288+4*65536].view(cp.float32)
    v16 = v5[3145728:3145728+4*1572864].view(cp.float32)
    del v16
    v18 = v5[9437184:9437184+4*24576].view(cp.int32)
    del v18
    v19 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v10[0:0+65536],v19[0:0+65536])
    del v10, v19
    v20 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v12[0:0+65536],v20[0:0+65536])
    del v12, v20
    v21 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v14[0:0+65536],v21[0:0+65536])
    del v14, v21
    v24 = "{}\n"
    v25 = "Here are the weight matrices."
    print(v24.format(v25),end="")
    del v25
    v27 = v6[0:0+4*65536].view(cp.float32)
    v29 = v6[262144:262144+4*65536].view(cp.float32)
    v31 = v6[524288:524288+4*65536].view(cp.float32)
    v59 = 0
    v60 = "{}"
    print(v60.format('['),end="")
    v61 = 0
    while method9(v61):
        v63 = v59
        v64 = v63 >= 100
        del v63
        if v64:
            v65 = " ..."
            print(v60.format(v65),end="")
            del v65
            break
        else:
            pass
        del v64
        v66 = v61 == 0
        v67 = v66 != True
        del v66
        if v67:
            v68 = "; "
            print(v60.format(v68),end="")
            del v68
        else:
            pass
        del v67
        print(v60.format('['),end="")
        v69 = 0
        while method7(v69):
            v71 = v59
            v72 = v71 >= 100
            del v71
            if v72:
                v73 = " ..."
                print(v60.format(v73),end="")
                del v73
                break
            else:
                pass
            del v72
            v74 = v69 == 0
            v75 = v74 != True
            del v74
            if v75:
                v76 = "; "
                print(v60.format(v76),end="")
                del v76
            else:
                pass
            del v75
            print(v60.format('['),end="")
            v77 = 0
            while method7(v77):
                v79 = v59
                v80 = v79 >= 100
                del v79
                if v80:
                    v81 = " ..."
                    print(v60.format(v81),end="")
                    del v81
                    break
                else:
                    pass
                del v80
                v82 = v77 == 0
                v83 = v82 != True
                del v82
                if v83:
                    v84 = "; "
                    print(v60.format(v84),end="")
                    del v84
                else:
                    pass
                del v83
                v85 = v59 + 1
                v59 = v85
                del v85
                v86 = v61 * 4096
                v87 = v69 * 64
                v88 = v86 + v87
                del v86, v87
                v89 = v88 + v77
                del v88
                v90 = v27[v89].item()
                del v89
                v91 = "{:.6f}"
                print(v91.format(v90),end="")
                del v90, v91
                v77 += 1 
            del v77
            print(v60.format(']'),end="")
            v69 += 1 
        del v69
        print(v60.format(']'),end="")
        v61 += 1 
    del v27, v59, v61
    print(v60.format(']'),end="")
    v92 = "\n"
    print(v92.format(),end="")
    v120 = 0
    print(v60.format('['),end="")
    v121 = 0
    while method9(v121):
        v123 = v120
        v124 = v123 >= 100
        del v123
        if v124:
            v125 = " ..."
            print(v60.format(v125),end="")
            del v125
            break
        else:
            pass
        del v124
        v126 = v121 == 0
        v127 = v126 != True
        del v126
        if v127:
            v128 = "; "
            print(v60.format(v128),end="")
            del v128
        else:
            pass
        del v127
        print(v60.format('['),end="")
        v129 = 0
        while method7(v129):
            v131 = v120
            v132 = v131 >= 100
            del v131
            if v132:
                v133 = " ..."
                print(v60.format(v133),end="")
                del v133
                break
            else:
                pass
            del v132
            v134 = v129 == 0
            v135 = v134 != True
            del v134
            if v135:
                v136 = "; "
                print(v60.format(v136),end="")
                del v136
            else:
                pass
            del v135
            print(v60.format('['),end="")
            v137 = 0
            while method7(v137):
                v139 = v120
                v140 = v139 >= 100
                del v139
                if v140:
                    v141 = " ..."
                    print(v60.format(v141),end="")
                    del v141
                    break
                else:
                    pass
                del v140
                v142 = v137 == 0
                v143 = v142 != True
                del v142
                if v143:
                    v144 = "; "
                    print(v60.format(v144),end="")
                    del v144
                else:
                    pass
                del v143
                v145 = v120 + 1
                v120 = v145
                del v145
                v146 = v121 * 4096
                v147 = v129 * 64
                v148 = v146 + v147
                del v146, v147
                v149 = v148 + v137
                del v148
                v150 = v29[v149].item()
                del v149
                v151 = "{:.6f}"
                print(v151.format(v150),end="")
                del v150, v151
                v137 += 1 
            del v137
            print(v60.format(']'),end="")
            v129 += 1 
        del v129
        print(v60.format(']'),end="")
        v121 += 1 
    del v29, v120, v121
    print(v60.format(']'),end="")
    print(v92.format(),end="")
    v179 = 0
    print(v60.format('['),end="")
    v180 = 0
    while method9(v180):
        v182 = v179
        v183 = v182 >= 100
        del v182
        if v183:
            v184 = " ..."
            print(v60.format(v184),end="")
            del v184
            break
        else:
            pass
        del v183
        v185 = v180 == 0
        v186 = v185 != True
        del v185
        if v186:
            v187 = "; "
            print(v60.format(v187),end="")
            del v187
        else:
            pass
        del v186
        print(v60.format('['),end="")
        v188 = 0
        while method7(v188):
            v190 = v179
            v191 = v190 >= 100
            del v190
            if v191:
                v192 = " ..."
                print(v60.format(v192),end="")
                del v192
                break
            else:
                pass
            del v191
            v193 = v188 == 0
            v194 = v193 != True
            del v193
            if v194:
                v195 = "; "
                print(v60.format(v195),end="")
                del v195
            else:
                pass
            del v194
            print(v60.format('['),end="")
            v196 = 0
            while method7(v196):
                v198 = v179
                v199 = v198 >= 100
                del v198
                if v199:
                    v200 = " ..."
                    print(v60.format(v200),end="")
                    del v200
                    break
                else:
                    pass
                del v199
                v201 = v196 == 0
                v202 = v201 != True
                del v201
                if v202:
                    v203 = "; "
                    print(v60.format(v203),end="")
                    del v203
                else:
                    pass
                del v202
                v204 = v179 + 1
                v179 = v204
                del v204
                v205 = v180 * 4096
                v206 = v188 * 64
                v207 = v205 + v206
                del v205, v206
                v208 = v207 + v196
                del v207
                v209 = v31[v208].item()
                del v208
                v210 = "{:.6f}"
                print(v210.format(v209),end="")
                del v209, v210
                v196 += 1 
            del v196
            print(v60.format(']'),end="")
            v188 += 1 
        del v188
        print(v60.format(']'),end="")
        v180 += 1 
    del v31, v179, v180
    print(v60.format(']'),end="")
    print(v92.format(),end="")
    v212 = v5[0:0+4*98304].view(cp.float32)
    v213 = cp.random.normal(0.0,1.0,98304,dtype=cp.float32) # type: ignore
    cp.copyto(v212[0:0+98304],v213[0:0+98304])
    del v213
    v241 = 0
    print(v60.format('['),end="")
    v242 = 0
    while method4(v242):
        v244 = v241
        v245 = v244 >= 100
        del v244
        if v245:
            v246 = " ..."
            print(v60.format(v246),end="")
            del v246
            break
        else:
            pass
        del v245
        v247 = v242 == 0
        v248 = v247 != True
        del v247
        if v248:
            v249 = "; "
            print(v60.format(v249),end="")
            del v249
        else:
            pass
        del v248
        print(v60.format('['),end="")
        v250 = 0
        while method7(v250):
            v252 = v241
            v253 = v252 >= 100
            del v252
            if v253:
                v254 = " ..."
                print(v60.format(v254),end="")
                del v254
                break
            else:
                pass
            del v253
            v255 = v250 == 0
            v256 = v255 != True
            del v255
            if v256:
                v257 = "; "
                print(v60.format(v257),end="")
                del v257
            else:
                pass
            del v256
            print(v60.format('['),end="")
            v258 = 0
            while method7(v258):
                v260 = v241
                v261 = v260 >= 100
                del v260
                if v261:
                    v262 = " ..."
                    print(v60.format(v262),end="")
                    del v262
                    break
                else:
                    pass
                del v261
                v263 = v258 == 0
                v264 = v263 != True
                del v263
                if v264:
                    v265 = "; "
                    print(v60.format(v265),end="")
                    del v265
                else:
                    pass
                del v264
                v266 = v241 + 1
                v241 = v266
                del v266
                v267 = v242 * 4096
                v268 = v250 * 64
                v269 = v267 + v268
                del v267, v268
                v270 = v269 + v258
                del v269
                v271 = v212[v270].item()
                del v270
                v272 = "{:.6f}"
                print(v272.format(v271),end="")
                del v271, v272
                v258 += 1 
            del v258
            print(v60.format(']'),end="")
            v250 += 1 
        del v250
        print(v60.format(']'),end="")
        v242 += 1 
    del v212, v241, v242
    print(v60.format(']'),end="")
    print(v92.format(),end="")
    v275 = "Here is the output tensor."
    print(v24.format(v275),end="")
    del v24, v275
    v276 = cp.cuda.Device().attributes['MultiProcessorCount']
    v277 = v276 == 24
    del v276
    v278 = v277 == False
    if v278:
        v279 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v277, v279
        del v279
    else:
        pass
    del v277, v278
    v280 = 1
    v281 = raw_module.get_function(f"entry{v280}")
    del v280
    v281.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v281((24,),(256,),(v6, v5, v4),shared_mem=98304)
    del v4, v6, v281
    v283 = v5[9437184:9437184+4*24576].view(cp.int32)
    del v5
    v317 = 0
    print(v60.format('['),end="")
    v318 = 0
    while method9(v318):
        v320 = v317
        v321 = v320 >= 2147483647
        del v320
        if v321:
            v322 = " ..."
            print(v60.format(v322),end="")
            del v322
            break
        else:
            pass
        del v321
        v323 = v318 == 0
        v324 = v323 != True
        del v323
        if v324:
            v325 = "; "
            print(v60.format(v325),end="")
            del v325
        else:
            pass
        del v324
        print(v60.format('['),end="")
        v326 = 0
        while method4(v326):
            v328 = v317
            v329 = v328 >= 2147483647
            del v328
            if v329:
                v330 = " ..."
                print(v60.format(v330),end="")
                del v330
                break
            else:
                pass
            del v329
            v331 = v326 == 0
            v332 = v331 != True
            del v331
            if v332:
                v333 = "; "
                print(v60.format(v333),end="")
                del v333
            else:
                pass
            del v332
            print(v60.format('['),end="")
            v334 = 0
            while method7(v334):
                v336 = v317
                v337 = v336 >= 2147483647
                del v336
                if v337:
                    v338 = " ..."
                    print(v60.format(v338),end="")
                    del v338
                    break
                else:
                    pass
                del v337
                v339 = v334 == 0
                v340 = v339 != True
                del v339
                if v340:
                    v341 = "; "
                    print(v60.format(v341),end="")
                    del v341
                else:
                    pass
                del v340
                v342 = v317 + 1
                v317 = v342
                del v342
                v343 = v318 * 1536
                v344 = v326 * 64
                v345 = v343 + v344
                del v343, v344
                v346 = v345 + v334
                del v345
                v347 = v283[v346].item()
                del v346
                print(v60.format(v347),end="")
                del v347
                v334 += 1 
            del v334
            print(v60.format(']'),end="")
            v326 += 1 
        del v326
        print(v60.format(']'),end="")
        v318 += 1 
    del v283, v317, v318
    print(v60.format(']'),end="")
    del v60
    print(v92.format(),end="")
    del v92
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method10() -> None:
    v0 = "test_text_outputs/layers/"
    v1 = "test5"
    v2 = "layers.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.empty(0,dtype=cp.uint8)
    v5 = cp.empty(2383872,dtype=cp.uint8)
    v6 = cp.empty(65536,dtype=cp.uint8)
    v8 = v5[0:0+4*98304].view(cp.float32)
    del v8
    v10 = v6[0:0+4*16384].view(cp.float32)
    v12 = v5[786432:786432+4*393216].view(cp.float32)
    del v12
    v14 = v5[2359296:2359296+4*6144].view(cp.int32)
    del v14
    v15 = cp.random.normal(0.0,0.0078125,16384,dtype=cp.float32) # type: ignore
    cp.copyto(v10[0:0+16384],v15[0:0+16384])
    del v10, v15
    v17 = v5[0:0+4*98304].view(cp.float32)
    v18 = cp.random.normal(0.0,1.0,98304,dtype=cp.float32) # type: ignore
    cp.copyto(v17[0:0+98304],v18[0:0+98304])
    del v17, v18
    v19 = cp.cuda.Device().attributes['MultiProcessorCount']
    v20 = v19 == 24
    del v19
    v21 = v20 == False
    if v21:
        v22 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v20, v22
        del v22
    else:
        pass
    del v20, v21
    v23 = 2
    v24 = raw_module.get_function(f"entry{v23}")
    del v23
    v24.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v24((24,),(256,),(v6, v5, v4),shared_mem=98304)
    del v4, v6, v24
    v26 = v5[2359296:2359296+4*6144].view(cp.int32)
    del v5
    v62 = 0
    v63 = "{}"
    print(v63.format('['),end="")
    v64 = 0
    while method1(v64):
        v66 = v62
        v67 = v66 >= 2147483647
        del v66
        if v67:
            v68 = " ..."
            print(v63.format(v68),end="")
            del v68
            break
        else:
            pass
        del v67
        v69 = v64 == 0
        v70 = v69 != True
        del v69
        if v70:
            v71 = "; "
            print(v63.format(v71),end="")
            del v71
        else:
            pass
        del v70
        print(v63.format('['),end="")
        v72 = 0
        while method4(v72):
            v74 = v62
            v75 = v74 >= 2147483647
            del v74
            if v75:
                v76 = " ..."
                print(v63.format(v76),end="")
                del v76
                break
            else:
                pass
            del v75
            v77 = v72 == 0
            v78 = v77 != True
            del v77
            if v78:
                v79 = "; "
                print(v63.format(v79),end="")
                del v79
            else:
                pass
            del v78
            print(v63.format('['),end="")
            v80 = 0
            while method7(v80):
                v82 = v62
                v83 = v82 >= 2147483647
                del v82
                if v83:
                    v84 = " ..."
                    print(v63.format(v84),end="")
                    del v84
                    break
                else:
                    pass
                del v83
                v85 = v80 == 0
                v86 = v85 != True
                del v85
                if v86:
                    v87 = "; "
                    print(v63.format(v87),end="")
                    del v87
                else:
                    pass
                del v86
                v88 = v62 + 1
                v62 = v88
                del v88
                v89 = v64 * 1536
                v90 = v72 * 64
                v91 = v89 + v90
                del v89, v90
                v92 = v91 + v80
                del v91
                v93 = v26[v92].item()
                del v92
                print(v63.format(v93),end="")
                del v93
                v80 += 1 
            del v80
            print(v63.format(']'),end="")
            v72 += 1 
        del v72
        print(v63.format(']'),end="")
        v64 += 1 
    del v26, v62, v64
    print(v63.format(']'),end="")
    del v63
    v94 = "\n"
    print(v94.format(),end="")
    del v94
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def main_body():
    cp.random.seed(12344321)
    method0()
    cp.random.seed(12344321)
    method3()
    cp.random.seed(12344321)
    method6()
    cp.random.seed(12344321)
    method8()
    cp.random.seed(12344321)
    return method10()

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
