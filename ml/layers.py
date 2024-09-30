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
__device__ void method_0(float * v0, float * v1);
__device__ void method_1(float * v0, float * v1);
struct Tuple0;
struct Tuple1;
struct Tuple2;
struct Tuple3;
__device__ void method_2(int * v0, float * v1, float * v2, curandStatePhilox4_32_10_t & v3);
__device__ void method_3(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5);
__device__ void method_4(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5);
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
__device__ void method_0(float * v0, float * v1){
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
__device__ void method_1(float * v0, float * v1){
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
__device__ void method_2(int * v0, float * v1, float * v2, curandStatePhilox4_32_10_t & v3){
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
__device__ void method_3(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5){
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
__device__ void method_4(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5){
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
    v9 = reinterpret_cast<float *>(&v1[0ull]);
    float * v11;
    v11 = reinterpret_cast<float *>(&v0[0ull]);
    float * v13;
    v13 = reinterpret_cast<float *>(&v1[393216ull]);
    int v15;
    v15 = blockIdx.x;
    assert("Tensor range check" && 0 <= v15 && v15 < 24);
    int v16;
    v16 = 4096 * v15;
    int v17;
    v17 = blockIdx.x;
    assert("Tensor range check" && 0 <= v17 && v17 < 24);
    int v18;
    v18 = 4096 * v17;
    cuda::pipeline<cuda::thread_scope_thread> v19 = cuda::make_pipeline();
    extern __shared__ unsigned char v20[];
    float * v21;
    v21 = reinterpret_cast<float *>(&v20[0ull]);
    float * v23;
    v23 = reinterpret_cast<float *>(&v20[17408ull]);
    float * v25;
    v25 = reinterpret_cast<float *>(&v20[0ull]);
    int v27;
    v27 = threadIdx.x;
    int v28;
    v28 = v27 / 32;
    bool v29;
    v29 = 0 <= v28;
    bool v30;
    v30 = v29 == false;
    if (v30){
        assert("The index needs to be zero or positive." && v29);
    } else {
    }
    int v32;
    v32 = v28 % 4;
    int v33;
    v33 = v28 / 4;
    bool v34;
    v34 = v33 < 2;
    bool v35;
    v35 = v34 == false;
    if (v35){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v34);
    } else {
    }
    assert("Tensor range check" && 0 <= v33 && v33 < 2);
    assert("Tensor range check" && 0 <= v32 && v32 < 4);
    int v37;
    v37 = 16 * v32;
    int v38;
    v38 = 2304 * v33;
    int v39;
    v39 = v38 + v37;
    float * v40;
    v40 = v25+v39;
    assert("Tensor range check" && 0 <= v33 && v33 < 2);
    int v42;
    v42 = 2176 * v33;
    int v43;
    v43 = threadIdx.x;
    int v44;
    v44 = v43 % 32;
    bool v45;
    v45 = 0 <= v44;
    bool v46;
    v46 = v45 == false;
    if (v46){
        assert("The index needs to be zero or positive." && v45);
    } else {
    }
    int v48;
    v48 = v44 % 4;
    int v49;
    v49 = v44 / 4;
    bool v50;
    v50 = v49 < 8;
    bool v51;
    v51 = v50 == false;
    if (v51){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v50);
    } else {
    }
    assert("Tensor range check" && 0 <= v49 && v49 < 8);
    assert("Tensor range check" && 0 <= v48 && v48 < 4);
    int v53;
    v53 = v48 + v42;
    int v54;
    v54 = 68 * v49;
    int v55;
    v55 = v54 + v53;
    float * v56;
    v56 = v21+v55;
    assert("Tensor range check" && 0 <= v32 && v32 < 4);
    int v58;
    v58 = 1088 * v32;
    int v59;
    v59 = threadIdx.x;
    int v60;
    v60 = v59 % 32;
    bool v61;
    v61 = 0 <= v60;
    bool v62;
    v62 = v61 == false;
    if (v62){
        assert("The index needs to be zero or positive." && v61);
    } else {
    }
    int v64;
    v64 = v60 % 4;
    int v65;
    v65 = v60 / 4;
    bool v66;
    v66 = v65 < 8;
    bool v67;
    v67 = v66 == false;
    if (v67){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v66);
    } else {
    }
    assert("Tensor range check" && 0 <= v65 && v65 < 8);
    assert("Tensor range check" && 0 <= v64 && v64 < 4);
    int v69;
    v69 = v64 + v58;
    int v70;
    v70 = 68 * v65;
    int v71;
    v71 = v70 + v69;
    float * v72;
    v72 = v23+v71;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v74[2];
    int v75;
    v75 = 0;
    while (while_method_0(v75)){
        int v77;
        v77 = 0;
        while (while_method_0(v77)){
            assert("Tensor range check" && 0 <= v75 && v75 < 1);
            assert("Tensor range check" && 0 <= v77 && v77 < 1);
            int v79;
            v79 = 64 * v77;
            int v80;
            v80 = v79 + v18;
            int v81;
            v81 = 4096 * v75;
            int v82;
            v82 = v81 + v80;
            float * v83;
            v83 = v13+v82;
            // Pushing the loop unrolling to: 0
            int v85;
            v85 = 0;
            #pragma unroll
            while (while_method_1(v85)){
                int v87;
                v87 = 0;
                #pragma unroll
                while (while_method_0(v87)){
                    assert("Tensor range check" && 0 <= v85 && v85 < 2);
                    assert("Tensor range check" && 0 <= v87 && v87 < 1);
                    int v89;
                    v89 = v85 + v87;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v90 = v74[v89];
                    wmma::fill_fragment(v90, 0.0f);
                    v87 += 1 ;
                }
                v85 += 1 ;
            }
            // Poping the loop unrolling to: 0
            int v91;
            v91 = 0;
            while (while_method_2(v91)){
                int v93;
                v93 = v91 + 1;
                bool v94;
                v94 = v91 == 0;
                int v95;
                v95 = v91 % 2;
                bool v96;
                v96 = 0 <= v91;
                bool v97;
                v97 = v96 == false;
                if (v97){
                    assert("The index needs to be zero or positive." && v96);
                } else {
                }
                bool v99;
                v99 = v91 < 1;
                bool v100;
                v100 = v99 == false;
                if (v100){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v99);
                } else {
                }
                bool v102;
                v102 = v93 < 1;
                Union0 v108;
                if (v102){
                    bool v103;
                    v103 = 0 <= v93;
                    bool v104;
                    v104 = v103 == false;
                    if (v104){
                        assert("The index needs to be zero or positive." && v103);
                    } else {
                    }
                    v108 = Union0{Union0_1{v93}};
                } else {
                    v108 = Union0{Union0_0{}};
                }
                assert("Tensor range check" && 0 <= v75 && v75 < 1);
                int v109;
                v109 = v81 + v16;
                assert("Tensor range check" && 0 <= v91 && v91 < 1);
                int v110;
                v110 = 64 * v91;
                int v111;
                v111 = v110 + v109;
                float * v112;
                v112 = v9+v111;
                assert("Tensor range check" && 0 <= v77 && v77 < 1);
                int v114;
                v114 = 4096 * v77;
                if (v94){
                    assert("Tensor range check" && 0 <= v91 && v91 < 1);
                    int v115;
                    v115 = v110 + v114;
                    float * v116;
                    v116 = v11+v115;
                    // Pushing the loop unrolling to: 0
                    v19.producer_acquire();
                    int v118;
                    v118 = threadIdx.x;
                    bool v119;
                    v119 = 0 <= v118;
                    bool v120;
                    v120 = v119 == false;
                    if (v120){
                        assert("The index needs to be zero or positive." && v119);
                    } else {
                    }
                    int v122;
                    v122 = v118 % 16;
                    int v123;
                    v123 = v118 / 16;
                    bool v124;
                    v124 = v123 < 16;
                    bool v125;
                    v125 = v124 == false;
                    if (v125){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v124);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v123 && v123 < 16);
                    assert("Tensor range check" && 0 <= v122 && v122 < 16);
                    int v127;
                    v127 = 4 * v122;
                    int v128;
                    v128 = 68 * v123;
                    int v129;
                    v129 = v128 + v127;
                    int v130;
                    v130 = 64 * v123;
                    int v131;
                    v131 = v130 + v127;
                    float * v132;
                    v132 = v23+v129;
                    float * v134;
                    v134 = v116+v131;
                    int v136;
                    v136 = 0;
                    #pragma unroll
                    while (while_method_3(v136)){
                        int v138;
                        v138 = 0;
                        #pragma unroll
                        while (while_method_0(v138)){
                            assert("Tensor range check" && 0 <= v136 && v136 < 4);
                            assert("Tensor range check" && 0 <= v138 && v138 < 1);
                            int v140;
                            v140 = 64 * v138;
                            int v141;
                            v141 = 1088 * v136;
                            int v142;
                            v142 = v141 + v140;
                            int v143;
                            v143 = 1024 * v136;
                            int v144;
                            v144 = v143 + v140;
                            constexpr int v145 = sizeof(float) * 4;
                            assert("Pointer alignment check" && (unsigned long long)(v134 + v144) % v145 == 0 && (unsigned long long)(v132 + v142) % v145 == 0);
                            cuda::memcpy_async(v132 + v142, v134 + v144, cuda::aligned_size_t<v145>(v145), v19);
                            v138 += 1 ;
                        }
                        v136 += 1 ;
                    }
                    v19.producer_commit();
                    // Poping the loop unrolling to: 0
                } else {
                }
                // Pushing the loop unrolling to: 0
                int v146;
                v146 = threadIdx.x;
                bool v147;
                v147 = 0 <= v146;
                bool v148;
                v148 = v147 == false;
                if (v148){
                    assert("The index needs to be zero or positive." && v147);
                } else {
                }
                int v150;
                v150 = v146 % 16;
                int v151;
                v151 = v146 / 16;
                bool v152;
                v152 = v151 < 16;
                bool v153;
                v153 = v152 == false;
                if (v153){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v152);
                } else {
                }
                assert("Tensor range check" && 0 <= v151 && v151 < 16);
                assert("Tensor range check" && 0 <= v150 && v150 < 16);
                int v155;
                v155 = 4 * v150;
                int v156;
                v156 = 68 * v151;
                int v157;
                v157 = v156 + v155;
                int v158;
                v158 = 64 * v151;
                int v159;
                v159 = v158 + v155;
                float * v160;
                v160 = v21+v157;
                float * v162;
                v162 = v112+v159;
                int v164;
                v164 = 0;
                #pragma unroll
                while (while_method_3(v164)){
                    int v166;
                    v166 = 0;
                    #pragma unroll
                    while (while_method_0(v166)){
                        assert("Tensor range check" && 0 <= v164 && v164 < 4);
                        assert("Tensor range check" && 0 <= v166 && v166 < 1);
                        int v168;
                        v168 = 64 * v166;
                        int v169;
                        v169 = 1088 * v164;
                        int v170;
                        v170 = v169 + v168;
                        int v171;
                        v171 = 1024 * v164;
                        int v172;
                        v172 = v171 + v168;
                        int4* v173;
                        v173 = reinterpret_cast<int4*>(v162 + v172);
                        int4* v174;
                        v174 = reinterpret_cast<int4*>(v160 + v170);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v173) % 16 == 0 && reinterpret_cast<unsigned long long>(v174) % 16 == 0);
                        *v174 = *v173;
                        v166 += 1 ;
                    }
                    v164 += 1 ;
                }
                // Poping the loop unrolling to: 0
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v175[1];
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v176[8];
                cuda::pipeline_consumer_wait_prior<0>(v19);;
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v177;
                v177 = 0;
                #pragma unroll
                while (while_method_0(v177)){
                    int v179;
                    v179 = 0;
                    #pragma unroll
                    while (while_method_4(v179)){
                        assert("Tensor range check" && 0 <= v177 && v177 < 1);
                        assert("Tensor range check" && 0 <= v179 && v179 < 8);
                        int v181;
                        v181 = 8 * v177;
                        int v182;
                        v182 = v181 + v179;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v183 = v176[v182];
                        assert("Tensor range check" && 0 <= v177 && v177 < 1);
                        int v184;
                        v184 = 1088 * v177;
                        assert("Tensor range check" && 0 <= v179 && v179 < 8);
                        int v185;
                        v185 = 8 * v179;
                        int v186;
                        v186 = v185 + v184;
                        int v187;
                        v187 = 0;
                        #pragma unroll
                        while (while_method_1(v187)){
                            int v189;
                            v189 = 0;
                            #pragma unroll
                            while (while_method_1(v189)){
                                assert("Tensor range check" && 0 <= v187 && v187 < 2);
                                assert("Tensor range check" && 0 <= v189 && v189 < 2);
                                int v191;
                                v191 = 4 * v189;
                                int v192;
                                v192 = v191 + v186;
                                int v193;
                                v193 = 544 * v187;
                                int v194;
                                v194 = v193 + v192;
                                float v195;
                                v195 = v72[v194];
                                bool v196;
                                v196 = 0 <= v189;
                                bool v198;
                                if (v196){
                                    bool v197;
                                    v197 = v189 < 2;
                                    v198 = v197;
                                } else {
                                    v198 = false;
                                }
                                bool v199;
                                v199 = v198 == false;
                                if (v199){
                                    assert("The indices should be inside the range of the dimension." && v198);
                                } else {
                                }
                                bool v201;
                                v201 = 0 <= v187;
                                bool v203;
                                if (v201){
                                    bool v202;
                                    v202 = v187 < 2;
                                    v203 = v202;
                                } else {
                                    v203 = false;
                                }
                                bool v204;
                                v204 = v203 == false;
                                if (v204){
                                    assert("The indices should be inside the range of the dimension." && v203);
                                } else {
                                }
                                int v206;
                                v206 = v187 * 2;
                                int v207;
                                v207 = v189 + v206;
                                v183.x[v207] = wmma::__float_to_tf32(v195);
                                v189 += 1 ;
                            }
                            v187 += 1 ;
                        }
                        v179 += 1 ;
                    }
                    v177 += 1 ;
                }
                // Poping the loop unrolling to: 0
                v19.consumer_release();
                switch (v108.tag) {
                    case 0: { // None
                        break;
                    }
                    case 1: { // Some
                        int v208 = v108.case1.v0;
                        assert("Tensor range check" && 0 <= v208 && v208 < 1);
                        int v209;
                        v209 = 64 * v208;
                        int v210;
                        v210 = v209 + v114;
                        float * v211;
                        v211 = v11+v210;
                        __syncthreads();
                        // Pushing the loop unrolling to: 0
                        v19.producer_acquire();
                        int v213;
                        v213 = threadIdx.x;
                        bool v214;
                        v214 = 0 <= v213;
                        bool v215;
                        v215 = v214 == false;
                        if (v215){
                            assert("The index needs to be zero or positive." && v214);
                        } else {
                        }
                        int v217;
                        v217 = v213 % 16;
                        int v218;
                        v218 = v213 / 16;
                        bool v219;
                        v219 = v218 < 16;
                        bool v220;
                        v220 = v219 == false;
                        if (v220){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v219);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v218 && v218 < 16);
                        assert("Tensor range check" && 0 <= v217 && v217 < 16);
                        int v222;
                        v222 = 4 * v217;
                        int v223;
                        v223 = 68 * v218;
                        int v224;
                        v224 = v223 + v222;
                        int v225;
                        v225 = 64 * v218;
                        int v226;
                        v226 = v225 + v222;
                        float * v227;
                        v227 = v23+v224;
                        float * v229;
                        v229 = v211+v226;
                        int v231;
                        v231 = 0;
                        #pragma unroll
                        while (while_method_3(v231)){
                            int v233;
                            v233 = 0;
                            #pragma unroll
                            while (while_method_0(v233)){
                                assert("Tensor range check" && 0 <= v231 && v231 < 4);
                                assert("Tensor range check" && 0 <= v233 && v233 < 1);
                                int v235;
                                v235 = 64 * v233;
                                int v236;
                                v236 = 1088 * v231;
                                int v237;
                                v237 = v236 + v235;
                                int v238;
                                v238 = 1024 * v231;
                                int v239;
                                v239 = v238 + v235;
                                constexpr int v240 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v229 + v239) % v240 == 0 && (unsigned long long)(v227 + v237) % v240 == 0);
                                cuda::memcpy_async(v227 + v237, v229 + v239, cuda::aligned_size_t<v240>(v240), v19);
                                v233 += 1 ;
                            }
                            v231 += 1 ;
                        }
                        v19.producer_commit();
                        // Poping the loop unrolling to: 0
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                // Pushing the loop unrolling to: 0
                int v241;
                v241 = 0;
                #pragma unroll
                while (while_method_1(v241)){
                    int v243;
                    v243 = 0;
                    #pragma unroll
                    while (while_method_4(v243)){
                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v245 = v175[0];
                        assert("Tensor range check" && 0 <= v241 && v241 < 2);
                        int v246;
                        v246 = 1088 * v241;
                        assert("Tensor range check" && 0 <= v243 && v243 < 8);
                        int v247;
                        v247 = 8 * v243;
                        int v248;
                        v248 = v247 + v246;
                        int v249;
                        v249 = 0;
                        #pragma unroll
                        while (while_method_1(v249)){
                            int v251;
                            v251 = 0;
                            #pragma unroll
                            while (while_method_1(v251)){
                                assert("Tensor range check" && 0 <= v249 && v249 < 2);
                                assert("Tensor range check" && 0 <= v251 && v251 < 2);
                                int v253;
                                v253 = 544 * v251;
                                int v254;
                                v254 = v253 + v248;
                                int v255;
                                v255 = 4 * v249;
                                int v256;
                                v256 = v255 + v254;
                                float v257;
                                v257 = v56[v256];
                                bool v258;
                                v258 = 0 <= v251;
                                bool v260;
                                if (v258){
                                    bool v259;
                                    v259 = v251 < 2;
                                    v260 = v259;
                                } else {
                                    v260 = false;
                                }
                                bool v261;
                                v261 = v260 == false;
                                if (v261){
                                    assert("The indices should be inside the range of the dimension." && v260);
                                } else {
                                }
                                bool v263;
                                v263 = 0 <= v249;
                                bool v265;
                                if (v263){
                                    bool v264;
                                    v264 = v249 < 2;
                                    v265 = v264;
                                } else {
                                    v265 = false;
                                }
                                bool v266;
                                v266 = v265 == false;
                                if (v266){
                                    assert("The indices should be inside the range of the dimension." && v265);
                                } else {
                                }
                                int v268;
                                v268 = v249 * 2;
                                int v269;
                                v269 = v251 + v268;
                                v245.x[v269] = wmma::__float_to_tf32(v257);
                                v251 += 1 ;
                            }
                            v249 += 1 ;
                        }
                        int v270;
                        v270 = 0;
                        #pragma unroll
                        while (while_method_0(v270)){
                            assert("Tensor range check" && 0 <= v241 && v241 < 2);
                            assert("Tensor range check" && 0 <= v270 && v270 < 1);
                            int v272;
                            v272 = v241 + v270;
                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v273 = v74[v272];
                            assert("Tensor range check" && 0 <= v270 && v270 < 1);
                            assert("Tensor range check" && 0 <= v243 && v243 < 8);
                            int v274;
                            v274 = 8 * v270;
                            int v275;
                            v275 = v274 + v243;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v276 = v176[v275];
                            wmma::mma_sync(v273, v245, v276, v273);
                            v270 += 1 ;
                        }
                        v243 += 1 ;
                    }
                    v241 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v91 = v93;
            }
            // Pushing the loop unrolling to: 0
            int v277;
            v277 = 0;
            #pragma unroll
            while (while_method_1(v277)){
                int v279;
                v279 = 0;
                #pragma unroll
                while (while_method_0(v279)){
                    assert("Tensor range check" && 0 <= v277 && v277 < 2);
                    assert("Tensor range check" && 0 <= v279 && v279 < 1);
                    int v281;
                    v281 = v277 + v279;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v282 = v74[v281];
                    assert("Tensor range check" && 0 <= v277 && v277 < 2);
                    assert("Tensor range check" && 0 <= v279 && v279 < 1);
                    int v283;
                    v283 = 16 * v279;
                    int v284;
                    v284 = 1152 * v277;
                    int v285;
                    v285 = v284 + v283;
                    float * v286;
                    v286 = v40+v285;
                    wmma::store_matrix_sync(v286, v282, 72, wmma::mem_row_major);
                    v279 += 1 ;
                }
                v277 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            // Pushing the loop unrolling to: 0
            int v288;
            v288 = threadIdx.x;
            bool v289;
            v289 = 0 <= v288;
            bool v290;
            v290 = v289 == false;
            if (v290){
                assert("The index needs to be zero or positive." && v289);
            } else {
            }
            int v292;
            v292 = v288 % 16;
            int v293;
            v293 = v288 / 16;
            bool v294;
            v294 = v293 < 16;
            bool v295;
            v295 = v294 == false;
            if (v295){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v294);
            } else {
            }
            assert("Tensor range check" && 0 <= v293 && v293 < 16);
            assert("Tensor range check" && 0 <= v292 && v292 < 16);
            int v297;
            v297 = 4 * v292;
            int v298;
            v298 = 64 * v293;
            int v299;
            v299 = v298 + v297;
            int v300;
            v300 = 72 * v293;
            int v301;
            v301 = v300 + v297;
            float * v302;
            v302 = v83+v299;
            float * v304;
            v304 = v25+v301;
            int v306;
            v306 = 0;
            #pragma unroll
            while (while_method_3(v306)){
                int v308;
                v308 = 0;
                #pragma unroll
                while (while_method_0(v308)){
                    assert("Tensor range check" && 0 <= v306 && v306 < 4);
                    assert("Tensor range check" && 0 <= v308 && v308 < 1);
                    int v310;
                    v310 = 64 * v308;
                    int v311;
                    v311 = 1024 * v306;
                    int v312;
                    v312 = v311 + v310;
                    int v313;
                    v313 = 1152 * v306;
                    int v314;
                    v314 = v313 + v310;
                    int4* v315;
                    v315 = reinterpret_cast<int4*>(v304 + v314);
                    int4* v316;
                    v316 = reinterpret_cast<int4*>(v302 + v312);
                    assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v315) % 16 == 0 && reinterpret_cast<unsigned long long>(v316) % 16 == 0);
                    *v316 = *v315;
                    v308 += 1 ;
                }
                v306 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            v77 += 1 ;
        }
        v75 += 1 ;
    }
    float * v317;
    v317 = reinterpret_cast<float *>(&v1[786432ull]);
    method_0(v317, v13);
    float * v319;
    v319 = reinterpret_cast<float *>(&v1[1179648ull]);
    method_1(v319, v317);
    float * v321;
    v321 = reinterpret_cast<float *>(&v0[16384ull]);
    float * v323;
    v323 = reinterpret_cast<float *>(&v1[1572864ull]);
    int v325;
    v325 = blockIdx.x;
    assert("Tensor range check" && 0 <= v325 && v325 < 24);
    int v326;
    v326 = 4096 * v325;
    int v327;
    v327 = blockIdx.x;
    assert("Tensor range check" && 0 <= v327 && v327 < 24);
    int v328;
    v328 = 4096 * v327;
    cuda::pipeline<cuda::thread_scope_thread> v329 = cuda::make_pipeline();
    extern __shared__ unsigned char v330[];
    float * v331;
    v331 = reinterpret_cast<float *>(&v330[0ull]);
    float * v333;
    v333 = reinterpret_cast<float *>(&v330[17408ull]);
    float * v335;
    v335 = reinterpret_cast<float *>(&v330[0ull]);
    int v337;
    v337 = threadIdx.x;
    int v338;
    v338 = v337 / 32;
    bool v339;
    v339 = 0 <= v338;
    bool v340;
    v340 = v339 == false;
    if (v340){
        assert("The index needs to be zero or positive." && v339);
    } else {
    }
    int v342;
    v342 = v338 % 4;
    int v343;
    v343 = v338 / 4;
    bool v344;
    v344 = v343 < 2;
    bool v345;
    v345 = v344 == false;
    if (v345){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v344);
    } else {
    }
    assert("Tensor range check" && 0 <= v343 && v343 < 2);
    assert("Tensor range check" && 0 <= v342 && v342 < 4);
    int v347;
    v347 = 16 * v342;
    int v348;
    v348 = 2304 * v343;
    int v349;
    v349 = v348 + v347;
    float * v350;
    v350 = v335+v349;
    assert("Tensor range check" && 0 <= v343 && v343 < 2);
    int v352;
    v352 = 2176 * v343;
    int v353;
    v353 = threadIdx.x;
    int v354;
    v354 = v353 % 32;
    bool v355;
    v355 = 0 <= v354;
    bool v356;
    v356 = v355 == false;
    if (v356){
        assert("The index needs to be zero or positive." && v355);
    } else {
    }
    int v358;
    v358 = v354 % 4;
    int v359;
    v359 = v354 / 4;
    bool v360;
    v360 = v359 < 8;
    bool v361;
    v361 = v360 == false;
    if (v361){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v360);
    } else {
    }
    assert("Tensor range check" && 0 <= v359 && v359 < 8);
    assert("Tensor range check" && 0 <= v358 && v358 < 4);
    int v363;
    v363 = v358 + v352;
    int v364;
    v364 = 68 * v359;
    int v365;
    v365 = v364 + v363;
    float * v366;
    v366 = v331+v365;
    assert("Tensor range check" && 0 <= v342 && v342 < 4);
    int v368;
    v368 = 1088 * v342;
    int v369;
    v369 = threadIdx.x;
    int v370;
    v370 = v369 % 32;
    bool v371;
    v371 = 0 <= v370;
    bool v372;
    v372 = v371 == false;
    if (v372){
        assert("The index needs to be zero or positive." && v371);
    } else {
    }
    int v374;
    v374 = v370 % 4;
    int v375;
    v375 = v370 / 4;
    bool v376;
    v376 = v375 < 8;
    bool v377;
    v377 = v376 == false;
    if (v377){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v376);
    } else {
    }
    assert("Tensor range check" && 0 <= v375 && v375 < 8);
    assert("Tensor range check" && 0 <= v374 && v374 < 4);
    int v379;
    v379 = v374 + v368;
    int v380;
    v380 = 68 * v375;
    int v381;
    v381 = v380 + v379;
    float * v382;
    v382 = v333+v381;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v384[2];
    int v385;
    v385 = 0;
    while (while_method_0(v385)){
        int v387;
        v387 = 0;
        while (while_method_0(v387)){
            assert("Tensor range check" && 0 <= v385 && v385 < 1);
            assert("Tensor range check" && 0 <= v387 && v387 < 1);
            int v389;
            v389 = 64 * v387;
            int v390;
            v390 = v389 + v328;
            int v391;
            v391 = 4096 * v385;
            int v392;
            v392 = v391 + v390;
            float * v393;
            v393 = v323+v392;
            // Pushing the loop unrolling to: 0
            int v395;
            v395 = 0;
            #pragma unroll
            while (while_method_1(v395)){
                int v397;
                v397 = 0;
                #pragma unroll
                while (while_method_0(v397)){
                    assert("Tensor range check" && 0 <= v395 && v395 < 2);
                    assert("Tensor range check" && 0 <= v397 && v397 < 1);
                    int v399;
                    v399 = v395 + v397;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v400 = v384[v399];
                    wmma::fill_fragment(v400, 0.0f);
                    v397 += 1 ;
                }
                v395 += 1 ;
            }
            // Poping the loop unrolling to: 0
            int v401;
            v401 = 0;
            while (while_method_2(v401)){
                int v403;
                v403 = v401 + 1;
                bool v404;
                v404 = v401 == 0;
                int v405;
                v405 = v401 % 2;
                bool v406;
                v406 = 0 <= v401;
                bool v407;
                v407 = v406 == false;
                if (v407){
                    assert("The index needs to be zero or positive." && v406);
                } else {
                }
                bool v409;
                v409 = v401 < 1;
                bool v410;
                v410 = v409 == false;
                if (v410){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v409);
                } else {
                }
                bool v412;
                v412 = v403 < 1;
                Union0 v418;
                if (v412){
                    bool v413;
                    v413 = 0 <= v403;
                    bool v414;
                    v414 = v413 == false;
                    if (v414){
                        assert("The index needs to be zero or positive." && v413);
                    } else {
                    }
                    v418 = Union0{Union0_1{v403}};
                } else {
                    v418 = Union0{Union0_0{}};
                }
                assert("Tensor range check" && 0 <= v385 && v385 < 1);
                int v419;
                v419 = v391 + v326;
                assert("Tensor range check" && 0 <= v401 && v401 < 1);
                int v420;
                v420 = 64 * v401;
                int v421;
                v421 = v420 + v419;
                float * v422;
                v422 = v319+v421;
                assert("Tensor range check" && 0 <= v387 && v387 < 1);
                int v424;
                v424 = 4096 * v387;
                if (v404){
                    assert("Tensor range check" && 0 <= v401 && v401 < 1);
                    int v425;
                    v425 = v420 + v424;
                    float * v426;
                    v426 = v321+v425;
                    // Pushing the loop unrolling to: 0
                    v329.producer_acquire();
                    int v428;
                    v428 = threadIdx.x;
                    bool v429;
                    v429 = 0 <= v428;
                    bool v430;
                    v430 = v429 == false;
                    if (v430){
                        assert("The index needs to be zero or positive." && v429);
                    } else {
                    }
                    int v432;
                    v432 = v428 % 16;
                    int v433;
                    v433 = v428 / 16;
                    bool v434;
                    v434 = v433 < 16;
                    bool v435;
                    v435 = v434 == false;
                    if (v435){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v434);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v433 && v433 < 16);
                    assert("Tensor range check" && 0 <= v432 && v432 < 16);
                    int v437;
                    v437 = 4 * v432;
                    int v438;
                    v438 = 68 * v433;
                    int v439;
                    v439 = v438 + v437;
                    int v440;
                    v440 = 64 * v433;
                    int v441;
                    v441 = v440 + v437;
                    float * v442;
                    v442 = v333+v439;
                    float * v444;
                    v444 = v426+v441;
                    int v446;
                    v446 = 0;
                    #pragma unroll
                    while (while_method_3(v446)){
                        int v448;
                        v448 = 0;
                        #pragma unroll
                        while (while_method_0(v448)){
                            assert("Tensor range check" && 0 <= v446 && v446 < 4);
                            assert("Tensor range check" && 0 <= v448 && v448 < 1);
                            int v450;
                            v450 = 64 * v448;
                            int v451;
                            v451 = 1088 * v446;
                            int v452;
                            v452 = v451 + v450;
                            int v453;
                            v453 = 1024 * v446;
                            int v454;
                            v454 = v453 + v450;
                            constexpr int v455 = sizeof(float) * 4;
                            assert("Pointer alignment check" && (unsigned long long)(v444 + v454) % v455 == 0 && (unsigned long long)(v442 + v452) % v455 == 0);
                            cuda::memcpy_async(v442 + v452, v444 + v454, cuda::aligned_size_t<v455>(v455), v329);
                            v448 += 1 ;
                        }
                        v446 += 1 ;
                    }
                    v329.producer_commit();
                    // Poping the loop unrolling to: 0
                } else {
                }
                // Pushing the loop unrolling to: 0
                int v456;
                v456 = threadIdx.x;
                bool v457;
                v457 = 0 <= v456;
                bool v458;
                v458 = v457 == false;
                if (v458){
                    assert("The index needs to be zero or positive." && v457);
                } else {
                }
                int v460;
                v460 = v456 % 16;
                int v461;
                v461 = v456 / 16;
                bool v462;
                v462 = v461 < 16;
                bool v463;
                v463 = v462 == false;
                if (v463){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v462);
                } else {
                }
                assert("Tensor range check" && 0 <= v461 && v461 < 16);
                assert("Tensor range check" && 0 <= v460 && v460 < 16);
                int v465;
                v465 = 4 * v460;
                int v466;
                v466 = 68 * v461;
                int v467;
                v467 = v466 + v465;
                int v468;
                v468 = 64 * v461;
                int v469;
                v469 = v468 + v465;
                float * v470;
                v470 = v331+v467;
                float * v472;
                v472 = v422+v469;
                int v474;
                v474 = 0;
                #pragma unroll
                while (while_method_3(v474)){
                    int v476;
                    v476 = 0;
                    #pragma unroll
                    while (while_method_0(v476)){
                        assert("Tensor range check" && 0 <= v474 && v474 < 4);
                        assert("Tensor range check" && 0 <= v476 && v476 < 1);
                        int v478;
                        v478 = 64 * v476;
                        int v479;
                        v479 = 1088 * v474;
                        int v480;
                        v480 = v479 + v478;
                        int v481;
                        v481 = 1024 * v474;
                        int v482;
                        v482 = v481 + v478;
                        int4* v483;
                        v483 = reinterpret_cast<int4*>(v472 + v482);
                        int4* v484;
                        v484 = reinterpret_cast<int4*>(v470 + v480);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v483) % 16 == 0 && reinterpret_cast<unsigned long long>(v484) % 16 == 0);
                        *v484 = *v483;
                        v476 += 1 ;
                    }
                    v474 += 1 ;
                }
                // Poping the loop unrolling to: 0
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v485[1];
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v486[8];
                cuda::pipeline_consumer_wait_prior<0>(v329);;
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v487;
                v487 = 0;
                #pragma unroll
                while (while_method_0(v487)){
                    int v489;
                    v489 = 0;
                    #pragma unroll
                    while (while_method_4(v489)){
                        assert("Tensor range check" && 0 <= v487 && v487 < 1);
                        assert("Tensor range check" && 0 <= v489 && v489 < 8);
                        int v491;
                        v491 = 8 * v487;
                        int v492;
                        v492 = v491 + v489;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v493 = v486[v492];
                        assert("Tensor range check" && 0 <= v487 && v487 < 1);
                        int v494;
                        v494 = 1088 * v487;
                        assert("Tensor range check" && 0 <= v489 && v489 < 8);
                        int v495;
                        v495 = 8 * v489;
                        int v496;
                        v496 = v495 + v494;
                        int v497;
                        v497 = 0;
                        #pragma unroll
                        while (while_method_1(v497)){
                            int v499;
                            v499 = 0;
                            #pragma unroll
                            while (while_method_1(v499)){
                                assert("Tensor range check" && 0 <= v497 && v497 < 2);
                                assert("Tensor range check" && 0 <= v499 && v499 < 2);
                                int v501;
                                v501 = 4 * v499;
                                int v502;
                                v502 = v501 + v496;
                                int v503;
                                v503 = 544 * v497;
                                int v504;
                                v504 = v503 + v502;
                                float v505;
                                v505 = v382[v504];
                                bool v506;
                                v506 = 0 <= v499;
                                bool v508;
                                if (v506){
                                    bool v507;
                                    v507 = v499 < 2;
                                    v508 = v507;
                                } else {
                                    v508 = false;
                                }
                                bool v509;
                                v509 = v508 == false;
                                if (v509){
                                    assert("The indices should be inside the range of the dimension." && v508);
                                } else {
                                }
                                bool v511;
                                v511 = 0 <= v497;
                                bool v513;
                                if (v511){
                                    bool v512;
                                    v512 = v497 < 2;
                                    v513 = v512;
                                } else {
                                    v513 = false;
                                }
                                bool v514;
                                v514 = v513 == false;
                                if (v514){
                                    assert("The indices should be inside the range of the dimension." && v513);
                                } else {
                                }
                                int v516;
                                v516 = v497 * 2;
                                int v517;
                                v517 = v499 + v516;
                                v493.x[v517] = wmma::__float_to_tf32(v505);
                                v499 += 1 ;
                            }
                            v497 += 1 ;
                        }
                        v489 += 1 ;
                    }
                    v487 += 1 ;
                }
                // Poping the loop unrolling to: 0
                v329.consumer_release();
                switch (v418.tag) {
                    case 0: { // None
                        break;
                    }
                    case 1: { // Some
                        int v518 = v418.case1.v0;
                        assert("Tensor range check" && 0 <= v518 && v518 < 1);
                        int v519;
                        v519 = 64 * v518;
                        int v520;
                        v520 = v519 + v424;
                        float * v521;
                        v521 = v321+v520;
                        __syncthreads();
                        // Pushing the loop unrolling to: 0
                        v329.producer_acquire();
                        int v523;
                        v523 = threadIdx.x;
                        bool v524;
                        v524 = 0 <= v523;
                        bool v525;
                        v525 = v524 == false;
                        if (v525){
                            assert("The index needs to be zero or positive." && v524);
                        } else {
                        }
                        int v527;
                        v527 = v523 % 16;
                        int v528;
                        v528 = v523 / 16;
                        bool v529;
                        v529 = v528 < 16;
                        bool v530;
                        v530 = v529 == false;
                        if (v530){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v529);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v528 && v528 < 16);
                        assert("Tensor range check" && 0 <= v527 && v527 < 16);
                        int v532;
                        v532 = 4 * v527;
                        int v533;
                        v533 = 68 * v528;
                        int v534;
                        v534 = v533 + v532;
                        int v535;
                        v535 = 64 * v528;
                        int v536;
                        v536 = v535 + v532;
                        float * v537;
                        v537 = v333+v534;
                        float * v539;
                        v539 = v521+v536;
                        int v541;
                        v541 = 0;
                        #pragma unroll
                        while (while_method_3(v541)){
                            int v543;
                            v543 = 0;
                            #pragma unroll
                            while (while_method_0(v543)){
                                assert("Tensor range check" && 0 <= v541 && v541 < 4);
                                assert("Tensor range check" && 0 <= v543 && v543 < 1);
                                int v545;
                                v545 = 64 * v543;
                                int v546;
                                v546 = 1088 * v541;
                                int v547;
                                v547 = v546 + v545;
                                int v548;
                                v548 = 1024 * v541;
                                int v549;
                                v549 = v548 + v545;
                                constexpr int v550 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v539 + v549) % v550 == 0 && (unsigned long long)(v537 + v547) % v550 == 0);
                                cuda::memcpy_async(v537 + v547, v539 + v549, cuda::aligned_size_t<v550>(v550), v329);
                                v543 += 1 ;
                            }
                            v541 += 1 ;
                        }
                        v329.producer_commit();
                        // Poping the loop unrolling to: 0
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                // Pushing the loop unrolling to: 0
                int v551;
                v551 = 0;
                #pragma unroll
                while (while_method_1(v551)){
                    int v553;
                    v553 = 0;
                    #pragma unroll
                    while (while_method_4(v553)){
                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v555 = v485[0];
                        assert("Tensor range check" && 0 <= v551 && v551 < 2);
                        int v556;
                        v556 = 1088 * v551;
                        assert("Tensor range check" && 0 <= v553 && v553 < 8);
                        int v557;
                        v557 = 8 * v553;
                        int v558;
                        v558 = v557 + v556;
                        int v559;
                        v559 = 0;
                        #pragma unroll
                        while (while_method_1(v559)){
                            int v561;
                            v561 = 0;
                            #pragma unroll
                            while (while_method_1(v561)){
                                assert("Tensor range check" && 0 <= v559 && v559 < 2);
                                assert("Tensor range check" && 0 <= v561 && v561 < 2);
                                int v563;
                                v563 = 544 * v561;
                                int v564;
                                v564 = v563 + v558;
                                int v565;
                                v565 = 4 * v559;
                                int v566;
                                v566 = v565 + v564;
                                float v567;
                                v567 = v366[v566];
                                bool v568;
                                v568 = 0 <= v561;
                                bool v570;
                                if (v568){
                                    bool v569;
                                    v569 = v561 < 2;
                                    v570 = v569;
                                } else {
                                    v570 = false;
                                }
                                bool v571;
                                v571 = v570 == false;
                                if (v571){
                                    assert("The indices should be inside the range of the dimension." && v570);
                                } else {
                                }
                                bool v573;
                                v573 = 0 <= v559;
                                bool v575;
                                if (v573){
                                    bool v574;
                                    v574 = v559 < 2;
                                    v575 = v574;
                                } else {
                                    v575 = false;
                                }
                                bool v576;
                                v576 = v575 == false;
                                if (v576){
                                    assert("The indices should be inside the range of the dimension." && v575);
                                } else {
                                }
                                int v578;
                                v578 = v559 * 2;
                                int v579;
                                v579 = v561 + v578;
                                v555.x[v579] = wmma::__float_to_tf32(v567);
                                v561 += 1 ;
                            }
                            v559 += 1 ;
                        }
                        int v580;
                        v580 = 0;
                        #pragma unroll
                        while (while_method_0(v580)){
                            assert("Tensor range check" && 0 <= v551 && v551 < 2);
                            assert("Tensor range check" && 0 <= v580 && v580 < 1);
                            int v582;
                            v582 = v551 + v580;
                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v583 = v384[v582];
                            assert("Tensor range check" && 0 <= v580 && v580 < 1);
                            assert("Tensor range check" && 0 <= v553 && v553 < 8);
                            int v584;
                            v584 = 8 * v580;
                            int v585;
                            v585 = v584 + v553;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v586 = v486[v585];
                            wmma::mma_sync(v583, v555, v586, v583);
                            v580 += 1 ;
                        }
                        v553 += 1 ;
                    }
                    v551 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v401 = v403;
            }
            // Pushing the loop unrolling to: 0
            int v587;
            v587 = 0;
            #pragma unroll
            while (while_method_1(v587)){
                int v589;
                v589 = 0;
                #pragma unroll
                while (while_method_0(v589)){
                    assert("Tensor range check" && 0 <= v587 && v587 < 2);
                    assert("Tensor range check" && 0 <= v589 && v589 < 1);
                    int v591;
                    v591 = v587 + v589;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v592 = v384[v591];
                    assert("Tensor range check" && 0 <= v587 && v587 < 2);
                    assert("Tensor range check" && 0 <= v589 && v589 < 1);
                    int v593;
                    v593 = 16 * v589;
                    int v594;
                    v594 = 1152 * v587;
                    int v595;
                    v595 = v594 + v593;
                    float * v596;
                    v596 = v350+v595;
                    wmma::store_matrix_sync(v596, v592, 72, wmma::mem_row_major);
                    v589 += 1 ;
                }
                v587 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            // Pushing the loop unrolling to: 0
            int v598;
            v598 = threadIdx.x;
            bool v599;
            v599 = 0 <= v598;
            bool v600;
            v600 = v599 == false;
            if (v600){
                assert("The index needs to be zero or positive." && v599);
            } else {
            }
            int v602;
            v602 = v598 % 16;
            int v603;
            v603 = v598 / 16;
            bool v604;
            v604 = v603 < 16;
            bool v605;
            v605 = v604 == false;
            if (v605){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v604);
            } else {
            }
            assert("Tensor range check" && 0 <= v603 && v603 < 16);
            assert("Tensor range check" && 0 <= v602 && v602 < 16);
            int v607;
            v607 = 4 * v602;
            int v608;
            v608 = 64 * v603;
            int v609;
            v609 = v608 + v607;
            int v610;
            v610 = 72 * v603;
            int v611;
            v611 = v610 + v607;
            float * v612;
            v612 = v393+v609;
            float * v614;
            v614 = v335+v611;
            int v616;
            v616 = 0;
            #pragma unroll
            while (while_method_3(v616)){
                int v618;
                v618 = 0;
                #pragma unroll
                while (while_method_0(v618)){
                    assert("Tensor range check" && 0 <= v616 && v616 < 4);
                    assert("Tensor range check" && 0 <= v618 && v618 < 1);
                    int v620;
                    v620 = 64 * v618;
                    int v621;
                    v621 = 1024 * v616;
                    int v622;
                    v622 = v621 + v620;
                    int v623;
                    v623 = 1152 * v616;
                    int v624;
                    v624 = v623 + v620;
                    int4* v625;
                    v625 = reinterpret_cast<int4*>(v614 + v624);
                    int4* v626;
                    v626 = reinterpret_cast<int4*>(v612 + v622);
                    assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v625) % 16 == 0 && reinterpret_cast<unsigned long long>(v626) % 16 == 0);
                    *v626 = *v625;
                    v618 += 1 ;
                }
                v616 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            v387 += 1 ;
        }
        v385 += 1 ;
    }
    float * v627;
    v627 = reinterpret_cast<float *>(&v1[1966080ull]);
    method_0(v627, v323);
    float * v629;
    v629 = reinterpret_cast<float *>(&v1[2359296ull]);
    method_1(v629, v627);
    float * v631;
    v631 = reinterpret_cast<float *>(&v0[32768ull]);
    float * v633;
    v633 = reinterpret_cast<float *>(&v1[2752512ull]);
    int v635;
    v635 = blockIdx.x;
    assert("Tensor range check" && 0 <= v635 && v635 < 24);
    int v636;
    v636 = 4096 * v635;
    int v637;
    v637 = blockIdx.x;
    assert("Tensor range check" && 0 <= v637 && v637 < 24);
    int v638;
    v638 = 4096 * v637;
    cuda::pipeline<cuda::thread_scope_thread> v639 = cuda::make_pipeline();
    extern __shared__ unsigned char v640[];
    float * v641;
    v641 = reinterpret_cast<float *>(&v640[0ull]);
    float * v643;
    v643 = reinterpret_cast<float *>(&v640[17408ull]);
    float * v645;
    v645 = reinterpret_cast<float *>(&v640[0ull]);
    int v647;
    v647 = threadIdx.x;
    int v648;
    v648 = v647 / 32;
    bool v649;
    v649 = 0 <= v648;
    bool v650;
    v650 = v649 == false;
    if (v650){
        assert("The index needs to be zero or positive." && v649);
    } else {
    }
    int v652;
    v652 = v648 % 4;
    int v653;
    v653 = v648 / 4;
    bool v654;
    v654 = v653 < 2;
    bool v655;
    v655 = v654 == false;
    if (v655){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v654);
    } else {
    }
    assert("Tensor range check" && 0 <= v653 && v653 < 2);
    assert("Tensor range check" && 0 <= v652 && v652 < 4);
    int v657;
    v657 = 16 * v652;
    int v658;
    v658 = 2304 * v653;
    int v659;
    v659 = v658 + v657;
    float * v660;
    v660 = v645+v659;
    assert("Tensor range check" && 0 <= v653 && v653 < 2);
    int v662;
    v662 = 2176 * v653;
    int v663;
    v663 = threadIdx.x;
    int v664;
    v664 = v663 % 32;
    bool v665;
    v665 = 0 <= v664;
    bool v666;
    v666 = v665 == false;
    if (v666){
        assert("The index needs to be zero or positive." && v665);
    } else {
    }
    int v668;
    v668 = v664 % 4;
    int v669;
    v669 = v664 / 4;
    bool v670;
    v670 = v669 < 8;
    bool v671;
    v671 = v670 == false;
    if (v671){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v670);
    } else {
    }
    assert("Tensor range check" && 0 <= v669 && v669 < 8);
    assert("Tensor range check" && 0 <= v668 && v668 < 4);
    int v673;
    v673 = v668 + v662;
    int v674;
    v674 = 68 * v669;
    int v675;
    v675 = v674 + v673;
    float * v676;
    v676 = v641+v675;
    assert("Tensor range check" && 0 <= v652 && v652 < 4);
    int v678;
    v678 = 1088 * v652;
    int v679;
    v679 = threadIdx.x;
    int v680;
    v680 = v679 % 32;
    bool v681;
    v681 = 0 <= v680;
    bool v682;
    v682 = v681 == false;
    if (v682){
        assert("The index needs to be zero or positive." && v681);
    } else {
    }
    int v684;
    v684 = v680 % 4;
    int v685;
    v685 = v680 / 4;
    bool v686;
    v686 = v685 < 8;
    bool v687;
    v687 = v686 == false;
    if (v687){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v686);
    } else {
    }
    assert("Tensor range check" && 0 <= v685 && v685 < 8);
    assert("Tensor range check" && 0 <= v684 && v684 < 4);
    int v689;
    v689 = v684 + v678;
    int v690;
    v690 = 68 * v685;
    int v691;
    v691 = v690 + v689;
    float * v692;
    v692 = v643+v691;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v694[2];
    int v695;
    v695 = 0;
    while (while_method_0(v695)){
        int v697;
        v697 = 0;
        while (while_method_0(v697)){
            assert("Tensor range check" && 0 <= v695 && v695 < 1);
            assert("Tensor range check" && 0 <= v697 && v697 < 1);
            int v699;
            v699 = 64 * v697;
            int v700;
            v700 = v699 + v638;
            int v701;
            v701 = 4096 * v695;
            int v702;
            v702 = v701 + v700;
            float * v703;
            v703 = v633+v702;
            // Pushing the loop unrolling to: 0
            int v705;
            v705 = 0;
            #pragma unroll
            while (while_method_1(v705)){
                int v707;
                v707 = 0;
                #pragma unroll
                while (while_method_0(v707)){
                    assert("Tensor range check" && 0 <= v705 && v705 < 2);
                    assert("Tensor range check" && 0 <= v707 && v707 < 1);
                    int v709;
                    v709 = v705 + v707;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v710 = v694[v709];
                    wmma::fill_fragment(v710, 0.0f);
                    v707 += 1 ;
                }
                v705 += 1 ;
            }
            // Poping the loop unrolling to: 0
            int v711;
            v711 = 0;
            while (while_method_2(v711)){
                int v713;
                v713 = v711 + 1;
                bool v714;
                v714 = v711 == 0;
                int v715;
                v715 = v711 % 2;
                bool v716;
                v716 = 0 <= v711;
                bool v717;
                v717 = v716 == false;
                if (v717){
                    assert("The index needs to be zero or positive." && v716);
                } else {
                }
                bool v719;
                v719 = v711 < 1;
                bool v720;
                v720 = v719 == false;
                if (v720){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v719);
                } else {
                }
                bool v722;
                v722 = v713 < 1;
                Union0 v728;
                if (v722){
                    bool v723;
                    v723 = 0 <= v713;
                    bool v724;
                    v724 = v723 == false;
                    if (v724){
                        assert("The index needs to be zero or positive." && v723);
                    } else {
                    }
                    v728 = Union0{Union0_1{v713}};
                } else {
                    v728 = Union0{Union0_0{}};
                }
                assert("Tensor range check" && 0 <= v695 && v695 < 1);
                int v729;
                v729 = v701 + v636;
                assert("Tensor range check" && 0 <= v711 && v711 < 1);
                int v730;
                v730 = 64 * v711;
                int v731;
                v731 = v730 + v729;
                float * v732;
                v732 = v629+v731;
                assert("Tensor range check" && 0 <= v697 && v697 < 1);
                int v734;
                v734 = 4096 * v697;
                if (v714){
                    assert("Tensor range check" && 0 <= v711 && v711 < 1);
                    int v735;
                    v735 = v730 + v734;
                    float * v736;
                    v736 = v631+v735;
                    // Pushing the loop unrolling to: 0
                    v639.producer_acquire();
                    int v738;
                    v738 = threadIdx.x;
                    bool v739;
                    v739 = 0 <= v738;
                    bool v740;
                    v740 = v739 == false;
                    if (v740){
                        assert("The index needs to be zero or positive." && v739);
                    } else {
                    }
                    int v742;
                    v742 = v738 % 16;
                    int v743;
                    v743 = v738 / 16;
                    bool v744;
                    v744 = v743 < 16;
                    bool v745;
                    v745 = v744 == false;
                    if (v745){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v744);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v743 && v743 < 16);
                    assert("Tensor range check" && 0 <= v742 && v742 < 16);
                    int v747;
                    v747 = 4 * v742;
                    int v748;
                    v748 = 68 * v743;
                    int v749;
                    v749 = v748 + v747;
                    int v750;
                    v750 = 64 * v743;
                    int v751;
                    v751 = v750 + v747;
                    float * v752;
                    v752 = v643+v749;
                    float * v754;
                    v754 = v736+v751;
                    int v756;
                    v756 = 0;
                    #pragma unroll
                    while (while_method_3(v756)){
                        int v758;
                        v758 = 0;
                        #pragma unroll
                        while (while_method_0(v758)){
                            assert("Tensor range check" && 0 <= v756 && v756 < 4);
                            assert("Tensor range check" && 0 <= v758 && v758 < 1);
                            int v760;
                            v760 = 64 * v758;
                            int v761;
                            v761 = 1088 * v756;
                            int v762;
                            v762 = v761 + v760;
                            int v763;
                            v763 = 1024 * v756;
                            int v764;
                            v764 = v763 + v760;
                            constexpr int v765 = sizeof(float) * 4;
                            assert("Pointer alignment check" && (unsigned long long)(v754 + v764) % v765 == 0 && (unsigned long long)(v752 + v762) % v765 == 0);
                            cuda::memcpy_async(v752 + v762, v754 + v764, cuda::aligned_size_t<v765>(v765), v639);
                            v758 += 1 ;
                        }
                        v756 += 1 ;
                    }
                    v639.producer_commit();
                    // Poping the loop unrolling to: 0
                } else {
                }
                // Pushing the loop unrolling to: 0
                int v766;
                v766 = threadIdx.x;
                bool v767;
                v767 = 0 <= v766;
                bool v768;
                v768 = v767 == false;
                if (v768){
                    assert("The index needs to be zero or positive." && v767);
                } else {
                }
                int v770;
                v770 = v766 % 16;
                int v771;
                v771 = v766 / 16;
                bool v772;
                v772 = v771 < 16;
                bool v773;
                v773 = v772 == false;
                if (v773){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v772);
                } else {
                }
                assert("Tensor range check" && 0 <= v771 && v771 < 16);
                assert("Tensor range check" && 0 <= v770 && v770 < 16);
                int v775;
                v775 = 4 * v770;
                int v776;
                v776 = 68 * v771;
                int v777;
                v777 = v776 + v775;
                int v778;
                v778 = 64 * v771;
                int v779;
                v779 = v778 + v775;
                float * v780;
                v780 = v641+v777;
                float * v782;
                v782 = v732+v779;
                int v784;
                v784 = 0;
                #pragma unroll
                while (while_method_3(v784)){
                    int v786;
                    v786 = 0;
                    #pragma unroll
                    while (while_method_0(v786)){
                        assert("Tensor range check" && 0 <= v784 && v784 < 4);
                        assert("Tensor range check" && 0 <= v786 && v786 < 1);
                        int v788;
                        v788 = 64 * v786;
                        int v789;
                        v789 = 1088 * v784;
                        int v790;
                        v790 = v789 + v788;
                        int v791;
                        v791 = 1024 * v784;
                        int v792;
                        v792 = v791 + v788;
                        int4* v793;
                        v793 = reinterpret_cast<int4*>(v782 + v792);
                        int4* v794;
                        v794 = reinterpret_cast<int4*>(v780 + v790);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v793) % 16 == 0 && reinterpret_cast<unsigned long long>(v794) % 16 == 0);
                        *v794 = *v793;
                        v786 += 1 ;
                    }
                    v784 += 1 ;
                }
                // Poping the loop unrolling to: 0
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v795[1];
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v796[8];
                cuda::pipeline_consumer_wait_prior<0>(v639);;
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v797;
                v797 = 0;
                #pragma unroll
                while (while_method_0(v797)){
                    int v799;
                    v799 = 0;
                    #pragma unroll
                    while (while_method_4(v799)){
                        assert("Tensor range check" && 0 <= v797 && v797 < 1);
                        assert("Tensor range check" && 0 <= v799 && v799 < 8);
                        int v801;
                        v801 = 8 * v797;
                        int v802;
                        v802 = v801 + v799;
                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v803 = v796[v802];
                        assert("Tensor range check" && 0 <= v797 && v797 < 1);
                        int v804;
                        v804 = 1088 * v797;
                        assert("Tensor range check" && 0 <= v799 && v799 < 8);
                        int v805;
                        v805 = 8 * v799;
                        int v806;
                        v806 = v805 + v804;
                        int v807;
                        v807 = 0;
                        #pragma unroll
                        while (while_method_1(v807)){
                            int v809;
                            v809 = 0;
                            #pragma unroll
                            while (while_method_1(v809)){
                                assert("Tensor range check" && 0 <= v807 && v807 < 2);
                                assert("Tensor range check" && 0 <= v809 && v809 < 2);
                                int v811;
                                v811 = 4 * v809;
                                int v812;
                                v812 = v811 + v806;
                                int v813;
                                v813 = 544 * v807;
                                int v814;
                                v814 = v813 + v812;
                                float v815;
                                v815 = v692[v814];
                                bool v816;
                                v816 = 0 <= v809;
                                bool v818;
                                if (v816){
                                    bool v817;
                                    v817 = v809 < 2;
                                    v818 = v817;
                                } else {
                                    v818 = false;
                                }
                                bool v819;
                                v819 = v818 == false;
                                if (v819){
                                    assert("The indices should be inside the range of the dimension." && v818);
                                } else {
                                }
                                bool v821;
                                v821 = 0 <= v807;
                                bool v823;
                                if (v821){
                                    bool v822;
                                    v822 = v807 < 2;
                                    v823 = v822;
                                } else {
                                    v823 = false;
                                }
                                bool v824;
                                v824 = v823 == false;
                                if (v824){
                                    assert("The indices should be inside the range of the dimension." && v823);
                                } else {
                                }
                                int v826;
                                v826 = v807 * 2;
                                int v827;
                                v827 = v809 + v826;
                                v803.x[v827] = wmma::__float_to_tf32(v815);
                                v809 += 1 ;
                            }
                            v807 += 1 ;
                        }
                        v799 += 1 ;
                    }
                    v797 += 1 ;
                }
                // Poping the loop unrolling to: 0
                v639.consumer_release();
                switch (v728.tag) {
                    case 0: { // None
                        break;
                    }
                    case 1: { // Some
                        int v828 = v728.case1.v0;
                        assert("Tensor range check" && 0 <= v828 && v828 < 1);
                        int v829;
                        v829 = 64 * v828;
                        int v830;
                        v830 = v829 + v734;
                        float * v831;
                        v831 = v631+v830;
                        __syncthreads();
                        // Pushing the loop unrolling to: 0
                        v639.producer_acquire();
                        int v833;
                        v833 = threadIdx.x;
                        bool v834;
                        v834 = 0 <= v833;
                        bool v835;
                        v835 = v834 == false;
                        if (v835){
                            assert("The index needs to be zero or positive." && v834);
                        } else {
                        }
                        int v837;
                        v837 = v833 % 16;
                        int v838;
                        v838 = v833 / 16;
                        bool v839;
                        v839 = v838 < 16;
                        bool v840;
                        v840 = v839 == false;
                        if (v840){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v839);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v838 && v838 < 16);
                        assert("Tensor range check" && 0 <= v837 && v837 < 16);
                        int v842;
                        v842 = 4 * v837;
                        int v843;
                        v843 = 68 * v838;
                        int v844;
                        v844 = v843 + v842;
                        int v845;
                        v845 = 64 * v838;
                        int v846;
                        v846 = v845 + v842;
                        float * v847;
                        v847 = v643+v844;
                        float * v849;
                        v849 = v831+v846;
                        int v851;
                        v851 = 0;
                        #pragma unroll
                        while (while_method_3(v851)){
                            int v853;
                            v853 = 0;
                            #pragma unroll
                            while (while_method_0(v853)){
                                assert("Tensor range check" && 0 <= v851 && v851 < 4);
                                assert("Tensor range check" && 0 <= v853 && v853 < 1);
                                int v855;
                                v855 = 64 * v853;
                                int v856;
                                v856 = 1088 * v851;
                                int v857;
                                v857 = v856 + v855;
                                int v858;
                                v858 = 1024 * v851;
                                int v859;
                                v859 = v858 + v855;
                                constexpr int v860 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v849 + v859) % v860 == 0 && (unsigned long long)(v847 + v857) % v860 == 0);
                                cuda::memcpy_async(v847 + v857, v849 + v859, cuda::aligned_size_t<v860>(v860), v639);
                                v853 += 1 ;
                            }
                            v851 += 1 ;
                        }
                        v639.producer_commit();
                        // Poping the loop unrolling to: 0
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                // Pushing the loop unrolling to: 0
                int v861;
                v861 = 0;
                #pragma unroll
                while (while_method_1(v861)){
                    int v863;
                    v863 = 0;
                    #pragma unroll
                    while (while_method_4(v863)){
                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v865 = v795[0];
                        assert("Tensor range check" && 0 <= v861 && v861 < 2);
                        int v866;
                        v866 = 1088 * v861;
                        assert("Tensor range check" && 0 <= v863 && v863 < 8);
                        int v867;
                        v867 = 8 * v863;
                        int v868;
                        v868 = v867 + v866;
                        int v869;
                        v869 = 0;
                        #pragma unroll
                        while (while_method_1(v869)){
                            int v871;
                            v871 = 0;
                            #pragma unroll
                            while (while_method_1(v871)){
                                assert("Tensor range check" && 0 <= v869 && v869 < 2);
                                assert("Tensor range check" && 0 <= v871 && v871 < 2);
                                int v873;
                                v873 = 544 * v871;
                                int v874;
                                v874 = v873 + v868;
                                int v875;
                                v875 = 4 * v869;
                                int v876;
                                v876 = v875 + v874;
                                float v877;
                                v877 = v676[v876];
                                bool v878;
                                v878 = 0 <= v871;
                                bool v880;
                                if (v878){
                                    bool v879;
                                    v879 = v871 < 2;
                                    v880 = v879;
                                } else {
                                    v880 = false;
                                }
                                bool v881;
                                v881 = v880 == false;
                                if (v881){
                                    assert("The indices should be inside the range of the dimension." && v880);
                                } else {
                                }
                                bool v883;
                                v883 = 0 <= v869;
                                bool v885;
                                if (v883){
                                    bool v884;
                                    v884 = v869 < 2;
                                    v885 = v884;
                                } else {
                                    v885 = false;
                                }
                                bool v886;
                                v886 = v885 == false;
                                if (v886){
                                    assert("The indices should be inside the range of the dimension." && v885);
                                } else {
                                }
                                int v888;
                                v888 = v869 * 2;
                                int v889;
                                v889 = v871 + v888;
                                v865.x[v889] = wmma::__float_to_tf32(v877);
                                v871 += 1 ;
                            }
                            v869 += 1 ;
                        }
                        int v890;
                        v890 = 0;
                        #pragma unroll
                        while (while_method_0(v890)){
                            assert("Tensor range check" && 0 <= v861 && v861 < 2);
                            assert("Tensor range check" && 0 <= v890 && v890 < 1);
                            int v892;
                            v892 = v861 + v890;
                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v893 = v694[v892];
                            assert("Tensor range check" && 0 <= v890 && v890 < 1);
                            assert("Tensor range check" && 0 <= v863 && v863 < 8);
                            int v894;
                            v894 = 8 * v890;
                            int v895;
                            v895 = v894 + v863;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v896 = v796[v895];
                            wmma::mma_sync(v893, v865, v896, v893);
                            v890 += 1 ;
                        }
                        v863 += 1 ;
                    }
                    v861 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v711 = v713;
            }
            // Pushing the loop unrolling to: 0
            int v897;
            v897 = 0;
            #pragma unroll
            while (while_method_1(v897)){
                int v899;
                v899 = 0;
                #pragma unroll
                while (while_method_0(v899)){
                    assert("Tensor range check" && 0 <= v897 && v897 < 2);
                    assert("Tensor range check" && 0 <= v899 && v899 < 1);
                    int v901;
                    v901 = v897 + v899;
                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v902 = v694[v901];
                    assert("Tensor range check" && 0 <= v897 && v897 < 2);
                    assert("Tensor range check" && 0 <= v899 && v899 < 1);
                    int v903;
                    v903 = 16 * v899;
                    int v904;
                    v904 = 1152 * v897;
                    int v905;
                    v905 = v904 + v903;
                    float * v906;
                    v906 = v660+v905;
                    wmma::store_matrix_sync(v906, v902, 72, wmma::mem_row_major);
                    v899 += 1 ;
                }
                v897 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            // Pushing the loop unrolling to: 0
            int v908;
            v908 = threadIdx.x;
            bool v909;
            v909 = 0 <= v908;
            bool v910;
            v910 = v909 == false;
            if (v910){
                assert("The index needs to be zero or positive." && v909);
            } else {
            }
            int v912;
            v912 = v908 % 16;
            int v913;
            v913 = v908 / 16;
            bool v914;
            v914 = v913 < 16;
            bool v915;
            v915 = v914 == false;
            if (v915){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v914);
            } else {
            }
            assert("Tensor range check" && 0 <= v913 && v913 < 16);
            assert("Tensor range check" && 0 <= v912 && v912 < 16);
            int v917;
            v917 = 4 * v912;
            int v918;
            v918 = 64 * v913;
            int v919;
            v919 = v918 + v917;
            int v920;
            v920 = 72 * v913;
            int v921;
            v921 = v920 + v917;
            float * v922;
            v922 = v703+v919;
            float * v924;
            v924 = v645+v921;
            int v926;
            v926 = 0;
            #pragma unroll
            while (while_method_3(v926)){
                int v928;
                v928 = 0;
                #pragma unroll
                while (while_method_0(v928)){
                    assert("Tensor range check" && 0 <= v926 && v926 < 4);
                    assert("Tensor range check" && 0 <= v928 && v928 < 1);
                    int v930;
                    v930 = 64 * v928;
                    int v931;
                    v931 = 1024 * v926;
                    int v932;
                    v932 = v931 + v930;
                    int v933;
                    v933 = 1152 * v926;
                    int v934;
                    v934 = v933 + v930;
                    int4* v935;
                    v935 = reinterpret_cast<int4*>(v924 + v934);
                    int4* v936;
                    v936 = reinterpret_cast<int4*>(v922 + v932);
                    assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v935) % 16 == 0 && reinterpret_cast<unsigned long long>(v936) % 16 == 0);
                    *v936 = *v935;
                    v928 += 1 ;
                }
                v926 += 1 ;
            }
            // Poping the loop unrolling to: 0
            __syncthreads();
            v697 += 1 ;
        }
        v695 += 1 ;
    }
    float * v937;
    v937 = reinterpret_cast<float *>(&v1[3145728ull]);
    int * v939;
    v939 = reinterpret_cast<int *>(&v1[3538944ull]);
    return method_2(v939, v937, v633, v8);
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
        v11 = reinterpret_cast<float *>(&v1[0ull]);
        float * v13;
        v13 = reinterpret_cast<float *>(&v0[0ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16);
        int v15;
        v15 = 4096 * v9;
        float * v16;
        v16 = reinterpret_cast<float *>(&v1[393216ull]);
        int v18;
        v18 = blockIdx.x;
        assert("Tensor range check" && 0 <= v18 && v18 < 24);
        int v19;
        v19 = 4096 * v18;
        int v20;
        v20 = blockIdx.x;
        assert("Tensor range check" && 0 <= v20 && v20 < 24);
        int v21;
        v21 = 4096 * v20;
        cuda::pipeline<cuda::thread_scope_thread> v22 = cuda::make_pipeline();
        extern __shared__ unsigned char v23[];
        float * v24;
        v24 = reinterpret_cast<float *>(&v23[0ull]);
        float * v26;
        v26 = reinterpret_cast<float *>(&v23[17408ull]);
        float * v28;
        v28 = reinterpret_cast<float *>(&v23[0ull]);
        int v30;
        v30 = threadIdx.x;
        int v31;
        v31 = v30 / 32;
        bool v32;
        v32 = 0 <= v31;
        bool v33;
        v33 = v32 == false;
        if (v33){
            assert("The index needs to be zero or positive." && v32);
        } else {
        }
        int v35;
        v35 = v31 % 4;
        int v36;
        v36 = v31 / 4;
        bool v37;
        v37 = v36 < 2;
        bool v38;
        v38 = v37 == false;
        if (v38){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v37);
        } else {
        }
        assert("Tensor range check" && 0 <= v36 && v36 < 2);
        assert("Tensor range check" && 0 <= v35 && v35 < 4);
        int v40;
        v40 = 16 * v35;
        int v41;
        v41 = 2304 * v36;
        int v42;
        v42 = v41 + v40;
        float * v43;
        v43 = v28+v42;
        assert("Tensor range check" && 0 <= v36 && v36 < 2);
        int v45;
        v45 = 2176 * v36;
        int v46;
        v46 = threadIdx.x;
        int v47;
        v47 = v46 % 32;
        bool v48;
        v48 = 0 <= v47;
        bool v49;
        v49 = v48 == false;
        if (v49){
            assert("The index needs to be zero or positive." && v48);
        } else {
        }
        int v51;
        v51 = v47 % 4;
        int v52;
        v52 = v47 / 4;
        bool v53;
        v53 = v52 < 8;
        bool v54;
        v54 = v53 == false;
        if (v54){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v53);
        } else {
        }
        assert("Tensor range check" && 0 <= v52 && v52 < 8);
        assert("Tensor range check" && 0 <= v51 && v51 < 4);
        int v56;
        v56 = v51 + v45;
        int v57;
        v57 = 68 * v52;
        int v58;
        v58 = v57 + v56;
        float * v59;
        v59 = v24+v58;
        assert("Tensor range check" && 0 <= v35 && v35 < 4);
        int v61;
        v61 = 1088 * v35;
        int v62;
        v62 = threadIdx.x;
        int v63;
        v63 = v62 % 32;
        bool v64;
        v64 = 0 <= v63;
        bool v65;
        v65 = v64 == false;
        if (v65){
            assert("The index needs to be zero or positive." && v64);
        } else {
        }
        int v67;
        v67 = v63 % 4;
        int v68;
        v68 = v63 / 4;
        bool v69;
        v69 = v68 < 8;
        bool v70;
        v70 = v69 == false;
        if (v70){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v69);
        } else {
        }
        assert("Tensor range check" && 0 <= v68 && v68 < 8);
        assert("Tensor range check" && 0 <= v67 && v67 < 4);
        int v72;
        v72 = v67 + v61;
        int v73;
        v73 = 68 * v68;
        int v74;
        v74 = v73 + v72;
        float * v75;
        v75 = v26+v74;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> v77[2];
        int v78;
        v78 = 0;
        while (while_method_0(v78)){
            int v80;
            v80 = 0;
            while (while_method_0(v80)){
                assert("Tensor range check" && 0 <= v78 && v78 < 1);
                assert("Tensor range check" && 0 <= v80 && v80 < 1);
                int v82;
                v82 = 64 * v80;
                int v83;
                v83 = v82 + v21;
                int v84;
                v84 = 4096 * v78;
                int v85;
                v85 = v84 + v83;
                float * v86;
                v86 = v16+v85;
                // Pushing the loop unrolling to: 0
                int v88;
                v88 = 0;
                #pragma unroll
                while (while_method_1(v88)){
                    int v90;
                    v90 = 0;
                    #pragma unroll
                    while (while_method_0(v90)){
                        assert("Tensor range check" && 0 <= v88 && v88 < 2);
                        assert("Tensor range check" && 0 <= v90 && v90 < 1);
                        int v92;
                        v92 = v88 + v90;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v93 = v77[v92];
                        wmma::fill_fragment(v93, 0.0f);
                        v90 += 1 ;
                    }
                    v88 += 1 ;
                }
                // Poping the loop unrolling to: 0
                int v94;
                v94 = 0;
                while (while_method_2(v94)){
                    int v96;
                    v96 = v94 + 1;
                    bool v97;
                    v97 = v94 == 0;
                    int v98;
                    v98 = v94 % 2;
                    bool v99;
                    v99 = 0 <= v94;
                    bool v100;
                    v100 = v99 == false;
                    if (v100){
                        assert("The index needs to be zero or positive." && v99);
                    } else {
                    }
                    bool v102;
                    v102 = v94 < 1;
                    bool v103;
                    v103 = v102 == false;
                    if (v103){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v102);
                    } else {
                    }
                    bool v105;
                    v105 = v96 < 1;
                    Union0 v111;
                    if (v105){
                        bool v106;
                        v106 = 0 <= v96;
                        bool v107;
                        v107 = v106 == false;
                        if (v107){
                            assert("The index needs to be zero or positive." && v106);
                        } else {
                        }
                        v111 = Union0{Union0_1{v96}};
                    } else {
                        v111 = Union0{Union0_0{}};
                    }
                    assert("Tensor range check" && 0 <= v78 && v78 < 1);
                    int v112;
                    v112 = v84 + v19;
                    assert("Tensor range check" && 0 <= v94 && v94 < 1);
                    int v113;
                    v113 = 64 * v94;
                    int v114;
                    v114 = v113 + v112;
                    float * v115;
                    v115 = v11+v114;
                    assert("Tensor range check" && 0 <= v80 && v80 < 1);
                    int v117;
                    v117 = 4096 * v80;
                    int v118;
                    v118 = v117 + v15;
                    if (v97){
                        assert("Tensor range check" && 0 <= v94 && v94 < 1);
                        int v119;
                        v119 = v113 + v118;
                        float * v120;
                        v120 = v13+v119;
                        // Pushing the loop unrolling to: 0
                        v22.producer_acquire();
                        int v122;
                        v122 = threadIdx.x;
                        bool v123;
                        v123 = 0 <= v122;
                        bool v124;
                        v124 = v123 == false;
                        if (v124){
                            assert("The index needs to be zero or positive." && v123);
                        } else {
                        }
                        int v126;
                        v126 = v122 % 16;
                        int v127;
                        v127 = v122 / 16;
                        bool v128;
                        v128 = v127 < 16;
                        bool v129;
                        v129 = v128 == false;
                        if (v129){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v128);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v127 && v127 < 16);
                        assert("Tensor range check" && 0 <= v126 && v126 < 16);
                        int v131;
                        v131 = 4 * v126;
                        int v132;
                        v132 = 68 * v127;
                        int v133;
                        v133 = v132 + v131;
                        int v134;
                        v134 = 64 * v127;
                        int v135;
                        v135 = v134 + v131;
                        float * v136;
                        v136 = v26+v133;
                        float * v138;
                        v138 = v120+v135;
                        int v140;
                        v140 = 0;
                        #pragma unroll
                        while (while_method_3(v140)){
                            int v142;
                            v142 = 0;
                            #pragma unroll
                            while (while_method_0(v142)){
                                assert("Tensor range check" && 0 <= v140 && v140 < 4);
                                assert("Tensor range check" && 0 <= v142 && v142 < 1);
                                int v144;
                                v144 = 64 * v142;
                                int v145;
                                v145 = 1088 * v140;
                                int v146;
                                v146 = v145 + v144;
                                int v147;
                                v147 = 1024 * v140;
                                int v148;
                                v148 = v147 + v144;
                                constexpr int v149 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v138 + v148) % v149 == 0 && (unsigned long long)(v136 + v146) % v149 == 0);
                                cuda::memcpy_async(v136 + v146, v138 + v148, cuda::aligned_size_t<v149>(v149), v22);
                                v142 += 1 ;
                            }
                            v140 += 1 ;
                        }
                        v22.producer_commit();
                        // Poping the loop unrolling to: 0
                    } else {
                    }
                    // Pushing the loop unrolling to: 0
                    int v150;
                    v150 = threadIdx.x;
                    bool v151;
                    v151 = 0 <= v150;
                    bool v152;
                    v152 = v151 == false;
                    if (v152){
                        assert("The index needs to be zero or positive." && v151);
                    } else {
                    }
                    int v154;
                    v154 = v150 % 16;
                    int v155;
                    v155 = v150 / 16;
                    bool v156;
                    v156 = v155 < 16;
                    bool v157;
                    v157 = v156 == false;
                    if (v157){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v156);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v155 && v155 < 16);
                    assert("Tensor range check" && 0 <= v154 && v154 < 16);
                    int v159;
                    v159 = 4 * v154;
                    int v160;
                    v160 = 68 * v155;
                    int v161;
                    v161 = v160 + v159;
                    int v162;
                    v162 = 64 * v155;
                    int v163;
                    v163 = v162 + v159;
                    float * v164;
                    v164 = v24+v161;
                    float * v166;
                    v166 = v115+v163;
                    int v168;
                    v168 = 0;
                    #pragma unroll
                    while (while_method_3(v168)){
                        int v170;
                        v170 = 0;
                        #pragma unroll
                        while (while_method_0(v170)){
                            assert("Tensor range check" && 0 <= v168 && v168 < 4);
                            assert("Tensor range check" && 0 <= v170 && v170 < 1);
                            int v172;
                            v172 = 64 * v170;
                            int v173;
                            v173 = 1088 * v168;
                            int v174;
                            v174 = v173 + v172;
                            int v175;
                            v175 = 1024 * v168;
                            int v176;
                            v176 = v175 + v172;
                            int4* v177;
                            v177 = reinterpret_cast<int4*>(v166 + v176);
                            int4* v178;
                            v178 = reinterpret_cast<int4*>(v164 + v174);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v177) % 16 == 0 && reinterpret_cast<unsigned long long>(v178) % 16 == 0);
                            *v178 = *v177;
                            v170 += 1 ;
                        }
                        v168 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v179[1];
                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v180[8];
                    cuda::pipeline_consumer_wait_prior<0>(v22);;
                    __syncthreads();
                    // Pushing the loop unrolling to: 0
                    int v181;
                    v181 = 0;
                    #pragma unroll
                    while (while_method_0(v181)){
                        int v183;
                        v183 = 0;
                        #pragma unroll
                        while (while_method_4(v183)){
                            assert("Tensor range check" && 0 <= v181 && v181 < 1);
                            assert("Tensor range check" && 0 <= v183 && v183 < 8);
                            int v185;
                            v185 = 8 * v181;
                            int v186;
                            v186 = v185 + v183;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v187 = v180[v186];
                            assert("Tensor range check" && 0 <= v181 && v181 < 1);
                            int v188;
                            v188 = 1088 * v181;
                            assert("Tensor range check" && 0 <= v183 && v183 < 8);
                            int v189;
                            v189 = 8 * v183;
                            int v190;
                            v190 = v189 + v188;
                            int v191;
                            v191 = 0;
                            #pragma unroll
                            while (while_method_1(v191)){
                                int v193;
                                v193 = 0;
                                #pragma unroll
                                while (while_method_1(v193)){
                                    assert("Tensor range check" && 0 <= v191 && v191 < 2);
                                    assert("Tensor range check" && 0 <= v193 && v193 < 2);
                                    int v195;
                                    v195 = 4 * v193;
                                    int v196;
                                    v196 = v195 + v190;
                                    int v197;
                                    v197 = 544 * v191;
                                    int v198;
                                    v198 = v197 + v196;
                                    float v199;
                                    v199 = v75[v198];
                                    bool v200;
                                    v200 = 0 <= v193;
                                    bool v202;
                                    if (v200){
                                        bool v201;
                                        v201 = v193 < 2;
                                        v202 = v201;
                                    } else {
                                        v202 = false;
                                    }
                                    bool v203;
                                    v203 = v202 == false;
                                    if (v203){
                                        assert("The indices should be inside the range of the dimension." && v202);
                                    } else {
                                    }
                                    bool v205;
                                    v205 = 0 <= v191;
                                    bool v207;
                                    if (v205){
                                        bool v206;
                                        v206 = v191 < 2;
                                        v207 = v206;
                                    } else {
                                        v207 = false;
                                    }
                                    bool v208;
                                    v208 = v207 == false;
                                    if (v208){
                                        assert("The indices should be inside the range of the dimension." && v207);
                                    } else {
                                    }
                                    int v210;
                                    v210 = v191 * 2;
                                    int v211;
                                    v211 = v193 + v210;
                                    v187.x[v211] = wmma::__float_to_tf32(v199);
                                    v193 += 1 ;
                                }
                                v191 += 1 ;
                            }
                            v183 += 1 ;
                        }
                        v181 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    v22.consumer_release();
                    switch (v111.tag) {
                        case 0: { // None
                            break;
                        }
                        case 1: { // Some
                            int v212 = v111.case1.v0;
                            assert("Tensor range check" && 0 <= v212 && v212 < 1);
                            int v213;
                            v213 = 64 * v212;
                            int v214;
                            v214 = v213 + v118;
                            float * v215;
                            v215 = v13+v214;
                            __syncthreads();
                            // Pushing the loop unrolling to: 0
                            v22.producer_acquire();
                            int v217;
                            v217 = threadIdx.x;
                            bool v218;
                            v218 = 0 <= v217;
                            bool v219;
                            v219 = v218 == false;
                            if (v219){
                                assert("The index needs to be zero or positive." && v218);
                            } else {
                            }
                            int v221;
                            v221 = v217 % 16;
                            int v222;
                            v222 = v217 / 16;
                            bool v223;
                            v223 = v222 < 16;
                            bool v224;
                            v224 = v223 == false;
                            if (v224){
                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v223);
                            } else {
                            }
                            assert("Tensor range check" && 0 <= v222 && v222 < 16);
                            assert("Tensor range check" && 0 <= v221 && v221 < 16);
                            int v226;
                            v226 = 4 * v221;
                            int v227;
                            v227 = 68 * v222;
                            int v228;
                            v228 = v227 + v226;
                            int v229;
                            v229 = 64 * v222;
                            int v230;
                            v230 = v229 + v226;
                            float * v231;
                            v231 = v26+v228;
                            float * v233;
                            v233 = v215+v230;
                            int v235;
                            v235 = 0;
                            #pragma unroll
                            while (while_method_3(v235)){
                                int v237;
                                v237 = 0;
                                #pragma unroll
                                while (while_method_0(v237)){
                                    assert("Tensor range check" && 0 <= v235 && v235 < 4);
                                    assert("Tensor range check" && 0 <= v237 && v237 < 1);
                                    int v239;
                                    v239 = 64 * v237;
                                    int v240;
                                    v240 = 1088 * v235;
                                    int v241;
                                    v241 = v240 + v239;
                                    int v242;
                                    v242 = 1024 * v235;
                                    int v243;
                                    v243 = v242 + v239;
                                    constexpr int v244 = sizeof(float) * 4;
                                    assert("Pointer alignment check" && (unsigned long long)(v233 + v243) % v244 == 0 && (unsigned long long)(v231 + v241) % v244 == 0);
                                    cuda::memcpy_async(v231 + v241, v233 + v243, cuda::aligned_size_t<v244>(v244), v22);
                                    v237 += 1 ;
                                }
                                v235 += 1 ;
                            }
                            v22.producer_commit();
                            // Poping the loop unrolling to: 0
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false); __trap();
                        }
                    }
                    // Pushing the loop unrolling to: 0
                    int v245;
                    v245 = 0;
                    #pragma unroll
                    while (while_method_1(v245)){
                        int v247;
                        v247 = 0;
                        #pragma unroll
                        while (while_method_4(v247)){
                            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v249 = v179[0];
                            assert("Tensor range check" && 0 <= v245 && v245 < 2);
                            int v250;
                            v250 = 1088 * v245;
                            assert("Tensor range check" && 0 <= v247 && v247 < 8);
                            int v251;
                            v251 = 8 * v247;
                            int v252;
                            v252 = v251 + v250;
                            int v253;
                            v253 = 0;
                            #pragma unroll
                            while (while_method_1(v253)){
                                int v255;
                                v255 = 0;
                                #pragma unroll
                                while (while_method_1(v255)){
                                    assert("Tensor range check" && 0 <= v253 && v253 < 2);
                                    assert("Tensor range check" && 0 <= v255 && v255 < 2);
                                    int v257;
                                    v257 = 544 * v255;
                                    int v258;
                                    v258 = v257 + v252;
                                    int v259;
                                    v259 = 4 * v253;
                                    int v260;
                                    v260 = v259 + v258;
                                    float v261;
                                    v261 = v59[v260];
                                    bool v262;
                                    v262 = 0 <= v255;
                                    bool v264;
                                    if (v262){
                                        bool v263;
                                        v263 = v255 < 2;
                                        v264 = v263;
                                    } else {
                                        v264 = false;
                                    }
                                    bool v265;
                                    v265 = v264 == false;
                                    if (v265){
                                        assert("The indices should be inside the range of the dimension." && v264);
                                    } else {
                                    }
                                    bool v267;
                                    v267 = 0 <= v253;
                                    bool v269;
                                    if (v267){
                                        bool v268;
                                        v268 = v253 < 2;
                                        v269 = v268;
                                    } else {
                                        v269 = false;
                                    }
                                    bool v270;
                                    v270 = v269 == false;
                                    if (v270){
                                        assert("The indices should be inside the range of the dimension." && v269);
                                    } else {
                                    }
                                    int v272;
                                    v272 = v253 * 2;
                                    int v273;
                                    v273 = v255 + v272;
                                    v249.x[v273] = wmma::__float_to_tf32(v261);
                                    v255 += 1 ;
                                }
                                v253 += 1 ;
                            }
                            int v274;
                            v274 = 0;
                            #pragma unroll
                            while (while_method_0(v274)){
                                assert("Tensor range check" && 0 <= v245 && v245 < 2);
                                assert("Tensor range check" && 0 <= v274 && v274 < 1);
                                int v276;
                                v276 = v245 + v274;
                                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v277 = v77[v276];
                                assert("Tensor range check" && 0 <= v274 && v274 < 1);
                                assert("Tensor range check" && 0 <= v247 && v247 < 8);
                                int v278;
                                v278 = 8 * v274;
                                int v279;
                                v279 = v278 + v247;
                                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v280 = v180[v279];
                                wmma::mma_sync(v277, v249, v280, v277);
                                v274 += 1 ;
                            }
                            v247 += 1 ;
                        }
                        v245 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    __syncthreads();
                    v94 = v96;
                }
                // Pushing the loop unrolling to: 0
                int v281;
                v281 = 0;
                #pragma unroll
                while (while_method_1(v281)){
                    int v283;
                    v283 = 0;
                    #pragma unroll
                    while (while_method_0(v283)){
                        assert("Tensor range check" && 0 <= v281 && v281 < 2);
                        assert("Tensor range check" && 0 <= v283 && v283 < 1);
                        int v285;
                        v285 = v281 + v283;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v286 = v77[v285];
                        assert("Tensor range check" && 0 <= v281 && v281 < 2);
                        assert("Tensor range check" && 0 <= v283 && v283 < 1);
                        int v287;
                        v287 = 16 * v283;
                        int v288;
                        v288 = 1152 * v281;
                        int v289;
                        v289 = v288 + v287;
                        float * v290;
                        v290 = v43+v289;
                        wmma::store_matrix_sync(v290, v286, 72, wmma::mem_row_major);
                        v283 += 1 ;
                    }
                    v281 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v292;
                v292 = threadIdx.x;
                bool v293;
                v293 = 0 <= v292;
                bool v294;
                v294 = v293 == false;
                if (v294){
                    assert("The index needs to be zero or positive." && v293);
                } else {
                }
                int v296;
                v296 = v292 % 16;
                int v297;
                v297 = v292 / 16;
                bool v298;
                v298 = v297 < 16;
                bool v299;
                v299 = v298 == false;
                if (v299){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v298);
                } else {
                }
                assert("Tensor range check" && 0 <= v297 && v297 < 16);
                assert("Tensor range check" && 0 <= v296 && v296 < 16);
                int v301;
                v301 = 4 * v296;
                int v302;
                v302 = 64 * v297;
                int v303;
                v303 = v302 + v301;
                int v304;
                v304 = 72 * v297;
                int v305;
                v305 = v304 + v301;
                float * v306;
                v306 = v86+v303;
                float * v308;
                v308 = v28+v305;
                int v310;
                v310 = 0;
                #pragma unroll
                while (while_method_3(v310)){
                    int v312;
                    v312 = 0;
                    #pragma unroll
                    while (while_method_0(v312)){
                        assert("Tensor range check" && 0 <= v310 && v310 < 4);
                        assert("Tensor range check" && 0 <= v312 && v312 < 1);
                        int v314;
                        v314 = 64 * v312;
                        int v315;
                        v315 = 1024 * v310;
                        int v316;
                        v316 = v315 + v314;
                        int v317;
                        v317 = 1152 * v310;
                        int v318;
                        v318 = v317 + v314;
                        int4* v319;
                        v319 = reinterpret_cast<int4*>(v308 + v318);
                        int4* v320;
                        v320 = reinterpret_cast<int4*>(v306 + v316);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v319) % 16 == 0 && reinterpret_cast<unsigned long long>(v320) % 16 == 0);
                        *v320 = *v319;
                        v312 += 1 ;
                    }
                    v310 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v80 += 1 ;
            }
            v78 += 1 ;
        }
        float * v321;
        v321 = reinterpret_cast<float *>(&v1[786432ull]);
        method_0(v321, v16);
        float * v323;
        v323 = reinterpret_cast<float *>(&v1[1179648ull]);
        method_1(v323, v321);
        float * v325;
        v325 = reinterpret_cast<float *>(&v0[262144ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16);
        float * v327;
        v327 = reinterpret_cast<float *>(&v1[1572864ull]);
        int v329;
        v329 = blockIdx.x;
        assert("Tensor range check" && 0 <= v329 && v329 < 24);
        int v330;
        v330 = 4096 * v329;
        int v331;
        v331 = blockIdx.x;
        assert("Tensor range check" && 0 <= v331 && v331 < 24);
        int v332;
        v332 = 4096 * v331;
        cuda::pipeline<cuda::thread_scope_thread> v333 = cuda::make_pipeline();
        extern __shared__ unsigned char v334[];
        float * v335;
        v335 = reinterpret_cast<float *>(&v334[0ull]);
        float * v337;
        v337 = reinterpret_cast<float *>(&v334[17408ull]);
        float * v339;
        v339 = reinterpret_cast<float *>(&v334[0ull]);
        int v341;
        v341 = threadIdx.x;
        int v342;
        v342 = v341 / 32;
        bool v343;
        v343 = 0 <= v342;
        bool v344;
        v344 = v343 == false;
        if (v344){
            assert("The index needs to be zero or positive." && v343);
        } else {
        }
        int v346;
        v346 = v342 % 4;
        int v347;
        v347 = v342 / 4;
        bool v348;
        v348 = v347 < 2;
        bool v349;
        v349 = v348 == false;
        if (v349){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v348);
        } else {
        }
        assert("Tensor range check" && 0 <= v347 && v347 < 2);
        assert("Tensor range check" && 0 <= v346 && v346 < 4);
        int v351;
        v351 = 16 * v346;
        int v352;
        v352 = 2304 * v347;
        int v353;
        v353 = v352 + v351;
        float * v354;
        v354 = v339+v353;
        assert("Tensor range check" && 0 <= v347 && v347 < 2);
        int v356;
        v356 = 2176 * v347;
        int v357;
        v357 = threadIdx.x;
        int v358;
        v358 = v357 % 32;
        bool v359;
        v359 = 0 <= v358;
        bool v360;
        v360 = v359 == false;
        if (v360){
            assert("The index needs to be zero or positive." && v359);
        } else {
        }
        int v362;
        v362 = v358 % 4;
        int v363;
        v363 = v358 / 4;
        bool v364;
        v364 = v363 < 8;
        bool v365;
        v365 = v364 == false;
        if (v365){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v364);
        } else {
        }
        assert("Tensor range check" && 0 <= v363 && v363 < 8);
        assert("Tensor range check" && 0 <= v362 && v362 < 4);
        int v367;
        v367 = v362 + v356;
        int v368;
        v368 = 68 * v363;
        int v369;
        v369 = v368 + v367;
        float * v370;
        v370 = v335+v369;
        assert("Tensor range check" && 0 <= v346 && v346 < 4);
        int v372;
        v372 = 1088 * v346;
        int v373;
        v373 = threadIdx.x;
        int v374;
        v374 = v373 % 32;
        bool v375;
        v375 = 0 <= v374;
        bool v376;
        v376 = v375 == false;
        if (v376){
            assert("The index needs to be zero or positive." && v375);
        } else {
        }
        int v378;
        v378 = v374 % 4;
        int v379;
        v379 = v374 / 4;
        bool v380;
        v380 = v379 < 8;
        bool v381;
        v381 = v380 == false;
        if (v381){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v380);
        } else {
        }
        assert("Tensor range check" && 0 <= v379 && v379 < 8);
        assert("Tensor range check" && 0 <= v378 && v378 < 4);
        int v383;
        v383 = v378 + v372;
        int v384;
        v384 = 68 * v379;
        int v385;
        v385 = v384 + v383;
        float * v386;
        v386 = v337+v385;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> v388[2];
        int v389;
        v389 = 0;
        while (while_method_0(v389)){
            int v391;
            v391 = 0;
            while (while_method_0(v391)){
                assert("Tensor range check" && 0 <= v389 && v389 < 1);
                assert("Tensor range check" && 0 <= v391 && v391 < 1);
                int v393;
                v393 = 64 * v391;
                int v394;
                v394 = v393 + v332;
                int v395;
                v395 = 4096 * v389;
                int v396;
                v396 = v395 + v394;
                float * v397;
                v397 = v327+v396;
                // Pushing the loop unrolling to: 0
                int v399;
                v399 = 0;
                #pragma unroll
                while (while_method_1(v399)){
                    int v401;
                    v401 = 0;
                    #pragma unroll
                    while (while_method_0(v401)){
                        assert("Tensor range check" && 0 <= v399 && v399 < 2);
                        assert("Tensor range check" && 0 <= v401 && v401 < 1);
                        int v403;
                        v403 = v399 + v401;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v404 = v388[v403];
                        wmma::fill_fragment(v404, 0.0f);
                        v401 += 1 ;
                    }
                    v399 += 1 ;
                }
                // Poping the loop unrolling to: 0
                int v405;
                v405 = 0;
                while (while_method_2(v405)){
                    int v407;
                    v407 = v405 + 1;
                    bool v408;
                    v408 = v405 == 0;
                    int v409;
                    v409 = v405 % 2;
                    bool v410;
                    v410 = 0 <= v405;
                    bool v411;
                    v411 = v410 == false;
                    if (v411){
                        assert("The index needs to be zero or positive." && v410);
                    } else {
                    }
                    bool v413;
                    v413 = v405 < 1;
                    bool v414;
                    v414 = v413 == false;
                    if (v414){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v413);
                    } else {
                    }
                    bool v416;
                    v416 = v407 < 1;
                    Union0 v422;
                    if (v416){
                        bool v417;
                        v417 = 0 <= v407;
                        bool v418;
                        v418 = v417 == false;
                        if (v418){
                            assert("The index needs to be zero or positive." && v417);
                        } else {
                        }
                        v422 = Union0{Union0_1{v407}};
                    } else {
                        v422 = Union0{Union0_0{}};
                    }
                    assert("Tensor range check" && 0 <= v389 && v389 < 1);
                    int v423;
                    v423 = v395 + v330;
                    assert("Tensor range check" && 0 <= v405 && v405 < 1);
                    int v424;
                    v424 = 64 * v405;
                    int v425;
                    v425 = v424 + v423;
                    float * v426;
                    v426 = v323+v425;
                    assert("Tensor range check" && 0 <= v391 && v391 < 1);
                    int v428;
                    v428 = 4096 * v391;
                    int v429;
                    v429 = v428 + v15;
                    if (v408){
                        assert("Tensor range check" && 0 <= v405 && v405 < 1);
                        int v430;
                        v430 = v424 + v429;
                        float * v431;
                        v431 = v325+v430;
                        // Pushing the loop unrolling to: 0
                        v333.producer_acquire();
                        int v433;
                        v433 = threadIdx.x;
                        bool v434;
                        v434 = 0 <= v433;
                        bool v435;
                        v435 = v434 == false;
                        if (v435){
                            assert("The index needs to be zero or positive." && v434);
                        } else {
                        }
                        int v437;
                        v437 = v433 % 16;
                        int v438;
                        v438 = v433 / 16;
                        bool v439;
                        v439 = v438 < 16;
                        bool v440;
                        v440 = v439 == false;
                        if (v440){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v439);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v438 && v438 < 16);
                        assert("Tensor range check" && 0 <= v437 && v437 < 16);
                        int v442;
                        v442 = 4 * v437;
                        int v443;
                        v443 = 68 * v438;
                        int v444;
                        v444 = v443 + v442;
                        int v445;
                        v445 = 64 * v438;
                        int v446;
                        v446 = v445 + v442;
                        float * v447;
                        v447 = v337+v444;
                        float * v449;
                        v449 = v431+v446;
                        int v451;
                        v451 = 0;
                        #pragma unroll
                        while (while_method_3(v451)){
                            int v453;
                            v453 = 0;
                            #pragma unroll
                            while (while_method_0(v453)){
                                assert("Tensor range check" && 0 <= v451 && v451 < 4);
                                assert("Tensor range check" && 0 <= v453 && v453 < 1);
                                int v455;
                                v455 = 64 * v453;
                                int v456;
                                v456 = 1088 * v451;
                                int v457;
                                v457 = v456 + v455;
                                int v458;
                                v458 = 1024 * v451;
                                int v459;
                                v459 = v458 + v455;
                                constexpr int v460 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v449 + v459) % v460 == 0 && (unsigned long long)(v447 + v457) % v460 == 0);
                                cuda::memcpy_async(v447 + v457, v449 + v459, cuda::aligned_size_t<v460>(v460), v333);
                                v453 += 1 ;
                            }
                            v451 += 1 ;
                        }
                        v333.producer_commit();
                        // Poping the loop unrolling to: 0
                    } else {
                    }
                    // Pushing the loop unrolling to: 0
                    int v461;
                    v461 = threadIdx.x;
                    bool v462;
                    v462 = 0 <= v461;
                    bool v463;
                    v463 = v462 == false;
                    if (v463){
                        assert("The index needs to be zero or positive." && v462);
                    } else {
                    }
                    int v465;
                    v465 = v461 % 16;
                    int v466;
                    v466 = v461 / 16;
                    bool v467;
                    v467 = v466 < 16;
                    bool v468;
                    v468 = v467 == false;
                    if (v468){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v467);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v466 && v466 < 16);
                    assert("Tensor range check" && 0 <= v465 && v465 < 16);
                    int v470;
                    v470 = 4 * v465;
                    int v471;
                    v471 = 68 * v466;
                    int v472;
                    v472 = v471 + v470;
                    int v473;
                    v473 = 64 * v466;
                    int v474;
                    v474 = v473 + v470;
                    float * v475;
                    v475 = v335+v472;
                    float * v477;
                    v477 = v426+v474;
                    int v479;
                    v479 = 0;
                    #pragma unroll
                    while (while_method_3(v479)){
                        int v481;
                        v481 = 0;
                        #pragma unroll
                        while (while_method_0(v481)){
                            assert("Tensor range check" && 0 <= v479 && v479 < 4);
                            assert("Tensor range check" && 0 <= v481 && v481 < 1);
                            int v483;
                            v483 = 64 * v481;
                            int v484;
                            v484 = 1088 * v479;
                            int v485;
                            v485 = v484 + v483;
                            int v486;
                            v486 = 1024 * v479;
                            int v487;
                            v487 = v486 + v483;
                            int4* v488;
                            v488 = reinterpret_cast<int4*>(v477 + v487);
                            int4* v489;
                            v489 = reinterpret_cast<int4*>(v475 + v485);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v488) % 16 == 0 && reinterpret_cast<unsigned long long>(v489) % 16 == 0);
                            *v489 = *v488;
                            v481 += 1 ;
                        }
                        v479 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v490[1];
                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v491[8];
                    cuda::pipeline_consumer_wait_prior<0>(v333);;
                    __syncthreads();
                    // Pushing the loop unrolling to: 0
                    int v492;
                    v492 = 0;
                    #pragma unroll
                    while (while_method_0(v492)){
                        int v494;
                        v494 = 0;
                        #pragma unroll
                        while (while_method_4(v494)){
                            assert("Tensor range check" && 0 <= v492 && v492 < 1);
                            assert("Tensor range check" && 0 <= v494 && v494 < 8);
                            int v496;
                            v496 = 8 * v492;
                            int v497;
                            v497 = v496 + v494;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v498 = v491[v497];
                            assert("Tensor range check" && 0 <= v492 && v492 < 1);
                            int v499;
                            v499 = 1088 * v492;
                            assert("Tensor range check" && 0 <= v494 && v494 < 8);
                            int v500;
                            v500 = 8 * v494;
                            int v501;
                            v501 = v500 + v499;
                            int v502;
                            v502 = 0;
                            #pragma unroll
                            while (while_method_1(v502)){
                                int v504;
                                v504 = 0;
                                #pragma unroll
                                while (while_method_1(v504)){
                                    assert("Tensor range check" && 0 <= v502 && v502 < 2);
                                    assert("Tensor range check" && 0 <= v504 && v504 < 2);
                                    int v506;
                                    v506 = 4 * v504;
                                    int v507;
                                    v507 = v506 + v501;
                                    int v508;
                                    v508 = 544 * v502;
                                    int v509;
                                    v509 = v508 + v507;
                                    float v510;
                                    v510 = v386[v509];
                                    bool v511;
                                    v511 = 0 <= v504;
                                    bool v513;
                                    if (v511){
                                        bool v512;
                                        v512 = v504 < 2;
                                        v513 = v512;
                                    } else {
                                        v513 = false;
                                    }
                                    bool v514;
                                    v514 = v513 == false;
                                    if (v514){
                                        assert("The indices should be inside the range of the dimension." && v513);
                                    } else {
                                    }
                                    bool v516;
                                    v516 = 0 <= v502;
                                    bool v518;
                                    if (v516){
                                        bool v517;
                                        v517 = v502 < 2;
                                        v518 = v517;
                                    } else {
                                        v518 = false;
                                    }
                                    bool v519;
                                    v519 = v518 == false;
                                    if (v519){
                                        assert("The indices should be inside the range of the dimension." && v518);
                                    } else {
                                    }
                                    int v521;
                                    v521 = v502 * 2;
                                    int v522;
                                    v522 = v504 + v521;
                                    v498.x[v522] = wmma::__float_to_tf32(v510);
                                    v504 += 1 ;
                                }
                                v502 += 1 ;
                            }
                            v494 += 1 ;
                        }
                        v492 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    v333.consumer_release();
                    switch (v422.tag) {
                        case 0: { // None
                            break;
                        }
                        case 1: { // Some
                            int v523 = v422.case1.v0;
                            assert("Tensor range check" && 0 <= v523 && v523 < 1);
                            int v524;
                            v524 = 64 * v523;
                            int v525;
                            v525 = v524 + v429;
                            float * v526;
                            v526 = v325+v525;
                            __syncthreads();
                            // Pushing the loop unrolling to: 0
                            v333.producer_acquire();
                            int v528;
                            v528 = threadIdx.x;
                            bool v529;
                            v529 = 0 <= v528;
                            bool v530;
                            v530 = v529 == false;
                            if (v530){
                                assert("The index needs to be zero or positive." && v529);
                            } else {
                            }
                            int v532;
                            v532 = v528 % 16;
                            int v533;
                            v533 = v528 / 16;
                            bool v534;
                            v534 = v533 < 16;
                            bool v535;
                            v535 = v534 == false;
                            if (v535){
                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v534);
                            } else {
                            }
                            assert("Tensor range check" && 0 <= v533 && v533 < 16);
                            assert("Tensor range check" && 0 <= v532 && v532 < 16);
                            int v537;
                            v537 = 4 * v532;
                            int v538;
                            v538 = 68 * v533;
                            int v539;
                            v539 = v538 + v537;
                            int v540;
                            v540 = 64 * v533;
                            int v541;
                            v541 = v540 + v537;
                            float * v542;
                            v542 = v337+v539;
                            float * v544;
                            v544 = v526+v541;
                            int v546;
                            v546 = 0;
                            #pragma unroll
                            while (while_method_3(v546)){
                                int v548;
                                v548 = 0;
                                #pragma unroll
                                while (while_method_0(v548)){
                                    assert("Tensor range check" && 0 <= v546 && v546 < 4);
                                    assert("Tensor range check" && 0 <= v548 && v548 < 1);
                                    int v550;
                                    v550 = 64 * v548;
                                    int v551;
                                    v551 = 1088 * v546;
                                    int v552;
                                    v552 = v551 + v550;
                                    int v553;
                                    v553 = 1024 * v546;
                                    int v554;
                                    v554 = v553 + v550;
                                    constexpr int v555 = sizeof(float) * 4;
                                    assert("Pointer alignment check" && (unsigned long long)(v544 + v554) % v555 == 0 && (unsigned long long)(v542 + v552) % v555 == 0);
                                    cuda::memcpy_async(v542 + v552, v544 + v554, cuda::aligned_size_t<v555>(v555), v333);
                                    v548 += 1 ;
                                }
                                v546 += 1 ;
                            }
                            v333.producer_commit();
                            // Poping the loop unrolling to: 0
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false); __trap();
                        }
                    }
                    // Pushing the loop unrolling to: 0
                    int v556;
                    v556 = 0;
                    #pragma unroll
                    while (while_method_1(v556)){
                        int v558;
                        v558 = 0;
                        #pragma unroll
                        while (while_method_4(v558)){
                            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v560 = v490[0];
                            assert("Tensor range check" && 0 <= v556 && v556 < 2);
                            int v561;
                            v561 = 1088 * v556;
                            assert("Tensor range check" && 0 <= v558 && v558 < 8);
                            int v562;
                            v562 = 8 * v558;
                            int v563;
                            v563 = v562 + v561;
                            int v564;
                            v564 = 0;
                            #pragma unroll
                            while (while_method_1(v564)){
                                int v566;
                                v566 = 0;
                                #pragma unroll
                                while (while_method_1(v566)){
                                    assert("Tensor range check" && 0 <= v564 && v564 < 2);
                                    assert("Tensor range check" && 0 <= v566 && v566 < 2);
                                    int v568;
                                    v568 = 544 * v566;
                                    int v569;
                                    v569 = v568 + v563;
                                    int v570;
                                    v570 = 4 * v564;
                                    int v571;
                                    v571 = v570 + v569;
                                    float v572;
                                    v572 = v370[v571];
                                    bool v573;
                                    v573 = 0 <= v566;
                                    bool v575;
                                    if (v573){
                                        bool v574;
                                        v574 = v566 < 2;
                                        v575 = v574;
                                    } else {
                                        v575 = false;
                                    }
                                    bool v576;
                                    v576 = v575 == false;
                                    if (v576){
                                        assert("The indices should be inside the range of the dimension." && v575);
                                    } else {
                                    }
                                    bool v578;
                                    v578 = 0 <= v564;
                                    bool v580;
                                    if (v578){
                                        bool v579;
                                        v579 = v564 < 2;
                                        v580 = v579;
                                    } else {
                                        v580 = false;
                                    }
                                    bool v581;
                                    v581 = v580 == false;
                                    if (v581){
                                        assert("The indices should be inside the range of the dimension." && v580);
                                    } else {
                                    }
                                    int v583;
                                    v583 = v564 * 2;
                                    int v584;
                                    v584 = v566 + v583;
                                    v560.x[v584] = wmma::__float_to_tf32(v572);
                                    v566 += 1 ;
                                }
                                v564 += 1 ;
                            }
                            int v585;
                            v585 = 0;
                            #pragma unroll
                            while (while_method_0(v585)){
                                assert("Tensor range check" && 0 <= v556 && v556 < 2);
                                assert("Tensor range check" && 0 <= v585 && v585 < 1);
                                int v587;
                                v587 = v556 + v585;
                                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v588 = v388[v587];
                                assert("Tensor range check" && 0 <= v585 && v585 < 1);
                                assert("Tensor range check" && 0 <= v558 && v558 < 8);
                                int v589;
                                v589 = 8 * v585;
                                int v590;
                                v590 = v589 + v558;
                                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v591 = v491[v590];
                                wmma::mma_sync(v588, v560, v591, v588);
                                v585 += 1 ;
                            }
                            v558 += 1 ;
                        }
                        v556 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    __syncthreads();
                    v405 = v407;
                }
                // Pushing the loop unrolling to: 0
                int v592;
                v592 = 0;
                #pragma unroll
                while (while_method_1(v592)){
                    int v594;
                    v594 = 0;
                    #pragma unroll
                    while (while_method_0(v594)){
                        assert("Tensor range check" && 0 <= v592 && v592 < 2);
                        assert("Tensor range check" && 0 <= v594 && v594 < 1);
                        int v596;
                        v596 = v592 + v594;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v597 = v388[v596];
                        assert("Tensor range check" && 0 <= v592 && v592 < 2);
                        assert("Tensor range check" && 0 <= v594 && v594 < 1);
                        int v598;
                        v598 = 16 * v594;
                        int v599;
                        v599 = 1152 * v592;
                        int v600;
                        v600 = v599 + v598;
                        float * v601;
                        v601 = v354+v600;
                        wmma::store_matrix_sync(v601, v597, 72, wmma::mem_row_major);
                        v594 += 1 ;
                    }
                    v592 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v603;
                v603 = threadIdx.x;
                bool v604;
                v604 = 0 <= v603;
                bool v605;
                v605 = v604 == false;
                if (v605){
                    assert("The index needs to be zero or positive." && v604);
                } else {
                }
                int v607;
                v607 = v603 % 16;
                int v608;
                v608 = v603 / 16;
                bool v609;
                v609 = v608 < 16;
                bool v610;
                v610 = v609 == false;
                if (v610){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v609);
                } else {
                }
                assert("Tensor range check" && 0 <= v608 && v608 < 16);
                assert("Tensor range check" && 0 <= v607 && v607 < 16);
                int v612;
                v612 = 4 * v607;
                int v613;
                v613 = 64 * v608;
                int v614;
                v614 = v613 + v612;
                int v615;
                v615 = 72 * v608;
                int v616;
                v616 = v615 + v612;
                float * v617;
                v617 = v397+v614;
                float * v619;
                v619 = v339+v616;
                int v621;
                v621 = 0;
                #pragma unroll
                while (while_method_3(v621)){
                    int v623;
                    v623 = 0;
                    #pragma unroll
                    while (while_method_0(v623)){
                        assert("Tensor range check" && 0 <= v621 && v621 < 4);
                        assert("Tensor range check" && 0 <= v623 && v623 < 1);
                        int v625;
                        v625 = 64 * v623;
                        int v626;
                        v626 = 1024 * v621;
                        int v627;
                        v627 = v626 + v625;
                        int v628;
                        v628 = 1152 * v621;
                        int v629;
                        v629 = v628 + v625;
                        int4* v630;
                        v630 = reinterpret_cast<int4*>(v619 + v629);
                        int4* v631;
                        v631 = reinterpret_cast<int4*>(v617 + v627);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v630) % 16 == 0 && reinterpret_cast<unsigned long long>(v631) % 16 == 0);
                        *v631 = *v630;
                        v623 += 1 ;
                    }
                    v621 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v391 += 1 ;
            }
            v389 += 1 ;
        }
        float * v632;
        v632 = reinterpret_cast<float *>(&v1[1966080ull]);
        method_0(v632, v327);
        float * v634;
        v634 = reinterpret_cast<float *>(&v1[2359296ull]);
        method_1(v634, v632);
        float * v636;
        v636 = reinterpret_cast<float *>(&v0[524288ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16);
        float * v638;
        v638 = reinterpret_cast<float *>(&v1[2752512ull]);
        int v640;
        v640 = blockIdx.x;
        assert("Tensor range check" && 0 <= v640 && v640 < 24);
        int v641;
        v641 = 4096 * v640;
        int v642;
        v642 = blockIdx.x;
        assert("Tensor range check" && 0 <= v642 && v642 < 24);
        int v643;
        v643 = 4096 * v642;
        cuda::pipeline<cuda::thread_scope_thread> v644 = cuda::make_pipeline();
        extern __shared__ unsigned char v645[];
        float * v646;
        v646 = reinterpret_cast<float *>(&v645[0ull]);
        float * v648;
        v648 = reinterpret_cast<float *>(&v645[17408ull]);
        float * v650;
        v650 = reinterpret_cast<float *>(&v645[0ull]);
        int v652;
        v652 = threadIdx.x;
        int v653;
        v653 = v652 / 32;
        bool v654;
        v654 = 0 <= v653;
        bool v655;
        v655 = v654 == false;
        if (v655){
            assert("The index needs to be zero or positive." && v654);
        } else {
        }
        int v657;
        v657 = v653 % 4;
        int v658;
        v658 = v653 / 4;
        bool v659;
        v659 = v658 < 2;
        bool v660;
        v660 = v659 == false;
        if (v660){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v659);
        } else {
        }
        assert("Tensor range check" && 0 <= v658 && v658 < 2);
        assert("Tensor range check" && 0 <= v657 && v657 < 4);
        int v662;
        v662 = 16 * v657;
        int v663;
        v663 = 2304 * v658;
        int v664;
        v664 = v663 + v662;
        float * v665;
        v665 = v650+v664;
        assert("Tensor range check" && 0 <= v658 && v658 < 2);
        int v667;
        v667 = 2176 * v658;
        int v668;
        v668 = threadIdx.x;
        int v669;
        v669 = v668 % 32;
        bool v670;
        v670 = 0 <= v669;
        bool v671;
        v671 = v670 == false;
        if (v671){
            assert("The index needs to be zero or positive." && v670);
        } else {
        }
        int v673;
        v673 = v669 % 4;
        int v674;
        v674 = v669 / 4;
        bool v675;
        v675 = v674 < 8;
        bool v676;
        v676 = v675 == false;
        if (v676){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v675);
        } else {
        }
        assert("Tensor range check" && 0 <= v674 && v674 < 8);
        assert("Tensor range check" && 0 <= v673 && v673 < 4);
        int v678;
        v678 = v673 + v667;
        int v679;
        v679 = 68 * v674;
        int v680;
        v680 = v679 + v678;
        float * v681;
        v681 = v646+v680;
        assert("Tensor range check" && 0 <= v657 && v657 < 4);
        int v683;
        v683 = 1088 * v657;
        int v684;
        v684 = threadIdx.x;
        int v685;
        v685 = v684 % 32;
        bool v686;
        v686 = 0 <= v685;
        bool v687;
        v687 = v686 == false;
        if (v687){
            assert("The index needs to be zero or positive." && v686);
        } else {
        }
        int v689;
        v689 = v685 % 4;
        int v690;
        v690 = v685 / 4;
        bool v691;
        v691 = v690 < 8;
        bool v692;
        v692 = v691 == false;
        if (v692){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v691);
        } else {
        }
        assert("Tensor range check" && 0 <= v690 && v690 < 8);
        assert("Tensor range check" && 0 <= v689 && v689 < 4);
        int v694;
        v694 = v689 + v683;
        int v695;
        v695 = 68 * v690;
        int v696;
        v696 = v695 + v694;
        float * v697;
        v697 = v648+v696;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> v699[2];
        int v700;
        v700 = 0;
        while (while_method_0(v700)){
            int v702;
            v702 = 0;
            while (while_method_0(v702)){
                assert("Tensor range check" && 0 <= v700 && v700 < 1);
                assert("Tensor range check" && 0 <= v702 && v702 < 1);
                int v704;
                v704 = 64 * v702;
                int v705;
                v705 = v704 + v643;
                int v706;
                v706 = 4096 * v700;
                int v707;
                v707 = v706 + v705;
                float * v708;
                v708 = v638+v707;
                // Pushing the loop unrolling to: 0
                int v710;
                v710 = 0;
                #pragma unroll
                while (while_method_1(v710)){
                    int v712;
                    v712 = 0;
                    #pragma unroll
                    while (while_method_0(v712)){
                        assert("Tensor range check" && 0 <= v710 && v710 < 2);
                        assert("Tensor range check" && 0 <= v712 && v712 < 1);
                        int v714;
                        v714 = v710 + v712;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v715 = v699[v714];
                        wmma::fill_fragment(v715, 0.0f);
                        v712 += 1 ;
                    }
                    v710 += 1 ;
                }
                // Poping the loop unrolling to: 0
                int v716;
                v716 = 0;
                while (while_method_2(v716)){
                    int v718;
                    v718 = v716 + 1;
                    bool v719;
                    v719 = v716 == 0;
                    int v720;
                    v720 = v716 % 2;
                    bool v721;
                    v721 = 0 <= v716;
                    bool v722;
                    v722 = v721 == false;
                    if (v722){
                        assert("The index needs to be zero or positive." && v721);
                    } else {
                    }
                    bool v724;
                    v724 = v716 < 1;
                    bool v725;
                    v725 = v724 == false;
                    if (v725){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v724);
                    } else {
                    }
                    bool v727;
                    v727 = v718 < 1;
                    Union0 v733;
                    if (v727){
                        bool v728;
                        v728 = 0 <= v718;
                        bool v729;
                        v729 = v728 == false;
                        if (v729){
                            assert("The index needs to be zero or positive." && v728);
                        } else {
                        }
                        v733 = Union0{Union0_1{v718}};
                    } else {
                        v733 = Union0{Union0_0{}};
                    }
                    assert("Tensor range check" && 0 <= v700 && v700 < 1);
                    int v734;
                    v734 = v706 + v641;
                    assert("Tensor range check" && 0 <= v716 && v716 < 1);
                    int v735;
                    v735 = 64 * v716;
                    int v736;
                    v736 = v735 + v734;
                    float * v737;
                    v737 = v634+v736;
                    assert("Tensor range check" && 0 <= v702 && v702 < 1);
                    int v739;
                    v739 = 4096 * v702;
                    int v740;
                    v740 = v739 + v15;
                    if (v719){
                        assert("Tensor range check" && 0 <= v716 && v716 < 1);
                        int v741;
                        v741 = v735 + v740;
                        float * v742;
                        v742 = v636+v741;
                        // Pushing the loop unrolling to: 0
                        v644.producer_acquire();
                        int v744;
                        v744 = threadIdx.x;
                        bool v745;
                        v745 = 0 <= v744;
                        bool v746;
                        v746 = v745 == false;
                        if (v746){
                            assert("The index needs to be zero or positive." && v745);
                        } else {
                        }
                        int v748;
                        v748 = v744 % 16;
                        int v749;
                        v749 = v744 / 16;
                        bool v750;
                        v750 = v749 < 16;
                        bool v751;
                        v751 = v750 == false;
                        if (v751){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v750);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v749 && v749 < 16);
                        assert("Tensor range check" && 0 <= v748 && v748 < 16);
                        int v753;
                        v753 = 4 * v748;
                        int v754;
                        v754 = 68 * v749;
                        int v755;
                        v755 = v754 + v753;
                        int v756;
                        v756 = 64 * v749;
                        int v757;
                        v757 = v756 + v753;
                        float * v758;
                        v758 = v648+v755;
                        float * v760;
                        v760 = v742+v757;
                        int v762;
                        v762 = 0;
                        #pragma unroll
                        while (while_method_3(v762)){
                            int v764;
                            v764 = 0;
                            #pragma unroll
                            while (while_method_0(v764)){
                                assert("Tensor range check" && 0 <= v762 && v762 < 4);
                                assert("Tensor range check" && 0 <= v764 && v764 < 1);
                                int v766;
                                v766 = 64 * v764;
                                int v767;
                                v767 = 1088 * v762;
                                int v768;
                                v768 = v767 + v766;
                                int v769;
                                v769 = 1024 * v762;
                                int v770;
                                v770 = v769 + v766;
                                constexpr int v771 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v760 + v770) % v771 == 0 && (unsigned long long)(v758 + v768) % v771 == 0);
                                cuda::memcpy_async(v758 + v768, v760 + v770, cuda::aligned_size_t<v771>(v771), v644);
                                v764 += 1 ;
                            }
                            v762 += 1 ;
                        }
                        v644.producer_commit();
                        // Poping the loop unrolling to: 0
                    } else {
                    }
                    // Pushing the loop unrolling to: 0
                    int v772;
                    v772 = threadIdx.x;
                    bool v773;
                    v773 = 0 <= v772;
                    bool v774;
                    v774 = v773 == false;
                    if (v774){
                        assert("The index needs to be zero or positive." && v773);
                    } else {
                    }
                    int v776;
                    v776 = v772 % 16;
                    int v777;
                    v777 = v772 / 16;
                    bool v778;
                    v778 = v777 < 16;
                    bool v779;
                    v779 = v778 == false;
                    if (v779){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v778);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v777 && v777 < 16);
                    assert("Tensor range check" && 0 <= v776 && v776 < 16);
                    int v781;
                    v781 = 4 * v776;
                    int v782;
                    v782 = 68 * v777;
                    int v783;
                    v783 = v782 + v781;
                    int v784;
                    v784 = 64 * v777;
                    int v785;
                    v785 = v784 + v781;
                    float * v786;
                    v786 = v646+v783;
                    float * v788;
                    v788 = v737+v785;
                    int v790;
                    v790 = 0;
                    #pragma unroll
                    while (while_method_3(v790)){
                        int v792;
                        v792 = 0;
                        #pragma unroll
                        while (while_method_0(v792)){
                            assert("Tensor range check" && 0 <= v790 && v790 < 4);
                            assert("Tensor range check" && 0 <= v792 && v792 < 1);
                            int v794;
                            v794 = 64 * v792;
                            int v795;
                            v795 = 1088 * v790;
                            int v796;
                            v796 = v795 + v794;
                            int v797;
                            v797 = 1024 * v790;
                            int v798;
                            v798 = v797 + v794;
                            int4* v799;
                            v799 = reinterpret_cast<int4*>(v788 + v798);
                            int4* v800;
                            v800 = reinterpret_cast<int4*>(v786 + v796);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v799) % 16 == 0 && reinterpret_cast<unsigned long long>(v800) % 16 == 0);
                            *v800 = *v799;
                            v792 += 1 ;
                        }
                        v790 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v801[1];
                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v802[8];
                    cuda::pipeline_consumer_wait_prior<0>(v644);;
                    __syncthreads();
                    // Pushing the loop unrolling to: 0
                    int v803;
                    v803 = 0;
                    #pragma unroll
                    while (while_method_0(v803)){
                        int v805;
                        v805 = 0;
                        #pragma unroll
                        while (while_method_4(v805)){
                            assert("Tensor range check" && 0 <= v803 && v803 < 1);
                            assert("Tensor range check" && 0 <= v805 && v805 < 8);
                            int v807;
                            v807 = 8 * v803;
                            int v808;
                            v808 = v807 + v805;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v809 = v802[v808];
                            assert("Tensor range check" && 0 <= v803 && v803 < 1);
                            int v810;
                            v810 = 1088 * v803;
                            assert("Tensor range check" && 0 <= v805 && v805 < 8);
                            int v811;
                            v811 = 8 * v805;
                            int v812;
                            v812 = v811 + v810;
                            int v813;
                            v813 = 0;
                            #pragma unroll
                            while (while_method_1(v813)){
                                int v815;
                                v815 = 0;
                                #pragma unroll
                                while (while_method_1(v815)){
                                    assert("Tensor range check" && 0 <= v813 && v813 < 2);
                                    assert("Tensor range check" && 0 <= v815 && v815 < 2);
                                    int v817;
                                    v817 = 4 * v815;
                                    int v818;
                                    v818 = v817 + v812;
                                    int v819;
                                    v819 = 544 * v813;
                                    int v820;
                                    v820 = v819 + v818;
                                    float v821;
                                    v821 = v697[v820];
                                    bool v822;
                                    v822 = 0 <= v815;
                                    bool v824;
                                    if (v822){
                                        bool v823;
                                        v823 = v815 < 2;
                                        v824 = v823;
                                    } else {
                                        v824 = false;
                                    }
                                    bool v825;
                                    v825 = v824 == false;
                                    if (v825){
                                        assert("The indices should be inside the range of the dimension." && v824);
                                    } else {
                                    }
                                    bool v827;
                                    v827 = 0 <= v813;
                                    bool v829;
                                    if (v827){
                                        bool v828;
                                        v828 = v813 < 2;
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
                                    int v832;
                                    v832 = v813 * 2;
                                    int v833;
                                    v833 = v815 + v832;
                                    v809.x[v833] = wmma::__float_to_tf32(v821);
                                    v815 += 1 ;
                                }
                                v813 += 1 ;
                            }
                            v805 += 1 ;
                        }
                        v803 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    v644.consumer_release();
                    switch (v733.tag) {
                        case 0: { // None
                            break;
                        }
                        case 1: { // Some
                            int v834 = v733.case1.v0;
                            assert("Tensor range check" && 0 <= v834 && v834 < 1);
                            int v835;
                            v835 = 64 * v834;
                            int v836;
                            v836 = v835 + v740;
                            float * v837;
                            v837 = v636+v836;
                            __syncthreads();
                            // Pushing the loop unrolling to: 0
                            v644.producer_acquire();
                            int v839;
                            v839 = threadIdx.x;
                            bool v840;
                            v840 = 0 <= v839;
                            bool v841;
                            v841 = v840 == false;
                            if (v841){
                                assert("The index needs to be zero or positive." && v840);
                            } else {
                            }
                            int v843;
                            v843 = v839 % 16;
                            int v844;
                            v844 = v839 / 16;
                            bool v845;
                            v845 = v844 < 16;
                            bool v846;
                            v846 = v845 == false;
                            if (v846){
                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v845);
                            } else {
                            }
                            assert("Tensor range check" && 0 <= v844 && v844 < 16);
                            assert("Tensor range check" && 0 <= v843 && v843 < 16);
                            int v848;
                            v848 = 4 * v843;
                            int v849;
                            v849 = 68 * v844;
                            int v850;
                            v850 = v849 + v848;
                            int v851;
                            v851 = 64 * v844;
                            int v852;
                            v852 = v851 + v848;
                            float * v853;
                            v853 = v648+v850;
                            float * v855;
                            v855 = v837+v852;
                            int v857;
                            v857 = 0;
                            #pragma unroll
                            while (while_method_3(v857)){
                                int v859;
                                v859 = 0;
                                #pragma unroll
                                while (while_method_0(v859)){
                                    assert("Tensor range check" && 0 <= v857 && v857 < 4);
                                    assert("Tensor range check" && 0 <= v859 && v859 < 1);
                                    int v861;
                                    v861 = 64 * v859;
                                    int v862;
                                    v862 = 1088 * v857;
                                    int v863;
                                    v863 = v862 + v861;
                                    int v864;
                                    v864 = 1024 * v857;
                                    int v865;
                                    v865 = v864 + v861;
                                    constexpr int v866 = sizeof(float) * 4;
                                    assert("Pointer alignment check" && (unsigned long long)(v855 + v865) % v866 == 0 && (unsigned long long)(v853 + v863) % v866 == 0);
                                    cuda::memcpy_async(v853 + v863, v855 + v865, cuda::aligned_size_t<v866>(v866), v644);
                                    v859 += 1 ;
                                }
                                v857 += 1 ;
                            }
                            v644.producer_commit();
                            // Poping the loop unrolling to: 0
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false); __trap();
                        }
                    }
                    // Pushing the loop unrolling to: 0
                    int v867;
                    v867 = 0;
                    #pragma unroll
                    while (while_method_1(v867)){
                        int v869;
                        v869 = 0;
                        #pragma unroll
                        while (while_method_4(v869)){
                            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v871 = v801[0];
                            assert("Tensor range check" && 0 <= v867 && v867 < 2);
                            int v872;
                            v872 = 1088 * v867;
                            assert("Tensor range check" && 0 <= v869 && v869 < 8);
                            int v873;
                            v873 = 8 * v869;
                            int v874;
                            v874 = v873 + v872;
                            int v875;
                            v875 = 0;
                            #pragma unroll
                            while (while_method_1(v875)){
                                int v877;
                                v877 = 0;
                                #pragma unroll
                                while (while_method_1(v877)){
                                    assert("Tensor range check" && 0 <= v875 && v875 < 2);
                                    assert("Tensor range check" && 0 <= v877 && v877 < 2);
                                    int v879;
                                    v879 = 544 * v877;
                                    int v880;
                                    v880 = v879 + v874;
                                    int v881;
                                    v881 = 4 * v875;
                                    int v882;
                                    v882 = v881 + v880;
                                    float v883;
                                    v883 = v681[v882];
                                    bool v884;
                                    v884 = 0 <= v877;
                                    bool v886;
                                    if (v884){
                                        bool v885;
                                        v885 = v877 < 2;
                                        v886 = v885;
                                    } else {
                                        v886 = false;
                                    }
                                    bool v887;
                                    v887 = v886 == false;
                                    if (v887){
                                        assert("The indices should be inside the range of the dimension." && v886);
                                    } else {
                                    }
                                    bool v889;
                                    v889 = 0 <= v875;
                                    bool v891;
                                    if (v889){
                                        bool v890;
                                        v890 = v875 < 2;
                                        v891 = v890;
                                    } else {
                                        v891 = false;
                                    }
                                    bool v892;
                                    v892 = v891 == false;
                                    if (v892){
                                        assert("The indices should be inside the range of the dimension." && v891);
                                    } else {
                                    }
                                    int v894;
                                    v894 = v875 * 2;
                                    int v895;
                                    v895 = v877 + v894;
                                    v871.x[v895] = wmma::__float_to_tf32(v883);
                                    v877 += 1 ;
                                }
                                v875 += 1 ;
                            }
                            int v896;
                            v896 = 0;
                            #pragma unroll
                            while (while_method_0(v896)){
                                assert("Tensor range check" && 0 <= v867 && v867 < 2);
                                assert("Tensor range check" && 0 <= v896 && v896 < 1);
                                int v898;
                                v898 = v867 + v896;
                                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v899 = v699[v898];
                                assert("Tensor range check" && 0 <= v896 && v896 < 1);
                                assert("Tensor range check" && 0 <= v869 && v869 < 8);
                                int v900;
                                v900 = 8 * v896;
                                int v901;
                                v901 = v900 + v869;
                                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v902 = v802[v901];
                                wmma::mma_sync(v899, v871, v902, v899);
                                v896 += 1 ;
                            }
                            v869 += 1 ;
                        }
                        v867 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    __syncthreads();
                    v716 = v718;
                }
                // Pushing the loop unrolling to: 0
                int v903;
                v903 = 0;
                #pragma unroll
                while (while_method_1(v903)){
                    int v905;
                    v905 = 0;
                    #pragma unroll
                    while (while_method_0(v905)){
                        assert("Tensor range check" && 0 <= v903 && v903 < 2);
                        assert("Tensor range check" && 0 <= v905 && v905 < 1);
                        int v907;
                        v907 = v903 + v905;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v908 = v699[v907];
                        assert("Tensor range check" && 0 <= v903 && v903 < 2);
                        assert("Tensor range check" && 0 <= v905 && v905 < 1);
                        int v909;
                        v909 = 16 * v905;
                        int v910;
                        v910 = 1152 * v903;
                        int v911;
                        v911 = v910 + v909;
                        float * v912;
                        v912 = v665+v911;
                        wmma::store_matrix_sync(v912, v908, 72, wmma::mem_row_major);
                        v905 += 1 ;
                    }
                    v903 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v914;
                v914 = threadIdx.x;
                bool v915;
                v915 = 0 <= v914;
                bool v916;
                v916 = v915 == false;
                if (v916){
                    assert("The index needs to be zero or positive." && v915);
                } else {
                }
                int v918;
                v918 = v914 % 16;
                int v919;
                v919 = v914 / 16;
                bool v920;
                v920 = v919 < 16;
                bool v921;
                v921 = v920 == false;
                if (v921){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v920);
                } else {
                }
                assert("Tensor range check" && 0 <= v919 && v919 < 16);
                assert("Tensor range check" && 0 <= v918 && v918 < 16);
                int v923;
                v923 = 4 * v918;
                int v924;
                v924 = 64 * v919;
                int v925;
                v925 = v924 + v923;
                int v926;
                v926 = 72 * v919;
                int v927;
                v927 = v926 + v923;
                float * v928;
                v928 = v708+v925;
                float * v930;
                v930 = v650+v927;
                int v932;
                v932 = 0;
                #pragma unroll
                while (while_method_3(v932)){
                    int v934;
                    v934 = 0;
                    #pragma unroll
                    while (while_method_0(v934)){
                        assert("Tensor range check" && 0 <= v932 && v932 < 4);
                        assert("Tensor range check" && 0 <= v934 && v934 < 1);
                        int v936;
                        v936 = 64 * v934;
                        int v937;
                        v937 = 1024 * v932;
                        int v938;
                        v938 = v937 + v936;
                        int v939;
                        v939 = 1152 * v932;
                        int v940;
                        v940 = v939 + v936;
                        int4* v941;
                        v941 = reinterpret_cast<int4*>(v930 + v940);
                        int4* v942;
                        v942 = reinterpret_cast<int4*>(v928 + v938);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v941) % 16 == 0 && reinterpret_cast<unsigned long long>(v942) % 16 == 0);
                        *v942 = *v941;
                        v934 += 1 ;
                    }
                    v932 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v702 += 1 ;
            }
            v700 += 1 ;
        }
        float * v943;
        v943 = reinterpret_cast<float *>(&v1[3145728ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16);
        int v945;
        v945 = 98304 * v9;
        int * v946;
        v946 = reinterpret_cast<int *>(&v1[9437184ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 16);
        int v948;
        v948 = 1536 * v9;
        method_3(v946, v948, v943, v945, v638, v8);
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
        v11 = reinterpret_cast<float *>(&v1[0ull]);
        float * v13;
        v13 = reinterpret_cast<float *>(&v0[0ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 4);
        int v15;
        v15 = 4096 * v9;
        float * v16;
        v16 = reinterpret_cast<float *>(&v1[393216ull]);
        int v18;
        v18 = blockIdx.x;
        assert("Tensor range check" && 0 <= v18 && v18 < 24);
        int v19;
        v19 = 4096 * v18;
        int v20;
        v20 = blockIdx.x;
        assert("Tensor range check" && 0 <= v20 && v20 < 24);
        int v21;
        v21 = 4096 * v20;
        cuda::pipeline<cuda::thread_scope_thread> v22 = cuda::make_pipeline();
        extern __shared__ unsigned char v23[];
        float * v24;
        v24 = reinterpret_cast<float *>(&v23[0ull]);
        float * v26;
        v26 = reinterpret_cast<float *>(&v23[17408ull]);
        float * v28;
        v28 = reinterpret_cast<float *>(&v23[0ull]);
        int v30;
        v30 = threadIdx.x;
        int v31;
        v31 = v30 / 32;
        bool v32;
        v32 = 0 <= v31;
        bool v33;
        v33 = v32 == false;
        if (v33){
            assert("The index needs to be zero or positive." && v32);
        } else {
        }
        int v35;
        v35 = v31 % 4;
        int v36;
        v36 = v31 / 4;
        bool v37;
        v37 = v36 < 2;
        bool v38;
        v38 = v37 == false;
        if (v38){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v37);
        } else {
        }
        assert("Tensor range check" && 0 <= v36 && v36 < 2);
        assert("Tensor range check" && 0 <= v35 && v35 < 4);
        int v40;
        v40 = 16 * v35;
        int v41;
        v41 = 2304 * v36;
        int v42;
        v42 = v41 + v40;
        float * v43;
        v43 = v28+v42;
        assert("Tensor range check" && 0 <= v36 && v36 < 2);
        int v45;
        v45 = 2176 * v36;
        int v46;
        v46 = threadIdx.x;
        int v47;
        v47 = v46 % 32;
        bool v48;
        v48 = 0 <= v47;
        bool v49;
        v49 = v48 == false;
        if (v49){
            assert("The index needs to be zero or positive." && v48);
        } else {
        }
        int v51;
        v51 = v47 % 4;
        int v52;
        v52 = v47 / 4;
        bool v53;
        v53 = v52 < 8;
        bool v54;
        v54 = v53 == false;
        if (v54){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v53);
        } else {
        }
        assert("Tensor range check" && 0 <= v52 && v52 < 8);
        assert("Tensor range check" && 0 <= v51 && v51 < 4);
        int v56;
        v56 = v51 + v45;
        int v57;
        v57 = 68 * v52;
        int v58;
        v58 = v57 + v56;
        float * v59;
        v59 = v24+v58;
        assert("Tensor range check" && 0 <= v35 && v35 < 4);
        int v61;
        v61 = 1088 * v35;
        int v62;
        v62 = threadIdx.x;
        int v63;
        v63 = v62 % 32;
        bool v64;
        v64 = 0 <= v63;
        bool v65;
        v65 = v64 == false;
        if (v65){
            assert("The index needs to be zero or positive." && v64);
        } else {
        }
        int v67;
        v67 = v63 % 4;
        int v68;
        v68 = v63 / 4;
        bool v69;
        v69 = v68 < 8;
        bool v70;
        v70 = v69 == false;
        if (v70){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v69);
        } else {
        }
        assert("Tensor range check" && 0 <= v68 && v68 < 8);
        assert("Tensor range check" && 0 <= v67 && v67 < 4);
        int v72;
        v72 = v67 + v61;
        int v73;
        v73 = 68 * v68;
        int v74;
        v74 = v73 + v72;
        float * v75;
        v75 = v26+v74;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> v77[2];
        int v78;
        v78 = 0;
        while (while_method_0(v78)){
            int v80;
            v80 = 0;
            while (while_method_0(v80)){
                assert("Tensor range check" && 0 <= v78 && v78 < 1);
                assert("Tensor range check" && 0 <= v80 && v80 < 1);
                int v82;
                v82 = 64 * v80;
                int v83;
                v83 = v82 + v21;
                int v84;
                v84 = 4096 * v78;
                int v85;
                v85 = v84 + v83;
                float * v86;
                v86 = v16+v85;
                // Pushing the loop unrolling to: 0
                int v88;
                v88 = 0;
                #pragma unroll
                while (while_method_1(v88)){
                    int v90;
                    v90 = 0;
                    #pragma unroll
                    while (while_method_0(v90)){
                        assert("Tensor range check" && 0 <= v88 && v88 < 2);
                        assert("Tensor range check" && 0 <= v90 && v90 < 1);
                        int v92;
                        v92 = v88 + v90;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v93 = v77[v92];
                        wmma::fill_fragment(v93, 0.0f);
                        v90 += 1 ;
                    }
                    v88 += 1 ;
                }
                // Poping the loop unrolling to: 0
                int v94;
                v94 = 0;
                while (while_method_2(v94)){
                    int v96;
                    v96 = v94 + 1;
                    bool v97;
                    v97 = v94 == 0;
                    int v98;
                    v98 = v94 % 2;
                    bool v99;
                    v99 = 0 <= v94;
                    bool v100;
                    v100 = v99 == false;
                    if (v100){
                        assert("The index needs to be zero or positive." && v99);
                    } else {
                    }
                    bool v102;
                    v102 = v94 < 1;
                    bool v103;
                    v103 = v102 == false;
                    if (v103){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v102);
                    } else {
                    }
                    bool v105;
                    v105 = v96 < 1;
                    Union0 v111;
                    if (v105){
                        bool v106;
                        v106 = 0 <= v96;
                        bool v107;
                        v107 = v106 == false;
                        if (v107){
                            assert("The index needs to be zero or positive." && v106);
                        } else {
                        }
                        v111 = Union0{Union0_1{v96}};
                    } else {
                        v111 = Union0{Union0_0{}};
                    }
                    assert("Tensor range check" && 0 <= v78 && v78 < 1);
                    int v112;
                    v112 = v84 + v19;
                    assert("Tensor range check" && 0 <= v94 && v94 < 1);
                    int v113;
                    v113 = 64 * v94;
                    int v114;
                    v114 = v113 + v112;
                    float * v115;
                    v115 = v11+v114;
                    assert("Tensor range check" && 0 <= v80 && v80 < 1);
                    int v117;
                    v117 = 4096 * v80;
                    int v118;
                    v118 = v117 + v15;
                    if (v97){
                        assert("Tensor range check" && 0 <= v94 && v94 < 1);
                        int v119;
                        v119 = v113 + v118;
                        float * v120;
                        v120 = v13+v119;
                        // Pushing the loop unrolling to: 0
                        v22.producer_acquire();
                        int v122;
                        v122 = threadIdx.x;
                        bool v123;
                        v123 = 0 <= v122;
                        bool v124;
                        v124 = v123 == false;
                        if (v124){
                            assert("The index needs to be zero or positive." && v123);
                        } else {
                        }
                        int v126;
                        v126 = v122 % 16;
                        int v127;
                        v127 = v122 / 16;
                        bool v128;
                        v128 = v127 < 16;
                        bool v129;
                        v129 = v128 == false;
                        if (v129){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v128);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v127 && v127 < 16);
                        assert("Tensor range check" && 0 <= v126 && v126 < 16);
                        int v131;
                        v131 = 4 * v126;
                        int v132;
                        v132 = 68 * v127;
                        int v133;
                        v133 = v132 + v131;
                        int v134;
                        v134 = 64 * v127;
                        int v135;
                        v135 = v134 + v131;
                        float * v136;
                        v136 = v26+v133;
                        float * v138;
                        v138 = v120+v135;
                        int v140;
                        v140 = 0;
                        #pragma unroll
                        while (while_method_3(v140)){
                            int v142;
                            v142 = 0;
                            #pragma unroll
                            while (while_method_0(v142)){
                                assert("Tensor range check" && 0 <= v140 && v140 < 4);
                                assert("Tensor range check" && 0 <= v142 && v142 < 1);
                                int v144;
                                v144 = 64 * v142;
                                int v145;
                                v145 = 1088 * v140;
                                int v146;
                                v146 = v145 + v144;
                                int v147;
                                v147 = 1024 * v140;
                                int v148;
                                v148 = v147 + v144;
                                constexpr int v149 = sizeof(float) * 4;
                                assert("Pointer alignment check" && (unsigned long long)(v138 + v148) % v149 == 0 && (unsigned long long)(v136 + v146) % v149 == 0);
                                cuda::memcpy_async(v136 + v146, v138 + v148, cuda::aligned_size_t<v149>(v149), v22);
                                v142 += 1 ;
                            }
                            v140 += 1 ;
                        }
                        v22.producer_commit();
                        // Poping the loop unrolling to: 0
                    } else {
                    }
                    // Pushing the loop unrolling to: 0
                    int v150;
                    v150 = threadIdx.x;
                    bool v151;
                    v151 = 0 <= v150;
                    bool v152;
                    v152 = v151 == false;
                    if (v152){
                        assert("The index needs to be zero or positive." && v151);
                    } else {
                    }
                    int v154;
                    v154 = v150 % 16;
                    int v155;
                    v155 = v150 / 16;
                    bool v156;
                    v156 = v155 < 16;
                    bool v157;
                    v157 = v156 == false;
                    if (v157){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v156);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v155 && v155 < 16);
                    assert("Tensor range check" && 0 <= v154 && v154 < 16);
                    int v159;
                    v159 = 4 * v154;
                    int v160;
                    v160 = 68 * v155;
                    int v161;
                    v161 = v160 + v159;
                    int v162;
                    v162 = 64 * v155;
                    int v163;
                    v163 = v162 + v159;
                    float * v164;
                    v164 = v24+v161;
                    float * v166;
                    v166 = v115+v163;
                    int v168;
                    v168 = 0;
                    #pragma unroll
                    while (while_method_3(v168)){
                        int v170;
                        v170 = 0;
                        #pragma unroll
                        while (while_method_0(v170)){
                            assert("Tensor range check" && 0 <= v168 && v168 < 4);
                            assert("Tensor range check" && 0 <= v170 && v170 < 1);
                            int v172;
                            v172 = 64 * v170;
                            int v173;
                            v173 = 1088 * v168;
                            int v174;
                            v174 = v173 + v172;
                            int v175;
                            v175 = 1024 * v168;
                            int v176;
                            v176 = v175 + v172;
                            int4* v177;
                            v177 = reinterpret_cast<int4*>(v166 + v176);
                            int4* v178;
                            v178 = reinterpret_cast<int4*>(v164 + v174);
                            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v177) % 16 == 0 && reinterpret_cast<unsigned long long>(v178) % 16 == 0);
                            *v178 = *v177;
                            v170 += 1 ;
                        }
                        v168 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v179[1];
                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v180[8];
                    cuda::pipeline_consumer_wait_prior<0>(v22);;
                    __syncthreads();
                    // Pushing the loop unrolling to: 0
                    int v181;
                    v181 = 0;
                    #pragma unroll
                    while (while_method_0(v181)){
                        int v183;
                        v183 = 0;
                        #pragma unroll
                        while (while_method_4(v183)){
                            assert("Tensor range check" && 0 <= v181 && v181 < 1);
                            assert("Tensor range check" && 0 <= v183 && v183 < 8);
                            int v185;
                            v185 = 8 * v181;
                            int v186;
                            v186 = v185 + v183;
                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v187 = v180[v186];
                            assert("Tensor range check" && 0 <= v181 && v181 < 1);
                            int v188;
                            v188 = 1088 * v181;
                            assert("Tensor range check" && 0 <= v183 && v183 < 8);
                            int v189;
                            v189 = 8 * v183;
                            int v190;
                            v190 = v189 + v188;
                            int v191;
                            v191 = 0;
                            #pragma unroll
                            while (while_method_1(v191)){
                                int v193;
                                v193 = 0;
                                #pragma unroll
                                while (while_method_1(v193)){
                                    assert("Tensor range check" && 0 <= v191 && v191 < 2);
                                    assert("Tensor range check" && 0 <= v193 && v193 < 2);
                                    int v195;
                                    v195 = 4 * v193;
                                    int v196;
                                    v196 = v195 + v190;
                                    int v197;
                                    v197 = 544 * v191;
                                    int v198;
                                    v198 = v197 + v196;
                                    float v199;
                                    v199 = v75[v198];
                                    bool v200;
                                    v200 = 0 <= v193;
                                    bool v202;
                                    if (v200){
                                        bool v201;
                                        v201 = v193 < 2;
                                        v202 = v201;
                                    } else {
                                        v202 = false;
                                    }
                                    bool v203;
                                    v203 = v202 == false;
                                    if (v203){
                                        assert("The indices should be inside the range of the dimension." && v202);
                                    } else {
                                    }
                                    bool v205;
                                    v205 = 0 <= v191;
                                    bool v207;
                                    if (v205){
                                        bool v206;
                                        v206 = v191 < 2;
                                        v207 = v206;
                                    } else {
                                        v207 = false;
                                    }
                                    bool v208;
                                    v208 = v207 == false;
                                    if (v208){
                                        assert("The indices should be inside the range of the dimension." && v207);
                                    } else {
                                    }
                                    int v210;
                                    v210 = v191 * 2;
                                    int v211;
                                    v211 = v193 + v210;
                                    v187.x[v211] = wmma::__float_to_tf32(v199);
                                    v193 += 1 ;
                                }
                                v191 += 1 ;
                            }
                            v183 += 1 ;
                        }
                        v181 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    v22.consumer_release();
                    switch (v111.tag) {
                        case 0: { // None
                            break;
                        }
                        case 1: { // Some
                            int v212 = v111.case1.v0;
                            assert("Tensor range check" && 0 <= v212 && v212 < 1);
                            int v213;
                            v213 = 64 * v212;
                            int v214;
                            v214 = v213 + v118;
                            float * v215;
                            v215 = v13+v214;
                            __syncthreads();
                            // Pushing the loop unrolling to: 0
                            v22.producer_acquire();
                            int v217;
                            v217 = threadIdx.x;
                            bool v218;
                            v218 = 0 <= v217;
                            bool v219;
                            v219 = v218 == false;
                            if (v219){
                                assert("The index needs to be zero or positive." && v218);
                            } else {
                            }
                            int v221;
                            v221 = v217 % 16;
                            int v222;
                            v222 = v217 / 16;
                            bool v223;
                            v223 = v222 < 16;
                            bool v224;
                            v224 = v223 == false;
                            if (v224){
                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v223);
                            } else {
                            }
                            assert("Tensor range check" && 0 <= v222 && v222 < 16);
                            assert("Tensor range check" && 0 <= v221 && v221 < 16);
                            int v226;
                            v226 = 4 * v221;
                            int v227;
                            v227 = 68 * v222;
                            int v228;
                            v228 = v227 + v226;
                            int v229;
                            v229 = 64 * v222;
                            int v230;
                            v230 = v229 + v226;
                            float * v231;
                            v231 = v26+v228;
                            float * v233;
                            v233 = v215+v230;
                            int v235;
                            v235 = 0;
                            #pragma unroll
                            while (while_method_3(v235)){
                                int v237;
                                v237 = 0;
                                #pragma unroll
                                while (while_method_0(v237)){
                                    assert("Tensor range check" && 0 <= v235 && v235 < 4);
                                    assert("Tensor range check" && 0 <= v237 && v237 < 1);
                                    int v239;
                                    v239 = 64 * v237;
                                    int v240;
                                    v240 = 1088 * v235;
                                    int v241;
                                    v241 = v240 + v239;
                                    int v242;
                                    v242 = 1024 * v235;
                                    int v243;
                                    v243 = v242 + v239;
                                    constexpr int v244 = sizeof(float) * 4;
                                    assert("Pointer alignment check" && (unsigned long long)(v233 + v243) % v244 == 0 && (unsigned long long)(v231 + v241) % v244 == 0);
                                    cuda::memcpy_async(v231 + v241, v233 + v243, cuda::aligned_size_t<v244>(v244), v22);
                                    v237 += 1 ;
                                }
                                v235 += 1 ;
                            }
                            v22.producer_commit();
                            // Poping the loop unrolling to: 0
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false); __trap();
                        }
                    }
                    // Pushing the loop unrolling to: 0
                    int v245;
                    v245 = 0;
                    #pragma unroll
                    while (while_method_1(v245)){
                        int v247;
                        v247 = 0;
                        #pragma unroll
                        while (while_method_4(v247)){
                            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v249 = v179[0];
                            assert("Tensor range check" && 0 <= v245 && v245 < 2);
                            int v250;
                            v250 = 1088 * v245;
                            assert("Tensor range check" && 0 <= v247 && v247 < 8);
                            int v251;
                            v251 = 8 * v247;
                            int v252;
                            v252 = v251 + v250;
                            int v253;
                            v253 = 0;
                            #pragma unroll
                            while (while_method_1(v253)){
                                int v255;
                                v255 = 0;
                                #pragma unroll
                                while (while_method_1(v255)){
                                    assert("Tensor range check" && 0 <= v253 && v253 < 2);
                                    assert("Tensor range check" && 0 <= v255 && v255 < 2);
                                    int v257;
                                    v257 = 544 * v255;
                                    int v258;
                                    v258 = v257 + v252;
                                    int v259;
                                    v259 = 4 * v253;
                                    int v260;
                                    v260 = v259 + v258;
                                    float v261;
                                    v261 = v59[v260];
                                    bool v262;
                                    v262 = 0 <= v255;
                                    bool v264;
                                    if (v262){
                                        bool v263;
                                        v263 = v255 < 2;
                                        v264 = v263;
                                    } else {
                                        v264 = false;
                                    }
                                    bool v265;
                                    v265 = v264 == false;
                                    if (v265){
                                        assert("The indices should be inside the range of the dimension." && v264);
                                    } else {
                                    }
                                    bool v267;
                                    v267 = 0 <= v253;
                                    bool v269;
                                    if (v267){
                                        bool v268;
                                        v268 = v253 < 2;
                                        v269 = v268;
                                    } else {
                                        v269 = false;
                                    }
                                    bool v270;
                                    v270 = v269 == false;
                                    if (v270){
                                        assert("The indices should be inside the range of the dimension." && v269);
                                    } else {
                                    }
                                    int v272;
                                    v272 = v253 * 2;
                                    int v273;
                                    v273 = v255 + v272;
                                    v249.x[v273] = wmma::__float_to_tf32(v261);
                                    v255 += 1 ;
                                }
                                v253 += 1 ;
                            }
                            int v274;
                            v274 = 0;
                            #pragma unroll
                            while (while_method_0(v274)){
                                assert("Tensor range check" && 0 <= v245 && v245 < 2);
                                assert("Tensor range check" && 0 <= v274 && v274 < 1);
                                int v276;
                                v276 = v245 + v274;
                                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v277 = v77[v276];
                                assert("Tensor range check" && 0 <= v274 && v274 < 1);
                                assert("Tensor range check" && 0 <= v247 && v247 < 8);
                                int v278;
                                v278 = 8 * v274;
                                int v279;
                                v279 = v278 + v247;
                                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v280 = v180[v279];
                                wmma::mma_sync(v277, v249, v280, v277);
                                v274 += 1 ;
                            }
                            v247 += 1 ;
                        }
                        v245 += 1 ;
                    }
                    // Poping the loop unrolling to: 0
                    __syncthreads();
                    v94 = v96;
                }
                // Pushing the loop unrolling to: 0
                int v281;
                v281 = 0;
                #pragma unroll
                while (while_method_1(v281)){
                    int v283;
                    v283 = 0;
                    #pragma unroll
                    while (while_method_0(v283)){
                        assert("Tensor range check" && 0 <= v281 && v281 < 2);
                        assert("Tensor range check" && 0 <= v283 && v283 < 1);
                        int v285;
                        v285 = v281 + v283;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v286 = v77[v285];
                        assert("Tensor range check" && 0 <= v281 && v281 < 2);
                        assert("Tensor range check" && 0 <= v283 && v283 < 1);
                        int v287;
                        v287 = 16 * v283;
                        int v288;
                        v288 = 1152 * v281;
                        int v289;
                        v289 = v288 + v287;
                        float * v290;
                        v290 = v43+v289;
                        wmma::store_matrix_sync(v290, v286, 72, wmma::mem_row_major);
                        v283 += 1 ;
                    }
                    v281 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                // Pushing the loop unrolling to: 0
                int v292;
                v292 = threadIdx.x;
                bool v293;
                v293 = 0 <= v292;
                bool v294;
                v294 = v293 == false;
                if (v294){
                    assert("The index needs to be zero or positive." && v293);
                } else {
                }
                int v296;
                v296 = v292 % 16;
                int v297;
                v297 = v292 / 16;
                bool v298;
                v298 = v297 < 16;
                bool v299;
                v299 = v298 == false;
                if (v299){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v298);
                } else {
                }
                assert("Tensor range check" && 0 <= v297 && v297 < 16);
                assert("Tensor range check" && 0 <= v296 && v296 < 16);
                int v301;
                v301 = 4 * v296;
                int v302;
                v302 = 64 * v297;
                int v303;
                v303 = v302 + v301;
                int v304;
                v304 = 72 * v297;
                int v305;
                v305 = v304 + v301;
                float * v306;
                v306 = v86+v303;
                float * v308;
                v308 = v28+v305;
                int v310;
                v310 = 0;
                #pragma unroll
                while (while_method_3(v310)){
                    int v312;
                    v312 = 0;
                    #pragma unroll
                    while (while_method_0(v312)){
                        assert("Tensor range check" && 0 <= v310 && v310 < 4);
                        assert("Tensor range check" && 0 <= v312 && v312 < 1);
                        int v314;
                        v314 = 64 * v312;
                        int v315;
                        v315 = 1024 * v310;
                        int v316;
                        v316 = v315 + v314;
                        int v317;
                        v317 = 1152 * v310;
                        int v318;
                        v318 = v317 + v314;
                        int4* v319;
                        v319 = reinterpret_cast<int4*>(v308 + v318);
                        int4* v320;
                        v320 = reinterpret_cast<int4*>(v306 + v316);
                        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v319) % 16 == 0 && reinterpret_cast<unsigned long long>(v320) % 16 == 0);
                        *v320 = *v319;
                        v312 += 1 ;
                    }
                    v310 += 1 ;
                }
                // Poping the loop unrolling to: 0
                __syncthreads();
                v80 += 1 ;
            }
            v78 += 1 ;
        }
        float * v321;
        v321 = reinterpret_cast<float *>(&v1[786432ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 4);
        int v323;
        v323 = 98304 * v9;
        int * v324;
        v324 = reinterpret_cast<int *>(&v1[2359296ull]);
        assert("Tensor range check" && 0 <= v9 && v9 < 4);
        int v326;
        v326 = 1536 * v9;
        method_4(v324, v326, v321, v323, v16, v8);
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
    v32 = 0
    v33 = "{}"
    print(v33.format('['),end="")
    v34 = 0
    while method1(v34):
        v36 = v32
        v37 = v36 >= 100
        del v36
        if v37:
            v38 = " ..."
            print(v33.format(v38),end="")
            del v38
            break
        else:
            pass
        del v37
        v39 = v34 == 0
        v40 = v39 != True
        del v39
        if v40:
            v41 = "; "
            print(v33.format(v41),end="")
            del v41
        else:
            pass
        del v40
        print(v33.format('['),end="")
        v42 = 0
        while method1(v42):
            v44 = v32
            v45 = v44 >= 100
            del v44
            if v45:
                v46 = " ..."
                print(v33.format(v46),end="")
                del v46
                break
            else:
                pass
            del v45
            v47 = v42 == 0
            v48 = v47 != True
            del v47
            if v48:
                v49 = "; "
                print(v33.format(v49),end="")
                del v49
            else:
                pass
            del v48
            v50 = v32 + 1
            v32 = v50
            del v50
            v51 = v34 * 4
            v52 = v51 + v42
            del v51
            v53 = v12[v52].item()
            del v52
            v54 = "{:.6f}"
            print(v54.format(v53),end="")
            del v53, v54
            v42 += 1 
        del v42
        print(v33.format(']'),end="")
        v34 += 1 
    del v12, v32, v34
    print(v33.format(']'),end="")
    v55 = "\n"
    print(v55.format(),end="")
    v57 = v6[64:64+4*16].view(cp.float32)
    v77 = 0
    print(v33.format('['),end="")
    v78 = 0
    while method1(v78):
        v80 = v77
        v81 = v80 >= 100
        del v80
        if v81:
            v82 = " ..."
            print(v33.format(v82),end="")
            del v82
            break
        else:
            pass
        del v81
        v83 = v78 == 0
        v84 = v83 != True
        del v83
        if v84:
            v85 = "; "
            print(v33.format(v85),end="")
            del v85
        else:
            pass
        del v84
        print(v33.format('['),end="")
        v86 = 0
        while method1(v86):
            v88 = v77
            v89 = v88 >= 100
            del v88
            if v89:
                v90 = " ..."
                print(v33.format(v90),end="")
                del v90
                break
            else:
                pass
            del v89
            v91 = v86 == 0
            v92 = v91 != True
            del v91
            if v92:
                v93 = "; "
                print(v33.format(v93),end="")
                del v93
            else:
                pass
            del v92
            v94 = v77 + 1
            v77 = v94
            del v94
            v95 = v78 * 4
            v96 = v95 + v86
            del v95
            v97 = v57[v96].item()
            del v96
            v98 = "{:.6f}"
            print(v98.format(v97),end="")
            del v97, v98
            v86 += 1 
        del v86
        print(v33.format(']'),end="")
        v78 += 1 
    del v57, v77, v78
    print(v33.format(']'),end="")
    print(v55.format(),end="")
    v100 = v6[128:128+4*8].view(cp.float32)
    v120 = 0
    print(v33.format('['),end="")
    v121 = 0
    while method2(v121):
        v123 = v120
        v124 = v123 >= 100
        del v123
        if v124:
            v125 = " ..."
            print(v33.format(v125),end="")
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
            print(v33.format(v128),end="")
            del v128
        else:
            pass
        del v127
        print(v33.format('['),end="")
        v129 = 0
        while method1(v129):
            v131 = v120
            v132 = v131 >= 100
            del v131
            if v132:
                v133 = " ..."
                print(v33.format(v133),end="")
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
                print(v33.format(v136),end="")
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
            v140 = v100[v139].item()
            del v139
            v141 = "{:.6f}"
            print(v141.format(v140),end="")
            del v140, v141
            v129 += 1 
        del v129
        print(v33.format(']'),end="")
        v121 += 1 
    del v100, v120, v121
    print(v33.format(']'),end="")
    print(v55.format(),end="")
    v143 = v5[0:0+4*96].view(cp.float32)
    del v5, v143
    v145 = v6[0:0+4*16].view(cp.float32)
    v146 = cp.random.normal(0.0,0.25,16,dtype=cp.float32) # type: ignore
    cp.copyto(v145[0:0+16],v146[0:0+16])
    del v145, v146
    v148 = v6[64:64+4*16].view(cp.float32)
    v149 = cp.random.normal(0.0,0.25,16,dtype=cp.float32) # type: ignore
    cp.copyto(v148[0:0+16],v149[0:0+16])
    del v148, v149
    v151 = v6[128:128+4*8].view(cp.float32)
    v152 = cp.random.normal(0.0,0.35355338,8,dtype=cp.float32) # type: ignore
    cp.copyto(v151[0:0+8],v152[0:0+8])
    del v151, v152
    v155 = "Done initing."
    print(v9.format(v155),end="")
    del v9, v155
    v157 = v6[0:0+4*16].view(cp.float32)
    v177 = 0
    print(v33.format('['),end="")
    v178 = 0
    while method1(v178):
        v180 = v177
        v181 = v180 >= 100
        del v180
        if v181:
            v182 = " ..."
            print(v33.format(v182),end="")
            del v182
            break
        else:
            pass
        del v181
        v183 = v178 == 0
        v184 = v183 != True
        del v183
        if v184:
            v185 = "; "
            print(v33.format(v185),end="")
            del v185
        else:
            pass
        del v184
        print(v33.format('['),end="")
        v186 = 0
        while method1(v186):
            v188 = v177
            v189 = v188 >= 100
            del v188
            if v189:
                v190 = " ..."
                print(v33.format(v190),end="")
                del v190
                break
            else:
                pass
            del v189
            v191 = v186 == 0
            v192 = v191 != True
            del v191
            if v192:
                v193 = "; "
                print(v33.format(v193),end="")
                del v193
            else:
                pass
            del v192
            v194 = v177 + 1
            v177 = v194
            del v194
            v195 = v178 * 4
            v196 = v195 + v186
            del v195
            v197 = v157[v196].item()
            del v196
            v198 = "{:.6f}"
            print(v198.format(v197),end="")
            del v197, v198
            v186 += 1 
        del v186
        print(v33.format(']'),end="")
        v178 += 1 
    del v157, v177, v178
    print(v33.format(']'),end="")
    print(v55.format(),end="")
    v200 = v6[64:64+4*16].view(cp.float32)
    v220 = 0
    print(v33.format('['),end="")
    v221 = 0
    while method1(v221):
        v223 = v220
        v224 = v223 >= 100
        del v223
        if v224:
            v225 = " ..."
            print(v33.format(v225),end="")
            del v225
            break
        else:
            pass
        del v224
        v226 = v221 == 0
        v227 = v226 != True
        del v226
        if v227:
            v228 = "; "
            print(v33.format(v228),end="")
            del v228
        else:
            pass
        del v227
        print(v33.format('['),end="")
        v229 = 0
        while method1(v229):
            v231 = v220
            v232 = v231 >= 100
            del v231
            if v232:
                v233 = " ..."
                print(v33.format(v233),end="")
                del v233
                break
            else:
                pass
            del v232
            v234 = v229 == 0
            v235 = v234 != True
            del v234
            if v235:
                v236 = "; "
                print(v33.format(v236),end="")
                del v236
            else:
                pass
            del v235
            v237 = v220 + 1
            v220 = v237
            del v237
            v238 = v221 * 4
            v239 = v238 + v229
            del v238
            v240 = v200[v239].item()
            del v239
            v241 = "{:.6f}"
            print(v241.format(v240),end="")
            del v240, v241
            v229 += 1 
        del v229
        print(v33.format(']'),end="")
        v221 += 1 
    del v200, v220, v221
    print(v33.format(']'),end="")
    print(v55.format(),end="")
    v243 = v6[128:128+4*8].view(cp.float32)
    del v6
    v263 = 0
    print(v33.format('['),end="")
    v264 = 0
    while method2(v264):
        v266 = v263
        v267 = v266 >= 100
        del v266
        if v267:
            v268 = " ..."
            print(v33.format(v268),end="")
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
            print(v33.format(v271),end="")
            del v271
        else:
            pass
        del v270
        print(v33.format('['),end="")
        v272 = 0
        while method1(v272):
            v274 = v263
            v275 = v274 >= 100
            del v274
            if v275:
                v276 = " ..."
                print(v33.format(v276),end="")
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
                print(v33.format(v279),end="")
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
            v283 = v243[v282].item()
            del v282
            v284 = "{:.6f}"
            print(v284.format(v283),end="")
            del v283, v284
            v272 += 1 
        del v272
        print(v33.format(']'),end="")
        v264 += 1 
    del v243, v263, v264
    print(v33.format(']'),end="")
    del v33
    print(v55.format(),end="")
    del v55
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
    v15 = cp.random.normal(0.0,0.35355338,8,dtype=cp.float32) # type: ignore
    cp.copyto(v14[0:0+8],v15[0:0+8])
    del v14, v15
    v17 = v6[32:32+4*16].view(cp.float32)
    v18 = cp.random.normal(0.0,0.25,16,dtype=cp.float32) # type: ignore
    cp.copyto(v17[0:0+16],v18[0:0+16])
    del v17, v18
    v20 = v6[96:96+4*8].view(cp.float32)
    v21 = cp.random.normal(0.0,0.35355338,8,dtype=cp.float32) # type: ignore
    cp.copyto(v20[0:0+8],v21[0:0+8])
    del v20, v21
    v24 = "Here are the weight matrices."
    print(v9.format(v24),end="")
    del v24
    v26 = v6[0:0+4*8].view(cp.float32)
    v46 = 0
    v47 = "{}"
    print(v47.format('['),end="")
    v48 = 0
    while method1(v48):
        v50 = v46
        v51 = v50 >= 100
        del v50
        if v51:
            v52 = " ..."
            print(v47.format(v52),end="")
            del v52
            break
        else:
            pass
        del v51
        v53 = v48 == 0
        v54 = v53 != True
        del v53
        if v54:
            v55 = "; "
            print(v47.format(v55),end="")
            del v55
        else:
            pass
        del v54
        print(v47.format('['),end="")
        v56 = 0
        while method2(v56):
            v58 = v46
            v59 = v58 >= 100
            del v58
            if v59:
                v60 = " ..."
                print(v47.format(v60),end="")
                del v60
                break
            else:
                pass
            del v59
            v61 = v56 == 0
            v62 = v61 != True
            del v61
            if v62:
                v63 = "; "
                print(v47.format(v63),end="")
                del v63
            else:
                pass
            del v62
            v64 = v46 + 1
            v46 = v64
            del v64
            v65 = v48 * 2
            v66 = v65 + v56
            del v65
            v67 = v26[v66].item()
            del v66
            v68 = "{:.6f}"
            print(v68.format(v67),end="")
            del v67, v68
            v56 += 1 
        del v56
        print(v47.format(']'),end="")
        v48 += 1 
    del v26, v46, v48
    print(v47.format(']'),end="")
    v69 = "\n"
    print(v69.format(),end="")
    v71 = v6[32:32+4*16].view(cp.float32)
    v91 = 0
    print(v47.format('['),end="")
    v92 = 0
    while method1(v92):
        v94 = v91
        v95 = v94 >= 100
        del v94
        if v95:
            v96 = " ..."
            print(v47.format(v96),end="")
            del v96
            break
        else:
            pass
        del v95
        v97 = v92 == 0
        v98 = v97 != True
        del v97
        if v98:
            v99 = "; "
            print(v47.format(v99),end="")
            del v99
        else:
            pass
        del v98
        print(v47.format('['),end="")
        v100 = 0
        while method1(v100):
            v102 = v91
            v103 = v102 >= 100
            del v102
            if v103:
                v104 = " ..."
                print(v47.format(v104),end="")
                del v104
                break
            else:
                pass
            del v103
            v105 = v100 == 0
            v106 = v105 != True
            del v105
            if v106:
                v107 = "; "
                print(v47.format(v107),end="")
                del v107
            else:
                pass
            del v106
            v108 = v91 + 1
            v91 = v108
            del v108
            v109 = v92 * 4
            v110 = v109 + v100
            del v109
            v111 = v71[v110].item()
            del v110
            v112 = "{:.6f}"
            print(v112.format(v111),end="")
            del v111, v112
            v100 += 1 
        del v100
        print(v47.format(']'),end="")
        v92 += 1 
    del v71, v91, v92
    print(v47.format(']'),end="")
    print(v69.format(),end="")
    v114 = v6[96:96+4*8].view(cp.float32)
    del v6
    v134 = 0
    print(v47.format('['),end="")
    v135 = 0
    while method2(v135):
        v137 = v134
        v138 = v137 >= 100
        del v137
        if v138:
            v139 = " ..."
            print(v47.format(v139),end="")
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
            print(v47.format(v142),end="")
            del v142
        else:
            pass
        del v141
        print(v47.format('['),end="")
        v143 = 0
        while method1(v143):
            v145 = v134
            v146 = v145 >= 100
            del v145
            if v146:
                v147 = " ..."
                print(v47.format(v147),end="")
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
                print(v47.format(v150),end="")
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
            v154 = v114[v153].item()
            del v153
            v155 = "{:.6f}"
            print(v155.format(v154),end="")
            del v154, v155
            v143 += 1 
        del v143
        print(v47.format(']'),end="")
        v135 += 1 
    del v114, v134, v135
    print(v47.format(']'),end="")
    print(v69.format(),end="")
    v158 = "Here is the input tensor."
    print(v9.format(v158),end="")
    del v9, v158
    v160 = v5[0:0+4*48].view(cp.float32)
    del v5
    v161 = cp.random.normal(0.0,1.0,48,dtype=cp.float32) # type: ignore
    cp.copyto(v160[0:0+48],v161[0:0+48])
    del v161
    v189 = 0
    print(v47.format('['),end="")
    v190 = 0
    while method4(v190):
        v192 = v189
        v193 = v192 >= 100
        del v192
        if v193:
            v194 = " ..."
            print(v47.format(v194),end="")
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
            print(v47.format(v197),end="")
            del v197
        else:
            pass
        del v196
        print(v47.format('['),end="")
        v198 = 0
        while method5(v198):
            v200 = v189
            v201 = v200 >= 100
            del v200
            if v201:
                v202 = " ..."
                print(v47.format(v202),end="")
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
                print(v47.format(v205),end="")
                del v205
            else:
                pass
            del v204
            print(v47.format('['),end="")
            v206 = 0
            while method2(v206):
                v208 = v189
                v209 = v208 >= 100
                del v208
                if v209:
                    v210 = " ..."
                    print(v47.format(v210),end="")
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
                    print(v47.format(v213),end="")
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
            print(v47.format(']'),end="")
            v198 += 1 
        del v198
        print(v47.format(']'),end="")
        v190 += 1 
    del v160, v189, v190
    print(v47.format(']'),end="")
    del v47
    print(v69.format(),end="")
    del v69
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
    v11 = cp.random.normal(0.0,0.015625,4096,dtype=cp.float32) # type: ignore
    cp.copyto(v10[0:0+4096],v11[0:0+4096])
    del v10, v11
    v13 = v6[16384:16384+4*4096].view(cp.float32)
    v14 = cp.random.normal(0.0,0.015625,4096,dtype=cp.float32) # type: ignore
    cp.copyto(v13[0:0+4096],v14[0:0+4096])
    del v13, v14
    v16 = v6[32768:32768+4*4096].view(cp.float32)
    v17 = cp.random.normal(0.0,0.015625,4096,dtype=cp.float32) # type: ignore
    cp.copyto(v16[0:0+4096],v17[0:0+4096])
    del v16, v17
    v20 = "{}\n"
    v21 = "Here are the weight matrices."
    print(v20.format(v21),end="")
    del v21
    v23 = v6[0:0+4*4096].view(cp.float32)
    v43 = 0
    v44 = "{}"
    print(v44.format('['),end="")
    v45 = 0
    while method7(v45):
        v47 = v43
        v48 = v47 >= 100
        del v47
        if v48:
            v49 = " ..."
            print(v44.format(v49),end="")
            del v49
            break
        else:
            pass
        del v48
        v50 = v45 == 0
        v51 = v50 != True
        del v50
        if v51:
            v52 = "; "
            print(v44.format(v52),end="")
            del v52
        else:
            pass
        del v51
        print(v44.format('['),end="")
        v53 = 0
        while method7(v53):
            v55 = v43
            v56 = v55 >= 100
            del v55
            if v56:
                v57 = " ..."
                print(v44.format(v57),end="")
                del v57
                break
            else:
                pass
            del v56
            v58 = v53 == 0
            v59 = v58 != True
            del v58
            if v59:
                v60 = "; "
                print(v44.format(v60),end="")
                del v60
            else:
                pass
            del v59
            v61 = v43 + 1
            v43 = v61
            del v61
            v62 = v45 * 64
            v63 = v62 + v53
            del v62
            v64 = v23[v63].item()
            del v63
            v65 = "{:.6f}"
            print(v65.format(v64),end="")
            del v64, v65
            v53 += 1 
        del v53
        print(v44.format(']'),end="")
        v45 += 1 
    del v23, v43, v45
    print(v44.format(']'),end="")
    v66 = "\n"
    print(v66.format(),end="")
    v68 = v6[16384:16384+4*4096].view(cp.float32)
    v88 = 0
    print(v44.format('['),end="")
    v89 = 0
    while method7(v89):
        v91 = v88
        v92 = v91 >= 100
        del v91
        if v92:
            v93 = " ..."
            print(v44.format(v93),end="")
            del v93
            break
        else:
            pass
        del v92
        v94 = v89 == 0
        v95 = v94 != True
        del v94
        if v95:
            v96 = "; "
            print(v44.format(v96),end="")
            del v96
        else:
            pass
        del v95
        print(v44.format('['),end="")
        v97 = 0
        while method7(v97):
            v99 = v88
            v100 = v99 >= 100
            del v99
            if v100:
                v101 = " ..."
                print(v44.format(v101),end="")
                del v101
                break
            else:
                pass
            del v100
            v102 = v97 == 0
            v103 = v102 != True
            del v102
            if v103:
                v104 = "; "
                print(v44.format(v104),end="")
                del v104
            else:
                pass
            del v103
            v105 = v88 + 1
            v88 = v105
            del v105
            v106 = v89 * 64
            v107 = v106 + v97
            del v106
            v108 = v68[v107].item()
            del v107
            v109 = "{:.6f}"
            print(v109.format(v108),end="")
            del v108, v109
            v97 += 1 
        del v97
        print(v44.format(']'),end="")
        v89 += 1 
    del v68, v88, v89
    print(v44.format(']'),end="")
    print(v66.format(),end="")
    v111 = v6[32768:32768+4*4096].view(cp.float32)
    v131 = 0
    print(v44.format('['),end="")
    v132 = 0
    while method7(v132):
        v134 = v131
        v135 = v134 >= 100
        del v134
        if v135:
            v136 = " ..."
            print(v44.format(v136),end="")
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
            print(v44.format(v139),end="")
            del v139
        else:
            pass
        del v138
        print(v44.format('['),end="")
        v140 = 0
        while method7(v140):
            v142 = v131
            v143 = v142 >= 100
            del v142
            if v143:
                v144 = " ..."
                print(v44.format(v144),end="")
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
                print(v44.format(v147),end="")
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
            v151 = v111[v150].item()
            del v150
            v152 = "{:.6f}"
            print(v152.format(v151),end="")
            del v151, v152
            v140 += 1 
        del v140
        print(v44.format(']'),end="")
        v132 += 1 
    del v111, v131, v132
    print(v44.format(']'),end="")
    print(v66.format(),end="")
    v154 = v5[0:0+4*98304].view(cp.float32)
    v155 = cp.random.normal(0.0,1.0,98304,dtype=cp.float32) # type: ignore
    cp.copyto(v154[0:0+98304],v155[0:0+98304])
    del v155
    v183 = 0
    print(v44.format('['),end="")
    v184 = 0
    while method4(v184):
        v186 = v183
        v187 = v186 >= 100
        del v186
        if v187:
            v188 = " ..."
            print(v44.format(v188),end="")
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
            print(v44.format(v191),end="")
            del v191
        else:
            pass
        del v190
        print(v44.format('['),end="")
        v192 = 0
        while method7(v192):
            v194 = v183
            v195 = v194 >= 100
            del v194
            if v195:
                v196 = " ..."
                print(v44.format(v196),end="")
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
                print(v44.format(v199),end="")
                del v199
            else:
                pass
            del v198
            print(v44.format('['),end="")
            v200 = 0
            while method7(v200):
                v202 = v183
                v203 = v202 >= 100
                del v202
                if v203:
                    v204 = " ..."
                    print(v44.format(v204),end="")
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
                    print(v44.format(v207),end="")
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
            print(v44.format(']'),end="")
            v192 += 1 
        del v192
        print(v44.format(']'),end="")
        v184 += 1 
    del v154, v183, v184
    print(v44.format(']'),end="")
    print(v66.format(),end="")
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
    print(v44.format('['),end="")
    v273 = 0
    while method4(v273):
        v275 = v272
        v276 = v275 >= 100
        del v275
        if v276:
            v277 = " ..."
            print(v44.format(v277),end="")
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
            print(v44.format(v280),end="")
            del v280
        else:
            pass
        del v279
        print(v44.format('['),end="")
        v281 = 0
        while method7(v281):
            v283 = v272
            v284 = v283 >= 100
            del v283
            if v284:
                v285 = " ..."
                print(v44.format(v285),end="")
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
                print(v44.format(v288),end="")
                del v288
            else:
                pass
            del v287
            print(v44.format('['),end="")
            v289 = 0
            while method7(v289):
                v291 = v272
                v292 = v291 >= 100
                del v291
                if v292:
                    v293 = " ..."
                    print(v44.format(v293),end="")
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
                    print(v44.format(v296),end="")
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
            print(v44.format(']'),end="")
            v281 += 1 
        del v281
        print(v44.format(']'),end="")
        v273 += 1 
    del v225, v272, v273
    print(v44.format(']'),end="")
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
            print(v44.format(v310),end="")
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
            print(v44.format(v313),end="")
            del v313
        else:
            pass
        del v312
        print(v44.format('['),end="")
        v314 = 0
        while method7(v314):
            v316 = v304
            v317 = v316 >= 100
            del v316
            if v317:
                v318 = " ..."
                print(v44.format(v318),end="")
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
                print(v44.format(v321),end="")
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
            print(v44.format(v325),end="")
            del v325
            v314 += 1 
        del v314
        print(v44.format(']'),end="")
        v306 += 1 
    del v227, v304, v306
    print(v44.format(']'),end="")
    del v44
    print(v66.format(),end="")
    del v66
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
    v11 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v10[0:0+65536],v11[0:0+65536])
    del v10, v11
    v13 = v6[262144:262144+4*65536].view(cp.float32)
    v14 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v13[0:0+65536],v14[0:0+65536])
    del v13, v14
    v16 = v6[524288:524288+4*65536].view(cp.float32)
    v17 = cp.random.normal(0.0,0.00390625,65536,dtype=cp.float32) # type: ignore
    cp.copyto(v16[0:0+65536],v17[0:0+65536])
    del v16, v17
    v19 = v5[3145728:3145728+4*1572864].view(cp.float32)
    del v19
    v21 = v5[9437184:9437184+4*24576].view(cp.int32)
    del v21
    v24 = "{}\n"
    v25 = "Here are the weight matrices."
    print(v24.format(v25),end="")
    del v25
    v27 = v6[0:0+4*65536].view(cp.float32)
    v55 = 0
    v56 = "{}"
    print(v56.format('['),end="")
    v57 = 0
    while method9(v57):
        v59 = v55
        v60 = v59 >= 100
        del v59
        if v60:
            v61 = " ..."
            print(v56.format(v61),end="")
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
            print(v56.format(v64),end="")
            del v64
        else:
            pass
        del v63
        print(v56.format('['),end="")
        v65 = 0
        while method7(v65):
            v67 = v55
            v68 = v67 >= 100
            del v67
            if v68:
                v69 = " ..."
                print(v56.format(v69),end="")
                del v69
                break
            else:
                pass
            del v68
            v70 = v65 == 0
            v71 = v70 != True
            del v70
            if v71:
                v72 = "; "
                print(v56.format(v72),end="")
                del v72
            else:
                pass
            del v71
            print(v56.format('['),end="")
            v73 = 0
            while method7(v73):
                v75 = v55
                v76 = v75 >= 100
                del v75
                if v76:
                    v77 = " ..."
                    print(v56.format(v77),end="")
                    del v77
                    break
                else:
                    pass
                del v76
                v78 = v73 == 0
                v79 = v78 != True
                del v78
                if v79:
                    v80 = "; "
                    print(v56.format(v80),end="")
                    del v80
                else:
                    pass
                del v79
                v81 = v55 + 1
                v55 = v81
                del v81
                v82 = v57 * 4096
                v83 = v65 * 64
                v84 = v82 + v83
                del v82, v83
                v85 = v84 + v73
                del v84
                v86 = v27[v85].item()
                del v85
                v87 = "{:.6f}"
                print(v87.format(v86),end="")
                del v86, v87
                v73 += 1 
            del v73
            print(v56.format(']'),end="")
            v65 += 1 
        del v65
        print(v56.format(']'),end="")
        v57 += 1 
    del v27, v55, v57
    print(v56.format(']'),end="")
    v88 = "\n"
    print(v88.format(),end="")
    v90 = v6[262144:262144+4*65536].view(cp.float32)
    v118 = 0
    print(v56.format('['),end="")
    v119 = 0
    while method9(v119):
        v121 = v118
        v122 = v121 >= 100
        del v121
        if v122:
            v123 = " ..."
            print(v56.format(v123),end="")
            del v123
            break
        else:
            pass
        del v122
        v124 = v119 == 0
        v125 = v124 != True
        del v124
        if v125:
            v126 = "; "
            print(v56.format(v126),end="")
            del v126
        else:
            pass
        del v125
        print(v56.format('['),end="")
        v127 = 0
        while method7(v127):
            v129 = v118
            v130 = v129 >= 100
            del v129
            if v130:
                v131 = " ..."
                print(v56.format(v131),end="")
                del v131
                break
            else:
                pass
            del v130
            v132 = v127 == 0
            v133 = v132 != True
            del v132
            if v133:
                v134 = "; "
                print(v56.format(v134),end="")
                del v134
            else:
                pass
            del v133
            print(v56.format('['),end="")
            v135 = 0
            while method7(v135):
                v137 = v118
                v138 = v137 >= 100
                del v137
                if v138:
                    v139 = " ..."
                    print(v56.format(v139),end="")
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
                    print(v56.format(v142),end="")
                    del v142
                else:
                    pass
                del v141
                v143 = v118 + 1
                v118 = v143
                del v143
                v144 = v119 * 4096
                v145 = v127 * 64
                v146 = v144 + v145
                del v144, v145
                v147 = v146 + v135
                del v146
                v148 = v90[v147].item()
                del v147
                v149 = "{:.6f}"
                print(v149.format(v148),end="")
                del v148, v149
                v135 += 1 
            del v135
            print(v56.format(']'),end="")
            v127 += 1 
        del v127
        print(v56.format(']'),end="")
        v119 += 1 
    del v90, v118, v119
    print(v56.format(']'),end="")
    print(v88.format(),end="")
    v151 = v6[524288:524288+4*65536].view(cp.float32)
    v179 = 0
    print(v56.format('['),end="")
    v180 = 0
    while method9(v180):
        v182 = v179
        v183 = v182 >= 100
        del v182
        if v183:
            v184 = " ..."
            print(v56.format(v184),end="")
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
            print(v56.format(v187),end="")
            del v187
        else:
            pass
        del v186
        print(v56.format('['),end="")
        v188 = 0
        while method7(v188):
            v190 = v179
            v191 = v190 >= 100
            del v190
            if v191:
                v192 = " ..."
                print(v56.format(v192),end="")
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
                print(v56.format(v195),end="")
                del v195
            else:
                pass
            del v194
            print(v56.format('['),end="")
            v196 = 0
            while method7(v196):
                v198 = v179
                v199 = v198 >= 100
                del v198
                if v199:
                    v200 = " ..."
                    print(v56.format(v200),end="")
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
                    print(v56.format(v203),end="")
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
                v209 = v151[v208].item()
                del v208
                v210 = "{:.6f}"
                print(v210.format(v209),end="")
                del v209, v210
                v196 += 1 
            del v196
            print(v56.format(']'),end="")
            v188 += 1 
        del v188
        print(v56.format(']'),end="")
        v180 += 1 
    del v151, v179, v180
    print(v56.format(']'),end="")
    print(v88.format(),end="")
    v212 = v5[0:0+4*98304].view(cp.float32)
    v213 = cp.random.normal(0.0,1.0,98304,dtype=cp.float32) # type: ignore
    cp.copyto(v212[0:0+98304],v213[0:0+98304])
    del v213
    v241 = 0
    print(v56.format('['),end="")
    v242 = 0
    while method4(v242):
        v244 = v241
        v245 = v244 >= 100
        del v244
        if v245:
            v246 = " ..."
            print(v56.format(v246),end="")
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
            print(v56.format(v249),end="")
            del v249
        else:
            pass
        del v248
        print(v56.format('['),end="")
        v250 = 0
        while method7(v250):
            v252 = v241
            v253 = v252 >= 100
            del v252
            if v253:
                v254 = " ..."
                print(v56.format(v254),end="")
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
                print(v56.format(v257),end="")
                del v257
            else:
                pass
            del v256
            print(v56.format('['),end="")
            v258 = 0
            while method7(v258):
                v260 = v241
                v261 = v260 >= 100
                del v260
                if v261:
                    v262 = " ..."
                    print(v56.format(v262),end="")
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
                    print(v56.format(v265),end="")
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
            print(v56.format(']'),end="")
            v250 += 1 
        del v250
        print(v56.format(']'),end="")
        v242 += 1 
    del v212, v241, v242
    print(v56.format(']'),end="")
    print(v88.format(),end="")
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
    print(v56.format('['),end="")
    v318 = 0
    while method9(v318):
        v320 = v317
        v321 = v320 >= 2147483647
        del v320
        if v321:
            v322 = " ..."
            print(v56.format(v322),end="")
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
            print(v56.format(v325),end="")
            del v325
        else:
            pass
        del v324
        print(v56.format('['),end="")
        v326 = 0
        while method4(v326):
            v328 = v317
            v329 = v328 >= 2147483647
            del v328
            if v329:
                v330 = " ..."
                print(v56.format(v330),end="")
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
                print(v56.format(v333),end="")
                del v333
            else:
                pass
            del v332
            print(v56.format('['),end="")
            v334 = 0
            while method7(v334):
                v336 = v317
                v337 = v336 >= 2147483647
                del v336
                if v337:
                    v338 = " ..."
                    print(v56.format(v338),end="")
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
                    print(v56.format(v341),end="")
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
                print(v56.format(v347),end="")
                del v347
                v334 += 1 
            del v334
            print(v56.format(']'),end="")
            v326 += 1 
        del v326
        print(v56.format(']'),end="")
        v318 += 1 
    del v283, v317, v318
    print(v56.format(']'),end="")
    del v56
    print(v88.format(),end="")
    del v88
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
    v11 = cp.random.normal(0.0,0.0078125,16384,dtype=cp.float32) # type: ignore
    cp.copyto(v10[0:0+16384],v11[0:0+16384])
    del v10, v11
    v13 = v5[786432:786432+4*393216].view(cp.float32)
    del v13
    v15 = v5[2359296:2359296+4*6144].view(cp.int32)
    del v15
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
