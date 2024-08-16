kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <cooperative_groups/reduce.h>
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
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 2l;
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
extern "C" __global__ void entry0(float * v0, float * v1) {
    unsigned long long v2;
    v2 = clock64();
    int v3;
    v3 = threadIdx.x;
    unsigned long long v4;
    v4 = (unsigned long long)v3;
    curandStatePhilox4_32_10_t v5;
    curand_init(v2,v4,0ull,&v5);
    int v6;
    v6 = threadIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 32l);
    int v7;
    v7 = 8l * v6;
    int v8;
    v8 = threadIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 32l);
    int v9;
    v9 = 8l * v8;
    __shared__ float * v10[32l];
    __shared__ float * v11[32l];
    int v12;
    v12 = threadIdx.x;
    float * v13;
    v13 = v0+v7;
    assert("Tensor range check" && 0 <= v12 && v12 < 32l);
    v10[v12] = v13;
    int v15;
    v15 = threadIdx.x;
    float * v16;
    v16 = v1+v9;
    assert("Tensor range check" && 0 <= v15 && v15 < 32l);
    v11[v15] = v16;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v18;
    v18 = threadIdx.x;
    bool v19;
    v19 = 0l <= v18;
    bool v20;
    v20 = v19 == false;
    if (v20){
        assert("The index needs to be zero or positive." && v19);
    } else {
    }
    int v22;
    v22 = v18 % 2l;
    int v23;
    v23 = v18 / 2l;
    bool v24;
    v24 = v23 < 16l;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
    } else {
    }
    assert("Tensor range check" && 0 <= v23 && v23 < 16l);
    int v27;
    v27 = 2l * v23;
    assert("Tensor range check" && 0 <= v23 && v23 < 16l);
    int v28;
    v28 = 0l;
    while (while_method_0(v28)){
        assert("Tensor range check" && 0 <= v28 && v28 < 2l);
        int v30;
        v30 = v28 + v27;
        float * v31;
        v31 = v10[v30];
        assert("Tensor range check" && 0 <= v28 && v28 < 2l);
        float * v32;
        v32 = v11[v30];
        assert("Tensor range check" && 0 <= v22 && v22 < 2l);
        int v33;
        v33 = 4l * v22;
        float v34[4l];
        int v35[4l];
        int v36;
        v36 = 0l;
        while (while_method_1(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 1l);
            int v38;
            v38 = 4l * v36;
            assert("Tensor range check" && 0 <= v36 && v36 < 1l);
            int v39;
            v39 = 8l * v36;
            int v40;
            v40 = v39 + v33;
            int4* v41;
            v41 = reinterpret_cast<int4*>(v31 + v40);
            int4* v42;
            v42 = reinterpret_cast<int4*>(v34 + v38);
            assert("Pointer alignment check" && (unsigned long long)(v41) % 4l == 0 && (unsigned long long)(v42) % 4l == 0);
            *v42 = *v41;
            v36 += 1l ;
        }
        int v43;
        v43 = 0l;
        while (while_method_1(v43)){
            int v45;
            v45 = 0l;
            while (while_method_2(v45)){
                bool v47;
                v47 = 0l <= v45;
                bool v49;
                if (v47){
                    bool v48;
                    v48 = v45 < 4l;
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
                v52 = 0l <= v22;
                bool v54;
                if (v52){
                    bool v53;
                    v53 = v22 < 2l;
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
                v57 = v22 * 4l;
                int v58;
                v58 = v45 + v57;
                bool v59;
                v59 = 0l <= v43;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v43 < 1l;
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
                v64 = v43 * 8l;
                int v65;
                v65 = v58 + v64;
                assert("Tensor range check" && 0 <= v43 && v43 < 1l);
                assert("Tensor range check" && 0 <= v45 && v45 < 4l);
                int v66;
                v66 = 4l * v43;
                int v67;
                v67 = v66 + v45;
                v35[v67] = v65;
                v45 += 1l ;
            }
            v43 += 1l ;
        }
        bool v68;
        v68 = 0l <= v28;
        bool v70;
        if (v68){
            bool v69;
            v69 = v28 < 2l;
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
        bool v73;
        v73 = 0l <= v23;
        bool v74;
        v74 = v73 && v24;
        bool v75;
        v75 = v74 == false;
        if (v75){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v74);
        } else {
        }
        int v77;
        v77 = v23 * 2l;
        int v78;
        v78 = v77 + v28;
        bool v79[4l];
        int v80;
        v80 = 0l;
        while (while_method_1(v80)){
            int v82;
            v82 = 0l;
            while (while_method_2(v82)){
                assert("Tensor range check" && 0 <= v80 && v80 < 1l);
                assert("Tensor range check" && 0 <= v82 && v82 < 4l);
                int v84;
                v84 = 4l * v80;
                int v85;
                v85 = v84 + v82;
                float v86;
                v86 = v34[v85];
                int v87;
                v87 = v35[v85];
                bool v88;
                v88 = v87 < 3l;
                assert("Tensor range check" && 0 <= v80 && v80 < 1l);
                assert("Tensor range check" && 0 <= v82 && v82 < 4l);
                v79[v85] = v88;
                v82 += 1l ;
            }
            v80 += 1l ;
        }
        float v89[4l];
        int v90;
        v90 = 0l;
        while (while_method_1(v90)){
            int v92;
            v92 = 0l;
            while (while_method_2(v92)){
                assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                assert("Tensor range check" && 0 <= v92 && v92 < 4l);
                int v94;
                v94 = 4l * v90;
                int v95;
                v95 = v94 + v92;
                float v96;
                v96 = v34[v95];
                bool v97;
                v97 = v79[v95];
                float v100;
                if (v97){
                    bool v98;
                    v98 = 0.0f >= v96;
                    if (v98){
                        v100 = 0.0f;
                    } else {
                        v100 = v96;
                    }
                } else {
                    v100 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v90 && v90 < 1l);
                assert("Tensor range check" && 0 <= v92 && v92 < 4l);
                v89[v95] = v100;
                v92 += 1l ;
            }
            v90 += 1l ;
        }
        float v101;
        v101 = 0.0f;
        int v102;
        v102 = 0l;
        while (while_method_1(v102)){
            int v104;
            v104 = 0l;
            while (while_method_2(v104)){
                assert("Tensor range check" && 0 <= v102 && v102 < 1l);
                assert("Tensor range check" && 0 <= v104 && v104 < 4l);
                int v106;
                v106 = 4l * v102;
                int v107;
                v107 = v106 + v104;
                float v108;
                v108 = v89[v107];
                float v109;
                v109 = v101 + v108;
                v101 = v109;
                v104 += 1l ;
            }
            v102 += 1l ;
        }
        auto v110 = cooperative_groups::coalesced_threads();
        int v111;
        v111 = threadIdx.x;
        int v112;
        v112 = v111 / 2l;
        auto v113 = cooperative_groups::labeled_partition(v110,v112);
        Closure0 v114{};
        float v115;
        v115 = cooperative_groups::reduce(v113, v101, v114);
        int v116[4l];
        int v117;
        v117 = 0l;
        while (while_method_1(v117)){
            int v119;
            v119 = 0l;
            while (while_method_2(v119)){
                assert("Tensor range check" && 0 <= v117 && v117 < 1l);
                assert("Tensor range check" && 0 <= v119 && v119 < 4l);
                int v121;
                v121 = 4l * v117;
                int v122;
                v122 = v121 + v119;
                bool v123;
                v123 = v79[v122];
                int v124;
                if (v123){
                    v124 = 1l;
                } else {
                    v124 = 0l;
                }
                assert("Tensor range check" && 0 <= v117 && v117 < 1l);
                assert("Tensor range check" && 0 <= v119 && v119 < 4l);
                v116[v122] = v124;
                v119 += 1l ;
            }
            v117 += 1l ;
        }
        int v125;
        v125 = 0l;
        int v126;
        v126 = 0l;
        while (while_method_1(v126)){
            int v128;
            v128 = 0l;
            while (while_method_2(v128)){
                assert("Tensor range check" && 0 <= v126 && v126 < 1l);
                assert("Tensor range check" && 0 <= v128 && v128 < 4l);
                int v130;
                v130 = 4l * v126;
                int v131;
                v131 = v130 + v128;
                int v132;
                v132 = v116[v131];
                int v133;
                v133 = v125 + v132;
                v125 = v133;
                v128 += 1l ;
            }
            v126 += 1l ;
        }
        auto v134 = cooperative_groups::coalesced_threads();
        int v135;
        v135 = threadIdx.x;
        int v136;
        v136 = v135 / 2l;
        auto v137 = cooperative_groups::labeled_partition(v134,v136);
        Closure1 v138{};
        int v139;
        v139 = cooperative_groups::reduce(v137, v125, v138);
        float v140;
        v140 = (float)v139;
        float v141;
        v141 = 1.0f / v140;
        float v142[4l];
        int v143;
        v143 = 0l;
        while (while_method_1(v143)){
            int v145;
            v145 = 0l;
            while (while_method_2(v145)){
                assert("Tensor range check" && 0 <= v143 && v143 < 1l);
                assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                int v147;
                v147 = 4l * v143;
                int v148;
                v148 = v147 + v145;
                float v149;
                v149 = v89[v148];
                bool v150;
                v150 = v79[v148];
                bool v151;
                v151 = v150 == false;
                float v156;
                if (v151){
                    v156 = 0.0f;
                } else {
                    bool v152;
                    v152 = v115 == 0.0f;
                    bool v153;
                    v153 = v152 != true;
                    if (v153){
                        float v154;
                        v154 = v149 / v115;
                        v156 = v154;
                    } else {
                        v156 = v141;
                    }
                }
                assert("Tensor range check" && 0 <= v143 && v143 < 1l);
                assert("Tensor range check" && 0 <= v145 && v145 < 4l);
                v142[v148] = v156;
                v145 += 1l ;
            }
            v143 += 1l ;
        }
        assert("Tensor range check" && 0 <= v22 && v22 < 2l);
        int v157;
        v157 = 0l;
        while (while_method_1(v157)){
            assert("Tensor range check" && 0 <= v157 && v157 < 1l);
            int v159;
            v159 = 8l * v157;
            int v160;
            v160 = v159 + v33;
            assert("Tensor range check" && 0 <= v157 && v157 < 1l);
            int v161;
            v161 = 4l * v157;
            int4* v162;
            v162 = reinterpret_cast<int4*>(v142 + v161);
            int4* v163;
            v163 = reinterpret_cast<int4*>(v32 + v160);
            assert("Pointer alignment check" && (unsigned long long)(v162) % 4l == 0 && (unsigned long long)(v163) % 4l == 0);
            *v163 = *v162;
            v157 += 1l ;
        }
        v28 += 1l ;
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
    v1 = v0 < 32
    del v0
    return v1
def method1(v0 : i32) -> bool:
    v1 = v0 < 8
    del v0
    return v1
def main():
    v0 = cp.arange(0,256,1,dtype=cp.float32) # type: ignore
    v1 = v0.size
    del v0
    v2 = 256 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.random.normal(0.0,1.0,256,dtype=cp.float32) # type: ignore
    v6 = cp.empty(256,dtype=cp.float32)
    v7 = 0
    v8 = raw_module.get_function(f"entry{v7}")
    del v7
    v8.max_dynamic_shared_size_bytes = 0 
    v8((1,),(32,),(v5, v6),shared_mem=0)
    del v5, v8
    v35 = 0
    v36 = "{}"
    print(v36.format('['),end="")
    v37 = 0
    while method0(v37):
        v39 = v35
        v40 = v39 >= 2147483647
        del v39
        if v40:
            v41 = " ..."
            print(v36.format(v41),end="")
            del v41
            break
        else:
            pass
        del v40
        v42 = v37 == 0
        v43 = v42 != True
        del v42
        if v43:
            v44 = "; "
            print(v36.format(v44),end="")
            del v44
        else:
            pass
        del v43
        print(v36.format('['),end="")
        v45 = 0
        while method1(v45):
            v47 = v35
            v48 = v47 >= 2147483647
            del v47
            if v48:
                v49 = " ..."
                print(v36.format(v49),end="")
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
                print(v36.format(v52),end="")
                del v52
            else:
                pass
            del v51
            v53 = v35 + 1
            v35 = v53
            del v53
            v54 = v37 * 8
            v55 = v54 + v45
            del v54
            v56 = v6[v55].item()
            del v55
            v57 = "{:.6f}"
            print(v57.format(v56),end="")
            del v56, v57
            v45 += 1 
        del v45
        print(v36.format(']'),end="")
        v37 += 1 
    del v6, v35, v37
    print(v36.format(']'),end="")
    del v36
    v58 = "\n"
    print(v58,end="")
    del v58
    return 

if __name__ == '__main__': print(main())
