kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using default_int = long;
using default_uint = unsigned long;
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
struct Tuple0 {
    long v0;
    float v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(long t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure0 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
__device__ inline bool while_method_0(long v0){
    bool v1;
    v1 = v0 < 512l;
    return v1;
}
__device__ inline bool while_method_1(long v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_2(long v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ inline bool while_method_3(long v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1, long * v2, long * v3) {
    auto v4 = cooperative_groups::this_thread_block();
    float v5;
    v5 = 0.0f;
    long v6;
    v6 = threadIdx.x;
    long v7;
    v7 = v6;
    while (while_method_0(v7)){
        bool v9;
        v9 = 0l <= v7;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The index needs to be zero or positive." && v9);
        } else {
        }
        long v11;
        v11 = v7 % 16l;
        long v12;
        v12 = v7 / 16l;
        bool v13;
        v13 = v12 < 32l;
        bool v14;
        v14 = v13 == false;
        if (v14){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v13);
        } else {
        }
        assert("Tensor range check" && 0 <= v12 && v12 < 32l);
        assert("Tensor range check" && 0 <= v11 && v11 < 16l);
        long v15;
        v15 = 4l * v11;
        long v16;
        v16 = 64l * v12;
        long v17;
        v17 = v16 + v15;
        float v18[4l];
        int4* v19;
        v19 = reinterpret_cast<int4*>(v0 + v17);
        int4* v20;
        v20 = reinterpret_cast<int4*>(v18 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v19) % 4l == 0 && (unsigned long long)(v20) % 4l == 0);
        *v20 = *v19;
        long v21; float v22;
        Tuple0 tmp0 = Tuple0{0l, v5};
        v21 = tmp0.v0; v22 = tmp0.v1;
        while (while_method_1(v21)){
            assert("Tensor range check" && 0 <= v21 && v21 < 4l);
            float v24;
            v24 = v18[v21];
            float v25;
            v25 = v24 + v22;
            v22 = v25;
            v21 += 1l ;
        }
        v5 = v22;
        v7 += 32l ;
    }
    auto v26 = cooperative_groups::coalesced_threads();
    Closure0 v27{};
    float v28;
    v28 = cooperative_groups::reduce(v26, v5, v27);
    long v29;
    v29 = threadIdx.x;
    long v30;
    v30 = v29 / 32l;
    __shared__ float v31[1l];
    assert("Tensor range check" && 0 <= v30 && v30 < 1l);
    v31[v30] = v28;
    __syncthreads();
    long v32;
    v32 = threadIdx.x;
    long v33;
    v33 = v32 % 32l;
    bool v34;
    v34 = v30 == 0l;
    bool v36;
    if (v34){
        bool v35;
        v35 = v33 < 1l;
        v36 = v35;
    } else {
        v36 = false;
    }
    if (v36){
        auto v37 = cooperative_groups::coalesced_threads();
        assert("Tensor range check" && 0 <= v33 && v33 < 1l);
        float v38;
        v38 = v31[v33];
        float v39;
        v39 = cooperative_groups::reduce(v37, v38, v27);
        v1[0l] = v39;
    } else {
    }
    __syncthreads();
    long v40;
    v40 = threadIdx.x;
    bool v41;
    v41 = 0l <= v40;
    bool v42;
    v42 = v41 == false;
    if (v42){
        assert("The index needs to be zero or positive." && v41);
    } else {
    }
    long v43;
    v43 = v40 % 16l;
    long v44;
    v44 = v40 / 16l;
    bool v45;
    v45 = v44 < 2l;
    bool v46;
    v46 = v45 == false;
    if (v46){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v45);
    } else {
    }
    cooperative_groups::thread_block_tile<16l, cooperative_groups::thread_block> v47 = cooperative_groups::tiled_partition<16l>(v4);
    assert("Tensor range check" && 0 <= v44 && v44 < 2l);
    assert("Tensor range check" && 0 <= v43 && v43 < 16l);
    long v48;
    v48 = 4l * v43;
    long v49;
    v49 = 64l * v44;
    long v50;
    v50 = v49 + v48;
    assert("Tensor range check" && 0 <= v44 && v44 < 2l);
    assert("Tensor range check" && 0 <= v43 && v43 < 16l);
    long v51;
    v51 = 0l;
    while (while_method_2(v51)){
        assert("Tensor range check" && 0 <= v51 && v51 < 16l);
        long v53;
        v53 = 128l * v51;
        long v54;
        v54 = v53 + v50;
        assert("Tensor range check" && 0 <= v51 && v51 < 16l);
        float v55[4l];
        long v56[4l];
        long v57;
        v57 = 0l;
        while (while_method_3(v57)){
            assert("Tensor range check" && 0 <= v57 && v57 < 1l);
            long v59;
            v59 = 4l * v57;
            assert("Tensor range check" && 0 <= v57 && v57 < 1l);
            long v60;
            v60 = 64l * v57;
            long v61;
            v61 = v60 + v54;
            int4* v62;
            v62 = reinterpret_cast<int4*>(v0 + v61);
            int4* v63;
            v63 = reinterpret_cast<int4*>(v55 + v59);
            assert("Pointer alignment check" && (unsigned long long)(v62) % 4l == 0 && (unsigned long long)(v63) % 4l == 0);
            *v63 = *v62;
            v57 += 1l ;
        }
        long v64;
        v64 = 0l;
        while (while_method_3(v64)){
            long v66;
            v66 = 0l;
            while (while_method_1(v66)){
                bool v68;
                v68 = 0l <= v66;
                bool v70;
                if (v68){
                    bool v69;
                    v69 = v66 < 4l;
                    v70 = v69;
                } else {
                    v70 = false;
                }
                bool v71;
                v71 = v70 == false;
                if (v71){
                    assert("The indices should be inside the range of the dimension." && v70);
                } else {
                }
                bool v72;
                v72 = 0l <= v43;
                bool v74;
                if (v72){
                    bool v73;
                    v73 = v43 < 16l;
                    v74 = v73;
                } else {
                    v74 = false;
                }
                bool v75;
                v75 = v74 == false;
                if (v75){
                    assert("The indices should be inside the range of the dimension." && v74);
                } else {
                }
                long v76;
                v76 = v43 * 4l;
                long v77;
                v77 = v66 + v76;
                bool v78;
                v78 = 0l <= v64;
                bool v80;
                if (v78){
                    bool v79;
                    v79 = v64 < 1l;
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
                long v82;
                v82 = v64 * 64l;
                long v83;
                v83 = v77 + v82;
                assert("Tensor range check" && 0 <= v64 && v64 < 1l);
                assert("Tensor range check" && 0 <= v66 && v66 < 4l);
                long v84;
                v84 = 4l * v64;
                long v85;
                v85 = v84 + v66;
                v56[v85] = v83;
                v66 += 1l ;
            }
            v64 += 1l ;
        }
        bool v86;
        v86 = 0l <= v44;
        bool v87;
        v87 = v86 && v45;
        bool v88;
        v88 = v87 == false;
        if (v88){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v87);
        } else {
        }
        bool v89;
        v89 = 0l <= v51;
        bool v91;
        if (v89){
            bool v90;
            v90 = v51 < 16l;
            v91 = v90;
        } else {
            v91 = false;
        }
        bool v92;
        v92 = v91 == false;
        if (v92){
            assert("The rigid merge indices have to be non-zero and less than the dimensions." && v91);
        } else {
        }
        long v93;
        v93 = v51 * 2l;
        long v94;
        v94 = v93 + v44;
        long v95[4l];
        long v96[4l];
        long v97;
        v97 = 0l;
        while (while_method_3(v97)){
            long v99;
            v99 = 0l;
            while (while_method_1(v99)){
                assert("Tensor range check" && 0 <= v97 && v97 < 1l);
                assert("Tensor range check" && 0 <= v99 && v99 < 4l);
                long v101;
                v101 = 4l * v97;
                long v102;
                v102 = v101 + v99;
                long v103;
                v103 = v56[v102];
                assert("Tensor range check" && 0 <= v97 && v97 < 1l);
                assert("Tensor range check" && 0 <= v99 && v99 < 4l);
                v95[v102] = v94;
                v96[v102] = v103;
                v99 += 1l ;
            }
            v97 += 1l ;
        }
        long v104;
        v104 = 0l;
        while (while_method_3(v104)){
            assert("Tensor range check" && 0 <= v104 && v104 < 1l);
            long v106;
            v106 = 64l * v104;
            long v107;
            v107 = v106 + v54;
            assert("Tensor range check" && 0 <= v104 && v104 < 1l);
            long v108;
            v108 = 4l * v104;
            int4* v109;
            v109 = reinterpret_cast<int4*>(v95 + v108);
            int4* v110;
            v110 = reinterpret_cast<int4*>(v2 + v107);
            assert("Pointer alignment check" && (unsigned long long)(v109) % 4l == 0 && (unsigned long long)(v110) % 4l == 0);
            *v110 = *v109;
            int4* v111;
            v111 = reinterpret_cast<int4*>(v96 + v108);
            int4* v112;
            v112 = reinterpret_cast<int4*>(v3 + v107);
            assert("Pointer alignment check" && (unsigned long long)(v111) % 4l == 0 && (unsigned long long)(v112) % 4l == 0);
            *v112 = *v111;
            v104 += 1l ;
        }
        v51 += 1l ;
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
    v1 = v0 < 32
    del v0
    return v1
def method2(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method3(v0 : i32) -> bool:
    v1 = v0 < 64
    del v0
    return v1
def method5(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def method4(v0 : i32, v1 : i32) -> None:
    method5(v0)
    del v0
    v2 = ", "
    method2(v2)
    del v2
    return method5(v1)
def method6() -> None:
    return 
def main():
    v0 = cp.arange(0,2048,1,dtype=cp.float32) # type: ignore
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
    v6 = cp.random.uniform(size=32,dtype=cp.float32) # type: ignore
    del v6
    v7 = cp.empty(1,dtype=cp.float32)
    v8 = cp.empty(2048,dtype=cp.int32)
    v9 = cp.empty(2048,dtype=cp.int32)
    v10 = 0
    v11 = raw_module.get_function(f"entry{v10}")
    del v10
    v11.max_dynamic_shared_size_bytes = 0 
    v11((1,),(32,),(v5, v7, v8, v9),shared_mem=0)
    del v5, v7, v11
    v12 = 0
    v13 = '['
    method0(v13)
    del v13
    v14 = 0
    while method1(v14):
        v16 = v12
        v17 = v16 >= 2048
        del v16
        if v17:
            v18 = " ..."
            method2(v18)
            del v18
            break
        else:
            pass
        del v17
        v19 = v14 == 0
        v20 = v19 != True
        del v19
        if v20:
            v21 = "; "
            method2(v21)
        else:
            pass
        del v20
        v22 = '['
        method0(v22)
        del v22
        v23 = 0
        while method3(v23):
            v25 = v12
            v26 = v25 >= 2048
            del v25
            if v26:
                v27 = " ..."
                method2(v27)
                del v27
                break
            else:
                pass
            del v26
            v28 = v23 == 0
            v29 = v28 != True
            del v28
            if v29:
                v30 = "; "
                method2(v30)
            else:
                pass
            del v29
            v31 = v12 + 1
            v12 = v31
            del v31
            v32 = v14 * 64
            v33 = v32 + v23
            del v32
            v34 = v8[v33].item()
            v35 = v9[v33].item()
            del v33
            method4(v34, v35)
            del v34, v35
            v23 += 1 
        del v23
        v36 = ']'
        method0(v36)
        del v36
        v14 += 1 
    del v8, v9, v12, v14
    v37 = ']'
    method0(v37)
    del v37
    method6()
    print()
    return 

if __name__ == '__main__': print(main())
