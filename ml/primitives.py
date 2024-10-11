kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
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

struct Tuple0;
struct Tuple0 {
    unsigned long long v1;
    int v0;
    __device__ Tuple0() = default;
    __device__ Tuple0(int t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Closure0 {
    __device__ unsigned long long operator()(unsigned long long tup0, unsigned long long tup1){
        unsigned long long v0 = tup0; unsigned long long v1 = tup1;
        unsigned long long v2;
        v2 = v0 + v1;
        return v2;
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 8192;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 24;
    return v1;
}
extern "C" __global__ void entry0(unsigned long long * v0, unsigned long long * v1, unsigned long long * v2) {
    auto v3 = cooperative_groups::this_grid();
    unsigned long long v4;
    v4 = 0ull;
    int v5;
    v5 = threadIdx.x;
    int v6;
    v6 = v5;
    while (while_method_0(v6)){
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
        Tuple0 tmp0 = Tuple0{0, v4};
        v22 = tmp0.v0; v23 = tmp0.v1;
        while (while_method_1(v22)){
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
    auto v27 = cooperative_groups::coalesced_threads();
    Closure0 v28{};
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
    while (while_method_0(v47)){
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
        Tuple0 tmp1 = Tuple0{0, v42};
        v63 = tmp1.v0; v64 = tmp1.v1;
        while (while_method_1(v63)){
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
    auto v84 = cooperative_groups::coalesced_threads();
    unsigned long long v85;
    v85 = cooperative_groups::reduce(v84, v81, v28);
    int v86;
    v86 = threadIdx.x;
    int v87;
    v87 = v86 / 32;
    extern __shared__ unsigned char v88[];
    unsigned long long * v89;
    v89 = reinterpret_cast<unsigned long long *>(&v88[0ull]);
    assert("Tensor range check" && 0 <= v87 && v87 < 8);
    v89[v87] = v85;
    __syncthreads();
    int v91;
    v91 = threadIdx.x;
    int v92;
    v92 = v91 % 32;
    bool v93;
    v93 = v92 < 8;
    unsigned long long v95;
    if (v93){
        assert("Tensor range check" && 0 <= v92 && v92 < 8);
        unsigned long long v94;
        v94 = v89[v92];
        v95 = v94;
    } else {
        v95 = 0ull;
    }
    __syncthreads();
    auto v96 = cooperative_groups::coalesced_threads();
    unsigned long long v97;
    v97 = cooperative_groups::reduce(v96, v95, v28);
    assert("Tensor range check" && 0 <= v82 && v82 < 24);
    v83[v82] = v97;
    v3.sync() ;
    unsigned long long v98;
    v98 = 0ull;
    int v99;
    v99 = threadIdx.x;
    int v100;
    v100 = v99 % 32;
    int v101;
    v101 = v100;
    while (while_method_2(v101)){
        bool v103;
        v103 = 0 <= v101;
        bool v104;
        v104 = v103 == false;
        if (v104){
            assert("The index needs to be zero or positive." && v103);
        } else {
        }
        bool v106;
        v106 = v101 < 24;
        bool v107;
        v107 = v106 == false;
        if (v107){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v106);
        } else {
        }
        assert("Tensor range check" && 0 <= v101 && v101 < 24);
        unsigned long long v109;
        v109 = v83[v101];
        unsigned long long v110;
        v110 = v98 + v109;
        v98 = v110;
        v101 += 32 ;
    }
    auto v111 = cooperative_groups::coalesced_threads();
    unsigned long long v112;
    v112 = cooperative_groups::reduce(v111, v98, v28);
    v2[0] = v112;
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
def method0(v0 : cp.ndarray, v1 : cp.ndarray, v2 : cp.ndarray) -> None:
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
    v11 = 0
    v12 = raw_module.get_function(f"entry{v11}")
    del v11
    v12.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v12((24,),(256,),(v0, v1, v2),shared_mem=98304)
    del v0, v1, v2, v12
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method2(v0 : i32) -> bool:
    v1 = v0 < 128
    del v0
    return v1
def method1(v0 : cp.ndarray) -> None:
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
def method3(v0 : cp.ndarray) -> None:
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
def method4(v0 : cp.ndarray) -> None:
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
    v0 = cp.arange(0,16384,1,dtype=cp.uint64) # type: ignore
    v1 = v0.size
    v2 = 16384 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.empty(1,dtype=cp.uint64)
    v6 = cp.empty(1,dtype=cp.uint64)
    method0(v0, v5, v6)
    method1(v0)
    del v0
    method3(v5)
    del v5
    return method4(v6)

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
