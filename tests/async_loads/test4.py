kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>
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

__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 262144l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1) {
    cuda::pipeline<cuda::thread_scope_thread> v2 = cuda::make_pipeline();
    int v3;
    v3 = threadIdx.x;
    assert("Tensor range check" && 0 <= v3 && v3 < 256l);
    int v4;
    v4 = 4l * v3;
    extern __shared__ unsigned char v5[];
    float * v6;
    v6 = reinterpret_cast<float *>(&v5[0ull]);
    int v8;
    v8 = threadIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 256l);
    int v9;
    v9 = 4l * v8;
    float v10[4l];
    float v11[4l];
    int v12;
    v12 = blockIdx.x;
    int v13;
    v13 = v12;
    while (while_method_0(v13)){
        bool v15;
        v15 = 0l <= v13;
        bool v16;
        v16 = v15 == false;
        if (v16){
            assert("The index needs to be zero or positive." && v15);
        } else {
        }
        bool v18;
        v18 = v13 < 262144l;
        bool v19;
        v19 = v18 == false;
        if (v19){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v18);
        } else {
        }
        assert("Tensor range check" && 0 <= v13 && v13 < 262144l);
        int v21;
        v21 = 1024l * v13;
        int v22;
        v22 = v21 + v4;
        v2.producer_acquire();
        constexpr int v23 = sizeof(float) * 4l;
        assert("Pointer alignment check" && (unsigned long long)(v0 + v22) % v23 == 0 && (unsigned long long)(v6 + v9) % v23 == 0);
        cuda::memcpy_async(v6 + v9, v0 + v22, cuda::aligned_size_t<v23>(v23), v2);
        v2.producer_commit();
        cuda::pipeline_consumer_wait_prior<0>(v2);;
        int4* v24;
        v24 = reinterpret_cast<int4*>(v6 + v9);
        int4* v25;
        v25 = reinterpret_cast<int4*>(v10 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v24) % 4l == 0 && (unsigned long long)(v25) % 4l == 0);
        *v25 = *v24;
        // Pushing the loop unrolling to: 0
        int v26;
        v26 = 0l;
        #pragma unroll
        while (while_method_1(v26)){
            assert("Tensor range check" && 0 <= v26 && v26 < 4l);
            float v28;
            v28 = v10[v26];
            __nanosleep(128ul);
            float v29;
            v29 = v28 + 10.0f;
            assert("Tensor range check" && 0 <= v26 && v26 < 4l);
            v11[v26] = v29;
            v26 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v30;
        v30 = reinterpret_cast<int4*>(v11 + 0l);
        int4* v31;
        v31 = reinterpret_cast<int4*>(v1 + v22);
        assert("Pointer alignment check" && (unsigned long long)(v30) % 4l == 0 && (unsigned long long)(v31) % 4l == 0);
        *v31 = *v30;
        v2.consumer_release();
        v13 += 24l ;
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

options = []
options.append('--define-macro=NDEBUG')
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=256')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def main_body():
    v2 = "{}\n"
    v3 = "Running test 4. Non-interleaved asynchronous loads."
    print(v2.format(v3),end="")
    del v2, v3
    v4 = cp.ones(268435456,dtype=cp.float32) # type: ignore
    v5 = cp.empty(268435456,dtype=cp.float32)
    v6 = cp.cuda.Device().attributes['MultiProcessorCount']
    v7 = v6 == 24
    del v6
    v8 = v7 == False
    if v8:
        v9 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v7, v9
        del v9
    else:
        pass
    del v7, v8
    v10 = 0
    v11 = raw_module.get_function(f"entry{v10}")
    del v10
    v11.max_dynamic_shared_size_bytes = 81920 
    v11((24,),(256,),(v4, v5),shared_mem=81920)
    del v4, v11
    v23 = 0
    v24 = "{}"
    print(v24.format('['),end="")
    v25 = 0
    while method0(v25):
        v27 = v23
        v28 = v27 >= 100
        del v27
        if v28:
            v29 = " ..."
            print(v24.format(v29),end="")
            del v29
            break
        else:
            pass
        del v28
        v30 = v25 == 0
        v31 = v30 != True
        del v30
        if v31:
            v32 = "; "
            print(v24.format(v32),end="")
            del v32
        else:
            pass
        del v31
        v33 = v23 + 1
        v23 = v33
        del v33
        v34 = v5[v25].item()
        v35 = "{:.6f}"
        print(v35.format(v34),end="")
        del v34, v35
        v25 += 1 
    del v5, v23, v25
    print(v24.format(']'),end="")
    del v24
    v36 = "\n"
    print(v36.format(),end="")
    del v36
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
