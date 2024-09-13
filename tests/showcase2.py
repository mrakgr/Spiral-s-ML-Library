kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
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

__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 5l;
    return v1;
}
extern "C" __global__ void entry0(float * v0) {
    int v1;
    v1 = threadIdx.x;
    int v2;
    v2 = blockIdx.x;
    int v3;
    v3 = v2 * 256l;
    int v4;
    v4 = v1 + v3;
    bool v5;
    v5 = v4 < 4l;
    if (v5){
        cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v6 = console_lock;
        auto v7 = cooperative_groups::coalesced_threads();
        v6.acquire();
        int v8;
        v8 = 0l;
        printf("{%s = %c","tensor", '[');
        int v9;
        v9 = 0l;
        while (while_method_0(v9)){
            int v11;
            v11 = v8;
            bool v12;
            v12 = v11 >= 100l;
            if (v12){
                printf("%s"," ...");
                break;
            } else {
            }
            bool v13;
            v13 = v9 == 0l;
            bool v14;
            v14 = v13 != true;
            if (v14){
                printf("%s","; ");
            } else {
            }
            printf("%c",'[');
            int v15;
            v15 = 0l;
            while (while_method_0(v15)){
                int v17;
                v17 = v8;
                bool v18;
                v18 = v17 >= 100l;
                if (v18){
                    printf("%s"," ...");
                    break;
                } else {
                }
                bool v19;
                v19 = v15 == 0l;
                bool v20;
                v20 = v19 != true;
                if (v20){
                    printf("%s","; ");
                } else {
                }
                int v21;
                v21 = v8 + 1l;
                v8 = v21;
                int v22;
                v22 = v9 * 5l;
                int v23;
                v23 = v22 + v15;
                float v24;
                v24 = v0[v23];
                printf("%f",v24);
                v15 += 1l ;
            }
            printf("%c",']');
            v9 += 1l ;
        }
        printf("%c",']');
        printf("; %s = %d}\n","tid", v4);
        v6.release();
        v7.sync() ;
        return ;
    } else {
        return ;
    }
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
    v1 = v0 < 5
    del v0
    return v1
def main_body():
    v0 = cp.random.normal(0.0,1.0,25,dtype=cp.float32) # type: ignore
    v3 = "{}\n"
    v4 = "Printing this on host."
    print(v3.format(v4),end="")
    del v3, v4
    v24 = 0
    v25 = "{}"
    print(v25.format('['),end="")
    v26 = 0
    while method0(v26):
        v28 = v24
        v29 = v28 >= 100
        del v28
        if v29:
            v30 = " ..."
            print(v25.format(v30),end="")
            del v30
            break
        else:
            pass
        del v29
        v31 = v26 == 0
        v32 = v31 != True
        del v31
        if v32:
            v33 = "; "
            print(v25.format(v33),end="")
            del v33
        else:
            pass
        del v32
        print(v25.format('['),end="")
        v34 = 0
        while method0(v34):
            v36 = v24
            v37 = v36 >= 100
            del v36
            if v37:
                v38 = " ..."
                print(v25.format(v38),end="")
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
                print(v25.format(v41),end="")
                del v41
            else:
                pass
            del v40
            v42 = v24 + 1
            v24 = v42
            del v42
            v43 = v26 * 5
            v44 = v43 + v34
            del v43
            v45 = v0[v44].item()
            del v44
            v46 = "{:.6f}"
            print(v46.format(v45),end="")
            del v45, v46
            v34 += 1 
        del v34
        print(v25.format(']'),end="")
        v26 += 1 
    del v24, v26
    print(v25.format(']'),end="")
    del v25
    v47 = "\n"
    print(v47,end="")
    del v47
    v48 = cp.cuda.Device().attributes['MultiProcessorCount']
    v49 = v48 == 24
    del v48
    v50 = v49 == False
    if v50:
        v51 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v49, v51
        del v51
    else:
        pass
    del v49, v50
    v52 = 0
    v53 = raw_module.get_function(f"entry{v52}")
    del v52
    v53.max_dynamic_shared_size_bytes = 81920 
    v53((24,),(256,),v0,shared_mem=81920)
    del v0, v53
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
