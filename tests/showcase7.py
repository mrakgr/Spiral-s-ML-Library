kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
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
    v1 = v0 < 8l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1) {
    int v2;
    v2 = threadIdx.x;
    int v3;
    v3 = v2;
    while (while_method_0(v3)){
        bool v5;
        v5 = 0l <= v3;
        bool v6;
        v6 = v5 == false;
        if (v6){
            assert("The index needs to be zero or positive." && v5);
        } else {
        }
        int v8;
        v8 = v3 % 2l;
        int v9;
        v9 = v3 / 2l;
        bool v10;
        v10 = v9 < 4l;
        bool v11;
        v11 = v10 == false;
        if (v11){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v10);
        } else {
        }
        assert("Tensor range check" && 0 <= v9 && v9 < 4l);
        assert("Tensor range check" && 0 <= v8 && v8 < 2l);
        int v13;
        v13 = 4l * v8;
        int v14;
        v14 = 8l * v9;
        int v15;
        v15 = v14 + v13;
        assert("Tensor range check" && 0 <= v9 && v9 < 4l);
        assert("Tensor range check" && 0 <= v8 && v8 < 2l);
        float v16[4l];
        float v17[4l];
        int4* v18;
        v18 = reinterpret_cast<int4*>(v0 + v15);
        int4* v19;
        v19 = reinterpret_cast<int4*>(v16 + 0l);
        assert("Pointer alignment check" && (unsigned long long)(v18) % 4l == 0 && (unsigned long long)(v19) % 4l == 0);
        *v19 = *v18;
        // Pushing the loop unrolling to: 0
        int v20;
        v20 = 0l;
        #pragma unroll
        while (while_method_1(v20)){
            assert("Tensor range check" && 0 <= v20 && v20 < 4l);
            float v22;
            v22 = v16[v20];
            float v23;
            v23 = v22 + 5.0f;
            assert("Tensor range check" && 0 <= v20 && v20 < 4l);
            v17[v20] = v23;
            v20 += 1l ;
        }
        // Poping the loop unrolling to: 0
        int4* v24;
        v24 = reinterpret_cast<int4*>(v17 + 0l);
        int4* v25;
        v25 = reinterpret_cast<int4*>(v1 + v15);
        assert("Pointer alignment check" && (unsigned long long)(v24) % 4l == 0 && (unsigned long long)(v25) % 4l == 0);
        *v25 = *v24;
        v3 += 256l ;
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
    v1 = v0 < 4
    del v0
    return v1
def method1(v0 : i32) -> bool:
    v1 = v0 < 8
    del v0
    return v1
def main_body():
    v0 = cp.zeros(32,dtype=cp.float32) # type: ignore
    v1 = cp.empty(32,dtype=cp.float32)
    v2 = cp.cuda.Device().attributes['MultiProcessorCount']
    v3 = v2 == 24
    del v2
    v4 = v3 == False
    if v4:
        v5 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v6 = 0
    v7 = raw_module.get_function(f"entry{v6}")
    del v6
    v7.max_dynamic_shared_size_bytes = 81920 
    v7((24,),(256,),(v0, v1),shared_mem=81920)
    del v7
    v44 = 0
    v45 = "{{{} = {}"
    v46 = "in_"
    print(v45.format(v46, '['),end="")
    del v45, v46
    v47 = 0
    while method0(v47):
        v49 = v44
        v50 = v49 >= 100
        del v49
        if v50:
            v51 = "{}"
            v52 = " ..."
            print(v51.format(v52),end="")
            del v51, v52
            break
        else:
            pass
        del v50
        v53 = v47 == 0
        v54 = v53 != True
        del v53
        if v54:
            v55 = "{}"
            v56 = "; "
            print(v55.format(v56),end="")
            del v55, v56
        else:
            pass
        del v54
        v57 = "{}"
        print(v57.format('['),end="")
        v58 = 0
        while method1(v58):
            v60 = v44
            v61 = v60 >= 100
            del v60
            if v61:
                v62 = " ..."
                print(v57.format(v62),end="")
                del v62
                break
            else:
                pass
            del v61
            v63 = v58 == 0
            v64 = v63 != True
            del v63
            if v64:
                v65 = "; "
                print(v57.format(v65),end="")
                del v65
            else:
                pass
            del v64
            v66 = v44 + 1
            v44 = v66
            del v66
            v67 = v47 * 8
            v68 = v67 + v58
            del v67
            v69 = v0[v68].item()
            del v68
            v70 = "{:.6f}"
            print(v70.format(v69),end="")
            del v69, v70
            v58 += 1 
        del v58
        print(v57.format(']'),end="")
        del v57
        v47 += 1 
    del v0, v44, v47
    v71 = "{}"
    print(v71.format(']'),end="")
    v72 = 0
    v73 = "; {} = {}"
    v74 = "out"
    print(v73.format(v74, '['),end="")
    del v73, v74
    v75 = 0
    while method0(v75):
        v77 = v72
        v78 = v77 >= 100
        del v77
        if v78:
            v79 = " ..."
            print(v71.format(v79),end="")
            del v79
            break
        else:
            pass
        del v78
        v80 = v75 == 0
        v81 = v80 != True
        del v80
        if v81:
            v82 = "; "
            print(v71.format(v82),end="")
            del v82
        else:
            pass
        del v81
        print(v71.format('['),end="")
        v83 = 0
        while method1(v83):
            v85 = v72
            v86 = v85 >= 100
            del v85
            if v86:
                v87 = " ..."
                print(v71.format(v87),end="")
                del v87
                break
            else:
                pass
            del v86
            v88 = v83 == 0
            v89 = v88 != True
            del v88
            if v89:
                v90 = "; "
                print(v71.format(v90),end="")
                del v90
            else:
                pass
            del v89
            v91 = v72 + 1
            v72 = v91
            del v91
            v92 = v75 * 8
            v93 = v92 + v83
            del v92
            v94 = v1[v93].item()
            del v93
            v95 = "{:.6f}"
            print(v95.format(v94),end="")
            del v94, v95
            v83 += 1 
        del v83
        print(v71.format(']'),end="")
        v75 += 1 
    del v1, v72, v75
    print(v71.format(']'),end="")
    del v71
    v96 = "}}\n"
    print(v96,end="")
    del v96
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
