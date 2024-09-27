kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
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
    v1 = v0 < 1;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
extern "C" __global__ void entry0(int * v0, int * v1) {
    auto v2 = cooperative_groups::this_grid();
    extern __shared__ unsigned char v3[];
    int * v4;
    v4 = reinterpret_cast<int *>(&v3[0ull]);
    int v6;
    v6 = blockIdx.x;
    int v7;
    v7 = v6;
    while (while_method_0(v7)){
        bool v9;
        v9 = 0 <= v7;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The index needs to be zero or positive." && v9);
        } else {
        }
        int v12;
        v12 = v7 % 1;
        bool v13;
        v13 = v7 < 1;
        bool v14;
        v14 = v13 == false;
        if (v14){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v13);
        } else {
        }
        assert("Tensor range check" && 0 <= v7 && v7 < 1);
        assert("Tensor range check" && 0 <= v12 && v12 < 1);
        int v16;
        v16 = 8 * v12;
        int v17;
        v17 = 32 * v7;
        int v18;
        v18 = v17 + v16;
        int v19;
        v19 = 32 * v12;
        int v20;
        v20 = 4 * v7;
        int v21;
        v21 = v20 + v19;
        int v22;
        v22 = threadIdx.x;
        int v23;
        v23 = v22;
        while (while_method_1(v23)){
            bool v25;
            v25 = 0 <= v23;
            bool v26;
            v26 = v25 == false;
            if (v26){
                assert("The index needs to be zero or positive." && v25);
            } else {
            }
            int v28;
            v28 = v23 % 2;
            int v29;
            v29 = v23 / 2;
            bool v30;
            v30 = v29 < 4;
            bool v31;
            v31 = v30 == false;
            if (v31){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v30);
            } else {
            }
            assert("Tensor range check" && 0 <= v29 && v29 < 4);
            assert("Tensor range check" && 0 <= v28 && v28 < 2);
            int v33;
            v33 = 4 * v28;
            int v34;
            v34 = 129 * v29;
            int v35;
            v35 = v34 + v33;
            int v36;
            v36 = v33 + v18;
            int v37;
            v37 = 8 * v29;
            int v38;
            v38 = v37 + v36;
            int4* v39;
            v39 = reinterpret_cast<int4*>(v0 + v38);
            int4* v40;
            v40 = reinterpret_cast<int4*>(v4 + v35);
            assert("Pointer alignment check" && (unsigned long long)(v39) % 4 == 0 && (unsigned long long)(v40) % 4 == 0);
            *v40 = *v39;
            v23 += 256 ;
        }
        int v41;
        v41 = threadIdx.x;
        int v42;
        v42 = v41;
        while (while_method_1(v42)){
            bool v44;
            v44 = 0 <= v42;
            bool v45;
            v45 = v44 == false;
            if (v45){
                assert("The index needs to be zero or positive." && v44);
            } else {
            }
            int v47;
            v47 = v42 % 1;
            bool v48;
            v48 = v42 < 8;
            bool v49;
            v49 = v48 == false;
            if (v49){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v48);
            } else {
            }
            assert("Tensor range check" && 0 <= v42 && v42 < 8);
            assert("Tensor range check" && 0 <= v47 && v47 < 1);
            int v51;
            v51 = 4 * v47;
            int v52;
            v52 = v51 + v21;
            int v53;
            v53 = 4 * v42;
            int v54;
            v54 = v53 + v52;
            int v55;
            v55 = 516 * v47;
            int v56;
            v56 = v42 + v55;
            int v57[4];
            int v58;
            v58 = 0;
            while (while_method_2(v58)){
                assert("Tensor range check" && 0 <= v58 && v58 < 4);
                int v60;
                v60 = 129 * v58;
                int v61;
                v61 = v60 + v56;
                int v62;
                v62 = v4[v61];
                assert("Tensor range check" && 0 <= v58 && v58 < 4);
                v57[v58] = v62;
                v58 += 1 ;
            }
            int4* v63;
            v63 = reinterpret_cast<int4*>(v57 + 0);
            int4* v64;
            v64 = reinterpret_cast<int4*>(v1 + v54);
            assert("Pointer alignment check" && (unsigned long long)(v63) % 4 == 0 && (unsigned long long)(v64) % 4 == 0);
            *v64 = *v63;
            v42 += 256 ;
        }
        __syncthreads();
        v7 += 24 ;
    }
    v2.sync() ;
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
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=255')
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
    v0 = cp.arange(0,32,1,dtype=cp.int32) # type: ignore
    v1 = v0.size
    v2 = 32 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.empty(32,dtype=cp.int32)
    v25 = 0
    v26 = "{}"
    print(v26.format('['),end="")
    v27 = 0
    while method0(v27):
        v29 = v25
        v30 = v29 >= 100
        del v29
        if v30:
            v31 = " ..."
            print(v26.format(v31),end="")
            del v31
            break
        else:
            pass
        del v30
        v32 = v27 == 0
        v33 = v32 != True
        del v32
        if v33:
            v34 = "; "
            print(v26.format(v34),end="")
            del v34
        else:
            pass
        del v33
        print(v26.format('['),end="")
        v35 = 0
        while method1(v35):
            v37 = v25
            v38 = v37 >= 100
            del v37
            if v38:
                v39 = " ..."
                print(v26.format(v39),end="")
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
                print(v26.format(v42),end="")
                del v42
            else:
                pass
            del v41
            v43 = v25 + 1
            v25 = v43
            del v43
            v44 = v27 * 8
            v45 = v44 + v35
            del v44
            v46 = v0[v45].item()
            del v45
            print(v26.format(v46),end="")
            del v46
            v35 += 1 
        del v35
        print(v26.format(']'),end="")
        v27 += 1 
    del v25, v27
    print(v26.format(']'),end="")
    v47 = "\n"
    print(v47.format(),end="")
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
    v53.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v53((24,),(256,),(v0, v5),shared_mem=98304)
    del v0, v53
    v73 = 0
    print(v26.format('['),end="")
    v74 = 0
    while method1(v74):
        v76 = v73
        v77 = v76 >= 100
        del v76
        if v77:
            v78 = " ..."
            print(v26.format(v78),end="")
            del v78
            break
        else:
            pass
        del v77
        v79 = v74 == 0
        v80 = v79 != True
        del v79
        if v80:
            v81 = "; "
            print(v26.format(v81),end="")
            del v81
        else:
            pass
        del v80
        print(v26.format('['),end="")
        v82 = 0
        while method0(v82):
            v84 = v73
            v85 = v84 >= 100
            del v84
            if v85:
                v86 = " ..."
                print(v26.format(v86),end="")
                del v86
                break
            else:
                pass
            del v85
            v87 = v82 == 0
            v88 = v87 != True
            del v87
            if v88:
                v89 = "; "
                print(v26.format(v89),end="")
                del v89
            else:
                pass
            del v88
            v90 = v73 + 1
            v73 = v90
            del v90
            v91 = v74 * 4
            v92 = v91 + v82
            del v91
            v93 = v5[v92].item()
            del v92
            print(v26.format(v93),end="")
            del v93
            v82 += 1 
        del v82
        print(v26.format(']'),end="")
        v74 += 1 
    del v5, v73, v74
    print(v26.format(']'),end="")
    del v26
    print(v47.format(),end="")
    del v47
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
