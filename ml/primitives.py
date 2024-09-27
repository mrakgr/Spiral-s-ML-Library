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
    v1 = v0 < 2;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 32;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_3(int v0){
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
        v13 = v7 < 2;
        bool v14;
        v14 = v13 == false;
        if (v14){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v13);
        } else {
        }
        assert("Tensor range check" && 0 <= v7 && v7 < 2);
        assert("Tensor range check" && 0 <= v12 && v12 < 1);
        assert("Tensor range check" && 0 <= v12 && v12 < 1);
        int v16;
        v16 = 8 * v12;
        int v17;
        v17 = 32 * v12;
        int v18;
        v18 = v17 + v16;
        int v19;
        v19 = 32 * v7;
        int v20;
        v20 = v19 + v18;
        int v21;
        v21 = 4 * v12;
        int v22;
        v22 = v21 + v17;
        int v23;
        v23 = v19 + v22;
        int v24;
        v24 = threadIdx.x;
        int v25;
        v25 = v24;
        while (while_method_1(v25)){
            bool v27;
            v27 = 0 <= v25;
            bool v28;
            v28 = v27 == false;
            if (v28){
                assert("The index needs to be zero or positive." && v27);
            } else {
            }
            int v30;
            v30 = v25 % 8;
            int v31;
            v31 = v25 / 8;
            bool v32;
            v32 = v31 < 4;
            bool v33;
            v33 = v32 == false;
            if (v33){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v32);
            } else {
            }
            assert("Tensor range check" && 0 <= v31 && v31 < 4);
            assert("Tensor range check" && 0 <= v30 && v30 < 8);
            int v35;
            v35 = v30 + v20;
            int v36;
            v36 = 8 * v31;
            int v37;
            v37 = v36 + v35;
            int v38;
            v38 = v0[v37];
            assert("Tensor range check" && 0 <= v31 && v31 < 4);
            assert("Tensor range check" && 0 <= v30 && v30 < 8);
            int v39;
            v39 = 129 * v31;
            int v40;
            v40 = v39 + v30;
            v4[v40] = v38;
            v25 += 256 ;
        }
        int v41;
        v41 = threadIdx.x;
        int v42;
        v42 = v41;
        while (while_method_2(v42)){
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
            v52 = v51 + v23;
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
            while (while_method_3(v58)){
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
            assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v63) % 16 == 0 && reinterpret_cast<unsigned long long>(v64) % 16 == 0);
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
    v1 = v0 < 2
    del v0
    return v1
def method1(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method2(v0 : i32) -> bool:
    v1 = v0 < 8
    del v0
    return v1
def main_body():
    v0 = cp.arange(0,64,1,dtype=cp.int32) # type: ignore
    v1 = v0.size
    v2 = 64 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.empty(64,dtype=cp.int32)
    v33 = 0
    v34 = "{}"
    print(v34.format('['),end="")
    v35 = 0
    while method0(v35):
        v37 = v33
        v38 = v37 >= 100
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
        while method1(v43):
            v45 = v33
            v46 = v45 >= 100
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
            print(v34.format('['),end="")
            v51 = 0
            while method2(v51):
                v53 = v33
                v54 = v53 >= 100
                del v53
                if v54:
                    v55 = " ..."
                    print(v34.format(v55),end="")
                    del v55
                    break
                else:
                    pass
                del v54
                v56 = v51 == 0
                v57 = v56 != True
                del v56
                if v57:
                    v58 = "; "
                    print(v34.format(v58),end="")
                    del v58
                else:
                    pass
                del v57
                v59 = v33 + 1
                v33 = v59
                del v59
                v60 = v35 * 32
                v61 = v43 * 8
                v62 = v60 + v61
                del v60, v61
                v63 = v62 + v51
                del v62
                v64 = v0[v63].item()
                del v63
                print(v34.format(v64),end="")
                del v64
                v51 += 1 
            del v51
            print(v34.format(']'),end="")
            v43 += 1 
        del v43
        print(v34.format(']'),end="")
        v35 += 1 
    del v33, v35
    print(v34.format(']'),end="")
    v65 = "\n"
    print(v65.format(),end="")
    v66 = cp.cuda.Device().attributes['MultiProcessorCount']
    v67 = v66 == 24
    del v66
    v68 = v67 == False
    if v68:
        v69 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v67, v69
        del v69
    else:
        pass
    del v67, v68
    v70 = 0
    v71 = raw_module.get_function(f"entry{v70}")
    del v70
    v71.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v71((24,),(256,),(v0, v5),shared_mem=98304)
    del v0, v71
    v99 = 0
    print(v34.format('['),end="")
    v100 = 0
    while method0(v100):
        v102 = v99
        v103 = v102 >= 100
        del v102
        if v103:
            v104 = " ..."
            print(v34.format(v104),end="")
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
            print(v34.format(v107),end="")
            del v107
        else:
            pass
        del v106
        print(v34.format('['),end="")
        v108 = 0
        while method2(v108):
            v110 = v99
            v111 = v110 >= 100
            del v110
            if v111:
                v112 = " ..."
                print(v34.format(v112),end="")
                del v112
                break
            else:
                pass
            del v111
            v113 = v108 == 0
            v114 = v113 != True
            del v113
            if v114:
                v115 = "; "
                print(v34.format(v115),end="")
                del v115
            else:
                pass
            del v114
            print(v34.format('['),end="")
            v116 = 0
            while method1(v116):
                v118 = v99
                v119 = v118 >= 100
                del v118
                if v119:
                    v120 = " ..."
                    print(v34.format(v120),end="")
                    del v120
                    break
                else:
                    pass
                del v119
                v121 = v116 == 0
                v122 = v121 != True
                del v121
                if v122:
                    v123 = "; "
                    print(v34.format(v123),end="")
                    del v123
                else:
                    pass
                del v122
                v124 = v99 + 1
                v99 = v124
                del v124
                v125 = v100 * 32
                v126 = v108 * 4
                v127 = v125 + v126
                del v125, v126
                v128 = v127 + v116
                del v127
                v129 = v5[v128].item()
                del v128
                print(v34.format(v129),end="")
                del v129
                v116 += 1 
            del v116
            print(v34.format(']'),end="")
            v108 += 1 
        del v108
        print(v34.format(']'),end="")
        v100 += 1 
    del v5, v99, v100
    print(v34.format(']'),end="")
    del v34
    print(v65.format(),end="")
    del v65
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
