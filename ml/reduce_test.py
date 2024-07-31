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

__device__ int return_snd_0(int v0, int v1);
struct Closure0 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        return return_snd_0(v0, v1);
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 32l;
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
__device__ int return_snd_0(int v0, int v1){
    bool v2;
    v2 = v0 < v1;
    bool v3;
    v3 = v2 == false;
    if (v3){
        assert("Expected leftmost argument to be passed first." && v2);
    } else {
    }
    return v1;
}
extern "C" __global__ void entry0(int * v0, int * v1) {
    int v2;
    v2 = threadIdx.x;
    bool v3;
    v3 = 0l <= v2;
    bool v4;
    v4 = v3 == false;
    if (v4){
        assert("The index needs to be zero or positive." && v3);
    } else {
    }
    int v6;
    v6 = v2 % 4l;
    int v7;
    v7 = v2 / 4l;
    bool v8;
    v8 = v7 < 8l;
    bool v9;
    v9 = v8 == false;
    if (v9){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v8);
    } else {
    }
    assert("Tensor range check" && 0 <= v7 && v7 < 8l);
    assert("Tensor range check" && 0 <= v6 && v6 < 4l);
    int v11;
    v11 = 4l * v6;
    int v12;
    v12 = 16l * v7;
    int v13;
    v13 = v12 + v11;
    assert("Tensor range check" && 0 <= v7 && v7 < 8l);
    int v14;
    v14 = 0l;
    while (while_method_0(v14)){
        assert("Tensor range check" && 0 <= v14 && v14 < 32l);
        int v16;
        v16 = 128l * v14;
        int v17;
        v17 = v16 + v13;
        int v18[4l];
        int v19[4l];
        int v20;
        v20 = 0l;
        while (while_method_1(v20)){
            assert("Tensor range check" && 0 <= v20 && v20 < 1l);
            int v22;
            v22 = 4l * v20;
            assert("Tensor range check" && 0 <= v20 && v20 < 1l);
            int v23;
            v23 = 16l * v20;
            int v24;
            v24 = v23 + v17;
            int4* v25;
            v25 = reinterpret_cast<int4*>(v0 + v24);
            int4* v26;
            v26 = reinterpret_cast<int4*>(v18 + v22);
            assert("Pointer alignment check" && (unsigned long long)(v25) % 4l == 0 && (unsigned long long)(v26) % 4l == 0);
            *v26 = *v25;
            v20 += 1l ;
        }
        int v27;
        v27 = 0l;
        while (while_method_1(v27)){
            int v29;
            v29 = 0l;
            while (while_method_2(v29)){
                bool v31;
                v31 = 0l <= v29;
                bool v33;
                if (v31){
                    bool v32;
                    v32 = v29 < 4l;
                    v33 = v32;
                } else {
                    v33 = false;
                }
                bool v34;
                v34 = v33 == false;
                if (v34){
                    assert("The indices should be inside the range of the dimension." && v33);
                } else {
                }
                bool v36;
                v36 = 0l <= v6;
                bool v38;
                if (v36){
                    bool v37;
                    v37 = v6 < 4l;
                    v38 = v37;
                } else {
                    v38 = false;
                }
                bool v39;
                v39 = v38 == false;
                if (v39){
                    assert("The indices should be inside the range of the dimension." && v38);
                } else {
                }
                int v41;
                v41 = v6 * 4l;
                int v42;
                v42 = v29 + v41;
                bool v43;
                v43 = 0l <= v27;
                bool v45;
                if (v43){
                    bool v44;
                    v44 = v27 < 1l;
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
                v48 = v27 * 16l;
                int v49;
                v49 = v42 + v48;
                assert("Tensor range check" && 0 <= v27 && v27 < 1l);
                assert("Tensor range check" && 0 <= v29 && v29 < 4l);
                int v50;
                v50 = 4l * v27;
                int v51;
                v51 = v50 + v29;
                v19[v51] = v49;
                v29 += 1l ;
            }
            v27 += 1l ;
        }
        bool v52;
        v52 = 0l <= v7;
        bool v53;
        v53 = v52 && v8;
        bool v54;
        v54 = v53 == false;
        if (v54){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v53);
        } else {
        }
        bool v56;
        v56 = 0l <= v14;
        bool v58;
        if (v56){
            bool v57;
            v57 = v14 < 32l;
            v58 = v57;
        } else {
            v58 = false;
        }
        bool v59;
        v59 = v58 == false;
        if (v59){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v58);
        } else {
        }
        int v61;
        v61 = v14 * 8l;
        int v62;
        v62 = v61 + v7;
        int v63;
        v63 = -1l;
        int v64;
        v64 = 0l;
        while (while_method_1(v64)){
            int v66;
            v66 = 0l;
            while (while_method_2(v66)){
                assert("Tensor range check" && 0 <= v64 && v64 < 1l);
                assert("Tensor range check" && 0 <= v66 && v66 < 4l);
                int v68;
                v68 = 4l * v64;
                int v69;
                v69 = v68 + v66;
                int v70;
                v70 = v18[v69];
                int v71;
                v71 = return_snd_0(v63, v70);
                v63 = v71;
                v66 += 1l ;
            }
            v64 += 1l ;
        }
        auto v72 = cooperative_groups::coalesced_threads();
        int v73;
        v73 = threadIdx.x;
        int v74;
        v74 = v73 / 4l;
        auto v75 = cooperative_groups::labeled_partition(v72,v74);
        Closure0 v76{};
        int v77;
        v77 = cooperative_groups::reduce(v75, v63, v76);
        assert("Tensor range check" && 0 <= v14 && v14 < 32l);
        int v78;
        v78 = 8l * v14;
        int v79;
        v79 = v78 + v7;
        v1[v79] = v77;
        v14 += 1l ;
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
def method0(v0 : string) -> None:
    print(v0, end="")
    del v0
    return 
def method1(v0 : char) -> None:
    print(v0, end="")
    del v0
    return 
def method2(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method3(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def method4(v0 : i32) -> None:
    print(v0, end="")
    del v0
    return 
def method5() -> None:
    return 
def main():
    v0 = cp.arange(0,4096,1,dtype=cp.int32) # type: ignore
    v1 = v0.size
    v2 = 4096 == v1
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = cp.empty(256,dtype=cp.int32)
    v6 = "The input is:"
    method0(v6)
    del v6
    print()
    v7 = 0
    v8 = '['
    method1(v8)
    del v8
    v9 = 0
    while method2(v9):
        v11 = v7
        v12 = v11 >= 2147483647
        del v11
        if v12:
            v13 = " ..."
            method0(v13)
            del v13
            break
        else:
            pass
        del v12
        v14 = v9 == 0
        v15 = v14 != True
        del v14
        if v15:
            v16 = "; "
            method0(v16)
        else:
            pass
        del v15
        v17 = '['
        method1(v17)
        del v17
        v18 = 0
        while method3(v18):
            v20 = v7
            v21 = v20 >= 2147483647
            del v20
            if v21:
                v22 = " ..."
                method0(v22)
                del v22
                break
            else:
                pass
            del v21
            v23 = v18 == 0
            v24 = v23 != True
            del v23
            if v24:
                v25 = "; "
                method0(v25)
            else:
                pass
            del v24
            v26 = v7 + 1
            v7 = v26
            del v26
            v27 = v9 * 16
            v28 = v27 + v18
            del v27
            v29 = v0[v28].item()
            del v28
            method4(v29)
            del v29
            v18 += 1 
        del v18
        v30 = ']'
        method1(v30)
        del v30
        v9 += 1 
    del v7, v9
    v31 = ']'
    method1(v31)
    del v31
    method5()
    print()
    v32 = 0
    v33 = raw_module.get_function(f"entry{v32}")
    del v32
    v33.max_dynamic_shared_size_bytes = 0 
    v33((1,),(32,),(v0, v5),shared_mem=0)
    del v0, v33
    v34 = "The output is:"
    method0(v34)
    del v34
    print()
    v35 = 0
    v36 = '['
    method1(v36)
    del v36
    v37 = 0
    while method2(v37):
        v39 = v35
        v40 = v39 >= 2147483647
        del v39
        if v40:
            v41 = " ..."
            method0(v41)
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
            method0(v44)
        else:
            pass
        del v43
        v45 = v35 + 1
        v35 = v45
        del v45
        v46 = v5[v37].item()
        method4(v46)
        del v46
        v37 += 1 
    del v5, v35, v37
    v47 = ']'
    method1(v47)
    del v47
    method5()
    print()
    return 

if __name__ == '__main__': print(main())
