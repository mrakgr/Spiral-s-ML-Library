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
    v1 = v0 < 4l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, int * v1) {
    int v2;
    v2 = threadIdx.x;
    assert("Tensor range check" && 0 <= v2 && v2 < 32l);
    int v3;
    v3 = 16l * v2;
    int v4;
    v4 = threadIdx.x;
    assert("Tensor range check" && 0 <= v4 && v4 < 32l);
    int v5;
    v5 = 16l * v4;
    __shared__ float * v6[32l];
    __shared__ int * v7[32l];
    int v8;
    v8 = threadIdx.x;
    float * v9;
    v9 = v0+v3;
    int * v11;
    v11 = v1+v5;
    assert("Tensor range check" && 0 <= v8 && v8 < 32l);
    v6[v8] = v9;
    v7[v8] = v11;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v13;
    v13 = threadIdx.x;
    bool v14;
    v14 = 0l <= v13;
    bool v15;
    v15 = v14 == false;
    if (v15){
        assert("The index needs to be zero or positive." && v14);
    } else {
    }
    int v17;
    v17 = v13 % 4l;
    int v18;
    v18 = v13 / 4l;
    bool v19;
    v19 = v18 < 8l;
    bool v20;
    v20 = v19 == false;
    if (v20){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v19);
    } else {
    }
    assert("Tensor range check" && 0 <= v18 && v18 < 8l);
    int v22;
    v22 = 4l * v18;
    int v23;
    v23 = 0l;
    while (while_method_0(v23)){
        assert("Tensor range check" && 0 <= v23 && v23 < 4l);
        int v25;
        v25 = v23 + v22;
        float * v26;
        v26 = v6[v25];
        int * v27;
        v27 = v7[v25];
        assert("Tensor range check" && 0 <= v17 && v17 < 4l);
        int v28;
        v28 = 4l * v17;
        assert("Tensor range check" && 0 <= v17 && v17 < 4l);
        float v29[4l];
        int v30[4l];
        int v31;
        v31 = 0l;
        while (while_method_1(v31)){
            assert("Tensor range check" && 0 <= v31 && v31 < 1l);
            int v33;
            v33 = 4l * v31;
            assert("Tensor range check" && 0 <= v31 && v31 < 1l);
            int v34;
            v34 = v33 + v28;
            int4* v35;
            v35 = reinterpret_cast<int4*>(v26 + v34);
            int4* v36;
            v36 = reinterpret_cast<int4*>(v29 + v33);
            assert("Pointer alignment check" && (unsigned long long)(v35) % 4l == 0 && (unsigned long long)(v36) % 4l == 0);
            *v36 = *v35;
            v31 += 1l ;
        }
        int v37;
        v37 = 0l;
        while (while_method_1(v37)){
            int v39;
            v39 = 0l;
            while (while_method_0(v39)){
                bool v41;
                v41 = 0l <= v39;
                bool v43;
                if (v41){
                    bool v42;
                    v42 = v39 < 4l;
                    v43 = v42;
                } else {
                    v43 = false;
                }
                bool v44;
                v44 = v43 == false;
                if (v44){
                    assert("The indices should be inside the range of the dimension." && v43);
                } else {
                }
                bool v46;
                v46 = 0l <= v17;
                bool v48;
                if (v46){
                    bool v47;
                    v47 = v17 < 4l;
                    v48 = v47;
                } else {
                    v48 = false;
                }
                bool v49;
                v49 = v48 == false;
                if (v49){
                    assert("The indices should be inside the range of the dimension." && v48);
                } else {
                }
                int v51;
                v51 = v17 * 4l;
                int v52;
                v52 = v39 + v51;
                bool v53;
                v53 = 0l <= v37;
                bool v55;
                if (v53){
                    bool v54;
                    v54 = v37 < 1l;
                    v55 = v54;
                } else {
                    v55 = false;
                }
                bool v56;
                v56 = v55 == false;
                if (v56){
                    assert("The indices should be inside the range of the dimension." && v55);
                } else {
                }
                int v58;
                v58 = v37 * 16l;
                int v59;
                v59 = v52 + v58;
                assert("Tensor range check" && 0 <= v37 && v37 < 1l);
                assert("Tensor range check" && 0 <= v39 && v39 < 4l);
                int v60;
                v60 = 4l * v37;
                int v61;
                v61 = v60 + v39;
                v30[v61] = v59;
                v39 += 1l ;
            }
            v37 += 1l ;
        }
        bool v62;
        v62 = 0l <= v18;
        bool v63;
        v63 = v62 && v19;
        bool v64;
        v64 = v63 == false;
        if (v64){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v63);
        } else {
        }
        bool v66;
        v66 = 0l <= v23;
        bool v68;
        if (v66){
            bool v67;
            v67 = v23 < 4l;
            v68 = v67;
        } else {
            v68 = false;
        }
        bool v69;
        v69 = v68 == false;
        if (v69){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v68);
        } else {
        }
        int v71;
        v71 = v23 * 8l;
        int v72;
        v72 = v71 + v18;
        int v73;
        v73 = 0l;
        while (while_method_1(v73)){
            assert("Tensor range check" && 0 <= v73 && v73 < 1l);
            int v75;
            v75 = 4l * v73;
            int v76;
            v76 = v75 + v28;
            assert("Tensor range check" && 0 <= v73 && v73 < 1l);
            int4* v77;
            v77 = reinterpret_cast<int4*>(v30 + v75);
            int4* v78;
            v78 = reinterpret_cast<int4*>(v27 + v76);
            assert("Pointer alignment check" && (unsigned long long)(v77) % 4l == 0 && (unsigned long long)(v78) % 4l == 0);
            *v78 = *v77;
            v73 += 1l ;
        }
        v23 += 1l ;
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
    v1 = v0 < 16
    del v0
    return v1
def main():
    v0 = cp.random.normal(0.0,1.0,512,dtype=cp.float32) # type: ignore
    v1 = cp.random.uniform(size=32,dtype=cp.float32) # type: ignore
    del v1
    v2 = cp.empty(512,dtype=cp.int32)
    v3 = 0
    v4 = raw_module.get_function(f"entry{v3}")
    del v3
    v4.max_dynamic_shared_size_bytes = 0 
    v4((1,),(32,),(v0, v2),shared_mem=0)
    del v0, v4
    v30 = 0
    v31 = "{}"
    print(v31.format('['),end="")
    v32 = 0
    while method0(v32):
        v34 = v30
        v35 = v34 >= 2147483647
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
        while method1(v40):
            v42 = v30
            v43 = v42 >= 2147483647
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
            v49 = v32 * 16
            v50 = v49 + v40
            del v49
            v51 = v2[v50].item()
            del v50
            print(v31.format(v51),end="")
            del v51
            v40 += 1 
        del v40
        print(v31.format(']'),end="")
        v32 += 1 
    del v2, v30, v32
    print(v31.format(']'),end="")
    del v31
    v52 = "\n"
    print(v52,end="")
    del v52
    return 

if __name__ == '__main__': print(main())
