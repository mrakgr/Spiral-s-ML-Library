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
    v1 = v0 < 32l;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 8l;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
extern "C" __global__ void entry0(float * v0, int * v1, int * v2) {
    int v3;
    v3 = threadIdx.x;
    bool v4;
    v4 = 0l <= v3;
    bool v5;
    v5 = v4 == false;
    if (v5){
        assert("The index needs to be zero or positive." && v4);
    } else {
    }
    int v7;
    v7 = v3 % 32l;
    int v8;
    v8 = v3 / 32l;
    bool v9;
    v9 = v8 < 1l;
    bool v10;
    v10 = v9 == false;
    if (v10){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v9);
    } else {
    }
    assert("Tensor range check" && 0 <= v8 && v8 < 1l);
    assert("Tensor range check" && 0 <= v7 && v7 < 32l);
    int v12;
    v12 = 4l * v7;
    int v13;
    v13 = 1024l * v8;
    int v14;
    v14 = v13 + v12;
    assert("Tensor range check" && 0 <= v8 && v8 < 1l);
    assert("Tensor range check" && 0 <= v7 && v7 < 32l);
    int v15;
    v15 = 0l;
    while (while_method_0(v15)){
        assert("Tensor range check" && 0 <= v15 && v15 < 32l);
        int v17;
        v17 = 1024l * v15;
        int v18;
        v18 = v17 + v14;
        float v19[32l];
        int v20[32l];
        int v21;
        v21 = 0l;
        while (while_method_1(v21)){
            assert("Tensor range check" && 0 <= v21 && v21 < 8l);
            int v23;
            v23 = 4l * v21;
            assert("Tensor range check" && 0 <= v21 && v21 < 8l);
            int v24;
            v24 = 128l * v21;
            int v25;
            v25 = v24 + v18;
            int4* v26;
            v26 = reinterpret_cast<int4*>(v0 + v25);
            int4* v27;
            v27 = reinterpret_cast<int4*>(v19 + v23);
            assert("Pointer alignment check" && (unsigned long long)(v26) % 4l == 0 && (unsigned long long)(v27) % 4l == 0);
            *v27 = *v26;
            v21 += 1l ;
        }
        int v28;
        v28 = 0l;
        while (while_method_1(v28)){
            int v30;
            v30 = 0l;
            while (while_method_2(v30)){
                bool v32;
                v32 = 0l <= v30;
                bool v34;
                if (v32){
                    bool v33;
                    v33 = v30 < 4l;
                    v34 = v33;
                } else {
                    v34 = false;
                }
                bool v35;
                v35 = v34 == false;
                if (v35){
                    assert("The indices should be inside the range of the dimension." && v34);
                } else {
                }
                bool v37;
                v37 = 0l <= v7;
                bool v39;
                if (v37){
                    bool v38;
                    v38 = v7 < 32l;
                    v39 = v38;
                } else {
                    v39 = false;
                }
                bool v40;
                v40 = v39 == false;
                if (v40){
                    assert("The indices should be inside the range of the dimension." && v39);
                } else {
                }
                int v42;
                v42 = v7 * 4l;
                int v43;
                v43 = v30 + v42;
                bool v44;
                v44 = 0l <= v28;
                bool v46;
                if (v44){
                    bool v45;
                    v45 = v28 < 8l;
                    v46 = v45;
                } else {
                    v46 = false;
                }
                bool v47;
                v47 = v46 == false;
                if (v47){
                    assert("The indices should be inside the range of the dimension." && v46);
                } else {
                }
                int v49;
                v49 = v28 * 128l;
                int v50;
                v50 = v43 + v49;
                assert("Tensor range check" && 0 <= v28 && v28 < 8l);
                assert("Tensor range check" && 0 <= v30 && v30 < 4l);
                int v51;
                v51 = 4l * v28;
                int v52;
                v52 = v51 + v30;
                v20[v52] = v50;
                v30 += 1l ;
            }
            v28 += 1l ;
        }
        bool v53;
        v53 = 0l <= v8;
        bool v54;
        v54 = v53 && v9;
        bool v55;
        v55 = v54 == false;
        if (v55){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v54);
        } else {
        }
        bool v57;
        v57 = 0l <= v15;
        bool v59;
        if (v57){
            bool v58;
            v58 = v15 < 32l;
            v59 = v58;
        } else {
            v59 = false;
        }
        bool v60;
        v60 = v59 == false;
        if (v60){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v59);
        } else {
        }
        int v62;
        v62 = v15 + v8;
        int v63[32l];
        int v64[32l];
        int v65;
        v65 = 0l;
        while (while_method_1(v65)){
            int v67;
            v67 = 0l;
            while (while_method_2(v67)){
                assert("Tensor range check" && 0 <= v65 && v65 < 8l);
                assert("Tensor range check" && 0 <= v67 && v67 < 4l);
                int v69;
                v69 = 4l * v65;
                int v70;
                v70 = v69 + v67;
                int v71;
                v71 = v20[v70];
                assert("Tensor range check" && 0 <= v65 && v65 < 8l);
                assert("Tensor range check" && 0 <= v67 && v67 < 4l);
                v63[v70] = v62;
                v64[v70] = v71;
                v67 += 1l ;
            }
            v65 += 1l ;
        }
        assert("Tensor range check" && 0 <= v15 && v15 < 32l);
        int v72;
        v72 = 0l;
        while (while_method_1(v72)){
            assert("Tensor range check" && 0 <= v72 && v72 < 8l);
            int v74;
            v74 = 128l * v72;
            int v75;
            v75 = v74 + v18;
            assert("Tensor range check" && 0 <= v72 && v72 < 8l);
            int v76;
            v76 = 4l * v72;
            int4* v77;
            v77 = reinterpret_cast<int4*>(v63 + v76);
            int4* v78;
            v78 = reinterpret_cast<int4*>(v1 + v75);
            assert("Pointer alignment check" && (unsigned long long)(v77) % 4l == 0 && (unsigned long long)(v78) % 4l == 0);
            *v78 = *v77;
            int4* v79;
            v79 = reinterpret_cast<int4*>(v64 + v76);
            int4* v80;
            v80 = reinterpret_cast<int4*>(v2 + v75);
            assert("Pointer alignment check" && (unsigned long long)(v79) % 4l == 0 && (unsigned long long)(v80) % 4l == 0);
            *v80 = *v79;
            v72 += 1l ;
        }
        v15 += 1l ;
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
    v1 = v0 < 1024
    del v0
    return v1
def main():
    v0 = cp.random.normal(0.0,1.0,32768,dtype=cp.float32) # type: ignore
    v1 = cp.empty(32768,dtype=cp.int32)
    v2 = cp.empty(32768,dtype=cp.int32)
    v3 = 0
    v4 = raw_module.get_function(f"entry{v3}")
    del v3
    v4.max_dynamic_shared_size_bytes = 0 
    v4((1,),(32,),(v0, v1, v2),shared_mem=0)
    del v0, v4
    v32 = 0
    v33 = "{}"
    print(v33.format('['),end="")
    v34 = 0
    while method0(v34):
        v36 = v32
        v37 = v36 >= 2147483647
        del v36
        if v37:
            v38 = " ..."
            print(v33.format(v38),end="")
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
            print(v33.format(v41),end="")
            del v41
        else:
            pass
        del v40
        print(v33.format('['),end="")
        v42 = 0
        while method1(v42):
            v44 = v32
            v45 = v44 >= 2147483647
            del v44
            if v45:
                v46 = " ..."
                print(v33.format(v46),end="")
                del v46
                break
            else:
                pass
            del v45
            v47 = v42 == 0
            v48 = v47 != True
            del v47
            if v48:
                v49 = "; "
                print(v33.format(v49),end="")
                del v49
            else:
                pass
            del v48
            v50 = v32 + 1
            v32 = v50
            del v50
            v51 = v34 * 1024
            v52 = v51 + v42
            del v51
            v53 = v1[v52].item()
            v54 = v2[v52].item()
            del v52
            v55 = "{}, {}"
            print(v55.format(v53, v54),end="")
            del v53, v54, v55
            v42 += 1 
        del v42
        print(v33.format(']'),end="")
        v34 += 1 
    del v1, v2, v32, v34
    print(v33.format(']'),end="")
    del v33
    v56 = "\n"
    print(v56,end="")
    del v56
    return 

if __name__ == '__main__': print(main())
