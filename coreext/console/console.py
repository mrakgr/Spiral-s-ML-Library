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

struct Union0;
struct Union1;
__device__ void method_0(Union0 v0);
__device__ void method_1(sptr<Union1> v0);
struct Union0_0 { // None
};
struct Union0_1 { // Some
    long long v1;
    float v0;
    unsigned char v2;
    __device__ Union0_1(float t0, long long t1, unsigned char t2) : v0(t0), v1(t1), v2(t2) {}
    __device__ Union0_1() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // None
        Union0_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // None
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Some
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // None
            case 1: new (&this->case1) Union0_1(x.case1); break; // Some
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{x};
        }
        return *this;
    }
    __device__ Union0 & operator=(Union0 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // None
            case 1: this->case1.~Union0_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union1_0 { // Cons
    sptr<Union1> v1;
    int v0;
    __device__ Union1_0(int t0, sptr<Union1> t1) : v0(t0), v1(t1) {}
    __device__ Union1_0() = delete;
};
struct Union1_1 { // Nil
};
struct Union1 {
    union {
        Union1_0 case0; // Cons
        Union1_1 case1; // Nil
    };
    int refc{0};
    unsigned char tag{255};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // Cons
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // Nil
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // Cons
            case 1: new (&this->case1) Union1_1(x.case1); break; // Nil
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // Cons
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // Nil
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Cons
                case 1: this->case1 = x.case1; break; // Nil
            }
        } else {
            this->~Union1();
            new (this) Union1{x};
        }
        return *this;
    }
    __device__ Union1 & operator=(Union1 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Cons
                case 1: this->case1 = std::move(x.case1); break; // Nil
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // Cons
            case 1: this->case1.~Union1_1(); break; // Nil
        }
        this->tag = 255;
    }
};
__device__ void method_0(Union0 v0){
    switch (v0.tag) {
        case 0: { // None
            printf("%s","None");
            return ;
            break;
        }
        case 1: { // Some
            float v2 = v0.case1.v0; long long v3 = v0.case1.v1; unsigned char v4 = v0.case1.v2;
            printf("%s(%f, %d, %u)","Some", v2, v3, v4);
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void method_1(sptr<Union1> v0){
    switch (v0.base->tag) {
        case 0: { // Cons
            int v1 = v0.base->case0.v0; sptr<Union1> v2 = v0.base->case0.v1;
            printf("%s(%d, ","Cons", v1);
            method_1(v2);
            printf(")");
            return ;
            break;
        }
        case 1: { // Nil
            printf("%s","Nil");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
extern "C" __global__ void entry0() {
    float v0;
    v0 = 3.0f;
    long long v1;
    v1 = 4ll;
    unsigned char v2;
    v2 = 5u;
    Union0 v3;
    v3 = Union0{Union0_1{v0, v1, v2}};
    int v4;
    v4 = 1l;
    int v5;
    v5 = 2l;
    int v6;
    v6 = 3l;
    int v7;
    v7 = 4l;
    sptr<Union1> v8;
    v8 = sptr<Union1>{new Union1{Union1_1{}}};
    sptr<Union1> v9;
    v9 = sptr<Union1>{new Union1{Union1_0{v7, v8}}};
    sptr<Union1> v10;
    v10 = sptr<Union1>{new Union1{Union1_0{v6, v9}}};
    sptr<Union1> v11;
    v11 = sptr<Union1>{new Union1{Union1_0{v5, v10}}};
    sptr<Union1> v12;
    v12 = sptr<Union1>{new Union1{Union1_0{v4, v11}}};
    cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v13 = console_lock;
    auto v14 = cooperative_groups::coalesced_threads();
    v13.acquire();
    printf("{%s = %s; %s = %d; %s = ","a", "false", "b", 2l, "c");
    method_0(v3);
    printf("; %s = ","d");
    method_1(v12);
    printf("}\n");
    v13.release();
    v14.sync() ;
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
UH0 = Union["UH0_0", "UH0_1"]
class US0_0(NamedTuple): # None
    tag = 0
class US0_1(NamedTuple): # Some
    v0 : f32
    v1 : i64
    v2 : u8
    tag = 1
US0 = Union[US0_0, US0_1]
class UH0_0(NamedTuple): # Cons
    v0 : i32
    v1 : UH0
    tag = 0
class UH0_1(NamedTuple): # Nil
    tag = 1
def method0(v0 : US0) -> None:
    match v0:
        case US0_0(): # None
            del v0
            v2 = "{}"
            v3 = "None"
            print(v2.format(v3),end="")
            del v2, v3
            return 
        case US0_1(v4, v5, v6): # Some
            del v0
            v8 = "{}({:.6f}, {}, {})"
            v9 = "Some"
            print(v8.format(v9, v4, v5, v6),end="")
            del v4, v5, v6, v8, v9
            return 
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method1(v0 : UH0) -> None:
    match v0:
        case UH0_0(v1, v2): # Cons
            del v0
            v4 = "{}({}, "
            v5 = "Cons"
            print(v4.format(v5, v1),end="")
            del v1, v4, v5
            method1(v2)
            del v2
            v6 = ")"
            print(v6,end="")
            del v6
            return 
        case UH0_1(): # Nil
            del v0
            v8 = "{}"
            v9 = "Nil"
            print(v8.format(v9),end="")
            del v8, v9
            return 
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def main():
    v0 = 3.0
    v1 = 4
    v2 = 5
    v3 = US0_1(v0, v1, v2)
    del v0, v1, v2
    v4 = 1
    v5 = 2
    v6 = 3
    v7 = 4
    v8 = UH0_1()
    v9 = UH0_0(v7, v8)
    del v7, v8
    v10 = UH0_0(v6, v9)
    del v6, v9
    v11 = UH0_0(v5, v10)
    del v5, v10
    v12 = UH0_0(v4, v11)
    del v4, v11
    v15 = "{{{} = {}; {} = {}; {} = "
    v16 = "a"
    v17 = "true"
    v18 = "b"
    v19 = "c"
    print(v15.format(v16, v17, v18, 2, v19),end="")
    del v15, v16, v17, v18, v19
    method0(v3)
    del v3
    v20 = "; {} = "
    v21 = "d"
    print(v20.format(v21),end="")
    del v20, v21
    method1(v12)
    del v12
    v22 = "}}\n"
    print(v22,end="")
    del v22
    v23 = 0
    v24 = raw_module.get_function(f"entry{v23}")
    del v23
    v24.max_dynamic_shared_size_bytes = 0 
    v24((1,),(32,),(),shared_mem=0)
    del v24
    return 

if __name__ == '__main__': print(main())
