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

struct Closure0 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 1l;
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
    bool v3;
    v3 = 0l <= v2;
    bool v4;
    v4 = v3 == false;
    if (v4){
        assert("The index needs to be zero or positive." && v3);
    } else {
    }
    int v6;
    v6 = v2 % 1l;
    bool v7;
    v7 = v2 < 32l;
    bool v8;
    v8 = v7 == false;
    if (v8){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v7);
    } else {
    }
    assert("Tensor range check" && 0 <= v2 && v2 < 32l);
    assert("Tensor range check" && 0 <= v6 && v6 < 1l);
    int v10;
    v10 = 4l * v6;
    int v11;
    v11 = 4l * v2;
    int v12;
    v12 = v11 + v10;
    assert("Tensor range check" && 0 <= v2 && v2 < 32l);
    assert("Tensor range check" && 0 <= v6 && v6 < 1l);
    int v13;
    v13 = 0l;
    while (while_method_0(v13)){
        assert("Tensor range check" && 0 <= v13 && v13 < 1l);
        int v15;
        v15 = 128l * v13;
        int v16;
        v16 = v15 + v12;
        float v17[4l];
        int v18[4l];
        int v19;
        v19 = 0l;
        while (while_method_0(v19)){
            assert("Tensor range check" && 0 <= v19 && v19 < 1l);
            int v21;
            v21 = 4l * v19;
            assert("Tensor range check" && 0 <= v19 && v19 < 1l);
            int v22;
            v22 = v21 + v16;
            int4* v23;
            v23 = reinterpret_cast<int4*>(v0 + v22);
            int4* v24;
            v24 = reinterpret_cast<int4*>(v17 + v21);
            assert("Pointer alignment check" && (unsigned long long)(v23) % 4l == 0 && (unsigned long long)(v24) % 4l == 0);
            *v24 = *v23;
            v19 += 1l ;
        }
        int v25;
        v25 = 0l;
        while (while_method_0(v25)){
            int v27;
            v27 = 0l;
            while (while_method_1(v27)){
                bool v29;
                v29 = 0l <= v27;
                bool v31;
                if (v29){
                    bool v30;
                    v30 = v27 < 4l;
                    v31 = v30;
                } else {
                    v31 = false;
                }
                bool v32;
                v32 = v31 == false;
                if (v32){
                    assert("The indices should be inside the range of the dimension." && v31);
                } else {
                }
                bool v34;
                v34 = 0l <= v6;
                bool v36;
                if (v34){
                    bool v35;
                    v35 = v6 < 1l;
                    v36 = v35;
                } else {
                    v36 = false;
                }
                bool v37;
                v37 = v36 == false;
                if (v37){
                    assert("The indices should be inside the range of the dimension." && v36);
                } else {
                }
                int v39;
                v39 = v6 * 4l;
                int v40;
                v40 = v27 + v39;
                bool v41;
                v41 = 0l <= v25;
                bool v43;
                if (v41){
                    bool v42;
                    v42 = v25 < 1l;
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
                int v46;
                v46 = v25 * 4l;
                int v47;
                v47 = v40 + v46;
                assert("Tensor range check" && 0 <= v25 && v25 < 1l);
                assert("Tensor range check" && 0 <= v27 && v27 < 4l);
                int v48;
                v48 = 4l * v25;
                int v49;
                v49 = v48 + v27;
                v18[v49] = v47;
                v27 += 1l ;
            }
            v25 += 1l ;
        }
        bool v50;
        v50 = v3 && v7;
        bool v51;
        v51 = v50 == false;
        if (v51){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v50);
        } else {
        }
        bool v53;
        v53 = 0l <= v13;
        bool v55;
        if (v53){
            bool v54;
            v54 = v13 < 1l;
            v55 = v54;
        } else {
            v55 = false;
        }
        bool v56;
        v56 = v55 == false;
        if (v56){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v55);
        } else {
        }
        int v58;
        v58 = v13 * 32l;
        int v59;
        v59 = v58 + v2;
        float v60;
        v60 = 0.0f;
        int v61;
        v61 = 0l;
        while (while_method_0(v61)){
            int v63;
            v63 = 0l;
            while (while_method_1(v63)){
                assert("Tensor range check" && 0 <= v61 && v61 < 1l);
                assert("Tensor range check" && 0 <= v63 && v63 < 4l);
                int v65;
                v65 = 4l * v61;
                int v66;
                v66 = v65 + v63;
                float v67;
                v67 = v17[v66];
                float v68;
                v68 = v60 + v67;
                v60 = v68;
                v63 += 1l ;
            }
            v61 += 1l ;
        }
        auto v69 = cooperative_groups::coalesced_threads();
        int v70;
        v70 = threadIdx.x;
        auto v71 = cooperative_groups::labeled_partition(v69,v70);
        Closure0 v72{};
        float v73;
        v73 = cooperative_groups::reduce(v71, v60, v72);
        float v74;
        v74 = v73 / 4.0f;
        float v75[4l];
        int v76;
        v76 = 0l;
        while (while_method_0(v76)){
            int v78;
            v78 = 0l;
            while (while_method_1(v78)){
                assert("Tensor range check" && 0 <= v76 && v76 < 1l);
                assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                int v80;
                v80 = 4l * v76;
                int v81;
                v81 = v80 + v78;
                float v82;
                v82 = v17[v81];
                float v83;
                v83 = v82 - v74;
                float v84;
                v84 = exp(v83);
                assert("Tensor range check" && 0 <= v76 && v76 < 1l);
                assert("Tensor range check" && 0 <= v78 && v78 < 4l);
                v75[v81] = v84;
                v78 += 1l ;
            }
            v76 += 1l ;
        }
        float v85;
        v85 = 0.0f;
        int v86;
        v86 = 0l;
        while (while_method_0(v86)){
            int v88;
            v88 = 0l;
            while (while_method_1(v88)){
                assert("Tensor range check" && 0 <= v86 && v86 < 1l);
                assert("Tensor range check" && 0 <= v88 && v88 < 4l);
                int v90;
                v90 = 4l * v86;
                int v91;
                v91 = v90 + v88;
                float v92;
                v92 = v75[v91];
                float v93;
                v93 = v85 + v92;
                v85 = v93;
                v88 += 1l ;
            }
            v86 += 1l ;
        }
        auto v94 = cooperative_groups::coalesced_threads();
        int v95;
        v95 = threadIdx.x;
        auto v96 = cooperative_groups::labeled_partition(v94,v95);
        float v97;
        v97 = cooperative_groups::reduce(v96, v85, v72);
        float v98[4l];
        int v99;
        v99 = 0l;
        while (while_method_0(v99)){
            int v101;
            v101 = 0l;
            while (while_method_1(v101)){
                assert("Tensor range check" && 0 <= v99 && v99 < 1l);
                assert("Tensor range check" && 0 <= v101 && v101 < 4l);
                int v103;
                v103 = 4l * v99;
                int v104;
                v104 = v103 + v101;
                float v105;
                v105 = v75[v104];
                float v106;
                v106 = v105 / v97;
                assert("Tensor range check" && 0 <= v99 && v99 < 1l);
                assert("Tensor range check" && 0 <= v101 && v101 < 4l);
                v98[v104] = v106;
                v101 += 1l ;
            }
            v99 += 1l ;
        }
        assert("Tensor range check" && 0 <= v13 && v13 < 1l);
        int v107;
        v107 = 0l;
        while (while_method_0(v107)){
            assert("Tensor range check" && 0 <= v107 && v107 < 1l);
            int v109;
            v109 = 4l * v107;
            int v110;
            v110 = v109 + v16;
            assert("Tensor range check" && 0 <= v107 && v107 < 1l);
            int4* v111;
            v111 = reinterpret_cast<int4*>(v98 + v109);
            int4* v112;
            v112 = reinterpret_cast<int4*>(v1 + v110);
            assert("Pointer alignment check" && (unsigned long long)(v111) % 4l == 0 && (unsigned long long)(v112) % 4l == 0);
            *v112 = *v111;
            v107 += 1l ;
        }
        v13 += 1l ;
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
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def main_body():
    v0 = cp.random.normal(0.0,1.0,128,dtype=cp.float32) # type: ignore
    v1 = cp.empty(128,dtype=cp.float32)
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
    v7((1,),(32,),(v0, v1),shared_mem=81920)
    del v0, v7
    v27 = 0
    v28 = "{}"
    print(v28.format('['),end="")
    v29 = 0
    while method0(v29):
        v31 = v27
        v32 = v31 >= 100
        del v31
        if v32:
            v33 = " ..."
            print(v28.format(v33),end="")
            del v33
            break
        else:
            pass
        del v32
        v34 = v29 == 0
        v35 = v34 != True
        del v34
        if v35:
            v36 = "; "
            print(v28.format(v36),end="")
            del v36
        else:
            pass
        del v35
        print(v28.format('['),end="")
        v37 = 0
        while method0(v37):
            v39 = v27
            v40 = v39 >= 100
            del v39
            if v40:
                v41 = " ..."
                print(v28.format(v41),end="")
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
                print(v28.format(v44),end="")
                del v44
            else:
                pass
            del v43
            v45 = v27 + 1
            v27 = v45
            del v45
            v46 = v29 * 4
            v47 = v46 + v37
            del v46
            v48 = v1[v47].item()
            del v47
            v49 = "{:.6f}"
            print(v49.format(v48),end="")
            del v48, v49
            v37 += 1 
        del v37
        print(v28.format(']'),end="")
        v29 += 1 
    del v1, v27, v29
    print(v28.format(']'),end="")
    del v28
    v50 = "\n"
    print(v50.format(),end="")
    del v50
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
