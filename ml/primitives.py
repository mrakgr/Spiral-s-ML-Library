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
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 16384;
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
            v47 = v42 % 4;
            int v48;
            v48 = v42 / 4;
            bool v49;
            v49 = v48 < 8;
            bool v50;
            v50 = v49 == false;
            if (v50){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v49);
            } else {
            }
            assert("Tensor range check" && 0 <= v48 && v48 < 8);
            assert("Tensor range check" && 0 <= v47 && v47 < 4);
            int v52;
            v52 = 129 * v47;
            int v53;
            v53 = v48 + v52;
            int v54;
            v54 = v4[v53];
            assert("Tensor range check" && 0 <= v48 && v48 < 8);
            assert("Tensor range check" && 0 <= v47 && v47 < 4);
            int v55;
            v55 = v47 + v23;
            int v56;
            v56 = 4 * v48;
            int v57;
            v57 = v56 + v55;
            v1[v57] = v54;
            v42 += 256 ;
        }
        __syncthreads();
        v7 += 24 ;
    }
    v2.sync() ;
    return ;
}
extern "C" __global__ void entry1(int * v0, int * v1) {
    auto v2 = cooperative_groups::this_grid();
    extern __shared__ unsigned char v3[];
    int * v4;
    v4 = reinterpret_cast<int *>(&v3[0ull]);
    int v6;
    v6 = blockIdx.x;
    int v7;
    v7 = v6;
    while (while_method_2(v7)){
        bool v9;
        v9 = 0 <= v7;
        bool v10;
        v10 = v9 == false;
        if (v10){
            assert("The index needs to be zero or positive." && v9);
        } else {
        }
        int v12;
        v12 = v7 % 2;
        int v13;
        v13 = v7 / 2;
        int v14;
        v14 = v13 % 1;
        bool v15;
        v15 = v13 < 2;
        bool v16;
        v16 = v15 == false;
        if (v16){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v15);
        } else {
        }
        assert("Tensor range check" && 0 <= v13 && v13 < 2);
        assert("Tensor range check" && 0 <= v14 && v14 < 1);
        assert("Tensor range check" && 0 <= v12 && v12 < 2);
        int v18;
        v18 = 128 * v12;
        int v19;
        v19 = 32768 * v14;
        int v20;
        v20 = v19 + v18;
        int v21;
        v21 = 32768 * v13;
        int v22;
        v22 = v21 + v20;
        int v23;
        v23 = 16384 * v12;
        int v24;
        v24 = 128 * v14;
        int v25;
        v25 = v24 + v23;
        int v26;
        v26 = v21 + v25;
        int v27;
        v27 = threadIdx.x;
        int v28;
        v28 = v27;
        while (while_method_3(v28)){
            bool v30;
            v30 = 0 <= v28;
            bool v31;
            v31 = v30 == false;
            if (v31){
                assert("The index needs to be zero or positive." && v30);
            } else {
            }
            int v33;
            v33 = v28 % 128;
            int v34;
            v34 = v28 / 128;
            bool v35;
            v35 = v34 < 128;
            bool v36;
            v36 = v35 == false;
            if (v36){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v35);
            } else {
            }
            assert("Tensor range check" && 0 <= v34 && v34 < 128);
            assert("Tensor range check" && 0 <= v33 && v33 < 128);
            int v38;
            v38 = v33 + v22;
            int v39;
            v39 = 256 * v34;
            int v40;
            v40 = v39 + v38;
            int v41;
            v41 = v0[v40];
            assert("Tensor range check" && 0 <= v34 && v34 < 128);
            assert("Tensor range check" && 0 <= v33 && v33 < 128);
            int v42;
            v42 = 129 * v34;
            int v43;
            v43 = v42 + v33;
            v4[v43] = v41;
            v28 += 256 ;
        }
        int v44;
        v44 = threadIdx.x;
        int v45;
        v45 = v44;
        while (while_method_3(v45)){
            bool v47;
            v47 = 0 <= v45;
            bool v48;
            v48 = v47 == false;
            if (v48){
                assert("The index needs to be zero or positive." && v47);
            } else {
            }
            int v50;
            v50 = v45 % 128;
            int v51;
            v51 = v45 / 128;
            bool v52;
            v52 = v51 < 128;
            bool v53;
            v53 = v52 == false;
            if (v53){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v52);
            } else {
            }
            assert("Tensor range check" && 0 <= v51 && v51 < 128);
            assert("Tensor range check" && 0 <= v50 && v50 < 128);
            int v55;
            v55 = 129 * v50;
            int v56;
            v56 = v51 + v55;
            int v57;
            v57 = v4[v56];
            assert("Tensor range check" && 0 <= v51 && v51 < 128);
            assert("Tensor range check" && 0 <= v50 && v50 < 128);
            int v58;
            v58 = v50 + v26;
            int v59;
            v59 = 128 * v51;
            int v60;
            v60 = v59 + v58;
            v1[v60] = v57;
            v45 += 256 ;
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

import sys
import pathlib
options = []
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=255')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method1(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method2(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method3(v0 : i32) -> bool:
    v1 = v0 < 8
    del v0
    return v1
def method0() -> None:
    v0 = "test_text_outputs/primitives/"
    v1 = "test5/a"
    v2 = "transpose.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.arange(0,64,1,dtype=cp.int32) # type: ignore
    v5 = v4.size
    v6 = 64 == v5
    del v5
    v7 = v6 == False
    if v7:
        v8 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v6, v8
        del v8
    else:
        pass
    del v6, v7
    v9 = cp.empty(64,dtype=cp.int32)
    v45 = 0
    v46 = "{}"
    print(v46.format('['),end="")
    v47 = 0
    while method1(v47):
        v49 = v45
        v50 = v49 >= 2147483647
        del v49
        if v50:
            v51 = " ..."
            print(v46.format(v51),end="")
            del v51
            break
        else:
            pass
        del v50
        v52 = v47 == 0
        v53 = v52 != True
        del v52
        if v53:
            v54 = "; "
            print(v46.format(v54),end="")
            del v54
        else:
            pass
        del v53
        print(v46.format('['),end="")
        v55 = 0
        while method2(v55):
            v57 = v45
            v58 = v57 >= 2147483647
            del v57
            if v58:
                v59 = " ..."
                print(v46.format(v59),end="")
                del v59
                break
            else:
                pass
            del v58
            v60 = v55 == 0
            v61 = v60 != True
            del v60
            if v61:
                v62 = "; "
                print(v46.format(v62),end="")
                del v62
            else:
                pass
            del v61
            print(v46.format('['),end="")
            v63 = 0
            while method3(v63):
                v65 = v45
                v66 = v65 >= 2147483647
                del v65
                if v66:
                    v67 = " ..."
                    print(v46.format(v67),end="")
                    del v67
                    break
                else:
                    pass
                del v66
                v68 = v63 == 0
                v69 = v68 != True
                del v68
                if v69:
                    v70 = "; "
                    print(v46.format(v70),end="")
                    del v70
                else:
                    pass
                del v69
                v71 = v45 + 1
                v45 = v71
                del v71
                v72 = v47 * 32
                v73 = v55 * 8
                v74 = v72 + v73
                del v72, v73
                v75 = v74 + v63
                del v74
                v76 = v4[v75].item()
                del v75
                print(v46.format(v76),end="")
                del v76
                v63 += 1 
            del v63
            print(v46.format(']'),end="")
            v55 += 1 
        del v55
        print(v46.format(']'),end="")
        v47 += 1 
    del v45, v47
    print(v46.format(']'),end="")
    v77 = "\n"
    print(v77.format(),end="")
    v78 = cp.cuda.Device().attributes['MultiProcessorCount']
    v79 = v78 == 24
    del v78
    v80 = v79 == False
    if v80:
        v81 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v79, v81
        del v81
    else:
        pass
    del v79, v80
    v82 = 0
    v83 = raw_module.get_function(f"entry{v82}")
    del v82
    v83.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v83((24,),(256,),(v4, v9),shared_mem=98304)
    del v4, v83
    v117 = 0
    print(v46.format('['),end="")
    v118 = 0
    while method1(v118):
        v120 = v117
        v121 = v120 >= 2147483647
        del v120
        if v121:
            v122 = " ..."
            print(v46.format(v122),end="")
            del v122
            break
        else:
            pass
        del v121
        v123 = v118 == 0
        v124 = v123 != True
        del v123
        if v124:
            v125 = "; "
            print(v46.format(v125),end="")
            del v125
        else:
            pass
        del v124
        print(v46.format('['),end="")
        v126 = 0
        while method3(v126):
            v128 = v117
            v129 = v128 >= 2147483647
            del v128
            if v129:
                v130 = " ..."
                print(v46.format(v130),end="")
                del v130
                break
            else:
                pass
            del v129
            v131 = v126 == 0
            v132 = v131 != True
            del v131
            if v132:
                v133 = "; "
                print(v46.format(v133),end="")
                del v133
            else:
                pass
            del v132
            print(v46.format('['),end="")
            v134 = 0
            while method2(v134):
                v136 = v117
                v137 = v136 >= 2147483647
                del v136
                if v137:
                    v138 = " ..."
                    print(v46.format(v138),end="")
                    del v138
                    break
                else:
                    pass
                del v137
                v139 = v134 == 0
                v140 = v139 != True
                del v139
                if v140:
                    v141 = "; "
                    print(v46.format(v141),end="")
                    del v141
                else:
                    pass
                del v140
                v142 = v117 + 1
                v117 = v142
                del v142
                v143 = v118 * 32
                v144 = v126 * 4
                v145 = v143 + v144
                del v143, v144
                v146 = v145 + v134
                del v145
                v147 = v9[v146].item()
                del v146
                print(v46.format(v147),end="")
                del v147
                v134 += 1 
            del v134
            print(v46.format(']'),end="")
            v126 += 1 
        del v126
        print(v46.format(']'),end="")
        v118 += 1 
    del v9, v117, v118
    print(v46.format(']'),end="")
    del v46
    print(v77.format(),end="")
    del v77
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def method5(v0 : i32) -> bool:
    v1 = v0 < 128
    del v0
    return v1
def method6(v0 : i32) -> bool:
    v1 = v0 < 256
    del v0
    return v1
def method4() -> None:
    v0 = "test_text_outputs/primitives/"
    v1 = "test5/b"
    v2 = "transpose.txt"
    v3 = pathlib.Path(v0,v1,v2)
    del v0, v1, v2
    v3.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(str(v3),'w')
    del v3
    v4 = cp.arange(0,65536,1,dtype=cp.int32) # type: ignore
    v5 = v4.size
    v6 = 65536 == v5
    del v5
    v7 = v6 == False
    if v7:
        v8 = "The total length of the reshaped tensor dimension must match that of the original one."
        assert v6, v8
        del v8
    else:
        pass
    del v6, v7
    v9 = cp.empty(65536,dtype=cp.int32)
    v45 = 0
    v46 = "{}"
    print(v46.format('['),end="")
    v47 = 0
    while method1(v47):
        v49 = v45
        v50 = v49 >= 2147483647
        del v49
        if v50:
            v51 = " ..."
            print(v46.format(v51),end="")
            del v51
            break
        else:
            pass
        del v50
        v52 = v47 == 0
        v53 = v52 != True
        del v52
        if v53:
            v54 = "; "
            print(v46.format(v54),end="")
            del v54
        else:
            pass
        del v53
        print(v46.format('['),end="")
        v55 = 0
        while method5(v55):
            v57 = v45
            v58 = v57 >= 2147483647
            del v57
            if v58:
                v59 = " ..."
                print(v46.format(v59),end="")
                del v59
                break
            else:
                pass
            del v58
            v60 = v55 == 0
            v61 = v60 != True
            del v60
            if v61:
                v62 = "; "
                print(v46.format(v62),end="")
                del v62
            else:
                pass
            del v61
            print(v46.format('['),end="")
            v63 = 0
            while method6(v63):
                v65 = v45
                v66 = v65 >= 2147483647
                del v65
                if v66:
                    v67 = " ..."
                    print(v46.format(v67),end="")
                    del v67
                    break
                else:
                    pass
                del v66
                v68 = v63 == 0
                v69 = v68 != True
                del v68
                if v69:
                    v70 = "; "
                    print(v46.format(v70),end="")
                    del v70
                else:
                    pass
                del v69
                v71 = v45 + 1
                v45 = v71
                del v71
                v72 = v47 * 32768
                v73 = v55 * 256
                v74 = v72 + v73
                del v72, v73
                v75 = v74 + v63
                del v74
                v76 = v4[v75].item()
                del v75
                print(v46.format(v76),end="")
                del v76
                v63 += 1 
            del v63
            print(v46.format(']'),end="")
            v55 += 1 
        del v55
        print(v46.format(']'),end="")
        v47 += 1 
    del v45, v47
    print(v46.format(']'),end="")
    v77 = "\n"
    print(v77.format(),end="")
    v78 = cp.cuda.Device().attributes['MultiProcessorCount']
    v79 = v78 == 24
    del v78
    v80 = v79 == False
    if v80:
        v81 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v79, v81
        del v81
    else:
        pass
    del v79, v80
    v82 = 1
    v83 = raw_module.get_function(f"entry{v82}")
    del v82
    v83.max_dynamic_shared_size_bytes = 98304 
    print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
    v83((24,),(256,),(v4, v9),shared_mem=98304)
    del v4, v83
    v117 = 0
    print(v46.format('['),end="")
    v118 = 0
    while method1(v118):
        v120 = v117
        v121 = v120 >= 2147483647
        del v120
        if v121:
            v122 = " ..."
            print(v46.format(v122),end="")
            del v122
            break
        else:
            pass
        del v121
        v123 = v118 == 0
        v124 = v123 != True
        del v123
        if v124:
            v125 = "; "
            print(v46.format(v125),end="")
            del v125
        else:
            pass
        del v124
        print(v46.format('['),end="")
        v126 = 0
        while method6(v126):
            v128 = v117
            v129 = v128 >= 2147483647
            del v128
            if v129:
                v130 = " ..."
                print(v46.format(v130),end="")
                del v130
                break
            else:
                pass
            del v129
            v131 = v126 == 0
            v132 = v131 != True
            del v131
            if v132:
                v133 = "; "
                print(v46.format(v133),end="")
                del v133
            else:
                pass
            del v132
            print(v46.format('['),end="")
            v134 = 0
            while method5(v134):
                v136 = v117
                v137 = v136 >= 2147483647
                del v136
                if v137:
                    v138 = " ..."
                    print(v46.format(v138),end="")
                    del v138
                    break
                else:
                    pass
                del v137
                v139 = v134 == 0
                v140 = v139 != True
                del v139
                if v140:
                    v141 = "; "
                    print(v46.format(v141),end="")
                    del v141
                else:
                    pass
                del v140
                v142 = v117 + 1
                v117 = v142
                del v142
                v143 = v118 * 32768
                v144 = v126 * 128
                v145 = v143 + v144
                del v143, v144
                v146 = v145 + v134
                del v145
                v147 = v9[v146].item()
                del v146
                print(v46.format(v147),end="")
                del v147
                v134 += 1 
            del v134
            print(v46.format(']'),end="")
            v126 += 1 
        del v126
        print(v46.format(']'),end="")
        v118 += 1 
    del v9, v117, v118
    print(v46.format(']'),end="")
    del v46
    print(v77.format(),end="")
    del v77
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return 
def main_body():
    method0()
    return method4()

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
