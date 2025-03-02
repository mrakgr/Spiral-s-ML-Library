kernel = r"""
struct ClosureBase0 { int refc{0}; __device__ virtual int operator()(int, int) = 0; __device__ virtual ~ClosureBase0() = default; };
struct Closure0 : ClosureBase0 {
    __device__ int operator()(int tup0, int tup1) override {
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
    ~Closure0() override = default;
};
extern "C" __global__ void entry0() {
    auto x = new Closure0{};
    delete x;
}
"""

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
def main():
    v0 = 0
    v1 = raw_module.get_function(f"entry{v0}")
    del v0
    v1.max_dynamic_shared_size_bytes = 0 
    v1((1,),(1,),(),shared_mem=0)
    del v1
    cp.cuda.get_current_stream().synchronize()
    return 

if __name__ == '__main__': print(main())
