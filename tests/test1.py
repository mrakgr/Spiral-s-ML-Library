kernel = r"""
#include "reference_counting.h"
struct ClosureBase0 { int refc = 0; virtual long operator()(long, long) = 0; virtual ~ClosureBase0() = default; };
typedef csptr<ClosureBase0> Fun0;
__device__ void write_0(long v0);
struct Closure0 : public ClosureBase0 {
    __device__ long operator()(long tup0, long tup1) override {
        long v0 = tup0; long v1 = tup1;
        long v2;
        v2 = v0 + v1;
        return v2;
    }
    ~Closure0() override = default;
};
__device__ void write_0(long v0){
    const char * v1;
    v1 = "%d";
    printf(v1,v0);
    return ;
}
extern "C" __global__ void entry0() {
    long v0;
    v0 = threadIdx.x;
    long v1;
    v1 = blockIdx.x;
    long v2;
    v2 = v1 * 32l;
    long v3;
    v3 = v0 + v2;
    bool v4;
    v4 = v3 == 0l;
    if (v4){
        Fun0 v5{new Closure0{}};
        long v6;
        v6 = v5(1l, 2l);
        write_0(v6);
        printf("\n");
        return ;
    } else {
        return ;
    }
}
"""
class static_array(list):
    def __init__(self, length):
        for _ in range(length):
            self.append(None)

class static_array_list(static_array):
    def __init__(self, length):
        super().__init__(length)
        self.length = 0
import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

options = []
options.append('--diag-suppress=550,20012')
options.append('--dopt=on')
options.append('--restrict')
options.append('-I C:/Spiral\'s ML Library/cpplib')
raw_module = cp.RawModule(code=kernel, backend='nvrtc', enable_cooperative_groups=True, options=tuple(options))
def main():
    v0 = 0
    v1 = raw_module.get_function(f"entry{v0}")
    del v0
    v1.max_dynamic_shared_size_bytes = 0 
    v1((1,),(32,),(),shared_mem=0)
    del v1
    return 

if __name__ == '__main__': print(main())
